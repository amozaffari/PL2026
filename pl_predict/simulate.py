"""Monte Carlo simulation of the upcoming Premier League season."""

from typing import NamedTuple

import numpy as np
import pandas as pd

from .config import UPCOMING_SEASON
from .data_sources.fpl import load_fixtures
from .goals_model import MAX_GOALS, GoalsModel
from .pipeline import replay_history


class SeasonContext(NamedTuple):
    """Everything expensive, computed once: fixtures, fitted model, Elo ratings."""

    fixtures: pd.DataFrame
    model: GoalsModel
    ratings: dict[str, float]


def build_context() -> SeasonContext:
    pl, elo = replay_history()
    # Summer regression toward the mean — but only while the predicted season is
    # absent from the data. Once its matches appear, replay_history has already
    # applied the regression at the season boundary.
    if elo.last_season != UPCOMING_SEASON:
        elo.new_season()

    model = GoalsModel()
    model.fit(pl)

    fixtures = load_fixtures()
    missing = set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])
    missing -= set(elo.ratings)
    if missing:
        raise ValueError(f"No Elo rating for: {missing} — check team-name mapping.")
    return SeasonContext(fixtures, model, dict(elo.ratings))


def match_probabilities(
    ctx: SeasonContext | None = None,
    elo_offsets: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, GoalsModel, dict]:
    """Fixture list with H/D/A probabilities and expected goals for every match.

    elo_offsets are per-team rating adjustments (e.g. market-implied) applied
    on top of the historical Elo before probabilities are computed.
    """
    if ctx is None:
        ctx = build_context()
    off = elo_offsets or {}
    fixtures = ctx.fixtures.copy()
    fixtures["EloHome"] = fixtures["HomeTeam"].map(
        lambda t: ctx.ratings[t] + off.get(t, 0.0))
    fixtures["EloAway"] = fixtures["AwayTeam"].map(
        lambda t: ctx.ratings[t] + off.get(t, 0.0))
    fixtures["EloDiff"] = fixtures["EloHome"] - fixtures["EloAway"]

    lam_h, lam_a = ctx.model.lambdas(fixtures["EloDiff"].to_numpy())
    probs = ctx.model.outcome_probs(fixtures["EloDiff"].to_numpy())
    fixtures["xG_home"] = lam_h
    fixtures["xG_away"] = lam_a
    fixtures[["p_home", "p_draw", "p_away"]] = probs
    return fixtures, ctx.model, ctx.ratings


def simulate_season(
    n_sims: int = 10_000,
    seed: int = 42,
    ctx: SeasonContext | None = None,
    elo_offsets: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-match probabilities, season projection table)."""
    fixtures, model, _ = match_probabilities(ctx, elo_offsets)
    teams = sorted(set(fixtures["HomeTeam"]))
    t_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_matches = len(fixtures)

    rng = np.random.default_rng(seed)
    n_cells = (MAX_GOALS + 1) ** 2

    # Sample a full scoreline (not just H/D/A) so goal difference breaks ties.
    sampled = np.empty((n_matches, n_sims), dtype=np.int16)
    for i, (lh, la) in enumerate(zip(fixtures["xG_home"], fixtures["xG_away"])):
        flat = model.score_matrix(lh, la).ravel()
        sampled[i] = rng.choice(n_cells, size=n_sims, p=flat / flat.sum())
    hg = sampled // (MAX_GOALS + 1)
    ag = sampled % (MAX_GOALS + 1)

    # Matches already played are fixed to their actual result in every run,
    # so mid-season the simulation only randomises the remaining fixtures.
    for i, r in enumerate(fixtures.itertuples()):
        if r.finished and pd.notna(r.FTHG):
            hg[i, :] = min(int(r.FTHG), MAX_GOALS)
            ag[i, :] = min(int(r.FTAG), MAX_GOALS)

    home_idx = fixtures["HomeTeam"].map(t_idx).to_numpy()
    away_idx = fixtures["AwayTeam"].map(t_idx).to_numpy()

    pts = np.zeros((n_sims, n_teams))
    gd = np.zeros((n_sims, n_teams))
    gf = np.zeros((n_sims, n_teams))
    home_pts = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
    away_pts = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
    for i in range(n_matches):
        pts[:, home_idx[i]] += home_pts[i]
        pts[:, away_idx[i]] += away_pts[i]
        gd[:, home_idx[i]] += hg[i] - ag[i]
        gd[:, away_idx[i]] += ag[i] - hg[i]
        gf[:, home_idx[i]] += hg[i]
        gf[:, away_idx[i]] += ag[i]

    # Rank within each simulation: points, then GD, then goals for.
    key = pts * 1e6 + gd * 1e3 + gf
    order = np.argsort(-key, axis=1, kind="stable")
    position = np.empty_like(order)
    rows_ix = np.arange(n_sims)[:, None]
    position[rows_ix, order] = np.arange(1, n_teams + 1)

    table = pd.DataFrame(
        {
            "team": teams,
            "exp_points": pts.mean(axis=0).round(1),
            "exp_gd": gd.mean(axis=0).round(1),
            "p_champion": (position == 1).mean(axis=0),
            "p_top4": (position <= 4).mean(axis=0),
            "p_top6": (position <= 6).mean(axis=0),
            "p_relegation": (position >= n_teams - 2).mean(axis=0),
            "median_position": np.median(position, axis=0).astype(int),
        }
    ).sort_values("exp_points", ascending=False).reset_index(drop=True)

    match_cols = ["gameweek", "kickoff_utc", "HomeTeam", "AwayTeam", "finished",
                  "FTHG", "FTAG", "xG_home", "xG_away", "p_home", "p_draw", "p_away"]
    return fixtures[match_cols], table
