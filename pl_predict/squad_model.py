"""Player-level squad strength model.

Bottom-up team strength from the full current roster, replacing the aggregate
transfer/injury heuristics:

- Each player's value is last season's FPL points; players with no PL history
  (foreign signings, promoted-club players, academy debuts) are valued from
  their FPL price via a per-position regression on players who have both.
- In-season, a player's rate blends last season with current form.
- Team strength = the top-15 player values (season horizon ignores short
  injuries; match horizon discounts by availability and chance-of-playing).
- The points -> Elo scale is CALIBRATED, not invented: end-of-season Elo is
  regressed on squad strength across past seasons of the vaastav archive.
"""

import io

import numpy as np
import pandas as pd
import requests

from .config import FPL_BOOTSTRAP_URL, RAW_DIR, USER_AGENT, fpl_to_fd
from .data_sources.football_data import load_matches
from .data_sources.squads import last_season_players
from .elo import Elo

VAASTAV_ROOT = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
# vaastav season dir -> football-data season code; teams.csv exists from 2019-20.
CALIBRATION_SEASONS = {
    "2019-20": "1920", "2020-21": "2021", "2021-22": "2122",
    "2022-23": "2223", "2023-24": "2324", "2024-25": "2425", "2025-26": "2526",
}
TOP_N = 15
OFFSET_CAP = 75.0
# Weight of the squad-implied rating when shrinking Elo toward it. Elo already
# encodes team strength, so the squad signal adjusts rather than adds.
BLEND_W = 0.4

AVAILABILITY = {"a": 1.0, "d": 0.5, "i": 0.0, "s": 0.0, "u": 0.0}


def _get_csv(url: str, cache_name: str) -> pd.DataFrame | None:
    cache = RAW_DIR / cache_name
    if not cache.exists():
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    return pd.read_csv(io.BytesIO(cache.read_bytes()))


def _squad_points(players: pd.DataFrame, value_col: str) -> pd.Series:
    """Top-N player values summed per team."""
    return (players.sort_values(value_col, ascending=False)
            .groupby("team_name")[value_col]
            .apply(lambda s: s.head(TOP_N).sum()))


def _season_end_elos() -> dict[str, dict[str, float]]:
    """Final Elo per team for every season in the data (E0+E1 replay)."""
    matches = load_matches()
    elo = Elo()
    out: dict[str, dict[str, float]] = {}
    current = None
    for row in matches.itertuples(index=False):
        if row.Season != current:
            if current is not None:
                out[current] = dict(elo.ratings)
                elo.new_season()
            current = row.Season
        elo.update(row.HomeTeam, row.AwayTeam, row.FTHG, row.FTAG, row.Div)
    out[current] = dict(elo.ratings)
    return out


def calibrate_pts_to_elo(verbose: bool = False) -> tuple[float, float]:
    """(slope, intercept) mapping top-15 squad FPL points to Elo, fit on
    past seasons — so squad strength converts to the Elo scale empirically."""
    end_elos = _season_end_elos()
    rows = []
    for vaastav_season, code in CALIBRATION_SEASONS.items():
        players = _get_csv(f"{VAASTAV_ROOT}/{vaastav_season}/players_raw.csv",
                           f"vaastav_{code}_players_raw.csv")
        teams = _get_csv(f"{VAASTAV_ROOT}/{vaastav_season}/teams.csv",
                         f"vaastav_{code}_teams.csv")
        if players is None or teams is None or code not in end_elos:
            continue
        names = {r.id: fpl_to_fd(r.name) for r in teams.itertuples()}
        players = players.assign(team_name=players["team"].map(names))
        squad = _squad_points(players, "total_points")
        for team, pts in squad.items():
            if team in end_elos[code]:
                rows.append({"squad_pts": pts, "elo": end_elos[code][team]})
    df = pd.DataFrame(rows)
    slope, intercept = np.polyfit(df["squad_pts"], df["elo"], 1)
    if verbose:
        pred = slope * df["squad_pts"] + intercept
        r2 = 1 - ((df["elo"] - pred) ** 2).sum() / ((df["elo"] - df["elo"].mean()) ** 2).sum()
        print(f"calibration: {len(df)} team-seasons, "
              f"{slope:.3f} Elo per squad point, R^2 = {r2:.2f}")
    return float(slope), float(intercept)


def current_roster() -> pd.DataFrame:
    """Every current player with a value in expected FPL points.

    value_season ignores current fitness (season horizon); value_match
    discounts by availability. In-season, form blends into the rate.
    """
    data = requests.get(FPL_BOOTSTRAP_URL,
                        headers={"User-Agent": USER_AGENT}, timeout=30).json()
    teams = {t["id"]: fpl_to_fd(t["name"]) for t in data["teams"]}
    df = pd.DataFrame(data["elements"])
    df["team_name"] = df["team"].map(teams)
    for col in ("form", "points_per_game", "selected_by_percent"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    prev = last_season_players()[["code", "total_points"]].rename(
        columns={"total_points": "pts_prev"})
    df = df.merge(prev, on="code", how="left")

    # Value newcomers from price: fit pts_prev ~ now_cost per position.
    known = df["pts_prev"].notna() & (df["pts_prev"] > 0)
    df["value_base"] = df["pts_prev"]
    for pos, group in df.groupby("element_type"):
        g = group[known.reindex(group.index, fill_value=False)]
        if len(g) < 10:
            continue
        slope, intercept = np.polyfit(g["now_cost"], g["pts_prev"], 1)
        estimate = (slope * group["now_cost"] + intercept).clip(lower=0)
        fill = group["value_base"].isna() | (group["value_base"] == 0)
        df.loc[group.index[fill], "value_base"] = estimate[fill]

    # In-season: blend last season's level with current form (pts per game
    # over the last 30 days, scaled to a season). Pre-season form is 0 for
    # everyone, so the blend is a no-op until matches are played.
    season_started = df["minutes"].sum() > 0 and (df["form"] > 0).any()
    if season_started:
        df["value_base"] = 0.7 * df["value_base"] + 0.3 * (df["form"] * 38)

    chance = pd.to_numeric(df["chance_of_playing_next_round"],
                           errors="coerce") / 100.0
    availability = df["status"].map(AVAILABILITY).fillna(0.0)
    df["avail"] = chance.fillna(availability).where(df["status"] != "a", 1.0)

    df["value_season"] = df["value_base"].where(df["status"] != "u", 0.0)
    df["value_match"] = df["value_base"] * df["avail"]
    return df[["code", "web_name", "team_name", "element_type", "now_cost",
               "status", "news", "form", "ep_next", "pts_prev", "avail",
               "value_season", "value_match"]]


def squad_strength(horizon: str, ratings: dict[str, float]) -> pd.DataFrame:
    """Per-team squad strength and the Elo adjustment it implies.

    The offset shrinks the team's current Elo toward its squad-implied rating
    (weight BLEND_W) rather than adding on top — Elo already measures strength,
    so the roster signal corrects it instead of double-counting it.
    """
    roster = current_roster()
    col = "value_season" if horizon == "season" else "value_match"
    strength = _squad_points(roster, col).rename("squad_pts").to_frame()
    slope, intercept = calibrate_pts_to_elo()
    strength["implied_elo"] = slope * strength["squad_pts"] + intercept
    strength["current_elo"] = pd.Series(ratings).reindex(strength.index)
    strength["elo_offset"] = (BLEND_W
                              * (strength["implied_elo"] - strength["current_elo"])
                              ).clip(-OFFSET_CAP, OFFSET_CAP)
    return strength.sort_values("squad_pts", ascending=False)


def squad_elo_offsets(horizon: str, ratings: dict[str, float]) -> dict[str, float]:
    return squad_strength(horizon, ratings)["elo_offset"].to_dict()
