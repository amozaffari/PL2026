"""Prediction history: weekly projection snapshots and a per-match prediction log.

Both live in `history/` (committed to the repo by the daily CI run, unlike the
regenerated `output/` directory), so past predictions stay comparable as real
results come in.

- `history/projections/YYYY-MM-DD.csv`: the projected table, at most one per
  ISO week (the first run of each week takes the snapshot).
- `history/match_predictions.csv`: one row per fixture. Probabilities are
  updated on every run until kickoff, then frozen — so each row holds the last
  pre-match prediction — and the actual result is filled in once played.
"""

import pandas as pd

from .config import OUT_DIR, PROJECT_ROOT

HISTORY_DIR = PROJECT_ROOT / "history"
PROJECTIONS_DIR = HISTORY_DIR / "projections"
MATCH_LOG = HISTORY_DIR / "match_predictions.csv"

PROB_COLS = ["p_home", "p_draw", "p_away"]
KEY_COLS = ["HomeTeam", "AwayTeam"]


def _read_output(preferred: str, fallback: str) -> tuple[pd.DataFrame, bool]:
    """Prefer the market-calibrated file, but never a stale one: if the plain
    file is newer (e.g. the calibration step failed this run), use it."""
    pref, fall = OUT_DIR / preferred, OUT_DIR / fallback
    use_pref = pref.exists() and (
        not fall.exists() or pref.stat().st_mtime >= fall.stat().st_mtime)
    return pd.read_csv(pref if use_pref else fall), use_pref


def snapshot_projection(now: pd.Timestamp | None = None) -> bool:
    """Write this week's projection snapshot; returns False if one exists already."""
    now = now or pd.Timestamp.now(tz="UTC")
    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    this_week = now.isocalendar()
    for existing in PROJECTIONS_DIR.glob("*.csv"):
        week = pd.Timestamp(existing.stem).isocalendar()
        if (week.year, week.week) == (this_week.year, this_week.week):
            return False
    table, calibrated = _read_output("season_projection_market_implied.csv",
                                     "season_projection.csv")
    table.insert(0, "date", now.strftime("%Y-%m-%d"))
    table["source"] = "market_implied" if calibrated else "pure_elo"
    table.to_csv(PROJECTIONS_DIR / f"{now.strftime('%Y-%m-%d')}.csv", index=False)
    return True


def update_match_log(now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Upsert pre-kickoff probabilities and fill results for played matches."""
    now = now or pd.Timestamp.now(tz="UTC")
    HISTORY_DIR.mkdir(exist_ok=True)
    current, _ = _read_output("match_probabilities_market_implied.csv",
                              "match_probabilities.csv")
    current["kickoff_utc"] = pd.to_datetime(current["kickoff_utc"], utc=True)

    if MATCH_LOG.exists():
        log = pd.read_csv(MATCH_LOG)
        log["kickoff_utc"] = pd.to_datetime(log["kickoff_utc"], utc=True)
        # An all-empty string column round-trips through CSV as float64 NaN,
        # and pandas 3 then refuses string assignments into it.
        log = log.astype({"outcome": "object", "pred_date": "object"})
        log = log.set_index(KEY_COLS)
    else:
        log = pd.DataFrame(columns=["gameweek", "kickoff_utc", *PROB_COLS,
                                    "pred_date", "finished", "FTHG", "FTAG",
                                    "outcome"]).set_index(pd.MultiIndex.from_arrays(
                                        [[], []], names=KEY_COLS))

    today = now.strftime("%Y-%m-%d")
    for r in current.itertuples():
        key = (r.HomeTeam, r.AwayTeam)
        known = key in log.index
        if not known:
            log.loc[key, ["gameweek", "kickoff_utc"]] = [r.gameweek, r.kickoff_utc]
        # Probabilities move until kickoff, then the prediction is frozen.
        if not known or r.kickoff_utc > now:
            log.loc[key, PROB_COLS] = [r.p_home, r.p_draw, r.p_away]
            log.loc[key, "pred_date"] = today
        log.loc[key, "finished"] = bool(r.finished)
        if r.finished and pd.notna(r.FTHG):
            hg, ag = int(r.FTHG), int(r.FTAG)
            log.loc[key, ["FTHG", "FTAG"]] = [hg, ag]
            log.loc[key, "outcome"] = "H" if hg > ag else "D" if hg == ag else "A"

    log = log.reset_index().sort_values("kickoff_utc")
    log.to_csv(MATCH_LOG, index=False)
    return log


def load_projection_history() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(PROJECTIONS_DIR.glob("*.csv"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def scoreboard(log: pd.DataFrame | None = None) -> dict | None:
    """Accuracy / log-loss / Brier of frozen predictions on played matches."""
    import numpy as np

    if log is None:
        if not MATCH_LOG.exists():
            return None
        log = pd.read_csv(MATCH_LOG)
    played = log[log["outcome"].notna() & log[PROB_COLS].notna().all(axis=1)]
    if played.empty:
        return None
    probs = played[PROB_COLS].to_numpy()
    idx = played["outcome"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    picked = np.clip(probs[np.arange(len(idx)), idx], 1e-12, None)
    onehot = np.eye(3)[idx]
    return {
        "n": len(played),
        "accuracy": float((probs.argmax(axis=1) == idx).mean()),
        "log_loss": float(-np.log(picked).mean()),
        "brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
    }
