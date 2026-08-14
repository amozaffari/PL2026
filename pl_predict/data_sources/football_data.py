"""Download and load historical results + bookmaker odds from football-data.co.uk."""

import io

import pandas as pd
import requests

from ..config import DIVISIONS, FOOTBALL_DATA_URL, RAW_DIR, SEASONS, USER_AGENT

KEEP_COLS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Odds column preference: closing average, pre-match average, Bet Brain average, Bet365.
ODDS_PREFERENCE = [
    ("AvgCH", "AvgCD", "AvgCA"),
    ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"),
    ("B365CH", "B365CD", "B365CA"),
    ("B365H", "B365D", "B365A"),
]


def fetch_all(force: bool = False) -> None:
    """Download every season/division CSV into data/raw.

    Completed seasons are skipped when already present; the current season's
    file is always re-downloaded (it grows every matchweek) and tolerated as a
    404 before the season's first results are published.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for season in SEASONS:
        current = season == SEASONS[-1]
        for div in DIVISIONS:
            dest = RAW_DIR / f"{season}_{div}.csv"
            if dest.exists() and not force and not current:
                continue
            url = FOOTBALL_DATA_URL.format(season=season, div=div)
            resp = session.get(url, timeout=30)
            if resp.status_code == 404 and current:
                print(f"{season}/{div} not published yet — skipping")
                continue
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"downloaded {dest.name} ({len(resp.content) // 1024} KB)")


def _read_one(path) -> pd.DataFrame:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    # Placeholder files early in a season may lack the result columns entirely.
    required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    df = df.dropna(subset=required)
    if df.empty:
        return pd.DataFrame()

    out = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    out["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")

    # Attach best-available 1X2 odds under uniform names.
    for h, d, a in ODDS_PREFERENCE:
        if h in df.columns and df[h].notna().mean() > 0.5:
            out["OddsH"] = pd.to_numeric(df[h], errors="coerce")
            out["OddsD"] = pd.to_numeric(df[d], errors="coerce")
            out["OddsA"] = pd.to_numeric(df[a], errors="coerce")
            break
    else:
        out["OddsH"] = out["OddsD"] = out["OddsA"] = float("nan")

    season = path.name.split("_")[0]
    out["Season"] = season
    return out


def load_matches() -> pd.DataFrame:
    """All downloaded matches, chronological, with FTHG/FTAG/FTR and OddsH/D/A."""
    frames = []
    for path in sorted(RAW_DIR.glob("*_E*.csv")):
        df = _read_one(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No raw data found — run `python -m pl_predict fetch` first.")
    matches = pd.concat(frames, ignore_index=True)
    matches["FTHG"] = matches["FTHG"].astype(int)
    matches["FTAG"] = matches["FTAG"].astype(int)
    matches = matches.sort_values(["Date", "Div"], kind="stable").reset_index(drop=True)
    return matches
