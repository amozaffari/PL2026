"""Prediction-market probabilities from Polymarket's public (keyless) Gamma API.

Two things are read: the season champion market, and per-match markets whose
slugs follow ``epl-{home}-{away}-{YYYY-MM-DD}`` with 3-letter team codes.
"""

import json

import pandas as pd
import requests

from ..config import USER_AGENT

GAMMA = "https://gamma-api.polymarket.com"
CHAMPION_EVENT_SLUG = "epl-2027-champion-20260701200428749"

# Polymarket display name -> football-data canonical name.
PM_TO_FD = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Chelsea": "Chelsea",
    "Coventry City": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
}

# 3-letter codes used in Polymarket match-event slugs.
MATCH_CODES = {
    "Arsenal": "ars", "Aston Villa": "ast", "Bournemouth": "bou",
    "Brentford": "bre", "Brighton": "bri", "Chelsea": "che",
    "Coventry": "cov", "Crystal Palace": "cry", "Everton": "eve",
    "Fulham": "ful", "Hull": "hul", "Ipswich": "ips", "Leeds": "lee",
    "Liverpool": "liv", "Man City": "mac", "Man United": "mun",
    "Newcastle": "new", "Nott'm Forest": "not", "Sunderland": "sun",
    "Tottenham": "tot",
}

# Substring of the match-market question that identifies the team.
QUESTION_NAMES = {
    "Man City": "Manchester City", "Man United": "Manchester United",
    "Newcastle": "Newcastle", "Nott'm Forest": "Nottingham",
    "Tottenham": "Tottenham", "Coventry": "Coventry", "Hull": "Hull",
    "Ipswich": "Ipswich", "Leeds": "Leeds", "Brighton": "Brighton",
}


def _get(path: str, **params):
    resp = requests.get(f"{GAMMA}{path}", params=params,
                        headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _yes_price(market: dict) -> float:
    return float(json.loads(market["outcomePrices"])[0])


def title_probs() -> pd.DataFrame:
    """De-vigged champion probabilities per team from the season winner market."""
    event = _get("/events", slug=CHAMPION_EVENT_SLUG)[0]
    rows = []
    for m in event["markets"]:
        team = PM_TO_FD.get(m.get("groupItemTitle", ""))
        if team is not None:  # skips "Team A/B/C"-style placeholder markets
            rows.append({"team": team, "p_title_market": _yes_price(m)})
    df = pd.DataFrame(rows)
    df["p_title_market"] /= df["p_title_market"].sum()
    return df


def match_probs(home: str, away: str, kickoff_utc) -> dict | None:
    """De-vigged H/D/A probabilities for one fixture, or None if no market exists."""
    date = pd.Timestamp(kickoff_utc).strftime("%Y-%m-%d")
    slug = f"epl-{MATCH_CODES[home]}-{MATCH_CODES[away]}-{date}"
    events = _get("/events", slug=slug)
    if not events:
        return None

    p_home = p_draw = p_away = None
    home_q = QUESTION_NAMES.get(home, home)
    away_q = QUESTION_NAMES.get(away, away)
    for m in events[0]["markets"]:
        q = m["question"].lower()
        if "draw" in q:
            p_draw = _yes_price(m)
        elif home_q.lower() in q:
            p_home = _yes_price(m)
        elif away_q.lower() in q:
            p_away = _yes_price(m)
    if None in (p_home, p_draw, p_away):
        return None
    total = p_home + p_draw + p_away
    return {"pm_home": p_home / total, "pm_draw": p_draw / total,
            "pm_away": p_away / total, "slug": slug}
