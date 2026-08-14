"""Live pre-match bookmaker odds from The Odds API (free key, 500 requests/month).

Get a key at https://the-odds-api.com/ and export it as ODDS_API_KEY. Without a
key every function quietly returns None and the rest of the pipeline works
unchanged; with one, `predict` gains de-vigged bookmaker probabilities and the
backtested model/odds blend for upcoming fixtures.
"""

import os

import numpy as np
import pandas as pd
import requests

from ..config import USER_AGENT

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"

# The Odds API team name -> football-data canonical name.
ODDS_TO_FD = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton and Hove Albion": "Brighton",
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
    "Tottenham Hotspur": "Tottenham",
}


def parse_events(events: list) -> pd.DataFrame:
    """Average h2h odds across bookmakers, de-vig into probabilities."""
    rows = []
    for ev in events:
        home = ODDS_TO_FD.get(ev["home_team"])
        away = ODDS_TO_FD.get(ev["away_team"])
        if home is None or away is None:
            continue
        prices = {ev["home_team"]: [], ev["away_team"]: [], "Draw": []}
        for book in ev.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market["outcomes"]:
                    prices.setdefault(outcome["name"], []).append(outcome["price"])
        if not all(prices[k] for k in (ev["home_team"], ev["away_team"], "Draw")):
            continue
        inv = np.array([1.0 / np.mean(prices[ev["home_team"]]),
                        1.0 / np.mean(prices["Draw"]),
                        1.0 / np.mean(prices[ev["away_team"]])])
        p = inv / inv.sum()
        rows.append({
            "HomeTeam": home, "AwayTeam": away,
            "book_H": p[0], "book_D": p[1], "book_A": p[2],
            "n_books": len(prices[ev["home_team"]]),
        })
    return pd.DataFrame(rows)


def fetch_h2h(api_key: str | None = None) -> pd.DataFrame | None:
    """Upcoming-fixture bookmaker probabilities, or None when no key is set."""
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        return None
    resp = requests.get(
        ODDS_API_URL,
        params={"apiKey": key, "regions": "uk,eu",
                "markets": "h2h", "oddsFormat": "decimal"},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"(The Odds API: {remaining} requests remaining this month)")
    return parse_events(resp.json())
