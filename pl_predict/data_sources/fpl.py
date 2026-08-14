"""Fixtures and teams for the upcoming season from the (keyless) FPL API."""

import pandas as pd
import requests

from ..config import FPL_BOOTSTRAP_URL, FPL_FIXTURES_URL, USER_AGENT, fpl_to_fd


def load_fixtures() -> pd.DataFrame:
    """All 380 fixtures of the upcoming season with canonical (football-data) team names."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    teams = {
        t["id"]: fpl_to_fd(t["name"])
        for t in session.get(FPL_BOOTSTRAP_URL, timeout=30).json()["teams"]
    }
    fixtures = session.get(FPL_FIXTURES_URL, timeout=30).json()

    rows = []
    for f in fixtures:
        rows.append(
            {
                "gameweek": f["event"],
                "kickoff_utc": f["kickoff_time"],
                "HomeTeam": teams[f["team_h"]],
                "AwayTeam": teams[f["team_a"]],
                "finished": f["finished"],
                "FTHG": f["team_h_score"],
                "FTAG": f["team_a_score"],
            }
        )
    df = pd.DataFrame(rows)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"])
    return df.sort_values(["kickoff_utc", "HomeTeam"]).reset_index(drop=True)
