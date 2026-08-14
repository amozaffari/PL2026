"""Shared pipeline step: replay history chronologically to get pre-match Elo diffs."""

import pandas as pd

from .elo import Elo
from .data_sources.football_data import load_matches


def replay_history() -> tuple[pd.DataFrame, Elo]:
    """Run Elo over every E0+E1 match in order.

    Returns (premier_league_matches with pre-match EloDiff, final Elo state).
    Season regression is applied at each season boundary.
    """
    matches = load_matches()
    elo = Elo()
    current_season = None
    diffs = []
    for row in matches.itertuples(index=False):
        if row.Season != current_season:
            if current_season is not None:
                elo.new_season()
            current_season = row.Season
        diff = elo.get(row.HomeTeam, row.Div) - elo.get(row.AwayTeam, row.Div)
        diffs.append(diff)
        elo.update(row.HomeTeam, row.AwayTeam, row.FTHG, row.FTAG, row.Div)

    elo.last_season = current_season  # lets callers know how far the data runs
    matches = matches.assign(EloDiff=diffs)
    pl = matches[matches["Div"] == "E0"].reset_index(drop=True)
    return pl, elo
