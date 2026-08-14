"""Goal-margin-weighted Elo ratings maintained across Premier League and Championship.

Running Elo over both divisions means promoted clubs arrive with a rating earned
from real Championship matches instead of an arbitrary prior.
"""

import math

BASE_K = 20.0
HOME_ADV = 60.0  # Elo points added to the home side inside the expectation
INIT_E0 = 1450.0  # debut rating when a club first appears in the Premier League
INIT_E1 = 1350.0  # debut rating when a club first appears in the Championship
MEAN_RATING = 1500.0
SEASON_REGRESSION = 0.25  # fraction reverted toward the league mean each summer


class Elo:
    def __init__(self):
        self.ratings: dict[str, float] = {}
        self.last_season: str | None = None  # season code of the last match replayed

    def get(self, team: str, div: str = "E0") -> float:
        if team not in self.ratings:
            self.ratings[team] = INIT_E0 if div == "E0" else INIT_E1
        return self.ratings[team]

    @staticmethod
    def expected_home(diff: float) -> float:
        """Win expectancy for the home side given (home - away + HOME_ADV) rating diff."""
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update(self, home: str, away: str, hg: int, ag: int, div: str) -> None:
        rh, ra = self.get(home, div), self.get(away, div)
        exp_home = self.expected_home(rh + HOME_ADV - ra)
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margin_mult = math.log(abs(hg - ag) + 1.0) + 1.0
        delta = BASE_K * margin_mult * (score - exp_home)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta

    def new_season(self) -> None:
        for team in self.ratings:
            self.ratings[team] += SEASON_REGRESSION * (MEAN_RATING - self.ratings[team])
