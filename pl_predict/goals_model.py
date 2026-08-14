"""Map Elo difference to expected goals, then to match-outcome probabilities.

Two Poisson GLMs (home goals, away goals) are fit on Premier League matches with
the pre-match Elo difference as the covariate and exponential time-decay weights.
The independent-Poisson score matrix gets the Dixon-Coles low-score correction,
with rho chosen by grid search on the training log-likelihood.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson

MAX_GOALS = 10
DECAY_HALF_LIFE_DAYS = 750.0


class GoalsModel:
    def __init__(self):
        self.home_params = None  # (intercept, slope) for log lambda_home
        self.away_params = None
        self.rho = 0.0

    @staticmethod
    def _design(elo_diff):
        x = np.asarray(elo_diff, float) / 100.0
        return sm.add_constant(x, has_constant="add")

    def fit(self, df: pd.DataFrame) -> None:
        """df needs columns: EloDiff (home - away, pre-match), FTHG, FTAG, Date."""
        age_days = (df["Date"].max() - df["Date"]).dt.days.to_numpy(float)
        w = 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS)
        X = self._design(df["EloDiff"])
        self.home_params = sm.GLM(df["FTHG"].to_numpy(), X,
                                  family=sm.families.Poisson(), freq_weights=w).fit().params
        self.away_params = sm.GLM(df["FTAG"].to_numpy(), X,
                                  family=sm.families.Poisson(), freq_weights=w).fit().params
        self._fit_rho(df, w)

    def lambdas(self, elo_diff):
        X = self._design(np.atleast_1d(elo_diff))
        lam_h = np.exp(X @ self.home_params)
        lam_a = np.exp(X @ self.away_params)
        return lam_h, lam_a

    @staticmethod
    def _dc_tau(matrix, lam_h, lam_a, rho):
        """Apply the Dixon-Coles adjustment to the 0/1 goal cells, renormalise."""
        m = matrix.copy()
        m[0, 0] *= 1.0 - lam_h * lam_a * rho
        m[0, 1] *= 1.0 + lam_h * rho
        m[1, 0] *= 1.0 + lam_a * rho
        m[1, 1] *= 1.0 - rho
        return m / m.sum()

    def score_matrix(self, lam_h: float, lam_a: float) -> np.ndarray:
        goals = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
        return self._dc_tau(m, lam_h, lam_a, self.rho)

    def _fit_rho(self, df: pd.DataFrame, weights: np.ndarray) -> None:
        lam_h, lam_a = self.lambdas(df["EloDiff"])
        hg = df["FTHG"].to_numpy()
        ag = df["FTAG"].to_numpy()
        best_rho, best_ll = 0.0, -np.inf
        for rho in np.arange(-0.15, 0.151, 0.01):
            # Base independent-Poisson pmf, tau multiplier on low-score cells only.
            pmf = poisson.pmf(hg, lam_h) * poisson.pmf(ag, lam_a)
            tau = np.ones_like(pmf)
            tau[(hg == 0) & (ag == 0)] = 1.0 - (lam_h * lam_a * rho)[(hg == 0) & (ag == 0)]
            tau[(hg == 0) & (ag == 1)] = 1.0 + (lam_h * rho)[(hg == 0) & (ag == 1)]
            tau[(hg == 1) & (ag == 0)] = 1.0 + (lam_a * rho)[(hg == 1) & (ag == 0)]
            tau[(hg == 1) & (ag == 1)] = 1.0 - rho
            vals = np.clip(pmf * tau, 1e-12, None)
            ll = float(np.sum(weights * np.log(vals)))
            if ll > best_ll:
                best_ll, best_rho = ll, rho
        self.rho = best_rho

    def outcome_probs(self, elo_diff):
        """P(home win), P(draw), P(away win) for one or many Elo differences."""
        lam_h, lam_a = self.lambdas(elo_diff)
        out = np.empty((len(lam_h), 3))
        for i, (lh, la) in enumerate(zip(lam_h, lam_a)):
            m = self.score_matrix(lh, la)
            out[i] = [np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]
        return out
