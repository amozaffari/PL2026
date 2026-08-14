"""Convert bookmaker 1X2 odds into fair probabilities (remove the overround)."""

import numpy as np


def implied_probs(odds_h, odds_d, odds_a):
    """De-vig by proportional normalisation. Accepts scalars or arrays; NaN-safe."""
    inv = np.array([1.0 / np.asarray(odds_h, float),
                    1.0 / np.asarray(odds_d, float),
                    1.0 / np.asarray(odds_a, float)])
    return inv / inv.sum(axis=0)
