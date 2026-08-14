"""Walk-forward backtest: Elo->goals model vs de-vigged bookmaker odds vs blend.

For each evaluated season the goals model is fit only on earlier seasons; Elo is
sequential by construction, so nothing leaks from the future.
"""

import numpy as np
import pandas as pd

from .goals_model import GoalsModel
from .odds import implied_probs
from .pipeline import replay_history

BLEND_W = 0.5  # weight on odds in the model/odds probability blend

OUTCOME_IDX = {"H": 0, "D": 1, "A": 2}


def _metrics(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    onehot = np.eye(3)[outcomes]
    picked = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-12, None)
    return {
        "log_loss": float(-np.mean(np.log(picked))),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "accuracy": float(np.mean(np.argmax(probs, axis=1) == outcomes)),
    }


def run(n_eval_seasons: int = 3) -> pd.DataFrame:
    pl, _ = replay_history()
    seasons = sorted(pl["Season"].unique())
    eval_seasons = seasons[-n_eval_seasons:]

    rows = []
    for season in eval_seasons:
        train = pl[pl["Season"] < season]
        test = pl[pl["Season"] == season].copy()

        model = GoalsModel()
        model.fit(train)
        model_p = model.outcome_probs(test["EloDiff"].to_numpy())

        outcomes = test["FTR"].map(OUTCOME_IDX).to_numpy()

        # Compare all three on the subset that has odds, so numbers are comparable.
        has_odds = test[["OddsH", "OddsD", "OddsA"]].notna().all(axis=1).to_numpy()
        odds_p = implied_probs(test.loc[has_odds, "OddsH"],
                               test.loc[has_odds, "OddsD"],
                               test.loc[has_odds, "OddsA"]).T
        blend_p = BLEND_W * odds_p + (1 - BLEND_W) * model_p[has_odds]

        for name, probs, mask in [
            ("elo_poisson", model_p[has_odds], has_odds),
            ("odds_implied", odds_p, has_odds),
            ("blend_50_50", blend_p, has_odds),
        ]:
            m = _metrics(probs, outcomes[mask])
            rows.append({"season": season, "model": name,
                         "n_matches": int(mask.sum()), **m})

    return pd.DataFrame(rows)
