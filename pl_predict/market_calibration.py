"""Market-implied Elo: nudge team ratings until simulated title odds match Polymarket.

Iteratively simulates the season, compares each team's simulated championship
probability with the market's, and moves the team's Elo proportionally to the
log-ratio. Only teams where either side puts >=1% on the title are adjusted —
below that the title market carries no information (its floor prices and Monte
Carlo noise would produce arbitrary offsets for mid-table and bottom clubs).

The result is a set of per-team Elo offsets that propagate the market's
squad-level knowledge (transfers, injuries, managers) to all 380 fixtures.
"""

import math

from .config import OUT_DIR
from .data_sources.polymarket import title_probs
from .simulate import SeasonContext, simulate_season

LEARNING_RATE = 25.0  # Elo points per unit of log(p_market / p_model)
MAX_STEP = 50.0
MIN_INFORMATIVE_P = 0.01


def market_implied_offsets(
    ctx: SeasonContext,
    iterations: int = 8,
    n_sims: int = 4000,
    verbose: bool = True,
) -> dict[str, float]:
    market_df = title_probs()
    OUT_DIR.mkdir(exist_ok=True)
    market_df.to_csv(OUT_DIR / "title_market.csv", index=False)
    market = dict(zip(market_df["team"], market_df["p_title_market"]))
    offsets: dict[str, float] = {}
    floor = 1.0 / (2 * n_sims)

    for it in range(iterations):
        _, table = simulate_season(n_sims=n_sims, seed=100 + it,
                                   ctx=ctx, elo_offsets=offsets)
        sim = dict(zip(table["team"], table["p_champion"]))
        max_move = 0.0
        for team, p_mkt in market.items():
            p_sim = sim.get(team, 0.0)
            if max(p_mkt, p_sim) < MIN_INFORMATIVE_P:
                continue
            step = LEARNING_RATE * math.log(max(p_mkt, floor) / max(p_sim, floor))
            step = max(-MAX_STEP, min(MAX_STEP, step))
            offsets[team] = offsets.get(team, 0.0) + step
            max_move = max(max_move, abs(step))
        if verbose:
            adjusted = {t: round(o) for t, o in sorted(offsets.items(),
                                                       key=lambda x: -x[1])}
            print(f"  iter {it + 1}: max step {max_move:.0f} Elo, offsets {adjusted}")
        if max_move < 3.0:
            break
    return offsets
