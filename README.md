# PL2026 — Premier League match prediction

**Live site:** <https://amozaffari.github.io/PL2026/> — rebuilt daily at 05:17
UTC by [GitHub Actions](.github/workflows/update.yml): fresh results and odds
are fetched, Elo is updated, the season is re-simulated (played matches locked
to their actual results), the model is re-calibrated against Polymarket, and
the dashboard + site are redeployed. One-time setup: repo Settings → Pages →
Source → **GitHub Actions**.

Predicts Premier League match outcomes and simulates the 2026-27 season, built
entirely on free, keyless data sources.

## Data sources (all free, no API key)

| Source | What it provides | Used for |
| --- | --- | --- |
| [football-data.co.uk](https://www.football-data.co.uk/englandm.php) | CSVs of every PL + Championship match since 2000, incl. closing odds from ~10 bookmakers | Training data, odds baseline |
| [FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) | Official 2026-27 team list and all 380 fixtures with kickoff times | Season simulation, upcoming predictions |
| [Open-Meteo](https://open-meteo.com/) | Hourly weather forecast at any coordinates | Kickoff weather at the home stadium (context only) |
| [Polymarket Gamma API](https://gamma-api.polymarket.com/) | Real-money prediction-market prices: season champion + per-match H/D/A markets | `markets` command: market probabilities, model-vs-market comparison, blend |
| [vaastav FPL dataset](https://github.com/vaastav/Fantasy-Premier-League) | Last season's FPL rosters (players keyed by stable code) | `transfers` command: roster diff reveals real transfer-window moves |

Optional upgrades that need a (free-tier) API key, not wired in: [football-data.org](https://www.football-data.org/)
(live fixtures/standings), [The Odds API](https://the-odds-api.com/) (live pre-match odds
— would let the blend model run on future matches), Understat (shot/xG data, scraping).

## How it works

1. **Elo ratings** are maintained across both the Premier League and the
   Championship (goal-margin weighted, home advantage, 25% regression to the
   mean each summer). Running Elo across both divisions means promoted clubs
   arrive with a rating earned from real matches.
2. **Expected goals**: two Poisson GLMs (home and away goals) map the pre-match
   Elo difference to scoring rates, fit with exponential time-decay weights.
3. **Match probabilities**: independent-Poisson score matrix with the
   Dixon-Coles low-score correction (rho fit by grid search).
4. **Odds baseline / blend**: bookmaker closing odds are de-vigged into fair
   probabilities; the backtest compares model vs. odds vs. a 50/50 blend.
5. **Season simulation**: 10,000 Monte Carlo runs of all 380 fixtures, sampling
   full scorelines so goal difference breaks ties.

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pl_predict fetch              # download 26 seasons of data
.venv/bin/python -m pl_predict backtest           # walk-forward eval, last 3 seasons
.venv/bin/python -m pl_predict simulate           # 2026-27 Monte Carlo -> output/*.csv
.venv/bin/python -m pl_predict predict            # next gameweek probs + weather
.venv/bin/python -m pl_predict predict --gameweek 5
.venv/bin/python -m pl_predict markets            # model vs Polymarket (title + gameweek)
.venv/bin/python -m pl_predict transfers          # live transfer-window activity per club
.venv/bin/python -m pl_predict simulate --market-implied   # Elo calibrated to Polymarket
ODDS_API_KEY=... .venv/bin/python -m pl_predict predict    # adds live bookmaker odds + blend
.venv/bin/python -m pl_predict simulate --squad-adjust     # transfer-window Elo offsets
.venv/bin/python -m pl_predict history                     # snapshot + match-prediction log
```

**Player-level squad model** (`pl_predict/squad_model.py`, `squad` command).
Team strength is built bottom-up from the full current roster: each player is
valued at last season's FPL points; players with no PL history (foreign
signings, promoted-club squads) are valued from their FPL price via a
per-position regression; in-season, values blend 30% current form. Team
strength is the top-15 sum, and the points→Elo conversion is *calibrated* by
regressing end-of-season Elo on squad points across 140 historical
team-seasons (R² ≈ 0.83) — not an invented scale. The resulting rating
*shrinks* each club's Elo 40% toward its squad-implied level (capped ±75)
rather than adding on top, since Elo already measures strength. `predict`
applies the match horizon (availability/chance-of-playing discounts) by
default (`--raw` disables); `simulate --squad-adjust` applies the season
horizon as a market-free alternative to `--market-implied` (never stack the
two). Known limits: promoted squads are undervalued (price-only estimates,
hence the cap) and FPL points are an attack-tilted quality proxy.

**Prediction history.** `history` writes a weekly snapshot of the projected
table to `history/projections/` and maintains `history/match_predictions.csv`,
where each fixture's probabilities update until kickoff and then freeze — the
model is judged on what it said before the game. The daily CI run commits
`history/` back to the repo and the site shows the probability evolution, last
gameweek's predictions vs. results, and a running scoreboard (accuracy, log
loss, Brier).

`--market-implied` iteratively nudges team Elo ratings until simulated title
odds match the Polymarket champion market, then re-simulates all 380 fixtures —
propagating the market's squad-level knowledge (transfers, injuries, managers)
to every match and to the relegation picture. Only teams with >=1% title
probability on either side get calibrated; the title market says nothing
reliable about the rest.

Live bookmaker odds need the one key-based source: a free key from
[The Odds API](https://the-odds-api.com/) (500 requests/month) exported as
`ODDS_API_KEY`. `predict` then shows de-vigged bookmaker probabilities next to
the model's and the 50/50 blend that won the backtest.

Outputs land in `output/`: `season_projection.csv` (title/top-4/relegation
probabilities, expected points), `match_probabilities.csv` (H/D/A probabilities
and xG for all 380 fixtures), `backtest.csv`.

## Backtest results (2023-24 → 2025-26, walk-forward)

| Model | Log loss | Brier | Accuracy |
| --- | --- | --- | --- |
| Bookmaker odds (de-vigged) | 0.960 | 0.570 | 55.0% |
| 50/50 blend | 0.973 | 0.579 | 53.5% |
| Elo → Poisson (this model) | 0.997 | 0.596 | 51.8% |

The market is (as expected) the strongest predictor — decades of research say
beating closing odds without inside information is close to impossible. The
value of the model is that it predicts *any* future fixture months ahead,
before odds exist, and its gap to the market (~0.04 log loss) is in line with
published academic models. When pre-match odds are available, blending moves
you toward the market. Weather is attached to predictions as context; its
measurable effect on outcomes is negligible, which is why it stays out of the
model.

## Honest limitations

- No squad-level information inside the model: transfers, injuries, managerial
  changes, and European-competition fatigue are invisible to Elo until results
  reflect them. The `markets` and `transfers` commands surface these signals
  (prediction markets price them in; the roster diff shows the moves), but the
  transfer signal counts arrivals from abroad as 0 points, and neither is fed
  back into the Elo automatically.
- Expected points from simulation are compressed relative to a realized table
  (the eventual champion usually overperforms its pre-season expectation).
- The model cannot beat the betting market; treat outputs as calibrated
  probabilities, not an edge.
