"""Command-line entry points: fetch, backtest, simulate, predict."""

import argparse

import pandas as pd

from .config import OUT_DIR


def cmd_fetch(args):
    from .data_sources.football_data import fetch_all

    fetch_all(force=args.force)
    print("done.")


def cmd_backtest(args):
    from .backtest import run

    results = run(n_eval_seasons=args.seasons)
    OUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUT_DIR / "backtest.csv", index=False)
    print(results.to_string(index=False))
    print("\noverall (mean across seasons):")
    print(results.groupby("model")[["log_loss", "brier", "accuracy"]]
          .mean().round(4).to_string())


def cmd_simulate(args):
    from .simulate import build_context, simulate_season

    if args.market_implied and args.squad_adjust:
        raise SystemExit("--market-implied and --squad-adjust must not be combined: "
                         "the market already prices transfers in.")
    ctx = build_context()
    offsets = None
    suffix = ""
    if args.market_implied:
        from .market_calibration import market_implied_offsets

        print("calibrating Elo offsets against the Polymarket title market...")
        offsets = market_implied_offsets(ctx)
        suffix = "_market_implied"
    elif args.squad_adjust:
        from .data_sources.squads import squad_elo_offsets

        offsets = squad_elo_offsets()
        suffix = "_squad_adjusted"
        shown = {t: round(o) for t, o in sorted(offsets.items(), key=lambda x: -x[1])
                 if abs(o) >= 1}
        print(f"transfer-window Elo offsets: {shown}")

    match_probs, table = simulate_season(n_sims=args.sims, ctx=ctx,
                                         elo_offsets=offsets)
    OUT_DIR.mkdir(exist_ok=True)
    match_probs.to_csv(OUT_DIR / f"match_probabilities{suffix}.csv", index=False)
    table.to_csv(OUT_DIR / f"season_projection{suffix}.csv", index=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(table.to_string(index=False))
    print(f"\nwrote {OUT_DIR / f'season_projection{suffix}.csv'} "
          f"and match_probabilities{suffix}.csv")


def cmd_predict(args):
    from .data_sources.weather import kickoff_weather
    from .simulate import match_probabilities

    offsets = None
    if not args.raw:
        from .data_sources.squads import injury_elo_penalties, squad_elo_offsets

        transfers = squad_elo_offsets()
        injuries = injury_elo_penalties()
        offsets = {t: transfers.get(t, 0.0) + injuries.get(t, 0.0)
                   for t in set(transfers) | set(injuries)}
        notable = {t: round(o) for t, o in sorted(offsets.items(), key=lambda x: x[1])
                   if abs(o) >= 10}
        print(f"squad adjustments applied (transfers + current injuries), "
              f"largest: {notable}  [--raw to disable]")
    fixtures, _, _ = match_probabilities(elo_offsets=offsets)
    upcoming = fixtures[~fixtures["finished"]]
    gw = args.gameweek or int(upcoming["gameweek"].min())
    sel = upcoming[upcoming["gameweek"] == gw].copy()

    weather = [kickoff_weather(r.HomeTeam, r.kickoff_utc) for r in sel.itertuples()]
    for k in ("temp_c", "precip_mm", "wind_kmh"):
        sel[k] = [w.get(k) for w in weather]

    cols = ["kickoff_utc", "HomeTeam", "AwayTeam", "p_home", "p_draw", "p_away",
            "xG_home", "xG_away", "temp_c", "precip_mm", "wind_kmh"]

    from .data_sources.odds_api import fetch_h2h

    book = fetch_h2h()
    if book is None:
        print("(no ODDS_API_KEY set — export one from https://the-odds-api.com/ "
              "to add live bookmaker odds and the model/odds blend)\n")
    elif not book.empty:
        sel = sel.merge(book, on=["HomeTeam", "AwayTeam"], how="left")
        for side, model_col in [("H", "p_home"), ("D", "p_draw"), ("A", "p_away")]:
            sel[f"blend_{side}"] = (sel[f"book_{side}"] + sel[model_col]) / 2
        cols += ["book_H", "book_D", "book_A", "blend_H", "blend_D", "blend_A"]
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(f"Gameweek {gw}\n" + sel[cols].to_string(index=False))


def cmd_history(args):
    from .history import scoreboard, snapshot_projection, update_match_log

    took = snapshot_projection()
    print("projection snapshot:", "written" if took else "already have one this week")
    log = update_match_log()
    frozen = log[log["outcome"].notna()]
    print(f"match log: {len(log)} fixtures tracked, {len(frozen)} with results")
    board = scoreboard(log)
    if board:
        print(f"scoreboard: n={board['n']} accuracy={board['accuracy']:.3f} "
              f"log_loss={board['log_loss']:.3f} brier={board['brier']:.3f}")


def cmd_markets(args):
    from .data_sources.polymarket import match_probs, title_probs
    from .simulate import simulate_season

    print("fetching Polymarket prices and running season simulation...")
    _, table = simulate_season(n_sims=args.sims)
    title = table[["team", "p_champion"]].merge(title_probs(), on="team", how="left")
    title["edge"] = title["p_champion"] - title["p_title_market"]
    title = title.sort_values("p_title_market", ascending=False)
    print("\nTitle probabilities — model vs Polymarket:")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(title.head(12).to_string(index=False))

    from .simulate import match_probabilities

    fixtures, _, _ = match_probabilities()
    upcoming = fixtures[~fixtures["finished"]]
    gw = args.gameweek or int(upcoming["gameweek"].min())
    sel = upcoming[upcoming["gameweek"] == gw].copy()
    rows = []
    for r in sel.itertuples():
        pm = match_probs(r.HomeTeam, r.AwayTeam, r.kickoff_utc)
        row = {"fixture": f"{r.HomeTeam} v {r.AwayTeam}",
               "model_H": r.p_home, "model_D": r.p_draw, "model_A": r.p_away}
        if pm:
            row |= {"mkt_H": pm["pm_home"], "mkt_D": pm["pm_draw"], "mkt_A": pm["pm_away"],
                    "blend_H": (r.p_home + pm["pm_home"]) / 2,
                    "blend_A": (r.p_away + pm["pm_away"]) / 2}
        rows.append(row)
    print(f"\nGameweek {gw} — model vs Polymarket (de-vigged):")
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(pd.DataFrame(rows).to_string(index=False))


def cmd_transfers(args):
    from .data_sources.squads import transfer_activity

    summary, moves = transfer_activity()
    print("Transfer window by club (quality proxied by last-season FPL points;")
    print("arrivals from abroad count 0, so buying clubs are understated):\n")
    print(summary.to_string(index=False))
    print("\nBiggest intra-league moves and departures:")
    named = moves[moves["last_season_pts"] > 0].head(args.top)
    print(named.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(prog="pl_predict",
                                     description="Premier League match prediction pipeline")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("fetch", help="download historical results + odds")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("backtest", help="walk-forward evaluation vs bookmaker odds")
    p.add_argument("--seasons", type=int, default=3)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("simulate", help="Monte Carlo simulation of the upcoming season")
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--market-implied", action="store_true",
                   help="calibrate Elo against the Polymarket title market first")
    p.add_argument("--squad-adjust", action="store_true",
                   help="apply transfer-window Elo offsets (market-free alternative)")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("predict", help="probabilities + weather for a gameweek")
    p.add_argument("--gameweek", type=int, default=None)
    p.add_argument("--raw", action="store_true",
                   help="skip transfer/injury squad adjustments")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("history", help="snapshot projections and update the match log")
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("markets", help="compare model vs Polymarket prediction markets")
    p.add_argument("--gameweek", type=int, default=None)
    p.add_argument("--sims", type=int, default=10_000)
    p.set_defaults(func=cmd_markets)

    p = sub.add_parser("transfers", help="current transfer-window activity per club")
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=cmd_transfers)

    args = parser.parse_args()
    args.func(args)
