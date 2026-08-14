"""Build the static GitHub Pages site from the pipeline's output CSVs.

Writes site/index.html and copies the dashboard PNG alongside it. Prefers the
market-calibrated outputs and falls back to pure-Elo files, so a Polymarket
outage degrades the page rather than breaking the build.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pl_predict.config import OUT_DIR, PROJECT_ROOT  # noqa: E402
from pl_predict.history import MATCH_LOG, scoreboard  # noqa: E402

SITE_DIR = PROJECT_ROOT / "site"

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7;
  --home: #2a78d6; --draw: #eb6834; --away: #1baf7a;
  --border: rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
    --home: #3987e5; --draw: #d95926; --away: #199e70;
    --border: rgba(255,255,255,0.10);
  }
}
* { box-sizing: border-box; margin: 0; }
body {
  background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.5; padding: 2rem 1rem 4rem;
}
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 1.7rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.15rem; margin: 2.2rem 0 0.7rem; }
.sub { color: var(--ink-2); font-size: 0.92rem; }
.updated { color: var(--muted); font-size: 0.85rem; margin-top: 0.3rem; }
img.dash { width: 100%; height: auto; margin-top: 1.5rem; border: 1px solid var(--border); border-radius: 8px; background: #fcfcfb; }
.tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { padding: 0.42rem 0.7rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
th { color: var(--muted); font-weight: 600; font-size: 0.78rem; border-bottom: 1px solid var(--baseline); }
td.team, th.team { text-align: left; }
tbody tr:nth-child(even) { background: color-mix(in srgb, var(--grid) 28%, transparent); }
tr.cutline td { border-bottom: 1.5px dashed var(--baseline); }
tr.played td { color: var(--muted); }
.meter { display: inline-flex; width: 130px; height: 10px; border-radius: 3px; overflow: hidden; vertical-align: middle; gap: 1px; background: var(--surface); }
.meter span { height: 100%; }
.m-h { background: var(--home); } .m-d { background: var(--draw); } .m-a { background: var(--away); }
.legend { color: var(--ink-2); font-size: 0.8rem; margin: 0.4rem 0 0.8rem; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin: 0 0.25rem 0 0.9rem; }
footer { margin-top: 3rem; color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--grid); padding-top: 1rem; }
footer a { color: var(--ink-2); }
"""


def pct(v: float) -> str:
    return "–" if pd.isna(v) else f"{v * 100:.0f}%"


def projection_rows(table: pd.DataFrame) -> str:
    rows = []
    for pos, r in enumerate(table.itertuples(), start=1):
        cls = ' class="cutline"' if pos in (4, 17) else ""
        rows.append(
            f"<tr{cls}><td>{pos}</td><td class=team>{r.team}</td>"
            f"<td>{r.exp_points:.0f}</td><td>{r.exp_gd:+.0f}</td>"
            f"<td>{pct(r.p_champion)}</td><td>{pct(r.p_top4)}</td>"
            f"<td>{pct(r.p_relegation)}</td></tr>")
    return "\n".join(rows)


def fixture_rows(matches: pd.DataFrame) -> str:
    rows = []
    for r in matches.itertuples():
        when = pd.Timestamp(r.kickoff_utc).strftime("%a %d %b %H:%M")
        if r.finished and pd.notna(r.FTHG):
            result = f"{int(r.FTHG)}–{int(r.FTAG)}"
            rows.append(
                f"<tr class=played><td>{when}</td>"
                f"<td class=team>{r.HomeTeam} v {r.AwayTeam}</td>"
                f"<td colspan=4>played: {result}</td></tr>")
            continue
        meter = (f'<span class="meter">'
                 f'<span class=m-h style="width:{r.p_home * 100:.0f}%"></span>'
                 f'<span class=m-d style="width:{r.p_draw * 100:.0f}%"></span>'
                 f'<span class=m-a style="width:{r.p_away * 100:.0f}%"></span></span>')
        rows.append(
            f"<tr><td>{when}</td><td class=team>{r.HomeTeam} v {r.AwayTeam}</td>"
            f"<td>{pct(r.p_home)}</td><td>{pct(r.p_draw)}</td>"
            f"<td>{pct(r.p_away)}</td><td>{meter}</td></tr>")
    return "\n".join(rows)


def history_section() -> str:
    parts = []
    if (OUT_DIR / "history.png").exists():
        parts.append(
            '<img class="dash" src="history.png" alt="Weekly evolution of '
            'title and relegation probabilities">')

    if MATCH_LOG.exists():
        log = pd.read_csv(MATCH_LOG)
        played = log[log["outcome"].notna()].sort_values("kickoff_utc")
        board = scoreboard(log)
        if board:
            parts.append(
                f'<p class="sub">Frozen pre-match predictions scored on '
                f'{board["n"]} played matches: '
                f'<strong>{board["accuracy"]:.0%}</strong> correct picks · '
                f'log loss <strong>{board["log_loss"]:.3f}</strong> · '
                f'Brier <strong>{board["brier"]:.3f}</strong></p>')
        if not played.empty:
            last_gw = int(played["gameweek"].max())
            rows = []
            for r in played[played["gameweek"] == last_gw].itertuples():
                probs = {"H": r.p_home, "D": r.p_draw, "A": r.p_away}
                pick = max(probs, key=probs.get)
                hit = "✓" if pick == r.outcome else "✗"
                rows.append(
                    f"<tr><td class=team>{r.HomeTeam} v {r.AwayTeam}</td>"
                    f"<td>{pct(r.p_home)}</td><td>{pct(r.p_draw)}</td>"
                    f"<td>{pct(r.p_away)}</td>"
                    f"<td>{int(r.FTHG)}–{int(r.FTAG)}</td><td>{hit}</td></tr>")
            parts.append(f"""
  <h3 style="font-size:1rem;margin:1.2rem 0 0.5rem">Gameweek {last_gw} —
  predicted vs actual</h3>
  <div class="tablewrap"><table>
    <thead><tr><th class=team>fixture</th><th>home</th><th>draw</th>
    <th>away</th><th>result</th><th>pick</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>""")

    if not parts:
        return ""
    return ('\n  <h2>Prediction history</h2>\n  <p class="sub">Snapshots are '
            "taken weekly; match predictions freeze at kickoff, so the model "
            "is judged on what it said before the game.</p>\n  "
            + "\n".join(parts))


def main():
    def read_pref(preferred, fallback):
        p = OUT_DIR / preferred
        return pd.read_csv(p if p.exists() else OUT_DIR / fallback), p.exists()

    table, calibrated = read_pref("season_projection_market_implied.csv",
                                  "season_projection.csv")
    matches, _ = read_pref("match_probabilities_market_implied.csv",
                           "match_probabilities.csv")

    unfinished = matches[~matches["finished"]]
    gw = int(unfinished["gameweek"].min()) if not unfinished.empty else None
    gw_matches = (matches[matches["gameweek"] == gw]
                  .sort_values("kickoff_utc") if gw else matches.iloc[0:0])

    updated = pd.Timestamp.now(tz="UTC").strftime("%d %b %Y, %H:%M UTC")
    calib_note = ("model calibrated against the Polymarket title market"
                  if calibrated else
                  "pure-Elo model (market calibration unavailable this run)")

    fixtures_section = "" if gw is None else f"""
  <h2>Gameweek {gw} — outcome probabilities</h2>
  <div class="legend">
    <span class="dot m-h"></span>home win
    <span class="dot m-d"></span>draw
    <span class="dot m-a"></span>away win
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th>kickoff (UTC)</th><th class=team>fixture</th>
    <th>home</th><th>draw</th><th>away</th><th></th></tr></thead>
    <tbody>{fixture_rows(gw_matches)}</tbody>
  </table></div>"""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PL 2026-27 Predictions</title>
<style>{CSS}</style>
</head>
<body>
<main>
  <h1>Premier League 2026-27 — predictions</h1>
  <p class="sub">Elo + Poisson model on 26 seasons of free data · {calib_note}
  · 10,000 season simulations, played matches locked to their results</p>
  <p class="updated">Last updated {updated} · rebuilt daily</p>

  <img class="dash" src="predictions_2026_27.png"
       alt="Prediction dashboard: expected points, title and relegation probabilities, next-gameweek outcome probabilities">
{fixtures_section}

{history_section()}

  <h2>Projected final table</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>#</th><th class=team>team</th><th>xPts</th><th>xGD</th>
    <th>title</th><th>top 4</th><th>relegated</th></tr></thead>
    <tbody>{projection_rows(table)}</tbody>
  </table></div>

  <footer>
    Data: <a href="https://www.football-data.co.uk/englandm.php">football-data.co.uk</a> ·
    <a href="https://fantasy.premierleague.com/">FPL API</a> ·
    <a href="https://polymarket.com/">Polymarket</a> ·
    code on <a href="https://github.com/amozaffari/PL2026">GitHub</a>.<br>
    Probabilities are model output, not betting advice. The model cannot beat the market.
  </footer>
</main>
</body>
</html>"""

    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "index.html").write_text(html)
    for name in ("predictions_2026_27.png", "history.png"):
        if (OUT_DIR / name).exists():
            shutil.copy(OUT_DIR / name, SITE_DIR / name)
    print(f"wrote {SITE_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
