"""Render the prediction results as a single dashboard PNG.

Reads the CSVs produced by `simulate` / `simulate --market-implied` and the live
Polymarket title market. Output: output/predictions_2026_27.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pl_predict.config import OUT_DIR  # noqa: E402

# Reference palette (validated): categorical slots 1-3, blue ordinal pair, chrome.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#1c5cab"
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
    "axes.titlecolor": INK, "figure.facecolor": PAGE, "axes.facecolor": SURFACE,
})


def style(ax, xgrid=False):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _read_with_fallback(preferred: str, fallback: str) -> pd.DataFrame:
    """Use market-calibrated output when present, else the pure-Elo file —
    so a Polymarket outage degrades the dashboard instead of breaking it."""
    pref, fall = OUT_DIR / preferred, OUT_DIR / fallback
    use_pref = pref.exists() and (
        not fall.exists() or pref.stat().st_mtime >= fall.stat().st_mtime)
    return pd.read_csv(pref if use_pref else fall)


def main():
    elo = pd.read_csv(OUT_DIR / "season_projection.csv")
    mi = _read_with_fallback("season_projection_market_implied.csv",
                             "season_projection.csv")
    matches = _read_with_fallback("match_probabilities_market_implied.csv",
                                  "match_probabilities.csv")
    market_path = OUT_DIR / "title_market.csv"
    market = (pd.read_csv(market_path) if market_path.exists()
              else elo[["team"]].assign(p_title_market=float("nan")))

    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], hspace=0.45, wspace=0.28,
                          left=0.09, right=0.97, top=0.90, bottom=0.06)
    ax_pts = fig.add_subplot(gs[:, 0])
    ax_title = fig.add_subplot(gs[0, 1])
    ax_rel = fig.add_subplot(gs[1, 1])
    ax_gw1 = fig.add_subplot(gs[2, 1])

    fig.suptitle("Premier League 2026-27 — model predictions", x=0.09, y=0.98,
                 ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.09, 0.945, "Elo + Poisson model on 26 seasons of free data, "
             "calibrated against the Polymarket title market  ·  10,000 season "
             "simulations  ·  " + pd.Timestamp.today().strftime("%d %b %Y"),
             fontsize=9.5, color=INK_2)

    # --- Expected points: pure Elo -> market-calibrated (dumbbell) ---
    both = (elo[["team", "exp_points"]].rename(columns={"exp_points": "elo"})
            .merge(mi[["team", "exp_points"]].rename(columns={"exp_points": "mi"}))
            .sort_values("mi").reset_index(drop=True))
    y = range(len(both))
    ax_pts.hlines(y, both["elo"], both["mi"], color=BLUE_LIGHT, linewidth=1.6,
                  alpha=0.7, zorder=2)
    ax_pts.scatter(both["elo"], y, s=52, color=BLUE_LIGHT, zorder=3,
                   label="pure Elo")
    ax_pts.scatter(both["mi"], y, s=64, color=BLUE_DARK, zorder=4,
                   label="market-calibrated")
    ax_pts.set_yticks(list(y), both["team"], fontsize=9.5)
    ax_pts.set_xlabel("expected points", fontsize=9)
    style(ax_pts, xgrid=True)
    ax_pts.set_title("Expected points — what the Polymarket calibration changes",
                     loc="left", fontsize=11, fontweight="bold", pad=10)
    ax_pts.legend(loc="lower right", frameon=False, fontsize=9,
                  labelcolor=INK_2, handletextpad=0.2)
    ax_pts.margins(y=0.02)

    # --- Title probabilities: model vs market (grouped bars) ---
    tp = elo[["team", "p_champion"]].merge(market, on="team", how="left")
    has_market = tp["p_title_market"].notna().any()
    sort_col = "p_title_market" if has_market else "p_champion"
    tp = tp.sort_values(sort_col, ascending=False).head(7)[::-1]
    y = range(len(tp))
    h = 0.34 if has_market else 0.55
    off = h / 2 + 0.03 if has_market else 0.0
    ax_title.barh([i + off for i in y], tp["p_champion"], height=h,
                  color=BLUE, label="model (pure Elo)")
    if has_market:
        ax_title.barh([i - off for i in y], tp["p_title_market"], height=h,
                      color=ORANGE, label="Polymarket")
    for i, r in zip(y, tp.itertuples()):
        ax_title.text(r.p_champion + 0.008, i + off,
                      f"{r.p_champion:.0%}", va="center", fontsize=8, color=INK_2)
        if has_market:
            ax_title.text(r.p_title_market + 0.008, i - off,
                          f"{r.p_title_market:.0%}", va="center", fontsize=8,
                          color=INK_2)
    ax_title.set_yticks(list(y), tp["team"], fontsize=9.5)
    ax_title.set_xlim(0, 0.56)
    ax_title.set_xticks([])
    style(ax_title)
    ax_title.spines["bottom"].set_visible(False)
    ax_title.set_title("Title probability — model vs prediction market"
                       if has_market else "Title probability (model)",
                       loc="left", fontsize=11, fontweight="bold", pad=10)
    if has_market:
        ax_title.legend(loc="lower right", frameon=False, fontsize=9,
                        labelcolor=INK_2, handletextpad=0.2)

    # --- Relegation probabilities (market-calibrated) ---
    rel = mi.sort_values("p_relegation", ascending=False).head(8)[::-1]
    y = range(len(rel))
    ax_rel.barh(list(y), rel["p_relegation"], height=0.55, color=BLUE)
    for i, v in zip(y, rel["p_relegation"]):
        ax_rel.text(v + 0.012, i, f"{v:.0%}", va="center", fontsize=8, color=INK_2)
    ax_rel.set_yticks(list(y), rel["team"], fontsize=9.5)
    ax_rel.set_xlim(0, 0.85)
    ax_rel.set_xticks([])
    style(ax_rel)
    ax_rel.spines["bottom"].set_visible(False)
    ax_rel.set_title("Relegation probability (market-calibrated)",
                     loc="left", fontsize=11, fontweight="bold", pad=10)

    # --- Gameweek 1 outcome probabilities (100% stacked) ---
    gw1 = matches[matches["gameweek"] == 1].sort_values("kickoff_utc")[::-1]
    labels = [f"{r.HomeTeam} – {r.AwayTeam}" for r in gw1.itertuples()]
    y = range(len(gw1))
    left = pd.Series(0.0, index=gw1.index)
    for col, color, name in [("p_home", BLUE, "home win"),
                             ("p_draw", ORANGE, "draw"),
                             ("p_away", AQUA, "away win")]:
        vals = gw1[col]
        ax_gw1.barh(list(y), vals, left=left, height=0.62, color=color,
                    edgecolor=SURFACE, linewidth=1.6, label=name)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.10:
                txt_color = "#ffffff" if color == BLUE else INK
                ax_gw1.text(l + v / 2, i, f"{v:.0%}", va="center", ha="center",
                            fontsize=7.5, color=txt_color)
        left = left + vals
    ax_gw1.set_yticks(list(y), labels, fontsize=8.5)
    ax_gw1.set_xlim(0, 1)
    ax_gw1.set_xticks([])
    style(ax_gw1)
    ax_gw1.spines["bottom"].set_visible(False)
    ax_gw1.set_title("Gameweek 1 — outcome probabilities (market-calibrated)",
                     loc="left", fontsize=11, fontweight="bold", pad=10)
    ax_gw1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3,
                  frameon=False, fontsize=8.5, labelcolor=INK_2,
                  handletextpad=0.2, columnspacing=1.2)

    out = OUT_DIR / "predictions_2026_27.png"
    fig.savefig(out, dpi=150, facecolor=PAGE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
