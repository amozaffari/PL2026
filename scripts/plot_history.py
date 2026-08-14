"""Render the prediction-history chart: how title and relegation probabilities
have moved across weekly snapshots. Output: output/history.png (skipped
gracefully while there are no snapshots yet)."""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pl_predict.config import OUT_DIR  # noqa: E402
from pl_predict.history import load_projection_history  # noqa: E402

# Validated categorical order (slots 1-5) + chrome, from the reference palette.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": PAGE, "axes.facecolor": SURFACE,
})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_series(ax, hist, prob_col, teams):
    for color, team in zip(SERIES, teams):
        rows = hist[hist["team"] == team].sort_values("date")
        ax.plot(rows["date"], rows[prob_col], color=color, linewidth=2,
                marker="o", markersize=5.5, label=team)
    ax.set_ylim(bottom=0)
    ax.set_xlim(pd.Timestamp("2026-08-01"), pd.Timestamp("2027-06-10"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_2,
              handletextpad=0.4, ncol=2, columnspacing=1.0)


def main():
    hist = load_projection_history()
    if hist.empty:
        print("no snapshots yet — skipping history chart")
        return
    hist["date"] = pd.to_datetime(hist["date"])
    latest = hist[hist["date"] == hist["date"].max()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.12, wspace=0.18)
    fig.suptitle("How the predictions have moved (weekly snapshots)",
                 x=0.06, y=0.95, ha="left", fontsize=13, fontweight="bold")

    top = latest.nlargest(5, "p_champion")["team"]
    plot_series(ax1, hist, "p_champion", top)
    ax1.set_title("Title probability", loc="left", fontsize=10.5,
                  fontweight="bold", pad=8)

    bottom = latest.nlargest(5, "p_relegation")["team"]
    plot_series(ax2, hist, "p_relegation", bottom)
    ax2.set_title("Relegation probability", loc="left", fontsize=10.5,
                  fontweight="bold", pad=8)

    out = OUT_DIR / "history.png"
    fig.savefig(out, dpi=150, facecolor=PAGE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
