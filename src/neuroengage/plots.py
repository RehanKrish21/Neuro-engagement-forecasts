"""Summary figure: brain features vs. baselines, side by side."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Fixed categorical order: baseline role, brain-only, combined.
ROLE_COLORS = {"baseline": "#2a78d6", "brain": "#eb6834", "combined": "#1baf7a"}
SET_ROLE = {
    "baseline": "baseline",
    "position": "baseline",
    "brain": "brain",
    "baseline+brain": "combined",
    "position+brain": "combined",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK_2, linewidth=0.8)


def _bars(ax, labels, means, stds, sets, y0=0.0, height=0.26, gap=0.03):
    ys = []
    for i, (m, s, name) in enumerate(zip(means, stds, sets)):
        y = y0 + i * (height + gap)
        ax.barh(y, m, height=height, color=ROLE_COLORS[SET_ROLE[name]],
                edgecolor=SURFACE, linewidth=2)
        ax.errorbar(m, y, xerr=s, fmt="none", ecolor=INK_2, elinewidth=1, capsize=0)
        ax.text(max(m + s, 0) + 0.02, y, f"{m:.2f}", va="center", fontsize=8.5, color=INK)
        ys.append(y)
    return ys


def plot_summary(
    video_summary: pd.DataFrame,
    drop_summary: pd.DataFrame | None,
    path: str | Path,
    title: str = "Results",
) -> Path:
    targets = list(dict.fromkeys(video_summary["target"]))
    sets = list(dict.fromkeys(video_summary["feature_set"]))
    if drop_summary is None:
        fig, a1 = plt.subplots(1, 1, figsize=(6.4, 1.6 + 1.0 * len(targets)), facecolor=SURFACE)
        a2 = None
    else:
        fig, (a1, a2) = plt.subplots(
            1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.4, 1]}, facecolor=SURFACE
        )
    group_h = len(sets) * 0.29 + 0.35
    yticks = []
    for gi, target in enumerate(targets):
        rows = video_summary[video_summary.target == target].set_index("feature_set").loc[sets]
        ys = _bars(a1, sets, rows["spearman_mean"], rows["spearman_std"].fillna(0), sets,
                   y0=gi * group_h)
        yticks.append(np.mean(ys))
    a1.set_yticks(yticks, [t.replace("_", " ") for t in targets])
    a1.invert_yaxis()
    _style(a1)
    if drop_summary is None:
        xlabel, panel = "Spearman ρ, held-out videos (mean ± sd over folds)", "Which videos hold attention?"
    else:
        xlabel, panel = ("Spearman ρ, held-out creators (mean ± sd over folds)",
                         "Which videos beat the creator's usual?")
    a1.set_xlabel(xlabel, color=INK_2, fontsize=9)
    a1.set_title(panel, loc="left", color=INK, fontsize=11)

    axes = [a1]
    if a2 is not None:
        axes.append(a2)
        _plot_dropoff(a2, drop_summary, len(targets) * group_h)

    hi = max(*(ax.get_xlim()[1] for ax in axes), 0.5) + 0.12
    for ax in axes:
        ax.set_xlim(min(ax.get_xlim()[0], -0.05), hi)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ROLE_COLORS.values()]
    base_label = "baseline (metadata / position)" if a2 is not None else "baseline (metadata)"
    fig.legend(handles, [base_label, "brain only", "baseline + brain"],
               loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    bottom = 0.07 if a2 is not None else 0.14
    fig.tight_layout(rect=(0, bottom, 1, 0.95))
    path = Path(path)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def _plot_dropoff(a2, drop_summary: pd.DataFrame, span: float) -> None:
    dsets = list(drop_summary["feature_set"])
    ys = _bars(a2, dsets, drop_summary["within_video_spearman_mean"],
               drop_summary["within_video_spearman_std"].fillna(0), dsets)
    a2.set_yticks(ys, dsets)
    # Match the left panel's vertical scale so bars keep the same thickness.
    mid = float(np.mean(ys))
    a2.set_ylim(mid + span / 2, mid - span / 2)
    _style(a2)
    a2.set_xlabel("Within-video Spearman ρ, held-out videos", color=INK_2, fontsize=9)
    a2.set_title("Where do viewers drop off?", loc="left", color=INK, fontsize=11)

