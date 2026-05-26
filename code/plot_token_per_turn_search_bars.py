#!/usr/bin/env python3
"""Publication-quality grouped bar chart (with bar-top trend lines) comparing
WebExplorer-8B vs ContextPilot-8B per-turn average token counts.

For every turn we draw two side-by-side bars (one per model) and overlay a
polyline that connects the *top centre* of consecutive bars belonging to the
same model. This makes both the per-turn comparison (bar heights) and the
overall trend (line slope) immediately readable.

Data source: ``analysis/token_count/results/token_count_stats_search.md``
(the ``WebExplorer-8B_all-tool_step176`` and
``ContextPilot-8B-NoThinking_AllTools__fsm_ctx`` blocks).

Per-dataset turn-axis end is clipped to ``min(len(model_a), len(model_b))``
so both bar groups cover exactly the same span.

Style mirrors:
    - analysis/tool-call_statics/plot_training_curve.py
    - analysis/tool_errors/plot_failure_curve_SFT-RL.py
    - analysis/token_count/plot_token_per_turn_search.py
(serif font, Okabe-Ito palette, no titles, vector-friendly PDF + 300-dpi PNG).

Output:
    analysis/token_count/figures_per_turn_search_bars/

Usage:
    python3 plot_token_per_turn_search_bars.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "figures_per_turn_search_bars"

# ---------------------------------------------------------------------------
# Data (transcribed from results/token_count_stats_search.md, same as
# plot_token_per_turn_search.py).
# ---------------------------------------------------------------------------
DATA: Dict[str, Dict[str, List[float]]] = {
    "browsecomp_sample200": {
        "WebExplorer-8B": [
            513.6, 3845.1, 7162.8, 10293.5, 11524.3, 13901.5, 15823.5,
            17539.3, 19559.3, 21748.9, 23438.0, 26159.8, 28259.2, 30642.8,
            31803.2, 36225.8, 37416.4,
        ],
        "ContextPilot-8B": [
            646.5, 4415.3, 8096.8, 8928.0, 5652.4, 8118.1, 9028.0, 8936.1,
            9834.5, 9144.6, 8865.0, 8746.5, 8469.6, 8290.2, 8378.1,
        ],
    },
    "browsecomp_zh": {
        "WebExplorer-8B": [
            448.7, 2582.0, 4590.4, 6397.0, 8674.0, 11030.1, 13103.4, 15058.1,
            17374.7, 19741.1, 21957.6, 24293.1, 26477.3, 28536.3, 30711.2,
            31510.1, 34830.7, 36946.6, 39094.4, 41112.2, 43370.8, 45389.8,
            47485.7,
        ],
        "ContextPilot-8B": [
            518.0, 2909.7, 5636.4, 7623.2, 6337.8, 9341.8, 10161.7, 8346.6,
            9723.8, 9719.8, 9584.0, 9608.4, 9547.5, 9560.9, 9078.9,
        ],
    },
}

DATASET_PRETTY: Dict[str, str] = {
    "browsecomp_sample200": "BrowseComp",
    "browsecomp_zh": "BrowseComp-ZH",
}

# Order in which to draw the two models (controls bar ordering and legend).
MODEL_ORDER: List[str] = ["WebExplorer-8B", "ContextPilot-8B"]

# Color-blind friendly Okabe-Ito palette, aligned with the line-plot script.
MODEL_STYLE: Dict[str, Dict[str, object]] = {
    "WebExplorer-8B":   {"color": "#0072B2", "marker": "o"},
    "ContextPilot-8B": {"color": "#D55E00", "marker": "s"},
}


# ---------------------------------------------------------------------------
# Style (mirrors plot_training_curve.py / plot_failure_curve_SFT-RL.py)
# ---------------------------------------------------------------------------

def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "DejaVu Serif", "Times", "STIXGeneral",
        ],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        # Lines & markers.
        "lines.linewidth": 1.8,
        "lines.markersize": 3.2,
        "lines.markeredgewidth": 0.7,
        # Axes & grid.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "axes.axisbelow": True,
        # Legend.
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#666666",
        "legend.fancybox": False,
        # Saving.
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_k(v: float, _pos: int) -> str:
    """Y-tick formatter: print kilo-tokens (e.g. '12k') for compactness."""
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        kv = v / 1000.0
        if abs(kv - round(kv)) < 1e-9:
            return f"{int(round(kv))}k"
        return f"{kv:.1f}k"
    return f"{v:.0f}"


def _slugify(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Layout constants for the grouped-bar chart. Each "group" lives at integer
# x position k (k = 1..n_common, matching the turn index). Within a group,
# the two bars are placed at x = k - BAR_OFFSET and x = k + BAR_OFFSET so
# the group is centered on the integer tick. ``BAR_WIDTH`` is slightly
# narrower than ``BAR_OFFSET * 2`` so adjacent bars don't visually touch.
BAR_WIDTH = 0.38
BAR_OFFSET = BAR_WIDTH / 2.0  # half of the bar width => centers separated by BAR_WIDTH


def plot_one(dataset: str, series_by_model: Dict[str, List[float]]) -> Path:
    """Plot a single dataset as grouped bars + bar-top trend lines."""
    n_common = min(len(series_by_model[m]) for m in MODEL_ORDER)
    turn_idx = np.arange(1, n_common + 1)  # group centers (T1..Tn)

    fig, ax = plt.subplots(figsize=(7.4, 3.2))

    # Per-model bar x-positions: WebExplorer on the left of each group,
    # ContextPilot on the right. Computed from MODEL_ORDER so adding more
    # models would just need a wider offset spec.
    bar_offsets = {
        MODEL_ORDER[0]: -BAR_OFFSET,
        MODEL_ORDER[1]: +BAR_OFFSET,
    }

    bar_handles = []  # for legend (one bar handle per model)
    line_handles = []  # for legend (one line handle per model)

    for model in MODEL_ORDER:
        ys = np.asarray(series_by_model[model][:n_common], dtype=float)
        xs = turn_idx + bar_offsets[model]
        style = MODEL_STYLE[model]

        # Bars: filled with the model's colour at moderate alpha so the
        # overlaid trend line stays clearly readable on top.
        bars = ax.bar(
            xs,
            ys,
            width=BAR_WIDTH,
            color=style["color"],
            edgecolor=style["color"],
            linewidth=0.7,
            alpha=0.55,
            label=model,
            zorder=2,
        )
        bar_handles.append(bars)

        # Trend line connecting bar tops: same colour as the bars but fully
        # opaque, with the same Okabe-Ito marker the line-plot variant uses
        # so the two scripts read as a matched pair.
        (line,) = ax.plot(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=1.8,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.0,
            markersize=5.0,
            alpha=0.95,
            zorder=4,
            label=f"{model} (trend)",
        )
        line_handles.append(line)

    # X axis: integer turn ticks centered on each group.
    ax.set_xlabel("Interaction Turn Index")
    ax.set_xticks(turn_idx)
    # If there are many turns, thin the labels so they don't overlap.
    if n_common > 20:
        # Show every other label.
        labels = [str(t) if (t % 2 == 1) else "" for t in turn_idx]
        ax.set_xticklabels(labels)
    # Tight x-limits with a small margin so end bars don't kiss the spines.
    ax.set_xlim(turn_idx[0] - 0.6, turn_idx[-1] + 0.6)

    # Y axis: leave 12% headroom above the observed max; format as
    # kilo-tokens, matching plot_token_per_turn_search.py.
    max_y = max(
        max(series_by_model[m][:n_common]) for m in MODEL_ORDER
    )
    ax.set_ylim(0, max_y * 1.12 if max_y > 0 else 1.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_format_k))
    ax.set_ylabel("Input Tokens")

    # Keep gridlines on the y-axis only -- a vertical grid behind the bars
    # adds visual noise without aiding readability.
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(False)

    # Legend: one entry per model. We use the line handles (which carry the
    # marker) because they convey both bar colour and trend marker in a
    # single glyph, keeping the legend compact.
    leg = ax.legend(
        line_handles,
        [h.get_label().replace(" (trend)", "") for h in line_handles],
        loc="upper left",
        ncol=1,
        fontsize=14,
        frameon=True,
        framealpha=0.85,
        edgecolor="#999999",
        fancybox=False,
        borderpad=0.4,
        labelspacing=0.32,
        handlelength=1.7,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"token_per_turn_bars__{_slugify(DATASET_PRETTY[dataset])}"
    pdf_path = PLOTS_DIR / f"{stem}.pdf"
    png_path = PLOTS_DIR / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def plot_all() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, series_by_model in DATA.items():
        out_path = plot_one(dataset, series_by_model)
        print(f"[INFO] plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()
    plot_all()
    print(f"[INFO] Plots written under: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
