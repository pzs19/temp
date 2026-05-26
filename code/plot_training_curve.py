#!/usr/bin/env python3
"""Publication-quality line plots for the interpolated tool-call training curve.

Reads `training_curve.json` produced by `interpolate_training_curve.py` and
plots, for every dataset, the average tool-call count *per tool category* over
the 64 interpolated training steps.

For every step we normalise each category by the TOTAL tool calls at that
step, so each curve represents the *share* (proportion) of that category
rather than its absolute count. The six curves at any x sum to 1.

Outputs (under `figures/`):
    <dataset>__categories.{pdf,png}   -- one figure per dataset (4 in total).

Style follows the same publication-friendly conventions as
`plot_ours_per_turn.py`: serif font, Okabe-Ito-style palette, thin gridlines,
vector-friendly PDF + 300-dpi PNG.

Usage:
    python3 plot_training_curve.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_JSON = RESULTS_DIR / "training_curve.json"
DEFAULT_OUT_DIR = SCRIPT_DIR / "figures"

# Per-variant defaults: which curve JSON to plot, where to write the figures,
# and what suffix to add to the output stem so the two model variants don't
# overwrite each other.
# Per-variant legend location override.
#
# Most variants behave like 8B/E4B/RL where Information Retrieval drops
# quickly into the 30-40%% band, so an 'upper right' legend doesn't
# clash with any line. The 8B-SFT run, however, keeps Information
# Retrieval in the 50-70%% band for the entire training trajectory, so
# the default 'upper right' legend ends up sitting on top of that line.
# We therefore park the legend on the right-hand vertical mid-band for
# 8B-SFT, which is the area cleanest of curves in those plots.
VARIANT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "8b": {
        "json": RESULTS_DIR / "training_curve.json",
        "out_dir": SCRIPT_DIR / "figures" / "8B",
        "stem_suffix": "",
        "legend_loc": "upper right",
    },
    "e4b": {
        "json": RESULTS_DIR / "training_curve_e4b.json",
        "out_dir": SCRIPT_DIR / "figures" / "E4B",
        "stem_suffix": "",
        "legend_loc": "upper right",
    },
    "8b-sft": {
        "json": RESULTS_DIR / "training_curve_sft.json",
        "out_dir": SCRIPT_DIR / "figures" / "8B-SFT",
        "stem_suffix": "",
        # 8B-SFT keeps Information Retrieval high (40-70%%) for the
        # whole run on every dataset. On LongMemEval the in-axes layout
        # is especially crowded -- IR ~40%%, CO ~25%%, Planning ~13%%,
        # plus stacked Mem./Other lines at 5-10%% leave no clean band
        # for an in-axes legend. We therefore park the legend OUTSIDE
        # the axes on the right for this variant.
        "legend_loc": "center left",
        "legend_outside": True,
    },
    "8b-rl": {
        "json": RESULTS_DIR / "training_curve_rl.json",
        "out_dir": SCRIPT_DIR / "figures" / "8B-RL",
        "stem_suffix": "",
        "legend_loc": "upper right",
    },
    "e4b-sft": {
        "json": RESULTS_DIR / "training_curve_e4b_sft.json",
        "out_dir": SCRIPT_DIR / "figures" / "E4B-SFT",
        "stem_suffix": "",
        "legend_loc": "upper right",
    },
    "e4b-rl": {
        "json": RESULTS_DIR / "training_curve_e4b_rl.json",
        "out_dir": SCRIPT_DIR / "figures" / "E4B-RL",
        "stem_suffix": "",
        "legend_loc": "upper right",
    },
}

DATASETS: List[str] = [
    "NovelQA",
    "InfBench-longbook_choice_eng",
    "LongMemEval-s_cleaned",
    "BrowseComp_Plus-decrypted",
]

# Pretty names for titles.
BENCH_PRETTY: Dict[str, str] = {
    "NovelQA": "NovelQA",
    "InfBench-longbook_choice_eng": "InfBench (LongBook Choice, EN)",
    "LongMemEval-s_cleaned": "LongMemEval-S",
    "BrowseComp_Plus-decrypted": "BrowseComp-Plus",
}

CATEGORIES: List[str] = [
    "Planning & Perception",
    "Information Retrieval",
    "Context Offloading",
    "Short-Term Memory",
    "Long-Term Memory",
    "Other",
]

# Color-blind friendly palette (Okabe-Ito + one neutral grey for "Other").
CATEGORY_STYLE: Dict[str, Dict[str, Any]] = {
    "Planning & Perception": {
        "color": "#0072B2",  # blue
        "marker": "o",
        "linestyle": "-",
    },
    "Information Retrieval": {
        "color": "#D55E00",  # vermillion
        "marker": "s",
        "linestyle": "-",
    },
    "Context Offloading": {
        "color": "#009E73",  # bluish green
        "marker": "^",
        "linestyle": "-",
    },
    "Short-Term Memory": {
        "color": "#CC79A7",  # reddish purple
        "marker": "D",
        "linestyle": "-",
    },
    "Long-Term Memory": {
        "color": "#E69F00",  # orange
        "marker": "v",
        "linestyle": "-",
    },
    "Other": {
        "color": "#666666",  # neutral grey -- de-emphasised
        "marker": "x",
        "linestyle": "--",
    },
}


# ---------------------------------------------------------------------------
# Plot styling (matches plot_ours_per_turn.py)
# ---------------------------------------------------------------------------

def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "DejaVu Serif", "Times", "STIXGeneral",
        ],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
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

def slugify(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def load_curve(path: Path) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Load `training_curve.json` and convert string step keys back to int."""
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): v for k, v in raw.items()}


def build_series(
    curve: Dict[int, Dict[str, Dict[str, float]]],
    dataset: str,
    steps: List[int],
) -> Dict[str, List[float]]:
    """Return {category: [share at each step]} restricted to the given steps.

    Each value is the category's count divided by the TOTAL across all
    categories at the same step, i.e. the proportion of tool calls that fall
    into this category. The six categories at any step sum to 1.
    """
    out: Dict[str, List[float]] = {cat: [] for cat in CATEGORIES}
    for s in steps:
        per_cat = curve[s]
        total = sum(per_cat[cat][dataset] for cat in CATEGORIES)
        # Guard against (vanishingly unlikely) division-by-zero.
        denom = total if total > 0 else 1.0
        for cat in CATEGORIES:
            out[cat].append(per_cat[cat][dataset] / denom)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_single(
    ax: plt.Axes,
    xs: List[int],
    series: Dict[str, List[float]],
    *,
    show_ylabel: bool,
    show_legend: bool,
    legend_loc: str = "upper right",
    legend_outside: bool = False,
) -> List[Any]:
    """Draw one panel; return the line handles for an external legend."""
    handles: List[Any] = []
    # Plot "Other" first so it sits behind the meaningful categories.
    plot_order = ["Other"] + [c for c in CATEGORIES if c != "Other"]
    for cat in plot_order:
        ys = series[cat]
        style = CATEGORY_STYLE[cat]
        # Slightly transparent line + opaque markers => clean & readable.
        (line,) = ax.plot(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markevery=1,           # sparse curve (~14 points): mark each one
            alpha=0.95 if cat != "Other" else 0.7,
            label=cat,
            zorder=3 if cat != "Other" else 2,
        )
        handles.append(line)

    ax.set_xlabel("Training step")
    if show_ylabel:
        ax.set_ylabel("Percentage")

    # X axis: show ~9 integer ticks across the full step range (0..LAST_STEP).
    # Margin scales with the run length so endpoints don't kiss the spines.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    x_margin = max(1.0, (max(xs) - min(xs)) * 0.02)
    ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)

    # Y axis: shares are in [0, 1]. Leave 8% headroom above the observed max
    # so the top line doesn't kiss the spine, and format ticks as percentages.
    max_y = max(max(v) for v in series.values()) if series else 1.0
    ax.set_ylim(0, min(1.0, max_y * 1.12) if max_y > 0 else 1.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _pos: f"{v * 100:.0f}%")
    )

    if show_legend:
        # Re-order so legend reads in the canonical CATEGORIES order.
        by_label = {h.get_label(): h for h in handles}
        ordered = [by_label[c] for c in CATEGORIES if c in by_label]
        # Display labels apply LEGEND_LABEL_OVERRIDES (e.g. "Memory" -> "Mem.")
        # so the in-figure legend stays compact without renaming the actual
        # line labels (which remain useful for downstream consumers).
        display_labels = [
            LEGEND_LABEL_OVERRIDES.get(h.get_label(), h.get_label())
            for h in ordered
        ]
        if legend_outside:
            # Park the legend OUTSIDE the axes on the right. This is used
            # for variants where every in-axes location collides with at
            # least one curve (e.g. 8B-SFT on LongMemEval has Information
            # Retrieval near 40%%, Context Offloading near 25%%, Planning
            # near 13%%, and Other / Mem. lines stacked at 5-10%% --
            # leaving no clean horizontal band in the figure).
            leg = ax.legend(
                ordered,
                display_labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                ncol=1,
                fontsize=12,
                frameon=True,
                framealpha=0.9,
                edgecolor="#999999",
                fancybox=False,
                borderaxespad=0.0,
                borderpad=0.4,
                labelspacing=0.4,
                handlelength=1.7,
                handletextpad=0.5,
            )
        else:
            # Place a compact legend *inside* the axes. The default
            # location (upper right) works for variants where
            # Information Retrieval quickly drops into the lower band,
            # leaving the top-right empty. For variants where the top
            # line stays high (e.g. 8B-SFT keeps Information Retrieval
            # at 50-70%% the whole run), callers can override
            # `legend_loc` to e.g. 'center right' so the legend doesn't
            # sit on top of any curve.
            leg = ax.legend(
                ordered,
                display_labels,
                loc=legend_loc,
                ncol=3,
                fontsize=10,
                frameon=True,
                framealpha=0.85,
                edgecolor="#999999",
                fancybox=False,
                borderpad=0.4,
                labelspacing=0.32,
                handlelength=1.7,
                handletextpad=0.5,
                columnspacing=0.8,
            )
        leg.get_frame().set_linewidth(0.6)
    return handles


def plot_per_dataset(
    curve: Dict[int, Dict[str, Dict[str, float]]],
    dataset: str,
    steps: List[int],
    out_dir: Path,
    stem_suffix: str = "",
    legend_loc: str = "upper right",
    legend_outside: bool = False,
) -> Path:
    series = build_series(curve, dataset, steps)
    # Widen the figure a bit when the legend lives outside so the axes
    # itself keeps roughly the same aspect ratio as the in-axes variant.
    fig_w = 9.2 if legend_outside else 7.4
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))
    _draw_single(
        ax,
        steps,
        series,
        show_ylabel=True,
        # Per-dataset panels carry their own two-column legend (see the
        # `ncol=2` block inside `_draw_single`). A standalone legend
        # figure is still emitted by `plot_legend_only` for layouts that
        # prefer a shared legend, but each panel is also self-contained.
        show_legend=True,
        legend_loc=legend_loc,
        legend_outside=legend_outside,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{slugify(dataset)}__categories{stem_suffix}"
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path


# Display-only label overrides for the standalone legend. The canonical
# names (used in the JSON / Markdown / CSV artifacts) keep saying
# "Memory", but in the figure legend we abbreviate to "Mem." so the
# single-row layout fits without crowding.
LEGEND_LABEL_OVERRIDES: Dict[str, str] = {
    "Short-Term Memory": "Short-Term Mem.",
    "Long-Term Memory": "Long-Term Mem.",
}


def plot_legend_only(out_dir: Path, stem_suffix: str = "") -> Path:
    """Render a standalone figure that contains only the category legend.

    The per-dataset panels deliberately omit their legends to save space;
    this single legend can then be placed once next to a grid of small-
    multiple plots. Per the user's spec the legend is laid out as a
    **single row** (one column per category), and "Memory" is
    abbreviated to "Mem." so the row fits at a comfortable width.
    """
    # Build proxy line handles in the canonical CATEGORIES order so the
    # legend matches the per-panel plotting style exactly (same colour,
    # marker, linestyle).
    handles: List[Any] = []
    labels: List[str] = []
    for cat in CATEGORIES:
        style = CATEGORY_STYLE[cat]
        display_label = LEGEND_LABEL_OVERRIDES.get(cat, cat)
        (line,) = plt.plot(
            [], [],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            label=display_label,
        )
        handles.append(line)
        labels.append(display_label)
    plt.close()  # discard the throwaway axes used to create the proxies

    # Wide, short figure sized for a single-row legend (one column per
    # category). The width scales with the number of entries so adding /
    # removing categories doesn't crowd the row.
    fig = plt.figure(figsize=(2.0 * len(CATEGORIES), 0.7))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(CATEGORIES),
        fontsize=14,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
        handletextpad=0.6,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"legend{stem_suffix}"
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_DEFAULTS.keys()),
        default="8b",
        help=(
            "Which model variant to plot. Selects the default curve JSON "
            "and output sub-directory (figures/8B or figures/E4B). "
            "Override either with --json / --out-dir."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to training_curve.json (default depends on --variant).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for output figures (default depends on --variant).",
    )
    parser.add_argument(
        "--stem-suffix",
        type=str,
        default=None,
        help=(
            "Optional extra suffix appended to each figure's filename stem "
            "(e.g. '_e4b'). Defaults to empty since per-variant subdirs "
            "already keep figures from overwriting each other."
        ),
    )
    parser.add_argument(
        "--drop-endpoints",
        action="store_true",
        help=(
            "Drop the two endpoints (BEFORE / AFTER) from the plot. By "
            "default we keep them so the figure starts at step 0 and ends "
            "at the post-training step."
        ),
    )
    args = parser.parse_args()

    defaults = VARIANT_DEFAULTS[args.variant]
    json_path: Path = args.json if args.json is not None else defaults["json"]
    out_dir: Path = args.out_dir if args.out_dir is not None else defaults["out_dir"]
    stem_suffix: str = (
        args.stem_suffix if args.stem_suffix is not None else defaults["stem_suffix"]
    )
    legend_loc: str = defaults.get("legend_loc", "upper right")
    legend_outside: bool = bool(defaults.get("legend_outside", False))

    setup_style()

    print(f"[info] variant: {args.variant}")
    print(f"[info] reading {json_path}")
    curve = load_curve(json_path)
    all_steps = sorted(curve.keys())

    if args.drop_endpoints:
        # Drop the two endpoints: keep only the interior interpolated points.
        steps = [s for s in all_steps if s not in (min(all_steps), max(all_steps))]
    else:
        steps = all_steps

    print(f"[info] plotting {len(steps)} steps: {steps[0]}..{steps[-1]}")

    n_written = 0
    for ds in DATASETS:
        p = plot_per_dataset(
            curve,
            ds,
            steps,
            out_dir,
            stem_suffix=stem_suffix,
            legend_loc=legend_loc,
            legend_outside=legend_outside,
        )
        n_written += 1
        print(f"[plot] {p}")

    # Standalone legend (2 columns) shared by all dataset panels.
    legend_path = plot_legend_only(out_dir, stem_suffix=stem_suffix)
    n_written += 1
    print(f"[plot] {legend_path}")

    print(f"[done] wrote {n_written} figure(s) to {out_dir}")


if __name__ == "__main__":
    main()
