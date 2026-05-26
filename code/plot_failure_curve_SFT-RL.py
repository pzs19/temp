#!/usr/bin/env python3
"""Render publication-quality figures for the SFT->RL failure-rate curves
produced by ``interpolate_failure_curve_SFT-RL.py``.

Reads ``results/failure_curve_SFT-RL.json`` and writes one figure per
(benchmark, pair_group) under ``figures/``. Each figure has 4 lines, one
per tool category (Long-Term Mem., Short-Term Mem., Context Offloading,
Information Retrieval).

Style is aligned with
``analysis/tool-call_statics/plot_training_curve.py`` (same serif font,
Okabe-Ito palette, figsize, legend formatting, etc.).

Usage:
    python3 plot_failure_curve_SFT-RL.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
CURVE_JSON = RESULTS_DIR / "failure_curve_SFT-RL.json"
PLOTS_DIR = SCRIPT_DIR / "figures"

# ---------------------------------------------------------------------------
# Categories / styles -- must match interpolate_failure_curve_SFT-RL.py
# ---------------------------------------------------------------------------
CATEGORIES: List[str] = [
    "Long-Term Mem.",
    "Short-Term Mem.",
    "Context Offloading",
    "Information Retrieval",
]

# Color-blind friendly Okabe-Ito palette, aligned with
# analysis/tool-call_statics/plot_training_curve.py.
CATEGORY_STYLE: Dict[str, Dict[str, object]] = {
    "Long-Term Mem.":        {"color": "#E69F00", "marker": "v", "linestyle": "-"},
    "Short-Term Mem.":       {"color": "#CC79A7", "marker": "D", "linestyle": "-"},
    "Context Offloading":    {"color": "#009E73", "marker": "^", "linestyle": "-"},
    "Information Retrieval": {"color": "#D55E00", "marker": "s", "linestyle": "-"},
}


# ---------------------------------------------------------------------------
# Style (mirrors plot_training_curve.py exactly)
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

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(name: str) -> str:
    return _SAFE_RE.sub("_", name).strip("_")


def load_curves(path: Path) -> Dict[str, Dict[str, Dict[str, Dict[int, float]]]]:
    """Return {benchmark: {pair_group: {category: {step(int): rate(float)}}}}."""
    raw = json.loads(path.read_text())
    out: Dict[str, Dict[str, Dict[str, Dict[int, float]]]] = {}
    for bench, group_map in raw.items():
        out[bench] = {}
        for group, cat_map in group_map.items():
            out[bench][group] = {}
            for cat, series in cat_map.items():
                out[bench][group][cat] = {
                    int(step): float(rate) for step, rate in series.items()
                }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_one(
    cat_map: Dict[str, Dict[int, float]],
    out_dir: Path,
    stem: str,
) -> Path:
    any_series = next(iter(cat_map.values()))
    step_grid = sorted(any_series.keys())

    # figsize matches analysis/tool-call_statics/plot_training_curve.py
    # exactly so the two figure families render at the same size.
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    handles = []
    for cat in CATEGORIES:
        if cat not in cat_map:
            continue
        series = cat_map[cat]
        ys = [series[s] * 100.0 for s in step_grid]  # to %
        style = CATEGORY_STYLE[cat]
        (line,) = ax.plot(
            step_grid,
            ys,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markevery=1,
            alpha=0.95,
            label=cat,
            zorder=3,
        )
        handles.append(line)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Failure Rate")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    x_margin = max(1.0, (step_grid[-1] - step_grid[0]) * 0.02)
    ax.set_xlim(step_grid[0] - x_margin, step_grid[-1] + x_margin)

    max_y = max(
        max(series[s] for s in step_grid) for series in cat_map.values()
    )
    upper = min(1.0, max_y * 1.12) if max_y > 0 else 1.0
    ax.set_ylim(0, upper * 100.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _pos: f"{v:.0f}%")
    )

    leg = ax.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper right",
        ncol=2,
        fontsize=12,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def plot_all(curves: Dict[str, Dict[str, Dict[str, Dict[int, float]]]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for bench, group_map in curves.items():
        bench_safe = _sanitize(bench)
        for group_label, cat_map in group_map.items():
            # Strip the " SFT -> RL" suffix from the group label so the file
            # stem looks like ``<bench>__<model>__SFT-RL`` rather than the
            # noisier ``<bench>__<model>_SFT_-_RL__SFT-RL``.
            short_group = group_label.split(" SFT")[0].strip()
            group_safe = _sanitize(short_group)
            stem = f"{bench_safe}__{group_safe}__SFT-RL"
            out_path = plot_one(cat_map, PLOTS_DIR, stem)
            print(f"[INFO] plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()
    if not CURVE_JSON.exists():
        raise SystemExit(
            f"[ERROR] {CURVE_JSON} not found. "
            "Run interpolate_failure_curve_SFT-RL.py first."
        )
    curves = load_curves(CURVE_JSON)
    plot_all(curves)
    print(f"[INFO] Plots written under: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
