#!/usr/bin/env python3
"""Render dual-y-axis figures for SFT->RL transition.
Left Y: Tool-call Failure Rate (%)
Right Y: Task Score
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FAILURE_JSON = RESULTS_DIR / "failure_curve_SFT-RL.json"
SCORE_JSON = RESULTS_DIR / "task_score_curve_SFT-RL.json"
PLOTS_DIR = SCRIPT_DIR / "figures_dual_y"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
CATEGORY_STYLE: Dict[str, Dict[str, object]] = {
    "Long-Term Mem.":        {"color": "#E69F00", "marker": "v", "linestyle": "--", "alpha": 0.8},
    "Short-Term Mem.":       {"color": "#CC79A7", "marker": "D", "linestyle": "--", "alpha": 0.8},
    "Context Offloading":    {"color": "#009E73", "marker": "^", "linestyle": "--", "alpha": 0.8},
    "Information Retrieval": {"color": "#D55E00", "marker": "s", "linestyle": "--", "alpha": 0.8},
}

SCORE_STYLE = {"color": "#0072B2", "marker": "o", "linestyle": "-", "linewidth": 2.5, "markersize": 5}

def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "axes.axisbelow": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def plot_dual_y(
    benchmark: str,
    group: str,
    failure_data: Dict[str, Dict[str, float]],
    score_data: Dict[str, float]
) -> None:
    # Make it flatter (e.g., 9x4 instead of 8x5)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    
    # Left axis: Failure Rate
    steps = sorted([int(s) for s in score_data.keys()])
    
    for cat, style in CATEGORY_STYLE.items():
        if cat not in failure_data: continue
        vals = [failure_data[cat][str(s)] * 100 for s in steps]
        ax1.plot(steps, vals, label=cat, **style)
    
    ax1.set_xlabel("Training Step") # Simplified X-axis label
    ax1.set_ylabel("Failure Rate (%)", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.set_ylim(bottom=0)
    
    # Right axis: Task Score
    ax2 = ax1.twinx()
    score_vals = [score_data[str(s)] for s in steps]
    ax2.plot(steps, score_vals, label="Task Score", **SCORE_STYLE)
    
    ax2.set_ylabel("Task Score (%)", color=SCORE_STYLE["color"]) 
    ax2.tick_params(axis='y', labelcolor=SCORE_STYLE["color"])

    # Legend - Moved outside to the top, 3 columns, spanning width
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center", 
               bbox_to_anchor=(0.0, 1.02, 1.0, 0.1), mode="expand", 
               ncol=3, frameon=False, borderaxespad=0.)

    # Save
    safe_group = group.replace(" ", "_").replace("->", "to")
    safe_bench = benchmark.replace("/", "_")
    
    base_name = PLOTS_DIR / f"dual_y_{safe_bench}_{safe_group}"
    
    plt.savefig(f"{base_name}.png")
    plt.savefig(f"{base_name}.pdf")
    plt.close()
    print(f"[INFO] Saved plots to {base_name}.png/pdf")

def main():
    setup_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(FAILURE_JSON) as f:
        failure_curves = json.load(f)
    with open(SCORE_JSON) as f:
        score_data_map = json.load(f)
    
    for bench, group_map in score_data_map.items():
        if bench not in failure_curves: continue
        for group, score_data in group_map.items():
            if group not in failure_curves[bench]: continue
            
            plot_dual_y(
                bench, group, 
                failure_curves[bench][group], 
                score_data
            )

if __name__ == "__main__":
    main()
