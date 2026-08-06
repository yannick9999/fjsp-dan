"""Renders the IQM bar charts from the analysis cache written by analyze.py.

Run analyze.py first (or whenever the underlying data changes). Re-run this
script alone to restyle plots, it never recomputes the bootstrap CIs.

Outputs:
    iqm_synthetic_machines.png   20 jobs, machines varied (20x5 ... 20x30)
    iqm_synthetic_jobs.png       10 machines, jobs varied (10x10 ... 200x10)
    iqm_hurink.png               Hurink edata / rdata / vdata
    iqm_brandimarte.png          one bar group per Mk instance
"""

from __future__ import annotations

import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from common import (
    ANALYSIS_CACHE,
    BASELINE_COLORS,
    BASELINE_LABELS,
    BRANDIMARTE_INSTANCES,
    HURINK_DATASETS,
    HURINK_LABELS,
    IQM_BASELINE_KEYS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
    MODE_HATCHES,
    MODE_LABELS,
    MODES,
    PLOTS_DIR,
    SYNTHETIC_JOBS,
    SYNTHETIC_MACHINES,
    combo_key,
)

mpl.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def load_analysis() -> dict:
    if not ANALYSIS_CACHE.exists():
        raise FileNotFoundError(f"{ANALYSIS_CACHE} not found - run analyze.py first.")
    with open(ANALYSIS_CACHE, "rb") as f:
        return pickle.load(f)


def plot_iqm_bars(data: dict, sizes: list[str], out_name: str, title: str,
                  figsize: tuple[float, float],
                  size_labels: dict[str, str] | None = None):
    """IQM as a grouped bar chart, one group per size/dataset/instance.

    Bar order within each group: CP-SAT, Best DR, then the (method, mode)
    combos (greedy before sampling). Baselines are deterministic, so they get
    no error bar; the method bars carry the 95% bootstrap CIs. Missing values
    are NaN so matplotlib simply skips drawing that bar, instead of a
    misleading zero-height stub.
    """
    size_labels = size_labels or {}
    combos = [(method, mode) for method in METHODS for mode in MODES]

    n_sizes = len(sizes)
    n_items = len(IQM_BASELINE_KEYS) + len(combos)
    bar_width = 0.18
    group_gap = 0.5
    group_positions = np.arange(n_sizes) * (n_items * bar_width + group_gap)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F9F9F9')

    max_top = 1.02
    min_bottom = 0.9

    # Baseline bars first (CP-SAT, Best DR)
    baseline_bar_handles = []
    for b_idx, baseline in enumerate(IQM_BASELINE_KEYS):
        offsets = group_positions + b_idx * bar_width
        means = []
        for size in sizes:
            entry = data.get(size)
            val = entry["baseline_iqm"].get(baseline) if entry else None
            means.append(val if val is not None else np.nan)
            if val is not None:
                max_top = max(max_top, val)
                min_bottom = min(min_bottom, val)

        bars = ax.bar(offsets, means, width=bar_width,
                      color=BASELINE_COLORS.get(baseline, "gray"),
                      edgecolor="white", linewidth=0.6, zorder=3)
        baseline_bar_handles.append(bars[0])

    # Method bars with bootstrap CIs
    combo_bar_handles = []
    for c_idx, (method, mode) in enumerate(combos):
        key = combo_key(method, mode)
        offsets = group_positions + (len(IQM_BASELINE_KEYS) + c_idx) * bar_width
        means, err_low, err_high = [], [], []

        for size in sizes:
            entry = data.get(size)
            if entry and key in entry["means"]:
                val = entry["means"][key]
                means.append(val)
                err_low.append(val - entry["cis"][key][0])
                err_high.append(entry["cis"][key][1] - val)
                max_top = max(max_top, entry["cis"][key][1])
                min_bottom = min(min_bottom, entry["cis"][key][0])
            else:
                means.append(np.nan)
                err_low.append(0)
                err_high.append(0)

        bars = ax.bar(offsets, means, width=bar_width,
                      color=METHOD_COLORS[method], hatch=MODE_HATCHES[mode],
                      yerr=[err_low, err_high],
                      capsize=2, error_kw={"elinewidth": 0.8, "capthick": 0.8},
                      edgecolor="white", linewidth=0.6, zorder=3)
        combo_bar_handles.append(bars[0])

    # X-axis group labels
    group_centers = group_positions + (n_items - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels([size_labels.get(s, s) for s in sizes], fontsize=13)
    ax.set_xlim(group_positions[0] - 0.4, group_positions[-1] + n_items * bar_width + 0.4)

    # Y-axis, ceiling/floor grow with the data so no bar or error bar clips
    ax.set_ylim(min_bottom - 0.02, max_top + 0.02)
    ax.set_ylabel("IQM Score (C_best / C_method)", fontsize=15, labelpad=8)
    ax.tick_params(axis='y', labelsize=13)

    # Grid and spines
    ax.grid(True, axis="y", color="#E0E0E0", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    all_handles = baseline_bar_handles + combo_bar_handles
    all_labels = [BASELINE_LABELS.get(b, b) for b in IQM_BASELINE_KEYS] + \
        [f"{METHOD_LABELS[m]} ({MODE_LABELS[mo]})" for m, mo in combos]
    ax.legend(all_handles, all_labels,
              loc="lower right", fontsize=11,
              frameon=True, framealpha=0.9,
              edgecolor="#CCCCCC", handlelength=2.0, ncol=2)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def main():
    print("=" * 70)
    print("Model Comparison Plotting")
    print("=" * 70)
    print(f"Cache in:  {ANALYSIS_CACHE}")
    print(f"Plots out: {PLOTS_DIR}")
    print()

    data = load_analysis()

    steps = [
        ("IQM Synthetic (machines varied)",
         lambda: plot_iqm_bars(data["iqm"], SYNTHETIC_MACHINES,
                               "iqm_synthetic_machines.png",
                               "IQM with 95% Bootstrap CIs (20 Jobs, Machines Varied)",
                               (9, 5.5))),
        ("IQM Synthetic (jobs varied)",
         lambda: plot_iqm_bars(data["iqm"], SYNTHETIC_JOBS,
                               "iqm_synthetic_jobs.png",
                               "IQM with 95% Bootstrap CIs (10 Machines, Jobs Varied)",
                               (14, 5.5))),
        ("IQM Hurink",
         lambda: plot_iqm_bars(data["iqm"], HURINK_DATASETS,
                               "iqm_hurink.png",
                               "IQM with 95% Bootstrap CIs (Hurink)",
                               (8, 5.5), size_labels=HURINK_LABELS)),
        ("IQM Brandimarte",
         lambda: plot_iqm_bars(data["iqm_brandimarte"], BRANDIMARTE_INSTANCES,
                               "iqm_brandimarte.png",
                               "IQM with 95% Bootstrap CIs (Brandimarte)",
                               (15, 5.5))),
    ]

    total = len(steps)
    for i, (name, fn) in enumerate(steps, 1):
        print(f"[{i}/{total}] {name} ...")
        fn()

    print()
    print("All plots done.")


if __name__ == "__main__":
    main()