"""Loads the model comparison result data, runs the rliable bootstrap
analysis for the IQM bar charts, and writes a cache for plot.py.

Run this whenever the underlying data changes. Run plot.py (no
recomputation) whenever only the plot styling should change.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rliable import library as rly
from rliable import metrics

from common import (
    ALL_SIZES,
    ANALYSIS_CACHE,
    BASELINES,
    BENCH_FILE_MAP,
    BENCHMARKS_DIR,
    BOOTSTRAP_REPS,
    BRANDIMARTE,
    BRANDIMARTE_INSTANCES,
    DISPATCHING_RULES,
    METHODS,
    MODES,
    RESULT_FOLDER_MAP,
    SCRIPT_DIR,
    SEEDS,
    combo_key,
)


# Data loading

def load_drl_test_makespans(method: str, size: str, seed: int, mode: str) -> dict[str, float] | None:
    """Loads test makespans per instance for a method, size, seed, mode.

    Returns:
        Dict {instance_name: makespan} or None if the file is missing.
    """
    folder = RESULT_FOLDER_MAP[size]
    if method == "dan":
        excel = SCRIPT_DIR / f"seed{seed}" / f"{folder}_{mode}" / "results.xlsx"
    else:
        # Add a path branch here when edsp/song/sagc results are copied in.
        raise ValueError(f"No loader defined for method '{method}'")
    if not excel.exists():
        return None
    df = pd.read_excel(excel, sheet_name="makespan")
    instance_col = df.columns[0]   # file_name
    makespan_col = df.columns[1]   # makespan
    return dict(zip(df[instance_col].astype(str), df[makespan_col].astype(float)))


def load_benchmark_makespans(rule: str, size: str) -> dict[str, float] | None:
    """Loads baseline makespans from CSV, stripping the .fjs extension so the
    instance names match the xlsx result files.

    Returns:
        Dict {instance_name: makespan} or None if the file is missing.
    """
    csv = BENCHMARKS_DIR / rule / f"{BENCH_FILE_MAP[size]}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    names = df["instance_name"].astype(str).str.replace(r"\.fjs$", "", regex=True)
    return dict(zip(names, df["makespan"].astype(float)))


# Score matrices

def get_baseline_makespans(size: str) -> dict[str, dict[str, float]]:
    """Collects all available baseline makespans for a size.

    Returns:
        Dict {baseline_name: {instance_name: makespan}}
    """
    result = {}
    for b in BASELINES:
        m = load_benchmark_makespans(b, size)
        if m is not None:
            result[b] = m
        else:
            print(f"  [warn] Baseline {b} missing for {size}")
    return result


def compute_c_best(baseline_data: dict[str, dict[str, float]]) -> dict[str, float]:
    """Computes C_best per instance as the minimum over all baselines."""
    if not baseline_data:
        return {}
    all_instances = set()
    for b_data in baseline_data.values():
        all_instances.update(b_data.keys())
    c_best = {}
    for inst in all_instances:
        values = [b_data[inst] for b_data in baseline_data.values() if inst in b_data]
        if values:
            c_best[inst] = min(values)
    return c_best


def compute_c_best_dr(baseline_data: dict[str, dict[str, float]]) -> dict[str, float]:
    """Computes the best-dispatching-rule makespan per instance (CP-SAT excluded)."""
    dr_data = {b: d for b, d in baseline_data.items() if b in DISPATCHING_RULES}
    return compute_c_best(dr_data)


def build_score_matrix(method: str, size: str, c_best: dict[str, float],
                       mode: str) -> tuple[np.ndarray, list[str]]:
    """Builds the normalized score matrix for a method, size, mode.

    Score = C_best / C_method (higher = better).

    Returns:
        (matrix shape (num_seeds, num_instances), list of instance_names in
         the same order as the matrix columns)
    """
    per_seed_dicts = []
    for s in SEEDS:
        d = load_drl_test_makespans(method, size, s, mode)
        if d is None:
            print(f"  [warn] Test data missing: {method} seed{s} {size} {mode}")
            return np.array([]), []
        per_seed_dicts.append(d)

    # Common instances present in all seeds AND in c_best
    common = set(per_seed_dicts[0].keys())
    for d in per_seed_dicts[1:]:
        common &= set(d.keys())
    if c_best:
        common &= set(c_best.keys())
    instances = sorted(common)

    if not instances:
        return np.array([]), []

    matrix = np.zeros((len(SEEDS), len(instances)))
    for i, s in enumerate(SEEDS):
        for j, inst in enumerate(instances):
            matrix[i, j] = c_best[inst] / per_seed_dicts[i][inst]
    return matrix, instances


def build_baseline_score(baseline_makespans: dict[str, float], c_best: dict[str, float],
                         instances: list[str]) -> np.ndarray:
    """Score array for a deterministic baseline (shape (1, num_instances))."""
    scores = np.array([c_best[i] / baseline_makespans[i] for i in instances if i in baseline_makespans])
    return scores.reshape(1, -1)


def build_scores_for_size(size: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Builds all (method, mode) score matrices plus baseline score arrays
    for one size/dataset.

    Returns:
        (score_dict, baseline_scores). Both empty if data is missing.
    """
    baseline_data = get_baseline_makespans(size)
    c_best = compute_c_best(baseline_data)
    if not c_best:
        print(f"  [warn] No C_best for {size}, skipping")
        return {}, {}

    score_dict = {}
    instances_ref = None
    for method in METHODS:
        for mode in MODES:
            matrix, instances = build_score_matrix(method, size, c_best, mode)
            if matrix.size == 0:
                continue
            score_dict[combo_key(method, mode)] = matrix
            if instances_ref is None:
                instances_ref = instances
            iqm = metrics.aggregate_iqm(matrix)
            print(f"    {combo_key(method, mode)}: {matrix.shape}, IQM={iqm:.4f}")

    baseline_scores = {}
    if instances_ref is not None:
        for b, b_data in baseline_data.items():
            arr = build_baseline_score(b_data, c_best, instances_ref)
            if arr.size:
                baseline_scores[b] = arr
        best_dr = compute_c_best_dr(baseline_data)
        arr = build_baseline_score(best_dr, c_best, instances_ref)
        if arr.size:
            baseline_scores["BestDR"] = arr

    return score_dict, baseline_scores


# Bootstrap analysis

def analyze_iqm_bars(score_dict_per_size: dict[str, dict[str, np.ndarray]],
                     baseline_scores_per_size: dict[str, dict[str, np.ndarray]],
                     sizes: list[str]) -> dict:
    """Bootstraps IQM + 95% CI per (method, mode), per size; plus baseline IQM points."""
    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x)])
    result = {}
    n = len(sizes)

    for si, size in enumerate(sizes):
        score_dict = score_dict_per_size.get(size, {})
        if not score_dict:
            result[size] = None
            continue

        print(f"    bootstrap {size} ({si+1}/{n}) ...", end=" ", flush=True)
        t0 = time.time()
        iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm_fn, reps=BOOTSTRAP_REPS)
        print(f"{time.time()-t0:.1f}s")

        methods = list(score_dict.keys())
        means = {m: float(iqm_scores[m][0]) for m in methods}
        cis = {m: (float(iqm_cis[m][0, 0]), float(iqm_cis[m][1, 0])) for m in methods}

        baseline_scores = baseline_scores_per_size.get(size, {})
        baseline_iqm = {}
        for b, arr in baseline_scores.items():
            if arr.size:
                baseline_iqm[b] = float(metrics.aggregate_iqm(arr))

        result[size] = {"methods": methods, "means": means, "cis": cis, "baseline_iqm": baseline_iqm}

    return result


def build_brandimarte_per_instance() -> tuple[dict[str, dict[str, np.ndarray]],
                                              dict[str, dict[str, np.ndarray]]]:
    """Builds per-instance score dicts for Brandimarte, so that the IQM chart
    can show one bar group per Mk instance.

    Each "size" key is one instance (Mk01 ... Mk10). The method matrices have
    shape (num_seeds, 1), the bootstrap CI then reflects seed variation only.

    Returns:
        (score_dict_per_instance, baseline_scores_per_instance)
    """
    baseline_data = get_baseline_makespans(BRANDIMARTE)
    c_best = compute_c_best(baseline_data)
    best_dr = compute_c_best_dr(baseline_data)

    # Load all method data once
    method_data = {}
    for method in METHODS:
        for mode in MODES:
            per_seed = []
            for s in SEEDS:
                d = load_drl_test_makespans(method, BRANDIMARTE, s, mode)
                if d is None:
                    print(f"  [warn] Test data missing: {method} seed{s} {BRANDIMARTE} {mode}")
                    per_seed = None
                    break
                per_seed.append(d)
            if per_seed is not None:
                method_data[combo_key(method, mode)] = per_seed

    score_dict_per_instance = {}
    baseline_scores_per_instance = {}
    for inst in BRANDIMARTE_INSTANCES:
        if inst not in c_best:
            print(f"  [warn] No baseline data for {inst}, skipping")
            continue

        score_dict = {}
        for key, per_seed in method_data.items():
            if all(inst in d for d in per_seed):
                col = np.array([[c_best[inst] / d[inst]] for d in per_seed])
                score_dict[key] = col
        score_dict_per_instance[inst] = score_dict

        baseline_scores = {}
        for b, b_data in baseline_data.items():
            if inst in b_data:
                baseline_scores[b] = np.array([[c_best[inst] / b_data[inst]]])
        if inst in best_dr:
            baseline_scores["BestDR"] = np.array([[c_best[inst] / best_dr[inst]]])
        baseline_scores_per_instance[inst] = baseline_scores

    return score_dict_per_instance, baseline_scores_per_instance


# Main

def main():
    print("=" * 70)
    print("Model Comparison Analysis")
    print("=" * 70)
    print(f"Script dir:  {SCRIPT_DIR}")
    print(f"Benchmarks:  {BENCHMARKS_DIR}")
    print(f"Cache out:   {ANALYSIS_CACHE}")
    print(f"Methods:     {METHODS}")
    print(f"Modes:       {MODES}")
    print(f"Seeds:       {SEEDS}")
    print(f"Sizes:       {ALL_SIZES}")
    print(f"Brandimarte: {BRANDIMARTE_INSTANCES}")
    print(f"Baselines:   {BASELINES}")
    print()

    t_start = time.time()

    # 1. Score matrices for all sizes/datasets (20x10 shared between the two
    # synthetic groups, so it's loaded and bootstrapped exactly once)
    print("[1/3] Loading data and building score matrices ...")
    score_dict_per_size = {}
    baseline_scores_per_size = {}
    for size in ALL_SIZES:
        print(f"  Size {size}")
        score_dict, baseline_scores = build_scores_for_size(size)
        if score_dict:
            score_dict_per_size[size] = score_dict
            baseline_scores_per_size[size] = baseline_scores

    # 2. Bootstrap IQM per size
    print("[2/3] IQM bootstrap per size ...")
    iqm = analyze_iqm_bars(score_dict_per_size, baseline_scores_per_size, ALL_SIZES)

    # 3. Brandimarte per instance
    print("[3/3] IQM bootstrap per Brandimarte instance ...")
    mk_scores, mk_baselines = build_brandimarte_per_instance()
    iqm_brandimarte = analyze_iqm_bars(mk_scores, mk_baselines, BRANDIMARTE_INSTANCES)

    cache = {"iqm": iqm, "iqm_brandimarte": iqm_brandimarte}
    with open(ANALYSIS_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nSaved analysis cache to {ANALYSIS_CACHE}")

    total_elapsed = time.time() - t_start
    print(f"All done in {total_elapsed:.1f}s. Run plot.py to (re)generate plots.")


if __name__ == "__main__":
    main()