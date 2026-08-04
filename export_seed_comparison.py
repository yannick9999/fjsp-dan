"""
Exports a comparison table of the DANIEL model (trained on 20x10) evaluated
with 3 different seeds (0, 1, 2), for both the greedy and the sampling
strategy, across all SD1 and SD2 test set sizes.

Rows: SD1 10x5, 20x5, 15x10, 20x10, 30x10, 40x10, then the same order for SD2
Columns: Greedy_seed0, Greedy_seed1, Greedy_seed2, Greedy_avg,
         Sampling_seed0, Sampling_seed1, Sampling_seed2, Sampling_avg
Cell value: mean makespan over the 100 test instances of that data set.
"""
import os
import numpy as np
import pandas as pd

SIZES = ["10x5", "20x5", "15x10", "20x10", "30x10", "40x10"]
SOURCES = ["SD1", "SD2"]
SEEDS = [0, 1, 2]
MODEL_BASE = "20x10"  # model was trained on 20x10

OUT_PATH = "./TestDataToExcel/seed_comparison.xlsx"


def data_name(source, size):
    return f"{size}+mix" if source == "SD2" else size


def load_mean_makespan(source, model_name, data_name_):
    file_path = f"./test_results/{source}/{data_name_}/Result_{model_name}_{data_name_}.npy"
    if not os.path.exists(file_path):
        print(f"[WARN] missing file: {file_path}")
        return np.nan
    result = np.load(file_path)
    return float(np.mean(result[:, 0]))


def main():
    rows = []
    row_labels = []

    for source in SOURCES:
        for size in SIZES:
            dname = data_name(source, size)
            row = {}

            greedy_vals = []
            sampling_vals = []
            for seed in SEEDS:
                g_model = f"DANIELG+{MODEL_BASE}+seed{seed}"
                s_model = f"DANIELS+{MODEL_BASE}+seed{seed}"

                g_val = load_mean_makespan(source, g_model, dname)
                s_val = load_mean_makespan(source, s_model, dname)

                row[f"Greedy_seed{seed}"] = g_val
                row[f"Sampling_seed{seed}"] = s_val

                greedy_vals.append(g_val)
                sampling_vals.append(s_val)

            row["Greedy_avg"] = float(np.nanmean(greedy_vals))
            row["Sampling_avg"] = float(np.nanmean(sampling_vals))

            rows.append(row)
            row_labels.append(f"{source} {size}")

    columns = [
        "Greedy_seed0", "Greedy_seed1", "Greedy_seed2", "Greedy_avg",
        "Sampling_seed0", "Sampling_seed1", "Sampling_seed2", "Sampling_avg",
    ]
    df = pd.DataFrame(rows, index=row_labels, columns=columns)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_excel(OUT_PATH, sheet_name="makespan", index=True)

    print(df)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
