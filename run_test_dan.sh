#!/bin/bash
#SBATCH --job-name=fjsp_test_song
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/test_song_seed%a_%j.out
#SBATCH --error=logs/test_song_seed%a_%j.err

# Evaluates one of our own SD1 20x10 models (seed = array task id) against
# Song's original SD1 and SD2 benchmark test sets, greedy + sampling, using
# the original test_trained_model.py pipeline (so results are directly
# comparable to the paper's tables via print_test_result.py afterwards).
# Submit with: sbatch run_test_song.sh   (after run_train.sh has finished)

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from repo root}"
cd "${REPO_ROOT}"
mkdir -p logs

module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="/home/grafyann/master_thesis_env/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing ${PYTHON}" >&2; exit 1; }

SEED=${SLURM_ARRAY_TASK_ID}
MODEL_SOURCE=SD1
MODEL_NAME="20x10+seed${SEED}"
MODEL_PATH="./trained_network/${MODEL_SOURCE}/${MODEL_NAME}.pth"

SD1_TEST_DATA=(10x5 15x10 20x10 20x5 30x10 40x10)
SD2_TEST_DATA=(10x5+mix 15x10+mix 20x5+mix 20x10+mix 30x10+mix 40x10+mix)

# Verify the model exists before launching
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: No model file found at ${MODEL_PATH}" >&2
    exit 1
fi

echo "=== Testing seed=${SEED} model=${MODEL_PATH} against SD1 and SD2 ==="
srun nvidia-smi || true

for DATA_SOURCE in SD1 SD2; do
    if [[ "${DATA_SOURCE}" == "SD1" ]]; then
        TEST_DATA=("${SD1_TEST_DATA[@]}")
    else
        TEST_DATA=("${SD2_TEST_DATA[@]}")
    fi

    for TEST_MODE in false true; do
        echo "--- data_source=${DATA_SOURCE} test_mode(sampling)=${TEST_MODE} ---"
        srun "${PYTHON}" test_trained_model.py \
            --model_source "${MODEL_SOURCE}" \
            --test_model "${MODEL_NAME}" \
            --data_source "${DATA_SOURCE}" \
            --test_data "${TEST_DATA[@]}" \
            --test_mode "${TEST_MODE}"
    done
done
