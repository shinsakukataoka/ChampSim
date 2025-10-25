#!/usr/bin/env bash
#SBATCH --partition=cpu-test-q
#SBATCH --job-name=hyb_eval
#SBATCH --array=1-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

ROOT="/home/skataoka26/ChampSim"
cd "$ROOT"

PARAMS_FILE="${1:-tools/hybrid/params_eval.tsv}"
L3MB="${2:-}"                  # << optional capacity (MB)
TAG_SUFFIX="${3:-}"            # << optional tag suffix (e.g., L3_32MB)

# pick the Nth line (no header expected)
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE")
IFS=$'\t' read -r bench pi_way pi_miss tS tMr tMw tag <<< "$line"

# add suffix if provided (keeps runs separated by capacity)
if [[ -n "$TAG_SUFFIX" ]]; then
  tag="${tag}_${TAG_SUFFIX}"
fi

echo "TASK ${SLURM_ARRAY_TASK_ID}: $bench pw=$pi_way pm=$pi_miss tS=$tS tMr=$tMr tMw=$tMw tag=$tag L3MB=${L3MB:-<default>}"

# build args
args=( "$bench" "$pi_way" "$pi_miss" "$tS" "$tMr" "$tMw" "$tag" )
if [[ -n "$L3MB" ]]; then
  args+=( --l3-mb "$L3MB" )
fi

# run
srun --cpu-bind=cores python3 tools/hybrid/eval_design.py "${args[@]}"

