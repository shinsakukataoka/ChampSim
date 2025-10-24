#!/usr/bin/env bash
#SBATCH --partition=cpu-q
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

# pick the Nth line (no header expected)
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE")
IFS=$'\t' read -r bench pi_way pi_miss tS tMr tMw tag <<< "$line"

echo "TASK ${SLURM_ARRAY_TASK_ID}: $bench pw=$pi_way pm=$pi_miss tag=$tag"

# call the evaluator (writes results/eval/<bench>/<tag>/*)
python3 tools/hybrid/eval_design.py "$bench" "$pi_way" "$pi_miss" "$tS" "$tMr" "$tMw" "$tag"

