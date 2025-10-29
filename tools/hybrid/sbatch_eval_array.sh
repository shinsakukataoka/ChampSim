#!/usr/bin/env bash
#SBATCH --partition=cpu-q
#SBATCH --job-name=hyb_eval
#SBATCH --array=1-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
#SBATCH --chdir=/home/skataoka26/ChampSim
#SBATCH --export=ALL

set -euo pipefail
ROOT="/home/skataoka26/ChampSim"
cd "$ROOT"

PARAMS_FILE="${1:-tools/hybrid/params_eval.tsv}"
L3MB="${2:-}"          # capacity MB (e.g., 2)
TAG_SUFFIX="${3:-}"      # optional tag suffix (e.g., L3_2MB)

# Get Nth TSV line
line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE" || true)"
if [[ -z "$line" ]]; then echo "Empty TSV line for task $SLURM_ARRAY_TASK_ID in $PARAMS_FILE" >&2; exit 3; fi
IFS=$'\t' read -r bench pi_way pi_miss tS tMr tMw tag <<< "$line"

# Suffix tag to disambiguate datasets
if [[ -n "$TAG_SUFFIX" ]]; then tag="${tag}_${TAG_SUFFIX}"; fi

# Capacity-specific binary
if [[ -n "$L3MB" ]]; then export CHAMPSIM_BIN="$ROOT/bin/champsim_L3_${L3MB}MB"; fi

echo "TASK ${SLURM_ARRAY_TASK_ID}: bench=$bench pw=$pi_way pm=$pi_miss tS=$tS tMr=$tMr tMw=$tMw tag=$tag L3MB=${L3MB:-<default>} BIN=${CHAMPSIM_BIN:-$ROOT/bin/champsim}"

args=( "$bench" "$pi_way" "$pi_miss" "$tS" "$tMr" "$tMw" "$tag" )
if [[ -n "$L3MB" ]]; then args+=( --l3-mb "$L3MB" ); fi

# Run one eval (writes results/eval/<bench>/<tag>/metrics.json)
srun --cpu-bind=cores python3 tools/hybrid/eval_design.py "${args[@]}"
