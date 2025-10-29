#!/usr/bin/env bash
# tools/hybrid/run_2mb_devspot.sh
# One-button pipeline for L3=2MB:
#  - build per-capacity binary
#  - characterize benches
#  - submit dataset eval grid
#  - aggregate + split per-cap
#  - submit devspot device-latency ground-truth
#  - aggregate + split per-cap

set -euo pipefail

# ------------------ CONFIG ------------------
CAP="${CAP:-2}"              # L3 capacity (MB)
CPU_PART="${CPU_PART:-cpu-q}"    # cluster partition (used by the sbatch scripts you already have)
EVAL_THROTTLE="${EVAL_THROTTLE:-64}"  # max concurrent eval tasks
export TRACES_ROOT="${TRACES_ROOT:-$HOME/traces/speccpu}"
# --------------------------------------------

# Repo root relative to this script
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm results/tmp

command -v jq >/dev/null || { echo "Need 'jq' (sudo apt install -y jq)"; exit 1; }

echo "[env] ROOT=$ROOT  CAP=${CAP}MB  TRACES_ROOT=$TRACES_ROOT"

# ========== Ensure the eval array script is clean (drop-in) ==========
cat > tools/hybrid/sbatch_eval_array.sh <<'SH'
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
SH
sed -i 's/\r$//' tools/hybrid/sbatch_eval_array.sh
chmod +x tools/hybrid/sbatch_eval_array.sh
bash -n tools/hybrid/sbatch_eval_array.sh

# ========== Build per-capacity ChampSim binary ==========
echo "[build] configuring champsim for L3=${CAP}MB"
BL=$(jq -r '.block_size' champsim_config.json)
WAYS=$(jq -r '.LLC.ways'  champsim_config.json)
SETS=$(( CAP*1024*1024 / (BL*WAYS) ))

jq --argjson sets "$SETS" --arg exe "champsim_L3_${CAP}MB" \
   '.LLC.sets=$sets | .executable_name=$exe' \
   champsim_config.json > "champsim_config.L3_${CAP}.json"

./config.sh "champsim_config.L3_${CAP}.json"
make -j"$(nproc)"
ls -l "bin/champsim_L3_${CAP}MB"

# ========== Characterization (L3=CAP) ==========
echo "[submit] characterization array (L3=${CAP}MB)"
CHAR_JID=$(sbatch tools/hybrid/sbatch_characterize_mb.sh "$CAP" | sed -n 's/Submitted batch job //p')
echo "[char] job id = $CHAR_JID"

# Point generic path at this capacity’s characterization (safe even while queueing)
ln -sfn "$PWD/results/characterization_L3_${CAP}" results/characterization

# ========== Dataset eval grid (L3=CAP) ==========
echo "[prep] generating dataset eval TSV"
python3 tools/hybrid/make_eval_params.py > tools/hybrid/params_eval_all.tsv
N=$(wc -l < tools/hybrid/params_eval_all.tsv)
echo "[submit] eval array ($N tasks) with dependency afterok:$CHAR_JID"
EVAL_JID=$(sbatch --dependency=afterok:$CHAR_JID \
        --array=1-"$N"%${EVAL_THROTTLE} \
        tools/hybrid/sbatch_eval_array.sh tools/hybrid/params_eval_all.tsv "$CAP" "L3_${CAP}MB" \
        | sed -n 's/Submitted batch job //p')
echo "[eval] job id = $EVAL_JID"

# --------- Wait for eval array to complete (optional but tidy) ----------
echo "[wait] eval array $EVAL_JID finishing…"
while [[ -n "$(squeue -h -j "$EVAL_JID")" ]]; do sleep 20; done

# ========== Aggregate and split per-capacity ==========
echo "[aggregate] building dataset.csv and per-cap splits"
python3 tools/hybrid/aggregate_eval.py
python3 - <<'PY'
import pandas as pd, os
p="results/eval/dataset.csv"
if os.path.exists(p):
    df=pd.read_csv(p)
    for mb,sub in df.groupby(df['l3_mb'].round().astype(int)):
        out=f"results/eval/dataset_L3_{mb}MB.csv"
        os.makedirs("results/eval", exist_ok=True)
        sub.to_csv(out,index=False)
        print("   wrote", out, len(sub), "rows")
else:
    print("[warn] results/eval/dataset.csv not found")
PY

# ========== Devspot ground-truth at device latencies ==========
echo "[prep] generating devspot params"
python3 - <<'PY' > tools/hybrid/params_devspot.tsv
import json, os
benches = [
  "600.perlbench_s-570B","602.gcc_s-1850B","620.omnetpp_s-874B","605.mcf_s-994B",
  "623.xalancbmk_s-700B","657.xz_s-3167B","641.leela_s-1083B","631.deepsjeng_s-928B",
  "619.lbm_s-2677B","621.wrf_s-6673B","648.exchange2_s-1247B","649.fotonik3d_s-7084B",
]
profiles = ["val_tS16_MR28_MW60","val_tS16_MR28_MW100","val_tS16_MR28_MW120"]
pw, pm = 0.50, 0.50
for prof in profiles:
    dj = json.load(open(os.path.join("devices", prof, "base.json")))
    tS  = int(dj["latency"]["t_sram_hit"])
    tMr = int(dj["latency"]["t_mram_rd"])
    tMw = int(dj["latency"]["t_mram_wr"])
    for b in benches:
        print(b, pw, pm, tS, tMr, tMw, f"devspot_{prof}", sep="\t")
PY

N_DEV=$(wc -l < tools/hybrid/params_devspot.tsv)
echo "[submit] devspot array ($N_DEV tasks)"
DEV_JID=$(sbatch --dependency=afterok:$EVAL_JID \
        --array=1-"$N_DEV"%${EVAL_THROTTLE} \
        tools/hybrid/sbatch_eval_array.sh tools/hybrid/params_devspot.tsv "$CAP" "L3_${CAP}MB" \
        | sed -n 's/Submitted batch job //p')
echo "[devspot] job id = $DEV_JID"

echo "[wait] devspot array $DEV_JID finishing…"
while [[ -n "$(squeue -h -j "$DEV_JID")" ]]; do sleep 20; done

# ========== Final aggregate + split & small report ==========
echo "[aggregate] final dataset rebuild/split"
python3 tools/hybrid/aggregate_eval.py
python3 - <<'PY'
import pandas as pd, os
df=pd.read_csv("results/eval/dataset.csv")
for mb,sub in df.groupby(df['l3_mb'].round().astype(int)):
    out=f"results/eval/dataset_L3_{mb}MB.csv"
    os.makedirs("results/eval", exist_ok=True); sub.to_csv(out,index=False)
print("[report] devspot rows total:", df["tag"].astype(str).str.startswith("devspot_").sum())
PY

echo "[DONE] L3_${CAP}MB build+char+eval+devspot complete."
