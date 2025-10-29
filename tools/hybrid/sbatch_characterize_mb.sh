#!/bin/bash
#SBATCH --partition=cpu-q
#SBATCH --job-name=char_mb
#SBATCH --array=1-12%4             # 12 benches; throttle=4 (tune as you like)
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail

ROOT="/home/skataoka26/ChampSim"
TR="/home/skataoka26/traces/speccpu"
cd "$ROOT"

MB="${1:?usage: sbatch ... sbatch_characterize_mb.sh <MB>}"
CHAMPSIM_BIN="${CHAMPSIM_BIN:-$ROOT/bin/champsim_L3_${MB}MB}"
OUTROOT="results/characterization_L3_${MB}"
mkdir -p "$OUTROOT" "results/tmp/char_${MB}" "logs/slurm"

# Bench list (12)
BENCHES=(
  600.perlbench_s-570B
  602.gcc_s-1850B
  620.omnetpp_s-874B
  605.mcf_s-994B
  623.xalancbmk_s-700B
  657.xz_s-3167B
  641.leela_s-1083B
  631.deepsjeng_s-928B
  619.lbm_s-2677B
  621.wrf_s-6673B
  648.exchange2_s-1247B
  649.fotonik3d_s-7084B
)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
BENCH="${BENCHES[$IDX]}"

# Nominal settings for characterization
WARM=2000000; SIM=5000000
PIWAY=0.5; PIMISS=0.5
TS=16; TMR=28; TMW=60

# Unique working dir + stage capacity-specific config
RUN_CWD="results/tmp/char_${MB}/${BENCH}_$RANDOM"
mkdir -p "$RUN_CWD"

python3 - "$ROOT" "$MB" "$RUN_CWD" <<'PY'
import sys, json, os
root, mb, cwd = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cfg = json.load(open(os.path.join(root,"champsim_config.json")))
bl   = int(cfg.get("block_size",64))
ways = int(cfg.get("LLC",{}).get("ways",16))
cfg.setdefault("LLC",{})["sets"] = int((mb*1024*1024)//(bl*ways))
json.dump(cfg, open(os.path.join(cwd,"champsim_config.json"),"w"), indent=2)
PY

echo "== characterize: ${BENCH}  (L3=${MB}MB) =="

# Run ChampSim in the staged CWD
srun --cpu-bind=cores bash -c "
  cd '$RUN_CWD' && '$CHAMPSIM_BIN' \
    --warmup-instructions $WARM \
    --simulation-instructions $SIM \
    --hybrid-llc \
    --pi-way $PIWAY --pi-miss $PIMISS \
    --t-sram-hit $TS --t-mram-rd $TMR --t-mram-wr $TMW \
    '$TR/${BENCH}.champsimtrace.xz'
"

OUTDIR="$OUTROOT/$BENCH"
mkdir -p "$OUTDIR"
if [[ ! -f "$RUN_CWD/results/LLC.llc.win.csv" ]]; then
  echo "ERROR: no CSV produced for $BENCH"; exit 2
fi

mv "$RUN_CWD/results/LLC.llc.win.csv" "$OUTDIR/LLC.llc.win.csv"
python3 tools/hybrid/extract_llc_window_features.py "$OUTDIR/LLC.llc.win.csv"
python3 tools/hybrid/select_representative_windows.py "$OUTDIR/LLC.window_features.csv" 3 0.10 1.0
echo "done -> $OUTDIR"
