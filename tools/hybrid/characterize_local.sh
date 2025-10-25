#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${TRACES_ROOT:?Set TRACES_ROOT to the directory with *.champsimtrace.xz}"
command -v jq >/dev/null || { echo "need 'jq' (sudo apt install -y jq)"; exit 1; }
command -v parallel >/dev/null || { echo "need 'parallel' (sudo apt install -y parallel)"; exit 1; }
export ROOT TRACES_ROOT K STDN_THR MPKC_MIN

JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)}"
K=5            # reps (↑ from 3)
STDN_THR=0.10  # stability threshold (can relax to 0.15 if needed)
MPKC_MIN=1.0
CAPS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs) JOBS="$2"; shift 2;;
    -K|--reps) K="$2"; shift 2;;
    --stdn) STDN_THR="$2"; shift 2;;
    --mpkc) MPKC_MIN="$2"; shift 2;;
    *) CAPS+=("$1"); shift;;
  esac
done
[[ ${#CAPS[@]} -gt 0 ]] || { echo "usage: $0 [-j N] [-K 5] [--stdn 0.10] [--mpkc 1.0] <MB...>"; exit 1; }

BENCHES=(
  600.perlbench_s-570B 602.gcc_s-1850B 605.mcf_s-994B 619.lbm_s-2677B
  620.omnetpp_s-874B   621.wrf_s-6673B 623.xalancbmk_s-700B 631.deepsjeng_s-928B
  641.leela_s-1083B    648.exchange2_s-1247B 649.fotonik3d_s-7084B 657.xz_s-3167B
)

run_one() {  # MB BENCH
  local MB="$1" BENCH="$2"
  local WARM=2000000 SIM=5000000 PIWAY=0.5 PIMISS=0.5 TS=16 TMR=28 TMW=60
  local RUN_CWD="results/tmp/char_${MB}/${BENCH}_$RANDOM"
  mkdir -p "$RUN_CWD" "results/characterization_L3_${MB}/${BENCH}"

  # stage LLC capacity into a local config
  python3 - "$ROOT" "$MB" "$RUN_CWD" <<'PY'
import sys, json, os
root, mb, cwd = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cfg = json.load(open(os.path.join(root,"champsim_config.json")))
for c in cfg.get("caches",[]):
    if c.get("name","").upper()=="LLC":
        bl=int(c.get("block_size",64)); ways=int(c.get("ways",16))
        c["sets"]=int((mb*1024*1024)//(bl*ways))
json.dump(cfg, open(os.path.join(cwd,"champsim_config.json"),"w"), indent=2)
PY

  echo "[char] ${BENCH}  L3=${MB}MB"
  ( cd "$RUN_CWD" && "$ROOT/bin/champsim" \
      --warmup-instructions $WARM --simulation-instructions $SIM \
      --hybrid-llc --pi-way $PIWAY --pi-miss $PIMISS \
      --t-sram-hit $TS --t-mram-rd $TMR --t-mM-wr $TMW \
      "$TRACES_ROOT/${BENCH}.champsimtrace.xz" )

  local OUTDIR="results/characterization_L3_${MB}/${BENCH}"
  mv "$RUN_CWD/results/LLC.llc.win.csv" "$OUTDIR/LLC.llc.win.csv"
  python3 tools/hybrid/extract_llc_window_features.py "$OUTDIR/LLC.llc.win.csv" >/dev/null
  python3 tools/hybrid/select_representative_windows.py \
      "$OUTDIR/LLC.window_features.csv" "$K" "$STDN_THR" "$MPKC_MIN" >/dev/null
  echo "[char] reps=$(jq '.representatives|length' "$OUTDIR/LLC.representatives.json") -> $OUTDIR"
}

export -f run_one
# Note: Other variables are exported at the top

for MB in "${CAPS[@]}"; do
  parallel --env ROOT,TRACES_ROOT,K,STDN_THR,MPKC_MIN \
           -j "$JOBS" --halt now,fail=1 \
           tools/hybrid/char_one.sh "$MB" ::: "${BENCHES[@]}"
done
