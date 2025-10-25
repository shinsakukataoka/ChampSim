#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:?}"; TRACES_ROOT="${TRACES_ROOT:?}"

MB="$1"; BENCH="$2"
WARM=2000000; SIM=5000000; PIWAY=0.5; PIMISS=0.5; TS=16; TMR=28; TMW=60

RUN_CWD="results/tmp/char_${MB}/${BENCH}_$RANDOM"
mkdir -p "$RUN_CWD" "results/characterization_L3_${MB}/${BENCH}"

# stage LLC capacity
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
    --t-sram-hit $TS --t-mram-rd $TMR --t-mram-wr $TMW \
    "$TRACES_ROOT/${BENCH}.champsimtrace.xz" )

OUTDIR="results/characterization_L3_${MB}/${BENCH}"
mv "$RUN_CWD/results/LLC.llc.win.csv" "$OUTDIR/LLC.llc.win.csv"
python3 tools/hybrid/extract_llc_window_features.py "$OUTDIR/LLC.llc.win.csv" >/dev/null
python3 tools/hybrid/select_representative_windows.py \
    "$OUTDIR/LLC.window_features.csv" "${K:-5}" "${STDN_THR:-0.10}" "${MPKC_MIN:-1.0}" >/dev/null
echo "[char] reps=$(jq '.representatives|length' "$OUTDIR/LLC.representatives.json") -> $OUTDIR"

