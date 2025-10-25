#!/usr/bin/env bash
set -euo pipefail

MB="${1:-8}"
TR="/home/skataoka26/traces/speccpu"
OUTROOT="results/characterization_L3_${MB}"
mkdir -p "$OUTROOT"

# Nominal policy/latency for characterization
WARM=2000000; SIM=5000000
PIWAY=0.5; PIMISS=0.5
TS=16; TMR=28; TMW=60

BENCHES=(
  600.perlbench_s-570B.champsimtrace.xz
  602.gcc_s-1850B.champsimtrace.xz
  620.omnetpp_s-874B.champsimtrace.xz
  605.mcf_s-994B.champsimtrace.xz
  623.xalancbmk_s-700B.champsimtrace.xz
  657.xz_s-3167B.champsimtrace.xz
  641.leela_s-1083B.champsimtrace.xz
  631.deepsjeng_s-928B.champsimtrace.xz
  619.lbm_s-2677B.champsimtrace.xz
  621.wrf_s-6673B.champsimtrace.xz
  648.exchange2_s-1247B.champsimtrace.xz
  649.fotonik3d_s-7084B.champsimtrace.xz
)

ROOT="/home/skataoka26/ChampSim"

for f in "${BENCHES[@]}"; do
  name="${f%.champsimtrace.xz}"
  outdir="$OUTROOT/$name"
  mkdir -p "$outdir"

  # unique working dir; stage capacity-specific config there
  run_cwd="results/tmp/char_${MB}/${name}_$RANDOM"
  mkdir -p "$run_cwd"

  # write a config with LLC≈MB into the run_cwd
  python3 - "$ROOT" "$MB" "$run_cwd" <<'PY'
import json, sys, os
root, mb, cwd = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cfg = json.load(open(os.path.join(root, "champsim_config.json")))
for c in cfg.get("caches", []):
    if c.get("name","").upper()=="LLC":
        bl=int(c.get("block_size",64)); ways=int(c.get("ways",16))
        sets=int((mb*1024*1024)//(bl*ways)); c["sets"]=sets
with open(os.path.join(cwd,"champsim_config.json"),"w") as f: json.dump(cfg,f,indent=2)
PY

  echo "== characterize: $name  (L3=${MB}MB) =="
  ( cd "$run_cwd" && "$ROOT/bin/champsim" \
      --warmup-instructions "$WARM" \
      --simulation-instructions "$SIM" \
      --hybrid-llc \
      --pi-way "$PIWAY" --pi-miss "$PIMISS" \
      --t-sram-hit "$TS" --t-mram-rd "$TMR" --t-mram-wr "$TMW" \
      "$TR/$f" )

  if [[ ! -f "$run_cwd/results/LLC.llc.win.csv" ]]; then
    echo "ERROR: $name produced no CSV"; exit 1
  fi

  mv "$run_cwd/results/LLC.llc.win.csv" "$outdir/LLC.llc.win.csv"
  python3 tools/hybrid/extract_llc_window_features.py "$outdir/LLC.llc.win.csv"
  python3 tools/hybrid/select_representative_windows.py "$outdir/LLC.window_features.csv" 3 0.10 1.0
done

echo "characterization complete -> $OUTROOT/<bench>/"

