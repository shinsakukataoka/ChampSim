#!/usr/bin/env bash
set -euo pipefail

TR="/home/skataoka26/traces/speccpu"
OUTROOT="results/characterization"
mkdir -p "$OUTROOT"

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

WARM=2000000
SIM=5000000
PIWAY=0.5
PIMISS=0.5
TS=16
TMR=28
TMW=60

for b in "${BENCHES[@]}"; do
  name="${b%.champsimtrace.xz}"
  outdir="$OUTROOT/$name"
  mkdir -p "$outdir"

  echo "== characterize: $name =="

  bin/champsim \
    --warmup-instructions "$WARM" \
    --simulation-instructions "$SIM" \
    --hybrid-llc \
    --pi-way "$PIWAY" \
    --pi-miss "$PIMISS" \
    --t-sram-hit "$TS" --t-mram-rd "$TMR" --t-mram-wr "$TMW" \
    "$TR/$b"

  if [[ ! -f results/LLC.llc.win.csv ]]; then
    echo "ERROR: results/LLC.llc.win.csv not produced for $name"; exit 1
  fi

  mv results/LLC.llc.win.csv "$outdir/LLC.llc.win.csv"

  python3 tools/hybrid/extract_llc_window_features.py "$outdir/LLC.llc.win.csv"
  python3 tools/hybrid/select_representative_windows.py "$outdir/LLC.window_features.csv" 3 0.10 1.0
done

echo "characterization complete -> $OUTROOT/<bench>/"
