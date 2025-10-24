#!/usr/bin/env bash
set -euo pipefail

TR="/home/skataoka26/traces/speccpu"
OUTDIR="logs"
mkdir -p "$OUTDIR"

# MUST benchmarks (the ones you downloaded)
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

PI_VALUES="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

for f in "${BENCHES[@]}"; do
  for p in $PI_VALUES; do
    echo "== $f  pi_miss=$p =="
    bin/champsim \
      --warmup-instructions 2000000 \
      --simulation-instructions 5000000 \
      --hybrid-llc \
      --pi-miss "$p" --t-sram-hit 16 --t-mram-rd 28 --t-mram-wr 60 \
      "$TR/$f" | tee "$OUTDIR/${f%.xz}.pi${p}.log"
  done
done
