#!/usr/bin/env bash
set -euo pipefail

OUT="results/eval"
mkdir -p "$OUT"

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

PI_WAYS=(0.25 0.50 0.75)
PI_MISS=(0.25 0.50 0.75)

for b in "${BENCHES[@]}"; do
  for pw in "${PI_WAYS[@]}"; do
    for pm in "${PI_MISS[@]}"; do
      tag=$(printf "pw%.2f_pm%.2f" "$pw" "$pm")
      echo "== eval $b $tag =="
      python3 tools/hybrid/eval_design.py "$b" "$pw" "$pm" 16 28 60 "$tag"
    done
  done
done
