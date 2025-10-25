#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MB="${1:?usage: $0 <L3_MB> [-j N] (seeds HM for perlbench/leela/exchange2)}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)}"
[[ "${2:-}" == "-j" || "${2:-}" == "--jobs" ]] && { JOBS="$3"; }

BENCHES=(600.perlbench_s-570B 641.leela_s-1083B 648.exchange2_s-1247B)
PWS=(0.05 0.50 0.95); PM=0.50; TS=16; TMR=28; TMW=60

gen() {
  for b in "${BENCHES[@]}"; do
    for pw in "${PWS[@]}"; do
      echo -e "$b\t$pw\t$PM\t$TS\t$TMR\t$TMW\tseed_pw$(printf %.2f "$pw")_pm0.50"
    done
  done
}
gen | parallel -j "$JOBS" --colsep '\t' --halt now,fail=1 \
 'python3 tools/hybrid/eval_design.py {1} {2} {3} {4} {5} {6} {7}_L3_'"${MB}"'MB --l3-mb '"${MB}"

