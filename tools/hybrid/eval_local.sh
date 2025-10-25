#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)}"
TSV="${1:-tools/hybrid/params_eval_all.tsv}"
MB="${2:?usage: $0 <params.tsv> <L3_MB> [-j N]}"
shift 2 || true
[[ "${1:-}" == "-j" || "${1:-}" == "--jobs" ]] && { JOBS="$2"; shift 2; }

echo "[eval] TSV=$TSV  L3=${MB}MB  jobs=$JOBS"

# TSV columns: bench  pi_way  pi_miss  tS  tMr  tMw  tag
parallel -j "$JOBS" --colsep '\t' --halt now,fail=1 \
 'python3 tools/hybrid/eval_design.py {1} {2} {3} {4} {5} {6} {7}_L3_'"${MB}"'MB --l3-mb '"${MB}" \
 :::: "$TSV"

