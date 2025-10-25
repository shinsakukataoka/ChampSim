# DROP-IN replacement for tools/hybrid/device_sweep_and_bounds.sh
#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/skataoka26/ChampSim"; cd "$ROOT"

DEVICE_ROOT="devices"
LATENCY_MODE="device"          # default: use device latencies
WITH_BOUNDS=1                  # compute policy bounds after postprocess
CAPS=""                        # optional: comma-separated, e.g. "32" or "2,128"

# parse flags
PROFILES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device-root) DEVICE_ROOT="$2"; shift 2;;
    --latency-mode) LATENCY_MODE="$2"; shift 2;;
    --caps) CAPS="$2"; shift 2;;
    --no-bounds) WITH_BOUNDS=0; shift;;
    -h|--help) echo "usage: $0 [--device-root DIR] [--latency-mode dataset|device] [--caps 32|2,128] [--no-bounds] <profile1> [profile2 ...]"; exit 0;;
    *) PROFILES+=("$1"); shift;;
  esac
done
[[ ${#PROFILES[@]} -gt 0 ]] || { echo "need at least one profile"; exit 1; }

# build cap flag if provided
CAP_FLAG=()
[[ -n "$CAPS" ]] && CAP_FLAG=(--caps "$CAPS")

for P in "${PROFILES[@]}"; do
  echo "=== DEVICE PROFILE: $P ==="
  bash tools/hybrid/postprocess_after_eval.sh \
       --device-root "$DEVICE_ROOT" \
       --device-profile "$P" \
       --latency-mode "$LATENCY_MODE" \
       "${CAP_FLAG[@]}"

  [[ $WITH_BOUNDS -eq 1 ]] && python3 tools/hybrid/policy_bounds.py \
  --device-root "$DEVICE_ROOT" \
  --device-profile "$P" \
  --latency-mode "$LATENCY_MODE" \
  "${CAP_FLAG[@]}"

  # archive outputs per profile/cap
  for f in results/eval/dataset_L3_*MB.csv; do
    [[ -e "$f" ]] || continue
    cap=$(basename "$f" | sed -E 's/.*_L3_([0-9]+)MB.*/\1/')
    out="results/surrogate/profiles/$P/L3_${cap}"
    mkdir -p "$out"
    find "results/surrogate/L3_${cap}" -maxdepth 1 -type f -exec cp -f {} "$out"/ \;
  done
done
echo "[done] profiles: ${PROFILES[*]}"

