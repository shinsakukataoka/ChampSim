#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/skataoka26/ChampSim"
cd "$ROOT"

# fixed config
CAP_LIST=(2 32 128)
CPU_PART="cpu-q"
CHAR_TIME="02:00:00"
EVAL_TIME="02:00:00"
CHAR_THROTTLE=6
EVAL_THROTTLE=64
CPUS_PER_TASK=1
MEM_PER_TASK="2G"

mkdir -p logs/slurm results/tmp

# build ChampSim if needed
if [[ ! -x bin/champsim ]]; then
  echo "[build] bin/champsim missing -> make -j"
  make -j
fi

# 1) generate sweep TSV (576 lines)
python3 - <<'PY' > tools/hybrid/params_eval_all.tsv
benches=[ "600.perlbench_s-570B","602.gcc_s-1850B","620.omnetpp_s-874B","605.mcf_s-994B",
          "623.xalancbmk_s-700B","657.xz_s-3167B","641.leela_s-1083B","631.deepsjeng_s-928B",
          "619.lbm_s-2677B","621.wrf_s-6673B","648.exchange2_s-1247B","649.fotonik3d_s-7084B"]
def emit(b,pw,pm,tS,tMr,tMw,tag): print(b,pw,pm,tS,tMr,tMw,tag,sep="\t")
# Dense pi_way for pm=0.25/0.50/0.75
pways=[round(x/10,2) for x in range(1,10)]
for b in benches:
  for pm in (0.25,0.50,0.75):
    for pw in pways: emit(b,pw,pm,16,28,60,f"pway{pw:.2f}_pm{pm:.2f}")
# tS sensitivity at three pi_way (pm=0.50)
for b in benches:
  for pw in (0.25,0.50,0.75):
    for tS in (12,20): emit(b,pw,0.50,tS,28,60,f"latSRAM{tS}_pw{pw:.2f}")
# Wider tMr edges (pm=0.50)
for b in benches:
  for pw in (0.25,0.50,0.75):
    for tMr in (18,22,45): emit(b,pw,0.50,16,tMr,60,f"latRD{tMr}_pw{pw:.2f}")
# Wider tMw edges (pm=0.50)
for b in benches:
  for pw in (0.25,0.50,0.75):
    for tMw in (100,120): emit(b,pw,0.50,16,28,tMw,f"latWR{tMw}_pw{pw:.2f}")
PY
N=$(wc -l < tools/hybrid/params_eval_all.tsv)
echo "[params] tools/hybrid/params_eval_all.tsv lines = $N"

# 2) submit characterization arrays per capacity (GPU nodes for CPU time)
for MB in "${CAP_LIST[@]}"; do
  echo "[submit] characterization L3_${MB}MB"
  sbatch -p "$CPU_PART" \
         --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" --time="$CHAR_TIME" \
         --array=1-12%${CHAR_THROTTLE} \
         tools/hybrid/sbatch_characterize_mb.sh "$MB"
done

# 3) submit eval arrays per capacity (tags auto-suffixed by CLI args)
for MB in "${CAP_LIST[@]}"; do
  echo "[submit] eval L3_${MB}MB"
  sbatch -p "$CPU_PART" \
         --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" --time="$EVAL_TIME" \
         --array=1-"$N"%${EVAL_THROTTLE} --job-name="hyb_eval_gpu_L3_${MB}" \
         tools/hybrid/sbatch_eval_array.sh tools/hybrid/params_eval_all.tsv "$MB" "L3_${MB}MB"
done

echo "[done] submitted characterization + eval arrays for: ${CAP_LIST[*]}"
echo "Tip: squeue -u $USER -o \"%.18i %.9P %.8T %.10M %.12K %.20R\" --array"
