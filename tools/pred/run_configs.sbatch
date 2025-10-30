#!/bin/bash
#SBATCH -J csim_lhs
#SBATCH -p cpu-q
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --output=/home/skataoka26/ChampSim/out/sims/slurm_%A_%a.out
#SBATCH --error=/home/skataoka26/ChampSim/out/sims/slurm_%A_%a.err

# Usage to submit:
# N=$(($(wc -l < /home/skataoka26/ChampSim/out/configs.csv)-1))
# sbatch --array=0-$((N-1)) /home/skataoka26/ChampSim/tools/pred/run_configs.sbatch

ROW=${SLURM_ARRAY_TASK_ID}
python3 /home/skataoka26/ChampSim/tools/pred/run_one.py \
  --row ${ROW} \
  --configs /home/skataoka26/ChampSim/out/configs.csv \
  --out-root /home/skataoka26/ChampSim/out/sims \
  --champsim-root /home/skataoka26/ChampSim
