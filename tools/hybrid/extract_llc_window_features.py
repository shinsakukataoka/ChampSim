#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys, os

# Usage:
#   python3 tools/hybrid/extract_llc_window_features.py <path/to/LLC.llc.win.csv>
# Writes:
#   <same_dir>/LLC.window_features.csv

CSV = sys.argv[1] if len(sys.argv) > 1 else "results/LLC.llc.win.csv"
df = pd.read_csv(CSV)

# basic counts
H_s_rd = df['hit_sram_rd'].to_numpy()
H_s_wr = df['hit_sram_wr'].to_numpy()
H_m_rd = df['hit_mram_rd'].to_numpy()
H_m_wr = df['hit_mram_wr'].to_numpy()
M_rd   = df['miss_rd'].to_numpy()
M_wr   = df['miss_wr'].to_numpy()
M      = M_rd + M_wr

# window cycles
win_cycles = (df['end_cycle'] - df['start_cycle']).to_numpy().astype(float)
win_cycles[win_cycles <= 0] = 1.0

# accesses & mixes
A = H_s_rd+H_s_wr+H_m_rd+H_m_wr+M
acc_per_1kcyc = A / (win_cycles/1000.0)
miss_rate     = M / np.maximum(A, 1)
read_frac     = (H_s_rd+H_m_rd+M_rd) / np.maximum(A, 1)
total_hits    = (H_s_rd+H_s_wr+H_m_rd+H_m_wr)
mram_hit_frac = (H_m_rd+H_m_wr) / np.maximum(total_hits, 1)

# MLPs
mlp_hit  = df['mlp_hit' ].to_numpy()
mlp_miss = df['mlp_miss'].to_numpy()

# offline latencies (adjust if needed)
T_S, T_MR, T_MW, T_DR = 16.0, 28.0, 60.0, 136.4
d_rd, d_wr, d_mis = T_MR-T_S, T_MW-T_S, T_DR-T_S

stall_win = (H_m_rd*d_rd + H_m_wr*d_wr)/np.maximum(mlp_hit,1) + (M*d_mis)/np.maximum(mlp_miss,1)
stall_per_1kcyc = stall_win / (win_cycles/1000.0)

# wave & stability signals
cover = df['cover_time_miss'].to_numpy()
wave_done = (cover > 0).astype(int)

mpkc = M / (win_cycles/1000.0)
roll = pd.Series(mpkc).rolling(window=5, center=True)
std5 = roll.std().fillna(method='bfill').fillna(method='ffill').to_numpy()
mean5 = pd.Series(mpkc).rolling(5, center=True).mean().fillna(method='bfill').fillna(method='ffill').to_numpy()
std5_norm = std5 / np.maximum(mean5, 1e-6)

out = pd.DataFrame({
    'window_id': df['window_id'],
    'start_cycle': df['start_cycle'],
    'end_cycle': df['end_cycle'],
    'win_cycles': win_cycles,
    'A': A,
    'M': M,
    'miss_rate': miss_rate,
    'acc_per_1kcyc': acc_per_1kcyc,
    'read_frac': read_frac,
    'mram_hit_frac': mram_hit_frac,
    'mlp_hit': mlp_hit,
    'mlp_miss': mlp_miss,
    'stall_per_1kcyc': stall_per_1kcyc,
    'mpkc': mpkc,
    'std5': std5,
    'std5_norm': std5_norm,
    'wave_done': wave_done
})

dst = os.path.join(os.path.dirname(CSV), "LLC.window_features.csv")
out.to_csv(dst, index=False)
print(f"[extract_llc_window_features] wrote {dst}")
