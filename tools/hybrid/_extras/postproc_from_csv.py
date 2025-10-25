import pandas as pd
import sys

CSV = "results/LLC.llc.win.csv" if len(sys.argv) < 2 else sys.argv[1]

# Device latencies used offline
T_S  = 16.0   # SRAM hit
T_MR = 28.0   # MRAM read hit
T_MW = 60.0   # MRAM write hit
T_DR = 136.4  # from "cpu0->LLC AVERAGE MISS LATENCY: 136.4 cycles" in your log

df = pd.read_csv(CSV)

# counts per window
H_s_rd = df['hit_sram_rd'].to_numpy()
H_s_wr = df['hit_sram_wr'].to_numpy()
H_m_rd = df['hit_mram_rd'].to_numpy()
H_m_wr = df['hit_mram_wr'].to_numpy()
M      = (df['miss_rd'] + df['miss_wr']).to_numpy()

mlp_hit  = df['mlp_hit'].clip(lower=1.0).to_numpy()
mlp_miss = df['mlp_miss'].clip(lower=1.0).to_numpy()

# deltas vs SRAM
dM_rd  = T_MR - T_S
dM_wr  = T_MW - T_S
dMiss  = T_DR - T_S

stall_w = (H_m_rd*dM_rd + H_m_wr*dM_wr)/mlp_hit + (M*dMiss)/mlp_miss
stall_total = stall_w.sum()

# approximate total cycles from windows (end-start); last row dominates
cycles_total = (df['end_cycle'] - df['start_cycle']).sum()

print(f"Windows: {len(df)}")
print(f"Total stall cycles (MLP-aware): {stall_total:.0f}")
print(f"Total window cycles approx:    {cycles_total:.0f}")
print(f"stall% ≈ {100.0*stall_total/cycles_total:.2f}%")

