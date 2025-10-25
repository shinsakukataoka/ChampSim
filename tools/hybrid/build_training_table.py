#!/usr/bin/env python3
import os, pandas as pd

DS  = "results/eval/dataset.csv"
FP  = "results/surrogate/benchmark_fingerprints.csv"
OUT = "results/surrogate/training_table.csv"

df  = pd.read_csv(DS).dropna(subset=["stall_pct"])
fp  = pd.read_csv(FP)

knob_cols = ["pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr"]
label_cols= ["stall_pct","E_total_J"]
meta_cols = ["bench","tag"]

tbl = df[meta_cols + knob_cols + label_cols].merge(fp, on="bench", how="left")

# add/compute richer features if available
if 't_dram' in df.columns:
    tbl['t_dram'] = df['t_dram']
else:
    tbl['t_dram'] = 0.0

if 'l3_mb' in df.columns:
    tbl['l3_mb'] = df['l3_mb']
else:
    tbl['l3_mb'] = 8.0

tbl['pi_way_x_pi_miss'] = tbl['pi_way'] * tbl['pi_miss']

tbl = tbl.dropna(subset=["char_miss_rate"]).sort_values(["bench","pi_way","pi_miss","tag"])

os.makedirs("results/surrogate", exist_ok=True)
tbl.to_csv(OUT, index=False)
print("wrote", OUT, len(tbl), "rows")
