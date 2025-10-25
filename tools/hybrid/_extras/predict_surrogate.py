#!/usr/bin/env python3
import os, sys, json, joblib, pandas as pd, numpy as np

if len(sys.argv)<8:
    print("Usage: predict_surrogate.py <bench> <pi_way> <pi_miss> <tS> <tMr> <tMw> results/surrogate/benchmark_fingerprints.csv")
    sys.exit(1)

bench, pw, pm = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
tS, tMr, tMw  = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
fp_csv        = sys.argv[7]

stall_path_b = f"results/surrogate/stallpct_hgb_{bench}.joblib"
ener_path_b  = f"results/surrogate/energy_hgb_{bench}.joblib"
stall_path_g = "results/surrogate/surrogate_stallpct_hgb_global.joblib"
ener_path_g  = "results/surrogate/surrogate_energy_hgb_global.joblib"

if os.path.exists(stall_path_b) and os.path.exists(ener_path_b):
    stall = joblib.load(stall_path_b); energy = joblib.load(ener_path_b)
else:
    stall = joblib.load(stall_path_g); energy = joblib.load(ener_path_g)
feat_cols = stall["feat_cols"]

fp = pd.read_csv(fp_csv)
row = fp[fp["bench"]==bench]
if row.empty:
    print(json.dumps({"error": f"no fingerprint for {bench}"})); sys.exit(1)
row = row.iloc[0]

x = {
 "pi_way": pw, "pi_miss": pm, "pi_way_x_pi_miss": pw*pm,
 "t_sram_hit": tS, "t_mram_rd": tMr, "t_mram_wr": tMw, "t_dram": 0.0, "l3_mb": 8.0,
 "char_miss_rate": row["char_miss_rate"], "char_acc_per_1kcyc": row["char_acc_per_1kcyc"],
 "char_read_frac": row["char_read_frac"], "char_mram_hit_frac": row["char_mram_hit_frac"],
 "char_mlp_hit": row["char_mlp_hit"], "char_mlp_miss": row["char_mlp_miss"],
 "char_mpkc": row["char_mpkc"], "char_stall_per_1kcyc": row["char_stall_per_1kcyc"],
}
X = np.array([[x[c] for c in feat_cols]])
print(json.dumps({
  "bench": bench, "pi_way": pw, "pi_miss": pm,
  "stall_pct": float(stall["model"].predict(X)[0]),
  "E_total_J": float(energy["model"].predict(X)[0])
}, indent=2))
