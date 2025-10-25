#!/usr/bin/env python3
import os, json, glob
import pandas as pd

CHAR_ROOT = "results/characterization"
rows=[]
for feat_path in glob.glob(os.path.join(CHAR_ROOT, "*", "LLC.window_features.csv")):
    bench = feat_path.split(os.sep)[-2]
    reps_path = os.path.join(CHAR_ROOT, bench, "LLC.representatives.json")
    if not os.path.exists(reps_path): continue
    reps = json.load(open(reps_path)).get("representatives", [])
    df = pd.read_csv(feat_path)
    if reps: df = df[df["window_id"].isin(reps)].copy()
    if df.empty: continue
    rows.append({
        "bench": bench,
        "char_miss_rate": df["miss_rate"].mean(),
        "char_acc_per_1kcyc": df["acc_per_1kcyc"].mean(),
        "char_read_frac": df["read_frac"].mean(),
        "char_mram_hit_frac": df["mram_hit_frac"].mean(),
        "char_mlp_hit": df["mlp_hit"].mean(),
        "char_mlp_miss": df["mlp_miss"].mean(),
        "char_mpkc": df["mpkc"].mean(),
        "char_stall_per_1kcyc": df["stall_per_1kcyc"].mean(),
    })

out = pd.DataFrame(rows).sort_values("bench")
os.makedirs("results/surrogate", exist_ok=True)
out.to_csv("results/surrogate/benchmark_fingerprints.csv", index=False)
print("wrote results/surrogate/benchmark_fingerprints.csv", len(out), "benchmarks")

