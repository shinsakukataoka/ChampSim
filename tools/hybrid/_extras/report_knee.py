#!/usr/bin/env python3
import sys, pandas as pd

if len(sys.argv) < 3:
    print("Usage: report_knee.py results/eval/dataset.csv <bench_name> [pi_miss]")
    sys.exit(1)

csv  = sys.argv[1]
bench= sys.argv[2]
pm   = float(sys.argv[3]) if len(sys.argv) > 3 else None

df = pd.read_csv(csv)
sub = df[df.bench == bench].copy()
if pm is not None:
    sub = sub[sub.pi_miss == pm].copy()

if sub.empty:
    print("No rows for that selection")
    sys.exit(0)

sub = sub.sort_values("pi_way")
print(sub[["bench","pi_way","pi_miss","stall_pct","tag"]])
print("\nASCII knee:\n")
for _,r in sub.iterrows():
    bar = '#' * int(r["stall_pct"]*50)
    print(f"{r['pi_way']:>4.2f} | {bar}")

