#!/usr/bin/env python3
import sys, pandas as pd

if len(sys.argv)<3:
    print("Usage: report_pareto.py results/eval/dataset.csv <bench>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
b  = sys.argv[2]
sub = df[df.bench==b].copy()
if sub.empty:
    print("no rows for", b); sys.exit(0)

sub = sub[['pi_way','pi_miss','stall_pct','E_total_J','tag']].sort_values(['stall_pct','E_total_J'])
pareto = []
bestE = float('inf')
for _,r in sub.iterrows():
    if r['E_total_J']<=bestE:
        pareto.append(r)
        bestE = r['E_total_J']

out = pd.DataFrame(pareto)
print(out.to_string(index=False))

