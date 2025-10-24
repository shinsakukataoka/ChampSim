#!/usr/bin/env python3
import json, glob, pandas as pd

rows=[]
for f in glob.glob('results/eval/*/*/metrics.json'):
    with open(f) as h:
        rows.append(json.load(h))

if not rows:
    print("no metrics found")
else:
    out = pd.DataFrame(rows)
    out.sort_values(["bench","pi_way","pi_miss","tag"], inplace=True)
    out.to_csv("results/eval/dataset.csv", index=False)
    print("wrote results/eval/dataset.csv", len(out), "rows")

