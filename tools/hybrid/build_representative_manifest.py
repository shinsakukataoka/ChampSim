#!/usr/bin/env python3
import os, json, glob

ROOT = "results/characterization"
out = {"benchmarks": {}}

for jpath in glob.glob(os.path.join(ROOT, "*", "LLC.representatives.json")):
    bench = jpath.split(os.sep)[-2]
    with open(jpath) as f:
        data = json.load(f)
    out["benchmarks"][bench] = {
        "representatives": data.get("representatives", []),
        "stable_windows": data.get("stable_windows", []),
        "n_clusters": data.get("n_clusters", 0)
    }

dst = os.path.join(ROOT, "representatives_manifest.json")
with open(dst, "w") as f:
    json.dump(out, f, indent=2)
print(f"[build_representative_manifest] wrote {dst} with {len(out['benchmarks'])} benchmarks")
