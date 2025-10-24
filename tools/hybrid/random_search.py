#!/usr/bin/env python3
import os, json, random, subprocess
BENCHES = [
  "600.perlbench_s-570B","602.gcc_s-1850B","620.omnetpp_s-874B","605.mcf_s-994B",
  "623.xalancbmk_s-700B","657.xz_s-3167B","641.leela_s-1083B","631.deepsjeng_s-928B",
  "619.lbm_s-2677B","621.wrf_s-6673B","648.exchange2_s-1247B","649.fotonik3d_s-7084B"
]
N = 20  # designs per benchmark
for b in BENCHES:
    for i in range(N):
        pw = round(random.uniform(0.1,0.9), 2)
        pm = round(random.uniform(0.1,0.9), 2)
        tag = f"rand{i:03d}_pw{pw}_pm{pm}"
        cmd = ["python3","tools/hybrid/eval_design.py", b, str(pw), str(pm), "16","28","60", tag]
        print("==", " ".join(cmd))
        subprocess.run(cmd, check=True)
print("random search done")

