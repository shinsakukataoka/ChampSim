#!/usr/bin/env python3
# Writes a TSV to stdout (no header): bench  pi_way  pi_miss  tS  tMr  tMw  tag
benches = [
  "600.perlbench_s-570B",
  "602.gcc_s-1850B",
  "620.omnetpp_s-874B",
  "605.mcf_s-994B",
  "623.xalancbmk_s-700B",
  "657.xz_s-3167B",
  "641.leela_s-1083B",
  "631.deepsjeng_s-928B",
  "619.lbm_s-2677B",
  "621.wrf_s-6673B",
  "648.exchange2_s-1247B",
  "649.fotonik3d_s-7084B",
]
pi_ways  = [0.25, 0.50, 0.75]
pi_miss  = [0.25, 0.50, 0.75]
tS, tMr, tMw = 16, 28, 60

rows = 0
for b in benches:
    for pw in pi_ways:
        for pm in pi_miss:
            tag = f"pw{pw:.2f}_pm{pm:.2f}"
            print(b, pw, pm, tS, tMr, tMw, tag, sep="\t")
            rows += 1
# (optional) print count to stderr so the stdout stays clean TSV
import sys
print(f"# rows={rows}", file=sys.stderr)

