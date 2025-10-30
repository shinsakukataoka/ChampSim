#!/usr/bin/env python3
import argparse, json, os, subprocess, tempfile
ap=argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--datasets-root", default="/home/skataoka26/ChampSim/out/datasets")
ap.add_argument("--model-dir", default="/home/skataoka26/ChampSim/model")
ap.add_argument("--out", default=None)
args=ap.parse_args()
cfg=json.load(open(args.config))

bench   = cfg["bench"]
sim     = cfg.get("sim_instrs", None)
pi_miss = cfg["policy"]["pi_miss"]; pi_way = cfg["policy"]["pi_way"]
tS      = cfg["latency"]["tS"];     tMr    = cfg["latency"]["tMr"]; tMw = cfg["latency"]["tMw"]

# If user supplies a different energy table, temporarily swap it in for predict.py
energy_tbl = cfg.get("energy", {}).get("table")
energy_path= cfg.get("energy", {}).get("table_path", "/home/skataoka26/ChampSim/model/energy_table.json")
tmp_path   = None
if energy_tbl is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    json.dump(energy_tbl, open(tmp.name, "w"), indent=2)
    tmp_path = tmp.name
    energy_path = tmp_path

# predict.py reads the fixed path; briefly point it there by setting ENV var for subprocess if you want,
# or just rely on your standard file path (we’ll copy if needed)
env=os.environ.copy()
env["ENERGY_TABLE_PATH"] = energy_path  # predict.py uses default path; if you later modify it to read ENV, this works.

cmd = [
  "python3", "/home/skataoka26/ChampSim/tools/pred/predict.py",
  "--bench", bench, "--pi-miss", str(pi_miss), "--pi-way", str(pi_way),
  "--tS", str(tS), "--tMr", str(tMr), "--tMw", str(tMw)
]
if sim: cmd += ["--sim", str(sim)]
if args.out: cmd += ["--out", args.out]
subprocess.run(cmd, env=env, check=True)
if tmp_path: os.unlink(tmp_path)
