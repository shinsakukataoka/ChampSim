#!/usr/bin/env python3
import argparse, os, subprocess, csv, shutil, json

def read_row(csv_path, row_idx):
    # 0-based row_idx over *data* rows (skip header)
    with open(csv_path, newline="") as fp:
        r = list(csv.DictReader(fp))
    return r[row_idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="/home/skataoka26/ChampSim/out/configs.csv")
    ap.add_argument("--row", type=int, required=True, help="0-based index over data rows")
    ap.add_argument("--out-root", default="/home/skataoka26/ChampSim/out/sims")
    ap.add_argument("--champsim-root", default="/home/skataoka26/ChampSim")
    args = ap.parse_args()

    row = read_row(args.configs, args.row)
    bench = row["bench"]
    config_id = int(row["config_id"])
    tr = row["trace_path"]
    exec_bin = row["exec_bin"]
    warmup = int(row["warmup"])
    sim = int(row["sim"])
    pi_miss = row["pi_miss"]; pi_way = row["pi_way"]
    tS = row["tS"]; tMr = row["tMr"]; tMw = row["tMw"]
    is_baseline = int(row["is_baseline"])

    out_dir = os.path.join(args.out_root, bench, f"cfg_{config_id}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        exec_bin,
        "--hide-heartbeat",
        "--hybrid-llc",
        "--pi-miss", str(pi_miss),
        "--pi-way",  str(pi_way),
        "--pi-read", "0.0",
        "--pi-write","0.0",
        "--t-sram-hit", str(tS),
        "--t-mram-rd",  str(tMr),
        "--t-mram-wr",  str(tMw),
        "--warmup-instructions", str(warmup),
        "--simulation-instructions", str(sim),
        "--json", "stats.json",
        tr
    ]

    with open(os.path.join(out_dir, "run.log"), "w") as logf:
        subprocess.run(cmd, cwd=out_dir, check=True, stdout=logf, stderr=subprocess.STDOUT)

    # Expect window CSV at results/LLC.llc.win.csv
    win_csv = os.path.join(out_dir, "results", "LLC.llc.win.csv")
    if not os.path.exists(win_csv):
        raise SystemExit(f"Expected window CSV not found at {win_csv}. Did you run with --hybrid-llc?")

    # Save a small meta.json
    meta = dict(
        config_id=config_id, bench=bench, trace_path=tr, is_baseline=is_baseline,
        pi_miss=float(pi_miss), pi_way=float(pi_way), tS=int(tS), tMr=int(tMr), tMw=int(tMw),
        warmup=warmup, sim=sim, exec_bin=exec_bin
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    print(f"Done: {bench} cfg_{config_id} -> {out_dir}")

if __name__ == "__main__":
    main()
