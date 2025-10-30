#!/usr/bin/env python3
import argparse, os, glob, csv, math, random
import numpy as np

def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, round(x))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--exec-bin", default="/home/skataoka26/ChampSim/bin/champsim_L3_2MB")
    ap.add_argument("--out", default="/home/skataoka26/ChampSim/out/configs.csv")
    ap.add_argument("--N", type=int, default=10, help="LHS samples per benchmark (not counting baseline/anchors)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--warmup", type=int, default=200_000_000)
    ap.add_argument("--sim", type=int, default=500_000_000)
    # tech-faithful global boxes
    ap.add_argument("--tS_lo", type=int, default=6)
    ap.add_argument("--tS_hi", type=int, default=40)
    ap.add_argument("--tMr_lo", type=int, default=12)
    ap.add_argument("--tMr_hi", type=int, default=80)
    ap.add_argument("--tMw_lo", type=int, default=32)
    ap.add_argument("--tMw_hi", type=int, default=120)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    traces = sorted(glob.glob(os.path.join(args.traces_dir, "*.champsimtrace.xz")))
    if not traces:
        raise SystemExit(f"No traces found in {args.traces_dir}")

    # utility: sample LHS in [0,1], then scale to ranges
    def lhs_u01(dim, n):
        # simple stratified + random within bin + random permutation per dim
        base = (np.arange(n) + np.random.rand(n)) / n
        X = np.zeros((n, dim))
        for d in range(dim):
            X[:, d] = np.random.permutation(base)
        return X

    rows = []
    config_id = 0
    for tr in traces:
        bench = os.path.basename(tr)
        # baseline
        rows.append(dict(
            config_id=config_id, bench=bench, trace_path=tr, is_baseline=1,
            pi_miss=0.5, pi_way=0.5, tS=16, tMr=28, tMw=60,
            warmup=args.warmup, sim=args.sim, exec_bin=args.exec_bin
        ))
        config_id += 1

        # LHS samples
        X = lhs_u01(5, args.N)  # pi_miss, pi_way, tS, tMr, tMw
        for i in range(args.N):
            pi_miss = float(X[i,0])                       # [0,1]
            pi_way  = float(X[i,1])                       # [0,1]
            tS  = args.tS_lo  + X[i,2]*(args.tS_hi-args.tS_lo)
            tMr = args.tMr_lo + X[i,3]*(args.tMr_hi-args.tMr_lo)
            tMw = args.tMw_lo + X[i,4]*(args.tMw_hi-args.tMw_lo)
            # enforce tS <= tMr <= tMw (monotone latencies)
            vals = sorted([tS, tMr, tMw])
            tS, tMr, tMw = [clamp_int(v, lo, hi) for v, lo, hi in
                            [(vals[0], args.tS_lo, args.tS_hi),
                             (vals[1], args.tMr_lo, args.tMr_hi),
                             (vals[2], args.tMw_lo, args.tMw_hi)]]
            rows.append(dict(
                config_id=config_id, bench=bench, trace_path=tr, is_baseline=0,
                pi_miss=round(pi_miss,5), pi_way=round(pi_way,5),
                tS=tS, tMr=tMr, tMw=tMw,
                warmup=args.warmup, sim=args.sim, exec_bin=args.exec_bin
            ))
            config_id += 1

        # two anchors at pi_miss {0,1}
        for pi_miss in (0.0, 1.0):
            rows.append(dict(
                config_id=config_id, bench=bench, trace_path=tr, is_baseline=0,
                pi_miss=pi_miss, pi_way=0.5, tS=16, tMr=28, tMw=60,
                warmup=args.warmup, sim=args.sim, exec_bin=args.exec_bin
            ))
            config_id += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} configs to {args.out}")

if __name__ == "__main__":
    main()
