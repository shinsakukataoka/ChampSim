#!/usr/bin/env python3
"""
Visualize f_crit least-squares fit:
Scatter of (S, y) with the fitted line y = f_crit * S, colored by split (train/val).

Inputs created by train.py:
  - model/diag/<bench>_fcrit_points.csv  # columns: config_id, window_id, S, y, split
  - model/<bench>_consts.json            # contains {"f_crit": <value>}

Usage examples:
  python3 tools/pred/plot_fcrit.py --bench 602.gcc_s-1850B.champsimtrace.xz
  python3 tools/pred/plot_fcrit.py --rows 3 --cols 4
"""
import argparse, os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHORT = {
  "400.perlbench-41B.champsimtrace.xz":"perlbench",
  "600.perlbench_s-570B.champsimtrace.xz":"perlbench_s",
  "602.gcc_s-1850B.champsimtrace.xz":"gcc",
  "605.mcf_s-994B.champsimtrace.xz":"mcf",
  "619.lbm_s-2677B.champsimtrace.xz":"lbm",
  "620.omnetpp_s-874B.champsimtrace.xz":"omnetpp",
  "621.wrf_s-6673B.champsimtrace.xz":"wrf",
  "623.xalancbmk_s-700B.champsimtrace.xz":"xalancbmk",
  "631.deepsjeng_s-928B.champsimtrace.xz":"deepsjeng",
  "641.leela_s-1083B.champsimtrace.xz":"leela",
  "648.exchange2_s-1247B.champsimtrace.xz":"exchange2",
  "649.fotonik3d_s-7084B.champsimtrace.xz":"fotonik3d",
  "657.xz_s-3167B.champsimtrace.xz":"xz",
}

def list_benches(model_dir):
    paths = sorted(glob.glob(os.path.join(model_dir, "diag", "*_fcrit_points.csv")))
    return [os.path.basename(p).replace("_fcrit_points.csv","") for p in paths]

def load_points(model_dir, bench):
    p = os.path.join(model_dir, "diag", f"{bench}_fcrit_points.csv")
    if not os.path.exists(p):
        raise SystemExit(f"Missing {p} (rerun train.py after adding the diagnostic dump)")
    df = pd.read_csv(p)
    # Ensure types
    df["S"] = df["S"].astype(float)
    df["y"] = df["y"].astype(float)
    return df

def load_fcrit(model_dir, bench):
    p = os.path.join(model_dir, f"{bench}_consts.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("f_crit", None)

def ls_slope(S, y):
    num = float(np.dot(S, y))
    den = float(np.dot(S, S)) if np.any(S!=0) else 0.0
    return num/den if den > 0 else np.nan

def r2_score_line(S, y, slope):
    yhat = slope * S
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) if len(y) else 0.0
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

def plot_one(ax, df, bench, f_file):
    # Color by split
    colors = {"train":"C0", "val":"C1", "other":"C2"}
    for sp, sub in df.groupby("split"):
        ax.scatter(sub["S"], sub["y"], s=10, alpha=0.6, label=sp, color=colors.get(sp,"C2"))
    # Compute slope from points as sanity
    S = df["S"].to_numpy()
    y = df["y"].to_numpy()
    f_ls = ls_slope(S, y)
    r2_ls = r2_score_line(S, y, f_ls)
    # Plot lines: the stored f_crit (if any) and the LS recomputed
    xx = np.linspace(S.min() if len(S) else -1, S.max() if len(S) else 1, 100)
    if f_file is not None:
        ax.plot(xx, f_file*xx, lw=2, color="k", label=f"file f_crit={f_file:.3f}")
    if not np.isnan(f_ls):
        ax.plot(xx, f_ls*xx, lw=1.5, color="C3", linestyle="--", label=f"LS slope={f_ls:.3f}, R²={r2_ls:.3f}")
    ax.set_title(SHORT.get(bench, bench), fontsize=10)
    ax.set_xlabel("S = cycles^(f=1) − cycles_base")
    ax.set_ylabel("y = cycles_true − cycles_base")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/skataoka26/ChampSim/model")
    ap.add_argument("--out-dir",   default="/home/skataoka26/ChampSim/out/figs")
    ap.add_argument("--bench",     default=None, help="single bench; else draw a grid")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.bench:
        df = load_points(args.model_dir, args.bench)
        f_file = load_fcrit(args.model_dir, args.bench)
        fig, ax = plt.subplots(figsize=(5,4))
        plot_one(ax, df, args.bench, f_file)
        outp = os.path.join(args.out_dir, f"fcrit_{SHORT.get(args.bench, args.bench)}.png")
        fig.tight_layout(); fig.savefig(outp, dpi=160); plt.close(fig)
        print(f"[plot_fcrit] Saved {outp}")
        return

    benches = list_benches(args.model_dir)[:args.rows*args.cols]
    fig, axs = plt.subplots(args.rows, args.cols, figsize=(args.cols*4, args.rows*3), squeeze=False)
    for k, bench in enumerate(benches):
        r, c = divmod(k, args.cols)
        df = load_points(args.model_dir, bench)
        f_file = load_fcrit(args.model_dir, bench)
        plot_one(axs[r,c], df, bench, f_file)
    # hide unused
    for k in range(len(benches), args.rows*args.cols):
        r,c = divmod(k, args.cols); axs[r,c].axis('off')
    fig.suptitle("f_crit least-squares diagnostics", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    outp = os.path.join(args.out_dir, "fcrit_grid.png")
    fig.savefig(outp, dpi=160); plt.close(fig)
    print(f"[plot_fcrit] Saved {outp}")

if __name__ == "__main__":
    main()

