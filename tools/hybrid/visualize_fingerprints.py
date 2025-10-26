#!/usr/bin/env python3
"""
Visualize benchmark fingerprints (per-capacity) from:
  results/surrogate/L3_<cap>/benchmark_fingerprints.csv

Two modes:
  - heatmap : benches × features heatmap (normalized per feature)
  - radar   : per-bench radar/spider plot (normalized per feature)

Examples:
  python3 tools/hybrid/visualize_fingerprints.py --cap 32 --mode heatmap --top 12
  python3 tools/hybrid/visualize_fingerprints.py --cap 32 --mode radar --benches 602.gcc_s-1850B,620.omnetpp_s-874B
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = [
    "char_miss_rate",
    "char_acc_per_1kcyc",
    "char_read_frac",
    "char_mram_hit_frac",
    "char_mlp_hit",
    "char_mlp_miss",
    "char_mpkc",
    "char_stall_per_1kcyc",
]

def load_fingerprints(cap:int):
    p = os.path.join("results","surrogate",f"L3_{cap}","benchmark_fingerprints.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing fingerprints: {p}")
    df = pd.read_csv(p)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"fingerprints missing columns: {missing}")
    return df

def minmax_normalize(df_feat: pd.DataFrame):
    # per feature min-max; handle constant columns
    mins = df_feat.min(axis=0)
    maxs = df_feat.max(axis=0)
    rng  = (maxs - mins).replace(0, 1.0)
    norm = (df_feat - mins) / rng
    return norm, mins, maxs

def plot_heatmap(df, benches, outpng, title):
    X = df.set_index("bench").loc[benches, FEATURES]
    Xn, mins, maxs = minmax_normalize(X)

    fig, ax = plt.subplots(figsize=(min(16, 1.1*len(FEATURES)+4), 0.45*len(benches)+2.5))
    im = ax.imshow(Xn.values, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(benches)))
    ax.set_yticklabels([b.split('.',1)[-1] for b in benches])
    ax.set_title(title)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("normalized (per feature min–max)", rotation=90)

    # optional: annotate raw (de-normalized) values under the hood?
    # Keep clean by default; uncomment if you want numbers:
    # for i in range(len(benches)):
    #     for j in range(len(FEATURES)):
    #         ax.text(j, i, f"{X.iloc[i, j]:.3g}", va="center", ha="center", color="w" if Xn.iloc[i,j]>0.5 else "k", fontsize=7)

    fig.tight_layout()
    fig.savefig(outpng, dpi=180)
    print(f"[ok] wrote {outpng}")

def plot_radar(df, bench, outpng, title):
    # one bench radar
    x = df.set_index("bench").loc[bench, FEATURES]
    # normalize against all benches min-max (so multiple radars comparable)
    Xall = df[FEATURES]
    Xn, mins, maxs = minmax_normalize(Xall)
    xn = Xn.iloc[df.index[df["bench"]==bench][0]]

    # radar axis
    labels = FEATURES
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    vals = xn.values
    vals = np.concatenate([vals, vals[:1]])
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.subplots(figsize=(6.5,6.5), subplot_kw=dict(polar=True))[1]
    fig.set_title(title, y=1.08)
    fig.plot(angles, vals, color="tab:blue", linewidth=2)
    fig.fill(angles, vals, color="tab:blue", alpha=0.20)
    fig.set_thetagrids(angles[:-1] * 180/np.pi, labels, fontsize=9)
    fig.set_rlim(0,1)
    fig.set_rlabel_position(0)
    fig.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpng, dpi=180)
    print(f"[ok] wrote {outpng}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--mode", choices=["heatmap","radar"], default="heatmap")
    ap.add_argument("--benches", default="", help="comma list; if empty, auto-pick")
    ap.add_argument("--top", type=int, default=12, help="heatmap: take top-N benches by mpkc if --benches empty")
    args = ap.parse_args()

    df = load_fingerprints(args.cap)
    figs_dir = os.path.join("results","figs", f"L3_{args.cap}")
    os.makedirs(figs_dir, exist_ok=True)

    if args.benches.strip():
        benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    else:
        # auto-pick top-N by mpkc (common proxy for “busy” benches)
        benches = df.sort_values("char_mpkc", ascending=False)["bench"].head(args.top).tolist()

    if args.mode == "heatmap":
        outpng = os.path.join(figs_dir, f"fingerprints_heatmap_top{len(benches)}.png")
        title  = f"Benchmark fingerprints (L3={args.cap}MB) — min–max normalized"
        plot_heatmap(df, benches, outpng, title)
    else:
        # Radar: one file per bench
        for b in benches:
            outpng = os.path.join(figs_dir, f"fingerprint_radar_{b.replace('/','-')}.png")
            title  = f"{b} — fingerprint (L3={args.cap}MB, normalized)"
            plot_radar(df, b, outpng, title)

if __name__ == "__main__":
    main()

