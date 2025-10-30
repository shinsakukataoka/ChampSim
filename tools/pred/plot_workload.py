#!/usr/bin/env python3
"""
Workload characterization plots.

Outputs (saved under --out-dir):
  - dynamics_A_tot.png        # 3x4 grid: A_tot time series per bench
  - dynamics_P_miss.png       # 3x4 grid: P_miss time series per bench
  - dynamics_mlp.png          # 3x4 grid: mlp_hit & mlp_miss time series per bench
  - hitmix_stacked.png        # stacked bars: P_sram_rd/wr, P_mram_rd/wr, P_miss per bench
  - dist_P_miss.png           # 3x4 grid: P_miss histogram per bench
  - knobs_pairwise_overview.png       # 3x4 grid: (pi_miss vs pi_way) per bench (from configs.csv)
  - knobs_pairwise/<bench>.png        # full 5x5 scatter matrix per bench (saved per-bench)
"""
import argparse, os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def list_benches(datasets_root):
    benches = [d for d in sorted(os.listdir(datasets_root))
               if os.path.isdir(os.path.join(datasets_root, d))]
    return benches

def load_baseline(datasets_root, bench):
    p = os.path.join(datasets_root, bench, "baseline_windows.csv")
    if not os.path.exists(p):
        raise SystemExit(f"Missing {p}")
    return pd.read_csv(p)

def load_configs(config_csv):
    if not os.path.exists(config_csv):
        return None
    return pd.read_csv(config_csv)

def _grid_indices(n, rows, cols):
    idx = [(r,c) for r in range(rows) for c in range(cols)]
    return idx[:n]

def _small_multiples(benches, rows, cols, title, ylabels, plot_fn, out_path, sharex=False, sharey=False):
    n = min(len(benches), rows*cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*2.6), sharex=sharex, sharey=sharey)
    axs = np.array(axs).reshape(rows, cols)
    for k,(r,c) in enumerate(_grid_indices(n, rows, cols)):
        ax = axs[r,c]
        plot_fn(benches[k], ax)
        ax.set_title(benches[k], fontsize=9)
        ax.grid(alpha=0.2, linewidth=0.5)
    # clean empty cells
    for k in range(n, rows*cols):
        r,c = divmod(k, cols)
        axs[r,c].axis('off')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="/home/skataoka26/ChampSim/out/datasets")
    ap.add_argument("--configs-csv",   default="/home/skataoka26/ChampSim/out/configs.csv")
    ap.add_argument("--out-dir",       default="/home/skataoka26/ChampSim/out/figs")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--benches", nargs="*", help="subset of benches; default: all (paginated by rows*cols)")
    args = ap.parse_args()

    benches = args.benches if args.benches else list_benches(args.datasets_root)
    benches = benches[:args.rows*args.cols]  # first page
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Plot 1: A_tot dynamics ----
    def plot_atot(bench, ax):
        df = load_baseline(args.datasets_root, bench)
        ax.plot(df["window_id"], df["A_tot"], lw=1.1)
        ax.set_xlabel("window_id"); ax.set_ylabel("A_tot")
    _small_multiples(benches, args.rows, args.cols, "Window dynamics: A_tot", ("A_tot",), plot_atot,
                     os.path.join(args.out_dir, "dynamics_A_tot.png"), sharex=False, sharey=False)

    # ---- Plot 2: P_miss dynamics ----
    def plot_pmiss(bench, ax):
        df = load_baseline(args.datasets_root, bench)
        ax.plot(df["window_id"], df["P_miss"], lw=1.1)
        ax.set_xlabel("window_id"); ax.set_ylabel("P_miss")
    _small_multiples(benches, args.rows, args.cols, "Window dynamics: P_miss", ("P_miss",), plot_pmiss,
                     os.path.join(args.out_dir, "dynamics_P_miss.png"))

    # ---- Plot 3: mlp_hit / mlp_miss dynamics ----
    def plot_mlp(bench, ax):
        df = load_baseline(args.datasets_root, bench)
        ax.plot(df["window_id"], df["mlp_hit"], lw=1.1, label="mlp_hit")
        ax.plot(df["window_id"], df["mlp_miss"], lw=1.1, label="mlp_miss")
        ax.legend(fontsize=8)
        ax.set_xlabel("window_id"); ax.set_ylabel("MLP")
    _small_multiples(benches, args.rows, args.cols, "Window dynamics: MLP (hit & miss)", ("MLP",), plot_mlp,
                     os.path.join(args.out_dir, "dynamics_mlp.png"))

    # ---- Plot 4: hit-mix stacked bars (per bench) ----
    labels, stacks = [], {"P_sram_rd":[], "P_sram_wr":[], "P_mram_rd":[], "P_mram_wr":[], "P_miss":[]}
    for b in benches:
        df = load_baseline(args.datasets_root, b)
        labels.append(b)
        stacks["P_sram_rd"].append(df["P_sram_rd"].mean())
        stacks["P_sram_wr"].append(df["P_sram_wr"].mean())
        stacks["P_mram_rd"].append(df["P_mram_rd"].mean())
        stacks["P_mram_wr"].append(df["P_mram_wr"].mean())
        stacks["P_miss"].append(df["P_miss"].mean())
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.9), 4))
    bottom = np.zeros(len(labels))
    for key in ["P_sram_rd","P_sram_wr","P_mram_rd","P_mram_wr","P_miss"]:
        ax.bar(x, stacks[key], bottom=bottom, label=key)
        bottom += np.array(stacks[key])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("share"); ax.set_title("Hit/miss mix (averaged over windows)")
    ax.legend(ncols=3, fontsize=8); ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "hitmix_stacked.png"), dpi=160)
    plt.close(fig)

    # ---- Plot 5: P_miss distributions (per bench) ----
    def plot_dist_pmiss(bench, ax):
        df = load_baseline(args.datasets_root, bench)
        ax.hist(df["P_miss"], bins=40, alpha=0.85)
        ax.set_xlabel("P_miss"); ax.set_ylabel("count")
    _small_multiples(benches, args.rows, args.cols, "Distribution of P_miss (per bench)", ("P_miss",), plot_dist_pmiss,
                     os.path.join(args.out_dir, "dist_P_miss.png"), sharey=True)

    # ---- Plot 6: Pairwise knobs coverage ----
    cfg = load_configs(args.configs_csv)
    if cfg is not None:
        # (a) overview: (pi_miss vs pi_way) per bench
        def plot_knobs_overview(bench, ax):
            sub = cfg[cfg["bench"]==bench] if "bench" in cfg.columns else cfg
            if sub.empty: 
                ax.text(0.5,0.5,"no samples", ha="center"); return
            ax.scatter(sub["pi_miss"], sub["pi_way"], s=16, alpha=0.7)
            ax.set_xlabel("pi_miss"); ax.set_ylabel("pi_way"); ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
        _small_multiples(benches, args.rows, args.cols, "LHS coverage: pi_miss vs pi_way", ("",""), plot_knobs_overview,
                         os.path.join(args.out_dir, "knobs_pairwise_overview.png"), sharex=True, sharey=True)

        # (b) full scatter matrix per bench (saved individually)
        out_sub = os.path.join(args.out_dir, "knobs_pairwise")
        os.makedirs(out_sub, exist_ok=True)
        cols = [c for c in ["pi_miss","pi_way","tS","tMr","tMw"] if c in cfg.columns]
        for b in benches:
            sub = cfg[cfg["bench"]==b] if "bench" in cfg.columns else cfg
            if sub.empty or not all(col in sub.columns for col in cols): continue
            fig, axs = plt.subplots(len(cols), len(cols), figsize=(len(cols)*2.4, len(cols)*2.4))
            for i,ci in enumerate(cols):
                for j,cj in enumerate(cols):
                    ax = axs[i,j]
                    if i==j:
                        ax.hist(sub[ci], bins=20)
                        ax.set_ylabel(""); ax.set_xlabel(ci)
                    else:
                        ax.scatter(sub[cj], sub[ci], s=10, alpha=0.6)
                        if i==len(cols)-1: ax.set_xlabel(cj)
                        if j==0: ax.set_ylabel(ci)
                    ax.grid(alpha=0.2)
            fig.suptitle(f"Knobs scatter matrix — {b}", fontsize=11)
            fig.tight_layout(rect=[0,0,1,0.96])
            fig.savefig(os.path.join(out_sub, f"{b}.png"), dpi=160)
            plt.close(fig)

    print(f"[plot_workload] Saved plots to {args.out_dir}")

if __name__ == "__main__":
    main()

