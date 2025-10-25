#!/usr/bin/env python3
import os, sys, argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- utils ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def pareto_front(points):
    # points: array of (x=E, y=stall)
    nd = []
    for i,p in enumerate(points):
        dominated = False
        for j,q in enumerate(points):
            if j==i: continue
            if q[0] <= p[0]+1e-12 and q[1] <= p[1]+1e-12 and (q[0] < p[0]-1e-12 or q[1] < p[1]-1e-12):
                dominated = True; break
        if not dominated: nd.append(i)
    return np.array(nd, dtype=int)

def nice_bench_list(df, limit=None):
    benches = sorted(df['bench'].unique().tolist())
    return benches if (limit is None or len(benches)<=limit) else benches[:limit]

def read_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # normalize tags/knobs if needed
    if 'stall_pct' not in df.columns:
        raise RuntimeError("dataset missing stall_pct")
    # drop any malformed rows
    df = df.dropna(subset=['stall_pct'])
    # enforce floats
    for c in ['pi_way','pi_miss','t_sram_hit','t_mram_rd','t_mram_wr','E_total_J']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# ---------- plotting ----------
def plot_knees_one(df, bench, out_png):
    sub = df[df['bench']==bench].copy()
    if sub.empty:
        print(f"[knees] no rows for {bench}")
        return
    sub = sub.dropna(subset=['stall_pct'])
    sub = sub.sort_values(['pi_miss','pi_way'])
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(6,4))
    for pm, grp in sub.groupby('pi_miss'):
        ax.plot(grp['pi_way'], grp['stall_pct']*100.0, marker='o', label=f"pi_miss={pm:.2f}")
    ax.set_title(f"Stall% vs pi_way — {bench}")
    ax.set_xlabel("pi_way (MRAM fraction of ways)")
    ax.set_ylabel("stall % (MLP-aware)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[knees] wrote {out_png}")

def plot_pareto_one(df, bench, out_png):
    sub = df[df['bench']==bench].dropna(subset=['E_total_J','stall_pct']).copy()
    if sub.empty:
        print(f"[pareto] no rows for {bench}"); return
    X = sub[['E_total_J','stall_pct']].to_numpy()
    idx = pareto_front(X)
    on = sub.iloc[idx].copy()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(sub['E_total_J']*1e6, sub['stall_pct']*100.0, s=12, c='#bdbdbd', label='designs')
    on = on.sort_values(['E_total_J'])
    ax.scatter(on['E_total_J']*1e6, on['stall_pct']*100.0, s=30, c='tab:blue', label='Pareto front')
    ax.plot(on['E_total_J']*1e6, on['stall_pct']*100.0, c='tab:blue', alpha=0.7)
    ax.set_title(f"Pareto: stall% vs energy — {bench}")
    ax.set_xlabel("E_total (μJ) — relative")
    ax.set_ylabel("stall % (MLP-aware)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[pareto] wrote {out_png}")

def plot_rep_coverage(df, out_png):
    if 'rep_rows' not in df.columns:
        print("[rep_coverage] rep_rows not in dataset; skip"); return
    fig, ax = plt.subplots(figsize=(5,3))
    df['rep_rows'].value_counts().sort_index().plot(kind='bar', ax=ax, color='tab:green')
    ax.set_title("Representative window mapping count across designs")
    ax.set_xlabel("rep_rows (should equal K)")
    ax.set_ylabel("# designs")
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[rep_coverage] wrote {out_png}")

def plot_grid_overview_knees(df, benches, out_png, rows=4, cols=3, pi_miss=None):
    n = min(len(benches), rows*cols)
    if n==0:
        print("[grid_knees] no benches"); return
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), sharex=True, sharey=False)
    axes = axes.flatten()
    for i,b in enumerate(benches[:n]):
        ax = axes[i]
        sub = df[df['bench']==b].copy()
        if pi_miss is not None:
            sub = sub[np.isclose(sub['pi_miss'], float(pi_miss))]
        sub = sub.sort_values(['pi_way','pi_miss'])
        if sub.empty:
            ax.text(0.5,0.5,"no data", ha='center', va='center'); ax.set_axis_off(); continue
        for pm, grp in sub.groupby('pi_miss'):
            ax.plot(grp['pi_way'], grp['stall_pct']*100.0, marker='o', label=f"pm={pm:.2f}", linewidth=1.4, markersize=3)
        ax.set_title(b, fontsize=9)
        ax.grid(True, alpha=0.3)
        if i%cols==0: ax.set_ylabel("stall %")
        if i//cols==rows-1: ax.set_xlabel("pi_way")
        if i==0: ax.legend(fontsize=7)
    # hide unused
    for j in range(n, rows*cols):
        axes[j].set_axis_off()
    fig.suptitle(f"Overview: Knee curves (rows={rows}, cols={cols})" + (f", pi_miss={pi_miss}" if pi_miss is not None else ""), fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[grid_knees] wrote {out_png}")

def plot_grid_overview_pareto(df, benches, out_png, rows=4, cols=3):
    n = min(len(benches), rows*cols)
    if n==0:
        print("[grid_pareto] no benches"); return
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), sharex=True, sharey=True)
    axes = axes.flatten()
    for i,b in enumerate(benches[:n]):
        ax = axes[i]
        sub = df[df['bench']==b].dropna(subset=['E_total_J','stall_pct']).copy()
        if sub.empty:
            ax.text(0.5,0.5,"no data", ha='center', va='center'); ax.set_axis_off(); continue
        X = sub[['E_total_J','stall_pct']].to_numpy()
        idx = pareto_front(X)
        on = sub.iloc[idx].copy().sort_values(['E_total_J'])
        ax.scatter(sub['E_total_J']*1e6, sub['stall_pct']*100.0, s=10, c='#bdbdbd')
        ax.plot(on['E_total_J']*1e6, on['stall_pct']*100.0, c='tab:blue', linewidth=1.5)
        ax.scatter(on['E_total_J']*1e6, on['stall_pct']*100.0, s=18, c='tab:blue')
        ax.set_title(b, fontsize=9)
        ax.grid(True, alpha=0.3)
        if i%cols==0: ax.set_ylabel("stall %")
        if i//cols==rows-1: ax.set_xlabel("E_total (μJ)")
    # hide unused
    for j in range(n, rows*cols):
        axes[j].set_axis_off()
    fig.suptitle(f"Overview: Pareto fronts (rows={rows}, cols={cols})", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[grid_pareto] wrote {out_png}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Hybrid LLC plotting utility")
    ap.add_argument("--dataset", required=True, help="results/eval/dataset.csv")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("knees", help="Knee plot for one benchmark")
    s1.add_argument("--bench", required=True)
    s1.add_argument("--out", default=None)

    s2 = sub.add_parser("pareto", help="Pareto plot for one benchmark")
    s2.add_argument("--bench", required=True)
    s2.add_argument("--out", default=None)

    s3 = sub.add_parser("coverage", help="Histogram of rep_rows across designs")
    s3.add_argument("--out", default="results/figs/coverage/rep_rows_hist.png")

    s4 = sub.add_parser("grid", help="Grid overview (knees or pareto)")
    s4.add_argument("--kind", choices=["knees","pareto"], required=True)
    s4.add_argument("--rows", type=int, default=4)
    s4.add_argument("--cols", type=int, default=3)
    s4.add_argument("--pi-miss", type=float, default=None, help="only used for knees")
    s4.add_argument("--benches", default=None, help="comma-separated; default=first rows*cols in dataset")
    s4.add_argument("--out", default=None)

    s5 = sub.add_parser("all", help="Produce common set of plots (per-bench knees/paretos + coverage + overview grids)")
    s5.add_argument("--rows", type=int, default=4)
    s5.add_argument("--cols", type=int, default=3)
    s5.add_argument("--pi-miss", type=float, default=0.50)

    args = ap.parse_args()
    df = read_dataset(args.dataset)

    # output dirs
    ensure_dir("results/figs/knees")
    ensure_dir("results/figs/pareto")
    ensure_dir("results/figs/coverage")
    ensure_dir("results/figs/overview")

    if args.cmd == "knees":
        out = args.out or f"results/figs/knees/{args.bench}_knees.png"
        plot_knees_one(df, args.bench, out)

    elif args.cmd == "pareto":
        out = args.out or f"results/figs/pareto/{args.bench}_pareto.png"
        plot_pareto_one(df, args.bench, out)

    elif args.cmd == "coverage":
        out = args.out
        plot_rep_coverage(df, out)

    elif args.cmd == "grid":
        benches = args.benches.split(",") if args.benches else nice_bench_list(df, args.rows*args.cols)
        out = args.out or f"results/figs/overview/grid_{args.kind}_{args.rows}x{args.cols}" + (f"_pm{args.pi_miss:.2f}" if args.kind=="knees" and args.pi_miss is not None else "") + ".png"
        if args.kind == "knees":
            plot_grid_overview_knees(df, benches, out, rows=args.rows, cols=args.cols, pi_miss=args.pi_miss)
        else:
            plot_grid_overview_pareto(df, benches, out, rows=args.rows, cols=args.cols)

    elif args.cmd == "all":
        benches12 = nice_bench_list(df, args.rows*args.cols)
        # per-bench plots
        for b in benches12:
            plot_knees_one(df, b, f"results/figs/knees/{b}_knees.png")
            plot_pareto_one(df, b, f"results/figs/pareto/{b}_pareto.png")
        # coverage
        plot_rep_coverage(df, "results/figs/coverage/rep_rows_hist.png")
        # overview grids
        plot_grid_overview_knees(df, benches12, f"results/figs/overview/grid_knees_{args.rows}x{args.cols}_pm{args.pi_miss:.2f}.png",
                                 rows=args.rows, cols=args.cols, pi_miss=args.pi_miss)
        plot_grid_overview_pareto(df, benches12, f"results/figs/overview/grid_pareto_{args.rows}x{args.cols}.png",
                                  rows=args.rows, cols=args.cols)

if __name__ == "__main__":
    main()

