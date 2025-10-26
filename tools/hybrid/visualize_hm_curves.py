#!/usr/bin/env python3
"""
Plot HM (hit-mix) curves per bench, overlaying observed points and the fitted isotonic model.

Reads:
  - results/surrogate/L3_<cap>/hm_curves.joblib
  - results/characterization_L3_<cap>/<bench>/LLC.representatives.json & LLC.window_features.csv
  - results/eval/<bench>/<tag>/metrics.json and LLC.llc.win.csv

Example:
  python3 tools/hybrid/visualize_hm_curves.py --cap 32
  python3 tools/hybrid/visualize_hm_curves.py --cap 32 --benches 602.gcc_s-1850B,605.mcf_s-994B --pm-slices 0.25,0.50,0.75
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def compute_mid_fractions(df):
    mids = 0.5*(df['start_cycle'].to_numpy() + df['end_cycle'].to_numpy())
    total = float(df['end_cycle'].max() - df['start_cycle'].min())
    return mids / (total if total>0 else 1.0)

def load_rep_fracs(char_dir):
    reps_json = os.path.join(char_dir, "LLC.representatives.json")
    feats_csv = os.path.join(char_dir, "LLC.window_features.csv")
    if not (os.path.exists(reps_json) and os.path.exists(feats_csv)):
        return None
    reps = json.load(open(reps_json)).get("representatives", [])
    feats= pd.read_csv(feats_csv, engine="python", on_bad_lines="skip")
    fracs_all = compute_mid_fractions(feats)
    id2frac = dict(zip(feats['window_id'].to_numpy(), fracs_all))
    rep_fracs = [id2frac[w] for w in reps if w in id2frac]
    return rep_fracs

def select_by_fraction(new_win_csv, rep_fracs):
    df = pd.read_csv(new_win_csv, engine="python", on_bad_lines="skip")
    if df.empty or not rep_fracs:
        return df.iloc[0:0].copy()
    fracs = compute_mid_fractions(df)
    used=set(); chosen_idx=[]
    for f in rep_fracs:
        order = np.argsort(np.abs(fracs - f))
        pick = None
        for j in order:
            j = int(j)
            if j in used: 
                continue
            hits = (df.loc[j,"hit_sram_rd"]+df.loc[j,"hit_sram_wr"]+
                    df.loc[j,"hit_mram_rd"]+df.loc[j,"hit_mram_wr"])
            if hits > 0:    # prefer a window that actually has hits
                pick = j; break
        if pick is None:
            pick = int(order[0])   # as absolute fallback
        if pick is not None:
            used.add(pick); chosen_idx.append(pick)
    return df.iloc[chosen_idx].copy()

def collect_points_for_bench(cap, bench):
    """Return dict: pm -> list of (pi_way, frac_mram_hits_among_hits) using mapped reps."""
    bench_dir = os.path.join("results","eval", bench)
    char_dir  = os.path.join("results", f"characterization_L3_{cap}", bench)
    rep_fracs = load_rep_fracs(char_dir)
    if rep_fracs is None:
        return {}
    out = {}
    for tag_dir in glob.glob(os.path.join(bench_dir, "*")):
        mpath = os.path.join(tag_dir, "metrics.json")
        csvp  = os.path.join(tag_dir, "LLC.llc.win.csv")
        if not (os.path.exists(mpath) and os.path.exists(csvp)):
            continue
        try:
            meta = json.load(open(mpath))
        except Exception:
            continue
        pw = float(meta.get("pi_way", 0.0))
        pm = float(meta.get("pi_miss", 0.5))
        df_rep = select_by_fraction(csvp, rep_fracs)
        if df_rep.empty:
            continue
        Hm = (df_rep['hit_mram_rd'] + df_rep['hit_mram_wr']).sum()
        Hs = (df_rep['hit_sram_rd'] + df_rep['hit_sram_wr']).sum()
        Ht = Hm + Hs
        if Ht <= 0:
            continue
        frac = float(Hm / Ht)
        out.setdefault(pm, []).append((pw, frac))
    return out

def plot_hm_for_bench(cap, bench, curves, pm_slices, outdir):
    points = collect_points_for_bench(cap, bench)
    if bench not in curves:
        print(f"[warn] no curves for {bench}")
        return

    # Which slices to show
    keyset = set(curves[bench].keys())
    keyset.discard("__pooled__")
    if pm_slices is None:
        pm_list = sorted([float(k) for k in keyset])
    else:
        pm_list = pm_slices

    fig, ax = plt.subplots(figsize=(7.5,5.5))
    colors = plt.get_cmap("tab10").colors

    # observed points
    for idx, pm in enumerate(pm_list):
        pts = points.get(pm, [])
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, s=28, color=colors[idx%10], alpha=0.8, label=f"obs pm={pm:.2f}")

    # fitted slices
    for idx, pm in enumerate(pm_list):
        key = f"{pm:.4f}"
        if key in curves[bench]:
            xs = np.asarray(curves[bench][key]["x"], float)
            ys = np.asarray(curves[bench][key]["y"], float)
            ax.plot(xs, ys, color=colors[idx%10], linewidth=2.0, alpha=0.9, label=f"fit pm={pm:.2f}")

    # pooled
    if "__pooled__" in curves[bench]:
        xs = np.asarray(curves[bench]["__pooled__"]["x"], float)
        ys = np.asarray(curves[bench]["__pooled__"]["y"], float)
        ax.plot(xs, ys, color="black", linewidth=2.2, linestyle="--", alpha=0.9, label="fit pooled")

    ax.set_xlabel("π_way (fraction of MRAM capacity)")
    ax.set_ylabel("MRAM hit fraction among hits")
    ax.set_title(f"{bench} — HM(π_way, π_miss)  (L3={cap}MB)")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    # de-duplicate legend
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, frameon=False, loc="best")

    ensure_dir(outdir)
    outpng = os.path.join(outdir, f"hm_curves_{bench.replace('/','-')}.png")
    fig.tight_layout()
    fig.savefig(outpng, dpi=180)
    print(f"[ok] wrote {outpng}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--benches", default="", help="comma list; if empty, use benches in hm_curves.joblib")
    ap.add_argument("--pm-slices", default="", help="comma list like 0.25,0.50,0.75; empty → all available")
    args = ap.parse_args()

    hm_p = os.path.join("results","surrogate", f"L3_{args.cap}", "hm_curves.joblib")
    if not os.path.exists(hm_p):
        raise FileNotFoundError(f"missing curves: {hm_p}")
    curves = joblib.load(hm_p)

    if args.benches.strip():
        benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    else:
        benches = sorted(curves.keys())

    pm_list = None
    if args.pm_slices.strip():
        pm_list = [float(x) for x in args.pm_slices.split(",") if x.strip()]

    outdir = os.path.join("results","figs", f"L3_{args.cap}", "hm_curves")
    ensure_dir(outdir)

    for b in benches:
        plot_hm_for_bench(args.cap, b, curves, pm_list, outdir)

if __name__ == "__main__":
    main()
