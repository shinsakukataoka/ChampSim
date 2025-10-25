#!/usr/bin/env python3
import os, glob, argparse, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- utilities ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ux = np.unique(x)
    if ux.size < 2: return np.nan
    return float(np.polyfit(x, y, 1)[0])

def round_keys(df):
    df = df.copy()
    for c in ["pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(6)
    return df

def autodetect_profiles(cap):
    roots = glob.glob(f"results/surrogate/profiles/*/L3_{cap}/physics_pred.csv")
    return sorted({p.split(os.sep)[3] for p in roots})

def hm_benches(cap):
    hp = f"results/surrogate/L3_{cap}/hm_curves.joblib"
    if not os.path.exists(hp): return []
    curves = joblib.load(hp)
    # keys are benches; take up to 9 benches deterministically
    benches = sorted(curves.keys())
    # Try to prefer the known 9 if present
    preferred = ['602.gcc_s-1850B','605.mcf_s-994B','619.lbm_s-2677B','620.omnetpp_s-874B',
                 '621.wrf_s-6673B','623.xalancbmk_s-700B','631.deepsjeng_s-928B',
                 '649.fotonik3d_s-7084B','657.xz_s-3167B']
    order = [b for b in preferred if b in benches] + [b for b in benches if b not in preferred]
    return order[:9]

def load_dataset(cap):
    p = f"results/eval/dataset_L3_{cap}MB.csv"
    if not os.path.exists(p): raise FileNotFoundError(p)
    return pd.read_csv(p).dropna(subset=["stall_pct","E_total_J"])

def load_policy_bounds(cap, profile=None):
    if profile:
        p = f"results/surrogate/profiles/{profile}/L3_{cap}/policy_bounds.csv"
    else:
        p = f"results/surrogate/L3_{cap}/policy_bounds.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

def load_physics(cap, profile=None):
    if profile:
        p = f"results/surrogate/profiles/{profile}/L3_{cap}/physics_pred.csv"
        if os.path.exists(p): return pd.read_csv(p)
    p = f"results/surrogate/L3_{cap}/physics_pred.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

# ---------- plots ----------
def plot_device_grid(cap, benches9, profiles, out_png):
    """Stall vs Energy by device profile on 3x3 grid."""
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    prof2col = {p: colors[i % len(colors)] for i,p in enumerate(profiles)}
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    any_data = False
    for i, b in enumerate(benches9):
        ax = axes[i]
        for p in profiles:
            path = f"results/surrogate/profiles/{p}/L3_{cap}/physics_pred.csv"
            if not os.path.exists(path):  # fallback to current run
                path = f"results/surrogate/L3_{cap}/physics_pred.csv"
                if not os.path.exists(path): continue
                df = pd.read_csv(path)
            else:
                df = pd.read_csv(path)
            sub = df[df["bench"]==b]
            if sub.empty: continue
            any_data = True
            ax.scatter(sub["E_total_J"]*1e6, sub["stall_pct"]*100.0, s=8, alpha=0.7,
                       label=p, color=prof2col[p])
        ax.set_title(b, fontsize=9)
        ax.grid(True, alpha=0.3)
        if i%3==0: ax.set_ylabel("Stall (%)")
        if i//3==2: ax.set_xlabel("Energy (μJ)")
    if any_data:
        # one legend outside
        handles = [plt.Line2D([],[],marker='o',linestyle='',color=prof2col[p],label=p) for p in profiles]
        fig.legend(handles=handles, labels=profiles, loc="lower center", ncol=min(4, len(profiles)), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0,0.03,1,1])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_policy_bounds_grid(cap, benches9, profile_for_bounds, out_png):
    """For each bench, arrow from best to worst median (stall, energy)."""
    pb = load_policy_bounds(cap, profile_for_bounds)
    if pb is None or pb.empty: return
    pv = pb.pivot_table(index=["bench","tag"], columns="bound",
                        values=["stall_pct","E_total_J"]).dropna()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    any_data = False
    for i,b in enumerate(benches9):
        ax = axes[i]
        pbb = pv.loc[pv.index.get_level_values(0)==b]
        if pbb.empty: 
            ax.set_title(b, fontsize=9); ax.grid(True, alpha=0.3); continue
        any_data = True
        # medians for clarity
        sb = pbb["stall_pct"]["stall_best"].median()
        sw = pbb["stall_pct"]["stall_worst"].median()
        eb = pbb["E_total_J"]["energy_best"].median()*1e6
        ew = pbb["E_total_J"]["energy_worst"].median()*1e6
        ax.scatter([eb,ew],[sb*100,sw*100], s=16, c=["tab:green","tab:red"])
        ax.arrow(eb, sb*100, (ew-eb), (sw*100 - sb*100),
                 length_includes_head=True, head_width=0.8, head_length=max(1.0, 0.02*(ew-eb)),
                 fc="tab:gray", ec="tab:gray", alpha=0.7)
        ax.set_title(b, fontsize=9)
        ax.grid(True, alpha=0.3)
        if i%3==0: ax.set_ylabel("Stall (%)")
        if i//3==2: ax.set_xlabel("Energy (μJ)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_sensitivity_panel(cap, benches9, out_png):
    df = load_dataset(cap)
    # pi_way slope @ pm=0.50
    sub = df[np.isclose(df["pi_miss"], 0.50)]
    s_pw = sub.groupby("bench", group_keys=False).apply(lambda g: slope(g["pi_way"], g["stall_pct"]))
    # tMr/tMw slope @ pw∈{.25,.5,.75}, pm=0.50 (median over benches per pw)
    sub2 = sub[sub["pi_way"].isin([0.25,0.50,0.75])]
    s_mr = sub2.groupby(["bench","pi_way"], group_keys=False).apply(lambda g: slope(g["t_mram_rd"], g["stall_pct"])).unstack()
    s_mw = sub2.groupby(["bench","pi_way"], group_keys=False).apply(lambda g: slope(g["t_mram_wr"], g["stall_pct"])).unstack()

    fig, axs = plt.subplots(1,3, figsize=(13,3.8))
    # A) pi_way slopes per bench
    bench_order = benches9 if benches9 else sorted(s_pw.index.tolist())
    y = [s_pw.get(b, np.nan) for b in bench_order]
    axs[0].bar(range(len(bench_order)), y)
    axs[0].set_xticks(range(len(bench_order))); axs[0].set_xticklabels([b.split('.')[1] for b in bench_order], rotation=60, fontsize=8)
    axs[0].set_title("π_way → stall slope (pm=0.50)")
    axs[0].set_ylabel("Δstall / Δπ_way"); axs[0].grid(True, axis='y', alpha=0.3)

    # B) tMr slope median by pw
    med_mr = {pw: float(np.nanmedian(s_mr.get(pw, pd.Series(dtype=float)).values)) for pw in [0.25,0.50,0.75]}
    axs[1].bar([0,1,2], [med_mr[0.25], med_mr[0.50], med_mr[0.75]])
    axs[1].set_xticks([0,1,2]); axs[1].set_xticklabels(["pw=.25",".50",".75"])
    axs[1].set_title("tMr → stall slope (median over benches)"); axs[1].grid(True, axis='y', alpha=0.3)

    # C) tMw slope median by pw
    med_mw = {pw: float(np.nanmedian(s_mw.get(pw, pd.Series(dtype=float)).values)) for pw in [0.25,0.50,0.75]}
    axs[2].bar([0,1,2], [med_mw[0.25], med_mw[0.50], med_mw[0.75]])
    axs[2].set_xticks([0,1,2]); axs[2].set_xticklabels(["pw=.25",".50",".75"])
    axs[2].set_title("tMw → stall slope (median over benches)"); axs[2].grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_parity_grids(cap, benches9, profile_for_parity, out_stall_png, out_energy_png):
    ds = load_dataset(cap)
    ph = load_physics(cap, profile_for_parity)
    if ph is None or ph.empty: return
    JOIN = ["bench","pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr","tag"]
    m = round_keys(ds).merge(round_keys(ph), on=JOIN, how="inner", suffixes=("_eval","_phys"))
    if m.empty: return

    # stall parity
    fig, axes = plt.subplots(3,3, figsize=(12,10), sharex=False, sharey=False)
    axes = axes.flatten()
    for i,b in enumerate(benches9):
        ax = axes[i]
        sub = m[m["bench"]==b]
        if sub.empty: ax.set_title(b); ax.grid(True, alpha=0.3); continue
        ax.scatter(sub["stall_pct_eval"], sub["stall_pct_phys"], s=8, alpha=0.7)
        lo = min(sub["stall_pct_eval"].min(), sub["stall_pct_phys"].min())
        hi = max(sub["stall_pct_eval"].max(), sub["stall_pct_phys"].max())
        ax.plot([lo,hi],[lo,hi], color="tab:gray", linewidth=1)
        ax.set_title(b, fontsize=9); ax.grid(True, alpha=0.3)
        if i%3==0: ax.set_ylabel("physics"); 
        if i//3==2: ax.set_xlabel("eval")
    fig.suptitle("Stall parity (fraction)", y=0.92)
    fig.tight_layout()
    fig.savefig(out_stall_png, dpi=180)
    plt.close(fig)

    # energy parity
    fig, axes = plt.subplots(3,3, figsize=(12,10), sharex=False, sharey=False)
    axes = axes.flatten()
    for i,b in enumerate(benches9):
        ax = axes[i]
        sub = m[m["bench"]==b]
        if sub.empty: ax.set_title(b); ax.grid(True, alpha=0.3); continue
        x = sub["E_total_J_eval"]*1e6; y = sub["E_total_J_phys"]*1e6
        ax.scatter(x, y, s=8, alpha=0.7)
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        ax.plot([lo,hi],[lo,hi], color="tab:gray", linewidth=1)
        ax.set_title(b, fontsize=9); ax.grid(True, alpha=0.3)
        if i%3==0: ax.set_ylabel("physics μJ"); 
        if i//3==2: ax.set_xlabel("eval μJ")
    fig.suptitle("Energy parity (μJ)", y=0.92)
    fig.tight_layout()
    fig.savefig(out_energy_png, dpi=180)
    plt.close(fig)

def plot_profile_medians(cap, profiles, out_png):
    """Median stall and energy per profile (bar chart)."""
    rows=[]
    for p in profiles:
        path = f"results/surrogate/profiles/{p}/L3_{cap}/physics_pred.csv"
        if os.path.exists(path): 
            df = pd.read_csv(path).assign(profile=p)
            rows.append(df)
    if not rows: return
    df = pd.concat(rows, ignore_index=True)
    med = df.groupby("profile")[["E_total_J","stall_pct"]].median().reset_index()
    med["E_total_uJ"] = med["E_total_J"]*1e6
    profiles = med["profile"].tolist()
    fig, ax1 = plt.subplots(figsize=(8,3.5))
    x = np.arange(len(profiles))
    w = 0.35
    ax1.bar(x-w/2, med["stall_pct"]*100.0, width=w, label="Stall (%)")
    ax2 = ax1.twinx()
    ax2.bar(x+w/2, med["E_total_uJ"], width=w, color="tab:orange", label="Energy (μJ)")
    ax1.set_xticks(x); ax1.set_xticklabels(profiles, rotation=30, ha="right")
    ax1.set_ylabel("Stall (%)"); ax2.set_ylabel("Energy (μJ)")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

# ---------- CLI / main ----------
def main():
    ap = argparse.ArgumentParser(description="Create presentation plots")
    ap.add_argument("--cap", type=int, required=True, help="capacity MB (e.g., 32)")
    ap.add_argument("--profiles", default="", help="comma-separated device profiles; auto-detect if empty")
    ap.add_argument("--bounds-profile", default="", help="profile to use for policy_bounds (default=current run)")
    ap.add_argument("--parity-profile", default="", help="profile to use for parity plots (default=current run if present)")
    ap.add_argument("--outdir", default="", help="output directory (default: results/figs/L3_<cap>)")
    args = ap.parse_args()

    cap = args.cap
    outdir = args.outdir or f"results/figs/L3_{cap}"
    ensure_dir(outdir)

    # choose benches (9 with HM curves)
    benches9 = hm_benches(cap)
    if not benches9:
        print(f"[warn] no HM benches found for L3_{cap}; falling back to first 9 benches in dataset")
        benches_all = load_dataset(cap)["bench"].unique().tolist()
        benches9 = sorted(benches_all)[:9]

    # profiles
    profiles = [p for p in args.profiles.split(",") if p] or autodetect_profiles(cap)
    if not profiles:
        print("[warn] no profiles detected; using current run as single profile 'current'")
        profiles = ["current"]

    # 1) Device comparison grid
    plot_device_grid(cap, benches9, profiles, os.path.join(outdir, "device_grid_stall_vs_energy.png"))

    # 2) Policy bounds grid (use given bounds profile, else current)
    bp = args.bounds_profile if args.bounds_profile else None
    plot_policy_bounds_grid(cap, benches9, bp, os.path.join(outdir, "policy_bounds_grid.png"))

    # 3) Sensitivity panel from dataset
    plot_sensitivity_panel(cap, benches9, os.path.join(outdir, "sensitivity_panel.png"))

    # 4) Parity grids (if physics available for chosen profile)
    pp = args.parity_profile if args.parity_profile else None
    plot_parity_grids(cap, benches9, pp,
                      os.path.join(outdir, "parity_stall_grid.png"),
                      os.path.join(outdir, "parity_energy_grid.png"))

    # 5) Device profile medians
    plot_profile_medians(cap, profiles, os.path.join(outdir, "device_profile_medians.png"))

    print(f"[done] wrote figures to {outdir}")

if __name__ == "__main__":
    main()

