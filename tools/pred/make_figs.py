#!/usr/bin/env python3
import argparse, os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from physics import DEFAULTS as PHY_CONSTS, physics_predict_row
plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight"})

def load_validations(model_dir):
    paths = sorted(glob.glob(os.path.join(model_dir, "*_validation.json")))
    rows = []
    for p in paths:
        j = json.load(open(p))
        bench = j["bench"]
        rows.append({
            "bench": bench,
            "mape_cycles_phys": j.get("mape_cycles_phys"),
            "mape_cycles_ours": j.get("mape_cycles_avg"),
            "mape_cycles_gbm":  j.get("mape_cycles_gbm"),
            "mape_cycles_ridge":j.get("mape_cycles_ridge"),
            "mape_energy_phys": j.get("mape_energy_phys"),
            "mape_energy_pred": j.get("mape_energy_pred"),
        })
    return pd.DataFrame(rows)

def bar_with_numbers(ax, labels, vals, title, ylabel="%"):
    x = np.arange(len(labels))
    bars = ax.bar(x, vals)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title); ax.set_ylabel(ylabel)
    for b,v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f"{v*100:.2f}%", ha="center", va="bottom", fontsize=8)

def make_validation_figs(df, out_dir):
    benches = df["bench"].tolist()
    # Runtime MAPE bars
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(benches, df["mape_cycles_phys"]*100, "o-", label="Physics-only")
    ax.plot(benches, df["mape_cycles_ours"]*100, "o-", label="Ours (phys+res)")
    ax.plot(benches, df["mape_cycles_gbm"]*100, "o-", label="Pure GBM")
    ax.plot(benches, df["mape_cycles_ridge"]*100, "o-", label="Ridge")
    ax.set_ylabel("ROI cycles MAPE (%)"); ax.set_xticklabels(benches, rotation=45, ha="right")
    ax.legend(); ax.grid(alpha=0.2)
    fig.savefig(os.path.join(out_dir, "val_runtime_bars.png")); plt.close(fig)

    # Energy MAPE bars
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(benches, (df["mape_energy_phys"]*100), "o-", label="Physics-only")
    ax.plot(benches, (df["mape_energy_pred"]*100), "o-", label="Ours (phys+res)")
    ax.set_ylabel("ROI energy MAPE (%)"); ax.set_xticklabels(benches, rotation=45, ha="right")
    ax.legend(); ax.grid(alpha=0.2)
    fig.savefig(os.path.join(out_dir, "val_energy_bars.png")); plt.close(fig)

    # CDFs
    def cdf_plot(series, title, fname):
        s = series.dropna().values
        xs = np.sort(s); ys = np.arange(1, len(xs)+1)/len(xs)
        fig, ax = plt.subplots(figsize=(4.5,3.5))
        ax.plot(xs*100, ys, lw=2)
        ax.set_xlabel("MAPE (%)"); ax.set_ylabel("CDF"); ax.set_title(title); ax.grid(alpha=0.2)
        fig.savefig(os.path.join(out_dir, fname)); plt.close(fig)
    cdf_plot(df["mape_cycles_ours"], "ROI cycles MAPE CDF (ours)", "val_runtime_cdf.png")
    if df["mape_energy_pred"].notna().any():
        cdf_plot(df["mape_energy_pred"], "ROI energy MAPE CDF (ours)", "val_energy_cdf.png")

def make_pareto_figs(dse_dir, out_dir, benches=None):
    # For each bench, scatter energy vs IPC
    paths = sorted(glob.glob(os.path.join(dse_dir, "*_dse.csv")))
    for p in paths:
        bench = os.path.basename(p).replace("_dse.csv","")
        if benches and bench not in benches: continue
        df = pd.read_csv(p)
        if "ipc_pred" not in df.columns:
            df["ipc_pred"] = np.nan
        # Pareto: minimize energy for given IPC, just scatter
        fig, ax = plt.subplots(figsize=(4.5,3.5))
        ax.scatter(df["energy_pred_pJ"]/1e6, df["ipc_pred"], s=8, alpha=0.5)
        ax.set_xlabel("Energy (mJ)"); ax.set_ylabel("IPC"); ax.set_title(f"Pareto: {bench}")
        ax.grid(alpha=0.2)
        fig.savefig(os.path.join(out_dir, f"pareto_{bench}.png")); plt.close(fig)

def make_window_fig(bench, datasets_root, model_dir, out_dir,
                    demo_knobs=dict(pi_miss=0.6,pi_way=0.4,tS=16,tMr=28,tMw=60)):
    base_csv = os.path.join(datasets_root, bench, "baseline_windows.csv")
    dfb = pd.read_csv(base_csv)
    # quick per-window predicted cycles for demo_knobs
    model = joblib.load(os.path.join(model_dir, f"{bench}.pkl"))
    consts = PHY_CONSTS.copy()
    const_path = os.path.join(model_dir, f"{bench}_consts.json")
    if os.path.exists(const_path): consts.update(json.load(open(const_path)))
    # run
    cyc_pred = []
    for _, brow in dfb.iterrows():
        cyc_phys, aux = physics_predict_row(brow, demo_knobs, consts)
        X = np.array([
            float(brow["A_tot"]), float(brow["P_miss"]), float(brow["P_sram_rd"]), float(brow["P_mram_rd"]),
            float(brow["mlp_hit"]), float(brow["mlp_miss"]),
            float(brow["T_miss_base"]), float(brow["AMAT_base"]),
            float(demo_knobs["tS"]), float(demo_knobs["tMr"]), float(demo_knobs["tMw"]),
            float(demo_knobs["pi_miss"]), float(demo_knobs["pi_way"]),
            float(aux["AMAT"]), float(aux["Tmiss"])
        ]).reshape(1,-1)
        cyc_pred.append(cyc_phys + float(model.predict(X)[0]))
    dfb["cycles_pred_demo"] = cyc_pred

    # plot
    fig, axs = plt.subplots(3,1, figsize=(10,6), sharex=True)
    axs[0].plot(dfb["window_id"], dfb["A_tot"], lw=1.2); axs[0].set_ylabel("A_tot"); axs[0].grid(alpha=0.2)
    axs[1].plot(dfb["window_id"], dfb["P_miss"], lw=1.2, label="P_miss")
    axs[1].plot(dfb["window_id"], dfb["mlp_miss"], lw=1.2, label="mlp_miss")
    axs[1].plot(dfb["window_id"], dfb["mlp_hit"], lw=1.2, label="mlp_hit")
    axs[1].legend(); axs[1].grid(alpha=0.2)
    axs[2].plot(dfb["window_id"], dfb["cycles_base"], lw=1.2, label="cycles_base")
    axs[2].plot(dfb["window_id"], dfb["cycles_pred_demo"], lw=1.2, label="cycles_pred(demo)")
    axs[2].set_xlabel("window_id"); axs[2].legend(); axs[2].grid(alpha=0.2)
    fig.suptitle(f"Windowing & dynamics: {bench}")
    fig.savefig(os.path.join(out_dir, f"windowing_{bench}.png")); plt.close(fig)

def feature_characterization(datasets_root, out_dir):
    # distributions of P_miss and A_tot across benches
    paths = sorted(glob.glob(os.path.join(datasets_root, "*", "baseline_windows.csv")))
    frames = []
    for p in paths:
        bench = p.split(os.sep)[-2]
        df = pd.read_csv(p)[["P_miss","A_tot"]].copy()
        df["bench"] = bench
        frames.append(df)
    allf = pd.concat(frames, ignore_index=True)
    # plots
    fig, ax = plt.subplots(figsize=(6,3.5))
    allf["P_miss"].hist(bins=50, ax=ax)
    ax.set_title("Distribution of P_miss (all benches)"); ax.set_xlabel("P_miss"); ax.set_ylabel("count")
    fig.savefig(os.path.join(out_dir, "feat_Pmiss_hist.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,3.5))
    np.log10(allf["A_tot"].clip(lower=1)).hist(bins=50, ax=ax)
    ax.set_title("Distribution of log10(A_tot) (all benches)"); ax.set_xlabel("log10(A_tot)"); ax.set_ylabel("count")
    fig.savefig(os.path.join(out_dir, "feat_Atot_hist.png")); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/skataoka26/ChampSim/model")
    ap.add_argument("--datasets-root", default="/home/skataoka26/ChampSim/out/datasets")
    ap.add_argument("--dse-dir", default="/home/skataoka26/ChampSim/out/dse")
    ap.add_argument("--out-figs", default="/home/skataoka26/ChampSim/out/figs")
    ap.add_argument("--window-bench", default="600.perlbench_s-570B.champsimtrace.xz")
    args = ap.parse_args()

    os.makedirs(args.out_figs, exist_ok=True)

    # (A,B) Validation figs
    dfv = load_validations(args.model_dir)
    if not dfv.empty:
        make_validation_figs(dfv, args.out_figs)

    # (C) Pareto figs from DSE
    make_pareto_figs(args.dse_dir, args.out_figs)

    # (D) Windowing dynamics fig for one representative benchmark
    make_window_fig(args.window_bench, args.datasets_root, args.model_dir, args.out_figs)

    # (E) Feature characterization
    feature_characterization(args.datasets_root, args.out_figs)

    print(f"Figures saved to {args.out_figs}")

if __name__ == "__main__":
    main()
