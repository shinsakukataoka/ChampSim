#!/usr/bin/env python3
"""
Validation plots.

Outputs (under --out-dir):
  - validation_mape_bars.png     # 1x2 grid: cycles & energy MAPE, physics-only vs phys+residual
  - importance_physics_share.png # one chart: per-bench share of importance from physics augment features (AMAT,Tmiss)
"""
import argparse, os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Dictionary for short benchmark names
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

def load_validations(model_dir):
    rows=[]
    for p in sorted(glob.glob(os.path.join(model_dir, "*_validation.json"))):
        j=json.load(open(p))
        rows.append({
            "bench": j["bench"],
            "mape_cycles_phys": j.get("mape_cycles_phys"),
            "mape_cycles_ours": j.get("mape_cycles_avg"),
            "mape_energy_phys": j.get("mape_energy_phys"),
            "mape_energy_ours": j.get("mape_energy_pred"),
        })
    return pd.DataFrame(rows)

def autolabel(rects, ax, format_str='{:.2f}%'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if pd.isna(height): # Skip NaN values
            continue
        ax.annotate(format_str.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6,
                    rotation=90) # Rotate text 90 degrees

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/skataoka26/ChampSim/model")
    ap.add_argument("--out-dir",   default="/home/skataoka26/ChampSim/out/figs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_validations(args.model_dir).sort_values("bench")
    benches = df["bench"].tolist()
    # Create list of short names, falling back to original if not in SHORT dict
    short_benches = [SHORT.get(b, b) for b in benches]
    x = np.arange(len(benches)); w = 0.35

    # ---- Plot 1: per-bench MAPE bars (cycles & energy), physics-only vs phys+res ----
    fig, axs = plt.subplots(1,2, figsize=(max(7, len(benches)*0.9), 4), sharey=False)
    # cycles
    ax = axs[0]
    rects1 = ax.bar(x - w/2, np.array(df["mape_cycles_phys"])*100.0, width=w, label="Physics-only")
    rects2 = ax.bar(x + w/2, np.array(df["mape_cycles_ours"])*100.0, width=w, label="Phys+Residual")
    ax.set_title("ROI Cycles MAPE"); ax.set_ylabel("MAPE (%)")
    ax.set_xticks(x); ax.set_xticklabels(short_benches, rotation=40, ha="right", fontsize=8); ax.grid(axis='y', alpha=0.2)
    ax.legend(fontsize=8)
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35) # Add more headroom for rotated labels

    # energy
    ax = axs[1]
    ephys = df["mape_energy_phys"].fillna(0.0).to_numpy()*100.0
    eours = df["mape_energy_ours"].fillna(0.0).to_numpy()*100.0
    rects3 = ax.bar(x - w/2, ephys, width=w, label="Physics-only")
    rects4 = ax.bar(x + w/2, eours, width=w, label="Phys+Residual")
    ax.set_title("ROI Energy MAPE")
    ax.set_xticks(x); ax.set_xticklabels(short_benches, rotation=40, ha="right", fontsize=8); ax.grid(axis='y', alpha=0.2)
    ax.legend(fontsize=8)
    autolabel(rects3, ax)
    autolabel(rects4, ax)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35) # Add more headroom for rotated labels

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "validation_mape_bars.png"), dpi=160)
    plt.close(fig)

    # ---- Plot 2: feature importance (per-bench physics augmentation share) ----
    # Our training features put AMAT and Tmiss at the end of build_X (2 features).
    # We'll report the share of total importance attributable to those two features.
    physics_share = []
    for b in benches:
        model_path = os.path.join(args.model_dir, f"{b}.pkl")
        if not os.path.exists(model_path):
            physics_share.append(np.nan); continue
        m = joblib.load(model_path)
        fi = getattr(m, "feature_importances_", None)
        if fi is None or len(fi) < 2:
            physics_share.append(np.nan); continue
        # Assumes build_X: 15 features, last two are [AMAT, Tmiss]
        share = float((fi[-2:].sum()) / max(1e-12, fi.sum()))
        physics_share.append(share*100.0)
    
    fig, ax = plt.subplots(figsize=(max(7, len(benches)*0.9), 3.8))
    rects5 = ax.bar(x, physics_share)
    ax.set_xticks(x); ax.set_xticklabels(short_benches, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Importance share from [AMAT, Tmiss] (%)")
    ax.set_title("Feature importance: physics augmentation contribution")
    ax.grid(axis='y', alpha=0.2)
    
    autolabel(rects5, ax)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35) # Add more headroom for rotated labels
    
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "importance_physics_share.png"), dpi=160)
    plt.close(fig)

    print(f"[plot_validation] Saved plots to {args.out_dir}")

if __name__ == "__main__":
    main()
