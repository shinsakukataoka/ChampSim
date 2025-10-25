#!/usr/bin/env python3
import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JOIN = ["bench","pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr","tag"]

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def short(b):
    try: return b.split('.',1)[1].split('_',1)[0]
    except: return b

def round_keys(df):
    df = df.copy()
    for c in ["pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(6)
    return df

def load_dataset(cap):
    p = f"results/eval/dataset_L3_{cap}MB.csv"
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing {p} (run postprocess_after_eval.sh)")
    return pd.read_csv(p).dropna(subset=["stall_pct","E_total_J"])

def load_physics(cap, profile=None):
    if profile:
        p = f"results/surrogate/profiles/{profile}/L3_{cap}/physics_pred.csv"
        if os.path.exists(p): return pd.read_csv(p)
    p = f"results/surrogate/L3_{cap}/physics_pred.csv"
    if not os.path.exists(p):
        raise FileNotFoundError("physics_pred.csv not found (run postprocess_after_eval.sh with a device profile).")
    return pd.read_csv(p)

# ---------- HELPERS FOR RE-WEIGHTING ----------
def load_device(profile, cap, root="devices"):
    if not profile: return None
    base=os.path.join(root, profile, "base.json")
    over=os.path.join(root, profile, f"L3_{cap}.json")
    def _ld(p): return json.load(open(p)) if os.path.exists(p) else {}
    dj=_ld(base); ov=_ld(over)
    for k,v in ov.items():
        if isinstance(v,dict) and k in dj and isinstance(dj[k],dict): dj[k].update(v)
        else: dj[k]=v
    return dj

def reweighted_eval_energy(cap, row, device_json):
    """Compute eval energy with a given device, using the run's CSV + rep mapping."""
    # device params
    f_ghz = float(device_json.get("f_clk_ghz",1.0))
    E_RS = device_json["sram"]["e_rd_pj"]*1e-12
    E_WS = device_json["sram"]["e_wr_pj"]*1e-12
    E_RM = device_json["mram"]["e_rd_pj"]*1e-12
    E_WM = device_json["mram"]["e_wr_pj"]*1e-12
    P_LS = device_json["sram"]["leak_mw_per_mb"]*1e-3
    P_LM = device_json["mram"]["leak_mw_per_mb"]*1e-3
    dram  = device_json.get("dram", {})
    E_MISS_BASE   = dram.get("e_miss_pj", 3000.0)*1e-12
    E_MISS_PER_NS = dram.get("e_miss_per_ns_pj", 0.0)*1e-12

    bench=row["bench"]; tag=row["tag"]; L3=float(row.get("l3_mb", cap))
    # reps → pick windows in this run
    char_dir=f"results/characterization_L3_{int(L3)}/{bench}"
    reps=json.load(open(os.path.join(char_dir,"LLC.representatives.json")))["representatives"]
    feats=pd.read_csv(os.path.join(char_dir,"LLC.window_features.csv"), engine="python", on_bad_lines="skip")
    mids=0.5*(feats.start_cycle.to_numpy()+feats.end_cycle.to_numpy()); T=float(feats.end_cycle.max()-feats.start_cycle.min()); fr= mids/(T if T>0 else 1.0)
    id2frac=dict(zip(feats.window_id.to_numpy(), fr))
    rep_fracs=[id2frac[w] for w in reps if w in id2frac]

    run_csv=f"results/eval/{bench}/{tag}/LLC.llc.win.csv"
    df=pd.read_csv(run_csv, engine="python", on_bad_lines="skip")
    if df.empty or not rep_fracs: return np.nan
    f=0.5*(df.start_cycle.to_numpy()+df.end_cycle.to_numpy()); TT=float(df.end_cycle.max()-df.start_cycle.min()); f=f/(TT if TT>0 else 1.0)
    used=set(); idx=[]
    for x in rep_fracs:
        for j in np.argsort(np.abs(f-x)):
            if int(j) not in used: used.add(int(j)); idx.append(int(j)); break
    sel=df.iloc[idx].copy()
    # counts
    HS_rd = sel['hit_sram_rd'].sum(); HS_wr = sel['hit_sram_wr'].sum()
    HM_rd = sel['hit_mram_rd'].sum(); HM_wr = sel['hit_mram_wr'].sum()
    M     = (sel['miss_rd']+sel['miss_wr']).sum()
    # dynamic energy
    E_dyn = HS_rd*E_RS + HS_wr*E_WS + HM_rd*E_RM + HM_wr*E_WM + M*(E_MISS_BASE + E_MISS_PER_NS*(float(row.get("t_dram",200.0))/f_ghz))
    # time over selected windows
    Tk_kcyc = float(((sel.end_cycle - sel.start_cycle).sum())/1000.0)
    # leakage with this run's partition
    pw=float(row["pi_way"]); C_S=(1.0-pw)*L3; C_M=pw*L3
    T_1KSEC = 1e-6 / f_ghz
    E_leak = (P_LS*C_S + P_LM*C_M) * Tk_kcyc * T_1KSEC
    return E_dyn + E_leak

# ---------- VALIDATION ----------
def make_validation(cap, profile, outdir, reweight_profile=""):
    ds = load_dataset(cap)
    ph = load_physics(cap, profile)
    m = round_keys(ds).merge(round_keys(ph), on=JOIN, how="inner", suffixes=("_eval","_phys"))
    if m.empty:
        raise RuntimeError("No overlap between eval and physics predictions.")

    # Optional: recompute eval energy with a device profile (e.g., NVSim)
    if reweight_profile:
        print(f"[validation] re-weighting eval energy with device profile: {reweight_profile}")
        dj = load_device(reweight_profile, cap)
        if not dj:
            raise FileNotFoundError(f"device not found: devices/{reweight_profile}")
        E_eval_re = []
        for _,row in m.iterrows():
            try:
                E_eval_re.append(reweighted_eval_energy(cap, row, dj))
            except Exception as e:
                # print(f"Warning: reweighting failed for {row['bench']}/{row['tag']}: {e}")
                E_eval_re.append(np.nan)
        m["E_total_J_eval"] = np.array(E_eval_re)  # overwrite eval energy with reweighted values
        m = m.dropna(subset=["E_total_J_eval"]) # drop rows where reweighting failed
        if m.empty:
            raise RuntimeError("No data remaining after re-weighting energy. Check for errors.")

    # per-bench errors
    rows=[]
    for b, g in m.groupby("bench"):
        e_stall = np.abs(g["stall_pct_phys"] - g["stall_pct_eval"]).to_numpy()         # fraction
        e_energy_abs = np.abs(g["E_total_J_phys"] - g["E_total_J_eval"]).to_numpy()   # J
        denom = np.where(g["E_total_J_eval"].to_numpy() > 0, g["E_total_J_eval"].to_numpy(), np.nan)
        e_energy_pct = e_energy_abs/denom                                             # fraction
        rows.append({
            "bench": b,
            "MAE_stall_frac": np.nanmean(e_stall),
            "P90_stall_frac": np.nanpercentile(e_stall, 90),
            "MAE_energy_J": np.nanmean(e_energy_abs),
            "MAPE_energy": np.nanmean(e_energy_pct),
            "P90_APE_energy": np.nanpercentile(e_energy_pct, 90)
        })
    err = pd.DataFrame(rows).sort_values("bench")
    ensure_dir(outdir)
    reweight_tag = f"_reweighted_{reweight_profile}" if reweight_profile else ""
    csv = os.path.join(outdir, f"validation_errors_L3_{cap}_{profile or 'current'}{reweight_tag}.csv")
    err.to_csv(csv, index=False)

    # figure: per-bench bars (stall MAE %, energy MAPE %)
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    x = np.arange(len(err))
    axs[0].bar(x, err["MAE_stall_frac"].values*100.0, width=0.7)
    axs[0].set_title("Stall MAE"); axs[0].set_ylabel("%"); axs[0].grid(True, axis='y', alpha=0.3)
    axs[1].bar(x, err["MAPE_energy"].values*100.0, width=0.7, color="tab:orange")
    axs[1].set_title("Energy MAPE"); axs[1].set_ylabel("%"); axs[1].grid(True, axis='y', alpha=0.3)
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels([short(b) for b in err["bench"].tolist()], rotation=60, fontsize=8)
    title = f"Validation vs eval — cap={cap}MB  profile={profile or 'current'}"
    if reweight_profile:
        title += f"\n(Eval energy reweighted with {reweight_profile})"
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0,0,1,0.95])
    png = os.path.join(outdir, f"validation_errors_L3_{cap}_{profile or 'current'}{reweight_tag}.png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[validation] wrote\n  {csv}\n  {png}")

# ---------- WORKLOAD CHARACTERIZATION ----------
def make_workload(cap, outdir):
    fp = f"results/surrogate/L3_{cap}/benchmark_fingerprints.csv"
    if not os.path.exists(fp):
        raise FileNotFoundError(f"missing {fp} (run postprocess_after_eval.sh)")
    df = pd.read_csv(fp).sort_values("bench")
    # figure: 2x2 – miss rate, acc/1kcyc, read frac, mpkc vs miss scatter
    fig, axs = plt.subplots(2,2, figsize=(12,7))
    x = np.arange(len(df))

    axs[0,0].bar(x, df["char_miss_rate"].values*100.0, width=0.7)
    axs[0,0].set_title("Miss rate (%)"); axs[0,0].grid(True, axis='y', alpha=0.3)

    axs[0,1].bar(x, df["char_acc_per_1kcyc"].values, width=0.7, color="tab:green")
    axs[0,1].set_title("Accesses per 1k cycles"); axs[0,1].grid(True, axis='y', alpha=0.3)

    axs[1,0].bar(x, df["char_read_frac"].values*100.0, width=0.7, color="tab:orange")
    axs[1,0].set_title("Read fraction (%)"); axs[1,0].grid(True, axis='y', alpha=0.3)

    axs[1,1].scatter(df["char_miss_rate"].values*100.0, df["char_mpkc"].values, s=20)
    for i, b in enumerate(df["bench"].tolist()):
        axs[1,1].annotate(short(b), (df["char_miss_rate"].iloc[i]*100.0, df["char_mpkc"].iloc[i]),
                          textcoords="offset points", xytext=(2,2), fontsize=8)
    axs[1,1].set_xlabel("Miss rate (%)"); axs[1,1].set_ylabel("MPKC")

    for ax in axs.flat:
        if ax in (axs[0,0], axs[0,1], axs[1,0]):
            ax.set_xticks(x); ax.set_xticklabels([short(b) for b in df["bench"].tolist()], rotation=60, fontsize=8)

    fig.suptitle(f"Workload characterization — cap={cap}MB (fingerprints over reps)", y=0.98)
    fig.tight_layout(rect=[0,0,1,0.95])
    png = os.path.join(outdir, f"workload_fingerprints_L3_{cap}.png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[workload] wrote\n  {png}")

def main():
    ap = argparse.ArgumentParser(description="Validation (error) + workload characterization")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--profile", default="", help="device profile name for physics parity (e.g., stt_28nm_1p0v)")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--reweight-energy-with-device", default="",
                    help="Device profile to reweight eval energy (e.g., nvsim_n32_readedp)")
    args = ap.parse_args()
    outdir = args.outdir or f"results/figs/L3_{args.cap}/validation"
    ensure_dir(outdir)
    try:
        if args.profile:
            make_validation(args.cap, args.profile, outdir, reweight_profile=args.reweight_energy_with_device)
        else:
            print("[validation] skipped (no --profile given)")
    except Exception as e:
        print(f"[validation] FAILED: {e}")

    try:
        make_workload(args.cap, outdir)
    except Exception as e:
        print(f"[workload] FAILED: {e}")


if __name__=="__main__":
    main()
