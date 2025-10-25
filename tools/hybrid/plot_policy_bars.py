#!/usr/bin/env python3
import os, json, argparse, numpy as np, pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JOIN = ["bench","pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr","tag"]

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- device JSON (profile/base + optional per-cap override) ----------
def load_device(profile:str, cap:int, root:str="devices"):
    if not profile: return None
    base=os.path.join(root, profile, "base.json")
    over=os.path.join(root, profile, f"L3_{cap}.json")
    def _load(p): return json.load(open(p)) if os.path.exists(p) else {}
    if not os.path.exists(base) and not os.path.exists(over): return None
    dj=_load(base); ov=_load(over)
    for k,v in ov.items():
        if isinstance(v,dict) and k in dj and isinstance(dj[k],dict): dj[k].update(v)
        else: dj[k]=v
    return dj

# ---------- time mapping helpers (rep-window scaling) ----------
def mid_fracs(df):
    mids=0.5*(df.start_cycle.to_numpy()+df.end_cycle.to_numpy())
    T=float(df.end_cycle.max()-df.start_cycle.min())
    return mids/(T if T>0 else 1.0)

def load_rep_fracs(char_dir):
    reps=json.load(open(os.path.join(char_dir,"LLC.representatives.json")))["representatives"]
    feats=pd.read_csv(os.path.join(char_dir,"LLC.window_features.csv"), engine="python", on_bad_lines="skip")
    d=dict(zip(feats.window_id.to_numpy(), mid_fracs(feats)))
    return [d[w] for w in reps if w in d]

def pick_unique(csv, fracs):
    df=pd.read_csv(csv, engine="python", on_bad_lines="skip")
    if df.empty or not fracs: return df.iloc[0:0].copy()
    f=mid_fracs(df); used=set(); idx=[]
    for x in fracs:
        for j in np.argsort(np.abs(f-x)):
            if int(j) not in used:
                used.add(int(j)); idx.append(int(j)); break
    return df.iloc[idx].copy()

# ---------- choose canonical row per bench (closest to target knobs) ----------
def pick_canonical_rows(ds:pd.DataFrame, target_pw=0.50, target_pm=0.50):
    rows=[]
    for b,grp in ds.groupby("bench"):
        # L2 distance in (pi_way, pi_miss); tie-break by near-default latencies
        d=(grp["pi_way"]-target_pw)**2 + (grp["pi_miss"]-target_pm)**2 \
           + 1e-4*((grp["t_sram_hit"]-16)**2 + (grp["t_mram_rd"]-28)**2 + (grp["t_mram_wr"]-60)**2)
        rows.append(grp.iloc[int(np.argmin(d.values))])
    return pd.DataFrame(rows).reset_index(drop=True)

# ---------- physics of one case, with dynamic breakdown ----------
def eval_case(row, fp_row, L3_MB, Tk_kcyc, f_ghz, E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS,
              HM_rd, HM_wr, latency):
    # latencies
    tS,tMr,tMw,tDR = latency
    # counts per 1k cycles (from fingerprint)
    A1k=float(fp_row.char_acc_per_1kcyc); mr=float(fp_row.char_miss_rate); rf=float(fp_row.char_read_frac)
    M1k=mr*A1k; H1k=A1k-M1k
    HS_rd = rf*H1k - HM_rd
    HS_wr = (1-rf)*H1k - HM_wr
    # stall (per 1k → fraction)
    mlp_hit  = max(1.0, float(fp_row.char_mlp_hit))
    mlp_miss = max(1.0, float(fp_row.char_mlp_miss))
    d_rd=tMr-tS; d_wr=tMw-tS; d_mi=tDR-tS
    stall_k = (HM_rd*d_rd + HM_wr*d_wr)/mlp_hit + (M1k*d_mi)/mlp_miss
    stall_frac = stall_k/1000.0
    # energy per 1k
    E_rd_hits_1k = HS_rd*E_RD_SRAM + HM_rd*E_RD_MRAM
    E_wr_hits_1k = HS_wr*E_WR_SRAM + HM_wr*E_WR_MRAM
    E_miss_1k    = M1k*E_MISS
    E_dyn_1k     = E_rd_hits_1k + E_wr_hits_1k + E_miss_1k
    # leakage over Tk
    C_S=(1.0-row.pi_way)*L3_MB; C_M=row.pi_way*L3_MB
    T_1KSEC = 1e-6 / f_ghz
    E_leak = (P_LEAK_S*C_S + P_LEAK_M*C_M) * Tk_kcyc*T_1KSEC
    # totals
    E_tot = E_leak + E_dyn_1k*Tk_kcyc
    # execution time: base + stall, base time = Tk_kcyc * (1k cycles) / f
    exec_sec = Tk_kcyc*T_1KSEC*(1.0 + stall_frac)
    return dict(stall_frac=stall_frac, exec_sec=exec_sec,
                E_tot=E_tot, E_dyn=E_dyn_1k*Tk_kcyc, E_leak=E_leak,
                E_rd_hits=E_rd_hits_1k*Tk_kcyc, E_wr_hits=E_wr_hits_1k*Tk_kcyc, E_miss=E_miss_1k*Tk_kcyc)

def main():
    ap=argparse.ArgumentParser(description="Policy bounds bars (OPT/WORST vs SRAM baseline) with dynamic breakdown")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--device-profile", required=True)
    ap.add_argument("--device-root", default="devices")
    ap.add_argument("--latency-mode", choices=["dataset","device"], default="device")
    ap.add_argument("--target-pw", type=float, default=0.50)
    ap.add_argument("--target-pm", type=float, default=0.50)
    ap.add_argument("--outdir", default="")
    args=ap.parse_args()

    cap=args.cap
    outdir=args.outdir or f"results/figs/L3_{cap}/{args.device_profile}"
    ensure_dir(outdir)

    # load datasets
    ds = pd.read_csv(f"results/eval/dataset_L3_{cap}MB.csv")
    fp = pd.read_csv(f"results/surrogate/L3_{cap}/benchmark_fingerprints.csv").set_index("bench")

    # device params
    dj = load_device(args.device_profile, cap, args.device_root)
    if not dj:
        raise FileNotFoundError(f"device profile not found: {args.device_root}/{args.device_profile}")
    f_ghz = float(dj.get("f_clk_ghz",1.0))
    E_RD_SRAM = dj["sram"]["e_rd_pj"]*1e-12
    E_WR_SRAM = dj["sram"]["e_wr_pj"]*1e-12
    E_RD_MRAM = dj["mram"]["e_rd_pj"]*1e-12
    E_WR_MRAM = dj["mram"]["e_wr_pj"]*1e-12
    P_LEAK_S  = dj["sram"]["leak_mw_per_mb"]*1e-3
    P_LEAK_M  = dj["mram"]["leak_mw_per_mb"]*1e-3
    dram      = dj.get("dram", {})
    E_MISS_BASE   = dram.get("e_miss_pj", 3000.0)*1e-12
    E_MISS_PER_NS = dram.get("e_miss_per_ns_pj", 0.0)*1e-12
    LAT          = dj.get("latency", {})

    # pick canonical row per bench near target pw/pm
    rows = pick_canonical_rows(ds, target_pw=args.target_pw, target_pm=args.target_pm)
    rows = rows.sort_values("bench").reset_index(drop=True)

    benches = rows["bench"].tolist()
    # preallocate outputs
    exec_opt=[]; exec_worst=[]; exec_base=[]
    Etot_opt=[]; Etot_worst=[]; Etot_base=[]
    Edyn_opt=[]; Edyn_worst=[]; Edyn_base=[]
    Eleak_opt=[]; Eleak_worst=[]; Eleak_base=[]
    # dynamic breakdown per bench for OPT/WORST
    Edyn_rd_opt=[]; Edyn_wr_opt=[]; Edyn_ms_opt=[]
    Edyn_rd_wst=[]; Edyn_wr_wst=[]; Edyn_ms_wst=[]

    # loop benches
    for _,r in rows.iterrows():
        b=r["bench"]; L3 = float(r.get("l3_mb", cap))
        # map reps to this tag to get Tk
        cdir=f"results/characterization_L3_{int(L3)}/{b}"
        rep_fr=load_rep_fracs(cdir)
        run_csv=f"results/eval/{b}/{r['tag']}/LLC.llc.win.csv"
        reps=pick_unique(run_csv, rep_fr)
        Tk = float(((reps.end_cycle - reps.start_cycle).sum())/1000.0)  # kcycles

        # per-row latencies: dataset or device
        tS=float(r["t_sram_hit"]); tMr=float(r["t_mram_rd"]); tMw=float(r["t_mram_wr"]); tDR=float(r.get("t_dram",200.0))
        if args.latency_mode=="device":
            tS  = float(LAT.get("t_sram_hit", tS))
            tMr = float(LAT.get("t_mram_rd",  tMr))
            tMw = float(LAT.get("t_mram_wr",  tMw))
            tDR = float(LAT.get("t_miss_cycles", tDR))

        # miss energy possibly depends on tDR (ns = cycles / f)
        E_MISS = E_MISS_BASE + E_MISS_PER_NS*(tDR / f_ghz)

        # fingerprint row
        if b not in fp.index: continue
        fpr = fp.loc[b]

        # Event-wise OPT/WORST for dynamic energy
        Hrd = float(fpr.char_read_frac) * (float(fpr.char_acc_per_1kcyc) - float(fpr.char_miss_rate)*float(fpr.char_acc_per_1kcyc))
        Hwr = (1.0 - float(fpr.char_read_frac)) * (float(fpr.char_acc_per_1kcyc) - float(fpr.char_miss_rate)*float(fpr.char_acc_per_1kcyc))
        # OPT: put events on lower dynamic energy medium
        HM_rd_opt = Hrd if E_RD_MRAM < E_RD_SRAM else 0.0
        HM_wr_opt = Hwr if E_WR_MRAM < E_WR_SRAM else 0.0
        # WORST: opposite
        HM_rd_wst = Hrd if E_RD_MRAM > E_RD_SRAM else 0.0
        HM_wr_wst = Hwr if E_WR_MRAM > E_WR_SRAM else 0.0
        # BASE: SRAM-only (HM=0)
        HM_rd_base = 0.0; HM_wr_base=0.0

        lat_tuple = (tS,tMr,tMw,tDR)

        opt  = eval_case(r, fpr, L3, Tk, f_ghz,E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS, HM_rd_opt, HM_wr_opt, lat_tuple)
        worst= eval_case(r, fpr, L3, Tk, f_ghz,E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS, HM_rd_wst, HM_wr_wst, lat_tuple)
        base = eval_case(r, fpr, L3, Tk, f_ghz,E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS, HM_rd_base, HM_wr_base, lat_tuple)

        exec_opt.append(opt["exec_sec"]); exec_worst.append(worst["exec_sec"]); exec_base.append(base["exec_sec"])
        Etot_opt.append(opt["E_tot"]);    Etot_worst.append(worst["E_tot"]);    Etot_base.append(base["E_tot"])
        Edyn_opt.append(opt["E_dyn"]);    Edyn_worst.append(worst["E_dyn"]);    Edyn_base.append(base["E_dyn"])
        Eleak_opt.append(opt["E_leak"]);  Eleak_worst.append(worst["E_leak"]);  Eleak_base.append(base["E_leak"])
        Edyn_rd_opt.append(opt["E_rd_hits"]); Edyn_wr_opt.append(opt["E_wr_hits"]); Edyn_ms_opt.append(opt["E_miss"])
        Edyn_rd_wst.append(worst["E_rd_hits"]); Edyn_wr_wst.append(worst["E_wr_hits"]); Edyn_ms_wst.append(worst["E_miss"])

    # convert to arrays
    exec_opt=np.array(exec_opt); exec_worst=np.array(exec_worst); exec_base=np.array(exec_base)
    Etot_opt=np.array(Etot_opt); Etot_worst=np.array(Etot_worst); Etot_base=np.array(Etot_base)
    Edyn_opt=np.array(Edyn_opt); Edyn_worst=np.array(Edyn_worst); Edyn_base=np.array(Edyn_base)
    Eleak_opt=np.array(Eleak_opt); Eleak_worst=np.array(Eleak_worst)
    Edyn_rd_opt=np.array(Edyn_rd_opt); Edyn_wr_opt=np.array(Edyn_wr_opt); Edyn_ms_opt=np.array(Edyn_ms_opt)
    Edyn_rd_wst=np.array(Edyn_rd_wst); Edyn_wr_wst=np.array(Edyn_wr_wst); Edyn_ms_wst=np.array(Edyn_ms_wst)

    # x positions
    N=len(benches); x=np.arange(N); w=0.38

    fig, axs = plt.subplots(3,2, figsize=(14,10))
    # ---- Exec time: raw ----
    ax=axs[0,0]
    ax.bar(x-w/2, exec_opt, width=w, label="Event-wise OPT (dyn)", color="tab:green")
    ax.bar(x+w/2, exec_worst, width=w, label="Event-wise WORST (dyn)", color="tab:red")
    ax.set_title(f"Execution time — raw  (cap={cap}MB, π_way≈{args.target_pw:.2f}, π_miss≈{args.target_pm:.2f})")
    ax.set_ylabel("seconds"); ax.set_xticks(x); ax.set_xticklabels([b.split('.')[1] for b in benches], rotation=60, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3); ax.legend(fontsize=8)

    # ---- Exec time: ratio vs SRAM baseline ----
    ax=axs[0,1]
    ax.bar(x-w/2, exec_opt/exec_base, width=w, color="tab:green")
    ax.bar(x+w/2, exec_worst/exec_base, width=w, color="tab:red")
    ax.axhline(1.0, color='tab:blue', linestyle='--', linewidth=1)
    ax.set_title("Execution time — ratio vs SRAM baseline")
    ax.set_ylabel("× baseline"); ax.set_xticks(x); ax.set_xticklabels([b.split('.')[1] for b in benches], rotation=60, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    # ---- Total energy: raw (stack dynamic+leak) ----
    ax=axs[1,0]
    ax.bar(x-w/2, Edyn_opt, width=w, color="tab:gray", label="Dynamic")
    ax.bar(x-w/2, Eleak_opt, bottom=Edyn_opt, width=w, color="white", edgecolor="tab:gray", hatch="//", label="Leakage")
    ax.bar(x+w/2, Edyn_worst, width=w, color="tab:gray")
    ax.bar(x+w/2, Eleak_worst, bottom=Edyn_worst, width=w, color="white", edgecolor="tab:gray", hatch="//")
    ax.set_title("Total LLC energy — raw")
    ax.set_ylabel("Joules"); ax.set_xticks(x); ax.set_xticklabels([b.split('.')[1] for b in benches], rotation=60, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # ---- Total energy: ratio vs SRAM baseline ----
    ax=axs[1,1]
    ax.bar(x-w/2, Etot_opt/Etot_base, width=w, color="tab:green")
    ax.bar(x+w/2, Etot_worst/Etot_base, width=w, color="tab:red")
    ax.axhline(1.0, color='tab:blue', linestyle='--', linewidth=1)
    ax.set_title("Total LLC energy — ratio vs SRAM baseline")
    ax.set_ylabel("× baseline"); ax.set_xticks(x); ax.set_xticklabels([b.split('.')[1] for b in benches], rotation=60, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    # ---- Dynamic energy: raw breakdown ----
    ax=axs[2,0]
    # OPT stacks
    ax.bar(x-w/2, Edyn_rd_opt, width=w, color="tab:blue", label="Read hits")
    ax.bar(x-w/2, Edyn_wr_opt, bottom=Edyn_rd_opt, width=w, color="tab:orange", label="Write hits")
    ax.bar(x-w/2, Edyn_ms_opt, bottom=Edyn_rd_opt+Edyn_wr_opt, width=w, color="tab:green", label="Miss installs")
    # WORST stacks
    ax.bar(x+w/2, Edyn_rd_wst, width=w, color="tab:blue")
    ax.bar(x+w/2, Edyn_wr_wst, bottom=Edyn_rd_wst, width=w, color="tab:orange")
    ax.bar(x+w/2, Edyn_ms_wst, bottom=Edyn_rd_wst+Edyn_wr_wst, width=w, color="tab:green")
    ax.set_title("Dynamic energy — raw breakdown")
    ax.set_ylabel("Joules"); ax.set_xticks(x); ax.set_xticklabels([b.split('.')[1] for b in benches], rotation=60, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, title="Dynamic components")

    # ---- Dynamic energy: ratio vs SRAM baseline ----
    ax=axs[2,1]
    ax.bar(x-w/2, Edyn_opt/Edyn_base, width=w, color="tab:green")
    ax.bar(x+w/2, Edyn_worst/Edyn_base, width=w, color="tab:red")
    ax.axhline(1.0, color='tab:blue', linestyle='--', linewidth=1)
    ax.set_title("Dynamic energy — ratio vs SRAM baseline")
    ax.set_ylabel("× baseline"); ax.set_xticks(x); ax.set_xticklabels([b.split('.')[1] for b in benches], rotation=60, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    out = os.path.join(outdir, "policy_bars.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"[done] wrote {out}")

if __name__ == "__main__":
    main()

