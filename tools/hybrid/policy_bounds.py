#!/usr/bin/env python3
import os, re, json, argparse, numpy as np, pandas as pd

JOIN=["bench","pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr","tag"]

def mid_fracs(df):
    mids=0.5*(df.start_cycle.to_numpy()+df.end_cycle.to_numpy())
    T=float(df.end_cycle.max()-df.start_cycle.min()); return mids/(T if T>0 else 1.0)

def load_rep_fracs(cdir):
    reps=json.load(open(os.path.join(cdir,"LLC.representatives.json")))["representatives"]
    feats=pd.read_csv(os.path.join(cdir,"LLC.window_features.csv"), engine="python", on_bad_lines="skip")
    d=dict(zip(feats.window_id.to_numpy(), mid_fracs(feats))); return [d[w] for w in reps if w in d]

def pick_unique(csv, fracs):
    df=pd.read_csv(csv, engine="python", on_bad_lines="skip")
    if df.empty or not fracs: return df.iloc[0:0].copy()
    f=mid_fracs(df); used=set(); idx=[]
    for x in fracs:
        for j in np.argsort(np.abs(f-x)):
            if int(j) not in used: used.add(int(j)); idx.append(int(j)); break
    return df.iloc[idx].copy()

def load_device(profile, cap, root):
    if not profile: return None
    base=os.path.join(root, profile, "base.json")
    over=os.path.join(root, profile, f"L3_{cap}.json")
    def _load(p): return json.load(open(p)) if os.path.exists(p) else {}
    dj=_load(base); ov=_load(over)
    for k,v in ov.items():
        if isinstance(v,dict) and k in dj and isinstance(dj[k],dict): dj[k].update(v)
        else: dj[k]=v
    return dj

ap=argparse.ArgumentParser()
ap.add_argument("--device-root", default="devices")
ap.add_argument("--device-profile", default="")
ap.add_argument("--latency-mode", choices=["dataset","device"], default="dataset")
ap.add_argument("--caps", default="")                 # e.g., "32" or "2,128"
args=ap.parse_args()

# discover capacities from datasets
caps=[]
for f in os.listdir("results/eval"):
    m=re.match(r"dataset_L3_(\d+)MB\.csv", f)
    if m: caps.append(int(m.group(1)))
if args.caps:
    caps=[int(x) for x in args.caps.split(",")]

for CAP in caps:
    ds=f"results/eval/dataset_L3_{CAP}MB.csv"
    fp=f"results/surrogate/L3_{CAP}/benchmark_fingerprints.csv"
    if not (os.path.exists(ds) and os.path.exists(fp)):
        print(f"[L3_{CAP}] missing inputs"); continue

    # defaults
    E_RD_SRAM=0.3e-9; E_WR_SRAM=0.4e-9; E_RD_MRAM=0.6e-9; E_WR_MRAM=0.8e-9
    P_LEAK_S=5e-3; P_LEAK_M=0.5e-3; E_MISS_BASE=3.0e-12; E_MISS_PER_NS=0.0
    f_ghz=1.0; T_1KSEC=1e-6; LAT={}
    dj=load_device(args.device_profile, CAP, args.device_root)
    if dj:
        f_ghz=float(dj.get("f_clk_ghz",1.0)); T_1KSEC=1e-6/f_ghz
        E_RD_SRAM=dj["sram"]["e_rd_pj"]*1e-12; E_WR_SRAM=dj["sram"]["e_wr_pj"]*1e-12
        E_RD_MRAM=dj["mram"]["e_rd_pj"]*1e-12; E_WR_MRAM=dj["mram"]["e_wr_pj"]*1e-12
        P_LEAK_S=dj["sram"]["leak_mw_per_mb"]*1e-3; P_LEAK_M=dj["mram"]["leak_mw_per_mb"]*1e-3
        d=dj.get("dram",{}); E_MISS_BASE=d.get("e_miss_pj",3.0)*1e-12; E_MISS_PER_NS=d.get("e_miss_per_ns_pj",0.0)*1e-12
        LAT=dj.get("latency",{})

    D=pd.read_csv(ds); FP=pd.read_csv(fp).set_index("bench"); rows=[]
    for _,r in D.iterrows():
        b=r.bench; tag=r.tag
        if b not in FP.index: continue
        pw=float(r.pi_way); pm=float(r.pi_miss)
        # latencies
        tS=float(r.t_sram_hit); tMr=float(r.t_mram_rd); tMw=float(r.t_mram_wr); tDR=float(r.get("t_dram",200.0))
        if args.latency_mode=="device" and dj:
            tS=float(LAT.get("t_sram_hit",tS)); tMr=float(LAT.get("t_mram_rd",tMr))
            tMw=float(LAT.get("t_mram_wr",tMw)); tDR=float(LAT.get("t_miss_cycles",tDR))
        # counts per 1k cycles
        f=FP.loc[b]; A1k=float(f.char_acc_per_1kcyc); mr=float(f.char_miss_rate); rf=float(f.char_read_frac)
        M1k=mr*A1k; H1k=A1k-M1k; Hrd=rf*H1k; Hwr=(1-rf)*H1k
        # time (selected reps) 
        cdir=f"results/characterization_L3_{int(float(r.get('l3_mb',CAP)))}/{b}"
        fr=load_rep_fracs(cdir); csv=f"results/eval/{b}/{tag}/LLC.llc.win.csv"
        if not os.path.exists(csv): continue
        reps=pick_unique(csv, fr); Tk=float(((reps.end_cycle-reps.start_cycle).sum())/1000.0)
        # leakage over Tk
        C_S=(1-pw)*float(r.get('l3_mb',CAP)); C_M=pw*float(r.get('l3_mb',CAP))
        E_leak=(P_LEAK_S*C_S + P_LEAK_M*C_M)*Tk*(1e-6/f_ghz)
        E_MISS = E_MISS_BASE + E_MISS_PER_NS * (tDR / f_ghz)

        def case(HM_rd,HM_wr,label):
            HS_rd=Hrd-HM_rd; HS_wr=Hwr-HM_wr
            d_rd=tMr-tS; d_wr=tMw-tS; d_mi=tDR-tS
            mh=max(1.0,float(f.char_mlp_hit)); mm=max(1.0,float(f.char_mlp_miss))
            stall_k=(HM_rd*d_rd + HM_wr*d_wr)/mh + (M1k*d_mi)/mm
            E_dyn_1k=HS_rd*E_RD_SRAM + HS_wr*E_WR_SRAM + HM_rd*E_RD_MRAM + HM_wr*E_WR_MRAM + M1k*E_MISS
            rows.append({**r[JOIN].to_dict(),"bound":label,
                         "stall_pct":stall_k/1000.0, "E_total_J":E_leak + E_dyn_1k*Tk})

        # latency-best/worst
        case(Hrd if tMr<tS else 0.0, Hwr if tMw<tS else 0.0, "stall_best")
        case(Hrd if tMr>tS else 0.0, Hwr if tMw>tS else 0.0, "stall_worst")
        # energy-best/worst (per-hit dynamic)
        case(Hrd if E_RD_MRAM<E_RD_SRAM else 0.0, Hwr if E_WR_MRAM<E_WR_SRAM else 0.0, "energy_best")
        case(Hrd if E_RD_MRAM>E_RD_SRAM else 0.0, Hwr if E_WR_MRAM>E_WR_SRAM else 0.0, "energy_worst")

    out=pd.DataFrame(rows)
    os.makedirs(f"results/surrogate/L3_{CAP}", exist_ok=True)
    out.to_csv(f"results/surrogate/L3_{CAP}/policy_bounds.csv", index=False)
    print(f"[L3_{CAP}] wrote results/surrogate/L3_{CAP}/policy_bounds.csv  rows={len(out)}")
