#!/usr/bin/env python3
import os, json, glob, argparse, numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JOIN=["bench","pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr","tag"]

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def bshort(b): 
    # "620.omnetpp_s-874B" -> "omnetpp"; "605.mcf_s-994B" -> "mcf"
    try: return b.split('.',1)[1].split('_',1)[0]
    except: return b

# ---------- device JSON ----------
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

# ---------- rep-window mapping (time scaling) ----------
def mid_fracs(df):
    mids=0.5*(df.start_cycle.to_numpy()+df.end_cycle.to_numpy())
    T=float(df.end_cycle.max()-df.start_cycle.min())
    return mids/(T if T>0 else 1.0)

def load_rep_fracs(cdir):
    reps=json.load(open(os.path.join(cdir,"LLC.representatives.json")))["representatives"]
    feats=pd.read_csv(os.path.join(cdir,"LLC.window_features.csv"), engine="python", on_bad_lines="skip")
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

# ---------- HM curve ----------
def iso_eval(curve, x):
    xs=np.asarray(curve["x"], float); ys=np.asarray(curve["y"], float)
    return float(np.interp(x, xs, ys)) if xs.size>0 else 0.5

def get_frac_mram(curves, bench, pm, pw):
    tb=curves.get(bench, {})
    key=f"{pm:.4f}" if f"{pm:.4f}" in tb else ("__pooled__" if "__pooled__" in tb else None)
    if key is None: return None
    return iso_eval(tb[key], pw)

def hm_benches(cap):
    hp=f"results/surrogate/L3_{cap}/hm_curves.joblib"
    if not os.path.exists(hp): return []
    curves=joblib.load(hp)
    preferred=['602.gcc_s-1850B','605.mcf_s-994B','619.lbm_s-2677B','620.omnetpp_s-874B',
               '621.wrf_s-6673B','623.xalancbmk_s-700B','631.deepsjeng_s-928B',
               '649.fotonik3d_s-7084B','657.xz_s-3167B']
    benches=sorted(curves.keys())
    order=[b for b in preferred if b in benches]+[b for b in benches if b not in preferred]
    return order[:9]

# ---------- canonical row per bench ----------
def pick_canonical_rows(ds:pd.DataFrame, target_pw=0.50, target_pm=0.50, only_benches=None):
    if only_benches:
        ds=ds[ds["bench"].isin(only_benches)]
    rows=[]
    for b,grp in ds.groupby("bench"):
        d=(grp["pi_way"]-target_pw)**2 + (grp["pi_miss"]-target_pm)**2 \
           + 1e-4*((grp["t_sram_hit"]-16)**2 + (grp["t_mram_rd"]-28)**2 + (grp["t_mram_wr"]-60)**2)
        rows.append(grp.iloc[int(np.argmin(d.values))])
    return pd.DataFrame(rows).reset_index(drop=True)

# ---------- physics ----------
def eval_case(row, fp_row, L3_MB, Tk_kcyc, f_ghz,
              E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS,
              HM_rd, HM_wr, latency, leak_pi_way=None):
    tS,tMr,tMw,tDR = latency
    A1k=float(fp_row.char_acc_per_1kcyc); mr=float(fp_row.char_miss_rate); rf=float(fp_row.char_read_frac)
    M1k=mr*A1k; H1k=A1k-M1k
    HS_rd = rf*H1k - HM_rd
    HS_wr = (1-rf)*H1k - HM_wr
    mlp_hit  = max(1.0, float(fp_row.char_mlp_hit))
    mlp_miss = max(1.0, float(fp_row.char_mlp_miss))
    d_rd=tMr-tS; d_wr=tMw-tS; d_mi=tDR-tS
    stall_k = (HM_rd*d_rd + HM_wr*d_wr)/mlp_hit + (M1k*d_mi)/mlp_miss
    stall_frac = stall_k/1000.0
    E_rd_hits_1k = HS_rd*E_RD_SRAM + HM_rd*E_RD_MRAM
    E_wr_hits_1k = HS_wr*E_WR_SRAM + HM_wr*E_WR_MRAM
    E_miss_1k    = M1k*E_MISS
    E_dyn_1k     = E_rd_hits_1k + E_wr_hits_1k + E_miss_1k
    pw_leak = float(row.pi_way) if (leak_pi_way is None) else float(leak_pi_way)
    C_S=(1.0-pw_leak)*L3_MB; C_M=pw_leak*L3_MB
    T_1KSEC = 1e-6 / f_ghz
    E_leak = (P_LEAK_S*C_S + P_LEAK_M*C_M) * Tk_kcyc*T_1KSEC
    E_tot = E_leak + E_dyn_1k*Tk_kcyc
    exec_sec = Tk_kcyc*T_1KSEC*(1.0 + stall_frac)
    return dict(stall_frac=stall_frac, exec_sec=exec_sec,
                E_tot=E_tot, E_dyn=E_dyn_1k*Tk_kcyc, E_leak=E_leak,
                E_rd_hits=E_rd_hits_1k*Tk_kcyc, E_wr_hits=E_wr_hits_1k*Tk_kcyc, E_miss=E_miss_1k*Tk_kcyc)

# ---------- fixed-partition allocation ----------
def alloc_fixed_partition(HM_tot, Hrd, Hwr, mode, costs):
    if (Hrd+Hwr) <= 0 or HM_tot<=0: return 0.0, 0.0
    if mode=="mid":
        hm_rd = min(HM_tot * Hrd/(Hrd+Hwr), Hrd)
        hm_wr = min(HM_tot - hm_rd, Hwr)
        return hm_rd, hm_wr
    def greedy(first):
        hm_rd=hm_wr=0.0; rem=HM_tot
        if first=="rd":
            take=min(rem, Hrd); hm_rd+=take; rem-=take
            hm_wr=min(rem, Hwr); rem-=hm_wr
        else:
            take=min(rem, Hwr); hm_wr+=take; rem-=take
            hm_rd=min(rem, Hrd); rem-=hm_rd
        return hm_rd, hm_wr
    if mode=="optE":  first = "rd" if costs["e_rd"] <= costs["e_wr"] else "wr"
    if mode=="wstE":  first = "rd" if costs["e_rd"] >= costs["e_wr"] else "wr"
    if mode=="optS":  first = "rd" if costs["t_rd"] <= costs["t_wr"] else "wr"
    if mode=="wstS":  first = "rd" if costs["t_rd"] >= costs["t_wr"] else "wr"
    return greedy(first)

# ---------- plot builder ----------
def make_bars(benches, variants, exec_raw, exec_base, Etot_raw, Etot_base,
              Edyn_raw, Edyn_base, Eleak_raw, dyn_breakdown, labels, title, out_png,
              sort_by="energy", ratio_pad=0.10):
    # sort benches by baseline energy or time (median across variants)
    idx = np.arange(len(benches))
    if sort_by!="none":
        if sort_by=="energy":
            key = np.nanmedian(Etot_base)
            # same base for all benches; but bar order is clearer if we sort by Etot of first variant
            order = np.argsort(np.nan_to_num(Etot_raw[0], nan=np.inf))[::-1]
        else:
            order = np.argsort(np.nan_to_num(exec_raw[0], nan=np.inf))[::-1]
        benches = [benches[i] for i in order]
        exec_base = exec_base[order]
        Etot_base = Etot_base[order]
        Edyn_base = Edyn_base[order]
        for j in range(len(variants)):
            exec_raw[j] = exec_raw[j][order]
            Etot_raw[j] = Etot_raw[j][order]
            Edyn_raw[j] = Edyn_raw[j][order]
            Eleak_raw[j]= Eleak_raw[j][order]

    # common x
    N=len(benches); G=len(variants)
    x=np.arange(N); w=min(0.8/G, 0.22)

    # helper: format bench labels compactly
    xt = [bshort(b) for b in benches]

    fig, axs = plt.subplots(3,2, figsize=(14,10))
    # Exec raw
    ax=axs[0,0]
    for j in range(G):
        ax.bar(x + (j-(G-1)/2)*w, exec_raw[j], width=w, label=labels[j])
    ax.set_title(f"Exec time — raw  |  {title}")
    ax.set_ylabel("seconds"); ax.set_xticks(x); ax.set_xticklabels(xt, rotation=60, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3); ax.legend(fontsize=8, ncol=min(4,G))

    # Exec ratio
    ax=axs[0,1]
    for j in range(G):
        ax.bar(x + (j-(G-1)/2)*w, exec_raw[j]/exec_base, width=w)
    ax.axhline(1.0, color='tab:blue', linestyle='--', linewidth=1)
    ax.set_title("Exec time — × baseline"); ax.set_ylabel("× baseline"); ax.set_xticks(x); ax.set_xticklabels(xt, rotation=60, fontsize=8)
    ax.set_ylim(1.0-ratio_pad, 1.0+ratio_pad)
    ax.grid(True, axis='y', alpha=0.3)

    # Energy raw (µJ, stacked: dyn + leak)
    ax=axs[1,0]
    for j in range(G):
        ax.bar(x + (j-(G-1)/2)*w, Etot_raw[j]*1e6 - Eleak_raw[j]*1e6, width=w, color="tab:gray", label="Dynamic" if j==0 else None)
        ax.bar(x + (j-(G-1)/2)*w, Eleak_raw[j]*1e6, width=w, bottom=(Etot_raw[j]*1e6 - Eleak_raw[j]*1e6),
               color="white", edgecolor="tab:gray", hatch="//", label="Leakage" if j==0 else None)
    ax.set_title("Total LLC energy — raw"); ax.set_ylabel("μJ"); ax.set_xticks(x); ax.set_xticklabels(xt, rotation=60, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3); ax.legend(loc="upper right", fontsize=8)

    # Energy ratio
    ax=axs[1,1]
    for j in range(G):
        ax.bar(x + (j-(G-1)/2)*w, Etot_raw[j]/Etot_base, width=w)
    ax.axhline(1.0, color='tab:blue', linestyle='--', linewidth=1)
    ax.set_title("Total LLC energy — × baseline"); ax.set_ylabel("× baseline"); ax.set_xticks(x); ax.set_xticklabels(xt, rotation=60, fontsize=8)
    ax.set_ylim(1.0-ratio_pad, 1.0+ratio_pad)
    ax.grid(True, axis='y', alpha=0.3)

    # Dynamic breakdown (median across benches, µJ)
    ax=axs[2,0]
    med = []
    for j in range(G):
        rd = np.nanmedian(dyn_breakdown[j]["rd"]*1e6)
        wr = np.nanmedian(dyn_breakdown[j]["wr"]*1e6)
        ms = np.nanmedian(dyn_breakdown[j]["ms"]*1e6)
        med.append((rd,wr,ms))
    X = np.arange(G)
    rd = [m[0] for m in med]; wr = [m[1] for m in med]; ms = [m[2] for m in med]
    ax.bar(X, rd, width=0.6, color="tab:blue", label="Read hits")
    ax.bar(X, wr, width=0.6, bottom=rd, color="tab:orange", label="Write hits")
    ax.bar(X, ms, width=0.6, bottom=np.array(rd)+np.array(wr), color="tab:green", label="Miss installs")
    ax.set_xticks(X); ax.set_xticklabels(labels, rotation=0)
    ax.set_title("Dynamic energy — median breakdown"); ax.set_ylabel("μJ")
    ax.grid(True, axis='y', alpha=0.3); ax.legend(fontsize=8)

    # Dynamic ratio vs baseline (median across benches)
    ax=axs[2,1]
    dyn_med = [np.nanmedian(Edyn_raw[j] / Edyn_base) for j in range(G)]
    ax.bar(X, dyn_med, width=0.6, color="tab:green")
    ax.axhline(1.0, color='tab:blue', linestyle='--', linewidth=1)
    ax.set_title("Dynamic energy — median × baseline"); ax.set_ylabel("× baseline")
    ax.set_xticks(X); ax.set_xticklabels(labels, rotation=0); ax.set_ylim(1.0-ratio_pad, 1.0+ratio_pad)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser(description="Variation bars: by pi_way or by device")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--mode", choices=["piway","device"], required=True)
    ap.add_argument("--device-profile", default="", help="used in mode=piway")
    ap.add_argument("--device-root", default="devices")
    ap.add_argument("--piways", default="0.00,0.25,0.50,0.75,1.00", help="mode=piway list")
    ap.add_argument("--profiles", default="", help="mode=device comma list; auto-detect if empty")
    ap.add_argument("--latency-mode", choices=["dataset","device"], default="device")
    ap.add_argument("--policy-mode", choices=["mid","optE","wstE","optS","wstS"], default="mid",
                    help="allocation under fixed partition")
    ap.add_argument("--target-pm", type=float, default=0.50, help="π_miss for HM curves")
    ap.add_argument("--target-pw", type=float, default=0.50, help="canonical row selection for mode=device")
    ap.add_argument("--baseline", choices=["sram","row"], default="sram")
    ap.add_argument("--sort-by", choices=["energy","time","none"], default="energy")
    ap.add_argument("--ratio-pad", type=float, default=0.10)
    ap.add_argument("--outdir", default="", help="output dir")
    args=ap.parse_args()

    cap=args.cap
    outdir = args.outdir or f"results/figs/L3_{cap}/variations"
    ensure_dir(outdir)

    # Load datasets + HM benches → filter to hide unsuccessful benches
    ds = pd.read_csv(f"results/eval/dataset_L3_{cap}MB.csv")
    fp = pd.read_csv(f"results/surrogate/L3_{cap}/benchmark_fingerprints.csv").set_index("bench")
    curves = joblib.load(f"results/surrogate/L3_{cap}/hm_curves.joblib") if os.path.exists(f"results/surrogate/L3_{cap}/hm_curves.joblib") else {}
    benches_keep = hm_benches(cap) or sorted(ds["bench"].unique().tolist())[:9]
    rows = pick_canonical_rows(ds, target_pw=(args.target_pW if hasattr(args,'target_pW') else args.target_pw),
                               target_pm=args.target_pm, only_benches=benches_keep).sort_values("bench").reset_index(drop=True)
    benches = rows["bench"].tolist()

    # helper: one variant across benches
    def compute_variant(device_profile, latency_mode, pi_way_override=None):
        dj = load_device(device_profile, cap, args.device_root) if device_profile else None
        if dj is None and device_profile:
            raise FileNotFoundError(f"device profile not found: {args.device_root}/{device_profile}")
        f_ghz = float(dj.get("f_clk_ghz",1.0)) if dj else 1.0
        E_RD_SRAM = dj["sram"]["e_rd_pj"]*1e-12 if dj else 0.3e-9
        E_WR_SRAM = dj["sram"]["e_wr_pj"]*1e-12 if dj else 0.4e-9
        E_RD_MRAM = dj["mram"]["e_rd_pj"]*1e-12 if dj else 0.6e-9
        E_WR_MRAM = dj["mram"]["e_wr_pj"]*1e-12 if dj else 0.8e-9
        P_LEAK_S  = dj["sram"]["leak_mw_per_mb"]*1e-3 if dj else 5e-3
        P_LEAK_M  = dj["mram"]["leak_mw_per_mb"]*1e-3 if dj else 0.5e-3
        dram      = dj.get("dram", {}) if dj else {}
        E_MISS_BASE   = dram.get("e_miss_pj", 3000.0)*1e-12
        E_MISS_PER_NS = dram.get("e_miss_per_ns_pj", 0.0)*1e-12
        lat_json      = dj.get("latency", {}) if dj else {}

        exec_raw=[]; Etot_raw=[]; Edyn_raw=[]; Eleak_raw=[]
        Edyn_rd=[]; Edyn_wr=[]; Edyn_ms=[]
        exec_base=[]; Etot_base=[]; Edyn_base=[]

        for _,r in rows.iterrows():
            b=r["bench"]; L3 = float(r.get("l3_mb", cap))
            if b not in fp.index or (curves and b not in curves): 
                # skip benches without fingerprints or HM curve
                exec_raw.append(np.nan); Etot_raw.append(np.nan); Edyn_raw.append(np.nan); Eleak_raw.append(np.nan)
                Edyn_rd.append(np.nan); Edyn_wr.append(np.nan); Edyn_ms.append(np.nan)
                exec_base.append(np.nan); Etot_base.append(np.nan); Edyn_base.append(np.nan)
                continue
            # rep-window time
            cdir=f"results/characterization_L3_{int(L3)}/{b}"
            rep_fr=load_rep_fracs(cdir)
            run_csv=f"results/eval/{b}/{r['tag']}/LLC.llc.win.csv"
            reps=pick_unique(run_csv, rep_fr)
            Tk = float(((reps.end_cycle - reps.start_cycle).sum())/1000.0)
            # latencies
            tS=float(r["t_sram_hit"]); tMr=float(r["t_mram_rd"]); tMw=float(r["t_mram_wr"]); tDR=float(r.get("t_dram",200.0))
            if latency_mode=="device" and dj:
                tS  = float(lat_json.get("t_sram_hit", tS))
                tMr = float(lat_json.get("t_mram_rd",  tMr))
                tMw = float(lat_json.get("t_mram_wr",  tMw))
                tDR = float(lat_json.get("t_miss_cycles", tDR))
            E_MISS = E_MISS_BASE + E_MISS_PER_NS*(tDR / f_ghz)

            fpr = fp.loc[b]
            # counts per 1k
            A1k=float(fpr.char_acc_per_1kcyc); mr=float(fpr.char_miss_rate); rf=float(fpr.char_read_frac)
            M1k=mr*A1k; H1k=A1k-M1k
            Hrd = rf*H1k; Hwr = (1-rf)*H1k

            # partition (π_way to use for HM curve and leakage)
            pw = float(r["pi_way"]) if (pi_way_override is None) else float(pi_way_override)
            frac = get_frac_mram(curves, b, float(r["pi_miss"]), pw) if curves else None
            if frac is None:
                exec_raw.append(np.nan); Etot_raw.append(np.nan); Edyn_raw.append(np.nan); Eleak_raw.append(np.nan)
                Edyn_rd.append(np.nan); Edyn_wr.append(np.nan); Edyn_ms.append(np.nan)
                exec_base.append(np.nan); Etot_base.append(np.nan); Edyn_base.append(np.nan)
                continue
            HM_tot = frac * H1k
            e_rd = (E_RD_MRAM - E_RD_SRAM)
            e_wr = (E_WR_MRAM - E_WR_SRAM)
            t_rd = (tMr - tS) / max(1.0, float(fpr.char_mlp_hit))
            t_wr = (tMw - tS) / max(1.0, float(fpr.char_mlp_hit))
            HM_rd, HM_wr = alloc_fixed_partition(HM_tot, Hrd, Hwr, args.policy_mode,
                                                 {"e_rd":e_rd,"e_wr":e_wr,"t_rd":t_rd,"t_wr":t_wr})
            lat_tuple=(tS,tMr,tMw,tDR)
            res = eval_case(r, fpr, L3, Tk, f_ghz,
                            E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS,
                            HM_rd, HM_wr, lat_tuple, leak_pi_way=pw)
            exec_raw.append(res["exec_sec"]); Etot_raw.append(res["E_tot"])
            Edyn_raw.append(res["E_dyn"]);    Eleak_raw.append(res["E_leak"])
            Edyn_rd.append(res["E_rd_hits"]); Edyn_wr.append(res["E_wr_hits"]); Edyn_ms.append(res["E_miss"])

            # baseline (SRAM-only or row)
            if args.baseline=="sram":
                base = eval_case(r, fpr, L3, Tk, f_ghz,
                                 E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS,
                                 HM_rd=0.0, HM_wr=0.0, latency=lat_tuple, leak_pi_way=0.0)
            else:
                base = eval_case(r, fpr, L3, Tk, f_ghz,
                                 E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS,
                                 HM_rd=0.0, HM_wr=0.0, latency=lat_tuple, leak_pi_way=float(r["pi_way"]))
            exec_base.append(base["exec_sec"]); Etot_base.append(base["E_tot"]); Edyn_base.append(base["E_dyn"])

        return (np.array(exec_raw), np.array(exec_base),
                np.array(Etot_raw), np.array(Etot_base),
                np.array(Edyn_raw), np.array(Elek_raw) if False else np.array(Eleak_raw),
                np.array(Edyn_base),
                dict(rd=np.array(Edyn_rd), wr=np.array(Edyn_wr), ms=np.array(Edyn_ms)))

    # ----------- MODE: vary pi_way -----------
    if args.mode=="piway":
        if not args.device_profile:
            raise SystemExit("--device-profile required for mode=piway")
        piways=[float(x) for x in args.piways.split(",")]
        exec_list=[]; Etot_list=[]; Edyn_list=[]; Eleak_list=[]; dyn_list=[]
        exec_base=Etot_base=Edyn_base=None
        labels=[f"pw={pw:.2f}" for pw in piways]
        for pw in piways:
            ex, exb, et, etb, ed, el, edb, db = compute_variant(args.device_profile, args.latency_mode, pi_way_override=pw)
            exec_list.append(ex); Etot_list.append(et); Edyn_list.append(ed); Eleak_list.append(el); dyn_list.append(db)
            exec_base = exb if exec_base is None else exec_base
            Etot_base = etb if Etot_base is None else Etot_base
            Edyn_base = edb if Edyn_base is None else Edyn_base
        out_png=os.path.join(outdir, f"bars_vary_piway_{args.policy_mode}_{args.device_profile}.png")
        make_bars(benches, piways, exec_list, exec_base, Etot_list, Etot_base,
                  Edyn_list, Edyn_base, Eleak_list, dyn_list, labels,
                  title=f"device={args.device_profile}, policy={args.policy_mode}, cap={cap}MB",
                  out_png=out_png, sort_by=args.sort_by, ratio_pad=args.ratio_pad)
        print(f"[done] {out_png}")

    # ----------- MODE: vary device profile -----------
    else:
        profiles = [p for p in args.profiles.split(",") if p] or \
                   sorted({p.split(os.sep)[3] for p in glob.glob(f"results/surrogate/profiles/*/L3_{cap}/physics_pred.csv")})
        if not profiles: raise RuntimeError("no profiles found; pass --profiles")
        exec_list=[]; Etot_list=[]; Edyn_list=[]; Eleak_list=[]; dyn_list=[]
        exec_base=Etot_base=Edyn_base=None
        labels=profiles
        for prof in profiles:
            ex, exb, et, etb, ed, el, edb, db = compute_variant(prof, args.latency_mode, pi_way_override=args.target_pw)
            exec_list.append(ex); Etot_list.append(et); Edyn_list.append(ed); Eleak_list.append(el); dyn_list.append(db)
            exec_base = exb if exec_base is None else exec_base
            Etot_base = etb if Etot_base is None else Etot_base
            Edyn_base = edb if Edyn_base is None else Edyn_base
        out_png=os.path.join(outdir, f"bars_vary_device_pw{args.target_pw:.2f}_{args.policy_mode}.png")
        make_bars(benches, profiles, exec_list, exec_base, Etot_list, Etot_base,
                  Edyn_list, Edyn_base, Eleak_list, dyn_list, labels,
                  title=f"π_way={args.target_pw:.2f}, policy={args.policy_mode}, cap={cap}MB",
                  out_png=out_png, sort_by=args.sort_by, ratio_pad=args.ratio_pad)
        print(f"[done] {out_png}")

if __name__ == "__main__":
    main()



