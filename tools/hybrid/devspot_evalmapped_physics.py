import os, json, argparse, numpy as np, pandas as pd, joblib

def load_device_json(profile, cap):
    base=os.path.join("devices", profile, "base.json")
    over=os.path.join("devices", profile, f"L3_{cap}.json")
    def _ld(p): return json.load(open(p)) if os.path.exists(p) else {}
    dj=_ld(base); ov=_ld(over)
    for k,v in ov.items():
        if isinstance(v,dict) and k in dj and isinstance(dj[k],dict): dj[k].update(v)
        else: dj[k]=v
    return dj

def load_eval_mapped_ids(run_dir):
    try:
        m=json.load(open(os.path.join(run_dir,"metrics.json")))
        return [int(x) for x in m.get("rep_windows_mapped",[])]
    except Exception: return []

def iso_eval(curve, x):
    xs=np.asarray(curve["x"], float); ys=np.asarray(curve["y"], float)
    return float(np.interp(x, xs, ys)) if xs.size>0 else 0.5

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--profiles", required=True, help="comma list of devspot profiles")
    ap.add_argument("--base", type=float, default=16.0, help="stall anchor")
    ap.add_argument("--counts-src", choices=["eval","char"], default="eval",
                    help="where to get per-window hit/miss counts for stall physics")
    args=ap.parse_args()

    cap=args.cap; BASE=args.base
    E=pd.read_csv(f"results/eval/dataset_L3_{cap}MB.csv")
    FP=pd.read_csv(f"results/surrogate/L3_{cap}/benchmark_fingerprints.csv").set_index("bench")
    curves=joblib.load(f"results/surrogate/L3_{cap}/hm_curves.joblib")

    for prof in [p for p in args.profiles.split(",") if p]:
        dj=load_device_json(prof, cap)
        f_ghz=float(dj.get("f_clk_ghz",1.0))
        E_RS=dj["sram"]["e_rd_pj"]*1e-12; E_WS=dj["sram"]["e_wr_pj"]*1e-12
        E_RM=dj["mram"]["e_rd_pj"]*1e-12; E_WM=dj["mram"]["e_wr_pj"]*1e-12
        P_LS=dj["sram"]["leak_mw_per_mb"]*1e-3; P_LM=dj["mram"]["leak_mw_per_mb"]*1e-3
        dram=dj.get("dram",{}); E_MISS_BASE=dram.get("e_miss_pj",3.0)*1e-12; E_MISS_PER_NS=dram.get("e_miss_per_ns_pj",0.0)*1e-12
        LAT=dj.get("latency",{})

        out=[]
        for _,r in E.iterrows():
            b=r["bench"]; tag=r["tag"]
            if not str(tag).startswith("devspot_"): continue
            if b not in FP.index or b not in curves: continue
            pw=float(r["pi_way"]); pm=float(r["pi_miss"])
            tS=float(LAT.get("t_sram_hit",   r["t_sram_hit"]))
            tMr=float(LAT.get("t_mram_rd",    r["t_mram_rd"]))
            tMw=float(LAT.get("t_mram_wr",    r["t_mram_wr"]))
            tDR=float(LAT.get("t_miss_cycles", r.get("t_dram",200.0)))
            tb=curves[b]; key=f"{pm:.4f}" if f"{pm:.4f}" in tb else ("__pooled__" if "__pooled__" in tb else None)
            if key is None: continue
            frac=iso_eval(tb[key], pw)

            run_dir=os.path.join("results","eval",b,tag)
            csv=os.path.join(run_dir,"LLC.llc.win.csv")
            if not os.path.exists(csv): continue
            ids=load_eval_mapped_ids(run_dir)
            df=pd.read_csv(csv, engine="python", on_bad_lines="skip")
            if "window_id" in df.columns:
                df["window_id"]=pd.to_numeric(df["window_id"], errors="coerce").fillna(-1).astype(int)
            reps=df[df["window_id"].isin(ids)].copy() if ids else df.iloc[0:0].copy()
            if reps.empty: continue

            win=(reps["end_cycle"]-reps["start_cycle"]).astype(float)
            T1K=np.maximum(win/1000.0,1e-9)

            # === Per-window counts source ===
            if args.counts_src == "eval":
                # use eval per-window totals (A2A transform; often near-zero MAE)
                Hs_rd=reps["hit_sram_rd"].to_numpy(float); Hs_wr=reps["hit_sram_wr"].to_numpy(float)
                Hm_rd_obs=reps["hit_mram_rd"].to_numpy(float); Hm_wr_obs=reps["hit_mram_wr"].to_numpy(float)
                M=(reps["miss_rd"]+reps["miss_wr"]).to_numpy(float)
                Hrd_tot=Hs_rd+Hm_rd_obs; Hwr_tot=Hs_wr+Hm_wr_obs; Htot=Hrd_tot+Hwr_tot
                rd_share=np.divide(Hrd_tot, Htot, out=np.full_like(Hrd_tot,0.5,dtype=float), where=(Htot>0))
                # MLP from eval (ok for A2A)
                mlp_h=np.maximum(1.0, reps["mlp_hit"].to_numpy(float))
                mlp_m=np.maximum(1.0, reps["mlp_miss"].to_numpy(float))
            else:
                # use characterization fingerprints (predictive HM, no eval per-window counts)
                cdir=f"results/characterization_L3_{int(float(r.get('l3_mb',cap)))}/{b}"
                feats=pd.read_csv(os.path.join(cdir,"LLC.window_features.csv"),
                                  engine="python", on_bad_lines="skip")
                # align features to reps window_id order
                id2row=feats.set_index("window_id")
                wids = reps["window_id"].to_numpy(int)
                acc1k = np.array([id2row.loc[i,"acc_per_1kcyc"] if i in id2row.index else 0.0 for i in wids], float)
                mr    = np.array([id2row.loc[i,"miss_rate"     ] if i in id2row.index else 0.0 for i in wids], float)
                rf    = np.array([id2row.loc[i,"read_frac"     ] if i in id2row.index else 0.5 for i in wids], float)
                mlp_h = np.maximum(1.0, np.array([id2row.loc[i,"mlp_hit" ] if i in id2row.index else 1.0 for i in wids], float))
                mlp_m = np.maximum(1.0, np.array([id2row.loc[i,"mlp_miss"] if i in id2row.index else 1.0 for i in wids], float))
                # derive per-window totals from fingerprints
                H1k   = acc1k*(1.0-mr)
                Hrd_tot = rf*H1k*T1K
                Hwr_tot = (1.0-rf)*H1k*T1K
                Htot = Hrd_tot + Hwr_tot
                rd_share=np.divide(Hrd_tot, Htot, out=np.full_like(Hrd_tot,0.5,dtype=float), where=(Htot>0))
                M = (mr*acc1k*T1K)  # per-window miss installs
            HM_tot_w=frac*Htot
            HM_rd_w=np.minimum(HM_tot_w*rd_share, Hrd_tot)
            HM_wr_w=np.minimum(HM_tot_w*(1-rd_share), Hwr_tot)
            HS_rd_w=Hrd_tot-HM_rd_w; HS_wr_w=Hwr_tot-HM_wr_w

            d_rd=tMr-BASE; d_wr=tMw-BASE; d_mi=tDR-BASE
            S=(HM_rd_w*d_rd + HM_wr_w*d_wr)/mlp_h + (M*d_mi)/mlp_m
            stall=float(np.nansum(S))/max(float(np.nansum(win)),1.0)

            E_MISS=E_MISS_BASE + E_MISS_PER_NS*(tDR / f_ghz)
            E_hits=(HS_rd_w*E_RS + HS_wr_w*E_WS + HM_rd_w*E_RM + HM_wr_w*E_WM).sum()
            Tk=float(np.nansum(T1K)); L3=float(r.get("l3_mb",cap)); C_S=(1.0-pw)*L3; C_M=pw*L3
            E_leak=(P_LS*C_S + P_LM*C_M) * Tk * 1e-6 / f_ghz
            E_tot=E_leak + E_hits + M.sum()*E_MISS

            out.append({"bench":b,"tag":tag,"pi_way":pw,"pi_miss":pm,
                        "t_sram_hit":tS,"t_mram_rd":tMr,"t_mram_wr":tMw,
                        "stall_pct":stall,"E_total_J":E_tot})

        df_out=pd.DataFrame(out)
        os.makedirs(f"results/surrogate/profiles/{prof}/L3_{cap}", exist_ok=True)
        df_out.to_csv(f"results/surrogate/profiles/{prof}/L3_{cap}/physics_pred.csv", index=False)
        print(f"[devspot_evalmapped_physics] {prof} rows={len(df_out)}")

if __name__=="__main__":
    main()
