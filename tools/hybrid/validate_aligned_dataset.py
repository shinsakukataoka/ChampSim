import os, json, argparse, pandas as pd, numpy as np

def mid_fracs(df):
    m=0.5*(df.start_cycle+df.end_cycle); T=float(df.end_cycle.max()-df.start_cycle.min())
    return m/(T if T>0 else 1.0)

def load_rep_fracs(cdir):
    reps=json.load(open(os.path.join(cdir,"LLC.representatives.json")))["representatives"]
    feats=pd.read_csv(os.path.join(cdir,"LLC.window_features.csv"), engine="python", on_bad_lines="skip")
    return [dict(zip(feats.window_id, mid_fracs(feats))).get(w) for w in reps if w in feats.window_id.to_list()]

def pick_unique(csv, fr):
    df=pd.read_csv(csv, engine="python", on_bad_lines="skip")
    if df.empty or not fr: return df.iloc[0:0]
    f=mid_fracs(df); used=set(); idx=[]
    for x in fr:
        for j in np.argsort(np.abs(f-x)):
            j=int(j)
            if j not in used:
                used.add(j); idx.append(j); break
    return df.iloc[idx].copy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--base", type=float, default=16.0, help="stall anchor (default 16)")
    args=ap.parse_args()

    cap=args.cap; prof=args.profile; BASE=args.base
    E=pd.read_csv(f"results/eval/dataset_L3_{cap}MB.csv")
    rows=[]
    for _,r in E.iterrows():
        b=r["bench"]; tag=r["tag"]; L3=int(round(r.get("l3_mb",cap)))
        cdir=f"results/characterization_L3_{L3}/{b}"
        if not os.path.exists(os.path.join(cdir,"LLC.representatives.json")): continue
        fr=load_rep_fracs(cdir)
        run=os.path.join("results","eval",b,tag,"LLC.llc.win.csv")
        if not os.path.exists(run): continue
        reps=pick_unique(run, fr)
        if reps.empty: continue
        mlp_h=np.maximum(1.0, reps['mlp_hit' ].to_numpy(float))
        mlp_m=np.maximum(1.0, reps['mlp_miss'].to_numpy(float))
        tS=float(r["t_sram_hit"]); tMr=float(r["t_mram_rd"]); tMw=float(r["t_mram_wr"]); tDR=float(r.get("t_dram",200.0))
        d_rd=tMr-BASE; d_wr=tMw-BASE; d_mi=tDR-BASE
        win=float((reps.end_cycle-reps.start_cycle).sum())
        S=(reps['hit_mram_rd']*d_rd + reps['hit_mram_wr']*d_wr)/mlp_h + ((reps['miss_rd']+reps['miss_wr'])*d_mi)/mlp_m
        rows.append({"bench":b,"tag":tag,"stall_eval_aligned":float(S.sum())/max(win,1.0)})
    A=pd.DataFrame(rows)

    P=pd.read_csv(f"results/surrogate/profiles/{prof}/L3_{cap}/physics_pred.csv")
    JOIN=["bench","pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr","tag"]
    M=E.merge(A,on=["bench","tag"],how="inner").merge(P[JOIN+["stall_pct"]], on=JOIN, how="inner", suffixes=("_eval","_phys"))
    err=(M["stall_pct_phys"]-M["stall_eval_aligned"]).abs()*100
    print(f"[Aligned parity] cap={cap} profile={prof} rows={len(M)}  Stall MAE med={err.median():.2f}%  p90={err.quantile(0.9):.2f}%")
    print("Worst 8:")
    print(pd.DataFrame({"bench":M["bench"],"stall_mae%":err}).sort_values("stall_mae%",ascending=False).head(8).to_string(index=False))

if __name__=="__main__":
    main()
