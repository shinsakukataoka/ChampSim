#!/usr/bin/env python3
import os, json, argparse, numpy as np, pandas as pd, joblib

JOIN = ["bench","pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr","tag"]

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_dataset(cap):
    p = f"results/eval/dataset_L3_{cap}MB.csv"
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing {p}. Run postprocess_after_eval.sh --caps {cap}.")
    return pd.read_csv(p).dropna(subset=["stall_pct","E_total_J"])

def load_fp(cap):
    p = f"results/surrogate/L3_{cap}/benchmark_fingerprints.csv"
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing {p}. Run postprocess_after_eval.sh --caps {cap}.")
    return pd.read_csv(p).set_index("bench")

def pick_canonical_rows(ds, target_pw=0.50, target_pm=0.50, only=None):
    if only: ds = ds[ds["bench"].isin(only)]
    rows=[]
    for b, g in ds.groupby("bench"):
        d=(g["pi_way"]-target_pw)**2 + (g["pi_miss"]-target_pm)**2 \
          + 1e-4*((g["t_sram_hit"]-16)**2 + (g["t_mram_rd"]-28)**2 + (g["t_mram_wr"]-60)**2)
        rows.append(g.iloc[int(np.argmin(d.values))])
    return pd.DataFrame(rows).sort_values("bench").reset_index(drop=True)

# ---- mapped reps → time & contributions ----
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
            if int(j) not in used: used.add(int(j)); idx.append(int(j)); break
    return df.iloc[idx].copy()

def stall_components_from_run(cap, row):
    """Return (stall_frac, miss_share, mram_share). Uses the eval row’s mapped reps & latencies."""
    b=row["bench"]; L3=float(row.get("l3_mb", cap))
    cdir=f"results/characterization_L3_{int(L3)}/{b}"
    fr=load_rep_fracs(cdir)
    run_csv=f"results/eval/{b}/{row['tag']}/LLC.llc.win.csv"
    reps=pick_unique(run_csv, fr)
    if reps.empty: return np.nan, np.nan, np.nan

    # counts
    Hm_rd=reps['hit_mram_rd'].to_numpy(); Hm_wr=reps['hit_mram_wr'].to_numpy()
    M = (reps['miss_rd']+reps['miss_wr']).to_numpy()
    mlp_hit=np.maximum(1.0, reps['mlp_hit'].to_numpy()); mlp_miss=np.maximum(1.0, reps['mlp_miss'].to_numpy())

    # latencies from dataset row
    tS=float(row["t_sram_hit"]); tMr=float(row["t_mram_rd"]); tMw=float(row["t_mram_wr"]); tDR=float(row.get("t_dram",200.0))
    d_rd=tMr-tS; d_wr=tMw-tS; d_mi=tDR-tS

    # per window stall contributions
    S_mram = (Hm_rd*d_rd + Hm_wr*d_wr)/mlp_hit
    S_miss = (M*d_mi)/mlp_miss
    S_tot = S_mram + S_miss
    stall_k = float(np.nansum(S_tot))                                  # stall cycles per 1k cycles aggregate
    win_cycles = float(((reps.end_cycle - reps.start_cycle).sum()))     # cycles (not k)
    stall_frac = stall_k / max(win_cycles/1000.0, 1.0)

    mram_share = float(np.nansum(S_mram)) / max(float(np.nansum(S_tot)), 1e-9)
    miss_share = 1.0 - mram_share
    return stall_frac, miss_share, mram_share

def slope(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if np.unique(x).size<2: return np.nan
    return float(np.polyfit(x, y, 1)[0])

def make_table(cap, benches=None, target_pw=0.50, target_pm=0.50, outcsv=""):
    ds = load_dataset(cap)
    fp = load_fp(cap)
    rows = pick_canonical_rows(ds, target_pw, target_pm, benches)
    benches = rows["bench"].tolist()

    # Sensitivity: restrict to pm=target_pm and pw in {0.25, 0.50, 0.75}
    sub = ds[np.isclose(ds["pi_miss"], target_pm)]
    sub3 = sub[sub["pi_way"].isin([0.25,0.50,0.75])]

    out=[]
    for r in rows.itertuples(index=False):
        b=r.bench
        # fingerprint stats
        fpr = fp.loc[b] if b in fp.index else None
        miss_rate = float(fpr.char_miss_rate)*100.0 if fpr is not None else np.nan
        mpkc = float(fpr.char_mpkc) if fpr is not None else np.nan
        mlp_miss = float(fpr.char_mlp_miss) if fpr is not None else np.nan
        mlp_hit  = float(fpr.char_mlp_hit ) if fpr is not None else np.nan
        stall_char_pct = float(fpr.char_stall_per_1kcyc)/10.0 if fpr is not None else np.nan  # /1000*100

        # eval stall at canonical
        stall_eval_pct = float(r.stall_pct)*100.0

        # sensitivity slopes (Δstall / Δlatency) @ pw=0.25/.50/.75 (median)
        g = sub3[sub3["bench"]==b]
        s_mr = g.groupby("pi_way", group_keys=False).apply(lambda gg: slope(gg["t_mram_rd"], gg["stall_pct"])).median()
        s_mw = g.groupby("pi_way", group_keys=False).apply(lambda gg: slope(gg["t_mram_wr"], gg["stall_pct"])).median()
        # convert to % per 10 cycles for readability
        slope_mr_pct10 = float(s_mr)*100.0*10.0 if pd.notna(s_mr) else np.nan
        slope_mw_pct10 = float(s_mw)*100.0*10.0 if pd.notna(s_mw) else np.nan

        # breakdown at the chosen row
        stall_frac, miss_share, mram_share = stall_components_from_run(cap, r._asdict())

        # simple memory-bound score (heuristic ranking aid)
        mbs = stall_eval_pct * (1.0 + (mpkc if not np.isnan(mpkc) else 0.0)/10.0)

        out.append({
            "bench": b,
            "miss_rate_%": miss_rate,
            "MPKC": mpkc,
            "MLP_miss": mlp_miss,
            "MLP_hit": mlp_hit,
            "stall_eval_%": stall_eval_pct,
            "stall_char_%": stall_char_pct,
            "stall_miss_share_%": miss_share*100.0,
            "stall_mram_share_%": mram_share*100.0,
            "slope_mr_(%/10cy)": slope_mr_pct10,
            "slope_mw_(%/10cy)": slope_mw_pct10,
            "MemoryBoundScore": mbs
        })

    tbl = pd.DataFrame(out).sort_values("MemoryBoundScore", ascending=False)
    if outcsv:
        ensure_dir(os.path.dirname(outcsv))
        tbl.to_csv(outcsv, index=False)
        print(f"[wrote] {outcsv}")
    return tbl

def main():
    ap = argparse.ArgumentParser(description="Quantify how memory-bound each bench is")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--benches", default="", help="comma list; empty = all")
    ap.add_argument("--target-pw", type=float, default=0.50)
    ap.add_argument("--target-pm", type=float, default=0.50)
    ap.add_argument("--outcsv", default="")
    args = ap.parse_args()

    only = [b for b in args.benches.split(",") if b] or None
    outcsv = args.outcsv or f"results/figs/L3_{args.cap}/validation/memory_bound_L3_{args.cap}.csv"
    tbl = make_table(args.cap, only, args.target_pw, args.target_pm, outcsv)
    # Pretty print small view
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(tbl.to_string(index=False))

if __name__ == "__main__":
    main()

