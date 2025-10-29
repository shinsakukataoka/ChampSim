#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

cd /home/skataoka26/ChampSim
CAP=2
DEVSPOT_PROFILES="val_tS16_MR28_MW60 val_tS16_MR28_MW100 val_tS16_MR28_MW120"

echo "--- Check 1: Frozen HM ---"
# 1) Frozen HM really used (bitwise compare)
cmp -s backups/hm_curves_L3_${CAP}_SEEDS.joblib results/surrogate/L3_${CAP}/hm_curves.joblib \
  && echo "[HM] OK: frozen HM in use" \
  || echo "[HM] FAIL: current hm_curves differs from frozen (restore before plotting)"

echo "--- Check 2: Physics Files ---"
# 2) Physics files exist & look like devspot
for P in $DEVSPOT_PROFILES; do
  F="results/surrogate/profiles/${P}/L3_${CAP}/physics_pred.csv"
  if [ ! -f "$F" ]; then echo "[PHY/$P] FAIL: missing $F"; continue; fi
  # Pass "$F" as an argument to python3
  python3 - "$F" <<PY
import os,pandas as pd,sys
F=sys.argv[1]
df=pd.read_csv(F)
ok_tags = df["tag"].astype(str).str.startswith("devspot_").all()
print(f"[PHY/{os.path.basename(os.path.dirname(F))}] rows={len(df)} benches={df['bench'].nunique()} devspot_tags={ok_tags}")
PY
done

echo "--- Check 3: A2A Window Mapping ---"
# 3) A2A window mapping: recompute stall from eval CSV mapped windows (device t*), compare to physics
python3 - << 'PY'
import os, json, numpy as np, pandas as pd

cap=2
profiles=["val_tS16_MR28_MW60","val_tS16_MR28_MW100","val_tS16_MR28_MW120"]
JOIN=["bench","pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr","tag"]

def load_dev(profile,cap):
    base=f"devices/{profile}/base.json"; over=f"devices/{profile}/L3_{cap}.json"
    dj=json.load(open(base)) if os.path.exists(base) else {}
    if os.path.exists(over):
        ov=json.load(open(over))
        for k,v in ov.items():
            if isinstance(v,dict) and k in dj and isinstance(dj[k],dict): dj[k].update(v)
            else: dj[k]=v
    return dj

def load_eval_mapped_ids(run_dir):
    try:
        m=json.load(open(os.path.join(run_dir,"metrics.json")))
        return [int(x) for x in m.get("rep_windows_mapped",[])]
    except Exception: return []

for prof in profiles:
    phy=f"results/surrogate/profiles/{prof}/L3_{cap}/physics_pred.csv"
    if not os.path.exists(phy):
        print(f"[A2A/{prof}] SKIP: {phy} missing"); continue
    try:
        P=pd.read_csv(phy)
        E=pd.read_csv(f"results/eval/dataset_L3_{cap}MB.csv")
    except Exception as e:
        print(f"[A2A/{prof}] FAIL: could not read CSV. Error: {e}"); continue
        
    dj=load_dev(prof, cap); LAT=dj.get("latency",{}); BASE=16.0
    # join to filter only tags we can re-evaluate
    M=P.merge(E, on=JOIN, how="inner", suffixes=("_phys",""))
    diffs=[]
    for r in M.itertuples(index=False):
        run=os.path.join("results","eval",r.bench,r.tag,"LLC.llc.win.csv")
        ids=load_eval_mapped_ids(os.path.join("results","eval",r.bench,r.tag))
        if not (os.path.exists(run) and ids): continue
        try:
            df=pd.read_csv(run, engine="python", on_bad_lines="skip")
        except Exception:
            continue # Skip bad/empty files
        if "window_id" in df.columns:
            df["window_id"]=pd.to_numeric(df["window_id"],errors="coerce").fillna(-1).astype(int)
        reps=df[df["window_id"].isin(ids)].copy()
        if reps.empty: continue
        # device t*
        tS=float(LAT.get("t_sram_hit", r.t_sram_hit))
        tMr=float(LAT.get("t_mram_rd",  r.t_mram_rd))
        tMw=float(LAT.get("t_mram_wr",  r.t_mram_wr))
        tDR=float(LAT.get("t_miss_cycles", getattr(r,"t_dram",200.0)))
        d_rd=tMr-BASE; d_wr=tMw-BASE; d_mi=tDR-BASE
        mlp_h=np.maximum(1.0, reps["mlp_hit"].to_numpy(float))
        mlp_m=np.maximum(1.0, reps["mlp_miss"].to_numpy(float))
        win=float((reps["end_cycle"]-reps["start_cycle"]).sum())
        S=(reps["hit_mram_rd"]*d_rd + reps["hit_mram_wr"]*d_wr)/mlp_h + ((reps["miss_rd"]+reps["miss_wr"])*d_mi)/mlp_m
        stall_eval=float(S.sum())/max(win,1.0)
        diffs.append(abs(stall_eval - r.stall_pct_phys))
    diffs=np.array(diffs, float)
    if diffs.size==0:
        print(f"[A2A/{prof}] FAIL: no rows rechecked (M={len(M)}, P={len(P)}, E={len(E)})")
    else:
        print(f"[A2A/{prof}] OK: n={diffs.size}  max_abs_diff={diffs.max():.3e}  median_diff={np.median(diffs):.3e}")
PY

echo "--- Check 4: Validation CSVs ---"
# 4) Validation CSVs join to physics/eval; print medians
python3 - << 'PY'
import os, pandas as pd
cap=2
def summarize(csv):
    try:
        d=pd.read_csv(csv)
    except Exception as e:
        return f"ERROR: could not read {csv}. {e}"
    s=(d["MAE_stall_frac"]*100)
    e=(d["MAPE_energy"]*100)
    return f"rows={len(d)}  stall_med={s.median():.2f}%  stall_p90={s.quantile(0.9):.2f}%  energy_med={e.median():.2f}%"

print("[Dataset-latency] ", summarize(f"results/figs/L3_{cap}/validation/validation_errors_L3_{cap}_nvsim_n10_readedp_reweighted_nvsim_n10_readedp.csv"))
for P in ["val_tS16_MR28_MW60","val_tS16_MR28_MW100","val_tS16_MR28_MW120"]:
    f=f"results/figs/L3_{cap}/validation/validation_errors_L3_{cap}_{P}_reweighted_{P}.csv"
    if os.path.exists(f):
        print(f"[Devspot/{P}] ", summarize(f))
    else:
        print(f"[Devspot/{P}] CSV missing")
PY

echo "--- All checks complete ---"
