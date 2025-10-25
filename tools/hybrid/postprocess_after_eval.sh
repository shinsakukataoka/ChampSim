#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/skataoka26/ChampSim"
cd "$ROOT"

echo "[1/3] Aggregate eval -> dataset.csv"
python3 tools/hybrid/aggregate_eval.py

echo "[2/3] Split dataset per capacity"
python3 - <<'PY'
import pandas as pd, os
df=pd.read_csv("results/eval/dataset.csv")
for mb,sub in df.groupby(df['l3_mb'].round().astype(int)):
    out=f"results/eval/dataset_L3_{mb}MB.csv"
    os.makedirs("results/eval", exist_ok=True)
    sub.to_csv(out,index=False)
    print("  wrote", out, len(sub), "rows")
PY

echo "[3/3] Per-capacity: fingerprints/HM -> train ML -> ML preds -> PHYSICS preds (ROI+capacity correct)"
for CAP in 2 32 128; do
  echo "===== L3_${CAP}MB ====="

  # Point builders at the matching characterization (absolute path)
  rm -f results/characterization
  ln -sfn "$ROOT/results/characterization_L3_${CAP}" results/characterization

  # Build fingerprints + HM curves
  python3 tools/hybrid/build_benchmark_fingerprints.py
  python3 tools/hybrid/build_empirical_hm_curves.py

  # Stash builder outputs per capacity
  mkdir -p "results/surrogate/L3_${CAP}"
  mv -f results/surrogate/benchmark_fingerprints.csv "results/surrogate/L3_${CAP}/"
  mv -f results/surrogate/hm_curves.joblib           "results/surrogate/L3_${CAP}/"

  # Use per-cap dataset+fingerprints -> training table -> train models
  \cp -f "results/eval/dataset_L3_${CAP}MB.csv"                   results/eval/dataset.csv
  \cp -f "results/surrogate/L3_${CAP}/benchmark_fingerprints.csv"   results/surrogate/benchmark_fingerprints.csv
  python3 tools/hybrid/build_training_table.py
  python3 tools/hybrid/train_surrogate_perbench.py

  # Archive training artifacts
  \cp -f results/surrogate/training_table.csv "results/surrogate/L3_${CAP}/training_table.csv"
  \cp -f results/surrogate/surrogate_*        "results/surrogate/L3_${CAP}/" 2>/dev/null || true
  \cp -f results/surrogate/*_hgb_*.joblib     "results/surrogate/L3_${CAP}/" 2>/dev/null || true

  # ---------- ML predictions ----------
  python3 - "$CAP" <<'PY'
import sys,pandas as pd,joblib
cap=int(sys.argv[1])
tt=pd.read_csv(f"results/surrogate/L3_{cap}/training_table.csv")
stall=joblib.load(f"results/surrogate/L3_{cap}/surrogate_stallpct_hgb_global.joblib")
ener =joblib.load(f"results/surrogate/L3_{cap}/surrogate_energy_hgb_global.joblib")
X=tt[stall["feat_cols"]].values
tt["stall_pct_pred_ml"]=stall["model"].predict(X)
tt["E_total_pred_ml"]  =ener["model"].predict(X)
cols=["bench","tag","pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr","stall_pct_pred_ml","E_total_pred_ml"]
tt[cols].to_csv(f"results/surrogate/L3_{cap}/ml_pred.csv",index=False)
print(f"  [ML] wrote results/surrogate/L3_{cap}/ml_pred.csv  rows={len(tt)}")
PY

  # ---------- PHYSICS predictions (capacity-aware + rep-window scaling, no calibration) ----------
  python3 - "$CAP" <<'PY'
import os, json, numpy as np, pandas as pd, joblib, sys

cap = int(sys.argv[1])
ds = f"results/eval/dataset_L3_{cap}MB.csv"
fp = f"results/surrogate/L3_{cap}/benchmark_fingerprints.csv"
hm = f"results/surrogate/L3_{cap}/hm_curves.joblib"

# energy constants (same as eval_design.py)
E_RD_SRAM=0.3e-9; E_WR_SRAM=0.4e-9
E_RD_MRAM=0.6e-9; E_WR_MRAM=0.8e-9
E_MISS   =3.0e-9
P_LEAK_S =5.0e-3; P_LEAK_M=0.5e-3
T_1KSEC  =1e-6

def mid_fracs(df):
    mids = 0.5*(df['start_cycle'].to_numpy() + df['end_cycle'].to_numpy())
    tot  = float(df['end_cycle'].max() - df['start_cycle'].min())
    return mids/(tot if tot>0 else 1.0)

def load_rep_fracs(char_dir):
    reps = json.load(open(os.path.join(char_dir,"LLC.representatives.json")))["representatives"]
    feats= pd.read_csv(os.path.join(char_dir,"LLC.window_features.csv"), engine="python", on_bad_lines="skip")
    id2f = dict(zip(feats['window_id'].to_numpy(), mid_fracs(feats)))
    return [id2f[w] for w in reps if w in id2f]

def pick_unique(run_csv, rep_fracs):
    df = pd.read_csv(run_csv, engine="python", on_bad_lines="skip")
    if df.empty or not rep_fracs: return df.iloc[0:0].copy()
    fr = mid_fracs(df); used=set(); idx=[]
    for f in rep_fracs:
        for j in np.argsort(np.abs(fr-f)):
            if int(j) not in used:
                used.add(int(j)); idx.append(int(j)); break
    return df.iloc[idx].copy()

def isotonic_eval(curve, x):
    xs=np.asarray(curve["x"], float); ys=np.asarray(curve["y"], float)
    return float(np.interp(x, xs, ys)) if xs.size>0 else 0.5

D  = pd.read_csv(ds)
FP = pd.read_csv(fp).set_index("bench")
curves = joblib.load(hm)

rows=[]
for _,r in D.iterrows():
    bench=r["bench"]; tag=r["tag"]; pw=float(r["pi_way"]); pm=float(r["pi_miss"])
    tS=float(r["t_sram_hit"]); tMr=float(r["t_mram_rd"]); tMw=float(r["t_mram_wr"]); tDR=float(r.get("t_dram",200.0))
    L3=float(r.get("l3_mb", cap))
    if bench not in FP.index or bench not in curves: continue

    tb=curves[bench]; key=f"{pm:.4f}" if f"{pm:.4f}" in tb else ("__pooled__" if "__pooled__" in tb else None)
    if key is None: continue
    frac_mram = isotonic_eval(tb[key], pw)

    f=FP.loc[bench]
    A_k=float(f["char_acc_per_1kcyc"]); mr=float(f["char_miss_rate"]); rf=float(f["char_read_frac"])
    M1k=mr*A_k; H1k=A_k-M1k; HM=frac_mram*H1k; HS=H1k-HM
    HM_rd=rf*HM; HM_wr=(1-rf)*HM; HS_rd=rf*HS; HS_wr=(1-rf)*HS

    E_dyn_1k = HS_rd*E_RD_SRAM + HS_wr*E_WR_SRAM + HM_rd*E_RD_MRAM + HM_wr*E_WR_MRAM + M1k*E_MISS
    C_S=(1.0-pw)*L3; C_M=pw*L3
    E_leak_1k=(P_LEAK_S*C_S + P_LEAK_M*C_M)*T_1KSEC

    char_dir=f"results/characterization_L3_{cap}/{bench}"
    rep_fracs=load_rep_fracs(char_dir)
    run_csv=f"results/eval/{bench}/{tag}/LLC.llc.win.csv"
    if not os.path.exists(run_csv): continue
    reps=pick_unique(run_csv, rep_fracs)
    rep_kcyc=float(((reps['end_cycle'] - reps['start_cycle']).sum())/1000.0)

    d_rd=tMr-tS; d_wr=tMw-tS; d_mi=tDR-tS
    mlp_hit=max(1.0,float(f["char_mlp_hit"])); mlp_miss=max(1.0,float(f["char_mlp_miss"]))
    stall_k=(HM_rd*d_rd + HM_wr*d_wr)/mlp_hit + (M1k*d_mi)/mlp_miss
    stall_pct=stall_k/1000.0

    rows.append({
      "bench":bench,"tag":tag,"pi_way":pw,"pi_miss":pm,
      "t_sram_hit":tS,"t_mram_rd":tMr,"t_mram_wr":tMw,
      "stall_pct":stall_pct, "E_total_J": (E_dyn_1k+E_leak_1k)*rep_kcyc
    })

pred=pd.DataFrame(rows)
os.makedirs(f"results/surrogate/L3_{cap}", exist_ok=True)
pred.to_csv(f"results/surrogate/L3_{cap}/physics_pred.csv", index=False)
print(f"  [PHY] wrote results/surrogate/L3_{cap}/physics_pred.csv rows={len(pred)}")
PY

done
echo "[done] Post-process completed."

