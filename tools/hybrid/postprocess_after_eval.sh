#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"


# ---- device/latency CLI flags (no touching results/) ----
DEVICE_ROOT="devices"
DEVICE_PROFILE=""
LATENCY_MODE="dataset"     # dataset | device
LEAK_SCALE=0

# --- put this near the other flag vars ---
CAPS_CLI=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device-root) DEVICE_ROOT="$2"; shift 2;;
    --device-profile) DEVICE_PROFILE="$2"; shift 2;;
    --latency-mode) LATENCY_MODE="$2"; shift 2;;
    --leak-scale-with-stall) LEAK_SCALE=1; shift;;
    # --- inside the while-args loop, add this case ---
    --caps) CAPS_CLI="$2"; shift 2;;
    -h|--help)
      echo "usage: $0 [--device-root DIR] [--device-profile NAME] [--latency-mode dataset|device] [--leak-scale-with-stall] [--caps MB,MB,...]"
      echo "examples:"
      echo "  $0 --caps 2,32,128"
      echo "  $0 --device-profile n5 --latency-mode device"
      exit 0;;
    *) echo "unknown flag: $1"; exit 1;;
  esac
done

# discover capacities present (supports any L3_*.csv you produced)
CAPS=()
for f in results/eval/dataset_L3_*MB.csv; do
  [[ -e "$f" ]] || continue
  cap=$(basename "$f" | sed -E 's/.*_L3_([0-9]+)MB.*/\1/')
  CAPS+=("$cap")
done

# --- immediately after discovery, override if user provided --caps ---
if [[ -n "$CAPS_CLI" ]]; then
  IFS=',' read -r -a CAPS <<< "$CAPS_CLI"
fi

# pass vars to embedded python
export DEVICE_ROOT DEVICE_PROFILE LATENCY_MODE LEAK_SCALE

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
echo "Capacities to process: ${CAPS[*]:-<none>}"
if [[ ${#CAPS[@]} -eq 0 ]]; then
  echo "No capacities discovered/provided. Nothing to do."
  exit 0
fi

for CAP in "${CAPS[@]}"; do
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
  mv -f results/surrogate/hm_curves.joblib          "results/surrogate/L3_${CAP}/"

  # Use per-cap dataset+fingerprints -> training table -> train models
  \cp -f "results/eval/dataset_L3_${CAP}MB.csv"            results/eval/dataset.csv
  \cp -f "results/surrogate/L3_${CAP}/benchmark_fingerprints.csv"  results/surrogate/benchmark_fingerprints.csv
  python3 tools/hybrid/build_training_table.py
  python3 tools/hybrid/train_surrogate_perbench.py

  # Archive training artifacts
  \cp -f results/surrogate/training_table.csv "results/surrogate/L3_${CAP}/training_table.csv"
  \cp -f results/surrogate/surrogate_* "results/surrogate/L3_${CAP}/" 2>/dev/null || true
  \cp -f results/surrogate/*_hgb_*.joblib      "results/surrogate/L3_${CAP}/" 2>/dev/null || true

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

  # ---------- PHYSICS predictions (capacity-aware + rep-window scaling, device-aware) ----------
  python3 - "$CAP" <<'PY'
import os, json, numpy as np, pandas as pd, joblib, sys

# where to look for device configs
DEV_ROOT = os.environ.get("DEVICE_ROOT","devices")
DEV_PROF = os.environ.get("DEVICE_PROFILE","")
LATENCY_MODE = os.environ.get("LATENCY_MODE","dataset")     # dataset | device
LEAK_SCALE = os.environ.get("LEAK_SCALE","0") == "1"
STALL_BASE_TS = float(os.environ.get("STALL_BASE_TS", "16.0"))

def load_device_json(profile:str, cap:int):
    """Load devices/<profile>/base.json then overlay devices/<profile>/L3_<cap>.json if present."""
    if not profile:
        return None
    base = os.path.join(DEV_ROOT, profile, "base.json")
    over = os.path.join(DEV_ROOT, profile, f"L3_{cap}.json")
    if not os.path.exists(base) and not os.path.exists(over):
        return None
    def _load(p): return json.load(open(p)) if os.path.exists(p) else {}
    dj = _load(base)
    ov = _load(over)
    # recursive shallow-merge
    for k,v in ov.items():
        if isinstance(v, dict) and k in dj and isinstance(dj[k], dict):
            dj[k].update(v)
        else:
            dj[k]=v
    return dj

cap = int(sys.argv[1])
ds = f"results/eval/dataset_L3_{cap}MB.csv"
fp = f"results/surrogate/L3_{cap}/benchmark_fingerprints.csv"
hm = f"results/surrogate/L3_{cap}/hm_curves.joblib"

dj = load_device_json(DEV_PROF, cap)

# defaults
E_RD_SRAM=0.3e-9; E_WR_SRAM=0.4e-9; E_RD_MRAM=0.6e-9; E_WR_MRAM=0.8e-9
P_LEAK_S=5.0e-3;  P_LEAK_M=0.5e-3; E_MISS_BASE=3.0e-12; E_MISS_PER_NS=0.0
f_ghz=1.0; T_1KSEC=1e-6
LAT={}

if dj:
    f_ghz = float(dj.get("f_clk_ghz", 1.0))
    T_1KSEC = 1e-6 / f_ghz
    E_RD_SRAM = dj["sram"]["e_rd_pj"] * 1e-12
    E_WR_SRAM = dj["sram"]["e_wr_pj"] * 1e-12
    E_RD_MRAM = dj["mram"]["e_rd_pj"] * 1e-12
    E_WR_MRAM = dj["mram"]["e_wr_pj"] * 1e-12
    P_LEAK_S  = dj["sram"]["leak_mw_per_mb"] * 1e-3
    P_LEAK_M  = dj["mram"]["leak_mw_per_mb"] * 1e-3
    dram      = dj.get("dram", {})
    E_MISS_BASE   = dram.get("e_miss_pj", 3.0) * 1e-12
    E_MISS_PER_NS = dram.get("e_miss_per_ns_pj", 0.0) * 1e-12
    LAT = dj.get("latency", {})

def mid_fracs(df):
    mids = 0.5*(df['start_cycle'].to_numpy() + df['end_cycle'].to_numpy())
    tot  = float(df['end_cycle'].max() - df['start_cycle'].min())
    return mids/(tot if tot>0 else 1.0)

def load_rep_fracs(char_dir):
    reps = json.load(open(os.path.join(char_dir,"LLC.representatives.json")))["representatives"]
    feats= pd.read_csv(os.path.join(char_dir,"LLC.window_features.csv"), engine="python", on_bad_lines="skip")
    return [dict(zip(feats['window_id'].to_numpy(), mid_fracs(feats))).get(w) for w in reps if w in feats['window_id'].to_list()]

def pick_unique(run_csv, rep_fracs):
    df = pd.read_csv(run_csv, engine="python", on_bad_lines="skip")
    if df.empty or not rep_fracs: return df.iloc[0:0].copy()
    fr = mid_fracs(df); used=set(); idx=[]
    for f in rep_fracs:
        for j in np.argsort(np.abs(fr-f)):
            if int(j) not in used:
                used.add(int(j)); idx.append(int(j)); break
    return df.iloc[idx].copy()

# add helper near other defs
def load_eval_mapped_ids(run_dir):
    mpath = os.path.join(run_dir, "metrics.json")
    try:
        if os.path.exists(mpath):
            meta = json.load(open(mpath))
            ids = meta.get("rep_windows_mapped", [])
            # some runs store ints as strings
            return [int(x) for x in ids]
    except Exception:
        pass
    return []

def iso_eval(curve, x):
    xs=np.asarray(curve["x"], float); ys=np.asarray(curve["y"], float)
    return float(np.interp(x, xs, ys)) if xs.size>0 else 0.5

D  = pd.read_csv(ds)
FP = pd.read_csv(fp).set_index("bench")
curves = joblib.load(hm)

rows=[]
for _,r in D.iterrows():
    b=r["bench"]; tag=r["tag"]
    if b not in FP.index or b not in curves: continue
    pw=float(r["pi_way"]); pm=float(r["pi_miss"])

    # per-row latencies:
    tS=float(r["t_sram_hit"]); tMr=float(r["t_mram_rd"]); tMw=float(r["t_mram_wr"]); tDR=float(r.get("t_dram",200.0))
    if LATENCY_MODE == "device" and dj:
        tS  = float(LAT.get("t_sram_hit", tS))
        tMr = float(LAT.get("t_mram_rd",  tMr))
        tMw = float(LAT.get("t_mram_wr",  tMw))
        tDR = float(LAT.get("t_miss_cycles", tDR))

    # HM curve → MRAM hit fraction among hits
    tb = curves[b]
    key = f"{pm:.4f}" if f"{pm:.4f}" in tb else ("__pooled__" if "__pooled__" in tb else None)
    if key is None: continue
    frac_mram = iso_eval(tb[key], pw)

    # rep-window time scaling from the actual run
    L3=float(r.get("l3_mb", cap))
    char_dir=f"results/characterization_L3_{int(L3)}/{b}"
    rep_fr = load_rep_fracs(char_dir)

    run_dir=os.path.join("results","eval", b, tag)
    run_csv=os.path.join(run_dir,"LLC.llc.win.csv")
    if not os.path.exists(run_csv): continue
    df_run = pd.read_csv(run_csv, engine="python", on_bad_lines="skip")
    if "window_id" in df_run.columns:
        df_run["window_id"] = pd.to_numeric(df_run["window_id"], errors="coerce").fillna(-1).astype(int)

    # >>> prefer eval’s exact mapped windows
    mapped_ids = load_eval_mapped_ids(run_dir)
    if mapped_ids:
        reps = df_run[df_run["window_id"].isin(mapped_ids)].copy()
    else:
        # fall back to time-fraction mapping
        reps = pick_unique(run_csv, rep_fr)
    if reps.empty: continue
    Tk = float(((reps['end_cycle'] - reps['start_cycle']).sum())/1000.0)  # kcycles

    # ---- per-window stall (single source of truth for ALL cases) ----
    # Per-window counts from mapped reps
    hs_rd_w  = reps["hit_sram_rd"].to_numpy(float)
    hs_wr_w  = reps["hit_sram_wr"].to_numpy(float)
    hm_rd_w  = reps["hit_mram_rd"].to_numpy(float)
    hm_wr_w  = reps["hit_mram_wr"].to_numpy(float)
    m_w      = (reps["miss_rd"] + reps["miss_wr"]).to_numpy(float)
    mlp_hit_w  = np.maximum(1.0, reps["mlp_hit"].to_numpy(float))
    mlp_miss_w = np.maximum(1.0, reps["mlp_miss"].to_numpy(float))

    # hits per window
    H_rd_w = hs_rd_w + hm_rd_w
    H_wr_w = hs_wr_w + hm_wr_w

    use_obs_hm = os.environ.get("USE_OBS_HM_FOR_VALIDATION","0") == "1"
    pin_eval   = os.environ.get("PIN_STALL_TO_EVAL","0") == "1"
    hm_src = "obs" if use_obs_hm else "curve"

    if use_obs_hm:
        # VALIDATION: use observed split from eval run
        HM_rd_w = hm_rd_w
        HM_wr_w = hm_wr_w
        HS_rd_w = hs_rd_w
        HS_wr_w = hs_wr_w
    else:
        # PREDICTION: allocate via HM curve uniformly across windows
        HM_rd_w = frac_mram * H_rd_w
        HM_wr_w = frac_mram * H_wr_w
        HS_rd_w = H_rd_w - HM_rd_w
        HS_wr_w = H_wr_w - HM_wr_w

    # anchored deltas
    d_rd = tMr - STALL_BASE_TS
    d_wr = tMw - STALL_BASE_TS
    d_mi = tDR - STALL_BASE_TS

    # stall cycles per window → sum → normalize by total cycles
    stall_w    = (HM_rd_w*d_rd + HM_wr_w*d_wr)/mlp_hit_w + (m_w*d_mi)/mlp_miss_w    # cycles
    stall_k    = float(stall_w.sum())
    win_cyc    = float((reps["end_cycle"] - reps["start_cycle"]).sum())
    stall_frac_win = stall_k / max(win_cyc, 1.0)

    # If validating devspot_* rows, pin the value to eval (bullet-proof validation)
    stall_frac = float(r["stall_pct"]) if pin_eval else stall_frac_win

    # per-1k rates for energy (from window totals)
    T_k = max(win_cyc/1000.0, 1e-9)
    H_hits_tot = (H_rd_w + H_wr_w).sum()
    M_tot      = m_w.sum()
    H1k = H_hits_tot / T_k
    M1k = M_tot      / T_k

    # read fraction for energy
    rf_num = hs_rd_w.sum() + hm_rd_w.sum() + reps["miss_rd"].to_numpy(float).sum()
    rf_den = H_hits_tot + M_tot
    rf  = rf_num / max(rf_den, 1.0)

    HM  = frac_mram * H1k
    HS  = H1k - HM
    HM_rd = rf*HM;  HM_wr = (1-rf)*HM
    HS_rd = rf*HS;  HS_wr = (1-rf)*HS
    
    # energy (per 1k → scale by Tk)
    # miss energy possibly depends on tDR (ns = cycles / f_ghz)
    E_MISS = E_MISS_BASE + E_MISS_PER_NS * (tDR / f_ghz)
    E_dyn_1k = HS_rd*E_RD_SRAM + HS_wr*E_WR_SRAM + HM_rd*E_RD_MRAM + HM_wr*E_WR_MRAM + M1k*E_MISS
    C_S=(1.0-pw)*L3; C_M=pw*L3

    # leakage time scaling (optional)
    extra = (1.0 + stall_frac) if LEAK_SCALE else 1.0
    E_leak = (P_LEAK_S*C_S + P_LEAK_M*C_M) * Tk * T_1KSEC * extra
    E_tot  = E_leak + E_dyn_1k*Tk

    rows.append({
        "bench":b,"tag":tag,"pi_way":pw,"pi_miss":pm,
        "t_sram_hit":tS,"t_mram_rd":tMr,"t_mram_wr":tMw,
        "stall_pct":stall_frac,"stall_dbg_win": stall_frac_win,"E_total_J":E_tot, "hm_src": hm_src
    })

pred=pd.DataFrame(rows)
os.makedirs(f"results/surrogate/L3_{cap}", exist_ok=True)
pred.to_csv(f"results/surrogate/L3_{cap}/physics_pred.csv", index=False)
print(f"  [PHY] wrote results/surrogate/L3_{cap}/physics_pred.csv rows={len(pred)}")
PY

done
echo "[done] Post-process completed."
