# tools/quick/hybrid_wholetrace.py
#!/usr/bin/env python3
"""
Hybrid LLC whole-trace quick printer (no file outputs)

- Reads one JSON config (device physics + knobs)
- Uses existing dataset/HM artifacts:
    results/eval/dataset_L3_<cap>MB.csv
    results/surrogate/L3_<cap>/benchmark_fingerprints.csv
    results/surrogate/L3_<cap>/hm_curves.joblib
- Fine-grained policy:
    * presets: mid | optE | wstE | optS | wstS
    * bias:           HM_rd = r * HM_tot
    * weighted:       minimize λ·E_dyn_1k + (1−λ)·stall_frac
    * bias_grid / weighted_search: try grids, pick best
    * constraints: hm_rd_min/max_frac, hm_wr_min/max_frac
    * per-bench rules + overrides
- Optional: time-coupled leakage (JSON flag or CLI)
- NEW:
    * --compare-sram: also print an SRAM-only baseline (π_way=0) with ratios
    * --policy-bounds: also print energy policy bounds at fixed (π_way, π_miss)
        (optE for best energy, wstE for worst energy)
"""

import os, sys, json, math, argparse
import numpy as np
import pandas as pd
import joblib

# ------------------------------ utils ------------------------------

def eprint(*a, **k): print(*a, **k, file=sys.stderr)

def load_cfg(path:str)->dict:
    with open(path, "r") as f: return json.load(f)

def overlay(base, extra):
    """Shallow overlay; tolerate non-dict bases (e.g., 'optE')."""
    if base is None:
        base = {}
    elif not isinstance(base, dict):
        base = {"mode": str(base)}
    out = dict(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = overlay(out[k], v)
        else:
            out[k] = v
    return out

def mid_fracs(df: pd.DataFrame):
    mids = 0.5*(df["start_cycle"].to_numpy() + df["end_cycle"].to_numpy())
    total = float(df["end_cycle"].max() - df["start_cycle"].min())
    return mids / (total if total > 0 else 1.0)

def load_rep_fracs(char_dir: str):
    try:
        reps = json.load(open(os.path.join(char_dir, "LLC.representatives.json")))["representatives"]
        feats = pd.read_csv(os.path.join(char_dir, "LLC.window_features.csv"), engine="python", on_bad_lines="skip")
    except Exception:
        return []
    d = dict(zip(feats["window_id"].to_numpy(), mid_fracs(feats)))
    return [d[w] for w in reps if w in d]

def pick_unique(csv_path: str, fracs):
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    if df.empty or not fracs: return df.iloc[0:0].copy()
    f = mid_fracs(df); used=set(); idx=[]
    for x in fracs:
        for j in np.argsort(np.abs(f-x)):
            j=int(j)
            if j not in used:
                used.add(j); idx.append(j); break
    return df.iloc[idx].copy()

def pick_rows(ds: pd.DataFrame, tpw: float, tpm: float, only=None):
    if only: ds = ds[ds["bench"].isin(only)]
    rows=[]
    for b,g in ds.groupby("bench"):
        d=(g["pi_way"]-tpw)**2 + (g["pi_miss"]-tpm)**2 + 1e-4*((g["t_sram_hit"]-16)**2+(g["t_mram_rd"]-28)**2+(g["t_mram_wr"]-60)**2)
        rows.append(g.iloc[int(np.argmin(d.values))])
    return (pd.DataFrame(rows).sort_values("bench").reset_index(drop=True)) if rows else pd.DataFrame(columns=ds.columns)

def frac_from_curve(curves: dict, bench: str, pm: float, pw: float):
    tb = curves.get(bench, {})
    key = f"{pm:.4f}" if f"{pm:.4f}" in tb else ("__pooled__" if "__pooled__" in tb else None)
    if key is None: return float("nan")
    xs=np.asarray(tb[key]["x"], float); ys=np.asarray(tb[key]["y"], float)
    return float(np.interp(pw, xs, ys)) if xs.size>0 else float("nan")

# --------------------------- policy logic ---------------------------

def clamp_allocation(HM_tot, Hrd, Hwr, HM_rd, constraints:dict):
    """enforce hm_rd/wr min/max fractions if provided"""
    HM_rd = max(0.0, min(HM_rd, HM_tot))
    HM_wr = HM_tot - HM_rd
    c = constraints or {}
    rd_min = c.get("hm_rd_min_frac"); rd_max = c.get("hm_rd_max_frac")
    wr_min = c.get("hm_wr_min_frac"); wr_max = c.get("hm_wr_max_frac")
    # wr bounds by moving HM_rd
    if wr_min is not None and HM_wr < wr_min*HM_tot: HM_rd = HM_tot - wr_min*HM_tot
    if wr_max is not None and HM_wr > wr_max*HM_tot: HM_rd = HM_tot - wr_max*HM_tot
    # rd bounds
    if rd_min is not None and HM_rd < rd_min*HM_tot: HM_rd = rd_min*HM_tot
    if rd_max is not None and HM_rd > rd_max*HM_tot: HM_rd = rd_max*HM_tot
    HM_rd = max(0.0, min(HM_rd, HM_tot)); HM_wr = HM_tot - HM_rd
    # clamp to available hits
    HM_rd = min(HM_rd, Hrd); HM_wr = min(HM_wr, Hwr)
    # re-assign any leftover to the side with headroom
    leftover = HM_tot - (HM_rd + HM_wr)
    if leftover > 0:
        if (Hrd - HM_rd) >= (Hwr - HM_wr): HM_rd = min(Hrd, HM_rd + leftover)
        else:                              HM_wr = min(Hwr, HM_wr + leftover)
    return HM_rd, HM_wr

def preset_allocation(mode, HM_tot, Hrd, Hwr, costs, lat3, mlpH):
    """mid/optE/wstE/optS/wstS"""
    eRS,eWS,eRM,eWM = costs
    tS,tMr,tMw = lat3
    if HM_tot <= 0 or (Hrd+Hwr) <= 0: return 0.0, 0.0
    if mode == "mid":
        HM_rd = min(HM_tot * Hrd/(Hrd+Hwr), Hrd); HM_wr = HM_tot - HM_rd; return HM_rd, HM_wr
    def greedy(first):
        hm_rd=hm_wr=0.0; rem=HM_tot
        if first=="rd":
            take=min(rem,Hrd); hm_rd+=take; rem-=take; hm_wr=min(rem,Hwr)
        else:
            take=min(rem,Hwr); hm_wr+=take; rem-=take; hm_rd=min(rem,Hrd)
        return hm_rd,hm_wr
    if mode in ("optE","wstE"):
        d_rd=(eRM-eRS); d_wr=(eWM-eWS)
        first="rd" if (d_rd<=d_wr) else "wr"
        if mode=="wstE": first="rd" if (d_rd>=d_wr) else "wr"
        return greedy(first)
    if mode in ("optS","wstS"):
        d_rd=(tMr-tS)/max(1.0,mlpH); d_wr=(tMw-tS)/max(1.0,mlpH)
        first="rd" if (d_rd<=d_wr) else "wr"
        if mode=="wstS": first="rd" if (d_rd>=d_wr) else "wr"
        return greedy(first)
    return preset_allocation("mid", HM_tot, Hrd, Hwr, costs, (tS,tMr,tMw), mlpH)

def eval_window(HM_rd, HM_wr, Hrd, Hwr, A1k, mr, mlpH, mlpM, costs, tS,tMr,tMw,tDR,f_ghz, E_MISS_BASE, E_MISS_PER_NS):
    eRS,eWS,eRM,eWM = costs
    stall_k = (HM_rd*(tMr-tS) + HM_wr*(tMw-tS))/mlpH + (mr*A1k*(tDR - tS))/mlpM
    stall_frac = stall_k/1000.0
    E_MISS = E_MISS_BASE + E_MISS_PER_NS*(tDR/f_ghz)
    HS_rd = Hrd - HM_rd; HS_wr = Hwr - HM_wr
    E_dyn_1k = HS_rd*eRS + HS_wr*eWS + HM_rd*eRM + HM_wr*eWM + (mr*A1k)*E_MISS
    return stall_frac, E_dyn_1k

def search_weighted(HM_tot, Hrd, Hwr, A1k, mr, mlpH, mlpM, costs, lat, lam, E_MISS_BASE, E_MISS_PER_NS, steps=101):
    tS,tMr,tMw,tDR,f_ghz = lat
    best = (None, None, float("inf"))
    for k in range(steps):
        HM_rd = HM_tot * (k/(steps-1))
        HM_wr = HM_tot - HM_rd
        HM_rd = min(HM_rd, Hrd); HM_wr = min(HM_wr, Hwr)
        stall_frac, E_dyn_1k = eval_window(HM_rd,HM_wr,Hrd,Hwr,A1k,mr,mlpH,mlpM,costs,tS,tMr,tMw,tDR,f_ghz,E_MISS_BASE,E_MISS_PER_NS)
        J = lam*E_dyn_1k + (1.0-lam)*stall_frac
        if J < best[2]:
            best = (HM_rd, HM_wr, J)
    return best[0], best[1]

def apply_policy_selection(policy_spec, constraints, HM_tot, Hrd, Hwr, A1k, mr, mlpH, mlpM, costs, lat, E_MISS_BASE, E_MISS_PER_NS):
    """Return HM_rd, HM_wr given a (possibly complex) policy spec"""
    if isinstance(policy_spec, str) or policy_spec is None:
        HM_rd, HM_wr = preset_allocation(policy_spec or "mid", HM_tot, Hrd, Hwr, costs, lat[:3], mlpH)
        return clamp_allocation(HM_tot,Hrd,Hwr,HM_rd,constraints)

    m = policy_spec.get("mode","mid")
    if m in ("mid","optE","wstE","optS","wstS"):
        HM_rd, HM_wr = preset_allocation(m, HM_tot, Hrd, Hwr, costs, lat[:3], mlpH)
        return clamp_allocation(HM_tot,Hrd,Hwr,HM_rd,constraints)

    if m == "bias":
        rb = float(policy_spec.get("rw_bias", 0.5))
        HM_rd = rb*HM_tot
        return clamp_allocation(HM_tot,Hrd,Hwr,HM_rd,constraints)

    if m == "bias_grid":
        grid = policy_spec.get("bias_grid",[0.0,0.25,0.5,0.75,1.0])
        optimize = policy_spec.get("optimize","energy")  # energy | stall | weighted
        lam = float(policy_spec.get("lambda",1.0))
        best=None
        for rb in grid:
            HM_rd = min(rb*HM_tot, Hrd); HM_wr = min(HM_tot-HM_rd, Hwr)
            stall_frac, E_dyn_1k = eval_window(HM_rd,HM_wr,Hrd,Hwr,A1k,mr,mlpH,mlpM,costs,*lat,E_MISS_BASE,E_MISS_PER_NS)
            J = {"stall": stall_frac, "weighted": lam*E_dyn_1k + (1-lam)*stall_frac}.get(optimize, E_dyn_1k)
            if (best is None) or (J < best[2]): best=(HM_rd,HM_wr,J)
        HM_rd,HM_wr,_=best
        return clamp_allocation(HM_tot,Hrd,Hwr,HM_rd,constraints)

    if m == "weighted":
        lam = float(policy_spec.get("lambda", 0.5))
        HM_rd, HM_wr = search_weighted(HM_tot,Hrd,Hwr,A1k,mr,mlpH,mlpM,costs,lat,lam,E_MISS_BASE,E_MISS_PER_NS)
        return clamp_allocation(HM_tot,Hrd,Hwr,HM_rd,constraints)

    if m == "weighted_search":
        lambdas = policy_spec.get("lambda_grid",[0.0,0.25,0.5,0.75,1.0])
        best=None
        for lam in lambdas:
            HM_rd, HM_wr = search_weighted(HM_tot,Hrd,Hwr,A1k,mr,mlpH,mlpM,costs,lat,lam,E_MISS_BASE,E_MISS_PER_NS)
            stall_frac, E_dyn_1k = eval_window(HM_rd,HM_wr,Hrd,Hwr,A1k,mr,mlpH,mlpM,costs,*lat,E_MISS_BASE,E_MISS_PER_NS)
            J = lam*E_dyn_1k + (1-lam)*stall_frac
            if (best is None) or (J < best[2]): best=(HM_rd,HM_wr,J,lam)
        HM_rd, HM_wr, *_ = best
        return clamp_allocation(HM_tot,Hrd,Hwr,HM_rd,constraints)

    # default
    HM_rd,HM_wr = preset_allocation("mid",HM_tot,Hrd,Hwr,costs,lat[:3],mlpH)
    return clamp_allocation(HM_tot,Hrd,Hwr,HM_rd,constraints)

def match_rule(cond:dict, fpr_row:pd.Series)->bool:
    if not cond: return False
    ops = {
        "_gte": lambda a,b: a >= b,
        "_lte": lambda a,b: a <= b,
        "_gt":  lambda a,b: a >  b,
        "_lt":  lambda a,b: a <  b,
        "_eq":  lambda a,b: abs(a-b) < 1e-12
    }
    for key,val in cond.items():
        for suf,op in ops.items():
            if key.endswith(suf):
                base=key[:-len(suf)]
                if base not in fpr_row.index: return False
                if not op(float(fpr_row[base]), float(val)): return False
                break
        else:
            if key not in fpr_row.index or abs(float(fpr_row[key])-float(val))>1e-12:
                return False
    return True

# --------------------------- main computation ---------------------------

def main():
    p = argparse.ArgumentParser(description="Whole-trace printer for Hybrid LLC (no file outputs).")
    p.add_argument("--cfg", required=True, help="Path to config JSON")
    p.add_argument("--cap", type=int, help="Override cap MB")
    p.add_argument("--pw", type=float, help="Override pi_way [0..1]")
    p.add_argument("--pm", type=float, help="Override pi_miss [0..1]")
    p.add_argument("--policy", help="Override policy preset (mid|optE|wstE|optS|wstS)")
    p.add_argument("--lat-mode", choices=["device","dataset"], help="Use device config latencies or dataset-row latencies")
    p.add_argument("--bench", action="append", help="Repeatable; include only these benches")
    p.add_argument("--leak-scale-with-stall", action="store_true", help="Scale leakage by (1+stall_frac)")
    p.add_argument("--compare-sram", action="store_true", help="Also print SRAM-only baseline (π_way=0) and ratios")
    p.add_argument("--policy-bounds", action="store_true", help="Also print energy policy bounds (optE/wstE) at fixed (π_way, π_miss)")
    args = p.parse_args()

    cfg = load_cfg(args.cfg)

    cap   = args.cap     if args.cap     is not None else int(cfg.get("cap_mb", 2))
    pw    = args.pw      if args.pw      is not None else float(cfg.get("pi_way", 0.5))
    pm    = args.pm      if args.pm      is not None else float(cfg.get("pi_miss", 0.5))
    latmd = args.lat_mode if args.lat_mode is not None else str(cfg.get("latency_mode", "device"))
    onlyb = args.bench if args.bench else (cfg.get("benches") or None)
    leak_scale = args.leak_scale_with_stall or bool(cfg.get("leak_scale_with_stall", False))
    compare_sram = args.compare_sram or bool(cfg.get("compare_sram", False))
    show_bounds  = args.policy_bounds or bool(cfg.get("policy_bounds", False))
    policy_overridden = args.policy is not None

    # Physics from config
    f_ghz = float(cfg.get("f_clk_ghz", 1.0))
    LAT   = cfg.get("latency", {}) or {}
    SRAM  = cfg.get("sram", {}) or {}
    MRAM  = cfg.get("mram", {}) or {}
    DRAM  = cfg.get("dram", {}) or {}

    global_policy = cfg.get("policy", "mid")
    constraints   = cfg.get("constraints", {}) or {}
    overrides     = cfg.get("policy_overrides", {}) or {}
    rules         = cfg.get("policy_rules", []) or {}
    budget        = cfg.get("budget", {}) or {}

    global_policy = args.policy if args.policy else global_policy  # CLI preset override

    # energies/leak
    E_RS = SRAM.get("e_rd_pj", 300)*1e-12; E_WS = SRAM.get("e_wr_pj", 400)*1e-12
    E_RM = MRAM.get("e_rd_pj", 600)*1e-12; E_WM = MRAM.get("e_wr_pj", 900)*1e-12
    P_LS = SRAM.get("leak_mw_per_mb", 5.0)*1e-3
    P_LM = MRAM.get("leak_mw_per_mb", 0.5)*1e-3
    E_MISS_BASE   = DRAM.get("e_miss_pj", 3000.0)*1e-12
    E_MISS_PER_NS = DRAM.get("e_miss_per_ns_pj", 0.0)*1e-12
    costs = (E_RS,E_WS,E_RM,E_WM)

    # inputs
    ds_path = os.path.join("results","eval",f"dataset_L3_{cap}MB.csv")
    fp_path = os.path.join("results","surrogate",f"L3_{cap}","benchmark_fingerprints.csv")
    hm_path = os.path.join("results","surrogate",f"L3_{cap}","hm_curves.joblib")
    if not os.path.exists(ds_path): eprint(f"[error] missing {ds_path}"); sys.exit(2)
    if not os.path.exists(fp_path): eprint(f"[error] missing {fp_path}"); sys.exit(2)
    if not os.path.exists(hm_path): eprint(f"[error] missing {hm_path}"); sys.exit(2)

    ds = pd.read_csv(ds_path).dropna(subset=["stall_pct","E_total_J"])
    fp = pd.read_csv(fp_path).set_index("bench")
    curves = joblib.load(hm_path)
    rows = pick_rows(ds, pw, pm, onlyb)
    if rows.empty:
        eprint("[warn] no rows selected"); sys.exit(0)

    out=[]
    for _,r in rows.iterrows():
        b=r["bench"]; L3=float(r.get("l3_mb",cap))
        if b not in fp.index or b not in curves: continue

        L3i=int(round(L3))
        cdir=os.path.join("results",f"characterization_L3_{L3i}",b)
        reps = load_rep_fracs(cdir)
        run_csv=os.path.join("results","eval",b,r["tag"],"LLC.llc.win.csv")
        Tk = None
        if os.path.exists(run_csv) and reps:
            sel=pick_unique(run_csv,reps)
            if not sel.empty: Tk=float(((sel["end_cycle"]-sel["start_cycle"]).sum())/1000.0)
        if not (Tk and math.isfinite(Tk) and Tk>0): Tk = float(r.get("window_cycles_sum", 1000.0))/1000.0
        Ttot = float(r.get("roi_cycles", 1000.0))/1000.0
        scale = Ttot/max(Tk,1e-9)

        # latencies
        tS=float(r["t_sram_hit"]); tMr=float(r["t_mram_rd"]); tMw=float(r["t_mram_wr"]); tDR=float(r.get("t_dram",200.0))
        if latmd=="device":
            tS=float(LAT.get("t_sram_hit",tS)); tMr=float(LAT.get("t_mram_rd",tMr))
            tMw=float(LAT.get("t_mram_wr",tMw)); tDR=float(LAT.get("t_miss_cycles",tDR))
        lat_tuple = (tS,tMr,tMw,tDR,f_ghz)

        # HM curve fraction for this (pw,pm)
        frac = frac_from_curve(curves, b, pm, pw)

        # per-1k counts
        fpr = fp.loc[b]
        A1k=float(fpr.char_acc_per_1kcyc); mr=float(fpr.char_miss_rate); rf=float(fpr.char_read_frac)
        H1k=A1k - mr*A1k; Hrd=rf*H1k; Hwr=(1-rf)*H1k
        mlpH=max(1.0,float(fpr.char_mlp_hit)); mlpM=max(1.0,float(fpr.char_mlp_miss))
        HM_tot = max(0.0, float(frac)*H1k if not math.isnan(frac) else 0.0)

        # compose bench-specific policy
        bench_policy = global_policy
        if isinstance(bench_policy, dict):
            bench_policy = dict(bench_policy)

        # rule overlay & overrides only if CLI did NOT force a preset
        if not policy_overridden:
            if isinstance(rules, list):
                for rule in rules:
                    cond = (rule or {}).get("if",{})
                    if cond and match_rule({k.replace("char_",""):v for k,v in cond.items()},
                                           fpr.rename(lambda x:x.replace("char_",""))):
                        bench_policy = overlay(bench_policy, (rule.get("set") or {}))
                        break
            if overrides.get(b):
                bench_policy = overlay(bench_policy, overrides[b])

        # compute allocation for main case
        HM_rd, HM_wr = apply_policy_selection(
            bench_policy, constraints, HM_tot,Hrd,Hwr, A1k,mr,mlpH,mlpM, costs, lat_tuple,
            E_MISS_BASE, E_MISS_PER_NS
        )

        # window-level eval (main)
        stall_frac, E_dyn_1k = eval_window(
            HM_rd,HM_wr,Hrd,Hwr,A1k,mr,mlpH,mlpM,costs,*lat_tuple,E_MISS_BASE,E_MISS_PER_NS
        )

        # whole-trace time & energy (main)
        T_1KSEC = 1e-6/f_ghz
        exec_whole_s = T_1KSEC * Ttot * (1.0 + stall_frac)
        C_S=(1.0-pw)*L3; C_M=pw*L3
        leak_time_sec = T_1KSEC * Ttot * ((1.0 + stall_frac) if leak_scale else 1.0)
        E_dyn_whole  = (E_dyn_1k*Tk) * scale
        E_leak_whole = (P_LS*C_S + P_LM*C_M) * leak_time_sec
        E_total_whole = E_dyn_whole + E_leak_whole

        row = {
            "bench":b,"pw":pw,"pm":pm,
            "policy": (bench_policy.get("mode","preset") if isinstance(bench_policy,dict) else str(bench_policy)),
            "lat":latmd, "Tk_kcyc":Tk, "Ttot_kcyc":Ttot, "scale":scale,
            "exec_whole_s":exec_whole_s,
            "E_total_whole_uJ":E_total_whole*1e6,
            "E_dyn_whole_uJ":E_dyn_whole*1e6,
            "E_leak_whole_uJ":E_leak_whole*1e6,
            "HM_frac":frac
        }

        # SRAM-only baseline (π_way=0), policy irrelevant; FORCE HM_tot=0
        if compare_sram:
            stall_frac_s = ((mr*A1k*(tDR - tS))/mlpM) / 1000.0
            exec_sram = T_1KSEC * Ttot * (1.0 + stall_frac_s)
            # dynamic with all hits on SRAM
            E_MISS = E_MISS_BASE + E_MISS_PER_NS*(tDR/f_ghz)
            E_dyn_1k_s = Hrd*E_RS + Hwr*E_WS + (mr*A1k)*E_MISS
            E_dyn_whole_s = (E_dyn_1k_s*Tk) * scale
            C_S_s=(1.0-0.0)*L3; C_M_s=0.0
            leak_time_s = T_1KSEC * Ttot * ((1.0 + stall_frac_s) if leak_scale else 1.0)
            E_leak_whole_s = (P_LS*C_S_s + P_LM*C_M_s) * leak_time_s
            E_tot_s = E_dyn_whole_s + E_leak_whole_s
            row.update({
                "exec_whole_s_sram": exec_sram,
                "E_total_whole_uJ_sram": E_tot_s*1e6,
                "E_total_vs_sram": (E_total_whole / max(E_tot_s,1e-30)),
                "exec_vs_sram": (exec_whole_s / max(exec_sram,1e-30))
            })

        # Energy policy bounds at fixed (pw, pm)
        if show_bounds:
            # optE / wstE allocations
            HM_rd_optE, HM_wr_optE = preset_allocation("optE", HM_tot,Hrd,Hwr, costs, lat_tuple[:3], mlpH)
            HM_rd_wstE, HM_wr_wstE = preset_allocation("wstE", HM_tot,Hrd,Hwr, costs, lat_tuple[:3], mlpH)
            # optE
            stall_optE, E1k_optE = eval_window(HM_rd_optE,HM_wr_optE,Hrd,Hwr,A1k,mr,mlpH,mlpM,costs,*lat_tuple,E_MISS_BASE,E_MISS_PER_NS)
            E_dyn_whole_optE  = (E1k_optE*Tk) * scale
            leak_time_optE = T_1KSEC * Ttot * ((1.0 + stall_optE) if leak_scale else 1.0)
            E_leak_whole_optE = (P_LS*C_S + P_LM*C_M) * leak_time_optE
            E_tot_optE = E_dyn_whole_optE + E_leak_whole_optE
            # wstE
            stall_wstE, E1k_wstE = eval_window(HM_rd_wstE,HM_wr_wstE,Hrd,Hwr,A1k,mr,mlpH,mlpM,costs,*lat_tuple,E_MISS_BASE,E_MISS_PER_NS)
            E_dyn_whole_wstE  = (E1k_wstE*Tk) * scale
            leak_time_wstE = T_1KSEC * Ttot * ((1.0 + stall_wstE) if leak_scale else 1.0)
            E_leak_whole_wstE = (P_LS*C_S + P_LM*C_M) * leak_time_wstE
            E_tot_wstE = E_dyn_whole_wstE + E_leak_whole_wstE
            row.update({
                "E_total_whole_uJ_optE": E_tot_optE*1e6,
                "E_total_whole_uJ_wstE": E_tot_wstE*1e6
            })

        out.append(row)

    df=pd.DataFrame(out).sort_values("bench")
    if df.empty:
        eprint("[warn] no outputs"); sys.exit(0)

    # budget note (informative)
    if budget:
        emax = budget.get("E_total_uJ_max")
        if emax is not None:
            bad = (df["E_total_whole_uJ"] > float(emax)+1e-12).sum()
            if bad>0: eprint(f"[budget] {bad} benches exceed E_total_uJ_max={emax}")

    # columns to print
    cols=["bench","pw","pm","policy","lat","Tk_kcyc","Ttot_kcyc","scale","exec_whole_s",
          "E_total_whole_uJ","E_dyn_whole_uJ","E_leak_whole_uJ","HM_frac"]
    if compare_sram:
        cols += ["exec_whole_s_sram","E_total_whole_uJ_sram","E_total_vs_sram","exec_vs_sram"]
    if show_bounds:
        cols += ["E_total_whole_uJ_optE","E_total_whole_uJ_wstE"]

    fmt={"Tk_kcyc":"{:.1f}".format,"Ttot_kcyc":"{:.1f}".format,"scale":"{:.2f}".format,
         "exec_whole_s":"{:.4f}".format,"E_total_whole_uJ":"{:.3f}".format,
         "E_dyn_whole_uJ":"{:.3f}".format,"E_leak_whole_uJ":"{:.3f}".format,
         "exec_whole_s_sram":"{:.4f}".format if compare_sram else str,
         "E_total_whole_uJ_sram":"{:.3f}".format if compare_sram else str,
         "E_total_vs_sram":"{:.3f}x".format if compare_sram else str,
         "exec_vs_sram":"{:.3f}x".format if compare_sram else str,
         "E_total_whole_uJ_optE":"{:.3f}".format if show_bounds else str,
         "E_total_whole_uJ_wstE":"{:.3f}".format if show_bounds else str,
         "HM_frac": (lambda x: "nan" if pd.isna(x) else f"{x:.3f}")}

    print(df[cols].to_string(index=False, formatters=fmt))

    # median summary
    label_policy = (args.policy if args.policy
                    else (global_policy if isinstance(global_policy, str)
                          else global_policy.get("mode","preset")))
    print("\n[medians]  exec_whole={:.4f}s  E_total_whole={:.3f} µJ  (cap={}MB, pw={:.2f}, pm={:.2f}, policy={}, lat={}, leak_scale={})"
          .format(df["exec_whole_s"].median(), df["E_total_whole_uJ"].median(),
                  cap, pw, pm, label_policy, latmd, leak_scale))
    if compare_sram:
        print("[medians]  vs SRAM: exec={:.3f}×  energy={:.3f}×"
              .format((df["exec_whole_s"]/df["exec_whole_s_sram"]).median(),
                      (df["E_total_whole_uJ"]/df["E_total_whole_uJ_sram"]).median()))
    if show_bounds:
        print("[medians]  bounds (energy): optE={:.3f} µJ  wstE={:.3f} µJ"
              .format(df["E_total_whole_uJ_optE"].median(),
                      df["E_total_whole_uJ_wstE"].median()))

if __name__ == "__main__":
    main()

