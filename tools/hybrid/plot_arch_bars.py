#!/usr/bin/env python3
import os, json, glob, argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- small utils -----------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def short_bench(b):
    try: return b.split('.',1)[1].split('_',1)[0]
    except: return b

def safe_ratio(num, den):
    num = np.asarray(num, float); den = np.asarray(den, float)
    out = np.full_like(num, np.nan)
    mask = np.isfinite(den) & (den > 0)
    out[mask] = num[mask] / den[mask]
    return out

def auto_unit(values_list):
    vals = np.concatenate([np.ravel(v[np.isfinite(v)]) for v in values_list if getattr(v, "size", 0)>0] or [np.array([1.0])])
    vmax = np.nanmax(vals) if vals.size else 1.0
    if vmax >= 1.0:   return 1.0, "J"
    if vmax >= 1e-3:  return 1e-3, "mJ"
    if vmax >= 1e-6:  return 1e-6, "µJ"
    return 1e-9, "nJ"

def clamp_ratio_axis(ax, ratios, pad_frac=0.05):
    r = np.asarray(ratios, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        ax.set_ylim(0.9, 1.1); return
    rmin, rmax = float(r.min()), float(r.max())
    span = max(1e-6, rmax-rmin)
    low  = rmin - pad_frac*span
    high = rmax + pad_frac*span
    if high-low < 0.1:
        mid = 0.5*(low+high); low, high = mid-0.05, mid+0.05
    ax.set_ylim(low, high)

def alloc_fixed_partition(HM_tot, Hrd, Hwr, mode):
    """Allocate MRAM hits under a fixed π_way:
       mode='optE' (min dynamic energy) or 'wstE' (max). Greedy by cheaper/pricier event first."""
    if HM_tot <= 0 or (Hrd+Hwr) <= 0:
        return 0.0, 0.0
    def greedy(first):
        hm_rd=hm_wr=0.0; rem=HM_tot
        if first=="rd":
            take=min(rem, Hrd); hm_rd+=take; rem-=take
            take=min(rem, Hwr); hm_wr+=take; rem-=take
        else:
            take=min(rem, Hwr); hm_wr+=take; rem-=take
            take=min(rem, Hrd); hm_rd+=take; rem-=take
        return hm_rd, hm_wr
    return greedy(first=mode)

# ----------------------------- dataset & selection -----------------------------
def load_dataset(cap:int):
    p = f"results/eval/dataset_L3_{cap}MB.csv"
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing {p}. Run postprocess_after_eval.sh --caps {cap}.")
    return pd.read_csv(p).dropna(subset=["stall_pct","E_total_J"])

def hm_benches(cap:int):
    hp = f"results/surrogate/L3_{cap}/hm_curves.joblib"
    if not os.path.exists(hp): return []
    curves = joblib.load(hp)
    benches = sorted(curves.keys())
    preferred = ['602.gcc_s-1850B','605.mcf_s-994B','619.lbm_s-2677B','620.omnetpp_s-874B',
                 '621.wrf_s-6673B','623.xalancbmk_s-700B','631.deepsjeng_s-928B',
                 '649.fotonik3d_s-7084B','657.xz_s-3167B']
    order = [b for b in preferred if b in benches] + [b for b in benches if b not in preferred]
    return order[:9]

def pick_canonical_rows(ds:pd.DataFrame, target_pw=0.50, target_pm=0.50, only_benches=None):
    if only_benches:
        ds = ds[ds["bench"].isin(only_benches)]
    rows=[]
    for b,grp in ds.groupby("bench"):
        d = (grp["pi_way"]-target_pw)**2 + (grp["pi_miss"]-target_pm)**2 \
            + 1e-4*((grp["t_sram_hit"]-16)**2 + (grp["t_mram_rd"]-28)**2 + (grp["t_mram_wr"]-60)**2)
        rows.append(grp.iloc[int(np.argmin(d.values))])
    return pd.DataFrame(rows).sort_values("bench").reset_index(drop=True)

# ----------------------------- reps → Tk (time mapping) -----------------------------
def mid_fracs(df):
    mids = 0.5*(df.start_cycle.to_numpy()+df.end_cycle.to_numpy())
    T = float(df.end_cycle.max()-df.start_cycle.min())
    return mids/(T if T>0 else 1.0)

def load_rep_fracs(char_dir):
    reps = json.load(open(os.path.join(char_dir,"LLC.representatives.json")))["representatives"]
    feats= pd.read_csv(os.path.join(char_dir,"LLC.window_features.csv"), engine="python", on_bad_lines="skip")
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

def get_Tk_for_row(cap:int, row:pd.Series):
    b=row["bench"]; L3=float(row.get("l3_mb", cap))
    cdir=f"results/characterization_L3_{int(L3)}/{b}"
    rep_fr=load_rep_fracs(cdir)
    run_csv=f"results/eval/{b}/{row['tag']}/LLC.llc.win.csv"
    reps=pick_unique(run_csv, rep_fr)
    Tk = float(((reps.end_cycle - reps.start_cycle).sum())/1000.0)  # kcycles
    return Tk

# ----------------------------- device json -----------------------------
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

# ----------------------------- physics kernel -----------------------------
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

# ----------------------------- figure builder -----------------------------
def make_6panel_bars(benches, labels, variants, baseline_idx, title, out_png, ratio_err=None):
    """
    variants: dict with keys exec, etot, edyn, eleak, rd, wr, ms -> arrays [G,N] (MID values)
    baseline_idx: which row is the baseline for ratios
    ratio_err: optional dict with keys for raw and ratio plots, e.g.:
               {"exec": (low, high), "etot": (low, high), "edyn": (low, high),
                "exec_raw": (low, high), "etot_raw": (low, high), "edyn_raw": (low, high)}
               Values are (low_delta[G,N], high_delta[G,N])
    """
    G, N = variants["exec"].shape
    x = np.arange(N); w = min(0.8/G, 0.22)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    variant_colors = []
    for j in range(G):
        variant_colors.append("lightgray" if j==baseline_idx else colors[(j-(baseline_idx+1)) % len(colors)]) # Adjusted index
    edge_kws = dict(edgecolor="black", linewidth=0.6)

    scale, unit = auto_unit([variants["etot"], variants["eleak"], variants["edyn"]])
    Etot_s = variants["etot"]/scale; Eleak_s = variants["eleak"]/scale; Edyn_s = variants["edyn"]/scale
    rd_s   = variants["rd"]/scale;   wr_s    = variants["wr"]/scale;   ms_s   = variants["ms"]/scale

    base_exec = variants["exec"][baseline_idx]
    base_etot = variants["etot"][baseline_idx]
    base_edyn = variants["edyn"][baseline_idx]

    R_exec = np.array([safe_ratio(variants["exec"][j], base_exec) for j in range(G)])
    R_etot = np.array([safe_ratio(variants["etot"][j], base_etot) for j in range(G)])
    R_edyn = np.array([safe_ratio(variants["edyn"][j], base_edyn) for j in range(G)])

    fig, axs = plt.subplots(3,2, figsize=(14,10))
    ebar_kws = dict(fmt='none', ecolor='black', elinewidth=1.0, capsize=2)

    # Exec raw
    ax=axs[0,0]
    for j in range(G):
        xpos = x+(j-(G-1)/2)*w
        ax.bar(xpos, variants["exec"][j], width=w, label=labels[j],
               color=variant_colors[j], alpha=0.95, **edge_kws)
        if ratio_err and "exec_raw" in ratio_err:
            low, high = ratio_err["exec_raw"]
            ax.errorbar(xpos, variants["exec"][j], yerr=[low[j], high[j]], **ebar_kws)
    ax.set_title("Execution time — raw"); ax.set_ylabel("seconds")
    ax.set_xticks(x); ax.set_xticklabels([short_bench(b) for b in benches], rotation=60, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3); ax.legend(fontsize=8, ncol=min(G,4))

    # Exec ratio (+ error bars if provided)
    ax=axs[0,1]
    for j in range(G):
        xpos = x+(j-(G-1)/2)*w
        ax.bar(xpos, R_exec[j], width=w, color=variant_colors[j], alpha=0.95, **edge_kws)
        if ratio_err and "exec" in ratio_err:
            low, high = ratio_err["exec"]
            ax.errorbar(xpos, R_exec[j], yerr=[low[j], high[j]], **ebar_kws)
    ax.axhline(1.0, color='tab:blue', linestyle='--', linewidth=1)
    ax.set_title("Execution time — × baseline"); ax.set_ylabel("× baseline")
    ax.set_xticks(x); ax.set_xticklabels([short_bench(b) for b in benches], rotation=60, fontsize=8)
    clamp_ratio_axis(ax, R_exec); ax.grid(True, axis='y', alpha=0.3)

    # Total energy raw (stack dyn+leak)
    ax=axs[1,0]
    for j in range(G):
        xpos = x+(j-(G-1)/2)*w
        v_color = variant_colors[j]
        # Dynamic part (solid variant color, black edge)
        ax.bar(xpos, Edyn_s[j], width=w,
               color=v_color,
               alpha=0.95,
               label="Dynamic" if j==0 else None,
               **edge_kws)
        # Leakage part (white fill, variant color hatch/edge)
        ax.bar(xpos, Eleak_s[j], width=w, bottom=Edyn_s[j],
               color="white",
               edgecolor=v_color, # This colors the hatch lines AND the box border
               hatch="//",
               linewidth=0.6, # Match edge_kws
               label="Leakage" if j==0 else None)

        if ratio_err and "etot_raw" in ratio_err:
            low, high = ratio_err["etot_raw"]
            ax.errorbar(xpos, Etot_s[j], yerr=[low[j]/scale, high[j]/scale], **ebar_kws)
    ax.set_title("Total LLC energy — raw"); ax.set_ylabel(unit)
    ax.set_xticks(x); ax.set_xticklabels([short_bench(b) for b in benches], rotation=60, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3); ax.legend(loc="upper right", fontsize=8)

    # Total energy ratio (+ error bars if provided)
    ax=axs[1,1]
    for j in range(G):
        xpos = x+(j-(G-1)/2)*w
        ax.bar(xpos, R_etot[j], width=w, color=variant_colors[j], alpha=0.95, **edge_kws)
        if ratio_err and "etot" in ratio_err:
            low, high = ratio_err["etot"]
            ax.errorbar(xpos, R_etot[j], yerr=[low[j], high[j]], **ebar_kws)
    ax.axhline(1.0, color='tab:blue', linestyle='--', linewidth=1)
    ax.set_title("Total LLC energy — × baseline"); ax.set_ylabel("× baseline")
    ax.set_xticks(x); ax.set_xticklabels([short_bench(b) for b in benches], rotation=60, fontsize=8)
    clamp_ratio_axis(ax, R_etot); ax.grid(True, axis='y', alpha=0.3)

    # Dynamic raw (stack components)
    ax=axs[2,0]
    for j in range(G):
        xj = x+(j-(G-1)/2)*w
        ax.bar(xj, rd_s[j], width=w, color="tab:blue", **edge_kws, label="Read hits" if j==0 else None)
        ax.bar(xj, wr_s[j], width=w, bottom=rd_s[j], color="tab:orange", **edge_kws, label="Write hits" if j==0 else None)
        ax.bar(xj, ms_s[j], width=w, bottom=rd_s[j]+wr_s[j], color="tab:green", **edge_kws, label="Miss installs" if j==0 else None)
        if ratio_err and "edyn_raw" in ratio_err:
            low, high = ratio_err["edyn_raw"]
            ax.errorbar(xj, Edyn_s[j], yerr=[low[j]/scale, high[j]/scale], **ebar_kws)
    ax.set_title("Dynamic energy — raw breakdown"); ax.set_ylabel(unit)
    ax.set_xticks(x); ax.set_xticklabels([short_bench(b) for b in benches], rotation=60, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3); ax.legend(loc="upper right", fontsize=8, title="Dynamic components")

    # Dynamic ratio
    ax=axs[2,1]
    for j in range(G):
        xpos = x+(j-(G-1)/2)*w
        ax.bar(xpos, R_edyn[j], width=w, color=variant_colors[j], alpha=0.95, **edge_kws)
        if ratio_err and "edyn" in ratio_err:
            low, high = ratio_err["edyn"]
            ax.errorbar(xpos, R_edyn[j], yerr=[low[j], high[j]], **ebar_kws)
    ax.axhline(1.0, color='tab:blue', linestyle='--', linewidth=1)
    ax.set_title("Dynamic energy — × baseline"); ax.set_ylabel("× baseline")
    ax.set_xticks(x); ax.set_xticklabels([short_bench(b) for b in benches], rotation=60, fontsize=8)
    clamp_ratio_axis(ax, R_edyn); ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0,0,1,0.98])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

# ----------------------------- runners for each mode -----------------------------
def run_policy(cap, device_profile, device_root, latency_mode, target_pw, target_pm, outdir):
    ds = load_dataset(cap); benches9 = hm_benches(cap)
    rows = pick_canonical_rows(ds, target_pw, target_pm, benches9)
    fp = pd.read_csv(f"results/surrogate/L3_{cap}/benchmark_fingerprints.csv").set_index("bench")

    dj = load_device(device_profile, cap, device_root)
    if not dj: raise FileNotFoundError(f"device profile not found: {device_root}/{device_profile}")
    f_ghz = float(dj.get("f_clk_ghz",1.0))
    E_RD_SRAM = dj["sram"]["e_rd_pj"]*1e-12; E_WR_SRAM = dj["sram"]["e_wr_pj"]*1e-12
    E_RD_MRAM = dj["mram"]["e_rd_pj"]*1e-12; E_WR_MRAM = dj["mram"]["e_wr_pj"]*1e-12
    P_LEAK_S  = dj["sram"]["leak_mw_per_mb"]*1e-3; P_LEAK_M = dj["mram"]["leak_mw_per_mb"]*1e-3
    dram      = dj.get("dram", {})
    E_MISS_BASE   = dram.get("e_miss_pj", 3000.0)*1e-12
    E_MISS_PER_NS = dram.get("e_miss_per_ns_pj", 0.0)*1e-12
    LAT           = dj.get("latency", {})

    benches = rows["bench"].tolist(); N=len(benches)
    labels = ["PESS","OPT"]          # baseline first
    G=2
    variants = {k:np.full((G,N), np.nan) for k in ["exec","etot","edyn","eleak","rd","wr","ms"]}

    for i, r in enumerate(rows.itertuples(index=False)):
        b=r.bench; L3=float(getattr(r,"l3_mb", cap))
        if b not in fp.index: continue
        Tk = get_Tk_for_row(cap, pd.Series(r._asdict()))
        tS=float(r.t_sram_hit); tMr=float(r.t_mram_rd); tMw=float(r.t_mram_wr); tDR=float(getattr(r,"t_dram",200.0))
        if latency_mode=="device":
            tS=float(LAT.get("t_sram_hit",tS)); tMr=float(LAT.get("t_mram_rd",tMr))
            tMw=float(LAT.get("t_mram_wr",tMw)); tDR=float(LAT.get("t_miss_cycles",tDR))
        E_MISS = E_MISS_BASE + E_MISS_PER_NS*(tDR / f_ghz)
        fpr=fp.loc[b]
        A1k=float(fpr.char_acc_per_1kcyc); mr=float(fpr.char_miss_rate); rf=float(fpr.char_read_frac)
        H1k=A1k-mr*A1k; Hrd=rf*H1k; Hwr=(1-rf)*H1k
        lat=(tS,tMr,tMw,tDR)

        # PESS baseline
        pess = eval_case(r, fpr, L3, Tk, f_ghz, E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,
                         P_LEAK_S,P_LEAK_M,E_MISS,
                         HM_rd=Hrd if E_RD_MRAM>E_RD_SRAM else 0.0,
                         HM_wr=Hwr if E_WR_MRAM>E_WR_SRAM else 0.0,
                         latency=lat, leak_pi_way=target_pw)
        opt  = eval_case(r, fpr, L3, Tk, f_ghz, E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,
                         P_LEAK_S,P_LEAK_M,E_MISS,
                         HM_rd=Hrd if E_RD_MRAM<E_RD_SRAM else 0.0,
                         HM_wr=Hwr if E_WR_MRAM<E_WR_SRAM else 0.0,
                         latency=lat, leak_pi_way=target_pw)

        for j,res in enumerate([pess,opt]):
            variants["exec"][j,i]=res["exec_sec"]; variants["etot"][j,i]=res["E_tot"]
            variants["edyn"][j,i]=res["E_dyn"];   variants["eleak"][j,i]=res["E_leak"]
            variants["rd"][j,i]=res["E_rd_hits"];  variants["wr"][j,i]=res["E_wr_hits"]; variants["ms"][j,i]=res["E_miss"]

    outdir = outdir or f"results/figs/L3_{cap}/arch"
    ensure_dir(outdir)
    out = os.path.join(outdir, f"bars_policy_{device_profile}_pw{target_pw:.2f}_pm{target_pm:.2f}.png")
    make_6panel_bars(benches, labels, variants, baseline_idx=0,
                     title=f"cap={cap}MB, policy=PESS baseline, device={device_profile}, π_way={target_pw:.2f}, π_miss={target_pm:.2f}",
                     out_png=out)

def run_device(cap, profiles, device_root, latency_mode, target_pw, target_pm, outdir):
    ds = load_dataset(cap); benches9 = hm_benches(cap) or sorted(ds["bench"].unique().tolist())[:9]
    rows = pick_canonical_rows(ds, target_pw, target_pm, benches9)
    fp = pd.read_csv(f"results/surrogate/L3_{cap}/benchmark_fingerprints.csv").set_index("bench")
    curves = joblib.load(f"results/surrogate/L3_{cap}/hm_curves.joblib") if os.path.exists(f"results/surrogate/L3_{cap}/hm_curves.joblib") else {}

    benches = rows["bench"].tolist(); N=len(benches)
    labels = ["SRAM"] + profiles      # baseline first
    G = 1 + len(profiles)
    variants = {k:np.full((G,N), np.nan) for k in ["exec","etot","edyn","eleak","rd","wr","ms"]}

    # helper to evaluate one (HM_rd,HM_wr, leak_pi_way)
    def eval_variant(j_idx, i_idx, E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS,f_ghz,LAT, leak_pw, HM_rd, HM_wr, r, fpr, L3, Tk):
        lat_tuple=(float(r.t_sram_hit), float(r.t_mram_rd), float(r.t_mram_wr), float(getattr(r,"t_dram",200.0)))
        if latency_mode=="device":
            lat_tuple=(float(LAT.get("t_sram_hit",lat_tuple[0])),
                         float(LAT.get("t_mram_rd", lat_tuple[1])),
                         float(LAT.get("t_mram_wr", lat_tuple[2])),
                         float(LAT.get("t_miss_cycles", lat_tuple[3])))
        res = eval_case(r, fpr, L3, Tk, f_ghz, E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS,
                         HM_rd, HM_wr, lat_tuple, leak_pi_way=leak_pw)
        variants["exec"][j_idx, i_idx]=res["exec_sec"]
        variants["etot"][j_idx, i_idx]=res["E_tot"]
        variants["edyn"][j_idx, i_idx]=res["E_dyn"]
        variants["eleak"][j_idx, i_idx]=res["E_leak"]
        variants["rd"][j_idx, i_idx]=res["E_rd_hits"]
        variants["wr"][j_idx, i_idx]=res["E_wr_hits"]
        variants["ms"][j_idx, i_idx]=res["E_miss"]

    # baseline SRAM row
    for i, r in enumerate(rows.itertuples(index=False)):
        b=r.bench; L3=float(getattr(r,"l3_mb", cap))
        if b not in fp.index: continue
        Tk = get_Tk_for_row(cap, pd.Series(r._asdict()))
        # Any profile for energies is OK; use first device for unit consistency (or default constants)
        if profiles:
            dj0 = load_device(profiles[0], cap, device_root)
        else:
            dj0 = {"f_clk_ghz":1.0, "sram":{"e_rd_pj":300,"e_wr_pj":400,"leak_mw_per_mb":5.0},
                   "mram":{"e_rd_pj":600,"e_wr_pj":900,"leak_mw_per_mb":0.6}, "dram":{"e_miss_pj":3000,"e_miss_per_ns_pj":0.0}, "latency":{}}
        f_ghz = float(dj0.get("f_clk_ghz",1.0))
        LAT0  = dj0.get("latency",{})
        E_RD_SRAM = dj0["sram"]["e_rd_pj"]*1e-12; E_WR_SRAM = dj0["sram"]["e_wr_pj"]*1e-12
        E_RD_MRAM = dj0["mram"]["e_rd_pj"]*1e-12; E_WR_MRAM = dj0["mram"]["e_wr_pj"]*1e-12
        P_LEAK_S  = dj0["sram"]["leak_mw_per_mb"]*1e-3; P_LEAK_M = dj0["mram"]["leak_mw_per_mb"]*1e-3
        dram0     = dj0.get("dram", {})
        E_MISS_BASE   = dram0.get("e_miss_pj", 3000.0)*1e-12
        E_MISS_PER_NS = dram0.get("e_miss_per_ns_pj", 0.0)*1e-12
        fpr=fp.loc[b]
        # baseline: HM=0, leak π_way=0
        eval_variant(0, i, E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS_BASE, f_ghz, LAT0,
                     leak_pw=0.0, HM_rd=0.0, HM_wr=0.0, r=r, fpr=fpr, L3=L3, Tk=Tk)

    # device variants (policy=mid)
    for j, prof in enumerate(profiles, start=1):
        dj = load_device(prof, cap, device_root)
        if not dj: raise FileNotFoundError(f"device profile not found: {device_root}/{prof}")
        f_ghz = float(dj.get("f_clk_ghz",1.0))
        E_RD_SRAM = dj["sram"]["e_rd_pj"]*1e-12; E_WR_SRAM = dj["sram"]["e_wr_pj"]*1e-12
        E_RD_MRAM = dj["mram"]["e_rd_pj"]*1e-12; E_WR_MRAM = dj["mram"]["e_wr_pj"]*1e-12
        P_LEAK_S  = dj["sram"]["leak_mw_per_mb"]*1e-3; P_LEAK_M = dj["mram"]["leak_mw_per_mb"]*1e-3
        dram      = dj.get("dram", {})
        E_MISS_BASE   = dram.get("e_miss_pj", 3000.0)*1e-12
        E_MISS_PER_NS = dram.get("e_miss_per_ns_pj", 0.0)*1e-12
        LAT           = dj.get("latency", {})

        for i, r in enumerate(rows.itertuples(index=False)):
            b=r.bench; L3=float(getattr(r,"l3_mb", cap))
            if b not in fp.index or b not in curves: continue
            Tk = get_Tk_for_row(cap, pd.Series(r._asdict()))
            fpr=fp.loc[b]
            # HM curve at target_pw, target_pm
            tb = curves[b]; key = f"{target_pm:.4f}" if f"{target_pm:.4f}" in tb else ("__pooled__" if "__pooled__" in tb else None)
            if key is None: continue
            xs=np.asarray(tb[key]["x"], float); ys=np.asarray(tb[key]["y"], float)
            frac = float(np.interp(target_pw, xs, ys)) if xs.size>0 else 0.5
            A1k=float(fpr.char_acc_per_1kcyc); mr=float(fpr.char_miss_rate); rf=float(fpr.char_read_frac)
            H1k=A1k-mr*A1k; Hrd=rf*H1k; Hwr=(1-rf)*H1k
            HM_tot = frac * H1k
            HM_rd  = min(HM_tot * Hrd/(Hrd+Hwr+1e-12), Hrd)  # proportional "mid"
            HM_wr  = HM_tot - HM_rd

            eval_variant(j, i, E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,P_LEAK_S,P_LEAK_M,E_MISS_BASE,
                         f_ghz, LAT, leak_pw=target_pw, HM_rd=HM_rd, HM_wr=HM_wr, r=r, fpr=fpr, L3=L3, Tk=Tk)

    outdir = outdir or f"results/figs/L3_{cap}/arch"
    ensure_dir(outdir)
    out = os.path.join(outdir, f"bars_device_pw{target_pw:.2f}_mid.png")
    make_6panel_bars(benches, labels, variants, baseline_idx=0,
                     title=f"cap={cap}MB, device-sweep, policy=mid, baseline=SRAM, π_way={target_pw:.2f}",
                     out_png=out)

def run_partition(cap, device_profile, device_root, latency_mode, piways, target_pm, outdir, extra_sram_profile=""):
    ds = load_dataset(cap); benches9 = hm_benches(cap) or sorted(ds["bench"].unique().tolist())[:9]
    rows = pick_canonical_rows(ds, target_pw=0.50, target_pm=target_pm, only_benches=benches9)
    fp = pd.read_csv(f"results/surrogate/L3_{cap}/benchmark_fingerprints.csv").set_index("bench")
    curves = joblib.load(f"results/surrogate/L3_{cap}/hm_curves.joblib") if os.path.exists(f"results/surrogate/L3_{cap}/hm_curves.joblib") else {}

    dj = load_device(device_profile, cap, device_root)
    if not dj: raise FileNotFoundError(f"device profile not found: {device_root}/{device_profile}")
    f_ghz = float(dj.get("f_clk_ghz",1.0))
    E_RD_SRAM = dj["sram"]["e_rd_pj"]*1e-12; E_WR_SRAM = dj["sram"]["e_wr_pj"]*1e-12
    E_RD_MRAM = dj["mram"]["e_rd_pj"]*1e-12; E_WR_MRAM = dj["mram"]["e_wr_pj"]*1e-12
    P_LEAK_S  = dj["sram"]["leak_mw_per_mb"]*1e-3; P_LEAK_M = dj["mram"]["leak_mw_per_mb"]*1e-3
    dram      = dj.get("dram", {})
    E_MISS_BASE   = dram.get("e_miss_pj", 3000.0)*1e-12
    E_MISS_PER_NS = dram.get("e_miss_per_ns_pj", 0.0)*1e-12
    LAT           = dj.get("latency", {})

    benches = rows["bench"].tolist(); N=len(benches)
    labels = [f"pw={pw:.2f}" for pw in piways]
    G=len(piways)
    variants = {k:np.full((G,N), np.nan) for k in ["exec","etot","edyn","eleak","rd","wr","ms"]}

    # For ratio error bars
    R_exec_low = np.zeros((G,N)); R_exec_high = np.zeros((G,N))
    R_etot_low = np.zeros((G,N)); R_etot_high = np.zeros((G,N))
    R_edyn_low = np.zeros((G,N)); R_edyn_high = np.zeros((G,N))
    # For raw error bars
    R_exec_raw_low = np.zeros((G,N)); R_exec_raw_high = np.zeros((G,N))
    R_etot_raw_low = np.zeros((G,N)); R_etot_raw_high = np.zeros((G,N))
    R_edyn_raw_low = np.zeros((G,N)); R_edyn_raw_high = np.zeros((G,N))


    # evaluate MID + OPT/PESS for each (variant j, bench i)
    base_exec_mid = np.full(N, np.nan); base_etot_mid = np.full(N, np.nan)
    base_edyn_mid = np.full(N, np.nan) # baseline row uses pw=0 if present

    for j, pw in enumerate(piways):
        for i, r in enumerate(rows.itertuples(index=False)):
            b=r.bench; L3=float(getattr(r,"l3_mb", cap))
            if b not in fp.index or b not in curves: continue
            Tk = get_Tk_for_row(cap, pd.Series(r._asdict()))
            tS=float(r.t_sram_hit); tMr=float(r.t_mram_rd); tMw=float(r.t_mram_wr); tDR=float(getattr(r,"t_dram",200.0))
            if latency_mode=="device":
                tS=float(LAT.get("t_sram_hit",tS)); tMr=float(LAT.get("t_mram_rd",tMr))
                tMw=float(LAT.get("t_mram_wr",tMw)); tDR=float(LAT.get("t_miss_cycles",tDR))
            E_MISS = E_MISS_BASE + E_MISS_PER_NS*(tDR / f_ghz)

            fpr=fp.loc[b]
            tb = curves[b]; key = f"{target_pm:.4f}" if f"{target_pm:.4f}" in tb else ("__pooled__" if "__pooled__" in tb else None)
            if key is None: continue
            xs=np.asarray(tb[key]["x"], float); ys=np.asarray(tb[key]["y"], float)
            frac = float(np.interp(pw, xs, ys)) if xs.size>0 else 0.5

            A1k=float(fpr.char_acc_per_1kcyc); mr=float(fpr.char_miss_rate); rf=float(fpr.char_read_frac)
            H1k=A1k-mr*A1k; Hrd=rf*H1k; Hwr=(1-rf)*H1k
            HM_tot = frac * H1k

            # --- MID allocation (proportional) ---
            HM_rd_mid = min(HM_tot * Hrd/(Hrd+Hwr+1e-12), Hrd)
            HM_wr_mid = HM_tot - HM_rd_mid
            res_mid = eval_case(r, fpr, L3, Tk, f_ghz, E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,
                                P_LEAK_S,P_LEAK_M,E_MISS, HM_rd_mid, HM_wr_mid, (tS,tMr,tMw,tDR), leak_pi_way=pw)
            # Populate all "mid" variants for the bars
            variants["exec"][j,i]=res_mid["exec_sec"]; variants["etot"][j,i]=res_mid["E_tot"]
            variants["edyn"][j,i]=res_mid["E_dyn"];   variants["eleak"][j,i]=res_mid["E_leak"]
            variants["rd"][j,i]=res_mid["E_rd_hits"];  variants["wr"][j,i]=res_mid["E_wr_hits"]; variants["ms"][j,i]=res_mid["E_miss"]

            # Baseline (SRAM-only) arrays for ratios
            if np.isclose(pw, 0.0):
                base_exec_mid[i] = res_mid["exec_sec"]
                base_etot_mid[i] = res_mid["E_tot"]
                base_edyn_mid[i] = res_mid["E_dyn"]

            # --- OPT/PESS allocation for error bars (energy bounds) ---
            e_rd_diff = (E_RD_MRAM - E_RD_SRAM)
            e_wr_diff = (E_WR_MRAM - E_WR_SRAM)
            first_opt = "rd" if e_rd_diff <= e_wr_diff else "wr"
            first_wst = "rd" if e_rd_diff >= e_wr_diff else "wr"
            HM_rd_opt, HM_wr_opt = alloc_fixed_partition(HM_tot, Hrd, Hwr, first_opt)
            HM_rd_wst, HM_wr_wst = alloc_fixed_partition(HM_tot, Hrd, Hwr, first_wst)

            res_opt  = eval_case(r, fpr, L3, Tk, f_ghz, E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,
                                 P_LEAK_S,P_LEAK_M,E_MISS, HM_rd_opt, HM_wr_opt, (tS,tMr,tMw,tDR), leak_pi_way=pw)
            res_wst  = eval_case(r, fpr, L3, Tk, f_ghz, E_RD_SRAM,E_WR_SRAM,E_RD_MRAM,E_WR_MRAM,
                                 P_LEAK_S,P_LEAK_M,E_MISS, HM_rd_wst, HM_wr_wst, (tS,tMr,tMw,tDR), leak_pi_way=pw)

            # --- Calculate Raw Errors (Deltas) ---
            exec_mid = res_mid["exec_sec"]
            exec_min = min(res_opt["exec_sec"], res_wst["exec_sec"])
            exec_max = max(res_opt["exec_sec"], res_wst["exec_sec"])
            R_exec_raw_low[j,i] = max(0.0, exec_mid - exec_min)
            R_exec_raw_high[j,i] = max(0.0, exec_max - exec_mid)

            etot_mid = res_mid["E_tot"]
            etot_min = min(res_opt["E_tot"], res_wst["E_tot"])
            etot_max = max(res_opt["E_tot"], res_wst["E_tot"])
            R_etot_raw_low[j,i] = max(0.0, etot_mid - etot_min)
            R_etot_raw_high[j,i] = max(0.0, etot_max - etot_mid)

            edyn_mid = res_mid["E_dyn"]
            edyn_min = min(res_opt["E_dyn"], res_wst["E_dyn"])
            edyn_max = max(res_opt["E_dyn"], res_wst["E_dyn"])
            R_edyn_raw_low[j,i] = max(0.0, edyn_mid - edyn_min)
            R_edyn_raw_high[j,i] = max(0.0, edyn_max - edyn_mid)

            # --- Calculate Ratio Errors (Deltas) ---
            base_exec = base_exec_mid[i] if np.isfinite(base_exec_mid[i]) else variants["exec"][j,i]
            base_etot = base_etot_mid[i] if np.isfinite(base_etot_mid[i]) else variants["etot"][j,i]
            base_edyn = base_edyn_mid[i] if np.isfinite(base_edyn_mid[i]) else variants["edyn"][j,i]

            mid_R_exec = safe_ratio(np.array([res_mid["exec_sec"]]), np.array([base_exec]))[0]
            opt_R_exec = safe_ratio(np.array([res_opt["exec_sec"]]), np.array([base_exec]))[0]
            wst_R_exec = safe_ratio(np.array([res_wst["exec_sec"]]), np.array([base_exec]))[0]
            R_exec_low[j,i]  = max(0.0, mid_R_exec - min(opt_R_exec, wst_R_exec))
            R_exec_high[j,i] = max(0.0, max(opt_R_exec, wst_R_exec) - mid_R_exec)

            mid_R_et  = safe_ratio(np.array([res_mid["E_tot"]]), np.array([base_etot]))[0]
            opt_R_et  = safe_ratio(np.array([res_opt["E_tot"]]), np.array([base_etot]))[0]
            wst_R_et  = safe_ratio(np.array([res_wst["E_tot"]]), np.array([base_etot]))[0]
            R_etot_low[j,i]  = max(0.0, mid_R_et - min(opt_R_et, wst_R_et))
            R_etot_high[j,i] = max(0.0, max(opt_R_et, wst_R_et) - mid_R_et)

            mid_R_edyn = safe_ratio(np.array([res_mid["E_dyn"]]), np.array([base_edyn]))[0]
            opt_R_edyn = safe_ratio(np.array([res_opt["E_dyn"]]), np.array([base_edyn]))[0]
            wst_R_edyn = safe_ratio(np.array([res_wst["E_dyn"]]), np.array([base_edyn]))[0]
            R_edyn_low[j,i]  = max(0.0, mid_R_edyn - min(opt_R_edyn, wst_R_edyn))
            R_edyn_high[j,i] = max(0.0, max(opt_R_edyn, wst_R_edyn) - mid_R_edyn)

    # --- Start of new block to add extra SRAM bar ---
    if extra_sram_profile:
        djx = load_device(extra_sram_profile, cap, device_root)
        if not djx:
            raise FileNotFoundError(f"device profile not found: {device_root}/{extra_sram_profile}")

        f_ghz_x = float(djx.get("f_clk_ghz",1.0))
        E_RSx = djx["sram"]["e_rd_pj"]*1e-12; E_WSx = djx["sram"]["e_wr_pj"]*1e-12
        E_RMx = djx["mram"]["e_rd_pj"]*1e-12; E_WMx = djx["mram"]["e_wr_pj"]*1e-12
        P_LSx = djx["sram"]["leak_mw_per_mb"]*1e-3; P_LMx = djx["mram"]["leak_mw_per_mb"]*1e-3
        dramx = djx.get("dram", {})
        E_MISS_BASE_x   = dramx.get("e_miss_pj", 3000.0)*1e-12
        E_MISS_PER_NS_x = dramx.get("e_miss_per_ns_pj", 0.0)*1e-12
        LATx            = djx.get("latency", {})

        # build one extra row (N benches) with HM=0, leak π_way=0
        extra_exec  = np.full((1, N), np.nan)
        extra_etot  = np.full((1, N), np.nan)
        extra_edyn  = np.full((1, N), np.nan)
        extra_eleak = np.full((1, N), np.nan)
        extra_rd    = np.full((1, N), np.nan)
        extra_wr    = np.full((1, N), np.nan)
        extra_ms    = np.full((1, N), np.nan)

        for i, r in enumerate(rows.itertuples(index=False)):
            b=r.bench; L3=float(getattr(r,"l3_mb", cap))
            if b not in fp.index: continue
            Tk = get_Tk_for_row(cap, pd.Series(r._asdict()))
            tS=float(r.t_sram_hit); tMr=float(r.t_mram_rd); tMw=float(r.t_mram_wr); tDR=float(getattr(r,"t_dram",200.0))
            if latency_mode=="device":
                tS=float(LATx.get("t_sram_hit",tS)); tMr=float(LATx.get("t_mram_rd",tMr))
                tMw=float(LATx.get("t_mram_wr",tMw)); tDR=float(LATx.get("t_miss_cycles",tDR))
            E_MISS_x = E_MISS_BASE_x + E_MISS_PER_NS_x*(tDR / f_ghz_x)
            fpr = fp.loc[b]

            res = eval_case(r, fpr, L3, Tk, f_ghz_x,
                            E_RSx,E_WSx,E_RMx,E_WMx,P_LSx,P_LMx,E_MISS_x,
                            HM_rd=0.0, HM_wr=0.0, latency=(tS,tMr,tMw,tDR), leak_pi_way=0.0)

            extra_exec[0,i]  = res["exec_sec"]
            extra_etot[0,i]  = res["E_tot"]
            extra_edyn[0,i]  = res.get("E_dyn", np.nan)
            extra_eleak[0,i] = res.get("E_leak", np.nan)
            extra_rd[0,i]    = res.get("E_rd_hits", np.nan)
            extra_wr[0,i]    = res.get("E_wr_hits", np.nan)
            extra_ms[0,i]    = res.get("E_miss", np.nan)

        # append to variants
        variants["exec"]  = np.vstack([variants["exec"],  extra_exec])
        variants["etot"]  = np.vstack([variants["etot"],  extra_etot])
        variants["edyn"]  = np.vstack([variants["edyn"],  extra_edyn])
        variants["eleak"] = np.vstack([variants["eleak"], extra_eleak])
        variants["rd"]    = np.vstack([variants["rd"],    extra_rd])
        variants["wr"]    = np.vstack([variants["wr"],    extra_wr])
        variants["ms"]    = np.vstack([variants["ms"],    extra_ms])

        # error bars for SRAM-only = zero (OPT=PESS=MID)
        zeros = np.zeros((1, N))
        R_exec_low  = np.vstack([R_exec_low,  zeros]); R_exec_high = np.vstack([R_exec_high, zeros])
        R_etot_low  = np.vstack([R_etot_low,  zeros]); R_etot_high = np.vstack([R_etot_high, zeros])
        R_edyn_low  = np.vstack([R_edyn_low,  zeros]); R_edyn_high = np.vstack([R_edyn_high, zeros])
        R_exec_raw_low  = np.vstack([R_exec_raw_low,  zeros]); R_exec_raw_high = np.vstack([R_exec_raw_high, zeros])
        R_etot_raw_low  = np.vstack([R_etot_raw_low,  zeros]); R_etot_raw_high = np.vstack([R_etot_raw_high, zeros])
        R_edyn_raw_low  = np.vstack([R_edyn_raw_low,  zeros]); R_edyn_raw_high = np.vstack([R_edyn_raw_high, zeros])

        labels.append("n10 SRAM")
    # --- End of new block ---


    # choose baseline index = pw==0.00 if present else first
    try: baseline_idx = piways.index(0.0)
    except ValueError: baseline_idx = 0

    outdir = outdir or f"results/figs/L3_{cap}/arch"
    ensure_dir(outdir)
    out = os.path.join(outdir, f"bars_partition_mid_pm{target_pm:.2f}.png")
    
    err_dict = {"exec": (R_exec_low, R_exec_high),
                "etot": (R_etot_low, R_etot_high),
                "edyn": (R_edyn_low, R_edyn_high),
                "exec_raw": (R_exec_raw_low, R_exec_raw_high),
                "etot_raw": (R_etot_raw_low, R_etot_raw_high),
                "edyn_raw": (R_edyn_raw_low, R_edyn_raw_high)
               }
    
    make_6panel_bars(benches, labels, variants, baseline_idx=baseline_idx,
                     title=f"cap={cap}MB, partition sweep (policy=mid), baseline=pw=0.00; error bars=OPT↔PESS",
                     out_png=out,
                     ratio_err=err_dict)

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Hybrid-LLC bars with explicit baselines")
    ap.add_argument("--mode", choices=["policy","device","partition"], required=True)
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--device-root", default="devices")
    ap.add_argument("--latency-mode", choices=["dataset","device"], default="device")
    ap.add_argument("--target-pw", type=float, default=0.50, help="used for policy/device modes")
    ap.add_argument("--target-pm", type=float, default=0.50)
    ap.add_argument("--outdir", default="")

    ap.add_argument("--device-profile", default="", help="for policy/partition modes")
    ap.add_argument("--profiles", default="", help="comma list for device mode")
    ap.add_argument("--piways", default="0.00,0.25,0.50,0.75,1.00", help="for partition mode")
    ap.add_argument("--extra-sram-profile", default="",
                    help="Append a pure-SRAM device (e.g., nvsim_n10_readedp) as an extra bar")

    args=ap.parse_args()

    if args.mode=="policy":
        if not args.device_profile: raise SystemExit("--device-profile required for mode=policy")
        run_policy(args.cap, args.device_profile, args.device_root, args.latency_mode,
                   args.target_pw, args.target_pm, args.outdir)
    elif args.mode=="device":
        profiles = [p for p in args.profiles.split(",") if p]
        if not profiles:
            roots = glob.glob(f"results/surrogate/profiles/*/L3_{args.cap}/physics_pred.csv")
            profiles = sorted({p.split(os.sep)[3] for p in roots})
        if not profiles: raise SystemExit("no device profiles; pass --profiles")
        run_device(args.cap, profiles, args.device_root, args.latency_mode, args.target_pw, args.target_pm, args.outdir)
    else:
        if not args.device_profile: raise SystemExit("--device-profile required for mode=partition")
        piways = [float(x) for x in args.piways.split(",")]
        run_partition(args.cap, args.device_profile, args.device_root, args.latency_mode,
                      piways, args.target_pm, args.outdir, args.extra_sram_profile)

    print("[done]")

if __name__=="__main__":
    main()
