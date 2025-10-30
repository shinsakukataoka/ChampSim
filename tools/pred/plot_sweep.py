#!/usr/bin/env python3

"""
Paper-style sweep figure (2x2 dashboard):

Top row:
  [L] Execution time — raw (seconds)
  [R] Execution time — × baseline

Bottom row:
  [L] Total LLC energy — raw (stacked: Dynamic + Leakage)
  [R] Total LLC energy — × baseline (bars) + policy envelope (whiskers)

Bars within each benchmark = different pi_way values (legend).
Whiskers = best↔worst across {optL,optDynE,optTotalE} and their pessimistic counterparts.

Run:
  python3 tools/pred/plot_sweep.py \
    --datasets-root /home/skataoka26/ChampSim/out/datasets \
    --model-dir       /home/skataoka26/ChampSim/model \
    --out-dir         /home/skataoka26/ChampSim/out/figs \
    --pi-ways 0.00 0.25 0.50 0.80 1.00 \
    --pi-miss 0.5 --tS 16 --tMr 28 --tMw 60 --sim 20000000
"""
import argparse, os, json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, yaml

THIS_DIR = os.path.dirname(__file__)
sys.path.append(THIS_DIR)
from physics import physics_predict_row, DEFAULTS as PHY_CONSTS
from bounds import accumulate_bounds

CLK_NS = 0.25  # seconds per cycle = CLK_NS * 1e-9

# --- ADDED: Benchmark abbreviations ---
SHORT = {
  "400.perlbench-41B.champsimtrace.xz":"perlbench",
  "600.perlbench_s-570B.champsimtrace.xz":"perlbench_s",
  "602.gcc_s-1850B.champsimtrace.xz":"gcc",
  "605.mcf_s-994B.champsimtrace.xz":"mcf",
  "619.lbm_s-2677B.champsimtrace.xz":"lbm",
  "620.omnetpp_s-874B.champsimtrace.xz":"omnetpp",
  "621.wrf_s-6673B.champsimtrace.xz":"wrf",
  "623.xalancbmk_s-700B.champsimtrace.xz":"xalancbmk",
  "631.deepsjeng_s-928B.champsimtrace.xz":"deepsjeng",
  "641.leela_s-1083B.champsimtrace.xz":"leela",
  "648.exchange2_s-1247B.champsimtrace.xz":"exchange2",
  "649.fotonik3d_s-7084B.champsimtrace.xz":"fotonik3d",
  "657.xz_s-3167B.champsimtrace.xz":"xz",
}

def list_benches(datasets_root):
    return [d for d in sorted(os.listdir(datasets_root))
            if os.path.isdir(os.path.join(datasets_root, d))]

def load_baseline(datasets_root, bench):
    p = os.path.join(datasets_root, bench, "baseline_windows.csv")
    if not os.path.exists(p): raise SystemExit(f"Missing {p}")
    return pd.read_csv(p)

def predict_cycles_sum(dfb, model, knobs, consts):
    tot = 0.0
    for _, brow in dfb.iterrows():
        cyc_phys, aux = physics_predict_row(brow, knobs, consts)
        X = np.array([
            float(brow["A_tot"]),
            float(brow["P_miss"]), float(brow["P_sram_rd"]), float(brow["P_mram_rd"]),
            float(brow["mlp_hit"]), float(brow["mlp_miss"]),
            float(brow["T_miss_base"]), float(brow["AMAT_base"]),
            float(knobs["tS"]), float(knobs["tMr"]), float(knobs["tMw"]),
            float(knobs["pi_miss"]), float(knobs["pi_way"]),
            float(aux["AMAT"]), float(aux["Tmiss"])
        ], dtype=float).reshape(1,-1)
        tot += (cyc_phys + float(model.predict(X)[0]))
    return tot

def dynamic_energy_for_knobs(dfb, knobs, consts, etab):
    """Medium-aware dynamic energy (pJ) using policy-dependent partitions from physics."""
    E_sum = 0.0
    for _, brow in dfb.iterrows():
        _, aux = physics_predict_row(brow, knobs, consts)
        A     = float(brow["A_tot"])
        Psr_r = float(aux.get("P_sram_rd", brow["P_sram_rd"]))
        Psr_w = float(aux.get("P_sram_wr", brow["P_sram_wr"]))
        Pmr_r = float(aux.get("P_mram_rd", brow["P_mram_rd"]))
        Pmr_w = float(aux.get("P_mram_wr", brow["P_mram_wr"]))
        Pmis  = float(brow["P_miss"])
        E = A*( Psr_r*etab["sram"]["read_hit_pJ"]  + Psr_w*etab["sram"]["write_hit_pJ"]
              + Pmr_r*etab["mram"]["read_hit_pJ"]  + Pmr_w*etab["mram"]["write_hit_pJ"]
              + Pmis  *etab["miss_path"]["per_miss_pJ"] )
        E_sum += E
    return E_sum

# --- REMOVED: dynamic_breakdown_for_knobs (no longer needed) ---

def seconds_from_cycles(cyc): return float(cyc) * CLK_NS * 1e-9
def uJ(pJ): return float(pJ) * 1e-6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="/home/skataoka26/ChampSim/out/datasets")
    ap.add_argument("--model-dir",     default="/home/skataoka26/ChampSim/model")
    ap.add_argument("--out-dir",       default="/home/skataoka26/ChampSim/out/figs")
    ap.add_argument("--benches", nargs="*", help="subset; default: first 10")
    ap.add_argument("--pi-ways", nargs="*", type=float, default=[0.00, 0.25, 0.50, 0.80, 1.00])
    ap.add_argument("--pi-miss", type=float, default=0.5)
    ap.add_argument("--tS", type=int, default=16)
    ap.add_argument("--tMr", type=int, default=28)
    ap.add_argument("--tMw", type=int, default=60)
    ap.add_argument("--sim", type=int, default=20_000_000, help="for reference IPC if needed; not plotted in this figure")
    ap.add_argument("--sort-by", choices=["name","pred","none"], default="name")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    benches = args.benches if args.benches else list_benches(args.datasets_root)
    benches = benches[:10]

    # constants & energy table
    consts = PHY_CONSTS.copy()
    yaml_path = os.path.join(args.model_dir, "physics_consts.yaml")
    if os.path.exists(yaml_path):
        consts.update(yaml.safe_load(open(yaml_path)))
    etab = json.load(open(os.path.join(args.model_dir, "energy_table.json")))

    # preload
    base = {b: load_baseline(args.datasets_root, b) for b in benches}
    models = {b: joblib.load(os.path.join(args.model_dir, f"{b}.pkl")) for b in benches}

    # compute baseline (reference) — keep using first pi_way passed (you can set it to all-SRAM)
    piw_ref = args.pi_ways[0]
    ref_time = {}
    ref_E_tot = {}
    ref_E_dyn = {} # Still needed for middle-left plot logic
    for b in benches:
        dfb, model = base[b], models[b]
        knobs = dict(tS=args.tS, tMr=args.tMr, tMw=args.tMw, pi_miss=args.pi_miss, pi_way=float(piw_ref))
        cyc = predict_cycles_sum(dfb, model, knobs, consts)
        time_s = seconds_from_cycles(cyc)
        E_dyn = dynamic_energy_for_knobs(dfb, knobs, consts, etab)
        theta = float(knobs["pi_way"])
        leak_mW = etab["sram"]["leak_mW"]*(1.0-theta) + etab["mram"]["leak_mW"]*theta
        E_leak = leak_mW * time_s * 1e9  # mW * ns = pJ
        ref_time[b] = time_s
        ref_E_tot[b] = E_dyn + E_leak
        ref_E_dyn[b] = E_dyn

    # sweep over pi_way values
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    K = len(args.pi_ways)
    all_time = {pw:[] for pw in args.pi_ways}
    all_time_lo = {pw:[] for pw in args.pi_ways}
    all_time_hi = {pw:[] for pw in args.pi_ways}
    all_Edyn = {pw:[] for pw in args.pi_ways}
    all_Etot = {pw:[] for pw in args.pi_ways}
    all_Etot_lo = {pw:[] for pw in args.pi_ways}
    all_Etot_hi = {pw:[] for pw in args.pi_ways}

    # --- REMOVED: dyn_break calculation (no longer needed) ---

    for pw in args.pi_ways:
        for b in benches:
            dfb, model = base[b], models[b]
            knobs = dict(tS=args.tS, tMr=args.tMr, tMw=args.tMw, pi_miss=args.pi_miss, pi_way=float(pw))
            # predicted cycles/time
            cyc_pred = predict_cycles_sum(dfb, model, knobs, consts)
            time_s = seconds_from_cycles(cyc_pred)

            # bounds for time (policy envelope)
            bnds = accumulate_bounds(dfb, knobs, consts, etab, clk_ns=CLK_NS)
            cyc_best  = min(bnds["cycles_optL"], bnds["cycles_optDynE"], bnds["cycles_optTotalE"])
            cyc_worst = max(bnds["cycles_optL_pess"], bnds["cycles_optDynE_pess"], bnds["cycles_optTotalE_pess"])
            time_best, time_worst = seconds_from_cycles(cyc_best), seconds_from_cycles(cyc_worst)

            # energies
            E_dyn = dynamic_energy_for_knobs(dfb, knobs, consts, etab)
            theta = float(knobs["pi_way"])
            leak_mW = etab["sram"]["leak_mW"]*(1.0-theta) + etab["mram"]["leak_mW"]*theta
            E_leak = leak_mW * time_s * 1e9
            E_tot = E_dyn + E_leak

            # energy envelope: same θ weighting and policy choices influence time; dynamic stays policy-aware via chosen medium in bounds.py
            E_leak_best  = leak_mW * time_best  * 1e9
            E_leak_worst = leak_mW * time_worst * 1e9
            Etot_best, Etot_worst = E_dyn + E_leak_best, E_dyn + E_leak_worst

            all_time[pw].append(time_s)
            all_time_lo[pw].append(max(0.0, time_s - time_best))
            all_time_hi[pw].append(max(0.0, time_worst - time_s))
            all_Edyn[pw].append(E_dyn)
            all_Etot[pw].append(E_tot)
            all_Etot_lo[pw].append(max(0.0, E_tot - Etot_best))
            all_Etot_hi[pw].append(max(0.0, Etot_worst - E_tot))

    # optional sort: by name or by predicted time at reference pi_way
    order = list(range(len(benches)))
    if args.sort_by == "name":
        order = np.argsort(benches)
    elif args.sort_by == "pred":
        order = np.argsort(all_time[piw_ref])
    
    # --- MODIFIED: Use full names for sorting, then create short names for labels ---
    benches = [benches[i] for i in order]
    short_benches = [SHORT.get(b, b) for b in benches] # Use abbreviations

    # figure
    # --- MODIFIED: Changed from 3,2 to 2,2 and adjusted figsize ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    
    # ----------------- top-left: Execution time — raw
    ax = axs[0,0]
    x = np.arange(len(benches))
    W = 0.8; bw = W / K
    for k,pw in enumerate(args.pi_ways):
        y = [all_time[pw][i] for i in order]
        lo = [all_time_lo[pw][i] for i in order]
        hi = [all_time_hi[pw][i] for i in order]
        ax.bar(x + (k - (K-1)/2)*bw, y, width=bw, color=colors[k%len(colors)], label=f"pw={pw:.2f}")
        # --- ADDED as requested ---
        ax.errorbar(x + (k - (K-1)/2)*bw, y, yerr=[lo, hi], fmt='none', ecolor='k', capsize=3, linewidth=1)
    ax.set_title("Execution time — raw"); ax.set_ylabel("seconds")
    # --- MODIFIED: Use short_benches for labels ---
    ax.set_xticks(x); ax.set_xticklabels(short_benches, rotation=40, ha="right"); ax.grid(axis='y', alpha=0.25)
    ax.legend(ncols=min(5,K), fontsize=8)

    # ----------------- top-right: Execution time — × baseline
    ax = axs[0,1]
    for k,pw in enumerate(args.pi_ways):
        y = []
        lo = []
        hi = []
        for i,b in enumerate(benches):
            base_val = ref_time[b]
            cur = all_time[pw][order[i]] / base_val if base_val>0 else 1.0
            y.append(cur)
            # envelope normalized likewise
            lo.append((all_time_lo[pw][order[i]] / base_val) if base_val>0 else 0.0)
            hi.append((all_time_hi[pw][order[i]] / base_val) if base_val>0 else 0.0)
        ax.bar(x + (k - (K-1)/2)*bw, y, width=bw, color=colors[k%len(colors)])
        # draw whiskers centered on bar top (not stacked)
        ax.errorbar(x + (k - (K-1)/2)*bw, y, yerr=[lo, hi], fmt='none', ecolor='k', capsize=3)
    ax.axhline(1.0, color='tab:blue', alpha=0.3, linestyle='--')
    ax.set_title("Execution time — × baseline")
    # --- MODIFIED: Use short_benches for labels ---
    ax.set_xticks(x); ax.set_xticklabels(short_benches, rotation=40, ha="right"); ax.grid(axis='y', alpha=0.25)

    # ----------------- middle-left: Total LLC energy — raw (stacked Dyn + Leak)
    # --- MODIFIED: Now axs[1,0] ---
    ax = axs[1,0]
    for k,pw in enumerate(args.pi_ways):
        Etot = [uJ(all_Etot[pw][order[i]]) for i in range(len(benches))]
        Edyn = [uJ(all_Edyn[pw][order[i]]) for i in range(len(benches))]
        Eleak = (np.array(Etot) - np.array(Edyn)).tolist()
        xoff = x + (k - (K-1)/2)*bw
        ax.bar(xoff, Edyn, width=bw, color=colors[k%len(colors)], label=(f"pw={pw:.2f}" if k==0 else None))
        ax.bar(xoff, Eleak, bottom=Edyn, width=bw, color='none', edgecolor=colors[k%len(colors)],
               hatch='///', label=( "Leakage" if k==0 else None))
        # --- ADDED as requested ---
        # whiskers at total (dynamic+leak) bar tops, using pJ→µJ conversion for deltas too
        Etot_lo = [uJ(all_Etot_lo[pw][order[i]]) for i in range(len(benches))]
        Etot_hi = [uJ(all_Etot_hi[pw][order[i]]) for i in range(len(benches))]
        ax.errorbar(xoff, Etot, yerr=[Etot_lo, Etot_hi], fmt='none', ecolor='k', capsize=3, linewidth=1)
    ax.set_title("Total LLC energy — raw"); ax.set_ylabel("µJ")
    # --- MODFIED: Use short_benches for labels ---
    ax.set_xticks(x); ax.set_xticklabels(short_benches, rotation=40, ha="right"); ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=8)

    # ----------------- middle-right: Total LLC energy — × baseline (with whiskers)
    # --- MODIFIED: Now axs[1,1] ---
    ax = axs[1,1]
    for k,pw in enumerate(args.pi_ways):
        y = []
        lo = []
        hi = []
        for i,b in enumerate(benches):
            base_val = ref_E_tot[b]
            cur = all_Etot[pw][order[i]] / base_val if base_val>0 else 1.0
            y.append(cur)
            lo.append((all_Etot_lo[pw][order[i]] / base_val) if base_val>0 else 0.0)
            hi.append((all_Etot_hi[pw][order[i]] / base_val) if base_val>0 else 0.0)
        ax.bar(x + (k - (K-1)/2)*bw, y, width=bw, color=colors[k%len(colors)])
        ax.errorbar(x + (k - (K-1)/2)*bw, y, yerr=[lo, hi], fmt='none', ecolor='k', capsize=3)
    ax.axhline(1.0, color='tab:blue', alpha=0.3, linestyle='--')
    ax.set_title("Total LLC energy — × baseline")
    # --- MODIFIED: Use short_benches for labels ---
    ax.set_xticks(x); ax.set_xticklabels(short_benches, rotation=40, ha="right"); ax.grid(axis='y', alpha=0.25)

    # --- REMOVED: bottom-left plot (axs[2,0]) ---

    # --- REMOVED: bottom-right plot (axs[2,1]) ---

    # ---- save
    outp = os.path.join(args.out_dir, "sweep_dashboard.png")
    fig.savefig(outp, dpi=170)
    plt.close(fig)
    print(f"[plot_sweep] Saved {outp}")

if __name__ == "__main__":
    main()
