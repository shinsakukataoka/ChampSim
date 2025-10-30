#!/usr/bin/env python3
import argparse, os, json, itertools
import numpy as np
import pandas as pd
import joblib

from physics import physics_predict_row, DEFAULTS as PHY_CONSTS

def energy_phys_row(brow, aux, cycles_phys, etab, clk_ns=0.25):
    # Fallback to baseline partitions if aux lacks them
    A = float(brow["A_tot"])
    Psr_rd = float(aux.get("P_sram_rd", brow["P_sram_rd"]))
    Psr_wr = float(aux.get("P_sram_wr", brow["P_sram_wr"]))
    Pmr_rd = float(aux.get("P_mram_rd", brow["P_mram_rd"]))
    Pmr_wr = float(aux.get("P_mram_wr", brow["P_mram_wr"]))
    Pmiss  = float(brow["P_miss"])
    e = (
        A*Psr_rd*etab["sram"]["read_hit_pJ"] +
        A*Psr_wr*etab["sram"]["write_hit_pJ"] +
        A*Pmr_rd*etab["mram"]["read_hit_pJ"] +
        A*Pmr_wr*etab["mram"]["write_hit_pJ"] +
        A*Pmiss *etab["miss_path"]["per_miss_pJ"]
    )
    leak_mW = etab["sram"]["leak_mW"] + etab["mram"]["leak_mW"]
    e += leak_mW * cycles_phys * clk_ns
    return e

def build_X(brow, knobs, aux):
    return np.array([
        float(brow["A_tot"]),
        float(brow["P_miss"]), float(brow["P_sram_rd"]), float(brow["P_mram_rd"]),
        float(brow["mlp_hit"]), float(brow["mlp_miss"]),
        float(brow["T_miss_base"]), float(brow["AMAT_base"]),
        float(knobs["tS"]), float(knobs["tMr"]), float(knobs["tMw"]),
        float(knobs["pi_miss"]), float(knobs["pi_way"]),
        float(aux["AMAT"]), float(aux["Tmiss"])
    ], dtype=float).reshape(1,-1)

def sweep_for_bench(bench, datasets_root, model_dir, out_dir,
                    pi_step=0.1, tS_vals=(12,16,24,32,40), tMr_vals=(24,28,36,48,64), tMw_vals=(50,60,80,100,120),
                    sim_instrs=None):
    base_csv = os.path.join(datasets_root, bench, "baseline_windows.csv")
    dfb = pd.read_csv(base_csv)

    # models and consts
    model = joblib.load(os.path.join(model_dir, f"{bench}.pkl"))
    consts = PHY_CONSTS.copy()
    const_path = os.path.join(model_dir, f"{bench}_consts.json")
    if os.path.exists(const_path):
        consts.update(json.load(open(const_path)))
    etab = json.load(open("/home/skataoka26/ChampSim/model/energy_table.json"))
    modelE_path = os.path.join(model_dir, f"{bench}_energy.pkl")
    modelE = joblib.load(modelE_path) if os.path.exists(modelE_path) else None

    # grid
    pis = np.round(np.arange(0.0, 1.0+1e-9, pi_step), 3)
    grid = list(itertools.product(pis, pis, tS_vals, tMr_vals, tMw_vals))

    rows = []
    for pi_miss, pi_way, tS, tMr, tMw in grid:
        # enforce order tS <= tMr <= tMw (skip invalid combos)
        if not (tS <= tMr <= tMw):
            continue
        knobs = dict(pi_miss=float(pi_miss), pi_way=float(pi_way), tS=int(tS), tMr=int(tMr), tMw=int(tMw))
        cyc_sum, en_sum = 0.0, 0.0
        for _, brow in dfb.iterrows():
            cycles_phys, aux = physics_predict_row(brow, knobs, consts)
            X = build_X(brow, knobs, aux)
            delta_cyc = float(model.predict(X)[0])
            cycles = cycles_phys + delta_cyc
            cyc_sum += cycles

            E_phys = energy_phys_row(brow, aux, cycles_phys, etab)
            if modelE is not None:
                E_phys += float(modelE.predict(X)[0])
            en_sum += E_phys
        ipc = (sim_instrs / cyc_sum) if sim_instrs else None
        rows.append(dict(bench=bench, pi_miss=pi_miss, pi_way=pi_way, tS=tS, tMr=tMr, tMw=tMw,
                         cycles_pred=cyc_sum, energy_pred_pJ=en_sum, ipc_pred=ipc))
    out_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(os.path.join(out_dir, f"{bench}_dse.csv"), index=False)
    return out_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="/home/skataoka26/ChampSim/out/datasets")
    ap.add_argument("--model-dir", default="/home/skataoka26/ChampSim/model")
    ap.add_argument("--out-root",  default="/home/skataoka26/ChampSim/out")
    ap.add_argument("--benches", nargs="*", help="Bench names under datasets root (default: all found)")
    ap.add_argument("--pi-step", type=float, default=0.1)
    ap.add_argument("--tS", nargs="*", type=int, default=[12,16,24,32,40])
    ap.add_argument("--tMr", nargs="*", type=int, default=[24,28,36,48,64])
    ap.add_argument("--tMw", nargs="*", type=int, default=[50,60,80,100,120])
    ap.add_argument("--sim", type=int, default=20_000_000, help="Total simulated instructions (for IPC)")
    args = ap.parse_args()

    if args.benches:
        benches = args.benches
    else:
        benches = sorted([d for d in os.listdir(args.datasets_root) if os.path.isdir(os.path.join(args.datasets_root, d))])

    combined = []
    out_dir = os.path.join(args.out_root, "dse")
    for b in benches:
        df = sweep_for_bench(b, args.datasets_root, args.model_dir, out_dir,
                             pi_step=args.pi_step, tS_vals=args.tS, tMr_vals=args.tMr, tMw_vals=args.tMw, sim_instrs=args.sim)
        combined.append(df)
        print(f"[DSE] {b}: {len(df)} points")
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(os.path.join(out_dir, "combined_dse.csv"), index=False)

if __name__ == "__main__":
    main()
