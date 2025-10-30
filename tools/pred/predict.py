#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd
import joblib
from physics import physics_predict_row, DEFAULTS as PHY_CONSTS
# --- added near top imports ---
import os, json
from bounds import accumulate_bounds
import yaml

def energy_phys_row(brow, aux, cycles_phys, etab, clk_ns=0.25):
    # Fall back to baseline partitions if aux lacks them
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--datasets-root", default="/home/skataoka26/ChampSim/out/datasets")
    ap.add_argument("--model-dir", default="/home/skataoka26/ChampSim/model")
    ap.add_argument("--out", default=None)
    # knobs
    ap.add_argument("--pi-miss", type=float, required=True)
    ap.add_argument("--pi-way",  type=float, required=True)
    ap.add_argument("--tS",      type=int,   required=True)
    ap.add_argument("--tMr",     type=int,   required=True)
    ap.add_argument("--tMw",     type=int,   required=True)
    ap.add_argument("--sim",     type=int,   default=None)  # for IPC conversion
    args = ap.parse_args()

    # load baseline windows
    base_csv = os.path.join(args.datasets_root, args.bench, "baseline_windows.csv")
    dfb = pd.read_csv(base_csv)

    # load residual model
    model_path = os.path.join(args.model_dir, f"{args.bench}.pkl")
    model = joblib.load(model_path)

    # per-benchmark physics consts (if any)
    consts = PHY_CONSTS.copy()
    const_path = os.path.join(args.model_dir, f"{args.bench}_consts.json")
    if os.path.exists(const_path):
        consts.update(json.load(open(const_path)))
    # ALSO load YAML physics defaults if present (so alpha_m/beta_w/kappa/f_crit from your yaml apply)
    yaml_path = "/home/skataoka26/ChampSim/model/physics_consts.yaml"
    if os.path.exists(yaml_path):
        consts.update(yaml.safe_load(open(yaml_path)))

    # energy table + optional energy residual
    etab = json.load(open(os.environ.get("ENERGY_TABLE_PATH","/home/skataoka26/ChampSim/model/energy_table.json")))
    modelE_path = os.path.join(args.model_dir, f"{args.bench}_energy.pkl")
    modelE = joblib.load(modelE_path) if os.path.exists(modelE_path) else None

    knobs = dict(tS=args.tS, tMr=args.tMr, tMw=args.tMw, pi_miss=args.pi_miss, pi_way=args.pi_way)

    cycles_sum = 0.0
    energy_sum = 0.0
    for _, brow in dfb.iterrows():
        cycles_phys, aux = physics_predict_row(brow, knobs, consts)

        # features for residuals
        X = np.array([
            float(brow["A_tot"]),
            float(brow["P_miss"]), float(brow["P_sram_rd"]), float(brow["P_mram_rd"]),
            float(brow["mlp_hit"]), float(brow["mlp_miss"]),
            float(brow["T_miss_base"]), float(brow["AMAT_base"]),
            float(knobs["tS"]), float(knobs["tMr"]), float(knobs["tMw"]),
            float(knobs["pi_miss"]), float(knobs["pi_way"]),
            float(aux["AMAT"]), float(aux["Tmiss"])
        ], dtype=float).reshape(1,-1)

        # cycles
        delta_cyc = float(model.predict(X)[0])
        cycles = cycles_phys + delta_cyc
        cycles_sum += cycles

        # energy (medium-aware dynamic + θ-weighted leakage; keep residual if available)
        A     = float(brow["A_tot"])
        Psr_r = float(aux.get("P_sram_rd", brow["P_sram_rd"]))
        Psr_w = float(aux.get("P_sram_wr", brow["P_sram_wr"]))
        Pmr_r = float(aux.get("P_mram_rd", brow["P_mram_rd"]))
        Pmr_w = float(aux.get("P_mram_wr", brow["P_mram_wr"]))
        Pmis  = float(brow["P_miss"])
        E_dyn = A*( Psr_r*etab["sram"]["read_hit_pJ"] + Psr_w*etab["sram"]["write_hit_pJ"]
                  + Pmr_r*etab["mram"]["read_hit_pJ"] + Pmr_w*etab["mram"]["write_hit_pJ"]
                  + Pmis  *etab["miss_path"]["per_miss_pJ"] )
        theta = float(knobs["pi_miss"]) * 0.0 + float(knobs["pi_way"])  # simple θ ≈ capacity split
        leak_mW = etab["sram"]["leak_mW"]*(1.0-theta) + etab["mram"]["leak_mW"]*theta
        time_s = cycles * 0.25e-9  # CLK_NS = 0.25 ns
        E_tot = E_dyn + leak_mW * time_s * 1e9
        if modelE is not None:
            E_tot += float(modelE.predict(X)[0])
        energy_sum += E_tot

    # --- Start of modified section ---
    bounds = accumulate_bounds(dfb, knobs, consts, etab, clk_ns=0.25)

    result = dict(bench=args.bench, knobs=knobs,
                  cycles_pred=cycles_sum, energy_pred_pJ=energy_sum)
    if args.sim:
        result["ipc_pred"] = float(args.sim) / cycles_sum

    # attach bounds (and IPCs if --sim was given)
    result.update(bounds)
    if args.sim:
        for k in list(bounds.keys()):
            if k.startswith("cycles_"):
                tag = k.replace("cycles_","ipc_")
                result[tag] = float(args.sim) / bounds[k] if bounds[k] > 0 else None

    # write file / print (existing code)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)

    print(json.dumps(result, indent=2))
    # --- End of modified section ---

if __name__ == "__main__":
    main()
