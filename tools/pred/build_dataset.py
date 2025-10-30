#!/usr/bin/env python3
import argparse, os, json, glob
import pandas as pd
import numpy as np

def load_configs(cfg_csv):
    df = pd.read_csv(cfg_csv)
    return df


def energy_from_counts(df, etab, clk_ns=0.25):
    # counts to pJ; leakage pJ = P_mW * cycles * clk_ns
    e = (
        df["hit_sram_rd"] * etab["sram"]["read_hit_pJ"] +
        df["hit_sram_wr"] * etab["sram"]["write_hit_pJ"] +
        df["hit_mram_rd"] * etab["mram"]["read_hit_pJ"] +
        df["hit_mram_wr"] * etab["mram"]["write_hit_pJ"] +
        (df["miss_rd"]+df["miss_wr"]) * etab["miss_path"]["per_miss_pJ"]
    )
    leak_mW = etab["sram"]["leak_mW"] + etab["mram"]["leak_mW"]
    e += leak_mW * df["cycles_win"] * clk_ns
    return e


def load_windows(win_csv):
    df = pd.read_csv(win_csv)
    # expected columns from your code:
    # window_id,start_cycle,end_cycle,start_inst,end_inst,
    # hit_sram_rd,hit_sram_wr,hit_mram_rd,hit_mram_wr,miss_rd,miss_wr,
    # sum_lat_miss,cover_time_miss,mlp_miss,sum_inflight_hits,hit_cover_cycles,mlp_hit
    need = ["window_id","start_cycle","end_cycle",
            "hit_sram_rd","hit_sram_wr","hit_mram_rd","hit_mram_wr","miss_rd","miss_wr",
            "sum_lat_miss","cover_time_miss","mlp_miss","mlp_hit"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"Missing {c} in {win_csv}")
    # add deriveds
    df["cycles_win"] = df["end_cycle"] - df["start_cycle"]
    df["hits"] = df["hit_sram_rd"]+df["hit_sram_wr"]+df["hit_mram_rd"]+df["hit_mram_wr"]
    df["miss"] = df["miss_rd"] + df["miss_wr"]
    df["A_tot"] = df["hits"] + df["miss"]
    # avoid invalid mlp denom
    df["mlp_hit"]  = df["mlp_hit"].fillna(1.0).clip(lower=1.0)
    df["mlp_miss"] = df["mlp_miss"].fillna(1.0).clip(lower=1.0)
    # per-miss raw latency and effective baseline miss time
    df["T_miss_base_raw"] = np.where(df["miss"]>0, df["sum_lat_miss"]/df["miss"], 0.0)
    df["T_miss_base"]     = df["T_miss_base_raw"] / df["mlp_miss"].clip(lower=1.0)
    return df

def build(basedir, cfg_csv, out_root):
    configs = load_configs(cfg_csv)
    benches = configs["bench"].unique()
    # baseline latencies we used in the baseline rows
    BASE_tS, BASE_tMr, BASE_tMw = 16, 28, 60

    for bench in benches:
        bdir = os.path.join(out_root, "datasets", bench)
        os.makedirs(bdir, exist_ok=True)
        # baseline row
        base_row = configs[(configs["bench"]==bench) & (configs["is_baseline"]==1)].iloc[0]
        base_dir = os.path.join(out_root, "sims", bench, f"cfg_{base_row['config_id']}")
        base_win = os.path.join(base_dir, "results", "LLC.llc.win.csv")
        dfb = load_windows(base_win)
        # baseline partitions
        Pmiss = np.where(dfb["A_tot"]>0, dfb["miss"]/dfb["A_tot"], 0.0)
        Psr_rd = np.where(dfb["A_tot"]>0, dfb["hit_sram_rd"]/dfb["A_tot"], 0.0)
        Psr_wr = np.where(dfb["A_tot"]>0, dfb["hit_sram_wr"]/dfb["A_tot"], 0.0)
        Pmr_rd = np.where(dfb["A_tot"]>0, dfb["hit_mram_rd"]/dfb["A_tot"], 0.0)
        Pmr_wr = np.where(dfb["A_tot"]>0, dfb["hit_mram_wr"]/dfb["A_tot"], 0.0)
        dfb["P_miss"]    = Pmiss
        dfb["P_sram_rd"] = Psr_rd
        dfb["P_sram_wr"] = Psr_wr
        dfb["P_mram_rd"] = Pmr_rd
        dfb["P_mram_wr"] = Pmr_wr
        # baseline AMAT with baseline latencies and effective T_miss_base
        dfb["AMAT_base"] = (Psr_rd+Psr_wr)*BASE_tS + Pmr_rd*BASE_tMr + Pmr_wr*BASE_tMw + Pmiss*dfb["T_miss_base"]
        dfb["cycles_base"] = dfb["cycles_win"]  # we use window cycles from baseline
        # save
        dfb.to_csv(os.path.join(bdir, "baseline_windows.csv"), index=False)

        # training rows (each non-baseline config)
        trows = []
        roi_rows = []
        for _, cfg in configs[(configs["bench"]==bench) & (configs["is_baseline"]==0)].iterrows():
            run_dir = os.path.join(out_root, "sims", bench, f"cfg_{cfg['config_id']}")
            run_win = os.path.join(run_dir, "results", "LLC.llc.win.csv")
            if not os.path.exists(run_win):
                continue
            dfr_full = load_windows(run_win)
            dfr_full["energy_true_pJ"] = energy_from_counts(dfr_full, json.load(open("/home/skataoka26/ChampSim/model/energy_table.json")))
            dfr = dfr_full[["window_id","cycles_win","energy_true_pJ"]].rename(columns={"cycles_win":"cycles_true"})
            merged = dfb.merge(dfr, on="window_id", how="inner")
            merged["config_id"] = int(cfg["config_id"])
            merged["tS"] = int(cfg["tS"]); merged["tMr"] = int(cfg["tMr"]); merged["tMw"] = int(cfg["tMw"])
            merged["pi_miss"] = float(cfg["pi_miss"]); merged["pi_way"] = float(cfg["pi_way"])
            # store also per-config ROI true cycles (from stats.json if available)
            stats_path = os.path.join(run_dir, "stats.json")
            roi_cycles = None; roi_instr = None
            if os.path.exists(stats_path):
                try:
                    # old:
                    # with open(stats_path) as fp: jj=json.load(fp)
                    # roi_instr = jj["roi"]["cores"][0]["instructions"]
                    # roi_cycles = jj["roi"]["cores"][0]["cycles"]
                    with open(stats_path) as fp:
                        jj = json.load(fp)
                    # stats.json may be a list of phases; pick the Simulation phase if present
                    if isinstance(jj, list):
                        sim = next((x for x in jj if isinstance(x, dict) and x.get("name")=="Simulation"), jj[0] if jj else {})
                    else:
                        sim = jj
                    roi = sim.get("roi", {})
                    cores = roi.get("cores", [])
                    if isinstance(cores, list) and cores:
                        roi_instr = cores[0].get("instructions")
                        roi_cycles = cores[0].get("cycles")
                    else:
                        roi_instr = roi_cycles = None
                except Exception:
                    pass
            merged["roi_cycles_true"] = roi_cycles if roi_cycles is not None else np.nan
            merged["roi_instr"]       = roi_instr  if roi_instr  is not None else np.nan
            trows.append(merged)

        if trows:
            train_df = pd.concat(trows, ignore_index=True)
            train_df.to_csv(os.path.join(bdir, "training_windows.csv"), index=False)
            print(f"[{bench}] baseline+training: {len(dfb)} windows baseline, {len(train_df)} rows training")
        else:
            print(f"[{bench}] no training rows found (did jobs finish?)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/home/skataoka26/ChampSim/out/configs.csv")
    ap.add_argument("--out-root", default="/home/skataoka26/ChampSim/out")
    args = ap.parse_args()
    build("/home/skataoka26/ChampSim", args.cfg, args.out_root)
