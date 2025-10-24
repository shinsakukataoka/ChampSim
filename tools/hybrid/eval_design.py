#!/usr/bin/env python3
import os, sys, json, time, subprocess, re
import numpy as np
import pandas as pd

# Resolve ChampSim root and absolute binary path
THIS = os.path.abspath(os.path.dirname(__file__))                # .../ChampSim/tools/hybrid
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))           # .../ChampSim
BIN  = os.path.join(ROOT, "bin", "champsim")

TRACES_ROOT = "/home/skataoka26/traces/speccpu"
WARM = 2_000_000
SIM  = 5_000_000


# energy placeholders (optional)
L3_MB         = 8.0
LEAK_SRAM_WPM = 5.0e-3
LEAK_MRAM_WPM = 0.5e-3
E_RD_SRAM = 0.3e-9;  E_WR_SRAM = 0.4e-9
E_RD_MRAM = 0.6e-9;  E_WR_MRAM = 0.8e-9
E_MISS_PATH = 3.0e-9

def run_cmd(cmd, log_path, cwd=None):
    with open(log_path, "w") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    with open(log_path, "r") as f:
        out = f.read()
    return p.returncode, out

def parse_dram_latency_and_cycles(stdout_text):
    tDR = None; cycles = None
    m = re.search(r'LLC AVERAGE MISS LATENCY:\s+([\d\.]+)\s+cycles', stdout_text)
    if m: tDR = float(m.group(1))
    m2 = re.search(r'CPU 0 cumulative IPC:\s+[\d\.]+\s+instructions:\s+\d+\s+cycles:\s+(\d+)', stdout_text)
    if m2: cycles = int(m2.group(1))
    return tDR, cycles

def compute_mid_fractions(win_df):
    mids = 0.5*(win_df['start_cycle'].to_numpy() + win_df['end_cycle'].to_numpy())
    total = float((win_df['end_cycle'].max() - win_df['start_cycle'].min()))
    return (mids / (total if total > 0 else 1.0))

def load_rep_fractions(char_dir):
    reps_json = os.path.join(char_dir, "LLC.representatives.json")
    feats_csv = os.path.join(char_dir, "LLC.window_features.csv")
    reps = json.load(open(reps_json))["representatives"]
    feats = pd.read_csv(feats_csv)
    frac_all = compute_mid_fractions(feats)
    id_to_frac = dict(zip(feats['window_id'].to_numpy(), frac_all))
    rep_fracs = [id_to_frac[w] for w in reps if w in id_to_frac]
    return rep_fracs, reps

def select_by_fraction(new_win_csv, rep_fracs):
    df = pd.read_csv(new_win_csv)
    if df.empty or len(rep_fracs)==0:
        return df.iloc[0:0].copy(), []
    fracs = compute_mid_fractions(df)
    chosen_idx = [int(np.argmin(np.abs(fracs - f))) for f in rep_fracs]
    chosen = df.iloc[chosen_idx].copy()
    chosen['__selected_by_frac__'] = rep_fracs
    return chosen, chosen['window_id'].to_list()

def compute_metrics(df_reps, t_s, t_mr, t_mw, t_dr, l3_mb, pi_way):
    if df_reps.empty:
        return {"stall_cycles":0.0,"window_cycles_sum":0.0,"stall_pct":0.0,
                "E_dynamic_J":0.0,"E_leakage_J":0.0,"E_total_J":0.0}
    Hs_rd = df_reps['hit_sram_rd'].to_numpy()
    Hs_wr = df_reps['hit_sram_wr'].to_numpy()
    Hm_rd = df_reps['hit_mram_rd'].to_numpy()
    Hm_wr = df_reps['hit_mram_wr'].to_numpy()
    M     = (df_reps['miss_rd'] + df_reps['miss_wr']).to_numpy()
    mlp_hit  = df_reps['mlp_hit' ].clip(lower=1.0).to_numpy()
    mlp_miss = df_reps['mlp_miss'].clip(lower=1.0).to_numpy()
    d_rd  = t_mr - t_s
    d_wr  = t_mw - t_s
    d_mis = t_dr - t_s
    stall_w = (Hm_rd*d_rd + Hm_wr*d_wr)/mlp_hit + (M*d_mis)/mlp_miss
    stall_cycles = float(stall_w.sum())
    win_cycles   = float(((df_reps['end_cycle'] - df_reps['start_cycle']).to_numpy()).sum())
    stall_pct    = stall_cycles / max(win_cycles, 1.0)
    E_dyn = (Hs_rd*E_RD_SRAM + Hs_wr*E_WR_SRAM + Hm_rd*E_RD_MRAM + Hm_wr*E_WR_MRAM + M*E_MISS_PATH).sum()
    T_sec_approx = win_cycles / 1e9
    C_S = (1.0 - pi_way)*l3_mb; C_M = pi_way*l3_mb
    E_leak = (LEAK_SRAM_WPM*C_S + LEAK_MRAM_WPM*C_M) * T_sec_approx
    return {"stall_cycles":stall_cycles,"window_cycles_sum":win_cycles,"stall_pct":stall_pct,
            "E_dynamic_J":E_dyn,"E_leakage_J":E_leak,"E_total_J":E_dyn+E_leak}

def main():
    if len(sys.argv) < 8:
        print("Usage: eval_design.py <bench_name> <pi_way> <pi_miss> <t_sram_hit> <t_mram_rd> <t_mram_wr> <run_tag> [--l3-mb X]")
        sys.exit(1)
    bench, pi_way, pi_miss = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    tS, tMr, tMw, tag = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
    l3_mb = L3_MB
    if len(sys.argv) > 8 and sys.argv[8] == "--l3-mb":
        l3_mb = float(sys.argv[9])

    trace_path = os.path.join(TRACES_ROOT, f"{bench}.champsimtrace.xz")
    char_dir   = os.path.join("results", "characterization", bench)
    reps_json  = os.path.join(char_dir, "LLC.representatives.json")
    feats_csv  = os.path.join(char_dir, "LLC.window_features.csv")
    if not os.path.exists(reps_json) or not os.path.exists(feats_csv):
        raise FileNotFoundError(f"Missing reps or features in {char_dir}")

    # load target rep fractions from characterization
    rep_fracs, rep_ids = load_rep_fractions(char_dir)

    out_dir = os.path.join("results", "eval", bench, tag)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")

    # create a unique working dir for this run
    tmp_root = os.path.join("results","tmp", bench)
    os.makedirs(tmp_root, exist_ok=True)
    uniq = f"{tag}_{int(time.time())}_{os.getpid()}"
    run_cwd = os.path.join(tmp_root, uniq)
    os.makedirs(run_cwd, exist_ok=True)

    # run champsim in the unique cwd so it writes to <cwd>/results/LLC.llc.win.csv
    cmd = [
        BIN,
        "--warmup-instructions", str(WARM),
        "--simulation-instructions", str(SIM),
        "--hybrid-llc",
        "--pi-way", str(pi_way),
        "--pi-miss", str(pi_miss),
        "--t-sram-hit", str(int(tS)),
        "--t-mram-rd",  str(int(tMr)),
        "--t-mram-wr",  str(int(tMw)),
        trace_path
    ]

    rc, logtxt = run_cmd(cmd, log_path, cwd=run_cwd)
    if rc != 0:
        print(f"ERROR: champsim returned {rc}; see {log_path}")
        sys.exit(rc)

    tDR, roi_cycles = parse_dram_latency_and_cycles(logtxt)
    if tDR is None:
        tDR = 200.0

    src_win = os.path.join(run_cwd, "results", "LLC.llc.win.csv")
    if not os.path.exists(src_win):
        print("ERROR: results/LLC.llc.win.csv not produced")
        sys.exit(2)
    dst_win = os.path.join(out_dir, "LLC.llc.win.csv")
    os.replace(src_win, dst_win)

    df_reps, mapped_ids = select_by_fraction(dst_win, rep_fracs)
    metrics = compute_metrics(df_reps, tS, tMr, tMw, tDR, l3_mb, pi_way)
    metrics.update({
        "bench": bench, "tag": tag,
        "pi_way": pi_way, "pi_miss": pi_miss,
        "t_sram_hit": tS, "t_mram_rd": tMr, "t_mram_wr": tMw, "t_dram": tDR,
        "l3_mb": l3_mb, "roi_cycles": roi_cycles,
        "rep_windows_char": rep_ids,
        "rep_windows_mapped": mapped_ids,
        "rep_rows": int(len(df_reps))
    })
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    print(f"[eval_design] wrote {os.path.join(out_dir,'metrics.json')}")
    print(f"  stall%={metrics['stall_pct']*100:.2f}  E_total={metrics['E_total_J']:.3e} J  (rep_rows={metrics['rep_rows']})")

if __name__ == "__main__":
    main()

