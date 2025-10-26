#!/usr/bin/env python3
import os, sys, json, time, subprocess, re, shutil
import numpy as np
import pandas as pd

# ---------- paths / defaults ----------
THIS = os.path.abspath(os.path.dirname(__file__))                # .../ChampSim/tools/hybrid
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))           # .../ChampSim
BIN  = os.environ.get("CHAMPSIM_BIN", os.path.join(ROOT, "bin", "champsim"))

TRACES_ROOT = os.environ.get("TRACES_ROOT", os.path.expanduser("~/traces/speccpu"))
WARM = 2_000_000
SIM  = 5_000_000

# Stall baseline for cross-run comparability (fixed SRAM-hit reference)
STALL_BASE_TS = 16.0

# ---------- energy placeholders (optional) ----------
L3_MB         = 8.0
LEAK_SRAM_WPM = 5.0e-3
LEAK_MRAM_WPM = 0.5e-3
E_RD_SRAM = 0.3e-9;  E_WR_SRAM = 0.4e-9
E_RD_MRAM = 0.6e-9;  E_WR_MRAM = 0.8e-9
E_MISS_PATH = 3.0e-9

def find_char_dir(bench, l3_mb):
    cand = os.path.join("results", f"characterization_L3_{int(l3_mb)}", bench)
    if os.path.exists(os.path.join(cand, "LLC.representatives.json")):
        return cand
    return os.path.join("results", "characterization", bench)

def run_cmd(cmd, log_path, cwd=None):
    t0 = time.time()
    with open(log_path, "w") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    with open(log_path, "r") as f:
        out = f.read()
    return p.returncode, out, (time.time() - t0)

def parse_dram_latency_and_cycles(stdout_text):
    tDR = None; cycles = None
    m = re.search(r'LLC AVERAGE MISS LATENCY:\s+([\d\.]+)\s+cycles', stdout_text)
    if m: tDR = float(m.group(1))
    # be robust to thousand-separators
    m2 = re.search(r'CPU 0 cumulative IPC:\s+[\d\.]+\s+instructions:\s+[0-9,]+\s+cycles:\s+([0-9,]+)', stdout_text)
    if m2: cycles = int(m2.group(1).replace(',', ''))
    return tDR, cycles

def compute_mid_fractions(win_df):
    mids = 0.5*(win_df['start_cycle'].to_numpy() + win_df['end_cycle'].to_numpy())
    total = float((win_df['end_cycle'].max() - win_df['start_cycle'].min()))
    return (mids / (total if total > 0 else 1.0))

def load_rep_fractions(char_dir):
    reps_json = os.path.join(char_dir, "LLC.representatives.json")
    feats_csv = os.path.join(char_dir, "LLC.window_features.csv")
    reps = json.load(open(reps_json))["representatives"]
    feats = pd.read_csv(feats_csv, engine="python", on_bad_lines="skip")
    frac_all = compute_mid_fractions(feats)
    id_to_frac = dict(zip(feats['window_id'].to_numpy(), frac_all))
    rep_fracs = [id_to_frac[w] for w in reps if w in id_to_frac]
    return rep_fracs, reps

def select_by_fraction(new_win_csv, rep_fracs):
    """Map each representative fraction to a UNIQUE nearest window in the new run."""
    df = pd.read_csv(new_win_csv, engine="python", on_bad_lines="skip")
    if df.empty or len(rep_fracs)==0:
        return df.iloc[0:0].copy(), []
    # ensure window_id int
    if 'window_id' in df.columns:
        df['window_id'] = pd.to_numeric(df['window_id'], errors='coerce').fillna(-1).astype(int)
    fracs = compute_mid_fractions(df)
    used = set(); chosen_idx = []
    for f in rep_fracs:
        order = np.argsort(np.abs(fracs - f))
        pick = next((int(j) for j in order if int(j) not in used), None)
        if pick is not None:
            used.add(pick); chosen_idx.append(pick)
    chosen = df.iloc[chosen_idx].copy()
    chosen['__selected_by_frac__'] = rep_fracs[:len(chosen_idx)]
    return chosen, chosen['window_id'].astype(int).to_list()

def compute_metrics(df_reps, t_s, t_mr, t_mw, t_dr, l3_mb, pi_way, stall_base_ts=STALL_BASE_TS):
    """Return stall anchored to fixed baseline (for fair cross-tS comparison),
       plus raw stall using the run's t_s for reference."""
    if df_reps.empty:
        return {"stall_cycles":0.0,"stall_cycles_raw":0.0,"window_cycles_sum":0.0,
                "stall_pct":0.0,"stall_pct_raw":0.0,"stall_pct_base_ts":stall_base_ts,
                "E_dynamic_J":0.0,"E_leakage_J":0.0,"E_total_J":0.0}

    Hs_rd = df_reps['hit_sram_rd'].to_numpy()
    Hs_wr = df_reps['hit_sram_wr'].to_numpy()
    Hm_rd = df_reps['hit_mram_rd'].to_numpy()
    Hm_wr = df_reps['hit_mram_wr'].to_numpy()
    M     = (df_reps['miss_rd'] + df_reps['miss_wr']).to_numpy()
    mlp_hit  = df_reps['mlp_hit' ].clip(lower=1.0).to_numpy()
    mlp_miss = df_reps['mlp_miss'].clip(lower=1.0).to_numpy()

    # --- baseline-anchored deltas (fair across different t_s) ---
    d_rd_b  = t_mr - stall_base_ts
    d_wr_b  = t_mw - stall_base_ts
    d_mis_b = t_dr - stall_base_ts
    stall_w_b = (Hm_rd*d_rd_b + Hm_wr*d_wr_b)/mlp_hit + (M*d_mis_b)/mlp_miss

    # --- raw deltas vs the run's own t_s (kept for reference) ---
    d_rd_r  = t_mr - t_s
    d_wr_r  = t_mw - t_s
    d_mis_r = t_dr - t_s
    stall_w_r = (Hm_rd*d_rd_r + Hm_wr*d_wr_r)/mlp_hit + (M*d_mis_r)/mlp_miss

    stall_cycles     = float(stall_w_b.sum())
    stall_cycles_raw = float(stall_w_r.sum())
    win_cycles       = float(((df_reps['end_cycle'] - df_reps['start_cycle']).to_numpy()).sum())

    stall_pct     = stall_cycles     / max(win_cycles, 1.0)
    stall_pct_raw = stall_cycles_raw / max(win_cycles, 1.0)

    # energy
    E_dyn = (Hs_rd*E_RD_SRAM + Hs_wr*E_WR_SRAM + Hm_rd*E_RD_MRAM + Hm_wr*E_WR_MRAM + M*E_MISS_PATH).sum()
    T_sec_approx = win_cycles / 1e9  # rough @1GHz
    C_S = (1.0 - pi_way)*l3_mb; C_M = pi_way*l3_mb
    E_leak = (LEAK_SRAM_WPM*C_S + LEAK_MRAM_WPM*C_M) * T_sec_approx

    return {
        "stall_cycles":stall_cycles,
        "stall_cycles_raw":stall_cycles_raw,
        "window_cycles_sum":win_cycles,
        "stall_pct":stall_pct,                 # baseline-anchored (use this in analysis)
        "stall_pct_raw":stall_pct_raw,         # original (vs run's t_s)
        "stall_pct_base_ts":stall_base_ts,
        "E_dynamic_J":E_dyn,"E_leakage_J":E_leak,"E_total_J":E_dyn+E_leak
    }

def stage_llc_config(root, run_cwd, size_mb):
    src = os.path.join(root, "champsim_config.json")
    cfg = json.load(open(src))
    bl   = int(cfg.get("block_size", 64))
    ways = int(cfg.get("LLC", {}).get("ways", 16))
    sets = int((size_mb*1024*1024)//(bl*ways))
    cfg.setdefault("LLC", {})["sets"] = sets
    dst = os.path.join(run_cwd, "champsim_config.json")
    with open(dst, "w") as f:
        json.dump(cfg, f, indent=2)
    return dst

def git_commit_hash(root):
    try:
        return subprocess.check_output(["git","-C", root, "rev-parse","--short","HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def main():
    if len(sys.argv) < 8:
        print("Usage: eval_design.py <bench> <pi_way> <pi_miss> <t_sram_hit> <t_mram_rd> <t_mram_wr> <run_tag> [--l3-mb X]")
        sys.exit(1)
    bench, pi_way, pi_miss = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    tS, tMr, tMw, tag = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
    l3_mb = L3_MB
    if len(sys.argv) > 8 and sys.argv[8] == "--l3-mb":
        l3_mb = float(sys.argv[9])

    trace_path = os.path.join(TRACES_ROOT, f"{bench}.champsimtrace.xz")
    char_dir = find_char_dir(bench, l3_mb)
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

    # stage LLC config
    cfg_path = stage_llc_config(ROOT, run_cwd, l3_mb)
    print(f"[eval_design] staged {cfg_path} (LLC ≈ {l3_mb} MB)")

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

    rc, logtxt, runtime_s = run_cmd(cmd, log_path, cwd=run_cwd)
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

    # snapshot the exact config we used in this run
    try:
        shutil.copyfile(cfg_path, os.path.join(out_dir, "champsim_config.used.json"))
    except Exception as e:
        print("[warn] failed to snapshot config:", e)

    df_reps, mapped_ids = select_by_fraction(dst_win, rep_fracs)
    metrics = compute_metrics(df_reps, tS, tMr, tMw, tDR, l3_mb, pi_way, stall_base_ts=STALL_BASE_TS)
    metrics.update({
        "bench": bench, "tag": tag,
        "pi_way": pi_way, "pi_miss": pi_miss,
        "t_sram_hit": tS, "t_mram_rd": tMr, "t_mram_wr": tMw, "t_dram": tDR,
        "l3_mb": l3_mb, "roi_cycles": roi_cycles,
        "rep_windows_char": rep_ids,
        "rep_windows_mapped": mapped_ids,
        "rep_rows": int(len(df_reps)),
        "rep_expected": int(len(rep_fracs)),
        "runtime_sec": float(runtime_s),
        "git_commit": git_commit_hash(ROOT)
    })
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    print(f"[eval_design] wrote {os.path.join(out_dir,'metrics.json')}")
    print(f"  stall% (base {STALL_BASE_TS:.1f}cy) = {metrics['stall_pct']*100:.2f}"
          f"  E_total={metrics['E_total_J']:.3e} J  (rep_rows={metrics['rep_rows']}/{metrics['rep_expected']})")

if __name__ == "__main__":
    main()

