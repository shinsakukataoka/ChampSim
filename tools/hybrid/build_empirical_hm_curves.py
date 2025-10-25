#!/usr/bin/env python3
import os, sys, json, glob
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import joblib

CHAR_ROOT = "results/characterization"
EVAL_ROOT = "results/eval"
OUT_PATH  = "results/surrogate/hm_curves.joblib"

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def compute_mid_fractions(df):
    mids = 0.5*(df['start_cycle'].to_numpy() + df['end_cycle'].to_numpy())
    total = float(df['end_cycle'].max() - df['start_cycle'].min())
    return mids / (total if total>0 else 1.0)

def load_rep_fracs(char_dir):
    reps_json = os.path.join(char_dir, "LLC.representatives.json")
    feats_csv = os.path.join(char_dir, "LLC.window_features.csv")
    if not (os.path.exists(reps_json) and os.path.exists(feats_csv)):
        return None, None
    reps = json.load(open(reps_json)).get("representatives", [])
    feats= pd.read_csv(feats_csv)
    if feats.empty or not reps: return None, None
    # fraction by window mid-point within characterization run
    fracs_all = compute_mid_fractions(feats)
    id2frac = dict(zip(feats['window_id'].to_numpy(), fracs_all))
    rep_fracs = [id2frac[w] for w in reps if w in id2frac]
    return rep_fracs, reps


def select_by_fraction(new_win_csv, rep_fracs):
    # robust CSV read: skip malformed lines
    df = pd.read_csv(new_win_csv, engine="python", on_bad_lines="skip")
    if df.empty or not rep_fracs:
        return df.iloc[0:0].copy()
    fracs = compute_mid_fractions(df)
    idx = [int(np.argmin(np.abs(fracs - f))) for f in rep_fracs]
    return df.iloc[idx].copy()


def collect_points_for_bench(bench):
    """Return dict: pi_miss -> list of (pi_way, mram_hit_frac) from all tags."""
    out = {}
    bench_dir = os.path.join(EVAL_ROOT, bench)
    char_dir  = os.path.join(CHAR_ROOT, bench)
    rep_fracs, _ = load_rep_fracs(char_dir)
    if rep_fracs is None:
        return out

    for tag_dir in glob.glob(os.path.join(bench_dir, "*")):
        mpath = os.path.join(tag_dir, "metrics.json")
        csvp  = os.path.join(tag_dir, "LLC.llc.win.csv")
        if not (os.path.exists(mpath) and os.path.exists(csvp)): continue
        meta = json.load(open(mpath))
        pw   = float(meta.get("pi_way", 0.0))
        pm   = float(meta.get("pi_miss", 0.5))

        df_rep = select_by_fraction(csvp, rep_fracs)
        if df_rep.empty: continue

        Hm = (df_rep['hit_mram_rd'] + df_rep['hit_mram_wr']).sum()
        Hs = (df_rep['hit_sram_rd'] + df_rep['hit_sram_wr']).sum()
        Ht = Hm + Hs
        if Ht <= 0: continue

        frac = Hm / Ht
        out.setdefault(pm, []).append((pw, frac))
    return out

def fit_isotonic(xs, ys):
    # sort by x and dedup
    idx = np.argsort(xs)
    xs  = np.asarray(xs)[idx]
    ys  = np.asarray(ys)[idx]
    # If all x equal or too few points, return a constant “curve”
    if len(np.unique(xs)) < 2:
        return {"x": xs.tolist(), "y": [float(np.clip(np.mean(ys), 0, 1))]*len(xs)}
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    yfit = iso.fit_transform(xs, ys)
    return {"x": xs.tolist(), "y": yfit.tolist()}

def main():
    curves = {}  # curves[bench][pm_str] = {"x":[...], "y":[...]}
    ensure_dir(os.path.dirname(OUT_PATH))
    # iterate benches that have characterization
    for feat_path in glob.glob(os.path.join(CHAR_ROOT, "*", "LLC.window_features.csv")):
        bench = feat_path.split(os.sep)[-2]
        pts = collect_points_for_bench(bench)
        if not pts: 
            print(f"[hm] no eval points for {bench}")
            continue
        curves[bench] = {}
        for pm, lst in pts.items():
            if not lst: continue
            xs = [a for a,_ in lst]
            ys = [b for _,b in lst]
            model = fit_isotonic(xs, ys)
            curves[bench][f"{pm:.4f}"] = model
            # also aggregate across all pm: we’ll build one later if needed
        # add a pooled curve across all pi_miss as fallback
        all_x = []; all_y=[]
        for pm,lst in pts.items():
            for (a,b) in lst: all_x.append(a); all_y.append(b)
        if len(all_x)>=1:
            curves[bench]["__pooled__"] = fit_isotonic(all_x, all_y)

    joblib.dump(curves, OUT_PATH)
    print(f"[hm] wrote {OUT_PATH} for {len(curves)} benches")

if __name__ == "__main__":
    main()

