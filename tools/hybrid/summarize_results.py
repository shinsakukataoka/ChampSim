#!/usr/bin/env python3
import os, glob, argparse, numpy as np, pandas as pd

def slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ux = np.unique(x)
    if ux.size < 2: return np.nan
    return float(np.polyfit(x, y, 1)[0])

def load_dataset(cap):
    p = f"results/eval/dataset_L3_{cap}MB.csv"
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_csv(p).dropna(subset=["stall_pct","E_total_J"])

def summarize_policy_bounds(cap, device_root, profile, latency_mode):
    # pick bounds file (prefer per-profile; else current run)
    if profile:
        p = f"results/surrogate/profiles/{profile}/L3_{cap}/policy_bounds.csv"
    else:
        p = f"results/surrogate/L3_{cap}/policy_bounds.csv"
    if not os.path.exists(p):
        return None, None, None
    pb = pd.read_csv(p)
    pv = pb.pivot_table(index=["bench","tag"], columns="bound",
                        values=["stall_pct","E_total_J"]).dropna()
    d_stall = (pv["stall_pct"]["stall_worst"] - pv["stall_pct"]["stall_best"]).median()
    d_energy_uJ = ((pv["E_total_J"]["energy_worst"] - pv["E_total_J"]["energy_best"]) * 1e6).median()
    sens = pv["stall_pct"].groupby(level=0).apply(
        lambda g: (g["stall_worst"] - g["stall_best"]).median()
    ).sort_values(ascending=False)
    top5 = sens.head(5).reset_index().rename(columns={0:"delta_stall"})
    return d_stall, d_energy_uJ, top5

def device_profile_medians(cap, profiles):
    rows=[]
    for p in profiles:
        path = f"results/surrogate/profiles/{p}/L3_{cap}/physics_pred.csv"
        if not os.path.exists(path):  # fallback to current run
            path = f"results/surrogate/L3_{cap}/physics_pred.csv"
            if not os.path.exists(path): continue
            df = pd.read_csv(path).assign(profile=p)
        else:
            df = pd.read_csv(path).assign(profile=p)
        g = df.groupby("profile")[["E_total_J","stall_pct"]].median().reset_index()
        g["E_total_uJ"] = g["E_total_J"] * 1e6
        rows.append(g[["profile","stall_pct","E_total_uJ"]])
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["profile","stall_pct","E_total_uJ"])

def sensitivity(df):
    out = {}

    # pi_way sensitivity at pm=0.50 (all latencies)
    sub = df[np.isclose(df["pi_miss"], 0.50)]
    s_pw = sub.groupby("bench", group_keys=False).apply(
        lambda g: slope(g["pi_way"], g["stall_pct"])
    )
    out["pi_way_slope_median"] = float(np.nanmedian(s_pw.values))

    # tMr / tMw sensitivity at pw in {0.25,0.50,0.75}, pm=0.50
    sub2 = sub[sub["pi_way"].isin([0.25, 0.50, 0.75])]
    s_mr = sub2.groupby(["bench","pi_way"], group_keys=False).apply(
        lambda g: slope(g["t_mram_rd"], g["stall_pct"])
    ).unstack()
    s_mw = sub2.groupby(["bench","pi_way"], group_keys=False).apply(
        lambda g: slope(g["t_mram_wr"], g["stall_pct"])
    ).unstack()
    for pw in [0.25, 0.50, 0.75]:
        out[f"tMr_slope_median_pw{pw:.2f}"] = float(np.nanmedian(s_mr.get(pw, pd.Series(dtype=float)).values))
        out[f"tMw_slope_median_pw{pw:.2f}"] = float(np.nanmedian(s_mw.get(pw, pd.Series(dtype=float)).values))

    # Δstall for tS: 20 - 12 (same slice)
    ts = sub2[sub2["t_sram_hit"].isin([12, 20])]
    if not ts.empty:
        pt = ts.pivot_table(index=["bench","pi_way","t_mram_rd","t_mram_wr"],
                            columns="t_sram_hit", values="stall_pct", aggfunc="mean")
        d = (pt.get(20) - pt.get(12))
        if isinstance(d, pd.Series):
            d_pw = d.groupby(level=1).median()
            for pw in [0.25, 0.50, 0.75]:
                out[f"Delta_stall_tS_20-12_pw{pw:.2f}"] = float(d_pw.get(pw, np.nan))
    return pd.DataFrame([out])

def autodetect_profiles(cap):
    roots = glob.glob(f"results/surrogate/profiles/*/L3_{cap}/physics_pred.csv")
    profs = sorted({p.split(os.sep)[3] for p in roots})  # profiles/<name>/L3_cap/...
    return profs

def main():
    ap = argparse.ArgumentParser(description="Summarize sweep & sensitivity")
    ap.add_argument("--cap", type=int, required=True, help="capacity MB (e.g., 32)")
    ap.add_argument("--profiles", default="", help="comma-separated device profiles; auto-detect if empty")
    ap.add_argument("--bounds-profile", default="", help="profile to use for policy_bounds.csv (default: current run)")
    ap.add_argument("--outdir", default="", help="output dir (default: results/surrogate/L3_<cap>)")
    args = ap.parse_args()

    cap = args.cap
    outdir = args.outdir or f"results/surrogate/L3_{cap}"
    os.makedirs(outdir, exist_ok=True)

    df = load_dataset(cap)

    # Sensitivity
    sens_df = sensitivity(df)
    sens_df.to_csv(os.path.join(outdir, "sensitivity_summary.csv"), index=False)

    # Profiles medians
    profiles = [p for p in args.profiles.split(",") if p] or autodetect_profiles(cap)
    prof_df = device_profile_medians(cap, profiles)
    if not prof_df.empty:
        prof_df.to_csv(os.path.join(outdir, "device_profile_medians.csv"), index=False)

    # Policy bounds
    d_stall, d_energy_uJ, top5 = summarize_policy_bounds(cap, "devices", args.bounds_profile, "dataset")
    if d_stall is not None:
        sum_pb = pd.DataFrame([{"median_delta_stall": d_stall, "median_delta_energy_uJ": d_energy_uJ}])
        sum_pb.to_csv(os.path.join(outdir, "policy_bounds_summary.csv"), index=False)
        top5.to_csv(os.path.join(outdir, "policy_bounds_top5.csv"), index=False)

    # Console summary
    print(f"=== L3_{cap}MB SUMMARY ===")
    if not prof_df.empty:
        print("\nDevice profiles (median):")
        print(prof_df.to_string(index=False))
    if d_stall is not None:
        print(f"\nPolicy bounds: median Δstall={d_stall:.4f} frac, median Δenergy={d_energy_uJ:.2f} μJ")
        print("Most sensitive benches (Δstall median):")
        print(top5.to_string(index=False))
    print("\nSensitivity:")
    print(sens_df.to_string(index=False))

if __name__ == "__main__":
    main()
