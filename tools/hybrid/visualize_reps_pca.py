#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

FEATURES = [
    'miss_rate','acc_per_1kcyc','read_frac','mram_hit_frac',
    'mlp_hit','mlp_miss','stall_per_1kcyc','mpkc'
]

def load_data(char_dir):
    feat_csv = os.path.join(char_dir, "LLC.window_features.csv")
    reps_json = os.path.join(char_dir, "LLC.representatives.json")
    if not os.path.exists(feat_csv):
        raise FileNotFoundError(f"missing: {feat_csv}")
    df = pd.read_csv(feat_csv)
    reps = None
    if os.path.exists(reps_json):
        reps = json.load(open(reps_json)).get("representatives", [])
    return df, reps

def compute_stable_mask(df, stdn_thr=0.10, mpkc_min=1.0, cyc_min=50_000, wave_req=1):
    stdn = df['std5_norm'].fillna(1.0).to_numpy()
    mpkc = df['mpkc'].to_numpy()
    win_cycles = df['win_cycles'].to_numpy()
    wave_done = df['wave_done'].to_numpy() if 'wave_done' in df.columns else np.ones(len(df))
    return (stdn < stdn_thr) & (mpkc >= mpkc_min) & (win_cycles >= cyc_min) & (wave_done >= wave_req)

def nearest_rep_labels(emb2d, rep_idx):
    rep_pts = emb2d[rep_idx,:]
    # label each point by nearest rep in PCA space
    d2 = ((emb2d[:,None,:] - rep_pts[None,:,:])**2).sum(axis=2)
    return d2.argmin(axis=1)

def main():
    ap = argparse.ArgumentParser(description="PCA/KMeans viz of representative windows")
    ap.add_argument("--char-dir", required=True, help="results/characterization_L3_<cap>/<bench> directory")
    ap.add_argument("--out", default="", help="output PNG path (default: figures under results/figs/)")
    ap.add_argument("--K", type=int, default=5, help="K for KMeans (when --color-by=kmeans)")
    ap.add_argument("--color-by", choices=["kmeans","nearest-rep"], default="nearest-rep",
                    help="how to color clusters")
    ap.add_argument("--stdn-thr", type=float, default=0.10)
    ap.add_argument("--mpkc-min", type=float, default=1.0)
    ap.add_argument("--cyc-min", type=int, default=50_000)
    ap.add_argument("--wave-req", type=int, default=1)
    args = ap.parse_args()

    df, reps = load_data(args.char_dir)
    if not set(FEATURES).issubset(df.columns):
        missing = [c for c in FEATURES if c not in df.columns]
        raise RuntimeError(f"missing feature columns: {missing}")

    window_ids = df['window_id'].to_numpy() if 'window_id' in df.columns else np.arange(len(df))
    X = df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6))

    pca = PCA(n_components=2, svd_solver='full', random_state=0)
    Z = pca.fit_transform(Xs)  # (N,2)

    # labels
    if reps and args.color_by == "nearest-rep":
        # map rep IDs to their row indices; robust if window_id not contiguous
        id_to_idx = {int(wid): idx for idx, wid in enumerate(window_ids)}
        rep_idx = [id_to_idx[r] for r in reps if int(r) in id_to_idx]
        labels = nearest_rep_labels(Z, np.array(rep_idx, dtype=int))
        centers = Z[rep_idx,:]
    else:
        km = KMeans(n_clusters=args.K, n_init=10, random_state=0)
        labels = km.fit_predict(Z)
        centers = km.cluster_centers_

    stable = compute_stable_mask(df, args.stdn_thr, args.mpkc_min, args.cyc_min, args.wave_req)

    # Plot
    fig, ax = plt.subplots(figsize=(7.5,6.5))
    # light gray for unstable
    ax.scatter(Z[~stable,0], Z[~stable,1], c="#CCCCCC", s=12, alpha=0.5, label="unstable")

    # color map for clusters
    cmap = plt.get_cmap("tab10")
    for k in np.unique(labels):
        mask = (labels==k) & stable
        ax.scatter(Z[mask,0], Z[mask,1], s=16, alpha=0.75, color=cmap(k%10), label=f"cluster {k}")

    # Representatives: star markers, black edge
    if reps:
        id_to_idx = {int(wid): idx for idx, wid in enumerate(window_ids)}
        rep_idx = [id_to_idx[r] for r in reps if int(r) in id_to_idx]
        ax.scatter(Z[rep_idx,0], Z[rep_idx,1], s=140, marker="*", color="gold",
                   edgecolor="black", linewidth=0.8, label="representatives")
        # annotate rep IDs (window_id)
        for ridx in rep_idx:
            ax.annotate(str(int(window_ids[ridx])), (Z[ridx,0], Z[ridx,1]),
                        textcoords="offset points", xytext=(4,4), fontsize=7, color="black")

    # centers
    ax.scatter(centers[:,0], centers[:,1], s=90, marker="X", color="black", label="centers/means")

    var1, var2 = pca.explained_variance_ratio_[0]*100, pca.explained_variance_ratio_[1]*100
    ax.set_xlabel(f"PCA 1 ({var1:.1f}% var)")
    ax.set_ylabel(f"PCA 2 ({var2:.1f}% var)")
    ax.set_title(os.path.basename(args.char_dir))
    ax.grid(alpha=0.2)
    # de-duplicate legend entries
    handles, labels_txt = ax.get_legend_handles_labels()
    uniq = dict(zip(labels_txt, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, frameon=False, loc="best")

    out = args.out
    if not out:
        figs_dir = os.path.join("results","figs", os.path.basename(os.path.dirname(args.char_dir)))
        os.makedirs(figs_dir, exist_ok=True)
        out = os.path.join(figs_dir, f"pca_kmeans_{os.path.basename(args.char_dir)}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[ok] wrote {out}")

if __name__ == "__main__":
    main()

