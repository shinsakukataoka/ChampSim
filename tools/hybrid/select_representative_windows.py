#!/usr/bin/env python3
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Usage:
#   python3 tools/hybrid/select_representative_windows.py <path/to/LLC.window_features.csv> [K] [stdn_thr] [mpkc_min]
# Writes:
#   <same_dir>/LLC.representatives.json

PF        = sys.argv[1] if len(sys.argv) > 1 else "results/LLC.window_features.csv"
K_desired = int(sys.argv[2]) if len(sys.argv) > 2 else 3
STDN_THR  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
MPKC_MIN  = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0

df = pd.read_csv(PF)

# stability mask
CYC_MIN, WAVE_REQ = 50_000, 1
stdn = df['std5_norm'].fillna(1.0).to_numpy()
mpkc = df['mpkc'].to_numpy()
stable_mask = (
    (stdn < STDN_THR) &
    (mpkc >= MPKC_MIN) &
    (df['win_cycles'] >= CYC_MIN) &
    (df['wave_done'] >= WAVE_REQ)
)
df_st = df[stable_mask].copy()
if df_st.empty:
    print(f"[select_representative_windows] fallback to top-MPKC windows (stdn<{STDN_THR}, mpkc>={MPKC_MIN} too strict).")
    df_st = df.sort_values('mpkc', ascending=False).head(max(K_desired,5)).copy()

# ensure integer window_id
if 'window_id' not in df_st.columns:
    df_st['window_id'] = np.arange(len(df_st), dtype=int)
else:
    pos_ids = pd.Series(np.arange(len(df_st), dtype=int), index=df_st.index)
    df_st['window_id'] = df_st['window_id'].fillna(pos_ids).astype(int, errors='ignore')

n_samples = len(df_st)
if n_samples == 0:
    raise RuntimeError("No windows to cluster.")

feat_cols = [
    'miss_rate','acc_per_1kcyc','read_frac','mram_hit_frac',
    'mlp_hit','mlp_miss','stall_per_1kcyc','mpkc'
]
X = df_st[feat_cols].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

if n_samples == 1:
    rep_id = int(df_st.iloc[0]['window_id'])
    out = {
        "stable_windows": [rep_id],
        "representatives": [rep_id],
        "n_clusters": 1
    }
    dst = os.path.join(os.path.dirname(PF), "LLC.representatives.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[select_representative_windows] reps = [{rep_id}] -> {dst}")
    sys.exit(0)

scaler = StandardScaler()
Xp = scaler.fit_transform(X)
pca_n = max(1, min(6, n_samples, Xp.shape[1]))
Z = PCA(n_components=pca_n, svd_solver='full', random_state=0).fit_transform(Xp)

K = max(1, min(K_desired, n_samples))
kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
labels = kmeans.fit_predict(Z)

df_st = df_st.reset_index(drop=True)
labels = np.asarray(labels)

reps = []
for c in sorted(np.unique(labels)):
    pos_idx = np.where(labels == c)[0]
    Z_sub = Z[pos_idx]
    cent  = kmeans.cluster_centers_[c]
    d2 = ((Z_sub - cent) ** 2).sum(axis=1)
    rep_pos = int(pos_idx[np.argmin(d2)])
    rep_id  = int(df_st.iloc[rep_pos]['window_id'])
    reps.append(rep_id)

out = {
    "stable_windows": list(map(int, df_st['window_id'].tolist())),
    "representatives": reps,
    "n_clusters": int(K)
}
dst = os.path.join(os.path.dirname(PF), "LLC.representatives.json")
with open(dst, "w") as f:
    json.dump(out, f, indent=2)
print(f"[select_representative_windows] reps={reps} -> {dst}")
