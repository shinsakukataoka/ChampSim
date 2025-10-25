#!/usr/bin/env python3
import os, pandas as pd, joblib
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

IN  = "results/surrogate/training_table.csv"
OD  = "results/surrogate"

df = pd.read_csv(IN)

feat_cols = [
    "pi_way","pi_miss","pi_way_x_pi_miss",
    "t_sram_hit","t_mram_rd","t_mram_wr","t_dram","l3_mb",
    "char_miss_rate","char_acc_per_1kcyc","char_read_frac","char_mram_hit_frac",
    "char_mlp_hit","char_mlp_miss","char_mpkc","char_stall_per_1kcyc"
]

X = df[feat_cols].values
y_t = df["stall_pct"].values
y_e = df["E_total_J"].values
groups = df["bench"].values

# Bench-aware CV for a global model
gkf = GroupKFold(n_splits=5)
mae_t, mae_e = [], []
for tr, te in gkf.split(X, y_t, groups):
    mt = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=800, random_state=0).fit(X[tr], y_t[tr])
    me = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=800, random_state=0).fit(X[tr], y_e[tr])
    mae_t.append(mean_absolute_error(y_t[te], mt.predict(X[te])))
    mae_e.append(mean_absolute_error(y_e[te], me.predict(X[te])))

print(f"[GLOBAL] CV MAE stall_frac: {sum(mae_t)/len(mae_t):.4f}")
print(f"[GLOBAL] CV MAE energy_J : {sum(mae_e)/len(mae_e):.4e}")

# Train and save global models
os.makedirs(OD, exist_ok=True)
mt = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=1000, random_state=0).fit(X, y_t)
me = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=1000, random_state=0).fit(X, y_e)
joblib.dump({"model": mt, "feat_cols": feat_cols}, os.path.join(OD, "surrogate_stallpct_hgb_global.joblib"))
joblib.dump({"model": me, "feat_cols": feat_cols}, os.path.join(OD, "surrogate_energy_hgb_global.joblib"))
print("[GLOBAL] saved models")

# Per-bench models (train on each bench only)
for b, grp in df.groupby("bench"):
    Xb = grp[feat_cols].values
    ytb = grp["stall_pct"].values
    yeb = grp["E_total_J"].values
    mtb = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=1000, random_state=0).fit(Xb, ytb)
    meb = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=1000, random_state=0).fit(Xb, yeb)
    joblib.dump({"model": mtb, "feat_cols": feat_cols}, os.path.join(OD, f"stallpct_hgb_{b}.joblib"))
    joblib.dump({"model": meb, "feat_cols": feat_cols}, os.path.join(OD, f"energy_hgb_{b}.joblib"))
print("[PER-BENCH] saved models")

