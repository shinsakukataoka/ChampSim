#!/usr/bin/env python3
import os, pandas as pd, joblib
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

IN  = "results/surrogate/training_table.csv"
OD  = "results/surrogate"

df = pd.read_csv(IN)

feat_cols = [
    "pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr",
    "char_miss_rate","char_acc_per_1kcyc","char_read_frac","char_mram_hit_frac",
    "char_mlp_hit","char_mlp_miss","char_mpkc","char_stall_per_1kcyc"
]
X = df[feat_cols].values
y_t = df["stall_pct"].values
y_e = df["E_total_J"].values
groups = df["bench"].values

gkf = GroupKFold(n_splits=5)
mae_t, mae_e = [], []
for tr, te in gkf.split(X, y_t, groups):
    mt = RandomForestRegressor(n_estimators=400, random_state=0).fit(X[tr], y_t[tr])
    me = RandomForestRegressor(n_estimators=400, random_state=0).fit(X[tr], y_e[tr])
    mae_t.append(mean_absolute_error(y_t[te], mt.predict(X[te])))
    mae_e.append(mean_absolute_error(y_e[te], me.predict(X[te])))

print(f"CV MAE stall_frac: {sum(mae_t)/len(mae_t):.4f}  (~{(sum(mae_t)/len(mae_t))*100:.2f}%)")
print(f"CV MAE energy_J : {sum(mae_e)/len(mae_e):.4e}")

# train final models
mt = RandomForestRegressor(n_estimators=600, random_state=0).fit(X, y_t)
me = RandomForestRegressor(n_estimators=600, random_state=0).fit(X, y_e)

os.makedirs(OD, exist_ok=True)
joblib.dump({"model": mt, "feat_cols": feat_cols}, os.path.join(OD, "surrogate_stallpct_rf.joblib"))
joblib.dump({"model": me, "feat_cols": feat_cols}, os.path.join(OD, "surrogate_energy_rf.joblib"))
print("saved models to", OD)

