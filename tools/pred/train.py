#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from physics import physics_predict_row, DEFAULTS as PHY_CONSTS


def build_X(brow, knobs, aux):
    return np.array([
        float(brow["A_tot"]),
        float(brow["P_miss"]), float(brow["P_sram_rd"]), float(brow["P_mram_rd"]),
        float(brow["mlp_hit"]), float(brow["mlp_miss"]),
        float(brow["T_miss_base"]), float(brow["AMAT_base"]),
        float(knobs["tS"]), float(knobs["tMr"]), float(knobs["tMw"]),
        float(knobs["pi_miss"]), float(knobs["pi_way"]),
        float(aux["AMAT"]), float(aux["Tmiss"])
    ], dtype=float)


def build_X_base(brow, knobs):
    return np.array([
        float(brow["A_tot"]), float(brow["P_miss"]), float(brow["P_sram_rd"]), float(brow["P_mram_rd"]),
        float(brow["mlp_hit"]), float(brow["mlp_miss"]),
        float(brow["T_miss_base"]), float(brow["AMAT_base"]),
        float(knobs["tS"]), float(knobs["tMr"]), float(knobs["tMw"]),
        float(knobs["pi_miss"]), float(knobs["pi_way"])
    ], dtype=float)


def energy_phys_row(brow, aux, cycles_phys, etab, clk_ns=0.25):
    # Fall back to baseline partitions if aux lacks them
    A = float(brow["A_tot"])
    Psr_rd = float(aux.get("P_sram_rd", brow["P_sram_rd"]))
    Psr_wr = float(aux.get("P_sram_wr", brow["P_sram_wr"]))
    Pmr_rd = float(aux.get("P_mram_rd", brow["P_mram_rd"]))
    Pmr_wr = float(aux.get("P_mram_wr", brow["P_mram_wr"]))
    Pmiss = float(brow["P_miss"])
    e = (
        A * Psr_rd * etab["sram"]["read_hit_pJ"] +
        A * Psr_wr * etab["sram"]["write_hit_pJ"] +
        A * Pmr_rd * etab["mram"]["read_hit_pJ"] +
        A * Pmr_wr * etab["mram"]["write_hit_pJ"] +
        A * Pmiss  * etab["miss_path"]["per_miss_pJ"]
    )
    leak_mW = etab["sram"]["leak_mW"] + etab["mram"]["leak_mW"]
    e += leak_mW * cycles_phys * clk_ns
    return e


def train_for_bench(bench, datasets_root, model_dir, val_frac=0.25, seed=0):
    etab = json.load(open("/home/skataoka26/ChampSim/model/energy_table.json"))

    bdir = os.path.join(datasets_root, bench)
    base_csv = os.path.join(bdir, "baseline_windows.csv")
    train_csv = os.path.join(bdir, "training_windows.csv")
    if not (os.path.exists(base_csv) and os.path.exists(train_csv)):
        raise SystemExit(f"Missing dataset files for {bench}")

    dfb = pd.read_csv(base_csv)
    dft = pd.read_csv(train_csv)

    # split by config_id (holdout by configs)
    cfgs = dft["config_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(cfgs)
    n_val = max(1, int(len(cfgs) * val_frac))
    val_cfgs = set(cfgs[:n_val])
    trn_cfgs = set(cfgs[n_val:])

    # --- estimate f_crit per benchmark via least squares (cycles_true - cycles_base ≈ f * S) ---
    S_list, y_list = [], []
    for cid in trn_cfgs:
        sub = dft[dft["config_id"] == cid]
        for _, r in sub.iterrows():
            brow = dfb[dfb["window_id"] == r["window_id"]].iloc[0]
            knobs = dict(tS=r["tS"], tMr=r["tMr"], tMw=r["tMw"], pi_miss=r["pi_miss"], pi_way=r["pi_way"])
            c1, aux1 = physics_predict_row(brow, knobs, {**PHY_CONSTS, "f_crit": 1.0})
            S = c1 - float(brow["cycles_base"])
            y = float(r["cycles_true"]) - float(brow["cycles_base"])
            S_list.append(S)
            y_list.append(y)
    S_arr = np.array(S_list, dtype=float)
    y_arr = np.array(y_list, dtype=float)
    if np.any(S_arr != 0):
        f_opt = float(np.clip((S_arr * y_arr).sum() / max(1e-9, (S_arr * S_arr).sum()), 0.05, 0.9))
    else:
        f_opt = PHY_CONSTS["f_crit"]
    consts = PHY_CONSTS.copy()
    consts["f_crit"] = f_opt

    # --- NEW: dump LS points for visualization ---
    diag_rows = []
    for cid in cfgs:  # all configs (train+val)
        split = "train" if cid in trn_cfgs else ("val" if cid in val_cfgs else "other")
        sub = dft[dft["config_id"] == cid]
        for _, r in sub.iterrows():
            brow = dfb[dfb["window_id"] == r["window_id"]].iloc[0]
            knobs = dict(tS=r["tS"], tMr=r["tMr"], tMw=r["tMw"], pi_miss=r["pi_miss"], pi_way=r["pi_way"])
            c1, _ = physics_predict_row(brow, knobs, {**PHY_CONSTS, "f_crit": 1.0})
            S = c1 - float(brow["cycles_base"])
            y = float(r["cycles_true"]) - float(brow["cycles_base"])
            diag_rows.append(dict(
                config_id=int(cid), window_id=int(brow["window_id"]),
                S=S, y=y, split=split
            ))
    os.makedirs(os.path.join(model_dir, "diag"), exist_ok=True)
    pd.DataFrame(diag_rows).to_csv(os.path.join(model_dir, "diag", f"{bench}_fcrit_points.csv"), index=False)
    with open(os.path.join(model_dir, "diag", f"{bench}_fcrit.txt"), "w") as fp:
        fp.write(str(f_opt))

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f"{bench}_consts.json"), "w") as fp:
        json.dump(consts, fp, indent=2)

    # assemble training rows
    X_tr, y_tr = [], []
    XE_tr, yE_tr = [], []
    XB_tr, yB_tr = [], []

    for cid in trn_cfgs:
        sub = dft[dft["config_id"] == cid]
        for _, r in sub.iterrows():
            brow = dfb[dfb["window_id"] == r["window_id"]].iloc[0]
            knobs = dict(tS=r["tS"], tMr=r["tMr"], tMw=r["tMw"], pi_miss=r["pi_miss"], pi_way=r["pi_way"])

            cycles_phys, aux = physics_predict_row(brow, knobs, consts)
            y = float(r["cycles_true"]) - cycles_phys
            X = build_X(brow, knobs, aux)
            X_tr.append(X)
            y_tr.append(y)

            # energy residual (optional)
            if "energy_true_pJ" in r and not pd.isna(r["energy_true_pJ"]):
                E_phys = energy_phys_row(brow, aux, cycles_phys, etab)
                XE_tr.append(X)
                yE_tr.append(float(r["energy_true_pJ"]) - E_phys)

            # baselines (direct cycles)
            XB_tr.append(build_X_base(brow, knobs))
            yB_tr.append(float(r["cycles_true"]))

    X_tr = np.vstack(X_tr) if X_tr else np.zeros((0, 15))
    y_tr = np.array(y_tr, dtype=float)
    XE_tr = np.vstack(XE_tr) if XE_tr else None
    yE_tr = np.array(yE_tr, dtype=float) if XE_tr is not None else None
    XB_tr = np.vstack(XB_tr) if XB_tr else np.zeros((0, 13))
    yB_tr = np.array(yB_tr, dtype=float)

    # fit models
    model = GradientBoostingRegressor(
        random_state=seed, loss="squared_error",
        n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9
    )
    model.fit(X_tr, y_tr)

    modelE = None
    if XE_tr is not None and len(XE_tr) > 0:
        modelE = GradientBoostingRegressor(
            random_state=seed, loss="squared_error",
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9
        )
        modelE.fit(XE_tr, yE_tr)

    gbm_base = GradientBoostingRegressor(
        random_state=seed, loss="squared_error",
        n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.9
    )
    gbm_base.fit(XB_tr, yB_tr)

    ridge = Ridge(alpha=1.0)
    ridge.fit(XB_tr, yB_tr)

    # validation
    def mape(pred, true): 
        return abs(pred - true) / max(1.0, true)

    reports = []
    for cid in val_cfgs:
        sub = dft[dft["config_id"] == cid]
        cyc_phys_sum = cyc_ours_sum = cyc_true_sum = 0.0
        cyc_gbm_sum = cyc_ridge_sum = 0.0
        E_phys_sum = E_pred_sum = E_true_sum = 0.0

        for _, r in sub.iterrows():
            brow = dfb[dfb["window_id"] == r["window_id"]].iloc[0]
            knobs = dict(tS=r["tS"], tMr=r["tMr"], tMw=r["tMw"], pi_miss=r["pi_miss"], pi_way=r["pi_way"])

            cycles_phys, aux = physics_predict_row(brow, knobs, consts)
            X = build_X(brow, knobs, aux)
            delta = float(model.predict([X])[0])
            cycles_ours = cycles_phys + delta
            cyc_true = float(r["cycles_true"])
            cyc_gbm = float(gbm_base.predict([build_X_base(brow, knobs)])[0])
            cyc_rid = float(ridge.predict([build_X_base(brow, knobs)])[0])

            cyc_phys_sum += cycles_phys
            cyc_ours_sum += cycles_ours
            cyc_true_sum += cyc_true
            cyc_gbm_sum += cyc_gbm
            cyc_ridge_sum += cyc_rid

            # energy
            E_phys = energy_phys_row(brow, aux, cycles_phys, etab)
            E_phys_sum += E_phys
            if "energy_true_pJ" in r and not pd.isna(r["energy_true_pJ"]):
                E_true_sum += float(r["energy_true_pJ"])
                if modelE is not None:
                    E_pred_sum += E_phys + float(modelE.predict([X])[0])

        roi_cycles_true = sub["roi_cycles_true"].dropna().values
        cycles_true_roi = float(roi_cycles_true[0]) if roi_cycles_true.size > 0 else cyc_true_sum

        reports.append(dict(
            config_id=int(cid),
            cycles_true=cycles_true_roi,
            cycles_phys=cyc_phys_sum, mape_phys=mape(cyc_phys_sum, cycles_true_roi),
            cycles_ours=cyc_ours_sum, mape_ours=mape(cyc_ours_sum, cycles_true_roi),
            cycles_gbm=cyc_gbm_sum,   mape_gbm=mape(cyc_gbm_sum, cycles_true_roi),
            cycles_ridge=cyc_ridge_sum, mape_ridge=mape(cyc_ridge_sum, cycles_true_roi),
            energy_true_pJ=E_true_sum if E_true_sum > 0 else None,
            energy_phys_pJ=E_phys_sum if E_true_sum > 0 else None,
            energy_pred_pJ=E_pred_sum if (E_true_sum > 0 and modelE is not None) else None,
            mape_energy_phys=(abs(E_phys_sum - E_true_sum) / E_true_sum if E_true_sum > 0 else None),
            mape_energy_pred=(abs(E_pred_sum - E_true_sum) / E_true_sum if (E_true_sum > 0 and modelE is not None) else None)
        ))

    mape_avg = float(np.mean([x["mape_ours"] for x in reports])) if reports else None
    mape_phys_avg = float(np.mean([x["mape_phys"] for x in reports])) if reports else None
    mape_gbm_avg = float(np.mean([x["mape_gbm"] for x in reports])) if reports else None
    mape_ridge_avg = float(np.mean([x["mape_ridge"] for x in reports])) if reports else None
    mape_Ephys_avg = float(np.mean([x["mape_energy_phys"] for x in reports if x["mape_energy_phys"] is not None])) if reports else None
    mape_Epred_avg = float(np.mean([x["mape_energy_pred"] for x in reports if x["mape_energy_pred"] is not None])) if reports else None

    print(f"[{bench}] ROI cycles MAPE avg — phys:{mape_phys_avg} ours:{mape_avg} gbm:{mape_gbm_avg} ridge:{mape_ridge_avg}")
    if mape_Ephys_avg is not None:
        print(f"[{bench}] ROI energy MAPE avg — phys:{mape_Ephys_avg} pred:{mape_Epred_avg}")

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f"{bench}.pkl"))
    if modelE is not None:
        joblib.dump(modelE, os.path.join(model_dir, f"{bench}_energy.pkl"))
    with open(os.path.join(model_dir, f"{bench}_validation.json"), "w") as fp:
        json.dump(dict(
            bench=bench, consts=consts,
            mape_cycles_avg=mape_avg, mape_cycles_phys=mape_phys_avg,
            mape_cycles_gbm=mape_gbm_avg, mape_cycles_ridge=mape_ridge_avg,
            mape_energy_phys=mape_Ephys_avg, mape_energy_pred=mape_Epred_avg,
            reports=reports
        ), fp, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--datasets-root", default="/home/skataoka26/ChampSim/out/datasets")
    ap.add_argument("--model-dir", default="/home/skataoka26/ChampSim/model")
    ap.add_argument("--val-frac", type=float, default=0.25)
    args = ap.parse_args()
    train_for_bench(args.bench, args.datasets_root, args.model_dir, args.val_frac)

