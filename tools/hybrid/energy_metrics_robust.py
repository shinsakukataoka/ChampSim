import argparse, pandas as pd, numpy as np
ap=argparse.ArgumentParser()
ap.add_argument("--cap", type=int, required=True)
ap.add_argument("--profile", required=True)
ap.add_argument("--epsilon", type=float, default=1e-9)
args=ap.parse_args()

cap=args.cap; prof=args.profile
JOIN=["bench","pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr","tag"]
E=pd.read_csv(f"results/eval/dataset_L3_{cap}MB.csv")
P=pd.read_csv(f"results/surrogate/profiles/{prof}/L3_{cap}/physics_pred.csv")
M=E.merge(P[JOIN+["stall_pct","E_total_J"]], on=JOIN, how="inner", suffixes=("_eval","_phys"))
if M.empty: raise SystemExit("no overlap; ensure physics_pred.csv exists for the profile")

e_eval=M["E_total_J_eval"]; e_phys=M["E_total_J_phys"]
mae_uJ=(e_phys.sub(e_eval).abs()*1e6)
smape=200.0*(e_phys.sub(e_eval).abs())/np.maximum(e_phys.abs().add(e_eval.abs()), 1e-30)
mask=e_eval.gt(args.epsilon)
mape_trim=(e_phys.sub(e_eval).abs()/e_eval.clip(lower=args.epsilon))*100.0

print(f"Abs energy MAE (µJ): med={mae_uJ.median():.3f}  p90={mae_uJ.quantile(0.9):.3f}  rows={len(M)}")
print(f"SMAPE   (%):        med={smape.median():.2f}  p90={smape.quantile(0.9):.2f}")
print(f"Trim MAPE(>{args.epsilon}J): med={mape_trim[mask].median():.2f}%  p90={mape_trim[mask].quantile(0.9):.2f}%  rows={int(mask.sum())}")
