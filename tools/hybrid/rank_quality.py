import argparse, pandas as pd, numpy as np
from scipy.stats import spearmanr

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--metric", choices=["stall","energy"], default="stall")
    ap.add_argument("--k", type=int, nargs="+", default=[1,3,5])
    args=ap.parse_args()

    cap=args.cap; prof=args.profile
    E=pd.read_csv(f"results/eval/dataset_L3_{cap}MB.csv")
    P=pd.read_csv(f"results/surrogate/profiles/{prof}/L3_{cap}/physics_pred.csv")
    JOIN=["bench","pi_way","pi_miss","t_sram_hit","t_mram_rd","t_mram_wr","tag"]
    M=E.merge(P[JOIN+["stall_pct","E_total_J"]], on=JOIN, how="inner", suffixes=("_eval","_phys"))
    if args.metric=="stall":
        m_eval=M["stall_pct_eval"]; m_phys=M["stall_pct_phys"]
    else:
        m_eval=M["E_total_J_eval"]; m_phys=M["E_total_J_phys"]

    rows=[]
    for b,g in M.groupby("bench"):
        if len(g)<3: continue
        if args.metric=="stall":
            x=g["stall_pct_eval"].to_numpy(); y=g["stall_pct_phys"].to_numpy()
        else:
            x=g["E_total_J_eval"].to_numpy(); y=g["E_total_J_phys"].to_numpy()
        rho,_=spearmanr(x,y)
        row={"bench":b,"spearman":float(rho)}
        # top-k overlap on minimizing metric
        idx_eval=np.argsort(x)
        idx_phys=np.argsort(y)
        for K in args.k:
            topE=set(g.index[idx_eval[:K]])
            topP=set(g.index[idx_phys[:K]])
            row[f"top{K}_overlap"]=len(topE & topP)/max(1,K)
        rows.append(row)

    R=pd.DataFrame(rows)
    print(R.sort_values("spearman",ascending=True).to_string(index=False, float_format=lambda z: f"{z:.3f}"))
    print("\n[summary] spearman med={:.3f}  p90={:.3f}".format(R["spearman"].median(), R["spearman"].quantile(0.9)))

if __name__=="__main__":
    main()
