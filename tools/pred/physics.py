#!/usr/bin/env python3
import math

DEFAULTS = dict(alpha_m=0.0, beta_w=0.0, kappa=0.8, lam=0.0, f_crit=0.45,
                pi_miss_base=0.5, pi_way_base=0.5, eps=1e-9)

def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

def physics_predict_row(row, knobs, consts=DEFAULTS):
    """
    row: baseline window dict-like with keys:
      A_tot, P_sram_rd, P_sram_wr, P_mram_rd, P_mram_wr, P_miss, mlp_hit, mlp_miss,
      T_miss_base, AMAT_base, cycles_base
    knobs: dict with tS,tMr,tMw, pi_miss, pi_way
    returns: cycles_phys, aux_dict
    """
    eps = consts["eps"]
    # baseline partitions
    P_sram_rd = float(row["P_sram_rd"]); P_sram_wr = float(row["P_sram_wr"])
    P_mram_rd = float(row["P_mram_rd"]); P_mram_wr = float(row["P_mram_wr"])
    P_miss    = float(row["P_miss"])
    A_tot     = float(row["A_tot"])
    if A_tot <= 0:
        # degenerate window: return baseline cycles
        return float(row["cycles_base"]), {"AMAT": row["AMAT_base"], "Tmiss": row["T_miss_base"]}

    # partition nudges (very small first-order shift; residual will correct)
    d_fill = consts["alpha_m"]*(float(knobs["pi_miss"]) - consts["pi_miss_base"])
    d_way  = consts["beta_w"] *(float(knobs["pi_way"])  - consts["pi_way_base"])
    shift = d_fill + d_way
    w_rd, w_wr = 1.0, 0.5  # reads are usually more critical

    P_mram_rd_p = clamp01(P_mram_rd + w_rd*shift)
    P_sram_rd_p = clamp01(P_sram_rd - w_rd*shift)
    P_mram_wr_p = clamp01(P_mram_wr + w_wr*shift)
    P_sram_wr_p = clamp01(P_sram_wr - w_wr*shift)
    # keep P_miss unchanged here (first-order)
    total = P_mram_rd_p + P_sram_rd_p + P_mram_wr_p + P_sram_wr_p + P_miss
    if total > 0:
        scale = (P_mram_rd + P_sram_rd + P_mram_wr + P_sram_wr + P_miss) / total
        P_mram_rd_p *= scale; P_sram_rd_p *= scale
        P_mram_wr_p *= scale; P_sram_wr_p *= scale
        # P_miss remains ~same

    # effective hit latencies (overlap)
    mlp_hit = max(1.0, float(row["mlp_hit"]))
    tS_eff  = float(knobs["tS"])/mlp_hit
    tMr_eff = float(knobs["tMr"])/mlp_hit
    tMw_eff = float(knobs["tMw"])/mlp_hit

    # miss path (scale queueing by miss traffic, preserve overlap via mlp_miss)
    mlp_miss = max(1.0, float(row["mlp_miss"]))
    rho_rel  = max(eps, P_miss) / max(eps, float(row["P_miss"]))
    q_scale  = 1.0 + consts["kappa"]*(rho_rel - 1.0)
    Tmiss    = (float(row["T_miss_base"]) * q_scale) / mlp_miss
    if consts.get("lam", 0.0) != 0.0 and "cong_rel" in row:
        Tmiss *= (float(row["cong_rel"]))**consts["lam"]

    AMAT = (P_sram_rd_p+P_sram_wr_p)*tS_eff + (P_mram_rd_p)*tMr_eff + (P_mram_wr_p)*tMw_eff + P_miss*Tmiss
    d_stall = A_tot * (AMAT - float(row["AMAT_base"])) * consts["f_crit"]
    cycles_phys = float(row["cycles_base"]) + d_stall

    aux = dict(AMAT=AMAT, Tmiss=Tmiss,
               P_sram_rd=P_sram_rd_p, P_sram_wr=P_sram_wr_p,
               P_mram_rd=P_mram_rd_p, P_mram_wr=P_mram_wr_p,
               tS_eff=tS_eff, tMr_eff=tMr_eff, tMw_eff=tMw_eff)
    return cycles_phys, aux
