import math

def _eff_lats(knobs, mlp_hit):
    mh = max(1.0, float(mlp_hit))
    return float(knobs["tS"])/mh, float(knobs["tMr"])/mh, float(knobs["tMw"])/mh

def _tmiss(brow, consts):
    # match physics.py's treatment (use baseline T_miss scaled by mlp_miss and optional kappa)
    mlp_miss = max(1.0, float(brow["mlp_miss"]))
    T_base   = float(brow["T_miss_base"])
    # optional queueing sensitivity: q_scale = 1 + kappa*(rho_rel-1); with no policy change to P_miss, rho_rel≈1
    kappa    = float(consts.get("kappa", 0.0))
    q_scale  = 1.0  # we keep P_miss fixed for bounds; set 1.0 so bounds reflect hit-placement only
    return (T_base * q_scale) / mlp_miss

def _choose_optL(tS_eff, tMr_eff, tMw_eff, pessimistic=False):
    rd = (tMr_eff < tS_eff)
    wr = (tMw_eff < tS_eff)
    return (not rd if pessimistic else rd), (not wr if pessimistic else wr)

def _choose_optDynE(etab, pessimistic=False):
    rd = (etab["mram"]["read_hit_pJ"]  < etab["sram"]["read_hit_pJ"])
    wr = (etab["mram"]["write_hit_pJ"] < etab["sram"]["write_hit_pJ"])
    return (not rd if pessimistic else rd), (not wr if pessimistic else wr)

def _choose_optTotalE(tS_eff, tMr_eff, tMw_eff, A_tot, f_crit, clk_ns, etab, pessimistic=False):
    # NOTE: leak coefficient here multiplies Δt; actual leakage mW will be θ-weighted later.
    leak_coeff = (etab["sram"]["leak_mW"] + etab["mram"]["leak_mW"]) * clk_ns * float(f_crit) * float(A_tot)
    rd = ( (etab["mram"]["read_hit_pJ"]  - etab["sram"]["read_hit_pJ"])  + leak_coeff*(tMr_eff - tS_eff) ) < 0.0
    wr = ( (etab["mram"]["write_hit_pJ"] - etab["sram"]["write_hit_pJ"]) + leak_coeff*(tMw_eff - tS_eff) ) < 0.0
    return (not rd if pessimistic else rd), (not wr if pessimistic else wr)

def _theta(knobs):
    # First-order estimate of MRAM share; start with capacity split.
    return float(knobs.get("pi_way", 0.0))

def _assemble(brow, knobs, consts, etab, chooser, clk_ns=0.25, pessimistic=False):
    # unpack baseline window fields
    P_sr_rd = float(brow["P_sram_rd"]); P_sr_wr = float(brow["P_sram_wr"])
    P_mr_rd = float(brow["P_mram_rd"]); P_mr_wr = float(brow["P_mram_wr"])
    P_miss  = float(brow["P_miss"])
    A_tot   = float(brow["A_tot"])
    AMAT_b  = float(brow["AMAT_base"])
    cyc_b   = float(brow["cycles_base"])
    f_crit  = float(consts.get("f_crit", 0.6))

    # totals of read vs write hits (keep totals; only reassign medium)
    P_rd = P_sr_rd + P_mr_rd
    P_wr = P_sr_wr + P_mr_wr

    # effective latencies and miss time
    tS_eff, tMr_eff, tMw_eff = _eff_lats(knobs, brow["mlp_hit"])
    Tmiss = _tmiss(brow, consts)

    # decide where reads/writes go
    rd_to_mram, wr_to_mram = chooser(tS_eff, tMr_eff, tMw_eff) if chooser.__name__=="_choose_optL" \
                             else (chooser(etab) if chooser.__name__=="_choose_optDynE"
                                   else chooser(tS_eff, tMr_eff, tMw_eff, A_tot, f_crit, clk_ns, etab))
    if pessimistic:
        rd_to_mram, wr_to_mram = (not rd_to_mram), (not wr_to_mram)

    # AMAT under the bound choices
    t_rd = (tMr_eff if rd_to_mram else tS_eff)
    t_wr = (tMw_eff if wr_to_mram else tS_eff)
    AMAT = P_miss*Tmiss + P_rd*t_rd + P_wr*t_wr

    # cycles via critical fraction model
    cycles = cyc_b + f_crit * A_tot * (AMAT - AMAT_b)

    # Dynamic energy with medium-aware per-hit coefficients.
    E_hit_rd = (etab["mram"]["read_hit_pJ"]  if rd_to_mram else etab["sram"]["read_hit_pJ"])
    E_hit_wr = (etab["mram"]["write_hit_pJ"] if wr_to_mram else etab["sram"]["write_hit_pJ"])
    E_dyn = A_tot * (P_rd*E_hit_rd + P_wr*E_hit_wr + P_miss*etab["miss_path"]["per_miss_pJ"])
    # Leakage mW weighted by θ (MRAM share).
    theta = _theta(knobs)
    leak_mW = etab["sram"]["leak_mW"]*(1.0-theta) + etab["mram"]["leak_mW"]*theta
    E_leak = leak_mW * cycles * clk_ns
    E_tot = E_dyn + E_leak

    return dict(AMAT=AMAT, cycles=cycles, energy_pJ=E_tot,
                choice={"read":"MRAM" if rd_to_mram else "SRAM",
                        "write":"MRAM" if wr_to_mram else "SRAM"})

def bound_window(brow, knobs, consts, etab, policy="optL", pessimistic=False, clk_ns=0.25):
    chooser = {"optL": _choose_optL, "optDynE": _choose_optDynE, "optTotalE": _choose_optTotalE}[policy]
    return _assemble(brow, knobs, consts, etab, chooser, clk_ns=clk_ns, pessimistic=pessimistic)

def accumulate_bounds(dfb, knobs, consts, etab, clk_ns=0.25):
    acc = {k:0.0 for k in ["cycles_optL","energy_optL_pJ","cycles_optL_pess","energy_optL_pess_pJ",
                           "cycles_optDynE","energy_optDynE_pJ","cycles_optDynE_pess","energy_optDynE_pess_pJ",
                           "cycles_optTotalE","energy_optTotalE_pJ","cycles_optTotalE_pess","energy_optTotalE_pess_pJ"]}
    for _, brow in dfb.iterrows():
        for pol in ("optL","optDynE","optTotalE"):
            res = bound_window(brow, knobs, consts, etab, policy=pol, pessimistic=False, clk_ns=clk_ns)
            acc[f"cycles_{pol}"]         += res["cycles"]
            acc[f"energy_{pol}_pJ"]      += res["energy_pJ"]
            resP = bound_window(brow, knobs, consts, etab, policy=pol, pessimistic=True, clk_ns=clk_ns)
            acc[f"cycles_{pol}_pess"]    += resP["cycles"]
            acc[f"energy_{pol}_pess_pJ"] += resP["energy_pJ"]
    return acc
