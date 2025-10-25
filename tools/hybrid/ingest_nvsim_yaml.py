#!/usr/bin/env python3
"""
Ingest NVSim YAMLs -> devices/<prefix>_n{node}_{opt}/
- Picks one optimization target (default: ReadEDP)
- Builds base.json from 32MB if present, else largest cap
- Writes L3_<cap>.json overrides for every capacity found
- Fills both SRAM and MRAM; if MRAM missing for a node, can borrow from a fallback node (e.g., 32nm)

Usage (example):
  python3 tools/hybrid/ingest_nvsim_yaml.py \
    --src /home/skataoka26/COSC_498/miniMXE/config/nvsim_sniper/20251014T002353Z \
    --out devices --prefix nvsim --prefer ReadEDP --miss-cycles 200 --fallback-mram-node 32
"""
import os, re, json, argparse, math
from collections import defaultdict, namedtuple

try:
    import yaml  # PyYAML
except ImportError:
    raise SystemExit("Please: pip install pyyaml")

Meta = namedtuple("Meta", "cap_mb node_nm opt path")

PAT = re.compile(
    r"sniper_llc_(?P<cap>\d+)mb_.*?_n(?P<node>\d+)_?(?P<opt>[A-Za-z]+)?\.yaml$"
)
OPT_ORDER = ["ReadEDP","ReadLatency","ReadDynamicEnergy","WriteDynamicEnergy","WriteLatency"]

MRAM_KEYS = ["mram","jans","stt","sttmram"]  # be liberal

def find_meta(root):
    metas = []
    for fn in os.listdir(root):
        if not fn.endswith(".yaml"): continue
        m = PAT.match(fn)
        if not m: continue
        cap = int(m.group("cap"))
        node = int(m.group("node"))
        opt = m.group("opt") or ""
        metas.append(Meta(cap, node, opt, os.path.join(root, fn)))
    return metas

def safeget(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d

def get_mram_section(llc):
    if not isinstance(llc, dict): return None
    for k in MRAM_KEYS:
        if k in llc and isinstance(llc[k], dict):
            return llc[k]
    return None

def pick_opt(paths_for_node, prefer):
    # paths_for_node: dict opt -> list[Meta]
    if prefer in paths_for_node: return prefer
    for o in OPT_ORDER:
        if o in paths_for_node: return o
    return sorted(paths_for_node.keys())[0]

def to_leak_per_mb(total_mw, size_bytes):
    mb = max(size_bytes / (1024*1024), 1)
    return float(total_mw) / mb

def extract_from_yaml(path, sram_lat_choice="read"):
    y = yaml.safe_load(open(path))
    llc = safeget(y, "llc", default={})
    sram = safeget(llc, "sram", default=None)
    mram = get_mram_section(llc)

    def one_medium(sec, is_sram=True):
        if not isinstance(sec, dict): return None
        sizeB = float(sec.get("size_bytes", 0))
        # cycles
        rd_cyc = float(sec.get("read_hit_cycles", 0))
        wr_cyc = float(sec.get("write_hit_cycles", rd_cyc))
        if is_sram:
            if sram_lat_choice == "read":  t_hit = rd_cyc
            elif sram_lat_choice == "write": t_hit = wr_cyc
            elif sram_lat_choice == "max": t_hit = max(rd_cyc, wr_cyc)
            else: t_hit = 0.5*(rd_cyc+wr_cyc)
            lat = {"t_sram_hit": int(round(t_hit))}
        else:
            lat = {"t_mram_rd": int(round(rd_cyc)), "t_mram_wr": int(round(wr_cyc))}
        # energy (pJ)
        e_rd = float(safeget(sec, "energy","e_read_hit_pJ", default=0.0))
        e_wr = float(safeget(sec, "energy","e_write_hit_pJ", default=0.0))
        # leakage (mW/MB) from total p_leak_mW
        pleak = float(safeget(sec, "energy","p_leak_mW", default=0.0))
        leak_per_mb = to_leak_per_mb(pleak, sizeB)
        eng = {"e_rd_pj": e_rd, "e_wr_pj": e_wr, "leak_mw_per_mb": leak_per_mb}
        return lat, eng

    res = {"sram": None, "mram": None, "miss_ns": None}
    if sram:
        res["sram"] = one_medium(sram, True)
    if mram:
        res["mram"] = one_medium(mram, False)
    res["miss_ns"] = float(safeget(sram, "timing","miss_latency_ns", default=0.0) or 0.0)
    return res

def write_profile(outdir, profile_name, percap, base_cap=None, miss_cycles=200, f_clk_ghz=1.0):
    """
    percap: cap_mb -> {"sram":(lat,eng)|None, "mram":(lat,eng)|None}
    base_cap: prefer 32 else max
    """
    os.makedirs(os.path.join(outdir, profile_name), exist_ok=True)
    # choose base cap
    caps = sorted(percap.keys())
    if base_cap is None:
        base_cap = 32 if 32 in percap else (caps[len(caps)//2] if caps else None)
    if base_cap is None:
        return False

    def pack(lat_eng_sram, lat_eng_mram):
        # fallbacks if a side missing for some node
        s_lat, s_eng = lat_eng_sram if lat_eng_sram else ({"t_sram_hit":16},{"e_rd_pj":300,"e_wr_pj":400,"leak_mw_per_mb":5.0})
        m_lat, m_eng = lat_eng_mram if lat_eng_mram else ({"t_mram_rd":28,"t_mram_wr":60},{"e_rd_pj":600,"e_wr_pj":900,"leak_mw_per_mb":0.6})
        return {
            "f_clk_ghz": f_clk_ghz,
            "latency": { **s_lat, **m_lat, "t_miss_cycles": int(miss_cycles) },
            "sram": s_eng,
            "mram": m_eng,
            "dram": { "e_miss_pj": 3000.0, "e_miss_per_ns_pj": 0.0 }
        }

    # base.json
    base_rec = percap.get(base_cap, {})
    base_json = pack(base_rec.get("sram"), base_rec.get("mram"))
    with open(os.path.join(outdir, profile_name, "base.json"), "w") as f:
        json.dump(base_json, f, indent=2)

    # per-cap overrides
    for cap in caps:
        if cap == base_cap: continue
        rec = percap[cap]
        cap_json = pack(rec.get("sram"), rec.get("mram"))
        with open(os.path.join(outdir, profile_name, f"L3_{cap}.json"), "w") as f:
            json.dump(cap_json, f, indent=2)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with NVSim YAMLs")
    ap.add_argument("--out", default="devices", help="devices dir (repo relative)")
    ap.add_argument("--prefix", default="nvsim", help="profile name prefix")
    ap.add_argument("--prefer", default="ReadEDP", help="preferred optimization tag")
    ap.add_argument("--miss-cycles", type=int, default=200, help="t_miss_cycles to use")
    ap.add_argument("--sram-lat-choice", choices=["read","write","max","mean"], default="read")
    ap.add_argument("--fallback-mram-node", type=int, default=None, help="e.g., 32 to borrow MRAM from n32")
    args = ap.parse_args()

    metas = find_meta(args.src)
    if not metas:
        raise SystemExit("No NVSim YAMLs found.")

    # group by (node -> opt -> cap -> path)
    by_node = defaultdict(lambda: defaultdict(dict))
    for m in metas:
        by_node[m.node_nm][m.opt or ""][m.cap_mb] = m.path

    # pre-extract fallback MRAM per-cap (if requested)
    fallback_mram = {}
    if args.fallback_mram_node and args.fallback_mram_node in by_node:
        # pick an opt for the fallback node
        fb_opt = pick_opt(by_node[args.fallback_mram_node], args.prefer)
        for cap, p in by_node[args.fallback_mram_node][fb_opt].items():
            ex = extract_from_yaml(p, args.sram_lat_choice)
            fallback_mram[cap] = ex["mram"]

    made = []
    for node, opt_map in sorted(by_node.items()):
        opt = pick_opt(opt_map, args.prefer)
        paths = opt_map[opt]  # cap -> path

        percap = {}
        for cap, p in paths.items():
            ex = extract_from_yaml(p, args.sram_lat_choice)
            percap[cap] = {
                "sram": ex["sram"],
                "mram": ex["mram"] if ex["mram"] else (fallback_mram.get(cap) if fallback_mram else None)
            }

        prof = f"{args.prefix}_n{node}_{opt.lower() or 'default'}"
        ok = write_profile(args.out, prof, percap, base_cap=32, miss_cycles=args.miss_cycles, f_clk_ghz=1.0)
        if ok: made.append((prof, sorted(percap.keys())))

    print("[ingest] created profiles:")
    for name, caps in made:
        print(f"  - {name}  caps={caps}")

if __name__ == "__main__":
    main()

