import re, subprocess

# === CONFIG ===
TR = "/home/skataoka26/traces/speccpu"
TRACE = f"{TR}/602.gcc_s-1850B.champsimtrace.xz"   # change if you want a different single benchmark

# device latencies (offline timing)
T_SRAM_HIT = 16.0
T_MRAM_RD  = 28.0
T_MRAM_WR  = 60.0

def run(pi_miss):
    cmd = ["bin/champsim",
           "--warmup-instructions", "2000000",
           "--simulation-instructions", "5000000",
           "--hybrid-llc",
           "--pi-miss", str(pi_miss),
           "--t-sram-hit", str(int(T_SRAM_HIT)),
           "--t-mram-rd",  str(int(T_MRAM_RD)),
           "--t-mram-wr",  str(int(T_MRAM_WR)),
           TRACE]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

    # parse LLC average miss latency (as DRAM path)
    tDR = 200.0
    m = re.search(r"LLC AVERAGE MISS LATENCY:\s+([\d\.]+)\s+cycles", out)
    if m: tDR = float(m.group(1))

    # parse the hybrid counters line
    m2 = re.search(
        r"\[LLC\]\[HYB\].*SRAM_HIT_RD:(\d+)\s+SRAM_HIT_WR:(\d+)\s+MRAM_HIT_RD:(\d+)\s+MRAM_HIT_WR:(\d+)\s+MISS_RD:(\d+)\s+MISS_WR:(\d+)",
        out)
    if not m2:
        raise RuntimeError("hybrid counters line not found in output")
    srd, swr, mrd, mwr, msrd, mswr = map(int, m2.groups())

    # total cycles from ROI line
    cyc = None
    m3 = re.search(r"CPU 0 cumulative IPC:\s+[\d\.]+\s+instructions:\s+\d+\s+cycles:\s+(\d+)", out)
    if m3: cyc = int(m3.group(1))

    # rough timing deltas (no MLP yet)
    dM_rd  = T_MRAM_RD - T_SRAM_HIT
    dM_wr  = T_MRAM_WR - T_SRAM_HIT
    dMiss  = tDR        - T_SRAM_HIT
    MLP_hit  = 1.0
    MLP_miss = 4.0

    stall = (mrd*dM_rd + mwr*dM_wr)/MLP_hit + ((msrd+mswr)*dMiss)/MLP_miss
    return {"pi_miss":pi_miss, "stall":stall, "cycles":(cyc or 1)}

if __name__ == "__main__":
    print("pi_miss,stall_over_cycles")
    for i in range(0, 11):
        r = run(i/10.0)
        print(f"{r['pi_miss']},{r['stall']/r['cycles']}")
