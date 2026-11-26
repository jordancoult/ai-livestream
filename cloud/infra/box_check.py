"""Box acceptance check: run on a freshly rented GPU box BEFORE the ~20 min JoyAI install (cloud/infra/box_check.sh
runs it over SSH). ~1 minute. Measures what JoyAI's speed depends on and prints PASS / WARN / FAIL per item and a
verdict (exit 0 = accept, 1 = reject). Thresholds: cloud/infra/README.md "Box acceptance".

  python3 box_check.py [--json out.json]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import torch

RES = {}

# Requirements per pipeline (bring_up.sh passes its pipeline name). A threshold of None = not checked.
#   power_default: the host must not cap the GPU's power (a cap throttles clocks: JoyAI -15 %)
PROFILES = {
    # JoyAI live character swap: 8-frame chunks must stay >= ~25 fps (box-local ~27 on a good 5090)
    "joyai": dict(gpus=["RTX 5090"], vram_gb=30, power_default=True, gemm=205, attn=100, mem=1300, h2d=10,
                  cpu_s=0.45, ram_gb=48, disk_gb=100),
    # LongLive 1 relay: same GPU class and speed needs as JoyAI (per-chunk GPU time = the laptop's budget)
    "ll": dict(gpus=["RTX 5090"], vram_gb=30, power_default=True, gemm=205, attn=100, mem=1300, h2d=10,
               cpu_s=0.45, ram_gb=48, disk_gb=150),
    # LongLive 2.0: native 1280x704 at >= 25 fps (memory: ll2-native-resolution); heavier weights
    "ll2": dict(gpus=["RTX 5090"], vram_gb=30, power_default=True, gemm=205, attn=100, mem=1300, h2d=10,
                cpu_s=0.45, ram_gb=64, disk_gb=150),
    # voice only (RVC, ~17 ms per 200 ms block on a laptop 5080): any recent NVIDIA GPU, a power cap is fine
    "voice": dict(gpus=None, vram_gb=8, power_default=False, gemm=30, attn=None, mem=300, h2d=3,
                  cpu_s=0.6, ram_gb=16, disk_gb=30),
}


def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                          text=True).stdout.strip()


def timed(fn, sec):
    fn(); torch.cuda.synchronize()
    t0, n = time.time(), 0
    while time.time() - t0 < sec:
        fn(); n += 1
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--profile", default="joyai", choices=sorted(PROFILES))
    a = ap.parse_args()
    P = PROFILES[a.profile]
    RES["profile"] = a.profile
    dev = torch.cuda.get_device_properties(0)
    RES["gpu"] = dev.name
    RES["vram_gb"] = round(dev.total_memory / 2 ** 30)
    q = subprocess.run(["nvidia-smi", "-q", "-d", "POWER"], capture_output=True, text=True).stdout
    def watt(key):
        for ln in q.splitlines():
            if key in ln and "W" in ln:
                try:
                    return float(ln.split(":")[1].split()[0])
                except ValueError:
                    pass
        return None
    RES["power_limit_w"], RES["power_default_w"] = watt("Current Power Limit"), watt("Default Power Limit")
    RES["pcie"] = smi("pcie.link.gen.max,pcie.link.width.current")
    # 1) sustained tensor math (DiT GEMMs) -- 20 s so a power/thermal cap shows
    n = 8192
    A = torch.randn(n, n, device="cuda", dtype=torch.bfloat16); B = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    dt = timed(lambda: [A @ B for _ in range(10)], 20) / 10
    RES["gemm_tflops"] = round(2 * n ** 3 / dt / 1e12)
    RES["sm_clock_load_mhz"], RES["power_load_w"] = [float(x) for x in smi("clocks.sm,power.draw").split(", ")]
    # 2) attention (the DiT's other half; memory + tensor)
    qk = torch.randn(1, 24, 8192, 128, device="cuda", dtype=torch.bfloat16)
    dt = timed(lambda: torch.nn.functional.scaled_dot_product_attention(qk, qk, qk), 5)
    RES["attn_tflops"] = round(4 * 24 * 8192 ** 2 * 128 / dt / 1e12)
    # 3) GPU memory bandwidth (VAE, norms, elementwise)
    x = torch.empty(512 * 2 ** 20, device="cuda", dtype=torch.uint8); y = torch.empty_like(x)
    dt = timed(lambda: y.copy_(x), 3)
    RES["mem_gbs"] = round(2 * x.numel() / dt / 1e9)
    # 4) PCIe host <-> GPU (frames in/out every chunk)
    h = torch.empty(256 * 2 ** 20, dtype=torch.uint8).pin_memory()
    d = torch.empty(256 * 2 ** 20, device="cuda", dtype=torch.uint8)
    RES["h2d_gbs"] = round(h.numel() / timed(lambda: d.copy_(h, non_blocking=True), 2) / 1e9, 1)
    RES["d2h_gbs"] = round(h.numel() / timed(lambda: h.copy_(d, non_blocking=True), 2) / 1e9, 1)
    # 5) host CPU: single-thread speed (the server's Python, frame decode / encode)
    t0 = time.perf_counter(); s = 0
    for i in range(3_000_000):
        s += i * i % 7
    RES["cpu_single_s"] = round(time.perf_counter() - t0, 3)
    RES["cpu_cores"] = os.cpu_count()
    RES["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30)
    RES["disk_free_gb"] = round(shutil.disk_usage("/workspace" if os.path.isdir("/workspace") else "/").free / 2 ** 30)
    RES["load_avg"] = round(os.getloadavg()[0], 1)

    # verdicts (thresholds documented in cloud/infra/README.md)
    pw_ok = (RES["power_limit_w"] or 0) >= 0.95 * (RES["power_default_w"] or 1)
    rules = [   # (name, ok, hard, value)
        (f"gpu in {P['gpus'] or 'any'}", not P["gpus"] or any(g in RES["gpu"] for g in P["gpus"]), True, RES["gpu"]),
        (f"vram_gb >= {P['vram_gb']}", RES["vram_gb"] >= P["vram_gb"], True, RES["vram_gb"]),
        ("power limit = card default (no host cap)", pw_ok, P["power_default"], f"{RES['power_limit_w']} / {RES['power_default_w']} W"),
        (f"gemm_tflops >= {P['gemm']} (sustained, 20 s)", RES["gemm_tflops"] >= P["gemm"], True, RES["gemm_tflops"]),
        (f"attn_tflops >= {P['attn']}", P["attn"] is None or RES["attn_tflops"] >= P["attn"], True, RES["attn_tflops"]),
        (f"mem_gbs >= {P['mem']}", RES["mem_gbs"] >= P["mem"], True, RES["mem_gbs"]),
        (f"h2d_gbs >= {P['h2d']} (PCIe)", RES["h2d_gbs"] >= P["h2d"], True, RES["h2d_gbs"]),
        (f"cpu_single_s <= {P['cpu_s']}", RES["cpu_single_s"] <= P["cpu_s"], True, RES["cpu_single_s"]),
        (f"ram_gb >= {P['ram_gb']}", RES["ram_gb"] >= P["ram_gb"], True, RES["ram_gb"]),
        (f"disk_free_gb >= {P['disk_gb']} (before the install)", RES["disk_free_gb"] >= P["disk_gb"], False, RES["disk_free_gb"]),
        ("load_avg < 4 (no other tenant)", RES["load_avg"] < 4, False, RES["load_avg"]),
    ]
    fail = False
    for name, ok, hard, val in rules:
        tag = "PASS" if ok else ("FAIL" if hard else "WARN")
        fail |= (not ok) and hard
        print(f"  {tag:4s}  {name:44s} {val}")
    RES["verdict"] = "REJECT" if fail else "ACCEPT"
    print(f"VERDICT {RES['verdict']} (profile {a.profile})  {json.dumps(RES)}")
    if a.json:
        open(a.json, "w").write(json.dumps(RES, indent=2))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
