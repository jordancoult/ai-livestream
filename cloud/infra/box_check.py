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
    a = ap.parse_args()
    dev = torch.cuda.get_device_properties(0)
    RES["gpu"] = dev.name
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
    rules = [
        ("gpu", "RTX 5090" in RES["gpu"], True, RES["gpu"]),
        ("gemm_tflops >= 205 (sustained, 20 s)", RES["gemm_tflops"] >= 205, True, RES["gemm_tflops"]),
        ("attn_tflops >= 100", RES["attn_tflops"] >= 100, True, RES["attn_tflops"]),
        ("mem_gbs >= 1300", RES["mem_gbs"] >= 1300, True, RES["mem_gbs"]),
        ("h2d_gbs >= 10 (PCIe; frames are small)", RES["h2d_gbs"] >= 10, True, RES["h2d_gbs"]),
        ("cpu_single_s <= 0.45", RES["cpu_single_s"] <= 0.45, True, RES["cpu_single_s"]),
        ("ram_gb >= 48", RES["ram_gb"] >= 48, True, RES["ram_gb"]),
        ("disk_free_gb >= 100 (before the install)", RES["disk_free_gb"] >= 100, False, RES["disk_free_gb"]),
        ("power limit = card default (no host cap)", (RES["power_limit_w"] or 0) >= 0.95 * (RES["power_default_w"] or 1), True,
         f"{RES['power_limit_w']} / {RES['power_default_w']} W"),
        ("load_avg < 4 (no other tenant)", RES["load_avg"] < 4, False, RES["load_avg"]),
    ]
    fail = False
    for name, ok, hard, val in rules:
        tag = "PASS" if ok else ("FAIL" if hard else "WARN")
        fail |= (not ok) and hard
        print(f"  {tag:4s}  {name:44s} {val}")
    RES["verdict"] = "REJECT" if fail else "ACCEPT"
    print(f"VERDICT {RES['verdict']}  {json.dumps(RES)}")
    if a.json:
        open(a.json, "w").write(json.dumps(RES, indent=2))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
