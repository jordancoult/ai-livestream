# FP8 fast-accum GEMM tile sweep at the DiT serving shapes (RTX 5090). Keeps only configs whose output is
# bit-identical to the current table; prints current vs best vs cuBLAS. Run on the box: /root/venv/bin/python gemm_bench.py
import os, sys, time, itertools
os.environ["JOYOMNI_FP8_FAST_ACCUM"] = "1"
sys.path.insert(0, "/workspace/JoyAI-Video-Edit/deploy")
import torch, triton
from xvideo.models.dit import fp8_linear as F
torch.manual_seed(0)
dev = "cuda"
def bench(fn, it=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1000  # us
shapes = [(700, 12288, 4096), (700, 4096, 4096), (700, 16384, 4096), (700, 4096, 16384),
          (1376, 12288, 4096), (1376, 4096, 4096), (1376, 16384, 4096), (1376, 4096, 16384),
          (347, 12288, 4096), (347, 4096, 4096), (347, 16384, 4096), (347, 4096, 16384)]
configs = [(bm, bn, bk, w, st, g) for bm in (64, 128) for bn in (128, 256) for bk in (128, 256) for w in (4, 8) for st in (2, 3, 4) for g in (8,)]
tot_cur = tot_best = 0
for M, N, K in shapes:
    lin = torch.nn.Linear(K, N, bias=True, device=dev, dtype=torch.bfloat16)
    fl = F.Fp8Linear.from_linear(lin)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    xq = torch.empty((M, K), device=dev, dtype=F.FP8_DTYPE); xs = torch.empty((M, 1), device=dev, dtype=torch.float32)
    F._fa_quant_kernel[(M,)](x, xq, xs, M, K, x.stride(0), x.stride(1), BLOCK=1024, QMAX=F._FA_QMAX_A, num_warps=4)
    y = torch.empty((M, N), device=dev, dtype=torch.bfloat16)
    def run(cfg):
        bm, bn, bk, w, st, g = cfg
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        F._fa_mm_kernel[grid](xq, fl.weight_fp8, y, xs.view(-1), fl.weight_scale, fl.bias, M, N, K,
            xq.stride(0), xq.stride(1), fl.weight_fp8.stride(0), fl.weight_fp8.stride(1), y.stride(0), y.stride(1),
            HAS_BIAS=True, BM=bm, BN=bn, BK=bk, GROUP=g, PROMOTE=max(1, 512 // bk), num_warps=w, num_stages=st)
    cur = (*F._fa_pick_config(M, N, K), 8)   # (BM, BN, BK, warps, stages, GROUP)
    run(cur); ref = y.clone()
    t_cur = bench(lambda: run(cur))
    best = (t_cur, cur)
    for cfg in configs:
        try:
            run(cfg)
            if not torch.equal(y, ref):   # must be bit-identical to stay lossless
                d = (y.float() - ref.float()).abs().max().item()
                if d > 0: continue
            t = bench(lambda: run(cfg), it=20)
            if t < best[0]: best = (t, cfg)
        except Exception as ex:
            pass
    # cuBLAS FP8 for reference (different accumulation -> not bit-identical; speed reference only)
    try:
        wq = fl.weight_fp8; t_cb = bench(lambda: torch._scaled_mm(xq, wq, scale_a=xs, scale_b=fl.weight_scale.view(1, -1), out_dtype=torch.bfloat16, use_fast_accum=True))
    except Exception as ex:
        t_cb = float("nan")
    fl_ = 2 * M * N * K
    tot_cur += t_cur; tot_best += best[0]
    print(f"M={M:5d} N={N:5d} K={K:5d}  cur {t_cur:7.1f}us {fl_/t_cur/1e6:6.0f}TF | best {best[0]:7.1f}us {fl_/best[0]/1e6:6.0f}TF {best[1]} | cuBLAS {t_cb:7.1f}us {fl_/t_cb/1e6:6.0f}TF", flush=True)
print(f"sum cur {tot_cur:.0f}us best {tot_best:.0f}us -> {100*(1-tot_best/tot_cur):.1f}% faster")
