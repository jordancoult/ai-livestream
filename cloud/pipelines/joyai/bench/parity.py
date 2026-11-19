# Parity test for DiT speed patches: one real-size MMDoubleStreamBlock (FP8 fast-accum + SageAttention, as served on
# the 5090) with fixed random weights/inputs/KV cache. Run once with the original tree and once with the patched one,
# then parity_cmp.py a.pt b.pt must print bit-identical=True everywhere. usage: parity.py <deploy_dir> <out.pt>
# (original tree: git worktree add --detach /root/orig <upstream-or-previous-commit>)
import os, sys
os.environ.update(JOYOMNI_FP8_FAST_ACCUM="1", JOYOMNI_SAGE_ATTN="1", JOYOMNI_FP8_IMG="1", JOYOMNI_FP8_TXT="1")
sys.path.insert(0, sys.argv[1])
import torch
from xvideo.models.dit.dit import MMDoubleStreamBlock
torch.manual_seed(0)
dev, dt = "cuda", torch.bfloat16
D, H, hd = 4096, 32, 128
blk = MMDoubleStreamBlock(D, H, 4.0, dtype=dt, device=dev).to(device=dev, dtype=dt).eval()
with torch.no_grad():
    for n, p in blk.named_parameters():
        p.copy_(torch.randn_like(p, dtype=torch.float32).mul_(0.02 if p.dim() > 1 else 0.1).to(dt))
res = {}
def freqs(L):
    ang = torch.randn(L, hd // 2, device=dev).repeat_interleave(2, -1)
    return (ang.cos(), ang.sin())
for name, Li, Lt, nkv, skip in [("img700_cache", 700, 347, 3, False), ("img1376_cache", 1376, 404, 2, False),
                                ("img700_nocache", 700, 347, 0, False), ("img700_skiptxt_cache", 700, 0, 3, True),
                                ("img700_skiptxt_nocache", 700, 0, 0, True)]:
    g = torch.Generator(device=dev).manual_seed(sum(map(ord, name)))
    img = torch.randn(1, Li, D, device=dev, dtype=dt, generator=g)
    txt = torch.randn(1, max(Lt, 1), D, device=dev, dtype=dt, generator=g)
    vec = torch.randn(1, D, device=dev, dtype=dt, generator=g) * 0.1
    entries = [{"key": torch.randn(1, 700, H, hd, device=dev, dtype=dt, generator=g),
                "value": torch.randn(1, 700, H, hd, device=dev, dtype=dt, generator=g)} for _ in range(nkv)]
    kw = dict(vis_freqs_cis=freqs(Li), layer_idx=0, skip_text_stream=skip)
    if nkv:
        kw["kv_cache_reader"] = lambda i, e=entries: e
    with torch.no_grad():
        oi, ot = blk(img, txt, vec, **kw)
    res[name] = (oi.float().cpu(), None if ot is None else ot.float().cpu())
torch.save(res, sys.argv[2]); print("saved", sys.argv[2])
