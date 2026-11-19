import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
for k in a:
    for i, (x, y) in enumerate(zip(a[k], b[k])):
        if x is None: continue
        print(f"{k:24s} {'img' if i == 0 else 'txt'}: bit-identical={torch.equal(x, y)} maxdiff={(x - y).abs().max().item():.3g}")
