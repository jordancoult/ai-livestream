# Cloud GPU boxes (Vast.ai)

## Renting a box: rules (2026-11-26)

Never install onto a box that hasn't passed the acceptance check. The flow:

```bash
bash cloud/infra/vast.sh search                    # US/CA RTX 5090 offers, cheapest first; skips vast_blacklist.txt
bash cloud/infra/box_trial.sh OFFER_ID             # rent + ~1 min acceptance check: ACCEPT keeps it (.pod.trial),
                                                   # REJECT destroys it and blacklists its machine_id with the reason
cp cloud/infra/.pod.trial cloud/infra/.pod.joyai   # (keep the previous one as .pod.joyai_old until the new one works)
bash cloud/infra/bring_up.sh joyai                 # reruns the check first (FORCE=1 to override), then installs (~20 min)
bash cloud/infra/voice_up.sh joyai                 # the voice stage (RVC) from the private HF repo
POD_FILE=cloud/infra/.pod.joyai bash cloud/infra/tunnel.sh start
```

Choosing an offer: prefer the region network-closest to where the laptop is (every JoyAI round trip crosses the
network twice: from Medellín, Texas/Florida beat California's ~105 ms), then price. Avoid fleets with a blacklisted
machine (adjacent machine IDs, same RAM/host pattern).

## Box acceptance (`cloud/infra/box_check.py`, run by `box_check.sh` / `box_trial.sh` / `bring_up.sh`)

Hard rules (any FAIL = reject):

| Check | Threshold | Why |
|---|---|---|
| GPU | RTX 5090 | the pipelines are tuned for it |
| Power limit | = the card's default (no host cap) | a capped 5090 throttles its clocks: 400 W cap = 2.3 GHz sustained, JoyAI 23.4 fps box-local vs 27.3 |
| Sustained bf16 GEMM (20 s) | >= 205 TFLOPS | uncapped boxes give ~227; capped ones 192-200 |
| Attention (SDPA) | >= 100 TFLOPS | the DiT's other half |
| GPU memory bandwidth | >= 1300 GB/s | VAE, norms, elementwise |
| PCIe host -> GPU | >= 10 GB/s | frames in/out per chunk (small; gen5 x16 = ~52) |
| CPU single thread | <= 0.45 s on the loop | the server's Python, frame decode/encode |
| RAM | >= 48 GB | model loading |

Warnings (shown, not rejecting): disk free < 100 GB before the install, load average >= 4 (another tenant?).

Also check after the install: the box-local JoyAI benchmark (`cloud/pipelines/joyai/bench/bench_box.sh`) should
give ~27 fps (base model, its sample reference).

`vast_blacklist.txt`: machine IDs never to rent, with date and reason. `vast.sh check-gpu`: the power-limit check
alone.

## Measured boxes

| Machine | Where | Verdict | Notes |
|---|---|---|---|
| 156320 | California | REJECT | 400/575 W cap: GEMM 192 TFLOPS @ 2.3 GHz, JoyAI 23.4 fps box-local (DiT 302 ms/chunk) |
| 147713 | Texas | REJECT | 450/600 W cap: GEMM 200 TFLOPS @ 2.4 GHz, PCIe 14 GB/s |
| 136976 | Texas | ACCEPT | 575/575 W, GEMM 227 TFLOPS, PCIe 52.6 GB/s, CPU 0.093 s |

Other files: `vast.sh` (search / create / show / destroy), `pod.sh` (ssh/scp to the current box), `tunnel.sh`
(laptop:18080 -> box:8080), `providers.md` (provider notes), `artifacts.md` (what lives on the HF artifacts repo).
