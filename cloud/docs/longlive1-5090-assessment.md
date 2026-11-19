# LongLive 1 (Wan2.1-1.3B) on a rented RTX 5090: assessment (2026-11-19)

The question: what speed and quality do we gain by running NVlabs LongLive v1 on a rented RTX 5090 (32 GB, about 575 W) instead of the power-limited RTX 5080 Laptop (16 GB, 100–130 W)? The use case is webcam V2V at high noise (0.9–1.0): the user's pose and appearance move into prompt-defined worlds, at 25–30 fps with under 2 s glass-to-glass (g2g). This is analysis only. Nothing was rented or run.

Labels: **[meas]** = measured in this repo (source given). **[est]** = my estimate (reasoning shown). **[pub]** = published by others.

## 0. Correcting the baseline first

The laptop speed ladder (TAE fix 3.6→25, fp8 →29.8, no syncs →31.5, compiled blocks →33.8, fused KV →37.4, compiled TAE →38.7) is real, but it was measured on **StreamDiffusionV2** at 576×320, 1 step (`README_QUICKSTART.md`, "How fast is it?"). It is not a LongLive number. LongLive inherits most of those patches (fp8 row-wise, fused clean pass, compile, TAE) and is slower. At 640×352 through Scope plus the bridge, on the desk clip, it measures [meas, README_QUICKSTART model status, `experiments/2026-11-01-ll1-speed`]:

| LongLive 1 on the laptop | depth off | depth on |
|---|---|---|
| 1 step (+ clean pass) | 21.3 fps | 17.4 fps |
| 3 steps, clean every 2nd | 13.4 | 10.1 (11.4 with VACE hint reuse) |
| 2 steps, dream presets | — | ~12 |

- **Cost model.** Cost is linear in DiT passes per chunk, about 230 ms per pass at 640×352 (one chunk = 3 latents = 12 frames). Depth/VACE adds 20–45%.
- **Memory limits on 16 GB.** The model must be fp8 (bf16 + LoRA + VACE does not fit). 768×432 runs at about 11 fps using 15 of 16 GB, and 832×480, LongLive's native size, runs out of memory (`experiments/2026-10-31-dream-variants`).
- **The trained config is not what we run.** LongLive was trained at 832×480 with 4 steps (1000/750/500/250), sink 3 and a 12-latent window (`longlive2/LongLive1.0/configs/longlive_inference.yaml`). The laptop runs it below its native size and with fewer steps.

## 1. Speed headroom on the 5090

**Scaling factor, 2.4–3.2× per DiT pass, central 2.7× [est].** The best evidence is LongLive 2.0 run on both machines with the same code and NVFP4 KV:

| Same LL2 config | Laptop | 5090 | Ratio |
|---|---|---|---|
| 1024×576, 2 steps, clean every chunk, window 32 | 10.1 (≈11.1 with compile) | 31.7 | 2.9× |
| 1280×704, 2 steps, compiled | 5.6 | 18.5 | 3.3× |
| 1024×576, laptop window 24 vs 5090 window 32 | 12.7 | 31.7 | ≥2.5× |

Sources: `experiments/2026-11-01-longlive2`, `experiments/2026-11-04-longlive2-5090` [meas]. The spec ratio points the same way: about 2.5× the tensor throughput and 2× the memory bandwidth (1.79 vs 0.9 TB/s), with no 100 W clock cap. Published cross-check: LongLive does 20.7 fps (bf16) and 24.8 fps (fp8) on an H100 at 832×480, 4 steps [pub]. A 5090 typically reaches about 0.6× an H100, which gives 12–15 fps. That is consistent with the table below.

**Projected LongLive 1 on the 5090, fp8, compiled, fused KV, TAE [est].** Assumptions: 85 ms per pass at 640×352. Tokens are 1.47× at 768×432 and 1.77× at 832×480, plus a little extra for attention. Fixed TAE and miscellaneous cost is 40–65 ms per chunk.

| Size | Steps / clean pass | Passes per chunk | fps, depth off | fps, depth/VACE on |
|---|---|---|---|---|
| 640×352 | 1 / every chunk (today's recipe) | 2 | ~55 (camera-capped at 30) | ~40 |
| 640×352 | 2 / every chunk | 3 | ~40 | ~30 |
| 640×352 | 4 / every chunk (as trained) | 5 | ~26 | ~19 |
| 768×432 | 2 / every 2nd | 2.5 | ~32 | **~25** |
| **832×480 (native)** | 1 / every chunk | 2 | ~32 | ~24 |
| **832×480 (native)** | 2 / every 2nd | 2.5 | ~26 | ~20 |
| 832×480 | 4 / every chunk | 5 | ~14 | ~10 |

Two adjustments apply to every row:
- **bf16 instead of fp8:** about −15% (the fp8 step was +19% on SDv2).
- **Full Wan2.1 VAE instead of TAE/lighttae:** an estimated −20–30% at 832×480. On LL2 the full VAE decoded at 10 fps against 1300 fps for the TAE, and looked "almost the same".

**Bottom line on speed.** With the same recipe, the 5090 has roughly 2× more fps than needed, but the camera caps output at 30 fps. That surplus is worth about one of these, not all of them:
- **native resolution**, or
- **2 steps instead of 1**, or
- **depth/VACE kept on**.

Any two of them land at about 20–25 fps. Four steps (as trained) at native resolution does not reach 25 fps.

**Where to spend the headroom for "transport me into another world" (high noise):**

| Lever | Expected quality effect | Evidence | Rank |
|---|---|---|---|
| **Depth→VACE always on** | Pose adherence at noise 1.0. Without depth, noise ≥ 0.93 loses the body ("boxer turns into fish") | dream-variants [meas]; SDv2 adherence 0.32→0.47 with depth [meas] | 1, a prerequisite |
| **Resolution 768×432 → 832×480** | Crisper small/placed subject; faces get more pixels (a 40 px face at 640×352 with 50% placement limited mouth detail). 832×480 is the training size, so less drift is likely | 768 "crisper" [meas]; 832 untested (OOM on the laptop) | 2 |
| **2 steps (from 1)** | Sharper, more world detail when generating from near-pure noise. Does **not** fix ghosting on fast punches (that comes from motion within the 12-frame chunk) | "3 steps: sharper" [meas]; SDv2 realism 0.53→0.77 at 2 steps [meas] | 3 |
| Clean pass every chunk (not every 2nd) | Temporal consistency over long runs | LL2: every 2nd ≈ every chunk [meas] | low |
| bf16 (no fp8) | Unknown. fp8 row-wise is per-channel and looked fine; the measured damage came from FP4 (adherence 0.38→0.1) | No A/B exists | test cheaply |
| Full VAE | Small: TAE ≈ full VAE on LL2 [meas]; lighttae is what LongLive expects | — | low |
| Bigger KV window | Risky. LongLive was trained with 12 + sink 3. Shorter windows hurt (3 frames: adherence 0.38→0.30; LL2 at 16 got blobs), and longer is out of distribution | — | low, experimental only |
| Larger LongLive variant | **None exists.** LongLive v1 is 1.3B only. Its siblings (Reward/Causal Forcing, MemFlow, OPSD-V LoRA) are all 1.3B. The "bigger LongLive" is LL2-5B, already tested | — | n/a |

**Identity (keeping *your* appearance) is the weak axis, and the 5090 does not fix it.** Reference-image identity is weak in streaming 1.3B models [meas]. The working levers are the identity hint (0.3, inside the person mask), a self-description in the prompt, and sink 3. More pixels on the face and 2 steps should help detail, not identity transfer.

## 2. Latency

| Component | Local laptop (LongLive) | 5090 (LongLive 1) [est] | Reference points |
|---|---|---|---|
| Chunk fill (12 frames at 30 fps) | 0.40 s | 0.40 s | LL2 needs 32 frames = 1.33 s |
| Generation of one chunk | ~0.56 s (at 21 fps) | 0.40–0.48 s at 25–30 fps (less if faster than real time) | JoyAI 0.30 s / 8 frames [meas] |
| Uplink + downlink + codec | ~0.1 s (WebRTC on localhost) | 0.15 s (US host) to 0.23 s (EU) | JoyAI laptop→EU added ~0.23 s [meas] |
| Playout/jitter buffer | cushion 3 frames | 0.3–0.45 s (12-frame bursts need a deeper buffer than JoyAI's 6) | JoyAI buffer ≈ 0.34 s [meas] |
| **Total g2g** | **~1.0–1.1 s [est, never measured for LongLive]** | **~1.3–1.6 s** | JoyAI 1.03 s; LL2 3.0 s; SDv2 local 0.46 s [meas] |

- **Is <2 s realistic? Yes, with margin.** LongLive's 3-latent chunk is its big advantage over LL2, whose 8-latent chunk was the 3 s floor (4-latent chunks broke the image).
- **Bandwidth.** The 3–9 Mbit/s upload figure comes from the brief; I could not find it in the repo. H.264 at generation size (640×352–832×480, 30 fps, mostly static webcam) is about 1.5–4 Mbit/s at CRF 20 [est]. That fits, but at the 3 Mbit/s low end, use capped VBR (`-maxrate 2M -bufsize 1M`) instead of the bridge's plain `--crf 20`. MJPEG (16 Mbit/s) already failed once (`cloud/docs/lessons.md`). At noise 0.9–1.0 input detail is destroyed anyway, so a low uplink bitrate costs almost nothing in quality.
- **What lowers it:**
  - a US host (−0.1 s);
  - generation faster than real time (spend less of the headroom; every 10% slack trims queueing);
  - pacing on client consumption and dropping stale frames (LL2 lesson; a server clock let the buffer drift +2.5 s);
  - a tuned adaptive playout target;
  - sending camera frames already cropped/downscaled to generation size.

## 3. Infinite generation

**Already solved locally** (Scope patches 11–12, `scope` commits `32a94ab2`, `d556ae2d`):
- **Endless memory.** At the 1024-frame RoPE limit, stock Scope wiped the KV cache, a visible jump every ~3.4 min at 20 fps. The patch rotates the window keys back instead; the sink keeps its position. At 30 fps the stock jump would come every ~2.3 min, so the patch matters more on a faster GPU.
- **Forever noise counter.** Scenes no longer replay after a memory reset.
- **Other pieces:** keyframe memory (rotates fresh frames into the sink slots), sink 0/3 dial, stale-VACE clearing, and random noise by default (fixed noise piles up into confetti on LongLive).

**Still open:**
- **Prompt changes wipe memory.** `manage_cache: True`; LongLive's own KV re-cache would keep the scene. On LL2, re-caching was "essential", at ~0.55 s for 16 latents on the 5090.
- **Invented motion and cache feedback.** Invented motion stays at ~10× the camera's in every config (feedback through the attention cache, measured on SDv2), and a cache reset visibly cleans up accumulated degradation.
- **No long-run drift measurement for LongLive 1.** The longest runs were 50–80 s clips. JoyAI was checked over 2 min.

**What 32 GB adds:**
- bf16 KV and weights.
- Room for a larger sink/keyframe memory (e.g. identity keyframes from the start of the session), which is still within the trained attention pattern only if total tokens stay near 15 latents.
- T5 on the GPU, so prompt switches encode in ~20 ms (LL2 lesson).
- Depth and segmentation at full rate.

More VRAM does not by itself fix drift. That needs the re-cache test plus a 10–20 min soak with periodic contact sheets.

## 4. Practicalities

- **Cost.** Vast 5090 at $0.605/h, measured on the LL2 run (3.5 h ≈ $2.1, plus ~$0.2 downloads); RunPod $0.69 community / $1.19 secure. A one-day porting and test effort is about $3–6.
- **Reusable as is:**
  - `cloud/infra/vast.sh`, `pod.sh`, `tunnel.sh`;
  - `vast_watchdog.sh` + `vast_heartbeat.sh` (both are needed: the watchdog alone stopped a live box after 45 min);
  - the HF artifacts repo (sm_120 flash-attn/Sage wheels already built for torch 2.9.1 cu128);
  - `cloud/client/live_bridge.py` (H.264 over WebSocket, ack/ping flow control, adaptive jitter buffer, auto-reconnect, `--dashboard`);
  - `tools/bench_longlive.py`, `adherence.py`, `quality_metrics.py`, which run headless on the box.
- **Needs building (about 1.5–2.5 agent-days) [est]:**
  1. **`cloud/pipelines/longlive1/setup.sh`**: clone the Scope fork (`blackwell-realtime`), `uv sync`, pull ~25 GB of weights (LongLive-1.3B, Wan2.1 base/VACE/T5, autoencoders, Video-Depth-Anything). That is about half of JoyAI's 50 GB, so a fresh Vast host is ready in about 15 min. The frontend build is optional.
  2. **A box-side relay**: live_bridge's WebSocket H.264 → aiortc WebRTC to Scope on 127.0.0.1 (essentially `scope_bridge.py` with a WebSocket source instead of a camera) → output back over WebSocket. Scope's WebRTC needs UDP, which an SSH tunnel cannot carry. Vast does expose UDP, but NAT/ICE through a tunnel-free path is fragile.
  3. **The graph page and presets over the tunnel**: map Scope's port 8000 to e.g. 18000 locally, because laptop port 8000 means the user's own live session.

  The env-var patch set (`SCOPE_FP8_ROWWISE`, `SCOPE_FUSED_KV`, `SCOPE_COMPILE_*`, ...) carries over unchanged. The CPU side (aiortc, depth preprocessing) runs on a desktop CPU there, which is better than the E-core pinning on the laptop.

## 5. Hardware profiles (`HW_PROFILE=5080|5090`): keep the laptop exactly as it is

The user expects the biggest 5090 wins to come from **higher resolution** and **more diffusion steps**, and wants to try those first. The section 1 analysis agrees, with two caveats:
- Steps do not fix ghosting on fast motion.
- 4 steps at native resolution projects to only ~14 fps.

The hard requirement is that the 5080 laptop behaves exactly as it does today, so the design below is a pure bypass when the flag is absent.

### Where the 5080-tuned settings live today

| Layer | File | 5080-specific knobs |
|---|---|---|
| Workspace env | `env.sh` | `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `TORCHINDUCTOR_COMPILE_THREADS=2` (laptop CPU thermals), `__NV_PRIME_RENDER_OFFLOAD` (hybrid graphics), `USE_TAEHV=1` |
| Launcher | `start_scope.sh` | `SCOPE_FP8_ROWWISE=1` (fp8 forced by 16 GB on LongLive), `SCOPE_COMPILE_BLOCKS=1`, `SCOPE_COMPILE_TAE=1`, `SCOPE_FUSED_KV` per model (0 for longlive), `SCOPE_NOISE_T_OFFSET=0`, `SCOPE_PRELOAD_DEPTH=1`, output cushion `SCOPE_OUTPUT_TARGET_FRAMES` 14 / `SCOPE_SINK_QUEUE_MAX` 24 for the LongLive family, `SCOPE_INPUT_QUEUE_MAX=8`, `taskset -c 6-13` (E-cores), `thermal_watchdog.sh 90 103 10` |
| Bridge (presets) | `tools/scope_bridge.py` | `--size` default 576×320 (`--landscape` swaps it), `PRESETS` 30/24/20/15 (steps `[750]` / `[750,400]`, depth, fixed noise, bicubic-sharp upscale to 720×1280), `STEP_LISTS` 1–5, `--vae` (lighttae for LongLive), `--memory-window` → `local_attn_size`, page defaults (`clean_kv_every` 1, `depth_stride` 4, `attention_backend` default = Sage when installed) |
| Scope patches (env-read) | `scope/src/scope/core/pipelines/...` | Read `SCOPE_FP8_ROWWISE`, `SCOPE_FP4_FFN`, `SCOPE_COMPILE_*`, `SCOPE_FUSED_KV`, `SCOPE_KV_FRAMES`, `SCOPE_ROPE_SHIFT/LIMIT`, `SCOPE_NOISE_RESTART`, `SCOPE_SKIP_CLEAN_KV`, `DISABLE_SAGEATTENTION`. **There are no hardware checks in Scope itself; everything arrives through env vars and load params**, so Scope needs no change. |
| Pipeline defaults | `longlive/schema.py`, `model.yaml` | Stock values (576×320 schema default, steps 1000/750/500/250, `local_attn_size` 12, sink 3, VAE wan). The bridge always overrides these. |
| Saves | `saved_settings/*.json` | `bridge_state` (preset, steps, noise, depth, sink...) + `launch` (`model`, `size`, `env`). `/load` resets to defaults first. |

### Design

1. **One switch, read in one place.** Use `HW_PROFILE` env, or `./start_scope.sh --hw 5090` which sets it.
   - **Unset or `5080` = today's code path, byte for byte.** `start_scope.sh` sources nothing new, and the bridge gets no extra arguments.
   - **`5090`** sources `profiles/hw_5090.env` near the top of `start_scope.sh`, **before** the existing `export X="${X:-default}"` lines. Every existing line already respects a pre-set value, so the profile only pre-sets values and the 5080 defaults stay literally where they are. Explicit user env vars still win over the profile, because the profile file uses `${X:-...}` too.
   - This follows the memory rule (new features default to bypass): the laptop never reads the profile file.
2. **Engine knobs in the profile file** (env + `BRIDGE_ARGS` defaults):

   | Knob | 5080 (unchanged) | 5090 profile |
   |---|---|---|
   | Generation size | 576×320 / 640×352 (`--size`) | 768×432 default; 832×480 option (portrait 432×768 / 480×832) |
   | fp8 vs bf16 | `SCOPE_FP8_ROWWISE=1` | Keep 1 at first (speed); `0` = bf16 once the A/B in experiment 3 says it matters |
   | VAE | lighttae | lighttae (full `wan` only if the A/B shows a gain) |
   | Fused KV / clean-pass cadence | Fused off for longlive; `clean_kv_every` 1 | Fused on (bit-exact for LongLive, `check_fused_ll.py`); clean every 2nd at 2+ steps |
   | Compile | blocks + TAE, inductor threads 2 | Same flags, inductor threads ~8 (desktop/server CPU) |
   | KV window / sink | stock 12 / 3 | Same (trained values; longer windows only as an experiment) |
   | Output cushion | 14 frames | 14 (same chunking); tune with the cloud jitter buffer |
   | CPU pinning / thermal watchdog | `taskset -c 6-13`, `90 103 10` | No taskset (`PIN_CPUS=""`); GPU-only watchdog (the CPU sensor logic is laptop-specific) |
   | T5 placement | as today | On the GPU (fast prompt switches) |
3. **Presets per profile.** In `scope_bridge.py`, add `--hw` (default `5080`) choosing between the existing `PRESETS` dict (untouched) and a new `PRESETS_5090` dict. The slider keeps the same labels (30/24/20/15 = target fps), so the page and saves need no new UI.
4. **Saves record the profile.** New saves write `launch.hw`. A save without `hw` is a 5080 save and loads exactly as today. Loading a 5090 save on the laptop warns and keeps the laptop size; the bridge already checks size mismatches at line ~954. This matters because 832×480 runs out of memory on 16 GB.
5. **Guard.** On `--hw 5090`, check `nvidia-smi` for ≥ 30 GB and refuse otherwise. This matches the `setup.sh` sm_120 ≥ 32 GB check in `cloud/pipelines/joyai`.
6. **Verify the bypass:**
   - Run `bench_longlive.py` and `bench_scope.py` on the laptop with and without the patch, and diff the output with `clip_diff.py` (it must be bit-identical, same fps).
   - Diff the env that `start_scope.sh` exports with `HW_PROFILE` unset, before and after the change.

### Proposed 5090 presets and candidates (all fps are **[est]**)

Assumptions:
- Uses the section 1 model: fp8, compiled, fused, TAE, LongLive 1, 30 fps camera cap.
- Step lists come from the existing `STEP_LISTS`. In video mode the first timestep is set by the noise slider, so at noise 0.9–1.0 the lists act like the trained 1000/750/500/250 schedule.
- **bf16 ≈ ×0.85** on every row.

| Candidate | Size | Steps | Clean pass | fps, depth off | fps, depth on | Purpose |
|---|---|---|---|---|---|---|
| **5090 "30"** | 768×432 | 1 | every chunk | ~40 (capped at 30) | ~30 | fast; replaces today's 30 preset |
| **5090 "24"** | 768×432 | 2 | every 2nd | ~32 | **~25** | **default**: resolution + steps + depth together |
| **5090 "20"** | 832×480 (native) | 2 | every 2nd | ~26 | ~20 | native-res quality |
| **5090 "15"** | 640×352 or 768×432 | 3–4 | every 2nd | 640: ~36 (3) / ~28 (4); 768: ~24 (3) / ~19 (4) | 640: ~27 / ~21; 768: ~18 / ~14 | step-count test |
| native fast | 832×480 | 1 | every chunk | ~32 | ~24 | resolution-only test |
| as trained | 832×480 | 4 | every chunk | ~14 | ~10 | offline reference / demo only |

Two practical points:
- **Size changes need a pipeline reload.** The slider presets share one size so switching stays instant. So per profile, pick one launch size (768×432 suggested) and let the presets vary steps, depth and clean cadence. Leave 832×480 as a launch option (`--size 480 832`).
- **First test is a grid.** The headless sweep in experiment 2 is exactly this table. Run it first, and replace these estimates before freezing `PRESETS_5090`.

## 6. Against the alternatives already tested on the 5090

| | LongLive 1 on 5090 [est] | LongLive 2.0 (5B NVFP4) on 5090 [meas] | JoyAI-Video-Edit (16B) on 5090 [meas] |
|---|---|---|---|
| fps at the useful setting | 20–25 at 768–832 px with depth, 2 steps | 33 generated, 24 locked playout at 1024×576 | 25–27 at 840×480, 2 steps |
| g2g | ~1.3–1.6 s | **3.0 s** webcam in→out (over target) | **1.03 s** |
| High-noise world transport | Strong. Depth + noise 1.0 + physics prompts already work (`claude_immersed_*` presets) | V2V 0.90–0.98 works; person-keep 0.6 works. The first seconds are weak live, and prompt changes land partially | Untested. It is an instruction editor; "change the background to X, keep the person" is plausible but not run |
| Pose adherence | Good with depth (dedicated control) | Good at 0.90; rough at 0.98 | Good (gestures followed) |
| Own-appearance retention | Weak (prompt + identity hint) | Person-keep keeps you nearly exactly | Likely strongest (editor) but unmeasured for "keep me" |
| World fidelity / resolution | 1.3B at 480p: dreamy, flickery detail | Best (5B, 1024×576) | 840×480, edit-bound to the camera layout |
| Endless runs | Patched (RoPE rotate, noise counter) | Sink + re-cache; hard cut clears the sink | Memory auto-reset; 2 min clean |
| Setup | New (relay + setup.sh) | Done (`tools/longlive2`) | Done (`cloud/pipelines/joyai`) |

## 7. Recommendation

**Worth it, as a cheap (< $5, ~2 days) experiment, not as a sure win.**
- **It is the only candidate that combines all three:** radical prompt worlds driven by depth/pose, under 2 s latency, and 25 fps.
- **LongLive 2.0 is too slow on latency** (3 s, a chunk-size floor).
- **JoyAI is unproven for world transport.**
- **Expected gain:** going from today's 640×352, 1-step, ~17 fps depth preset to roughly 768×432–832×480 at 2 steps, ~20–25 fps, with depth.
- **Not expected:** stronger identity. That is a model limit.
- **Before porting, run one free check:** confirm the laptop's LongLive g2g (never measured) so the cloud comparison has a baseline.

**Top experiments, ranked by expected quality gain per effort:**

1. **JoyAI background/world instruction test (≈1 h, infra exists).** Run the desk clip with prompts like "place the person in an underwater coral reef, keep the person, pose and clothes". If identity is kept and the world is convincing, it may beat LongLive 1 outright at 1.03 s g2g. If it only recolors the room, LongLive 1 is the path.
2. **Headless LongLive 1 sweep on a Vast 5090 with `bench_longlive.py` (≈0.5 day, no relay needed).** This is the resolution × steps test the user wants first; it fills the section 5 preset table. Grid: {640×352, 768×432, 832×480} × {1, 2 steps} × depth on, fp8, clean every 2nd. Score each with `adherence.py` and `quality_metrics.py`, and make contact sheets for the user. This replaces every [est] in section 1 with numbers and picks the recipe.
3. **bf16 vs fp8 and lighttae vs full VAE A/B at the chosen size (≈1 h, same box).** Low effort; it settles whether to spend 15–30% fps on precision and decoder quality.
4. **Relay + live run, 2-min live and 15-min soak (≈1–1.5 days).** Measure g2g with the flash probe, stutters, and drift across a RoPE-limit rotation, with endless memory on. Use a US host and the capped uplink bitrate.
5. **`HW_PROFILE` switch (≈0.5 day, laptop-safe).** Build it once the sweep has picked the 5090 presets, and prove the bypass is bit-identical on the laptop (section 5).
6. **KV re-cache on prompt change instead of a wipe (≈0.5 day, works locally too).** This is what makes live world switches smooth during an endless stream. LL2 showed it is the difference between "the prompt barely shows" and real changes.

Missing evidence to keep in mind:
- No LongLive 1 number exists on any 5090.
- No laptop LongLive g2g measurement exists.
- No bf16 vs fp8 quality A/B exists.
- No LongLive 1 soak longer than ~80 s exists.
- JoyAI has not been tested for keeping the user's own identity in a new world.
- The 3–9 Mbit/s uplink figure comes from the brief and is not recorded in the repo.
