"""Physical A/V latency probe for the phone driver: where does the audio/video offset come from?

Flashes a full-screen window and beeps (1 kHz) at the same instants (av_sync.py blink), while it
  - taps the phone's H.264 (tools/phone_view.py feed on tcp 127.0.0.1:9026: every frame carries its capture time on
    the laptop clock) -> when each flash is CAPTURED (per the phone's timestamp) and when it ARRIVES here,
  - records the phone mic as it arrives (PulseAudio phone_mic.monitor) -> when each beep arrives.
All against the laptop's own clock (the flash/beep log), so it checks the timestamps the sync relies on:
  video: capture stamp - flash  (should be ~display + exposure, 20-60 ms)
  audio: beep arrival  - beep   (the mic's true lag, incl. the laptop speaker's output latency)
and compares the voice delay the bridge uses with the one the physics says it should use.

  cloud/facefix/.venv/bin/python tools/av_probe.py [seconds] [--save]   (point the phone camera at the laptop screen)
  --save: store the measured constant (phone audio input processing, ~70-120 ms on an iPhone with voice processing)
          as this input's correction, live and persisted (the bridge's av_chain); everything variable is tracked live
Writes nothing into the pipeline; the bridge keeps running (its dashboard on :8765 is read for the pipeline latency).
"""
import json
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import av
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SR = 48000
HDR = struct.Struct("<IdB")


def video_tap(stop, out):
    s = socket.create_connection(("127.0.0.1", 9026), timeout=5)
    s.settimeout(1.0)
    dec = av.CodecContext.create("h264", "r")
    buf = b""
    while not stop.is_set():
        try:
            c = s.recv(1 << 16)
        except socket.timeout:
            continue
        if not c:
            break
        buf += c
        while len(buf) >= HDR.size:
            n, cap, _fl = HDR.unpack_from(buf)
            if len(buf) < HDR.size + n:
                break
            data, buf = buf[HDR.size:HDR.size + n], buf[HDR.size + n:]
            t_arr = time.time()
            try:
                for fr in dec.decode(av.Packet(data)):
                    g = fr.to_ndarray(format="gray")
                    h, w = g.shape
                    out.append((cap / 1000, t_arr, float(g[h // 4:3 * h // 4, w // 4:3 * w // 4].mean())))
            except av.error.InvalidDataError:
                pass


def audio_tap(stop, out):
    p = subprocess.Popen(["parec", "-d", "phone_mic.monitor", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
                          "--latency-msec=10"], stdout=subprocess.PIPE)
    blk = 480
    while not stop.is_set():
        b = p.stdout.read(blk * 2)
        if len(b) < blk * 2:
            break
        out.append((time.time(), np.frombuffer(b, np.int16).astype(np.float32) / 32768))
    p.kill()


def onsets(times, vals, thr, min_gap=0.2):
    r, last = [], -9
    for i in range(1, len(vals)):
        if vals[i - 1] < thr <= vals[i] and times[i] - last > min_gap:
            r.append(times[i])
            last = times[i]
    return r


def match(events, flashes, lo=-0.1, hi=1.5):
    d = []
    for f in flashes:
        c = [e - f for e in events if lo < e - f < hi]
        if c:
            d.append(min(c, key=abs) if lo < 0 else min(c))
    return d


def main():
    nums = [a for a in sys.argv[1:] if not a.startswith("--")]
    sec = float(nums[0]) if nums else 15
    stop, vid, aud = threading.Event(), [], []
    th = [threading.Thread(target=video_tap, args=(stop, vid), daemon=True),
          threading.Thread(target=audio_tap, args=(stop, aud), daemon=True)]
    for t in th:
        t.start()
    time.sleep(1.5)                                   # keyframe + decoder warm-up
    log = REPO / ".tmp" / "av_probe_blink.txt"
    subprocess.run([sys.executable, str(REPO / "cloud/client/av_sync.py"), "blink", str(sec), str(log)])
    time.sleep(1.5)
    stop.set()
    try:
        st = json.loads(urllib.request.urlopen("http://127.0.0.1:8765/status", timeout=3).read())
    except Exception:  # noqa: BLE001
        st = {}
    flashes = [float(a) for a, lv in (ln.split() for ln in log.read_text().splitlines()) if lv == "1"]

    # video: brightness rising through the midpoint
    caps, arrs, br = (np.array([v[i] for v in vid]) for i in range(3))
    mid = (np.percentile(br, 10) + np.percentile(br, 90)) / 2
    v_cap = match(onsets(list(caps), list(br), mid), flashes, lo=-0.3)
    v_arr = match(onsets(list(arrs), list(br), mid), flashes)
    # audio: 1 kHz energy (Goertzel per 10 ms block) rising
    k = 2 * np.cos(2 * np.pi * 1000 / SR)
    def g1k(x):
        s0 = s1 = 0.0
        for v in x:
            s0, s1 = v + k * s0 - s1, s0
        return s0 * s0 + s1 * s1 - k * s0 * s1
    e = np.array([g1k(x) for _, x in aud])
    t_a = [t - 480 / SR for t, _ in aud]          # time of the block's first sample (read returns at its end)
    thr = np.percentile(e, 50) + 0.3 * (np.percentile(e, 99.5) - np.percentile(e, 50))
    a_arr = match(onsets(t_a, list(e), thr), flashes)

    # robust audio estimate: correlate the 1 kHz energy envelope with the whole beep schedule (all beeps at once,
    # survives noise suppression softening single beeps)
    ta, ea = np.array(t_a), np.log(e + 1e-9)
    grid = np.arange(flashes[0] - 0.5, flashes[-1] + 2.0, 0.005)
    env = np.interp(grid, ta, ea)
    env = (env - env.mean()) / (env.std() + 1e-9)
    best = (None, -9)
    for lag in np.arange(0.0, 1.2, 0.005):
        sched = np.zeros_like(grid)
        for f0 in flashes:
            sched[(grid >= f0 + lag) & (grid < f0 + lag + 0.12)] = 1
        sched = (sched - sched.mean()) / (sched.std() + 1e-9)
        c = float((env * sched).mean())
        if c > best[1]:
            best = (lag, c)
    a_corr_ms, a_corr = (round(best[0] * 1000), round(best[1], 2)) if best[0] is not None else (None, 0)
    med = lambda x: round(float(np.median(x)) * 1000) if x else None  # noqa: E731
    sync = st.get("sync") or {}
    res = {"flashes": len(flashes), "video_frames": len(vid),
           "video_capture_stamp_minus_flash_ms": med(v_cap), "n_vcap": len(v_cap),
           "video_arrival_minus_flash_ms": med(v_arr), "n_varr": len(v_arr),
           "mic_arrival_minus_beep_ms": a_corr_ms if a_corr >= 0.2 else med(a_arr), "audio_corr": a_corr,
           "mic_arrival_minus_beep_ms_onsets": med(a_arr), "n_audio": len(a_arr),
           "bridge_pipeline_ms(capture->played)": sync.get("pipeline_ms"), "bridge_voice_delay_ms": sync.get("delay_ms"),
           "bridge_mic_lag_adjust_ms": sync.get("live_adjust_ms"), "chain_ms": sync.get("chain_ms"),
           "adjust_ms": sync.get("adjust_ms")}
    if None not in (res["video_capture_stamp_minus_flash_ms"], res["mic_arrival_minus_beep_ms"], sync.get("pipeline_ms")):
        # what the voice delay should be: event -> played video  minus  event -> mic audio here (output latencies
        # of the laptop screen/speaker are on both sides and cancel; the 30 ms playback buffer is the bridge's)
        res["voice_delay_should_be_ms"] = res["video_capture_stamp_minus_flash_ms"] + sync["pipeline_ms"] \
            - res["mic_arrival_minus_beep_ms"] - 30
    if "--save" in sys.argv and "voice_delay_should_be_ms" in res and res.get("audio_corr", 0) >= 0.5:
        # the constant the timestamps cannot see (the phone's audio input processing): save it as this input's
        # camera-chain correction, live (bridge command av_chain; persisted like a sync test)
        used = sync.get("delay_ms")
        chain = (sync.get("chain_ms") or 0) + (res["voice_delay_should_be_ms"] - used)
        req = urllib.request.Request("http://127.0.0.1:8765/cmd", json.dumps({"av_chain": chain}).encode())
        urllib.request.urlopen(req, timeout=3).read()
        res["saved_chain_ms"] = round(chain)
    elif "--save" in sys.argv:
        res["saved_chain_ms"] = "not saved (weak audio match: beeps audible? flash in view?)"
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
