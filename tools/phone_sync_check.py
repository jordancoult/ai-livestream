"""A/V sync check of the phone path: a fake phone that plays back like tools/phone_view.py's page (audio jitter
buffer; picture shown when its audio is heard) and measures beep-vs-flash offsets of the flash/beep test clip.

  tools/character_live.py karen --test --phone --clip .tmp/avsync/av.mp4 --seconds 40 --out test-outputs/sync.mp4 &
  cloud/facefix/.venv/bin/python tools/phone_sync_check.py "https://<ip>:8443/?t=<token>" 30
(the clip comes from tools/av_sync_selftest.sh). + = audio late. The laptop recording (--out) can be checked the same way.
"""
import asyncio
import ssl
import sys
import time

import cv2
import numpy as np
from websockets.asyncio.client import connect

url, secs = sys.argv[1], float(sys.argv[2])
LAG = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
host, tok = url.split("/")[2], url.split("t=")[1]
ctx = ssl.create_default_context()
ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE


async def main():
    play_at = 0.0
    audio, video = [], []          # (presentation time, samples) / (presentation time, brightness)
    async with connect(f"wss://{host}/ws?t={tok}", ssl=ctx, max_size=2**22) as ws:
        t0 = time.monotonic()
        while time.monotonic() - t0 < secs:
            m = await ws.recv()
            now = time.monotonic()
            if m[:1] == b"A":
                s = np.frombuffer(m[1:], np.int16).astype(np.float32) / 32768
                if play_at < now + 0.04 or play_at > now + 0.6:
                    play_at = now + LAG
                audio.append((play_at, s))
                play_at += len(s) / 48000
            elif m[:1] == b"V":
                img = cv2.imdecode(np.frombuffer(m[1:], np.uint8), cv2.IMREAD_GRAYSCALE)
                video.append((now + max(0.0, play_at - now), float(img.mean())))
    # audio timeline at 1 kHz envelope
    ta = np.array([t for t, _ in audio])
    env_t, env_v = [], []
    for t, s in audio:
        for k in range(0, len(s), 48):
            env_t.append(t + k / 48000)
            env_v.append(float(np.abs(s[k:k + 48]).mean()))
    env_t, env_v = np.array(env_t), np.array(env_v)
    th = max(0.02, np.percentile(env_v, 99) * 0.25)
    on = env_v > th
    beeps = [env_t[i] for i in range(1, len(on)) if on[i] and not on[i - 1:max(0, i - 120) - 1:-1].any()]
    vt = np.array([t for t, _ in video])
    vb = np.array([b for _, b in video])
    mid = (np.percentile(vb, 10) + np.percentile(vb, 90)) / 2
    hi = vb > mid
    flashes = [vt[i] for i in range(1, len(hi)) if hi[i] and not hi[i - 1]]
    offs = []
    for b in beeps:
        if not flashes:
            break
        f = min(flashes, key=lambda x: abs(x - b))
        if abs(f - b) < 0.4:
            offs.append((b - f) * 1000)
    print(f"phone view: {len(video)} frames ({len(video) / secs:.1f} fps), {len(beeps)} beeps, {len(flashes)} flashes, "
          f"{len(offs)} matched")
    if offs:
        o = np.array(offs)
        print(f"PHONE A/V offset (audio minus video, + = audio late): median {np.median(o):+.0f} ms, "
              f"p10 {np.percentile(o, 10):+.0f}, p90 {np.percentile(o, 90):+.0f}")


asyncio.run(main())
