"""Local stand-in for the JoyAI server: replays a recorded JoyAI run with JoyAI's timing, so the bridge's live
playout / face pass can be tested without a GPU box (the face pass sees the same content it would live).

Speaks the server's protocol (start -> started; frame_meta + H.264 per input frame; output_frame + H.264 back,
acks). Every CHUNK input frames it answers with the recorded output frames that belong to those inputs (by
source_seq, from the run's input.json), as one burst, after a compute time like JoyAI's (~0.44 s per 8-frame chunk
on a 5090, chunks back to back) plus a one-way network delay with jitter.

  cloud/facefix/.venv/bin/python tools/joyai_standin.py --run cloud/lpfast/experiments/2026-11-25-karen [--port 18090]
Then point the bridge at ws://127.0.0.1:18090/ws and stream the same clip the run was recorded from.
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import av
import cv2
import numpy as np
import websockets


def load_run(run):
    cap, frames = cv2.VideoCapture(str(run / "joyai_raw.mp4")), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    seq = json.load(open(run / "input.json"))["source_seq"]
    n = min(len(frames), len(seq))
    by_seq = {}
    for j in range(n):
        if seq[j]:
            by_seq.setdefault(seq[j], j)
    keys = np.array(sorted(by_seq))
    return frames, by_seq, keys


async def session(ws, a, frames, by_seq, keys):
    start = json.loads(await ws.recv())
    h, w = frames[0].shape[:2]
    await ws.send(json.dumps({"type": "started", "width": w, "height": h}))
    print(f"[standin] session {w}x{h}, prompt {start.get('prompt', '')[:40]!r}", flush=True)
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p"
    enc.options = {"preset": "veryfast", "tune": "zerolatency", "crf": "18", "g": "16",
                   "x264-params": "scenecut=0:repeat-headers=1"}
    enc.open()
    out_q = asyncio.Queue()
    pending, meta = [], None
    busy_until = [0.0]
    last_out = [0.0]

    async def sender():
        while True:
            t_send, items = await out_q.get()
            await asyncio.sleep(max(0.0, t_send - time.time()))
            for sseq, tcap in items:
                k = keys[min(len(keys) - 1, np.searchsorted(keys, sseq))]
                img = frames[by_seq[int(k)]]
                vf = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")
                data = b"".join(bytes(p) for p in enc.encode(vf))
                await ws.send(json.dumps({"type": "output_frame", "source_seq": sseq, "t_capture_ms": tcap}))
                await ws.send(data)
                await asyncio.sleep(a.burst_gap_ms / 1000)

    st = asyncio.create_task(sender())
    try:
        async for m in ws:
            if isinstance(m, str):
                j = json.loads(m)
                if j.get("type") == "frame_meta":
                    meta = j
                continue
            if meta is None:
                continue
            pending.append((meta["seq"], meta.get("t_capture_ms")))
            meta = None
            if len(pending) >= a.chunk:
                now = time.time()
                start_c = max(now, busy_until[0])                    # chunks run back to back on the GPU
                busy_until[0] = start_c + a.compute_ms / 1000 * random.uniform(0.95, 1.08)
                t_out = busy_until[0] + a.net_ms / 1000 + random.uniform(0, a.jitter_ms / 1000)
                if random.random() < a.stall_prob:          # network hiccup: delivery late, the GPU keeps going
                    t_out += random.uniform(0.5, 1.0) * a.stall_ms / 1000
                t_out = max(t_out, last_out[0])             # in order: later bursts queue behind a stalled one
                last_out[0] = t_out
                await out_q.put((t_out, pending))
                pending = []
    finally:
        st.cancel()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--compute-ms", type=float, default=300, help="GPU time per chunk (the server keeps up with 26 "
                    "fps input: 8 frames in < 0.31 s; the measured 0.44 s 'elapsed' overlaps transfer)")
    ap.add_argument("--net-ms", type=float, default=200, help="extra fixed delay: network + decode/encode")
    ap.add_argument("--jitter-ms", type=float, default=60)
    ap.add_argument("--burst-gap-ms", type=float, default=3)
    ap.add_argument("--stall-prob", type=float, default=0.0, help="chance per chunk of a network stall")
    ap.add_argument("--stall-ms", type=float, default=300)
    a = ap.parse_args()
    frames, by_seq, keys = load_run(Path(a.run))
    print(f"[standin] {len(frames)} recorded frames, ws://127.0.0.1:{a.port}/ws", flush=True)
    async with websockets.serve(lambda ws: session(ws, a, frames, by_seq, keys), "127.0.0.1", a.port,
                                max_size=64 * 2**20):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
