"""Replay a video into JoyAI-VL-Interaction (webinfer adapter) at 1 frame/s in real time, as its WebUI does, with optional
spoken-style questions at set moments; record every decision (silence / speak / delegate), the reply and its latency;
optionally voice each reply with their Qwen3-TTS adapter (time to first audio + full clip).

  python stream_test.py --video clip.mp4 --out run.json [--queries "10:What am I holding?;30:How do I look?"]
      [--prompt "standing instruction sent with every frame"] [--tts ws://127.0.0.1:8992/ws/tts] [--voice ryan]
"""

import argparse
import asyncio
import base64
import json
import time
import urllib.request
import uuid

import cv2

p = argparse.ArgumentParser()
p.add_argument("--video", required=True)
p.add_argument("--out", required=True)
p.add_argument("--adapter", default="http://127.0.0.1:8070/v1/chat/completions")
p.add_argument("--model", default="jdopensource/JoyAI-VL-Interaction")
p.add_argument("--fps", type=float, default=1.0)
p.add_argument("--seconds", type=float, default=0)
p.add_argument("--queries", default="")
p.add_argument("--prompt", default="")
p.add_argument("--prompt-key", default="DEFAULT_SYSTEM_PROMPT_NO_DELEGATION")
p.add_argument("--tts", default="")
p.add_argument("--voice", default="ryan")
p.add_argument("--width", type=int, default=640)
p.add_argument("--voice-only", default="", help="skip the video run: voice the replies of this earlier run JSON")
a = p.parse_args()

cap = cv2.VideoCapture(a.video)
vfps = cap.get(cv2.CAP_PROP_FPS) or 30
dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / vfps
if a.seconds:
    dur = min(dur, a.seconds)
queries = {}
for item in [x for x in a.queries.split(";") if x.strip()]:
    t, q = item.split(":", 1)
    queries[int(float(t))] = q.strip()
session = f"test-{uuid.uuid4().hex[:8]}"


def frame_at(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * vfps))
    ok, f = cap.read()
    if not ok:
        return None
    h, w = f.shape[:2]
    f = cv2.resize(f, (a.width, round(h * a.width / w)), interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


def ask(t, img, text):
    content = ([{"type": "text", "text": text}] if text else []) + [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": 128,
            "frame_time_range": f"{t:.1f} seconds"}
    req = urllib.request.Request(a.adapter, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "x-streaming-session": session, "x-system-prompt-key": a.prompt_key})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"], time.perf_counter() - t0


async def tts(text):
    import websockets
    t0 = time.perf_counter()
    first, pcm = None, b""
    async with websockets.connect(a.tts, max_size=2 ** 26) as ws:
        await ws.send(json.dumps({"config": {"voice": a.voice, "output_audio_format": "pcm16", "sample_rate": 24000}}))
        await ws.send(json.dumps({"type": "input_text.append", "text": text}))
        await ws.send(json.dumps({"type": "input_text.commit"}))
        async for m in ws:
            if isinstance(m, bytes):
                first = first or time.perf_counter() - t0
                pcm += m
            elif json.loads(m).get("type") in ("response.done", "error"):
                break
    return pcm, first, time.perf_counter() - t0


def voice_only():
    import numpy as np
    import soundfile as sf
    res = json.load(open(a.voice_only))
    for i, e in enumerate(res["log"]):
        if e.get("said"):
            pcm, first, full = asyncio.run(tts(e["said"]))
            e["tts_first_audio_s"], e["tts_total_s"] = round(first or 0, 3), round(full, 3)
            e["wav"] = a.out.rsplit(".", 1)[0] + f"_reply{i:03d}.wav"
            e["audio_s"] = round(len(pcm) / 2 / 24000, 2)
            sf.write(e["wav"], np.frombuffer(pcm, np.int16), 24000)
            print(f"{e['t']:5.1f}s first audio {first:.2f}s, total {full:.2f}s for {e['audio_s']}s: {e['said'][:60]}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    if a.voice_only:
        return voice_only()
    log, t_start = [], time.perf_counter()
    step = 1.0 / a.fps
    t = 0.0
    while t < dur:
        wait = t_start + t - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        img = frame_at(t)
        if img is None:
            break
        q = queries.get(int(t))
        text = q or a.prompt
        try:
            reply, lat = ask(t, img, text)
        except Exception as e:  # noqa: BLE001
            reply, lat = f"ERROR {type(e).__name__}: {e}", None
        kind = "silence" if "</silence>" in reply else "delegate" if "</delegation>" in reply else \
            "speak" if "</response>" in reply else "other"
        said = reply.split("</response>", 1)[-1].split("</delegation>")[0].strip() if kind in ("speak", "delegate") else ""
        e = {"t": round(t, 1), "query": q, "kind": kind, "said": said, "latency_s": round(lat, 3) if lat else None,
             "raw": reply[:300]}
        log.append(e)
        print(f"{t:5.1f}s {kind:8s} {lat or 0:5.2f}s {('Q: ' + q + ' -> ') if q else ''}{said[:100]}", flush=True)
        # real time: if a reply took longer than a frame, skip ahead like a live stream does
        t = max(t + step, round(time.perf_counter() - t_start))
    if a.tts:
        import numpy as np
        import soundfile as sf
        for i, e in enumerate(log):
            if e["said"]:
                pcm, first, full = asyncio.run(tts(e["said"]))
                e["tts_first_audio_s"], e["tts_total_s"] = round(first or 0, 3), round(full, 3)
                wav = a.out.rsplit(".", 1)[0] + f"_reply{i:03d}.wav"
                sf.write(wav, np.frombuffer(pcm, np.int16), 24000)
                e["wav"] = wav
                print(f"  tts {e['t']}s: first audio {first:.2f}s, total {full:.2f}s", flush=True)
    lats = [e["latency_s"] for e in log if e["latency_s"]]
    res = {"session": session, "video": a.video, "prompt": a.prompt, "queries": queries, "frames": len(log),
           "spoke": sum(e["kind"] == "speak" for e in log), "latency_median_s": sorted(lats)[len(lats) // 2] if lats else None,
           "latency_max_s": max(lats) if lats else None, "log": log}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}))


if __name__ == "__main__":
    main()
