"""A Nuance Live AI persona as a live input for the bridge (non-commercial experiment).

Starts one Nuance Live conversation (docs.nuancelabs.ai), joins its LiveKit room and:
  persona video -> H.264 over SRT to the bridge (like a phone: --input srt://0.0.0.0:PORT)
  persona audio -> a PulseAudio sink (the bridge's --monitor-source <sink>.monitor, e.g. through --voice rvc)
  your mic      -> the persona (so you can talk with it; use headphones), or silence with --mic none
The API key comes from .secrets/nuance_api_key (never printed).

  tools/nuance_live/.venv/bin/python tools/nuance_live/nuance_source.py --srt srt://127.0.0.1:9022 \
      --sink nuance_audio [--persona sonia] [--mic default|none] [--say "Introduce yourself"] [--max-seconds 300]
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
from livekit import rtc

REPO = Path(__file__).resolve().parents[2]
API = "https://api.nuancelabs.ai"
TOPIC = "nuance_live"
SR = 48000

DEFAULT_PROMPT = ("You are Karen, a warm, chatty livestreamer talking with your viewers on a live stream. Keep "
                  "answers short and natural, like a real conversation.")


CAM_W, CAM_H = 480, 640   # user camera frames (portrait 3:4)


def log(m):
    print(f"[nuance] {m}", flush=True)


class Nuance:
    def __init__(self, a):
        self.a = a
        self.key = (REPO / ".secrets" / "nuance_api_key").read_text().strip()
        self.conv = None
        self.token = None
        self.ended = asyncio.Event()
        self.started = asyncio.Event()
        self.video_proc = None
        self.audio_proc = None
        self.n_video = 0
        self.n_audio = 0
        self.rec_v = None          # --record: raw persona video (h264, frames as they arrive) + its audio
        self.rec_a = None
        self.rec_t = {}

    async def http(self, s, method, path, token, body=None):
        h = {"Authorization": f"Bearer {token}"}
        kw = {"json": body} if body is not None else {"data": b"", "headers": {**h, "Content-Length": "0"}}
        async with s.request(method, API + path, headers=kw.pop("headers", h), timeout=aiohttp.ClientTimeout(total=40),
                             **kw) as r:
            txt = await r.text()
            if r.status >= 300:
                raise RuntimeError(f"{method} {path} -> {r.status}: {txt[:400]}")
            return json.loads(txt) if txt else {}

    async def run(self):
        a = self.a
        async with aiohttp.ClientSession() as s:
            u = await self.http(s, "GET", "/usage", self.key)
            log(f"credits left today {u['credits']['remaining']}/{u['limits']['daily_credits']}, active "
                f"{u['active_sessions']}/{u['limits']['concurrent_sessions']}")
            body = {"interface": "nuance_live",
                    "personality": {"persona_id": a.persona, "system_prompt": a.prompt},
                    "interface_setup": {"delegation": {"type": "none"},
                                        "video": {"input": {"camera": a.camera != "none"}}},
                    "expressiveness": a.expressiveness, "end_user_id": "ai-livestream-test"}
            r = await self.http(s, "POST", "/session-tokens", self.key, body)
            if r.get("warnings"):
                log(f"warnings: {r['warnings']}")
            self.token = r["token"]
            c = await self.http(s, "POST", "/conversation", self.token,
                                {"timezone": a.timezone, "max_session_seconds": a.max_seconds})
            self.conv = c["id"]
            log(f"conversation {self.conv} created ({a.persona}, max {a.max_seconds} s)")
            admitted = asyncio.get_running_loop().create_future()
            follow = asyncio.create_task(self.follow(s, admitted))
            try:
                ev = await asyncio.wait_for(admitted, 90)
                await self.room(ev["livekit_url"], ev["token"])
            finally:
                follow.cancel()
                if not self.ended.is_set():
                    try:
                        await self.http(s, "POST", f"/conversation/{self.conv}/end", self.key)
                    except Exception:  # noqa: BLE001
                        try:
                            await self.http(s, "POST", f"/conversation/{self.conv}/abandon", self.token)
                        except Exception:  # noqa: BLE001
                            pass

    async def follow(self, s, admitted):
        after = 0
        while True:
            try:
                r = await self.http(s, "GET", f"/conversation/{self.conv}/events?after={after}", self.token)
            except Exception as e:  # noqa: BLE001
                log(f"events: {e}")
                await asyncio.sleep(2)
                continue
            for ev in r.get("events", []):
                after = ev["id"]
                t = ev["type"]
                if t == "admitted" and not admitted.done():
                    admitted.set_result(ev)
                elif t == "started":
                    log(f"persona started (time left {ev.get('seconds_remaining')} s)")
                elif t == "ended":
                    log(f"conversation ended: {ev.get('reason')}")
                    self.ended.set()
                    return
                elif t != "admitted":
                    log(f"event {t}")

    # ------------------------------------------------------------------ room
    async def room(self, url, token):
        a = self.a
        room = rtc.Room()
        loop = asyncio.get_running_loop()

        @room.on("track_subscribed")
        def _sub(track, pub, participant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                asyncio.ensure_future(self.video(track))
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.ensure_future(self.audio(track))

        @room.on("data_received")
        def _data(pkt):
            if pkt.topic != TOPIC:
                return
            try:
                ev = json.loads(bytes(pkt.data).decode())
            except ValueError:
                return
            t = ev.get("type")
            if t == "session.started":
                log(f"session started (expires {ev.get('session', {}).get('expires_at')})")
                loop.call_soon(self.started.set)
            elif t == "session.closed":
                log(f"session closed: {ev.get('reason')}")
                loop.call_soon(self.ended.set)
            elif t == "error":
                log(f"error: {json.dumps(ev)[:300]}")
            elif t and "transcript" in t and ev.get("text") and a.verbose:
                log(f"{t}: {ev['text'][:120]}")

        await room.connect(url, token)
        log("joined the room")
        mic = rtc.AudioSource(SR, 1)
        track = rtc.LocalAudioTrack.create_audio_track("microphone", mic)
        await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        mic_task = asyncio.create_task(self.mic(mic))
        cam_task = None
        if a.camera != "none":   # the user's face: the persona sees them
            cam = rtc.VideoSource(CAM_W, CAM_H)
            ctrack = rtc.LocalVideoTrack.create_video_track("camera", cam)
            await room.local_participant.publish_track(ctrack, rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA))
            cam_task = asyncio.create_task(self.camera(cam))
        try:
            await asyncio.wait_for(self.started.wait(), 60)
            if a.say:
                await room.local_participant.publish_data(
                    json.dumps({"type": "response.create", "response": {"instructions": a.say}}).encode(),
                    reliable=True, topic=TOPIC)
            await self.ended.wait()
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            try:
                await room.local_participant.publish_data(json.dumps({"type": "session.close"}).encode(),
                                                          reliable=True, topic=TOPIC)
                await asyncio.wait_for(self.ended.wait(), 5)
            except Exception:  # noqa: BLE001
                pass
            mic_task.cancel()
            if cam_task:
                cam_task.cancel()
            await room.disconnect()
            for p in (self.video_proc, self.audio_proc):
                if p and p.poll() is None:
                    p.stdin.close()
                    p.kill()
            log(f"left: {self.n_video} video frames, {self.n_audio / SR:.0f} s audio forwarded")
            if self.a.record and self.rec_v is not None and self.rec_a is not None:
                self.finish_record()

    def finish_record(self):
        """Mux the recording: video at its measured frame rate, audio shifted by (first audio - first video)."""
        self.rec_v.stdin.close(); self.rec_v.wait(); self.rec_a.close()
        t = self.rec_t
        fps = (t["nv"] - 1) / max(1e-3, t["v1"] - t["v0"])
        off = t["a0"] - t["v0"]
        out = self.a.record
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-r", f"{fps:.4f}", "-i", out + ".v.mp4",
                        "-itsoffset", f"{off:.3f}", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", out + ".a.raw",
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", out], check=True)
        Path(out + ".v.mp4").unlink(); Path(out + ".a.raw").unlink()
        log(f"recorded {out} ({t['nv']} frames at {fps:.2f} fps, audio offset {off * 1000:+.0f} ms)")

    async def mic(self, src):
        """Your mic (PulseAudio source via ffmpeg) or silence, in 10 ms frames; the persona needs a constant track."""
        n = SR // 100
        if self.a.mic == "none":
            silence = rtc.AudioFrame(bytes(n * 2), SR, 1, n)
            t = time.monotonic()
            while True:
                await src.capture_frame(silence)
                t += 0.01
                await asyncio.sleep(max(0, t - time.monotonic()))
        p = await asyncio.create_subprocess_exec(
            "ffmpeg", "-loglevel", "error", "-f", "pulse", "-fragment_size", "1920", "-i", self.a.mic, "-ac", "1",
            "-ar", str(SR), "-f", "s16le", "-", stdout=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.DEVNULL)
        try:
            while True:
                b = await p.stdout.readexactly(n * 2)
                await src.capture_frame(rtc.AudioFrame(b, SR, 1, n))
        finally:
            p.kill()

    async def camera(self, src):
        """The user's camera: raw RGBA frames (CAM_W x CAM_H) from tcp://host:port (tools/phone_view.py
        --camera-port), forwarded as they come; reconnects."""
        host, port = self.a.camera.split("://", 1)[1].split(":")
        n, sent = CAM_W * CAM_H * 4, 0
        while True:
            try:
                r, _w = await asyncio.open_connection(host, int(port))
                log(f"camera source connected ({self.a.camera})")
                while True:
                    b = await r.readexactly(n)
                    src.capture_frame(rtc.VideoFrame(CAM_W, CAM_H, rtc.VideoBufferType.RGBA, b))
                    sent += 1
                    if sent == 1:
                        log("camera: first frame sent to the persona")
            except (OSError, asyncio.IncompleteReadError):
                await asyncio.sleep(1)

    async def video(self, track):
        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
        async for ev in stream:
            f = ev.frame
            if self.video_proc is None and self.a.srt == "none":
                self.size = (f.width, f.height)
                self.video_proc = False
            if self.video_proc is None:
                log(f"persona video {f.width}x{f.height} -> {self.a.srt}")
                self.video_proc = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{f.width}x{f.height}",
                     "-use_wallclock_as_timestamps", "1", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                     "-tune", "zerolatency", "-g", "30", "-b:v", "5M", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                     "-f", "mpegts", f"{self.a.srt}?mode=caller&latency=20"], stdin=subprocess.PIPE)
                self.size = (f.width, f.height)
            if (f.width, f.height) != self.size:
                continue
            if self.a.record:
                if self.rec_v is None:
                    self.rec_t["v0"] = time.time()
                    self.rec_v = subprocess.Popen(
                        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                         f"{f.width}x{f.height}", "-r", "25", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                         "-crf", "16", "-pix_fmt", "yuv420p", self.a.record + ".v.mp4"], stdin=subprocess.PIPE)
                self.rec_t["v1"] = time.time()
                self.rec_t["nv"] = self.rec_t.get("nv", 0) + 1
                self.rec_v.stdin.write(bytes(f.data))
            if self.video_proc is False:
                self.n_video += 1
                continue
            try:
                self.video_proc.stdin.write(bytes(f.data))
                self.n_video += 1
            except (BrokenPipeError, ValueError):
                return

    async def audio(self, track):
        stream = rtc.AudioStream(track, sample_rate=SR, num_channels=1)
        self.audio_proc = subprocess.Popen(["pacat", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
                                            "--latency-msec=20", f"--device={self.a.sink}", "--stream-name=nuance-persona"],
                                           stdin=subprocess.PIPE)
        log(f"persona audio -> {self.a.sink}")
        async for ev in stream:
            b = bytes(ev.frame.data)
            if self.a.record:
                if self.rec_a is None:
                    self.rec_t["a0"] = time.time()
                    self.rec_a = open(self.a.record + ".a.raw", "wb")
                self.rec_a.write(b)
            try:
                self.audio_proc.stdin.write(b)
                self.n_audio += len(b) // 2
            except (BrokenPipeError, ValueError):
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", default="srt://127.0.0.1:9022")
    ap.add_argument("--sink", default="nuance_audio")
    ap.add_argument("--persona", default="sonia", help="sam | sonia | mira (docs.nuancelabs.ai/personas)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--say", default="Say hi to your viewers and introduce yourself.",
                    help="first thing the persona says (it never speaks first); '' = wait for you")
    ap.add_argument("--mic", default="default", help="PulseAudio source that talks to the persona, or 'none'")
    ap.add_argument("--camera", default="none", help="the user's camera for the persona to see: none, or "
                    f"tcp://HOST:PORT sending raw RGBA {CAM_W}x{CAM_H} frames (tools/phone_view.py --camera-port)")
    ap.add_argument("--max-seconds", type=int, default=300)
    ap.add_argument("--expressiveness", type=float, default=1.0)
    ap.add_argument("--timezone", default="America/Los_Angeles")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--record", default=None, help="also save the persona's raw video + audio to this mp4 (a reusable "
                    "test input: tools/voice_swap_test.sh plays it like a live persona, no credits)")
    a = ap.parse_args()
    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        asyncio.run(Nuance(a).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
