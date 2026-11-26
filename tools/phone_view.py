"""Phone client for the live character mode: watch + hear the character and talk to it from a phone on the same Wi-Fi.

  laptop -> phone : the output video (dashboard /stream/final, JPEG) + the character's voice (PulseAudio sink OUT_SINK,
                    the bridge's audio monitor plays into it), over one WebSocket, re-synced on the phone
  phone  -> laptop: the phone's mic (echo-cancelled by the browser) -> PulseAudio sink MIC_SINK; its .monitor is the
                    "mic" the driver listens to (nuance_source --mic MIC_SINK.monitor)
                    + the phone's front camera (JPEG, ~15 fps) -> raw RGBA 480x640 on tcp://127.0.0.1:CAMERA_PORT
                    (nuance_source --camera tcp://127.0.0.1:CAMERA_PORT: the persona sees you)

HTTPS with a self-signed certificate (browsers only allow the mic on secure pages): the phone shows a warning once
(Safari: Show Details -> visit this website). A random token in the link keeps others on the Wi-Fi out.
Started by tools/character_live.py --phone; standalone:
  cloud/facefix/.venv/bin/python tools/phone_view.py --out-sink phone_out --mic-sink phone_mic [--port 8443]
"""
import argparse
import asyncio
import http
import json
import secrets
import socket
import ssl
import subprocess
import threading
import time
import collections
import urllib.request
from pathlib import Path

import struct

import cv2
import numpy as np

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

REPO = Path(__file__).resolve().parents[1]
MIC_LAG_FILE = REPO / ".tmp" / "phone_mic_lag.json"
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(REPO / "tools"))
from phone_cam_js import CAM_CSS, CAM_JS  # noqa: E402
SR, CHUNK = 48000, 960  # 20 ms audio chunks
CAM_W, CAM_H = 480, 640  # camera frames handed on (nuance_source CAM_W/CAM_H)
DRV_W, DRV_H = 720, 960  # --drive-srt: the phone camera as the pipeline's source video (3:4 portrait)


def lan_ip(iface=None):
    """The address the phone should open. Order: --iface; the Beryl LAN (192.168.8.0/24: the Seattle phone's only
    network -- it must never be offered another); Wi-Fi (wl*); Ethernet. (A VPN may own the default route, so ask the
    interfaces.)"""
    rows = [ln.split() for ln in subprocess.run(["ip", "-o", "-4", "addr", "show"], capture_output=True,
                                                 text=True).stdout.splitlines()]
    addrs = [(r[1], r[3].split("/")[0]) for r in rows if len(r) > 3]
    for name, ip in addrs:
        if iface and name == iface:
            return ip
    for name, ip in addrs:
        if ip.startswith("192.168.8."):
            return ip
    for pref in ("wl", "en", "eth"):
        for name, ip in addrs:
            if name.startswith(pref) and not ip.startswith("127."):
                return ip
    return "127.0.0.1"


def cert(ip):
    """Signed by the local CA (tools/local_ca.py; the phone trusts it once): no warning, and Safari's WebSockets
    connect (with a self-signed certificate they can hang)."""
    import sys as _s
    _s.path.insert(0, str(REPO / "tools"))
    from local_ca import server_cert
    crt, key = server_cert([ip], REPO / ".tmp" / "phone_cert")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    return ctx


def cert_selfsigned(ip):
    d = REPO / ".tmp" / "phone_cert"
    d.mkdir(parents=True, exist_ok=True)
    crt, key, tag = d / "cert.pem", d / "key.pem", d / "ip.txt"
    if not crt.exists() or not tag.exists() or tag.read_text() != ip:
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825", "-keyout", str(key),
                        "-out", str(crt), "-subj", "/CN=ai-livestream-phone", "-addext",
                        f"subjectAltName=IP:{ip},DNS:localhost"], check=True, capture_output=True)
        tag.write_text(ip)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    return ctx


class Clock:
    """Phone clock -> laptop clock: offset from the ping with the smallest round trip (last ~30 s)."""

    def __init__(self):
        self.samples = collections.deque(maxlen=60)

    def add(self, t_send, t_phone, t_recv):
        rtt = t_recv - t_send
        self.samples.append((rtt, t_phone - (t_send + rtt / 2)))

    def offset(self):
        return min(self.samples)[1] if self.samples else None


DRV_HDR = struct.Struct("<dIB")    # phone page 'H': capture time (phone clock, ms), seq, flags (bit0 key, bits1-2 rot)
FWD_HDR = struct.Struct("<IdB")    # to the bridge: length, capture time (laptop clock, ms), flags


class Hub:
    def __init__(self, a):
        self.a, self.clients, self.loop = a, set(), None
        self.cam_out = set()   # local camera readers (asyncio StreamWriters)
        self.enc = None        # --drive-srt encoder (ffmpeg -> SRT -> the bridge)
        # counters for the console (GET /stats): per second, over the last ~2 s
        self.cnt = {k: 0 for k in ("cam_in", "mic_in", "video_out", "audio_out")}
        self.rates, self.t_rate = {}, time.time()
        self.mic_rms = collections.deque(maxlen=25)
        self.cam_gaps, self.t_cam = collections.deque(maxlen=150), None   # camera frame arrival gaps (burstiness)
        self.clock = Clock()
        self.h264_out = set()       # --drive-h264-port readers (the bridge)
        self.link_ms = collections.deque(maxlen=150)   # phone capture -> here (ms)
        # phone mic capture -> here (ms): the bridge subtracts it from the voice delay (lip sync follows the uplink)
        self.mic_lag = collections.deque(maxlen=250)
        self.mic_t0, self.mic_w, self.mic_backlog_ms = None, 0, None
        self.want_key = False
        self.down = collections.deque()     # HTTPS transport: what goes to the phone in the next response
        self.http_last = self.http_ping = 0.0
        self.phone = {}        # what the page reports (audio buffer, frames shown, output latency, sync offset)
        self.n_cam = 0
        self.mic = subprocess.Popen(["pacat", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
                                     "--latency-msec=20", "-d", a.mic_sink], stdin=subprocess.PIPE)

    def mic_governor(self, pcm):
        """Keep the mic buffer in front of the phone_mic device at ~40 ms. The phone's audio clock runs ~1.3 % fast
        against the laptop's: written 1.013x real time, played 1.000x, the surplus piled up in pacat's buffer
        without limit (+13 ms/s: ~330 ms after 25 s; found 2026-11-26 with tools/av_probe.py) -- delay the
        timestamps could not see. Backlog = written - elapsed real time; surplus -> a slight time squeeze (<= 3 %,
        inaudible), a burst (> 250 ms) -> dropped, an underrun -> re-anchor."""
        now = time.time()
        if self.mic_t0 is None:
            self.mic_t0, self.mic_w = now, 0
        backlog = self.mic_w / SR - (now - self.mic_t0)
        if backlog < -0.05:                       # underrun (the phone paused): start counting again
            self.mic_t0, self.mic_w, backlog = now, 0, 0.0
        self.mic_backlog_ms = round(backlog * 1000)
        excess = backlog - 0.04
        if excess > 0.25:
            self.cnt["mic_dropped"] = self.cnt.get("mic_dropped", 0) + 1
            return None
        if excess > 0:
            ratio = 1.0 - min(0.03, 0.005 + excess * 0.1)
            n = max(1, int(round(len(pcm) * ratio)))
            pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm.astype(np.float32)).astype(np.int16)
        self.mic_w += len(pcm)
        return pcm

    def tick(self):
        now = time.time()
        if now - self.t_rate >= 2:
            dt = now - self.t_rate
            self.rates = {k: round(v / dt, 1) for k, v in self.cnt.items()}
            if len(self.mic_lag) >= 20:
                ml = sorted(self.mic_lag)[len(self.mic_lag) // 2]
                self.mic_lag_ms = round(ml)
                try:   # read by the bridge (live_bridge.py --mic-lag-file)
                    tmp = MIC_LAG_FILE.with_suffix(".tmp")       # atomic: the bridge never reads half a file
                    tmp.write_text(json.dumps({"ms": round(ml, 1), "t": time.time() * 1000}))
                    tmp.replace(MIC_LAG_FILE)
                except OSError:
                    pass
            self.cnt = {k: 0 for k in self.cnt}
            self.t_rate = now

    def stats(self):
        self.tick()
        m = max(self.mic_rms) if self.mic_rms else 0
        g = sorted(self.cam_gaps)
        burst = {"gap_p50_ms": round(g[len(g) // 2] * 1000), "gap_p90_ms": round(g[int(len(g) * 0.9)] * 1000),
                 "gap_max_ms": round(g[-1] * 1000)} if len(g) > 20 else None
        lk = sorted(self.link_ms)
        link = {"p50": round(lk[len(lk) // 2]), "p90": round(lk[int(len(lk) * 0.9)]), "max": round(lk[-1])} if len(lk) > 20 else None
        return {"clients": len(self.clients) + (1 if self.http_active() else 0), "rates": self.rates, "cam_arrival": burst, "link_ms": link, "mic_lag_ms": getattr(self, "mic_lag_ms", None), "mic_backlog_ms": self.mic_backlog_ms,
                "mic_db": round(20 * np.log10(max(1e-5, m)), 1), "phone": self.phone, "drive": bool(self.a.drive_srt)}

    def broadcast(self, data, video=False):
        self.cnt["video_out" if video else "audio_out"] += 1
        if self.http_active():
            self.loop.call_soon_threadsafe(self._enqueue, data, video)
        for ws in list(self.clients):
            tr = ws.transport
            if video and tr is not None and tr.get_write_buffer_size() > 600_000:
                continue   # slow phone link: skip this frame rather than build a backlog
            asyncio.run_coroutine_threadsafe(self._send(ws, data), self.loop)

    def _enqueue(self, data, video):
        if video:   # keep at most 2 frames waiting: a slow moment drops old frames instead of building delay
            nv = sum(1 for x in self.down if x[:1] == b"V")
            if nv >= 2:
                for x in list(self.down):
                    if x[:1] == b"V":
                        self.down.remove(x)
                        break
        elif sum(1 for x in self.down if x[:1] == b"A") >= 25:   # at most ~0.5 s of voice waiting for the phone
            return
        self.down.append(data)

    @staticmethod
    async def _send(ws, data):
        try:
            await ws.send(data)
        except Exception:  # noqa: BLE001
            pass

    def video_thread(self):
        url = f"http://127.0.0.1:{self.a.dashboard}/stream/{self.a.view}?fps={self.a.fps}"
        while True:
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    buf = b""
                    while True:
                        buf += r.read1(65536)
                        while True:
                            i = buf.find(b"Content-Length: ")
                            if i < 0:
                                break
                            j = buf.find(b"\r\n\r\n", i)
                            if j < 0:
                                break
                            n = int(buf[i + 16:buf.find(b"\r\n", i)])
                            if len(buf) < j + 4 + n:
                                break
                            jpg = buf[j + 4:j + 4 + n]
                            buf = buf[j + 4 + n:]
                            if self.clients or self.http_active():
                                self.broadcast(b"V" + jpg, video=True)
            except Exception:  # noqa: BLE001  (bridge not up yet / restarted)
                time.sleep(1)

    def audio_thread(self):
        while True:
            p = subprocess.Popen(["parec", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
                                  "--latency-msec=20", "-d", f"{self.a.out_sink}.monitor"], stdout=subprocess.PIPE)
            while True:
                b = p.stdout.read(CHUNK * 2)
                if len(b) < CHUNK * 2:
                    break
                if self.clients or self.http_active():
                    self.broadcast(b"A" + b)
            p.kill()
            time.sleep(1)

    async def handler(self, ws):
        if not ws.request.path.startswith(f"/ws?t={self.a.token}"):
            await ws.close(1008, "bad token")
            return
        for old in list(self.clients):   # one phone at a time: a reload / second tab replaces the old one
            self.clients.discard(old)
            asyncio.ensure_future(old.close(1000, "replaced by a newer connection"))
        self.clients.add(ws)
        ping_task = asyncio.create_task(self.pinger(ws))
        self.want_key = True
        print(f"[phone] connected ({len(self.clients)})", flush=True)
        try:
            async for m in ws:
                if isinstance(m, bytes) and m:
                    self.on_msg(m)
                    if self.want_key and m[:1] == b"H":
                        self.want_key = False
                        await ws.send(b"K")
        finally:
            ping_task.cancel()
            self.clients.discard(ws)
            print(f"[phone] disconnected ({len(self.clients)})", flush=True)

    def on_msg(self, m):
        if True:
                if m[:1] == b"m" and len(m) > 9:     # timestamped mic: [m][capture ms f64, phone clock][pcm]
                    cap = struct.unpack_from("<d", m, 1)[0]
                    off = self.clock.offset()
                    if off is not None:
                        self.mic_lag.append(time.time() * 1000 - (cap - off))
                    m = b"M" + m[9:]
                if m[:1] == b"M" and self.mic.poll() is None:
                    self.cnt["mic_in"] += 1
                    x = np.frombuffer(m[1:], np.int16).astype(np.float32) / 32768
                    self.mic_rms.append(float(np.sqrt((x ** 2).mean())) if len(x) else 0.0)
                    out = self.mic_governor(np.frombuffer(m[1:], np.int16))
                    if out is not None:
                        self.mic.stdin.write(out.tobytes())
                        self.mic.stdin.flush()
                elif m[:1] == b"S":
                    try:
                        self.phone = {**json.loads(m[1:].decode()), "t": time.time()}
                    except ValueError:
                        pass
                elif m[:1] == b"Q":
                    t_send, t_phone = struct.unpack_from("<dd", m, 1)
                    self.clock.add(t_send, t_phone, time.time() * 1000)
                elif m[:1] == b"H":
                    self.h264(m)
                elif m[:1] == b"C" and (self.cam_out or self.a.drive_srt):
                    self.camera_frame(m[1:])

    def http_active(self):
        return time.time() - self.http_last < 2

    async def http_link(self, r, w):
        """The phone's only transport (Safari's WebSocket hangs for minutes at times; plain HTTPS doesn't, and is as
        fast: 23 ms vs 25 measured): POST /up?t=TOKEN, keep-alive, body = [u32 len][message]... (camera H.264,
        mic, stats, clock replies); the RESPONSE carries what the laptop has for the phone ([u32 len][message]...:
        the character's video and voice, clock pings, keyframe requests). The page posts at least every ~25 ms."""
        try:
            while True:
                head = await r.readuntil(b"\r\n\r\n")
                lines = head.decode(errors="replace").split("\r\n")
                method, path = (lines[0].split(" ") + ["", ""])[:2]
                hdrs = {ln.split(":", 1)[0].strip().lower(): ln.split(":", 1)[1].strip() for ln in lines[1:] if ":" in ln}
                body = await r.readexactly(int(hdrs.get("content-length", 0) or 0))
                out, code = b"", "404 Not Found"
                if method == "POST" and path.startswith(f"/up?t={self.a.token}"):
                    code = "200 OK"
                    if not self.http_active():
                        print("[phone] connected (https)", flush=True)
                        self.want_key = True
                        self.down.clear()
                    self.http_last = time.time()
                    i = 0
                    while i + 4 <= len(body):
                        n = struct.unpack_from("<I", body, i)[0]
                        self.on_msg(body[i + 4:i + 4 + n])
                        i += 4 + n
                    if self.want_key:
                        self.want_key = False
                        self.down.append(b"K")
                    if time.time() - self.http_ping > 0.5:
                        self.http_ping = time.time()
                        self.down.append(b"P" + struct.pack("<d", time.time() * 1000))
                    msgs = []
                    while self.down:
                        msgs.append(self.down.popleft())
                    out = b"".join(struct.pack("<I", len(x)) + x for x in msgs)
                elif method == "OPTIONS":
                    code = "204 No Content"
                w.write((f"HTTP/1.1 {code}\r\nContent-Type: application/octet-stream\r\n"
                         f"Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n"
                         f"Access-Control-Allow-Methods: POST\r\nCache-Control: no-store\r\n"
                         f"Content-Length: {len(out)}\r\n\r\n").encode() + out)
                await w.drain()
        except (asyncio.IncompleteReadError, ConnectionError, OSError, asyncio.LimitOverrunError):
            pass
        finally:
            w.close()

    def h264(self, m):
        """The phone's hardware H.264 (WebCodecs), passed to the bridge with its capture time on the laptop clock."""
        now = time.time() * 1000
        cap, _seq, flags = DRV_HDR.unpack_from(m, 1)
        data = m[1 + DRV_HDR.size:]
        off = self.clock.offset()
        cap_l = now if off is None else cap - off
        if off is not None:
            self.link_ms.append(now - cap_l)
            if now - cap_l > 500 and now - getattr(self, "flush_t", 0) > 2:   # a backlog formed: drop it now
                self.flush_t = now
                self.down.append(b"F")
                print(f"[phone] frames arrive {now - cap_l:.0f} ms after capture: asking the phone to drop its backlog",
                      flush=True)
        self.cnt["cam_in"] += 1
        if self.t_cam:
            self.cam_gaps.append(now / 1000 - self.t_cam)
        self.t_cam = now / 1000
        pkt = FWD_HDR.pack(len(data), cap_l, flags) + data
        for wr in list(self.h264_out):
            # never queue: a reader behind by > ~0.3 s of video stops getting packets until the next keyframe (asked
            # for at once), so it skips ahead instead of building delay (a 2 MB buffer here was ~4 s)
            if wr.transport.get_write_buffer_size() > 150_000:
                if not getattr(wr, "skipping", False):
                    wr.skipping = True
                    self.down.append(b"K")
                continue
            if getattr(wr, "skipping", False):
                if not flags & 1:
                    continue
                wr.skipping = False
            wr.write(pkt)
        self.n_cam += 1
        if self.n_cam == 1:
            print("[phone] camera (H.264) -> the pipeline", flush=True)

    async def h264_reader(self, r, w):
        self.h264_out.add(w)
        self.want_key = True        # a new decoder needs a keyframe: ask the phone for one now
        print("[phone] H.264 reader connected (the bridge)", flush=True)
        try:
            await r.read()
        except (ConnectionError, OSError):
            pass
        finally:
            self.h264_out.discard(w)

    async def pinger(self, ws):
        while True:
            try:
                await ws.send(b"P" + struct.pack("<d", time.time() * 1000))
            except Exception:  # noqa: BLE001
                return
            await asyncio.sleep(0.5)

    def drive(self, img):
        """The phone camera as the source video: raw frames -> x264 -> SRT (the bridge listens)."""
        if self.enc is None or self.enc.poll() is not None:
            self.enc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{DRV_W}x{DRV_H}",
                 "-use_wallclock_as_timestamps", "1", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                 "-tune", "zerolatency", "-g", "30", "-b:v", "4M", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                 "-f", "mpegts", self.a.drive_srt], stdin=subprocess.PIPE)
            print(f"[phone] camera -> {self.a.drive_srt} (the source video)", flush=True)
        try:
            self.enc.stdin.write(img.tobytes())
        except (BrokenPipeError, OSError):
            self.enc = None

    def camera_frame(self, jpg):
        self.cnt["cam_in"] += 1
        now = time.time()
        if self.t_cam:
            self.cam_gaps.append(now - self.t_cam)
        self.t_cam = now
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        if self.a.drive_srt:
            h, w = img.shape[:2]
            if w * DRV_H > h * DRV_W:
                nw = h * DRV_W // DRV_H
                c = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
            else:
                nh = w * DRV_H // DRV_W
                c = img[(h - nh) // 2:(h - nh) // 2 + nh]
            self.drive(cv2.resize(c, (DRV_W, DRV_H), interpolation=cv2.INTER_AREA))
            self.n_cam += 1
            if self.n_cam == 1:
                print("[phone] camera: first frame -> the pipeline", flush=True)
            if not self.cam_out:
                return
        h, w = img.shape[:2]
        if w * CAM_H > h * CAM_W:            # crop to 3:4 portrait, then scale
            nw = h * CAM_W // CAM_H
            img = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
        else:
            nh = w * CAM_H // CAM_W
            img = img[(h - nh) // 2:(h - nh) // 2 + nh]
        rgba = cv2.cvtColor(cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGBA)
        data = rgba.tobytes()
        for wr in list(self.cam_out):
            if wr.transport.get_write_buffer_size() > 4 * len(data):
                continue   # reader behind: drop
            wr.write(data)
        self.n_cam += 1
        if self.n_cam == 1:
            print("[phone] camera: first frame handed on", flush=True)

    async def camera_reader(self, r, w):
        self.cam_out.add(w)
        print("[phone] camera reader connected", flush=True)
        try:
            await r.read()     # until it disconnects
        except (ConnectionError, OSError):
            pass
        finally:
            self.cam_out.discard(w)

    def http(self, conn, req):
        if req.path.startswith("/ws"):
            return None  # WebSocket upgrade
        if req.path.split("?")[0] == "/ca.crt":   # the local CA, for the phone to install + trust once
            from local_ca import CA_DER, ensure_ca
            ensure_ca()
            body = CA_DER.read_bytes()
            return Response(http.HTTPStatus.OK, "OK", Headers([("Content-Type", "application/x-x509-ca-cert"),
                                                               ("Content-Disposition", 'attachment; filename="ai-livestream-local-ca.crt"'),
                                                               ("Content-Length", str(len(body))), ("Connection", "close")]), body)

        if req.path.split("?")[0] == "/stats" and f"t={self.a.token}" in req.path:
            body = json.dumps(self.stats()).encode()
            return Response(http.HTTPStatus.OK, "OK", Headers([("Content-Type", "application/json"),
                                                               ("Content-Length", str(len(body))), ("Connection", "close")]), body)
        if req.path.split("?")[0] == "/" and f"t={self.a.token}" in req.path:
            body = (PAGE.replace("__CAM_JS__", CAM_JS).replace("__CAM_CSS__", CAM_CSS)
                    .replace("__HTTP_PORT__", str(self.a.http_port or self.a.port + 1))
                    .replace("__TOKEN__", self.a.token).replace("__TITLE__", self.a.title)
                    .replace("__CAMERA__", "true" if self.a.camera_port or self.a.drive_srt or self.a.drive_h264_port
                             else "false")
                    .replace("__DRIVE__", "true" if self.a.drive_srt or self.a.drive_h264_port else "false")
                    .replace("__H264__", "true" if self.a.drive_h264_port else "false")).encode()
            return Response(http.HTTPStatus.OK, "OK", Headers([("Content-Type", "text/html; charset=utf-8"),
                                                               ("Cache-Control", "no-store"),
                                                               ("Content-Length", str(len(body))), ("Connection", "close")]), body)
        return Response(http.HTTPStatus.NOT_FOUND, "Not Found", Headers([("Content-Length", "0"), ("Connection", "close")]), b"")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-sink", default="phone_out")
    ap.add_argument("--mic-sink", default="phone_mic")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--dashboard", type=int, default=8765)
    ap.add_argument("--view", default="final")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--token", default=None)
    ap.add_argument("--title", default="Character")
    ap.add_argument("--drive-srt", default=None, help="the phone camera becomes the source video: H.264 over SRT to "
                    "this URL (caller), e.g. srt://127.0.0.1:9024 (the bridge's --input srt://0.0.0.0:9024)")
    ap.add_argument("--iface", default=None, help="offer the phone this interface's address (default: the Beryl LAN "
                    "if present, else Wi-Fi)")
    ap.add_argument("--http-port", type=int, default=0, help="the phone's HTTPS transport (default: --port + 1)")
    ap.add_argument("--drive-h264-port", type=int, default=0, help="the phone camera becomes the source video as the "
                    "phone's own H.264 (WebCodecs) with capture times, served to the bridge on tcp://127.0.0.1:PORT "
                    "(bridge --input h264tcp://127.0.0.1:PORT)")
    ap.add_argument("--camera-port", type=int, default=0, help="hand the phone camera on to tcp://127.0.0.1:PORT "
                    "(0 = the page doesn't ask for the camera)")
    a = ap.parse_args()
    a.token = a.token or secrets.token_urlsafe(9)
    ip = lan_ip(a.iface)
    hub = Hub(a)
    hub.loop = asyncio.get_running_loop()
    threading.Thread(target=hub.video_thread, daemon=True).start()
    threading.Thread(target=hub.audio_thread, daemon=True).start()
    if a.camera_port:
        await asyncio.start_server(hub.camera_reader, "127.0.0.1", a.camera_port)
    if a.drive_h264_port:
        await asyncio.start_server(hub.h264_reader, "127.0.0.1", a.drive_h264_port)
    ctx = cert(ip)
    a.http_port = a.http_port or a.port + 1
    await asyncio.start_server(hub.http_link, "0.0.0.0", a.http_port, ssl=ctx, limit=16 * 2**20)
    async with serve(hub.handler, "0.0.0.0", a.port, ssl=ctx, process_request=hub.http, max_size=2**20,
                     ping_interval=20):
        print(f"[phone] open on the phone (same Wi-Fi): https://{ip}:{a.port}/?t={a.token}", flush=True)
        await asyncio.Future()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>__TITLE__</title>
<style>
[hidden]{display:none!important}
html,body{margin:0;height:100%;background:#000;color:#ddd;font:15px -apple-system,system-ui,sans-serif}
#v{position:fixed;inset:0;width:100%;height:100%;object-fit:contain}
#ui{position:fixed;left:0;right:0;bottom:0;padding:12px max(12px,env(safe-area-inset-left)) calc(12px + env(safe-area-inset-bottom));
 display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:linear-gradient(transparent,#000c)}
button{font:600 16px -apple-system,system-ui;border:0;border-radius:22px;padding:11px 18px;background:#2e7d32;color:#fff}
button.off{background:#8a3b2b} button.gray{background:#333} #st{font-size:12px;opacity:.8;flex:1;text-align:right;min-width:120px}
#sync{display:flex;align-items:center;gap:6px;font-size:12px;opacity:.85} #sync input{width:100px}
#go{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;background:#000}
#go p{max-width:300px;text-align:center;opacity:.75;margin:0}
#me{position:fixed;top:calc(10px + env(safe-area-inset-top));right:max(10px,env(safe-area-inset-right));width:24vw;
 max-width:130px;border-radius:10px;border:1px solid #fff4;background:#111}
#camctl{position:fixed;left:max(10px,env(safe-area-inset-left));right:max(10px,env(safe-area-inset-right));
 bottom:calc(80px + env(safe-area-inset-bottom));background:#111e;border-radius:14px;padding:10px}
#resume{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:#000a}
__CAM_CSS__
</style></head><body>
<img id="v" alt=""><video id="me" playsinline muted autoplay hidden></video>
<div id="go"><button id="start">Start talking to __TITLE__</button><p>Allow the microphone (and camera). Earbuds work best (the phone
cancels echo, but speakers can still leak into the mic).</p></div>
<div id="camctl" hidden><div id="cams"></div><div id="camrow"><input id="zoom" type="range" hidden><span id="zoomv"></span><button id="rot">↻ rotate</button></div></div>
<div id="resume" hidden><button id="resumeb">Tap to resume sound</button></div>
<div id="ui" hidden><button id="mute">Mic on</button><button id="camb" class="gray" hidden>Camera</button>
<button id="syncb" class="gray">Sync 0</button>
<label id="sync" hidden title="Voice comes late (e.g. Bluetooth earbuds): slide right. Voice early: slide left.">
<input id="off" type="range" min="-400" max="400" step="20" value="0"><button id="off0" class="gray">0</button></label><span id="st"></span></div>
<script>
__CAM_JS__
const T="__TOKEN__", CAM=__CAMERA__, DRIVE=__DRIVE__, H264=__H264__, HP=__HTTP_PORT__, $=id=>document.getElementById(id);
const nowMs=()=>performance.timeOrigin+performance.now();
var micLate=[], micMode='';
var ctx=null, micOn=true, playAt=0, frames=[], lastURL=null, rx=0, lag=0.15, userOff=0, cam=null, started=false, okT=0, shownN=0;
try{localStorage.removeItem("avoff");}catch(e){}   // sync starts at 0 on every load (not remembered)
function outLat(){return ctx?(ctx.outputLatency||0)+(ctx.baseLatency||0):0;}
// ---- transport: one HTTPS request loop (Safari's WebSocket can hang for minutes; this doesn't, and is as fast).
// The body carries what goes to the laptop, the response what comes back (the character's video + voice, clock pings).
let outq=[], outBytes=0, inflight=false, lastPost=0, flushes=0;
// latency budget: nothing waits more than BUDGET ms to go out. A backlog (the request loop briefly slower than
// real time) used to fill a 3 MB queue = ~5 s of delay that never drained (2026-11-26); now stale video is dropped
// and the encoder sends a keyframe at once (H.264 can't resume mid-stream without one), stale mic audio dropped too.
const BUDGET=250;
function out(m){ m.t=Date.now(); outq.push(m); outBytes+=m.byteLength; post(); return true; }
function trim(force){ const now=Date.now();
  const mic=m=>m[0]===77||m[0]===109;
  if(!force&&!outq.some(m=>(m[0]===72||mic(m))&&now-m.t>BUDGET)) return;
  const before=outq.length; outq=outq.filter(m=>!(m[0]===72||mic(m))||(!force&&now-m.t<=BUDGET&&mic(m)));
  if(outq.length!==before){ flushes++; if(cam) cam.key(); }
  outBytes=outq.reduce((a,m)=>a+m.byteLength,0); }
async function post(){ if(inflight) return; inflight=true; lastPost=Date.now(); trim(false);
  const n=outq.reduce((a,m)=>a+4+m.byteLength,0), b=new Uint8Array(n), dv=new DataView(b.buffer); let i=0;
  for(const m of outq){ dv.setUint32(i,m.byteLength,true); b.set(m,i+4); i+=4+m.byteLength; } outq=[]; outBytes=0;
  try{ const r=await fetch(`https://${location.hostname}:${HP}/up?t=${T}`,{method:"POST",body:b,headers:{"Content-Type":"text/plain"}});
    const a=new Uint8Array(await r.arrayBuffer()); rx+=a.length; okT=Date.now(); let j=0; const v=new DataView(a.buffer);
    while(j+4<=a.length){ const len=v.getUint32(j,true); onDown(a.subarray(j+4,j+4+len)); j+=4+len; }
  }catch(e){ await new Promise(r=>setTimeout(r,300)); }
  inflight=false; if(outq.length) post(); }
setInterval(()=>{ if(!inflight&&Date.now()-lastPost>25) post(); },20);   // keep pulling the downlink even when idle
function onDown(b){ const k=b[0], d=new DataView(b.buffer,b.byteOffset,b.byteLength);
  if(k===80){ const o=new Uint8Array(17), v=new DataView(o.buffer); v.setUint8(0,81); v.setFloat64(1,d.getFloat64(1,true),true); v.setFloat64(9,nowMs(),true); out(o); return; }
  if(k===75){ if(cam) cam.key(); return; }        // 'K': the pipeline (re)connected and needs a keyframe
  if(k===70){ trim(true); return; }                // 'F': the laptop sees old frames: drop the backlog
  if(k===65&&ctx){ // 'A' the character's voice: s16le 48 kHz mono, scheduled with a small jitter buffer
    const s=new Int16Array(b.slice(1).buffer), buf=ctx.createBuffer(1,s.length,48000), dd=buf.getChannelData(0);
    for(let i=0;i<s.length;i++)dd[i]=s[i]/32768;
    const now=ctx.currentTime, l=lag+Math.max(0,-userOff/1000); if(playAt<now+0.04||playAt>now+0.2+l)playAt=now+l;   // never let the voice buffer grow past lag+0.2 s
    const nd=ctx.createBufferSource(); nd.buffer=buf; nd.connect(ctx.destination); nd.start(playAt); playAt+=buf.duration; return; }
  if(k===86){ // 'V' JPEG: shown when its audio is heard, so picture and voice stay together
    frames.push([performance.now()+Math.max(0,ctx?(playAt-ctx.currentTime):0)*1000+outLat()*1000+Math.max(0,userOff), new Blob([b.subarray(1)],{type:"image/jpeg"})]);
    if(frames.length>40)frames.shift(); } }
function show(){const now=performance.now(); let f=null;
  while(frames.length&&frames[0][0]<=now)f=frames.shift();
  if(f){shownN++; const u=URL.createObjectURL(f[1]); $("v").src=u; if(lastURL)URL.revokeObjectURL(lastURL); lastURL=u;}
  requestAnimationFrame(show);}
setInterval(()=>{ if(!started) return;
  const c=cam?cam.stats():{}; const st={buf_ms:ctx?Math.round(Math.max(0,playAt-ctx.currentTime)*1000):0, out_lat_ms:Math.round(outLat()*1000),
    shown_fps:+(shownN/2).toFixed(1), queued:frames.length, up_queue:outq.length, mic_mode:micMode, mic_cb_late_ms:micLate.length?Math.round(micLate.slice().sort((a,b)=>a-b)[micLate.length>>1]):null, enc_restarts:cam?cam.restarts():0, up_kb:Math.round(outBytes/1024), flushes, offset_ms:userOff, sr:ctx?ctx.sampleRate:0, camera:c.label?`${c.label} ${c.camera}`:"", enc_fps:c.enc_fps!=null?+(c.enc_fps/2).toFixed(1):null};
  shownN=0; out(new TextEncoder().encode("S"+JSON.stringify(st)));
  $("st").textContent=(Date.now()-okT<2000?"connected":"reconnecting…")+` · ${(rx*8/1e6/2).toFixed(1)} Mbit/s`; rx=0; },2000);
async function start(){ if(started) return; started=true;
  $("start").disabled=true; $("start").textContent="Starting…";
  // never wait for the audio to unlock: iOS only allows it after a tap, and an automatic start (page reopened) has
  // none -- the page then hung at "Starting..." (2026-11-26). Camera, mic and link start regardless; sound resumes
  // on the next tap ("Tap to resume sound").
  ctx=new (window.AudioContext||window.webkitAudioContext)(); ctx.resume().catch(()=>{});
  const ac={echoCancellation:true,noiseSuppression:true,autoGainControl:true,channelCount:1};
  try{
    if(CAM&&H264){ // your camera IS the source video: any of the phone's cameras, hardware H.264, capture times
      if(!("VideoEncoder" in window)) throw new Error("this browser has no WebCodecs video encoder");
      cam=CamEngine({video:$("me"), bitrate:4e6, onerror:m=>{$("st").textContent=m;},
        send:(cap,sq,flags,data)=>{ const o=new Uint8Array(14+data.byteLength), v=new DataView(o.buffer); v.setUint8(0,72); v.setFloat64(1,cap,true);
          v.setUint32(9,sq,true); v.setUint8(13,flags); o.set(data,14); out(o); }});
      await cam.start(); $("me").hidden=false; $("camb").hidden=false;
    } else if(CAM){ // the persona sees you: JPEG ~15 fps
      const vs=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:640},height:{ideal:480}}});
      const me=$("me"); me.srcObject=vs; me.hidden=false; me.style.transform="scaleX(-1)"; await me.play().catch(()=>{});
      const cv=document.createElement("canvas"), g=cv.getContext("2d"); let busy=false;
      setInterval(()=>{ if(busy||!me.videoWidth||outBytes>300000)return; const s=Math.min(1,640/Math.max(me.videoWidth,me.videoHeight));
        cv.width=me.videoWidth*s|0; cv.height=me.videoHeight*s|0; g.drawImage(me,0,0,cv.width,cv.height); busy=true;
        cv.toBlob(async b=>{busy=false; if(!b)return; const d=new Uint8Array(await b.arrayBuffer()), o=new Uint8Array(d.length+1); o[0]=67; o.set(d,1); out(o);},"image/jpeg",0.7); },66);
    }
    const ms=await navigator.mediaDevices.getUserMedia({audio:ac});
    // Mic capture on the AUDIO THREAD (AudioWorklet, 128-sample quanta). The old ScriptProcessor delivered the mic
    // ~400 ms after the sound on iOS (physical flash/beep probe, tools/av_probe.py, 2026-11-26): real delay AND
    // timestamps that claimed old audio was fresh, so the voice played ~300 ms after the lips. Each chunk carries
    // the capture time of its first sample on the phone clock (the video's), mapped from the audio clock.
    const src=ctx.createMediaStreamSource(ms), r=48000/ctx.sampleRate, z=ctx.createGain(); z.gain.value=0; z.connect(ctx.destination);
    function sendMic(x, ctxT, cbMs){            // x: Float32 at ctx.sampleRate, ctxT: context time of x[0]
      const n=Math.floor(x.length*r), o=new Int16Array(n);
      for(let i=0;i<n;i++){const v=micOn?x[Math.min(x.length-1,Math.floor(i/r))]:0; o[i]=Math.max(-1,Math.min(1,v))*32767;}
      let cap=cbMs;
      try{ const ts=ctx.getOutputTimestamp&&ctx.getOutputTimestamp();
        if(ts&&ts.performanceTime>0&&ctxT!=null) cap=performance.timeOrigin+ts.performanceTime+(ctxT-ts.contextTime)*1000; }catch(_){}
      micLate.push(cbMs-cap); if(micLate.length>50) micLate.shift();
      const m=new Uint8Array(9+o.byteLength); m[0]=109; new DataView(m.buffer).setFloat64(1,cap,true); m.set(new Uint8Array(o.buffer),9); out(m); }
    let worklet=false;
    if(ctx.audioWorklet&&window.AudioWorkletNode){
      try{
        const code=`class M extends AudioWorkletProcessor{constructor(){super();this.b=[];this.n=0;}
          process(i){const x=i[0]&&i[0][0]; if(x){ if(!this.n) this.t=currentTime; this.b.push(x.slice()); this.n+=x.length;
            if(this.n>=1024){ const o=new Float32Array(this.n); let k=0; for(const a of this.b){o.set(a,k);k+=a.length;}
              this.port.postMessage({t:this.t,d:o},[o.buffer]); this.b=[]; this.n=0; } } return true; } }
          registerProcessor("mic",M);`;
        await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([code],{type:"application/javascript"})));
        const node=new AudioWorkletNode(ctx,"mic",{numberOfInputs:1,numberOfOutputs:1,channelCount:1});
        node.port.onmessage=({data})=>sendMic(data.d, data.t, nowMs()-data.d.length/ctx.sampleRate*1000);
        src.connect(node); node.connect(z); worklet=true;
      }catch(err){ console.log("audioWorklet failed, using ScriptProcessor: "+err); }
    }
    if(!worklet){                                // fallback (older browsers)
      const proc=ctx.createScriptProcessor(1024,1,1);
      proc.onaudioprocess=e=>{ const x=e.inputBuffer.getChannelData(0), dur=x.length/ctx.sampleRate;
        sendMic(x, e.playbackTime>0?e.playbackTime-2*dur:null, nowMs()-dur*1000); };
      src.connect(proc); proc.connect(z);
    }
    micMode=worklet?"worklet":"scriptprocessor";
  }catch(e){ started=false; $("start").disabled=false; $("start").textContent="Start talking to __TITLE__"; alert("Camera/microphone: "+(e.message||e)); return; }
  $("go").hidden=true; $("ui").hidden=false; requestAnimationFrame(show); wake();
  try{localStorage.setItem("autostart","1")}catch(_){}
  setTimeout(()=>{ if(ctx.state!=="running") $("resume").hidden=false; },600); }
$("start").onclick=()=>{ if(ctx&&ctx.state!=="running") ctx.resume().catch(()=>{}); if(!started) start(); };
document.addEventListener("touchend",()=>{ if(ctx&&ctx.state!=="running") ctx.resume().then(()=>{$("resume").hidden=ctx.state==="running";}).catch(()=>{}); },{passive:true});
$("resumeb").onclick=async()=>{ await ctx.resume().catch(()=>{}); $("resume").hidden=ctx.state==="running"; };
$("camb").onclick=()=>{ $("camctl").hidden=!$("camctl").hidden; };
function showMic(){ $("mute").textContent=micOn?"Mic on":"Mic muted"; $("mute").className=micOn?"":"off"; }
try{ micOn=localStorage.getItem("micOn")!=="0"; }catch(_){}           // mic state survives a reload
showMic();
$("mute").onclick=()=>{micOn=!micOn; showMic(); try{localStorage.setItem("micOn",micOn?"1":"0");}catch(_){} };
// the slider hides behind the Sync button (a stray swipe on the always-visible slider set -400 unnoticed, 2026-11-26)
function setOff(v){ userOff=v; $("off").value=v; $("syncb").textContent="Sync "+(v>0?"+":"")+v; }
setOff(userOff);
$("syncb").onclick=()=>{ $("sync").hidden=!$("sync").hidden; };
$("off").oninput=e=>setOff(+e.target.value);
$("off0").onclick=()=>setOff(0);
// resilience: coming back to the tab (or reopening the page) restores camera, sound and the link by itself
let wl=null; async function wake(){ try{ if(!wl||wl.released) wl=await navigator.wakeLock.request("screen"); }catch(_){} }
function revive(){ if(document.visibilityState!=="visible"||!started) return;
  if(cam) cam.ensure(); if(ctx&&ctx.state!=="running") ctx.resume().then(()=>{$("resume").hidden=ctx.state==="running";}).catch(()=>{$("resume").hidden=false;});
  post(); wake(); }
document.addEventListener("visibilitychange",revive); window.addEventListener("pageshow",revive); window.addEventListener("focus",revive);
try{ if(localStorage.getItem("autostart")) setTimeout(()=>{ start(); },300); }catch(_){}
</script></body></html>"""

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
