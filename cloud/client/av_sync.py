"""Model-agnostic audio/video sync for live edit pipelines (camera -> any AI model -> played video + your voice).

The picture comes back through a pipeline with a varying delay; your voice (the mic) does not. AVSync delays the mic by

    delay = camera chain (video arrives after its sound; measured by the sync test, saved per input + audio source;
                          ~0 when the audio travels with the video, e.g. a source that already lip-syncs it)
          + pipeline     (input frame arrived -> its output frame played; measured continuously, median of ~5 s)
          - the monitor's playback buffer + a manual adjust (normally 0)

and plays it to the default output (headphones), following the live pipeline delay in 10 ms steps. An optional voice
stage (voice_stage.py, e.g. RVC) converts the voice inside that delay: its own delay is counted, not added.

Any pipeline integrates it with these calls (live_bridge.py is the reference):

    sync = AVSync(input_kind="net", input_name=url, monitor_source="default")   # monitor_source=None: measure only
    source.on_frame = sync.on_input            # input thread, at every frame's arrival: (t_arrival, frame)
    sync.on_played(t_arrival, frame)           # at playout, with the ARRIVAL time of the input frame it came from
    sync.status() -> dict                      # pipeline_ms, chain_ms, delay_ms, monitor stats, last test
    sync.start_test()                          # beep + flash test (also measures and saves the camera chain)
    sync.record_av(path)                       # optional: played video + heard audio -> mp4 at close()
    sync.close()

Sync test: a full-screen window flashes and beeps at every flash (own process). The camera sees the flash, the mic hears
the beep. Input arrival vs undelayed mic -> camera chain (saved, applied from then on); played video vs delayed audio ->
the remaining A/V offset, plus a clip in test-outputs/av-sync-test.mp4. Unattended version with a virtual
camera + mic: tools/av_sync_selftest.sh.
"""

import bisect
import collections
import json
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TMP = REPO / ".tmp"
CHAIN_FILE = TMP / "av_chain.json"
CLIP = Path(__file__).resolve().parents[2] / "test-outputs" / "av-sync-test.mp4"
SR, BLOCK = 48000, 480   # 10 ms audio blocks
PLAY_LATENCY_MS = 30     # pacat --latency-msec
DEADBAND_BLOCKS = 3      # monitor: delay errors below 30 ms are left alone (no audible correction)
FORCE_BLOCKS = 8         # ... above 80 ms they are corrected even mid-speech (crossfaded)
FORCE_GAP = 20           # ... at most one mid-speech correction per 200 ms
JUMP_BLOCKS = 50         # ... above 500 ms (start, pipeline restart) it snaps at once
QUIET_RMS = 0.004        # ... otherwise only in pauses (block RMS below about -48 dBFS)


def _rms(b):
    x = np.frombuffer(b, np.int16).astype(np.float32) / 32768
    return float(np.sqrt((x * x).mean()))


def _fade(a, out_):
    """A 10 ms block fading out (out_=True) or in."""
    x = np.frombuffer(a, np.int16).astype(np.float32)
    w = np.sin(np.linspace(0, np.pi / 2, len(x), dtype=np.float32))
    return (x * (w[::-1] if out_ else w)).astype(np.int16).tobytes()


def _xfade(a, b):
    """One 10 ms block: a fading out into b fading in (equal power): drops 10 ms without a click."""
    x = np.frombuffer(a, np.int16).astype(np.float32)
    w = np.sin(np.linspace(0, np.pi / 2, len(x), dtype=np.float32))
    y = np.frombuffer(b, np.int16).astype(np.float32)
    return np.clip(x * w[::-1] + y * w, -32768, 32767).astype(np.int16).tobytes()
TEST_S = 15


# ---------------------------------------------------------------- delayed mic monitor

class AudioMonitor:
    """Mic (PulseAudio source via ffmpeg) -> delay line -> pacat (default output). get_delay_ms() is polled every
    10 ms block: too little delay plays silence while the buffer grows, too much drops the oldest blocks (+-10 ms)."""

    def __init__(self, source, get_delay_ms, voice=None):
        self.source, self.get_delay_ms, self.voice = source, get_delay_ms, voice
        self.stop = threading.Event()
        self.stats = {"delay_ms": None, "dropped": 0, "inserted": 0}
        self._adjusting = False
        self._smooth = None
        self._since_force = 999
        self.mute = False       # sync test: keep timing, play silence (no speaker -> mic feedback loop)
        self.record = None      # sync test: list -> (write time, block) of what is played
        self.record_in = None   # sync test: list -> (capture time, block) as captured (undelayed)
        self.record_av = None   # AV recording: list -> (write time, block) of what is played

    def start(self):
        self.cap = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-f", "pulse", "-fragment_size", "1920", "-i",
                                     self.source, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                                    stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
        self.play = subprocess.Popen(["pacat", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
                                      f"--latency-msec={PLAY_LATENCY_MS}", "--stream-name=bridge-monitor"],
                                     stdin=subprocess.PIPE)
        threading.Thread(target=self._run_logged, daemon=True).start()
        return self

    def _run_logged(self):
        try:
            self._run()
        except Exception:  # noqa: BLE001  -- never die silently: the audio would just stop
            import traceback
            print("[av_sync] audio monitor crashed:\n" + traceback.format_exc(), flush=True)
            self.stats["crashed"] = True

    def _run(self):
        """Delay accounting in 10 ms blocks: queued here + inside the voice stage (if any, incl. its algorithmic
        delay). The voice stage returns audio in bursts (its block size); counting what is inside it keeps the
        total exact, so a voice change costs no extra delay while the pipeline is slower than the stage."""
        buf = collections.deque()
        silence = bytes(BLOCK * 2)
        voice_on = self.voice is not None
        while not self.stop.is_set():
            b = self.cap.stdout.read(BLOCK * 2)
            if len(b) < BLOCK * 2:
                return
            if self.record_in is not None:
                self.record_in.append((time.time(), b))
            if voice_on and self.voice.failed:   # stage died: continue with the plain mic
                voice_on = False
                buf.clear()
            if voice_on:
                self.voice.feed(b)
                buf.extend(self.voice.drain())
                in_stage = self.voice.pending_blocks()
            else:
                buf.append(b)
                in_stage = 0
            target = self.get_delay_ms()
            if target is None:  # until the first pipeline measurement: play nothing, keep the last 3 s
                while len(buf) > 300:
                    buf.popleft()
                continue
            # smooth the target (time constant ~2 s): the pipeline median wobbles by a frame now and then; the
            # audio follows real changes, not the wobble
            sm = target if self._smooth is None or abs(target - self._smooth) > JUMP_BLOCKS * 10 else \
                self._smooth + (target - self._smooth) * 0.005
            self._smooth = sm
            target_blocks = max(1, int(round(sm / 10)))
            self.stats["delay_ms"] = round(sm)
            have = len(buf) + in_stage
            err = int(round(have - target_blocks))   # + = too much delay (drop), - = too little (insert)
            if err > JUMP_BLOCKS:                # first measurement or a big jump: snap at once
                for _ in range(err):
                    if len(buf) > 1:
                        buf.popleft()
                self.stats["dropped"] += err
                err = 0
            if not buf or err < -JUMP_BLOCKS:    # nothing to play yet, or far too little delay: silence
                out = silence
                self.stats["inserted"] += 1
            else:
                # Follow the pipeline without audible cuts: ignore wobbles below DEADBAND (the median moves by a
                # frame or two), correct during pauses (a dropped or repeated silent block is inaudible), and only a
                # large error is corrected mid-speech, crossfaded. One 10 ms correction per block at most.
                if abs(err) >= DEADBAND_BLOCKS:
                    adjusting = True
                elif abs(err) <= 1:
                    adjusting = False
                else:
                    adjusting = self._adjusting
                self._adjusting = adjusting
                head = buf[0]
                quiet = _rms(head) < QUIET_RMS
                # mid-speech only for a large error, and at most one 10 ms correction per FORCE_GAP blocks
                force = abs(err) >= FORCE_BLOCKS and self._since_force >= FORCE_GAP
                self._since_force += 1
                if adjusting and err > 0 and len(buf) > 1 and (quiet or force):
                    a, b = buf.popleft(), buf.popleft()          # two blocks -> one, crossfaded
                    out = a if quiet else _xfade(a, b)
                    self.stats["dropped"] += 1
                    if not quiet:
                        self._since_force = 0
                elif adjusting and err < 0 and (quiet or force):
                    if quiet:
                        out = silence                            # 10 ms more pause: inaudible
                    else:                                        # 10 ms dip: the head fades out now and is played
                        out = _fade(head, out_=True)             # next, fading in (never a repeated block: buzz)
                        buf[0] = _fade(head, out_=False)
                        self._since_force = 0
                    self.stats["inserted"] += 1
                else:
                    out = buf.popleft()
            try:
                if self.record is not None:
                    self.record.append((time.time(), out))
                if self.record_av is not None and not self.mute:
                    self.record_av.append((time.time(), out))
                self.play.stdin.write(silence if self.mute else out)
            except (BrokenPipeError, ValueError):
                return

    def close(self):
        self.stop.set()
        for p in (getattr(self, "cap", None), getattr(self, "play", None)):
            if p is not None and p.poll() is None:
                p.kill()
        if self.voice is not None:
            self.voice.close()


# ---------------------------------------------------------------- camera chain store

def load_chain():
    """{input kind: {"ms": video arrives this much after its sound, ...}} from the sync tests."""
    try:
        return json.loads(CHAIN_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_chain(kind, ms, source, score):
    d = load_chain()
    d[kind] = {"ms": round(ms), "source": source, "score": round(score, 3), "measured": time.strftime("%Y-%m-%d %H:%M:%S")}
    TMP.mkdir(exist_ok=True)
    CHAIN_FILE.write_text(json.dumps(d))


# ---------------------------------------------------------------- analysis

def sync_offset(frames, audio, pa_latency_ms=PLAY_LATENCY_MS, max_ms=600):
    """frames: [(time, mean brightness)], audio: [(time, s16 block)] -> (audio minus video ms, match score).
    Onsets = brightness rises vs beep energy rises, smoothed 40 ms; positive = audio LATE. Accurate to ~10-15 ms."""
    if len(frames) < 50 or len(audio) < 100:
        return None, 0.0
    t0 = max(frames[0][0], audio[0][0]); t1 = min(frames[-1][0], audio[-1][0])
    grid = np.arange(t0, t1, 0.005)
    ft = np.array([f[0] for f in frames]); fb = np.array([f[1] for f in frames])
    vb = fb[np.clip(np.searchsorted(ft, grid, side="right") - 1, 0, len(fb) - 1)]
    at = np.array([a[0] for a in audio]) + pa_latency_ms / 1000
    ae = np.array([np.abs(np.frombuffer(a[1], np.int16)).mean() for a in audio], np.float64)
    ag = ae[np.clip(np.searchsorted(at, grid, side="right") - 1, 0, len(ae) - 1)]
    box = np.ones(8) / 8
    v = np.convolve(np.maximum(0, np.diff(vb, prepend=vb[0])), box, "same")
    a = np.convolve(np.maximum(0, np.diff(ag, prepend=ag[0])), box, "same")
    best = (None, -1.0)
    for lag in range(-max_ms, max_ms + 1, 5):
        k = lag // 5
        x, y = (a[k:], v[:len(v) - k]) if k >= 0 else (a[:k], v[-k:])
        if len(x) < 200 or x.std() == 0 or y.std() == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best[1]:
            best = (lag, c)
    return best


def _brightness(frame):
    return float(frame[::16, ::16].mean())


# ---------------------------------------------------------------- the pipeline-facing object

class AVSync:
    def __init__(self, input_kind, input_name, monitor_source=None, adjust_ms=0.0, log=print, window_frames=125,
                 voice=None):
        # the camera chain depends on the input AND the audio path (e.g. AirPlay audio already matches its picture, a
        # laptop mic does not): saved per (input, audio source), re-measured by every sync test
        self.kind = f"{input_kind}|{input_name}|{monitor_source or '-'}"
        self.name, self.adjust_ms, self.log = input_name, adjust_ms, log
        self.lat = collections.deque(maxlen=window_frames)   # ms, input arrival -> played (display)
        # the audio follows the median of the last ~15 s: a stall of a few seconds (network, server) delays a
        # minority of frames and leaves the audio alone, instead of a gap and a skip in the voice
        self.lat_long = collections.deque(maxlen=window_frames * 3)
        self.chain_ms = load_chain().get(self.kind, {}).get("ms", 0)
        self.test = {"running": False}
        self._arrivals = None    # test: (t, brightness) at input arrival
        self._played = None      # test: (t, brightness, small frame) at playout
        self.monitor = AudioMonitor(monitor_source, self.delay_ms, voice).start() if monitor_source else None
        self._rec = None         # record_av: [(t, jpeg)] of played frames
        if self.monitor:
            self.log(f"audio monitor: {monitor_source} -> default output, delayed by camera chain {self.chain_ms} ms "
                     "(last sync test) + the live pipeline latency (use headphones)")

    # hooks
    def on_input(self, t_arrival, frame):
        if self._arrivals is not None:
            self._arrivals.append((t_arrival, _brightness(frame)))

    def on_played(self, t_arrival, frame, rec_frame=None):
        now = time.time()
        self.lat.append((now - t_arrival) * 1000)
        self.lat_long.append(self.lat[-1])
        rf = rec_frame if rec_frame is not None else frame
        if self._rec is not None and rf is not None and len(self._rec["frames"]) < 25 * 180:
            import cv2
            h, w = rf.shape[:2]
            sm = cv2.resize(rf, (self._rec["width"], round(self._rec["width"] * h / w / 2) * 2))
            self._rec["frames"].append((now, cv2.imencode(".jpg", sm, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()))
        if self._played is not None and frame is not None:
            import cv2
            h, w = frame.shape[:2]
            self._played.append((now, _brightness(frame), cv2.resize(frame, (240, round(240 * h / w / 2) * 2))))

    # numbers
    def pipeline_ms(self):
        l = sorted(self.lat)
        return l[len(l) // 2] if l else None

    def delay_ms(self):
        l = sorted(self.lat_long)
        p = l[len(l) // 2] if l else None
        return None if p is None else max(0.0, p + self.chain_ms - PLAY_LATENCY_MS + self.adjust_ms)

    def status(self):
        p, d = self.pipeline_ms(), self.delay_ms()
        return {"pipeline_ms": None if p is None else round(p), "chain_ms": self.chain_ms,
                "delay_ms": None if d is None else round(d),
                "monitor": dict(self.monitor.stats) if self.monitor else None, "test": dict(self.test),
                "voice": self.monitor.voice.stats() if self.monitor and self.monitor.voice else None}

    # A/V recording: what you see (played frames) + what you hear (the delayed, possibly converted audio)
    def record_av(self, path, width=480):  # width of the recorded frame (side by side: pass ~960)
        """Record from now until close() (max 3 min) and write an mp4 on one timeline, so the clip shows the real
        A/V sync of the live output."""
        if self.monitor is None:
            self.log("record_av needs the audio monitor")
            return
        self._rec = {"path": Path(path), "width": width, "frames": []}
        self.monitor.record_av = []

    def _finish_record(self):
        rec, audio = self._rec, self.monitor.record_av
        self._rec, self.monitor.record_av = None, None
        if not rec["frames"] or not audio:
            self.log("record_av: nothing recorded")
            return
        import cv2
        frames = [(t, cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)) for t, j in rec["frames"]]
        p = self._write_clip([(t, 0, f) for t, f in frames], audio, rec["path"])
        self.log(f"A/V recording: {p} ({frames[-1][0] - frames[0][0]:.0f} s)")

    # sync test
    def start_test(self, window=True, after_s=0.0):
        if self.test.get("running"):
            return
        threading.Thread(target=self._run_test, args=(window, after_s), daemon=True).start()

    def _run_test(self, window, after_s):
        time.sleep(after_s)
        if self.monitor is None:
            self.log("sync test needs the audio monitor (--audio-monitor)")
            return
        self.test = {"running": True}
        self.log(f"A/V sync test ({TEST_S} s): point the camera at the flashing window; beeps on the SPEAKERS "
                 "(not headphones) so the mic hears them")
        m = self.monitor
        m.mute, m.record, m.record_in, self._arrivals, self._played = True, [], [], [], []
        if window:
            log = TMP / "blink_log.txt"
            TMP.mkdir(exist_ok=True)
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "blink", str(TEST_S), str(log)]).wait()
        else:  # the input itself carries flashes + beeps (tools/av_sync_selftest.sh)
            time.sleep(TEST_S)
        time.sleep(((self.pipeline_ms() or 1000) + 500) / 1000)   # let the last flashes come through
        played, audio, audio_in, arrivals = self._played, m.record, m.record_in, self._arrivals
        self._played, m.record, m.record_in, self._arrivals, m.mute = None, None, None, None, False

        chain, c_chain = sync_offset(arrivals, audio_in, pa_latency_ms=0)
        if chain is not None and c_chain > 0.5:
            self.chain_ms = -chain
            save_chain(self.kind, -chain, self.name, c_chain)
        off, c = sync_offset([(t, b) for t, b, _ in played], audio)
        res = {"running": False, "offset_ms": off, "match": round(c, 2), "chain_ms": self.chain_ms,
               "chain_match": round(c_chain, 2), "pipeline_ms": round(self.pipeline_ms() or 0),
               "delay_ms": m.stats["delay_ms"]}
        if off is None or c < 0.3:
            msg = f"sync test inconclusive (match {c:.2f}): flash filling the camera view? beeps audible to the mic?"
        elif abs(off) > 30:  # below ~30 ms: within the test resolution and below perception
            msg = (f"audio was {abs(off)} ms {'LATE' if off > 0 else 'EARLY'} (match {c:.2f}); camera chain "
                   f"{self.chain_ms} ms now applied automatically and saved; run the test again to verify")
        else:
            msg = f"A/V in sync: audio {off:+d} ms (match {c:.2f}); camera chain {self.chain_ms} ms"
        res["msg"] = msg
        self.test = res
        self.log(msg)
        try:
            res["clip"] = str(self._write_clip(played, audio))
            self.log(f"sync test clip: {res['clip']}")
        except Exception as e:  # noqa: BLE001
            self.log(f"sync test clip failed: {e}")

    @staticmethod
    def _write_clip(played, audio, out=CLIP):
        """Played frames + the delayed audio on one timeline (what you'd see and hear), 25 fps mp4."""
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = max(played[0][0], audio[0][0]); t1 = min(played[-1][0], audio[-1][0])
        # the audio as played: one continuous stream (blocks back to back, as pacat got them), anchored at the
        # first block's time; placing each block at its own jittery timestamp would cut gaps and overlaps
        i0 = int(round((audio[0][0] + PLAY_LATENCY_MS / 1000 - t0) * SR))
        stream = np.frombuffer(b"".join(b for _, b in audio), np.int16)
        pcm = np.zeros(int((t1 - t0) * SR) + SR, np.int16)
        if i0 < 0:
            stream, i0 = stream[-i0:], 0
        n = min(len(stream), len(pcm) - i0)
        pcm[i0:i0 + n] = stream[:n]
        TMP.mkdir(exist_ok=True)
        wav = TMP / "synctest.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
        h, w_ = played[0][2].shape[:2]
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w_}x{h}",
                              "-r", "25", "-i", "-", "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                              "-profile:v", "high", "-level", "4.1", "-c:a", "aac", "-shortest", "-movflags",
                              "+faststart", str(out)], stdin=subprocess.PIPE)
        ft = [f[0] for f in played]
        for k in range(int((t1 - t0) * 25)):
            p.stdin.write(played[max(0, bisect.bisect_right(ft, t0 + k / 25) - 1)][2].tobytes())
        p.stdin.close(); p.wait()
        return out

    def close(self):
        if self._rec is not None:
            try:
                self._finish_record()
            except Exception as e:  # noqa: BLE001
                self.log(f"record_av failed: {e}")
        if self.monitor:
            self.monitor.close()


# ---------------------------------------------------------------- flashing + beeping window (own process)

def _blink(seconds, log_path):
    import cv2
    pa = subprocess.Popen(["pacat", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1", "--latency-msec=20",
                           "--stream-name=sync-test-beep"], stdin=subprocess.PIPE)
    tt = np.arange(int(0.12 * SR)) / SR   # 120 ms 1 kHz beep at every switch to white
    tone = (0.6 * 32767 * np.sin(2 * np.pi * 1000 * tt) * np.minimum(1, np.minimum(tt, 0.12 - tt) / 0.005)
            ).astype(np.int16).tobytes()
    w = "A/V sync test: point the camera at this window"
    cv2.namedWindow(w, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(w, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    black = np.zeros((720, 1280, 3), np.uint8); white = np.full((720, 1280, 3), 255, np.uint8)
    rng = np.random.default_rng(1)
    t_end, level = time.time() + float(seconds), 0
    with open(log_path, "w") as f:
        while time.time() < t_end:
            level = 1 - level
            cv2.imshow(w, white if level else black)
            cv2.waitKey(1)
            if level:
                pa.stdin.write(tone); pa.stdin.flush()
            f.write(f"{time.time():.4f} {level}\n"); f.flush()
            t_next = time.time() + 0.25 + 0.5 * rng.random()  # irregular: one unambiguous best lag
            while time.time() < t_next:
                cv2.waitKey(5)
    cv2.destroyAllWindows()
    pa.kill()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "blink":
    _blink(sys.argv[2], sys.argv[3])
