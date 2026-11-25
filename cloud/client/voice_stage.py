"""Optional voice stage for the delayed audio monitor (av_sync.AudioMonitor): changes the voice without changing the
A/V sync, and adds no latency as long as the stage is faster than the video pipeline.

The stage is a subprocess speaking a trivial protocol (tools/rvc_live/serve.py is the reference): raw s16le mono
48 kHz in on stdin, the converted stream out on stdout sample-for-sample, and 'READY delay_ms=N' on stderr when warm.
The monitor accounts for it exactly: delay of a sample = samples queued in the delay line + samples inside the stage
(sent, not yet returned) + the stage's algorithmic delay. So the voice change is absorbed into the delay the audio
needs anyway to wait for the video; only a stage slower than the whole pipeline would add any delay.

New converters (another RVC model, a zero-shot VC) = one more entry in PRESETS or --voice-cmd.
"""
import collections
import os
import shlex
import subprocess
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BLOCK_BYTES = 480 * 2   # 10 ms at 48 kHz, s16 mono (the monitor's block)

PRESETS = {
    # RVC v2 via Applio's realtime core on the local GPU (~15 ms compute per 100 ms block on an RTX 5080 Laptop)
    "rvc": "{repo}/tools/rvc_live/.venv/bin/python {repo}/tools/rvc_live/serve.py --model {model} --pitch {pitch} "
           "--index-rate {index_rate} --block-ms {block_ms} --gain {gain}",
}


def build_cmd(kind, model=None, pitch=5, index_rate=0.5, block_ms=100, gain=-6.0, cmd=None, host=None):
    """host: run the stage on a GPU machine (cloud/infra/.pod.<host>, set up by cloud/infra/voice_up.sh <host>)
    through an SSH pipe instead of the local GPU. The network round trip is counted in the A/V delay like the rest."""
    if cmd:
        return shlex.split(cmd)
    if host:
        model = f"models/rvc/{Path(model).parent.name}/{Path(model).name}" if model else "models/rvc/karen/karen.pth"
        remote = PRESETS[kind].format(repo=".", model=model, pitch=pitch, index_rate=index_rate, block_ms=block_ms,
                                      gain=gain)
        return ["env", f"POD_FILE={REPO}/cloud/infra/.pod.{host}", "bash", str(REPO / "cloud" / "infra" / "pod.sh"),
                f"cd /workspace/rvc && {remote}"]
    model = model or str(REPO / "models" / "rvc" / "karen" / "karen.pth")
    return shlex.split(PRESETS[kind].format(repo=REPO, model=model, pitch=pitch, index_rate=index_rate,
                                            block_ms=block_ms, gain=gain))


class VoiceStage:
    def __init__(self, cmd, log=print, ready_timeout=180, gpu_share=None):
        self.log, self.cmd = log, cmd
        self.out = collections.deque()      # converted 10 ms blocks
        self.sent = self.moved = 0          # 10 ms blocks fed / handed to the delay line
        self.delay_ms, self.block = None, None
        self.failed = False
        self._partial = b""
        env = dict(os.environ)
        if gpu_share:   # under CUDA MPS: the stage may use only this % of the GPU's SMs, the rest stays free for video
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(gpu_share)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self._ready = threading.Event()
        threading.Thread(target=self._stderr, daemon=True).start()
        if not self._ready.wait(ready_timeout) or self.delay_ms is None:
            self.close()
            raise RuntimeError(f"voice stage did not start: {' '.join(cmd)}")
        threading.Thread(target=self._stdout, daemon=True).start()
        self.log(f"voice stage ready: algorithmic delay {self.delay_ms} ms, block {self.block / 48:.0f} ms")

    def _stderr(self):
        tail = collections.deque(maxlen=20)
        for line in self.p.stderr:
            s = line.decode(errors="replace").rstrip()
            if s.startswith("READY"):
                kv = dict(x.split("=") for x in s.split()[1:])
                self.delay_ms, self.block = float(kv["delay_ms"]), int(kv["block"])
                self._ready.set()
            else:
                tail.append(s)
        if not self._ready.is_set():
            self.log("voice stage exited before READY:\n  " + "\n  ".join(tail))
        self._ready.set()
        self.failed = True

    def _stdout(self):
        while True:
            b = self.p.stdout.read(BLOCK_BYTES)
            if len(b) < BLOCK_BYTES:
                self.failed = True
                self.log("voice stage stopped: audio continues unconverted")
                return
            self.out.append(b)

    # monitor side (one thread)
    def feed(self, block):
        try:
            self.p.stdin.write(block)
            self.p.stdin.flush()
            self.sent += 1
        except (BrokenPipeError, ValueError, OSError):
            self.failed = True

    def drain(self):
        """Converted 10 ms blocks ready now (handed over in order)."""
        got = []
        while self.out:
            got.append(self.out.popleft())
        self.moved += len(got)
        return got

    def pending_blocks(self):
        """Delay inside the stage in 10 ms blocks: fed but not yet handed over (in the stage or ready) + its
        algorithmic delay."""
        return (self.sent - self.moved) + self.delay_ms / 10

    def stats(self):
        return {"delay_ms": self.delay_ms, "in_stage_ms": (self.sent - self.moved) * 10, "failed": self.failed}

    def close(self):
        if self.p.poll() is None:
            self.p.kill()
