"""Per model: stereo mix (user left, model right) + mono mix as mp3, a timeline PNG (who talks when), a transcript of
the model's speech (faster-whisper small.en, word timestamps) and the metrics (analyze.py) -> eval/report/<model>.*

  /workspace/tts/bin/python report_assets.py --eval /workspace/eval --models pp mo vc
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import segs, vad  # noqa: E402

NAMES = {"pp": "NVIDIA PersonaPlex-7B", "mo": "Kyutai Moshi (moshiko)", "vc": "NVIDIA NemotronLabs VoiceChat-11B",
         "cpm": "MiniCPM-o 4.5"}


def load(p, sr=24000):
    x, s = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if s != sr:
        import librosa
        x = librosa.resample(x, orig_sr=s, target_sr=sr)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", default="/workspace/eval")
    p.add_argument("--models", nargs="+", default=["pp", "mo", "vc"])
    a = p.parse_args()
    E = Path(a.eval)
    R = E / "report"
    R.mkdir(exist_ok=True)
    sc = json.load(open(E / "scenario.json"))
    user = load(E / "scenario.wav")
    from faster_whisper import WhisperModel
    asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=16)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = {}
    for m in a.models:
        wav = E / f"{m}.wav"
        if not wav.exists():
            print("skip", m)
            continue
        mod = load(wav)
        n = min(len(user), len(mod))
        mod = mod[:n] * (0.8 / max(1e-6, np.abs(mod[:n]).max()))
        st = np.stack([user[:n], mod], 1)
        sf.write(R / f"{m}_stereo.wav", st, 24000)
        sf.write(R / f"{m}_mono.wav", (user[:n] + mod) * 0.6, 24000)
        for k in ("stereo", "mono"):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(R / f"{m}_{k}.wav"), "-b:a", "96k",
                            str(R / f"{m}_{k}.mp3")], check=True)
        rtf = float((E / f"{m}.rtf").read_text()) if (E / f"{m}.rtf").exists() else None
        cmd = [sys.executable, str(Path(__file__).parent / "analyze.py"), "--scenario", str(E / "scenario.json"),
               "--model-wav", str(wav), "--name", NAMES.get(m, m), "--out", str(R / f"{m}.json")]
        if rtf is not None:
            cmd += ["--rtf", str(rtf)]
        subprocess.run(cmd, check=True)
        met = json.load(open(R / f"{m}.json"))
        import librosa
        segs_, _ = asr.transcribe(librosa.load(str(wav), sr=16000)[0], language="en", vad_filter=True,
                                  word_timestamps=False)  # array input: faster-whisper's own decoder needs another PyAV
        met["transcript"] = [(round(s.start, 1), round(s.end, 1), s.text.strip()) for s in segs_]
        json.dump(met, open(R / f"{m}.json", "w"), indent=1)
        # timeline
        on_u, _, hop = vad(user[:n], 24000)
        on_m, _, _ = vad(mod, 24000)
        fig, ax = plt.subplots(figsize=(11, 1.6), dpi=110)
        for y, msk, col, lab in ((1, on_u, "#5aa9ff", "you"), (0, on_m, "#f0b357", "model")):
            for s, e in segs(msk, hop):
                ax.broken_barh([(s, e - s)], (y - 0.35, 0.7), color=col)
            ax.text(-0.6, y, lab, ha="right", va="center", fontsize=9)
        for t in sc["turns"]:
            ax.axvline(t["start"], color="#888", lw=0.6, ls=":")
            ax.text(t["start"], 1.55, t["kind"], fontsize=7, color="#555")
        ax.set_xlim(0, n / 24000); ax.set_ylim(-0.6, 1.9); ax.set_yticks([]); ax.set_xlabel("seconds", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(R / f"{m}_timeline.png"); plt.close(fig)
        summary[m] = met
        print(m, {k: v for k, v in met.items() if k not in ("model_segments", "transcript")})
    json.dump(summary, open(R / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
