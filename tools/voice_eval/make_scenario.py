"""Scripted 'user' side for the full-duplex voice tests: one 24 kHz mono wav + the exact timing of every user turn.

Turns (same input for every model, so results compare directly):
  ask        a question, then silence -> response latency
  interrupt  cuts in while the model should still be answering -> how fast / how softly it yields (fade vs hard cut)
  backchannel x3  short "mm-hmm", "yeah", "right" while the model talks -> it should keep talking (not stop)
  talkover   a long statement over the model -> overlap behaviour, then its reply latency

  /workspace/tts/bin/python make_scenario.py --out scenario.wav     (writes scenario.wav + scenario.json)
"""

import argparse
import json

import numpy as np
import soundfile as sf

SR = 24000
TURNS = [  # (start s, kind, text)
    (0.5, "ask", "Hey! Can you tell me a bit about how volcanoes form?"),
    (8.0, "interrupt", "Wait, wait, hold on. Sorry, I actually meant underwater volcanoes. How do those work?"),
    (16.5, "backchannel", "Mm-hmm."),
    (18.5, "backchannel", "Yeah."),
    (20.5, "backchannel", "Right."),
    (24.0, "talkover", "Oh, that's so cool. So this weekend I'm going hiking near Mount Rainier with a couple of friends, "
                       "and I'm hoping we get to see some old lava flows along the trail."),
]
TOTAL = 44.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="scenario.wav")
    p.add_argument("--voice", default="am_michael")
    a = p.parse_args()
    from kokoro import KPipeline
    tts = KPipeline(lang_code="a")
    track = np.zeros(int(TOTAL * SR), np.float32)
    meta = []
    for t0, kind, text in TURNS:
        audio = np.concatenate([np.asarray(r.audio, np.float32) for r in tts(text, voice=a.voice, speed=1.05)])
        nz = np.flatnonzero(np.abs(audio) > 0.01)  # trim TTS padding
        audio = audio[max(0, nz[0] - 240):nz[-1] + 240] if len(nz) else audio
        s = int(t0 * SR)
        track[s:s + len(audio)] += audio[:len(track) - s]
        meta.append({"kind": kind, "text": text, "start": t0, "end": round(t0 + len(audio) / SR, 3)})
    track *= 0.8 / max(1e-6, np.abs(track).max())
    sf.write(a.out, track, SR)
    json.dump({"sr": SR, "total": TOTAL, "turns": meta}, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    for m in meta:
        print(f"{m['start']:5.1f}-{m['end']:5.1f}  {m['kind']:11s} {m['text'][:60]}")


if __name__ == "__main__":
    main()
