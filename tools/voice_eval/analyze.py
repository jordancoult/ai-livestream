"""Score a full-duplex model's output against the scripted user turns (scenario.json).

  python analyze.py --scenario scenario.json --model-wav out_model.wav [--rtf 0.4] --name PersonaPlex --out res.json

Model voice activity = 20 ms frames above (peak - 32 dB) with 250 ms hangover. Metrics:
  response_latency_s      user 'ask' ends -> model starts (negative = it started before you finished)
  interrupt_yield_s       you cut in -> the model goes quiet (None if it wasn't talking when you cut in)
  interrupt_cutoff        'fade' / 'hard cut' / 'n/a': how the model's voice ends after the interruption (level drop
                          over its last 160 ms; a hard cut drops >25 dB within 40 ms)
  backchannel_kept_talking  of the 'mm-hmm/yeah/right' moments where the model was talking: how many it talked through
  talkover_overlap_s      seconds the model spoke while you talked over it
  reply_after_talkover_s  your long turn ends -> model starts
  rtf                     wall-clock / audio duration of the streaming run (< 1 = keeps up in real time)
"""

import argparse
import json

import numpy as np
import soundfile as sf


def vad(x, sr, hop=0.02, rel_db=-32, hang=0.25):
    n = int(hop * sr)
    fr = x[:len(x) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    on = db > (np.percentile(db, 99) + rel_db)
    h = int(hang / hop)
    out = on.copy()
    last = -10 ** 9
    for i, v in enumerate(on):
        if v:
            last = i
        elif i - last <= h:
            out[i] = True
    return out, db, hop


def segs(mask, hop):
    s, res = None, []
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            res.append((s * hop, i * hop))
            s = None
    if s is not None:
        res.append((s * hop, len(mask) * hop))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--model-wav", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--rtf", type=float, default=None, help="real-time factor of the streaming run")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sc = json.load(open(a.scenario))
    x, sr = sf.read(a.model_wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    on, db, hop = vad(x, sr)
    S = segs(on, hop)
    speaking = lambda t: bool(on[min(len(on) - 1, int(t / hop))])  # noqa: E731
    next_on = lambda t: next((s for s, e in S if s >= t - 1e-9 or (s < t < e and False)), None)  # noqa: E731
    turns = {k: [t for t in sc["turns"] if t["kind"] == k] for k in ("ask", "interrupt", "backchannel", "talkover")}
    r = {"name": a.name, "model_segments": [(round(s, 2), round(e, 2)) for s, e in S]}
    ask = turns["ask"][0]
    st = next((s for s, e in S if e > ask["end"] - 1.5 and s > ask["start"]), None)
    r["response_latency_s"] = round(st - ask["end"], 2) if st is not None else None
    it = turns["interrupt"][0]
    if speaking(it["start"]):
        off = next(e for s, e in S if s <= it["start"] < e)
        r["interrupt_yield_s"] = round(off - it["start"], 2)
        i_end = int((off - 0.25) / hop)  # hangover removed -> the real last voiced frame
        tail = db[max(0, i_end - 8):i_end + 3]
        drop = float(tail.max() - tail.min())
        steep = float(max(tail[j] - tail[j + 2] for j in range(len(tail) - 2))) if len(tail) > 2 else 0.0
        r["interrupt_cutoff"] = "hard cut" if steep > 25 else "fade"
        r["interrupt_tail_drop_db"] = round(drop, 1)
        r["interrupt_steepest_db_per_40ms"] = round(steep, 1)
    else:
        r["interrupt_yield_s"], r["interrupt_cutoff"] = None, "n/a (not talking when you cut in)"
    r["reply_after_interrupt_s"] = (lambda s: round(s - it["end"], 2) if s is not None else None)(
        next((s for s, e in S if s > it["end"] - 0.5), None))
    kept, tested = 0, 0
    for b in turns["backchannel"]:
        if speaking(b["start"] - 0.1):
            tested += 1
            kept += all(speaking(t) for t in np.arange(b["start"], b["end"] + 0.6, 0.1))
    r["backchannel_kept_talking"] = f"{kept}/{tested}" if tested else "n/a (silent at those moments)"
    to = turns["talkover"][0]
    m = on[int(to["start"] / hop):int(to["end"] / hop)]
    r["talkover_overlap_s"] = round(float(m.sum()) * hop, 2)
    r["reply_after_talkover_s"] = (lambda s: round(s - to["end"], 2) if s is not None else None)(
        next((s for s, e in S if s > to["end"] - 0.5), None))
    r["model_talk_s"] = round(float(on.sum()) * hop, 1)
    if a.rtf is not None:
        r["rtf"] = a.rtf
    json.dump(r, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "model_segments"}))


if __name__ == "__main__":
    main()
