"""Build the voice-model comparison page (single HTML with embedded audio + timelines) from results_*/report."""
import base64, html, json, sys
from pathlib import Path
R = Path(sys.argv[1]); OUT = Path(sys.argv[2])
b64 = lambda p: base64.b64encode(Path(p).read_bytes()).decode()
sc = json.load(open(R.parent / "scenario.json"))
import subprocess
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(R.parent / "scenario.wav"), "-b:a", "64k", str(R / "you.mp3")], check=True)
M = {
 "pp": dict(name="NVIDIA PersonaPlex-7B", lic="NVIDIA Open Model License (commercial OK)", verdict="Fastest clean turn-taking",
     tone="good", notes=["Answers 0.36 s after you stop, the fastest of the three, and stays on topic (volcanoes, then underwater volcanoes, then your hike).",
     "Yields 0.56 s after you cut in, but the voice stops abruptly: a 60 dB drop within about 40 ms. That's the hard cut you want to avoid. A short audio fade on our side could soften it.",
     "Stays silent while you talk over it, then replies. Polite, but not 'dual talking'.",
     "Takes a persona prompt plus 18 preset voices, so it can be the character's personality."]),
 "mo": dict(name="Kyutai Moshi (moshiko)", lic="CC-BY-4.0 (commercial OK with attribution)", verdict="Most natural overlap, weakest manners",
     tone="warn", notes=["The only model that actually talks while you talk and fades out instead of cutting off (no drop steeper than 20 dB per 40 ms).",
     "It also barges in: it says 'Hey there, what's going on' during your question, which is why its response latency shows as negative.",
     "Takes 1.6 s to yield when interrupted, then acknowledges it ('Yes, underwater volcanoes'). It talked through 2 of the 3 'mm-hmm/yeah/right' moments, which is correct behaviour.",
     "Content is simpler and a little rambly. No persona prompt in this checkpoint (a fixed male voice)."]),
 "vc": dict(name="NVIDIA NemotronLabs VoiceChat-11B", lic="OpenMDW 1.1 (commercial OK)", verdict="Best content, stiffest turn-taking",
     tone="bad", notes=["The most fluent, polished answers. It's an 11B model built on Nemotron Nano, and it supports tool calls.",
     "Replies after 0.8 s. When interrupted it stops in 0.74 s, dropping straight to digital silence: the hardest cut of the three.",
     "Stopped at all 3 short backchannels instead of talking through them.",
     "Real-time factor 3.2 here: the offline checkpoint path is not NVIDIA's optimized streaming container (vLLM), so this does not reflect its live speed. Its live speed is untested."]),
}
def pill(t, k): return f'<span class="pill {k}">{html.escape(t)}</span>'
rows, sections = [], []
for k, m in M.items():
    d = json.load(open(R / f"{k}.json"))
    tr = [t for t in d["transcript"] if "thanks for watching" not in t[2].lower()]
    rl = d["response_latency_s"]; iy = d["interrupt_yield_s"]
    cut = d["interrupt_cutoff"]; bc = d["backchannel_kept_talking"]; ov = d["talkover_overlap_s"]; rtf = d.get("rtf")
    rows.append(f"""<tr><th scope="row">{m['name']}</th>
      <td class="num">{'started during your question' if rl is not None and rl < 0 else f'{rl:.2f} s'}</td>
      <td class="num">{f'{iy:.2f} s' if iy is not None else '–'}</td>
      <td>{pill('fade', 'good') if cut == 'fade' else pill('hard cut', 'bad') if cut == 'hard cut' else pill(cut, 'mute')}</td>
      <td class="num">{bc}</td><td class="num">{ov:.1f} s</td>
      <td class="num">{pill(f'{rtf:.2f}×', 'good' if rtf < 1 else 'warn') if rtf is not None else '–'}</td></tr>""")
    trh = "".join(f'<li><span class="ts">{s:.1f}–{e:.1f}s</span> {html.escape(t)}</li>' for s, e, t in tr)
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in m["notes"])
    sections.append(f"""<section class="model" id="{k}">
  <header><h2>{m['name']}</h2>{pill(m['verdict'], m['tone'])}</header>
  <p class="lic">{m['lic']}</p>
  <div class="players">
    <figure><figcaption>Headphones: you left, model right</figcaption><audio controls preload="none" src="data:audio/mpeg;base64,{b64(R / f'{k}_stereo.mp3')}"></audio></figure>
    <figure><figcaption>Speakers: mixed</figcaption><audio controls preload="none" src="data:audio/mpeg;base64,{b64(R / f'{k}_mono.mp3')}"></audio></figure>
  </div>
  <img class="tl" alt="Timeline of who talks when: you (blue) and {m['name']} (amber)" src="data:image/png;base64,{b64(R / f'{k}_timeline.png')}">
  <div class="cols"><div><h3>What I heard in the numbers</h3><ul>{notes}</ul></div>
  <div><h3>What it said</h3><ol class="tr">{trh}</ol></div></div>
</section>""")
turns = "".join(f'<li><span class="ts">{t["start"]:.1f}–{t["end"]:.1f}s</span> <b>{t["kind"]}</b> {html.escape(t["text"])}</li>' for t in sc["turns"])
page = f"""<title>Full-Duplex Voice Shootout</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one column of model "takes", like a studio session sheet; summary table first */
:root {{ --bg:#f6f7f9; --fg:#1b1f27; --mut:#5c6474; --card:#ffffff; --line:#dde1e8; --acc:#2f6fd6;
  --good:#1f8a4c; --goodbg:#e3f4ea; --warn:#9a6400; --warnbg:#fbf0d9; --bad:#b4372f; --badbg:#fbe5e3;
  --sans:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#11141a; --fg:#e6e9ef; --mut:#9aa3b2; --card:#181c24; --line:#2a303b; --acc:#6aa2ff;
  --good:#5fd38d; --goodbg:#16301f; --warn:#f0b357; --warnbg:#33270f; --bad:#ff7b72; --badbg:#3a1a18; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#11141a; --fg:#e6e9ef; --mut:#9aa3b2; --card:#181c24; --line:#2a303b; --acc:#6aa2ff;
  --good:#5fd38d; --goodbg:#16301f; --warn:#f0b357; --warnbg:#33270f; --bad:#ff7b72; --badbg:#3a1a18; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 var(--sans); }}
main {{ max-width:1040px; margin:0 auto; padding-block:28px 60px; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); display:grid; gap:28px; }}
h1 {{ font-size:1.9rem; margin:0; text-wrap:balance; font-weight:600 }} h2 {{ font-size:1.25rem; margin:0; font-weight:600 }}
h3 {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; color:var(--mut); margin:0 0 6px; font-weight:600 }}
.lede {{ color:var(--mut); max-width:68ch; margin:6px 0 0 }}
.tablewrap {{ overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px }}
table {{ border-collapse:collapse; width:100%; min-width:760px; font-size:.92rem }}
th, td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:middle }}
thead th {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:var(--mut); font-weight:600 }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom:0 }}
.num {{ font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap }}
.pill {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:.78rem; font-weight:500; white-space:nowrap }}
.good {{ color:var(--good); background:var(--goodbg) }} .warn {{ color:var(--warn); background:var(--warnbg) }}
.bad {{ color:var(--bad); background:var(--badbg) }} .mute {{ color:var(--mut); background:var(--line) }}
.model {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; display:grid; gap:12px; min-width:0 }}
.model header {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap }}
.lic {{ margin:0; color:var(--mut); font-size:.85rem }}
.players {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(260px,1fr)); gap:12px }}
figure {{ margin:0 }} figcaption {{ font-size:.8rem; color:var(--mut); margin-bottom:4px }} audio {{ width:100% }}
.tl {{ width:100%; border-radius:6px; background:#fff; border:1px solid var(--line) }}
.cols {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(300px,1fr)); gap:18px }}
.cols > div {{ min-width:0 }} ul, ol {{ margin:0; padding-left:1.2em }} li {{ margin:3px 0 }}
.tr {{ list-style:none; padding:0; font-size:.9rem }} .ts {{ font-family:var(--mono); color:var(--mut); font-size:.78rem; margin-right:6px }}
.box {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; display:grid; gap:8px }}
.box p {{ margin:0; max-width:75ch }}
</style>
<main>
<header><h1>Full-Duplex Voice Shootout</h1>
<p class="lede">Three open speech-to-speech models, none going through text, each given the same scripted conversation in real time:
a question, an interruption, three short backchannels and a long talk-over. Listen with headphones to hear who talks over whom.
Run 2026-11-23 on one RTX 6000 Ada (48 GB), one take per model.</p></header>

<div class="box"><h3>Recommendation</h3>
<p><b>PersonaPlex-7B</b> is the best base for the AI-persona mode. It has the quickest, most sensible turns, a persona prompt with voice choice, and runs at 0.26× real time.
Its weakness is the abrupt stop when you interrupt, which we can soften with a fade in our audio stage. <b>Moshi</b> is the only one with real dual talking and soft yields, but it barges in and has no persona prompt.
<b>VoiceChat-11B</b> gives the best answers but has the stiffest turn-taking, and its live speed is unproven.</p></div>

<div class="tablewrap"><table>
<thead><tr><th>Model</th><th>Response latency</th><th>Yield when interrupted</th><th>How it stops</th><th>Talked through backchannels</th><th>Talked during your talk-over</th><th>Speed (real-time factor)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>

<div class="box"><h3>Your side of the conversation</h3>
<audio controls preload="none" src="data:audio/mpeg;base64,{b64(R / 'you.mp3')}"></audio>
<ol class="tr">{turns}</ol>
<p class="lic">Voiced by Kokoro TTS. Real-time factor: compute time ÷ audio time for the streaming run, with model loading removed; below 1 means it keeps up live.
"How it stops" looks at the model's level over its last 160 ms: a drop of more than 25 dB within 40 ms counts as a hard cut.</p></div>
{''.join(sections)}
<div class="box"><h3>Limits of this test</h3>
<p>One take per model (they sample randomly, so a second take will differ), a synthetic user voice, and a fixed script that doesn't wait for the model.
Commercial APIs (OpenAI Realtime, Gemini Live, Hume EVI, Nova Sonic) aren't included because there are no keys for them yet.
MiniCPM-o 4.5 (Apache-2.0, full-duplex) is the next open model to add.</p></div>
</main>"""
OUT.write_text(page); print(OUT, OUT.stat().st_size // 1024, "KB")
