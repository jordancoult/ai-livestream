#!/bin/bash
# Unattended A/V sync test of the bridge's --audio-monitor: no camera, mic or speakers.
# A clip with flashes + a beep at every flash is streamed as the "phone" (SRT) while its audio plays into a virtual
# PulseAudio sink that stands in for the mic; the bridge delays that audio by its live pipeline latency and measures
# how far the delayed beeps are from the flashes in the played video (+ writes ~/Videos/ai-livestream-tests/av-sync-test.mp4).
#   tools/av_sync_selftest.sh [joyai ws url]      (JoyAI tunnel up, no other bridge session)
set -u
cd "$(dirname "$0")/.."
URL=${1:-ws://127.0.0.1:18080/ws}; PORT=9020; T=.tmp/avsync; mkdir -p $T
PY=cloud/facefix/.venv/bin/python
$PY - <<'PY'
import cv2, numpy as np, subprocess, wave
T=".tmp/avsync"; SR=48000; rng=np.random.default_rng(7)
flips=[]; t=0.5; l=0
while t<40: l=1-l; flips.append((t,l)); t+=0.3+0.5*rng.random()
def state(tt):
    s=0
    for a,b in flips:
        if a<=tt: s=b
        else: break
    return s
v=cv2.VideoCapture("assets/selftest_cam.mp4")  # optional (gitignored): any face clip; else a gray frame
fr=[]
while len(fr)<300:
    ok,f=v.read()
    if not ok: break
    fr.append(cv2.resize(f,(720,1280)))
p=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","bgr24","-s","720x1280","-r","30","-i","-",
    "-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p",f"{T}/v.mp4"],stdin=subprocess.PIPE)
for k in range(30*40):
    f=fr[k%len(fr)].copy() if fr else np.full((1280,720,3),90,np.uint8)
    if state(k/30): f=(0.25*f+0.75*255).astype(np.uint8)
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
pcm=np.zeros(SR*40,np.float32); tt=np.arange(int(0.12*SR))/SR
tone=0.5*np.sin(2*np.pi*1000*tt)*np.minimum(1,np.minimum(tt,0.12-tt)/0.005)
for a,b in flips:
    if b: i=int(a*SR); pcm[i:i+len(tone)]+=tone[:max(0,min(len(tone),len(pcm)-i))]
with wave.open(f"{T}/a.wav","wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((pcm*32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg","-v","error","-y","-i",f"{T}/v.mp4","-i",f"{T}/a.wav","-c:v","copy","-c:a","aac","-shortest",f"{T}/av.mp4"],check=True)
print("clip", len(flips)//2, "flashes")
PY
M1=$(pactl load-module module-null-sink sink_name=avsync_in sink_properties=device.description=avsync_in)
M2=$(pactl load-module module-null-sink sink_name=avsync_out sink_properties=device.description=avsync_out)
cleanup(){ [ -n "${SND:-}" ] && kill $SND 2>/dev/null; pactl unload-module $M1; pactl unload-module $M2; }
trap cleanup EXIT
PULSE_SINK=avsync_out timeout 110 $PY cloud/client/live_bridge.py --url "$URL" --input "srt://0.0.0.0:$PORT" \
  --output $T/played.mp4 --portrait --ref assets/char_face.png --prompt "$(cat cloud/pipelines/joyai/prompts/gstar.txt)" \
  --facefix-preset gstar --audio-monitor --monitor-source avsync_in.monitor --synctest-auto 25 --seconds 60 \
  --no-reconnect > $T/bridge.log 2>&1 &
BR=$!
sleep 8
timeout 70 ffmpeg -v error -re -i $T/av.mp4 -map 0:v -c:v libx264 -preset veryfast -tune zerolatency -g 30 -b:v 5M \
  -f mpegts "srt://127.0.0.1:$PORT?mode=caller&latency=20" -map 0:a -f pulse -device avsync_in avsync > $T/sender.log 2>&1 &
SND=$!
wait $BR
grep -aE "A/V|sync test|RESULT|Traceback" $T/bridge.log | cut -c1-400
