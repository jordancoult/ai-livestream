# iPhone (13 Pro) as the JoyAI input camera (Linux laptop, no extra software on the laptop)

The phone streams its camera over Wi-Fi to the bridge with **SRT** (low latency); the bridge receives it with the system
ffmpeg, scales/crops to 720p in the session's orientation and measures the input fps (dashboard: `input N fps`).
Tested 2026-11-22 with a simulated phone stream: input 31.6 fps -> JoyAI 26.9 fps, 0 repeated frames, lpfast 100 %.

## Recommended: AirPlay screen mirroring (no streaming app) — tools/iphone_airplay.sh
Works with any full-screen camera app (e.g. Blackmagic Camera). UxPlay receives the mirror, NVENC re-encodes it and
sends SRT into the bridge (2026-11-23: smooth after two fixes below).
```bash
# 1. bridge:  cloud/facefix/.venv/bin/python cloud/client/live_bridge.py --input srt://0.0.0.0:9000 --portrait ...
# 2. mirror:  tools/iphone_airplay.sh          (same Wi-Fi; HOTSPOT=1 needs the router OFF DFS channels 52-144)
# 3. iPhone:  Control Center -> Screen Mirroring -> JoyAI-Cam, camera app full screen (portrait)
```
- AirPlay delivers frames in pairs: the bridge sends the oldest of the last 2 frames (newest-only halved the rate to
  13 fps -> JoyAI chunks stalled 0.3 s every 0.3 s).
- SPS/PPS repeat on every keyframe (`h264parse config-interval=-1`) so restarting the bridge recovers on its own.
- The laptop hotspot fails on DFS channels (Intel can't host an AP there): move the router to 36-48 or 149-165 to use it.
- Larix Broadcaster (below) also works but was unreliable in practice.

## Alternative: Larix Broadcaster
### Phone (one time)
1. Install **Larix Broadcaster** (free, App Store).
2. Settings -> Connections -> New: URL `srt://<laptop IP>:9000` (laptop IP: `ip -4 addr show wlo1`, e.g. 192.168.78.142),
   mode **Caller**, latency **120 ms**.
3. Settings -> Video: H.264, **1280x720 (or 1920x1080)**, **30 fps**, bitrate 4-6 Mbps, keyframe 1 s; orientation
   **portrait** for TikTok (lock it). Audio can stay on (the bridge ignores it; the stream's audio comes from OBS).
4. Same Wi-Fi as the laptop (5 GHz if possible). Plug the phone in: streaming drains the battery and it throttles hot.

## Laptop
```bash
cloud/facefix/.venv/bin/python cloud/client/live_bridge.py --input srt://0.0.0.0:9000 --portrait ...   # rest as usual
```
Start the bridge first (`waiting for the stream on srt://0.0.0.0:9000?mode=listener`), then tap the red button in
Larix (`stream connected`). If the phone stops, the bridge waits for it again. Firewall: `sudo ufw allow 9000/udp` if
ufw is on. RTMP works too (`--input rtmp://0.0.0.0:1935/live/iphone`, Larix URL `rtmp://<laptop IP>:1935/live/iphone`)
but adds ~1 s; RTSP/UDP URLs are also accepted.

## Frame rate
The pipeline runs at the input's fresh-frame rate (the bridge sends the newest frame, up to `--fps 26`). The dashboard
warns when the input is slower than that. Phone: keep Larix at 30 fps and enough light (iOS lowers the camera's frame
rate in the dark too). Webcam: the bridge locks `exposure_dynamic_framerate=0` on start (`--no-camera-fix` to skip).
