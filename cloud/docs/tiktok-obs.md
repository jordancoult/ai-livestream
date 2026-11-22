# Streaming the JoyAI pipeline to TikTok LIVE (OBS on Linux)

```
camera / iPhone -> live_bridge (JoyAI on the 5090 + lpfast on the laptop) -> /dev/video10 (virtual cam "AI_Scope_Cam")
   -> OBS (portrait 1080x1920 scene + your mic, delayed to match) -> TikTok LIVE (RTMP stream key)
```

## One-time setup
1. **OBS**: `sudo apt install obs-studio` (or `flatpak install flathub com.obsproject.Studio`). Not installed yet.
2. **Virtual camera**: already set up (`/dev/video10`, v4l2loopback `exclusive_caps=1`: OBS only lists it while the
   bridge is writing to it, so start the bridge first).
3. **TikTok stream key**: TikTok LIVE Studio is Windows-only; on Linux use OBS with a stream key from TikTok's LIVE
   Center / LIVE Producer (web). Access depends on TikTok's eligibility rules for your account (age, followers, region);
   the key changes per LIVE session.
4. **OBS settings**
   - Video: base and output **1080x1920**, FPS **25** (the pipeline makes ~26 fps; 25 avoids duplicated frames;
     use 30 if TikTok rejects 25).
   - Output (Advanced): encoder **NVIDIA NVENC H.264**, rate control CBR **4500-6000 kbps**, keyframe interval **2 s**,
     profile high. Audio 128-160 kbps.
   - Stream: Service **Custom**, Server = TikTok's RTMP URL, Stream key = TikTok's key.
5. **Scene**: Sources -> + **Video Capture Device (V4L2)** -> device `AI_Scope_Cam` (/dev/video10), resolution
   1080x1920; right-click -> Transform -> Fit to screen. + **Audio Input Capture** (your mic).
6. **Lip sync**: the picture is ~0.8 s behind your voice (camera -> JoyAI -> lpfast; the dashboard shows the
   glass-to-glass latency). Audio Mixer -> gear -> Advanced Audio Properties -> mic **Sync Offset = 800 ms**
   (= the dashboard's glass-to-glass value).

## Going live
```bash
tools/tiktok_live.sh                      # webcam (or: INPUT=srt://0.0.0.0:9000 tools/tiktok_live.sh for the iPhone)
```
Then OBS -> Start Streaming. The bridge's dashboard (http://127.0.0.1:8766) shows input / server / playout fps and the
face pass; prompts, reference and gains change live there.
