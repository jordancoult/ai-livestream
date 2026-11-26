# Pipeline output → OBS → TikTok (and video calls)

The live character run (`tools/character_live.py`, or the console at http://127.0.0.1:8780) outputs:

| What | Device | Use in OBS / apps |
|---|---|---|
| Video (JoyAI + face pass, 480x840, ~26 fps) | v4l2loopback virtual camera `/dev/video10` "AI_Scope_Cam" (loaded at boot: `/etc/modprobe.d`, `exclusive_caps=1`) | OBS: Video Capture Device (V4L2) → AI_Scope_Cam. Calls: pick camera "AI_Scope_Cam" |
| Same video, a window | ffplay "ai-livestream" | watching only (not for OBS: Wayland window capture asks every session) |
| The character's voice, already delayed to match the video | null sink `phone_out` (kept between runs) | OBS: Audio Output Capture (PulseAudio) → "Monitor of phone_out" |
| Same voice as a microphone | `character_mic` "Character_Mic" (remap of phone_out.monitor, kept between runs) | Calls (Telegram, browsers): pick microphone "Character_Mic" |

`--obs-device ''` turns the camera output off. Neither output can stall the pipeline (live outputs keep the newest
frame only).

## OBS setup (once)

1. Install: `sudo apt install -y obs-studio` (Ubuntu 26.04: 32.x).
2. Settings → Video: Base and Output resolution **1080x1920**, FPS **30**.
3. Sources → + → **Video Capture Device (V4L2)** → Device "AI_Scope_Cam" → right-click the source → Transform → Fit to
   screen (480x840 is 4:7; leaves thin bars on 9:16 — or Stretch/Crop to fill).
4. Sources → + → **Audio Output Capture (PulseAudio)** → Device "Monitor of phone_out". Disable Desktop Audio / Mic
   in the mixer (they would add your laptop's sounds and your raw voice).
5. A/V sync: the pipeline aligns voice and picture before the outputs; if OBS shows lips early/late, Advanced Audio
   Properties → Sync Offset on the phone_out source (try +50..+100 ms; the V4L2 path adds about a frame).
6. Keep OBS running across pipeline restarts: the camera and sink stay; the V4L2 source shows the last/blank frame
   while a run restarts and resumes by itself (if not: source Properties → Deactivate/Activate).

## TikTok

TikTok gives a server URL + stream key only to accounts with LIVE access (TikTok's eligibility rules, roughly 18+ and
1,000 followers). With access: TikTok LIVE Center / LIVE Studio "Go LIVE" with a stream key → OBS Settings → Stream →
Service **Custom**, paste Server + Stream Key. Output: x264 or NVENC, 4500-6000 kbps CBR, keyframe interval 2 s,
1080x1920 at 30 fps. The key usually changes per LIVE session.

## Video calls

- **Telegram Desktop** (Linux: `flatpak install flathub org.telegram.desktop` or snap): Settings → Calls → Camera
  "AI_Scope_Cam", Microphone "Character_Mic".
- **Browser calls** (Google Meet, Telegram Web, WhatsApp Web where calls are offered): pick the camera and mic in the
  site's settings; Chrome/Firefox list AI_Scope_Cam because of `exclusive_caps=1`.
- WhatsApp has no Linux desktop app.
- You hear the other side on the laptop: use headphones there, and drive with the phone (its mic → your voice into
  the pipeline), so their voice never reaches the pipeline's input.
