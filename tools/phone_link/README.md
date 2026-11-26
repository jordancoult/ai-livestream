# Phone camera link (iPhone → laptop)

The phone is the pipeline's camera and mic: a Safari page (no app) encodes the camera in hardware H.264 (WebCodecs)
and sends it, with each frame's capture time, to the laptop. The laptop decodes it once and feeds the bridge
(`live_bridge.py --input h264tcp://…`), so glass-to-glass latency is measured from the camera.

- `tools/phone_link/phone_link.py`: the link **in isolation** (test + measure; no pipeline).
- `tools/phone_view.py --drive-h264-port 9026`: the same link inside the pipeline (character mode `--driver
  phonecam`, the console's "you: phone"): it also plays the character's video and voice back on the phone.
- `tools/phone_cam_js.py`: the shared camera engine (both pages).
- `tools/local_ca.py`: the local certificate authority the phone trusts.

## Network (Medellín setup, 2026-11-26)

```
iPhone 13 Pro (Seattle phone) ──5 GHz──► Beryl ──LAN port──USB-Ethernet──► laptop (enx00e04cd07fdd, 192.168.8.186)
   internet: ONLY through the Beryl's Seattle tunnel          laptop internet: wlo1 / SOMOS (unchanged)
```

- Phone ↔ laptop traffic stays on the Beryl LAN (switched locally): never routed, never the tunnel or the WAN.
- The laptop's Beryl link is **address-only**: NM profile "Wired connection 3" has `ipv4.never-default yes`,
  `ipv4.ignore-auto-dns yes`, IPv6 off (plugging the cable in once made the Beryl the laptop's default route: its
  traffic went through Seattle for a few minutes). Laptop FORWARD is dropped on the Beryl interfaces
  (`iptables -I FORWARD -i/-o enx00e04cd07fdd -j DROP`, not persistent).
- The Beryl itself allows only the iPhone 13 Pro out (MAC allowlist, everything else LAN-only); see
  `~/personal/beryl-restore/docs/CHANGELOG.md` (2026-11-26).
- USB Wi-Fi alternative: `sudo VIF=wlx3460f993bc3d bash ~/personal/beryl-restore/bin/beryl-link.sh up` (TP-Link
  Archer T3U, 5 GHz). Never a virtual interface on wlo1 (it crashed the Wi-Fi firmware).
- The phone pages load nothing from the internet, open by IP (no DNS), use no WebRTC/STUN.

## Transport: HTTPS only

One request loop: the body carries camera / mic / stats / clock replies, the response carries what the laptop has
for the phone (the character's video + voice, clock pings, keyframe requests). WebSockets were dropped: Safari's
WebSocket hung in "connecting" for minutes at times while plain HTTPS kept working (found with
`ios_webkit_debug_proxy` over USB); HTTPS measured as fast (23-25 ms). Plain HTTP responses close their connection
(`Connection: close`) so Safari never reuses one for something else.

## Certificates

The phone trusts a local CA once (laptop page → "First time on a phone"): Settings → Profile Downloaded → Install,
then General → About → Certificate Trust Settings → on. The CA is **name-constrained** to private IPv4 (10/8,
172.16/12, 192.168/16) + localhost: it can't vouch for any website (a leaf for www.google.com fails verification).
Key: `.secrets/local_ca/` (600, gitignored). Remove from the phone: General → VPN & Device Management.

## Measured (wired Beryl, iPhone 13 Pro, 2026-11-26)

| | |
|---|---|
| frame rate | 30.0 fps |
| latency, phone capture → decoded on the laptop | p50 23-26 ms, p90 26-30 ms, p99 ~31 ms (excl. the camera sensor, ~30-50 ms) |
| mic latency | 23 ms |
| bitrate / encode / decode | 2.8 Mbit/s / ~10 ms (phone, hardware) / ~3 ms (laptop) |
| stutters | occasional 80-130 ms gaps (~1 per 5-6 s) with no TCP retransmissions and clean Wi-Fi (ping 3 ms, 0 loss): under investigation (the one-request-at-a-time loop is the suspect) |

Raw camera frames are sent in sensor orientation and rotated on the laptop (a canvas redraw on the phone cost time).

## Run

```bash
cloud/facefix/.venv/bin/python tools/phone_link/phone_link.py --iface enx00e04cd07fdd
# laptop: http://127.0.0.1:8782 (QR codes, live numbers, preview); per-frame log .tmp/phone_link/frames.csv
# raw full-quality view of the phone's H.264:
ffplay -fflags nobuffer -flags low_delay -framedrop -f h264 tcp://127.0.0.1:8790 -vf transpose=1
```

Phone page: camera buttons (front / ultra wide / wide / tele …), zoom, ↻ rotate override; auto-starts after the
first Start; survives tab/app switches, page reloads and server restarts (reconnects by itself); keeps the screen on.
iOS stops the camera while Safari is in the background or the phone is locked: keep it in the foreground.

Debug the phone page from the laptop (USB, Web Inspector on in Safari settings):
`ios_webkit_debug_proxy -F` → pages at http://127.0.0.1:9222/json.
