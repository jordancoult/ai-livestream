"""Phone H.264 decoder, in its own process (live_bridge.py --input h264tcp://host:port starts it).

Decoding 30 fps of 720p inside the bridge competed with its send loop for the GIL (the bridge sent only 14-18 fps);
ffmpeg reading a raw H.264 pipe buffers for seconds. This does what the in-process decoder did (PyAV, a picture per
packet, no added delay), in a separate interpreter.

  stdin : nothing    stdout: per frame [capture ms f64 (laptop clock)][H*W*3 BGR bytes]
  python h264_decode_proc.py HOST PORT WIDTH HEIGHT
Reads tools/phone_view.py's feed: <len u32, capture ms f64, flags u8 (bit0 key, bits1-2 quarter turns cw)> + Annex-B.
Rotates upright, scales to cover WIDTHxHEIGHT and centre-crops. Reconnects.
"""
import socket
import struct
import sys
import time

import av
import cv2
import numpy as np

HDR = struct.Struct("<IdB")
ROTS = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)


def recv(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise ConnectionError
        b += c
    return bytes(b)


def main():
    host, port, iw, ih = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    out = sys.stdout.buffer
    while True:
        try:
            s = socket.create_connection((host, port), timeout=5)
            s.settimeout(None)
            dec = av.CodecContext.create("h264", "r")
            while True:
                n, cap, flags = HDR.unpack(recv(s, HDR.size))
                data = recv(s, n)
                try:
                    frames = dec.decode(av.Packet(data))
                except av.error.InvalidDataError:
                    continue
                for fr in frames:
                    img = fr.to_ndarray(format="bgr24")
                    rot = (flags >> 1) & 3
                    if rot:
                        img = cv2.rotate(img, ROTS[rot])
                    h, w = img.shape[:2]
                    sc = max(iw / w, ih / h)
                    img = cv2.resize(img, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
                    y0, x0 = (img.shape[0] - ih) // 2, (img.shape[1] - iw) // 2
                    out.write(struct.pack("<d", cap) + np.ascontiguousarray(img[y0:y0 + ih, x0:x0 + iw]).tobytes())
                    out.flush()
        except (OSError, ConnectionError, struct.error):
            time.sleep(1)


if __name__ == "__main__":
    main()
