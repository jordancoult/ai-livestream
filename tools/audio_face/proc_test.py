import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cloud", "lpfast"))
from audio_driver import LiveAudioMotionProc
if __name__ == "__main__":
    lm = LiveAudioMotionProc(sys.argv[1], source="file:" + sys.argv[2]).start()
    t_start = time.time()
    while lm.t0 is None and time.time() - t_start < 60:
        time.sleep(0.1)
    print("t0 after", round(time.time() - t_start, 1), "s", flush=True)
    time.sleep(6)
    t = time.time() - 0.6
    m = lm.motion_at(t, timeout=1.0)
    print("frames", lm.stats, "motion ok", m is not None, flush=True)
    lm.close(); os._exit(0)
