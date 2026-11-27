"""Fetch an RVC voice from our private Hugging Face artifacts repo into models/rvc/<name>/ (skips files already there).

  python tools/rvc_live/get_voice.py karen voices/karen/v3        (HF token: HF_TOKEN or the huggingface cache login)
  python tools/rvc_live/get_voice.py --all                         every character's voice + every voice in voices/catalog.json
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = os.environ.get("ARTIFACTS_REPO", "JordanCo/ai-livestream-artifacts")
ROOT = Path(__file__).resolve().parents[2]


def fetch(name, hf_dir):
    dst = ROOT / "models" / "rvc" / name
    dst.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_repo_files(REPO_ID) if f.startswith(hf_dir.rstrip("/") + "/")]
    if not files:
        sys.exit(f"nothing at {REPO_ID}/{hf_dir}")
    stale = [p for p in dst.iterdir() if p.suffix in (".pth", ".index") and p.name not in {Path(f).name for f in files}]
    for p in stale:   # another version's files in the folder: the engine would pick the wrong index
        p.unlink()
    for f in files:
        out = dst / Path(f).name
        if out.exists():
            continue
        shutil.copy(hf_hub_download(REPO_ID, f), out)
    print(f"{name}: {hf_dir} -> {dst} ({', '.join(sorted(p.name for p in dst.iterdir()))})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("hf_dir", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        seen = set()
        for c in sorted((ROOT / "characters").glob("*/character.json")):
            v = json.loads(c.read_text()).get("voice", {})
            if v.get("hf") and v["hf"] not in seen:
                seen.add(v["hf"])
                fetch(Path(v["model"]).parent.name if v.get("model") else c.parent.name, v["hf"])
        cat = ROOT / "voices" / "catalog.json"   # + the live-switchable voices (console Voice dropdown)
        if cat.exists():
            for k, v in json.loads(cat.read_text()).items():
                if not k.startswith("_") and v.get("hf") and v["hf"] not in seen:
                    seen.add(v["hf"])
                    fetch(Path(v["model"]).parent.name, v["hf"])
    else:
        fetch(a.name, a.hf_dir)
