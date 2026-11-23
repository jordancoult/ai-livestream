import sys
from huggingface_hub import snapshot_download as s
repo = sys.argv[1]; kw = {"local_dir": sys.argv[2]} if len(sys.argv) > 2 else {}
s(repo, max_workers=16, **kw); print("DONE", repo, flush=True)
