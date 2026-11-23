import sys
from huggingface_hub import snapshot_download as s
for repo, kw, tag in (("kyutai/moshiko-pytorch-bf16", {}, "W_MO_OK"),
                      ("nvidia/NVIDIA-NemotronLabs-VoiceChat-11B", {"local_dir": "/workspace/models/voicechat"}, "W_VC_OK"),
                      ("nvidia/personaplex-7b-v1", {}, "W_PP_OK")):
    try:
        s(repo, **kw); print(tag, flush=True)
    except Exception as e:
        print("FAILED", repo, type(e).__name__, str(e)[:200], flush=True)
