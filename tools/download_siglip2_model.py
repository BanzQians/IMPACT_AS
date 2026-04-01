#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser(description="Download a SigLIP2 model snapshot from Hugging Face.")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--local-dir", required=True)
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise SystemExit(f"[SIGLIP2][ERROR] huggingface_hub is required: {exc}")

    model_id = str(args.model_id or "").strip()
    local_dir = os.path.abspath(os.path.expanduser(str(args.local_dir or "").strip()))
    if not model_id or not local_dir:
        raise SystemExit("[SIGLIP2][ERROR] --model-id and --local-dir are required.")

    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[SIGLIP2] Downloaded {model_id} to {local_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
