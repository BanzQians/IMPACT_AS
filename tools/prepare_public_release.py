#!/usr/bin/env python3
"""Build a sanitized copy of the repository for public release.

This script copies the current workspace into a new output directory while
excluding files and folders that may disclose unpublished ideas or internal
experiments.

Usage:
  python tools/prepare_public_release.py --out-dir ..\\IMPACT_AS_public
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List


DEFAULT_EXCLUDES = [
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "backups",
]


def _normalize_rel(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return rel


def _match_exclude(rel: str, excludes: Iterable[str]) -> bool:
    for item in excludes:
        pat = str(item).replace("\\", "/").strip("/")
        if not pat:
            continue
        if rel == pat or rel.startswith(pat + "/"):
            return True
    return False


def _copy_tree(src_root: Path, dst_root: Path, excludes: List[str]) -> None:
    for path in src_root.rglob("*"):
        rel = _normalize_rel(path, src_root)
        if _match_exclude(rel, excludes):
            continue

        dst = dst_root / rel
        if path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(path), str(dst))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a sanitized public release copy.")
    parser.add_argument("--out-dir", required=True, help="Output directory for sanitized copy")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional relative path prefixes to exclude (can be repeated)",
    )
    args = parser.parse_args()

    src_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).resolve()

    if out_dir.exists():
        raise SystemExit(f"[ERROR] Output directory already exists: {out_dir}")

    excludes = list(DEFAULT_EXCLUDES)
    excludes.extend(args.exclude or [])

    out_dir.mkdir(parents=True, exist_ok=False)
    _copy_tree(src_root, out_dir, excludes)

    print("[OK] Sanitized public copy created:")
    print(f"  {out_dir}")
    print("[INFO] Excluded patterns:")
    for item in excludes:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
