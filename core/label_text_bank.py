from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_load(path: str) -> Any:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _safe_save(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _normalize_rows(table: np.ndarray) -> np.ndarray:
    arr = np.asarray(table, dtype=np.float32)
    if arr.ndim != 2 or arr.size <= 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return arr / norms


def _stable_hash(parts: Sequence[str]) -> str:
    blob = "||".join(str(x or "") for x in parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(blob).hexdigest()


def _runtime_dir(features_dir: str) -> str:
    return os.path.join(
        os.path.abspath(os.path.expanduser(str(features_dir or "").strip())),
        "text_bank_runtime",
    )


def _bank_path(features_dir: str) -> str:
    return os.path.join(_runtime_dir(features_dir), "label_text_bank.pkl")


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _normalize_labels(classes: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in classes or []:
        name = str(raw or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def _prompt_labels(labels: Sequence[str], prompt_template: str) -> List[str]:
    tpl = str(prompt_template or "assembly action: {}").strip() or "{}"
    out: List[str] = []
    for name in labels or []:
        if "{}" in tpl:
            out.append(tpl.format(str(name)))
        else:
            out.append(f"{tpl} {name}".strip())
    return out


def _lexical_hash_embeddings(texts: Sequence[str], dim: int) -> np.ndarray:
    dim = int(max(8, dim))
    table = np.zeros((len(texts or []), dim), dtype=np.float32)
    for row_idx, raw_text in enumerate(texts or []):
        text = str(raw_text or "").strip().lower()
        if not text:
            continue
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]
        feats = set(tokens)
        chars = text.replace(" ", "_")
        for n in (2, 3, 4):
            if len(chars) < n:
                continue
            for idx in range(len(chars) - n + 1):
                feats.add(chars[idx : idx + n])
        for feat in feats:
            digest = hashlib.sha1(feat.encode("utf-8")).digest()
            col = int.from_bytes(digest[:4], "little", signed=False) % dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            table[row_idx, int(col)] += np.float32(sign)
        if float(np.linalg.norm(table[row_idx])) <= 1e-6:
            table[row_idx, row_idx % dim] = 1.0
    return _normalize_rows(table)


def _project_embeddings(table: np.ndarray, out_dim: int, seed_key: str) -> np.ndarray:
    arr = np.asarray(table, dtype=np.float32)
    out_dim = int(max(8, out_dim))
    if arr.ndim != 2 or arr.shape[0] <= 0:
        return np.zeros((0, out_dim), dtype=np.float32)
    if int(arr.shape[1]) == out_dim:
        return _normalize_rows(arr.astype(np.float32))
    seed = int(_stable_hash([seed_key, str(arr.shape[1]), str(out_dim)])[:8], 16)
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((int(arr.shape[1]), out_dim), dtype=np.float32)
    proj = proj / np.sqrt(max(1, int(arr.shape[1])))
    out = np.asarray(arr @ proj, dtype=np.float32)
    return _normalize_rows(out)


def _allow_remote_model_fetch() -> bool:
    raw = str(os.environ.get("SIGLIP2_TEXT_BANK_ALLOW_REMOTE", "1") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _siglip2_runner_embeddings(
    features_dir: str,
    texts: Sequence[str],
    classes: Sequence[str],
    feature_dim: int,
    *,
    model_name: str,
) -> Optional[Dict[str, Any]]:
    repo_root = _repo_root()
    runner_path = os.path.join(repo_root, "tools", "runners", "run_in_env.py")
    script_path = os.path.join(repo_root, "tools", "siglip2_text_bank.py")
    if not (os.path.isfile(runner_path) and os.path.isfile(script_path)):
        return None

    runtime_dir = _runtime_dir(features_dir)
    try:
        os.makedirs(runtime_dir, exist_ok=True)
    except Exception:
        pass
    req_path = os.path.join(runtime_dir, "_text_bank_request.json")
    out_path = os.path.join(runtime_dir, "_text_bank_response.pkl")
    try:
        with open(req_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "texts": list(texts or []),
                    "classes": list(classes or []),
                    "feature_dim": int(feature_dim),
                    "model_name": str(model_name or "").strip(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        cmd = [
            sys.executable,
            runner_path,
            "--profile",
            "siglip2",
            "--",
            script_path,
            "--input-json",
            req_path,
            "--output",
            out_path,
            "--model-name",
            str(model_name or "").strip(),
        ]
        if not _allow_remote_model_fetch():
            cmd.append("--local-only")
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0 or not os.path.isfile(out_path):
            return None
        obj = _safe_load(out_path)
        if not isinstance(obj, dict):
            return None
        table = (
            np.asarray(obj.get("text_table"), dtype=np.float32)
            if obj.get("text_table") is not None
            else np.zeros((0, 0), dtype=np.float32)
        )
        if table.ndim != 2 or table.shape != (len(classes or []), int(feature_dim)):
            return None
        return {
            "backend": str(obj.get("backend") or "siglip2_runner"),
            "model_name": str(obj.get("model_name") or str(model_name or "").strip()),
            "raw_dim": int(obj.get("raw_dim", table.shape[1]) or table.shape[1]),
            "text_table": _normalize_rows(table),
        }
    except Exception:
        return None
    finally:
        for path in (req_path, out_path):
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass


def ensure_label_text_bank(
    features_dir: str,
    classes: Sequence[str],
    feature_dim: int,
    *,
    backend: str = "siglip2",
    model_name: str = "",
    prompt_template: str = "assembly action: {}",
) -> Dict[str, Any]:
    features_dir = os.path.abspath(os.path.expanduser(str(features_dir or "").strip()))
    runtime_dir = _runtime_dir(features_dir)
    os.makedirs(runtime_dir, exist_ok=True)
    labels = _normalize_labels(classes)
    feature_dim = int(max(1, feature_dim))
    backend = str(backend or "siglip2").strip().lower() or "siglip2"
    model_name = str(
        model_name
        or os.environ.get("TEXT_BANK_MODEL")
        or os.environ.get("SIGLIP2_TEXT_BANK_MODEL")
        or "external/huggingface/google--siglip2-base-patch16-224"
    ).strip()
    prompt_template = str(
        prompt_template
        or os.environ.get("TEXT_BANK_PROMPT_TEMPLATE")
        or "assembly action: {}"
    )
    bank_path = _bank_path(features_dir)
    label_hash = _stable_hash(
        [backend, model_name, prompt_template, str(feature_dim)] + list(labels)
    )

    existing = _safe_load(bank_path) if os.path.isfile(bank_path) else {}
    if isinstance(existing, dict):
        same = (
            str(existing.get("label_hash", "") or "") == label_hash
            and int(existing.get("feature_dim", 0) or 0) == feature_dim
        )
        if same and backend not in {"auto"}:
            saved_backend = str(existing.get("backend", "") or "").lower()
            if backend == "siglip2" and "siglip2" not in saved_backend:
                same = False
        table = (
            np.asarray(existing.get("text_table"), dtype=np.float32)
            if existing.get("text_table") is not None
            else np.zeros((0, 0), dtype=np.float32)
        )
        if same and table.ndim == 2 and table.shape == (len(labels), feature_dim):
            return {
                "ok": True,
                "changed": False,
                "path": bank_path,
                "backend": str(existing.get("backend", "") or "unknown"),
                "classes": list(labels),
                "feature_dim": int(feature_dim),
                "text_table": _normalize_rows(table),
            }

    texts = _prompt_labels(labels, prompt_template)
    text_table = None
    raw_dim = 0
    backend_used = backend
    resolved_model_name = model_name

    if backend in {"auto", "siglip2"}:
        result = _siglip2_runner_embeddings(
            features_dir,
            texts,
            labels,
            int(feature_dim),
            model_name=model_name,
        )
        if result is not None:
            backend_used = str(result.get("backend") or "siglip2_runner")
            resolved_model_name = str(result.get("model_name") or model_name)
            text_table = np.asarray(result.get("text_table"), dtype=np.float32)
            raw_dim = int(
                result.get("raw_dim", text_table.shape[1] if text_table.ndim == 2 else 0)
                or 0
            )

    if text_table is None:
        raw_dim = min(max(64, feature_dim), 512)
        raw_table = _lexical_hash_embeddings(texts, dim=raw_dim)
        backend_used = "hashed_lexical"
        resolved_model_name = ""
        text_table = _project_embeddings(
            raw_table,
            out_dim=feature_dim,
            seed_key=f"{backend_used}:{prompt_template}",
        )

    payload = {
        "version": 1,
        "kind": "label_text_bank",
        "created_at": _utc_now(),
        "backend": backend_used,
        "model_name": resolved_model_name,
        "prompt_template": prompt_template,
        "classes": list(labels),
        "label_hash": label_hash,
        "raw_dim": int(raw_dim),
        "feature_dim": int(feature_dim),
        "text_table": text_table.astype(np.float32),
    }
    _safe_save(payload, bank_path)
    return {
        "ok": True,
        "changed": True,
        "path": bank_path,
        "backend": backend_used,
        "classes": list(labels),
        "feature_dim": int(feature_dim),
        "text_table": text_table.astype(np.float32),
    }


def load_label_text_bank_map(
    features_dir: str,
    classes: Sequence[str],
    feature_dim: int,
) -> Dict[str, np.ndarray]:
    bank_path = _bank_path(features_dir)
    if not os.path.isfile(bank_path):
        return {}
    obj = _safe_load(bank_path)
    if not isinstance(obj, dict):
        return {}
    names = _normalize_labels(obj.get("classes") or [])
    table = (
        np.asarray(obj.get("text_table"), dtype=np.float32)
        if obj.get("text_table") is not None
        else np.zeros((0, 0), dtype=np.float32)
    )
    if table.ndim != 2 or table.shape != (len(names), int(feature_dim)):
        return {}
    class_set = set(_normalize_labels(classes))
    out: Dict[str, np.ndarray] = {}
    table = _normalize_rows(table)
    for idx, name in enumerate(names):
        if name in class_set:
            out[str(name)] = table[idx]
    return out
