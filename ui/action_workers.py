import os
import subprocess
import sys
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

from utils.feature_env import build_runner_cmd, load_feature_env_defaults
from utils.optional_deps import (
    MissingOptionalDependency,
    format_missing_dependency_message,
    import_optional_local_module,
)


ACTION_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_feature_env_defaults(repo_root=ACTION_REPO_ROOT)


def load_feature_extractor_module():
    return import_optional_local_module(
        "tools.feature_extractors",
        module_path=os.path.join(ACTION_REPO_ROOT, "tools", "feature_extractors.py"),
        feature_name="Feature extraction",
        install_hint=(
            "Install the optional feature-extraction dependencies first, "
            "for example: pip install torch torchvision opencv-python"
        ),
    )


def _run_streaming_command(cmd, progress_cb):
    log_lines = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                text = str(line or "").rstrip()
                log_lines.append(text + "\n")
                progress_cb(text)
    finally:
        proc.wait()
    return proc.returncode, log_lines


def _write_worker_log(log_path: str, log_lines, progress_cb) -> None:
    if not log_path:
        return
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(log_lines)
        progress_cb(f"[LOG] Saved to {log_path}")
    except Exception as ex:
        progress_cb(f"[WARN] failed to write log: {ex}")


class FeatureExtractWorker(QObject):
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int, int)
    done = pyqtSignal(object, bool)  # (features_dir, ok)

    def __init__(
        self,
        video_path: str,
        features_dir: str,
        batch_size: int = 128,
        frame_stride: int = 1,
        use_fp16: bool = True,
        backbone: Optional[str] = None,
    ):
        super().__init__()
        self.video_path = str(video_path or "")
        self.features_dir = str(features_dir or "")
        self.batch_size = int(batch_size)
        self.frame_stride = int(frame_stride)
        self.use_fp16 = bool(use_fp16)
        self.backbone = str(backbone or "").strip()

    def run(self):
        try:
            feat_path = os.path.join(self.features_dir, "features.npy")
            os.makedirs(self.features_dir, exist_ok=True)
            backbone = self.backbone or os.environ.get("FEATURE_BACKBONE", "i3d")
            self.progress.emit(f"[FEATS] Extracting {backbone} features to {feat_path}")
            feature_api = load_feature_extractor_module()

            def _emit_progress(done: int, total: int):
                self.progress_value.emit(int(done), int(total))

            feats, meta = feature_api.extract_video_features(
                self.video_path,
                backbone=backbone,
                batch_size=max(1, self.batch_size),
                frame_stride=max(1, self.frame_stride),
                use_fp16=self.use_fp16,
                progress_cb=_emit_progress,
            )
            if bool(meta.get("model_cached", False)):
                self.progress.emit(f"[FEATS] Reused cached {backbone} model.")
            else:
                self.progress.emit(f"[FEATS] Loaded {backbone} model.")
            feature_api.save_features(self.features_dir, feats, meta=meta)
            self.progress.emit(f"[FEATS] Saved features {tuple(feats.shape)} to {feat_path}")
            self.done.emit(self.features_dir, True)
        except MissingOptionalDependency as ex:
            self.progress.emit(f"[FEATS][ERROR] {format_missing_dependency_message(ex)}")
            self.done.emit(None, False)
        except Exception as ex:
            self.progress.emit(f"[FEATS][ERROR] {ex}")
            self.done.emit(None, False)


class ASOTInferWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(str, str)  # (txt_path, json_path)

    def __init__(
        self,
        features_dir: str,
        ckpt: str,
        class_names: Optional[str] = None,
        smooth_k: int = 3,
        out_prefix: str = "pred_asot",
        tool_path: str = "asot_full_infer_adapter.py",
        extra_args=None,
        log_path: Optional[str] = None,
    ):
        super().__init__()
        self.features_dir = str(features_dir or "")
        self.ckpt = str(ckpt or "")
        self.class_names = str(class_names or "")
        self.smooth_k = int(smooth_k)
        self.out_prefix = str(out_prefix or "pred_asot")
        self.tool_path = str(tool_path or "asot_full_infer_adapter.py")
        self.extra_args = list(extra_args or [])
        self.log_path = log_path or os.path.join(
            self.features_dir, f"{self.out_prefix}_infer.log"
        )

    def run(self):
        log_lines = []
        try:
            cmd = build_runner_cmd(
                repo_root=ACTION_REPO_ROOT,
                profile="asot",
                script_path=self.tool_path,
                script_args=[
                    "--features_dir",
                    self.features_dir,
                    "--ckpt",
                    self.ckpt,
                    "--out_prefix",
                    self.out_prefix,
                ],
                python_executable=sys.executable,
            )
            if self.class_names:
                cmd += ["--class_names", self.class_names]
            if self.smooth_k > 1:
                cmd += ["--smooth_k", str(self.smooth_k)]
            if self.extra_args:
                cmd += list(self.extra_args)

            returncode, log_lines = _run_streaming_command(cmd, self.progress.emit)
            txt_path = os.path.join(
                self.features_dir, f"{self.out_prefix}_segments.txt"
            )
            json_path = os.path.join(
                self.features_dir, f"{self.out_prefix}_segments.json"
            )
            ok = (returncode == 0) and os.path.isfile(txt_path)
            self.done.emit(txt_path if ok else "", json_path if ok else "")
        except Exception as ex:
            self.progress.emit(f"[ERROR] {ex}")
            self.done.emit("", "")
        finally:
            _write_worker_log(self.log_path, log_lines, self.progress.emit)
