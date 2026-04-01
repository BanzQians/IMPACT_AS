import os
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

from utils.feature_env import load_feature_env_defaults
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
