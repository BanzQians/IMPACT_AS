import argparse
import os
import sys


def _bootstrap_fontconfig_runtime() -> None:
    if os.name == "nt":
        return

    config_candidates = [
        os.path.join(sys.prefix, "etc", "fonts", "fonts.conf"),
        "/etc/fonts/fonts.conf",
    ]
    config_path = str(os.environ.get("FONTCONFIG_FILE", "") or "").strip()
    if not config_path:
        for candidate in config_candidates:
            if os.path.isfile(candidate):
                config_path = candidate
                os.environ["FONTCONFIG_FILE"] = candidate
                break

    config_dir = str(os.environ.get("FONTCONFIG_PATH", "") or "").strip()
    if not config_dir:
        dir_candidates = []
        if config_path:
            dir_candidates.append(os.path.dirname(config_path))
        dir_candidates.extend(
            [
                os.path.join(sys.prefix, "etc", "fonts"),
                "/etc/fonts",
            ]
        )
        for candidate in dir_candidates:
            if os.path.isdir(candidate):
                os.environ["FONTCONFIG_PATH"] = candidate
                break


def _bootstrap_qt_runtime() -> None:
    """
    Force Qt to use the PyQt5-bundled runtime/plugins.
    This avoids PATH/plugin conflicts in mixed environments (e.g. Anaconda + other Qt installs).
    """
    try:
        import PyQt5  # delay QtWidgets import until plugin paths are normalized
    except Exception:
        return
    pyqt_root = os.path.dirname(PyQt5.__file__)
    qt_root = os.path.join(pyqt_root, "Qt5")

    def _candidate_roots() -> list:
        roots = []

        def _append(plugin_root: str, qt_bin: str = "") -> None:
            platform_root = os.path.join(plugin_root, "platforms")
            if os.path.isdir(platform_root):
                roots.append((plugin_root, platform_root, qt_bin))

        conda_root = sys.prefix
        if os.name == "nt":
            _append(
                os.path.join(conda_root, "Library", "plugins"),
                os.path.join(conda_root, "Library", "bin"),
            )
        else:
            _append(os.path.join(conda_root, "plugins"), os.path.join(conda_root, "bin"))
            _append(
                os.path.join(conda_root, "lib", "qt5", "plugins"),
                os.path.join(conda_root, "bin"),
            )
            _append(
                os.path.join(conda_root, "lib", "qt", "plugins"),
                os.path.join(conda_root, "bin"),
            )
        _append(os.path.join(qt_root, "plugins"), os.path.join(qt_root, "bin"))
        return roots

    plugin_root = ""
    platform_root = ""
    qt_bin = ""
    candidates = _candidate_roots()
    if candidates:
        plugin_root, platform_root, qt_bin = candidates[0]

    # Clear externally injected plugin paths that often point to incompatible Qt builds.
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_FONTDIR", None)

    if os.path.isdir(plugin_root):
        os.environ["QT_PLUGIN_PATH"] = plugin_root
    if os.path.isdir(platform_root):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platform_root
    if os.path.isdir(qt_bin):
        cur_path = os.environ.get("PATH", "")
        parts = [p.strip() for p in cur_path.split(os.pathsep) if p.strip()]
        norm_parts = {os.path.normcase(os.path.normpath(p)) for p in parts}
        norm_qt_bin = os.path.normcase(os.path.normpath(qt_bin))
        if norm_qt_bin not in norm_parts:
            os.environ["PATH"] = qt_bin + os.pathsep + cur_path
        # Python 3.8+ on Windows: ensure dependent DLL lookup includes Qt bin.
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(qt_bin)
            except Exception:
                pass


_bootstrap_fontconfig_runtime()
_bootstrap_qt_runtime()

from utils.feature_env import load_feature_env_defaults  # noqa: E402
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
load_feature_env_defaults(repo_root=_REPO_ROOT)
# Some optional torch backends require this env var before torch-dependent imports.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from PyQt5.QtGui import QIcon  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from utils.op_logger import OperationLogger  # noqa: E402

# cv2 may overwrite Qt plugin env vars during import; restore them before QApplication starts.
_bootstrap_qt_runtime()


def _set_windows_app_id(app_id: str) -> None:
    if os.name != "nt" or not app_id:
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def _resolve_app_icon() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, "icon.ico"),
        os.path.join(root, "icon.png"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oplog",
        action="store_true",
        help="Enable operation logging (writes alongside annotation file).",
    )
    args, _ = parser.parse_known_args()

    _set_windows_app_id("cvhci.video.annotation.impact_as")
    app = QApplication(sys.argv)
    icon_path = _resolve_app_icon()
    if icon_path:
        icon = QIcon(icon_path)
        app.setWindowIcon(icon)
    logger = OperationLogger(enabled=bool(args.oplog))
    w = MainWindow(logger=logger)
    if icon_path:
        w.setWindowIcon(QIcon(icon_path))
    w.show()
    sys.exit(app.exec_())


# Subtitle conversion feature is not implemented
# Bbox Genera1tor feature is not implemented
# Segmantaion Assistant feature is not implemented
# Interactive Segmentation feature is not implemented
