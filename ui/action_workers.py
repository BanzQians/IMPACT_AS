import os
import subprocess
import sys

from PyQt5.QtCore import QObject, pyqtSignal

from utils.feature_env import build_runner_cmd, load_feature_env_defaults


ACTION_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_feature_env_defaults(repo_root=ACTION_REPO_ROOT)


def _quiet_windows_kwargs() -> dict:
    if os.name != "nt":
        return {}
    kwargs = {}
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0)
    if creationflags:
        kwargs["creationflags"] = creationflags
    startup_cls = getattr(subprocess, "STARTUPINFO", None)
    if startup_cls is not None:
        startupinfo = startup_cls()
        startupinfo.dwFlags |= int(getattr(subprocess, "STARTF_USESHOWWINDOW", 0) or 0)
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    return kwargs


def _run_streaming_command(cmd, progress_cb):
    log_lines = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **_quiet_windows_kwargs(),
    )
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip()
                log_lines.append(line + "\n")
                progress_cb(line)
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


class FactBatchWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(
        self,
        video_dir,
        output_dir,
        fact_repo,
        ckpt,
        fact_cfg,
        tool_path,
        class_names=None,
        log_path=None,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.fact_repo = fact_repo
        self.ckpt = ckpt
        self.fact_cfg = fact_cfg
        self.tool_path = tool_path
        self.class_names = class_names
        self.log_path = log_path or os.path.join(output_dir, "pred_fact_batch.log")

    def run(self):
        log_lines = []
        try:
            cmd = build_runner_cmd(
                repo_root=ACTION_REPO_ROOT,
                profile="current",
                script_path=self.tool_path,
                script_args=[
                    "--video_dir",
                    self.video_dir,
                    "--output_dir",
                    self.output_dir,
                    "--fact_repo",
                    self.fact_repo,
                    "--fact_cfg",
                    self.fact_cfg,
                    "--ckpt",
                    self.ckpt,
                ],
                python_executable=sys.executable,
            )
            if self.class_names:
                cmd += ["--class_names", self.class_names]

            returncode, log_lines = _run_streaming_command(cmd, self.progress.emit)
            self.done.emit(returncode == 0, self.output_dir)
        except Exception as ex:
            self.progress.emit(f"[ERROR] {ex}")
            self.done.emit(False, self.output_dir)
        finally:
            _write_worker_log(self.log_path, log_lines, self.progress.emit)
