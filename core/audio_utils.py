import os
import shutil
import subprocess
from typing import Tuple


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


def _run(cmd: list) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        **_quiet_windows_kwargs(),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} not found in PATH.")
    return path


def probe_audio_stream(video_path: str) -> Tuple[bool, str]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        code, out, err = _run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nk=1:nw=1",
                video_path,
            ]
        )
        return code == 0 and out.strip() != "", (out or err)
    ffmpeg = _bin("ffmpeg")
    code, out, err = _run([ffmpeg, "-hide_banner", "-i", video_path, "-f", "null", "-"])
    text = out + "\n" + err
    has_audio = ("Audio:" in text) or ("Stream" in text and "Audio" in text)
    return has_audio, text


def extract_wav_16k_mono_verbose(video_path: str, out_wav: str) -> Tuple[bool, str]:
    ffmpeg = _bin("ffmpeg")
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-map",
        "a:0?",
        "-f",
        "wav",
        out_wav,
    ]
    code, out, err = _run(cmd)
    return code == 0, (out or err)
