"""Small helper for success-gated episode MP4 recording."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


class EpisodeVideoWriter:
    """Stream an episode to a temp H.264 MP4, then either commit or discard it."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._temp_path: Path | None = None
        self._width = 0
        self._height = 0
        self._fps = 30

    @property
    def is_active(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, temp_path: str | os.PathLike, width: int, height: int, fps: int) -> None:
        self.abort()
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for H.264 video recording.")

        self._temp_path = Path(temp_path)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._temp_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(self._fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-preset",
            "medium",
            "-crf",
            "23",
            str(self._temp_path),
        ]
        self._process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write_frame(self, rgb_uint8_hwc: np.ndarray) -> None:
        if not self.is_active or self._process.stdin is None:
            return
        if rgb_uint8_hwc is None or rgb_uint8_hwc.size == 0:
            return

        frame = np.asarray(rgb_uint8_hwc)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.ndim != 3 or frame.shape[-1] != 3:
            return
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            return

        self._process.stdin.write(frame.tobytes())

    def _finalize_process(self) -> None:
        if self._process is None:
            return
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait()
        self._process = None
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while writing video (exit code {return_code}).")

    def commit(self, final_path: str | os.PathLike) -> Path | None:
        if self._process is None or self._temp_path is None:
            return None

        self._finalize_process()

        final = Path(final_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        if not self._temp_path.exists() or self._temp_path.stat().st_size == 0:
            self.abort()
            return None

        os.replace(self._temp_path, final)
        self._temp_path = None
        print(f"[INFO] Saved successful grasp video: {final}")
        return final

    def abort(self) -> None:
        if self._process is not None:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except OSError:
                    pass
            if self._process.poll() is None:
                self._process.kill()
            self._process.wait()
            self._process = None
        if self._temp_path is not None and self._temp_path.exists():
            self._temp_path.unlink()
        self._temp_path = None
