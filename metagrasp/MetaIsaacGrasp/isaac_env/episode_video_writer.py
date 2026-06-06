"""Small helper for success-gated episode MP4 recording."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


class EpisodeVideoWriter:
    """Stream an episode to a temp MP4, then either commit or discard it."""

    def __init__(self) -> None:
        self._writer: cv2.VideoWriter | None = None
        self._temp_path: Path | None = None

    @property
    def is_active(self) -> bool:
        return self._writer is not None

    def start(self, temp_path: str | os.PathLike, width: int, height: int, fps: int) -> None:
        self.abort()
        self._temp_path = Path(temp_path)
        self._temp_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._temp_path), fourcc, float(fps), (width, height))
        if not self._writer.isOpened():
            self._writer.release()
            self._writer = None
            raise RuntimeError(f"Failed to open video writer: {self._temp_path}")

    def write_frame(self, rgb_uint8_hwc: np.ndarray) -> None:
        if self._writer is None:
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

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def commit(self, final_path: str | os.PathLike) -> Path | None:
        if self._writer is None or self._temp_path is None:
            return None

        self._writer.release()
        self._writer = None

        final = Path(final_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self._temp_path, final)
        self._temp_path = None
        print(f"[INFO] Saved successful grasp video: {final}")
        return final

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._temp_path is not None and self._temp_path.exists():
            self._temp_path.unlink()
        self._temp_path = None
