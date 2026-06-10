"""Runtime Superquadrics Estimation bridge for the supersq grasp task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _rgbsq_root() -> Path:
    return _repo_root() / "rgbsq_grasp"


def _ensure_rgbsq_path() -> Path:
    root = _rgbsq_root()
    if not root.exists():
        raise FileNotFoundError(f"Missing rgbsq_grasp checkout: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def default_sq_config() -> Path:
    return _rgbsq_root() / "pretrained" / "sqnet_config" / "example" / "sqnet_config.yml"


def default_sq_checkpoint() -> Path:
    return _rgbsq_root() / "pretrained" / "sqnet_config" / "example" / "model_best.pkl"


def resolve_device(device: str | int | None = None) -> str:
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device_str = str(device)
    if device_str == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if device_str.startswith("cuda"):
        return device_str
    return f"cuda:{device_str}"


@dataclass
class SuperquadricEstimator:
    config_path: Path
    checkpoint_path: Path
    device: str = "cuda:0"
    num_fitting_points: int = 300

    def __post_init__(self) -> None:
        _ensure_rgbsq_path()
        from inference import load_recognition_model

        self.config_path = Path(self.config_path).expanduser().resolve()
        self.checkpoint_path = Path(self.checkpoint_path).expanduser().resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing SQNet config: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing SQNet checkpoint: {self.checkpoint_path}")

        self.device = resolve_device(self.device)
        self.model, self.cfg = load_recognition_model(
            self.config_path,
            self.checkpoint_path,
            device=self.device,
        )
        print(
            "[SuperSQ] Loaded SQNet "
            f"checkpoint={self.checkpoint_path} device={self.device}"
        )

    def fit_superquadric(self, partial_pc: np.ndarray) -> dict[str, Any]:
        from supersq_inference import recognize_superquadric

        points = np.asarray(partial_pc, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected partial_pc shape N x 3, got {points.shape}.")

        result = recognize_superquadric(
            points,
            self.model,
            num_fitting_points=self.num_fitting_points,
        )
        return {
            "se3_m": np.asarray(result["SE3"], dtype=np.float64),
            "parameters": result["parameters"],
            "debug": {
                "normalizer": float(result["normalizer"]),
                "segment_frame": np.asarray(result["segment_frame"], dtype=np.float64),
                "pred_fit": np.asarray(result["pred_fit"], dtype=np.float64),
                "num_input_points": int(points.shape[0]),
                "num_processed_points": int(np.asarray(result["processed_points"]).shape[0]),
            },
        }


def make_estimator(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | int | None = None,
    num_fitting_points: int = 300,
) -> SuperquadricEstimator:
    return SuperquadricEstimator(
        config_path=Path(config_path) if config_path else default_sq_config(),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else default_sq_checkpoint(),
        device=resolve_device(device),
        num_fitting_points=num_fitting_points,
    )
