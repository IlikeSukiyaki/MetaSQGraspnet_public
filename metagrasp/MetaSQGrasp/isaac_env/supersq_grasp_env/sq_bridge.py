"""Runtime SQ estimation bridge for the supersq grasp task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np
import torch

EstimatorKind = Literal["sq_esti_model", "extsq_esti_model"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _training_root() -> Path:
    return _project_root() / "sq_estimation_model" / "training"


def _ensure_training_path() -> Path:
    root = _training_root()
    if not root.exists():
        raise FileNotFoundError(f"Missing SQ estimation training package: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _pretrained_root() -> Path:
    return _project_root() / "sq_estimation_model" / "pretrained"


def default_sq_config(model_kind: EstimatorKind = "sq_esti_model") -> Path:
    if model_kind == "extsq_esti_model":
        return _pretrained_root() / "extsq_esti_model" / "extsq_esti_model_config.yml"
    return _pretrained_root() / "sq_esti_model" / "sq_esti_model_config.yml"


def default_sq_checkpoint(model_kind: EstimatorKind = "sq_esti_model") -> Path:
    if model_kind == "extsq_esti_model":
        return _pretrained_root() / "extsq_esti_model" / "model_best.pkl"
    return _pretrained_root() / "sq_esti_model" / "model_best.pkl"


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
    model_kind: EstimatorKind = "sq_esti_model"
    device: str = "cuda:0"
    num_fitting_points: int = 300

    def __post_init__(self) -> None:
        _ensure_training_path()
        if self.model_kind == "extsq_esti_model":
            from extsq_est_inference import load_extsq_model as load_model
            from extsq_est_inference import recognize_extsq as recognize
        elif self.model_kind == "sq_esti_model":
            from sq_est_inference import load_sq_model as load_model
            from sq_est_inference import recognize_sq as recognize
        else:
            raise ValueError(f"Unsupported SQ estimation model: {self.model_kind}")

        self.config_path = Path(self.config_path).expanduser().resolve()
        self.checkpoint_path = Path(self.checkpoint_path).expanduser().resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing SQ estimation config: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing SQ estimation checkpoint: {self.checkpoint_path}")

        self.device = resolve_device(self.device)
        self.model, self.cfg = load_model(
            self.config_path,
            self.checkpoint_path,
            device=self.device,
        )
        self._recognize = recognize
        print(
            "[SuperSQ] Loaded SQ estimation "
            f"model={self.model_kind} checkpoint={self.checkpoint_path} device={self.device}"
        )

    def fit_superquadric(self, partial_pc: np.ndarray) -> dict[str, Any]:
        points = np.asarray(partial_pc, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected partial_pc shape N x 3, got {points.shape}.")

        result = self._recognize(
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
    model_kind: EstimatorKind = "sq_esti_model",
    device: str | int | None = None,
    num_fitting_points: int = 300,
) -> SuperquadricEstimator:
    return SuperquadricEstimator(
        config_path=Path(config_path) if config_path else default_sq_config(model_kind),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else default_sq_checkpoint(model_kind),
        model_kind=model_kind,
        device=resolve_device(device),
        num_fitting_points=num_fitting_points,
    )
