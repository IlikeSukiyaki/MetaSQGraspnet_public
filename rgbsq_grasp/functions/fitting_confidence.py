"""Per-point Gross & Boult fitting error and confidence for deformable superquadrics."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]
ParameterDict = Mapping[str, float]

_EPS = 1e-4


def transform_points_to_primitive_frame(points_nx3: ArrayLike, se3: np.ndarray) -> np.ndarray:
    """Map world-frame points into the predicted primitive frame (same as dsq_loss)."""
    points = np.asarray(points_nx3, dtype=np.float64)
    se3 = np.asarray(se3, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected N x 3 points, got {points.shape}.")
    if se3.shape != (4, 4):
        raise ValueError(f"Expected 4 x 4 SE3, got {se3.shape}.")

    rotation = se3[:3, :3]
    translation = se3[:3, 3]
    centered = points - translation
    return np.dot(centered, rotation)


def _parameters_to_arrays(parameters: ParameterDict) -> Dict[str, np.ndarray]:
    a1 = float(parameters["a1"])
    a2 = float(parameters["a2"])
    a3 = float(parameters["a3"])
    return {
        "a1": np.array([[a1]]),
        "a2": np.array([[a2]]),
        "a3": np.array([[a3]]),
        "e1": np.array([[float(parameters["e1"])]]),
        "e2": np.array([[float(parameters["e2"])]]),
        "k": np.array([[float(parameters["k"])]]),
        "b": np.array([[float(parameters["b"]) / max(a1, a2)]]),
        "cos_alpha": np.array([[float(parameters["cos_alpha"])]]),
        "sin_alpha": np.array([[float(parameters["sin_alpha"])]]),
    }


def compute_dsq_gross_boult_distance(
    points_nx3: ArrayLike,
    parameters: ParameterDict,
) -> np.ndarray:
    """
    Per-point Gross & Boult implicit-surface distance F (NumPy port of dsq_loss.dsq_distance).

    Args:
        points_nx3: N x 3 points already expressed in the primitive frame.
        parameters: Deformable superquadric parameter dict.

    Returns:
        F with shape (N,).
    """
    x = np.asarray(points_nx3, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"Expected N x 3 points, got {x.shape}.")

    params = _parameters_to_arrays(parameters)
    a1 = params["a1"]
    a2 = params["a2"]
    a3 = params["a3"]
    e1 = params["e1"]
    e2 = params["e2"]
    k = params["k"]
    b = params["b"]
    cos_alpha = params["cos_alpha"]
    sin_alpha = params["sin_alpha"]

    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    x2 = x[:, 2:3]

    beta_angle = np.arctan2(x1, x0)
    radial_xy = np.sqrt(x0**2 + x1**2)
    radius_term = (cos_alpha * np.cos(beta_angle) + sin_alpha * np.sin(beta_angle)) * radial_xy
    r = 1.0 / b - np.sqrt(x2**2 + (1.0 / b - radius_term) ** 2)
    gamma = np.arctan2(x2, 1.0 / b - radius_term)
    x_unbent = x0 - cos_alpha * (radius_term - r)
    y_unbent = x1 - sin_alpha * (radius_term - r)
    z_unbent = gamma / b
    taper = k / a3 * z_unbent + 1.0

    beta_sq = np.squeeze(
        (
            (
                np.abs(x_unbent / (np.abs(a1 * taper) + _EPS)) ** (2.0 / e2)
                + np.abs(y_unbent / (np.abs(a2 * taper) + _EPS)) ** (2.0 / e2)
            )
            ** (e2 / e1)
            + np.abs(z_unbent / a3) ** (2.0 / e1)
            + _EPS
        )
        ** (-e1 / 2.0),
        axis=-1,
    )

    return np.linalg.norm(x, axis=1) * (1.0 - beta_sq)


def gross_boult_confidence(fitting_error: ArrayLike, sigma: float) -> np.ndarray:
    """Map per-point Gross & Boult error F to confidence in [0, 1]."""
    f = np.asarray(fitting_error, dtype=np.float64)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    return np.exp(-(f**2) / (sigma**2))


def resolve_sigma(
    fitting_error: ArrayLike,
    sigma: Optional[float],
    normalizer: float,
) -> float:
    """Choose sigma for confidence mapping."""
    if sigma is not None and sigma > 0:
        return float(sigma)
    if normalizer > 0:
        return float(normalizer)

    f = np.asarray(fitting_error, dtype=np.float64)
    positive = f[f > 0]
    if positive.size == 0:
        return 1.0
    return float(0.5 * np.percentile(positive, 95))


def confidence_to_colors(
    confidence: ArrayLike,
    *,
    colormap: str = "viridis",
) -> np.ndarray:
    """Map confidence scores to RGB colors in [0, 1] with shape (N, 3)."""
    scores = np.asarray(confidence, dtype=np.float64)
    scores = np.clip(scores, 0.0, 1.0)
    cmap = plt.get_cmap(colormap)
    return cmap(scores)[:, :3]


def compute_surface_confidence(
    surface_points_nx3: ArrayLike,
    observed_points_nx3: ArrayLike,
    observed_confidence: ArrayLike,
    sigma: float,
    *,
    k_neighbors: int = 8,
) -> np.ndarray:
    """
    Transfer observation confidence onto predicted surface samples.

    Each surface point receives a weighted average of nearby partial-point
    confidences, with Gaussian weights exp(-d² / σ²).
    """
    from scipy.spatial import cKDTree

    surface = np.asarray(surface_points_nx3, dtype=np.float64)
    observed = np.asarray(observed_points_nx3, dtype=np.float64)
    obs_conf = np.asarray(observed_confidence, dtype=np.float64)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if surface.ndim != 2 or surface.shape[1] != 3:
        raise ValueError(f"Expected N x 3 surface points, got {surface.shape}.")
    if observed.shape[0] == 0:
        raise ValueError("Need at least one observed point to transfer confidence.")

    k = min(max(1, k_neighbors), observed.shape[0])
    dists, indices = cKDTree(observed).query(surface, k=k)
    if k == 1:
        dists = dists[:, np.newaxis]
        indices = indices[:, np.newaxis]

    spatial_weight = np.exp(-(dists**2) / (sigma**2))
    neighbor_conf = obs_conf[indices]
    weight_sum = np.sum(spatial_weight, axis=1)
    return np.sum(spatial_weight * neighbor_conf, axis=1) / np.maximum(weight_sum, 1e-12)


def compute_local_confidence(
    points_world_nx3: ArrayLike,
    se3: np.ndarray,
    parameters: ParameterDict,
    *,
    sigma: Optional[float] = None,
    normalizer: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute per-point fitting error and confidence for observed points.

    Returns dict with keys: fitting_error, confidence, sigma, points_primitive.
    """
    points_primitive = transform_points_to_primitive_frame(points_world_nx3, se3)
    fitting_error = compute_dsq_gross_boult_distance(points_primitive, parameters)
    resolved_sigma = resolve_sigma(fitting_error, sigma, normalizer)
    confidence = gross_boult_confidence(fitting_error, resolved_sigma)
    return {
        "fitting_error": fitting_error,
        "confidence": confidence,
        "sigma": resolved_sigma,
        "points_primitive": points_primitive,
    }
