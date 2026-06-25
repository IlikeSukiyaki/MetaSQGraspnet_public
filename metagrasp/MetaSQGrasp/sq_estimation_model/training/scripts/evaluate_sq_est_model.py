#!/usr/bin/env python3
"""Offline SQ estimation evaluation on a manifest dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from data_loaders.manifest import filter_entries, get_split_entries, load_manifest
from geometry.utils_numpy import define_SE3, quaternions_to_rotation_matrices
from inference import load_recognition_model


PARAMETER_KEYS = ("a1", "a2", "a3", "e1", "e2")
_RUNTIME_IMPORT_WARNING_SHOWN = False


def parse_class_ids(raw_class_ids: list[str] | None) -> list[str] | None:
    if not raw_class_ids:
        return None
    class_ids = []
    for raw in raw_class_ids:
        class_ids.extend(part.strip() for part in raw.split(",") if part.strip())
    return class_ids or None


def metadata_summary(data: dict) -> dict:
    summary = {}
    for key, value in data.items():
        if key in {"partial_pc", "full_pc", "primitives"}:
            continue
        if isinstance(value, np.ndarray):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            summary[key] = value
    return summary


def load_sample(path: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    data = np.load(path, allow_pickle=True).item()
    if "partial_pc" not in data or "full_pc" not in data:
        raise KeyError(f"Sample {path} must contain partial_pc and full_pc.")
    if "primitives" not in data or len(data["primitives"]) != 1:
        raise ValueError(f"Sample {path} must contain exactly one primitive.")

    partial_pc = np.asarray(data["partial_pc"], dtype=np.float64)
    full_pc = np.asarray(data["full_pc"], dtype=np.float64)
    return partial_pc[:, :3], full_pc[:, :3], data["primitives"][0], metadata_summary(data)


def sq_parameter_errors(predicted: dict[str, float], gt: dict) -> dict[str, float]:
    gt_params = gt.get("parameters", {})
    if not all(key in gt_params for key in PARAMETER_KEYS):
        return {}
    return {
        f"abs_{key}": abs(float(predicted[key]) - float(gt_params[key]))
        for key in PARAMETER_KEYS
    }


def normalize_points_like_dataset(points: np.ndarray) -> tuple[np.ndarray, float]:
    max_xyz = np.max(points, axis=0)
    min_xyz = np.min(points, axis=0)
    diagonal_len = float(np.linalg.norm(max_xyz - min_xyz))
    if diagonal_len <= 0:
        raise ValueError("Cannot normalize a degenerate point cloud.")
    return points / diagonal_len, diagonal_len


def prediction_to_result(pred_fit: np.ndarray, normalizer: float, input_points: np.ndarray) -> dict:
    pred_fit = np.asarray(pred_fit, dtype=np.float64).reshape(-1)
    se3 = define_SE3(
        np.squeeze(quaternions_to_rotation_matrices(np.expand_dims(pred_fit[3:7], axis=0))),
        normalizer * pred_fit[:3],
    )
    parameters = {
        "a1": float(pred_fit[7] * normalizer),
        "a2": float(pred_fit[8] * normalizer),
        "a3": float(pred_fit[9] * normalizer),
        "e1": float(pred_fit[10]),
        "e2": float(pred_fit[11]),
    }
    return {
        "SE3": se3,
        "parameters": parameters,
        "pred_fit": pred_fit,
        "normalizer": float(normalizer),
        "segment_frame": np.eye(4),
        "input_points": input_points,
        "processed_points": input_points / normalizer,
    }


@torch.no_grad()
def recognize_sq_dataset_frame(point_cloud: np.ndarray, model: torch.nn.Module) -> dict:
    points = np.asarray(point_cloud, dtype=np.float64)
    normalized_points, normalizer = normalize_points_like_dataset(points)
    device = next(model.parameters()).device
    inputs = torch.tensor(
        np.expand_dims(normalized_points.transpose(), axis=0),
        dtype=torch.float32,
        device=device,
    )
    pred_fit = np.squeeze(model(inputs).detach().cpu().numpy())
    return prediction_to_result(pred_fit, normalizer, points)


def recognize_sample(point_cloud: np.ndarray, model: torch.nn.Module, preprocess: str, num_fitting_points: int) -> dict:
    global _RUNTIME_IMPORT_WARNING_SHOWN
    if preprocess == "runtime":
        try:
            from sq_est_inference import recognize_sq
        except ImportError as exc:
            if not _RUNTIME_IMPORT_WARNING_SHOWN:
                print(
                    "[WARN] Runtime SQ preprocessing unavailable; falling back to dataset-frame "
                    f"preprocessing ({exc})."
                )
                _RUNTIME_IMPORT_WARNING_SHOWN = True
            return recognize_sq_dataset_frame(point_cloud, model)
        return recognize_sq(point_cloud, model, num_fitting_points=num_fitting_points)
    return recognize_sq_dataset_frame(point_cloud, model)


def transform_points_to_sq_frame(points: np.ndarray, se3: np.ndarray) -> np.ndarray:
    rotation = se3[:3, :3]
    translation = se3[:3, 3]
    return (rotation.T @ (points - translation).T).T


def sq_distance(points: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    local = np.asarray(points, dtype=np.float64)
    eps = np.finfo(np.float64).eps
    a1 = max(float(parameters["a1"]), eps)
    a2 = max(float(parameters["a2"]), eps)
    a3 = max(float(parameters["a3"]), eps)
    e1 = max(float(parameters["e1"]), eps)
    e2 = max(float(parameters["e2"]), eps)

    xy = (np.abs(local[:, 0] / a1) ** (2.0 / e2) + np.abs(local[:, 1] / a2) ** (2.0 / e2)) ** (
        e2 / e1
    )
    z = np.abs(local[:, 2] / a3) ** (2.0 / e1)
    beta = (xy + z) ** (-e1 / 2.0)
    return np.linalg.norm(local, axis=1) * (1.0 - beta)


def fit_errors(points: np.ndarray, result: dict) -> dict[str, float]:
    local_points = transform_points_to_sq_frame(points, np.asarray(result["SE3"], dtype=np.float64))
    distances = sq_distance(local_points, result["parameters"])
    return {
        "mean_abs_sq_distance": float(np.mean(np.abs(distances))),
        "mean_sq_distance": float(np.mean(distances**2)),
    }


def pose_errors(result: dict, gt: dict, source_type: str | None) -> dict[str, float]:
    gt_se3 = np.asarray(gt.get("SE3"), dtype=np.float64)
    if gt_se3.shape != (4, 4):
        return {}

    pred_se3 = np.asarray(result["SE3"], dtype=np.float64)
    pred_axis = pred_se3[:3, 2]
    gt_axis_index = 1 if source_type == "truncated_torus" else 2
    gt_axis = gt_se3[:3, gt_axis_index]

    pred_axis_norm = np.linalg.norm(pred_axis)
    gt_axis_norm = np.linalg.norm(gt_axis)
    axis_error_deg = None
    if pred_axis_norm > 0 and gt_axis_norm > 0:
        cosine = np.clip(abs(np.dot(pred_axis, gt_axis) / (pred_axis_norm * gt_axis_norm)), -1.0, 1.0)
        axis_error_deg = float(np.degrees(np.arccos(cosine)))

    errors = {
        "translation_error": float(np.linalg.norm(pred_se3[:3, 3] - gt_se3[:3, 3])),
    }
    if axis_error_deg is not None:
        errors["axis_error_deg"] = axis_error_deg
    return errors


def summarize(rows: list[dict]) -> dict:
    error_groups = {
        "mean_parameter_abs_errors": "parameter_abs_errors",
        "mean_fit_errors": "fit_errors",
        "mean_pose_errors": "pose_errors",
    }
    summary = {"num_samples": len(rows)}
    for summary_key, row_key in error_groups.items():
        keys = sorted({key for row in rows for key in row[row_key]})
        summary[summary_key] = {
            key: float(np.mean([row[row_key][key] for row in rows if key in row[row_key]]))
            for key in keys
        }
    if rows:
        summary["mean_inference_time_s"] = float(np.mean([row["inference_time_s"] for row in rows]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an SQ estimation checkpoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", default=Path("datasets/primitive_dataset"), type=Path)
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--class-id", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-fitting-points", type=int, default=300)
    parser.add_argument(
        "--preprocess",
        choices=["runtime", "dataset"],
        default="runtime",
        help="runtime uses sq_est_inference OBB preprocessing; dataset matches PrimitiveDataset normalization.",
    )
    parser.add_argument("--output", type=Path, default=Path("eval_results/sq_est_eval.json"))
    args = parser.parse_args()

    model, _ = load_recognition_model(args.config, args.checkpoint, device=args.device)
    manifest, _, _ = load_manifest(args.dataset / args.manifest)
    entries = filter_entries(
        get_split_entries(manifest, args.split),
        class_ids=parse_class_ids(args.class_id),
    )
    if args.max_samples is not None:
        entries = entries[: args.max_samples]

    rows = []
    for entry in entries:
        sample_path = args.dataset / entry["path"]
        partial_pc, full_pc, gt, metadata = load_sample(sample_path)
        started_at = time.perf_counter()
        result = recognize_sample(partial_pc, model, args.preprocess, args.num_fitting_points)
        inference_time_s = time.perf_counter() - started_at
        source_type = entry.get("source_type", gt.get("type"))

        row = {
            "path": entry["path"],
            "class_id": entry.get("class_id"),
            "source_type": source_type,
            "num_input_points": int(partial_pc.shape[0]),
            "num_full_points": int(full_pc.shape[0]),
            "metadata": metadata,
            "parameters": result["parameters"],
            "parameter_abs_errors": sq_parameter_errors(result["parameters"], gt),
            "fit_errors": fit_errors(full_pc, result),
            "pose_errors": pose_errors(result, gt, source_type),
            "inference_time_s": float(inference_time_s),
        }
        rows.append(row)

    payload = {"summary": summarize(rows), "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
