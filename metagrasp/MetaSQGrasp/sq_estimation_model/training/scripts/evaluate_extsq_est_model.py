#!/usr/bin/env python3
"""Offline ExtSQ estimation evaluation on a manifest dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from data_loaders.manifest import filter_entries, get_split_entries, load_manifest
from extsq_est_inference import load_extsq_model, load_sample, recognize_extsq
from geometry.primitives import DeformableSuperquadric


PARAMETER_KEYS = ("a1", "a2", "a3", "e1", "e2", "k", "b", "cos_alpha", "sin_alpha")


def parameter_errors(predicted: dict[str, float], gt: dict | None) -> dict[str, float]:
    if not gt:
        return {}
    gt_params = gt.get("parameters", {})
    errors = {}
    for key in PARAMETER_KEYS:
        if key in gt_params and key in predicted:
            errors[f"abs_{key}"] = abs(float(predicted[key]) - float(gt_params[key]))
    return errors


def write_meshes(export_dir: Path, stem: str, result: dict, gt: dict | None) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    pred_mesh = DeformableSuperquadric(result["SE3"], result["parameters"]).mesh
    o3d.io.write_triangle_mesh(str(export_dir / f"{stem}_pred.ply"), pred_mesh)
    if gt:
        gt_mesh = DeformableSuperquadric(gt["SE3"], gt["parameters"]).mesh
        o3d.io.write_triangle_mesh(str(export_dir / f"{stem}_gt.ply"), gt_mesh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ExtSQ estimation checkpoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", default=Path("datasets/extsq_est_dataset"), type=Path)
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--class-id", default="extsq_est_001")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-fitting-points", type=int, default=300)
    parser.add_argument("--export-mesh-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("eval_results/extsq_est_eval.json"))
    args = parser.parse_args()

    model, _ = load_extsq_model(args.config, args.checkpoint, device=args.device)
    manifest, _, _ = load_manifest(args.dataset / args.manifest)
    entries = filter_entries(
        get_split_entries(manifest, args.split),
        class_ids=[args.class_id] if args.class_id else None,
    )
    if args.max_samples is not None:
        entries = entries[: args.max_samples]

    rows = []
    for idx, entry in enumerate(entries):
        sample_path = args.dataset / entry["path"]
        points, gt, metadata = load_sample(sample_path)
        result = recognize_extsq(points, model, num_fitting_points=args.num_fitting_points)
        row = {
            "path": entry["path"],
            "num_input_points": int(points.shape[0]),
            "metadata": metadata,
            "parameters": result["parameters"],
            "parameter_abs_errors": parameter_errors(result["parameters"], gt),
        }
        rows.append(row)
        if args.export_mesh_dir is not None:
            write_meshes(args.export_mesh_dir, f"{idx:04d}", result, gt)

    error_keys = sorted({key for row in rows for key in row["parameter_abs_errors"]})
    summary = {
        "num_samples": len(rows),
        "mean_abs_errors": {
            key: float(np.mean([row["parameter_abs_errors"][key] for row in rows if key in row["parameter_abs_errors"]]))
            for key in error_keys
        },
    }
    payload = {"summary": summary, "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
