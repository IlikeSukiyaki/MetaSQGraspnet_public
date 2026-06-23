"""Dataset-level metadata helpers for extended_sq generation."""

import csv
import json
import os
from datetime import datetime, timezone

import numpy as np

INSTANCE_MANIFEST_FIELDS = [
    "instance_id",
    "object_name",
    "profile",
    "a1",
    "a2",
    "a3",
    "axis_ratio",
    "e1",
    "e2",
    "k",
    "b",
    "alpha_rad",
    "alpha_deg",
    "cos_alpha",
    "sin_alpha",
    "num_saved_viewpoints",
]

SUPERQUADRIC_MANIFEST_FIELDS = [
    "instance_id",
    "object_name",
    "a1",
    "a2",
    "a3",
    "axis_ratio",
    "e1",
    "e2",
    "num_saved_viewpoints",
]


def write_generation_config(
    dir_name,
    *,
    config_mode,
    object_names,
    num_objects,
    num_cams,
    num_pnts,
    object_param_config,
    save_full_pc,
    save_membership,
):
    payload = {
        "config_mode": config_mode,
        "object_names": object_names,
        "num_objects": num_objects,
        "num_cams": num_cams,
        "num_pnts": num_pnts,
        "save_full_pc": save_full_pc,
        "save_membership": save_membership,
        "object_param_config": object_param_config,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(dir_name, "generation_config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def build_instance_manifest_row(object_name, object_num, profile, parameters, num_saved_viewpoints):
    a1 = float(parameters["a1"])
    a3 = float(parameters["a3"])
    cos_alpha = float(parameters["cos_alpha"])
    sin_alpha = float(parameters["sin_alpha"])
    alpha_rad = float(np.arctan2(sin_alpha, cos_alpha))
    alpha_deg = float(np.degrees(alpha_rad))
    return {
        "instance_id": f"{object_num:04d}",
        "object_name": object_name,
        "profile": profile,
        "a1": a1,
        "a2": float(parameters["a2"]),
        "a3": a3,
        "axis_ratio": a3 / max(a1, 1e-6),
        "e1": float(parameters["e1"]),
        "e2": float(parameters["e2"]),
        "k": float(parameters["k"]),
        "b": float(parameters["b"]),
        "alpha_rad": alpha_rad,
        "alpha_deg": alpha_deg,
        "cos_alpha": cos_alpha,
        "sin_alpha": sin_alpha,
        "num_saved_viewpoints": int(num_saved_viewpoints),
    }


def build_superquadric_manifest_row(object_name, object_num, parameters, num_saved_viewpoints):
    a1 = float(parameters["a1"])
    a3 = float(parameters["a3"])
    return {
        "instance_id": f"{object_num:04d}",
        "object_name": object_name,
        "a1": a1,
        "a2": float(parameters["a2"]),
        "a3": a3,
        "axis_ratio": a3 / max(a1, 1e-6),
        "e1": float(parameters["e1"]),
        "e2": float(parameters["e2"]),
        "num_saved_viewpoints": int(num_saved_viewpoints),
    }


def append_instance_manifest(dir_name, row, fieldnames=None):
    fieldnames = fieldnames or INSTANCE_MANIFEST_FIELDS
    path = os.path.join(dir_name, "instance_manifest.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path
