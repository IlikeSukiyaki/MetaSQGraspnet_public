#!/usr/bin/env python
"""Offline sanity checks for classical grasp frame conversion.

Compares the sampler frame, the active Hand-E EE command frame, and the
expected URDF finger axis mapping at the classical spawn pose.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_sampler import runtime as grasp_runtime  # noqa: E402
from grasp_sampler.gripper_profiles import get_profile  # noqa: E402

CLASSICAL_OBJ_ORIENTATION_WXYZ = (1.0, 0.0, 0.0, 0.0)
CLASSICAL_OBJ_POSE_OFFSET = (0.75, 0.20, 0.08)
CLASSICAL_MIN_TOPDOWN_ALIGNMENT = 0.75
desk_center = (0.75, 0.2, 0.0)


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


def _spawn_obj_pose_wxyz() -> np.ndarray:
    pos = np.array(CLASSICAL_OBJ_POSE_OFFSET, dtype=float)
    quat = np.asarray(CLASSICAL_OBJ_ORIENTATION_WXYZ, dtype=float)
    return np.concatenate([pos, quat])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj_id", default="001")
    parser.add_argument("--gripper_profile", default="hand_e", choices=["hand_e", "franka", "2f85"])
    parser.add_argument("--hdf5_suffix", default=None)
    args = parser.parse_args()

    profile = get_profile(args.gripper_profile)
    hdf5_suffix = profile.hdf5_suffix if args.hdf5_suffix is None else args.hdf5_suffix
    obj_pose = _spawn_obj_pose_wxyz()
    obj_pose_cm = grasp_runtime.pose_vector_to_matrix_cm(obj_pose)

    selection = grasp_runtime.select_best_grasp(
        obj_id=args.obj_id,
        object_pose_cm=obj_pose_cm,
        desk_center_m=desk_center,
        profile=profile,
        hdf5_suffix=hdf5_suffix,
        min_topdown_alignment=CLASSICAL_MIN_TOPDOWN_ALIGNMENT,
        rank=0,
    )
    if selection is None:
        print(f"[FAIL] No collision-free grasp found for object {args.obj_id} at spawn pose.")
        return 1

    ee_pose = selection.pose.detach().cpu().numpy()
    ee_rot = grasp_runtime.quaternion_wxyz_to_matrix(ee_pose[3:7])
    sampler_baseline = _normalize(selection.gripper_pose_cm[:3, 0])
    sampler_approach = _normalize(selection.gripper_pose_cm[:3, 2])
    topdown_alignment = float(sampler_approach @ np.array([0.0, 0.0, -1.0]))

    print("[Info] Classical spawn object pose (robot base, m):")
    print(f"  pos={obj_pose[:3].tolist()} quat_wxyz={obj_pose[3:].tolist()}")
    print("[Info] Selected grasp:")
    print(f"  profile={profile.name} hdf5_suffix={hdf5_suffix or '<default>'}")
    print(f"  score={selection.score:.3f} width_cm={selection.width_cm:.2f}")
    print(f"  approach_robot={sampler_approach.tolist()} topdown_alignment={topdown_alignment:.3f}")
    print(f"  ee_pos={ee_pose[:3].tolist()}")
    print(f"  ee_quat_wxyz={ee_pose[3:].tolist()}")
    print("[Info] Axis alignment (command frame vs sampler frame):")
    for idx, name in enumerate(("ee_x", "ee_y", "ee_z")):
        print(
            f"  {name}: "
            f"dot_baseline={float(ee_rot[:, idx] @ sampler_baseline):+.3f} "
            f"dot_approach={float(ee_rot[:, idx] @ sampler_approach):+.3f}"
        )

    axis_index = {"ee_x": 0, "ee_y": 1, "ee_z": 2}
    baseline_axis = max(
        axis_index,
        key=lambda axis: abs(float(ee_rot[:, axis_index[axis]] @ sampler_baseline)),
    )
    approach_axis = max(
        axis_index,
        key=lambda axis: abs(float(ee_rot[:, axis_index[axis]] @ sampler_approach)),
    )
    print(f"[Info] Sampler baseline maps to {baseline_axis}; approach maps to {approach_axis}.")

    if profile.name == "hand_e" and baseline_axis != "ee_x":
        print(
            "[WARN] Jaw baseline is not mapped to EE +X. "
            "Measured finger motion in the loaded Isaac USD is along hande_end +X, so this likely "
            "explains a correct-looking offline grasp but a rotated physical gripper."
        )
        return 2

    if profile.name == "hand_e" and approach_axis != "ee_y":
        print("[WARN] Approach is not mapped to EE +Y; pre-grasp retreat may be wrong.")
        return 2

    print("[OK] Grasp command frame is internally consistent for the selected profile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
