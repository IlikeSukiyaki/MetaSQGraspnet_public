#!/usr/bin/env python
"""Check the Hand-E sampler-frame to `hande_end` frame mapping."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_sampler.gripper_profiles import get_profile  # noqa: E402
from grasp_sampler.runtime import quaternion_wxyz_to_matrix  # noqa: E402


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


def candidate_rotations(gripper_pose_cm: np.ndarray) -> dict[str, np.ndarray]:
    baseline = _normalize(gripper_pose_cm[:3, 0])
    approach = _normalize(gripper_pose_cm[:3, 2])
    side = _normalize(np.cross(baseline, approach))
    return {
        "hande_current": np.stack([side, baseline, approach], axis=1),
        "sampler_direct": np.stack([baseline, -side, approach], axis=1),
        "legacy_franka": np.stack([baseline, approach, side], axis=1),
        "hande_flipped_side": np.stack([-side, baseline, approach], axis=1),
    }


def print_alignment(name: str, rotation: np.ndarray, gripper_pose_cm: np.ndarray) -> None:
    sampler_axes = {
        "sampler_x/baseline": _normalize(gripper_pose_cm[:3, 0]),
        "sampler_y": _normalize(gripper_pose_cm[:3, 1]),
        "sampler_z/approach": _normalize(gripper_pose_cm[:3, 2]),
    }
    print(f"\n[{name}]")
    for ee_idx, ee_name in enumerate(("ee_x", "ee_y", "ee_z")):
        values = {
            sampler_name: float(np.dot(rotation[:, ee_idx], sampler_axis))
            for sampler_name, sampler_axis in sampler_axes.items()
        }
        pretty = " ".join(f"{key}={value:+.2f}" for key, value in values.items())
        print(f"  {ee_name}: {pretty}")


def main() -> int:
    profile = get_profile("hand_e")

    approach = np.array([0.0, 0.0, 1.0])
    baseline = np.array([1.0, 0.0, 0.0])
    contact = np.zeros(3)
    width_cm = 4.0

    gripper_pose_cm = profile.convert_to_gripper_6d(approach, baseline, contact, width_cm)
    ee_pose = profile.gripper_pose_cm_to_ee_pose(gripper_pose_cm).cpu().numpy()
    ee_rot = quaternion_wxyz_to_matrix(ee_pose[3:7])

    print("[Info] Candidate remaps for a synthetic grasp:")
    for name, rotation in candidate_rotations(gripper_pose_cm).items():
        print_alignment(name, rotation, gripper_pose_cm)

    print_alignment("active_profile_output", ee_rot, gripper_pose_cm)

    # The loaded Isaac USD has jaw opening on EE +X and approach on EE +Y.
    np.testing.assert_allclose(ee_rot[:, 0], gripper_pose_cm[:3, 0], atol=1e-6)
    np.testing.assert_allclose(ee_rot[:, 1], gripper_pose_cm[:3, 2], atol=1e-6)

    retreat_m = 0.05
    pregrasp_pos = ee_pose[:3] - ee_rot[:, 1] * retreat_m
    np.testing.assert_allclose(pregrasp_pos, ee_pose[:3] - approach * retreat_m, atol=1e-6)

    print("[OK] Hand-E frame check passed: baseline->EE+X, approach->EE+Y.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
