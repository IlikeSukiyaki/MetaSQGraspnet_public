#!/usr/bin/env python
"""Synthetic frame check for the Robotiq 2F85 classical grasp profile."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_sampler import runtime as grasp_runtime  # noqa: E402
from grasp_sampler.gripper_profiles import get_profile  # noqa: E402


def _axis_name(rot: np.ndarray, vec: np.ndarray) -> str:
    names = ("ee_x", "ee_y", "ee_z")
    idx = int(np.argmax([abs(float(rot[:, i] @ vec)) for i in range(3)]))
    sign = "+" if float(rot[:, idx] @ vec) >= 0.0 else "-"
    return sign + names[idx]


def main() -> int:
    profile = get_profile("2f85")
    baseline = np.array([1.0, 0.0, 0.0])
    approach = np.array([0.0, 0.0, 1.0])
    contact = np.array([0.0, 0.0, 0.0])
    width = 6.0
    gripper_pose = profile.convert_to_gripper_6d(approach, baseline, contact, width)
    ee_pose = profile.gripper_pose_cm_to_ee_pose(gripper_pose).detach().cpu().numpy()
    ee_rot = grasp_runtime.quaternion_wxyz_to_matrix(ee_pose[3:7])

    baseline_axis = _axis_name(ee_rot, baseline)
    approach_axis = _axis_name(ee_rot, approach)
    print("[Info] Synthetic 2F85 frame mapping:")
    print(f"  baseline maps to {baseline_axis}")
    print(f"  approach maps to {approach_axis}")
    print(f"  contact_to_ee_offset_cm={profile.contact_to_ee_offset_cm:.2f}")
    print(f"  hdf5_suffix={profile.hdf5_suffix!r}")

    if baseline_axis != "+ee_x":
        print("[WARN] Baseline is expected to map to +EE X for parallel-jaw opening.")
        return 2
    if approach_axis != "+ee_y":
        print("[WARN] Approach currently maps to +EE Y; update after measuring the USD if needed.")
        return 2

    print("[OK] Synthetic 2F85 mapping is internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
