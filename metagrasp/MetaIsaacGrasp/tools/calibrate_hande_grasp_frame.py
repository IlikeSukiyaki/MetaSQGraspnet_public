#!/usr/bin/env python
"""Inspect candidate Hand-E command-frame remaps for one HDF5 grasp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_sampler.gripper_profiles import get_profile  # noqa: E402
from grasp_sampler.runtime import load_mesh_grasps, resolve_mesh_root, quaternion_wxyz_to_matrix  # noqa: E402
from tools.check_hande_frame import candidate_rotations, print_alignment  # noqa: E402


def _as_jsonable(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="001", help="Object ID in UR_grasp_database.")
    parser.add_argument("--grasp_idx", type=int, default=0, help="HDF5 grasp row to inspect.")
    parser.add_argument("--mesh_root", default=None, help="Object database root.")
    parser.add_argument("--json", type=Path, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mesh_root = resolve_mesh_root(args.mesh_root)
    _, grasp_dict = load_mesh_grasps(args.object, str(mesh_root))
    grasps = np.asarray(grasp_dict["paralleljaw_pregrasp_transform"])
    if args.grasp_idx < 0 or args.grasp_idx >= len(grasps):
        raise IndexError(f"grasp_idx {args.grasp_idx} outside [0, {len(grasps)})")

    grasp = grasps[args.grasp_idx]
    vec_a = grasp[:3]
    vec_b = grasp[3:6]
    contact = grasp[6:9]
    width = float(grasp[9])

    profile = get_profile("hand_e")
    gripper_pose_cm = profile.convert_to_gripper_6d(vec_a, vec_b, contact, width)
    active_pose = profile.gripper_pose_cm_to_ee_pose(gripper_pose_cm).cpu().numpy()
    active_rotation = quaternion_wxyz_to_matrix(active_pose[3:7])

    print(f"[Info] mesh_root={mesh_root}")
    print(f"[Info] object={args.object} grasp_idx={args.grasp_idx} width_cm={width:.3f}")
    print(f"[Info] vec_a/approach={vec_a.tolist()}")
    print(f"[Info] vec_b/baseline={vec_b.tolist()}")
    print(f"[Info] contact_pt_cm={contact.tolist()}")
    print(f"[Info] gripper_translation_cm={gripper_pose_cm[:3, 3].tolist()}")

    candidates = candidate_rotations(gripper_pose_cm)
    for name, rotation in candidates.items():
        print_alignment(name, rotation, gripper_pose_cm)
    print_alignment("active_profile_output", active_rotation, gripper_pose_cm)

    if args.json:
        payload = {
            "object": args.object,
            "grasp_idx": args.grasp_idx,
            "width_cm": width,
            "vec_a": vec_a.tolist(),
            "vec_b": vec_b.tolist(),
            "contact_pt_cm": contact.tolist(),
            "gripper_pose_cm": _as_jsonable(gripper_pose_cm),
            "candidate_rotations": {
                name: _as_jsonable(rotation)
                for name, rotation in candidates.items()
            },
            "active_profile_rotation": _as_jsonable(active_rotation),
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[Info] Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
