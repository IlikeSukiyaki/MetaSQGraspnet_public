#!/usr/bin/env python
"""Generate antipodal HDF5 grasps for one or more IFL object meshes.

This is a non-interactive wrapper around ``grasp_sampler/generate_grasp_object.py``.
It keeps the original sampler behavior while selecting the gripper profile used
for width limits, contact-frame offset, and collision geometry.
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import pathlib
import sys
from typing import Iterable

import numpy as np
import trimesh

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_sampler import generate_grasp_object as sampler
from grasp_sampler.gripper_profiles import get_profile


def _default_mesh_root() -> pathlib.Path:
    local_models = REPO_ROOT / "models" / "models_ifl"
    if local_models.exists():
        return local_models

    workspace_models = REPO_ROOT.parents[1] / "models_ifl"
    if workspace_models.exists():
        return workspace_models

    return local_models


def _build_config(args: argparse.Namespace) -> dict:
    profile = get_profile(args.gripper_profile)
    return {
        "max_angle_sampling": np.pi / 16,
        "max_angle_friction": np.pi / 8,
        "samples_per_point": args.samples_per_point,
        "max_translation": 0.5,
        "max_grasps": args.max_grasps,
        "antipodal": {
            "gripper_length": args.gripper_length,
            "distance_gripper": profile.contact_to_ee_offset_cm,
            "collision_samples": args.collision_samples,
        },
        "suction": {},
    }


def _install_gripper_profile(profile_name: str) -> None:
    profile = get_profile(profile_name)
    sampler.set_gripper_profile(profile)
    sampler.GRIPPER_WIDTH_MAX = profile.width_max_cm
    sampler.PREGRASP_PARALLEL_DISTANCE = profile.pregrasp_extra_cm


def _resolve_mesh_paths(mesh_root: pathlib.Path, categories: Iterable[str] | None) -> list[str]:
    if categories:
        obj_paths = []
        for category in categories:
            matches = glob.glob(str(mesh_root / category / "**" / "textured.obj"), recursive=True)
            if not matches:
                raise FileNotFoundError(f"No textured.obj found for category {category} under {mesh_root}")
            obj_paths.append(matches[0])
        return obj_paths

    return glob.glob(str(mesh_root / "**" / "textured.obj"), recursive=True)


def _generate_one(obj_path: str, mesh_root: pathlib.Path, args: argparse.Namespace, config: dict) -> bool:
    relative_obj_path = pathlib.Path(obj_path).relative_to(mesh_root)
    obj_mesh, _ = sampler.load_mesh(mesh_root, relative_obj_path, scale_factor=100)
    obj_mesh.fill_holes()

    hdf5_file = sampler.create_file(mesh_root, relative_obj_path)
    try:
        if args.com:
            contact_points, index_tri = sampler.sample_surface_points_com(
                mesh=obj_mesh,
                num_points=args.num_points,
            )
        else:
            contact_points, index_tri = trimesh.sample.sample_surface_even(
                mesh=obj_mesh,
                count=args.num_points,
                radius=None,
            )

        counter, _ = sampler.sample_antipodal_grasps_geometrically(
            obj_mesh,
            contact_points,
            contact_indexes=index_tri,
            hdf5_file=hdf5_file,
            mesh_name=relative_obj_path,
            **config,
        )
        print(f"[Info] Wrote {counter} grasps for {relative_obj_path}")
    finally:
        hdf5_file.close()

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh_root", type=pathlib.Path, default=_default_mesh_root())
    parser.add_argument("--categories", type=str, nargs="+", default=["001"])
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--max_grasps", type=int, default=2000)
    parser.add_argument("--samples_per_point", type=int, default=10)
    parser.add_argument("--collision_samples", type=int, default=360 // 24)
    parser.add_argument("--gripper_profile", type=str, default="hand_e", choices=["hand_e", "franka"])
    parser.add_argument("--gripper_length", type=float, default=None)
    parser.add_argument("--pool_size", type=int, default=1)
    parser.add_argument("--com", action="store_true")
    parser.add_argument("--paralleljaw", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mesh_root = args.mesh_root.expanduser().resolve()
    if not mesh_root.exists():
        raise FileNotFoundError(f"Mesh root does not exist: {mesh_root}")

    profile = get_profile(args.gripper_profile)
    if args.gripper_length is None:
        args.gripper_length = profile.width_max_cm
    _install_gripper_profile(args.gripper_profile)
    config = _build_config(args)
    obj_paths = _resolve_mesh_paths(mesh_root, args.categories)
    print(f"[Info] Found object meshes: {obj_paths}")

    if args.pool_size == 1:
        for obj_path in obj_paths:
            _generate_one(obj_path, mesh_root, args, config)
    else:
        with mp.Pool(args.pool_size) as pool:
            pool.starmap(
                _generate_one,
                [(obj_path, mesh_root, args, config) for obj_path in obj_paths],
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
