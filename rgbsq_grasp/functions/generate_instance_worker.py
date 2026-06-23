"""Generate one object instance (all viewpoints) in an isolated process."""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

# Ensure rgbsq_grasp root is importable when invoked as a script.
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Import Open3D-dependent modules only in the worker process.
import functions.object_class as object_class
from functions.data_reader import create_object_from_data
from functions.utils_datagen import get_cam_pnts, get_partial_pc_via_depth, transform_to_pc_frame
from functions.utils_numpy import inverse_SE3
from functions.dataset_metadata import (
    SUPERQUADRIC_MANIFEST_FIELDS,
    append_instance_manifest,
    build_instance_manifest_row,
    build_superquadric_manifest_row,
)


def generate_object_instance(
    object_name,
    object_num,
    *,
    dir_name,
    num_cams,
    num_pnts,
    save_full_pc,
    save_membership,
):
    config_path = os.path.join(WORKSPACE_ROOT, "object_params", f"{object_name}.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    object_class_name = object_name.replace("truncated_", "")
    obj = object_class.load_object[f"{object_class_name}"](config)
    print(
        f"[worker] rendering {object_name}_{object_num:04} "
        f"({obj.num_primitives} primitive(s))",
        flush=True,
    )

    cam_pnts = get_cam_pnts(num_cams)
    partial_pcds, voxel_size = get_partial_pc_via_depth(
        obj, cam_pnts, num_pcd_points=num_pnts, get_color=True
    )

    primitives = []
    for primitive_ind in range(obj.num_primitives):
        primitives.append(
            {
                "type": obj.primitives[primitive_ind].type,
                "SE3": obj.primitives[primitive_ind].SE3,
                "parameters": obj.primitives[primitive_ind].parameters,
            }
        )

    obj_dir = os.path.join(dir_name, object_name)
    os.makedirs(obj_dir, exist_ok=True)
    generation_metadata = {
        "object_name": object_name,
        "instance_id": object_num,
        "profile": getattr(obj, "sampling_profile", "unknown"),
    }
    instance_parameters = copy.deepcopy(obj.primitives[0].parameters)

    saved = 0
    for view_point_num in range(num_cams):
        partial_pcd = partial_pcds[view_point_num]
        partial_pcd, object_frame = transform_to_pc_frame(partial_pcd)

        if partial_pcd is None:
            print(
                f"----- discard view point {view_point_num} "
                f"since only a single plane is visible -----",
                flush=True,
            )
            continue

        data = {
            "partial_pc": np.asarray(partial_pcd.points),
            "primitives": copy.deepcopy(primitives),
            "generation_metadata": copy.deepcopy(generation_metadata),
        }
        if data["partial_pc"].shape[0] != num_pnts:
            raise RuntimeError(
                f"{object_name}_{object_num:04}_viewpoint_{view_point_num:02} "
                f"has {data['partial_pc'].shape[0]} points (expected {num_pnts})"
            )

        for data_primitive in data["primitives"]:
            data_primitive["SE3"] = np.matmul(inverse_SE3(object_frame), data_primitive["SE3"])

        if save_full_pc:
            obj_partial_pc_frame = create_object_from_data(data)
            full_obj_pcd = obj_partial_pc_frame.mesh.sample_points_poisson_disk(
                1500, use_triangle_normal=True
            )
            data["full_pc"] = np.concatenate(
                (np.asarray(full_obj_pcd.points), np.asarray(full_obj_pcd.normals)), axis=1
            )

        partial_pcd_colors = np.asarray(partial_pcd.colors)
        partial_pcd_tree = __import__("open3d").geometry.KDTreeFlann(partial_pcd)
        for pnt_idx in range(partial_pcd_colors.shape[0]):
            _, idx, _ = partial_pcd_tree.search_radius_vector_3d(
                partial_pcd.points[pnt_idx], 2 * voxel_size[view_point_num]
            )
            partial_pcd_colors[pnt_idx, :] = np.mean(
                partial_pcd_colors[np.asarray(idx), :], axis=0
            )

        clrs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 0]]
        color_dist = np.zeros((partial_pcd_colors.shape[0], len(clrs)))
        for clr in range(len(clrs)):
            color_dist[:, clr] = np.linalg.norm(partial_pcd_colors - clrs[clr], axis=1)
        membership = np.argmin(color_dist, axis=1)

        if len(set(membership)) < len(data["primitives"]):
            print(
                f"----- discard view point {view_point_num} "
                f"since some primitives are not visible -----",
                flush=True,
            )
            continue

        if save_membership:
            data["membership"] = membership

        out_path = os.path.join(
            obj_dir,
            f"{object_name}_{object_num:04}_viewpoint_{view_point_num:02}.npy",
        )
        np.save(out_path, data)
        saved += 1
        print(f"saved {object_name}_{object_num:04}_viewpoint_{view_point_num:02}", flush=True)

    if saved == 0:
        raise RuntimeError(f"No viewpoints saved for {object_name}_{object_num:04}")

    if object_name in {"extended_sq", "deformable_superquadric"}:
        append_instance_manifest(
            dir_name,
            build_instance_manifest_row(
                object_name,
                object_num,
                generation_metadata["profile"],
                instance_parameters,
                saved,
            ),
        )
    elif object_name == "superquadric":
        append_instance_manifest(
            dir_name,
            build_superquadric_manifest_row(
                object_name,
                object_num,
                instance_parameters,
                saved,
            ),
            fieldnames=SUPERQUADRIC_MANIFEST_FIELDS,
        )
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--object-num", type=int, required=True)
    parser.add_argument("--dir-name", required=True)
    parser.add_argument("--num-cams", type=int, default=16)
    parser.add_argument("--num-pnts", type=int, default=1500)
    parser.add_argument("--save-full-pc", action="store_true")
    parser.add_argument("--save-membership", action="store_true")
    args = parser.parse_args()

    generate_object_instance(
        args.object_name,
        args.object_num,
        dir_name=args.dir_name,
        num_cams=args.num_cams,
        num_pnts=args.num_pnts,
        save_full_pc=args.save_full_pc,
        save_membership=args.save_membership,
    )


if __name__ == "__main__":
    main()
