# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SQNet data scene constants.

This task intentionally reuses the AIR robot/camera/table configuration and
narrows the object set to the curated UR_grasp_database demo pool.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from grasp_sampler.database_paths import (
    DEMO_OBJ_IDS,
    legacy_models_ifl_roots,
    normalize_obj_id,
    resolve_grasp_database_root,
)
from isaac_env.air_env_base.element_cfg import *  # noqa: F403


N_MULTIPLE_CAM = 2

FIX_RAND_CAMERA = True
n_random_cam = 0
random_cam_names = []
CAMERA_RANDOM_CFG = []
SQ_DATA_ROOT = os.path.join("./", "datasets", "supersq_grasp_data")

TOPDOWN_CAMERA_XY = (1.0, 0.75)
TOPDOWN_CAMERA_HEIGHT = 2.1
TOPDOWN_CAMERA_QUAT = (0.0, 1.0, 0.0, 0.0)
TOPDOWN_CAMERA_ROLE_NAMES = {
    0: "camera_onhand",
    1: "camera_topdown",
}

NUM_OBJS = 0
OBJ_LABLE: list[str] = []
MGN_PATHS: list[str] = []
OBJ_PATH: list[str] = []
MGN_CFGs: list[RigidObjectCfg] = []
OBJ_CFGs: list[RigidObjectCfg] = []


def _resolve_model_path() -> str:
    database_root = resolve_grasp_database_root()
    if database_root.exists():
        return str(database_root)

    for legacy_root in legacy_models_ifl_roots():
        if legacy_root.exists():
            return str(legacy_root)

    return str(database_root)


MODEL_PATH = _resolve_model_path()


def make_topdown_camera_cfg(height: float = TOPDOWN_CAMERA_HEIGHT):
    """Create a world-fixed top-down camera at TOPDOWN_CAMERA_XY."""
    topdown_pos = (
        float(TOPDOWN_CAMERA_XY[0]),
        float(TOPDOWN_CAMERA_XY[1]),
        float(desk_center[2] + height),  # noqa: F405
    )
    return CAMERA_CFG.replace(  # noqa: F405
        prim_path="{ENV_REGEX_NS}/camera_1",
        offset=CameraCfg.OffsetCfg(  # noqa: F405
            pos=topdown_pos,
            rot=TOPDOWN_CAMERA_QUAT,
            convention="ros",
        ),
    )


def validate_data_assets() -> None:
    missing_usd = [path for path in MGN_PATHS if not os.path.exists(path)]
    missing_meshes = [
        os.path.join(MODEL_PATH, obj_id, "textured.obj")
        for obj_id in OBJ_LABLE
        if not os.path.exists(os.path.join(MODEL_PATH, obj_id, "textured.obj"))
    ]
    if missing_usd or missing_meshes:
        details = []
        if missing_usd:
            details.append(f"Missing USD assets: {missing_usd}")
        if missing_meshes:
            details.append(f"Missing object meshes: {missing_meshes}")
        raise FileNotFoundError(
            "SQ-v0-Data asset check failed. "
            "Ensure UR_grasp_database/<id>/orbit_obj.usd exists "
            "(python urdf_converter.py --input UR_grasp_database/<id>/textured.urdf). "
            + " ".join(details)
        )


def _make_object_cfg(obj_id: str, usd_path: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            semantic_tags=[("class", obj_id)],
            rigid_props=rigid_props,  # noqa: F405
            mass_props=sim_utils.MassPropertiesCfg(density=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=ROBOT_POS,  # noqa: F405
            rot=(1, 0.0, 0.0, 0),
        ),
        collision_group=0,
    )


def configure_data_objects(obj_ids: list[str] | None = None) -> None:
    """Configure the UR grasp database object pool used by SQ-v0-Data."""
    global NUM_OBJS, OBJ_LABLE, MGN_PATHS, OBJ_PATH, MGN_CFGs, OBJ_CFGs

    normalized_ids = [
        normalize_obj_id(obj_id)
        for obj_id in (DEMO_OBJ_IDS if obj_ids is None else obj_ids)
    ]
    if not normalized_ids:
        raise ValueError("SQ-v0-Data needs at least one object ID.")

    NUM_OBJS = len(normalized_ids)
    OBJ_LABLE = normalized_ids
    MGN_PATHS = [os.path.join(MODEL_PATH, obj_id, "orbit_obj.usd") for obj_id in OBJ_LABLE]
    OBJ_PATH = MGN_PATHS
    MGN_CFGs = [
        _make_object_cfg(obj_id, usd_path)
        for obj_id, usd_path in zip(OBJ_LABLE, MGN_PATHS, strict=True)
    ]
    OBJ_CFGs = MGN_CFGs


configure_data_objects()
validate_data_assets()
