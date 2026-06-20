"""Configuration for the 2F85 classical antipodal grasp task."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from grasp_sampler.database_paths import (
    legacy_models_ifl_roots,
    normalize_obj_id,
    resolve_grasp_database_root,
)
from isaac_env.air_env_base.element_cfg import *  # noqa: F403
from isaac_env.gripper_task_configs.robotiq_2f85 import apply_to_element_cfg


REMOTE_AGENT = False
READ_FROM_HDF5 = True
apply_to_element_cfg(globals())

CLASSICAL_OBJ_IDS = ["001"]


def _resolve_model_path() -> str:
    database_root = resolve_grasp_database_root()
    if database_root.exists():
        return str(database_root)

    for legacy_root in legacy_models_ifl_roots():
        if legacy_root.exists():
            return str(legacy_root)

    return str(database_root)


MODEL_PATH = _resolve_model_path()
NUM_OBJS = 0
OBJ_LABLE = []
MGN_PATHS = []
OBJ_PATH = MGN_PATHS

CLASSICAL_GRASP_MESH_ROOT = MODEL_PATH
CLASSICAL_MAX_GRASP_CANDIDATES = 500
CLASSICAL_MIN_GRASP_WIDTH_CM = 1.0
CLASSICAL_PREFER_TOPDOWN = True
CLASSICAL_TOPDOWN_WEIGHT = 0.5
CLASSICAL_MIN_TOPDOWN_ALIGNMENT = 0.75
CLASSICAL_TOPDOWN_AXIS = (0.0, 0.0, -1.0)

APPROACH_DISTANCE = 0.08
CLASSICAL_APPROACH_DURATION = 1.25
CLASSICAL_APPROACH_TIMEOUT = 2.0
CLASSICAL_PREGRASP_HOLD_TIME = 0.25

CLASSICAL_OBJ_POSE_OFFSET = (0.75, 0.20, 0.08)
CLASSICAL_OBJ_ORIENTATION_WXYZ = (1.0, 0.0, 0.0, 0.0)
CLASSICAL_OBJ_INIT_POS = (
    ROBOT_POS[0] + CLASSICAL_OBJ_POSE_OFFSET[0],  # noqa: F405
    ROBOT_POS[1] + CLASSICAL_OBJ_POSE_OFFSET[1],  # noqa: F405
    ROBOT_POS[2] + CLASSICAL_OBJ_POSE_OFFSET[2],  # noqa: F405
)


def _grasp_hdf5_path(obj_id: str) -> str:
    return os.path.join(MODEL_PATH, obj_id, f"textured.obj{CLASSICAL_HDF5_SUFFIX}.hdf5")


def validate_classical_assets() -> None:
    robot_usd = os.path.join(HOME_PATH, "models", "ur10e_with_2F85.usd")  # noqa: F405
    missing_robot_assets = [] if os.path.exists(robot_usd) else [robot_usd]
    missing_usd = [path for path in MGN_PATHS if not os.path.exists(path)]
    missing_meshes = [
        os.path.join(MODEL_PATH, obj_id, "textured.obj")
        for obj_id in OBJ_LABLE
        if not os.path.exists(os.path.join(MODEL_PATH, obj_id, "textured.obj"))
    ]
    missing_grasps = [
        _grasp_hdf5_path(obj_id)
        for obj_id in OBJ_LABLE
        if not os.path.exists(_grasp_hdf5_path(obj_id))
    ]
    if missing_robot_assets or missing_usd or missing_meshes or missing_grasps:
        details = []
        if missing_robot_assets:
            details.append(f"Missing robot USD assets: {missing_robot_assets}")
        if missing_usd:
            details.append(f"Missing USD assets: {missing_usd}")
        if missing_meshes:
            details.append(f"Missing object meshes: {missing_meshes}")
        if missing_grasps:
            details.append(
                "Missing 2F85 antipodal HDF5 grasps: "
                f"{missing_grasps}. Run tools/generate_object_grasps.py with --gripper_profile 2f85."
            )
        raise FileNotFoundError("object-v1-Classical-2f85 asset check failed. " + " ".join(details))


def _normalize_classical_obj_id(obj_id: str) -> str:
    return normalize_obj_id(obj_id)


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
            pos=CLASSICAL_OBJ_INIT_POS,
            rot=CLASSICAL_OBJ_ORIENTATION_WXYZ,
        ),
        collision_group=0,
    )


def configure_classical_objects(obj_ids: list[str] | None = None) -> None:
    """Configure the object pool used by object-v1-Classical-2f85."""
    global NUM_OBJS, OBJ_LABLE, MGN_PATHS, OBJ_PATH, MGN_CFGs, OBJ_CFGs

    normalized_ids = [
        _normalize_classical_obj_id(obj_id)
        for obj_id in (CLASSICAL_OBJ_IDS if obj_ids is None else obj_ids)
    ]
    if not normalized_ids:
        raise ValueError("object-v1-Classical-2f85 needs at least one object ID.")

    NUM_OBJS = len(normalized_ids)
    OBJ_LABLE = normalized_ids
    MGN_PATHS = [os.path.join(MODEL_PATH, obj_id, "orbit_obj.usd") for obj_id in OBJ_LABLE]
    OBJ_PATH = MGN_PATHS
    MGN_CFGs = [
        _make_object_cfg(obj_id, usd_path)
        for obj_id, usd_path in zip(OBJ_LABLE, MGN_PATHS, strict=True)
    ]
    OBJ_CFGs = MGN_CFGs


configure_classical_objects()
