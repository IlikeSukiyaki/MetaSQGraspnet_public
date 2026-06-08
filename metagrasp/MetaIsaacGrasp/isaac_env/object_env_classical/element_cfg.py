"""Configuration for the classical antipodal grasp smoke task."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from isaac_env.air_env_base.element_cfg import *  # noqa: F403


REMOTE_AGENT = False
READ_FROM_HDF5 = True
NUM_OBJS = 1
N_MULTIPLE_CAM = 1

CLASSICAL_OBJ_IDS = ["001"]


def _resolve_model_path() -> str:
    local_model_path = Path(HOME_PATH) / "models" / "models_ifl"  # noqa: F405
    if local_model_path.exists():
        return str(local_model_path)

    workspace_model_path = Path(HOME_PATH).parents[1] / "models_ifl"  # noqa: F405
    if workspace_model_path.exists():
        return str(workspace_model_path)

    return str(local_model_path)


MODEL_PATH = _resolve_model_path()
OBJ_LABLE = CLASSICAL_OBJ_IDS
MGN_PATHS = [os.path.join(MODEL_PATH, obj_id, "orbit_obj.usd") for obj_id in OBJ_LABLE]
OBJ_PATH = MGN_PATHS

CLASSICAL_GRASP_MESH_ROOT = MODEL_PATH
CLASSICAL_GRIPPER_PROFILE = "hand_e"
CLASSICAL_GRIPPER_COLLISION_ROOT = os.path.join(
    HOME_PATH,  # noqa: F405
    "models",
    "Gripper",
    "hande",
    "robotiq_hande_description",
    "meshes",
)
CLASSICAL_MAX_GRASP_CANDIDATES = 500
# Prefer grasps with at least this jaw opening (cm).  Filters out pinching contacts
# that score highly but cannot support a lift (common on cylindrical objects).
CLASSICAL_MIN_GRASP_WIDTH_CM = 1.0

# Fixed demo spawn: offset from the placeholder default root (ROBOT_POS), meters.
# Center of the reachable table patch, low z for gentle contact (no drop / tumble).
CLASSICAL_OBJ_POSE_OFFSET = (0.75, 0.20, 0.08)
CLASSICAL_OBJ_ORIENTATION_WXYZ = (1.0, 0.0, 0.0, 0.0)
CLASSICAL_OBJ_INIT_POS = (
    ROBOT_POS[0] + CLASSICAL_OBJ_POSE_OFFSET[0],  # noqa: F405
    ROBOT_POS[1] + CLASSICAL_OBJ_POSE_OFFSET[1],  # noqa: F405
    ROBOT_POS[2] + CLASSICAL_OBJ_POSE_OFFSET[2],  # noqa: F405
)


def validate_classical_assets() -> None:
    missing_usd = [path for path in MGN_PATHS if not os.path.exists(path)]
    missing_meshes = [
        os.path.join(MODEL_PATH, obj_id, "textured.obj")
        for obj_id in OBJ_LABLE
        if not os.path.exists(os.path.join(MODEL_PATH, obj_id, "textured.obj"))
    ]
    missing_grasps = [
        os.path.join(MODEL_PATH, obj_id, "textured.obj.hdf5")
        for obj_id in OBJ_LABLE
        if not os.path.exists(os.path.join(MODEL_PATH, obj_id, "textured.obj.hdf5"))
    ]
    if missing_usd or missing_meshes or missing_grasps:
        details = []
        if missing_usd:
            details.append(f"Missing USD assets: {missing_usd}")
        if missing_meshes:
            details.append(f"Missing object meshes: {missing_meshes}")
        if missing_grasps:
            details.append(
                "Missing antipodal HDF5 grasps: "
                f"{missing_grasps}. Run tools/generate_object_grasps.py first."
            )
        raise FileNotFoundError("object-v0-Classical asset check failed. " + " ".join(details))


MGN_CFGs = [
    RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=mgn,
            semantic_tags=[("class", OBJ_LABLE[i])],
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
    for i, mgn in enumerate(MGN_PATHS)
]

OBJ_CFGs = MGN_CFGs
