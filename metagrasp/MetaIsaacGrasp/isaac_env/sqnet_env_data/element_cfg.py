# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SQNet data scene constants.

This task intentionally reuses the AIR robot/camera/table configuration and
narrows the object set to selected IFL models for SQNet data collection.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from isaac_env.air_env_base.element_cfg import *  # noqa: F403


NUM_OBJS = 2
nums = ["000", "001"]
N_MULTIPLE_CAM = 2

MODEL_PATH = os.path.join(HOME_PATH, "models", "models_ifl")  # noqa: F405
OBJ_LABLE = ["000", "001"]
MGN_PATHS = [os.path.join(MODEL_PATH, obj_id, "orbit_obj.usd") for obj_id in OBJ_LABLE]
OBJ_PATH = MGN_PATHS

FIX_RAND_CAMERA = True
n_random_cam = 0
random_cam_names = []
CAMERA_RANDOM_CFG = []
SQ_DATA_ROOT = os.path.join("./", "datasets", "sqnet_isaac")

# Top-down camera defaults (world frame). Tune XY here; Z = desk_center[2] + height.
# TOPDOWN_CAMERA_XY = (
#     float(desk_center[0]),  # noqa: F405
#     float(desk_center[1]),  # noqa: F405
# )

TOPDOWN_CAMERA_XY = (1.0, 0.75)

TOPDOWN_CAMERA_HEIGHT = 2.1
TOPDOWN_CAMERA_QUAT = (0.0, 1.0, 0.0, 0.0)
TOPDOWN_CAMERA_ROLE_NAMES = {
    0: "camera_onhand",
    1: "camera_topdown",
}


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

_missing_assets = [path for path in MGN_PATHS if not os.path.exists(path)]
if _missing_assets:
    raise FileNotFoundError(
        "SQ-v0-Data requires physics-ready object USD files. "
        "Run from MetaIsaacGrasp: "
        "python urdf_converter.py --input models/models_ifl/*/textured.urdf. "
        f"Missing: {_missing_assets}"
    )

MGN_CFGs = [
    RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=mgn,
            semantic_tags=[("class", OBJ_LABLE[i])],  # noqa: F405
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
    for i, mgn in enumerate(MGN_PATHS)
]

OBJ_CFGs = MGN_CFGs
