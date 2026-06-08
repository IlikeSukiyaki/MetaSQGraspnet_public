"""Isaac Lab scene config for the classical antipodal grasp task."""

from __future__ import annotations

from math import pi

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass

from isaac_env.air_env_base.env_cfg import (
    depth_capture,
    get_camera_pose,
    grasp_success_compute,
    gripper_pose_capture,
    inst_capture,
    normal_capture,
    reset_robot_to_default,
    reset_root_state_sphere,
    rgb_capture,
)

from .element_cfg import *  # noqa: F403

# Third-person viewer presets for success videos (eye, lookat) in world frame (meters).
CLASSICAL_VIEW_LOOKAT = (0.7, 0.4, 1.0)
CLASSICAL_VIDEO_VIEW_PRESETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front": ((0.6, -1.25, 1.65), CLASSICAL_VIEW_LOOKAT),
    "left": ((-0.85, 0.4, 1.65), CLASSICAL_VIEW_LOOKAT),
    "right": ((1.85, 0.4, 1.65), CLASSICAL_VIEW_LOOKAT),
    "back": ((0.6, 1.35, 1.65), CLASSICAL_VIEW_LOOKAT),
}


def reset_classical_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objs"),
):
    """Place the object at a fixed, upright pose with zero velocity (stable demo spawn)."""
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_object_state[env_ids].clone()
    env_num, obj_num = root_states.size(0), root_states.size(1)

    orient = torch.tensor(CLASSICAL_OBJ_ORIENTATION_WXYZ, device=asset.device, dtype=root_states.dtype)  # noqa: F405

    positions = root_states[..., 0:3] + env.scene.env_origins[env_ids][:, None, :]
    orientations = orient.view(1, 1, 4).expand(env_num, obj_num, 4)
    velocities = torch.zeros(env_num, obj_num, 6, device=asset.device, dtype=root_states.dtype)

    asset.write_object_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_object_velocity_to_sim(velocities, env_ids=env_ids)


def classical_pcd_capture(env):
    pcds = []
    for cam_id in range(N_MULTIPLE_CAM):  # noqa: F405
        depths = torch.clip(
            env.scene[f"camera_{cam_id}"].data.output["distance_to_image_plane"],
            0.0,
            DEPTH_MAX,  # noqa: F405
        )
        pcd_map = []
        for env_id in range(env.scene.num_envs):
            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=env.scene[f"camera_{cam_id}"].data.intrinsic_matrices[env_id],
                depth=depths[env_id],
                position=env.get_camera_pose(cam_id, env_id)[:3],
                orientation=env.get_camera_pose(cam_id, env_id)[3:],
                device=env.device,
            )
            expected_size = CAM_WIDTH * CAM_HEIGHT * 3  # noqa: F405
            if pointcloud.numel() != expected_size:
                pcd_map.append(
                    torch.zeros(
                        CAM_HEIGHT,  # noqa: F405
                        CAM_WIDTH,  # noqa: F405
                        3,
                        device=env.device,
                        dtype=depths.dtype,
                    )
                )
            else:
                pcd_map.append(
                    pointcloud.view(CAM_WIDTH, CAM_HEIGHT, 3).permute(1, 0, 2)  # noqa: F405
                )
        pcds.append(torch.stack(pcd_map, 0))
    return torch.stack(pcds, 0).transpose(0, 1)


@configclass
class CellSceneCfg(InteractiveSceneCfg):
    """Configuration for the classical grasp smoke-test scene."""

    def __init__(self, disable_camera, **kwargs):
        self.dome_light = AssetBaseCfg(
            spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=300.0),
        ).replace(prim_path="{ENV_REGEX_NS}/Domelight")

        self.distant_light = AssetBaseCfg(
            spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=800.0),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 12), rot=(0, 0, 0, 0.0)),
        ).replace(prim_path="/World/Distantlight")

        self.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        )
        self.table = TABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Table")  # noqa: F405

        setattr(
            self,
            ROBOT_NAME,  # noqa: F405
            ROBOT_CFGs[ROBOT_NAME].replace(prim_path="{ENV_REGEX_NS}/" + ROBOT_NAME),  # noqa: F405
        )

        super().__init__(**kwargs)

        self.objs = RigidObjectCollectionCfg(
            rigid_objects={
                f"obj_{i}": OBJ_CFGs[i].replace(prim_path="{ENV_REGEX_NS}/obj_" + str(i))  # noqa: F405
                for i in range(NUM_OBJS)  # noqa: F405
            },
        )

        if not disable_camera:
            self.camera_0 = CAMERA_CFG.replace(  # noqa: F405
                prim_path="{ENV_REGEX_NS}" + f"/{ROBOT_NAME}/{CAMERA_NAME}camera_0"  # noqa: F405
            )
            if n_random_cam:  # noqa: F405
                for i in range(n_random_cam):  # noqa: F405
                    setattr(
                        self,
                        random_cam_names[i],  # noqa: F405
                        CAMERA_RANDOM_CFG[i].replace(  # noqa: F405
                            prim_path="{ENV_REGEX_NS}" + f"/{ROBOT_NAME}/{random_cam_names[i]}",  # noqa: F405
                            offset=CameraCfg.OffsetCfg(
                                pos=(0.1, 0.1, 0.1),
                                rot=(0.1, 0.1, 0.1, 0.1),
                                convention="ros",
                            ),
                        ),
                    )

        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/" + ROBOT_NAME + "/base_link",  # noqa: F405
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(  # noqa: F405
                prim_path="/Visuals/EndEffectorFrameTransformer"
            ),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/" + ROBOT_NAME + "/" + ee_name,  # noqa: F405
                    name=ee_name,  # noqa: F405
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                )
            ],
        )

        if ON_HAND:  # noqa: F405
            self.view_frame = FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/" + ROBOT_NAME + "/base_link",  # noqa: F405
                debug_vis=False,
                visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ViewFrameTransformer"),  # noqa: F405
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}" + f"/{ROBOT_NAME}/VMS3D_Femto_Mega_S_0",  # noqa: F405
                        name=ee_name,  # noqa: F405
                        offset=OffsetCfg(pos=VIEW_POS_CAM, rot=VIEW_QUAT_CAM),  # noqa: F405
                    )
                ],
            )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name=ROBOT_NAME, joint_names=list(ARM_JOINT.keys()), scale=1.0  # noqa: F405
    )
    gripper_pos = mdp.BinaryJointPositionActionCfg(
        asset_name=ROBOT_NAME,  # noqa: F405
        joint_names=[*gripper_joint_cfg[ROBOT_NAME]["default"]],  # noqa: F405
        open_command_expr=gripper_joint_cfg[ROBOT_NAME]["open"],  # noqa: F405
        close_command_expr=gripper_joint_cfg[ROBOT_NAME]["close"],  # noqa: F405
    )


@configclass
class ObservationsCfg:
    @configclass
    class ImageCfg(ObsGroup):
        def __init__(self):
            super().__init__()
            if "rgb" in MODALITIES:  # noqa: F405
                self.rgb = ObsTerm(func=rgb_capture)
            if "normals" in MODALITIES:  # noqa: F405
                self.normals = ObsTerm(func=normal_capture)
            if "instance_segmentation_fast" in MODALITIES:  # noqa: F405
                self.instance_segmentation_fast = ObsTerm(func=inst_capture)
            if "distance_to_image_plane" in MODALITIES:  # noqa: F405
                self.distance_to_image_plane = ObsTerm(func=depth_capture)
                self.pcd = ObsTerm(func=classical_pcd_capture)
            self.gripper_pose = ObsTerm(func=gripper_pose_capture)
            self.concatenate_terms = False

    @configclass
    class PolicyCfg(ObsGroup):
        pos = ObsTerm(func=get_camera_pose)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    cam_pos: PolicyCfg = PolicyCfg()
    policy: ImageCfg = ImageCfg()


@configclass
class EventCfg:
    def __init__(self):
        self.reset_robot_pos = EventTerm(func=reset_robot_to_default, mode="reset")
        self.reset_obj_pos = EventTerm(
            func=reset_classical_object_pose,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("objs")},
        )

        for ix in range(n_random_cam):  # noqa: F405
            setattr(
                self,
                f"reset_camera_pos_{random_cam_names[ix]}",  # noqa: F405
                EventTerm(
                    func=reset_root_state_sphere,
                    mode="reset",
                    params={
                        "SPHERE_RADIUS": SPHERE_RADIUS,  # noqa: F405
                        "asset_cfg": SceneEntityCfg(random_cam_names[ix]),  # noqa: F405
                    },
                ),
            )


@configclass
class RewardsCfg:
    grasp_success = RewTerm(func=grasp_success_compute, weight=1)


def time_out(env) -> torch.Tensor:
    cri1 = env.successive_grasp_failure == SUCCESSIVE_GRASP_FAILURE_LIMIT  # noqa: F405
    cri2 = ~env.env_reachable
    cri3 = env.epi_step_count[:, 1] >= STEP_TOTAL  # noqa: F405
    cri4 = env.sm_state == STATE_MACHINE["init_env"]  # noqa: F405
    return (cri1 | cri2 | cri3) & cri4


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class CommandsCfg:
    pass


@configclass
class CurriculumCfg:
    pass


@configclass
class CellEnvCfg(ManagerBasedRLEnvCfg):
    """Environment config for classical antipodal grasp execution."""

    def __init__(self, disable_camera=False, **kwargs):
        super().__init__(**kwargs)
        self.observations = ObservationsCfg()
        self.actions = ActionsCfg()
        self.curriculum = CurriculumCfg()
        self.rewards = RewardsCfg()
        self.terminations = TerminationsCfg()
        self.commands = CommandsCfg()

        self.scene = CellSceneCfg(
            num_envs=4096,
            env_spacing=2.5,
            replicate_physics=True,
            disable_camera=disable_camera,
        )
        self.events = EventCfg()
        self.num_envs = 1
        self.episode_length_s = 5

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5
        front_eye, front_lookat = CLASSICAL_VIDEO_VIEW_PRESETS["front"]
        self.viewer.eye = front_eye
        self.viewer.lookat = front_lookat
        self.sim.dt = 1 / 120
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.enable_ccd = True
