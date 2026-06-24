# from isaaclab.sensors.camera import Camera, PinholeCameraCfg
from dataclasses import MISSING
from math import pi

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import (
    ManagerBasedEnv,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    mdp,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
)

from isaaclab.utils import configclass
from isaaclab.utils.math import *

from isaac_env.env_base.env_cfg import *
from .element_cfg import *


def sq_rgb_capture(env):
    data = [
        env.scene[f"camera_{i}"].data.output["rgb"][..., :3].float()
        for i in range(N_MULTIPLE_CAM)  # noqa: F405
    ]
    return torch.stack(data, 0).transpose(0, 1)


def sq_normal_capture(env):
    data = [
        env.scene[f"camera_{i}"].data.output["normals"][..., :3]
        for i in range(N_MULTIPLE_CAM)  # noqa: F405
    ]
    return torch.stack(data, 0).transpose(0, 1)


def sq_inst_capture(env):
    data = [
        env.scene[f"camera_{i}"].data.output["instance_segmentation_fast"].unsqueeze(-1)
        for i in range(N_MULTIPLE_CAM)  # noqa: F405
    ]
    return torch.stack(data, 0).transpose(0, 1)


def sq_depth_capture(env):
    data = [
        env.scene[f"camera_{i}"].data.output["distance_to_image_plane"].unsqueeze(-1)
        for i in range(N_MULTIPLE_CAM)  # noqa: F405
    ]
    depth_data = torch.clip(torch.stack(data, 0).transpose(0, 1), 0.0, DEPTH_MAX)  # noqa: F405
    return depth_data


def sq_pcd_capture(env):
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


MIN_TABLE_SEP = 0.08
_PLACEMENT_MAX_RETRIES = 20


def _sample_table_poses(
    ranges: torch.Tensor,
    count: int,
    device: torch.device,
    min_sep: float = MIN_TABLE_SEP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample independent table poses with a minimum XY separation."""
    best_positions = None
    best_orientations = None

    for _ in range(_PLACEMENT_MAX_RETRIES):
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0],
            ranges[:, 1],
            (count, 6),
            device=device,
        )
        candidate_positions = rand_samples[:, :3]
        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_samples[:, 3],
            rand_samples[:, 4],
            rand_samples[:, 5],
        )
        if count < 2:
            return candidate_positions, orientations_delta

        too_close = False
        for i in range(count):
            for j in range(i + 1, count):
                dist = torch.linalg.norm(
                    candidate_positions[i, :2] - candidate_positions[j, :2]
                )
                if dist < min_sep:
                    too_close = True
                    break
            if too_close:
                break

        best_positions = candidate_positions
        best_orientations = orientations_delta
        if not too_close:
            return candidate_positions, orientations_delta

    return best_positions, best_orientations


def sq_reset_scene_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objs"),
    drop_height: float = 0.1,
):
    """Place K random objects on the table and move the rest off-screen."""
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_object_state[env_ids].clone()
    env_num, obj_num = root_states.size(0), root_states.size(1)
    table_count = min(int(getattr(env, "scene_objects", 1)), obj_num)

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)

    positions = (
        root_states[..., 0:3]
        + env.scene.env_origins[env_ids][:, None].repeat(1, obj_num, 1)
    )
    orientations = root_states[..., 3:7].clone()
    velocities = root_states[..., 7:13].clone()

    for batch_idx, env_id in enumerate(env_ids):
        selected = torch.randperm(obj_num, device=asset.device)[:table_count]
        target_idx = int(selected[torch.randint(0, table_count, (1,), device=asset.device)].item())
        env.obj_chosen[env_id] = target_idx

        table_positions, table_orientations_delta = _sample_table_poses(
            ranges,
            table_count,
            asset.device,
        )
        for table_slot, obj_idx in enumerate(selected.tolist()):
            positions[batch_idx, obj_idx] = (
                positions[batch_idx, obj_idx] + table_positions[table_slot]
            )
            orientations[batch_idx, obj_idx] = math_utils.quat_mul(
                root_states[batch_idx, obj_idx, 3:7],
                table_orientations_delta[table_slot],
            )

        selected_set = set(selected.tolist())
        for obj_idx in range(obj_num):
            if obj_idx in selected_set:
                continue
            positions[batch_idx, obj_idx] = env.scene.env_origins[env_id].clone()
            positions[batch_idx, obj_idx, 2] = drop_height
            orientations[batch_idx, obj_idx] = root_states[batch_idx, obj_idx, 3:7]

    asset.write_object_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_object_velocity_to_sim(velocities, env_ids=env_ids)


def sq_get_camera_pose(env, cam_id=None, env_id=None):
    if cam_id is None:
        return torch.stack(
            [env.get_camera_pose(i) for i in range(N_MULTIPLE_CAM)]  # noqa: F405
        )
    if env_id is None:
        return env.get_camera_pose(cam_id)
    return env.get_camera_pose(cam_id, env_id)


@configclass
class CellSceneCfg(InteractiveSceneCfg):
    """Configuration for the SQNet smoke-test scene."""

    def __init__(self, disable_camera, topdown_height=TOPDOWN_CAMERA_HEIGHT, **kwargs):  # noqa: F405
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

        self.table: AssetBaseCfg = TABLE_CFG.replace(  # noqa: F405
            prim_path="{ENV_REGEX_NS}/Table"
        )
        setattr(
            self,
            ROBOT_NAME,  # noqa: F405
            ROBOT_CFGs[ROBOT_NAME].replace(  # noqa: F405
                prim_path="{ENV_REGEX_NS}/" + ROBOT_NAME  # noqa: F405
            ),
        )

        super().__init__(**kwargs)

        self.objs: RigidObjectCollectionCfg = RigidObjectCollectionCfg(  # noqa: F405
            rigid_objects={
                f"obj_{i}": OBJ_CFGs[i].replace(  # noqa: F405
                    prim_path="{ENV_REGEX_NS}/obj_" + str(i)
                )
                for i in range(NUM_OBJS)  # noqa: F405
            },
        )

        if not disable_camera:
            self.camera_0 = CAMERA_CFG.replace(  # noqa: F405
                prim_path="{ENV_REGEX_NS}" + f"/{ROBOT_NAME}/{CAMERA_NAME}camera_0"  # noqa: F405
            )
            self.camera_1 = make_topdown_camera_cfg(topdown_height)  # noqa: F405

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
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                )
            ],
        )

        if ON_HAND:  # noqa: F405
            self.view_frame = FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/" + ROBOT_NAME + "/base_link",  # noqa: F405
                debug_vis=False,
                visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(  # noqa: F405
                    prim_path="/Visuals/ViewFrameTransformer"
                ),
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}" + f"/{ROBOT_NAME}/VMS3D_Femto_Mega_S_0",  # noqa: F405
                        name=ee_name,  # noqa: F405
                        offset=OffsetCfg(
                            pos=VIEW_POS_CAM,  # noqa: F405
                            rot=VIEW_QUAT_CAM,  # noqa: F405
                        ),
                    )
                ],
            )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

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
            self.rgb = ObsTerm(func=sq_rgb_capture)
            self.normals = ObsTerm(func=sq_normal_capture)
            self.instance_segmentation_fast = ObsTerm(func=sq_inst_capture)
            self.distance_to_image_plane = ObsTerm(func=sq_depth_capture)
            self.pcd = ObsTerm(func=sq_pcd_capture)
            self.gripper_pose = ObsTerm(func=gripper_pose_capture)  # noqa: F405
            self.concatenate_terms = False

    @configclass
    class PolicyCfg(ObsGroup):

        pos = ObsTerm(func=sq_get_camera_pose)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    cam_pos: PolicyCfg = PolicyCfg()
    policy: ImageCfg = ImageCfg()


@configclass
class EventCfg:
    def __init__(self):
        self.reset_robot_pos = EventTerm(
            func=reset_robot_to_default,  # noqa: F405
            mode="reset",
        )
        self.reset_obj_pos = EventTerm(
            func=sq_reset_scene_objects,
            mode="reset",
            params={
                "pose_range": {
                    "x": ee_obj_default[0],  # noqa: F405
                    "y": ee_obj_default[1],  # noqa: F405
                    "z": (0.10, 0.10),
                    "roll": (-1.0, 1.0),
                    "pitch": (-1.0, 1.0),
                    "yaw": (-1.0, 1.0),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("objs"),
            },
        )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    grasp_success = RewTerm(func=grasp_success_compute, weight=1)  # noqa: F405


def time_out(env) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    cri1 = env.successive_grasp_failure == SUCCESSIVE_GRASP_FAILURE_LIMIT  # noqa: F405
    cri2 = ~env.env_reachable
    cri3 = env.epi_step_count[:, 1] >= STEP_TOTAL  # noqa: F405
    cri4 = env.sm_state == STATE_MACHINE["init_env"]  # noqa: F405

    return (cri1 | cri2 | cri3) & cri4


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pass


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


@configclass
class CellEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the SQNet smoke-test environment."""

    def __init__(
        self,
        disable_camera=False,
        topdown_height=TOPDOWN_CAMERA_HEIGHT,  # noqa: F405
        **kwargs,
    ):
        """Initialization."""
        super().__init__(**kwargs)
        self.observations: ObservationsCfg = ObservationsCfg()
        self.actions: ActionsCfg = ActionsCfg()

        self.curriculum: CurriculumCfg = CurriculumCfg()
        self.rewards: RewardsCfg = RewardsCfg()
        self.terminations: TerminationsCfg = TerminationsCfg()
        self.commands: CommandsCfg = CommandsCfg()

        self.scene: CellSceneCfg = CellSceneCfg(
            num_envs=4096,
            env_spacing=2.5,
            replicate_physics=True,
            disable_camera=disable_camera,
            topdown_height=topdown_height,
        )
        self.events: EventCfg = EventCfg()
        self.num_envs = 1
        self.episode_length_s = 5

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 5
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.enable_ccd = True
