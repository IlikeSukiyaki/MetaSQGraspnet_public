"""Shared Isaac Lab scene config factory for classical grasp tasks."""

from __future__ import annotations

from types import SimpleNamespace

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
    get_camera_pose,
    grasp_success_compute,
    gripper_pose_capture,
    reset_robot_to_default,
    reset_root_state_sphere,
)


CLASSICAL_VIEW_LOOKAT = (0.7, 0.4, 1.0)
CLASSICAL_VIDEO_VIEW_PRESETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front": ((0.6, -1.25, 1.65), CLASSICAL_VIEW_LOOKAT),
    "left": ((-0.85, 0.4, 1.65), CLASSICAL_VIEW_LOOKAT),
    "right": ((1.85, 0.4, 1.65), CLASSICAL_VIEW_LOOKAT),
    "back": ((0.6, 1.35, 1.65), CLASSICAL_VIEW_LOOKAT),
}


def build_classical_env_cfg(element_cfg):
    """Build task-specific config classes from an element config module."""

    def reset_classical_random_object(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("objs"),
        drop_height: float = 0.1,
    ):
        """Place one random object at the stable demo pose and hide the rest."""
        asset = env.scene[asset_cfg.name]
        root_states = asset.data.default_object_state[env_ids].clone()
        env_num, obj_num = root_states.size(0), root_states.size(1)

        orient = torch.tensor(
            element_cfg.CLASSICAL_OBJ_ORIENTATION_WXYZ,
            device=asset.device,
            dtype=root_states.dtype,
        )
        positions = root_states[..., 0:3] + env.scene.env_origins[env_ids][:, None, :]
        orientations = orient.view(1, 1, 4).expand(env_num, obj_num, 4)
        velocities = torch.zeros(env_num, obj_num, 6, device=asset.device, dtype=root_states.dtype)

        for batch_idx, env_id in enumerate(env_ids):
            active_idx = int(torch.randint(0, obj_num, (1,), device=asset.device).item())
            env.obj_chosen[env_id] = active_idx

            for obj_idx in range(obj_num):
                if obj_idx == active_idx:
                    continue
                positions[batch_idx, obj_idx] = env.scene.env_origins[env_id].clone()
                positions[batch_idx, obj_idx, 2] = drop_height

        asset.write_object_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_object_velocity_to_sim(velocities, env_ids=env_ids)

    def classical_rgb_capture(env):
        data = [
            env.scene[f"camera_{cam_id}"].data.output["rgb"][..., :3].float()
            for cam_id in range(element_cfg.N_MULTIPLE_CAM)
        ]
        return torch.stack(data, 0).transpose(0, 1)

    def classical_normal_capture(env):
        data = [
            env.scene[f"camera_{cam_id}"].data.output["normals"][..., :3]
            for cam_id in range(element_cfg.N_MULTIPLE_CAM)
        ]
        return torch.stack(data, 0).transpose(0, 1)

    def classical_inst_capture(env):
        data = [
            env.scene[f"camera_{cam_id}"].data.output["instance_segmentation_fast"].unsqueeze(-1)
            for cam_id in range(element_cfg.N_MULTIPLE_CAM)
        ]
        return torch.stack(data, 0).transpose(0, 1)

    def classical_depth_capture(env):
        data = [
            env.scene[f"camera_{cam_id}"].data.output["distance_to_image_plane"].unsqueeze(-1)
            for cam_id in range(element_cfg.N_MULTIPLE_CAM)
        ]
        return torch.clip(torch.stack(data, 0).transpose(0, 1), 0.0, element_cfg.DEPTH_MAX)

    def classical_pcd_capture(env):
        pcds = []
        for cam_id in range(element_cfg.N_MULTIPLE_CAM):
            depths = torch.clip(
                env.scene[f"camera_{cam_id}"].data.output["distance_to_image_plane"],
                0.0,
                element_cfg.DEPTH_MAX,
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
                expected_size = element_cfg.CAM_WIDTH * element_cfg.CAM_HEIGHT * 3
                if pointcloud.numel() != expected_size:
                    pcd_map.append(
                        torch.zeros(
                            element_cfg.CAM_HEIGHT,
                            element_cfg.CAM_WIDTH,
                            3,
                            device=env.device,
                            dtype=depths.dtype,
                        )
                    )
                else:
                    pcd_map.append(
                        pointcloud.view(
                            element_cfg.CAM_WIDTH,
                            element_cfg.CAM_HEIGHT,
                            3,
                        ).permute(1, 0, 2)
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
            self.table = element_cfg.TABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Table")
            setattr(
                self,
                element_cfg.ROBOT_NAME,
                element_cfg.ROBOT_CFGs[element_cfg.ROBOT_NAME].replace(
                    prim_path="{ENV_REGEX_NS}/" + element_cfg.ROBOT_NAME
                ),
            )
            super().__init__(**kwargs)

            self.objs = RigidObjectCollectionCfg(
                rigid_objects={
                    f"obj_{i}": element_cfg.OBJ_CFGs[i].replace(prim_path="{ENV_REGEX_NS}/obj_" + str(i))
                    for i in range(element_cfg.NUM_OBJS)
                },
            )

            if not disable_camera:
                self.camera_0 = element_cfg.CAMERA_CFG.replace(
                    prim_path="{ENV_REGEX_NS}"
                    + f"/{element_cfg.ROBOT_NAME}/{element_cfg.CAMERA_NAME}camera_0"
                )
                if getattr(element_cfg, "SUPERSQ_ENABLE_TOPDOWN_CAMERA", False):
                    self.camera_1 = element_cfg.make_topdown_camera_cfg()
                if element_cfg.n_random_cam:
                    for i in range(element_cfg.n_random_cam):
                        setattr(
                            self,
                            element_cfg.random_cam_names[i],
                            element_cfg.CAMERA_RANDOM_CFG[i].replace(
                                prim_path="{ENV_REGEX_NS}"
                                + f"/{element_cfg.ROBOT_NAME}/{element_cfg.random_cam_names[i]}",
                                offset=CameraCfg.OffsetCfg(
                                    pos=(0.1, 0.1, 0.1),
                                    rot=(0.1, 0.1, 0.1, 0.1),
                                    convention="ros",
                                ),
                            ),
                        )

            self.ee_frame = FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/" + element_cfg.ROBOT_NAME + "/base_link",
                debug_vis=False,
                visualizer_cfg=element_cfg.FRAME_MARKER_SMALL_CFG.replace(
                    prim_path="/Visuals/EndEffectorFrameTransformer"
                ),
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/" + element_cfg.ROBOT_NAME + "/" + element_cfg.ee_name,
                        name=element_cfg.ee_name,
                        offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                    )
                ],
            )

            if element_cfg.ON_HAND:
                self.view_frame = FrameTransformerCfg(
                    prim_path="{ENV_REGEX_NS}/" + element_cfg.ROBOT_NAME + "/base_link",
                    debug_vis=False,
                    visualizer_cfg=element_cfg.FRAME_MARKER_SMALL_CFG.replace(
                        prim_path="/Visuals/ViewFrameTransformer"
                    ),
                    target_frames=[
                        FrameTransformerCfg.FrameCfg(
                            prim_path="{ENV_REGEX_NS}"
                            + f"/{element_cfg.ROBOT_NAME}/VMS3D_Femto_Mega_S_0",
                            name=element_cfg.ee_name,
                            offset=OffsetCfg(pos=element_cfg.VIEW_POS_CAM, rot=element_cfg.VIEW_QUAT_CAM),
                        )
                    ],
                )

    @configclass
    class ActionsCfg:
        joint_pos = mdp.JointPositionActionCfg(
            asset_name=element_cfg.ROBOT_NAME,
            joint_names=list(element_cfg.ARM_JOINT.keys()),
            scale=1.0,
        )
        gripper_pos = mdp.BinaryJointPositionActionCfg(
            asset_name=element_cfg.ROBOT_NAME,
            joint_names=[*element_cfg.gripper_joint_cfg[element_cfg.ROBOT_NAME]["default"]],
            open_command_expr=element_cfg.gripper_joint_cfg[element_cfg.ROBOT_NAME]["open"],
            close_command_expr=element_cfg.gripper_joint_cfg[element_cfg.ROBOT_NAME]["close"],
        )

    @configclass
    class ObservationsCfg:
        @configclass
        class ImageCfg(ObsGroup):
            def __init__(self):
                super().__init__()
                if "rgb" in element_cfg.MODALITIES:
                    self.rgb = ObsTerm(func=classical_rgb_capture)
                if "normals" in element_cfg.MODALITIES:
                    self.normals = ObsTerm(func=classical_normal_capture)
                if "instance_segmentation_fast" in element_cfg.MODALITIES:
                    self.instance_segmentation_fast = ObsTerm(func=classical_inst_capture)
                if "distance_to_image_plane" in element_cfg.MODALITIES:
                    self.distance_to_image_plane = ObsTerm(func=classical_depth_capture)
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
                func=reset_classical_random_object,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("objs")},
            )

            for ix in range(element_cfg.n_random_cam):
                setattr(
                    self,
                    f"reset_camera_pos_{element_cfg.random_cam_names[ix]}",
                    EventTerm(
                        func=reset_root_state_sphere,
                        mode="reset",
                        params={
                            "SPHERE_RADIUS": element_cfg.SPHERE_RADIUS,
                            "asset_cfg": SceneEntityCfg(element_cfg.random_cam_names[ix]),
                        },
                    ),
                )

    @configclass
    class RewardsCfg:
        grasp_success = RewTerm(func=grasp_success_compute, weight=1)

    def classical_time_out(env) -> torch.Tensor:
        cri1 = env.successive_grasp_failure == element_cfg.SUCCESSIVE_GRASP_FAILURE_LIMIT
        cri2 = ~env.env_reachable
        cri3 = env.epi_step_count[:, 1] >= element_cfg.STEP_TOTAL
        cri4 = env.sm_state == element_cfg.STATE_MACHINE["init_env"]
        return (cri1 | cri2 | cri3) & cri4

    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func=classical_time_out, time_out=True)

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

    return SimpleNamespace(
        CLASSICAL_VIDEO_VIEW_PRESETS=CLASSICAL_VIDEO_VIEW_PRESETS,
        reset_classical_random_object=reset_classical_random_object,
        CellEnvCfg=CellEnvCfg,
    )
