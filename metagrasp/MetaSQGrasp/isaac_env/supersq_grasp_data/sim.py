import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaac_env.air_env_base.sim import AIRPickSm
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from . import element_cfg as sq_element_cfg
from . import env_cfg as sq_env_cfg
from .element_cfg import NUM_OBJS, OBJ_LABLE, TOPDOWN_CAMERA_HEIGHT, make_topdown_camera_cfg


class SQPickSmData(AIRPickSm):
    def __init__(self, args_cli):
        cli_obj_ids = getattr(args_cli, "obj_ids", None)
        if cli_obj_ids:
            sq_element_cfg.configure_data_objects(
                [obj_id.strip() for obj_id in cli_obj_ids.split(",") if obj_id.strip()]
            )
        self._sync_data_object_config()
        sq_element_cfg.validate_data_assets()

        env_cfg = parse_env_cfg(
            args_cli.task,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        cli_height = getattr(args_cli, "topdown_height", None)
        self.topdown_height = float(
            TOPDOWN_CAMERA_HEIGHT if cli_height is None else cli_height
        )
        if hasattr(env_cfg.scene, "camera_1"):
            env_cfg.scene.camera_1 = make_topdown_camera_cfg(self.topdown_height)

        self.output_dir = getattr(args_cli, "output_dir", "datasets/supersq_grasp_data")
        self.max_episodes = getattr(args_cli, "max_episodes", None)
        self.export_profile = getattr(args_cli, "export_profile", "full")
        self.save_scene_pc = getattr(args_cli, "save_scene_pc", False)
        scene_objects = int(getattr(args_cli, "scene_objects", 1))
        if scene_objects < 1 or scene_objects > NUM_OBJS:
            raise ValueError(
                f"--scene_objects must be between 1 and {NUM_OBJS} (object pool size), "
                f"got {scene_objects}."
            )
        self.scene_objects = scene_objects
        self.env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            save_camera_data=args_cli.save_camera_data,
            output_dir=self.output_dir,
            topdown_height=self.topdown_height,
            export_profile=self.export_profile,
            save_scene_pc=self.save_scene_pc,
            scene_objects=self.scene_objects,
        )
        self.device = self.env_unwrapped.device
        self.num_envs = self.env_unwrapped.num_envs
        self.env_idx = torch.arange(
            args_cli.num_envs, dtype=torch.int64, device=self.device
        )
        self.inference_criteria = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.obj_label = OBJ_LABLE

    def _sync_data_object_config(self) -> None:
        for name in ("NUM_OBJS", "OBJ_LABLE", "MGN_PATHS", "OBJ_PATH", "MGN_CFGs", "OBJ_CFGs"):
            setattr(sq_env_cfg, name, getattr(sq_element_cfg, name))

    def should_stop(self) -> bool:
        if self.max_episodes is None:
            return False
        if self.env_unwrapped.save_camera_data:
            return self.env_unwrapped.num_saved_samples >= self.max_episodes
        return int(self.env_unwrapped.epi_step_count[:, 0].max().item()) >= self.max_episodes

    def propose_action(self, get_pcd=False):
        ids = self.env_idx.clone()[self.inference_criteria]
        actions = self.env_unwrapped.grasp_pose.clone()
        if len(ids) > 0:
            actions[ids] = self.env_unwrapped.get_action(ids, self.obs_buf)
        return actions

    def close(self):
        self.env.close()
