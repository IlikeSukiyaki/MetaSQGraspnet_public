import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaac_env.air_env_base.sim import AIRPickSm
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from .element_cfg import OBJ_LABLE, TOPDOWN_CAMERA_HEIGHT, make_topdown_camera_cfg


class SQPickSmData(AIRPickSm):
    def __init__(self, args_cli):
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

        self.output_dir = getattr(args_cli, "output_dir", "datasets/sqnet_isaac")
        self.max_episodes = getattr(args_cli, "max_episodes", None)
        self.env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            save_camera_data=args_cli.save_camera_data,
            output_dir=self.output_dir,
            topdown_height=self.topdown_height,
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
