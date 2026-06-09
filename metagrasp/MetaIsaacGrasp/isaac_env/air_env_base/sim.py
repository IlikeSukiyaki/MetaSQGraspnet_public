import gymnasium as gym
import os
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.math import subtract_frame_transforms, quat_from_matrix
from isaac_env.air_env_base.element_cfg import *
from isaac_env.utils import *
# initialize warp
from isaac_env.air_env_base.wp_cfg import *
from datetime import datetime
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.utils.io import dump_yaml
from isaac_env.io_utils import dump_pickle
from isaaclab.utils.dict import print_dict
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaac_env.agents.custom_extractor import CustomExtractor

class AIRPickSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. 

    """

    def __init__(self, args_cli):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # parse configuration
        env_cfg = parse_env_cfg(
            args_cli.task,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        saved_video = getattr(args_cli, "saved_video", False)
        video_output_dir = getattr(args_cli, "saved_video_dir", None)
        if saved_video and video_output_dir is None:
            video_output_dir = os.path.join(
                "logs",
                "videos",
                args_cli.task,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
        if saved_video:
            os.makedirs(video_output_dir, exist_ok=True)
            print(f"[INFO] Success video recording enabled: {video_output_dir}")

        video_view_presets = None
        if (
            saved_video
            and args_cli.task.startswith("object-")
            and getattr(args_cli, "saved_video_views", None)
        ):
            try:
                if args_cli.task == "object-v1-Classical-2f85":
                    from isaac_env.object_env_classical_2f85.env_cfg import CLASSICAL_VIDEO_VIEW_PRESETS
                else:
                    from isaac_env.object_env_classical.env_cfg import CLASSICAL_VIDEO_VIEW_PRESETS
            except ImportError:
                CLASSICAL_VIDEO_VIEW_PRESETS = {}
            requested = [name.strip() for name in args_cli.saved_video_views.split(",") if name.strip()]
            video_view_presets = {
                name: CLASSICAL_VIDEO_VIEW_PRESETS[name]
                for name in requested
                if name in CLASSICAL_VIDEO_VIEW_PRESETS
            }
            if video_view_presets:
                print(f"[INFO] Recording success videos from views: {', '.join(video_view_presets)}")
            else:
                video_view_presets = None

        self.env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if saved_video else None,
            save_camera_data=args_cli.save_camera_data,
            save_video=saved_video,
            video_output_dir=video_output_dir,
            video_fps=getattr(args_cli, "saved_video_fps", 30),
            video_view_presets=video_view_presets,
        )
        self.device = self.env_unwrapped.device
        self.num_envs = self.env_unwrapped.num_envs
        # Environment index
        self.env_idx = torch.arange(args_cli.num_envs, dtype=torch.int64, device=self.device)
        self.inference_criteria = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.max_episodes = getattr(args_cli, "max_episodes", None)
        self.max_successful_grasps = getattr(args_cli, "max_successful_grasps", None)

    def should_stop(self) -> bool:
        env = self.env_unwrapped
        if self.max_successful_grasps is not None:
            saved = int(getattr(env, "num_saved_videos", 0))
            if saved >= self.max_successful_grasps:
                print(
                    f"[INFO] Reached max successful grasps ({self.max_successful_grasps}). Stopping."
                )
                return True
        if self.max_episodes is not None:
            episodes = int(env.epi_step_count[:, 0].max().item())
            if episodes >= self.max_episodes:
                print(f"[INFO] Reached max episodes ({self.max_episodes}). Stopping.")
                return True
        return False

    def init_run(self):
        """Initialize the simulation loop."""
        # Environment step
        obs_buf = self.env.reset()
        self.obs_buf = obs_buf[0]
        self.env_unwrapped.update_env_state()
        print("-" * 80)
        print("[INFO]: Reset finish...")
    
    
    def run(self):
        """Runs the simulation loop."""
        # Get the grasp pose
        actions = self.propose_action()

        # Advance the environment and get the observations
        self.obs_buf, reward_buf, reset_terminated, dones, self.inference_criteria = self.env.step(actions)

        
    def propose_action(self, get_pcd = False):
        
        # Get the envs that are in the choose object state
        ids = self.env_idx.clone()[self.inference_criteria]
        if len(ids) == 0:
            return self.env_unwrapped.grasp_pose
        self.env_unwrapped.get_action(ids, self.obs_buf)
        return self.env_unwrapped.grasp_pose

    @property
    def env_unwrapped(self):
        return self.env.unwrapped

    def close(self):
        if hasattr(self.env_unwrapped, "close_video_writers"):
            self.env_unwrapped.close_video_writers()
        self.env.close()


