"""SQNet-estimated superquadric grasp task."""

import gymnasium as gym

from .. import agents
from .element_cfg import *  # noqa: F403
from .env import SuperSQGraspEnv
from .env_cfg import CellEnvCfg
from .sim import SuperSQPickSm

__all__ = ["SuperSQPickSm", "SuperSQGraspEnv"]


gym.register(
    id="supersq_grasp-v0",
    entry_point="isaac_env.supersq_grasp_env:SuperSQGraspEnv",
    kwargs={
        "env_cfg_entry_point": CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
