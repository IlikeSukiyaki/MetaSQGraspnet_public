"""2F85 classical antipodal grasp task."""

import gymnasium as gym

from .. import agents
from .element_cfg import *  # noqa: F403
from .env import ObjectEnvClassical2F85
from .env_cfg import CellEnvCfg
from .sim import ObjectPickSmClassical2F85

__all__ = ["ObjectEnvClassical2F85", "ObjectPickSmClassical2F85"]


gym.register(
    id="object-v1-Classical-2f85",
    entry_point="isaac_env.object_env_classical_2f85:ObjectEnvClassical2F85",
    kwargs={
        "env_cfg_entry_point": CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
