"""Classical antipodal grasp smoke task."""

import gymnasium as gym

from .. import agents
from .element_cfg import *  # noqa: F403
from .env import ObjectEnvClassical
from .env_cfg import CellEnvCfg
from .sim import ObjectPickSmClassical

__all__ = ["ObjectPickSmClassical", "ObjectEnvClassical"]


gym.register(
    id="object-v0-Classical",
    entry_point="isaac_env.object_env_classical:ObjectEnvClassical",
    kwargs={
        "env_cfg_entry_point": CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
