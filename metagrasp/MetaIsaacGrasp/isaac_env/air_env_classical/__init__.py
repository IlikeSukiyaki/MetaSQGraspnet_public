"""Classical antipodal grasp smoke task."""

import gymnasium as gym

from .. import agents
from .element_cfg import *  # noqa: F403
from .env import AIREnvClassical
from .env_cfg import CellEnvCfg
from .sim import AIRPickSmClassical

__all__ = ["AIRPickSmClassical", "AIREnvClassical"]


gym.register(
    id="AIR-v0-Classical",
    entry_point="isaac_env.air_env_classical:AIREnvClassical",
    kwargs={
        "env_cfg_entry_point": CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
