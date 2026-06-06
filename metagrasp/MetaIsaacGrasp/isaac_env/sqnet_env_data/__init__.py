import gymnasium as gym

from .. import agents
from .element_cfg import *
from .env import SQEnvData
from .env_cfg import CellEnvCfg
from .sim import SQPickSmData

__all__ = ["SQPickSmData", "SQEnvData"]


gym.register(
    id="SQ-v0-Data",
    entry_point="isaac_env.sqnet_env_data:SQEnvData",
    kwargs={
        "env_cfg_entry_point": CellEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
