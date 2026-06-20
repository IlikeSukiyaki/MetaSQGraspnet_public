"""Isaac Lab scene config for SQNet-estimated superquadric grasping."""

from isaac_env.classical_shared.env_cfg import build_classical_env_cfg

from . import element_cfg

_cfg = build_classical_env_cfg(element_cfg)

CLASSICAL_VIDEO_VIEW_PRESETS = _cfg.CLASSICAL_VIDEO_VIEW_PRESETS
reset_classical_random_object = _cfg.reset_classical_random_object
CellEnvCfg = _cfg.CellEnvCfg
