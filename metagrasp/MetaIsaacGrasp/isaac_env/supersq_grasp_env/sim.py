"""Simulator wrapper for SQNet-estimated superquadric grasping."""

import os
from datetime import datetime

from isaac_env.classical_shared.sim import ClassicalPickSmBase

from . import element_cfg
from . import env as env_module
from . import env_cfg
from .sq_bridge import make_estimator


class SuperSQPickSm(ClassicalPickSmBase):
    element_cfg = element_cfg
    env_module = env_module
    env_cfg_module = env_cfg

    def __init__(self, args_cli):
        super().__init__(args_cli)

        estimator = make_estimator(
            config_path=getattr(args_cli, "sq_config", None),
            checkpoint_path=getattr(args_cli, "sq_checkpoint", None),
            device=getattr(args_cli, "sq_device", None),
            num_fitting_points=getattr(args_cli, "sq_num_fitting_points", 300),
        )

        debug_dir = getattr(args_cli, "sq_debug_dir", None)
        if getattr(args_cli, "save_sq_debug", False) and debug_dir is None:
            debug_dir = os.path.join(
                "logs",
                "sq_debug",
                args_cli.task,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            print(f"[INFO] SuperSQ debug output: {debug_dir}")

        self.env_unwrapped.configure_superquadric_estimator(
            estimator,
            save_sq_debug=getattr(args_cli, "save_sq_debug", False),
            sq_debug_dir=debug_dir,
            sq_camera=getattr(args_cli, "sq_camera", "topdown"),
        )
