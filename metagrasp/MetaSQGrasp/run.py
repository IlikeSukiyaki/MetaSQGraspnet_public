import argparse
import torch

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")

parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--saved_video", action="store_true", default=False, help="Save MP4 videos only for successful grasp episodes.")
parser.add_argument("--saved_video_dir", type=str, default=None, help="Directory for success-only grasp videos.")
parser.add_argument("--saved_video_fps", type=int, default=30, help="Playback FPS for success-only grasp videos.")
parser.add_argument(
    "--saved_video_views",
    type=str,
    default="front,left,right,back",
    help="Comma-separated viewer presets for classical success videos. "
    "Choices: front, left, right, back.",
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--save_camera_data", action="store_true", default=False, help="Save camera data.")
parser.add_argument("--output_dir", type=str, default="datasets/supersq_grasp_data", help="Dataset output directory.")
parser.add_argument(
    "--max_episodes",
    type=int,
    default=None,
    help="Stop after this many full environment episodes (init_env resets).",
)
parser.add_argument(
    "--max_successful_grasps",
    type=int,
    default=None,
    help="Stop after saving this many successful grasp videos (--saved_video).",
)
parser.add_argument(
    "--topdown_height",
    type=float,
    default=None,
    help="SQ top-down camera height above the table (meters). "
    "Default: TOPDOWN_CAMERA_HEIGHT in isaac_env/supersq_grasp_data/element_cfg.py.",
)
parser.add_argument(
    "--obj_ids",
    type=str,
    default=None,
    help="Comma-separated object IDs for classical / SQ-v0-Data tasks (e.g. 001,004,011).",
)
parser.add_argument(
    "--scene_objects",
    type=int,
    default=1,
    help="SQ-v0-Data: number of objects placed on the table per episode (default: 1).",
)
parser.add_argument(
    "--export_profile",
    type=str,
    default="full",
    choices=["full", "lite"],
    help="SQ-v0-Data export bundle: full (all tensors + scene.pt) or lite (meta, PNGs, partial_pc).",
)
parser.add_argument(
    "--save_scene_pc",
    action="store_true",
    default=False,
    help="SQ-v0-Data: save per-camera pcd_map.npy (always saved in full profile; optional in lite).",
)
parser.add_argument(
    "--sq_model",
    type=str,
    default="sq_esti_model",
    choices=["sq_esti_model", "extsq_esti_model"],
    help="SQ estimator used by supersq_grasp-v0.",
)
parser.add_argument("--sq_config", type=str, default=None, help="SQ estimator config YAML for supersq_grasp-v0.")
parser.add_argument("--sq_checkpoint", type=str, default=None, help="SQ estimator checkpoint for supersq_grasp-v0.")
parser.add_argument("--sq_device", type=str, default=None, help="Device for SQ estimation, e.g. cpu or cuda:0.")
parser.add_argument(
    "--sq_num_fitting_points",
    type=int,
    default=300,
    help="Number of points used by SQ estimator preprocessing.",
)
parser.add_argument(
    "--sq_camera",
    type=str,
    default="topdown",
    choices=["topdown", "onhand"],
    help="Camera used for supersq_grasp-v0 partial point cloud extraction.",
)
parser.add_argument("--save_sq_debug", action="store_true", default=False, help="Save supersq fit debug artifacts.")
parser.add_argument("--sq_debug_dir", type=str, default=None, help="Directory for supersq debug artifacts.")
parser.add_argument("--task", type=str, default="object-v0-Classical", choices=[
                                                                         "object-v0-Classical",
                                                                         "object-v1-Classical-2f85",
                                                                         "SQ-v0-Data",
                                                                         "supersq_grasp-v0",
                                                                         ], help="Task name.")
parser.add_argument("--distributed", action="store_true", help="Run training with multiple GPUs or nodes.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.saved_video and args_cli.headless and not args_cli.enable_cameras:
    print("[WARN] --saved_video in headless mode needs --enable_cameras for off-screen rendering.")


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.settings
settings = carb.settings.get_settings()

# set different types into different keys
# guideline: each extension puts settings in /ext/[ext name]/ and lists them extension.toml for discoverability
settings.set("/renderer/multiGPU/enabled", True)
settings.set("/renderer/multiGPU/autoEnable", True)
settings.set("/rtx/realtime/mgpu/autoTiling/enabled", True)

"""Rest everything else."""

import time

if args_cli.task.startswith("SQ-"):
    from isaac_env.supersq_grasp_data import SQPickSmData as TaskSim
elif args_cli.task.startswith("supersq"):
    from isaac_env.supersq_grasp_env import SuperSQPickSm as TaskSim
elif args_cli.task == "object-v1-Classical-2f85":
    from isaac_env.object_env_classical_2f85 import ObjectPickSmClassical2F85 as TaskSim
elif args_cli.task.startswith("object-"):
    from isaac_env.object_env_classical import ObjectPickSmClassical as TaskSim
else:
    raise ValueError(f"Unsupported task: {args_cli.task}")

if __name__ == "__main__":
    # run the main function
    simulator = TaskSim(args_cli)
    simulator.init_run()
    while simulation_app.is_running():
        with torch.inference_mode():
            simulator.run()
        if hasattr(simulator, "should_stop") and simulator.should_stop():
            break
    # close the environment
    if hasattr(simulator, "close"):
        simulator.close()
    # close sim app
    simulation_app.close()