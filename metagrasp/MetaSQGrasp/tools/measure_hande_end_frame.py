#!/usr/bin/env python
"""Measure the effective Hand-E frame axes in Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--obj_ids", default="001")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--save_camera_data", action="store_true", default=False)
parser.add_argument("--saved_video", action="store_true", default=False)
parser.add_argument("--saved_video_dir", default=None)
parser.add_argument("--saved_video_fps", type=int, default=30)
parser.add_argument("--saved_video_views", default=None)
parser.add_argument("--max_episodes", type=int, default=None)
parser.add_argument("--max_successful_grasps", type=int, default=None)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.task = "object-v0-Classical"
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaac_env.object_env_classical import ObjectPickSmClassical  # noqa: E402
from isaac_env.object_env_classical.element_cfg import ROBOT_NAME  # noqa: E402
from grasp_sampler.runtime import quaternion_wxyz_to_matrix  # noqa: E402


def _normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / (torch.linalg.norm(vec) + 1e-12)


def _body_position(env, body_name: str, env_id: int = 0) -> torch.Tensor:
    body_names = list(env.robot.data.body_names)
    body_id = body_names.index(body_name)
    root_pos = env.scene[ROBOT_NAME].data.root_state_w[env_id, :3]
    return env.robot.data.body_state_w[env_id, body_id, :3] - root_pos


def main() -> int:
    sim_args = SimpleNamespace(**vars(args_cli))
    simulator = ObjectPickSmClassical(sim_args)
    simulator.init_run()
    env = simulator.env_unwrapped

    for _ in range(args_cli.steps):
        env.sim.step()
        env.scene.update(env.sim.get_physics_dt())

    ee_pose = env._get_ee_pose()[0].detach().cpu().numpy()
    ee_rot = quaternion_wxyz_to_matrix(ee_pose[3:7])
    hande_link = _body_position(env, "hande_link").detach().cpu()
    left = _body_position(env, "hande_left_finger").detach().cpu()
    right = _body_position(env, "hande_right_finger").detach().cpu()
    finger_axis = _normalize(right - left).numpy()
    palm_to_end_axis = _normalize(torch.tensor(ee_pose[:3]) - hande_link).numpy()
    finger_midpoint = 0.5 * (left + right)
    finger_mid_to_end_axis = _normalize(torch.tensor(ee_pose[:3]) - finger_midpoint).numpy()

    print("[Info] hande_end pose in robot base:")
    print(f"  pos={ee_pose[:3].tolist()}")
    print(f"  quat_wxyz={ee_pose[3:].tolist()}")
    print("[Info] EE basis columns in robot base:")
    for idx, name in enumerate(("ee_x", "ee_y", "ee_z")):
        print(
            f"  {name}={ee_rot[:, idx].tolist()} "
            f"dot_finger={float(ee_rot[:, idx] @ finger_axis):+.3f} "
            f"dot_palm_to_end={float(ee_rot[:, idx] @ palm_to_end_axis):+.3f} "
            f"dot_finger_mid_to_end={float(ee_rot[:, idx] @ finger_mid_to_end_axis):+.3f}"
        )
    print(f"[Info] measured finger left->right axis={finger_axis.tolist()}")
    print(f"[Info] measured palm->hande_end axis={palm_to_end_axis.tolist()}")
    print(f"[Info] measured finger_midpoint->hande_end axis={finger_mid_to_end_axis.tolist()}")

    simulator.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
