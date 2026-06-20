#!/usr/bin/env python
"""Measure the effective Robotiq 2F85 ``tool0`` frame axes in Isaac Sim."""

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
parser.add_argument("--left_body", default=None, help="Optional left finger body name override.")
parser.add_argument("--right_body", default=None, help="Optional right finger body name override.")
parser.add_argument("--palm_body", default=None, help="Optional palm/base body name override.")
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
args_cli.task = "object-v1-Classical-2f85"
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from grasp_sampler.runtime import quaternion_wxyz_to_matrix  # noqa: E402
from isaac_env.object_env_classical_2f85 import ObjectPickSmClassical2F85  # noqa: E402
from isaac_env.object_env_classical_2f85.element_cfg import ROBOT_NAME  # noqa: E402


def _normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / (torch.linalg.norm(vec) + 1e-12)


def _body_position(env, body_name: str, env_id: int = 0) -> torch.Tensor:
    body_names = list(env.robot.data.body_names)
    body_id = body_names.index(body_name)
    root_pos = env.scene[ROBOT_NAME].data.root_state_w[env_id, :3]
    return env.robot.data.body_state_w[env_id, body_id, :3] - root_pos


def _choose_body(body_names: list[str], override: str | None, keywords: tuple[str, ...]) -> str | None:
    if override:
        return override
    lowered = [(name, name.lower()) for name in body_names]
    for keyword in keywords:
        for name, lower in lowered:
            if keyword in lower:
                return name
    return None


def main() -> int:
    sim_args = SimpleNamespace(**vars(args_cli))
    simulator = ObjectPickSmClassical2F85(sim_args)
    simulator.init_run()
    env = simulator.env_unwrapped

    for _ in range(args_cli.steps):
        env.sim.step()
        env.scene.update(env.sim.get_physics_dt())

    body_names = list(env.robot.data.body_names)
    print("[Info] Available robot body names:")
    for name in body_names:
        print(f"  {name}")

    left_name = _choose_body(body_names, args_cli.left_body, ("left_outer_finger", "left_finger", "left"))
    right_name = _choose_body(body_names, args_cli.right_body, ("right_outer_finger", "right_finger", "right"))
    palm_name = _choose_body(body_names, args_cli.palm_body, ("robotiq", "gripper", "tool0", "wrist_3_link"))
    missing = [
        label
        for label, name in (("left_body", left_name), ("right_body", right_name), ("palm_body", palm_name))
        if name is None
    ]
    if missing:
        print(f"[FAIL] Could not infer these body names: {missing}. Re-run with explicit overrides.")
        simulator.close()
        simulation_app.close()
        return 1

    ee_pose = env._get_ee_pose()[0].detach().cpu().numpy()
    ee_rot = quaternion_wxyz_to_matrix(ee_pose[3:7])
    left = _body_position(env, left_name).detach().cpu()
    right = _body_position(env, right_name).detach().cpu()
    palm = _body_position(env, palm_name).detach().cpu()
    finger_axis = _normalize(right - left).numpy()
    palm_to_tool_axis = _normalize(torch.tensor(ee_pose[:3]) - palm).numpy()
    finger_midpoint = 0.5 * (left + right)
    tool_to_finger_mid = finger_midpoint - torch.tensor(ee_pose[:3])
    contact_offset_cm = float(torch.linalg.norm(tool_to_finger_mid) * 100.0)
    tool_to_finger_mid_axis = _normalize(tool_to_finger_mid).numpy()

    print("[Info] Selected bodies:")
    print(f"  left_body={left_name}")
    print(f"  right_body={right_name}")
    print(f"  palm_body={palm_name}")
    print("[Info] tool0 pose in robot base:")
    print(f"  pos={ee_pose[:3].tolist()}")
    print(f"  quat_wxyz={ee_pose[3:].tolist()}")
    print("[Info] EE basis columns in robot base:")
    for idx, name in enumerate(("ee_x", "ee_y", "ee_z")):
        print(
            f"  {name}={ee_rot[:, idx].tolist()} "
            f"dot_finger={float(ee_rot[:, idx] @ finger_axis):+.3f} "
            f"dot_palm_to_tool={float(ee_rot[:, idx] @ palm_to_tool_axis):+.3f} "
            f"dot_tool_to_finger_mid={float(ee_rot[:, idx] @ tool_to_finger_mid_axis):+.3f}"
        )
    print(f"[Info] measured finger left->right axis={finger_axis.tolist()}")
    print(f"[Info] measured tool0->finger_midpoint axis={tool_to_finger_mid_axis.tolist()}")
    print(f"[Info] measured contact_to_ee_offset_cm≈{contact_offset_cm:.2f}")

    simulator.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
