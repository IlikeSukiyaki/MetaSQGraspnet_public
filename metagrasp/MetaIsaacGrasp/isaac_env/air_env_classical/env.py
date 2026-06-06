"""Classical antipodal grasp environment."""

from __future__ import annotations

import torch
import warp as wp

from grasp_sampler import runtime as grasp_runtime
from grasp_sampler.gripper_profiles import get_profile
from isaac_env import AIREnvBase

from .element_cfg import (
    CLASSICAL_GRASP_MESH_ROOT,
    CLASSICAL_GRIPPER_COLLISION_ROOT,
    CLASSICAL_MAX_GRASP_CANDIDATES,
    CLASSICAL_MIN_GRASP_WIDTH_CM,
    CLASSICAL_GRIPPER_PROFILE,
    OBJ_LABLE,
    ROBOT_NAME,
    desk_center,
)
from .env_cfg import CellEnvCfg
from .wp_cfg import infer_state_machine_disc


class AIREnvClassical(AIREnvBase):
    def __init__(
        self,
        cfg: CellEnvCfg,
        render_mode: str | None = None,
        save_camera_data: bool = False,
        **kwargs,
    ):
        super().__init__(
            cfg,
            render_mode=render_mode,
            save_camera_data=save_camera_data,
            **kwargs,
        )
        self.obj_label = OBJ_LABLE
        self.classical_mesh_root = CLASSICAL_GRASP_MESH_ROOT
        self.classical_collision_root = CLASSICAL_GRIPPER_COLLISION_ROOT
        self.classical_max_grasp_candidates = CLASSICAL_MAX_GRASP_CANDIDATES
        self.classical_gripper_profile = get_profile(
            CLASSICAL_GRIPPER_PROFILE,
            mesh_root=self.classical_collision_root,
        )
        self._classical_last_log_episode = {}
        self._classical_last_fallback_log_episode = {}

    def _advance_state_machine(self):
        """Compute desired end-effector pose and gripper command."""
        ee_vel = self._get_ee_vel()
        ee_pose = self._get_ee_pose()[:, [0, 1, 2, 4, 5, 6, 3]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 4, 5, 6, 3]].to(self.device)

        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        ee_vel_wp = wp.from_torch(ee_vel, wp.float32)
        env_reachable_wp = wp.from_torch(self.env_reachable.contiguous(), wp.bool)
        env_reachable_and_stable_wp = wp.from_torch(
            self.env_reachable_and_stable.contiguous(), wp.bool
        )
        grasp_pose_wp = wp.from_torch(self.grasp_pose, wp.transform)

        wp.launch(
            kernel=infer_state_machine_disc,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                self.successive_grasp_failure_wp,
                self.epi_count_wp,
                self.step_count_wp,
                env_reachable_wp,
                env_reachable_and_stable_wp,
                ee_pose_wp,
                ee_vel_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.ee_quat_default_wp,
                grasp_pose_wp,
            ],
            device=self.device,
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat((des_ee_pose, self.des_gripper_state.unsqueeze(-1)), -1)

    def get_action(self, ids, obs_buf):
        """Select collision-filtered antipodal grasps for the requested environments."""
        fallback_ids = []
        for env_id_tensor in ids:
            env_id = int(env_id_tensor.detach().cpu().item())
            if int(self.obj_chosen[env_id].item()) < 0:
                self.obj_chosen[env_id] = 0

            obj_idx = int(self.obj_chosen[env_id].item())
            obj_id = self.obj_label[obj_idx]
            episode = int(self.epi_step_count[env_id, 0].detach().cpu().item())
            attempt = max(0, int(self.successive_grasp_failure[env_id].detach().cpu().item()))

            robot_root_pos_m = (
                self.scene[ROBOT_NAME].data.root_state_w[env_id, :3].detach().cpu().numpy()
            )
            obj_pose_cm = grasp_runtime.pose_vector_to_matrix_cm(
                self._get_obj_pose(obj_idx, env_id)
            )
            selection = grasp_runtime.select_best_grasp(
                obj_id=obj_id,
                object_pose_cm=obj_pose_cm,
                desk_center_m=desk_center,
                mesh_root=self.classical_mesh_root,
                collision_root=self.classical_collision_root,
                max_candidates=self.classical_max_grasp_candidates,
                device=self.device,
                profile=self.classical_gripper_profile,
                robot_root_pos_m=robot_root_pos_m,
                rank=attempt,
                min_width_cm=CLASSICAL_MIN_GRASP_WIDTH_CM,
            )

            if selection is None:
                if self._classical_last_fallback_log_episode.get(env_id) != episode:
                    print(
                        f"[Classical Grasp] No collision-free grasp for env {env_id} "
                        f"episode={episode} attempt={attempt}, "
                        "falling back to image-normal demo grasp."
                    )
                    self._classical_last_fallback_log_episode[env_id] = episode
                fallback_ids.append(env_id_tensor)
                continue

            self.grasp_pose[env_id] = selection.pose
            log_key = (env_id, episode, attempt)
            if self._classical_last_log_episode.get(env_id) != log_key:
                print(
                    "[Classical Grasp] "
                    f"env={env_id} episode={episode} attempt={attempt} obj={obj_id} "
                    f"profile={self.classical_gripper_profile.name} "
                    f"score={selection.score:.3f} "
                    f"valid_candidates={selection.candidate_count} "
                    f"width_cm={selection.width_cm:.2f} "
                    f"ee_pos={selection.pose[:3].detach().cpu().tolist()}"
                )
                self._classical_last_log_episode[env_id] = log_key

        if fallback_ids:
            fallback_tensor = torch.stack(fallback_ids).to(self.device)
            super().get_action_demo(fallback_tensor, obs_buf)

        return self.grasp_pose
