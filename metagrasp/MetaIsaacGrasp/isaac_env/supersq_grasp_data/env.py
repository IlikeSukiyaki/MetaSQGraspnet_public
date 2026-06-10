"""SQNet data environment with dual-camera dataset export."""

from __future__ import annotations

import torch
import warp as wp

from isaac_env import AIREnvBase

from .element_cfg import N_MULTIPLE_CAM, OBJ_LABLE, SQ_DATA_ROOT, TOPDOWN_CAMERA_HEIGHT
from .env_cfg import CellEnvCfg
from .export import save_scene_snapshot
from .wp_cfg import infer_state_machine_data


class SQEnvData(AIREnvBase):
    def __init__(
        self,
        cfg: CellEnvCfg,
        render_mode: str | None = None,
        save_camera_data: bool = False,
        output_dir: str = SQ_DATA_ROOT,
        topdown_height: float = TOPDOWN_CAMERA_HEIGHT,
        export_profile: str = "full",
        save_scene_pc: bool = False,
        scene_objects: int = 1,
        **kwargs,
    ):
        super().__init__(
            cfg,
            render_mode,
            random_drop_obj=False,
            save_camera_data=save_camera_data,
        )
        self.output_dir = output_dir
        self.topdown_height = topdown_height
        self.export_profile = export_profile
        self.save_scene_pc = save_scene_pc
        self.scene_objects = max(1, int(scene_objects))
        self.obj_label = OBJ_LABLE
        self.camera = [self.scene[f"camera_{i}"] for i in range(N_MULTIPLE_CAM)]
        self.saved_sample_dirs = []
        self._saved_episode_ids: set[tuple[int, int]] = set()

    def step(self, grasp_pose, policy_inference_criteria=torch.tensor([])):
        result = super().step(grasp_pose, policy_inference_criteria)
        self.camera_info = [
            self.camera[cam_id].data.info for cam_id in range(N_MULTIPLE_CAM)
        ]
        return result

    def _advance_state_machine(self):
        """Compute the desired state of the robot's end-effector and gripper."""

        ee_pose = self._get_ee_pose()[:, [0, 1, 2, 4, 5, 6, 3]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        env_reachable_and_stable_wp = wp.from_torch(
            self.env_reachable_and_stable.contiguous(), wp.bool
        )

        wp.launch(
            kernel=infer_state_machine_data,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                self.epi_count_wp,
                env_reachable_and_stable_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                ee_pose_wp,
            ],
            device=self.device,
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        self.grasp_pose = self.grasp_pose[:, [0, 1, 2, 6, 3, 4, 5]]

        return torch.cat((des_ee_pose, self.des_gripper_state.unsqueeze(-1)), -1)

    def get_action(self, ids, obs_buf):
        """Hold the robot pose and save stable scene snapshots."""
        if len(ids) > 0:
            if self.save_camera_data:
                for env_id_tensor in ids:
                    env_id = int(env_id_tensor.detach().cpu().item())
                    episode = int(
                        self.epi_step_count[env_id, 0].detach().cpu().item()
                    )
                    sample_key = (env_id, episode)
                    if sample_key not in self._saved_episode_ids:
                        sample_dir = save_scene_snapshot(
                            self,
                            env_id,
                            obs_buf,
                            self.output_dir,
                            export_profile=self.export_profile,
                            save_scene_pc=self.save_scene_pc,
                        )
                        self.saved_sample_dirs.append(sample_dir)
                        self._saved_episode_ids.add(sample_key)
            self.grasp_pose[ids] = self._get_ee_pose()[ids]
        return self.grasp_pose[ids]

    @property
    def num_saved_samples(self) -> int:
        return len(self.saved_sample_dirs)
