"""SQ-estimated superquadric grasp environment."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from grasp_sampler import runtime as grasp_runtime
from grasp_sampler.sq_mesh import superquadric_trimesh_cm
from isaac_env.classical_shared.env import ObjectEnvClassicalBase
from tools.sq_partial_pc import extract_partial_pc_from_obs

from . import element_cfg, env_cfg, wp_cfg


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class SuperSQGraspEnv(ObjectEnvClassicalBase):
    element_cfg = element_cfg
    env_cfg_module = env_cfg
    wp_cfg_module = wp_cfg

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera = [self.scene[f"camera_{i}"] for i in range(element_cfg.N_MULTIPLE_CAM)]
        self.supersq_estimator = None
        self.save_sq_debug = False
        self.sq_debug_dir: Path | None = None
        self.sq_camera_index = element_cfg.SUPERSQ_CAMERA_INDEX
        self._supersq_fit_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._supersq_obs_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._supersq_last_log_key = {}
        self._supersq_observation_shape_logged = False

    def configure_superquadric_estimator(
        self,
        estimator,
        *,
        save_sq_debug: bool = False,
        sq_debug_dir: str | Path | None = None,
        sq_camera: str = "topdown",
    ) -> None:
        self.supersq_estimator = estimator
        self.save_sq_debug = save_sq_debug
        self.sq_debug_dir = Path(sq_debug_dir) if sq_debug_dir else None
        if sq_camera == "onhand":
            self.sq_camera_index = 0
        elif sq_camera == "topdown":
            self.sq_camera_index = 1
        else:
            raise ValueError("sq_camera must be 'topdown' or 'onhand'.")
        print(
            "[SuperSQ] Camera config "
            f"N_MULTIPLE_CAM={element_cfg.N_MULTIPLE_CAM} "
            f"sq_camera={sq_camera} sq_camera_index={self.sq_camera_index}"
        )

    def step(self, grasp_pose, policy_inference_criteria=torch.tensor([])):
        result = super().step(grasp_pose, policy_inference_criteria)
        self.camera_info = [
            self.camera[cam_id].data.info for cam_id in range(element_cfg.N_MULTIPLE_CAM)
        ]
        return result

    def _remap_id_to_labels(self, id_to_label: dict[Any, Any] | None) -> dict[str, str]:
        if not id_to_label:
            return {}

        labels: dict[str, str] = {}
        for key, value in id_to_label.items():
            label = str(value)
            match = re.search(r"obj_(\d+)", label)
            if match:
                try:
                    obj_index = int(match.group(1))
                    label = self.obj_label[obj_index]
                except (ValueError, IndexError):
                    label = "-1"
            labels[str(key)] = label
        return labels

    def _camera_id_to_labels(self, cam_id: int, env_id: int) -> dict[str, str]:
        try:
            info = self.camera_info[cam_id][env_id]["instance_segmentation_fast"]
            return self._remap_id_to_labels(info["idToLabels"])
        except (AttributeError, KeyError, IndexError, TypeError):
            return {}

    @staticmethod
    def _se3_m_to_cm(se3_m: np.ndarray) -> np.ndarray:
        se3_cm = np.asarray(se3_m, dtype=np.float64).copy()
        se3_cm[:3, 3] *= 100.0
        return se3_cm

    @staticmethod
    def _pose_cm_to_m(pose_cm: np.ndarray) -> np.ndarray:
        pose_m = np.asarray(pose_cm, dtype=np.float64).copy()
        pose_m[:3, 3] /= 100.0
        return pose_m

    @staticmethod
    def _mesh_cm_to_m(mesh) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(mesh.vertices, dtype=np.float64) / 100.0, np.asarray(mesh.faces, dtype=np.int32)

    def _write_sq_observation_exports(self, out_dir: Path, observation: dict | None) -> dict:
        paths: dict[str, str] = {}
        if not observation:
            return paths

        try:
            import cv2
        except ImportError:
            return paths

        obs_dir = out_dir / "observation"
        obs_dir.mkdir(parents=True, exist_ok=True)

        depth = np.squeeze(np.asarray(observation.get("depth")))
        instance = np.squeeze(np.asarray(observation.get("instance")))
        if depth.size:
            np.save(obs_dir / "depth.npy", depth.astype(np.float32))
            depth_png = np.clip(depth, 0.0, element_cfg.DEPTH_MAX)
            depth_png = (depth_png / element_cfg.DEPTH_MAX * 255.0).astype(np.uint8)
            cv2.imwrite(str(obs_dir / "depth.png"), depth_png)
            paths["depth_png"] = str(obs_dir / "depth.png")

        if instance.size:
            np.save(obs_dir / "instance.npy", instance.astype(np.int32))
            labels = instance.astype(np.uint8)
            instance_png = cv2.applyColorMap((labels * 40) % 255, cv2.COLORMAP_JET)
            cv2.imwrite(str(obs_dir / "instance.png"), instance_png)
            paths["instance_png"] = str(obs_dir / "instance.png")

        rgb = observation.get("rgb")
        if rgb is not None:
            rgb_arr = np.asarray(rgb)[..., :3]
            np.save(obs_dir / "rgb.npy", rgb_arr.astype(np.uint8))
            cv2.imwrite(str(obs_dir / "rgb.png"), cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
            paths["rgb_png"] = str(obs_dir / "rgb.png")

        object_mask = observation.get("object_mask")
        if object_mask is not None:
            mask = np.asarray(object_mask, dtype=bool)
            np.save(obs_dir / "object_mask.npy", mask.astype(np.uint8))
            mask_png = (mask.astype(np.uint8) * 255)
            cv2.imwrite(str(obs_dir / "object_mask.png"), mask_png)
            paths["object_mask_png"] = str(obs_dir / "object_mask.png")

        meta = {
            "camera_index": observation.get("camera_index"),
            "camera_role": observation.get("camera_role"),
            "id_to_labels": observation.get("id_to_labels"),
            "depth_max_m": float(element_cfg.DEPTH_MAX),
        }
        with (obs_dir / "obs_meta.json").open("w", encoding="utf-8") as file:
            json.dump(_jsonable(meta), file, indent=2)
        paths["obs_meta_json"] = str(obs_dir / "obs_meta.json")
        return paths

    def _write_sq_debug_geometry(self, out_dir: Path, partial_payload: dict, sq_result: dict | None, selection) -> dict:
        paths: dict[str, str] = {}
        try:
            import open3d as o3d
        except ImportError:
            return paths

        try:
            partial_pc = np.asarray(partial_payload.get("partial_pc", []), dtype=np.float64)
            if partial_pc.size:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(partial_pc.reshape(-1, 3))
                partial_path = out_dir / "partial_pc.ply"
                o3d.io.write_point_cloud(str(partial_path), pcd)
                paths["partial_pc_ply"] = str(partial_path)

            if sq_result is not None:
                mesh = superquadric_trimesh_cm(
                    sq_result["parameters"],
                    resolution=element_cfg.SUPERSQ_MESH_RESOLUTION,
                )
                vertices_m, faces = self._mesh_cm_to_m(mesh)
                vertices_h = np.concatenate(
                    [vertices_m, np.ones((vertices_m.shape[0], 1), dtype=np.float64)],
                    axis=1,
                )
                vertices_robot = (np.asarray(sq_result["se3_m"], dtype=np.float64) @ vertices_h.T).T[:, :3]
                mesh_o3d = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(vertices_robot),
                    o3d.utility.Vector3iVector(faces),
                )
                mesh_o3d.compute_vertex_normals()
                mesh_path = out_dir / "sq_mesh_robot.ply"
                o3d.io.write_triangle_mesh(str(mesh_path), mesh_o3d)
                paths["sq_mesh_robot_ply"] = str(mesh_path)

            if selection is not None:
                grasp_pose_m = self._pose_cm_to_m(selection.gripper_pose_cm)
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                frame.transform(grasp_pose_m)
                grasp_path = out_dir / "grasp_axes.ply"
                o3d.io.write_triangle_mesh(str(grasp_path), frame)
                paths["grasp_axes_ply"] = str(grasp_path)
            else:
                grasp_path = out_dir / "grasp_axes.ply"
                o3d.io.write_point_cloud(str(grasp_path), o3d.geometry.PointCloud())
                paths["grasp_axes_ply"] = str(grasp_path)
        except Exception as exc:  # pragma: no cover - debug output must not break simulation
            print(f"[SuperSQ] Failed to save debug geometry in {out_dir}: {exc}")
        return paths

    def _save_sq_attempt_debug(
        self,
        *,
        env_id: int,
        episode: int,
        attempt: int,
        obj_id: str,
        partial_payload: dict,
        sq_result: dict | None,
        selection,
        observation: dict | None = None,
    ) -> None:
        if not self.save_sq_debug or self.sq_debug_dir is None:
            return

        out_dir = self.sq_debug_dir / f"env{env_id}_epi{episode:06d}_attempt{attempt}_obj{obj_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "partial_pc.npy", partial_payload, allow_pickle=True)
        geometry_paths = self._write_sq_debug_geometry(out_dir, partial_payload, sq_result, selection)
        observation_paths = self._write_sq_observation_exports(out_dir, observation)
        fit_payload = {
            "object_id": obj_id,
            "frame": {
                "partial_pc": "robot_base_m",
                "sq_mesh": "robot_base_m",
                "grasp_pose": "robot_base_cm",
            },
            "camera_index": self.sq_camera_index,
            "mask_method": partial_payload.get("mask_method"),
            "mask_mode": partial_payload.get("mask_mode"),
            "mask_debug": partial_payload.get("mask_debug"),
            "num_points": partial_payload.get("num_points"),
            "partial_pc_npy": str(out_dir / "partial_pc.npy"),
            "exports": {**geometry_paths, **observation_paths},
            "observation_dir": str(out_dir / "observation") if observation_paths else None,
            "se3_m": None if sq_result is None else sq_result["se3_m"],
            "parameters": None if sq_result is None else sq_result["parameters"],
            "debug": None if sq_result is None else sq_result["debug"],
            "selection": None
            if selection is None
            else {
                "score": selection.score,
                "candidate_count": selection.candidate_count,
                "width_cm": selection.width_cm,
                "approach_alignment": selection.approach_alignment,
                "gripper_pose_cm": selection.gripper_pose_cm,
                "contact_pose_cm": selection.contact_pose_cm,
            },
        }
        with (out_dir / "sq_fit.json").open("w", encoding="utf-8") as file:
            json.dump(_jsonable(fit_payload), file, indent=2)

    def _fit_superquadric_for_episode(
        self,
        *,
        env_id: int,
        episode: int,
        attempt: int,
        obj_idx: int,
        obj_id: str,
        obs_buf: dict,
        obj_pose_cm: np.ndarray,
    ) -> tuple[dict | None, dict | None]:
        cache_key = (env_id, episode)
        if cache_key in self._supersq_fit_cache:
            cached = self._supersq_fit_cache[cache_key]
            return cached["sq_result"], cached["partial_payload"]

        if self.supersq_estimator is None:
            raise RuntimeError("SuperSQ estimator was not configured by the simulator wrapper.")

        data_cam = obs_buf["policy"] if "policy" in obs_buf else obs_buf
        cam_id = self.sq_camera_index
        if not self._supersq_observation_shape_logged:
            depth_shape = tuple(data_cam["distance_to_image_plane"].shape)
            pcd_shape = tuple(data_cam["pcd"].shape)
            print(f"[SuperSQ] Observation shapes depth={depth_shape} pcd={pcd_shape}")
            self._supersq_observation_shape_logged = True
        depth = data_cam["distance_to_image_plane"][env_id, cam_id].detach().cpu().numpy()
        instance = data_cam["instance_segmentation_fast"][env_id, cam_id].detach().cpu().numpy()
        pcd_map = data_cam["pcd"][env_id, cam_id].detach().cpu().numpy()
        rgb = None
        if "rgb" in data_cam:
            rgb = data_cam["rgb"][env_id, cam_id].detach().cpu().numpy()
        mesh_path = Path(self.classical_mesh_root) / obj_id / "textured.obj"
        partial_payload = extract_partial_pc_from_obs(
            depth,
            instance,
            pcd_map,
            id_to_labels=self._camera_id_to_labels(cam_id, env_id),
            object_id=obj_id,
            object_pose_robot_m=self._pose_cm_to_m(obj_pose_cm),
            mesh_path=mesh_path,
            mask_mode="auto",
            color_mode="none",
        )
        observation_payload = {
            "camera_index": cam_id,
            "camera_role": "topdown" if cam_id == 1 else "onhand",
            "depth": np.squeeze(depth),
            "instance": np.squeeze(instance),
            "rgb": rgb,
            "id_to_labels": self._camera_id_to_labels(cam_id, env_id),
            "object_mask": partial_payload.get("instance_mask"),
        }
        self._supersq_obs_cache[cache_key] = observation_payload
        if partial_payload["num_points"] < element_cfg.SUPERSQ_MIN_PARTIAL_PC_POINTS:
            print(
                f"[SuperSQ] Too few partial PC points for env={env_id} episode={episode} "
                f"obj={obj_id}: {partial_payload['num_points']}"
            )
            self._save_sq_attempt_debug(
                env_id=env_id,
                episode=episode,
                attempt=attempt,
                obj_id=obj_id,
                partial_payload=partial_payload,
                sq_result=None,
                selection=None,
                observation=observation_payload,
            )
            return None, partial_payload

        sq_result = self.supersq_estimator.fit_superquadric(partial_payload["partial_pc"])
        self._supersq_fit_cache[cache_key] = {
            "obj_idx": obj_idx,
            "sq_result": sq_result,
            "partial_payload": partial_payload,
        }
        return sq_result, partial_payload

    def get_action(self, ids, obs_buf):
        """Fit a superquadric from partial PC and select a Hand-E grasp."""
        for env_id_tensor in ids:
            env_id = int(env_id_tensor.detach().cpu().item())
            if int(self.obj_chosen[env_id].item()) < 0:
                self.env_cfg_module.reset_classical_random_object(
                    self,
                    torch.tensor([env_id], device=self.device, dtype=torch.int64),
                )

            obj_idx = int(self.obj_chosen[env_id].item())
            obj_id = self.obj_label[obj_idx]
            episode = int(self.epi_step_count[env_id, 0].detach().cpu().item())
            attempt = max(0, int(self.successive_grasp_failure[env_id].detach().cpu().item()))

            robot_root_pos_m = (
                self.scene[element_cfg.ROBOT_NAME].data.root_state_w[env_id, :3].detach().cpu().numpy()
            )
            obj_pose_cm = grasp_runtime.pose_vector_to_matrix_cm(self._get_obj_pose(obj_idx, env_id))
            sq_result, partial_payload = self._fit_superquadric_for_episode(
                env_id=env_id,
                episode=episode,
                attempt=attempt,
                obj_idx=obj_idx,
                obj_id=obj_id,
                obs_buf=obs_buf,
                obj_pose_cm=obj_pose_cm,
            )
            if sq_result is None:
                self.grasp_pose[env_id] = self._get_ee_pose()[env_id]
                continue

            selection = grasp_runtime.select_best_grasp_on_sq_mesh(
                sq_parameters=sq_result["parameters"],
                object_pose_cm=self._se3_m_to_cm(sq_result["se3_m"]),
                desk_center_m=element_cfg.desk_center,
                collision_root=self.classical_collision_root,
                max_candidates=self.classical_max_grasp_candidates,
                device=self.device,
                profile=self.classical_gripper_profile,
                robot_root_pos_m=robot_root_pos_m,
                rank=attempt,
                min_width_cm=element_cfg.CLASSICAL_MIN_GRASP_WIDTH_CM,
                prefer_topdown=element_cfg.CLASSICAL_PREFER_TOPDOWN,
                topdown_weight=element_cfg.CLASSICAL_TOPDOWN_WEIGHT,
                min_topdown_alignment=element_cfg.CLASSICAL_MIN_TOPDOWN_ALIGNMENT,
                down_axis=element_cfg.CLASSICAL_TOPDOWN_AXIS,
                mesh_resolution=element_cfg.SUPERSQ_MESH_RESOLUTION,
                num_surface_points=element_cfg.SUPERSQ_NUM_SURFACE_POINTS,
                max_grasps=element_cfg.SUPERSQ_MAX_GRASPS,
            )
            self._save_sq_attempt_debug(
                env_id=env_id,
                episode=episode,
                attempt=attempt,
                obj_id=obj_id,
                partial_payload=partial_payload,
                sq_result=sq_result,
                selection=selection,
                observation=self._supersq_obs_cache.get((env_id, episode)),
            )
            if selection is None:
                print(
                    f"[SuperSQ] No collision-free SQ grasp for env={env_id} "
                    f"episode={episode} attempt={attempt} obj={obj_id}; retrying."
                )
                self.grasp_pose[env_id] = self._get_ee_pose()[env_id]
                continue

            self.grasp_pose[env_id] = selection.pose
            log_key = (env_id, episode, attempt)
            if self._supersq_last_log_key.get(env_id) != log_key:
                print(
                    "[SuperSQ] "
                    f"env={env_id} episode={episode} attempt={attempt} obj={obj_id} "
                    f"points={partial_payload['num_points']} "
                    f"mask={partial_payload['mask_method']} "
                    f"score={selection.score:.3f} width_cm={selection.width_cm:.2f} "
                    f"alignment={selection.approach_alignment:.2f} "
                    f"candidates={selection.candidate_count}"
                )
                self._supersq_last_log_key[env_id] = log_key

        return self.grasp_pose[ids]
