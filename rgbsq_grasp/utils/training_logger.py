import numpy as np
import wandb


def _strip_mesh_padding(vertices, faces=None, colors=None):
    """Remove zero-padded rows appended for batched mesh logging."""
    valid_vertices = np.any(vertices != 0, axis=1)
    if not np.any(valid_vertices):
        empty = vertices[:0]
        return empty, None if faces is None else faces[:0], None if colors is None else colors[:0]

    last_vertex = np.where(valid_vertices)[0][-1] + 1
    vertices = vertices[:last_vertex]
    if colors is not None:
        colors = colors[:last_vertex]

    if faces is not None:
        valid_faces = np.any(faces != 0, axis=1)
        if np.any(valid_faces):
            last_face = np.where(valid_faces)[0][-1] + 1
            faces = faces[:last_face]
        else:
            faces = faces[:0]

    return vertices, faces, colors


def _to_object3d(vertices, faces=None, colors=None):
    vertices = np.asarray(vertices, dtype=np.float32)
    if faces is None:
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
            points = np.concatenate([vertices, colors], axis=1)
            return wandb.Object3D(points)
        return wandb.Object3D(vertices)

    faces = np.asarray(faces, dtype=np.int32)
    mesh = {
        "type": "lidar/beta",
        "vertices": [list(map(float, v)) for v in vertices],
        "faces": [list(map(int, f)) for f in faces],
    }
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
        mesh["colors"] = [list(map(float, c)) for c in colors]
    return wandb.Object3D(mesh)


class WandbTrainingLogger:
    """Thin adapter matching the previous TensorBoard writer API."""

    def __init__(
        self,
        log_dir,
        project="rgbsq_grasp",
        entity=None,
        run_name=None,
        config=None,
        tags=None,
    ):
        self.log_dir = log_dir
        init_kwargs = {
            "project": project,
            "name": run_name,
            "dir": log_dir,
            "config": config,
        }
        if entity:
            init_kwargs["entity"] = entity
        if tags:
            init_kwargs["tags"] = tags
        self._run = wandb.init(**init_kwargs)

    def add_scalar(self, tag, scalar_value, global_step):
        wandb.log({tag: float(scalar_value)}, step=int(global_step))

    def add_mesh(self, tag, vertices, faces=None, colors=None, global_step=0):
        vertices = np.asarray(vertices)
        if vertices.ndim != 3:
            raise ValueError(f"Expected batched vertices (B, N, 3), got shape {vertices.shape}")

        if faces is not None:
            faces = np.asarray(faces)
        if colors is not None:
            colors = np.asarray(colors)

        log_dict = {}
        for batch_idx in range(vertices.shape[0]):
            batch_vertices = vertices[batch_idx]
            batch_faces = None if faces is None else faces[batch_idx]
            batch_colors = None if colors is None else colors[batch_idx]
            batch_vertices, batch_faces, batch_colors = _strip_mesh_padding(
                batch_vertices, batch_faces, batch_colors
            )
            if batch_vertices.shape[0] == 0:
                continue
            key = f"{tag}/sample_{batch_idx}"
            log_dict[key] = _to_object3d(batch_vertices, batch_faces, batch_colors)

        if log_dict:
            wandb.log(log_dict, step=int(global_step))

    def finish(self):
        if self._run is not None:
            wandb.finish()
            self._run = None
