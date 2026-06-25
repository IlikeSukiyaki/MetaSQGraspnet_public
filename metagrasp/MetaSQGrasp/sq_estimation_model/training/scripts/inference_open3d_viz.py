"""Shared Open3D visualization helpers for SQ / ExtSQ inference CLIs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import open3d as o3d

from geometry.primitives import ExtendedSuperquadric, Superquadric, gen_primitive

PAPER_COLORS = {
    "partial_pc": [0.28, 0.30, 0.34],
    "prediction": [0.86, 0.37, 0.24],
    "prediction_wire": [0.55, 0.18, 0.10],
    "ground_truth": [0.23, 0.45, 0.72],
    "ground_truth_wire": [0.12, 0.28, 0.48],
}

COLOR_LABELS = {
    "partial_pc": "dark gray points — partial observation (model input)",
    "prediction": "orange — predicted superquadric",
    "ground_truth": "blue — ground truth (debug only, not used at inference)",
}


def build_legend_entries(
    *,
    viz_mode: str = "overlay",
    show_gt: bool = False,
    surface_style: str = "point_cloud",
) -> list[tuple[list[float], str]]:
    pred_kind = "mesh" if surface_style == "mesh" else "points"
    entries = [
        (PAPER_COLORS["partial_pc"], "Partial observation (model input)"),
        (PAPER_COLORS["prediction"], f"Predicted superquadric ({pred_kind})"),
    ]
    if show_gt and viz_mode in ("overlay", "side_by_side"):
        gt_kind = "mesh" if surface_style == "mesh" else "points"
        entries.append((PAPER_COLORS["ground_truth"], f"Ground truth ({gt_kind}, debug only)"))
    return entries


def render_legend_rgba(
    entries: list[tuple[list[float], str]],
    *,
    width: int = 360,
    row_height: int = 30,
) -> np.ndarray:
    """Render a small RGBA legend panel for in-viewer overlay."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required for in-viewer legend rendering.") from exc

    title_height = 28
    height = 16 + title_height + row_height * len(entries) + 12
    image = Image.new("RGBA", (width, height), (255, 255, 255, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(160, 160, 160, 255), width=1)
    draw.text((14, 10), "Color legend", fill=(0, 0, 0, 255))

    y = 16 + title_height
    swatch_size = 16
    for rgb, label in entries:
        color = tuple(int(max(0, min(255, round(channel * 255)))) for channel in rgb[:3])
        draw.rectangle(
            [14, y + 4, 14 + swatch_size, y + 4 + swatch_size],
            fill=color + (255,),
            outline=(80, 80, 80, 255),
        )
        draw.text((14 + swatch_size + 10, y + 2), label, fill=(0, 0, 0, 255))
        y += row_height
    return np.asarray(image)


def print_color_legend(
    *,
    viz_mode: str = "overlay",
    show_gt: bool = False,
    surface_style: str = "point_cloud",
) -> None:
    """Print a short color key for the Open3D viewer."""
    pred_kind = "mesh" if surface_style == "mesh" else "point cloud"
    print("Visualization color legend:")
    print(f"  {COLOR_LABELS['partial_pc']}")
    print(f"  orange {pred_kind} — predicted superquadric fit")
    if show_gt and viz_mode in ("overlay", "side_by_side"):
        gt_kind = "mesh" if surface_style == "mesh" else "point cloud"
        print(f"  blue {gt_kind} — ground truth (shown only with --show-gt)")
    if viz_mode == "side_by_side" and show_gt:
        print("  Layout: ground truth on the left, observation + prediction on the right.")
    print("  ----------------------------------------------------------------------------")


def as_point_cloud_nx3(point_cloud) -> np.ndarray:
    points = np.asarray(point_cloud, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"Expected a 2D point cloud, got shape {tuple(points.shape)}.")
    if points.shape[1] == 3:
        return points
    if points.shape[0] == 3:
        return points.T
    raise ValueError(f"Expected point cloud shape N x 3 or 3 x N, got {tuple(points.shape)}.")


def load_sample(path: Path) -> tuple[np.ndarray, Optional[dict[str, Any]], Optional[dict[str, Any]], str]:
    """Load partial_pc, optional GT primitive, optional metadata, and label from a .npy file."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if getattr(data, "dtype", None) == object:
        payload = data.item()
        if isinstance(payload, dict):
            if "partial_pc" not in payload:
                raise KeyError(f"{path} dict must contain 'partial_pc'.")
            points = as_point_cloud_nx3(payload["partial_pc"])
            gt = payload["primitives"][0] if payload.get("primitives") else None
            return points, gt, payload.get("generation_metadata"), path.name
        return as_point_cloud_nx3(payload), None, None, path.name
    return as_point_cloud_nx3(data), None, None, path.name


def result_to_jsonable(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "SE3": np.asarray(result["SE3"]).tolist(),
        "parameters": result["parameters"],
        "normalizer": result["normalizer"],
        "segment_frame": np.asarray(result["segment_frame"]).tolist(),
        "pred_fit": np.asarray(result["pred_fit"]).tolist(),
        "num_input_points": int(np.asarray(result["input_points"]).shape[0]),
        "num_processed_points": int(np.asarray(result["processed_points"]).shape[0]),
    }


def gt_primitive_mesh(primitive: dict[str, Any]) -> o3d.geometry.TriangleMesh:
    """Build a renderable GT mesh from a dataset primitive dict."""
    ptype = primitive.get("type")
    se3 = np.asarray(primitive["SE3"], dtype=np.float64)
    params = primitive["parameters"]

    if ptype == "superquadric":
        mesh = Superquadric(se3, params).mesh
    elif ptype in ("extended_sq", "deformable_superquadric"):
        mesh = ExtendedSuperquadric(se3, params).mesh
    elif ptype in gen_primitive:
        mesh = gen_primitive[ptype](se3, params).mesh
    else:
        raise ValueError(f"Unsupported ground-truth primitive type for rendering: {ptype!r}")

    mesh.compute_vertex_normals()
    return mesh


def prediction_mesh(
    result: dict[str, Any],
    *,
    extended: bool = False,
) -> o3d.geometry.TriangleMesh:
    if extended:
        mesh = ExtendedSuperquadric(result["SE3"], result["parameters"]).mesh
    else:
        mesh = Superquadric(result["SE3"], result["parameters"]).mesh
    mesh.compute_vertex_normals()
    return mesh


def _point_cloud_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    *,
    color: list[float],
    num_points: int,
) -> o3d.geometry.PointCloud:
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=5)
    pcd.paint_uniform_color(color)
    return pcd


def _surface_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    color: list[float],
    *,
    surface_style: str = "point_cloud",
    num_points: int = 3500,
) -> o3d.geometry.Geometry:
    if surface_style == "mesh":
        colored = deepcopy(mesh)
        colored.paint_uniform_color(color)
        colored.compute_vertex_normals()
        return colored
    if surface_style == "point_cloud":
        return _point_cloud_from_mesh(mesh, color=color, num_points=num_points)
    raise ValueError(f"Unknown surface_style: {surface_style}")


def _surface_from_primitive(
    primitive: dict[str, Any],
    color: list[float],
    *,
    surface_style: str = "point_cloud",
    num_points: int = 3500,
) -> o3d.geometry.Geometry:
    mesh = gt_primitive_mesh(primitive)
    return _surface_from_mesh(
        mesh, color, surface_style=surface_style, num_points=num_points
    )


def _wireframe_from_mesh(mesh: o3d.geometry.TriangleMesh, color: list[float]) -> o3d.geometry.LineSet:
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(deepcopy(mesh))
    wire.paint_uniform_color(color)
    return wire


def _point_cloud_for_display(
    points: np.ndarray,
    *,
    color: list[float],
    max_points: int = 2500,
) -> o3d.geometry.PointCloud:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[indices]
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    pcd.paint_uniform_color(color)
    return pcd


def _combined_extent(geometries: list[o3d.geometry.Geometry]) -> np.ndarray:
    min_bound = np.min([g.get_min_bound() for g in geometries], axis=0)
    max_bound = np.max([g.get_max_bound() for g in geometries], axis=0)
    return max_bound - min_bound


def _offset_geometry(geometry: o3d.geometry.Geometry, translation: np.ndarray) -> o3d.geometry.Geometry:
    moved = deepcopy(geometry)
    moved.translate(np.asarray(translation, dtype=np.float64), relative=True)
    return moved


def build_overlay_scene(
    result: dict[str, Any],
    *,
    gt_primitive: Optional[dict[str, Any]] = None,
    surface_style: str = "point_cloud",
    fit_surface_points: int = 3500,
    extended: bool = False,
) -> list[o3d.geometry.Geometry]:
    geometries: list[o3d.geometry.Geometry] = [
        _point_cloud_for_display(result["input_points"], color=PAPER_COLORS["partial_pc"]),
    ]
    pred_mesh = prediction_mesh(result, extended=extended)
    geometries.append(
        _surface_from_mesh(
            pred_mesh,
            PAPER_COLORS["prediction"],
            surface_style=surface_style,
            num_points=fit_surface_points,
        )
    )
    if gt_primitive is not None:
        if surface_style == "mesh":
            gt_mesh = gt_primitive_mesh(gt_primitive)
            gt_mesh.paint_uniform_color(PAPER_COLORS["ground_truth"])
            geometries.append(gt_mesh)
            geometries.append(_wireframe_from_mesh(gt_mesh, PAPER_COLORS["ground_truth_wire"]))
        else:
            geometries.append(
                _surface_from_primitive(
                    gt_primitive,
                    PAPER_COLORS["ground_truth"],
                    surface_style="point_cloud",
                    num_points=fit_surface_points,
                )
            )
    return geometries


def build_side_by_side_scene(
    result: dict[str, Any],
    *,
    gt_primitive: Optional[dict[str, Any]] = None,
    surface_style: str = "point_cloud",
    fit_surface_points: int = 3500,
    extended: bool = False,
) -> list[o3d.geometry.Geometry]:
    pred_mesh = prediction_mesh(result, extended=extended)
    pred_surface = _surface_from_mesh(
        pred_mesh,
        PAPER_COLORS["prediction"],
        surface_style=surface_style,
        num_points=fit_surface_points,
    )
    input_pcd = _point_cloud_for_display(result["input_points"], color=PAPER_COLORS["partial_pc"])
    right_group: list[o3d.geometry.Geometry] = [input_pcd, pred_surface]

    if gt_primitive is None:
        return right_group

    gt_surface = _surface_from_primitive(
        gt_primitive,
        PAPER_COLORS["ground_truth"],
        surface_style=surface_style,
        num_points=fit_surface_points,
    )
    extent = _combined_extent([gt_surface, pred_surface, input_pcd])
    gap = 1.35 * extent[0]
    left_shift = np.array([-gap / 2, 0.0, 0.0])
    right_shift = np.array([gap / 2, 0.0, 0.0])
    geometries = [_offset_geometry(gt_surface, left_shift)]
    geometries.extend(_offset_geometry(item, right_shift) for item in right_group)
    return geometries


def build_pred_only_scene(
    result: dict[str, Any],
    *,
    surface_style: str = "point_cloud",
    fit_surface_points: int = 3500,
    extended: bool = False,
) -> list[o3d.geometry.Geometry]:
    pred_mesh = prediction_mesh(result, extended=extended)
    pred_surface = _surface_from_mesh(
        pred_mesh,
        PAPER_COLORS["prediction"],
        surface_style=surface_style,
        num_points=fit_surface_points,
    )
    return [
        _point_cloud_for_display(result["input_points"], color=PAPER_COLORS["partial_pc"]),
        pred_surface,
    ]


def configure_render_options(vis: o3d.visualization.Visualizer, *, surface_style: str = "point_cloud") -> None:
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = 3.2 if surface_style == "point_cloud" else 2.2
    opt.line_width = 1.0
    opt.show_coordinate_frame = False
    opt.mesh_show_back_face = True
    opt.light_on = surface_style == "mesh"
    if surface_style == "mesh":
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        if hasattr(o3d.visualization.MeshShadeOption, "Default"):
            opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Default


def _add_geometry_to_scene(
    scene: "o3d.visualization.rendering.Open3DScene",
    name: str,
    geometry: o3d.geometry.Geometry,
    *,
    surface_style: str,
) -> None:
    import open3d.visualization.rendering as rendering

    material = rendering.MaterialRecord()
    if isinstance(geometry, o3d.geometry.PointCloud):
        material.shader = "defaultUnlit"
        material.point_size = 3.2
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        material.shader = "defaultLit" if surface_style == "mesh" else "defaultUnlit"
    elif isinstance(geometry, o3d.geometry.LineSet):
        material.shader = "unlitLine"
        material.line_width = 1.0
    else:
        material.shader = "defaultUnlit"
    scene.add_geometry(name, geometry, material)


def _composite_legend_on_image(image_path: Path, legend_rgba: np.ndarray) -> None:
    from PIL import Image

    scene = Image.open(image_path).convert("RGBA")
    legend = Image.fromarray(legend_rgba, mode="RGBA")
    scene.paste(legend, (12, 12), legend)
    scene.convert("RGB").save(image_path)


def open3d_show(
    geometries: list[o3d.geometry.Geometry],
    *,
    title: str,
    width: int = 1600,
    height: int = 1000,
    surface_style: str = "point_cloud",
    legend_rgba: Optional[np.ndarray] = None,
) -> None:
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    class ViewerApp:
        def __init__(self) -> None:
            gui.Application.instance.initialize()
            self.window = gui.Application.instance.create_window(title, width, height)
            self.scene = gui.SceneWidget()
            self.scene.scene = rendering.Open3DScene(self.window.renderer)
            self.scene.scene.set_background([1.0, 1.0, 1.0, 1.0])
            self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

            for index, geometry in enumerate(geometries):
                _add_geometry_to_scene(
                    self.scene.scene,
                    f"geom_{index}",
                    geometry,
                    surface_style=surface_style,
                )

            bounds = self.scene.scene.bounding_box
            self.scene.setup_camera(60.0, bounds, bounds.get_center())

            self.legend_widget = None
            if legend_rgba is not None:
                legend_image = o3d.geometry.Image(legend_rgba)
                self.legend_widget = gui.ImageWidget(legend_image)

            self.window.add_child(self.scene)
            if self.legend_widget is not None:
                self.window.add_child(self.legend_widget)
            self.window.set_on_layout(self._on_layout)
            self.window.set_on_close(self._on_close)
            self.closed = False

        def _on_layout(self, layout_context: gui.LayoutContext) -> None:
            content = self.window.content_rect
            self.scene.frame = content
            if self.legend_widget is not None:
                legend_w = min(380, max(260, content.width // 3))
                legend_h = legend_rgba.shape[0] * legend_w // max(legend_rgba.shape[1], 1)
                legend_h = min(legend_h, content.height - 24)
                self.legend_widget.frame = gui.Rect(content.x + 12, content.y + 12, legend_w, legend_h)

        def _on_close(self) -> bool:
            self.closed = True
            return True

        def run(self) -> None:
            while not self.closed:
                gui.Application.instance.run_one_tick()
            gui.Application.instance.quit()

    ViewerApp().run()


def open3d_show_legacy(
    geometries: list[o3d.geometry.Geometry],
    *,
    title: str,
    width: int = 1600,
    height: int = 1000,
    save_image: Optional[Path] = None,
    surface_style: str = "point_cloud",
    legend_rgba: Optional[np.ndarray] = None,
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=title,
        width=width,
        height=height,
        visible=save_image is None,
    )
    for geometry in geometries:
        vis.add_geometry(geometry)

    configure_render_options(vis, surface_style=surface_style)
    vis.reset_view_point(True)

    if save_image is not None:
        save_image.parent.mkdir(parents=True, exist_ok=True)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(save_image), do_render=True)
        if legend_rgba is not None:
            _composite_legend_on_image(save_image, legend_rgba)
        print(f"Saved figure to {save_image}")
    else:
        vis.run()
    vis.destroy_window()


def visualize_result(
    result: dict[str, Any],
    *,
    title: str,
    gt_primitive: Optional[dict[str, Any]] = None,
    viz_mode: str = "overlay",
    surface_style: str = "point_cloud",
    fit_surface_points: int = 3500,
    save_image: Optional[Path] = None,
    extended: bool = False,
) -> None:
    if viz_mode == "overlay":
        geometries = build_overlay_scene(
            result,
            gt_primitive=gt_primitive,
            surface_style=surface_style,
            fit_surface_points=fit_surface_points,
            extended=extended,
        )
    elif viz_mode == "side_by_side":
        geometries = build_side_by_side_scene(
            result,
            gt_primitive=gt_primitive,
            surface_style=surface_style,
            fit_surface_points=fit_surface_points,
            extended=extended,
        )
    elif viz_mode == "pred_only":
        geometries = build_pred_only_scene(
            result,
            surface_style=surface_style,
            fit_surface_points=fit_surface_points,
            extended=extended,
        )
    else:
        raise ValueError(f"Unknown viz_mode: {viz_mode}")

    legend_entries = build_legend_entries(
        viz_mode=viz_mode,
        show_gt=gt_primitive is not None,
        surface_style=surface_style,
    )
    legend_rgba = render_legend_rgba(legend_entries)
    print_color_legend(
        viz_mode=viz_mode,
        show_gt=gt_primitive is not None,
        surface_style=surface_style,
    )

    try:
        if save_image is not None:
            open3d_show_legacy(
                geometries,
                title=title,
                save_image=save_image,
                surface_style=surface_style,
                legend_rgba=legend_rgba,
            )
        else:
            open3d_show(
                geometries,
                title=title,
                surface_style=surface_style,
                legend_rgba=legend_rgba,
            )
    except Exception as exc:
        print(f"[WARN] GUI viewer with in-window legend unavailable ({exc}); using legacy viewer.")
        open3d_show_legacy(
            geometries,
            title=title,
            save_image=save_image,
            surface_style=surface_style,
            legend_rgba=legend_rgba,
        )
