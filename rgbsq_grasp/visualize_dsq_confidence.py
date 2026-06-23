#!/usr/bin/env python3
"""Visualize observation-support confidence on the predicted deformable superquadric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dsq_supersq_inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_evaluation_segment,
    load_sample,
    recognize_deformable_superquadric,
    resolve_device,
)
from functions.fitting_confidence import (
    compute_local_confidence,
    compute_surface_confidence,
    confidence_to_colors,
)
from functions.primitives import DeformableSuperquadric
from inference import load_recognition_model

PARTIAL_PC_COLOR = "#9aa0a6"


def _subsample_for_display(
    points: np.ndarray,
    values: np.ndarray,
    max_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, values
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices], values[indices]


def _sample_fit_surface_points(
    se3: np.ndarray,
    parameters: Dict[str, float],
    num_points: int = 2500,
) -> np.ndarray:
    mesh = DeformableSuperquadric(se3, parameters).mesh
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=5)
    return np.asarray(pcd.points)


def _set_equal_axes(ax: plt.Axes, points_list: List[np.ndarray]) -> None:
    stacked = np.concatenate(points_list, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-3)
    for axis, value in zip("xyz", center):
        getattr(ax, f"set_{axis}lim")(value - radius, value + radius)


def save_confidence_figure(
    *,
    surface_points: np.ndarray,
    surface_confidence: np.ndarray,
    partial_points: Optional[np.ndarray],
    label: str,
    sigma: float,
    save_path: Path,
    colormap: str = "viridis",
    show_partial: bool = False,
    max_display_points: int = 4000,
) -> None:
    display_surface, display_confidence = _subsample_for_display(
        surface_points, surface_confidence, max_display_points, seed=0
    )

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    scatter = ax.scatter(
        display_surface[:, 0],
        display_surface[:, 1],
        display_surface[:, 2],
        c=display_confidence,
        cmap=colormap,
        vmin=0.0,
        vmax=1.0,
        s=2.5,
        depthshade=False,
    )

    points_for_axes = [display_surface]
    if show_partial and partial_points is not None and partial_points.size > 0:
        partial = _subsample_for_display(
            partial_points,
            np.ones(partial_points.shape[0]),
            max_display_points,
            seed=1,
        )[0]
        ax.scatter(
            partial[:, 0],
            partial[:, 1],
            partial[:, 2],
            c=PARTIAL_PC_COLOR,
            s=1.5,
            alpha=0.35,
            depthshade=False,
            label="Observed partial cloud",
        )
        points_for_axes.append(partial)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Predicted superquadric confidence\n{label}", fontsize=11)
    _set_equal_axes(ax, points_for_axes)
    if show_partial and partial_points is not None:
        ax.legend(loc="upper right", fontsize=9)

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.1)
    colorbar.set_label("Surface confidence")

    mean_conf = float(np.mean(surface_confidence))
    fig.suptitle(
        f"{label}\nσ={sigma:.5f}  mean surface confidence={mean_conf:.3f}",
        fontsize=12,
        y=0.98,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def show_confidence_interactive(
    *,
    surface_points: np.ndarray,
    surface_confidence: np.ndarray,
    partial_points: Optional[np.ndarray],
    label: str,
    sigma: float,
    colormap: str = "viridis",
    show_partial: bool = False,
    max_display_points: int = 4000,
) -> None:
    """Interactive 3D viewer: confidence colormap on predicted surface only."""
    display_surface, display_confidence = _subsample_for_display(
        surface_points, surface_confidence, max_display_points, seed=0
    )

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    scatter = ax.scatter(
        display_surface[:, 0],
        display_surface[:, 1],
        display_surface[:, 2],
        c=display_confidence,
        cmap=colormap,
        vmin=0.0,
        vmax=1.0,
        s=3.0,
        depthshade=False,
        label="Predicted surface",
    )

    points_for_axes = [display_surface]
    if show_partial and partial_points is not None and partial_points.size > 0:
        partial = _subsample_for_display(
            partial_points,
            np.ones(partial_points.shape[0]),
            max_display_points,
            seed=1,
        )[0]
        ax.scatter(
            partial[:, 0],
            partial[:, 1],
            partial[:, 2],
            c=PARTIAL_PC_COLOR,
            s=1.5,
            alpha=0.35,
            depthshade=False,
            label="Observed partial cloud",
        )
        points_for_axes.append(partial)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Predicted superquadric confidence\n{label}", fontsize=11)
    _set_equal_axes(ax, points_for_axes)
    ax.legend(loc="upper right", fontsize=9)

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.1)
    colorbar.set_label("Surface confidence")

    stats_text = (
        f"σ = {sigma:.5f}\n"
        f"surface conf:  min {surface_confidence.min():.3f}  "
        f"mean {surface_confidence.mean():.3f}  max {surface_confidence.max():.3f}"
    )
    fig.text(0.02, 0.98, stats_text, va="top", ha="left", fontsize=9, family="monospace")

    fig.tight_layout()
    plt.show()


def _show_colorbar_panel(colormap: str = "viridis", title: str = "Surface confidence") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(1.4, 5.0), facecolor="white")
    fig.subplots_adjust(left=0.35, right=0.75, top=0.95, bottom=0.05)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=ax)
    colorbar.set_label(title)
    ax.set_title(title, fontsize=10)
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(f"{title} colorbar")
    plt.show(block=False)
    return fig


def show_confidence_open3d(
    *,
    surface_points: np.ndarray,
    surface_confidence: np.ndarray,
    partial_points: Optional[np.ndarray],
    title: str,
    colormap: str = "viridis",
    show_partial: bool = False,
    max_display_points: int = 4000,
    sigma: float = 1.0,
) -> None:
    display_surface, display_confidence = _subsample_for_display(
        surface_points, surface_confidence, max_display_points, seed=0
    )
    colors = confidence_to_colors(display_confidence, colormap=colormap)

    geometries: List[o3d.geometry.Geometry] = []
    surface_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(display_surface))
    surface_pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(surface_pcd)

    if show_partial and partial_points is not None and partial_points.size > 0:
        partial = _subsample_for_display(
            partial_points,
            np.ones(partial_points.shape[0]),
            max_display_points,
            seed=1,
        )[0]
        partial_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(partial))
        partial_pcd.paint_uniform_color([0.60, 0.63, 0.67])
        geometries.append(partial_pcd)

    colorbar_fig = _show_colorbar_panel(colormap=colormap)
    window_title = (
        f"{title} | σ={sigma:.5f} | mean surface conf={surface_confidence.mean():.3f}"
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1600, height=900)
    for geometry in geometries:
        vis.add_geometry(geometry)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = 2.8
    opt.show_coordinate_frame = True
    opt.light_on = False

    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()
    plt.close(colorbar_fig)


def print_confidence_summary(
    observed_fitting_error: np.ndarray,
    observed_confidence: np.ndarray,
    surface_confidence: np.ndarray,
    sigma: float,
) -> None:
    print("Observed-point fitting (used for transfer):")
    print(f"  sigma: {sigma:.6f}")
    print(
        f"  F  min/median/mean/max: "
        f"{observed_fitting_error.min():.6f} / {np.median(observed_fitting_error):.6f} / "
        f"{observed_fitting_error.mean():.6f} / {observed_fitting_error.max():.6f}"
    )
    print(
        f"  observed confidence min/median/mean/max: "
        f"{observed_confidence.min():.6f} / {np.median(observed_confidence):.6f} / "
        f"{observed_confidence.mean():.6f} / {observed_confidence.max():.6f}"
    )
    print("Predicted surface confidence:")
    print(
        f"  min/median/mean/max: "
        f"{surface_confidence.min():.6f} / {np.median(surface_confidence):.6f} / "
        f"{surface_confidence.mean():.6f} / {surface_confidence.max():.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DSQNet on one partial point cloud and visualize observation-support "
            "confidence on the predicted deformable superquadric surface."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=str, help="Path to .npy partial point cloud sample.")
    source.add_argument(
        "--object",
        type=str,
        help="Evaluation dataset object name (e.g. ellipsoid). Use with --index.",
    )

    parser.add_argument("--index", type=int, default=0, help="Evaluation sample index (0-9).")
    parser.add_argument("--viewpoint", type=int, default=None, help="Evaluation viewpoint id (1-14).")
    parser.add_argument("--segment-id", type=int, default=None, help="Optional GT segment id.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="DSQNet config YAML.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="DSQNet checkpoint.")
    parser.add_argument(
        "--device",
        default=0,
        help="Torch device: GPU index (e.g. 0), cuda:0, or cpu.",
    )
    parser.add_argument(
        "--num-fitting-points",
        type=int,
        default=300,
        help="Resampled point count passed to DSQNet (matches training).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Confidence scale σ. Default: inference normalizer.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for surface confidence coloring.",
    )
    parser.add_argument(
        "--max-display-points",
        type=int,
        default=4000,
        help="Maximum points used for rendering.",
    )
    parser.add_argument(
        "--fit-surface-points",
        type=int,
        default=3500,
        help="Poisson-disk sample count on the predicted superquadric surface.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=8,
        help="Nearest observed points used to transfer confidence onto each surface sample.",
    )
    parser.add_argument(
        "--show-partial",
        action="store_true",
        help="Overlay the observed partial cloud in neutral gray (surface stays colored).",
    )
    parser.add_argument(
        "--save-image",
        type=str,
        default=None,
        help="Optional: save a PNG figure.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip the interactive viewer.",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        default="open3d",
        choices=["matplotlib", "open3d"],
        help="Interactive viewer backend.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional JSON path exporting surface confidence arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        points, _, metadata = load_sample(args.input)
        label = Path(args.input).name
    else:
        if args.object is None:
            raise SystemExit("--object is required when --input is not provided.")
        points, label, _, metadata = load_evaluation_segment(
            args.object,
            args.index,
            viewpoint=args.viewpoint,
            segment_id=args.segment_id,
        )

    device = resolve_device(args.device)
    if str(args.device) != "cpu" and device == "cpu":
        print("CUDA unavailable; falling back to CPU.")

    model, _ = load_recognition_model(args.config, args.checkpoint, device=device)
    result = recognize_deformable_superquadric(
        points,
        model,
        num_fitting_points=args.num_fitting_points,
    )

    observed = compute_local_confidence(
        result["input_points"],
        result["SE3"],
        result["parameters"],
        sigma=args.sigma,
        normalizer=float(result["normalizer"]),
    )
    fitting_error = observed["fitting_error"]
    observed_confidence = observed["confidence"]
    sigma = observed["sigma"]

    surface_points = _sample_fit_surface_points(
        result["SE3"],
        result["parameters"],
        num_points=args.fit_surface_points,
    )
    surface_confidence = compute_surface_confidence(
        surface_points,
        result["input_points"],
        observed_confidence,
        sigma,
        k_neighbors=args.k_neighbors,
    )

    print(f"Sample: {label}")
    print(f"Input points: {result['input_points'].shape[0]}")
    print(f"Surface samples: {surface_points.shape[0]}")
    if metadata:
        print(f"Profile: {metadata.get('profile', 'unknown')}")
    print_confidence_summary(fitting_error, observed_confidence, surface_confidence, sigma)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "sample": label,
            "sigma": sigma,
            "k_neighbors": args.k_neighbors,
            "observed_fitting_error": fitting_error.tolist(),
            "observed_confidence": observed_confidence.tolist(),
            "surface_points": surface_points.tolist(),
            "surface_confidence": surface_confidence.tolist(),
            "summary": {
                "observed_confidence_mean": float(observed_confidence.mean()),
                "surface_confidence_min": float(surface_confidence.min()),
                "surface_confidence_median": float(np.median(surface_confidence)),
                "surface_confidence_mean": float(surface_confidence.mean()),
                "surface_confidence_max": float(surface_confidence.max()),
            },
            "parameters": result["parameters"],
            "SE3": np.asarray(result["SE3"]).tolist(),
        }
        if metadata:
            payload["generation_metadata"] = metadata
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {output_path}")

    should_visualize = not args.no_visualize
    if args.save_image:
        save_confidence_figure(
            surface_points=surface_points,
            surface_confidence=surface_confidence,
            partial_points=result["input_points"],
            label=label,
            sigma=sigma,
            save_path=Path(args.save_image),
            colormap=args.colormap,
            show_partial=args.show_partial,
            max_display_points=args.max_display_points,
        )

    if should_visualize:
        if args.viewer == "open3d":
            show_confidence_open3d(
                surface_points=surface_points,
                surface_confidence=surface_confidence,
                partial_points=result["input_points"],
                title=f"DSQNet surface confidence: {label}",
                colormap=args.colormap,
                show_partial=args.show_partial,
                max_display_points=args.max_display_points,
                sigma=sigma,
            )
        else:
            show_confidence_interactive(
                surface_points=surface_points,
                surface_confidence=surface_confidence,
                partial_points=result["input_points"],
                label=label,
                sigma=sigma,
                colormap=args.colormap,
                show_partial=args.show_partial,
                max_display_points=args.max_display_points,
            )


if __name__ == "__main__":
    main()
