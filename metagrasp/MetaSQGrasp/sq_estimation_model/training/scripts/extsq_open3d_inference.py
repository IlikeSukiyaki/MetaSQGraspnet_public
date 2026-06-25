#!/usr/bin/env python3
"""Single-sample ExtSQ estimation inference with Open3D visualization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from extsq_est_inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    REQUIRED_PARAM_KEYS,
    load_extsq_model,
    recognize_extsq,
)
from inference_open3d_viz import (
    load_sample,
    result_to_jsonable,
    visualize_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extsq_esti_model on one partial point cloud with Open3D visualization."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to .npy file containing N x 3 points or a dict with partial_pc.",
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="ExtSQ model config YAML.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="ExtSQ model checkpoint.")
    parser.add_argument(
        "--device",
        default=0,
        help="Torch device: GPU index (e.g. 0), cuda:0, or cpu.",
    )
    parser.add_argument(
        "--num-fitting-points",
        type=int,
        default=300,
        help="Resampled point count passed to the estimator.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON path for predictions.")
    parser.add_argument(
        "--show-gt",
        action="store_true",
        help="Overlay ground-truth primitive from .npy when available.",
    )
    parser.add_argument("--visualize", action="store_true", help="Open an Open3D viewer.")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization.")
    parser.add_argument(
        "--viz-mode",
        type=str,
        default="overlay",
        choices=["overlay", "side_by_side", "pred_only"],
        help="Visualization style.",
    )
    parser.add_argument(
        "--surface-style",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "mesh"],
        help="Render fitted surfaces as sampled points or triangle mesh.",
    )
    parser.add_argument(
        "--fit-surface-points",
        type=int,
        default=3500,
        help="Poisson-disk surface points when --surface-style point_cloud.",
    )
    parser.add_argument(
        "--save-image",
        type=str,
        default=None,
        help="Save a PNG without opening the viewer (headless rendering).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points, gt_primitive, metadata, label = load_sample(args.input)

    model, _ = load_extsq_model(args.config, args.checkpoint, device=args.device)
    result = recognize_extsq(points, model, num_fitting_points=args.num_fitting_points)

    print(f"Sample: {label}")
    print(f"Input points: {result['input_points'].shape[0]}")
    if metadata:
        print(f"Metadata keys: {sorted(metadata.keys())}")
    print("Predicted extended superquadric parameters:")
    for key in REQUIRED_PARAM_KEYS:
        print(f"  {key}: {result['parameters'][key]:.6f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = result_to_jsonable(result)
        if metadata:
            payload["metadata"] = metadata
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {output_path}")

    should_visualize = (args.visualize or args.save_image) and not args.no_visualize
    if should_visualize:
        gt_for_viz = gt_primitive if args.show_gt else None
        if args.show_gt and gt_primitive is None:
            print("[WARN] --show-gt requested but no ground-truth primitive found in input.")
        visualize_result(
            result,
            title=f"ExtSQ fit: {label}",
            gt_primitive=gt_for_viz,
            viz_mode=args.viz_mode,
            surface_style=args.surface_style,
            fit_surface_points=args.fit_surface_points,
            save_image=Path(args.save_image) if args.save_image else None,
            extended=True,
        )


if __name__ == "__main__":
    main()
