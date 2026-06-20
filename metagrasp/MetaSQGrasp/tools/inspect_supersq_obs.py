#!/usr/bin/env python
"""Visualize depth and instance segmentation from one SuperSQ debug attempt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.supersq_debug_paths import (  # noqa: E402
    add_debug_dir_arguments,
    list_attempt_dirs,
    resolve_debug_dir_from_args,
    resolve_run_dir,
)

import numpy as np

PANEL_SPECS = (
    ("rgb", "rgb.png", "color"),
    ("depth", "depth.png", "gray"),
    ("instance_segmentation", "instance.png", "color"),
    ("object_mask", "object_mask.png", "gray"),
)


def _cv2():
    import cv2

    return cv2


def _load_meta(debug_dir: Path) -> dict:
    meta_path = debug_dir / "sq_fit.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing sq_fit.json in {debug_dir}")
    with meta_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_partial_payload(debug_dir: Path) -> dict | None:
    partial_path = debug_dir / "partial_pc.npy"
    if not partial_path.exists():
        return None
    payload = np.load(partial_path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        return payload.item()
    return None


def _depth_max_m(obs_dir: Path) -> float:
    meta_path = obs_dir / "obs_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as file:
            meta = json.load(file)
        return float(meta.get("depth_max_m", 2.0))
    return 2.0


def _export_pngs_from_arrays(obs_dir: Path, *, depth_max_m: float) -> dict[str, Path]:
    cv2 = _cv2()
    saved: dict[str, Path] = {}

    depth_path = obs_dir / "depth.npy"
    if depth_path.exists():
        depth = np.squeeze(np.load(depth_path))
        depth_png = np.clip(depth, 0.0, depth_max_m) / depth_max_m * 255.0
        out = obs_dir / "depth.png"
        cv2.imwrite(str(out), depth_png.astype(np.uint8))
        saved["depth"] = out

    instance_path = obs_dir / "instance.npy"
    if instance_path.exists():
        labels = np.squeeze(np.load(instance_path)).astype(np.uint8)
        out = obs_dir / "instance.png"
        cv2.imwrite(str(out), cv2.applyColorMap((labels * 40) % 255, cv2.COLORMAP_JET))
        saved["instance_segmentation"] = out

    rgb_path = obs_dir / "rgb.npy"
    if rgb_path.exists():
        rgb = np.asarray(np.load(rgb_path))[..., :3]
        out = obs_dir / "rgb.png"
        cv2.imwrite(str(out), cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
        saved["rgb"] = out

    mask_path = obs_dir / "object_mask.npy"
    if mask_path.exists():
        mask = np.asarray(np.load(mask_path), dtype=np.uint8) * 255
        out = obs_dir / "object_mask.png"
        cv2.imwrite(str(out), mask)
        saved["object_mask"] = out

    return saved


def _to_bgr_panel(image: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _annotate_panel(image: np.ndarray, title: str) -> np.ndarray:
    cv2 = _cv2()
    panel = _to_bgr_panel(image)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        panel,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _resize_panel(panel: np.ndarray, width: int, height: int) -> np.ndarray:
    return _cv2().resize(panel, (width, height), interpolation=_cv2().INTER_AREA)


def _save_montage(panels: dict[str, np.ndarray], obs_dir: Path) -> Path:
    if not panels:
        raise ValueError("No observation panels available to save.")

    ordered = [name for name, _, _ in PANEL_SPECS if name in panels]
    if not ordered:
        ordered = list(panels.keys())

    sample = _annotate_panel(panels[ordered[0]], ordered[0])
    panel_h, panel_w = sample.shape[:2]

    def _panel(name: str) -> np.ndarray:
        return _resize_panel(_annotate_panel(panels[name], name), panel_w, panel_h)

    if len(ordered) == 1:
        montage = _panel(ordered[0])
    elif len(ordered) == 2:
        montage = np.hstack([_panel(ordered[0]), _panel(ordered[1])])
    elif len(ordered) == 3:
        blank = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        row1 = np.hstack([_panel(ordered[0]), _panel(ordered[1])])
        row2 = np.hstack([_panel(ordered[2]), blank])
        montage = np.vstack([row1, row2])
    else:
        row1 = np.hstack([_panel(ordered[0]), _panel(ordered[1])])
        row2 = np.hstack([_panel(ordered[2]), _panel(ordered[3])])
        montage = np.vstack([row1, row2])

    out_path = obs_dir / "preview_montage.png"
    _cv2().imwrite(str(out_path), montage)
    return out_path


def _load_panels(obs_dir: Path, *, show_mask: bool) -> dict[str, np.ndarray]:
    cv2 = _cv2()
    panels: dict[str, np.ndarray] = {}
    for name, filename, mode in PANEL_SPECS:
        if name == "object_mask" and not show_mask:
            continue
        path = obs_dir / filename
        if not path.exists():
            continue
        imread_flag = cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_GRAYSCALE
        image = cv2.imread(str(path), imread_flag)
        if image is None:
            continue
        panels[name] = image
    return panels


def _show_panels(panels: dict[str, np.ndarray], wait_ms: int) -> None:
    cv2 = _cv2()
    for name, image in panels.items():
        cv2.imshow(name, image)
    if wait_ms == 0:
        cv2.waitKey(0)
    else:
        cv2.waitKey(wait_ms)
    cv2.destroyAllWindows()


def inspect_observation(
    debug_dir: Path,
    *,
    wait_ms: int = 0,
    show_mask: bool = True,
    save: bool = True,
    show: bool = True,
) -> dict[str, Path]:
    debug_dir = debug_dir.expanduser().resolve()
    meta = _load_meta(debug_dir)
    obs_dir = debug_dir / "observation"
    saved_paths: dict[str, Path] = {}

    print(f"Debug dir: {debug_dir}")
    print(f"Object: {meta.get('object_id')}")
    print(f"Camera index: {meta.get('camera_index')}")

    if obs_dir.exists():
        meta_path = obs_dir / "obs_meta.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as file:
                obs_meta = json.load(file)
            print(f"Camera role: {obs_meta.get('camera_role')}")
            print(f"ID to labels: {obs_meta.get('id_to_labels')}")

        if save:
            saved_paths.update(_export_pngs_from_arrays(obs_dir, depth_max_m=_depth_max_m(obs_dir)))

        panels = _load_panels(obs_dir, show_mask=show_mask)
        if not panels:
            raise FileNotFoundError(f"No observation PNGs found under {obs_dir}")

        if (obs_dir / "instance.npy").exists():
            print(f"Instance IDs: {np.unique(np.load(obs_dir / 'instance.npy')).tolist()}")
        if "depth" in panels:
            print(f"Depth shape: {panels['depth'].shape}")

        if save:
            for name, filename, _ in PANEL_SPECS:
                if name in panels:
                    saved_paths[name] = obs_dir / filename
            saved_paths["montage"] = _save_montage(panels, obs_dir)
            print("Saved observation previews:")
            for key, path in saved_paths.items():
                print(f"  {key}: {path}")

        if show:
            print("Showing:", ", ".join(panels.keys()))
            _show_panels(panels, wait_ms)
        return saved_paths

    payload = _load_partial_payload(debug_dir)
    if payload and "instance_mask" in payload:
        obs_dir.mkdir(parents=True, exist_ok=True)
        mask = np.asarray(payload["instance_mask"], dtype=np.uint8) * 255
        print(
            "This run predates full observation export. "
            "Recovered object_mask from partial_pc.npy."
        )
        print(f"Mask method: {payload.get('mask_method')}  points: {payload.get('num_points')}")
        if save:
            cv2 = _cv2()
            mask_path = obs_dir / "object_mask.png"
            cv2.imwrite(str(mask_path), mask)
            saved_paths["object_mask"] = mask_path
            print(f"Saved object_mask: {mask_path}")
        if show:
            cv2 = _cv2()
            cv2.imshow("object_mask_from_partial_pc", mask)
            if wait_ms == 0:
                cv2.waitKey(0)
            cv2.destroyAllWindows()
        print("Re-run with --save_sq_debug to export depth.png and instance.png.")
        return saved_paths

    raise FileNotFoundError(
        f"No observation artifacts under {obs_dir}. "
        "Re-run supersq_grasp-v0 with --save_sq_debug using the current codebase."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_debug_dir_arguments(parser)
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Process every attempt folder in --run (requires --run or --latest).",
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=0,
        help="OpenCV waitKey delay. Use 0 to wait for a key press.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Do not show or save the extracted object mask PNG.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Display only; do not write PNG previews.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save PNG previews without opening OpenCV windows.",
    )
    args = parser.parse_args()

    if args.list_runs or args.list_episodes:
        resolve_debug_dir_from_args(args)

    if args.all_episodes:
        if args.dir is not None:
            raise SystemExit("--all-episodes expects --run or --latest, not --dir.")
        if args.latest:
            from tools.supersq_debug_paths import latest_run_dir

            run_dir = latest_run_dir()
        elif args.run:
            run_dir = resolve_run_dir(args.run)
        else:
            raise SystemExit("--all-episodes requires --run or --latest.")
        attempt_dirs = list_attempt_dirs(run_dir)
        if not attempt_dirs:
            raise SystemExit(f"No attempt folders under {run_dir}")
        for attempt_dir in attempt_dirs:
            inspect_observation(
                attempt_dir,
                wait_ms=args.wait_ms,
                show_mask=not args.no_mask,
                save=not args.no_save,
                show=not args.no_show,
            )
        return

    debug_dir = resolve_debug_dir_from_args(args)
    inspect_observation(
        debug_dir,
        wait_ms=args.wait_ms,
        show_mask=not args.no_mask,
        save=not args.no_save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
