#!/usr/bin/env python
"""Convert a full SQ-v0-Data episode folder to the lite export bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


LITE_KEEP = {
    "meta.json",
    "rgb.png",
    "depth.png",
    "instance.png",
}

LITE_REMOVE = {
    "scene.pt",
    "depth.npy",
    "instance.npy",
    "normals.npy",
}


def prune_sample(sample_dir: Path, output_dir: Path | None, save_scene_pc: bool) -> Path:
    sample_dir = sample_dir.resolve()
    if output_dir is None:
        dest = sample_dir
    else:
        dest = output_dir.resolve()
        if dest != sample_dir:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(sample_dir, dest)

    for path in dest.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in LITE_REMOVE:
            path.unlink(missing_ok=True)
            continue
        if name == "pcd_map.npy" and not save_scene_pc:
            path.unlink(missing_ok=True)

    meta_path = dest / "meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as file:
            meta = json.load(file)
        meta["export"] = {
            "profile": "lite",
            "save_scene_pc": bool(save_scene_pc),
        }
        with meta_path.open("w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)

    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Full SQ-v0-Data episode directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination episode directory (default: prune in place).",
    )
    parser.add_argument(
        "--save_scene_pc",
        action="store_true",
        help="Keep per-camera pcd_map.npy in the lite bundle.",
    )
    args = parser.parse_args()
    dest = prune_sample(args.input, args.output, args.save_scene_pc)
    print(f"Lite sample ready: {dest}")


if __name__ == "__main__":
    main()
