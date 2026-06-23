#!/usr/bin/env python3
"""Merge two primitive datasets into one folder with renumbered supplement instances."""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from functions.data_generator import save_data

SHAPE_NAMES = [
    "box",
    "cone",
    "cylinder",
    "ellipsoid",
    "truncated_cone",
    "truncated_torus",
]

FILENAME_RE = re.compile(
    r"^(?P<shape>[a-z_]+)_(?P<instance>\d{4})_viewpoint_(?P<view>\d{2})\.npy$"
)


def _shape_dirs(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    shapes = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    return shapes


def _instance_ids(shape_dir):
    ids = set()
    for path in Path(shape_dir).glob("*.npy"):
        match = FILENAME_RE.match(path.name)
        if match:
            ids.add(int(match.group("instance")))
    return ids


def _count_instances(shape_dir):
    return len(_instance_ids(shape_dir))


def _max_instance_id(shape_dir):
    ids = _instance_ids(shape_dir)
    return max(ids) if ids else -1


def _transfer_file(src, dst, use_symlink):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing file: {dst}")
    if use_symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def _link_shape_folder(base_shape_dir, out_shape_dir, use_symlink):
    out_shape_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(base_shape_dir.glob("*.npy")):
        dst = out_shape_dir / src.name
        _transfer_file(src, dst, use_symlink)


def _link_shape_folder_renumbered(supp_shape_dir, out_shape_dir, shape, offset, use_symlink):
    out_shape_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(supp_shape_dir.glob("*.npy")):
        match = FILENAME_RE.match(src.name)
        if not match or match.group("shape") != shape:
            continue
        old_id = int(match.group("instance"))
        new_id = old_id + offset
        dst_name = f"{shape}_{new_id:04d}_viewpoint_{match.group('view')}.npy"
        dst = out_shape_dir / dst_name
        _transfer_file(src, dst, use_symlink)


def summarize_dataset(root, label):
    root = Path(root)
    print(f"\n{label}: {root}")
    total_instances = 0
    for shape in sorted(_shape_dirs(root)):
        shape_dir = root / shape
        count = _count_instances(shape_dir)
        max_id = _max_instance_id(shape_dir)
        total_instances += count
        print(f"  {shape}: {count} instances (max_id={max_id:04d})")
    print(f"  total instances: {total_instances}")
    return total_instances


def merge_datasets(base, supplement, output, use_symlink=True, force=False):
    base = Path(base)
    supplement = Path(supplement)
    output = Path(output)

    if output.exists():
        if not force:
            existing = any(output.iterdir()) if output.is_dir() else True
            if existing:
                raise FileExistsError(
                    f"Output directory already exists: {output}. "
                    "Use --force to remove and recreate it."
                )
        if output.is_dir():
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    base_shapes = set(_shape_dirs(base))
    supp_shapes = set(_shape_dirs(supplement)) if supplement.exists() else set()
    all_shapes = base_shapes | supp_shapes

    print("Merging datasets...")
    for shape in sorted(all_shapes):
        out_shape_dir = output / shape
        if shape in base_shapes:
            base_shape_dir = base / shape
            _link_shape_folder(base_shape_dir, out_shape_dir, use_symlink)

            if shape in supp_shapes:
                offset = _max_instance_id(base_shape_dir) + 1
                _link_shape_folder_renumbered(
                    supplement / shape,
                    out_shape_dir,
                    shape,
                    offset,
                    use_symlink,
                )
                print(
                    f"  {shape}: base -> output, supplement renumbered with offset +{offset}"
                )
            else:
                print(f"  {shape}: base -> output only (no supplement folder)")
        else:
            _link_shape_folder(supplement / shape, out_shape_dir, use_symlink)
            print(f"  {shape}: supplement-only -> output (no renumbering)")

    save_data(
        default_path=str(output),
        shuffle=False,
        train_test_ratio=0.9,
        train_val_ratio=0.9,
    )
    print(f"\nWrote train/validation/test CSVs under {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge base and supplement primitive datasets with instance renumbering."
    )
    parser.add_argument(
        "--base",
        default="datasets/primitive_dataset",
        help="Existing primitive dataset directory.",
    )
    parser.add_argument(
        "--supplement",
        default="datasets/primitive_dataset_supplement",
        help="Supplement primitive dataset directory.",
    )
    parser.add_argument(
        "--output",
        default="datasets/primitive_dataset_merged",
        help="Merged output directory.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks (default: symlink).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing output directory before merging.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Print per-shape instance counts and exit without merging.",
    )
    args = parser.parse_args()

    base = Path(args.base)
    supplement = Path(args.supplement)
    output = Path(args.output)

    if args.verify_only:
        summarize_dataset(base, "Base")
        if supplement.exists():
            summarize_dataset(supplement, "Supplement")
        if output.exists():
            summarize_dataset(output, "Merged output")
        return

    if not supplement.exists():
        raise FileNotFoundError(
            f"Supplement dataset not found: {supplement}. "
            "Generate it first with:\n"
            "  python data_generation.py --config primitive "
            "--savedir datasets/ --name primitive_dataset_supplement --num-objects 100"
        )

    summarize_dataset(base, "Base")
    summarize_dataset(supplement, "Supplement")

    merge_datasets(
        base,
        supplement,
        output,
        use_symlink=not args.copy,
        force=args.force,
    )

    summarize_dataset(output, "Merged output")


if __name__ == "__main__":
    main()
