#!/usr/bin/env python3
"""Migrate rgbsq_grasp datasets into MetaSQGrasp sq_estimation_model/training/datasets."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

PRIMITIVE_TYPES = [
    "box",
    "cone",
    "cylinder",
    "ellipsoid",
    "truncated_cone",
    "truncated_torus",
]

OBJECT_TYPES = [
    "bottle_cone",
    "box",
    "cone",
    "cup_with_lid",
    "cylinder",
    "dumbbell",
    "ellipsoid",
    "hammer_cylinder",
    "padlock",
    "screw_driver",
    "truncated_cone",
    "truncated_torus",
]

FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<instance>\d{4})_viewpoint_(?P<viewpoint>\d{2})\.npy$"
)


def build_class_map(types: list[str], prefix: str) -> dict[str, str]:
    return {f"{prefix}_{idx:03d}": semantic for idx, semantic in enumerate(types, start=1)}


def invert_class_map(class_map: dict[str, str]) -> dict[str, str]:
    return {semantic: class_id for class_id, semantic in class_map.items()}


def parse_filename(filename: str):
    match = FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")
    return match.group("prefix"), match.group("instance"), match.group("viewpoint")


def should_include(viewpoint: str, exclude_viewpoints: set[str]) -> bool:
    return viewpoint not in exclude_viewpoints


def copy_sample(
    src_root: Path,
    dst_root: Path,
    semantic_type: str,
    class_id: str,
    filename: str,
) -> dict:
    prefix, instance, viewpoint = parse_filename(filename)
    if prefix != semantic_type:
        raise ValueError(f"Filename prefix {prefix!r} does not match folder type {semantic_type!r}")

    new_filename = f"{class_id}_{instance}_viewpoint_{viewpoint}.npy"
    rel_path = Path(class_id) / new_filename
    src_path = src_root / semantic_type / filename
    dst_path = dst_root / rel_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return {
        "path": rel_path.as_posix(),
        "class_id": class_id,
        "source_type": semantic_type,
        "instance": instance,
        "viewpoint": viewpoint,
    }


def migrate_csv_splits(
    src_root: Path,
    dst_root: Path,
    class_map: dict[str, str],
    csv_splits: dict[str, str],
    exclude_viewpoints: set[str],
) -> dict:
    semantic_to_class = invert_class_map(class_map)
    manifest = {
        "dataset": dst_root.name,
        "class_map": f"{dst_root.name.split('_')[0]}_class_map.json",
        "splits": {"train": [], "validation": [], "test": []},
    }

    for split, csv_name in csv_splits.items():
        csv_path = src_root / csv_name
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                filename = line.strip()
                if not filename:
                    continue
                semantic_type, _, viewpoint = parse_filename(filename)
                if semantic_type not in semantic_to_class:
                    continue
                if not should_include(viewpoint, exclude_viewpoints):
                    continue
                class_id = semantic_to_class[semantic_type]
                entry = copy_sample(src_root, dst_root, semantic_type, class_id, filename)
                manifest["splits"][split].append(entry)
    return manifest


def migrate_evaluation_dataset(src_root: Path, dst_root: Path, class_map: dict[str, str]) -> dict:
    semantic_to_class = invert_class_map(class_map)
    manifest = {
        "dataset": dst_root.name,
        "class_map": "evaluation_class_map.json",
        "splits": {"all": []},
    }

    for semantic_type in sorted(semantic_to_class):
        class_id = semantic_to_class[semantic_type]
        src_dir = src_root / semantic_type
        if not src_dir.exists():
            continue
        for src_path in sorted(src_dir.glob("*.npy")):
            entry = copy_sample(src_root, dst_root, semantic_type, class_id, src_path.name)
            manifest["splits"]["all"].append(entry)
    return manifest


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def migrate_all(source_datasets: Path, target_datasets: Path) -> None:
    exclude_viewpoints = {"00", "15"}

    primitive_map = build_class_map(PRIMITIVE_TYPES, "primitive")
    primitive_dst = target_datasets / "primitive_dataset"
    if primitive_dst.exists():
        shutil.rmtree(primitive_dst)
    primitive_manifest = migrate_csv_splits(
        source_datasets / "primitive_dataset",
        primitive_dst,
        primitive_map,
        {
            "train": "train_datalist.csv",
            "validation": "validation_datalist.csv",
            "test": "test_datalist.csv",
        },
        exclude_viewpoints,
    )
    write_json(primitive_dst / "primitive_class_map.json", primitive_map)
    write_json(primitive_dst / "dataset_manifest.json", primitive_manifest)

    object_map = build_class_map(OBJECT_TYPES, "object")
    object_dst = target_datasets / "object_dataset"
    if object_dst.exists():
        shutil.rmtree(object_dst)
    object_manifest = migrate_csv_splits(
        source_datasets / "object_dataset",
        object_dst,
        object_map,
        {
            "train": "train_datalist.csv",
            "validation": "validation_datalist.csv",
            "test": "test_datalist.csv",
        },
        exclude_viewpoints,
    )
    write_json(object_dst / "object_class_map.json", object_map)
    write_json(object_dst / "dataset_manifest.json", object_manifest)

    evaluation_map = build_class_map(OBJECT_TYPES, "evaluation")
    evaluation_dst = target_datasets / "evaluation_dataset"
    if evaluation_dst.exists():
        shutil.rmtree(evaluation_dst)
    evaluation_manifest = migrate_evaluation_dataset(
        source_datasets / "evaluation_dataset",
        evaluation_dst,
        evaluation_map,
    )
    write_json(evaluation_dst / "evaluation_class_map.json", evaluation_map)
    write_json(evaluation_dst / "dataset_manifest.json", evaluation_manifest)

    print("Migration complete:")
    for name, manifest in [
        ("primitive_dataset", primitive_manifest),
        ("object_dataset", object_manifest),
        ("evaluation_dataset", evaluation_manifest),
    ]:
        split_counts = {split: len(entries) for split, entries in manifest["splits"].items()}
        print(f"  {name}: {split_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate rgbsq_grasp datasets with renamed IDs.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/sustech/桌面/DSQNet/rgbsq_grasp/datasets"),
        help="Source datasets root (rgbsq_grasp/datasets).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets",
        help="Target datasets root under sq_estimation_model/training.",
    )
    args = parser.parse_args()
    migrate_all(args.source, args.target)


if __name__ == "__main__":
    main()
