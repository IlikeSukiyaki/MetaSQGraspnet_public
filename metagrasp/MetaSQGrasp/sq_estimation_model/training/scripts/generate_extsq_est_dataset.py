#!/usr/bin/env python3
"""Generate or convert an ExtSQ estimation dataset with MetaSQGrasp manifests."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

CLASS_ID = "extsq_est_001"
SEMANTIC_TYPE = "extended_sq"
FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<instance>\d{4})_viewpoint_(?P<viewpoint>\d{2})\.npy$"
)
CSV_SPLITS = {
    "train": "train_datalist.csv",
    "validation": "validation_datalist.csv",
    "test": "test_datalist.csv",
}
REPO_ROOT = Path(__file__).resolve().parents[5]


def parse_filename(filename: str) -> tuple[str, str, str]:
    match = FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Unexpected dataset filename: {filename}")
    return match.group("prefix"), match.group("instance"), match.group("viewpoint")


def copy_sample(src_root: Path, dst_root: Path, filename: str) -> dict:
    prefix, instance, viewpoint = parse_filename(filename)
    if prefix != SEMANTIC_TYPE:
        raise ValueError(f"Expected {SEMANTIC_TYPE!r} sample, got {filename!r}")

    new_filename = f"{CLASS_ID}_{instance}_viewpoint_{viewpoint}.npy"
    rel_path = Path(CLASS_ID) / new_filename
    src_path = src_root / SEMANTIC_TYPE / filename
    dst_path = dst_root / rel_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return {
        "path": rel_path.as_posix(),
        "class_id": CLASS_ID,
        "source_type": SEMANTIC_TYPE,
        "instance": instance,
        "viewpoint": viewpoint,
    }


def build_manifest(src_dataset: Path, dst_dataset: Path, *, overwrite: bool) -> dict:
    if dst_dataset.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_dataset} exists; pass --overwrite to replace it.")
        shutil.rmtree(dst_dataset)
    dst_dataset.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": dst_dataset.name,
        "class_map": "extsq_est_class_map.json",
        "splits": {"train": [], "validation": [], "test": []},
    }
    for split, csv_name in CSV_SPLITS.items():
        csv_path = src_dataset / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split list: {csv_path}")
        with csv_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                filename = raw_line.strip().split(",")[0]
                if not filename:
                    continue
                _, _, viewpoint = parse_filename(filename)
                if viewpoint in {"00", "15"}:
                    continue
                manifest["splits"][split].append(copy_sample(src_dataset, dst_dataset, filename))

    write_json(dst_dataset / "extsq_est_class_map.json", {CLASS_ID: SEMANTIC_TYPE})
    write_json(dst_dataset / "dataset_manifest.json", manifest)
    for metadata_name in ("instance_manifest.csv", "generation_config.json"):
        metadata_src = src_dataset / metadata_name
        if metadata_src.exists():
            shutil.copy2(metadata_src, dst_dataset / metadata_name)
    return manifest


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def render_source_dataset(args: argparse.Namespace) -> Path:
    source_root = args.source_generator_root.resolve()
    entrypoint = source_root / "data_generation.py"
    if not entrypoint.exists():
        raise FileNotFoundError(f"Missing source generator entrypoint: {entrypoint}")

    raw_parent = args.raw_output_parent.resolve()
    raw_parent.mkdir(parents=True, exist_ok=True)
    raw_name = args.raw_name
    cmd = [
        sys.executable,
        str(entrypoint),
        "--config",
        "extended_sq",
        "--savedir",
        str(raw_parent),
        "--name",
        raw_name,
        "--num-objects",
        str(args.num_objects),
    ]
    subprocess.run(cmd, cwd=str(source_root), check=True)
    return raw_parent / raw_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a MetaSQGrasp ExtSQ estimation dataset with JSON manifests."
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=None,
        help="Existing generated extended_sq dataset root to convert.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Run the renderer before writing the MetaSQGrasp manifest dataset.",
    )
    parser.add_argument(
        "--source-generator-root",
        type=Path,
        default=REPO_ROOT / "rgbsq_grasp",
        help="Root containing the existing synthetic renderer.",
    )
    parser.add_argument(
        "--raw-output-parent",
        type=Path,
        default=Path("datasets/raw_extsq_est"),
        help="Parent directory for raw renderer output when --render is used.",
    )
    parser.add_argument("--raw-name", default="extended_sq_dataset")
    parser.add_argument("--num-objects", type=int, default=1000)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "extsq_est_dataset",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.render:
        source_dataset = render_source_dataset(args)
    elif args.source_dataset is not None:
        source_dataset = args.source_dataset
    else:
        raise ValueError("Pass either --render or --source-dataset.")

    manifest = build_manifest(source_dataset.resolve(), args.target.resolve(), overwrite=args.overwrite)
    split_counts = {split: len(entries) for split, entries in manifest["splits"].items()}
    print(f"ExtSQ dataset ready: {args.target.resolve()}")
    print(f"Split counts: {split_counts}")


if __name__ == "__main__":
    main()
