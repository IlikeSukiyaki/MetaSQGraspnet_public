#!/usr/bin/env python
"""Seed the curated UR + Hand-E grasp database from the full IFL asset root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_sampler.database_paths import (  # noqa: E402
    legacy_models_ifl_roots,
    normalize_obj_id,
    resolve_grasp_database_root,
)
from grasp_sampler.gripper_profile_defaults import HAND_E_HDF5_SUFFIX  # noqa: E402


GENERATED_NAMES = {"orbit_obj.usd", "textured.obj.hdf5", f"textured.obj{HAND_E_HDF5_SUFFIX}.hdf5"}
GENERATED_SUFFIXES = {".usd", ".hdf5"}


def _default_source() -> Path:
    for root in legacy_models_ifl_roots():
        if root.exists():
            return root
    return legacy_models_ifl_roots()[0]


def _parse_objects(value: str) -> list[str]:
    objects = [normalize_obj_id(item) for item in value.replace(",", " ").split()]
    if not objects:
        raise argparse.ArgumentTypeError("At least one object ID is required.")
    return objects


def _is_generated(path: Path) -> bool:
    return path.name in GENERATED_NAMES or path.suffix in GENERATED_SUFFIXES


def _copy_or_link_file(src: Path, dst: Path, *, symlink: bool, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return "exists"
        dst.unlink()

    if symlink:
        dst.symlink_to(src.resolve())
        return "symlinked"

    shutil.copy2(src, dst)
    return "copied"


def seed_object(source_root: Path, dest_root: Path, obj_id: str, *, symlink: bool, overwrite: bool) -> dict:
    source_dir = source_root / obj_id
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source object directory: {source_dir}")

    dest_dir = dest_root / obj_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for src in sorted(source_dir.rglob("*")):
        if not src.is_file() or _is_generated(src):
            continue
        relative = src.relative_to(source_dir)
        dst = dest_dir / relative
        status = _copy_or_link_file(src, dst, symlink=symlink, overwrite=overwrite)
        records.append(
            {
                "source": str(src),
                "target": str(dst),
                "status": status,
            }
        )

    manifest = {
        "object_id": obj_id,
        "source_dir": str(source_dir),
        "dest_dir": str(dest_dir),
        "mode": "symlink" if symlink else "copy",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    manifest_path = dest_dir / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=_default_source())
    parser.add_argument("--dest", type=Path, default=resolve_grasp_database_root())
    parser.add_argument("--objects", type=_parse_objects, default=_parse_objects("001,004,010,016"))
    parser.add_argument("--symlink", action="store_true", help="Symlink asset files instead of copying them.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing copied/symlinked asset files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source.expanduser().resolve()
    dest_root = args.dest.expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source asset root does not exist: {source_root}")

    dest_root.mkdir(parents=True, exist_ok=True)
    for obj_id in args.objects:
        manifest = seed_object(
            source_root,
            dest_root,
            obj_id,
            symlink=args.symlink,
            overwrite=args.overwrite,
        )
        print(
            f"[Info] Seeded {obj_id}: {len(manifest['files'])} files "
            f"from {manifest['source_dir']} -> {manifest['dest_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
