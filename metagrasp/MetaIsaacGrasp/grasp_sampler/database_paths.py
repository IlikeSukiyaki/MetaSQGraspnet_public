"""Shared path helpers for the curated UR + Hand-E grasp database."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_obj_id(obj_id: str | int) -> str:
    obj_id = str(obj_id).strip()
    if not obj_id:
        raise ValueError("Object ID cannot be empty.")
    return obj_id.zfill(3) if obj_id.isdigit() else obj_id


def resolve_grasp_database_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    return repo_root() / "UR_grasp_database"


def object_dir(obj_id: str | int, root: str | Path | None = None) -> Path:
    return resolve_grasp_database_root(root) / normalize_obj_id(obj_id)


def legacy_models_ifl_roots() -> list[Path]:
    root = repo_root()
    return [
        root / "models" / "models_ifl",
        root.parents[1] / "models_ifl",
    ]
