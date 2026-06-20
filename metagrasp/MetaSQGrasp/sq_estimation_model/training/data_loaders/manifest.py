import json
import os
from pathlib import Path


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    dataset_root = manifest_path.parent
    class_map_path = manifest.get("class_map")
    class_map = {}
    if class_map_path:
        with (dataset_root / class_map_path).open("r", encoding="utf-8") as handle:
            class_map = json.load(handle)
    return manifest, dataset_root, class_map


def get_split_entries(manifest, split):
    splits = manifest.get("splits", {})
    if split not in splits:
        raise KeyError(f"Split {split!r} not found in manifest splits: {list(splits.keys())}")
    return splits[split]


def filter_entries(entries, class_ids=None, exclude_viewpoints=None):
    exclude_viewpoints = exclude_viewpoints or {"00", "15"}
    filtered = []
    for entry in entries:
        if class_ids and entry.get("class_id") not in class_ids:
            continue
        viewpoint = entry.get("viewpoint")
        if viewpoint is None:
            filename = Path(entry["path"]).name
            viewpoint = filename.split("_")[-1].split(".")[0]
        if viewpoint in exclude_viewpoints:
            continue
        filtered.append(entry)
    return filtered
