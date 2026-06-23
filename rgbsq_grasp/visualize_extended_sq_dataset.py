#!/usr/bin/env python3
"""Quick matplotlib subplot viewer for extended_sq_dataset samples."""

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_manifest(dataset_dir):
    path = Path(dataset_dir) / "instance_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return pd.read_csv(path)


def pick_instances(manifest, num_samples, indices, profiles, seed):
    if indices:
        ids = [int(x) for x in indices.split(",")]
        rows = manifest[manifest["instance_id"].isin(ids)]
        if len(rows) == 0:
            raise ValueError(f"No manifest rows for instance_id in {ids}")
        return rows

    if profiles:
        wanted = [p.strip() for p in profiles.split(",")]
        rows = []
        for profile in wanted:
            match = manifest[manifest["profile"] == profile]
            if match.empty:
                raise ValueError(f"No instances with profile '{profile}'")
            rows.append(match.sample(1, random_state=seed))
        return pd.concat(rows, ignore_index=True)

    n = min(num_samples, len(manifest))
    return manifest.sample(n, random_state=seed)


def load_sample(dataset_dir, instance_id, viewpoint):
    rel = f"extended_sq/extended_sq_{instance_id:04d}_viewpoint_{viewpoint:02d}.npy"
    path = Path(dataset_dir) / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True).item()


def subsample_points(points, max_points, seed):
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def set_equal_axes(ax, points_list):
    stacked = np.concatenate(points_list, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-3)
    for axis, c in zip("xyz", center):
        getattr(ax, f"set_{axis}lim")(c - radius, c + radius)


def plot_sample(ax, partial_pc, full_pc, title):
    partial = subsample_points(partial_pc, max_points=400, seed=0)
    full = subsample_points(full_pc[:, :3], max_points=400, seed=1)

    ax.scatter(
        partial[:, 0],
        partial[:, 1],
        partial[:, 2],
        s=1,
        c="#e74c3c",
        alpha=0.7,
        label="partial",
    )
    ax.scatter(
        full[:, 0],
        full[:, 1],
        full[:, 2],
        s=1,
        c="#3498db",
        alpha=0.35,
        label="full GT",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_equal_axes(ax, [partial, full])
    ax.view_init(elev=20, azim=45)


def build_title(row, viewpoint):
    return (
        f"id={row.instance_id:04d}  profile={row.profile}\n"
        f"view={viewpoint:02d}  "
        f"a=({row.a1:.3f},{row.a2:.3f},{row.a3:.3f})  "
        f"e=({row.e1:.2f},{row.e2:.2f})  k={row.k:.2f}  b={row.b:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize extended_sq_dataset samples in matplotlib subplots."
    )
    parser.add_argument(
        "--dataset",
        default="datasets/extended_sq_dataset",
        help="Path to extended_sq_dataset folder.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=4,
        help="Number of random instances (ignored if --indices or --profiles set).",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma-separated instance ids, e.g. 0,12,27,999.",
    )
    parser.add_argument(
        "--profiles",
        default=None,
        help="Comma-separated profiles; picks one instance each "
        "(compact,elongated,flattened,high_deformation).",
    )
    parser.add_argument(
        "--viewpoint",
        type=int,
        default=None,
        help="Fixed viewpoint index 0-15. Default: random per sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Subplot columns.",
    )
    parser.add_argument(
        "--output",
        default="extended_sq_preview.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    manifest = load_manifest(dataset_dir)
    rows = pick_instances(manifest, args.num, args.indices, args.profiles, args.seed)

    n = len(rows)
    cols = max(1, args.cols)
    rows_n = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows_n))
    fig.suptitle("Extended SQ dataset: partial (red) vs full GT (blue)", fontsize=12)

    rng = random.Random(args.seed)
    for i, row in enumerate(rows.itertuples(index=False)):
        instance_id = int(row.instance_id)
        viewpoint = args.viewpoint
        if viewpoint is None:
            viewpoint = rng.randint(0, 15)

        data = load_sample(dataset_dir, instance_id, viewpoint)
        ax = fig.add_subplot(rows_n, cols, i + 1, projection="3d")
        plot_sample(
            ax,
            data["partial_pc"],
            data["full_pc"],
            build_title(row, viewpoint),
        )
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, markerscale=4)

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved preview to {output_path.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
