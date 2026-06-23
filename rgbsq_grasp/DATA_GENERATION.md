# Synthetic Data Generation

This document describes the synthetic dataset engine in `rgbsq_grasp`: how partial point clouds are rendered from parametric shapes, how datasets are organized on disk, and how generated data connects to DSQNet training.

Run all commands from the `rgbsq_grasp` directory:

```bash
cd /path/to/DSQNet/rgbsq_grasp
conda activate rgbsq_grasp
```

## Overview

The generation pipeline samples random shape parameters, builds a colored mesh, renders 16 camera viewpoints, and saves partial point clouds with ground-truth primitive parameters.

```text
data_generation.py
    └── functions/data_generator.py
            └── functions/generate_instance_worker.py  (one process per object instance)
                    ├── object_params/{name}.json
                    ├── functions/object_class.py
                    ├── functions/primitives.py
                    └── functions/utils_datagen.py  (Open3D depth rendering)
```

Each **object instance** gets its own worker subprocess. This isolates Open3D/Filament crashes and keeps memory usage bounded on long runs.

After generation finishes, `save_data()` writes `train_datalist.csv`, `validation_datalist.csv`, and `test_datalist.csv` under the dataset folder.

## Quick start

```bash
# Primitive shapes (boxes, cylinders, ellipsoids, …) for SQNet / DSQNet recognition
python data_generation.py --config primitive --savedir datasets/ --name primitive_dataset

# Multi-part objects with segmentation membership labels
python data_generation.py --config object --savedir datasets/ --name object_dataset

# Extended / deformable superquadrics for DSQNet recognition
python data_generation.py --config extended_sq --savedir datasets/ --name extended_sq_dataset
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | *(required)* | `primitive`, `object`, or `extended_sq` |
| `--savedir` | `datasets/` | Parent directory for output |
| `--name` | timestamp `YYYYMMDD-HHMM` | Dataset folder name |
| `--num-objects` | 100 (`primitive`/`object`), 1000 (`extended_sq`) | Instances per object type |

Smoke test (2 instances):

```bash
python data_generation.py --config extended_sq --savedir datasets/ --name extended_sq_smoke --num-objects 2
```

## Config modes

### `primitive`

Single-primitive shapes for superquadric / deformable superquadric **recognition** training.

| Setting | Value |
|---------|-------|
| Object types | `box`, `cylinder`, `ellipsoid`, `cone`, `truncated_cone`, `truncated_torus` |
| Points per partial PC | 1500 |
| Save full PC | yes (required by `PrimitiveDataset`) |
| Save membership | no |
| Default instances | 100 per type |

Parameter ranges live in `object_params/{type}.json`.

### `object`

Multi-part composed objects for **segmentation** training.

| Setting | Value |
|---------|-------|
| Object types | primitives above plus `bottle_cone`, `screw_driver`, `cup_with_lid`, `hammer_cylinder`, `padlock`, `dumbbell` |
| Points per partial PC | 3000 |
| Save full PC | no |
| Save membership | yes (per-primitive color labels) |
| Default instances | 100 per type |

### `extended_sq`

Single **extended superquadrics** (deformable superquadrics with taper and bend) for DSQNet recognition training with broad shape coverage.

| Setting | Value |
|---------|-------|
| Object type | `extended_sq` |
| Points per partial PC | 1500 |
| Save full PC | yes |
| Save membership | no |
| Default instances | 1000 |

Parameter sampling is defined in [`object_params/extended_sq.json`](object_params/extended_sq.json).

**Sampling profiles** (weighted random):

| Profile | Weight | Description |
|---------|--------|-------------|
| `compact` | 40% | Similar axis lengths, ratio ≤ 1.8 |
| `elongated` | 30% | One long axis, ratio up to 3.5 |
| `flattened` | 20% | One short axis, ratio up to 3.5 |
| `high_deformation` | 10% | Strong taper `k` and bend `b` |

**Parameter ranges:**

| Parameter | Range | Notes |
|-----------|-------|-------|
| `a1`, `a2`, `a3` | 0.03 – 0.14 m | `a1 ≤ a2 ≤ a3`, max ratio 3.5 |
| `e1`, `e2` | 0.25 – 1.6 | Shape exponents |
| `k` | −0.65 – 0.65 | Taper |
| `b` | 0.01 – 0.55 | Bend magnitude |
| `alpha` | 0 – 2π | Bend direction (`cos_alpha`, `sin_alpha`) |

Train with [`pretrained/dsqnet_config/example/dsqnet_extended_sq_config.yml`](pretrained/dsqnet_config/example/dsqnet_extended_sq_config.yml).

## Dataset layout

### Primitive / object datasets

```text
datasets/primitive_dataset/
├── box/
│   ├── box_0000_viewpoint_00.npy
│   └── ...
├── cylinder/
├── ...
├── train_datalist.csv
├── validation_datalist.csv
└── test_datalist.csv
```

Filename pattern: `{object_type}_{instance:04d}_viewpoint_{view:02d}.npy`

The loader parses the object type as everything before the last three `_`-separated segments (`instance`, `viewpoint`, `view_id`). Do not embed parameter values in filenames.

### Extended SQ dataset

```text
datasets/extended_sq_dataset/
├── generation_config.json      # global ranges and generation settings
├── instance_manifest.csv       # one row per object instance
├── train_datalist.csv
├── validation_datalist.csv
├── test_datalist.csv
└── extended_sq/
    ├── extended_sq_0000_viewpoint_00.npy
    ├── extended_sq_0000_viewpoint_01.npy
    └── ...
```

**Inspecting parameters without opening every `.npy`:**

```bash
# Per-instance summary (profile + all 9 shape parameters)
cat datasets/extended_sq_dataset/instance_manifest.csv

# Intended global sampling ranges
cat datasets/extended_sq_dataset/generation_config.json
```

Each `.npy` also stores `generation_metadata` with `object_name`, `instance_id`, and `profile`.

## Sample file format (`.npy`)

Each file is a Python dict saved with `numpy.save`:

| Key | Shape / type | When present |
|-----|--------------|--------------|
| `partial_pc` | `(N, 3)` | always |
| `full_pc` | `(N, 6)` points + normals | `primitive`, `extended_sq` |
| `primitives` | list of dicts | always |
| `membership` | `(N,)` int labels | `object` config |
| `generation_metadata` | dict | `extended_sq` |

Each primitive dict:

```python
{
    "type": "extended_sq",       # or "box", "superquadric", …
    "SE3": (4, 4) ndarray,
    "parameters": { ... }        # type-specific keys
}
```

Extended SQ parameters: `a1`, `a2`, `a3`, `e1`, `e2`, `k`, `b`, `cos_alpha`, `sin_alpha`.

## Training integration

Generated data is consumed by loaders in `loader/`:

| Config mode | Loader | Typical model |
|-------------|--------|---------------|
| `primitive` | `PrimitiveDataset` | SQNet, DSQNet recognition |
| `extended_sq` | `PrimitiveDataset` | DSQNet recognition |
| `object` | `ObjectDataset` | Segmentation network |

`PrimitiveDataset` filters datalists by `object_types` and excludes top/bottom viewpoints (`00` and `15`) by default.

Example training config for extended SQ:

```yaml
data:
  training:
    loader: primitive
    path: datasets/extended_sq_dataset
    csv_name: train_datalist.csv
    object_types:
      - extended_sq
    num_points: 1500
    num_gt_points: 1500
```

Legacy datasets generated with object name `deformable_superquadric` still load if you set `object_types: [deformable_superquadric]`.

## Customization

### Object parameters

Edit or add JSON files under `object_params/`. Each file supports:

- `"random": true` with `"min"` / `"max"` bounds per parameter, or
- Fixed numeric values when `"random": false`.

Register new composed objects in `functions/object_class.py` (`load_object` dict).

### Generation settings

Defaults are set in `data_generation.py` (`num_cams`, split ratios, point counts). Per-dataset overrides use CLI flags (`--num-objects`, `--name`, `--savedir`).

### Resume / append

Re-running the same `--name` with `append=True` (default) skips completed instances and fills gaps. Incomplete instances (fewer than 14 of 16 viewpoints after discards) are regenerated from the first missing view.

## Merging primitive datasets (SQNet co-training)

To double SQNet training data, generate a **supplement** dataset with the same `primitive` mode, then merge it with the existing `primitive_dataset` into one folder. Supplement instances are renumbered so IDs never collide.

### 1. Generate supplement (100 instances per shape)

```bash
python data_generation.py \
  --config primitive \
  --savedir datasets/ \
  --name primitive_dataset_supplement \
  --num-objects 100
```

### 2. Merge base + supplement

```bash
python scripts/merge_primitive_datasets.py \
  --base datasets/primitive_dataset \
  --supplement datasets/primitive_dataset_supplement \
  --output datasets/primitive_dataset_merged \
  --force
```

By default, merged files are **symlinks** back to the source datasets (use `--copy` to duplicate files). The script calls `save_data()` with the same split ratios as generation (`train_test_ratio=0.9`, `train_val_ratio=0.9`).

**Renumbering rule:** supplement instance `0000` becomes `max_base_id + 1` per shape. Example: base `box_0099` + supplement `box_0000` → merged `box_0100`.

**Expected merged instance counts** (given default 100-instance supplement and current base):

| Shape | Base | Supplement | Merged |
|-------|------|------------|--------|
| box, cone, cylinder, ellipsoid, truncated_cone | 100 | 100 | 200 |
| truncated_torus | 300 | 100 | 400 |

Verify counts without merging:

```bash
python scripts/merge_primitive_datasets.py --verify-only \
  --base datasets/primitive_dataset \
  --supplement datasets/primitive_dataset_supplement \
  --output datasets/primitive_dataset_merged
```

### 3. Train SQNet on merged data

```bash
python train.py --config configs/sqnet_config_merged.yml --run sqnet_merged
# or with TensorBoard logging:
python train_tensorboard.py --config configs/sqnet_config_merged.yml --run sqnet_merged_tb
```

The original `configs/sqnet_config.yml` and `datasets/primitive_dataset` remain unchanged.

## Rendering notes

- Default renderer: Open3D legacy `Visualizer` (needs a display).
- Headless / server: set `DSQNET_RENDERER=offscreen` for EGL/Filament offscreen rendering. See [Open3D headless rendering](http://www.open3d.org/docs/latest/tutorial/Advanced/headless_rendering.html).
- Worker retries up to 5 times on intermittent GPU/EGL failures.
- Inline generation (no subprocess isolation): set `DSQNET_INLINE_GENERATION=1`.

## Key source files

| File | Role |
|------|------|
| `data_generation.py` | CLI entry point |
| `functions/data_generator.py` | Orchestration, datalist splits, resume logic |
| `functions/generate_instance_worker.py` | Per-instance rendering and save |
| `functions/object_class.py` | Object samplers (`ExtendedSQ_Object`, …) |
| `functions/primitives.py` | Mesh builders (superquadric, deformable SQ, …) |
| `functions/utils_datagen.py` | Camera poses, depth capture, frame alignment |
| `functions/dataset_metadata.py` | `generation_config.json` and `instance_manifest.csv` |
| `object_params/*.json` | Sampling ranges per object type |
| `loader/primitive_dataset.py` | Training data loader for primitive / extended SQ |

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| Worker exit non-zero after retries | Upgrade Open3D (≥ 0.18); try `DSQNET_RENDERER=offscreen` |
| Empty or partial point cloud | View discarded if only a plane is visible; normal for some poses |
| Wrong point count error | Renderer zoom loop failed; check mesh size / GPU |
| Loader finds 0 samples | Check `object_types` matches folder prefix (`extended_sq`, not `deformable_superquadric`) |
| Cannot infer parameter range from filename | Use `instance_manifest.csv` or read `primitives[0]['parameters']` from any viewpoint of the same instance |
