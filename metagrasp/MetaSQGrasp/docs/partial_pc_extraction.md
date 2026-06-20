# SQ Partial Point Cloud Extraction

This guide explains how to extract **object-only** partial point clouds from SQ-v0-Data samples.

Tool: `tools/extract_partial_pc.py`

All extracted points are in the **robot base frame** (meters), stored in `partial_pc`.

---

## Quick start

```bash
cd metagrasp/MetaSQGrasp
conda activate metagrasp

python tools/extract_partial_pc.py \
  --sample datasets/supersq_grasp_data/001/epi_000001 \
  --camera topdown \
  --mask auto \
  --color instance \
  --show
```

---

## CLI reference

### Input scope (required, pick one)

| Flag | Description |
|------|-------------|
| `--sample PATH` | One episode folder, e.g. `datasets/supersq_grasp_data/001/epi_000001` |
| `--input PATH` | Object folder with many episodes, e.g. `datasets/supersq_grasp_data/001` |

### Camera

| Flag | Values | Default |
|------|--------|---------|
| `--camera` | `onhand`, `topdown`, `both` | `topdown` |

### Mask mode (`--mask`)

Controls **which pixels** are turned into 3D points.

| Value | Meaning | When to use |
|-------|---------|-------------|
| `auto` | Use instance segmentation; if the mask covers almost the whole image, fall back to mesh bounding box | **Recommended default** |
| `instance` | Instance segmentation only | When segmentation cleanly separates object vs background |
| `mesh` | Crop by object mesh bbox using `obj_poses_robot` + `models_ifl/<id>/textured.obj` | Current single-object data where instance mask is degenerate |

### Color mode (`--color`)

Controls **how points are colored** in `.npy`, `.ply`, and the Open3D viewer.

| Value | Meaning |
|-------|---------|
| `instance` | JET colormap from `instance.npy` (same style as `instance.png`) |
| `rgb` | Camera RGB from `rgb.png` at the selected pixels |
| `none` | No colors saved |

### Other flags

| Flag | Description |
|------|-------------|
| `--output PATH` | Custom output directory (default: `<sample>/partial_pc/`) |
| `--model_root PATH` | Mesh root containing `<object_id>/textured.obj` |
| `--margin FLOAT` | Mesh bbox padding in meters (default: `0.015`) |
| `--ply` | Also write a colored `.ply` file |
| `--show` | Open Open3D viewer after extraction |

---

## Output files

Default naming:

```
<sample>/partial_pc/
  camera_topdown_auto_instance.npy
  camera_topdown_auto_instance.ply      # with --ply
  camera_onhand_mesh_rgb.npy            # example with --mask mesh --color rgb
```

Each `.npy` dict contains:

| Key | Description |
|-----|-------------|
| `partial_pc` | `(N, 3)` float32 object points |
| `normals` | `(N, 3)` if available |
| `colors` | `(N, 3)` uint8 if `--color` is not `none` |
| `instance_ids` | `(N,)` if `--color instance` |
| `mask_mode` | CLI mask mode (`auto` / `instance` / `mesh`) |
| `mask_method` | Actual method used (`instance_label`, `mesh_bbox`, ...) |
| `color_mode` | CLI color mode |
| `color_source` | `instance_segmentation` or `rgb` |
| `camera_pose_7d`, `intrinsics_3x3` | Camera metadata |

---

## Common recipes

### 1. Visualize object cloud with instance-seg colors (recommended)

```bash
python tools/extract_partial_pc.py \
  --sample datasets/supersq_grasp_data/001/epi_000001 \
  --camera topdown \
  --mask auto \
  --color instance \
  --show
```

### 2. Force mesh bbox mask + camera RGB colors

```bash
python tools/extract_partial_pc.py \
  --sample datasets/supersq_grasp_data/001/epi_000001 \
  --camera topdown \
  --mask mesh \
  --color rgb \
  --ply --show
```

### 3. Batch both cameras for all episodes

```bash
python tools/extract_partial_pc.py \
  --input datasets/supersq_grasp_data/001 \
  --camera both \
  --mask auto \
  --color instance
```

### 4. Convert to rgbsq_grasp training format

```bash
python tools/convert_sq_to_npy.py \
  --input datasets/supersq_grasp_data/001 \
  --output ../../rgbsq_grasp/datasets/isaac_synth/001 \
  --camera topdown \
  --mask auto
```

---

## Auto-save during data collection

When running SQ-v0-Data with `--save_camera_data`, each new episode also writes:

```
<epi>/partial_pc/camera_onhand_auto_instance.npy
<epi>/partial_pc/camera_topdown_auto_instance.npy
```

Metadata is recorded in `meta.json` under `partial_pc`.

---

## Python API

```python
from pathlib import Path
from sq_partial_pc import extract_object_partial_pc

payload = extract_object_partial_pc(
    Path("datasets/supersq_grasp_data/001/epi_000001"),
    "topdown",              # or "camera_topdown"
    mask_mode="auto",       # auto | instance | mesh
    color_mode="instance",  # instance | rgb | none
)
points = payload["partial_pc"]
colors = payload.get("colors")
```

---

## Choosing the right mask mode

| Situation | Use |
|-----------|-----|
| Current SQ single-object data (instance mask covers full image) | `--mask auto` or `--mask mesh` |
| Future data with proper background/object instance IDs | `--mask instance` or `--mask auto` |
| Multi-object scenes (`meta.json` → `scene_objects > 1`) | `--mask instance` or `--mask auto` (never `--mask mesh`) |
| Need reproducible object crop independent of segmentation quality | `--mask mesh` (single-object only) |

### Multi-object occlusion scenes

When collecting with `--scene_objects 2` (or higher), distractor objects can occlude the target in depth and instance segmentation. Extraction uses the **target object instance label** only (`meta["object_id"]` / `meta["target_object_id"]`), so occluded target regions are naturally absent from `partial_pc`.

Saved payloads also include:

| Key | Description |
|-----|-------------|
| `occlusion_mode` | `single_object` or `multi_object` |
| `scene_object_ids` | All object IDs on the table |
| `distractor_ids` | Non-target scene objects |
| `visible_target_fraction` | Target instance pixels / valid depth pixels |

`mask=auto` skips the mesh-bbox fallback on multi-object samples because occluder points can fall inside the target mesh bounding box.

---

## Troubleshooting

**`mask=instance found zero object pixels`**

The instance label for your object is missing. Use `--mask auto` or `--mask mesh` (single-object only).

**`mask=mesh is unsafe for multi-object scenes`**

Re-collect with `--scene_objects 1`, or re-extract with `--mask instance` or `--mask auto`.

**All points have the same instance color**

Your sample may only contain one instance ID (e.g. `1`). This is expected until segmentation separates background.

**`Could not find models_ifl`**

Pass `--model_root /path/to/models_ifl` when using `--mask mesh` or `--mask auto`.
