# SQ-v0-Data (`supersq_grasp_data`)

Isaac Lab environment package for **SQ estimation synthetic data collection**. It extends the AIR grasping stack with a dual-camera setup, randomized object poses, and structured dataset export for downstream grasp-pose synthesis.

Registered Gym task: **`SQ-v0-Data`**

Related guides: [`datasets/README.md`](../../datasets/README.md), [`tools/README.md`](../../tools/README.md), [`multi-object-collection.md`](multi-object-collection.md), [`../../tools/partial-pc-extraction.md`](../../tools/partial-pc-extraction.md)

---

## Package overview

```
supersq_grasp_data/
├── __init__.py      # Gym registration
├── element_cfg.py   # Scene constants, object/camera defaults
├── env_cfg.py       # Scene, observations, events, MDP config
├── env.py           # SQEnvData runtime environment
├── sim.py           # CLI-facing simulation loop wrapper
├── export.py        # Dataset snapshot writer
├── wp_cfg.py        # Warp state-machine kernel re-export
└── README.md        # This file
```

### Data flow

```mermaid
flowchart LR
    A[run.py CLI] --> B[sim.SQPickSmData]
    B --> C[env.SQEnvData]
    C --> D[env_cfg observations]
    C --> E[export.save_scene_snapshot]
    E --> F[datasets/supersq_grasp_data/...]
    F --> G[tools/extract_partial_pc.py]
    F --> H[tools/convert_sq_to_npy.py]
    H --> I[SQ estimation datasets]
```

---

## File reference

### `__init__.py`

- Registers the Gym environment `SQ-v0-Data`
- Entry point: `SQEnvData`
- Config entry point: `CellEnvCfg`
- Exports `SQPickSmData`, `SQEnvData`

### `element_cfg.py`

Scene-level constants inherited from `env_base.element_cfg`, narrowed for SQ estimation smoke tests.

| Item | Value / role |
|------|----------------|
| `DEMO_OBJ_IDS` | `["000", "001", "004", "011"]` in `grasp_sampler/database_paths.py` |
| `NUM_OBJS` | len(`DEMO_OBJ_IDS`) by default |
| `OBJ_LABLE` | same as `DEMO_OBJ_IDS` |
| `N_MULTIPLE_CAM` | `2` (on-hand + top-down) |
| `MODEL_PATH` | `UR_grasp_database/` (fallback: legacy `models/models_ifl`) |
| `MGN_PATHS` | USD assets per object ID |
| `SQ_DATA_ROOT` | Default output root: `datasets/supersq_grasp_data` |

**Top-down camera tuning** (edit here):

```python
TOPDOWN_CAMERA_XY = (0.75, 0.75)   # world-frame x, y
TOPDOWN_CAMERA_HEIGHT = 1.8          # meters above desk_center[2]
TOPDOWN_CAMERA_QUAT = (0.0, 1.0, 0.0, 0.0)
```

- `make_topdown_camera_cfg(height)` builds a world-fixed `camera_1` prim.
- `TOPDOWN_CAMERA_ROLE_NAMES` maps camera indices to export folder names:
  - `0` → `camera_onhand`
  - `1` → `camera_topdown`

Objects are spawned with `semantic_tags` for instance segmentation.

**Prerequisite:** physics-ready USD at `UR_grasp_database/<id>/orbit_obj.usd` for each object. Convert from URDF via:

```bash
python urdf_converter.py --input UR_grasp_database/004/textured.urdf --headless
```

Override the object pool at runtime with `--obj_ids 000,001,004,011` (same IDs as `supersq_grasp-v0` and `object-v0-Classical`).

### `env_cfg.py`

Defines the full Isaac Lab configuration for SQ data collection.

#### Scene (`CellSceneCfg`)

Reuses AIR robot/table/lighting setup and adds:

- `camera_0` — on-hand camera on the UR10e + Hand-E
- `camera_1` — top-down camera from `make_topdown_camera_cfg()`
- `objs` — single rigid object collection (`obj_0`)
- `ee_frame`, `view_frame` — end-effector / camera frame transformers

#### Observations (`ObservationsCfg`)

Dual-camera capture functions (stacked over `N_MULTIPLE_CAM`):

| Term | Function | Output |
|------|----------|--------|
| `rgb` | `sq_rgb_capture` | RGB image per camera |
| `normals` | `sq_normal_capture` | Surface normals |
| `instance_segmentation_fast` | `sq_inst_capture` | Per-pixel instance IDs |
| `distance_to_image_plane` | `sq_depth_capture` | Depth (clipped to `DEPTH_MAX`) |
| `pcd` | `sq_pcd_capture` | H×W×3 point map in robot base (meters) |
| `gripper_pose` | inherited | Gripper state |

`sq_pcd_capture` back-projects depth with camera intrinsics and pose. Returns a zero map if point cloud size is unexpected (e.g. during init probing).

#### Events (`EventCfg`)

- `reset_robot_pos` — reset arm to default
- `reset_obj_pos` — **`sq_reset_scene_objects`**: each episode places **K objects** on the table (`--scene_objects`, default 1), picks one as the target (`env.obj_chosen`), and moves unselected pool objects off-screen

#### MDP (`CellEnvCfg`)

- Inherits rewards/terminations from AIR base
- Episode length: 5 s, decimation 2, sim dt 1/120

### `env.py` — `SQEnvData`

Runtime environment class extending `EnvBase`.

**Constructor kwargs:**

| Arg | Default | Purpose |
|-----|---------|---------|
| `save_camera_data` | `False` | Enable dataset export |
| `output_dir` | `SQ_DATA_ROOT` | Dataset root directory |
| `topdown_height` | `TOPDOWN_CAMERA_HEIGHT` | Top-down camera height override |
| `export_profile` | `"full"` | `"full"` or `"lite"` export bundle |
| `save_scene_pc` | `False` | Keep `pcd_map.npy` in lite mode |
| `scene_objects` | `1` | Number of objects placed on the table per episode |

**Key behavior:**

1. Maintains dual-camera handles (`camera_0`, `camera_1`).
2. Uses a **data-collection state machine** (`infer_state_machine_data` from `wp_cfg.py`) instead of the grasp-demo machine.
3. On `get_action()` when `save_camera_data=True`:
   - Holds the robot at current EE pose
   - Calls `save_scene_snapshot()` once per `(env_id, episode)`
   - Tracks saved samples via `num_saved_samples`

### `sim.py` — `SQPickSmData`

CLI-facing wrapper used by `run.py`.

**Responsibilities:**

- Parse env config for `SQ-v0-Data`
- Apply `--topdown_height` to `camera_1`
- Apply `--obj_ids` to `configure_data_objects()` before env creation
- Pass `--output_dir`, `--save_camera_data`, `--topdown_height`, `--export_profile`, `--save_scene_pc`, `--scene_objects` into `gym.make()`
- `propose_action()` — clones full-env `grasp_pose`, updates only inference envs (avoids empty-action crash)
- `should_stop()` — stops when `num_saved_samples >= --max_episodes`

### `export.py`

Writes one episode snapshot when the scene is stable.

#### Per-episode layout

**Full profile** (`--export_profile full`, default):

```
<output_dir>/001/epi_000123/
├── meta.json
├── scene.pt
├── camera_onhand/
│   ├── rgb.png, depth.png, instance.png
│   ├── depth.npy, instance.npy, normals.npy, pcd_map.npy
├── camera_topdown/
│   └── (same structure)
└── partial_pc/
    ├── camera_onhand_auto_instance.npy
    └── camera_topdown_auto_instance.npy
```

**Lite profile** (`--export_profile lite`):

```
<output_dir>/001/epi_000123/
├── meta.json
├── camera_onhand/
│   ├── rgb.png, depth.png, instance.png
│   └── pcd_map.npy          # only with --save_scene_pc
├── camera_topdown/
│   └── (same structure)
└── partial_pc/
    └── …
```

Curated demo samples: [`datasets/supersq_grasp_data/demo/`](../../datasets/supersq_grasp_data/demo/README.md).

#### `save_scene_snapshot(env, env_id, obs_buf, output_dir)`

1. Saves per-camera tensors and preview PNGs
2. Builds `scene.pt` (torch bundle with all camera tensors + object poses)
3. Writes `meta.json` (camera intrinsics, poses, frame conventions, id-to-label maps)
4. Auto-exports object partial point clouds via `tools/sq_partial_pc.py`
5. Appends a row to `<output_dir>/manifest.jsonl`

#### Coordinate frames (`meta.json → frames`)

| Field | Frame |
|-------|-------|
| `camera_pose` | Robot base |
| `pcd_map` | Robot base, meters |
| `obj_poses_robot` | Robot base, translation in **cm** |

### `wp_cfg.py`

Re-exports `infer_state_machine_data` from `env_data.wp_cfg` — the Warp GPU kernel that drives the data-collection state machine (reach stable pose → hold → advance episode).

---

## How to run

From `MetaSQGrasp/`:

```bash
conda activate metagrasp

python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras --headless \
  --save_camera_data \
  --export_profile lite \
  --obj_ids 000,001,004,011 \
  --output_dir datasets/supersq_grasp_data/demo \
  --max_episodes 4 \
  --topdown_height 2.1
```

Or collect the full demo set in one script:

```bash
bash tools/collect_demo_dataset.sh datasets/supersq_grasp_data/demo
```

Multi-object scenes (2–4 objects on table, random target + distractors):

```bash
# Lite export: smaller bundles, auto-saved target partial_pc
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras --headless \
  --save_camera_data --export_profile lite \
  --obj_ids 000,001,004,011 \
  --scene_objects 2 \
  --output_dir datasets/supersq_grasp_data/multi_obj \
  --max_episodes 20

# Full export: per-object partial PC re-extraction + visualization
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras --headless \
  --save_camera_data --export_profile full \
  --obj_ids 000,001,004,011 \
  --scene_objects 4 \
  --output_dir datasets/supersq_grasp_data/multi_obj_full \
  --max_episodes 1
```

See [`multi-object-collection.md`](multi-object-collection.md) for the full workflow, `meta.json` fields, and `tools/visualize_partial_pc.py` usage.

### CLI flags (via `run.py`)

| Flag | Description |
|------|-------------|
| `--save_camera_data` | Enable snapshot export |
| `--output_dir` | Dataset root (default: `datasets/supersq_grasp_data`) |
| `--max_episodes` | Stop after N saved samples |
| `--obj_ids` | Comma-separated object IDs (default: `DEMO_OBJ_IDS`) |
| `--scene_objects` | Objects on the table per episode (default: `1`; max: pool size) |
| `--export_profile` | `full` (all tensors + scene.pt) or `lite` (meta, PNGs, partial_pc) |
| `--save_scene_pc` | Keep `pcd_map.npy` per camera in lite mode |
| `--topdown_height` | Override top-down camera Z offset (meters above table) |
| `--enable_cameras` | Required for camera rendering |
| `--num_envs` | Parallel environments (typically `1` for data collection) |

Top-down XY position is configured in `element_cfg.py` (`TOPDOWN_CAMERA_XY`), not via CLI.

---

## Downstream tools

These live outside `supersq_grasp_data/` but consume its output:

| Tool | Purpose |
|------|---------|
| `tools/collect_demo_dataset.sh` | Collect lite demo episodes for all verified object IDs |
| `tools/prune_sq_sample_lite.py` | Convert a full episode folder to lite export |
| `tools/extract_partial_pc.py` | Extract object-only partial point clouds (`--mask`, `--color`) |
| `tools/convert_sq_to_npy.py` | Convert raw samples to estimator `.npy` format |
| `tools/inspect_sq_sample.py` | Inspect depth/seg/pcd of one sample |
| `tools/visualize_partial_pc.py` | Open3D viewer for target or all scene-object partial PCs |

See also: [`multi-object-collection.md`](multi-object-collection.md), [`../../tools/partial-pc-extraction.md`](../../tools/partial-pc-extraction.md)

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`README.md`](README.md) | This package reference |
| [`multi-object-collection.md`](multi-object-collection.md) | Cluttered-scene export workflow |

Related: [`../../datasets/README.md`](../../datasets/README.md), [`../../tools/partial-pc-extraction.md`](../../tools/partial-pc-extraction.md)

---

## Design notes

- **Inherits shared base** — robot, table, on-hand camera, controller, and most MDP terms come from `env_base` / `env_data`.
- **Multi-object pool** — default `DEMO_OBJ_IDS`; override with `--obj_ids`; `--scene_objects K` places K distinct objects on the table per episode (one is the grasp target, others are distractors for occlusion testing).
- **Dual-camera observations** — all image/depth/seg/pcd tensors are shaped `(num_envs, num_cameras, ...)`.
- **Degenerate instance masks** — current single-object data may label the full image as one instance ID; partial PC export uses mesh-bbox fallback (`mask=auto`). Multi-object scenes (`--scene_objects > 1`) rely on per-object instance labels and do not use mesh-bbox fallback.
- **No RL training here** — this package is for synthetic dataset generation, not policy learning.

---

## Quick troubleshooting

| Issue | Check |
|-------|-------|
| `FileNotFoundError` for USD | Run `urdf_converter.py` on `UR_grasp_database/<id>/textured.urdf` |
| Empty / wrong top-down view | Tune `TOPDOWN_CAMERA_XY` and `TOPDOWN_CAMERA_HEIGHT` in `element_cfg.py` |
| No files saved | Pass `--save_camera_data`; wait until state machine reaches stable snapshot state |
| Collection won't stop | Set `--max_episodes N` |
| Partial PC covers whole scene | Use `tools/extract_partial_pc.py --mask mesh` or `--mask auto` |
