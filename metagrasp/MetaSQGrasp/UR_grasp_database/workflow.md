# UR + Hand-E Grasp Database Workflow

Extended reference for the curated asset library `UR_grasp_database/` used by `object-v0-Classical`: seed from upstream IFL meshes, generate Hand-E grasps, convert URDF to USD, and verify in Isaac Sim.

Quick start: [`README.md`](README.md)

Related guides:

- [`../grasp_sampler/sampling.md`](../grasp_sampler/sampling.md) — generation and visualization CLI
- [`../grasp_sampler/pose-logistics.md`](../grasp_sampler/pose-logistics.md) — coordinate frames and runtime filtering
- [`../grasp_sampler/retargeting.md`](../grasp_sampler/retargeting.md) — sampler → `hande_end` retargeting and verification

---

## 1. Directory purpose

| Path | Role |
|------|------|
| `../../models_ifl/` | Full IFL object corpus (upstream read-only source) |
| `UR_grasp_database/` | Curated UR10e + Hand-E test subset |

**Verified objects (2026-06):** `000`, `001`, `004`, `011`.

**Historical default pool:** `001`, `004`, `010`, `016`. At runtime, pass any prepared object ID via `--obj_ids`.

### Gripper width configuration (single entry point)

Edit [`../grasp_sampler/gripper_profile_defaults.py`](../grasp_sampler/gripper_profile_defaults.py):

| Constant | Value | Meaning |
|----------|-------|---------|
| `HAND_E_WIDTH_MAX_CM` | **7.0** | Max jaw width for sampling and runtime (cm) |
| `HAND_E_HDF5_SUFFIX` | `.hand_e` | HDF5 filename suffix |
| `HAND_E_PREGRASP_EXTRA_CM` | `1.0` | Pre-grasp clearance |

Generation commands read profile width automatically — no need to pass `--gripper_length 5.0`.

> Note: URDF nominal max opening is 5 cm, but Isaac finger joint targets (0.0425 m per finger) allow wider effective grasps. This project uses **7 cm** as the sampling/filtering upper bound to match current sim behavior.

Each object directory should contain:

```text
UR_grasp_database/<id>/
├── textured.obj              # mesh for sampling
├── textured.urdf             # URDF source for USD conversion
├── textured.mtl              # materials (and textures/ sidecars)
├── asset_manifest.json       # copy record from setup tool
├── textured.obj.hand_e.hdf5  # Hand-E grasp library (generated)
└── orbit_obj.usd             # Isaac rigid-body asset (converted)
```

---

## 2. Environment setup

| Step | Conda env | Purpose |
|------|-----------|---------|
| Copy / generate HDF5 / visualize | `metagrasp` | Offline antipodal sampling |
| URDF → USD / Isaac runs | `metagrasp` | Isaac Lab **0.47.2** + Isaac Sim **5.1.0** (see [`../install.md`](../install.md)) |

Run all commands from the MetaSQGrasp root:

```bash
cd metagrasp/MetaSQGrasp
```

---

## 3. Full pipeline (recommended order)

### Step 0 — Seed curated library from IFL

Copy mesh / URDF / materials from full `models_ifl` into `UR_grasp_database`, **skipping** generated files (`textured.obj.hand_e.hdf5`, `orbit_obj.usd`).

```bash
conda activate metagrasp

python tools/setup_ur_grasp_database.py \
  --source ../../models_ifl \
  --objects 000,001,004,011 \
  --overwrite
```

Optional symlink to save disk:

```bash
python tools/setup_ur_grasp_database.py \
  --source ../../models_ifl \
  --objects 000,001,004,011 \
  --symlink --overwrite
```

### Step 1 — Generate Hand-E grasp HDF5

```bash
conda activate metagrasp

python tools/generate_object_grasps.py \
  --mesh_root UR_grasp_database \
  --categories 000 001 004 011 \
  --gripper_profile hand_e \
  --num_points 500 \
  --max_grasps 2000 \
  --pool_size 1
```

Output: `UR_grasp_database/<id>/textured.obj.hand_e.hdf5`

### Step 2 — Verify Hand-E frame mapping

```bash
conda activate metagrasp
python tools/check_hande_frame.py
```

Expected:

```text
[OK] Hand-E frame check passed: baseline->EE+X, approach->EE+Y.
```

### Step 3 — URDF to physics USD

Requires `metagrasp` (Isaac Lab 0.47 + Isaac Sim 5.1). Run once per object:

```bash
conda activate metagrasp

python urdf_converter.py --input UR_grasp_database/000/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/001/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/004/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/011/textured.urdf --headless
```

Output: `orbit_obj.usd` in the same directory.

### Step 4 — Optional: offline grasp visualization

```bash
conda activate metagrasp

python grasp_sampler/visualize_grasp_object.py \
  --root UR_grasp_database \
  --object 011 \
  --gripper_profile hand_e \
  --frame ee \
  --max_grasps 50 \
  --score_min 0.3
```

Default `--frame ee` shows the execution-equivalent `hande_end` frame. Use `--frame sampler` for the raw sampler frame.

### Step 5 — Optional: quick runtime grasp selection check

```bash
conda activate metagrasp

python - <<'PY'
import sys, numpy as np
sys.path.insert(0, ".")
from grasp_sampler.runtime import resolve_mesh_root, select_best_grasp

root = resolve_mesh_root()
print("mesh_root =", root)
pose = np.eye(4)
pose[:3, 3] = [75.0, 20.0, 8.0]  # cm, approximate classical table pose

for obj_id in ["000", "001", "004", "011"]:
    sel = select_best_grasp(
        obj_id=obj_id,
        object_pose_cm=pose,
        desk_center_m=(0.75, 0.2, 0.0),
        mesh_root=str(root),
        profile="hand_e",
    )
    if sel is None:
        print(f"{obj_id}: NONE")
    else:
        print(
            f"{obj_id}: score={sel.score:.3f} "
            f"candidates={sel.candidate_count} width={sel.width_cm:.2f} cm"
        )
PY
```

### Step 6 — Isaac simulation smoke test

```bash
conda activate metagrasp

# Single object
python run.py --task object-v0-Classical \
  --obj_ids 011 \
  --num_envs 1 --headless --enable_cameras --saved_video \
  --max_successful_grasps 1

# Multiple objects
python run.py --task object-v0-Classical \
  --obj_ids 000,001,004,011 \
  --num_envs 1 --headless --enable_cameras --saved_video \
  --max_successful_grasps 4
```

On success, expect logs like:

```text
[Classical Grasp] env=0 episode=0 attempt=0 obj=011 profile=hand_e score=... align=... valid_candidates=... width_cm=...
```

Videos: `logs/videos/object-v0-Classical/<timestamp>/`.

---

## 4. Adding a new test object

1. Place the object in upstream `models_ifl/<id>/` (or seed from there).
2. Run Step 0 with `<id>` in `--objects`.
3. Run Steps 1–3 to produce `textured.obj.hand_e.hdf5` and `orbit_obj.usd`.
4. Pass `--obj_ids` at runtime (`0` and `000` are equivalent; IDs are zero-padded).

```bash
python run.py --task object-v0-Classical --obj_ids 004 --num_envs 1 --enable_cameras
```

---

## 5. Code layout (classical task modules)

| Path | Role |
|------|------|
| `grasp_sampler/gripper_profile_defaults.py` | Hand-E / 2F85 width and HDF5 suffix (**single global config**) |
| `isaac_env/gripper_task_configs/hand_e.py` | Hand-E task constants (profile, EE axes, camera) |
| `isaac_env/classical_shared/` | Shared env cfg factory, sim base, scene config |
| `isaac_env/object_env_classical/` | Thin wrapper for `object-v0-Classical` |
| `isaac_env/object_env_classical_2f85/` | `object-v1-Classical-2f85` (Robotiq 2F85, experimental) |

Hand-E frame fix summary:

| Component | Behavior |
|-----------|----------|
| `HandEProfile.gripper_pose_cm_to_ee_pose()` | baseline → EE +X, approach → EE +Y |
| `object_env_classical/wp_cfg.py` | Pre-approach retreats along `CLASSICAL_APPROACH_AXIS_EE` (default EE +Y) |
| `visualize_grasp_object.py` | `--frame ee` shows execution-equivalent frame by default |

---

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Missing antipodal HDF5` | Re-run Step 1; confirm `textured.obj.hand_e.hdf5` exists |
| `Missing USD assets` | Run Step 3 or confirm `orbit_obj.usd` exists |
| Legacy `textured.obj.hdf5` only | Regenerate with `--gripper_profile hand_e` |
| No grasps for wide object (e.g. `016`) | Object too wide; try 2F85 task or another object |
| Execution still ~90° off | Run `tools/check_hande_frame.py` + Step 6 |
| `urdf_converter` fails | See [`../install.md`](../install.md); use pre-converted `orbit_obj.usd` |

---

## 7. Related files

| File | Role |
|------|------|
| `grasp_sampler/gripper_profile_defaults.py` | Gripper width / HDF5 suffix config |
| `grasp_sampler/database_paths.py` | `UR_grasp_database` path resolution |
| `tools/setup_ur_grasp_database.py` | Seed from `models_ifl` |
| `tools/generate_object_grasps.py` | Generate HDF5 |
| `tools/check_hande_frame.py` | Hand-E frame unit check |
| `grasp_sampler/gripper_profiles.py` | Hand-E / 2F85 profiles and EE conversion |
| `isaac_env/classical_shared/` | Shared classical Isaac config |
| `isaac_env/object_env_classical/` | Hand-E classical task |
| `command.txt` | Copy-paste command snippets |
