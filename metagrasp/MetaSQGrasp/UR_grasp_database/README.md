# UR + Hand-E Grasp Database

Curated object asset root for classical and SQ-based grasp tasks. Stores UR10e + Robotiq Hand-E test objects and their generated grasp assets.

Upstream source: full IFL corpus at [`../../models_ifl/`](../../models_ifl/) (MetaGraspNet meshes).

---

## Expected layout

```text
UR_grasp_database/
├── 000/
├── 001/
├── 004/
├── 011/
│   ├── textured.obj
│   ├── textured.urdf
│   ├── textured.obj.hand_e.hdf5
│   └── orbit_obj.usd
└── ...
```

**Verified smoke-test objects (2026-06):** `000`, `001`, `004`, `011`.

---

## Generated files

| File | Description |
|------|-------------|
| `textured.obj.hand_e.hdf5` | Hand-E antipodal grasp candidates (suffix and max jaw width from [`grasp_sampler/gripper_profile_defaults.py`](../grasp_sampler/gripper_profile_defaults.py)) |
| `orbit_obj.usd` | Isaac physics asset converted from `textured.urdf` |
| `asset_manifest.json` | Source and copy metadata from `tools/setup_ur_grasp_database.py` |

Legacy note: older runs may have plain `textured.obj.hdf5` (no suffix). Regenerate with `tools/generate_object_grasps.py --gripper_profile hand_e`.

---

## Full asset pipeline

Run from `metagrasp/MetaSQGrasp/` with `conda activate metagrasp`.

### 1. Seed from IFL corpus

```bash
python tools/setup_ur_grasp_database.py \
  --source ../../models_ifl \
  --objects 000,001,004,011 \
  --overwrite
```

Copies mesh, URDF, and texture sidecars. Skips generated outputs (`textured.obj.hand_e.hdf5`, `orbit_obj.usd`) so they can be rebuilt.

### 2. Generate Hand-E grasp HDF5

```bash
python tools/generate_object_grasps.py \
  --mesh_root UR_grasp_database \
  --categories 000 001 004 011 \
  --gripper_profile hand_e \
  --num_points 500 \
  --max_grasps 2000 \
  --pool_size 1
```

Gripper width and HDF5 suffix are read from [`grasp_sampler/gripper_profile_defaults.py`](../grasp_sampler/gripper_profile_defaults.py) — no need to pass `--gripper_length` manually.

### 3. Convert URDF to USD

```bash
python urdf_converter.py --input UR_grasp_database/000/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/001/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/004/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/011/textured.urdf --headless
```

Or batch: `bash tools/convert_models_ifl_to_usd.sh`

### 4. Verify classical task

```bash
python run.py --task object-v0-Classical \
  --obj_ids 011 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  --saved_video \
  --max_successful_grasps 1
```

Success videos: `logs/videos/object-v0-Classical/<timestamp>/`

---

## Gripper width configuration

Single entry point: [`grasp_sampler/gripper_profile_defaults.py`](../grasp_sampler/gripper_profile_defaults.py)

| Constant | Value | Meaning |
|----------|-------|---------|
| `HAND_E_WIDTH_MAX_CM` | 7.0 | Sampling and runtime max jaw width (cm) |
| `HAND_E_HDF5_SUFFIX` | `.hand_e` | HDF5 filename suffix |
| `HAND_E_PREGRASP_EXTRA_CM` | 1.0 | Pre-grasp clearance |

---

## Which tasks use this folder

| Task | Requires HDF5 | Requires USD |
|------|---------------|--------------|
| `object-v0-Classical` | Yes (`textured.obj.hand_e.hdf5`) | Yes |
| `object-v1-Classical-2f85` | Yes (2F85 suffix) | Yes |
| `SQ-v0-Data` | No | Yes |
| `supersq_grasp-v0` | No | Yes (+ `textured.obj`) |

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`README.md`](README.md) | This quick start |
| [`workflow.md`](workflow.md) | Extended asset pipeline reference |

Related: [`../grasp_sampler/sampling.md`](../grasp_sampler/sampling.md), [`../grasp_sampler/retargeting.md`](../grasp_sampler/retargeting.md), [`../tools/README.md`](../tools/README.md)
