# `object-v0-Classical`

UR10e + Robotiq Hand-E pick-and-lift smoke test using offline antipodal grasps from `UR_grasp_database/<id>/textured.obj.hand_e.hdf5`.

---

## Prerequisites

Run these from the MetaSQGrasp root (`metagrasp/MetaSQGrasp/`).

Full asset pipeline: [`UR_grasp_database/README.md`](../../UR_grasp_database/README.md)

### 1. Seed object assets

Each object ID needs a folder under `UR_grasp_database/<id>/` with mesh and URDF files (from the upstream IFL corpus):

```bash
conda activate metagrasp
python tools/setup_ur_grasp_database.py \
  --source ../../models_ifl \
  --objects 000,001,004,011 \
  --overwrite
```

### 2. Generate grasp HDF5

Produce `textured.obj.hand_e.hdf5` for each object:

```bash
python tools/generate_object_grasps.py \
  --mesh_root UR_grasp_database \
  --categories 000 001 004 011 \
  --gripper_profile hand_e \
  --num_points 500 \
  --max_grasps 2000 \
  --pool_size 1
```

Expected per object:

- `UR_grasp_database/<id>/textured.obj.hand_e.hdf5`

### 3. Convert URDF to USD

Isaac Sim loads `orbit_obj.usd` (converted from `textured.urdf`):

```bash
conda activate metagrasp
python urdf_converter.py --input UR_grasp_database/000/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/001/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/004/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/011/textured.urdf --headless
```

Expected per object:

- `UR_grasp_database/<id>/orbit_obj.usd`

---

## Verify the task

```bash
conda activate metagrasp

python run.py --task object-v0-Classical \
  --obj_ids 011 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  --saved_video \
  --max_successful_grasps 1
```

Success videos are written to `logs/videos/object-v0-Classical/<timestamp>/`.

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`README.md`](README.md) | This quick start |
| [`../../grasp_sampler/sampling.md`](../../grasp_sampler/sampling.md) | Grasp generation and runtime options |
| [`../../grasp_sampler/pose-logistics.md`](../../grasp_sampler/pose-logistics.md) | Coordinate frames and troubleshooting |

Related: [`../../UR_grasp_database/README.md`](../../UR_grasp_database/README.md), [`../../tools/README.md`](../../tools/README.md)
