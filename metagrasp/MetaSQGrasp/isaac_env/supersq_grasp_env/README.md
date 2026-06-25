# SuperSQ Grasp (`supersq_grasp-v0`)

Live perception-to-grasp pipeline: partial point cloud → SQ/ExtSQ shape fit → online Hand-E grasp sampling → pick-and-lift execution.

Package: `isaac_env/supersq_grasp_env/`  
Entry point: [`run.py`](../../run.py) with `--task supersq_grasp-v0`

---

## What this task does

| Layer | Component |
|-------|-----------|
| Robot / gripper | UR10e + Robotiq Hand-E |
| Scene | `UR_grasp_database/<id>/` objects |
| Cameras | On-hand `camera_0` + fixed top-down `camera_1` |
| Perception | Live object-only partial PC (default: top-down) |
| Shape | `sq_esti_model` or `extsq_esti_model` |
| Grasp | Online antipodal sampling on estimated SQ mesh |
| Execution | Classical pick-lift state machine |

Unlike `object-v0-Classical`, this task does **not** load precomputed HDF5 grasps.

---

## Pipeline

```mermaid
flowchart LR
  Cam[Camera obs] --> PC[Partial PC]
  PC --> SqEst[SQ/ExtSQ fit]
  SqEst --> Mesh[Estimated mesh]
  Mesh --> Sample[Online grasp sampling]
  Sample --> Exec[Pick-lift]
```

Runtime bridge: [`sq_bridge.py`](sq_bridge.py) loads checkpoints from `sq_estimation_model/pretrained/`.

---

## Prerequisites

- Conda env **`metagrasp`** with Isaac Lab 5.1 + Isaac Sim 5.1 (see [`../../install.md`](../../install.md))
- `pip install -r requirements.txt`
- Object assets under `UR_grasp_database/<id>/`:
  - `orbit_obj.usd` — simulation asset
  - `textured.obj` — mesh for partial-PC fallback
- SQ checkpoint under `sq_estimation_model/pretrained/` (`.pkl` files are local-only; configs are in git)

**Verified smoke-test object IDs:** `000`, `001`, `004`, `011`

SuperSQ does **not** require `textured.obj.hand_e.hdf5`.

---

## Quick start

From `metagrasp/MetaSQGrasp/`:

```bash
conda activate metagrasp

# Smoke test with debug export
python -u run.py --task supersq_grasp-v0 --obj_ids 004 --num_envs 1 \
  --headless --enable_cameras --max_successful_grasps 1 \
  --save_sq_debug

# Full run with success video
python -u run.py --task supersq_grasp-v0 --obj_ids 004 --num_envs 1 \
  --headless --enable_cameras --saved_video --max_successful_grasps 1 \
  --save_sq_debug
```

### Use ExtSQ model

```bash
python -u run.py --task supersq_grasp-v0 --obj_ids 004 --num_envs 1 \
  --headless --enable_cameras --max_successful_grasps 1 --save_sq_debug \
  --sq_model extsq_esti_model \
  --sq_config sq_estimation_model/pretrained/extsq_esti_model/extsq_esti_model_config.yml \
  --sq_checkpoint sq_estimation_model/pretrained/extsq_esti_model/extsq_esti_model_best.pkl
```

---

## Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--obj_ids` | demo pool | Comma-separated object IDs |
| `--sq_model` | `sq_esti_model` | `sq_esti_model` or `extsq_esti_model` |
| `--sq_config` | pretrained config path | Estimator YAML config |
| `--sq_checkpoint` | pretrained `.pkl` | Trained weights |
| `--save_sq_debug` | off | Export partial PC, SQ mesh, grasp axes to `logs/sq_debug/` |
| `--saved_video` | off | Write success montage to `logs/videos/` |
| `--max_successful_grasps` | task default | Stop after N successful grasps |

---

## Debug visualization

After a run with `--save_sq_debug`:

```bash
python tools/inspect_supersq_debug.py \
  --run logs/sq_debug/supersq_grasp-v0/<timestamp>/env0_epi000000_attempt0_obj004

python tools/inspect_supersq_grasps.py \
  --run logs/sq_debug/supersq_grasp-v0/<timestamp>/env0_epi000000_attempt0_obj004
```

---

## Compare with classical baseline

Run the oracle baseline on the same object:

```bash
python run.py --task object-v0-Classical --obj_ids 004 --num_envs 1 \
  --headless --enable_cameras --saved_video --max_successful_grasps 1
```

Classical uses precomputed HDF5 on the true mesh; SuperSQ uses live partial PC and estimated shape.

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`README.md`](README.md) | This quick start |
| [`workflow.md`](workflow.md) | Full pipeline, CLI flags, debug export, troubleshooting |

Related: [`../../sq_estimation_model/README.md`](../../sq_estimation_model/README.md), [`../supersq_grasp_data/README.md`](../supersq_grasp_data/README.md)
