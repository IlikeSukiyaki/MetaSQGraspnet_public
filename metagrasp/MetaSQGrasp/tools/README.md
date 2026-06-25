# Tools

CLI utilities for object asset setup, grasp generation, URDF conversion, dataset processing, and debug visualization. Run all commands from the MetaSQGrasp root (`metagrasp/MetaSQGrasp/`).

---

## Asset pipeline

| Script | Purpose |
|--------|---------|
| [`setup_ur_grasp_database.py`](setup_ur_grasp_database.py) | Copy mesh/URDF from `models_ifl` into `UR_grasp_database/<id>/` |
| [`generate_object_grasps.py`](generate_object_grasps.py) | Offline antipodal grasp sampling → `textured.obj.hand_e.hdf5` |
| [`convert_models_ifl_to_usd.sh`](convert_models_ifl_to_usd.sh) | Batch URDF → USD via `urdf_converter.py` |
| [`../urdf_converter.py`](../urdf_converter.py) | Single-object URDF → `orbit_obj.usd` for Isaac Sim |

### Seed curated objects

```bash
conda activate metagrasp

python tools/setup_ur_grasp_database.py \
  --source ../../models_ifl \
  --objects 000,001,004,011 \
  --overwrite
```

### Generate Hand-E grasps

```bash
python tools/generate_object_grasps.py \
  --mesh_root UR_grasp_database \
  --categories 000 001 004 011 \
  --gripper_profile hand_e \
  --max_grasps 2000
```

### Convert URDF to USD

```bash
python urdf_converter.py --input UR_grasp_database/001/textured.urdf --headless
```

Or batch:

```bash
bash tools/convert_models_ifl_to_usd.sh
```

---

## Dataset collection

| Script | Purpose |
|--------|---------|
| [`collect_demo_dataset.sh`](collect_demo_dataset.sh) | Collect lite demo episodes for objects 000, 001, 004, 011 |
| [`convert_sq_to_npy.py`](convert_sq_to_npy.py) | Convert `SQ-v0-Data` episodes to estimator `.npy` format |
| [`extract_partial_pc.py`](extract_partial_pc.py) | Extract object-only partial point clouds from saved episodes |
| [`prune_sq_sample_lite.py`](prune_sq_sample_lite.py) | Convert a full episode export to lite profile |
| [`inspect_sq_sample.py`](inspect_sq_sample.py) | Inspect depth/seg/pcd of one episode folder |

### Collect demo dataset

```bash
bash tools/collect_demo_dataset.sh datasets/supersq_grasp_data/demo
```

### Extract partial point cloud

```bash
python tools/extract_partial_pc.py \
  --sample datasets/supersq_grasp_data/001/epi_000000 \
  --output /tmp/partial_pc.ply
```

---

## Debug and verification

| Script | Purpose |
|--------|---------|
| [`inspect_supersq_debug.py`](inspect_supersq_debug.py) | Visualize SQ debug export (partial PC + fitted mesh) |
| [`inspect_supersq_grasps.py`](inspect_supersq_grasps.py) | Visualize grasp axes on SQ mesh |
| [`inspect_supersq_obs.py`](inspect_supersq_obs.py) | Inspect live camera observations |
| [`visualize_partial_pc.py`](visualize_partial_pc.py) | Multi-object partial PC visualization |
| [`verify_classical_grasp_frame.py`](verify_classical_grasp_frame.py) | Verify sampler → `hande_end` frame mapping |
| [`calibrate_hande_grasp_frame.py`](calibrate_hande_grasp_frame.py) | Hand-E grasp frame calibration |
| [`measure_hande_end_frame.py`](measure_hande_end_frame.py) | Measure Hand-E EE frame in sim |
| [`check_hande_frame.py`](check_hande_frame.py) | Quick Hand-E frame check |
| [`check_2f85_frame.py`](check_2f85_frame.py) | 2F85 frame check |
| [`measure_2f85_tool0_frame.py`](measure_2f85_tool0_frame.py) | 2F85 tool0 frame measurement |

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`README.md`](README.md) | This tool index |
| [`partial-pc-extraction.md`](partial-pc-extraction.md) | `extract_partial_pc.py` reference |

Related: [`../isaac_env/supersq_grasp_data/multi-object-collection.md`](../isaac_env/supersq_grasp_data/multi-object-collection.md), [`../UR_grasp_database/workflow.md`](../UR_grasp_database/workflow.md)
