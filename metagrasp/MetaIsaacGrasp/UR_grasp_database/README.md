# UR + Hand-E Grasp Database

This directory is the curated object asset root for `object-v0-Classical`.
It stores UR10e + Robotiq Hand-E test objects and their generated grasp assets.

Expected layout:

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

Generated files:

- `textured.obj.hand_e.hdf5` — Hand-E antipodal grasp candidates. Suffix and max
  jaw width come from `grasp_sampler/gripper_profile_defaults.py`
  (`HAND_E_HDF5_SUFFIX`, `HAND_E_WIDTH_MAX_CM`, currently **7.0 cm**).
- `orbit_obj.usd` — Isaac physics asset converted from `textured.urdf`.
- `asset_manifest.json` — source and copy metadata from
  `tools/setup_ur_grasp_database.py`.

Legacy note: older runs may still have plain `textured.obj.hdf5` (no suffix).
Regenerate with `tools/generate_object_grasps.py --gripper_profile hand_e` so
runtime loads `textured.obj.hand_e.hdf5`.

**Verified smoke-test objects (2026-06):** `000`, `001`, `004`, `011`.

The full IFL corpus lives outside this folder, typically at
`/home/sustech/桌面/DSQNet/models_ifl`. Seed new IDs with
`tools/setup_ur_grasp_database.py`.

**Full workflow:** [docs/ur_grasp_database_workflow.md](../docs/ur_grasp_database_workflow.md)
