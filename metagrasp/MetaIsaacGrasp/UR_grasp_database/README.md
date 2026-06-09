# UR + Hand-E Grasp Database

This directory is the curated object asset root for `object-v0-Classical`.
It stores only the UR10e + Robotiq Hand-E test objects and their generated
grasp assets.

Expected layout:

```text
UR_grasp_database/
├── 001/
│   ├── textured.obj
│   ├── textured.urdf
│   ├── textured.obj.hand_e.hdf5
│   └── orbit_obj.usd
├── 004/
├── 010/
└── 016/
```

Generated files:

- `textured.obj.hand_e.hdf5`: Hand-E antipodal grasp candidates (suffix from
  `HAND_E_HDF5_SUFFIX` in `grasp_sampler/gripper_profile_defaults.py`).
- `orbit_obj.usd`: Isaac physics asset converted from `textured.urdf`.
- `asset_manifest.json`: source and copy metadata written by
  `tools/setup_ur_grasp_database.py`.

The full IFL object corpus remains outside this folder, typically at
`/home/sustech/桌面/DSQNet/models_ifl`. Use it as the upstream source when
seeding this curated database.

**Full workflow (Chinese):** [docs/ur_grasp_database_workflow.md](../docs/ur_grasp_database_workflow.md)
