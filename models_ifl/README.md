# MetaGraspNet IFL object assets (not tracked in git)

This directory holds **MetaGraspNet** object meshes used by MetaIsaacGrasp and the antipodal grasp sampler. It is intentionally excluded from version control because of size (~20 GB+ for the full set).

## Where to put this folder

Clone or unpack assets at the **repository root**:

```text
superquadrics_metagrasp/models_ifl/
├── 000/
│   ├── textured.obj
│   ├── textured.urdf
│   └── orbit_obj.usd          # generated locally
├── 001/
│   ├── textured.obj
│   ├── textured.obj.hdf5      # generated locally
│   └── …
└── …
```

## How to obtain files

1. Download from the [MetaGraspNet IFL storage](https://nx25922.your-storageshare.de/s/9KrFffzwoTmtapR), or unzip `models_ifl.zip` next to this README.
2. Symlink into MetaIsaacGrasp:

   ```bash
   cd metagrasp/MetaIsaacGrasp/models
   ln -sfn ../../../models_ifl models_ifl
   ```

3. For each object used in simulation, run once:

   ```bash
   cd metagrasp/MetaIsaacGrasp
   python urdf_converter.py --input ../../models_ifl/001/textured.urdf
   python tools/generate_object_grasps.py --mesh_root ../../models_ifl --categories 001
   ```

Smoke-test default object id: **`001`**.

See the root `README.md` and `metagrasp/MetaIsaacGrasp/docs/classical_grasp_sampling.md`.
