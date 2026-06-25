# MetaGraspNet IFL Object Assets

This directory holds **MetaGraspNet** object meshes used by MetaSQGrasp and the antipodal grasp sampler. It is intentionally excluded from version control because of size (~20 GB+ for the full set).

Repository overview: [`../README.md`](../README.md)

---

## Where to put this folder

Clone or unpack assets at the **repository root**:

```text
MetaSQGrasp_public/
├── models_ifl/
│   ├── 000/
│   │   ├── textured.obj
│   │   ├── textured.urdf
│   │   └── orbit_obj.usd          # generated locally
│   ├── 001/
│   │   ├── textured.obj
│   │   ├── textured.obj.hdf5      # generated locally (optional)
│   │   └── …
│   └── …
└── metagrasp/MetaSQGrasp/
```

---

## Downloadable assets (this folder)

| Asset | In git? | User action |
|-------|---------|-------------|
| **Full IFL corpus** (`models_ifl/`) | No (this README only) | Download from [Google Drive](https://drive.google.com/drive/folders/1U7pqp0JVDFzGKmypEwJcGf--OJJ-5Xml?usp=sharing) → unzip here |
| **Curated sim objects** (`UR_grasp_database/`) | Yes (6 IDs in repo) | Use as-is for smoke tests; expand via [`UR_grasp_database/workflow.md`](../metagrasp/MetaSQGrasp/UR_grasp_database/workflow.md) |

The public repo ships a small curated subset under `metagrasp/MetaSQGrasp/UR_grasp_database/` (objects **000, 001, 004, 010, 011, 016**). Download the full IFL corpus here when you need additional object IDs.

---

## How to obtain files

1. Download from the [MetaGraspNet IFL Google Drive folder](https://drive.google.com/drive/folders/1U7pqp0JVDFzGKmypEwJcGf--OJJ-5Xml?usp=sharing), or unzip `models_ifl.zip` next to this README.
2. Symlink into MetaSQGrasp:

   ```bash
   cd metagrasp/MetaSQGrasp/models
   ln -sfn ../../../models_ifl models_ifl
   ```

3. For each object used in simulation, run once (from `metagrasp/MetaSQGrasp/`):

   ```bash
   conda activate metagrasp
   python urdf_converter.py --input ../../models_ifl/001/textured.urdf --headless
   python tools/generate_object_grasps.py --mesh_root ../../models_ifl --categories 001
   ```

Smoke-test default object id: **`001`**.

For the curated simulation subset, see [`metagrasp/MetaSQGrasp/UR_grasp_database/README.md`](../metagrasp/MetaSQGrasp/UR_grasp_database/README.md).

---

## Attribution

Object assets originate from [MetaGraspNet](https://github.com/maximiliangilles/MetaGraspNet). See the root [`README.md`](../README.md) for citation information.

License: **CC BY-NC-SA 3.0** (academic use only).
