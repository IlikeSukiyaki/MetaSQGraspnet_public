# MetaSQGrasp datasets

Runtime-collected simulation data lives here. Most of this tree is **gitignored**; see below for the curated demo exception.

## Typical layout

```text
datasets/
├── supersq_grasp_data/
│   ├── demo/                 # curated lite demo (whitelist for git when ready)
│   │   ├── manifest.jsonl
│   │   ├── 000/epi_000000/…
│   │   ├── 001/epi_000000/…
│   │   ├── 004/epi_000000/…
│   │   └── 011/epi_000000/…
│   ├── manifest.jsonl        # full local collection (ignored)
│   └── 001/epi_000000/…      # full exports (~46 MB/episode, ignored)
└── …
```

## Demo dataset

Verified objects **000, 001, 004, 011** from `UR_grasp_database/` (same pool as `supersq_grasp-v0` and `object-v0-Classical`).

Lite episodes (~1–3 MB each) contain `meta.json`, dual-camera RGB/depth/instance PNGs, and `partial_pc/` payloads.

See [`supersq_grasp_data/demo/README.md`](supersq_grasp_data/demo/README.md) for collection and git-tracking instructions.

Collect all demo episodes:

```bash
cd metagrasp/MetaSQGrasp
conda activate env_isaaclab
bash tools/collect_demo_dataset.sh datasets/supersq_grasp_data/demo
```

## Multi-object scenes

Collect cluttered-table episodes with `--scene_objects` (2–4 objects from the pool) and visualize all object partial PCs with `tools/visualize_partial_pc.py`.

Full guide: [`docs/multi_object_data_collection.md`](../docs/multi_object_data_collection.md)

```bash
# 4 objects on table, full export (per-object visualization)
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras --headless \
  --save_camera_data --export_profile full \
  --obj_ids 000,001,004,011 --scene_objects 4 \
  --output_dir datasets/supersq_grasp_data/multi_obj_full \
  --max_episodes 1 --topdown_height 2.1

conda activate metagrasp
python tools/visualize_partial_pc.py \
  --sample datasets/supersq_grasp_data/multi_obj_full/011/epi_000000 \
  --source extract --objects all --camera topdown
```

Lite multi-object collection (target partial PC only, smaller files):

```bash
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras --headless \
  --save_camera_data --export_profile lite \
  --obj_ids 000,001,004,011 --scene_objects 2 \
  --output_dir datasets/supersq_grasp_data/multi_obj \
  --max_episodes 20
```

## Full collection

```bash
cd metagrasp/MetaSQGrasp
conda activate env_isaaclab
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras --headless \
  --save_camera_data \
  --export_profile full \
  --obj_ids 000,001,004,011 \
  --output_dir datasets/supersq_grasp_data \
  --max_episodes 100 \
  --topdown_height 2.1
```

Lite export (smaller bundles, no `scene.pt`):

```bash
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras --headless \
  --save_camera_data --export_profile lite \
  --obj_ids 004 --output_dir datasets/supersq_grasp_data/demo \
  --max_episodes 1
```

Optional per-camera scene point maps in lite mode: add `--save_scene_pc`.

Convert episodes to rgbsq_grasp `.npy` samples:

```bash
python tools/convert_sq_to_npy.py \
  --input datasets/supersq_grasp_data/demo/001 \
  --output ../../rgbsq_grasp/datasets/isaac_synth/001 \
  --camera both
```

## Git-tracked demo sample

One curated multi-object episode is whitelisted in the repo root [`.gitignore`](../../.gitignore):

`supersq_grasp_data/multi_obj_full/011/epi_000000/` (~46 MB, 4-object cluttered table, full export).

All other collected data stays local. Object meshes and USD assets remain under `UR_grasp_database/` (tracked separately).

See `metagrasp/MetaSQGrasp/README.md` and the root `README.md` for full documentation.
