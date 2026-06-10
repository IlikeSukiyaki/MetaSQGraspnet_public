# MetaIsaacGrasp datasets

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
cd metagrasp/MetaIsaacGrasp
conda activate env_isaaclab
bash tools/collect_demo_dataset.sh datasets/supersq_grasp_data/demo
```

## Full collection

```bash
cd metagrasp/MetaIsaacGrasp
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

## Git tracking the demo folder (when ready to push)

Add to the repo root [`.gitignore`](../../.gitignore):

```gitignore
!metagrasp/MetaIsaacGrasp/datasets/supersq_grasp_data/demo/
!metagrasp/MetaIsaacGrasp/datasets/supersq_grasp_data/demo/**
```

Object meshes and USD assets remain under `UR_grasp_database/` (tracked separately).

See `metagrasp/MetaIsaacGrasp/README.md` and the root `README.md` for full documentation.
