# MetaIsaacGrasp datasets (not tracked in git)

Runtime-collected simulation data lives here. Nothing in this tree is committed to git.

## Typical layout

```text
datasets/
├── sqnet_isaac/              # SQ-v0-Data task (--output_dir datasets/sqnet_isaac)
│   ├── manifest.jsonl
│   └── 001/epi_000000/…
└── …
```

## How to populate

```bash
cd metagrasp/MetaIsaacGrasp
conda activate env_isaaclab
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras \
  --save_camera_data --output_dir datasets/sqnet_isaac --max_episodes 100
```

Convert episodes to rgbsq_grasp `.npy` samples:

```bash
python tools/convert_sq_to_npy.py \
  --input datasets/sqnet_isaac/001 \
  --output ../../rgbsq_grasp/datasets/isaac_synth/001 \
  --camera both
```

See `metagrasp/MetaIsaacGrasp/README.md` and the root `README.md` for full documentation.
