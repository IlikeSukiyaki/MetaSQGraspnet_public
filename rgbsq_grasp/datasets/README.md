# rgbsq_grasp datasets (not tracked in git)

Training, evaluation, and Isaac-synthetic samples are stored here and excluded from git.

## Expected layout

```text
datasets/
├── primitive_dataset/        # DSQNet primitive training (from upstream Drive)
├── object_dataset/
├── evaluation_dataset/
├── extended_sq_dataset/      # generated extended superquadrics (see DATA_GENERATION.md)
└── isaac_synth/              # converted from MetaIsaacGrasp SQ-v0-Data
    └── 001/
        ├── datalist.csv
        └── *.npy
```

## How to populate

| Subfolder | Source |
|-----------|--------|
| `primitive_dataset/`, `object_dataset/`, `evaluation_dataset/` | [DSQNet dataset Drive](https://drive.google.com/drive/folders/1PQ9dSeD0WmdESQemsnM1SPmpPDChQ95s) |
| `extended_sq_dataset/` | `python data_generation.py --config extended_sq --name extended_sq_dataset` — see [DATA_GENERATION.md](../DATA_GENERATION.md) |
| `isaac_synth/` | `metagrasp/MetaSQGrasp/tools/convert_sq_to_npy.py` on collected Isaac episodes |

Example Isaac conversion:

```bash
cd metagrasp/MetaSQGrasp
python tools/convert_sq_to_npy.py \
  --input datasets/sqnet_isaac/001 \
  --output ../../rgbsq_grasp/datasets/isaac_synth/001 \
  --camera both
```

See [DATA_GENERATION.md](../DATA_GENERATION.md), [rgbsq_grasp/README.md](../README.md), and the repository root [README.md](../../README.md).
