# SQ estimation training datasets (not tracked in git)

Large point-cloud datasets live here. Git tracks only JSON/CSV manifests and generation metadata so you can regenerate or download the full data locally.

## Expected layout

```text
datasets/
├── primitive_dataset/
│   └── dataset_manifest.json       # tracked
├── object_dataset/
├── evaluation_dataset/
└── extsq_est_dataset/              # extended superquadrics
    ├── dataset_manifest.json       # tracked
    ├── generation_config.json      # tracked
    ├── instance_manifest.csv       # tracked
    └── extsq_est_001/              # local point clouds (ignored)
```

## How to populate

| Subfolder | Command |
|-----------|---------|
| `primitive_dataset/`, `object_dataset/`, `evaluation_dataset/` | Included in repo manifests; regenerate `.npy` samples locally or use `python scripts/migrate_datasets.py --source /path/to/source --target datasets` |
| `extsq_est_dataset/` | `python scripts/generate_extsq_est_dataset.py` |

See [training/README.md](../README.md).
