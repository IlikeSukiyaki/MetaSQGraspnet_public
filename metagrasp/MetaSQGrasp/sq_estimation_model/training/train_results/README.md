# SQ estimation training outputs (not tracked in git)

Checkpoints and logs from `train.py` / `train_tensorboard.py` are written here.

## Expected layout

```text
train_results/
└── <run_name>/
    ├── sq_esti_model_best.pkl
    ├── model_iter_*.pkl
    └── <config_copy>.yml
```

See [training/README.md](../README.md).
