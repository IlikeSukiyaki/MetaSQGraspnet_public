# rgbsq_grasp training outputs (not tracked in git)

Checkpoints, logs, and W&B offline runs from `train.py` are written here.

## Expected layout

```text
train_results/
└── <run_name>/
    ├── model_best.pkl
    ├── model_iter_*.pkl
    └── <config_copy>.yml
```

For TensorBoard runs, use `train_results_tensorboard/` instead (see that folder's README).

See [rgbsq_grasp/README.md](../README.md).
