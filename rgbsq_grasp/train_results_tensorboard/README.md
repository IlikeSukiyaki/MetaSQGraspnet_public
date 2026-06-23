# rgbsq_grasp TensorBoard training outputs (not tracked in git)

Checkpoints and TensorBoard event files from `train_tensorboard.py` are written here.

## Expected layout

```text
train_results_tensorboard/
└── <config_name>/
    └── <run_name>/
        ├── model_best.pkl
        ├── model_iter_*.pkl
        ├── <config_copy>.yml
        └── tensorboard/
            └── events.out.tfevents.*
```

Launch TensorBoard:

```bash
tensorboard --logdir train_results_tensorboard/
```

See [rgbsq_grasp/README.md](../README.md) and [DATA_GENERATION.md](../DATA_GENERATION.md).
