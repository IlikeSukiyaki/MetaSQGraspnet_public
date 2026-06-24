# SQ estimation pretrained checkpoints (not tracked in git)

Example YAML configs are tracked; `.pkl` weights stay local.

## Expected layout

```text
pretrained/
├── sq_esti_model/
│   ├── sq_esti_model_config.yml
│   └── sq_esti_model_best.pkl      # local only
└── extsq_esti_model/
    ├── extsq_esti_model_config.yml
    └── extsq_esti_model_best.pkl   # local only
```

## How to populate

Train under `sq_estimation_model/training/` and copy the best checkpoint + config here, or download released weights when available.

See [sq_estimation_model/README.md](../README.md) and [training/README.md](../training/README.md).
