# SQ estimation pretrained checkpoints

Released SQ and ExtSQ weights are **tracked in git** alongside their YAML configs.

## Layout

```text
pretrained/
├── sq_esti_model/
│   ├── sq_esti_model_config.yml
│   └── sq_esti_model_best.pkl
└── extsq_esti_model/
    ├── extsq_esti_model_config.yml
    └── extsq_esti_model_best.pkl
```

Use as-is with `supersq_grasp-v0`:

```bash
python run.py --task supersq_grasp-v0 --sq_model sq_esti_model --obj_id 001
python run.py --task supersq_grasp-v0 --sq_model extsq_esti_model --obj_id 001
```

## Retrain or replace

Train under `sq_estimation_model/training/` and copy the best checkpoint + config here.

See [sq_estimation_model/README.md](../README.md) and [training/README.md](../training/README.md).
