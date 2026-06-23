# rgbsq_grasp pretrained checkpoints (not tracked in git)

Place trained model checkpoints here. Only YAML configs are tracked; `.pkl` weights stay local.

## Expected layout

```text
pretrained/
├── sqnet_config/example/
│   ├── sqnet_config.yml
│   └── model_best.pkl          # local only
├── dsqnet_config/example/
│   ├── dsqnet_config.yml
│   └── model_best.pkl          # local only
└── segnet_config/example/
    ├── segnet_config.yml
    └── model_best.pkl          # local only
```

## How to populate

| Model | Source |
|-------|--------|
| SQNet / SegNet | [DSQNet pretrained Drive](https://drive.google.com/drive/folders/1PQ9dSeD0WmdESQemsnM1SPmpPDChQ95s) |
| DSQNet (extended SQ) | Train with `train_tensorboard.py` — outputs go to `train_results_tensorboard/`; copy `model_best.pkl` + config here |

See [rgbsq_grasp/README.md](../README.md) and the repository root [README.md](../../README.md).
