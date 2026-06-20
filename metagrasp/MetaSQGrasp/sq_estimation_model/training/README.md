# SQ Estimation Model Training (Isolated)

This folder is the **isolated training package** for the SQ estimation model (`sq_esti_model`) and extended SQ estimation model (`extended_sq_esti_model`). It is intentionally separate from the Isaac Lab simulation code under `isaac_env/` and `run.py`.

The original source under `rgbsq_grasp/` is left untouched.

## Layout

```text
sq_estimation_model/
  training/                 # run all training commands from here
    train.py
    configs/
    datasets/               # copied + renamed datasets (JSON manifests)
    train_results/
    scripts/migrate_datasets.py
  pretrained/
    sq_esti_model/
```

## Environment

Use the same conda env as `rgbsq_grasp` (Python 3.10 + PyTorch), then install deps:

```bash
conda activate rgbsq_grasp
cd /home/sustech/桌面/DSQNet/metagrasp/MetaSQGrasp/sq_estimation_model/training
python -m pip install -r requirements.txt
```

Install PyTorch separately if needed (CUDA 12.8+ wheels for RTX 5090).

## Train SQ estimation model

```bash
cd /home/sustech/桌面/DSQNet/metagrasp/MetaSQGrasp/sq_estimation_model/training
conda activate rgbsq_grasp

python train.py \
  --config configs/sq_esti_model_config.yml \
  --device 0 \
  --run sq_esti_model_full \
  --wandb-project sq_estimation_model
```

## Train extended SQ estimation model

```bash
python train.py \
  --config configs/extended_sq_esti_model_config.yml \
  --device 0 \
  --run extended_sq_esti_model_full \
  --wandb-project sq_estimation_model
```

Offline logging (no W&B upload):

```bash
WANDB_MODE=offline python train.py --config configs/sq_esti_model_config.yml --device 0 --run my_run
```

Checkpoints are saved under:

```text
train_results/sq_esti_model_config/<run>/model_best.pkl
train_results/extended_sq_esti_model_config/<run>/model_best.pkl
```

## Smoke tests

```bash
WANDB_MODE=disabled python train.py --config configs/sq_esti_model_config_smoke.yml --device 0 --run smoke_sq
WANDB_MODE=disabled python train.py --config configs/extended_sq_esti_model_config_smoke.yml --device 0 --run smoke_extended
```

## Dataset layout (JSON manifests)

CSV datalists are **not** used. Each dataset has:

- `dataset_manifest.json` — split definitions
- `*_class_map.json` — stable ID → semantic type mapping

### Primitive dataset

| Class ID | Semantic type |
|----------|---------------|
| `primitive_001` | box |
| `primitive_002` | cone |
| `primitive_003` | cylinder |
| `primitive_004` | ellipsoid |
| `primitive_005` | truncated_cone |
| `primitive_006` | truncated_torus |

Example file:

```text
datasets/primitive_dataset/primitive_001/primitive_001_0000_viewpoint_01.npy
```

Manifest entry:

```json
{
  "path": "primitive_001/primitive_001_0000_viewpoint_01.npy",
  "class_id": "primitive_001",
  "source_type": "box",
  "instance": "0000",
  "viewpoint": "01"
}
```

Splits: `train` (6959), `validation` (2982), `test` (1106) after viewpoint filtering.

### Object dataset

Uses `object_001` … `object_012` (alphabetical semantic mapping). See `datasets/object_dataset/object_class_map.json`.

Splits: `train`, `validation`, `test`.

### Evaluation dataset

Uses `evaluation_001` … `evaluation_012` with the same semantic order as object classes. See `datasets/evaluation_dataset/evaluation_class_map.json`.

Single split: `all` (1701 samples).

## Re-migrate datasets from rgbsq_grasp

If you need to refresh copied data from the untouched source:

```bash
python scripts/migrate_datasets.py \
  --source /home/sustech/桌面/DSQNet/rgbsq_grasp/datasets \
  --target datasets
```

This physically copies `.npy` files, renames class folders/files to stable IDs, and regenerates JSON manifests.

## Inference helper

```python
from inference import load_recognition_model, predict_parameters

model, cfg = load_recognition_model(
    "configs/sq_esti_model_config.yml",
    "train_results/sq_esti_model_config/my_run/model_best.pkl",
    device="cuda:0",
)
params = predict_parameters(model, point_cloud_xyz, num_points=300)
```

## Naming reference

| Old (rgbsq_grasp) | New (MetaSQGrasp) |
|-------------------|----------------------|
| sqnet | sq_esti_model |
| dsqnet | extended_sq_esti_model |
| sqnet_config.yml | sq_esti_model_config.yml |
| dsqnet_config.yml | extended_sq_esti_model_config.yml |
| sq_loss | sq_esti_loss |
| dsq_loss | extended_sq_esti_loss |
