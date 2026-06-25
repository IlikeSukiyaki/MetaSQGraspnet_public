# SQ Estimation Model Training

Isolated training package for the SQ estimation model (`sq_esti_model`) and ExtSQ estimation model (`extsq_esti_model`). Separate from Isaac Lab simulation code under `isaac_env/` and `run.py`.

Package overview: [`../README.md`](../README.md)

---

## Layout

```text
sq_estimation_model/
  training/                 # run all training commands from here
    train.py                # W&B logging
    train_tensorboard.py    # TensorBoard logging
    configs/
    datasets/               # training datasets (JSON manifests)
    train_results/
    scripts/generate_extsq_est_dataset.py
  pretrained/
    sq_esti_model/
    extsq_esti_model/
```

---

## Environment

Use the separate **`rgbsq_grasp`** conda env for training (not `metagrasp`). This env name is kept for historical compatibility — it is **not** a folder in this repository.

Simulation versions: [install.md](../../install.md)

```bash
conda create -n rgbsq_grasp python=3.10 -y
conda activate rgbsq_grasp
pip install torch  # install a CUDA-matched wheel (2.7+ recommended)

cd metagrasp/MetaSQGrasp/sq_estimation_model/training
python -m pip install -r requirements.txt
```

---

## Train SQ estimation model

```bash
cd metagrasp/MetaSQGrasp/sq_estimation_model/training
conda activate rgbsq_grasp

python train.py \
  --config configs/sq_esti_model_config.yml \
  --device 0 \
  --run sq_esti_model_full \
  --wandb-project sq_estimation_model
```

## Train ExtSQ estimation model

```bash
python train.py \
  --config configs/extsq_esti_model_config.yml \
  --device 0 \
  --run extsq_esti_model_full \
  --wandb-project sq_estimation_model
```

Offline logging (no W&B upload):

```bash
WANDB_MODE=offline python train.py --config configs/sq_esti_model_config.yml --device 0 --run my_run
```

---

## TensorBoard training

```bash
python train_tensorboard.py \
  --config configs/sq_esti_model_config.yml \
  --device 0 \
  --run sq_esti_model_tb

python train_tensorboard.py \
  --config configs/extsq_esti_model_config.yml \
  --device 0 \
  --run extsq_esti_model_tb
```

View logs:

```bash
tensorboard --logdir train_results_tensorboard
```

TensorBoard smoke tests:

```bash
python train_tensorboard.py --config configs/sq_esti_model_config_smoke.yml --device cpu --run smoke_sq_tb
python train_tensorboard.py --config configs/extsq_esti_model_config_smoke.yml --device cpu --run smoke_extsq_tb
```

Checkpoints are saved under:

```text
train_results/sq_esti_model_config/<run>/sq_esti_model_best.pkl
train_results_tensorboard/sq_esti_model_config/<run>/sq_esti_model_best.pkl
```

Copy the best checkpoint for Isaac runtime:

```bash
cp train_results/sq_esti_model_config/sq_esti_model_full/sq_esti_model_best.pkl \
   ../pretrained/sq_esti_model/sq_esti_model_best.pkl
```

---

## Smoke tests

```bash
WANDB_MODE=disabled python train.py --config configs/sq_esti_model_config_smoke.yml --device 0 --run smoke_sq
WANDB_MODE=disabled python train.py --config configs/extsq_esti_model_config_smoke.yml --device 0 --run smoke_extsq
```

---

## Dataset layout (JSON manifests)

CSV datalists are **not** used. Each dataset has:

- `dataset_manifest.json` — split definitions
- `*_class_map.json` — stable ID → semantic type mapping

Datasets live under `sq_estimation_model/training/datasets/`. See [`datasets/README.md`](datasets/README.md) for schema details.

### Primitive dataset

| Class ID | Semantic type |
|----------|---------------|
| `primitive_001` | box |
| `primitive_002` | cone |
| `primitive_003` | cylinder |
| `primitive_004` | ellipsoid |
| `primitive_005` | truncated_cone |
| `primitive_006` | truncated_torus |

### Object / evaluation datasets

- `object_dataset/` — `object_001` … `object_012`
- `evaluation_dataset/` — `evaluation_001` … `evaluation_012`
- `extsq_est_dataset/` — extended superquadrics (`extsq_est_001`)

### Generate ExtSQ dataset

```bash
python scripts/generate_extsq_est_dataset.py \
  --render \
  --num-objects 1000 \
  --target datasets/extsq_est_dataset \
  --overwrite
```

Or from an existing generated folder:

```bash
python scripts/generate_extsq_est_dataset.py \
  --source-dataset /path/to/extended_sq_dataset \
  --target datasets/extsq_est_dataset \
  --overwrite
```

---

## ExtSQ offline evaluation

```bash
python scripts/evaluate_extsq_est_model.py \
  --config configs/extsq_esti_model_config.yml \
  --checkpoint train_results/extsq_esti_model_config/extsq_esti_model_full/extsq_esti_model_best.pkl \
  --dataset datasets/extsq_est_dataset \
  --split test \
  --max-samples 32 \
  --output eval_results/extsq_est_eval.json
```

---

## Inference helper

```python
from inference import load_recognition_model, predict_parameters

model, cfg = load_recognition_model(
    "configs/sq_esti_model_config.yml",
    "train_results/sq_esti_model_config/my_run/sq_esti_model_best.pkl",
    device="cuda:0",
)
params = predict_parameters(model, point_cloud_xyz, num_points=300)
```

---

## Deploy to simulation

After training, copy checkpoints to `sq_estimation_model/pretrained/` and run:

```bash
conda activate metagrasp
python run.py --task supersq_grasp-v0 --obj_ids 004 --num_envs 1 \
  --headless --enable_cameras --max_successful_grasps 1 --save_sq_debug
```

Guide: [`isaac_env/supersq_grasp_env/README.md`](../../isaac_env/supersq_grasp_env/README.md)

---

## Model naming reference

| Component | Name |
|-----------|------|
| Regular SQ estimator | `sq_esti_model` |
| ExtSQ estimator | `extsq_esti_model` |
| Regular SQ config | `sq_esti_model_config.yml` |
| ExtSQ config | `extsq_esti_model_config.yml` |
| Regular SQ loss | `sq_esti_loss` |
| ExtSQ loss | `extsq_esti_loss` |
