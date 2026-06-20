# DSQNet Migration Implementation

This folder is the migrated working copy of `DSQNet-public`. The reference folder at `/home/sustech/桌面/DSQNet/DSQNet-public` should remain untouched.

## What Was Implemented

- Copied the DSQNet source layout into `/home/sustech/桌面/DSQNet/rgbsq_grasp` while preserving the existing `datasets/` directory.
- Migrated training logging from TensorBoard to Weights & Biases via `utils/training_logger.py`.
- Added safer device handling so `--device cpu` works and CUDA seed calls are skipped when CUDA is unavailable.
- Added checkpoint loading compatibility for newer PyTorch versions through `models.safe_torch_load()`.
- Added clearer missing-checkpoint errors for `evaluation.py` and pretrained model loading.
- Fixed a Python comparison issue in `trainers/recognition_trainer.py`.
- Added `inference.py`, a lightweight inference helper intended for later IsaacLab use. It loads SQNet/DSQNet checkpoints and predicts raw primitive parameters from point cloud arrays/tensors without importing Open3D visualization paths.
- Added `supersq_inference.py` for SQNet-only superquadric fitting and Open3D visualization on a single partial point cloud (no SegNet required). Prefer this over `evaluation.py` when checking a trained SQNet checkpoint.
- Updated `requirements.txt` to use modern package names such as `scikit-learn` instead of deprecated `sklearn`.
- Added `environment.yml` documenting the Python 3.10 environment target.

## Environment Target

The target is Python 3.10 because IsaacSim 4.5 / IsaacLab 2.0 are Python 3.10 based.

RTX 5090 uses the Blackwell `sm_120` architecture, so use a PyTorch build with CUDA 12.8+ support. The commands below use the CUDA 12.8 PyTorch wheel index.

## Conda Installation

From a normal terminal, outside the sandbox:

```bash
conda create -n rgbsq_grasp python=3.10 -y
conda activate rgbsq_grasp

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r /home/sustech/桌面/DSQNet/rgbsq_grasp/requirements.txt
```

`python-fcl` (imported as `fcl`) is in `requirements.txt` for grasp collision checking. SQNet training does not need it at import time anymore.

If the environment already exists because it was partially created earlier, use:

```bash
conda activate rgbsq_grasp
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r /home/sustech/桌面/DSQNet/rgbsq_grasp/requirements.txt
```

Verify the environment:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

Expected: CUDA should be available and the device should be `NVIDIA GeForce RTX 5090`.

## Project Smoke Tests

**Important:** All project commands must run from `rgbsq_grasp`, not from `~/桌面/DSQNet`. If you see `ModuleNotFoundError: No module named 'models'` or `'loader'`, your shell is in the wrong directory.

```bash
cd /home/sustech/桌面/DSQNet/rgbsq_grasp
conda activate rgbsq_grasp
pwd   # should end with .../rgbsq_grasp
```

Check imports:

```bash
python - <<'PY'
import torch
import open3d
import omegaconf
import pandas
import sklearn
import scipy
import wandb
print("imports ok")
PY
```

Check SQNet construction and a tiny CUDA forward pass:

```bash
python - <<'PY'
import torch
from omegaconf import OmegaConf
from models import get_model

cfg = OmegaConf.load("configs/sqnet_config.yml")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = get_model(cfg).to(device).eval()
x = torch.randn(1, 3, 300, device=device)
with torch.no_grad():
    y = model(x)
print("output shape:", tuple(y.shape))
print("device:", device)
PY
```

Check dataset loading:

```bash
python - <<'PY'
from omegaconf import OmegaConf
from loader import get_dataloader

cfg = OmegaConf.load("configs/sqnet_config.yml")
cfg.data.training.batch_size = 1
cfg.data.training.num_workers = 0
loader = get_dataloader(cfg.data.training)
batch = next(iter(loader))
print("batch items:", len(batch))
print("partial pc shape:", tuple(batch[0].shape))
PY
```

Start SQNet training (logs to W&B; use offline mode on headless machines):

```bash
python train.py --config configs/sqnet_config.yml --device 0 --run sqnet_baseline
# or:
WANDB_MODE=offline python train.py --config configs/sqnet_config_smoke.yml --device 0 --run wandb_smoke
```

The full `sqnet_config.yml` run may take a long time. Checkpoints are saved under `train_results/`.

Evaluation requires pretrained checkpoints in:

```text
pretrained/segnet_config/example/segnet_config.yml
pretrained/segnet_config/example/model_best.pkl
pretrained/sqnet_config/example/sqnet_config.yml
pretrained/sqnet_config/example/model_best.pkl
pretrained/dsqnet_config/example/dsqnet_config.yml
pretrained/dsqnet_config/example/model_best.pkl
```

Then run:

```bash
python evaluation.py --object box --index 0 --device 0
```

Evaluation opens Open3D viewer windows. By default (`--viz-mode both`), each viewpoint shows:

1. Ground truth, segmentation, and observed point cloud (standalone)
2. SQNet prediction alone
3. SQNet vs GT side by side (offset horizontally so meshes do not overlap)
4. SQNet vs GT wireframe overlay (GT edges + solid prediction in the same pose)

Use `--viz-mode legacy` for the old opaque overlap view, or `--viz-mode minimal` for only the three standalone views. Use `--no-visualize` on headless machines.

### SQNet quick visualization (`supersq_inference.py`)

Recommended way to inspect a trained SQNet checkpoint on one partial point cloud. Unlike `evaluation.py`, this script does **not** require SegNet or DSQNet — it loads SQNet directly and opens a single Open3D window showing the input point cloud plus the fitted superquadric mesh (solid + wireframe).

Run from `rgbsq_grasp` with a display attached:

```bash
cd /home/sustech/桌面/DSQNet/rgbsq_grasp
conda activate rgbsq_grasp

python supersq_inference.py \
  --object box \
  --index 0 \
  --device 0 \
  --config train_results/sqnet_config/sqnet_full/sqnet_config.yml \
  --checkpoint train_results/sqnet_config/sqnet_full/model_best.pkl \
  --visualize
```

Use any checkpoint from the same run (`model_best.pkl`, `model_iter_420000.pkl`, etc.) as long as the matching config YAML is passed to `--config`.

**Arguments:**

| Flag | Description |
|------|-------------|
| `--object` | Evaluation object class (`box`, `cylinder`, `cone`, …) |
| `--index` | Sample index `0`–`9` (loads `{object}_{index+90:04}_viewpoint_XX.npy`) |
| `--viewpoint` | Optional viewpoint `1`–`14` (default: first file found) |
| `--segment-id` | Optional GT membership label to extract one primitive patch |
| `--config` | SQNet config YAML (use the copy saved under `train_results/…/`) |
| `--checkpoint` | SQNet `.pkl` checkpoint |
| `--device` | GPU index (`0`), `cuda:0`, or `cpu` |
| `--visualize` | Open Open3D viewer (partial cloud + fitted superquadric) |
| `--no-visualize` | Print parameters only |
| `--output` | Optional path to save predicted parameters as JSON |
| `--input` | Alternative to `--object`: path to a `.npy` point cloud file |

**Example — evaluation sample with a specific checkpoint:**

```bash
python supersq_inference.py \
  --object box \
  --index 0 \
  --device 0 \
  --config train_results/sqnet_config/sqnet_full/sqnet_config.yml \
  --checkpoint train_results/sqnet_config/sqnet_full/model_iter_420000.pkl \
  --visualize
```

**Example — custom `.npy` input:**

```bash
python supersq_inference.py \
  --input datasets/evaluation_dataset/box/box_0090_viewpoint_01.npy \
  --config train_results/sqnet_config/sqnet_full/sqnet_config.yml \
  --checkpoint train_results/sqnet_config/sqnet_full/model_best.pkl \
  --visualize
```

**Example — headless parameter dump (no GUI):**

```bash
python supersq_inference.py \
  --object cylinder \
  --index 3 \
  --device 0 \
  --config train_results/sqnet_config/sqnet_full/sqnet_config.yml \
  --checkpoint train_results/sqnet_config/sqnet_full/model_best.pkl \
  --no-visualize \
  --output results/cylinder_3_sqnet.json
```

Terminal output includes predicted superquadric parameters (`a1`, `a2`, `a3`, `e1`, `e2`). With `--visualize`, Open3D shows the gray partial point cloud, orange fitted mesh, and blue wireframe overlay in one window.

Default pretrained paths (when `--config` / `--checkpoint` are omitted) are `pretrained/sqnet_config/example/sqnet_config.yml` and `model_best.pkl`.

## IsaacLab Notes

Keep the DSQNet training environment separate from the IsaacLab environment. Later, IsaacLab can consume trained checkpoints through `inference.py`:

```python
from inference import load_recognition_model, predict_parameters

model, cfg = load_recognition_model(
    "configs/sqnet_config.yml",
    "path/to/model_best.pkl",
    device="cuda:0",
)
params = predict_parameters(model, point_cloud_xyz, num_points=300)
```

The helper accepts point clouds shaped `N x 3` or `3 x N` and returns the raw predicted SQ/DSQ parameter vector.
