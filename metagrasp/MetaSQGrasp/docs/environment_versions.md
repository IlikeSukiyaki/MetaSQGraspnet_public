# Environment Versions and Dependencies

This document records the **verified software stack** for MetaSQGrasp on this machine, plus pinned/special dependency notes discovered during setup and SQ-grasp debugging.

For end-to-end workflows see [general_workflow.md](./general_workflow.md).

---

## Conda environments

| Env | Python | Used for |
|-----|--------|----------|
| `metagrasp` | 3.11.x | Isaac Sim / Isaac Lab simulation, grasp tasks, SQ inference at runtime |
| `rgbsq_grasp` | 3.10+ | Offline SQ / ExtSQ training (`sq_estimation_model/training/`) |

Do **not** mix these stacks in one command: `run.py` uses `metagrasp`; `train.py` / `train_tensorboard.py` use `rgbsq_grasp`.

---

## Verified simulation stack (`metagrasp`)

Tested with live `supersq_grasp-v0` execution.

| Component | Version | Install location / notes |
|-----------|---------|---------------------------|
| **Isaac Sim (pip metapackage)** | **5.1.0.0** | `isaacsim`, `isaacsim-kernel`, … in `metagrasp` |
| **Isaac Sim (binary symlink)** | **5.1.0-rc.19** | `~/IsaacLab/_isaac_sim/VERSION` |
| **Isaac Lab core** | **0.47.2** | editable: `~/IsaacLab/source/isaaclab` |
| **isaaclab_assets** | **0.2.3** | `~/IsaacLab/source/isaaclab_assets` |
| **isaaclab_tasks** | **0.11.6** | `~/IsaacLab/source/isaaclab_tasks` |
| **isaaclab_rl** | **0.4.4** | `~/IsaacLab/source/isaaclab_rl` |
| **isaaclab_mimic** | **1.0.15** | `~/IsaacLab/source/isaaclab_mimic` |
| **PyTorch** | **2.7.0+cu128** | CUDA 12.8 wheel index (Isaac Lab default) |
| **NumPy** | **1.26.0** | Isaac Lab requires `numpy<2` |

Isaac Lab core declares compatibility with Isaac Sim **4.5 / 5.0 / 5.1** in its package metadata; this project is currently verified on **Isaac Sim 5.1 + Isaac Lab 0.47.2**.

---

## Isaac Lab pinned dependencies (from `~/IsaacLab/source/isaaclab/setup.py`)

These are installed automatically by `./isaaclab.sh -i` or `pip install -e ~/IsaacLab/source/isaaclab`:

| Package | Pinned version |
|---------|----------------|
| `numpy` | `<2` |
| `torch` | `>=2.7` (cu128 index) |
| `gymnasium` | `==1.2.0` |
| `prettytable` | `==3.3.0` |
| `hidapi` | `==0.14.0.post2` |
| `pillow` | `==11.3.0` |
| `starlette` | `==0.45.3` |
| `flatdict` | `==4.0.1` |
| `pyglet` | `<2` |

Also required (unpinned in Isaac Lab, pinned here after verification):

| Package | Verified in `metagrasp` |
|---------|-------------------------|
| `warp-lang` | 1.14.0 |
| `trimesh` | 4.5.1 |
| `transformers` | (Isaac Lab dependency) |

---

## MetaSQGrasp extra packages (`requirements.txt` → `metagrasp`)

Install after Isaac Lab setup:

```bash
conda activate metagrasp
cd metagrasp/MetaSQGrasp
pip install -r requirements.txt
```

| Package | Verified | Purpose |
|---------|----------|---------|
| `opencv-python` | (env-specific) | Image I/O in tools |
| `pandas` | (env-specific) | Dataset utilities |
| `trimesh` | 4.5.1 | Mesh / grasp geometry |
| `transforms3d` | (env-specific) | Pose transforms in `grasp_sampler/` |
| `python-fcl` | (env-specific) | Collision checking |
| `matplotlib` | (env-specific) | Plotting |
| `h5py` | (env-specific) | Classical grasp HDF5 |
| `warp-lang` | 1.14.0 | State-machine kernels in `isaac_env/` |
| `open3d` | 0.19.0 | Partial-PC / debug visualization |

### SQ runtime inside Isaac (`supersq_grasp-v0`)

The sim task imports `sq_estimation_model/training/` at runtime. Ensure these are available in **`metagrasp`**:

| Package | Verified |
|---------|----------|
| `omegaconf` | 2.3.0 |
| `scikit-learn` | (for PC preprocessing) |
| `open3d` | 0.19.0 |
| `trimesh` | 4.5.1 |

---

## Training stack (`rgbsq_grasp`)

See [sq_estimation_model/training/requirements.txt](../sq_estimation_model/training/requirements.txt):

```bash
conda activate rgbsq_grasp
cd sq_estimation_model/training
pip install -r requirements.txt
# Install PyTorch separately for your GPU (CUDA 12.8+ wheels recommended)
```

Key packages: `wandb`, `omegaconf`, `scikit-learn`, `scipy`, `open3d`, `tqdm`, `PyYAML`.

---

## Known pip resolver warnings (usually non-blocking)

After `./isaaclab.sh -i`, pip may report conflicts between newer packages and `isaacsim-kernel 5.1.0.0`:

| Package | Installed (verified working) | `isaacsim-kernel` expects |
|---------|------------------------------|---------------------------|
| `psutil` | 7.2.2 | `==5.9.8` |
| `click` | 8.4.1 | `==8.1.7` |

These warnings appeared during setup but **did not block** `supersq_grasp-v0` in testing. Downgrade only if you hit a runtime error tied to these packages.

---

## Setup procedure (simulation)

### 1. Install Isaac Lab into `metagrasp`

```bash
cd ~/IsaacLab
conda activate metagrasp

# Install / refresh all Isaac Lab extensions
./isaaclab.sh -i

# Wire conda activation hooks (Isaac Sim env vars, ISAACLAB_PATH)
./isaaclab.sh --conda metagrasp
```

### 2. Fix broken editable paths after moving Isaac Lab

If `import isaaclab` fails or points to an old folder (e.g. `DSQNet/IsaacLab`):

```bash
conda activate metagrasp
pip install "flatdict==4.0.1" --no-build-isolation   # if flatdict build fails
pip uninstall -y isaaclab
pip install -e ~/IsaacLab/source/isaaclab
python -c "import isaaclab; print(isaaclab.__file__)"
```

Expected output:

```text
/home/<user>/IsaacLab/source/isaaclab/isaaclab/__init__.py
```

### 3. Source Isaac Sim env vars (if hooks missing)

If `conda activate metagrasp` alone is not enough:

```bash
export ISAACLAB_PATH=~/IsaacLab
source ~/IsaacLab/_isaac_sim/setup_conda_env.sh
```

### 4. Install MetaSQGrasp extras

```bash
cd /path/to/MetaSQGrasp_public/metagrasp/MetaSQGrasp
pip install -r requirements.txt
```

### 5. Verify before `run.py`

```bash
python -c "import isaaclab; from isaaclab.app import AppLauncher; print('ok')"
```

---

## URDF conversion note

`urdf_converter.py` in this repo was originally written for **Isaac Lab 1.4 + Isaac Sim 4.2**. On Isaac Sim **5.1**, conversion may still work but is less tested than the main sim tasks. If conversion fails, use a legacy Isaac Lab 1.4 environment or pre-converted `orbit_obj.usd` under `UR_grasp_database/<id>/`.

---

## Quick version check command

```bash
conda activate metagrasp
python - <<'PY'
import isaaclab, isaacsim, torch, numpy
print("isaaclab:", isaaclab.__file__)
print("isaacsim:", isaacsim.__file__)
print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
PY
pip list | grep -iE "^isaaclab|^isaacsim |^torch |^numpy |^gymnasium |^flatdict |^warp-lang |^open3d |^trimesh "
```
