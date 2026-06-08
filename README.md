# Superquadrics MetaGrasp

Integration of **MetaIsaacGrasp** (Isaac Lab grasp simulation) with **DSQNet / rgbsq_grasp** (deformable superquadric shape estimation) for classical antipodal grasping, synthetic data collection, and superquadric-based grasp pose generation.

Remote repository: [superquadrics_metagrasp](https://github.com/IlikeSukiyaki/superquadrics_metagrasp)

## Directory layout

```text
superquadrics_metagrasp/          # repository root (clone target)
├── README.md                     # this file
├── models_ifl/                   # [local only] MetaGraspNet IFL object meshes + HDF5 grasps
├── models_ifl.zip                # [local only] optional archive of models_ifl
├── IsaacLab/                     # [local only] NVIDIA Isaac Lab (third-party)
├── metagrasp/
│   └── MetaIsaacGrasp/           # Isaac Sim environments, grasp sampler, tools
│       ├── run.py                # entry point for AIR / SQ / Classical tasks
│       ├── grasp_sampler/        # antipodal sampling + runtime collision filter
│       ├── isaac_env/            # task packages (Classical, Grasp, SQ-v0-Data, …)
│       ├── tools/                # offline utilities (generate grasps, convert SQ data)
│       ├── docs/                 # classical grasp guides
│       ├── models/               # robot / table / gripper USD assets (tracked)
│       │   └── models_ifl -> ../../../models_ifl   # symlink; create after cloning
│       └── datasets/             # [local only] collected Isaac episodes
└── rgbsq_grasp/                  # DSQNet network code + grasp pose generation
    ├── supersq_inference.py
    ├── grasp_pose_generation.py
    ├── datasets/                 # [local only] training / eval / Isaac-synth samples
    ├── pretrained/               # [local only] DSQNet checkpoints
    └── train_results/            # [local only] training logs and checkpoints
```

Large directories are **not** stored in git. Each excluded path has a short `README.md` (where applicable) explaining what to place there after cloning.

## Quick start after cloning

### 1. Isaac Lab (required for simulation)

Install Isaac Sim and Isaac Lab following the [official pip installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

Recommended layout (sibling to this repo):

```text
~/projects/
├── superquadrics_metagrasp/    # this repository
└── IsaacLab/                   # clone from https://github.com/isaac-sim/IsaacLab
```

Create a conda env (e.g. `env_isaaclab`) with Python 3.10+ as required by your Isaac Sim version. Point `ISAACSIM_PATH` and Isaac Lab paths in your shell profile as described in the Isaac Lab docs.

This repo expects Isaac Lab to be available on `PYTHONPATH` when running `metagrasp/MetaIsaacGrasp/run.py`.

### 2. MetaGraspNet object assets (`models_ifl/`)

Download IFL object models from the [MetaGraspNet storage](https://nx25922.your-storageshare.de/s/9KrFffzwoTmtapR) (or use your existing `models_ifl.zip`).

Place the unpacked folder at the **repository root**:

```text
superquadrics_metagrasp/models_ifl/
├── 000/
│   ├── textured.obj
│   ├── textured.urdf
│   └── …
├── 001/
│   └── …
└── …
```

Wire it into MetaIsaacGrasp:

```bash
cd metagrasp/MetaIsaacGrasp/models
ln -sfn ../../../models_ifl models_ifl
```

For object `001`, convert URDF to simulation USD (once per object):

```bash
cd metagrasp/MetaIsaacGrasp
conda activate env_isaaclab
python urdf_converter.py --input ../../models_ifl/001/textured.urdf
# creates models_ifl/001/orbit_obj.usd
```

Generate antipodal grasp HDF5 (once per object):

```bash
conda activate metagrasp   # or another env with trimesh / h5py
python tools/generate_object_grasps.py \
  --mesh_root ../../models_ifl \
  --categories 001 \
  --gripper_profile hand_e \
  --num_points 500 \
  --max_grasps 2000
# creates models_ifl/001/textured.obj.hdf5
```

See `metagrasp/MetaIsaacGrasp/docs/classical_grasp_sampling.md` for the full classical grasp pipeline.

### 3. MetaIsaacGrasp datasets (`metagrasp/MetaIsaacGrasp/datasets/`)

Created at runtime when collecting data. Example SQNet collection:

```bash
cd metagrasp/MetaIsaacGrasp
conda activate env_isaaclab
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras \
  --save_camera_data --output_dir datasets/sqnet_isaac --max_episodes 100
```

Output layout is documented in `metagrasp/MetaIsaacGrasp/README.md` and `metagrasp/MetaIsaacGrasp/datasets/README.md`.

### 4. rgbsq_grasp assets

| Path | Purpose | Source |
|------|---------|--------|
| `rgbsq_grasp/datasets/` | Training, evaluation, and Isaac-synth `.npy` samples | [DSQNet Google Drive datasets](https://drive.google.com/drive/folders/1PQ9dSeD0WmdESQemsnM1SPmpPDChQ95s) or convert from `datasets/sqnet_isaac` via `tools/convert_sq_to_npy.py` |
| `rgbsq_grasp/pretrained/` | DSQNet checkpoints | [Pretrained models Drive folder](https://drive.google.com/drive/folders/1PN7DF0iNL60iOuyA-QS2g7jMzXSOPD6a) |
| `rgbsq_grasp/train_results/` | Training outputs | Created by `train.py` |

Install the rgbsq_grasp env:

```bash
conda create -n rgbsq_grasp python=3.10 -y
conda activate rgbsq_grasp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r rgbsq_grasp/requirements.txt
```

See `rgbsq_grasp/README.md` and `rgbsq_grasp/IMPLEMENTATION.md` for details.

## Main tasks

### Classical antipodal grasp (smoke test)

```bash
cd metagrasp/MetaIsaacGrasp
conda activate env_isaaclab
python run.py --task object-v0-Classical --num_envs 1 --enable_cameras
```

Requires `models_ifl/001/{textured.obj, textured.obj.hdf5, orbit_obj.usd}`.

### SQNet data collection → rgbsq_grasp inference

```bash
# 1) Collect in Isaac (see section 3)
# 2) Convert to .npy
python tools/convert_sq_to_npy.py \
  --input datasets/sqnet_isaac/001 \
  --output ../../rgbsq_grasp/datasets/isaac_synth/001 \
  --camera both

# 3) Run DSQNet inference
cd ../../rgbsq_grasp
python supersq_inference.py \
  --input datasets/isaac_synth/001/001_0000_viewpoint_00.npy --device cpu
```

Command cheatsheet: `metagrasp/MetaIsaacGrasp/command.txt`.

## Conda environments (recommended)

| Env | Used for |
|-----|----------|
| `env_isaaclab` | Isaac Sim / MetaIsaacGrasp simulation |
| `metagrasp` | Offline grasp sampling (`tools/generate_object_grasps.py`) |
| `rgbsq_grasp` | DSQNet training and inference |

## Submodule notes

This repository is a **monorepo**. It extends [MetaIsaacGrasp](https://github.com/YitianShi/MetaIsaacGrasp) with classical grasp runtime, Hand-E gripper profiles, SQ-v0-Data, and rgbsq_grasp integration. Upstream MetaIsaacGrasp history is not preserved as a separate submodule.

## License

See `metagrasp/MetaIsaacGrasp/LICENSE` and third-party notices under `metagrasp/MetaIsaacGrasp/doc/`. rgbsq_grasp follows the original DSQNet license terms described in its README.
