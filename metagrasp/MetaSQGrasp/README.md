# MetaSQGrasp

Isaac Lab simulation stack for UR10e robotic grasping with superquadric shape estimation. All tasks are launched from this directory via `run.py`.

Parent repository overview: [`../../README.md`](../../README.md)

---

## Quick start

```bash
conda activate metagrasp
cd metagrasp/MetaSQGrasp   # from repo root

# Classical baseline (requires UR_grasp_database assets + HDF5)
python run.py --task object-v0-Classical --obj_ids 001 --num_envs 1 \
  --headless --enable_cameras --saved_video --max_successful_grasps 1

# SQ-based grasping (requires SQ checkpoint under sq_estimation_model/pretrained/)
python run.py --task supersq_grasp-v0 --obj_ids 004 --num_envs 1 \
  --headless --enable_cameras --max_successful_grasps 1 --save_sq_debug

# Synthetic data collection
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras --headless \
  --save_camera_data --obj_ids 001 --max_episodes 1
```

---

## Registered tasks

| Task ID | Package | Purpose |
|---------|---------|---------|
| `object-v0-Classical` | `isaac_env/object_env_classical/` | Oracle antipodal grasping (precomputed HDF5 on true mesh) |
| `object-v1-Classical-2f85` | `isaac_env/object_env_classical_2f85/` | Same with Robotiq 2F85 |
| `SQ-v0-Data` | `isaac_env/supersq_grasp_data/` | Dual-camera synthetic data export |
| `supersq_grasp-v0` | `isaac_env/supersq_grasp_env/` | Live partial PC → SQ fit → online grasp → pick-lift |

---

## Environments

| Conda env | Used for |
|-----------|----------|
| `metagrasp` | `run.py`, asset tools, SQ inference at runtime |
| `rgbsq_grasp` | `sq_estimation_model/training/train.py` (separate PyTorch stack) |

Install: [`install.md`](install.md)

---

## Project layout

```text
MetaSQGrasp/
├── run.py                    # main entry point
├── install.md                # conda envs, Isaac stack, pip deps
├── workflow.md               # end-to-end project guide
├── requirements.txt
├── urdf_converter.py         # URDF → USD for Isaac Sim
├── UR_grasp_database/        # curated objects + HDF5 + USD
├── grasp_sampler/            # antipodal sampling, runtime loading
├── sq_estimation_model/      # SQ/ExtSQ training + pretrained configs
├── isaac_env/                # all task environments
├── tools/                    # asset setup, converters, debug scripts
├── datasets/                 # runtime-collected sim data
├── models/                   # robot/gripper USD; symlink models_ifl here
└── assets/                   # demo images and GIFs
```

Each module folder also has co-located `README.md` (quick start) and topic `*.md` guides.

---

## Documentation

| Topic | Quick start | Deep dive |
|-------|-------------|-----------|
| Install and versions | — | [`install.md`](install.md) |
| End-to-end workflows | — | [`workflow.md`](workflow.md) |
| Object assets | [`UR_grasp_database/README.md`](UR_grasp_database/README.md) | [`UR_grasp_database/workflow.md`](UR_grasp_database/workflow.md) |
| Grasp sampling | [`grasp_sampler/README.md`](grasp_sampler/README.md) | [`grasp_sampler/sampling.md`](grasp_sampler/sampling.md), [`pose-logistics.md`](grasp_sampler/pose-logistics.md), [`retargeting.md`](grasp_sampler/retargeting.md) |
| CLI tools | [`tools/README.md`](tools/README.md) | [`tools/partial-pc-extraction.md`](tools/partial-pc-extraction.md) |
| Classical grasping | [`isaac_env/object_env_classical/README.md`](isaac_env/object_env_classical/README.md) | [`grasp_sampler/sampling.md`](grasp_sampler/sampling.md) |
| Data collection | [`isaac_env/supersq_grasp_data/README.md`](isaac_env/supersq_grasp_data/README.md) | [`isaac_env/supersq_grasp_data/multi-object-collection.md`](isaac_env/supersq_grasp_data/multi-object-collection.md) |
| SQ-based grasping | [`isaac_env/supersq_grasp_env/README.md`](isaac_env/supersq_grasp_env/README.md) | [`isaac_env/supersq_grasp_env/workflow.md`](isaac_env/supersq_grasp_env/workflow.md) |
| SQ training | [`sq_estimation_model/training/README.md`](sq_estimation_model/training/README.md) | — |

---

## Roadmap

| Item | Status |
|------|--------|
| Standard UR robotic grasping (`object-v0-Classical`) | Done |
| Synthetic dataset generation (`SQ-v0-Data`) | Done |
| SQ / ExtSQ estimation training | Done |
| SQ-based UR grasping (`supersq_grasp-v0`) | Done |
| SO-101 manipulator support | Planned |
| SQ-informed RL for UR | Planned |
| Real-world depth (`depth-pro/`) and segmentation (`SAM/`) at repo root | Planned |
