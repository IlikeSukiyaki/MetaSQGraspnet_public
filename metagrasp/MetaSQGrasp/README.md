# MetaSQGrasp: IsaacLab for Supervised / Reinforcement learning (Support IsaacLab 2.0 and IsaacSim 4.5)

A data generation engine and test bench for grasp learning, powered by [IsaacLab](https://github.com/isaac-sim/IsaacLab) and [MetaGraspNetv2](https://github.com/maximiliangilles/MetaGraspNet) (MGN) including: 

- Data generation `AIR-v0-Data`
- Policy evaluation `AIR-v0-Grasp`
- Teleoperation `AIR-v0-Tele`
- Reinforcement learning `AIR-v0-SB3`, `AIR-v0-SKRL`

### New features:

1. RL support with `stable baseline3` with `AIR-v0-SB3`, `AIR-v0-SKRL`
2. Support IsaacLab 2.0 and IsaacSim 4.5.0
3. Both single-shot grasp execution and continuous learning supported
4. Teleoperation environment `AIR-v0-Tele`
5. Remote grasp agent to work around the environment conflict (see [vMF-Contact](https://github.com/YitianShi/vMF-Contact)))

<div align="center">
<img src="pic/pic.png" width=500"/ >
<p>Grasp learning data collection and test bench</p>
</div>

<div align="center">
  <img src="pic/rl.gif" style="width: 500px;" />
  <p style="font-size: 12px;">Reinforcement learning with stable baseline3</p>
</div>

#### Demo video (Inference with remote agent functionality by [vMF-Contact](https://github.com/YitianShi/vMF-Contact))

(*Click to watch, all the successfully grasped objects will be put under the table.)

[![Video Title](https://img.youtube.com/vi/FSXTWSLbo68/0.jpg)](https://www.youtube.com/watch?v=FSXTWSLbo68)

#### Teleoperation

[![Video Title](https://img.youtube.com/vi/XxlxfCCyMCE/0.jpg)](https://www.youtube.com/watch?v=XxlxfCCyMCE)

Our teleoperation may also supported by vMF-Contact to reach objects:

[![Video Title](https://img.youtube.com/vi/SV-5fFmEhaA/0.jpg)](https://www.youtube.com/watch?v=SV-5fFmEhaA)

<div align="center">
<p float="left">
  <img src="pic/rgb.png" width="200" />
  <img src="pic/segmentation.png" width="200" /> 
  <img src="pic/depth.png" width="200" />
  <img src="pic/normals.png" width="200" />
</p>
<p>Captured images: RGB, Segmentation, Depth, Normals</p>
</div>

## Getting started


Make sure you already installed the Isaac-Sim in proper manner.

Install Isaac Lab following the [installation tutourial](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-sim). Please make sure that is under your home directory: `~/IsaacLab` and following environment variables are added into `.bashrc`:

```
# Isaac Sim python executable
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
# Add Isaac Lab's sh path for convenience when runing: isaaclab -p *.py
alias isaaclab=/home/{user_name}/IsaacLab/isaaclab.sh
```

## MetaGraspNet objects [available](https://github.com/maximiliangilles/MetaGraspNet/tree/master?tab=readme-ov-file)

Objects are now adapted from [models](https://nx25922.your-storageshare.de/s/9KrFffzwoTmtapR). Keep the full corpus at `/home/sustech/桌面/DSQNet/models_ifl` and seed the curated UR10e + Hand-E test set into `UR_grasp_database/<id>/` with `tools/setup_ur_grasp_database.py`. Run `isaaclab -p urdf_converter.py` to convert URDF files into USD files (!!! Please use isaaclab 1.4 version for urdf conversion since this is now out-of-date for isaaclab 2.0!). We don't use original USD files since all the collision meshes are in convex hull, which are unrealistic.

## Classical antipodal grasp task (`object-v0-Classical`)

UR10e + Robotiq Hand-E pick-and-lift smoke test using offline antipodal grasps from `UR_grasp_database/<id>/textured.obj.hand_e.hdf5`.

| Setting | Value |
|---------|-------|
| Task | `object-v0-Classical` |
| Robot / EE | `ur10e_hand_e` / `hande_end` |
| Max jaw width | **7.0 cm** — edit `grasp_sampler/gripper_profile_defaults.py` |
| HDF5 suffix | `.hand_e` |
| Verified objects | `000`, `001`, `004`, `011` |

**Docs:** [docs/ur_grasp_database_workflow.md](docs/ur_grasp_database_workflow.md) · [docs/classical_grasp_sampling.md](docs/classical_grasp_sampling.md) · [command.txt](command.txt)

Quick start (after seeding assets):

```bash
cd /home/sustech/桌面/DSQNet/metagrasp/MetaSQGrasp
conda activate metagrasp
python tools/generate_object_grasps.py \
  --mesh_root UR_grasp_database --categories 011 --gripper_profile hand_e

conda activate env_isaaclab
python urdf_converter.py --input UR_grasp_database/011/textured.urdf --headless
python run.py --task object-v0-Classical --obj_ids 011 \
  --num_envs 1 --headless --enable_cameras --saved_video --max_successful_grasps 1
```

Experimental wider-gripper task: `object-v1-Classical-2f85` (Robotiq 2F85, `textured.obj.2f85.hdf5`).

## SQ / extended SQ estimation model training (isolated)

Training for the SQ estimation model (`sq_esti_model`) and extended SQ estimation model (`extended_sq_esti_model`) lives in **`sq_estimation_model/training/`**, separate from Isaac sim code. Datasets use JSON manifests with renamed class IDs (`primitive_001` → box, etc.).

```bash
cd /home/sustech/桌面/DSQNet/metagrasp/MetaSQGrasp/sq_estimation_model/training
conda activate rgbsq_grasp

python train.py --config configs/sq_esti_model_config.yml --device 0 --run sq_esti_model_full
python train.py --config configs/extended_sq_esti_model_config.yml --device 0 --run extended_sq_esti_model_full
```

See [sq_estimation_model/training/README.md](sq_estimation_model/training/README.md) for dataset schema, smoke tests, and migration from `rgbsq_grasp/datasets/`.

## SQNet superquadric grasp task (`supersq_grasp-v0`)

`supersq_grasp-v0` combines the stable Classical UR10e + Hand-E execution stack with live SQNet perception. It uses UR_grasp_database objects, extracts an object-only top-down partial point cloud in memory, estimates a single SQNet superquadric, samples Hand-E antipodal grasps on the estimated mesh, and executes the normal pick-lift state machine.

**Full workflow:** [docs/supersq_grasp_workflow.md](docs/supersq_grasp_workflow.md)

```bash
cd /home/sustech/桌面/DSQNet/metagrasp/MetaSQGrasp
conda activate env_isaaclab
python run.py --task supersq_grasp-v0 --obj_ids 004 --num_envs 1 \
  --headless --enable_cameras --saved_video --max_successful_grasps 1
```

For a lightweight smoke test with perception artifacts:

```bash
python -u run.py --task supersq_grasp-v0 --obj_ids 004 --num_envs 1 \
  --headless --enable_cameras --max_successful_grasps 1 \
  --save_sq_debug
```

Use `--save_sq_debug` to write each attempt under `logs/sq_debug/supersq_grasp-v0/<timestamp>/`. Each attempt folder includes `partial_pc.npy`, `partial_pc.ply`, `sq_mesh_robot.ply`, `grasp_axes.ply`, and `sq_fit.json`, all in the robot-base frame except centimeter-valued grasp matrices recorded in JSON. Visualize one attempt with:

```bash
python tools/inspect_supersq_debug.py \
  --dir logs/sq_debug/supersq_grasp-v0/<timestamp>/env0_epi000001_attempt0_obj004

python tools/inspect_supersq_grasps.py \
  --dir logs/sq_debug/supersq_grasp-v0/<timestamp>/env0_epi000001_attempt0_obj004
```

The Isaac environment must be able to import `/home/sustech/桌面/DSQNet/rgbsq_grasp`; the task adds that checkout to `sys.path` and uses `rgbsq_grasp/pretrained/sqnet_config/example/model_best.pkl` by default. Make sure the Isaac Python environment has the SQNet-side runtime packages installed, especially `omegaconf`, `scikit-learn`, and `open3d`.

## SQNet dual-camera data task

`SQ-v0-Data` collects synthetic SQNet scene observations with the AIR setup's UR10e + Hand-E manipulator, object `models_ifl/001`, a fixed on-hand camera, and a second top-down camera above the table. Each stable randomized scene can be saved as raw depth/segmentation/point-cloud data and converted to `rgbsq_grasp` `.npy` samples.

Before the first run, convert the object URDF to a physics-ready USD if `/home/sustech/桌面/DSQNet/models_ifl/001/orbit_obj.usd` is missing:

```bash
cd /home/sustech/桌面/DSQNet/metagrasp/MetaSQGrasp
conda activate env_isaaclab
python urdf_converter.py --input /home/sustech/桌面/DSQNet/models_ifl/001/textured.urdf
```

Launch the scene without saving:

```bash
cd /home/sustech/桌面/DSQNet/metagrasp/MetaSQGrasp
conda activate env_isaaclab
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras
```

Collect a bounded dataset:

```bash
cd /home/sustech/桌面/DSQNet/metagrasp/MetaSQGrasp
conda activate env_isaaclab
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras \
  --save_camera_data --output_dir datasets/supersq_grasp_data \
  --max_episodes 100 --topdown_height 1.2
```

The top-down camera height is measured in meters above `desk_center`. The camera data uses the same pinhole model as AIR: 640 x 480, `focal_length_cm=1.69`, `horizontal_aperture_mm=2.59`, and modalities `rgb`, `distance_to_image_plane`, `normals`, and `instance_segmentation_fast`. Runtime intrinsics are saved from `camera.data.intrinsic_matrices`.

Dataset layout:

```text
datasets/supersq_grasp_data/
  manifest.jsonl
  001/
    epi_000001/
      meta.json
      scene.pt
      camera_onhand/
        rgb.png
        depth.npy
        depth.png
        instance.npy
        instance.png
        normals.npy
        pcd_map.npy
      camera_topdown/
        ...
```

`scene.pt` keeps the AIR-style structure with `camera_0`, `camera_1`, `obj_poses_robot`, and `obj_id`. The sidecar files are easier to inspect from NumPy/OpenCV. `pcd_map.npy` and `camera_pose` are in the robot base frame and meters. `obj_poses_robot` follows the AIR convention with translation in centimeters.

Inspect one saved sample:

```bash
python tools/inspect_sq_sample.py --sample datasets/supersq_grasp_data/001/epi_000001 --camera camera_topdown --show_pcd
```

Convert raw samples to `rgbsq_grasp` point-cloud samples:

```bash
python tools/convert_sq_to_npy.py \
  --input datasets/supersq_grasp_data/001 \
  --output ../../rgbsq_grasp/datasets/isaac_synth/001 \
  --camera both
```

Each converted `.npy` contains `partial_pc`, `membership`, a coarse box primitive for compatibility, `pcd_frame`, `source_camera`, and `source_sample`. For a quick network-side check, use:

```bash
cd /home/sustech/桌面/DSQNet/rgbsq_grasp
python supersq_inference.py --input datasets/isaac_synth/001/001_0000_viewpoint_00.npy --device cpu
```

## VSCode development

We provide the vscode debugging setup and setting file in ´.vscode´, please replace ´home/yitian´ with your home path.

## Potential Issues

If you cause core dump due to camera setting please run following command:

```
sudo prime-select nvidia
```

in case your 

```
sudo prime-select query
```

returns `on-demand`

## Citation

Please cite our paper which uses the whole framework for reference:

```
@article{shi2024vmf,
  title={vMF-Contact: Uncertainty-aware Evidential Learning for Probabilistic Contact-grasp in Noisy Clutter},
  author={Shi, Yitian and Welte, Edgar and Gilles, Maximilian and Rayyes, Rania},
  journal={arXiv preprint arXiv:2411.03591},
  year={2024}
}
```
