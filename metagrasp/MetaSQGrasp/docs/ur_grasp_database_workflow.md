# UR + Hand-E 抓取数据库工作流

本文档记录 `object-v0-Classical` 任务使用的 **curated 资产库** `UR_grasp_database/` 的完整流程：从上游 IFL 物体拷贝、Hand-E 抓取生成、USD 转换，到 Isaac 仿真验证。

更详细的采样/坐标系说明见：

- [classical_grasp_sampling.md](./classical_grasp_sampling.md) — 生成与可视化 CLI
- [classical_grasp_pose_logistics.md](./classical_grasp_pose_logistics.md) — 坐标系与运行时筛选
- [grasp_retargeting.md](./grasp_retargeting.md) — 采样器 → `hande_end` 重定向与验证流程

---

## 1. 目录用途

| 路径 | 用途 |
|------|------|
| `/home/sustech/桌面/MetaSQGrasp_public/models_ifl/` | 完整 IFL 物体语料（上游只读源） |
| `metagrasp/MetaSQGrasp/UR_grasp_database/` | UR10e + Hand-E 测试用 curated 子集 |

**当前已验证对象：** `000`, `001`, `004`, `011`（2026-06 smoke test 成功）。

**默认测试池（历史）：** `001`, `004`, `010`, `016`。运行时通过 `--obj_ids` 指定任意已准备资产的对象 ID。

### 夹爪宽度配置（单一入口）

编辑 `grasp_sampler/gripper_profile_defaults.py`：

| 常量 | 当前值 | 含义 |
|------|--------|------|
| `HAND_E_WIDTH_MAX_CM` | **7.0** | 采样与运行时最大 jaw 宽度（cm） |
| `HAND_E_HDF5_SUFFIX` | `.hand_e` | 输出 HDF5 后缀 |
| `HAND_E_PREGRASP_EXTRA_CM` | `1.0` | 预抓取额外 clearance |

生成命令 **无需** 再写 `--gripper_length 5.0`；CLI 默认读取 profile 宽度。

> 说明：URDF 标称最大开口 5 cm，但 Isaac 中 finger joint 目标（0.0425 m/指）允许更宽的有效抓取；项目用 **7 cm** 作为采样/筛选上限，与当前 sim 行为一致。

每个对象目录应包含：

```text
UR_grasp_database/<id>/
├── textured.obj              # 采样用 mesh
├── textured.urdf             # USD 转换源
├── textured.mtl              # 材质（及 textures/ 等 sidecar）
├── asset_manifest.json       # setup 工具写入的拷贝记录
├── textured.obj.hand_e.hdf5  # Hand-E 抓取库（生成）
└── orbit_obj.usd             # Isaac 刚体资产（转换）
```

---

## 2. 环境准备

| 步骤 | Conda 环境 | 用途 |
|------|------------|------|
| 拷贝 / 生成 HDF5 / 可视化 | `metagrasp` | 离线 antipodal 采样 |
| URDF → USD / Isaac 运行 | `metagrasp` | Isaac Lab **0.47.2** + Isaac Sim **5.1.0**（见 [environment_versions.md](./environment_versions.md)） |

所有命令默认在仓库根目录执行：

```bash
cd /home/sustech/桌面/MetaSQGrasp_public/metagrasp/MetaSQGrasp
```

---

## 3. 完整流程（推荐顺序）

### Step 0 — 从 IFL 语料初始化 curated 库

从完整 `models_ifl` 拷贝 mesh / URDF / 材质到 `UR_grasp_database`，**跳过**已生成文件（`textured.obj.hand_e.hdf5`, `orbit_obj.usd`）。

```bash
conda activate metagrasp

python tools/setup_ur_grasp_database.py \
  --source /home/sustech/桌面/MetaSQGrasp_public/models_ifl \
  --objects 000,001,004,011 \
  --overwrite
```

可选：用符号链接节省磁盘：

```bash
python tools/setup_ur_grasp_database.py \
  --source /home/sustech/桌面/MetaSQGrasp_public/models_ifl \
  --objects 000,001,004,011 \
  --symlink --overwrite
```

---

### Step 1 — 生成 Hand-E 抓取 HDF5

```bash
conda activate metagrasp

python tools/generate_object_grasps.py \
  --mesh_root UR_grasp_database \
  --categories 000 001 004 011 \
  --gripper_profile hand_e \
  --num_points 500 \
  --max_grasps 2000 \
  --pool_size 1
```

输出：`UR_grasp_database/<id>/textured.obj.hand_e.hdf5`

---

### Step 2 — 检查 Hand-E 坐标系映射

```bash
conda activate metagrasp

python tools/check_hande_frame.py
```

期望输出：

```text
[OK] Hand-E frame check passed: baseline->EE+X, approach->EE+Y.
```

---

### Step 3 — URDF 转 physics USD

需要 `metagrasp`（Isaac Lab 0.47 + Isaac Sim 5.1）。每个对象执行一次：

```bash
conda activate metagrasp

python urdf_converter.py --input UR_grasp_database/000/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/001/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/004/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/011/textured.urdf --headless
```

输出文件：同目录下的 `orbit_obj.usd`。

---

### Step 4 — 可选：离线可视化抓取

```bash
conda activate metagrasp

python grasp_sampler/visualize_grasp_object.py \
  --root UR_grasp_database \
  --object 011 \
  --gripper_profile hand_e \
  --frame ee \
  --max_grasps 50 \
  --score_min 0.3
```

默认 `--frame ee` 显示执行等价的 `hande_end` frame。原始采样 frame 用 `--frame sampler`。

---

### Step 5 — 可选：快速检查运行时能否选到抓取

```bash
conda activate metagrasp

python - <<'PY'
import sys, numpy as np
sys.path.insert(0, ".")
from grasp_sampler.runtime import resolve_mesh_root, select_best_grasp

root = resolve_mesh_root()
print("mesh_root =", root)
pose = np.eye(4)
pose[:3, 3] = [75.0, 20.0, 8.0]  # cm，近似 classical 桌面位姿

for obj_id in ["000", "001", "004", "011"]:
    sel = select_best_grasp(
        obj_id=obj_id,
        object_pose_cm=pose,
        desk_center_m=(0.75, 0.2, 0.0),
        mesh_root=str(root),
        profile="hand_e",
    )
    if sel is None:
        print(f"{obj_id}: NONE")
    else:
        print(
            f"{obj_id}: score={sel.score:.3f} "
            f"candidates={sel.candidate_count} width={sel.width_cm:.2f} cm"
        )
PY
```

---

### Step 6 — Isaac 仿真 smoke test

```bash
conda activate metagrasp

# 单对象
python run.py --task object-v0-Classical \
  --obj_ids 011 \
  --num_envs 1 --headless --enable_cameras --saved_video \
  --max_successful_grasps 1

# 多对象
python run.py --task object-v0-Classical \
  --obj_ids 000,001,004,011 \
  --num_envs 1 --headless --enable_cameras --saved_video \
  --max_successful_grasps 4
```

成功时终端会出现类似日志：

```text
[Classical Grasp] env=0 episode=0 attempt=0 obj=011 profile=hand_e score=... align=... valid_candidates=... width_cm=...
```

视频保存在 `logs/videos/object-v0-Classical/<timestamp>/`。

---

## 4. 新增测试对象

1. 将物体放入上游 `models_ifl/<id>/`（或直接从那里 seed）。
2. 运行 Step 0，把 `<id>` 加入 `--objects`。
3. 运行 Step 1–3 生成 `textured.obj.hand_e.hdf5` 与 `orbit_obj.usd`。
4. 运行时通过 `--obj_ids` 指定（`0` 与 `000` 等价，会自动 zero-pad）。

```bash
python run.py --task object-v0-Classical --obj_ids 004 --num_envs 1 --enable_cameras
```

---

## 5. 代码结构（Classical 任务模块化）

| 路径 | 作用 |
|------|------|
| `grasp_sampler/gripper_profile_defaults.py` | Hand-E / 2F85 宽度与 HDF5 后缀（**改这里即可全局生效**） |
| `isaac_env/gripper_task_configs/hand_e.py` | Hand-E 任务常量（profile、EE 轴、相机） |
| `isaac_env/classical_shared/` | 共享 env cfg 工厂、sim 基类、scene 配置 |
| `isaac_env/object_env_classical/` | `object-v0-Classical` 薄封装 |
| `isaac_env/object_env_classical_2f85/` | `object-v1-Classical-2f85`（Robotiq 2F85，实验性） |

Hand-E 坐标系修复摘要：

| 环节 | 内容 |
|------|------|
| `HandEProfile.gripper_pose_cm_to_ee_pose()` | baseline → EE +X，approach → EE +Y |
| `object_env_classical/wp_cfg.py` | 预接近沿 `CLASSICAL_APPROACH_AXIS_EE`（默认 EE +Y）后退 |
| `visualize_grasp_object.py` | `--frame ee` 默认显示执行等价 frame |

---

## 6. 常见问题

| 现象 | 处理 |
|------|------|
| `Missing antipodal HDF5` | 重新执行 Step 1；确认存在 `textured.obj.hand_e.hdf5` |
| `Missing USD assets` | 执行 Step 3，或确认 `orbit_obj.usd` 存在 |
| 只有旧版 `textured.obj.hdf5` | 用 `--gripper_profile hand_e` 重新生成 |
| `016` 无抓取 | 物体过宽；可试 2F85 任务或换物体 |
| 执行仍偏 ~90° | 跑 `tools/check_hande_frame.py` + Step 6 |
| `urdf_converter` 失败 | 见 [environment_versions.md](./environment_versions.md)；可改用已转换的 `orbit_obj.usd` |

---

## 7. 相关文件

| 文件 | 作用 |
|------|------|
| `grasp_sampler/gripper_profile_defaults.py` | 夹爪宽度 / HDF5 后缀单一配置 |
| `grasp_sampler/database_paths.py` | `UR_grasp_database` 路径解析 |
| `tools/setup_ur_grasp_database.py` | 从 `models_ifl` 初始化 curated 库 |
| `tools/generate_object_grasps.py` | 生成 HDF5 |
| `tools/check_hande_frame.py` | Hand-E 坐标系单元检查 |
| `grasp_sampler/gripper_profiles.py` | Hand-E / 2F85 profile 与 EE 转换 |
| `isaac_env/classical_shared/` | Classical 共享 Isaac 配置 |
| `isaac_env/object_env_classical/` | Hand-E classical 任务 |
| `command.txt` | 命令速查 |
