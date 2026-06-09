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
| `/home/sustech/桌面/DSQNet/models_ifl/` | 完整 IFL 物体语料（上游只读源） |
| `metagrasp/MetaIsaacGrasp/UR_grasp_database/` | UR10e + Hand-E 测试用 curated 子集 |

当前测试对象池：`001`, `004`, `010`, `016`。

每个对象目录应包含：

```text
UR_grasp_database/<id>/
├── textured.obj              # 采样用 mesh
├── textured.urdf             # USD 转换源
├── textured.mtl              # 材质（及 textures/ 等 sidecar）
├── asset_manifest.json       # setup 工具写入的拷贝记录
├── textured.obj.hdf5         # Hand-E 5 cm 抓取库（生成）
└── orbit_obj.usd             # Isaac 刚体资产（转换）
```

---

## 2. 环境准备

| 步骤 | Conda 环境 | 用途 |
|------|------------|------|
| 拷贝 / 生成 HDF5 / 可视化 | `metagrasp` | 离线 antipodal 采样 |
| URDF → USD / Isaac 运行 | `env_isaaclab` | Isaac Lab 1.4 + Isaac Sim 4.2 |

所有命令默认在仓库根目录执行：

```bash
cd /home/sustech/桌面/DSQNet/metagrasp/MetaIsaacGrasp
```

---

## 3. 完整流程（推荐顺序）

### Step 0 — 从 IFL 语料初始化 curated 库

从完整 `models_ifl` 拷贝 mesh / URDF / 材质到 `UR_grasp_database`，**跳过**已生成文件（`textured.obj.hdf5`, `orbit_obj.usd`）。

```bash
conda activate metagrasp

python tools/setup_ur_grasp_database.py \
  --source /home/sustech/桌面/DSQNet/models_ifl \
  --objects 001,004,010,016 \
  --overwrite
```

可选：用符号链接节省磁盘（不复制文件本体）：

```bash
python tools/setup_ur_grasp_database.py \
  --source /home/sustech/桌面/DSQNet/models_ifl \
  --objects 001,004,010,016 \
  --symlink --overwrite
```

---

### Step 1 — 生成 Hand-E 5 cm 抓取 HDF5

```bash
conda activate metagrasp

python tools/generate_object_grasps.py \
  --mesh_root UR_grasp_database \
  --categories 001 004 010 016 \
  --gripper_profile hand_e \
  --gripper_length 5.0 \
  --num_points 500 \
  --max_grasps 2000 \
  --pool_size 1
```

**最近一次生成结果（参考）：**

| 对象 | 写入抓取数 | 说明 |
|------|------------|------|
| `001` | 2000 | 正常 |
| `004` | 12 | 候选较少，运行时需依赖碰撞筛选 |
| `010` | 4 | 有效 antipodal 抓取很少 |
| `016` | 0 | 直径 ~7.4 cm，超出 Hand-E 5 cm 限制，预期为 0 |

---

### Step 2 — 检查 Hand-E 坐标系映射

修复后，采样器 approach（局部 +Z）应映射到加载后的 Isaac USD `hande_end` 局部 **+Y**，jaw baseline 映射到 `hande_end` 局部 **+X**。

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

需要 `env_isaaclab`（Isaac Lab 1.4）。每个对象执行一次：

```bash
conda activate env_isaaclab

python urdf_converter.py --input UR_grasp_database/001/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/004/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/010/textured.urdf --headless
python urdf_converter.py --input UR_grasp_database/016/textured.urdf --headless
```

输出文件：同目录下的 `orbit_obj.usd`。

---

### Step 4 — 可选：离线可视化抓取

```bash
conda activate metagrasp

python grasp_sampler/visualize_grasp_object.py \
  --root UR_grasp_database \
  --dataset_name . \
  --object 001 \
  --frame ee \
  --max_grasps 50 \
  --score_min 0.5
```

默认推荐 `--frame ee`，它显示执行等价的 Hand-E EE frame。若需要查看原始采样器 frame，可改成 `--frame sampler`。两者都不再使用旧 Franka 11.21 cm 偏移。

---

### Step 5 — 可选：快速检查运行时能否选到抓取

不启动 Isaac，仅验证 HDF5 加载 + 场景碰撞筛选：

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

for obj_id in ["001", "004", "010", "016"]:
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

验证 frame fix 后执行姿态是否与采样一致：

```bash
conda activate env_isaaclab

# 全测试池
python run.py --task object-v0-Classical \
  --obj_ids 001,004,010 \
  --num_envs 1 --headless --enable_cameras --saved_video \
  --max_successful_grasps 3

# 仅验证 001
python run.py --task object-v0-Classical \
  --obj_ids 001 \
  --num_envs 1 --headless --enable_cameras --saved_video \
  --max_successful_grasps 1
```

成功时终端会出现类似日志：

```text
[Classical Grasp] env=0 episode=0 attempt=0 obj=001 profile=hand_e score=1.000 align=0.92 valid_candidates=... width_cm=...
```

视频保存在 `logs/videos/object-v0-Classical/<timestamp>/`。

---

## 4. 新增测试对象

1. 将物体放入上游 `models_ifl/<id>/`（或直接从那里 seed）。
2. 运行 Step 0，把 `<id>` 加入 `--objects`。
3. 运行 Step 1–3 生成 HDF5 与 USD。
4. 运行时通过 `--obj_ids` 指定，或修改 `isaac_env/object_env_classical/element_cfg.py` 中的 `CLASSICAL_OBJ_IDS`。

```bash
python run.py --task object-v0-Classical --obj_ids 001,010 --num_envs 1 --enable_cameras
```

---

## 5. Hand-E 坐标系修复摘要

本次修复解决「采样看起来正、执行偏 ~90°」的问题：

| 环节 | 修复内容 |
|------|----------|
| `HandEProfile.gripper_pose_cm_to_ee_pose()` | baseline → EE +X，approach → EE +Y（匹配加载后的 Isaac USD） |
| `object_env_classical/wp_cfg.py` | 预接近沿 EE 局部 **+Y** 后退 |
| `visualize_grasp_object.py` | 新增 `--frame ee`，默认显示执行等价 frame |
| `tools/check_hande_frame.py` | 输出候选 remap 轴对齐并检查 active profile |
| `tools/calibrate_hande_grasp_frame.py` | 对单个 HDF5 grasp 打印候选 remap 对齐表 |
| `tools/measure_hande_end_frame.py` | 在 Isaac 中读取真实 `hande_end` / finger 轴 |

Legacy Franka 任务仍保留旧的 +Y approach 映射。

---

## 6. 常见问题

| 现象 | 处理 |
|------|------|
| `Missing antipodal HDF5` | 重新执行 Step 1 |
| `Missing USD assets` | 执行 Step 3，或确认 `orbit_obj.usd` 存在 |
| `016` 无抓取 | 正常；物体过宽，不适合 5 cm Hand-E |
| `010` 运行时 `valid_candidates` 很少 | HDF5 仅 4 个有效抓取；可增大 `--num_points` 或换物体 |
| 执行仍偏 | 确认已拉取最新代码并重新跑 Step 2 + Step 6 |
| `urdf_converter` 失败 | 使用 Isaac Lab 1.4 / Isaac Sim 4.2 的 `env_isaaclab` |

---

## 7. 相关文件

| 文件 | 作用 |
|------|------|
| `grasp_sampler/database_paths.py` | `UR_grasp_database` 路径解析 |
| `tools/setup_ur_grasp_database.py` | 从 `models_ifl` 初始化 curated 库 |
| `tools/generate_object_grasps.py` | 生成 HDF5 |
| `tools/check_hande_frame.py` | Hand-E 坐标系单元检查 |
| `grasp_sampler/gripper_profiles.py` | Hand-E / Franka profile 与 EE 转换 |
| `isaac_env/object_env_classical/` | Classical 任务配置与状态机 |
| `command.txt` | 命令速查 |
