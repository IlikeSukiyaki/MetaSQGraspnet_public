# Simulation logs

Runtime outputs from Isaac tasks land here. Most paths are **gitignored**; a small set of curated demo artifacts is tracked for smoke testing and documentation.

## Subfolders

| Folder | Written by | Contents |
|--------|------------|----------|
| `sq_debug/` | `supersq_grasp-v0` | Partial PC, SQ fit JSON, grasp axes, observation dumps |
| `videos/` | Grasp tasks with `--save_video` | Multi-view montage MP4s |
| `urdf_conversion/` | `urdf_converter.py` batch runs | Per-object conversion logs |

## In-repo demo paths

These paths are whitelisted in the root [`.gitignore`](../../../.gitignore):

| Path | What it shows |
|------|----------------|
| `sq_debug/supersq_grasp-v0/2026-06-25_16-40-32/env0_epi000000_attempt0_obj011/` | SQ grasp debug bundle for object **011** |
| `urdf_conversion/2026-06-09_21-20-10/` | Batch URDF→USD conversion logs |
| `videos/object-v0-Classical/2026-06-09_16-51-45/grasp_success_env0_epi1_obj000_step0_montage.mp4` | Classical grasp montage (obj **000**) |
| `videos/supersq_grasp-v0/2026-06-10_17-19-43/grasp_success_env0_epi0_obj001_step0_montage.mp4` | SQ grasp montage (obj **001**) |

## Reproduce demos

From `metagrasp/MetaSQGrasp/` with `conda activate metagrasp`:

```bash
# Classical grasp video
python run.py --task object-v0-Classical --num_envs 1 --enable_cameras \
  --obj_id 000 --save_video --headless

# SQ grasp video
python run.py --task supersq_grasp-v0 --num_envs 1 --enable_cameras \
  --obj_id 001 --sq_model sq_esti_model --save_video --headless

# SQ debug dump (writes under logs/sq_debug/)
python run.py --task supersq_grasp-v0 --num_envs 1 --enable_cameras \
  --obj_id 011 --sq_model sq_esti_model --save_sq_debug --headless
```

See task guides under [`isaac_env/`](../isaac_env/) for full CLI options.
