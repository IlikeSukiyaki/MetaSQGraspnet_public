# SQ Estimation Model Package

Isolated SQ / ExtSQ **training and inference** for MetaSQGrasp. Simulation and live grasp tasks remain under `isaac_env/` and `run.py`.

| Path | Purpose |
|------|---------|
| `training/` | Train `sq_esti_model` and `extsq_esti_model` |
| `pretrained/sq_esti_model/` | Example config + optional checkpoint for inference |
| `pretrained/extsq_esti_model/` | ExtSQ config + optional checkpoint for inference |

Environment versions (sim vs training): [install.md](../install.md)

Start here: [training/README.md](training/README.md)

---

## Runtime connection

At inference time, `isaac_env/supersq_grasp_env/sq_bridge.py` loads checkpoints from `pretrained/` (or custom paths) and calls the same inference code used during training.

Live grasp task guide: [`isaac_env/supersq_grasp_env/README.md`](../isaac_env/supersq_grasp_env/README.md)
