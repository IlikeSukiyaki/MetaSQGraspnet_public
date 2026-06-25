# supersq_grasp_data

Synthetic scene exports from the `SQ-v0-Data` task. Most collected data stays local; one curated multi-object episode is shipped in this public repo.

## Committed demo

```text
multi_obj_full/011/epi_000000/
```

- **~46 MB** full export (dual cameras, depth, instance masks, partial PC)
- **4-object** cluttered table; target object **011**
- Object pool: **000, 001, 004, 011** (same as `UR_grasp_database/` smoke-test set)

Visualize all object partial point clouds:

```bash
cd metagrasp/MetaSQGrasp
conda activate metagrasp
python tools/visualize_partial_pc.py \
  --sample datasets/supersq_grasp_data/multi_obj_full/011/epi_000000 \
  --source extract --objects all --camera topdown
```

## Collect more data locally

Full multi-object collection guide: [`isaac_env/supersq_grasp_data/multi-object-collection.md`](../../isaac_env/supersq_grasp_data/multi-object-collection.md)

Quick example (4 objects, full export):

```bash
python run.py --task SQ-v0-Data --num_envs 1 --enable_cameras --headless \
  --save_camera_data --export_profile full \
  --obj_ids 000,001,004,011 --scene_objects 4 \
  --output_dir datasets/supersq_grasp_data/multi_obj_full \
  --max_episodes 1 --topdown_height 2.1
```

See also [`../README.md`](../README.md) for lite vs full export profiles and `.npy` conversion.
