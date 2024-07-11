# Visualization Tools

## Debugging Generate Hand Config Dicts

Add `--use_wandb` to log to wandb during the optimization, which allows you to view plots of energy vs. iteration and visualize the grasps on wandb.

## Debugging Eval Grasp Config Dicts (Isaac Validator)

### Debug Prints

Change the `DEBUG` flag in `eval_grasp_config_dict.py` and/or `isaac_validator.py` to get more detailed debug info.

### Sim Visualization with GUI

You can visualize all grasp simulations in a GUI like so:

```
python get_a_grip/dataset_generation/scripts/eval_grasp_config_dict.py \
--meshdata_root_path data/large/meshes \
--input_grasp_config_dicts_path data/NEW_DATASET/grasp_config_dicts \
--output_evaled_grasp_config_dicts_path None \
--object_code_and_scale_str mug_0_1000 \
--max_grasps_per_batch 5000 \
--use_gui True \
--save_to_file False
```

You can visualize a specific grasp simulation in a GUI like so:

```
python get_a_grip/dataset_generation/scripts/eval_grasp_config_dict.py \
--meshdata_root_path data/large/meshes \
--input_grasp_config_dicts_path data/NEW_DATASET/grasp_config_dicts \
--output_evaled_grasp_config_dicts_path None \
--object_code_and_scale_str mug_0_1000 \
--max_grasps_per_batch 5000 \
--debug_index 0 \
--use_gui True \
--save_to_file False
```

When GUI is on, press space to pause/start, s to toggle step mode where each space press takes one sim step forward. Add `--start_with_step_mode True` to start with step mode, which is useful for detailed debugging and analysis.

Note: You can move fingers back first at pregrasp (different grasp strategy) by adding `--move_fingers_back_at_init True` to this evaluation command.

### Sim Visualization with Output Video Files

You can visualize specific simulations with output video files like so:

```
python get_a_grip/dataset_generation/scripts/eval_grasp_config_dict.py \
--meshdata_root_path data/large/meshes \
--input_grasp_config_dicts_path data/NEW_DATASET/grasp_config_dicts \
--output_evaled_grasp_config_dicts_path None \
--object_code_and_scale_str mug_0_1000 \
--max_grasps_per_batch 5000 \
--save_to_file False \
--record_indices 0 1 2 3 4 5 6 7 8 9
```

### Visualize Grasps with Plotly

Visualize one grasp on one object:

```
python get_a_grip/visualization/scripts/visualize_config_dict.py \
--meshdata_root_path data/large/meshes \
--input_config_dicts_path data/NEW_DATASET/grasp_config_dicts \
--object_code_and_scale_str mug_0_1000 \
--idx_to_visualize 0
```

Visualize multiple grasps on one object:

```
python get_a_grip/visualization/scripts/visualize_config_dict.py \
--meshdata_root_path data/large/meshes \
--input_config_dicts_path data/NEW_DATASET/grasp_config_dicts \
--object_code_and_scale_str mug_0_1000 \
--visualize_all True
```

Visualize the optimization of one grasp on one object (must have run stored mid optimization grasps during `generate_hand_config_dicts.py` with `--store_grasps_mid_optimization_freq 25` or `--store_grasps_mid_optimization_iters 7 11 15`, then run `generate_grasp_config_dicts.py` and `eval_all_grasp_config_dicts.py` with `--all_mid_optimization_steps True`):

```
python get_a_grip/visualization/scripts/visualize_config_dict_optimization.py \
--meshdata_root_path data/large/meshes \
--input_config_dicts_mid_optimization_path data/NEW_DATASET/evaled_grasp_config_dicts/mid_optimization \
--object_code_and_scale_str mug_0_1000 \
--idx_to_visualize 0
```

### Visualize Meshes with Plotly

Visualize multiple meshes files (very useful when looking at meshes generated by nerf):

```
python get_a_grip/visualization/scripts/visualize_objs.py \
--meshdata_root_path data/large/meshes
```

### Visualize Meshes with Isaacgym

Visualize the meshes in isaacgym like so:

```
python get_a_grip/visualization/scripts/visualize_objs_in_isaacgym.py \
--meshdata_root_path data/large/meshes
```

## Visualizing NeRFs

Run nerfstudio's interactive viewer:

```
ns-viewer --load-config data/NEW_DATASET/nerfcheckpoints/mug_0_1000/nerfacto/2024-05-11_212418/config.yml
```