# Grasp Motion Planning

Follow these instructions if you want to run grasp motion planning (arm + hand). To do this, you will need trained models like the following:

```
data/models/pretrained
├── bps_evaluator_model
├── bps_sampler_model
├── nerf_evaluator_model
└── nerf_sampler_model
```

and at least one fixed sampler grasp config dict like the following:

```
data/fixed_sampler_grasp_config_dicts/given
├── all_good_grasps.npy
└── one_good_grasp_per_object.npy
```

Grasp motion planning process:

```
1. Grasps are planned using a grasp planner, which outputs "floating hand" grasps
2. Approach trajectories are planned using a motion planner, which outputs "arm + hand" grasps
```

The grasp motion planning code will output a set of grasps in `data/grasp_motion_planning_outputs/...`. It also visualizes the planned grasps and motion plan in an interactive loop, where the user can select the grasp index to visualize and plot it using the terminal.

## Frames

In isaacgym, Y is the up direction. Thus all grasps and NeRFs from simulated data are specified in y-up coordinates. However, in some situations, Z will be the up direction. To convert between the two, we need to rotate the NeRFs by 90 degrees about the x-axis.

Please be sure to correctly specify the `--nerf_is_z_up` flag when running the grasp planning code. For all the simulated nerfs, we assume that the NeRFs are in y-up coordinates since they were generated using data from isaacgym. For all the real world nerfs, we assume that the NeRFs are in z-up coordinates.

First run the following (change as needed):
```
export DATASET_NAME=nano

export BPS_SAMPLER=data/models/pretrained/bps_sampler_model/ckpt.pth
export BPS_EVALUATOR=data/models/pretrained/bps_evaluator_model/ckpt.pth
export NERF_SAMPLER=data/models/pretrained/nerf_sampler_model/ckpt.pth
export NERF_EVALUATOR=data/models/pretrained/nerf_evaluator_model/TODO_EXPERIMENT_NAME/config.yaml
export FIXED_SAMPLER_GRASPS=data/fixed_sampler_grasp_config_dicts/given/all_good_grasps.npy

export OBJECT_CODE_AND_SCALE_STR=core-mug-5c48d471200d2bf16e8a121e6886e18d_0_0622
export NERF_CONFIG=data/dataset/${DATASET_NAME}/nerfcheckpoints/core-mug-5c48d471200d2bf16e8a121e6886e18d_0_0622/nerfacto/2024-07-13_111325/config.yml
```

## BPS Sampler + BPS Evaluator + BPS Random Sampling Optimizer

```
python get_a_grip/grasp_motion_planning/scripts/run_grasp_motion_planning.py \
--visualize_loop True \
--output_folder data/grasp_motion_planning_outputs/bps_sampler_bps_evaluator \
--overwrite True \
--nerf.nerf_is_z_up False \
--nerf.nerf_config ${NERF_CONFIG} \
planner.sampler:bps-sampler-config \
  --planner.sampler.ckpt_path ${BPS_SAMPLER} \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:bps-evaluator-config \
  --planner.evaluator.ckpt_path ${BPS_EVALUATOR} \
planner.optimizer:bps-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```

## BPS Sampler + No Evaluator + No Optimizer

```
python get_a_grip/grasp_motion_planning/scripts/run_grasp_motion_planning.py \
--visualize_loop True \
--output_folder data/grasp_motion_planning_outputs/bps_sampler \
--overwrite True \
--nerf.nerf_is_z_up False \
--nerf.nerf_config ${NERF_CONFIG} \
planner.sampler:bps-sampler-config \
  --planner.sampler.ckpt_path ${BPS_SAMPLER} \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:no-evaluator-config \
planner.optimizer:no-optimizer-config
```

## NeRF Sampler + NeRF Evaluator + NeRF Random Sampling Optimizer

```
python get_a_grip/grasp_motion_planning/scripts/run_grasp_motion_planning.py \
--visualize_loop True \
--output_folder data/grasp_motion_planning_outputs/nerf_sampler_nerf_evaluator \
--overwrite True \
--nerf.nerf_is_z_up False \
--nerf.nerf_config ${NERF_CONFIG} \
planner.sampler:nerf-sampler-config \
  --planner.sampler.ckpt_path ${NERF_SAMPLER}  \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:nerf-evaluator-config \
  --planner.evaluator.nerf-evaluator-model-config-path ${NERF_EVALUATOR} \
planner.optimizer:nerf-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```

## Fixed Sampler + NeRF Evaluator + NeRF Random Sampling Optimizer

```
python get_a_grip/grasp_motion_planning/scripts/run_grasp_motion_planning.py \
--visualize_loop True \
--output_folder data/grasp_motion_planning_outputs/fixed_sampler_nerf_evaluator  \
--overwrite True \
--nerf.nerf_is_z_up False \
--nerf.nerf_config ${NERF_CONFIG} \
planner.sampler:fixed-sampler-config \
  --planner.sampler.fixed_grasp_config_dict_path ${FIXED_SAMPLER_GRASPS}  \
planner.evaluator:nerf-evaluator-config \
  --planner.evaluator.nerf-evaluator-model-config-path ${NERF_EVALUATOR} \
planner.optimizer:nerf-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```
