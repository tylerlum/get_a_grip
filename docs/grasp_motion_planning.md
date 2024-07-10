# Grasp Motion Planning

Grasp motion planning process:

```
1. Grasps are planned using a grasp planner, which outputs "floating hand" grasps
2. Approach trajectories are planned using a motion planner, which outputs "arm + hand" grasps
```

## BPS Sampler + BPS Evaluator + BPS Random Sampling Optimizer

```
python get_a_grip/grasp_motion_planning/scripts/run_grasp_motion_planning.py \
--grasp_planner.output_folder data/grasp_planning_outputs \
--grasp_planner.overwrite True \
--grasp_planner.nerf.nerf_is_z_up False \
--grasp_planner.nerf.nerf_config data/NEW_DATASET/nerfcheckpoints/mug_0_1000/nerfacto/2024-05-11_212418/config.yml \
grasp_planner.planner.sampler:bps-sampler-config \
  --grasp_planner.planner.sampler.ckpt_path data/trained_models/bps_sampler_model/20240710140203/ckpt_1.pth \
  --grasp_planner.planner.sampler.num_grasps 3200 \
grasp_planner.planner.evaluator:bps-evaluator-config \
  --grasp_planner.planner.evaluator.ckpt_path data/trained_models/bps_evaluator_model/20240710_140229/ckpt-emrvib91-step-80.pth \
grasp_planner.planner.optimizer:bps-random-sampling-optimizer-config \
  --grasp_planner.planner.optimizer.num_grasps 32
```

## Fixed Sampler + NeRF Evaluator + NeRF Gradient Optimizer

```
python get_a_grip/grasp_motion_planning/scripts/run_grasp_motion_planning.py \
--output_folder data/grasp_planning_outputs \
--overwrite True \
--nerf.nerf_is_z_up False \
--nerf.nerf_config data/NEW_DATASET/nerfcheckpoints/mug_0_1000/nerfacto/2024-05-11_212418/config.yml \
planner.sampler:fixed-sampler-config \
  --planner.sampler.fixed_grasp_config_dict_path data/fixed_sampler_grasp_config_dicts/all_good_grasps.npy  \
planner.evaluator:nerf-evaluator-config \
  --planner.evaluator.nerf-evaluator-model-config-path data/trained_models/nerf_evaluator_model/MY_NERF_EXPERIMENT_NAME_2024-07-10_14-04-32-950499/config.yaml \
planner.optimizer:nerf-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```
