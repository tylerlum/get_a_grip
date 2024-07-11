# Grasp Planning

Grasp planning process:

```
1. A large batch of candidate grasps T is drawn from a sampler.
2. An initial evaluation culls all but the top K grasps sorted by the probability of grasp success (y_PGS), resulting in a smaller set S, which is a subset of T.
3. All grasps in S are iteratively refined, and the final grasp executed is G*, which is the grasp in S with the highest PGS after refinement.
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip_release/assets/26510814/c1fde4fb-b7d8-42de-bdb5-91f842b5be8f" alt="grasp_planning" style="width:50%;">
</p>

## NeRF Frames

In isaacgym, Y is the up direction. Thus all grasps and NeRFs from simulated data are specified in y-up coordinates. However, in some situations, Z will be the up direction. To convert between the two, we need to rotate the NeRFs by 90 degrees about the x-axis. Please be sure to correctly specify the `nerf_is_z_up` flag when running the grasp planning code.

## Create Fixed Sampler Grasp Config Dict

All good grasps:

```
python get_a_grip/grasp_planning/scripts/create_fixed_sampler_grasp_config_dict.py \
--input-evaled-grasp-config-dicts-path data/TINY_DATASET/final_evaled_grasp_config_dicts_train \
--output-grasp-config-dict-path data/fixed_sampler_grasp_config_dicts/all_good_grasps.npy \
--y_PGS_threshold 0.9 \
--max_grasps_per_object None \
--overwrite True
```

One good grasp per object:

```
python get_a_grip/grasp_planning/scripts/create_fixed_sampler_grasp_config_dict.py \
--input-evaled-grasp-config-dicts-path data/TINY_DATASET/final_evaled_grasp_config_dicts_train \
--output-grasp-config-dict-path data/fixed_sampler_grasp_config_dicts/one_good_grasp_per_object.npy \
--y_PGS_threshold 0.9 \
--max_grasps_per_object 1 \
--overwrite True
```

## BPS Sampler + BPS Evaluator + BPS Random Sampling Optimizer

```
python get_a_grip/grasp_planning/scripts/run_grasp_planning.py \
--output_folder data/grasp_planning_outputs \
--overwrite True \
--nerf.nerf_is_z_up False \
--nerf.nerf_config data/NEW_DATASET/nerfcheckpoints/mug_0_1000/nerfacto/2024-05-11_212418/config.yml \
planner.sampler:bps-sampler-config \
  --planner.sampler.ckpt_path data/trained_models/bps_sampler_model/20240710140203/ckpt_1.pth \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:bps-evaluator-config \
  --planner.evaluator.ckpt_path data/trained_models/bps_evaluator_model/20240710_140229/ckpt-emrvib91-step-80.pth \
planner.optimizer:bps-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```

## BPS Sampler + No Evaluator + No Optimizer

```
python get_a_grip/grasp_planning/scripts/run_grasp_planning.py \
--output_folder data/grasp_planning_outputs \
--overwrite True \
--nerf.nerf_is_z_up False \
--nerf.nerf_config data/NEW_DATASET/nerfcheckpoints/mug_0_1000/nerfacto/2024-05-11_212418/config.yml \
planner.sampler:bps-sampler-config \
  --planner.sampler.ckpt_path data/trained_models/bps_sampler_model/20240710140203/ckpt_1.pth \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:no-evaluator-config \
planner.optimizer:no-optimizer-config
```

## NeRF Sampler + NeRF Evaluator + NeRF Random Sampling Optimizer

```
python get_a_grip/grasp_planning/scripts/run_grasp_planning.py \
--output_folder data/grasp_planning_outputs \
--overwrite True \
--nerf.nerf_is_z_up False \
--nerf.nerf_config data/NEW_DATASET/nerfcheckpoints/mug_0_1000/nerfacto/2024-05-11_212418/config.yml \
planner.sampler:nerf-sampler-config \
  --planner.sampler.ckpt_path data/trained_models/nerf_sampler_model/20240710135846/ckpt_100.pth  \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:nerf-evaluator-config \
  --planner.evaluator.nerf-evaluator-model-config-path data/trained_models/nerf_evaluator_model/MY_NERF_EXPERIMENT_NAME_2024-07-10_14-04-32-950499/config.yaml \
planner.optimizer:nerf-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```

## Fixed Sampler + NeRF Evaluator + NeRF Gradient Optimizer

```
python get_a_grip/grasp_planning/scripts/run_grasp_planning.py \
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
