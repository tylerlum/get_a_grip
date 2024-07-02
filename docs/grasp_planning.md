# Grasp Planning

Grasp planning process:

```
1. A large batch of candidate grasps T is drawn from a sampler.
2. An initial evaluation culls all but the top K grasps by the probability of grasp success (PGS), resulting in a smaller set S, which is a subset of T.
3. All grasps in S are iteratively refined, and the final grasp executed is G*, which is the grasp in S with the highest PGS.
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip_release/assets/26510814/c1fde4fb-b7d8-42de-bdb5-91f842b5be8f" alt="grasp_planning" style="width:50%;">
</p>

## BPS Sampler

```
python nerf_grasping/sim_eval_scripts/dexdiffuser.py \
--output_folder sim_evals/frogger/grasp_config_dicts \
--nerfcheckpoints_path data/NEW_DATASET/nerfcheckpoints \
--num_grasps 5
```

## BPS Sampler + BPS Evaluator

```
python nerf_grasping/sim_eval_scripts/dexdiffuser_bps.py \
--output_folder sim_evals/dexdiffuser_bps/grasp_config_dicts \
--nerfcheckpoints_path data/NEW_DATASET/nerfcheckpoints \
--num_grasps 5
```

## BPS Sampler + NeRF Evaluator

```
python nerf_grasping/sim_eval_scripts/dexdiffuser_gg.py \
--output_folder sim_evals/dexdiffuser_gg/grasp_config_dicts \
--nerfcheckpoints_path data/NEW_DATASET/nerfcheckpoints \
--num_grasps 5
```

## Fixed Sampler

```
python nerf_grasping/sim_eval_scripts/offline_dataset.py \
--output_folder sim_evals/offline_dataset/grasp_config_dicts \
--nerfcheckpoints_path data/NEW_DATASET/nerfcheckpoints \
--num_grasps 5
```

## Fixed Sampler + BPS Evaluator

```
python nerf_grasping/sim_eval_scripts/offline_dataset_bps.py \
--output_folder sim_evals/offline_dataset_bps/grasp_config_dicts \
--nerfcheckpoints_path data/NEW_DATASET/nerfcheckpoints \
--num_grasps 5
```

## Fixed Sampler + NeRF Evaluator

```
python nerf_grasping/sim_eval_scripts/offline_dataset_gg.py \
--output_folder sim_evals/offline_dataset_gg/grasp_config_dicts \
--nerfcheckpoints_path data/NEW_DATASET/nerfcheckpoints \
--num_grasps 5
```
