# Model Training

## HDF5 File Generation

BPS dataset:

```
python nerf_grasping/dexdiffuser/create_grasp_bps_dataset.py \
--config-dict-folder data/NEW_DATASET/evaled_grasp_config_dicts \
--output-filepath data/NEW_DATASET/bps_dataset/dataset.h5
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip_release/assets/26510814/6da75267-280c-4e3a-921b-79b765842ab9" alt="dataset_gen" style="width:50%;">
</p>

Grid dataset for NeRF inputs:

```
python nerf_grasping/dataset/Create_DexGraspNet_NeRF_Grasps_Dataset.py grid \
--evaled-grasp-config-dicts-path data/NEW_DATASET/evaled_grasp_config_dicts \
--nerf-checkpoints-path data/NEW_DATASET/nerfcheckpoints \
--output-filepath data/NEW_DATASET/grid_dataset/dataset.h5
```

## BPS Evaluator Model Training

```
python nerf_grasping/dexdiffuser/train_dexdiffuser_evaluator.py
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/ccc2d2fa-7b08-4660-b202-28e0b5a8cee1" alt="bps_evaluator" style="width:50%;">
</p>

## NeRF Evaluator Model Training

```
python nerf_grasping/learned_metric/Train_DexGraspNet_NeRF_Grasp_Metric.py cnn-3d-xyz-global-cnn \
--task-type Y_PICK_AND_Y_COLL_AND_Y_PGS \
--train-dataset-filepath data/NEW_DATASET/grid_dataset/dataset.h5 \
--val-dataset-filepath data/NEW_DATASET/grid_dataset/dataset.h5 (TODO) \
--test-dataset-filepath data/NEW_DATASET/grid_dataset/dataset.h5 (TODO) \
--dataloader.batch-size 128 \
--name MY_NERF_EXPERIMENT_NAME \
--training.loss_fn l2 \
--training.save_checkpoint_freq 1
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/e958ffdf-44b4-4491-9f3d-aa37ffc20f21" alt="nerf_evaluator" style="width:50%;">
</p>

## BPS Sampler Model Training

```
python nerf_grasping/dexdiffuser/diffusion.py
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/782d7a18-8ac6-462d-b434-513387494fe2" alt="bps_sampler" style="width:30%;">
</p>
