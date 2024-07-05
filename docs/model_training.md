# Model Training

Follow these instructions if you want to train models yourself instead of using the given models. To do this, you will still need a dataset like the following:

```
data
└── large
    ├── final_evaled_grasp_config_dicts
    ├── meshes
    ├── nerfdata
    ├── nerfs
    └── point_clouds
```

## Train Val Test Split Across Objects

Create train val test split across objects (creates symlinks to <input*evaled_grasp_config_dicts_path>*_, where _ is train, val, or test):

```
python get_a_grip/model_training/scripts/create_train_val_test_split.py \
--input_evaled_grasp_config_dicts_path data/NEW_DATASET/final_evaled_grasp_config_dicts \
--frac_train 0.8 \
--frac_val 0.1
```

## BPS HDF5 File Generation

Create BPS dataset:

```
python get_a_grip/model_training/scripts/create_bps_grasp_dataset.py \
--input_point_clouds_path data/NEW_DATASET/point_clouds \
--input_evaled_grasp_config_dicts_path data/NEW_DATASET/final_evaled_grasp_config_dicts_train \
--output_filepath data/NEW_DATASET/bps_grasp_dataset/train_dataset.h5
```

```
python get_a_grip/model_training/scripts/create_bps_grasp_dataset.py \
--input_point_clouds_path data/NEW_DATASET/point_clouds \
--input_evaled_grasp_config_dicts_path data/NEW_DATASET/final_evaled_grasp_config_dicts_val \
--output_filepath data/NEW_DATASET/bps_grasp_dataset/val_dataset.h5
```

```
python get_a_grip/model_training/scripts/create_bps_grasp_dataset.py \
--input_point_clouds_path data/NEW_DATASET/point_clouds \
--input_evaled_grasp_config_dicts_path data/NEW_DATASET/final_evaled_grasp_config_dicts_test \
--output_filepath data/NEW_DATASET/bps_grasp_dataset/test_dataset.h5
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip_release/assets/26510814/6da75267-280c-4e3a-921b-79b765842ab9" alt="dataset_gen" style="width:50%;">
</p>

## NeRF HDF5 File Generation

Create NeRF dataset:

```
python get_a_grip/model_training/scripts/create_nerf_grasp_dataset.py \
--input_nerfcheckpoints_path data/NEW_DATASET/nerfcheckpoints \
--input_evaled_grasp_config_dicts_path data/NEW_DATASET/final_evaled_grasp_config_dicts_train \
--output_filepath data/NEW_DATASET/nerf_grasp_dataset/train_dataset.h5
```

```
python get_a_grip/model_training/scripts/create_nerf_grasp_dataset.py \
--input_nerfcheckpoints_path data/NEW_DATASET/nerfcheckpoints \
--input_evaled_grasp_config_dicts_path data/NEW_DATASET/final_evaled_grasp_config_dicts_val \
--output_filepath data/NEW_DATASET/nerf_grasp_dataset/val_dataset.h5
```

```
python get_a_grip/model_training/scripts/create_nerf_grasp_dataset.py \
--input_nerfcheckpoints_path data/NEW_DATASET/nerfcheckpoints \
--input_evaled_grasp_config_dicts_path data/NEW_DATASET/final_evaled_grasp_config_dicts_test \
--output_filepath data/NEW_DATASET/nerf_grasp_dataset/test_dataset.h5
```

## BPS Evaluator Model Training

```
python get_a_grip/model_training/scripts/train_bps_grasp_evaluator.py \
--train_dataset_path data/NEW_DATASET/bps_grasp_dataset/train_dataset.h5 \
--val_dataset_path data/NEW_DATASET/bps_grasp_dataset/val_dataset.h5
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/ccc2d2fa-7b08-4660-b202-28e0b5a8cee1" alt="bps_evaluator" style="width:50%;">
</p>

## NeRF Evaluator Model Training

```
python get_a_grip/model_training/scripts/train_nerf_grasp_evaluator.py \
cnn-3d-xyz-global-cnn \
--train-dataset-path data/NEW_DATASET/nerf_grasp_dataset/train_dataset.h5 \
--val-dataset-path data/NEW_DATASET/nerf_grasp_dataset/val_dataset.h5 \
--test-dataset-path data/NEW_DATASET/nerf_grasp_dataset/test_dataset.h5 \
--task-type Y_PICK_AND_Y_COLL_AND_Y_PGS \
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
python get_a_grip/model_training/scripts/train_bps_grasp_sampler.py \
--train_dataset_path data/NEW_DATASET/bps_grasp_dataset/train_dataset.h5 \
--val_dataset_path data/NEW_DATASET/bps_grasp_dataset/val_dataset.h5
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/782d7a18-8ac6-462d-b434-513387494fe2" alt="bps_sampler" style="width:30%;">
</p>

## NeRF Sampler Model Training

```
python get_a_grip/model_training/scripts/train_nerf_grasp_sampler.py \
--train_dataset_path data/NEW_DATASET/nerf_grasp_dataset/train_dataset.h5 \
--val_dataset_path data/NEW_DATASET/nerf_grasp_dataset/val_dataset.h5
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/782d7a18-8ac6-462d-b434-513387494fe2" alt="bps_sampler" style="width:30%;">
</p>
