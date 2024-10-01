# Model Training

Follow these instructions if you want to train models yourself instead of using the given models in `data/models/pretrained`. To do this, you will still need a dataset like the following:

```
data/dataset/large
├── final_evaled_grasp_config_dicts
├── meshes
├── nerfdata
├── nerfcheckpoints
└── point_clouds
```

For these next steps, we will be creating new models in `data/models/NEW` and creating new fixed sampler grasp config dicts in `data/fixed_sampler_grasp_config_dicts/NEW`.


First run the following (change as needed):
```
export DATASET_NAME=large
export MODELS_NAME=NEW
export FIXED_SAMPLER_GRASP_CONFIG_DICTS_NAME=NEW
```

## Train Val Test Split Across Objects

Create train val test split across objects (creates symlinks to `<input_evaled_grasp_config_dicts_path>_train`, `<input_evaled_grasp_config_dicts_path>_val`, `<input_evaled_grasp_config_dicts_path>_test`):

```
python get_a_grip/model_training/scripts/create_train_val_test_split.py \
--input_evaled_grasp_config_dicts_path data/dataset/${DATASET_NAME}/final_evaled_grasp_config_dicts \
--frac_train 0.8 \
--frac_val 0.1
```

## BPS HDF5 File Generation

Create BPS dataset:

```
python get_a_grip/model_training/scripts/create_bps_grasp_dataset.py \
--input_point_clouds_path data/dataset/${DATASET_NAME}/point_clouds \
--input_evaled_grasp_config_dicts_path data/dataset/${DATASET_NAME}/final_evaled_grasp_config_dicts_train \
--output_filepath data/dataset/${DATASET_NAME}/bps_grasp_dataset/train_dataset.h5
```

```
python get_a_grip/model_training/scripts/create_bps_grasp_dataset.py \
--input_point_clouds_path data/dataset/${DATASET_NAME}/point_clouds \
--input_evaled_grasp_config_dicts_path data/dataset/${DATASET_NAME}/final_evaled_grasp_config_dicts_val \
--output_filepath data/dataset/${DATASET_NAME}/bps_grasp_dataset/val_dataset.h5
```

```
python get_a_grip/model_training/scripts/create_bps_grasp_dataset.py \
--input_point_clouds_path data/dataset/${DATASET_NAME}/point_clouds \
--input_evaled_grasp_config_dicts_path data/dataset/${DATASET_NAME}/final_evaled_grasp_config_dicts_test \
--output_filepath data/dataset/${DATASET_NAME}/bps_grasp_dataset/test_dataset.h5
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip_release/assets/26510814/6da75267-280c-4e3a-921b-79b765842ab9" alt="dataset_gen" style="width:50%;">
</p>

## NeRF HDF5 File Generation

Create NeRF dataset:

```
python get_a_grip/model_training/scripts/create_nerf_grasp_dataset.py \
--input_nerfcheckpoints_path data/dataset/${DATASET_NAME}/nerfcheckpoints \
--input_evaled_grasp_config_dicts_path data/dataset/${DATASET_NAME}/final_evaled_grasp_config_dicts_train \
--output_filepath data/dataset/${DATASET_NAME}/nerf_grasp_dataset/train_dataset.h5
```

```
python get_a_grip/model_training/scripts/create_nerf_grasp_dataset.py \
--input_nerfcheckpoints_path data/dataset/${DATASET_NAME}/nerfcheckpoints \
--input_evaled_grasp_config_dicts_path data/dataset/${DATASET_NAME}/final_evaled_grasp_config_dicts_val \
--output_filepath data/dataset/${DATASET_NAME}/nerf_grasp_dataset/val_dataset.h5
```

```
python get_a_grip/model_training/scripts/create_nerf_grasp_dataset.py \
--input_nerfcheckpoints_path data/dataset/${DATASET_NAME}/nerfcheckpoints \
--input_evaled_grasp_config_dicts_path data/dataset/${DATASET_NAME}/final_evaled_grasp_config_dicts_test \
--output_filepath data/dataset/${DATASET_NAME}/nerf_grasp_dataset/test_dataset.h5
```

## BPS Evaluator Model Training

```
python get_a_grip/model_training/scripts/train_bps_evaluator_model.py \
--train_dataset_path data/dataset/${DATASET_NAME}/bps_grasp_dataset/train_dataset.h5 \
--val_dataset_path data/dataset/${DATASET_NAME}/bps_grasp_dataset/val_dataset.h5
--output_dir data/models/${MODELS_NAME}/bps_evaluator_model
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/ccc2d2fa-7b08-4660-b202-28e0b5a8cee1" alt="bps_evaluator" style="width:50%;">
</p>

## NeRF Evaluator Model Training

```
python get_a_grip/model_training/scripts/train_nerf_evaluator_model.py \
--nerf_grasp_dataset_config_path data/dataset/${DATASET_NAME}/nerf_grasp_dataset/config.yml \
--train-dataset-path data/dataset/${DATASET_NAME}/nerf_grasp_dataset/train_dataset.h5 \
--val-dataset-path data/dataset/${DATASET_NAME}/nerf_grasp_dataset/val_dataset.h5 \
--test-dataset-path data/dataset/${DATASET_NAME}/nerf_grasp_dataset/test_dataset.h5 \
--checkpoint_workspace.output_dir data/models/${MODELS_NAME}/nerf_evaluator_model
--task-type Y_PICK_AND_Y_COLL_AND_Y_PGS \
--dataloader.batch-size 128 \
--name MY_NERF_EXPERIMENT_NAME \
--training.loss_fn l2 \
--training.save_checkpoint_freq 1 \
model-config:cnn-xyz-global-cnn-model-config
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/e958ffdf-44b4-4491-9f3d-aa37ffc20f21" alt="nerf_evaluator" style="width:50%;">
</p>

## BPS Sampler Model Training

```
python get_a_grip/model_training/scripts/train_bps_sampler_model.py \
--train_dataset_path data/dataset/${DATASET_NAME}/bps_grasp_dataset/train_dataset.h5 \
--val_dataset_path data/dataset/${DATASET_NAME}/bps_grasp_dataset/val_dataset.h5
--output_dir data/models/${MODELS_NAME}/bps_sampler_model
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/782d7a18-8ac6-462d-b434-513387494fe2" alt="bps_sampler" style="width:30%;">
</p>

## NeRF Sampler Model Training

```
python get_a_grip/model_training/scripts/train_nerf_sampler_model.py \
--train_dataset_path data/dataset/${DATASET_NAME}/nerf_grasp_dataset/train_dataset.h5 \
--val_dataset_path data/dataset/${DATASET_NAME}/nerf_grasp_dataset/val_dataset.h5
--output_dir data/models/${MODELS_NAME}/nerf_sampler_model
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/782d7a18-8ac6-462d-b434-513387494fe2" alt="bps_sampler" style="width:30%;">
</p>

## Create Fixed Sampler Grasp Config Dict

All good grasps:

```
python get_a_grip/grasp_planning/scripts/create_fixed_sampler_grasp_config_dict.py \
--input-evaled-grasp-config-dicts-path data/dataset/${DATASET_NAME}/final_evaled_grasp_config_dicts_train \
--output-grasp-config-dict-path data/fixed_sampler_grasp_config_dicts/${FIXED_SAMPLER_GRASP_CONFIG_DICTS_NAME}/all_good_grasps.npy \
--y_PGS_threshold 0.9 \
--max_grasps_per_object None \
--overwrite True
```

One good grasp per object:

```
python get_a_grip/grasp_planning/scripts/create_fixed_sampler_grasp_config_dict.py \
--input-evaled-grasp-config-dicts-path data/dataset/${DATASET_NAME}/final_evaled_grasp_config_dicts_train \
--output-grasp-config-dict-path data/fixed_sampler_grasp_config_dicts/${FIXED_SAMPLER_GRASP_CONFIG_DICTS_NAME}/one_good_grasp_per_object.npy \
--y_PGS_threshold 0.9 \
--max_grasps_per_object 1 \
--overwrite True
```
