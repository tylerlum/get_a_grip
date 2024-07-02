# Dataset Generation

Follow these instructions if you want to generate the dataset yourself instead of using the dataset above. To do this, you will still need the `meshes` dataset above.

## 1. Grasp Generation

```
CUDA_VISIBLE_DEVICES=0 \
python get_a_grip/dataset_generation/scripts/generate_hand_config_dicts.py \
--meshdata_root_path data/large/meshes \
--output_hand_config_dicts_path data/NEW_DATASET/hand_config_dicts \
--use_penetration_energy \
--rand_object_scale \
--min_object_scale 0.05 \
--max_object_scale 0.1
```

```
CUDA_VISIBLE_DEVICES=0 \
python get_a_grip/dataset_generation/scripts/generate_grasp_config_dicts.py \
--meshdata_root_path data/large/meshes \
--input_hand_config_dicts_path data/NEW_DATASET/hand_config_dicts \
--output_grasp_config_dicts_path data/NEW_DATASET/grasp_config_dicts
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/4d3166e0-b6dc-48a8-a317-e2aa40caaf5b" alt="sim" style="width:50%;">
</p>

## 2. Grasp Evaluation

```
CUDA_VISIBLE_DEVICES=0 \
python get_a_grip/dataset_generation/scripts/eval_all_grasp_config_dicts.py \
--meshdata_root_path data/large/meshes \
--input_grasp_config_dicts_path data/NEW_DATASET/grasp_config_dicts \
--output_evaled_grasp_config_dicts_path data/NEW_DATASET/evaled_grasp_config_dicts \
--num_random_pose_noise_samples_per_grasp 5
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/9aaca7ba-f843-4a28-8a1f-77b6aa401dd8" alt="sim" style="width:70%;">
<!--   <img src="https://github.com/tylerlum/get_a_grip_release/assets/26510814/3874194f-86b4-4204-8710-49264fc5f734" alt="snowman" style="width:100%;"> -->
<!--   <img src="https://github.com/tylerlum/get_a_grip_release/assets/26510814/9b1920fb-1ead-431e-a657-6be0243ab058" alt="figurine" style="width:100%;"> -->
</p>

## 3. Grasp Dataset Augmentation

```
TODO
```

## 4. Image Data Generation

```
CUDA_VISIBLE_DEVICES=0 \
python get_a_grip/dataset_generation/scripts/generate_nerf_data.py \
--meshdata_root_path data/large/meshes \
--output_nerfdata_path data/NEW_DATASET/nerfdata \
--only_objects_in_this_path data/NEW_DATASET/hand_config_dicts \
--num_cameras 100
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip_release/assets/26510814/d5f90429-b7af-4950-a6e7-0fb8112c97fa" alt="image_data_gen" style="width:50%;">
</p>

## 5. NeRF Training

```
python get_a_grip/dataset_generation/scripts/train_nerfs.py \
--experiment_name NEW_DATASET \
--nerfdata_name nerfdata \
--output_nerfcheckpoints_name nerfcheckpoints \
--max_num_iterations 400
```

## 6. Point Cloud Generation

```
python get_a_grip/dataset_generation/export_pointclouds.py \
--experiment_name NEW_DATASET \
--nerf-is-z-up False \
--nerfcheckpoints_name nerfcheckpoints \
--output_pointclouds_name pointclouds \
--num_points 5000
```
