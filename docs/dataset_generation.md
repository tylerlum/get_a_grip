# Dataset Generation

Follow these instructions if you want to generate the dataset yourself, instead of using the given dataset in `data/dataset/large`. To do this, you will still need the given meshdata at `data/meshdata`. For these next steps, we will be creating a dataset in `data/dataset/NEW`.

## 0. Object Selection

Select objects and scales to generate data for:

```
python get_a_grip/dataset_generation/scripts/generate_object_code_and_scales_txt.py \
--meshdata_root_path data/meshdata \
--output_object_code_and_scales_txt_path data/dataset/NEW/object_code_and_scales.txt \
--min_object_scale 0.05 \
--max_object_scale 0.1 \
--num_scales_per_object 3
```

If you want to generate for fewer objects to test, you can set the `--max_num_object_codes` like so:

```
python get_a_grip/dataset_generation/scripts/generate_object_code_and_scales_txt.py \
--meshdata_root_path data/meshdata \
--output_object_code_and_scales_txt_path data/dataset/NEW/object_code_and_scales.txt \
--min_object_scale 0.05 \
--max_object_scale 0.1 \
--num_scales_per_object 3 \
--max_num_object_codes 20
```

## 1. NeRF Data Generation + Object Filtering

For each object, drop it on a table, wait for it to settle upright, then capture posed images of it on a table. If the object doesn't settle upright on the table, it is filtered out with a new txt file (<output_nerfdata_path_data>_settled_successes.txt).

```
python get_a_grip/dataset_generation/scripts/generate_nerfdata.py \
--meshdata_root_path data/meshdata \
--input_object_code_and_scales_txt_path data/dataset/NEW/object_code_and_scales.txt \
--output_nerfdata_path data/dataset/NEW/nerfdata \
--num_cameras 100
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip_release/assets/26510814/d5f90429-b7af-4950-a6e7-0fb8112c97fa" alt="image_data_gen" style="width:50%;">
</p>


## 2. Grasp Generation

Generate pre-grasp pose for each object and scale:

```
python get_a_grip/dataset_generation/scripts/generate_hand_config_dicts.py \
--meshdata_root_path data/meshdata \
--input_object_code_and_scales_txt_path data/dataset/NEW/nerfdata_settled_successes.txt \
--output_hand_config_dicts_path data/dataset/NEW/hand_config_dicts
```

Generate grasp directions for each grasp:

```
python get_a_grip/dataset_generation/scripts/generate_grasp_config_dicts.py \
--meshdata_root_path data/meshdata \
--input_hand_config_dicts_path data/dataset/NEW/hand_config_dicts \
--output_grasp_config_dicts_path data/dataset/NEW/grasp_config_dicts
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/4d3166e0-b6dc-48a8-a317-e2aa40caaf5b" alt="sim" style="width:50%;">
</p>

## 3. Grasp Evaluation

Evaluate grasps in simulation:

```
python get_a_grip/dataset_generation/scripts/eval_all_grasp_config_dicts.py \
--meshdata_root_path data/meshdata \
--input_grasp_config_dicts_path data/dataset/NEW/grasp_config_dicts \
--output_evaled_grasp_config_dicts_path data/dataset/NEW/evaled_grasp_config_dicts \
--num_random_pose_noise_samples_per_grasp 5
```

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/9aaca7ba-f843-4a28-8a1f-77b6aa401dd8" alt="sim" style="width:70%;">
<!--   <img src="https://github.com/tylerlum/get_a_grip_release/assets/26510814/3874194f-86b4-4204-8710-49264fc5f734" alt="snowman" style="width:100%;"> -->
<!--   <img src="https://github.com/tylerlum/get_a_grip_release/assets/26510814/9b1920fb-1ead-431e-a657-6be0243ab058" alt="figurine" style="width:100%;"> -->
</p>

## 4. Grasp Dataset Augmentation

Augment the grasp dataset with random noise:

```
python get_a_grip/dataset_generation/scripts/augment_grasp_config_dicts.py \
--input_evaled_grasp_config_dicts_path data/dataset/NEW/evaled_grasp_config_dicts \
--output_augmented_grasp_config_dicts_path data/dataset/NEW/augmented_grasp_config_dicts
```

Evaluate augmented grasps in simulation:

```
python get_a_grip/dataset_generation/scripts/eval_all_grasp_config_dicts.py \
--meshdata_root_path data/meshdata \
--input_grasp_config_dicts_path data/dataset/NEW/augmented_grasp_config_dicts \
--output_evaled_grasp_config_dicts_path data/dataset/NEW/evaled_augmented_grasp_config_dicts \
--num_random_pose_noise_samples_per_grasp 5
```

## 5. Merge Evaled Grasp Config Dicts

Merge the original and augmented grasp config dicts:

```
python get_a_grip/dataset_generation/scripts/merge_evaled_grasp_config_dicts.py \
--input_evaled_grasp_config_dicts_paths data/dataset/NEW/evaled_grasp_config_dicts data/dataset/NEW/evaled_augmented_grasp_config_dicts \
--output_evaled_grasp_config_dicts_path data/dataset/NEW/final_evaled_grasp_config_dicts
```

## 6. NeRF Training

Train a NeRF for each object:

```
python get_a_grip/dataset_generation/scripts/train_nerfs.py \
--input_nerfdata_path data/dataset/NEW/nerfdata \
--output_nerfcheckpoints_path data/dataset/NEW/nerfcheckpoints \
--max_num_iterations 400
```

## 7. Point Cloud Generation

Generate point clouds for each object:

```
python get_a_grip/dataset_generation/scripts/generate_point_clouds.py \
--nerf-is-z-up False \
--input_nerfcheckpoints_path data/dataset/NEW/nerfcheckpoints \
--output_point_clouds_path data/dataset/NEW/point_clouds \
--num_points 5000
```
