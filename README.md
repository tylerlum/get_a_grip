# Get a Grip: Multi-Finger Grasp Evaluation at Scale Enables Robust Sim-to-Real Transfer

Website: https://sites.google.com/view/get-a-grip-dataset

![image](https://github.com/tylerlum/get_a_grip_release/assets/26510814/257a96a4-bb33-4a77-a1f5-3c549b337fe0)

![media](https://github.com/tylerlum/get_a_grip/assets/26510814/aa1cb4a8-f640-4f50-a545-c56e4116a594)

## Overview

This repo contains the official implementation of Get a Grip. It consists of:

- A dataset of labeled grasps, object meshes, and perceptual data.

- Pre-trained sampler and evaluator models for grasp planning.

- Code for dataset generation, visualization, model training, grasp planning (floating hand), and grasp motion planning (arm + hand).

## Project Structure

```
get_a_grip
  ├── assets
  │   └── // Assets such as allegro hand urdf files
  ├── data
  │   └── // Store dataset, models, and output files here
  ├── docs
  │   └── // Documentation
  └── get_a_grip
      ├── // Source code
      └── dataset_generation
         └── // Generate dataset
      └── grasp_motion_planning
         └── // Motion planning to perform grasps with an arm and hand
      └── grasp_planning
         └── // Planning to perform grasps with a floating hand
      └── model_training
         └── // Neural network training for samplers and evaluators
      └── utils
         └── // Shared utilities
      └── visualization
         └── // Visualization tools
```

## Installation

Installation instructions [here](docs/installation.md).

## Quick Start

Please run all commands from the root directory of this repository.

### 1. Download Dataset

Fill out this [form](https://forms.gle/qERExXPn5wKrGr1G8) to download the Get a Grip dataset and pretrained models. After filling out this form, you will receive a URL <download_url>, which will be used next.

Next, we will download the small version of the dataset and the pretrained models. Change `<download_url>` to the URL you received, then run:

```
python download.py \
--download_url <download_url> \
--include_meshdata True \
--dataset_size small \
--include_final_evaled_grasp_config_dicts True \
--include_nerfdata True \
--include_point_clouds True \
--include_nerfcheckpoints True \
--include_pretrained_models True \
--include_fixed_sampler_grasp_config_dicts True \
--include_real_world_nerfdata True \
--include_real_world_nerfcheckpoints True
```

The resulting directory structure should look like this:

```
data
├── dataset
│   └── small
│       ├── final_evaled_grasp_config_dicts
│       ├── nerfdata
│       ├── nerfcheckpoints
│       └── point_clouds
├── fixed_sampler_grasp_config_dicts
│   └── given
│       ├── all_good_grasps.py
│       └── one_good_grasp_per_object.py
├── meshdata
│   ├── <object_code>
│   ├── ...
│   └── <object_code>
└── models
    └── pretrained
        ├── bps_evaluator_model
        ├── bps_sampler_model
        ├── nerf_evaluator_model
        └── nerf_sampler_model
```

You can change `small` to `large` for the large version, which is >1 TB (only recommended for model training).

See [here](docs/download.md) for more download details.

### 2. Visualize Grasps and NeRFs

Visualize a specific grasp on one object:

```
python get_a_grip/visualization/scripts/visualize_config_dict.py \
--meshdata_root_path data/meshdata \
--input_config_dicts_path data/dataset/small/grasp_config_dicts \
--object_code_and_scale_str mug_0_1000 \
--idx_to_visualize 0
```

Visualize a specific grasp simulation on one object in a GUI like so:

```
python get_a_grip/dataset_generation/scripts/eval_grasp_config_dict.py \
--meshdata_root_path data/meshdata \
--input_grasp_config_dicts_path data/dataset/small/grasp_config_dicts \
--output_evaled_grasp_config_dicts_path None \
--object_code_and_scale_str mug_0_1000 \
--max_grasps_per_batch 5000 \
--save_to_file False \
--use_gui True \
--debug_index 0
```

Run nerfstudio's interactive viewer:

```
ns-viewer --load-config data/dataset/small/nerfcheckpoints/mug_0_1000/nerfacto/2024-05-11_212418/config.yml
```

See [here](docs/visualization.md) for more visualization details.

### 3. Grasp Planning (Floating Hand)

Visualize BPS sampler and BPS evaluator on simulated NeRF:

```
python get_a_grip/grasp_planning/scripts/run_grasp_planning.py \
--visualize_loop True \
--output_folder data/grasp_planning_outputs/bps_sampler_bps_evaluator \
--overwrite True \
--nerf.nerf_is_z_up False \
--nerf.nerf_config data/dataset/small/nerfcheckpoints/mug_0_1000/nerfacto/2024-05-11_212418/config.yml \
planner.sampler:bps-sampler-config \
  --planner.sampler.ckpt_path data/models/pretrained/bps_sampler_model/20240710140203/ckpt_1.pth \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:bps-evaluator-config \
  --planner.evaluator.ckpt_path data/models/pretrained/bps_evaluator_model/20240710_140229/ckpt-emrvib91-step-80.pth \
planner.optimizer:bps-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```

Visualize BPS sampler and BPS evaluator on real world NeRF:

```
python get_a_grip/grasp_planning/scripts/run_grasp_planning.py \
--visualize_loop True \
--output_folder data/grasp_planning_outputs/bps_sampler_bps_evaluator \
--overwrite True \
--nerf.nerf_is_z_up True \
--nerf.nerf_config data/real_world/nerfcheckpoints/dragon_0_1000/nerfacto/2024-05-11_212418/config.yml \
planner.sampler:bps-sampler-config \
  --planner.sampler.ckpt_path data/models/pretrained/bps_sampler_model/20240710140203/ckpt_1.pth \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:bps-evaluator-config \
  --planner.evaluator.ckpt_path data/models/pretrained/bps_evaluator_model/20240710_140229/ckpt-emrvib91-step-80.pth \
planner.optimizer:bps-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```

See [here](docs/grasp_planning.md) for more grasp planning details.

### 4. Grasp Motion Planning (Arm + Hand)

Visualize BPS sampler and BPS evaluator on simulated NeRF:

```
python get_a_grip/grasp_motion_planning/scripts/run_grasp_motion_planning.py \
--visualize_loop True \
--output_folder data/grasp_planning_outputs/bps_sampler_bps_evaluator \
--overwrite True \
--nerf.nerf_is_z_up False \
--nerf.nerf_config data/dataset/small/nerfcheckpoints/mug_0_1000/nerfacto/2024-05-11_212418/config.yml \
planner.sampler:bps-sampler-config \
  --planner.sampler.ckpt_path data/models/pretrained/bps_sampler_model/20240710140203/ckpt_1.pth \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:bps-evaluator-config \
  --planner.evaluator.ckpt_path data/models/pretrained/bps_evaluator_model/20240710_140229/ckpt-emrvib91-step-80.pth \
planner.optimizer:bps-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```

Visualize BPS sampler and BPS evaluator on real world NeRF:

```
python get_a_grip/grasp_motion_planning/scripts/run_grasp_motion_planning.py \
--visualize_loop True \
--output_folder data/grasp_planning_outputs/bps_sampler_bps_evaluator \
--overwrite True \
--nerf.nerf_is_z_up True \
--nerf.nerf_config data/real_world/nerfcheckpoints/dragon_0_1000/nerfacto/2024-05-11_212418/config.yml \
planner.sampler:bps-sampler-config \
  --planner.sampler.ckpt_path data/models/pretrained/bps_sampler_model/20240710140203/ckpt_1.pth \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:bps-evaluator-config \
  --planner.evaluator.ckpt_path data/models/pretrained/bps_evaluator_model/20240710_140229/ckpt-emrvib91-step-80.pth \
planner.optimizer:bps-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```

See [here](docs/grasp_motion_planning.md) for more grasp motion planning details.

## Dataset Generation

Dataset generation instructions [here](docs/dataset_generation.md).

## Model Training

Model training information [here](docs/model_training.md).

## Additional Details

Additional details [here](docs/additional_details.md).

## Citation

```
@article{TODO}
```
