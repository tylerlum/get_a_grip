# Get a Grip: Multi-Finger Grasp Evaluation at Scale Enables Robust Sim-to-Real Transfer

[[Project Page](https://sites.google.com/view/get-a-grip-dataset)] [[Streamlit Data Visualization App](https://get-a-grip.streamlit.app/)]

![image](https://github.com/tylerlum/get_a_grip_release/assets/26510814/257a96a4-bb33-4a77-a1f5-3c549b337fe0)

![media](https://github.com/tylerlum/get_a_grip/assets/26510814/aa1cb4a8-f640-4f50-a545-c56e4116a594)

## Overview

This repo contains the official implementation of Get a Grip. It consists of:

- A dataset of labeled grasps, object meshes, and perceptual data.

- Pre-trained sampler and evaluator models for grasp planning.

- Code for dataset generation, visualization, model training, grasp planning (floating hand), and grasp motion planning (arm + hand).

## Streamlit Data Visualization App

Before continuing, we highly recommend you check out our [Streamlit Data Visualization App](https://get-a-grip.streamlit.app/) to interactively visualize the dataset. Here are a few videos to preview what the website will show!

### Grasp Visualization

[get-a-grip-streamlit-app-2024-09-27-16-09-96-ezgif.com-video-cutter.webm](https://github.com/user-attachments/assets/34dbc160-0e11-4910-b641-863cd80b0afe)

### Object Visualization

[get-a-grip-streamlit-app-2024-09-27-16-09-96-ezgif.com-video-cutter (1).webm](https://github.com/user-attachments/assets/c32f66b3-54d4-46b3-ac14-bcb255957123)

### NeRF Data Visualization

[get-a-grip-streamlit-app-2024-09-27-16-09-96-ezgif.com-video-cutter (4).webm](https://github.com/user-attachments/assets/0573b993-55b5-4eb4-8579-b1a8fd71a810)

### Real World Data Visualization

[get-a-grip-streamlit-app-2024-09-27-16-09-96-ezgif.com-video-cutter (5).webm](https://github.com/user-attachments/assets/6e31dcaa-e7be-476c-86b3-f2290e586d52)


## Project Structure

```
get_a_grip
  ├── assets
  │   └── // Assets such as Allegro hand URDF files
  ├── data
  │   └── // Store dataset, models, and output files here
  ├── docs
  │   └── // Documentation
  ├── get_a_grip
  │   ├── // Source code
  │   └── dataset_generation
  │   │  └── // Generate dataset
  │   └── grasp_motion_planning
  │   │   └── // Motion planning to perform grasps with an arm and hand
  │   └── grasp_planning
  │   │  └── // Planning to perform grasps with a floating hand
  │   └── model_training
  │   │  └── // Neural network training for samplers and evaluators
  │   └── utils
  │   │  └── // Shared utilities
  │   └── visualization
  │      └── // Visualization tools
  ├── streamlit
  │   ├── app.py            // Streamlit app for data visualization
  │   ├── requirements.txt  // Streamlit Python requirements
  │   └── ...
  └── packages.txt          // Streamlit required packages
                            // https://docs.streamlit.io/knowledge-base/dependencies/libgl
                            // https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
```

## Installation

Installation instructions [here](docs/installation.md).

## Quick Start

Please run all commands from the root directory of this repository.

### 0. Set Up Zsh Tab Completion (Optional)

Run this to get tab completion for most scripts in this codebase. This is experimental and only works for `zsh` (not `bash`). This is just a small quality of life improvement, so feel free to skip it!

One-time run to set up tab completion (takes a few minutes):
```
python get_a_grip/utils/setup_zsh_tab_completion.py
```

Run this in every new session (or put in `~/.zshrc`):
```
fpath+=`pwd`/.zsh_tab_completion
autoload -Uz compinit && compinit
```

Now, for most scripts in the database, you will have tab autocompletion. For example:

```
python get_a_grip/grasp_motion_planning/scripts/run_grasp_motion_planning.py \
-
```

After the script filepath, you can enter a `-` and then press "TAB" to get auto-completion!

[zsh_tab_completion.webm](https://github.com/user-attachments/assets/2eab49d0-c176-438f-9833-99bd79d7f172)

### 1. Download Dataset

Fill out this [form](https://forms.gle/qERExXPn5wKrGr1G8) to download the Get a Grip dataset and pretrained models. After filling out this form, you will receive a URL `<download_url>`, which will be used next.

NOTE: Navigating to `<download_url>` will take you to a forbidden page. This is expected. We use it in the steps below by setting this environment variable:

```
export DOWNLOAD_URL=<download_url>
```

First, we will download the meshdata for all objects (recommended for all use cases):

```
python get_a_grip/utils/download.py \
--download_url ${DOWNLOAD_URL} \
--include_meshdata True
```

We will be choosing the `nano` version of the dataset for testing and visualization:

```
export DATASET_NAME=nano
```

You can change `nano` (2 random objects) to `tiny_random` (25 random objects), `tiny_best` (25 best-grasped objects), `small_random` (100 random objects), `small_best` (100 best-grasped objects), or `large` (all objects). `large` is >2 TB (only recommended for model training).

Next, we will download the dataset:

```
python get_a_grip/utils/download.py \
--download_url ${DOWNLOAD_URL} \
--dataset_name ${DATASET_NAME} \
--include_final_evaled_grasp_config_dicts True \
--include_nerfdata True \
--include_nerfcheckpoints True \
--include_point_clouds True
```

Note: You can remove the `--include_...` if you don't need a certain component (for example, if you only want the grasps and not the perceptual data, remove all `--include_...` except for --include_final_evaled_grasp_config_dicts True`).

Next, we will download the pretrained models and the fixed sampler grasps:

```
python get_a_grip/utils/download.py \
--download_url ${DOWNLOAD_URL} \
--include_pretrained_models True \
--include_fixed_sampler_grasp_config_dicts True 
```

Next, we will download real-world object data (optional, takes a few minutes to download):

```
python get_a_grip/utils/download.py \
--download_url ${DOWNLOAD_URL} \
--include_real_world_nerfdata True \
--include_real_world_nerfcheckpoints True \
--include_real_world_point_clouds True
```

The resulting directory structure should look like this:

```
data
├── dataset
│   └── nano
│       ├── final_evaled_grasp_config_dicts
│       ├── nerfdata
│       ├── nerfcheckpoints
│       └── point_clouds
├── fixed_sampler_grasp_config_dicts
│   └── given
│       ├── all_good_grasps.npy
│       └── one_good_grasp_per_object.npy
├── meshdata
│   ├── <object_code>
│   ├── ...
│   └── <object_code>
└── models
│   └── pretrained
│       ├── bps_evaluator_model
│       ├── bps_sampler_model
│       ├── nerf_evaluator_model
│       └── nerf_sampler_model
└── real_world
    ├── nerfdata
    ├── nerfcheckpoints
    └── point_clouds
```

See [here](docs/download.md) for more download details.

### 2. Visualize Grasps and NeRFs

Set the following environment variable for the next steps:
```
export MESHDATA_ROOT_PATH=data/meshdata
```

We will be using the following object for the next steps:

```
export OBJECT_CODE_AND_SCALE_STR=core-mug-5c48d471200d2bf16e8a121e6886e18d_0_0622
export NERF_CONFIG=data/dataset/${DATASET_NAME}/nerfcheckpoints/core-mug-5c48d471200d2bf16e8a121e6886e18d_0_0622/nerfacto/2024-07-13_111325/config.yml
```

Visualize a specific grasp on one object:

```
python get_a_grip/visualization/scripts/visualize_config_dict.py \
--meshdata_root_path ${MESHDATA_ROOT_PATH} \
--input_config_dicts_path data/dataset/${DATASET_NAME}/final_evaled_grasp_config_dicts \
--object_code_and_scale_str ${OBJECT_CODE_AND_SCALE_STR} \
--idx_to_visualize 0
```

[Visualize_Grasp_Plotly_simplescreenrecorder-2024-09-27_17.35.20.mp4](https://github.com/user-attachments/assets/5b5f9014-3c48-4fc6-adb0-84e7670ffe36)

Visualize a specific grasp simulation on one object in a GUI like so:

```
python get_a_grip/dataset_generation/scripts/eval_grasp_config_dict.py \
--meshdata_root_path ${MESHDATA_ROOT_PATH} \
--input_grasp_config_dicts_path data/dataset/${DATASET_NAME}/final_evaled_grasp_config_dicts \
--output_evaled_grasp_config_dicts_path None \
--object_code_and_scale_str ${OBJECT_CODE_AND_SCALE_STR} \
--max_grasps_per_batch 5000 \
--save_to_file False \
--use_gui True \
--debug_index 0
```

[Visualize_Grasp_IsaacGym_simplescreenrecorder-2024-09-27_17.38.06.mp4](https://github.com/user-attachments/assets/bdfe121f-1c36-4741-a294-0e2604c5c05a)

Note: Isaacgym may not run on all GPUs (e.g., H100s seem to not work). Please refer to their documentation for more details (in `<path/to/isaacgym>/docs/index.html`).

If you get this error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, you can try `export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib`.

Run nerfstudio's interactive viewer:

```
ns-viewer --load-config ${NERF_CONFIG}
```

[Visualize_NeRF_simplescreenrecorder-2024-09-27_17.36.19.mp4](https://github.com/user-attachments/assets/0371651a-e05a-4e06-a93b-bedb1a96b0a6)

See [here](docs/visualization.md) for more visualization details.

### 3. Grasp Planning (Floating Hand)

We will be using the following models for the next steps:

```
export BPS_SAMPLER=data/models/pretrained/bps_sampler_model/ckpt.pth
export BPS_EVALUATOR=data/models/pretrained/bps_evaluator_model/ckpt.pth
export REAL_WORLD_NERF_CONFIG=data/real_world/nerfcheckpoints/squirrel_0_9999/nerfacto/2024-09-27_125722/config.yml
```

Visualize BPS sampler and BPS evaluator on simulated NeRF:

```
python get_a_grip/grasp_planning/scripts/run_grasp_planning.py \
--visualize_loop True \
--output_folder data/grasp_planning_outputs/bps_sampler_bps_evaluator \
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

[Visualize_Mug_Generated_Grasp_simplescreenrecorder-2024-09-30_15.46.14.mp4](https://github.com/user-attachments/assets/c89b84fc-55e6-4ac2-90ca-d0cab31f6933)

Visualize BPS sampler and BPS evaluator on real world NeRF (optional, only if you included the real world object data):

```
python get_a_grip/grasp_planning/scripts/run_grasp_planning.py \
--visualize_loop True \
--output_folder data/grasp_planning_outputs/bps_sampler_bps_evaluator \
--overwrite True \
--nerf.nerf_is_z_up True \
--nerf.nerf_config ${REAL_WORLD_NERF_CONFIG} \
planner.sampler:bps-sampler-config \
  --planner.sampler.ckpt_path ${BPS_SAMPLER} \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:bps-evaluator-config \
  --planner.evaluator.ckpt_path ${BPS_EVALUATOR} \
planner.optimizer:bps-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```

[Visualize_Squirrel_Generated_Grasp_simplescreenrecorder-2024-09-30_16.04.20.mp4](https://github.com/user-attachments/assets/062a8f28-097e-47d2-84e2-d2630be6e108)

See [here](docs/grasp_planning.md) for more grasp planning details.

### 4. Grasp Motion Planning (Arm + Hand)

Visualize BPS sampler and BPS evaluator on simulated NeRF:

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

[Visualize_Mug_Generated_Grasp_Motion_simplescreenrecorder-2024-09-30_15.55.17.mp4](https://github.com/user-attachments/assets/19bfc592-367b-47b2-99da-0ec9503ddcc5)

Visualize BPS sampler and BPS evaluator on real world NeRF (optional, only if you included the real world object data):

```
python get_a_grip/grasp_motion_planning/scripts/run_grasp_motion_planning.py \
--visualize_loop True \
--output_folder data/grasp_motion_planning_outputs/bps_sampler_bps_evaluator \
--overwrite True \
--nerf.nerf_is_z_up True \
--nerf.nerf_config ${REAL_WORLD_NERF_CONFIG} \
planner.sampler:bps-sampler-config \
  --planner.sampler.ckpt_path ${BPS_SAMPLER} \
  --planner.sampler.num_grasps 3200 \
planner.evaluator:bps-evaluator-config \
  --planner.evaluator.ckpt_path ${BPS_EVALUATOR} \
planner.optimizer:bps-random-sampling-optimizer-config \
  --planner.optimizer.num_grasps 32
```

[Visualize_Squirrel_Generated_Grasp_Motion_simplescreenrecorder-2024-09-30_16.02.24.mp4](https://github.com/user-attachments/assets/8609a248-5278-4a0a-8ee5-0745830de42e)

See [here](docs/grasp_motion_planning.md) for more grasp motion planning details.

## Dataset Generation

See [here](docs/dataset_generation.md) for details about generating the dataset yourself.

## Model Training

See [here](docs/model_training.md) for details about training your own models yourself.

## Additional Details

See [here](docs/additional_details.md) for additional useful details about this codebase (highly recommended).

## Streamlit Data Visualization App

To run our Streamlit data visualization app locally:

```
pip install -r streamlit/requirements.txt
streamlit run streamlit/app.py
```

## Citation

```
@inproceedings{lum2024get,
  title     = {Get a Grip: Multi-Finger Grasp Evaluation at Scale Enables Robust Sim-to-Real Transfer},
  author    = {Tyler Ga Wei Lum and Albert H. Li and Preston Culbertson and Krishnan Srinivasan and Aaron Ames and Mac Schwager and Jeannette Bohg},
  booktitle = {8th Annual Conference on Robot Learning},
  year      = {2024},
  url       = {https://openreview.net/forum?id=1jc2zA5Z6J}
}
```

## Contact

If you have any questions, issues, or feedback, please contact [Tyler Lum](https://tylerlum.github.io/).
