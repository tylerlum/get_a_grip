# Get a Grip: Multi-Finger Grasp Evaluation at Scale Enables Robust Sim-to-Real Transfer

Website: https://sites.google.com/view/get-a-grip-dataset

![image](https://github.com/tylerlum/get_a_grip_release/assets/26510814/257a96a4-bb33-4a77-a1f5-3c549b337fe0)

![media](https://github.com/tylerlum/get_a_grip/assets/26510814/aa1cb4a8-f640-4f50-a545-c56e4116a594)

## Overview

This repo contains the official implementation of Get a Grip. It consists of:

- A dataset of labeled grasps, object meshes, and perceptual data.

- Pre-trained sampler and evaluator models for grasp planning.

- Code for dataset generation, visualization, model training, grasp planning, and grasp motion planning.

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

### 1. Download Dataset

Fill out this [form](https://forms.gle/qERExXPn5wKrGr1G8) to download the Get a Grip dataset and pretrained models. After filling out this form, you will receive a URL, which will be used next.

### 2. Visualize Dataset

TODO

### 3. Grasp Planning with BPS Sampler and BPS Evaluator (Floating Hand)

TODO

### 4. Grasp Motion Planning with BPS Sampler and BPS Evaluator (Arm + Hand)

TODO

## Dataset

Dataset information [here](docs/dataset.md).

## Visualization

Visualization tools information [here](docs/visualization.md).

## Dataset Generation

Dataset generation instructions [here](docs/dataset_generation.md).

## Model Training

Model training information [here](docs/model_training.md).

## Grasp Planning

Grasp planning information [here](docs/grasp_planning.md).

## Grasp Motion Planning

Grasp motion planning information [here](docs/grasp_motion_planning.md).

## Additional Details

Additional details [here](docs/additional_details.md).

## Citation

```
@article{TODO}
```
