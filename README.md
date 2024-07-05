# Get a Grip: Multi-Finger Grasp Evaluation at Scale Enables Robust Sim-to-Real Transfer

Website: https://sites.google.com/view/get-a-grip-dataset

![image](https://github.com/tylerlum/get_a_grip_release/assets/26510814/257a96a4-bb33-4a77-a1f5-3c549b337fe0)

![media](https://github.com/tylerlum/get_a_grip/assets/26510814/aa1cb4a8-f640-4f50-a545-c56e4116a594)

## Overview

This repo contains the official implementation of Get a Grip. It consists of a dataset of labeled grasps, object meshes, and perceptual data. It also consists of the code for dataset generation and visualization, model training, and grasp planning.

## Project Structure

```
get_a_grip
  ├── assets
  │   └── // Assets such as allegro hand urdf files
  ├── data
  │   └── // Store dataset and output files here
  ├── docs
  │   └── // Documentation
  ├── get_a_grip
  │   ├── // Source code
  |   └── dataset_generation
  |      └── // Generate dataset
  |   └── grasp_planning
  |      └── // Grasp planning training and inference
  |   └── model_training
  |      └── // Neural network training
  |   └── visualization
  |      └── // Visualization tools
  └── scripts
      └── // Useful scripts
```

## Installation

Installation instructions [here](docs/installation.md).

## Quick Start

TODO: Note that we used 4090

### 1. Dataset Download

TODO

### 2. Dataset Visualization

TODO

### 3. Grasp Planning with BPS Sampler and BPS Evaluator (Floating Hand)

TODO

### 4. Grasp Planning with BPS Sampler and BPS Evaluator (Arm + Hand)

TODO

## Dataset

Dataset information [here](docs/dataset.md).

## Visualization Tools

Visualization tools information [here](docs/visualization_tools.md).

## Dataset Generation

Dataset generation instructions [here](docs/dataset_generation.md).

## Model Training

Model training information [here](docs/model_training.md).

## Grasp Planning

Grasp planning information [here](docs/grasp_planning.md).

## Additional Details

Additional details [here](docs/additional_details.md).

## Citation

```
@article{TODO}
```
