# Config management

For this project, we use `tyro` to manage our experiment configs.

In general, the configs are nested, with the structure:
```
GraspMetricConfig:
    - GraspOptimizerConfig
    - ClassifierConfig
        - NerfDatasetConfig
```

## Workflow

Our pipeline on the `nerf_grasping` side is as follows:
1. Run `dataset/Create_DexGraspNet_NeRF_Grasps_Dataset.py` to generate a dataset for the classifier, and an associated `NerfDatasetConfig`.
2. Run `learned_metric/Train_DexGraspNet_NeRF_Grasp_Metric.py` to train a classifier, and save an associated `ClassifierConfig`. The constructor for `ClassifierConfig` takes a special argument, `nerfdata_cfg_path`,
that will load the data config params from file.
3. Run `optimizer.py` to optimize grasps using the learned classifier, and save them to file. Similarly to the `ClassifierConfig`, the `GraspMetricConfig` takes a special argument `classifier_config_path` to determine which classifier to load from data. 