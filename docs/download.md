# Download

## Overview

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

Additional details:

- We recommend using the small version for testing and visualization, and we recommend using the large version for model training.

- For dataset generation, you only need the `meshdata`.

```
python download.py \
--download_url <download_url> \
--include_meshdata True
```

- For model training, you only need the `meshdata`, `final_evaled_grasp_config_dicts`, `nerfdata`, `nerfcheckpoints`, and `point_clouds`.

```
python download.py \
--download_url <download_url> \
--include_meshdata True \
--dataset_size large \
--include_final_evaled_grasp_config_dicts True \
--include_nerfdata True \
--include_point_clouds True \
--include_nerfcheckpoints True
```

- The largest part of the dataset is the nerfcheckpoints. This part contains trained NeRF model weights for each object at multiple scales. You can set `--include_nerfcheckpoints False` to exclude this part of the dataset, and then regenerate this yourself.

- The second largest part of the dataset is the nerfdata. This part contains 100 posed RGB images of each object at multiple scales. You can set `--include_nerfdata False` to exclude this part of the dataset, and then regenerate this yourself.

- Under the hood, each component of the dataset is contained in a zip file that can be accessed through a link. The download script will download these zip files and unzip them into the correct directory.

## Dataset Details

### final_evaled_grasp_config_dicts

Directory structure:

```
final_evaled_grasp_config_dicts
├── <object_code_and_scale_str>.npy
├── <object_code_and_scale_str>.npy
├── <object_code_and_scale_str>.npy
├── ...
└── <object_code_and_scale_str>.npy
```

Our definition of grasps from the paper:

```
We parameterize precision grasps as G = (T_OH, theta, d_1, ..., d_n_f):
- T_OH: The hand pose relative to the object, belongs to the space SE(3).
- theta: The pre-grasp joint configuration, belongs to the space R^n_j.
- d_i: The direction in which the ith fingertip moves during the grasp, belongs to the space S^2.

The grasp data are stored as tuples {(G^k, y^k_coll, y^k_pick, y^k_PGS)}, where each y^k_* is a distinct smooth label for grasp G^k generated in simulation. The labels are as follows:
- y_coll: Indicates a grasp with unwanted collisions.
- y_pick: Indicates that the simulated pick was successful.
- y_PGS: Represents the probability of grasp success, defined as the conjunction of the two previous conditions.

Each y^k_* belongs to the range [0, 1].
```

Next, we specify what the grasp file format should look like. Grasps are stored in .npy files as config dicts. Each stored .npy file will be in the form <object*code_and_scale_str>.npy (eg. mug_0_1000.npy for object_code=mug, object_scale=0.1000 <object_code>*<object_scale_4_decimals>.npy), which stores a config dict. Each config dict contains grasp information for a batch of grasps associated with this object code and object scale.

Note that all config dicts should store T_OH and d_i in Oy frame.

You can read in this file like so:

```
config_dict = np.load('mug_0_1000.npy', allow_pickle=True).item()
config_dict.keys()
```

There are a few types of config_dicts that typically stack upon one another. We assume that for all keys in the config_dict, the associated value is a numpy ndarray of shape (B, ...). For example (B,) or (B, 3) or (B, 3, 3).

#### Hand Config Dict

Specify the pregrasp wrist pose and joint angles of the hand:

```
hand_config_dict['trans'].shape == (batch_size, 3)
hand_config_dict['rot'].shape == (batch_size, 3, 3)
hand_config_dict['joint_angles'].shape == (batch_size, 16)
```

It may also have the start wrist pose and joint angles, which refers to what those values were from the start of optimization. This is the same as the above, but with keys ending in '\_start'. This is purely for debugging purposes.

#### Grasp Config Dict

Has the same as the hand_config_dict, but also has:

```
grasp_config_dict['grasp_orientations'].shape == (batch_size, n_fingers, 3, 3)
```

Note that `grasp_orientations` refer to rotation matrices that specify the direction and orientation that each finger should move along to complete a grasp, with the z-dim along the grasp approach direction and the y-dim along the finger to fingertip direction (modified to be perpendicular to z). The grasp directions (d_i) can be accessed by `grasp_config_dict['grasp_orientations'][:, i, :, 0]. The reason this full rotation matrix is stored is to have access to the fingertip orientation without requiring forward kinematics.

#### Evaled Grasp Config Dict

Has the same as the grasp_config_dict, but also has:

```
evaled_grasp_config_dict['y_PGS'].shape == (batch_size,)
evaled_grasp_config_dict['y_pick'].shape == (batch_size,)
evaled_grasp_config_dict['y_coll'].shape == (batch_size,)
```

#### Optimized Grasp Config Dict

Has the same as the grasp_config_dict, but also has:

```
optimized_grasp_config_dict['loss'].shape == (batch_size,)
```

Where loss refer to predicted failure probabilities (1 is bad, 0 is good)

#### Grasps in Neural Networks

When grasps are used as input to or output from neural networks, we need a good way to represent them. We choose to represent the grasps as a compact vector of (xyz, rot6d, theta, d_i, ..., d_n_f). This is 37D (3 + 6 + 16 + 4\*3) for the Allegro hand.

- xyz: The wrist position in Oy frame.
- rot6d: The wrist rotation in Oy frame, represented as a 6D vector (first two columns of the rotation matrix)
- theta: The pre-grasp joint configuration
- d_i: The direction in which the ith fingertip moves during the grasp in Oy frame, belongs to the space S^2.

We try to use the word "grasp config" or "grasp config dict" when referring to the information in the dicts described above, and "grasp" when referring to the compact vector representation.

### meshdata

Directory structure:

```
meshdata
├── <object_code>
│   └── coacd
│       ├── coacd_convex_piece_0.obj
│       ├── coacd_convex_piece_1.obj
│       ├── coacd_convex_piece_2.obj
│       ├── coacd_convex_piece_3.obj
│       ├── coacd_convex_piece_4.obj
│       ├── coacd_convex_piece_5.obj
│       ├── coacd_convex_piece_6.obj
│       ├── coacd_convex_piece_7.obj
│       ├── coacd.urdf
│       ├── decomposed_log.txt
│       ├── decomposed.obj
│       ├── decomposed.wrl
│       └── model.config
└── ...
```

`coacd.urdf` is used for simulations, and is made up of the `coacd_convex_piece_X.obj` meshes. `decomposed.obj` is used for grasp generation, and is a single mesh of the object.

The objects are all:

- Scaled such that the largest object bounding box extent is 2 meters (this is very large, so we use small object scales)

- Centered at the center of object bounding box

### nerfdata

Directory structure:

```
nerfdata
├── <object_code_and_scale_str>
│   ├── images
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── transforms.json
├── <object_code_and_scale_str>
│   ├── images
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── transforms.json
├── <object_code_and_scale_str>
│   ├── images
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── transforms.json
└── ...
```

`images` contains a set of images of the object from different angles. `transforms.json` has the camera pose associated with each image.

### nerfcheckpoints

Directory structure:

```
nerfcheckpoints
├── <object_code_and_scale_str>
│   └── nerfacto
│       └── <datetime_str>
│           ├── config.yml
│           ├── dataparser_transforms.json
│           └── nerfstudio_models
│               └── step-000000399.ckpt
├── <object_code_and_scale_str>
│   └── nerfacto
│       └── <datetime_str>
│           ├── config.yml
│           ├── dataparser_transforms.json
│           └── nerfstudio_models
│               └── step-000000399.ckpt
├── <object_code_and_scale_str>
│   └── nerfacto
│       └── <datetime_str>
│           ├── config.yml
│           ├── dataparser_transforms.json
│           └── nerfstudio_models
│               └── step-000000399.ckpt
└── ...
```

`step-000000399.ckpt` contains the NeRF model weights and `config.yml` contains the NeRF data and model configuration information.

Note: The `config.yml` files may store absolute paths depending on the exact parameters used for training. When used on a different machine with different paths, these absolute paths may need to be updated to relative paths to be used properly.

View a NeRF with:

```
ns-viewer --load-config <path_to_config.yml>
```

### point_clouds

Directory structure:

```
point_clouds
├── <object_code_and_scale_str>
│   └── point_cloud.ply
├── <object_code_and_scale_str>
│   └── point_cloud.ply
├── <object_code_and_scale_str>
│   └── point_cloud.ply
└── ...
```

`point_cloud.ply` can be read using Open3D (https://www.open3d.org/). It is generated using nerfstudio with a request of 5000 points. However, the number of points is not exactly 5000 due to nerfstudio's sampling method, so this can be handled downstream.

## Model Details

### bps_evaluator_model

```
bps_evaluator_model
└── <datetime>
    └── <filename>.pth
```

### bps_sampler_model

```
bps_sampler_model
└── <datetime>
    └── <filename>.pth
```

### nerf_evaluator_model

```
nerf_evaluator_model
└── <datetime>
    └── <filename>.pth
    └── config.yaml
```

### nerf_sampler_model

```
nerf_sampler_model
└── <datetime>
    └── <filename>.pth
```

## Fixed Sampler Grasp Config Dicts

Our fixed sampler requires a set of good grasps to sample from. We provide two types of fixed sampler grasp config dicts:

```
fixed_sampler_grasp_config_dicts
└── given
    ├── all_good_grasps.py
    └── one_good_grasp_per_object.py
```