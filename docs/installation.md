# Installation

```
cd <path_to_get_a_grip_root>
conda create -n get_a_grip_env python=3.8
conda activate get_a_grip_env

# Install nerf-studio https://docs.nerf.studio/quickstart/installation.html
python -m pip install --upgrade pip
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install setuptools==69.5.1  # Avoid ImportError: cannot import name 'packaging' from 'pkg_resources' (https://github.com/aws-neuron/aws-neuron-sdk/issues/893)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

mkdir thirdparty
cd thirdparty
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
ns-install-cli
cd ..  # back to thirdparty

pip install setuptools==69.5.1  # Avoid ImportError: cannot import name 'packaging' from 'pkg_resources' (https://github.com/aws-neuron/aws-neuron-sdk/issues/893)
pip install numpy==1.23.5  # Compatible with isaacgym (np.float removed in 1.24)

# Install pytorch3d
# If this doesn't work, can try the following from https://github.com/facebookresearch/pytorch3d/discussions/1752
# pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.7+pt2.1.2cu118
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
cd ..  # back to thirdparty

# Install custom kaolin (https://kaolin.readthedocs.io/en/latest/notes/installation.html)
git clone --recursive https://github.com/tylerlum/kaolin
cd kaolin
pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
export IGNORE_TORCH_VER=1  # Build from source on new version
python setup.py develop
cd ..  # back to thirdparty

# install isaacgym (https://developer.nvidia.com/isaac-gym)
cd thirdparty/isaacgym/python
pip install -e .
cd <path_to_get_a_grip_root>

# Install frogger (TODO: Do we keep this?)
git clone https://github.com/tylerlum/frogger.git
cd frogger
git checkout 2024-04-05_Tyler_Frogger_to_DGN
pip install -e .
cd ..  # back to thirdparty

# Install curobo (Library Installation step in https://curobo.org/get_started/1_install_instructions.html#library-installation, but with custom fork)
sudo apt install git-lfs
git clone https://github.com/tylerlum/curobo.git
cd curobo
git lfs pull  # Maybe need to add this (https://github.com/NVlabs/curobo/issues/10)
pip install -e . --no-build-isolation  # ~20 min
python3 -m pytest .  # To verify
cd ..  # back to thirdparty

# Install bps
pip3 install git+https://github.com/sergeyprokudin/bps

# Install other dependencies
# Main functionality
pip install pypose tyro wandb pytorch_kinematics arm_pytorch_utilities mujoco urdf_parser_py pybullet torchviz transforms3d trimesh scipy networkx rtree torchinfo positional_encodings diffusers

# Plotting and visualization
pip install matplotlib plotly kaleido

# Debugging and development
pip install rich ipdb jupyterlab jupytext pandas black clean_loop_timer isaacgym_stubs localscope "pyglet<2"

# Install get_a_grip
cd <path_to_get_a_grip_root>
pip install -e .
```

## Notes on Installation

- `isaacgym` requires Python 3.7 or 3.8.
- If you encounter issues installing third-party dependencies, please take a look at the installation instructions associated with these dependencies. We provide the installation instructions that worked for us, but those instructions may change over time or or change on different hardware.
- There can be challenges with `pytorch3d` depending on the exact python version, pytorch version, and cuda version. Please refer to the pytorch3d installation instructions (https://github.com/facebookresearch/pytorch3d) for more details. We also found the following issue pages to be helpful: https://github.com/facebookresearch/pytorch3d/issues/1401, https://github.com/facebookresearch/pytorch3d/discussions/1752
- The installation has been tested on Ubuntu 20.04 with a NVIDIA RTX 4090 GPU. The installation should work on similar setups, but has not been tested extensively.
