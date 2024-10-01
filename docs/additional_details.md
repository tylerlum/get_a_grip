# Additional Details

## Frames and Conventions

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/e12abaef-cbbe-4cb4-ae0a-0a25a7935708" alt="Ny" style="width:30%;">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/2274a609-7348-484e-9751-32f3bd04a60a" alt="Oy" style="width:30%;">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/2861558c-436f-4669-8a77-d187d6e57705" alt="H" style="width:30%;">
</p>

### Isaac Gym World Y-up Frame (Wy Frame)

Isaac Gym is set up so that Y-axis is up and gravity points along -Y. An object with 0 translation and 0 rotation is shown below.

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/507903bd-1e6d-4ac2-b4c5-fe68588c95b1" alt="Wy" style="width:30%;">
</p>

### NeRF Y-up Frame (Ny Frame)

The NeRF Y-up frame is the same as the Isaac Gym World Y-up frame. We place the table such that the table surface is at y=0.

When training the NeRF, we add training arguments that explicitly ensure that there is not adjustment to the translation, orientation, or scale so that the frames/dimensions are not different from world frame.

### Object Y-up Frame (Oy Frame)

Object Y-up frame is centered at the object bounding box's center with Y-up.

### Isaacgym => NeRF Image Frames

Following [NeRF Studio data conventions](https://docs.nerf.studio/quickstart/data_conventions.html), we define camera orientation such that X = right, Y = up, Z = optical axis into camera. The translation xyz are the same as Isaac Gym coordinates.

Note that in some intermediate computation, we first compute the camera transform with the following orientation convention (before adding a rotation to modify it).

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/5486b518-2abf-4009-bd7b-0ff58a7736aa" alt="nerfstudio_frames" style="width:30%;">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/b2bf5405-a5b4-4c44-824e-ffe591e6c1d4" alt="intermediate_frames" style="width:27%;">
</p>

## Robot

### Allegro Hand

Although our framework is agnostic to the specific hand hardware, this codebase currently only supports the Allegro hand, which has four fingers.

Allegro wrist pose frame:

- The Allegro Hand's origin is roughly at the base of its middle finger, with Y-axis (Green) along the direction from middle finger to index finger and Z-axis (Blue) along the direction from middle finger base to tip. An Allegro hand with 0 translation, 0 rotation, and 0 joint angles is shown below.

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/bc383cc4-3710-4187-ac5d-d30e0a361453" alt="allegro_frame" style="width:30%;">
</p>

Allegro urdf links:

- link 0 - 3 is fore finger (link 3 tip is fore fingertip)

- link 4 - 7 is middle finger (link 7 tip is middle fingertip)

- link 8 - 11 is ring finger (link 11 tip is ring fingertip)

- link 12 - 15 is thumb (link 15 tip is thumbtip)

Allegro urdf details:

- The urdf comes from [simlabrobotics/allegro_hand_ros_v4](https://github.com/simlabrobotics/allegro_hand_ros_v4/blob/master/src/allegro_hand_description/allegro_hand_description_right.urdf)

- We modify the urdf to have 6 "virtual joints" (one each for translation xyz an rotation RPY), which allows us to move the gripper in 6 DOF (used to move the gripper during grasp evaluation)

- We use the abbreviation "algr" for the Allegro hand in our codebase

### Other Hands

We use the Allegro Hand because it is widely used in the robotics research community and we had access to it for hardware experiments. Our framework is agnostic to the specific hand hardware, so you can use other hands by modifying the codebase. We have integrated the LEAP hand into the dataset generation part of our codebase. Please see `assets/leap_hand_simplified` and `get_a_grip/dataset_generation/utils/leap_hand_info.py` for more details about how to integrate a different hand into the dataset generation code.

### Franka Research 3

Although our framework is agnostic to the specific arm hardware, this codebase currently only supports the Franka hand, which has four fingers.

Franka urdf details:

- We use the abbreviation "fr3" for the Franka Research 3 arm in our codebase

- Our full hardware setup includes a Franka Research 3 arm, a Allegro hand, a wrist-mounted Zed 2i camera, and a table. These are all modeled in the `fr3_algr_zed2i.urdf` (in our custom fork of `curobo`)

- `curobo` requires that the robot be modeled as a set of collision spheres, which needed to be done manually

- `curobo` typically is visualized using Isaac Sim, but we had issues using it, so we use pybullet for visualization

## DexGraspNet

Many major components of the dataset generation code originated from [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet). For additional information about mesh processing, other hand models, and more, please refer to their repository.

We don't use DexGraspNet's [TorchSDF](https://github.com/wrc042/TorchSDF/tree/main) because of build issues on newer torch and cuda versions. Instead, we integrate its source code into [kaolin](https://github.com/NVIDIAGameWorks/kaolin), which is able to build it properly.

## tyro

This codebase makes extensive use of [tyro](https://brentyi.github.io/tyro/), which is a tool for generating command-line interfaces (CLIs) and configuration objects for Python functions. This results in:

- Easy-to-use CLIs for running experiments, where you can add `--help` to the command to get a helpful message.

- Nicely type-annotated Python code that is easy to read and understand for both humans and machines (e.g., IDEs, linters, etc.)

- `dataclass` objects used as configs or args to all scripts

- All scripts having an equivalent function that can be called from Python code with the same arguments passed through a `dataclass` object

Note: more complex scripts may have `Union` arguments that make the `--help` message more complex to use. Here is a quick example.

### Simple Example

```
from dataclasses import dataclass
import tyro


@dataclass
class FullName:
    first: str
    last: str


@dataclass
class Args:
    name: FullName
    age: int = 20


def run_test(args: Args) -> None:
    print(f"args = {tyro.extras.to_yaml(args)}")


def main() -> None:
    args = tyro.cli(Args)
    run_test(args)


if __name__ == "__main__":
    main()
```

```
python test.py --help

usage: test.py [-h] [--age INT] --name.first STR --name.last STR

╭─ options ───────────────────────────────────────────────╮
│ -h, --help              show this help message and exit │
│ --age INT               (default: 20)                   │
╰─────────────────────────────────────────────────────────╯
╭─ name options ──────────────────────────────────────────╮
│ --name.first STR        (required)                      │
│ --name.last STR         (required)                      │
╰─────────────────────────────────────────────────────────╯
```

```
python test.py --name.first Tyler --name.last Lum

args = # tyro YAML.
!dataclass:Args
age: 20
name: !dataclass:FullName
  first: Tyler
  last: Lum
```

### More Complex Example with Union

```
# test2.py
from dataclasses import dataclass
import tyro
from typing import Union


@dataclass
class FullName:
    first: str
    last: str


@dataclass
class Args:
    name: Union[FullName, str]
    age: int = 20


def run_test(args: Args) -> None:
    print(f"args = {tyro.extras.to_yaml(args)}")


def main() -> None:
    args = tyro.cli(Args)
    run_test(args)


if __name__ == "__main__":
    main()
```

Because tyro doesn't know if `name` is a `FullName` or a `str`, it can't provide the full help message right away.

```
python test2.py --help

usage: test2.py [-h] [--age INT] {name:full-name,name:str}

╭─ options ─────────────────────────────────────────╮
│ -h, --help        show this help message and exit │
│ --age INT         (default: 20)                   │
╰───────────────────────────────────────────────────╯
╭─ subcommands ─────────────────────────────────────╮
│ {name:full-name,name:str}                         │
│     name:full-name                                │
│     name:str                                      │
╰───────────────────────────────────────────────────╯
```

Here is an example of deciding to provide a `FullName`, but wanting to see the help message for that.

```
python test2.py name:full-name --help

usage: test2.py name:full-name [-h] --name.first STR --name.last STR

╭─ options ───────────────────────────────────────────────╮
│ -h, --help              show this help message and exit │
╰─────────────────────────────────────────────────────────╯
╭─ name options ──────────────────────────────────────────╮
│ --name.first STR        (required)                      │
│ --name.last STR         (required)                      │
╰─────────────────────────────────────────────────────────╯
```

Note that the `--age` argument is not longer shown in the `--help` message, as the position of `--help` and how many Union options you provide affects the output. This shows how the `--help` message can be more complex to use when `Union` arguments are used.

```
python test2.py name:full-name --name.first Tyler --name.last Lum

args = # tyro YAML.
!dataclass:Args
age: 20
name: !dataclass:FullName
  first: Tyler
  last: Lum
```

### Calling from Python Code

The following code shows how to call the `run_test` function from Python code rather than from the command line.

```
from test2 import run_test, Args, FullName


def main() -> None:
    args = Args(
        name=FullName(first="Tyler", last="Lum"),
    )
    run_test(args)


if __name__ == "__main__":
    main()
```

```
python test3.py

args = # tyro YAML.
!dataclass:Args
age: 20
name: !dataclass:FullName
  first: Tyler
  last: Lum
```
