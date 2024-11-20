# Custom Hand

Although our framework is agnostic to the specific hand hardware, this codebase primarily supports the Allegro hand, which has four fingers. 
We have also integrated the LEAP hand into the dataset generation part of our codebase. 

Here, we provide a rough guide on how to add your own hand, which we will call `custom_hand`.

## 1. Create custom hand files

You need to create the following files:

1. Hand urdf file: URDF file of the hand, should have a `palm_link` at the palm. E.g., `assets/leap_hand_simplified/leap_hand_right.urdf`
2. Hand urdf file with virtual joints: Same URDF file as above, but with 6 virtual links and joints for 6D pose control. E.g., `assets/leap_hand_simplified/leap_hand_right_with_virtual_joints.urdf`
3. Contact points file: Points on the hand that should make contact with the object during the grasp. E.g., `assets/leap_hand_simplified/contact_points_precision_grasp.json`
4. Penetration points file: Points on the hand that are used to check for self-penetration. E.g., `assets/leap_hand_simplified/penetration_points.json`

Note: It is best if you can have 0 or few mesh/STL files for collision geometry, as these take much more GPU memory to check for collisions, which slows things down significantly.

Note 2: Setting these points is a bit of a manual, iterative process. For the contact points, we recommend you create a script like so (you will need to manually test and visualize to set the parameters): `get_a_grip/dataset_generation/scripts/generate_leap_contact_points_precision_grasp.py`

Note 3: To make this process easier, you can visualize the hand and points using the following after you complete step 2 (you should complete step 2 even with incomplete files above):

```
python get_a_grip/visualization/scripts/visualize_hand_model.py --hand-model-type CUSTOM
```

It will create a plot like so:

![Visualize_Hand_Screenshot from 2024-11-20 13-44-30](https://github.com/user-attachments/assets/e4353b72-60b4-4120-89d7-4a2062915228)


## 2. Copy leap_hand_info.py and modify

* Copy `get_a_grip/dataset_generation/utils/leap_hand_info.py` to a new file like `get_a_grip/dataset_generation/utils/custom_hand_info.py`

* Modify all uses of `LEAP_HAND_...` with `CUSTOM_HAND_...`.

* Modify all variables in this file to match your hand. For example:

```
# Leap Hand Constants
LEAP_HAND_NUM_FINGERS = 4
LEAP_HAND_NUM_JOINTS = 16
...
LEAP_HAND_FILE = "leap_hand_simplified/leap_hand_right.urdf"
LEAP_HAND_FILE_WITH_VIRTUAL_JOINTS = (
    "leap_hand_simplified/leap_hand_right_with_virtual_joints.urdf"
)
```

Change to
```
# Custom Hand Constants
CUSTOM_HAND_NUM_FINGERS = 5
CUSTOM_HAND_NUM_JOINTS = 20
...
CUSTOM_HAND_FILE = "custom_hand/custom_hand_right.urdf"
CUSTOM_HAND_FILE_WITH_VIRTUAL_JOINTS = (
    "custom_hand_simplified/custom_hand_right_with_virtual_joints.urdf"
)
```

## 3. Update hand_model.py

* Open `get_a_grip/dataset_generation/utils/hand_model.py`.

* Add your hand to `HandModelType`, like

```
class HandModelType(Enum):
    ALLEGRO = auto()
    LEAP = auto()
    CUSTOM = auto()
```

## 4. Update code

Update all parts of the code that rely on these things. For example, search for

```
git grep "leap_hand_info"
git grep "HandModelType"
```

For all instances, modify the code accordingly to update. For example:

```
from get_a_grip.dataset_generation.utils.allegro_hand_info import (
    ALLEGRO_HAND_CONTACT_POINTS_PATH,
    ALLEGRO_HAND_DEFAULT_JOINT_ANGLES,
    ALLEGRO_HAND_DEFAULT_ORIENTATION,
    ALLEGRO_HAND_FINGERTIP_KEYWORDS,
    ALLEGRO_HAND_FINGERTIP_NAMES,
    ALLEGRO_HAND_JOINT_NAMES,
    ALLEGRO_HAND_NUM_FINGERS,
    ALLEGRO_HAND_NUM_JOINTS,
    ALLEGRO_HAND_PENETRATION_POINTS_PATH,
    ALLEGRO_HAND_URDF_PATH,
)
from get_a_grip.dataset_generation.utils.leap_hand_info import (
    LEAP_HAND_CONTACT_POINTS_PATH,
    LEAP_HAND_DEFAULT_JOINT_ANGLES,
    LEAP_HAND_DEFAULT_ORIENTATION,
    LEAP_HAND_FINGERTIP_KEYWORDS,
    LEAP_HAND_FINGERTIP_NAMES,
    LEAP_HAND_JOINT_NAMES,
    LEAP_HAND_NUM_FINGERS,
    LEAP_HAND_NUM_JOINTS,
    LEAP_HAND_PENETRATION_POINTS_PATH,
    LEAP_HAND_URDF_PATH,
)
...
    @property
    def penetration_points_path(self) -> pathlib.Path:
        if self.hand_model_type == HandModelType.ALLEGRO:
            return ALLEGRO_HAND_PENETRATION_POINTS_PATH
        elif self.hand_model_type == HandModelType.LEAP:
            return LEAP_HAND_PENETRATION_POINTS_PATH
        else:
            raise ValueError(f"Unknown hand model type: {self.hand_model_type}")
```

Add in your custom hand:
```
from get_a_grip.dataset_generation.utils.allegro_hand_info import (
    ALLEGRO_HAND_CONTACT_POINTS_PATH,
    ALLEGRO_HAND_DEFAULT_JOINT_ANGLES,
    ALLEGRO_HAND_DEFAULT_ORIENTATION,
    ALLEGRO_HAND_FINGERTIP_KEYWORDS,
    ALLEGRO_HAND_FINGERTIP_NAMES,
    ALLEGRO_HAND_JOINT_NAMES,
    ALLEGRO_HAND_NUM_FINGERS,
    ALLEGRO_HAND_NUM_JOINTS,
    ALLEGRO_HAND_PENETRATION_POINTS_PATH,
    ALLEGRO_HAND_URDF_PATH,
)
from get_a_grip.dataset_generation.utils.leap_hand_info import (
    LEAP_HAND_CONTACT_POINTS_PATH,
    LEAP_HAND_DEFAULT_JOINT_ANGLES,
    LEAP_HAND_DEFAULT_ORIENTATION,
    LEAP_HAND_FINGERTIP_KEYWORDS,
    LEAP_HAND_FINGERTIP_NAMES,
    LEAP_HAND_JOINT_NAMES,
    LEAP_HAND_NUM_FINGERS,
    LEAP_HAND_NUM_JOINTS,
    LEAP_HAND_PENETRATION_POINTS_PATH,
    LEAP_HAND_URDF_PATH,
)
from get_a_grip.dataset_generation.utils.custom_hand_info import (
    CUSTOM_HAND_CONTACT_POINTS_PATH,
    CUSTOM_HAND_DEFAULT_JOINT_ANGLES,
    CUSTOM_HAND_DEFAULT_ORIENTATION,
    CUSTOM_HAND_FINGERTIP_KEYWORDS,
    CUSTOM_HAND_FINGERTIP_NAMES,
    CUSTOM_HAND_JOINT_NAMES,
    CUSTOM_HAND_NUM_FINGERS,
    CUSTOM_HAND_NUM_JOINTS,
    CUSTOM_HAND_PENETRATION_POINTS_PATH,
    CUSTOM_HAND_URDF_PATH,
)
...
    @property
    def penetration_points_path(self) -> pathlib.Path:
        if self.hand_model_type == HandModelType.ALLEGRO:
            return ALLEGRO_HAND_PENETRATION_POINTS_PATH
        elif self.hand_model_type == HandModelType.LEAP:
            return LEAP_HAND_PENETRATION_POINTS_PATH
        elif self.hand_model_type == HandModelType.CUSTOM:
            return CUSTOM_HAND_PENETRATION_POINTS_PATH
        else:
            raise ValueError(f"Unknown hand model type: {self.hand_model_type}")
```

Please see `assets/leap_hand_simplified` and `get_a_grip/dataset_generation/utils/leap_hand_info.py` for more details about how to integrate a different hand into the dataset generation code.
