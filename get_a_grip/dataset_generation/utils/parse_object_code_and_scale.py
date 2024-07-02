from typing import Tuple

"""
ASSUMPTIONS:
* Exactly 4 decimal places for the scale
* All scales are less than 10
* The "." character is replaced with "_" in the scale
* Scale represented as a string of 6 characters (e.g., 0.1 => "0_1000")
"""


def parse_object_code_and_scale(object_code_and_scale_str: str) -> Tuple[str, float]:
    """
    Parses a string containing an object code and its scale into a tuple of (object_code, object_scale).

    Args:
        object_code_and_scale_str (str): The input string containing the object code and scale
                                         where the scale is represented as a string of 6 characters.

    Returns:
        Tuple[str, float]: A tuple containing the object code (str) and the object scale (float).

    Example:
        >>> parse_object_code_and_scale('mug_0_1000')
        ('mug', 0.1)
    """
    object_code = object_code_and_scale_str[:-7]
    object_scale = float(object_code_and_scale_str[-6:].replace("_", "."))
    return object_code, object_scale


def object_code_and_scale_to_str(object_code: str, object_scale: float) -> str:
    """
    Converts an object code and its scale into a formatted string.

    Args:
        object_code (str): The code representing the object.
        object_scale (float): The scale of the object.

    Returns:
        str: A formatted string containing the object code and scale
             where the scale is represented as a string of 6 characters.

    Example:
        >>> object_code_and_scale_to_str('mug', 0.1)
        'mug_0_1000'
    """
    object_code_and_scale_str = f"{object_code}_{object_scale:.4f}".replace(".", "_")
    return object_code_and_scale_str
