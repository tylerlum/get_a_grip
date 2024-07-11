import pathlib
from typing import List

from get_a_grip.utils.parse_object_code_and_scale import (
    is_object_code_and_scale_str,
    parse_object_code_and_scale,
)


def get_object_codes_and_scales_to_process(
    input_object_code_and_scale_strs: List[str],
    meshdata_root_path: pathlib.Path,
    output_folder_path: pathlib.Path,
    continue_ok: bool,
) -> List[str]:
    print(
        f"First 10 in input_object_code_and_scale_strs: {input_object_code_and_scale_strs[:10]}"
    )
    print(
        f"len(input_object_code_and_scale_strs): {len(input_object_code_and_scale_strs)}"
    )

    # Check that all object codes are valid
    for object_code_and_scale_str in input_object_code_and_scale_strs:
        assert is_object_code_and_scale_str(
            object_code_and_scale_str
        ), f"{object_code_and_scale_str} is not valid."

    input_object_codes_set = set(
        [parse_object_code_and_scale(x)[0] for x in input_object_code_and_scale_strs]
    )
    ALL_OBJECT_CODES_SET = set([path.name for path in meshdata_root_path.iterdir()])
    assert set(
        input_object_codes_set
    ).issubset(
        ALL_OBJECT_CODES_SET
    ), f"Only in input_object_codes_set: {set(input_object_codes_set) - ALL_OBJECT_CODES_SET}"

    # Check if any objects have already been processed
    existing_object_code_and_scale_strs = (
        [
            path.stem
            for path in output_folder_path.iterdir()
            if is_object_code_and_scale_str(path.stem)
        ]
        if output_folder_path.exists()
        else []
    )

    if len(existing_object_code_and_scale_strs) > 0 and not continue_ok:
        print(
            f"Found {len(existing_object_code_and_scale_strs)} objects in {output_folder_path}"
        )
        print("Exiting because --continue_ok False was specified.")
        exit()
    elif len(existing_object_code_and_scale_strs) > 0:
        print(
            f"Found {len(existing_object_code_and_scale_strs)} objects in {output_folder_path}"
        )
        print("Continuing because --continue_ok True was specified.")

        input_object_code_and_scale_strs = list(
            set(input_object_code_and_scale_strs)
            - set(existing_object_code_and_scale_strs)
        )
        print(
            f"Continuing with {len(input_object_code_and_scale_strs)} object codes after filtering."
        )
    else:
        print("No objects have been processed yet.")

    return sorted(input_object_code_and_scale_strs)
