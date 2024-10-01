import pathlib
from typing import List, Literal

from nerfstudio.fields.base_field import Field
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import eval_utils


def load_nerf_model(cfg_path: pathlib.Path) -> Model:
    return load_nerf_pipeline(cfg_path).model


def load_nerf_field(cfg_path: pathlib.Path) -> Field:
    return load_nerf_model(cfg_path).field


def load_nerf_pipeline(
    cfg_path: pathlib.Path, test_mode: Literal["test", "val", "inference"] = "test"
) -> Pipeline:
    _, pipeline, _, _ = eval_utils.eval_setup(cfg_path, test_mode=test_mode)
    return pipeline


def get_nerf_configs(nerfcheckpoints_path: pathlib.Path) -> List[pathlib.Path]:
    """
    Returns a list of all the NeRF configs in the given directory, searching recursively
    """
    return list(nerfcheckpoints_path.rglob("nerfacto/*/config.yml"))


def get_latest_nerf_config(nerfcheckpoint_path: pathlib.Path) -> pathlib.Path:
    nerf_configs = list(nerfcheckpoint_path.glob("nerfacto/*/config.yml"))
    assert len(nerf_configs) > 0, f"No NERF configs found in {nerfcheckpoint_path}"
    latest_nerf_config = max(nerf_configs, key=lambda p: p.stat().st_ctime)
    return latest_nerf_config


def get_nerf_configs_through_symlinks(
    nerfcheckpoints_path: pathlib.Path,
) -> List[pathlib.Path]:
    """
    Expects following directory structure:
    <nerfcheckpoints_path>
    ├── <object_name>
    |   ├── nerfacto
    |   |   ├── <timestamp>
    |   |   |   ├── config.yml
    ├── <object_name>
    |   ├── nerfacto
    |   |   ├── <timestamp>
    |   |   |   ├── config.yml
    ...

    rglob doesn't work through symlinks, so can use this instead if <object_name> directories are symlinks
    """
    object_nerfcheckpoint_paths = sorted(
        [
            object_nerfcheckpoint_path
            for object_nerfcheckpoint_path in nerfcheckpoints_path.iterdir()
        ]
    )
    nerf_configs = []
    for object_nerfcheckpoint_path in object_nerfcheckpoint_paths:
        nerfacto_path = object_nerfcheckpoint_path / "nerfacto"
        assert nerfacto_path.exists(), f"{nerfacto_path} does not exist"

        nerf_config = sorted(list(nerfacto_path.rglob("config.yml")))[-1]
        nerf_configs.append(nerf_config)
    return nerf_configs
