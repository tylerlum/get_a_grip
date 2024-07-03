from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.optimizer_utils import (
    GraspMetric,
    DepthImageGraspMetric,
    AllegroGraspConfig,
)
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
from tqdm import tqdm
import nerf_grasping
import tyro
import pathlib
import numpy as np
from typing import Optional
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt


@dataclass
class Args:
    grasp_metric: GraspMetricConfig
    grasp_config_dict_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
        / "2023-01-03_mugs_smaller0-075_noise_lightshake_mid_opt"
        / "evaled_grasp_config_dicts"
        / "core-mug-10f6e09036350e92b3f21f1137c3c347_0_0750.npy"
    )
    max_num_grasps: Optional[int] = None
    batch_size: int = 32


def main(cfg: Args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grasp_config_dict = np.load(cfg.grasp_config_dict_path, allow_pickle=True).item()

    if cfg.max_num_grasps is not None:
        print(f"Limiting number of grasps to {cfg.max_num_grasps}")
        for key in grasp_config_dict.keys():
            grasp_config_dict[key] = grasp_config_dict[key][: cfg.max_num_grasps]

    grasp_config = AllegroGraspConfig.from_grasp_config_dict(grasp_config_dict)

    # Create grasp metric
    USE_DEPTH_IMAGES = isinstance(
        cfg.grasp_metric.classifier_config.nerfdata_config, DepthImageNerfDataConfig
    )
    if USE_DEPTH_IMAGES:
        grasp_metric = DepthImageGraspMetric.from_config(
            cfg.grasp_metric,
        ).to(device)
    else:
        grasp_metric = GraspMetric.from_config(
            cfg.grasp_metric,
        ).to(device)
    grasp_metric = grasp_metric.to(device)
    grasp_metric.eval()

    # Evaluate grasp
    with torch.no_grad():
        predicted_pass_prob_list = []
        n_batches = len(grasp_config) // cfg.batch_size
        for batch_i in tqdm(range(n_batches)):
            batch_grasp_config = grasp_config[batch_i * cfg.batch_size : (batch_i + 1) * cfg.batch_size].to(device)
            predicted_failure_prob = grasp_metric.get_failure_probability(batch_grasp_config)
            predicted_pass_prob = 1 - predicted_failure_prob
            predicted_pass_prob_list += predicted_pass_prob.detach().cpu().numpy().tolist()
        if n_batches * cfg.batch_size < len(grasp_config):
            batch_grasp_config = grasp_config[n_batches * cfg.batch_size :].to(device)
            predicted_failure_prob = grasp_metric.get_failure_probability(batch_grasp_config)
            predicted_pass_prob = 1 - predicted_failure_prob
            predicted_pass_prob_list += predicted_pass_prob.detach().cpu().numpy().tolist()
    print(f"Grasp predicted_pass_prob_list: {predicted_pass_prob_list}")

    # Ensure grasp_config was not modified
    output_grasp_config_dict = grasp_config.as_dict()
    assert output_grasp_config_dict.keys()
    for key, val in output_grasp_config_dict.items():
        assert np.allclose(
            val, grasp_config_dict[key], atol=1e-5, rtol=1e-5
        ), f"Key {key} was modified!"

    # Compare to ground truth
    passed_eval = grasp_config_dict["passed_eval"]
    print(f"Passed eval: {passed_eval}")

    # Plot predicted vs. ground truth
    plt.scatter(passed_eval, predicted_pass_prob_list, label="Predicted")
    plt.plot([0, 1], [0, 1], c="r", label="Ground truth")
    plt.xlabel("Ground truth")
    plt.ylabel("Predicted")
    plt.title(
        f"Grasp metric: {cfg.grasp_metric.classifier_config.name} on {cfg.grasp_metric.object_name}"
    )
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = tyro.cli(Args)
    main(cfg)
