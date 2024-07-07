from __future__ import annotations

import pathlib
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pypose as pp
import torch
import tyro
from nerfstudio.fields.base_field import Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.hand_model import HandModel
from get_a_grip.dataset_generation.utils.pose_conversion import (
    hand_config_np_to_pose,
)
from get_a_grip.grasp_planning.config.nerf_evaluator_wrapper_config import (
    NerfEvaluatorWrapperConfig,
)
from get_a_grip.grasp_planning.utils.allegro_grasp_config import (
    AllegroGraspConfig,
)
from get_a_grip.grasp_planning.utils.plot_utils import (
    plot_mesh,
    plot_nerf_densities,
)
from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    lb_Oy,
    ub_Oy,
)
from get_a_grip.model_training.config.nerf_evaluator_config import NerfEvaluatorConfig
from get_a_grip.model_training.models.nerf_evaluator import (
    NerfEvaluator,
)
from get_a_grip.model_training.utils.nerf_grasp_evaluator_batch_data import (
    BatchDataInput,
)
from get_a_grip.model_training.utils.nerf_load_utils import (
    load_nerf_field,
)
from get_a_grip.model_training.utils.nerf_ray_utils import (
    get_ray_origins_finger_frame,
    get_ray_samples,
)
from get_a_grip.model_training.utils.nerf_utils import (
    get_densities_in_grid,
)
from get_a_grip.model_training.utils.point_utils import (
    transform_point,
)


class NerfEvaluatorWrapper(torch.nn.Module):
    """
    Wrapper for NeRF + grasp nerf_evaluator to evaluate
    a particular AllegroGraspConfig.
    """

    def __init__(
        self,
        nerf_field: Field,
        nerf_evaluator_model: NerfEvaluator,
        fingertip_config: EvenlySpacedFingertipConfig,
        X_N_Oy: np.ndarray,
    ) -> None:
        super().__init__()
        self.nerf_field = nerf_field
        self.nerf_evaluator_model = nerf_evaluator_model
        self.fingertip_config = fingertip_config
        self.X_N_Oy = X_N_Oy
        self.ray_origins_finger_frame = get_ray_origins_finger_frame(fingertip_config)

    def forward(
        self,
        grasp_config: AllegroGraspConfig,
    ) -> torch.Tensor:
        ray_samples = self.compute_ray_samples(grasp_config)

        # Query NeRF at RaySamples.
        densities = self.compute_nerf_densities(
            ray_samples,
        )
        assert densities.shape == (
            len(grasp_config),
            4,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )

        # Query NeRF in grid
        # BRITTLE: Require that the nerf_evaluator_model has the word "Global" in it if needed
        need_to_query_global = (
            "global" in self.nerf_evaluator_model.__class__.__name__.lower()
        )
        if need_to_query_global:
            lb_N = transform_point(T=self.X_N_Oy, point=lb_Oy)
            ub_N = transform_point(T=self.X_N_Oy, point=ub_Oy)
            nerf_densities_global, _query_points_global_N = get_densities_in_grid(
                field=self.nerf_field,
                lb=lb_N,
                ub=ub_N,
                num_pts_x=NERF_DENSITIES_GLOBAL_NUM_X,
                num_pts_y=NERF_DENSITIES_GLOBAL_NUM_Y,
                num_pts_z=NERF_DENSITIES_GLOBAL_NUM_Z,
            )
            nerf_densities_global = (
                torch.from_numpy(nerf_densities_global)
                .float()[None, ...]
                .repeat_interleave(len(grasp_config), dim=0)
            )
        else:
            nerf_densities_global, _query_points_global_N = None, None

        batch_data_input = BatchDataInput(
            nerf_densities=densities,
            grasp_transforms=grasp_config.grasp_frame_transforms,
            fingertip_config=self.fingertip_config,
            grasp_configs=grasp_config.as_tensor(),
            nerf_densities_global=(
                nerf_densities_global if nerf_densities_global is not None else None
            ),
        ).to(grasp_config.device)

        # Pass grasp transforms, densities into nerf_evaluator.
        return self.nerf_evaluator_model.get_failure_probability(batch_data_input)

    def compute_ray_samples(
        self,
        grasp_config: AllegroGraspConfig,
    ) -> torch.Tensor:
        # Let Oy be object yup frame (centroid of object)
        # Let N be nerf frame (where the nerf is defined)
        # For NeRFs trained from sim data, Oy and N are the same.
        # But for real-world data, Oy and N are different (N is a point on the table, used as NeRF origin)
        # When sampling from the NeRF, we must give ray samples in N frame
        # But nerf_evaluator is trained on Oy frame
        # Thus, we must transform grasp_frame_transforms from Oy frame to N frame
        # Let Fi be the finger frame (origin at each fingertip i)
        # Let p_Fi be points in Fi frame
        # self.ray_origins_finger_frame = p_Fi
        # grasp_frame_transforms = T_{Oy <- Fi}
        # X_N_Oy = T_{N <- Oy}
        # TODO: Batch this to avoid OOM (refer to create_nerf_grasp_dataset.py)

        # Prepare transforms
        T_Oy_Fi = grasp_config.grasp_frame_transforms
        assert T_Oy_Fi.lshape == (len(grasp_config), grasp_config.num_fingers)

        assert self.X_N_Oy.shape == (
            4,
            4,
        )
        X_N_Oy_repeated = (
            torch.from_numpy(self.X_N_Oy)
            .float()
            .unsqueeze(dim=0)
            .repeat_interleave(len(grasp_config) * grasp_config.num_fingers, dim=0)
            .reshape(len(grasp_config), grasp_config.num_fingers, 4, 4)
        )

        T_N_Oy = pp.from_matrix(
            X_N_Oy_repeated.to(T_Oy_Fi.device),
            pp.SE3_type,
        )

        # Transform grasp_frame_transforms to nerf frame
        T_N_Fi = T_N_Oy @ T_Oy_Fi

        # Generate RaySamples.
        ray_samples = get_ray_samples(
            self.ray_origins_finger_frame,
            T_N_Fi,
            self.fingertip_config,
        )
        return ray_samples

    def compute_nerf_densities(
        self,
        ray_samples,
    ) -> torch.Tensor:
        # Query NeRF at RaySamples.
        densities = self.nerf_field.get_density(ray_samples.to("cuda"))[0][
            ..., 0
        ]  # Shape [B, 4, n_x, n_y, n_z]
        return densities

    def get_failure_probability(
        self,
        grasp_config: AllegroGraspConfig,
    ) -> torch.Tensor:
        return self(grasp_config)

    def DEBUG_PLOT(
        self,
        grasp_config: AllegroGraspConfig,
    ) -> None:
        # TODO: Clean this up a lot
        # DEBUG PLOT
        ray_samples = self.compute_ray_samples(grasp_config)

        # Query NeRF at RaySamples.
        densities = self.compute_nerf_densities(
            ray_samples,
        )
        assert densities.shape == (
            len(grasp_config),
            4,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )

        # Query NeRF in grid
        # BRITTLE: Require that the nerf_evaluator_model has the word "Global" in it if needed
        need_to_query_global = (
            "global" in self.nerf_evaluator_model.__class__.__name__.lower()
        )
        if need_to_query_global:
            lb_N = transform_point(T=self.X_N_Oy, point=lb_Oy)
            ub_N = transform_point(T=self.X_N_Oy, point=ub_Oy)
            nerf_densities_global, query_points_global_N = get_densities_in_grid(
                field=self.nerf_field,
                lb=lb_N,
                ub=ub_N,
                num_pts_x=NERF_DENSITIES_GLOBAL_NUM_X,
                num_pts_y=NERF_DENSITIES_GLOBAL_NUM_Y,
                num_pts_z=NERF_DENSITIES_GLOBAL_NUM_Z,
            )
            nerf_densities_global = (
                torch.from_numpy(nerf_densities_global)
                .float()[None, ...]
                .repeat_interleave(len(grasp_config), dim=0)
            )
        else:
            nerf_densities_global, query_points_global_N = None, None

        from pathlib import Path

        import trimesh

        if Path("/tmp/mesh_viz_object.obj").exists():
            mesh = trimesh.load("/tmp/mesh_viz_object.obj")
        else:
            mesh = None

        # Do not need this, just for debugging
        query_points_global_N = (
            torch.from_numpy(query_points_global_N)
            .float()[None, ...]
            .repeat_interleave(len(grasp_config), dim=0)
        )
        all_query_points = ray_samples.frustums.get_positions()

        for batch_idx in range(2):
            fig = go.Figure()
            N_FINGERS = 4
            for i in range(N_FINGERS):
                plot_nerf_densities(
                    fig=fig,
                    densities=densities[batch_idx, i].reshape(-1),
                    query_points=all_query_points[batch_idx, i].reshape(-1, 3),
                    name=f"finger_{i}",
                    opacity=0.2,
                )
            if need_to_query_global and nerf_densities_global is not None:
                nerf_densities_global_flattened = nerf_densities_global[
                    batch_idx
                ].reshape(-1)
                query_points_global_N_flattened = query_points_global_N[
                    batch_idx
                ].reshape(-1, 3)
                plot_nerf_densities(
                    fig=fig,
                    densities=nerf_densities_global_flattened[
                        nerf_densities_global_flattened > 15
                    ],
                    query_points=query_points_global_N_flattened[
                        nerf_densities_global_flattened > 15
                    ],
                    name="global",
                )

            if mesh is not None:
                plot_mesh(fig=fig, mesh=mesh)

            wrist_trans_array = (
                grasp_config.wrist_pose.translation().detach().cpu().numpy()
            )
            wrist_rot_array = (
                grasp_config.wrist_pose.rotation().matrix().detach().cpu().numpy()
            )
            joint_angles_array = grasp_config.joint_angles.detach().cpu().numpy()

            # Put into transforms X_Oy_H_array
            B = len(grasp_config)
            X_Oy_H_array = np.repeat(np.eye(4)[None, ...], B, axis=0)
            assert X_Oy_H_array.shape == (B, 4, 4)
            X_Oy_H_array[:, :3, :3] = wrist_rot_array
            X_Oy_H_array[:, :3, 3] = wrist_trans_array

            X_N_H_array = np.repeat(np.eye(4)[None, ...], B, axis=0)
            for i in range(B):
                X_N_H_array[i] = self.X_N_Oy @ X_Oy_H_array[i]

            device = "cuda"
            hand_model = HandModel(device=device)

            # Compute pregrasp and target hand poses
            trans_array = X_N_H_array[:, :3, 3]
            rot_array = X_N_H_array[:, :3, :3]

            pregrasp_hand_pose = hand_config_np_to_pose(
                trans_array, rot_array, joint_angles_array
            ).to(device)

            # Get plotly data
            hand_model.set_parameters(pregrasp_hand_pose)
            pregrasp_plot_data = hand_model.get_plotly_data(i=batch_idx, opacity=1.0)

            for x in pregrasp_plot_data:
                fig.add_trace(x)
            # Add title with idx
            fig.update_layout(title_text=f"Batch idx {batch_idx}")
            fig.show()

    @classmethod
    def from_config(
        cls,
        nerf_evaluator_wrapper_config: NerfEvaluatorWrapperConfig,
        console: Optional[Console] = None,
    ) -> NerfEvaluatorWrapper:
        assert nerf_evaluator_wrapper_config.X_N_Oy is not None
        return cls.from_configs(
            nerf_config=nerf_evaluator_wrapper_config.nerf_config,
            nerf_evaluator_config=nerf_evaluator_wrapper_config.nerf_evaluator_config,
            X_N_Oy=nerf_evaluator_wrapper_config.X_N_Oy,
            nerf_evaluator_checkpoint=nerf_evaluator_wrapper_config.nerf_evaluator_checkpoint,
            console=console,
        )

    @classmethod
    def from_configs(
        cls,
        nerf_config: pathlib.Path,
        nerf_evaluator_config: NerfEvaluatorConfig,
        X_N_Oy: np.ndarray,
        nerf_evaluator_checkpoint: int = -1,
        console: Optional[Console] = None,
    ) -> NerfEvaluatorWrapper:
        # Load nerf
        with (
            Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description} "),
                TimeElapsedColumn(),
                console=console,
            )
            if console is not None
            else nullcontext()
        ) as progress:
            task = (
                progress.add_task("Loading NeRF", total=1)
                if progress is not None
                else None
            )

            nerf_field = load_nerf_field(nerf_config)

            if progress is not None and task is not None:
                progress.update(task, advance=1)

        # Load nerf_evaluator
        nerf_evaluator = load_nerf_evaluator(
            nerf_evaluator_config=nerf_evaluator_config,
            nerf_evaluator_checkpoint=nerf_evaluator_checkpoint,
            console=console,
        )

        return cls(
            nerf_field,
            nerf_evaluator,
            nerf_evaluator_config.nerfdata_config.fingertip_config,
            X_N_Oy,
        )


def load_nerf_evaluator(
    nerf_evaluator_config: NerfEvaluatorConfig,
    nerf_evaluator_checkpoint: int = -1,
    console: Optional[Console] = None,
) -> NerfEvaluator:
    # Load nerf_evaluator
    with (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description} "),
            console=console,
        )
        if console is not None
        else nullcontext()
    ) as progress:
        task = (
            progress.add_task("Loading nerf_evaluator", total=1)
            if progress is not None
            else None
        )

        # (should device thing be here? probably since saved on gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nerf_evaluator = (
            nerf_evaluator_config.model_config.get_nerf_evaluator_from_fingertip_config(
                fingertip_config=nerf_evaluator_config.nerfdata_config.fingertip_config,
                n_tasks=nerf_evaluator_config.task_type.n_tasks,
            )
        ).to(device)

        # Load nerf_evaluator weights
        assert nerf_evaluator_config.checkpoint_workspace.output_dir.exists(), f"checkpoint_workspace.output_dir does not exist at {nerf_evaluator_config.checkpoint_workspace.output_dir}"
        print(
            f"Loading checkpoint ({nerf_evaluator_config.checkpoint_workspace.output_dir})..."
        )

        output_checkpoint_paths = (
            nerf_evaluator_config.checkpoint_workspace.output_checkpoint_paths
        )
        assert (
            len(output_checkpoint_paths) > 0
        ), f"No checkpoints found in {nerf_evaluator_config.checkpoint_workspace.output_checkpoint_paths}"
        assert (
            nerf_evaluator_checkpoint < len(output_checkpoint_paths)
        ), f"Requested checkpoint {nerf_evaluator_checkpoint} does not exist in {nerf_evaluator_config.checkpoint_workspace.output_checkpoint_paths}"
        checkpoint_path = output_checkpoint_paths[nerf_evaluator_checkpoint]

        checkpoint = torch.load(checkpoint_path)
        nerf_evaluator.load_state_dict(checkpoint["nerf_evaluator"])
        nerf_evaluator.load_state_dict(torch.load(checkpoint_path)["nerf_evaluator"])

        if progress is not None and task is not None:
            progress.update(task, advance=1)

    return nerf_evaluator


@dataclass
class NerfEvaluatorWrapperCmdLineArgs:
    nerf_evaluator_wrapper: NerfEvaluatorWrapperConfig = field(
        default_factory=NerfEvaluatorWrapperConfig
    )
    grasp_config_dict_path: pathlib.Path = (
        get_data_folder()
        / "NEW_DATASET/final_evaled_grasp_config_dicts_train/core-bottle-11fc9827d6b467467d3aa3bae1f7b494_0_0726.npy"
    )
    max_num_grasps: Optional[int] = None
    batch_size: int = 32


def main(cfg: NerfEvaluatorWrapperCmdLineArgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grasp_config_dict = np.load(cfg.grasp_config_dict_path, allow_pickle=True).item()

    if cfg.max_num_grasps is not None:
        print(f"Limiting number of grasps to {cfg.max_num_grasps}")
        for key in grasp_config_dict.keys():
            grasp_config_dict[key] = grasp_config_dict[key][: cfg.max_num_grasps]

    grasp_config = AllegroGraspConfig.from_grasp_config_dict(grasp_config_dict)

    # Create grasp metric
    nerf_evaluator_wrapper = NerfEvaluatorWrapper.from_config(
        cfg.nerf_evaluator_wrapper,
    ).to(device)
    nerf_evaluator_wrapper = nerf_evaluator_wrapper.to(device)
    nerf_evaluator_wrapper.eval()

    # Evaluate grasp
    with torch.no_grad():
        predicted_pass_prob_list = []
        n_batches = len(grasp_config) // cfg.batch_size
        for batch_i in tqdm(range(n_batches)):
            batch_grasp_config = grasp_config[
                batch_i * cfg.batch_size : (batch_i + 1) * cfg.batch_size
            ].to(device)
            predicted_failure_prob = nerf_evaluator_wrapper.get_failure_probability(
                batch_grasp_config
            )
            predicted_pass_prob = 1 - predicted_failure_prob
            predicted_pass_prob_list += (
                predicted_pass_prob.detach().cpu().numpy().tolist()
            )
        if n_batches * cfg.batch_size < len(grasp_config):
            batch_grasp_config = grasp_config[n_batches * cfg.batch_size :].to(device)
            predicted_failure_prob = nerf_evaluator_wrapper.get_failure_probability(
                batch_grasp_config
            )
            predicted_pass_prob = 1 - predicted_failure_prob
            predicted_pass_prob_list += (
                predicted_pass_prob.detach().cpu().numpy().tolist()
            )
    print(f"Grasp predicted_pass_prob_list: {predicted_pass_prob_list}")

    # Ensure grasp_config was not modified
    output_grasp_config_dict = grasp_config.as_dict()
    assert output_grasp_config_dict.keys()
    for key, val in output_grasp_config_dict.items():
        assert np.allclose(
            val, grasp_config_dict[key], atol=1e-5, rtol=1e-5
        ), f"Key {key} was modified!"

    # Compare to ground truth
    y_PGS = grasp_config_dict["y_PGS"]
    print(f"Passed eval: {y_PGS}")

    # Plot predicted vs. ground truth
    plt.scatter(y_PGS, predicted_pass_prob_list, label="Predicted")
    plt.plot([0, 1], [0, 1], c="r", label="Ground truth")
    plt.xlabel("Ground truth")
    plt.ylabel("Predicted")
    plt.title(
        f"Grasp metric: {cfg.nerf_evaluator_wrapper.nerf_evaluator_config.name} on {cfg.nerf_evaluator_wrapper.object_name}"
    )
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = tyro.cli(NerfEvaluatorWrapperCmdLineArgs)
    main(cfg)
