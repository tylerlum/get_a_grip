import math
import multiprocessing
import os
import pathlib
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
import tyro
from clean_loop_timer import LoopTimer
from torch.multiprocessing import set_start_method
from tqdm import tqdm

import wandb
from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.energy import (
    ENERGY_NAME_TO_SHORTHAND_DICT,
    ENERGY_NAMES,
    cal_energy,
)
from get_a_grip.dataset_generation.utils.hand_model import HandModel, HandModelType
from get_a_grip.dataset_generation.utils.initializations import initialize_convex_hull
from get_a_grip.dataset_generation.utils.object_model import ObjectModel
from get_a_grip.dataset_generation.utils.optimizer import Annealing
from get_a_grip.dataset_generation.utils.pose_conversion import pose_to_hand_config
from get_a_grip.dataset_generation.utils.process_utils import (
    get_object_codes_and_scales_to_process,
)
from get_a_grip.utils.parse_object_code_and_scale import (
    object_code_and_scale_to_str,
    parse_object_code_and_scale,
)
from get_a_grip.utils.seed import set_seed

try:
    set_start_method("spawn")
except RuntimeError:
    pass


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
np.seterr(all="raise")


@dataclass
class GenerateHandConfigDictsArgs:
    # experiment settings
    meshdata_root_path: pathlib.Path = get_data_folder() / "meshdata"
    input_object_code_and_scales_txt_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/nerfdata_settled_successes.txt"
    )
    output_hand_config_dicts_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/hand_config_dicts"
    )
    hand_model_type: HandModelType = HandModelType.ALLEGRO
    randomize_order_seed: Optional[int] = datetime.now().microsecond
    optimization_seed: Optional[int] = None
    batch_size_each_object: int = 250
    n_objects_per_batch: int = (
        20  # Runs batch_size_each_object * n_objects_per_batch grasps per GPU
    )
    n_iter: int = 2000
    use_multiprocess: bool = False

    # Logging
    use_wandb: bool = False
    wandb_name: str = ""
    wandb_entity: str = "tylerlum"
    wandb_project: str = "get_a_grip_datagen"
    wandb_visualization_freq: Optional[int] = 50

    # hyper parameters
    switch_possibility: float = 0.5
    mu: float = 0.98
    step_size: float = 0.005
    stepsize_period: int = 50
    starting_temperature: float = 18
    annealing_period: int = 30
    temperature_decay: float = 0.95
    n_contacts_per_finger: int = 5
    w_fc: float = 0.5
    w_dis: float = 500
    w_pen: float = 300.0
    w_spen: float = 100.0
    w_joints: float = 1.0
    w_ff: float = 3.0
    w_fp: float = 0.0
    w_tpen: float = 100.0
    object_num_surface_samples: int = 5000
    object_num_samples_calc_penetration_energy: int = 500

    # initialization settings
    jitter_strength: float = 0.1
    distance_lower: float = 0.2
    distance_upper: float = 0.3
    theta_lower: float = -math.pi / 6
    theta_upper: float = math.pi / 6

    # energy function params
    thres_dis: float = 0.015
    thres_pen: float = 0.015

    # store extra grasps mid optimization
    store_grasps_mid_optimization_freq: Optional[int] = None
    store_grasps_mid_optimization_iters: Optional[List[int]] = None

    # Continue from previous run
    continue_ok: bool = True


def create_visualization_figure(
    hand_model: HandModel,
    object_model: ObjectModel,
    idx_to_visualize: int,
) -> Tuple[go.Figure, str]:
    fig_title = f"hand_object_visualization_{idx_to_visualize}"
    fig = go.Figure(
        layout=go.Layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
            ),
            showlegend=True,
            title=fig_title,
        )
    )
    plots = [
        *hand_model.get_plotly_data(
            i=idx_to_visualize,
            opacity=1.0,
            with_contact_points=True,
            with_contact_candidates=True,
        ),
        *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
    ]
    for plot in plots:
        fig.add_trace(plot)
    return fig, fig_title


def get_energy_term_log_dict(
    unweighted_energy_matrix: torch.Tensor,
    weighted_energy_matrix: torch.Tensor,
    idx_to_visualize: int,
) -> Dict[str, Any]:
    log_dict = {}
    for i, energy_name in enumerate(ENERGY_NAMES):
        shorthand = ENERGY_NAME_TO_SHORTHAND_DICT[energy_name]
        uw_shorthand = f"unweighted_{shorthand}"
        log_dict.update(
            {
                uw_shorthand: unweighted_energy_matrix[:, i].mean().item(),
                shorthand: weighted_energy_matrix[:, i].mean().item(),
                f"{uw_shorthand}_{idx_to_visualize}": unweighted_energy_matrix[
                    idx_to_visualize, i
                ].item(),
                f"{shorthand}_{idx_to_visualize}": weighted_energy_matrix[
                    idx_to_visualize, i
                ].item(),
            }
        )
    return log_dict


def save_hand_config_dicts(
    hand_model: HandModel,
    object_model: ObjectModel,
    object_codes: List[str],
    object_scales: List[float],
    hand_pose_start: torch.Tensor,
    energy: torch.Tensor,
    unweighted_energy_matrix: torch.Tensor,
    output_folder_path: pathlib.Path,
) -> None:
    """
    Save results to output_folder_path:

    <output_folder_path>
    ├── <object_code>_<object_scale>.npy
    ├── <object_code>_<object_scale>.npy
    ├── <object_code>_<object_scale>.npy
    ├── ...
    """
    assert object_model.object_scale_tensor is not None
    num_objects, num_grasps_per_object = object_model.object_scale_tensor.shape
    assert len(object_codes) == num_objects
    assert hand_pose_start.shape[0] == num_objects * num_grasps_per_object
    correct_object_scales = (
        torch.Tensor(object_scales)
        .unsqueeze(-1)
        .expand(-1, object_model.batch_size_each)
    ).to(device=object_model.object_scale_tensor.device)
    assert (object_model.object_scale_tensor == correct_object_scales).all()

    # Reshape hand poses and energy terms to be (num_objects, num_grasps_per_object, ...)
    # an aside: it's absolutely ridiculous that we have to do this 🙃
    assert hand_model.hand_pose is not None
    hand_pose = (
        hand_model.hand_pose.detach()
        .cpu()
        .reshape(num_objects, num_grasps_per_object, -1)
    )
    hand_pose_start = (
        hand_pose_start.detach().cpu().reshape(num_objects, num_grasps_per_object, -1)
    )

    unweighted_energy_matrix = unweighted_energy_matrix.reshape(
        num_objects, num_grasps_per_object, -1
    )
    energy = energy.reshape(num_objects, num_grasps_per_object)

    for ii, object_code in enumerate(object_codes):
        trans, rot, joint_angles = pose_to_hand_config(hand_pose=hand_pose[ii])

        trans_start, rot_start, joint_angles_start = pose_to_hand_config(
            hand_pose=hand_pose_start[ii]
        )

        energy_dict = {}
        for k, energy_name in enumerate(ENERGY_NAMES):
            energy_dict[energy_name] = (
                unweighted_energy_matrix[ii, :, k].detach().cpu().numpy()
            )
        energy_dict["Total Energy"] = energy[ii].detach().cpu().numpy()

        object_code_and_scale_str = object_code_and_scale_to_str(
            object_code, object_scales[ii]
        )

        hand_config_dict = {
            "trans": trans,
            "rot": rot,
            "joint_angles": joint_angles,
            "trans_start": trans_start,
            "rot_start": rot_start,
            "joint_angles_start": joint_angles_start,
            **energy_dict,
        }

        np.save(
            file=output_folder_path / f"{object_code_and_scale_str}.npy",
            arr=hand_config_dict,
            allow_pickle=True,
        )


def generate(
    args_tuple: Tuple[
        GenerateHandConfigDictsArgs,
        List[str],
        int,
        List[str],
    ],
) -> None:
    args, object_code_and_scale_strs, id, gpu_list = args_tuple

    if args.optimization_seed is not None:
        set_seed(args.optimization_seed)
    else:
        set_seed(datetime.now().microsecond)

    # Parse object codes and scales
    object_codes, object_scales = [], []
    for object_code_and_scale_str in object_code_and_scale_strs:
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )
        object_codes.append(object_code)
        object_scales.append(object_scale)

    try:
        loop_timer = LoopTimer()

        # Log to wandb
        with loop_timer.add_section_timer("wandb and setup"):
            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            name = (
                f"{args.wandb_name}_{time_str}"
                if len(args.wandb_name) > 0
                else time_str
            )
            if args.use_wandb:
                wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    name=name,
                    config=args,
                )

            if args.use_multiprocess:
                worker = multiprocessing.current_process()._identity[0]
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[worker - 1]
            else:
                assert len(gpu_list) > 0, f"len(gpu_list) = {len(gpu_list)}"
                os.environ["CUDA_VISIBLE_DEVICES"] = (
                    gpu_list[0] if len(gpu_list) == 1 else ",".join(gpu_list)
                )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare models
        with loop_timer.add_section_timer("create hand model"):
            hand_model = HandModel(
                hand_model_type=args.hand_model_type,
                device=device,
                n_surface_points=1000,  # Need this for table penetration
            )

        with loop_timer.add_section_timer("create object model"):
            object_model = ObjectModel(
                meshdata_root_path=str(args.meshdata_root_path),
                batch_size_each=args.batch_size_each_object,
                num_samples=args.object_num_surface_samples,
                num_calc_samples=args.object_num_samples_calc_penetration_energy,
                device=device,
            )
            object_model.initialize(object_codes, object_scales)

        with loop_timer.add_section_timer("init convex hull"):
            initialize_convex_hull(
                hand_model=hand_model,
                object_model=object_model,
                distance_lower=args.distance_lower,
                distance_upper=args.distance_upper,
                theta_lower=args.theta_lower,
                theta_upper=args.theta_upper,
                jitter_strength=args.jitter_strength,
                n_contacts_per_finger=args.n_contacts_per_finger,
            )
            assert hand_model.hand_pose is not None
            hand_pose_start = hand_model.hand_pose.detach()

        with loop_timer.add_section_timer("create optimizer"):
            optim_config = {
                "switch_possibility": args.switch_possibility,
                "starting_temperature": args.starting_temperature,
                "temperature_decay": args.temperature_decay,
                "annealing_period": args.annealing_period,
                "step_size": args.step_size,
                "stepsize_period": args.stepsize_period,
                "n_contacts_per_finger": args.n_contacts_per_finger,
                "mu": args.mu,
                "device": device,
            }
            optimizer = Annealing(hand_model, **optim_config)

        # optimize
        with loop_timer.add_section_timer("energy forward"):
            energy_name_to_weight_dict = {
                "Force Closure": args.w_fc,
                "Hand Contact Point to Object Distance": args.w_dis,
                "Hand Object Penetration": args.w_pen,
                "Hand Self Penetration": args.w_spen,
                "Joint Limits Violation": args.w_joints,
                "Finger Finger Distance": args.w_ff,
                "Finger Palm Distance": args.w_fp,
                "Hand Table Penetration": args.w_tpen,
            }
            energy, unweighted_energy_matrix, weighted_energy_matrix = cal_energy(
                hand_model,
                object_model,
                energy_name_to_weight_dict=energy_name_to_weight_dict,
                thres_dis=args.thres_dis,
                thres_pen=args.thres_pen,
            )

        with loop_timer.add_section_timer("energy backward"):
            energy.sum().backward(retain_graph=True)

        idx_to_visualize = 0
        pbar = tqdm(range(args.n_iter), desc="optimizing", dynamic_ncols=True)
        for step in pbar:
            with loop_timer.add_section_timer("wandb and setup"):
                wandb_log_dict = {}
                wandb_log_dict["optimization_step"] = step

            with loop_timer.add_section_timer("optimizer try step zero grad"):
                _ = optimizer.try_step()
                optimizer.zero_grad()

            with loop_timer.add_section_timer("energy forward"):
                (
                    new_energy,
                    new_unweighted_energy_matrix,
                    new_weighted_energy_matrix,
                ) = cal_energy(
                    hand_model,
                    object_model,
                    energy_name_to_weight_dict=energy_name_to_weight_dict,
                    thres_dis=args.thres_dis,
                    thres_pen=args.thres_pen,
                )
            with loop_timer.add_section_timer("energy backward"):
                new_energy.sum().backward(retain_graph=True)

            with loop_timer.add_section_timer("update energy"):
                with torch.no_grad():
                    accept, temperature = optimizer.accept_step(energy, new_energy)

                    energy[accept] = new_energy[accept]
                    unweighted_energy_matrix[accept] = new_unweighted_energy_matrix[
                        accept
                    ]
                    weighted_energy_matrix[accept] = new_weighted_energy_matrix[accept]

            # Store grasps mid optimization
            with loop_timer.add_section_timer("save mid optimization grasps"):
                if (
                    args.store_grasps_mid_optimization_freq is not None
                    and step % args.store_grasps_mid_optimization_freq == 0
                ) or (
                    args.store_grasps_mid_optimization_iters is not None
                    and step in args.store_grasps_mid_optimization_iters
                ):
                    new_output_folder = (
                        pathlib.Path(f"{args.output_hand_config_dicts_path}")
                        / "mid_optimization"
                        / str(step)
                    )
                    new_output_folder.mkdir(parents=True, exist_ok=True)
                    save_hand_config_dicts(
                        hand_model=hand_model,
                        object_model=object_model,
                        object_codes=object_codes,
                        object_scales=object_scales,
                        hand_pose_start=hand_pose_start,
                        energy=energy,
                        unweighted_energy_matrix=unweighted_energy_matrix,
                        output_folder_path=new_output_folder,
                    )

            # Log
            with loop_timer.add_section_timer("wandb and setup"):
                wandb_log_dict.update(
                    {
                        "accept": accept.sum().item(),
                        "temperature": temperature.item(),
                        "energy": energy.mean().item(),
                        f"accept_{idx_to_visualize}": accept[idx_to_visualize].item(),
                        f"energy_{idx_to_visualize}": energy[idx_to_visualize].item(),
                    }
                )
                wandb_log_dict.update(
                    get_energy_term_log_dict(
                        unweighted_energy_matrix=unweighted_energy_matrix,
                        weighted_energy_matrix=weighted_energy_matrix,
                        idx_to_visualize=idx_to_visualize,
                    )
                )

                # Visualize
                if (
                    args.wandb_visualization_freq is not None
                    and step % args.wandb_visualization_freq == 0
                ):
                    fig, fig_title = create_visualization_figure(
                        hand_model=hand_model,
                        object_model=object_model,
                        idx_to_visualize=idx_to_visualize,
                    )
                    wandb_log_dict[fig_title] = fig

                if args.use_wandb:
                    wandb.log(wandb_log_dict)

                pbar.set_description(
                    f"optimizing, mean energy: {energy.mean().item():.4f}"
                )

            PRINT_LOOP_TIMER_EVERY_LOOP = False
            if PRINT_LOOP_TIMER_EVERY_LOOP:
                loop_timer.pretty_print_section_times()

        with loop_timer.add_section_timer("save final grasps"):
            save_hand_config_dicts(
                hand_model=hand_model,
                object_model=object_model,
                object_codes=object_codes,
                object_scales=object_scales,
                hand_pose_start=hand_pose_start,
                energy=energy,
                unweighted_energy_matrix=unweighted_energy_matrix,
                output_folder_path=args.output_hand_config_dicts_path,
            )
        loop_timer.pretty_print_section_times()

    except Exception as e:
        print(f"Exception: {e}")
        print(f"Skipping {object_codes} and continuing")


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[GenerateHandConfigDictsArgs])

    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f"gpu_list: {gpu_list}")

    # check whether arguments are valid and process arguments
    args.output_hand_config_dicts_path.mkdir(parents=True, exist_ok=True)
    if not args.meshdata_root_path.exists():
        raise ValueError(f"meshdata_root_path {args.meshdata_root_path} doesn't exist")

    # Read in object codes and scales
    if not args.input_object_code_and_scales_txt_path.exists():
        raise ValueError(
            f"input_object_code_and_scales_txt_path {args.input_object_code_and_scales_txt_path} doesn't exist"
        )
    with open(args.input_object_code_and_scales_txt_path, "r") as f:
        input_object_code_and_scale_strs_from_file = f.read().splitlines()
    input_object_code_and_scale_strs = get_object_codes_and_scales_to_process(
        input_object_code_and_scale_strs=input_object_code_and_scale_strs_from_file,
        meshdata_root_path=args.meshdata_root_path,
        output_folder_path=args.output_hand_config_dicts_path,
        continue_ok=args.continue_ok,
    )

    if args.randomize_order_seed is not None:
        random.Random(args.randomize_order_seed).shuffle(
            input_object_code_and_scale_strs
        )

    object_code_and_scale_str_groups = [
        input_object_code_and_scale_strs[i : i + args.n_objects_per_batch]
        for i in range(
            0, len(input_object_code_and_scale_strs), args.n_objects_per_batch
        )
    ]

    process_args = []
    for id, object_code_and_scale_str_group in enumerate(
        object_code_and_scale_str_groups
    ):
        process_args.append((args, object_code_and_scale_str_group, id + 1, gpu_list))

    if args.use_multiprocess:
        with multiprocessing.Pool(len(gpu_list)) as p:
            it = tqdm(
                p.imap(generate, process_args),
                total=len(process_args),
                desc="generating",
                maxinterval=1000,
            )
            list(it)
    else:
        for process_arg in tqdm(process_args, desc="generating", maxinterval=1000):
            generate(process_arg)


if __name__ == "__main__":
    main()
