import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
import transforms3d
import trimesh
from nerfdata_viewer import visualize_nerfdata
from plot_grid import make_fig_grid, plot_mesh, plot_point_cloud, plot_table
from tqdm import tqdm

import streamlit as st
from get_a_grip import get_data_folder
from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.model_training.scripts.create_bps_grasp_dataset import (
    crop_single_point_cloud,
    get_bps,
    get_fixed_basis_points,
    read_and_process_single_point_cloud,
    read_raw_single_point_cloud,
)
from get_a_grip.model_training.utils.plot_utils import (
    plot_grasp_and_mesh_and_more,
)
from get_a_grip.utils.download import DownloadArgs, run_download
from get_a_grip.utils.parse_object_code_and_scale import parse_object_code_and_scale

# Caching parameters: balance fast loading with memory usage
# See https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data
CACHE_TIME_TO_LIVE = datetime.timedelta(minutes=30)
CACHE_MAX_ENTRIES = 5

# Centered or Wide mode
st.set_page_config(layout="centered")
# st.set_page_config(layout="wide")

########## TITLE ##########
st.title(
    "Get a Grip: Multi-Finger Grasp Evaluation at Scale Enables Robust Sim-to-Real Transfer"
)

########## AUTHOR INFORMATION ##########
st.write("""
***Tyler Ga Wei Lum\*$${}^1$$, Albert H. Li\*$${}^2$$, Preston Culbertson$${}^2$$, Krishnan Srinivasan$${}^1$$, Aaron Ames$${}^2$$, Mac Schwager$${}^1$$, Jeannette Bohg$${}^1$$***

**Conference on Robot Learning (CoRL) 2024**

$${}^1$$ Stanford University, $${}^2$$ California Institute of Technology

(\*) equal contribution
""")

BUTTON_COLUMNS = st.columns([1.5, 3, 2, 4, 1.5])
with BUTTON_COLUMNS[1]:
    st.link_button(
        label="Project Website 🌐",
        url="https://sites.google.com/view/get-a-grip-dataset",
    )
with BUTTON_COLUMNS[2]:
    st.link_button(
        label="Paper 📄",
        url="https://openreview.net/forum?id=1jc2zA5Z6J",
    )
with BUTTON_COLUMNS[3]:
    st.link_button(
        label="Code & Dataset 🧑‍💻💽",
        url="https://github.com/tylerlum/get_a_grip",
    )


########## OVERVIEW ##########
st.header("Overview")
st.write("""
The purpose of this app is to visualize the **Get a Grip** dataset, which includes:

* Grasps ✊
* Meshes 🧱
* Point Clouds ☁️
* Posed Images 📷
* Other Perceptual Data 👁️
* More! 🎉

We emphasize that these are **interactive visualizations** that allow you to pan, zoom, and rotate. You can also press the Full Screen button in the top right corner of the visualization to see it better!

The key takeaway of our work: We show that learned grasp evaluators enable robust real-world dexterous grasping when trained at sufficient scale with perceptual data. **We release a dataset of 3.5M labeled grasps with perceptual data** and show evaluators trained on it achieve SOTA hardware performance. See our project website for more details!
""")

########## NOTE ON PERFORMANCE ##########
st.header("Note on Performance")
st.write("""
We are doing our best to balance fast loading with memory constraints. We are currently using Streamlit Community Cloud's free service, which is a bit resource-constrained (otherwise it is great). Under the hood, we are using Streamlit's caching feature to speed up the app where possible. This means that **the first time you load a visualization, it may take a bit longer, but subsequent loads will be faster**.

Some signs that things are loading (we apologize and ask for your patience 🙏):

* You "feel" the lag (harder to scroll or interact with the page)

* You see "Running" on the top right

* You see "grayed out" elements

* You see a spinner or progress bar or "Running `some_function(...)`"

""")

DOWNLOAD_ALL_DATASETS = False

if DOWNLOAD_ALL_DATASETS:
    DATASET_NAMES = [
        "tiny_random",
        "tiny_best",
        "small_random",
        "small_best",
    ]
    PRETTY_DATASET_NAMES = [
        "Tiny Random (25 random objects)",
        "Tiny Best (25 best-grasped objects)",
        "Small Random (100 random objects)",
        "Small Best (100 best-grasped objects)",
    ]
else:
    DATASET_NAMES = [
        "small_random",
    ]
    PRETTY_DATASET_NAMES = [
        "Small Random (100 random objects)",
    ]


@st.cache_data(ttl=CACHE_TIME_TO_LIVE, max_entries=CACHE_MAX_ENTRIES)
def download_dataset() -> Path:
    download_url = os.environ["DOWNLOAD_URL"]
    data_folder = get_data_folder()

    progress = st.progress(0)
    status_text = st.empty()
    status_text.write(
        "*Downloading dataset. If this has not been downloaded before, this will take a few minutes (but will be fast after the first time)...*"
    )

    n_tasks = len(DATASET_NAMES) + 1

    # Download datasets
    for i, dataset_name in tqdm(enumerate(DATASET_NAMES), total=len(DATASET_NAMES)):
        run_download(
            args=DownloadArgs(
                download_url=download_url,
                data_folder=data_folder,
                include_meshdata_small=True,
                dataset_name=dataset_name,
                include_final_evaled_grasp_config_dicts=True,
                include_nerfdata=True,
                include_point_clouds=True,
                include_real_world_nerfdata=True,
                include_real_world_point_clouds=True,
            )
        )
        progress.progress(int((i + 1) / n_tasks * 100))

    run_download(
        args=DownloadArgs(
            download_url=download_url,
            data_folder=data_folder,
            include_real_world_nerfdata=True,
            include_real_world_point_clouds=True,
        )
    )

    progress.progress(100)
    status_text.write("*Done downloading!*")

    return data_folder


DATA_FOLDER = download_dataset()

# Hardcode this one dataset for simplicity
SELECTED_DATASET_NAME = "small_random"
SELECTED_PRETTY_DATASET_NAME = PRETTY_DATASET_NAMES[
    DATASET_NAMES.index(SELECTED_DATASET_NAME)
]
st.write(f"""
The full dataset includes millions of grasps across thousands of objects. For this app, we are focusing on the **{SELECTED_PRETTY_DATASET_NAME}** dataset. This is a good starting point to explore the dataset!
""")


########## VISUALIZATION SELECTION ##########
st.header("Visualization Selection")

GRASP_VISUALIZATION_HEADER = "Grasp Visualization"
OBJECT_VISUALIZATION_HEADER = "Object Visualization"
NERF_DATA_VISUALIZATION_HEADER = "Nerf Data Visualization"
REAL_WORLD_DATA_VISUALIZATION_HEADER = "Real World Data Visualization"
VISUALIZATION_HEADERS = [
    GRASP_VISUALIZATION_HEADER,
    OBJECT_VISUALIZATION_HEADER,
    NERF_DATA_VISUALIZATION_HEADER,
    REAL_WORLD_DATA_VISUALIZATION_HEADER,
]

SELECTED_VISUALIZATION_HEADER = st.radio(
    label="Selected Visualization:", options=VISUALIZATION_HEADERS, index=0
)
assert (
    SELECTED_VISUALIZATION_HEADER in VISUALIZATION_HEADERS
), f"{SELECTED_VISUALIZATION_HEADER} not in {VISUALIZATION_HEADERS}!"


@st.cache_data(ttl=CACHE_TIME_TO_LIVE, max_entries=CACHE_MAX_ENTRIES)
def get_object_codes_and_scale_strs(
    data_folder: Path, dataset_name: str
) -> Tuple[List[str], List[str], List[float]]:
    object_code_and_scale_strs = sorted(
        [
            x.stem
            for x in (
                data_folder
                / "dataset"
                / dataset_name
                / "final_evaled_grasp_config_dicts"
            ).iterdir()
        ]
    )
    object_codes_and_scales = [
        parse_object_code_and_scale(x) for x in object_code_and_scale_strs
    ]
    object_codes = [object_code for object_code, _ in object_codes_and_scales]
    object_scales = [object_scale for _, object_scale in object_codes_and_scales]
    return (
        object_code_and_scale_strs,
        object_codes,
        object_scales,
    )


(
    OBJECT_CODE_AND_SCALE_STRS,
    OBJECT_CODES,
    OBJECT_SCALES,
) = get_object_codes_and_scale_strs(
    data_folder=DATA_FOLDER, dataset_name=SELECTED_DATASET_NAME
)
PRETTY_OBJECT_CODE_AND_SCALE_STRS = [
    f"{object_code} ({object_scale:.4f})"
    for object_code, object_scale in zip(OBJECT_CODES, OBJECT_SCALES)
]


@st.cache_data(ttl=CACHE_TIME_TO_LIVE, max_entries=CACHE_MAX_ENTRIES)
def load_data_per_object(
    data_folder: Path, dataset_name: str, object_code_and_scale_str: str
) -> Tuple[
    trimesh.Trimesh,
    Dict[str, np.ndarray],
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
    Optional[np.ndarray],
]:
    progress = st.progress(0)
    progress_value = 0
    status_text = st.empty()

    def update_progress(msg: str, value: int, enabled: bool = False) -> None:
        if not enabled:
            return

        nonlocal progress
        nonlocal progress_value
        nonlocal status_text
        progress_value = value
        progress_value = np.clip(progress_value, a_min=None, a_max=100)
        progress.progress(value)
        status_text.write(msg)

    update_progress("*Parsing object code and scale*", 10)
    object_code, object_scale = parse_object_code_and_scale(object_code_and_scale_str)

    update_progress("*Validating downloaded filepaths*", 20)
    assert (
        data_folder / "meshdata_small"
    ).exists(), f"{data_folder / 'meshdata_small'} does not exist!"
    assert (
        data_folder / "dataset" / dataset_name / "final_evaled_grasp_config_dicts"
    ).exists(), f"{data_folder / 'dataset' / dataset_name / 'final_evaled_grasp_config_dicts'} does not exist!"
    assert (
        data_folder / "dataset" / dataset_name / "nerfdata"
    ).exists(), f"{data_folder / 'dataset' / dataset_name / 'nerfdata'} does not exist!"
    assert (
        data_folder / "dataset" / dataset_name / "point_clouds"
    ).exists(), "Point clouds not downloaded!"

    update_progress("*Loading mesh*", 30)
    mesh: trimesh.Trimesh = trimesh.load(
        data_folder / "meshdata_small" / object_code / "coacd" / "decomposed.obj"
    )
    mesh.apply_scale(object_scale)

    update_progress("*Loading evaled grasp config dict*", 40)
    evaled_grasp_config_dict = np.load(
        data_folder
        / "dataset"
        / dataset_name
        / "final_evaled_grasp_config_dicts"
        / f"{object_code_and_scale_str}.npy",
        allow_pickle=True,
    ).item()

    update_progress("*Loading point clouds*", 50)
    raw_point_cloud = read_raw_single_point_cloud(
        data_folder
        / "dataset"
        / dataset_name
        / "point_clouds"
        / object_code_and_scale_str
        / "point_cloud.ply"
    )

    update_progress("*Processing point clouds*", 60)
    processed_point_cloud = crop_single_point_cloud(
        read_and_process_single_point_cloud(
            data_folder
            / "dataset"
            / dataset_name
            / "point_clouds"
            / object_code_and_scale_str
            / "point_cloud.ply"
        )
    )

    update_progress("*Loading basis points*", 70)
    basis_points = get_fixed_basis_points()

    update_progress("*Calculating BPS*", 80)
    bps = (
        get_bps(
            all_points=processed_point_cloud[None, ...],
            basis_points=basis_points,
        ).squeeze(axis=0)
        if processed_point_cloud is not None
        else None
    )

    update_progress("*Done loading object data!*", 100)

    return (
        mesh,
        evaled_grasp_config_dict,
        raw_point_cloud,
        processed_point_cloud,
        basis_points,
        bps,
    )


def get_X_N_Oy(
    evaled_grasp_config_dict: Dict[str, Any], selected_grasp_index: int
) -> np.ndarray:
    object_state = evaled_grasp_config_dict["object_states_before_grasp"][
        selected_grasp_index, 0
    ]
    xyz, quat_xyzw = object_state[:3], object_state[3:7]
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    X_N_Oy = np.eye(4)
    X_N_Oy[:3, :3] = transforms3d.quaternions.quat2mat(quat_wxyz)
    X_N_Oy[:3, 3] = xyz
    return X_N_Oy


@st.cache_data(ttl=CACHE_TIME_TO_LIVE, max_entries=CACHE_MAX_ENTRIES)
def load_data_per_grasp(
    evaled_grasp_config_dict: Dict[str, Any], selected_grasp_index: int
) -> Tuple[float, float, float, torch.Tensor, np.ndarray]:
    progress = st.progress(0)
    progress_value = 0
    status_text = st.empty()

    def update_progress(msg: str, value: int, enabled: bool = False) -> None:
        if not enabled:
            return

        nonlocal progress
        nonlocal progress_value
        nonlocal status_text
        progress_value = value
        progress_value = np.clip(progress_value, a_min=None, a_max=100)
        progress.progress(value)
        status_text.write(msg)

    update_progress("*Loading y_* labels*", 25)
    y_pick = float(evaled_grasp_config_dict["y_pick"][selected_grasp_index])
    y_coll = float(evaled_grasp_config_dict["y_coll"][selected_grasp_index])
    y_PGS = float(evaled_grasp_config_dict["y_PGS"][selected_grasp_index])

    update_progress("*Loading grasp*", 50)
    grasp = AllegroGraspConfig.from_grasp_config_dict(
        grasp_config_dict=evaled_grasp_config_dict
    ).as_grasp()[selected_grasp_index]

    update_progress("*Loading X_N_Oy*", 75)
    X_N_Oy = get_X_N_Oy(
        evaled_grasp_config_dict=evaled_grasp_config_dict,
        selected_grasp_index=selected_grasp_index,
    )

    update_progress("*Done loading grasp data!*", 100)
    return (
        y_pick,
        y_coll,
        y_PGS,
        grasp,
        X_N_Oy,
    )


########## GRASP ##########
if SELECTED_VISUALIZATION_HEADER == GRASP_VISUALIZATION_HEADER:
    st.header(GRASP_VISUALIZATION_HEADER)

    st.write(f"""
    This dataset includes {len(OBJECT_CODE_AND_SCALE_STRS)} objects, and each object has many grasps.
    """)

    st.write("""
    In the visualization below, we show:

    * **The pre-grasp position**, where the fingers should not be in contact yet (:blue[blue])

    * **The post-grasp position**, where the fingers move towards to grasp the object (:green[green])

    * **The object mesh** used in simulation to evaluate the grasp quality and generate perceptual data

    * **The table** that the object is resting on, which the hand should avoid colliding with

    * **The grasp quality metrics** for the selected grasp, where each metric is a continuous value between 0 and 1, with 1 being higher quality:

      * We simulate each grasp multiple times with slight wrist pose perturbations and then average the results (see the paper for more details)

      * $$y_{pick}$$: The probability of successfully picking up the object

      * $$y_{coll}$$: The probability of avoiding undesired collisions with the object and table in the pre-grasp position

      * $$y_{PGS}$$: The probability of grasp success, which is the logical conjunction of $$y_{pick}$$ and $$y_{coll}$$

    We optionally show:

    * **The raw point cloud** of the object generated from a NeRF (:blue[blue])

    * **The processed point cloud** of the object to remove noise and distractors (black)

    * **The basis point set** used as input to the grasp evaluator, where each basis point is colored by its distance to the processed point cloud (:rainbow[rainbow])
    """)

    SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STR = st.selectbox(
        label=f"Selected Object Code and Scale (for {GRASP_VISUALIZATION_HEADER}) ({len(PRETTY_OBJECT_CODE_AND_SCALE_STRS)} options):",
        options=PRETTY_OBJECT_CODE_AND_SCALE_STRS,
        index=0,
    )
    assert (
        SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STR in PRETTY_OBJECT_CODE_AND_SCALE_STRS
    ), f"{SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STR} not in {PRETTY_OBJECT_CODE_AND_SCALE_STRS}!"

    SELECTED_OBJECT_CODE_AND_SCALE_STR = OBJECT_CODE_AND_SCALE_STRS[
        PRETTY_OBJECT_CODE_AND_SCALE_STRS.index(
            SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STR
        )
    ]

    (
        MESH,
        EVALED_GRASP_CONFIG_DICT,
        RAW_POINT_CLOUD,
        PROCESSED_POINT_CLOUD,
        BASIS_POINTS,
        BPS,
    ) = load_data_per_object(
        data_folder=DATA_FOLDER,
        dataset_name=SELECTED_DATASET_NAME,
        object_code_and_scale_str=SELECTED_OBJECT_CODE_AND_SCALE_STR,
    )

    # Extract data per grasp
    N_GRASPS = len(EVALED_GRASP_CONFIG_DICT["y_pick"])
    SELECTED_GRASP_INDEX = st.slider(
        label="Selected Grasp Index:", min_value=0, max_value=N_GRASPS - 1, value=0
    )

    INCLUDE_RAW_POINT_CLOUD = st.checkbox(
        label=f"Include Raw Point Cloud ({GRASP_VISUALIZATION_HEADER})", value=False
    )
    INCLUDE_PROCESSED_POINT_CLOUD = st.checkbox(
        label=f"Include Processed Point Cloud ({GRASP_VISUALIZATION_HEADER})",
        value=False,
    )
    INCLUDE_BASIS_POINTS = st.checkbox(
        label=f"Include Basis Points ({GRASP_VISUALIZATION_HEADER})", value=False
    )

    (Y_PICK, Y_COLL, Y_PGS, GRASP, GLOBAL_X_N_Oy) = load_data_per_grasp(
        evaled_grasp_config_dict=EVALED_GRASP_CONFIG_DICT,
        selected_grasp_index=SELECTED_GRASP_INDEX,
    )

    TRANSFORMED_MESH = MESH.copy()
    TRANSFORMED_MESH.apply_transform(
        GLOBAL_X_N_Oy
    )  # Set object pose when the grasp is applied

    @st.cache_data(ttl=CACHE_TIME_TO_LIVE, max_entries=CACHE_MAX_ENTRIES)
    def plot_grasp_visualization(
        data_folder: Path,
        dataset_name: str,
        object_code_and_scale_str: str,
        selected_grasp_index: int,
        include_raw_point_cloud: bool,
        include_processed_point_cloud: bool,
        include_basis_points: bool,
    ) -> go.Figure:
        # Display the grasp, mesh, and more
        fig_all = plot_grasp_and_mesh_and_more(
            grasp=GRASP,
            X_N_Oy=GLOBAL_X_N_Oy,
            visualize_target_hand=True,
            visualize_pre_hand=False,
            mesh=TRANSFORMED_MESH,
            basis_points=BASIS_POINTS if include_basis_points else None,
            bps=BPS if include_basis_points else None,
            raw_point_cloud_points=RAW_POINT_CLOUD if include_raw_point_cloud else None,
            processed_point_cloud_points=PROCESSED_POINT_CLOUD
            if include_processed_point_cloud
            else None,
            title=f"y_pick: {np.round(Y_PICK, 2)}, y_coll: {np.round(Y_COLL, 2)}, y_PGS: {np.round(Y_PGS, 2)}",
        )

        # Hide these traces by default to keep it cleaner
        # for trace in fig_all.data:
        #     if trace.name.lower() in [
        #         "raw point cloud",
        #         "processed point cloud",
        #         "basis points",
        #     ]:
        #         trace.visible = "legendonly"

        fig_all = plot_table(bounds=TRANSFORMED_MESH.bounds, fig=fig_all, is_z_up=False)
        return fig_all

    st.plotly_chart(
        plot_grasp_visualization(
            data_folder=DATA_FOLDER,
            dataset_name=SELECTED_DATASET_NAME,
            object_code_and_scale_str=SELECTED_OBJECT_CODE_AND_SCALE_STR,
            selected_grasp_index=SELECTED_GRASP_INDEX,
            include_raw_point_cloud=INCLUDE_RAW_POINT_CLOUD,
            include_processed_point_cloud=INCLUDE_PROCESSED_POINT_CLOUD,
            include_basis_points=INCLUDE_BASIS_POINTS,
        )
    )

########## OBJECT ##########
if SELECTED_VISUALIZATION_HEADER == OBJECT_VISUALIZATION_HEADER:
    st.header(OBJECT_VISUALIZATION_HEADER)

    st.write("""
    Next, we allow you to **visualize many object meshes simultaneously**, and **optionally include the processed point cloud**.

    We recommend pressing the Full Screen button in the top right corner of the visualization to see the objects better!
    """)

    MAX_NUM_OBJECTS_SELECTED = 10
    OBJECTS_SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STRS = st.multiselect(
        label=f"Selected Object Codes and Scales (for {OBJECT_VISUALIZATION_HEADER}) (up to {MAX_NUM_OBJECTS_SELECTED}) ({len(PRETTY_OBJECT_CODE_AND_SCALE_STRS)} options):",
        options=PRETTY_OBJECT_CODE_AND_SCALE_STRS,
        default=PRETTY_OBJECT_CODE_AND_SCALE_STRS[:2],
    )
    OBJECTS_SELECTED_OBJECT_CODE_AND_SCALE_STRS = [
        OBJECT_CODE_AND_SCALE_STRS[PRETTY_OBJECT_CODE_AND_SCALE_STRS.index(x)]
        for x in OBJECTS_SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STRS
    ]

    OBJECTS_INCLUDE_PROCESSED_POINT_CLOUD = st.checkbox(
        label=f"Include Processed Point Cloud ({OBJECT_VISUALIZATION_HEADER})",
        value=False,
    )
    if OBJECTS_INCLUDE_PROCESSED_POINT_CLOUD:
        st.write("""
        Feel free to change the size of each point in the point cloud below!
        """)
        OBJECTS_SELECTED_POINT_SIZE = st.slider(
            label="Selected Point Size:", min_value=1, max_value=10, value=2
        )
    else:
        OBJECTS_SELECTED_POINT_SIZE = None

    @st.cache_data(ttl=CACHE_TIME_TO_LIVE, max_entries=CACHE_MAX_ENTRIES)
    def plot_object_grid_cache(
        data_folder: Path,
        dataset_name: str,
        object_code_and_scale_strs: List[str],
        point_size: Optional[int],
    ) -> go.Figure:
        meshes, evaled_grasp_config_dicts, processed_point_clouds = [], [], []
        for object_code_and_scale_str in object_code_and_scale_strs:
            mesh, evaled_grasp_config_dict, _, processed_point_cloud, _, _ = (
                load_data_per_object(
                    data_folder=data_folder,
                    dataset_name=dataset_name,
                    object_code_and_scale_str=object_code_and_scale_str,
                )
            )
            meshes.append(mesh)
            evaled_grasp_config_dicts.append(evaled_grasp_config_dict)
            processed_point_clouds.append(processed_point_cloud)

        X_N_Oys = [
            get_X_N_Oy(
                evaled_grasp_config_dict, selected_grasp_index=0
            )  # First grasp, arbitrary
            for evaled_grasp_config_dict in evaled_grasp_config_dicts
        ]

        figs = []
        for mesh, X_N_Oy, processed_point_cloud in zip(
            meshes, X_N_Oys, processed_point_clouds
        ):
            mesh.apply_transform(X_N_Oy)  # Set object pose to match the point cloud

            fig = go.Figure()
            fig = plot_mesh(mesh=mesh, fig=fig, is_z_up=False)
            fig = plot_table(bounds=mesh.bounds, fig=fig, is_z_up=False)

            if processed_point_cloud is not None and point_size is not None:
                fig = plot_point_cloud(
                    points=processed_point_cloud,
                    size=point_size,
                    fig=fig,
                    is_z_up=False,
                )
            figs.append(fig)

        return make_fig_grid(
            figs=figs,
            fig_titles=object_code_and_scale_strs,
            title="Object Grid",
        )

    if (
        MAX_NUM_OBJECTS_SELECTED
        >= len(OBJECTS_SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STRS)
        > 0
    ):
        st.plotly_chart(
            plot_object_grid_cache(
                data_folder=DATA_FOLDER,
                dataset_name=SELECTED_DATASET_NAME,
                object_code_and_scale_strs=OBJECTS_SELECTED_OBJECT_CODE_AND_SCALE_STRS,
                point_size=OBJECTS_SELECTED_POINT_SIZE
                if OBJECTS_INCLUDE_PROCESSED_POINT_CLOUD
                else None,
            )
        )
    elif len(OBJECTS_SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STRS) == 0:
        st.write("*No objects selected!*")
    elif (
        len(OBJECTS_SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STRS)
        > MAX_NUM_OBJECTS_SELECTED
    ):
        st.write(
            f"*{len(OBJECTS_SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STRS)} objects selected, but only {MAX_NUM_OBJECTS_SELECTED} can be displayed at once!*"
        )
    else:
        raise ValueError("Should never get here")


########## NERF DATA ##########
if SELECTED_VISUALIZATION_HEADER == NERF_DATA_VISUALIZATION_HEADER:
    st.header(NERF_DATA_VISUALIZATION_HEADER)

    st.write("""
    Next, we allow you to **visualize the NeRF data for a single object**, which consists of **RGB images and camera poses**.

    We show the camera poses as cones that show its position and orientation with respect to the object, and we show the RGB image associated with each cone. We also show the ground-truth object mesh.
    """)

    @st.cache_data(ttl=CACHE_TIME_TO_LIVE, max_entries=CACHE_MAX_ENTRIES)
    def plot_nerfdata_cache(
        data_folder: Path,
        dataset_name: str,
        object_code_and_scale_str: str,
        max_num_cameras: int,
    ) -> go.Figure:
        nerdata_path = data_folder / "dataset" / dataset_name / "nerfdata"
        assert nerdata_path.exists(), f"{nerdata_path} does not exist!"

        object_nerfdata_path = nerdata_path / object_code_and_scale_str
        assert object_nerfdata_path.exists(), f"{object_nerfdata_path} does not exist!"

        (
            mesh,
            evaled_grasp_config_dict,
            _,
            _,
            _,
            _,
        ) = load_data_per_object(
            data_folder=data_folder,
            dataset_name=dataset_name,
            object_code_and_scale_str=object_code_and_scale_str,
        )

        (_, _, _, _, X_N_Oy) = load_data_per_grasp(
            evaled_grasp_config_dict=evaled_grasp_config_dict,
            selected_grasp_index=0,  # First grasp, arbitrary
        )

        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(
            X_N_Oy
        )  # Set object pose when the grasp is applied

        fig = visualize_nerfdata(
            nerfdata_path=object_nerfdata_path,
            mesh=transformed_mesh,
            max_num_cameras=max_num_cameras,
            show_background=True,
            show_grid=True,
            show_ticklabels=True,
            show_legend=True,
            is_z_up=False,
        )

        fig = plot_table(bounds=transformed_mesh.bounds, fig=fig, is_z_up=False)
        return fig

    NERFDATA_SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STR = st.selectbox(
        label=f"Selected Object Code and Scale (for {NERF_DATA_VISUALIZATION_HEADER}) ({len(PRETTY_OBJECT_CODE_AND_SCALE_STRS)} options):",
        options=PRETTY_OBJECT_CODE_AND_SCALE_STRS,
        index=0,
    )
    if NERFDATA_SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STR is not None:
        assert (
            NERFDATA_SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STR
            in PRETTY_OBJECT_CODE_AND_SCALE_STRS
        ), f"{NERFDATA_SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STR} not in {PRETTY_OBJECT_CODE_AND_SCALE_STRS}!"

        NERFDATA_SELECTED_OBJECT_CODE_AND_SCALE_STR = OBJECT_CODE_AND_SCALE_STRS[
            PRETTY_OBJECT_CODE_AND_SCALE_STRS.index(
                NERFDATA_SELECTED_PRETTY_OBJECT_CODE_AND_SCALE_STR
            )
        ]

        st.write("""
        We have 100 images per object, but visualizing all of them at once would be too much!
        """)
        MAX_NUM_CAMERAS = st.slider(
            label="Selected Max Num Cameras:",
            min_value=1,
            max_value=20,
            value=5,
        )
        st.plotly_chart(
            plot_nerfdata_cache(
                data_folder=DATA_FOLDER,
                dataset_name=SELECTED_DATASET_NAME,
                object_code_and_scale_str=NERFDATA_SELECTED_OBJECT_CODE_AND_SCALE_STR,
                max_num_cameras=MAX_NUM_CAMERAS,
            )
        )
    else:
        st.write("*No object selected!*")

if SELECTED_VISUALIZATION_HEADER == REAL_WORLD_DATA_VISUALIZATION_HEADER:
    ########## REAL WORLD DATA ##########
    st.header(REAL_WORLD_DATA_VISUALIZATION_HEADER)

    st.write("""
    In this section, we allow you to **visualize a small amount of real-world perception data**, which consists of **RGB images, camera poses, and point clouds**.

    We show the camera poses as cones that show its position and orientation with respect to the object, and we show the RGB image associated with each cone. We also show the object's associated point cloud (no ground-truth mesh for real world objects).
    """)

    @st.cache_data(ttl=CACHE_TIME_TO_LIVE, max_entries=CACHE_MAX_ENTRIES)
    def plot_real_world_nerfdata_cache(
        data_folder: Path,
        object_code_and_scale_str: str,
        max_num_cameras: int,
        show_raw_point_cloud: bool = True,
        show_processed_point_cloud: bool = False,
    ) -> go.Figure:
        nerdata_path = data_folder / "real_world" / "nerfdata"
        assert nerdata_path.exists(), f"{nerdata_path} does not exist!"

        object_nerfdata_path = nerdata_path / object_code_and_scale_str
        assert object_nerfdata_path.exists(), f"{object_nerfdata_path} does not exist!"

        point_cloud_path = data_folder / "real_world" / "point_clouds"
        assert point_cloud_path.exists(), f"{point_cloud_path} does not exist!"

        object_point_cloud_path = (
            point_cloud_path / object_code_and_scale_str / "point_cloud.ply"
        )
        assert (
            object_point_cloud_path.exists()
        ), f"{object_point_cloud_path} does not exist!"

        fig = visualize_nerfdata(
            nerfdata_path=object_nerfdata_path,
            mesh=None,
            max_num_cameras=max_num_cameras,
            randomize_camera_order_seed=42,  # Real world camera images are time-correlated, so randomize for more variety
            show_background=True,
            show_grid=True,
            show_ticklabels=True,
            show_legend=False,  # legend overlaps with the point cloud colorbar
            is_z_up=True,
        )

        raw_point_cloud = read_raw_single_point_cloud(object_point_cloud_path)
        processed_point_cloud = crop_single_point_cloud(
            read_and_process_single_point_cloud(object_point_cloud_path)
        )

        N_pts = raw_point_cloud.shape[0]
        assert raw_point_cloud.shape == (N_pts, 3), raw_point_cloud.shape
        bounds = np.zeros((2, 3))
        for i in range(3):
            bounds[0, i] = np.min(raw_point_cloud[:, i])
            bounds[1, i] = np.max(raw_point_cloud[:, i])
        fig = plot_table(bounds=bounds, fig=fig, is_z_up=True)

        if show_raw_point_cloud:
            fig = plot_point_cloud(
                points=raw_point_cloud, size=2, fig=fig, is_z_up=True
            )

        if show_processed_point_cloud and processed_point_cloud is not None:
            fig = plot_point_cloud(
                points=processed_point_cloud, size=2, fig=fig, is_z_up=True
            )
        return fig

    @st.cache_data(ttl=CACHE_TIME_TO_LIVE, max_entries=CACHE_MAX_ENTRIES)
    def get_real_world_object_codes_and_scale_strs(
        data_folder: Path,
    ) -> Tuple[List[str], List[str], List[float]]:
        assert (
            data_folder / "real_world" / "nerfdata"
        ).exists(), f"{data_folder / 'real_world' / 'nerfdata'} does not exist!"

        object_code_and_scale_strs = sorted(
            [x.stem for x in (data_folder / "real_world" / "nerfdata").iterdir()]
        )
        object_codes_and_scales = [
            parse_object_code_and_scale(x) for x in object_code_and_scale_strs
        ]
        object_codes = [object_code for object_code, _ in object_codes_and_scales]
        object_scales = [object_scale for _, object_scale in object_codes_and_scales]
        return (
            object_code_and_scale_strs,
            object_codes,
            object_scales,
        )

    (
        REAL_WORLD_OBJECT_CODE_AND_SCALE_STRS,
        REAL_WORLD_OBJECT_CODES,
        REAL_WORLD_OBJECT_SCALES,
    ) = get_real_world_object_codes_and_scale_strs(data_folder=DATA_FOLDER)

    SELECTED_REAL_WORLD_OBJECT_CODE = st.selectbox(
        label=f"Selected Real World Object Code (for {REAL_WORLD_DATA_VISUALIZATION_HEADER}) ({len(REAL_WORLD_OBJECT_CODES)} options):",
        options=REAL_WORLD_OBJECT_CODES,
        index=0,
    )

    if SELECTED_REAL_WORLD_OBJECT_CODE is not None:
        assert (
            SELECTED_REAL_WORLD_OBJECT_CODE in REAL_WORLD_OBJECT_CODES
        ), f"{SELECTED_REAL_WORLD_OBJECT_CODE} not in {REAL_WORLD_OBJECT_CODES}!"

        SELECTED_REAL_WORLD_OBJECT_CODE_AND_SCALE_STR = (
            REAL_WORLD_OBJECT_CODE_AND_SCALE_STRS[
                REAL_WORLD_OBJECT_CODES.index(SELECTED_REAL_WORLD_OBJECT_CODE)
            ]
        )

        st.write("""
        We have 100 images per object, but visualizing all of them at once would be too much!
        """)
        REAL_WORLD_MAX_NUM_CAMERAS = st.slider(
            label="Selected Real World Max Num Cameras:",
            min_value=1,
            max_value=20,
            value=5,
        )
        st.plotly_chart(
            plot_real_world_nerfdata_cache(
                data_folder=DATA_FOLDER,
                object_code_and_scale_str=SELECTED_REAL_WORLD_OBJECT_CODE_AND_SCALE_STR,
                max_num_cameras=REAL_WORLD_MAX_NUM_CAMERAS,
            )
        )
    else:
        st.write("*No object selected!*")
