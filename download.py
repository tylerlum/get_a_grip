import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin

import requests
import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder


@dataclass
class DownloadArgs:
    download_url: str

    # Meshdata
    include_meshdata: bool = False

    # Dataset
    dataset_size: Literal["small", "large"] = "small"
    include_final_evaled_grasp_config_dicts: bool = False
    include_nerfdata: bool = False
    include_point_clouds: bool = False
    include_nerfcheckpoints: bool = False

    # Models
    include_pretrained_models: bool = False

    # Fixed sampler grasp config dicts
    include_fixed_sampler_grasp_config_dicts: bool = False

    # Real world
    include_real_world_nerfdata: bool = False
    include_real_world_nerfcheckpoints: bool = False


def download_and_extract_zip(url: str, extract_to: Path) -> None:
    # Check if the path exists
    if extract_to.exists():
        print(f"Path '{extract_to}' already exists. Aborting to avoid overwrite.")
        return

    # Make the directory
    extract_to.mkdir(parents=True, exist_ok=True)

    # Stream the download and show progress
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    zip_path = extract_to / "downloaded_file.zip"

    with zip_path.open("wb") as file, tqdm(
        desc=f"Downloading from {url}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    # Extract the zip file with progress bar
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_info_list = zip_ref.infolist()
        total_files = len(zip_info_list)

        with tqdm(
            desc=f"Extracting files to {extract_to}", total=total_files, unit="file"
        ) as bar:
            for zip_info in zip_info_list:
                zip_ref.extract(zip_info, extract_to)
                bar.update(1)

    # Clean up the downloaded zip file
    zip_path.unlink()


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[DownloadArgs])

    # Meshdata
    if args.include_meshdata:
        url = urljoin(args.download_url, "meshdata.zip")
        print(f"url = {url}")
        extract_to = get_data_folder() / "meshdata"
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Final evaled grasp config dicts
    if args.include_final_evaled_grasp_config_dicts:
        url = urljoin(
            args.download_url,
            f"dataset/{args.dataset_size}/final_evaled_grasp_config_dicts.zip",
        )
        extract_to = (
            get_data_folder()
            / "dataset"
            / args.dataset_size
            / "final_evaled_grasp_config_dicts"
        )
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Nerfdata
    if args.include_nerfdata:
        url = urljoin(args.download_url, f"dataset/{args.dataset_size}/nerfdata.zip")
        extract_to = get_data_folder() / "dataset" / args.dataset_size / "nerfdata"
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Point clouds
    if args.include_point_clouds:
        url = urljoin(
            args.download_url, f"dataset/{args.dataset_size}/point_clouds.zip"
        )
        extract_to = get_data_folder() / "dataset" / args.dataset_size / "point_clouds"
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Nerf checkpoints
    if args.include_nerfcheckpoints:
        url = urljoin(
            args.download_url, f"dataset/{args.dataset_size}/nerfcheckpoints.zip"
        )
        extract_to = (
            get_data_folder() / "dataset" / args.dataset_size / "nerfcheckpoints"
        )
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Pretrained models
    if args.include_pretrained_models:
        url = urljoin(args.download_url, "models/pretrained.zip")
        extract_to = get_data_folder() / "models" / "pretrained"
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Fixed sampler grasp config dicts
    if args.include_fixed_sampler_grasp_config_dicts:
        url = urljoin(args.download_url, "fixed_sampler_grasp_config_dicts.zip")
        extract_to = get_data_folder() / "fixed_sampler_grasp_config_dicts" / "given"
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Real world nerfdata
    if args.include_real_world_nerfdata:
        url = urljoin(args.download_url, "real_world/nerfdata.zip")
        extract_to = get_data_folder() / "real_world" / "nerfdata"
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Real world nerf checkpoints
    if args.include_real_world_nerfcheckpoints:
        url = urljoin(args.download_url, "real_world/nerfcheckpoints.zip")
        extract_to = get_data_folder() / "real_world" / "nerfcheckpoints"
        download_and_extract_zip(url=url, extract_to=extract_to)


if __name__ == "__main__":
    main()
