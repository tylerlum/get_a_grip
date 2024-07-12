import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

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

    @property
    def download_url_no_trailing_slash(self) -> str:
        if self.download_url.endswith("/"):
            return self.download_url[:-1]
        return self.download_url


def download_and_extract_zip(url: str, extract_to: Path) -> None:
    assert url.endswith(".zip"), f"URL must end with .zip, got {url}"

    url_filename_without_ext = Path(urlparse(url).path).stem
    output_zip_path = extract_to / f"{url_filename_without_ext}.zip"
    output_folder = extract_to / url_filename_without_ext
    print("=" * 80)
    print(f"Downloading {url} => {output_zip_path}")
    print(f"Then extracting {output_zip_path} => {extract_to}")
    print(f"Then expecting to end with {output_folder}")
    print("=" * 80 + "\n")

    if output_folder.exists():
        print("!" * 80)
        print(f"Folder {output_folder} already exists, skipping download.")
        print("!" * 80 + "\n")
        return

    # Make the directory
    extract_to.mkdir(parents=True, exist_ok=True)

    # Stream the download and show progress
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with output_zip_path.open("wb") as file, tqdm(
        desc=f"Downloading {url}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    # Extract the zip file with progress bar
    with zipfile.ZipFile(output_zip_path, "r") as zip_ref:
        zip_info_list = zip_ref.infolist()
        total_files = len(zip_info_list)

        with tqdm(
            desc=f"Extracting {output_zip_path}",
            total=total_files,
            unit="file",
            dynamic_ncols=True,
        ) as bar:
            for zip_info in zip_info_list:
                zip_ref.extract(zip_info, extract_to)
                bar.update(1)

    assert output_folder.exists(), f"Expected {output_folder} to exist"

    # Clean up the downloaded zip file
    print(f"Removing {output_zip_path}")
    output_zip_path.unlink()

    print("DONE!")
    print("~" * 80 + "\n")


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[DownloadArgs])

    # Meshdata
    if args.include_meshdata:
        url = f"{args.download_url_no_trailing_slash}/meshdata.zip"
        extract_to = get_data_folder()
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Final evaled grasp config dicts
    if args.include_final_evaled_grasp_config_dicts:
        url = f"{args.download_url_no_trailing_slash}/dataset/{args.dataset_size}/final_evaled_grasp_config_dicts.zip"
        extract_to = get_data_folder() / "dataset" / args.dataset_size
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Nerfdata
    if args.include_nerfdata:
        url = f"{args.download_url_no_trailing_slash}/dataset/{args.dataset_size}/nerfdata.zip"
        extract_to = get_data_folder() / "dataset" / args.dataset_size
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Point clouds
    if args.include_point_clouds:
        url = f"{args.download_url_no_trailing_slash}/dataset/{args.dataset_size}/point_clouds.zip"
        extract_to = get_data_folder() / "dataset" / args.dataset_size
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Nerf checkpoints
    if args.include_nerfcheckpoints:
        url = f"{args.download_url_no_trailing_slash}/dataset/{args.dataset_size}/nerfcheckpoints.zip"
        extract_to = get_data_folder() / "dataset" / args.dataset_size
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Pretrained models
    if args.include_pretrained_models:
        url = f"{args.download_url_no_trailing_slash}/models/pretrained.zip"
        extract_to = get_data_folder() / "models"
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Fixed sampler grasp config dicts
    if args.include_fixed_sampler_grasp_config_dicts:
        url = f"{args.download_url_no_trailing_slash}/fixed_sampler_grasp_config_dicts.zip"
        extract_to = get_data_folder()
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Real world nerfdata
    if args.include_real_world_nerfdata:
        url = f"{args.download_url_no_trailing_slash}/real_world/nerfdata.zip"
        extract_to = get_data_folder() / "real_world"
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Real world nerf checkpoints
    if args.include_real_world_nerfcheckpoints:
        url = f"{args.download_url_no_trailing_slash}/real_world/nerfcheckpoints.zip"
        extract_to = get_data_folder() / "real_world"
        download_and_extract_zip(url=url, extract_to=extract_to)


if __name__ == "__main__":
    main()
