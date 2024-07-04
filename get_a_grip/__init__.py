from pathlib import Path


def get_package_folder() -> Path:
    """Returns the path to the package folder."""
    # Get the current file path
    package_file = __file__
    # Convert to a pathlib.Path object
    package_path = Path(package_file).parent
    return package_path


def get_repo_folder() -> Path:
    """Returns the path to the root of the repository."""
    # Get the package folder path
    package_folder = get_package_folder()
    # Navigate to the root of the repository (one level up from the package folder)
    repo_root = package_folder.parent
    return repo_root


def get_assets_folder() -> Path:
    """Returns the path to the assets folder located in the repository root."""
    # Get the repository root
    repo_root = get_repo_folder()
    # Define the path to the assets folder
    assets_folder = repo_root / "assets"
    return assets_folder


def get_data_folder() -> Path:
    """Returns the path to the data folder located in the repository root."""
    # Get the repository root
    repo_root = get_repo_folder()
    # Define the path to the data folder
    data_folder = repo_root / "data"
    return data_folder
