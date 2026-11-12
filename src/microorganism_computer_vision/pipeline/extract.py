import tomllib
from pathlib import Path

import requests

from microorganism_computer_vision.paths import DATA_FOLDER, SETTINGS_FOLDER

DOWNLOAD_TIMEOUT = 60
CHUNK = 8192
DATASET_SETTINGS = SETTINGS_FOLDER / "dataset.toml"


def video_urls(settings_path: Path = DATASET_SETTINGS) -> tuple[str, ...]:
    """The videos of the source dataset, one URL per file id of the settings."""
    with settings_path.open("rb") as file:
        dataset = tomllib.load(file)
    return tuple(f"{dataset['base-url']}{file_id}" for file_id in dataset["files"])


def download_file(url: str, output_folder: Path) -> Path:
    destination = output_folder / (Path(url).name + ".avi")
    if destination.exists():
        return destination

    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK):
                file.write(chunk)
    return destination


def download_videos(urls: tuple[str, ...], output_folder: Path) -> tuple[Path, ...]:
    return tuple(download_file(url, output_folder) for url in urls)


def process() -> tuple[Path, ...]:
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    downloaded = download_videos(video_urls(), DATA_FOLDER)
    print(f"Extracted {len(downloaded)} videos to {DATA_FOLDER}")
    return downloaded
