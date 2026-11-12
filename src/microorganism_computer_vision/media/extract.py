import tomllib
from pathlib import Path

import requests

from microorganism_computer_vision.media.folders import DATA, DATASET
from microorganism_computer_vision.media.video import UsageError

DOWNLOAD_TIMEOUT = 60
CHUNK = 8192


def video_urls(settings_path: Path = DATASET) -> tuple[str, ...]:
    """The videos of the source dataset, one URL per file id of the settings."""
    with settings_path.open("rb") as file:
        dataset = tomllib.load(file)
    return tuple(f"{dataset['base-url']}{file_id}" for file_id in dataset["files"])


def download_file(url: str, output_folder: Path) -> Path:
    """Fetch one file, written whole or not at all: a partial or empty one is removed.

    A host that answers a script with a challenge page sends nothing and a 2xx;
    an empty file is that, never a video.
    """
    destination = output_folder / (Path(url).name + ".avi")
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    partial = destination.with_suffix(".part")
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with partial.open("wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK):
                file.write(chunk)
    if partial.stat().st_size == 0:
        partial.unlink()
        raise UsageError(f"{url} sent an empty file: the host refused the download")
    return partial.replace(destination)


def download_videos(urls: tuple[str, ...], output_folder: Path) -> tuple[Path, ...]:
    return tuple(download_file(url, output_folder) for url in urls)


def process() -> tuple[Path, ...]:
    DATA.mkdir(parents=True, exist_ok=True)
    downloaded = download_videos(video_urls(), DATA)
    return downloaded
