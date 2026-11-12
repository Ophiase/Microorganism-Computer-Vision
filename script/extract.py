from pathlib import Path

import requests

from common import DATA_FOLDER

DOWNLOAD_TIMEOUT = 60
CHUNK = 8192
FILE_IDS = (
    "342430",
    "342517",
    "342580",
    "342655",
    "342737",
    "342783",
    "342843",
    "342910",
    "342998",
)


def video_urls() -> tuple[str, ...]:
    """The nine videos of the source dataset, one URL per file id."""
    # hardcoded-url: the dataset of the study
    base = "https://plos.figshare.com/ndownloader/files/"
    return tuple(f"{base}{file_id}" for file_id in FILE_IDS)


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
