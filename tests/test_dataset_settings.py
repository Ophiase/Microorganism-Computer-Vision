import tomllib

from microorganism_computer_vision.media.extract import video_urls
from microorganism_computer_vision.media.folders import DATASET


def test_every_file_of_the_settings_gets_its_own_address() -> None:
    with DATASET.open("rb") as file:
        files = tomllib.load(file)["files"]

    urls = video_urls()

    assert len(set(urls)) == len(files)
    assert all(url.endswith(file_id) for url, file_id in zip(urls, files, strict=True))
