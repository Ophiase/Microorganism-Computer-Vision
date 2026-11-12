import tomllib

from microorganism_computer_vision.pipeline.extract import DATASET_SETTINGS, video_urls


def test_every_file_of_the_settings_gets_its_own_address() -> None:
    with DATASET_SETTINGS.open("rb") as file:
        files = tomllib.load(file)["files"]

    urls = video_urls()

    assert len(set(urls)) == len(files)
    assert all(url.endswith(file_id) for url, file_id in zip(urls, files, strict=True))
