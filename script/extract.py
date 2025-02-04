import os
import requests
from pathlib import Path
from common import DATA_FOLDER

###################################################################################


# source:
# https://plos.figshare.com/articles/dataset/Modification_of_Salmonella_Typhimurium_Motility_by_the_Probiotic_Yeast_Strain_Saccharomyces_boulardii_/127695
DATA_PATHS = [
    f"https://plos.figshare.com/ndownloader/files/{x}" for x in [
        "342430", "342517", "342580",
        "342655", "342737", "342783",
        "342843", "342910", "342998"
    ]
]

###################################################################################


def create_data_dir(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)


def download_file(url: str, output_dir: str) -> None:
    filename = os.path.join(output_dir, Path(url).name + ".avi")
    if not os.path.exists(filename):
        response = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


def download_videos(data_paths: list[str], output_dir: str) -> None:
    for url in data_paths:
        download_file(url, output_dir)

###################################################################################


def process():
    create_data_dir(DATA_FOLDER)
    download_videos(DATA_PATHS, DATA_FOLDER)
    print(f"Extracted and moved to: {DATA_FOLDER}")


def main() -> None:
    process()


if __name__ == "__main__":
    main()
