import os
import requests
from pathlib import Path

DATA_DIR = "data"

# source: https://plos.figshare.com/articles/dataset/Modification_of_Salmonella_Typhimurium_Motility_by_the_Probiotic_Yeast_Strain_Saccharomyces_boulardii_/127695
DATA_PATHS = [
    "https://plos.figshare.com/ndownloader/files/342430",
    "https://plos.figshare.com/ndownloader/files/342517",
    "https://plos.figshare.com/ndownloader/files/342580",
    "https://plos.figshare.com/ndownloader/files/342655",
    "https://plos.figshare.com/ndownloader/files/342737",
    "https://plos.figshare.com/ndownloader/files/342783",
    "https://plos.figshare.com/ndownloader/files/342843",
    "https://plos.figshare.com/ndownloader/files/342910",
    "https://plos.figshare.com/ndownloader/files/342998",
]

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

def main() -> None:
    create_data_dir(DATA_DIR)
    download_videos(DATA_PATHS, DATA_DIR)
    print(f"Extracted and moved to: {DATA_DIR}")

if __name__ == "__main__":
    main()
