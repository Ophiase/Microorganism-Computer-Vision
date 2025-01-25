import os
from typing import Tuple
import numpy as np
from enum import Enum
from common import DATA_FOLDER, DEFAULT_VIDEO, PREPROCESSED_FOLDER
from logic.filters import pre_transform, transform_video
from logic.load import load_video
from logic.optical_flow import compute_optical_flow
from visualization.grayscale_visualization import show_grayscale
from logic.kernel import KernelType, KERNELS

###################################################################################

VIDEO_PATH = os.path.join(DATA_FOLDER, DEFAULT_VIDEO)
FOLDER = PREPROCESSED_FOLDER
INTERVAL = (0, 40)
DEBUG = True

###################################################################################


def save_processed_video(
        processed_video: np.ndarray,
        output_folder: str = FOLDER,
        filename: str = "processed_video.npy"
):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, filename)
    np.save(output_path, processed_video)
    print(f"Processed video saved to {output_path}")


def process(
        video_path: str = VIDEO_PATH, output_folder: str = FOLDER,
        interval: Tuple[int] = INTERVAL,
        debug: bool = DEBUG) -> None:
    video = load_video(video_path, interval=interval, verbose=True)
    video = pre_transform(video)

    video_optical_flow: np.ndarray = compute_optical_flow(video)

    if debug:
        print(video.shape)
        print(video_optical_flow.shape)

    processed_video: np.ndarray = transform_video(video, video_optical_flow)

    if debug:
        print(f"Processed video shape: {processed_video.shape}")

        FRAME_TO_SHOW = 0
        to_show = [
            processed_video[FRAME_TO_SHOW, :, :, i, j]
            for i in range(3)
            for j in range(4)]
        show_grayscale(to_show).show()

    save_processed_video(processed_video, output_folder,
                         os.path.basename(video_path))

###################################################################################


def main():
    process()

if __name__ == "__main__":
    main()
