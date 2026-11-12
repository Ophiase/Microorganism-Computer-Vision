from pathlib import Path

import numpy as np
from motility.vision.filters import pre_transform, transform_video
from motility.vision.optical_flow import compute_optical_flow

from microorganism_computer_vision.paths import (
    DATA_FOLDER,
    DEFAULT_VIDEO,
    DEFAULT_VIDEO_INTERVAL,
    PREPROCESSED_FOLDER,
)
from microorganism_computer_vision.video import load_video
from microorganism_computer_vision.visualization.panels import show_grayscale

VIDEO_PATH = DATA_FOLDER / DEFAULT_VIDEO
CHANNEL_ROWS = 3
CHANNEL_COLUMNS = 4


def save_processed_video(
    processed_video: np.ndarray,
    output_folder: Path = PREPROCESSED_FOLDER,
    filename: str = "processed_video.npy",
) -> Path:
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / filename
    np.save(output_path, processed_video)
    print(f"Processed video saved to {output_path}")
    return output_path


def process(
    video_path: Path = VIDEO_PATH,
    output_folder: Path = PREPROCESSED_FOLDER,
    interval: tuple[int, int] = DEFAULT_VIDEO_INTERVAL,
    debug: bool = False,
) -> Path:
    """Turn a video into the channel tensor the tracker and the figures read."""
    video = pre_transform(load_video(video_path, interval=interval, verbose=True))

    print("\tCompute optical flow..")
    optical_flow = compute_optical_flow(video)
    print("\tTransform video...")
    processed_video = transform_video(video, optical_flow)

    if debug:
        print(f"Processed video shape: {processed_video.shape}")
        show_grayscale([
            processed_video[0, :, :, row, column]
            for row in range(CHANNEL_ROWS)
            for column in range(CHANNEL_COLUMNS)
        ]).show()

    print("\tSaving...")
    return save_processed_video(processed_video, output_folder, video_path.name)
