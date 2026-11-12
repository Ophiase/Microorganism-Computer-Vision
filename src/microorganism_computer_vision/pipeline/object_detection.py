from pathlib import Path

import numpy as np
from motility.tracking.bounding_box import BoundingBox
from motility.tracking.kalman_filter import BacterialTracker
from motility.vision.shape_detection import detect_shapes

from microorganism_computer_vision.paths import (
    DEFAULT_VIDEO,
    PREPROCESSED_FOLDER,
    TRACKING_FOLDER,
)
from microorganism_computer_vision.visualization.videos import plot_tracked_video

DEFAULT_NPY_FILE = PREPROCESSED_FOLDER / f"{DEFAULT_VIDEO}.npy"
MIN_AREA = 10
MAX_AREA = 800
TREND_ROW = 0
FLOW_X_ROW = 1
FLOW_Y_ROW = 2
FIRST_COLUMN = 0


def split_channels(
    full_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the channel tensor as frame, flow x and flow y."""
    frames = full_data[:, :, :, TREND_ROW, FIRST_COLUMN]
    flow_x = full_data[:, :, :, FLOW_X_ROW, FIRST_COLUMN]
    flow_y = full_data[:, :, :, FLOW_Y_ROW, FIRST_COLUMN]
    return frames, flow_x, flow_y


def process_with_tracking(
    file_path: Path = DEFAULT_NPY_FILE,
    output_folder: Path = TRACKING_FOLDER,
    interval: tuple[int, int] | None = None,
) -> list[list[BoundingBox]]:
    """Track the cells of a transformed video and save the boxes per frame."""
    full_data = np.load(file_path)
    if interval is not None:
        full_data = full_data[interval[0] : interval[1]]
    frames, _, _ = split_channels(full_data)

    tracker = BacterialTracker()
    tracked_frames = []
    for frame_index in range(len(frames)):
        detections = detect_shapes(frames[frame_index], MIN_AREA, MAX_AREA)
        tracked_frames.append(list(tracker.update_tracks(detections, frame_index)))

    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / file_path.name
    np.save(output_file, np.array(tracked_frames, dtype=object))
    print(f"Tracked boxes saved to {output_file}")
    return tracked_frames


def show_tracks(
    file_path: Path = DEFAULT_NPY_FILE,
    interval: tuple[int, int] | None = None,
) -> None:
    """Draw the tracks over the frames, with the flow of each frame."""
    tracked_frames = process_with_tracking(file_path, interval=interval)

    full_data = np.load(file_path)
    if interval is not None:
        full_data = full_data[interval[0] : interval[1]]
    frames, flow_x, flow_y = split_channels(full_data)

    plot_tracked_video(
        video=frames,
        tracked_boxes=tracked_frames,
        optical_flow_video=np.stack([flow_x, flow_y], axis=-1),
        show_flow=True,
    ).show()
