from dataclasses import dataclass
import numpy as np
import os
from typing import List, Tuple
import cv2
import plotly.graph_objects as go

from bounding_box import BoundingBox
from common import BOUNDING_BOX_FOLDER, PREPROCESSED_FOLDER, TRACKING_FOLDER
from kalman_filter import BacterialTracker
from visualization import plot_bboxes, plot_bboxes_video, plot_tracked_video

###################################################################################


DEFAULT_NPY_FILE = os.path.join(PREPROCESSED_FOLDER, "342843.avi.npy")

###################################################################################

def extract_high_optical_flow_areas(frame: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    optical_flow_magnitude = np.sqrt(
        frame[..., 1, 0] ** 2 + frame[..., 1, 1] ** 2)
    high_flow_mask = optical_flow_magnitude > threshold
    return high_flow_mask


def detect_shapes(frame: np.ndarray,
                  min_size: int = 10,
                  max_size: int = 1000) -> List[Tuple[int, int, int, int]]:
    """
    Detect bounding boxes of white regions in a grayscale image.

    Args:
        frame: 2D numpy array of shape (height, width) with values in [0, 1]
        min_size: Minimum area of bounding boxes to include
        max_size: Maximum area of bounding boxes to include

    Returns:
        List of bounding boxes as (x, y, width, height) tuples
    """
    # Convert to 8-bit and apply threshold
    frame_uint8 = (frame * 255).astype(np.uint8)
    _, thresh = cv2.threshold(frame_uint8, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract and filter bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if min_size <= area <= max_size:
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes

###################################################################################


def process(
    file_path: str = DEFAULT_NPY_FILE,
    output_folder: str = BOUNDING_BOX_FOLDER
) -> List[List[Tuple[int, int, int, int]]]:
    video_tensor = np.load(file_path)[:, :, :, 0, 0]
    bounding_boxes_per_frame = [
        detect_shapes(frame) for frame in video_tensor
    ]

    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, file_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    np.save(output_file, bounding_boxes_per_frame)
    print(f"Bounding boxes saved to {output_file}")
    return bounding_boxes_per_frame


def process_with_tracking(
    file_path: str = DEFAULT_NPY_FILE,
    output_folder: str = TRACKING_FOLDER,
    interval: Tuple[int, int] = None
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Process video with tracking and save results
    """
    
    # Load full data cube
    full_data = np.load(file_path)
    if interval is not None:
        full_data = full_data[interval[0]:interval[1]]

    video_tensor = full_data[:, :, :, 0, 0]  # Trend channel
    optical_flow = np.stack([
        full_data[:, :, :, 1, 0],  # Optical flow x (transformed)
        full_data[:, :, :, 2, 0]   # Optical flow y (transformed)
    ], axis=-1)

    # Initialize tracker
    tracker = BacterialTracker(optical_flow)
    tracked_boxes: List[List[BoundingBox]] = []

    # Process frames
    for frame_idx in range(len(video_tensor)):
        frame = video_tensor[frame_idx]
        bboxes = detect_shapes(frame, 15, 800)
        tracked = tracker.update_tracks(bboxes, frame_idx)
        tracked_boxes.append(tracked)

    # Save results
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, f"{file_name}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, tracked_boxes)

    print(f"Tracked boxes saved to {output_file}")
    return tracked_boxes

#########################################


def test_detect_shapes(
        file_path: str = DEFAULT_NPY_FILE,
        output_folder: str = BOUNDING_BOX_FOLDER
):
    video_tensor = np.load(file_path)[:, :, :, 0, 0]

    # show_grayscale(
    #     [video_tensor[i, :, :, 0, 0] for i in range(video_tensor.shape[0])]
    # ).show()

    bounding_boxes_per_frame = [
        detect_shapes(frame, min_size=10, max_size=800)
        for frame in video_tensor
    ]

    # print(bounding_boxes_per_frame[0].shape)

    plot_bboxes_video(video_tensor, bounding_boxes_per_frame).show()


def test_kalman_filter(
        file_path: str = DEFAULT_NPY_FILE,
        output_folder: str = TRACKING_FOLDER,
        interval: Tuple[int, int] = None,
):
    tracked_boxes = process_with_tracking(file_path, output_folder, interval)

    #########################################

    full_data = np.load(file_path)
    if interval is not None:
        full_data = full_data[interval[0]:interval[1]]

    video_tensor = full_data[:, :, :, 0, 0]
    optical_flow = np.stack(
        [full_data[:, :, :, 1, 0], full_data[:, :, :, 2, 0]], -1)

    fig = plot_tracked_video(
        video=video_tensor,
        tracked_boxes=tracked_boxes,
        optical_flow_video=optical_flow,
        show_flow=True
    )
    fig.show()

#########################################


def main():
    # process()
    # test()

    # process_with_tracking()
    test_kalman_filter(interval=(0, 20))


if __name__ == "__main__":
    main()
