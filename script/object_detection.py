import numpy as np
import os
from typing import List, Tuple
from logic.bounding_box import BoundingBox
from common import BOUNDING_BOX_FOLDER, PREPROCESSED_FOLDER, TRACKING_FOLDER
from logic.kalman_filter import BacterialTracker
from logic.shape_detection import detect_shapes
from visualization.bbox_visualization import plot_bboxes_video, plot_tracked_video

###################################################################################


DEFAULT_NPY_FILE = os.path.join(PREPROCESSED_FOLDER, "342843.avi.npy")

###################################################################################


def process(
    file_path: str = DEFAULT_NPY_FILE,
    output_folder: str = BOUNDING_BOX_FOLDER
) -> List[List[BoundingBox]]:
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
) -> List[List[BoundingBox]]:
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
        bboxes = detect_shapes(frame, 10, 800)
        tracked = tracker.update_tracks(bboxes, frame_idx)
        tracked_boxes.append(tracked)

    # Save results
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, f"{file_name}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, np.array(tracked_boxes, dtype=object))

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
    # test_detect_shape()
    # process_with_tracking()
    test_kalman_filter(interval=(0, 40))


if __name__ == "__main__":
    main()
