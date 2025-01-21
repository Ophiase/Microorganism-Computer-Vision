import numpy as np
import os
from typing import List, Tuple
import cv2
import plotly.graph_objects as go

from common import OBJECTS_FOLDER, PREPROCESSED_FOLDER
from visualization import plot_bboxes, show_grayscale

###################################################################################


DEFAULT_NPY_FILE = os.path.join(PREPROCESSED_FOLDER, "342843.avi.npy")

###################################################################################


def extract_high_optical_flow_areas(frame: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    optical_flow_magnitude = np.sqrt(
        frame[..., 1, 0] ** 2 + frame[..., 1, 1] ** 2)
    high_flow_mask = optical_flow_magnitude > threshold
    return high_flow_mask


def detect_shapes(frame: np.ndarray,
                  min_size: int = 100,
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


def load_npy_file(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return np.load(file_path)


def process(file_path: str = DEFAULT_NPY_FILE, output_folder: str = OBJECTS_FOLDER):
    video_tensor = load_npy_file(file_path)
    bounding_boxes_per_frame = detect_shapes(video_tensor)

    output_file = os.path.join(output_folder, "bounding_boxes.npy")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    np.save(output_file, bounding_boxes_per_frame)
    print(f"Bounding boxes saved to {output_file}")


#########################################

def test(file_path: str = DEFAULT_NPY_FILE, output_folder: str = OBJECTS_FOLDER):
    video_tensor = load_npy_file(file_path)

    # show_grayscale(
    #     [video_tensor[i, :, :, 0, 0] for i in range(video_tensor.shape[0])]
    # ).show()

    frame = np.clip(video_tensor[0][:, :, 0, 0] * 4, 0, 1)
    print(frame.shape)

    bboxes = detect_shapes(frame, min_size=5, max_size=800)
    # TODO: multiple size detection to exclude intersecting samples
    plot_bboxes(frame, bboxes).show()


#########################################


def main():
    # process()
    test()


if __name__ == "__main__":
    main()
