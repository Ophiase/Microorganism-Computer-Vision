import numpy as np
import os
from typing import List, Tuple
import cv2
import plotly.graph_objects as go

from common import BOUNDING_BOX_FOLDER, PREPROCESSED_FOLDER
from visualization import plot_bboxes, plot_bboxes_video, show_grayscale

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


def process(file_path: str = DEFAULT_NPY_FILE, output_folder: str = BOUNDING_BOX_FOLDER):
    video_tensor = load_npy_file(file_path)
    bounding_boxes_per_frame = [
        detect_shapes(frame) for frame in video_tensor
    ]

    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, file_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    np.save(output_file, bounding_boxes_per_frame)
    print(f"Bounding boxes saved to {output_file}")


#########################################

def test(file_path: str = DEFAULT_NPY_FILE, output_folder: str = BOUNDING_BOX_FOLDER):
    video_tensor = load_npy_file(file_path)[:,:,:,0,0]

    # show_grayscale(
    #     [video_tensor[i, :, :, 0, 0] for i in range(video_tensor.shape[0])]
    # ).show()

    bounding_boxes_per_frame = [
        detect_shapes(frame, min_size=10, max_size=800) 
        for frame in video_tensor
    ]

    # print(bounding_boxes_per_frame[0].shape)

    plot_bboxes_video(video_tensor, bounding_boxes_per_frame).show()


#########################################


# TODO: multiple size detection to exclude intersecting samples
def main():
    # process()
    test()


if __name__ == "__main__":
    main()
