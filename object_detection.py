import numpy as np
import os
from typing import List, Tuple

from common import OBJECTS_FOLDER, PREPROCESSED_FOLDER
from visualization import show_grayscale

###################################################################################


DEFAULT_NPY_FILE = os.path.join(PREPROCESSED_FOLDER, "342843.avi.npy")
YOLO_CONFIDENCE_THRESHOLD = 0.5

###################################################################################


def extract_high_optical_flow_areas(frame: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    optical_flow_magnitude = np.sqrt(
        frame[..., 1, 0] ** 2 + frame[..., 1, 1] ** 2)
    high_flow_mask = optical_flow_magnitude > threshold
    return high_flow_mask


def find_bounding_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    from scipy.ndimage import label, find_objects

    labeled_array, _ = label(mask)
    bounding_boxes = []
    for slice_x, slice_y in find_objects(labeled_array):
        x_start, x_end = slice_x.start, slice_x.stop
        y_start, y_end = slice_y.start, slice_y.stop
        bounding_boxes.append((x_start, y_start, x_end, y_end))
    return bounding_boxes


def apply_yolo_on_frame(frame: np.ndarray) -> List[np.ndarray]:
    layer = frame[:, :, 0, 0]
    # high_flow_mask: np.ndarray = extract_high_optical_flow_areas(frame)
    bounding_boxes = find_bounding_boxes(layer)
    return bounding_boxes


def apply_yolo(video_tensor: np.ndarray) -> List[List[Tuple[int, int, int, int]]]:
    return [apply_yolo_on_frame(frame) for frame in video_tensor]

###################################################################################


def load_npy_file(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return np.load(file_path)


def process(file_path: str = DEFAULT_NPY_FILE, output_folder: str = OBJECTS_FOLDER):
    video_tensor = load_npy_file(file_path)
    bounding_boxes_per_frame = apply_yolo(video_tensor)

    output_file = os.path.join(output_folder, "bounding_boxes.npy")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    np.save(output_file, bounding_boxes_per_frame)
    print(f"Bounding boxes saved to {output_file}")


#########################################

def test(file_path: str = DEFAULT_NPY_FILE, output_folder: str = OBJECTS_FOLDER):
    video_tensor = load_npy_file(file_path)
    show_grayscale(
        [video_tensor[i, :, :, 0, 0] for i in range(video_tensor.shape[0])]
    ).show()

    exit()

    bounding_boxes = apply_yolo_on_frame(video_tensor[0])
    print(bounding_boxes)


def main():
    # process()
    test()


if __name__ == "__main__":
    main()
