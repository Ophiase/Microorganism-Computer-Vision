import os
from typing import Tuple
import numpy as np
from enum import Enum
from scipy.ndimage import convolve

from common import DATA_FOLDER, PREPROCESSED_FOLDER
from load import load_video
from optical_flow import compute_optical_flow
from visualization import show_grayscale


###################################################################################

VIDEO = os.path.join(DATA_FOLDER, "342843.avi")
FOLDER = PREPROCESSED_FOLDER
INTERVAL = (0, 20)

HIGH = 0.49
LOW = 0.2

DEBUG = True

#########################################


class KernelType(Enum):
    DEFAULT = "default"
    GAUSSIAN = "gaussian"
    SOFT_MEDIAN = "soft_median"


KERNELS = {
    KernelType.DEFAULT: np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
    KernelType.GAUSSIAN: np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    KernelType.SOFT_MEDIAN: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8,
}

#########################################


def apply_kernel(tensor, kernel, i, j, width, height):
    local_sum, weight = 0, 0
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni, nj = i + di, j + dj
            if 0 <= ni < width and 0 <= nj < height:
                local_sum += kernel[di + 1, dj + 1] * tensor[ni, nj]
                weight += kernel[di + 1, dj + 1]
    return local_sum / weight if weight > 0 else 0


def average_neighbors(tensor: np.ndarray, kernel_type: KernelType = KernelType.DEFAULT, use_library: bool = False) -> np.ndarray:
    kernel = KERNELS[kernel_type]

    if use_library:
        return convolve(tensor, kernel, mode='constant', cval=0)

    result = np.zeros_like(tensor)
    width, height = tensor.shape

    for i in range(width):
        for j in range(height):
            result[i, j] = apply_kernel(tensor, kernel, i, j, width, height)
    return result


def tensor_gradient(tensor: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(tensor)
    dy = np.zeros_like(tensor)
    dx[:, 1:] = tensor[:, 1:] - tensor[:, :-1]
    dy[1:, :] = tensor[1:, :] - tensor[:-1]
    return np.stack((dx, dy), axis=-1)


def pre_transform(tensor: np.ndarray) -> np.ndarray:
    result = np.clip(tensor, LOW, HIGH)
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    result = 1.0 - result
    return result


def transform(tensor: np.ndarray) -> np.ndarray:
    trend = average_neighbors(tensor)
    normalized = tensor - trend
    gradient = tensor_gradient(tensor)
    return np.stack((
        trend,
        normalized,
        gradient[..., 0], gradient[..., 1]
    ), axis=-1)


def transform_frame(frame: np.ndarray, optical_flow: np.ndarray) -> np.ndarray:
    frame_transformed = transform(frame)

    optical_flow_x_transformed = transform(optical_flow[:, :, 0])
    optical_flow_y_transformed = transform(optical_flow[:, :, 1])

    return np.stack((
        frame_transformed,
        optical_flow_x_transformed,
        optical_flow_y_transformed
    ), axis=-2)


def transform_video(video: np.ndarray, optical_flow_video: np.ndarray) -> np.ndarray:
    return np.stack([
        transform_frame(frame, optical_flow)
        for frame, optical_flow in zip(video, optical_flow_video)
    ])


###################################################################################

def full_test(
        video: np.ndarray = None,
        optical_flow_video: np.ndarray = None,
        frame: int = 1
) -> None:
    np.random.seed(100)

    RES = 32
    if video is None:
        video = np.random.rand(2, RES, RES)
    if optical_flow_video is None:
        optical_flow_video = np.random.rand(2, RES, RES, 2)

    processed_video = transform_video(video, optical_flow_video)

    print("\nTransformed Video Shape:")
    print(processed_video.shape)

    to_show = [
        processed_video[frame, :, :, i, j]
        for i in range(3)
        for j in range(4)]

    show_grayscale(to_show).show()


def check_gradient() -> None:
    tensor = np.random.rand(40, 60)
    grad = tensor_gradient(tensor)

    avg = tensor
    for _ in range(1):
        avg = average_neighbors(avg, KernelType.SOFT_MEDIAN)
    grad_avg = tensor_gradient(avg)
    show_grayscale([
        tensor, grad[:, :, 0], grad[:, :, 0],
        avg, grad_avg[:, :, 0], grad_avg[:, :, 1]
    ]).show()


def check_kernel() -> None:
    tensor = np.random.rand(40, 60)
    avg = tensor
    for _ in range(20):
        avg = average_neighbors(avg, KernelType.SOFT_MEDIAN)

    show_grayscale([
        tensor, avg
    ]).show()

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
        video_path: str = VIDEO, output_folder: str = FOLDER,
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
        print("Processed video shape: " + processed_video)

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
    # process()
    full_test()


if __name__ == "__main__":
    main()
