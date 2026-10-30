
import unittest
import numpy as np
from logic.filters import tensor_gradient, transform_video
from logic.kernel import KernelType, average_neighbors
from visualization.grayscale_visualization import show_grayscale


def test_full_on_noise(
    self,
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

def test_check_gradient(self) -> None:
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

def test_check_kernel(self) -> None:
    tensor = np.random.rand(40, 60)
    avg = tensor
    for _ in range(20):
        avg = average_neighbors(avg, KernelType.SOFT_MEDIAN)

    show_grayscale([
        tensor, avg
    ]).show()
