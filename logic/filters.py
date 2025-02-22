import os
from typing import Tuple
import numpy as np
from enum import Enum
from scipy.ndimage import convolve
from common import DATA_FOLDER, PREPROCESSED_FOLDER
from logic.kernel import KernelType, KERNELS, average_neighbors
from numba import jit

###################################################################################

HIGH = 0.49
LOW = 0.2

###################################################################################


@jit(nopython=True)
def tensor_gradient(tensor: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(tensor)
    dy = np.zeros_like(tensor)
    dx[:, 1:] = tensor[:, 1:] - tensor[:, :-1]
    dy[1:, :] = tensor[1:, :] - tensor[:-1]
    return np.stack((dx, dy), axis=-1)


@jit(nopython=True)
def pre_transform(tensor: np.ndarray) -> np.ndarray:
    result = np.clip(tensor, LOW, HIGH)
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    result = 1.0 - result
    result = np.clip(result * 4, 0.0, 1.0)
    return result


@jit(nopython=True)
def _transform_jit(tensor: np.ndarray, trend: np.ndarray) -> np.ndarray:
    normalized = tensor - trend
    gradient = tensor_gradient(tensor)
    return np.stack((
        trend,
        normalized,
        gradient[..., 0], gradient[..., 1]
    ), axis=-1)


def transform(tensor: np.ndarray) -> np.ndarray:
    return _transform_jit(tensor, average_neighbors(tensor))


def transform_frame(frame: np.ndarray, optical_flow: np.ndarray) -> np.ndarray:
    frame_transformed = transform(frame)

    optical_flow_x_transformed = transform(optical_flow[:, :, 0])
    optical_flow_y_transformed = transform(optical_flow[:, :, 1])

    return np.stack((
        frame_transformed,
        optical_flow_x_transformed,
        optical_flow_y_transformed
    ), axis=-2)


# @jit(nopython=True)
def transform_video(video: np.ndarray, optical_flow_video: np.ndarray) -> np.ndarray:
    return np.stack([
        transform_frame(frame, optical_flow)
        for frame, optical_flow in zip(video, optical_flow_video)
    ])
