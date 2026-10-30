import numpy as np
from numba import njit

from logic.kernel import average_neighbors

LOWER_CLIP = 0.2
UPPER_CLIP = 0.49

CHANNELS_PER_INPUT = 4
THREE_INPUTS = 3


@njit
def tensor_gradient(tensor: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(tensor)
    dy = np.zeros_like(tensor)
    dx[:, 1:] = tensor[:, 1:] - tensor[:, :-1]
    dy[1:, :] = tensor[1:, :] - tensor[:-1]
    return np.stack((dx, dy), axis=-1)


@njit
def pre_transform(tensor: np.ndarray) -> np.ndarray:
    result = np.clip(tensor, LOWER_CLIP, UPPER_CLIP)
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    result = 1.0 - result
    return np.clip(result * 4, 0.0, 1.0)


@njit
def _transform_jit(tensor: np.ndarray, trend: np.ndarray) -> np.ndarray:
    normalized = tensor - trend
    gradient = tensor_gradient(tensor)
    return np.stack((trend, normalized, gradient[..., 0], gradient[..., 1]), axis=-1)


def transform(tensor: np.ndarray) -> np.ndarray:
    """Stack the trend, the detail and the two gradients of a tensor."""
    return _transform_jit(tensor, average_neighbors(tensor))


def transform_frame(frame: np.ndarray, optical_flow: np.ndarray) -> np.ndarray:
    """Three transforms of one frame, stacked as the input channels of a pixel."""
    return np.stack(
        (
            transform(frame),
            transform(optical_flow[:, :, 0]),
            transform(optical_flow[:, :, 1]),
        ),
        axis=-2,
    )


def transform_video(video: np.ndarray, optical_flow_video: np.ndarray) -> np.ndarray:
    return np.stack([
        transform_frame(frame, optical_flow)
        for frame, optical_flow in zip(video, optical_flow_video, strict=True)
    ])
