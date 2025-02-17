from enum import Enum
from scipy.ndimage import convolve
import numpy as np
from numba import jit


class KernelType(Enum):
    DEFAULT = "default"
    GAUSSIAN = "gaussian"
    SOFT_MEDIAN = "soft_median"


KERNELS = {
    KernelType.DEFAULT: np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
    KernelType.GAUSSIAN: np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    KernelType.SOFT_MEDIAN: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8,
}


@jit(nopython=True)
def apply_kernel(tensor: np.ndarray, kernel: np.ndarray, i: int, j: int, width: int, height: int):
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

    return _average_neighbors_numba(tensor, kernel)


@jit(nopython=True)
def _average_neighbors_numba(tensor: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    width, height = tensor.shape
    result = np.zeros_like(tensor)

    for i in range(width):
        for j in range(height):
            result[i, j] = apply_kernel(tensor, kernel, i, j, width, height)

    return result
