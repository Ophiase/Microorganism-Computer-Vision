from enum import Enum

import numpy as np


class KernelType(Enum):
    DEFAULT = "default"
    GAUSSIAN = "gaussian"
    SOFT_MEDIAN = "soft_median"


KERNELS = {
    KernelType.DEFAULT: np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
    KernelType.GAUSSIAN: np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    KernelType.SOFT_MEDIAN: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8,
}
