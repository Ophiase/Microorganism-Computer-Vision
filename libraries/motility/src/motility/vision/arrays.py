import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.float32]
"""One grayscale image, rows by columns, in [0, 1]."""

Video = NDArray[np.float32]
"""Grayscale frames stacked on the first axis, each in [0, 1]."""


def as_video(frames: NDArray[np.generic]) -> Video:
    """Any stack of grayscale frames, bytes or floats, as a float32 video in [0, 1]."""
    if np.issubdtype(frames.dtype, np.integer):
        maximum = float(np.iinfo(frames.dtype.str).max)
        return (frames.astype(np.float32) / maximum).astype(np.float32)
    return np.clip(frames, 0.0, 1.0).astype(np.float32)
