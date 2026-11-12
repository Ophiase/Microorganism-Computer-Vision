import cv2
import numpy as np


def compute_optical_flow(video: np.ndarray) -> np.ndarray:
    """Farneback flow between consecutive frames, one flow per frame.

    The first flow is the one of the second frame: the pair (0, 1) has no
    predecessor to be compared with, and every later stage expects one flow
    per frame.
    """
    if len(video) < 2:
        raise ValueError("a video of at least two frames is needed to measure a flow")

    flows = []
    previous = video[0]
    for index in range(1, len(video)):
        current = video[index]
        flow = np.zeros((*previous.shape, 2), dtype=np.float32)
        cv2.calcOpticalFlowFarneback(previous, current, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        previous = current

    return np.array([flows[0], *flows])
