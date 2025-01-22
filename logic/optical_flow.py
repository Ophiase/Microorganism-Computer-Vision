import cv2
import numpy as np

VERBOSE = False


def compute_optical_flow(video: np.ndarray, verbose: bool = VERBOSE) -> np.ndarray:
    optical_flow = []
    prev_frame = video[0]
    for i in range(1, len(video)):
        next_frame = video[i]
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        optical_flow.append(flow)
        prev_frame = next_frame
        if verbose and (i % 30 == 0):
            print(
                f"Computed optical flow for frame {i}, flow shape: {flow.shape}")

    # Duplicate the first flow to match the video length
    optical_flow.insert(0, optical_flow[0])
    return np.array(optical_flow)
