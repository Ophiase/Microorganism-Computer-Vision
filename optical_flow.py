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
    return np.array(optical_flow)


def preview_optical_flow(video: np.ndarray, optical_flow: np.ndarray, verbose: bool = False):
    for i in range(len(optical_flow)):
        frame = video[i]
        flow = optical_flow[i]
        hsv = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('Optical Flow', rgb_flow)
        if cv2.waitKey(30) & 0xFF == 27:
            break
        if verbose:
            print(
                f"Previewing optical flow for frame {i}, rgb_flow shape: {rgb_flow.shape}")
    cv2.destroyAllWindows()
