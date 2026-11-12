from pathlib import Path

import cv2
import numpy as np


def load_video(
    video_path: Path,
    interval: tuple[int, int] | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Read the frames of a video as floats in [0, 1], grayscale.

    `interval` keeps the half-open frame range `[start, end)`; frames outside it
    are decoded and dropped, the codec gives no cheap seek per frame.
    """
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    frame_count = 0
    while capture.isOpened():
        read, frame = capture.read()
        if not read:
            break
        if interval is None or interval[0] <= frame_count < interval[1]:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(grayscale.astype(np.float32) / 255.0)
        frame_count += 1
    capture.release()

    video = np.array(frames)
    if verbose:
        print(f"Loaded video {video_path} shape: {video.shape}")
    return video
