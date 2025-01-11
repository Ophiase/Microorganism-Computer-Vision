import cv2
import numpy as np

VERBOSE = False


def load_video(video_path: str, color_mode: str = 'grayscale', interval: tuple = None, verbose: bool = VERBOSE) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if interval is None or (frame_count >= interval[0] and frame_count < interval[1]):
            if color_mode == 'grayscale':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        frame_count += 1
    cap.release()
    video = np.array(frames)
    if verbose:
        print(f"Loaded video {video_path} shape: {video.shape}")
    return video
