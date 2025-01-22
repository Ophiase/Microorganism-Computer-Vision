import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio
import cv2

from common import DATA_FOLDER, PREPROCESSED_FOLDER, TRACKING_FOLDER, OUTPUT_FOLDER, DEFAULT_FONT

#########################################


def render_gifs(tracking_path: Path, output_dir: Path, fps: int = 10):
    """Render both original and transformed GIFs"""
    # Load tracking data
    tracked_data = np.load(tracking_path, allow_pickle=True)
    num_frames = len(tracked_data)

    # Base name handling
    video_stem = tracking_path.stem.replace(
        '.avi', '')  # "342843.avi.npy" -> "342843"
    orig_video_path = Path(DATA_FOLDER) / f"{video_stem}.avi"
    preprocessed_path = Path(PREPROCESSED_FOLDER) / tracking_path.name

    # Create output paths
    orig_gif = output_dir / f"{video_stem}_original.gif"
    transformed_gif = output_dir / f"{video_stem}_transformed.gif"

    # Render original GIF
    if orig_video_path.exists():
        _render_video_gif(orig_video_path, tracked_data, orig_gif, fps)

    # Render transformed GIF
    if preprocessed_path.exists():
        _render_transformed_gif(
            preprocessed_path, tracked_data, transformed_gif, fps)


def _render_video_gif(video_path: Path, tracked_data, output_path: Path, fps: int):
    """Render GIF from original video file"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_count = 0

    while frame_count < len(tracked_data):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1

    cap.release()
    _save_gif(frames, tracked_data, output_path, fps)


def _render_transformed_gif(npy_path: Path, tracked_data, output_path: Path, fps: int):
    """Render GIF from preprocessed numpy array"""
    preprocessed = np.load(npy_path)
    video_tensor = preprocessed[:len(
        tracked_data), :, :, 0, 0]  # Trend channel

    frames = []
    for frame in (video_tensor * 255).astype(np.uint8):
        frames.append(np.stack([frame]*3, axis=-1))  # Convert to RGB

    _save_gif(frames, tracked_data, output_path, fps)


def _save_gif(frames: list, tracked_data, output_path: Path, fps: int):
    """Common GIF saving logic"""
    if not frames:
        return

    # Prepare font
    try:
        font = ImageFont.truetype(DEFAULT_FONT, 12)
    except:
        font = ImageFont.load_default()

    pil_frames = []
    for frame_idx, frame in enumerate(frames):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        if frame_idx < len(tracked_data):
            for bbox in tracked_data[frame_idx]:
                bbox_id, x, y, w, h = bbox
                draw.rectangle([x, y, x+w, y+h], outline="red", width=1)
                draw.text((x + w//2, y + h//2), str(bbox_id),
                          fill="red", font=font)

        pil_frames.append(img)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000//fps,
        loop=0,
        optimize=True
    )
    print(f"Saved GIF to {output_path}")


###################################################################################


def main():
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(exist_ok=True)

    for tracking_path in Path(TRACKING_FOLDER).glob("*.npy"):
        try:
            print(f"Processing {tracking_path.name}...")
            render_gifs(tracking_path, output_dir)
        except Exception as e:
            print(f"Error processing {tracking_path}: {str(e)}")


if __name__ == "__main__":
    main()
