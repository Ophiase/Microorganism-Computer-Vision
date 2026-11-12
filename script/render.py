from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import (
    DATA_FOLDER,
    DEFAULT_FONT,
    OUTPUT_FOLDER,
    PREPROCESSED_FOLDER,
    TRACKING_FOLDER,
)
from logic.structure.bounding_box import BoundingBox

FRAMES_PER_SECOND = 10
FONT_SIZE = 12
BYTE_MAX = 255
TREND_ROW = 0
FIRST_COLUMN = 0
BOX_COLOUR = "red"


def render_gifs(
    tracking_path: Path, output_folder: Path, frames_per_second: int = FRAMES_PER_SECOND
) -> tuple[Path, ...]:
    """Write the original and the transformed video as GIFs, boxes drawn."""
    tracked_frames: list[list[BoundingBox]] = np.load(
        tracking_path, allow_pickle=True
    ).tolist()
    stem = tracking_path.stem.removesuffix(".avi")

    original = output_folder / f"{stem}_original.gif"
    transformed = output_folder / f"{stem}_transformed.gif"

    _render_video(
        DATA_FOLDER / f"{stem}.avi", tracked_frames, original, frames_per_second
    )
    _render_tensor(
        PREPROCESSED_FOLDER / tracking_path.name,
        tracked_frames,
        transformed,
        frames_per_second,
    )
    return (original, transformed)


def _render_video(
    video_path: Path,
    tracked_frames: list[list[BoundingBox]],
    output_path: Path,
    frames_per_second: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(len(tracked_frames)):
        read, frame = capture.read()
        if not read:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    _save_gif(frames, tracked_frames, output_path, frames_per_second)


def _render_tensor(
    tensor_path: Path,
    tracked_frames: list[list[BoundingBox]],
    output_path: Path,
    frames_per_second: int,
) -> None:
    tensor = np.load(tensor_path)
    frames = [
        np.stack([frame] * 3, axis=-1)
        for frame in (
            tensor[: len(tracked_frames), :, :, TREND_ROW, FIRST_COLUMN] * BYTE_MAX
        ).astype(np.uint8)
    ]
    _save_gif(frames, tracked_frames, output_path, frames_per_second)


def _save_gif(
    frames: list[np.ndarray],
    tracked_frames: list[list[BoundingBox]],
    output_path: Path,
    frames_per_second: int,
) -> None:
    if not frames:
        return

    try:
        font = ImageFont.truetype(DEFAULT_FONT, FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    images = []
    for frame_index, frame in enumerate(frames):
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        for box in tracked_frames[frame_index]:
            draw.rectangle(
                [box.x, box.y, box.x + box.width, box.y + box.height],
                outline=BOX_COLOUR,
                width=1,
            )
            draw.text(
                (box.x + box.width // 2, box.y + box.height // 2),
                str(box.index),
                fill=BOX_COLOUR,
                font=font,
            )
        images.append(image)

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
        optimize=True,
    )
    print(f"Saved GIF to {output_path}")


def process(output_folder: Path = OUTPUT_FOLDER) -> tuple[Path, ...]:
    output_folder.mkdir(parents=True, exist_ok=True)
    written = []
    for tracking_path in sorted(TRACKING_FOLDER.glob("*avi.npy")):
        written.extend(render_gifs(tracking_path, output_folder))
    return tuple(written)
