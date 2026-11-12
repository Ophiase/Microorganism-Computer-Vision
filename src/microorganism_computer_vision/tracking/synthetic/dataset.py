from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from microorganism_computer_vision.pipeline.paths import TRACKING_FOLDER
from microorganism_computer_vision.tracking.bounding_box import BoundingBox
from microorganism_computer_vision.tracking.synthetic.generator import (
    SyntheticGenerator,
)

DEFAULT_CENTER = (512.0, 512.0)


@dataclass(frozen=True, slots=True)
class MotionSpec:
    """A named way of moving, built for a number of frames."""

    name: str
    build: Callable[[SyntheticGenerator, int], list[BoundingBox]]


MOTIONS = (
    MotionSpec(
        "brownian",
        lambda generator, frames: generator.brownian_motion(frames, speed=2.0),
    ),
    MotionSpec(
        "directed",
        lambda generator, frames: generator.directed_motion(
            frames, speed=1.5, angle=float(np.random.uniform(0, 360))
        ),
    ),
    MotionSpec(
        "confined",
        lambda generator, frames: generator.confined_motion(
            frames, center=DEFAULT_CENTER, radius=50.0
        ),
    ),
    MotionSpec(
        "ctrw", lambda generator, frames: generator.ctrw_motion(frames, base_speed=3.0)
    ),
    MotionSpec(
        "sinusoidal",
        lambda generator, frames: generator.sinusoidal_motion(
            frames, speed=2.2, amplitude=10.0, frequency=0.01
        ),
    ),
)


def generate_synthetic_data(
    output_folder: Path = TRACKING_FOLDER,
    trajectories_per_type: int = 25,
    num_frames: int = 200,
    verbose: bool = False,
) -> tuple[Path, ...]:
    """Write one tracking file per motion, each with its truth to compare against."""
    output_folder.mkdir(parents=True, exist_ok=True)
    generator = SyntheticGenerator()
    written = []

    for motion in MOTIONS:
        trajectories = [
            motion.build(generator, num_frames) for _ in range(trajectories_per_type)
        ]
        frames: list[list[BoundingBox]] = [[] for _ in range(num_frames)]
        for trajectory in trajectories:
            for frame_index, box in enumerate(trajectory):
                frames[frame_index].append(box)

        output_path = output_folder / f"synthetic_{motion.name}.npy"
        np.save(output_path, np.array(frames, dtype=object))
        written.append(output_path)
        if verbose:
            print(f"Saved {trajectories_per_type} {motion.name} to {output_path}")

    return tuple(written)
