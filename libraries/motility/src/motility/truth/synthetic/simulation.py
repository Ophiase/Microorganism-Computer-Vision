from dataclasses import dataclass

import numpy as np

from motility.diffusion.models.motion import Motion
from motility.truth.synthetic.motions import walk
from motility.truth.synthetic.render import render
from motility.truth.synthetic.scene import SceneParameters
from motility.truth.synthetic.truth import GroundTruth
from motility.vision.arrays import Video


@dataclass(frozen=True, slots=True)
class Simulation:
    """A rendered synthetic video and the truth it was drawn from."""

    video: Video
    truth: GroundTruth


def simulate(scene: SceneParameters) -> Simulation:
    """Draw the population, walk every cell, render the frames; one seed for all."""
    generator = np.random.default_rng(scene.seed)
    paths = [
        walk(motion, scene.kinetics, scene.frames, _spawn(scene, generator), generator)
        for motion in _population(scene)
    ]
    video = render(scene, paths, generator)
    truth = GroundTruth.from_paths(paths, scene.width, scene.height)
    return Simulation(video, truth)


def _population(scene: SceneParameters) -> list[Motion]:
    counts = scene.population.model_dump()
    return [Motion(name) for name, count in counts.items() for _ in range(count)]


def _spawn(
    scene: SceneParameters, generator: np.random.Generator
) -> tuple[float, float]:
    return (generator.uniform(0, scene.width), generator.uniform(0, scene.height))
