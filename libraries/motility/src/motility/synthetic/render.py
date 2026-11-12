from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from motility.synthetic.motions import Path
from motility.synthetic.scene import Appearance, SceneParameters
from motility.vision.arrays import Frame, Video
from motility.vision.preprocessing import Polarity

FWHM_TO_SIGMA = 1 / 2.355
TEXTURE_SCALE = 15.0
DUST_SIZE = 0.8


def render(
    scene: SceneParameters, paths: Sequence[Path], generator: np.random.Generator
) -> Video:
    """Draw every cell of every frame over a static background, then blur and noise."""
    appearance = scene.appearance
    static = _background(scene, generator)
    _draw_dust(static, scene, generator)
    contrasts = appearance.contrast * np.clip(
        1 + appearance.contrast_spread * generator.normal(size=len(paths)), 0.2, 2.0
    )
    sign = -1.0 if appearance.polarity is Polarity.DARK else 1.0
    size = (appearance.length, appearance.width)
    frames = []
    for frame in range(scene.frames):
        image = static.copy()
        for path, contrast in zip(paths, contrasts.tolist(), strict=True):
            x, y = path.positions[frame].tolist()
            _stamp(image, (x, y), float(path.headings[frame]), size, sign * contrast)
        frames.append(_optics(image, appearance, generator))
    return np.stack(frames)


def _background(scene: SceneParameters, generator: np.random.Generator) -> Frame:
    appearance = scene.appearance
    columns = np.linspace(-0.5, 0.5, scene.width, dtype=np.float32)
    image = np.tile(
        appearance.background - appearance.gradient * columns, (scene.height, 1)
    )
    noise = generator.normal(size=(scene.height, scene.width)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), TEXTURE_SCALE)
    deviation = max(float(texture.std()), 1e-9)
    return (image + appearance.texture * texture / deviation).astype(np.float32)


def _draw_dust(
    image: Frame, scene: SceneParameters, generator: np.random.Generator
) -> None:
    appearance = scene.appearance
    sign = -1.0 if appearance.polarity is Polarity.DARK else 1.0
    size = appearance.width * (1 + DUST_SIZE)
    for _ in range(appearance.dust):
        centre = (generator.uniform(0, scene.width), generator.uniform(0, scene.height))
        _stamp(image, centre, 0.0, (size, size), sign * appearance.dust_contrast)


def _stamp(
    image: Frame,
    centre: tuple[float, float],
    heading: float,
    size: tuple[float, float],
    amplitude: float,
) -> None:
    """Add an elongated Gaussian, its long axis along `heading`, sizes as FWHM."""
    length, width = size
    radius = int(np.ceil(length))
    x, y = centre
    left, top = int(x) - radius, int(y) - radius
    first_row, last_row = max(top, 0), min(top + 2 * radius + 1, image.shape[0])
    first_column = max(left, 0)
    last_column = min(left + 2 * radius + 1, image.shape[1])
    if first_row >= last_row or first_column >= last_column:
        return
    rows, columns = np.mgrid[first_row:last_row, first_column:last_column]
    along = (columns - x) * np.cos(heading) + (rows - y) * np.sin(heading)
    across = -(columns - x) * np.sin(heading) + (rows - y) * np.cos(heading)
    spread_along = length * FWHM_TO_SIGMA
    spread_across = width * FWHM_TO_SIGMA
    profile = np.exp(
        -(along**2) / (2 * spread_along**2) - across**2 / (2 * spread_across**2)
    )
    image[rows, columns] += amplitude * profile.astype(np.float32)


def _optics(
    image: Frame, appearance: Appearance, generator: np.random.Generator
) -> NDArray[np.float32]:
    blurred = (
        cv2.GaussianBlur(image, (0, 0), appearance.blur)
        if appearance.blur > 0
        else image
    )
    noisy = blurred + generator.normal(0, appearance.noise, image.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)
