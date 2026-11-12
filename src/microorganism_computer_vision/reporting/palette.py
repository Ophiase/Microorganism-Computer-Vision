from motility.diffusion import Motion

Colour = tuple[int, int, int]
"""Blue, green, red bytes, the order OpenCV draws in."""


def _bgr(hexadecimal: str) -> Colour:
    red, green, blue = (int(hexadecimal[index : index + 2], 16) for index in (1, 3, 5))
    return (blue, green, red)


MOTION_HEX = {
    Motion.DIRECTED: "#2a78d6",
    Motion.DIFFUSIVE: "#eb6834",
    Motion.CONFINED: "#1baf7a",
    Motion.CIRCULAR: "#eda100",
    Motion.STATIONARY: "#e87ba4",
    Motion.UNDETERMINED: "#8c8b86",
}
MOTION_COLOURS = {motion: _bgr(value) for motion, value in MOTION_HEX.items()}
DETECTION = _bgr("#f0efec")
SERIES = "#2a78d6"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
SURFACE = "#fcfcfb"
