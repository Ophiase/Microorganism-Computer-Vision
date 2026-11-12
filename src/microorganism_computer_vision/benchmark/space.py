from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from pydantic import ValidationError

from microorganism_computer_vision.configuration import Configuration

type Value = float | int | str

ATTEMPTS: Final = 100
STEP: Final = 0.3
KEEP_CHOICE: Final = 0.8


@dataclass(frozen=True, slots=True)
class Knob:
    """One searchable parameter: its dotted path and its range or its choices.

    A range is drawn on a log scale when `logarithmic`, so a factor counts the
    same at every size; an integer knob is rounded after the draw.
    """

    path: str
    low: float = 0.0
    high: float = 1.0
    choices: tuple[Value, ...] = ()
    logarithmic: bool = False
    integer: bool = False

    def draw(self, generator: np.random.Generator) -> Value:
        if self.choices:
            return self.choices[int(generator.integers(len(self.choices)))]
        if self.logarithmic:
            value = float(
                np.exp(generator.uniform(np.log(self.low), np.log(self.high)))
            )
        else:
            value = float(generator.uniform(self.low, self.high))
        return round(value) if self.integer else value

    def nudge(self, current: Value, generator: np.random.Generator) -> Value:
        """A step near `current`: a factor of about 30 % or, rarely, another choice."""
        if self.choices:
            return current if generator.random() < KEEP_CHOICE else self.draw(generator)
        value = float(current) * float(np.exp(generator.normal(0, STEP)))
        value = min(max(value, self.low), self.high)
        return round(value) if self.integer else value


KNOBS: Final = (
    Knob("preprocessing.background", choices=("none", "temporal-median")),
    Knob("preprocessing.polarity", choices=("dark", "bright")),
    Knob("preprocessing.window_low", 0.01, 0.6),
    Knob("preprocessing.window_high", 0.4, 1.0),
    Knob("preprocessing.gain", 0.5, 8.0, logarithmic=True),
    Knob("preprocessing.smoothing", choices=(1, 3, 5)),
    Knob("detection.threshold", 0.15, 0.85),
    Knob("detection.opening", choices=(0, 1)),
    Knob("detection.min_area", 2, 80, logarithmic=True, integer=True),
    Knob("detection.max_area", 40, 2000, logarithmic=True, integer=True),
    Knob("tracking.max_assignment_distance", 3.0, 40.0, logarithmic=True),
    Knob("tracking.max_missed_frames", 0, 15, integer=True),
)


class Pick(Protocol):
    """Chooses a knob's next value, given its current one."""

    def __call__(self, knob: Knob, current: Value, /) -> Value: ...


class SearchSpace:
    """Draws valid configurations, blindly or around a starting one."""

    def __init__(self, knobs: tuple[Knob, ...] = KNOBS) -> None:
        self._knobs = knobs

    def blind(
        self, base: Configuration, generator: np.random.Generator
    ) -> Configuration:
        return self._draw(base, lambda knob, _: knob.draw(generator))

    def around(
        self, base: Configuration, generator: np.random.Generator
    ) -> Configuration:
        return self._draw(base, lambda knob, current: knob.nudge(current, generator))

    def _draw(self, base: Configuration, pick: Pick) -> Configuration:
        """Redraw until the model accepts: some draws break a constraint, low < high."""
        for _ in range(ATTEMPTS):
            values = base.model_dump(mode="json")
            for knob in self._knobs:
                section, name = knob.path.split(".")
                values[section][name] = pick(knob, values[section][name])
            try:
                return Configuration.model_validate(values)
            except ValidationError:
                continue
        raise ValueError("no valid configuration drawn; the space contradicts itself")
