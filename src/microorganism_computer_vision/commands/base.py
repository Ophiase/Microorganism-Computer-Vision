import argparse
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.video import Clip
from microorganism_computer_vision.reporting.looks import Zoom

type Output = BaseModel | str
"""A model is printed as JSON for the agent; a text is printed as it is."""


class Command(Protocol):
    """One subcommand of `mcv`: it declares its arguments, then answers."""

    name: str
    summary: str

    def configure(self, parser: argparse.ArgumentParser) -> None: ...

    def execute(self, arguments: argparse.Namespace, /) -> Output: ...


def add_configuration(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="a configuration TOML file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="change one parameter, e.g. detection.threshold=0.4; repeatable",
    )
    parser.add_argument("--reason", help="why this configuration, kept with the run")


def configuration_of(arguments: argparse.Namespace) -> Configuration:
    configuration = Configuration.load(arguments.config).with_overrides(arguments.set)
    if arguments.reason:
        return configuration.model_copy(update={"reason": arguments.reason})
    return configuration


def clip_of(text: str | None) -> Clip | None:
    """`START:STOP`, either side optional: `:80`, `100:`, `20:60`."""
    if text is None:
        return None
    start, _, stop = text.partition(":")
    return Clip(start=int(start or 0), stop=int(stop) if stop else None)


def zoom_of(text: str | None) -> Zoom | None:
    """`X,Y,SIZE` in pixels of the frame."""
    if text is None:
        return None
    x, y, size = (int(part) for part in text.split(","))
    return Zoom(x, y, size)


def frames_of(text: str | None) -> tuple[int, ...]:
    return () if text is None else tuple(int(part) for part in text.split(","))
