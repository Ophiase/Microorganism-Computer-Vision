import argparse
import sys
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ValidationError

from microorganism_computer_vision.commands.base import Command
from microorganism_computer_vision.commands.inform import (
    FetchCommand,
    GuideCommand,
    HistoryCommand,
    SchemaCommand,
    SourcesCommand,
)
from microorganism_computer_vision.commands.probe import ProbeCommand
from microorganism_computer_vision.commands.review import (
    CorrectCommand,
    GalleryCommand,
    InspectCommand,
    MissesCommand,
    SuspectsCommand,
)
from microorganism_computer_vision.commands.run import RunCommand, SweepCommand
from microorganism_computer_vision.commands.truth import (
    BenchCommand,
    SearchCommand,
    SynthCommand,
)
from microorganism_computer_vision.media.video import UsageError

COMMANDS: Final[tuple[Command, ...]] = (
    GuideCommand(),
    SchemaCommand(),
    SourcesCommand(),
    ProbeCommand(),
    RunCommand(),
    SweepCommand(),
    HistoryCommand(),
    SuspectsCommand(),
    MissesCommand(),
    InspectCommand(),
    GalleryCommand(),
    CorrectCommand(),
    SynthCommand(),
    BenchCommand(),
    SearchCommand(),
    FetchCommand(),
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="mcv",
        description="Read bacterial motility from video; every answer is JSON.",
    )
    subcommands = root.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command.configure(subcommands.add_parser(command.name, help=command.summary))
    return root


def main(argv: Sequence[str] | None = None) -> int:
    """Answer on stdout; a refused input is one line on stderr and exit code 2."""
    arguments = parser().parse_args(argv)
    command = next(item for item in COMMANDS if item.name == arguments.command)
    try:
        output = command.execute(arguments)
    except (ValidationError, UsageError, FileNotFoundError) as error:
        print(f"mcv {command.name}: {error}", file=sys.stderr)
        return 2
    print(output.model_dump_json(indent=2) if isinstance(output, BaseModel) else output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
