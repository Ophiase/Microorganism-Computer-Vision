import argparse
from pathlib import Path

import yaml
from pydantic import BaseModel

from microorganism_computer_vision.media.fetch import (
    Planned,
    bring,
    load_catalogue,
    plan,
)
from microorganism_computer_vision.media.folders import CATALOGUE, DATA
from microorganism_computer_vision.media.video import UsageError


class Sources(BaseModel):
    count: int
    sources: tuple[dict[str, object], ...]


class SourcesCommand:
    name = "sources"
    summary = "the catalogue of motility videos, their truth, licence and trust"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--with-truth", action="store_true", help="only those with a truth"
        )

    def execute(self, arguments: argparse.Namespace) -> Sources:
        entries = yaml.safe_load(CATALOGUE.read_text())
        if arguments.with_truth:
            entries = [entry for entry in entries if _has_truth(entry)]
        return Sources(count=len(entries), sources=tuple(entries))


class Fetched(BaseModel):
    files: tuple[Path, ...]
    megabytes: float
    brought: bool
    next: str


class FetchCommand:
    name = "fetch"
    summary = "bring catalogued videos into data/: by name, tag, truth or disk budget"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("names", nargs="*", help="source ids, see mcv sources")
        parser.add_argument("--tag", help="every source with this tag, e.g. swimming")
        parser.add_argument(
            "--with-truth", action="store_true", help="only sources with a truth"
        )
        parser.add_argument(
            "--max-mb", type=float, help="the smallest files first, up to this"
        )
        parser.add_argument(
            "--yes", action="store_true", help="download; without it, only the plan"
        )

    def execute(self, arguments: argparse.Namespace) -> Fetched:
        if not (
            arguments.names or arguments.tag or arguments.with_truth or arguments.max_mb
        ):
            raise UsageError(
                "say what to bring: names, --tag, --with-truth or --max-mb"
            )
        planned = plan(
            load_catalogue(CATALOGUE),
            DATA,
            arguments.names,
            arguments.tag,
            arguments.with_truth,
            arguments.max_mb,
        )
        total = sum(item.download.megabytes for item in planned)
        files = (
            tuple(bring(item) for item in planned)
            if arguments.yes
            else tuple(item.destination for item in planned)
        )
        return Fetched(
            files=files,
            megabytes=round(total, 1),
            brought=arguments.yes,
            next="done"
            if arguments.yes
            else f"{total:.0f} MB to bring: add --yes to download",
        )


def _has_truth(entry: dict[str, object]) -> bool:
    return not str(entry.get("truth", "none")).startswith("none")


def _has_truth(entry: dict[str, object]) -> bool:
    return not str(entry.get("truth", "none")).startswith("none")


def _bring_all(planned: tuple[Planned, ...]) -> tuple[Path, ...]:
    """Every file that comes; a refused one is said, and the others still come."""
    brought, refused = [], []
    for item in planned:
        try:
            brought.append(bring(item))
        except (UsageError, OSError) as error:
            refused.append(f"{item.download.file}: {error}")
    if refused and not brought:
        raise UsageError("nothing came: " + "; ".join(refused))
    return tuple(brought)
