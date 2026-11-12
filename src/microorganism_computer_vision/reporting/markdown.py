from collections.abc import Sequence
from pathlib import Path

from motility.diffusion import Motion, Verdict
from motility.pipeline import Scale

from microorganism_computer_vision.reporting.summary import RunSummary, Spread

SHOWN_TRACKS = 25


def report(
    title: str,
    reason: str,
    summary: RunSummary,
    verdicts: Sequence[Verdict],
    scale: Scale,
    pictures: Sequence[Path],
) -> str:
    """The run as a page a researcher reads and checks: numbers, pictures, tracks."""
    sections = (
        f"# {title}",
        f"**Why this configuration**: {reason or 'not given'}",
        _numbers(summary),
        "\n".join(f"- {hint}" for hint in summary.hints)
        or "No hint: the numbers look sane.",
        "\n\n".join(f"![{picture.stem}]({picture.name})" for picture in pictures),
        _tracks(verdicts, scale),
        "The configuration is `configuration.toml`; the tracks, `tracks.csv`;"
        " every verdict, `verdicts.csv`.",
    )
    return "\n\n".join(sections) + "\n"


def _numbers(summary: RunSummary) -> str:
    motions = ", ".join(
        f"{count} {motion.value}" for motion, count in summary.motions.items() if count
    )
    rows = (
        ("frames", str(summary.frames)),
        ("detections per frame", _spread(summary.detections_per_frame)),
        ("tracks", f"{summary.tracks}, {summary.classified} classified"),
        ("observations per track", _spread(summary.track_length)),
        ("motions", motions or "none"),
        (f"speed ({summary.speed_unit})", _spread(summary.speed)),
        ("anomalous exponent α", _spread(summary.alpha)),
        (f"diffusion ({summary.diffusion_unit})", _spread(summary.diffusion)),
    )
    table = "\n".join(f"| {name} | {value} |" for name, value in rows)
    return f"| Measure | Value |\n| --- | --- |\n{table}"


def _spread(spread: Spread) -> str:
    if spread.median is None:
        return "none"
    return f"{spread.median:.3g} (10 % {spread.low:.3g}, 90 % {spread.high:.3g})"


def _tracks(verdicts: Sequence[Verdict], scale: Scale) -> str:
    classified = sorted(
        (item for item in verdicts if item.motion is not Motion.UNDETERMINED),
        key=lambda item: -item.length,
    )[:SHOWN_TRACKS]
    if not classified:
        return "No track was long enough to be classified."
    header = (
        f"| Track | Length | Motion | Confidence | α | Speed ({scale.speed_unit}) |"
    )
    rows = [
        f"| {item.track} | {item.length} | {item.motion.value} | {item.confidence:.2f}"
        f" | {_exponent(item)} | {scale.speed(item.speed):.3g} |"
        for item in classified
    ]
    return "\n".join((
        f"The {len(classified)} longest classified tracks:",
        "",
        header,
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
    ))


def _exponent(verdict: Verdict) -> str:
    if verdict.exponent is None:
        return "none"
    exponent = verdict.exponent
    return f"{exponent.value:.2f} [{exponent.low:.2f}, {exponent.high:.2f}]"
