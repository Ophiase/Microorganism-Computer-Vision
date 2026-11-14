import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from motility.diffusion import Verdict
from motility.diffusion.evidence import Evidence
from motility.pipeline import Scale
from motility.tracking import TrackTable
from motility.tracking.table import COLUMNS


def write_tracks(path: Path, table: TrackTable) -> Path:
    """One row per observation: track, frame, x, y, width, height."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        writer.writerows(zip(*table.columns().values(), strict=True))
    return path


def read_tracks(path: Path) -> TrackTable:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return TrackTable.from_columns({
        name: np.array([float(row[name]) for row in rows])
        for name in COLUMNS
        if not rows or name in rows[0]
    })


VERDICT_COLUMNS: Final = (
    "track",
    "length",
    "motion",
    "confidence",
    "alpha",
    "alpha_low",
    "alpha_high",
    "speed",
    "diffusion",
)


def write_verdicts(path: Path, verdicts: Sequence[Verdict], scale: Scale) -> Path:
    """One row per track: its motion and the numbers behind it, in report units."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow((*VERDICT_COLUMNS, "speed_unit", "diffusion_unit"))
        writer.writerows(
            (*_verdict_row(verdict, scale), scale.speed_unit, scale.diffusion_unit)
            for verdict in verdicts
        )
    return path


def _verdict_row(verdict: Verdict, scale: Scale) -> tuple[object, ...]:
    exponent = verdict.exponent
    return (
        verdict.track,
        verdict.length,
        verdict.motion.value,
        round(verdict.confidence, 4),
        *(
            (round(exponent.value, 4), round(exponent.low, 4), round(exponent.high, 4))
            if exponent
            else ("", "", "")
        ),
        round(scale.speed(verdict.speed), 5),
        ""
        if verdict.diffusion is None
        else round(scale.diffusion(verdict.diffusion), 6),
    )


def write_evidence(path: Path, evidence: Sequence[Evidence]) -> Path:
    """One row per test on one track: its statistic, p, q and effect."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow((
            "hypothesis",
            "track",
            "statistic",
            "p",
            "q",
            "effect",
            "samples",
        ))
        writer.writerows(
            (
                item.hypothesis,
                item.subject,
                round(item.statistic, 5),
                "" if item.p_value is None else f"{item.p_value:.3g}",
                "" if item.q_value is None else f"{item.q_value:.3g}",
                round(item.effect, 5),
                item.samples,
            )
            for item in evidence
        )
    return path
