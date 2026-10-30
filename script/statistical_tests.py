import csv
from pathlib import Path

import numpy as np

from common import (
    ANALYSIS_GRAPHICS_PATH,
    DEFAULT_VIDEO,
    DEFAULT_VIDEO_RESOLUTION,
    TRACKING_FOLDER,
)
from logic.diffusion import (
    CircularMotionTest,
    DiffusionTest,
    DirectionalSwitchTest,
    GaussianRandomWalkTest,
    MSDLinearTest,
    PersistentMotionTest,
    SubDiffusionTest,
)
from logic.trajectories import Trajectory
from visualization.trajectories import (
    plot_angular_distribution,
    plot_speed_distribution,
    plot_speed_distribution_per_trajectory,
    plot_trajectories,
)

TRACKING_PATH = TRACKING_FOLDER / f"{DEFAULT_VIDEO}.npy"
HYPOTHESES: dict[str, type[DiffusionTest]] = {
    "Gaussian": GaussianRandomWalkTest,
    "Linear MSD": MSDLinearTest,
    "Sub Diffusion": SubDiffusionTest,
    "Persistent Motion": PersistentMotionTest,
    "Circular Motion": CircularMotionTest,
    "Directional Switch": DirectionalSwitchTest,
}
TOP_PERCENT = 0.05
MINIMAL_SPEED = 3.0


def load_trajectories(tracking_file: Path = TRACKING_PATH) -> tuple[Trajectory, ...]:
    tracked_frames = np.load(tracking_file, allow_pickle=True).tolist()
    return Trajectory.from_frames(tracked_frames)


def fastest(
    trajectories: tuple[Trajectory, ...], percent: float = TOP_PERCENT
) -> tuple[Trajectory, ...]:
    """Keep the fastest share of the tracks, the ones that actually moved."""
    ranked = sorted(
        trajectories, key=lambda trajectory: trajectory.average_speed, reverse=True
    )
    return tuple(ranked[: max(1, int(len(ranked) * percent))])


def moving(trajectories: tuple[Trajectory, ...]) -> tuple[Trajectory, ...]:
    return tuple(
        trajectory
        for trajectory in trajectories
        if trajectory.average_speed >= MINIMAL_SPEED
    )


def analyze(trajectories: tuple[Trajectory, ...]) -> dict[str, list[bool]]:
    """Run the six hypotheses and report how many tracks each one fits."""
    print("Hypothesis:")
    results = {}
    for name, hypothesis in HYPOTHESES.items():
        analyzer = hypothesis()
        verdicts = [analyzer.analyze(trajectory) for trajectory in trajectories]
        print(f"- {name} fit: {sum(verdicts)}/{len(verdicts)}")
        results[name] = verdicts
    return results


def write_csv(results_path: Path, results: dict[str, list[bool]]) -> Path:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open(mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Trajectory", *results.keys()])
        for index, row in enumerate(zip(*results.values(), strict=True)):
            writer.writerow([index, *row])
    return results_path


def write_figures(
    trajectories: tuple[Trajectory, ...],
    output_folder: Path,
    resolution: tuple[int, int] = DEFAULT_VIDEO_RESOLUTION,
) -> tuple[Path, ...]:
    figures = {
        "speed_distribution": plot_speed_distribution(trajectories),
        "speed_distribution_per_trajectory": plot_speed_distribution_per_trajectory(
            trajectories
        ),
        "angular_distribution": plot_angular_distribution(trajectories),
        "trajectories": plot_trajectories(trajectories, resolution=resolution),
    }

    output_folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, figure in figures.items():
        image_path = output_folder / f"{name}.png"
        figure.write_image(image_path)
        print(f"Saved {name} to {image_path}")
        written.append(image_path)
    return tuple(written)


def process(
    video: str = DEFAULT_VIDEO,
    tracking_file: Path = TRACKING_PATH,
    keep: str = "all",
    debug: bool = False,
) -> tuple[Path, ...]:
    """Test every trajectory against the hypotheses, then draw what was found.

    `keep` is `all`, `moving` or `fastest`: on a real video most tracks are
    dust that never moved, and they drown the distributions.
    """
    trajectories = load_trajectories(tracking_file)
    match keep:
        case "moving":
            trajectories = moving(trajectories)
        case "fastest":
            trajectories = fastest(trajectories)

    output_folder = ANALYSIS_GRAPHICS_PATH / Path(video).stem
    write_csv(output_folder / "diffusion_results.csv", analyze(trajectories))
    written = write_figures(trajectories, output_folder)
    if debug:
        print(f"Output folder: {output_folder}")
    return written
