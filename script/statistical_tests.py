import os
from typing import Dict, List
import numpy as np
from common import ANALYSIS_GRAPHICS_PATH, DEFAULT_VIDEO, TRACKING_FOLDER
from logic.diffusion.base import DiffusionTest
from logic.structure.bounding_box import BoundingBox
from logic.trajectories import Trajectory, restructure_data
from logic.diffusion.gaussian_random_walk_test import GaussianRandomWalkTest
from logic.diffusion.msd_linear_test import MSDLinearTest
from logic.diffusion.sub_diffusion_test import SubDiffusionTest
from visualization.trajectories_visualization import plot_trajectories, plot_speed_distribution, plot_angular_distribution, plot_speed_distribution_per_trajectory
import csv

###################################################################################

TRACKING_PATH = os.path.join(TRACKING_FOLDER, DEFAULT_VIDEO + ".npy")

ONLY_MOVING_BACTERIA = False
MOVING_BACTERIA_MINIMAL_SPEED = 3
ONLY_MOVING_BACTERIA_BY_PERCENT = False
MOVING_BACTERIA_PERCENT = 0.05
DEBUG = True

DIFFUSION_HYPOTHESIS: Dict[str, DiffusionTest] = {
    'Gaussian': GaussianRandomWalkTest,
    'Linear MSD': MSDLinearTest,
    'Sub Diffusion': SubDiffusionTest
}

###################################################################################


def load_trajectories(
        tracking_file: str = TRACKING_PATH
) -> List[Trajectory]:
    tracked_data: List[List[BoundingBox]] = np.load(
        tracking_file, allow_pickle=True
    ).tolist()
    restructured = restructure_data(tracked_data)
    return [Trajectory(bboxes) for bboxes in restructured]


def filter_trajectories(trajectories: List[Trajectory]) -> List[Trajectory]:
    if ONLY_MOVING_BACTERIA_BY_PERCENT:
        sorted_trajs = sorted(
            trajectories,
            key=lambda t: t.average_speed(),
            reverse=True
        )

        keep_count = max(1, int(len(sorted_trajs) * MOVING_BACTERIA_PERCENT))
        return sorted_trajs[:keep_count]
    elif ONLY_MOVING_BACTERIA:
        return [
            t for t in trajectories
            if t.average_speed() >= MOVING_BACTERIA_MINIMAL_SPEED
        ]
    return trajectories


def diffusion_analysis(
        trajectories: List[Trajectory],
        video: str = DEFAULT_VIDEO
):
    os.makedirs(os.path.join(ANALYSIS_GRAPHICS_PATH, video), exist_ok=True)
    results_path = os.path.join(
        ANALYSIS_GRAPHICS_PATH, video, "diffusion_results.csv"
    )

    results_per_hypothesis = analyze_trajectories(trajectories)

    write_csv(results_path, results_per_hypothesis)


def analyze_trajectories(trajectories: List[Trajectory]) -> Dict[str, List[bool]]:
    results = {}
    for name, analysis_class in DIFFUSION_HYPOTHESIS.items():
        analyzer = analysis_class()
        result = [analyzer.analyze(trajectory) for trajectory in trajectories]
        print(f"{name} fit: {sum(result)}/{len(result)}")
        results[name] = result
    return results


def write_csv(results_path: str, results: Dict[str, List[bool]]):
    with open(results_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Trajectory"] + list(results.keys())
        writer.writerow(header)

        rows = zip(*results.values())
        for i, row in enumerate(rows):
            writer.writerow([i] + list(row))

def statistics(
    trajectories: List[Trajectory],
    video: str = DEFAULT_VIDEO,
    debug: bool = DEBUG
) -> None:
    plots = {
        "speed_distribution": plot_speed_distribution(trajectories),
        "speed_distribution_per_trajectory": plot_speed_distribution_per_trajectory(trajectories),
        "angular_distribution": plot_angular_distribution(trajectories),
        "trajectories": plot_trajectories(trajectories)
    }

    print(f"Output folder: {ANALYSIS_GRAPHICS_PATH}")
    for name, fig in plots.items():
        img_path = os.path.join(ANALYSIS_GRAPHICS_PATH, video, f"{name}.png")
        fig.write_image(img_path, engine="kaleido")
        print(f"Saved {name}")

        if debug:
            fig.show()

###################################################################################


def process(
        video: str = DEFAULT_VIDEO,
        tracking_file: str = TRACKING_PATH,
        debug: bool = DEBUG
):
    trajectories = load_trajectories(tracking_file)
    trajectories = filter_trajectories(trajectories)
    diffusion_analysis(trajectories, video)
    statistics(trajectories, video, debug)


def main():
    process()


if __name__ == "__main__":
    main()
