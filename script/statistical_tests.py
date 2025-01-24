import os
from typing import List
import numpy as np
from common import ANALYSIS_GRAPHICS_PATH, TRACKING_FOLDER
from logic.structure.bounding_box import BoundingBox, restructure_data
from logic.trajectories import Trajectory
from logic.diffusion.gaussian_random_walk_analysis import GaussianRandomWalkTest
from logic.diffusion.msd_linear_analysis import MSDLinearTest
from visualization.trajectories_visualization import plot_trajectories, plot_speed_distribution, plot_angular_distribution, plot_speed_distribution_per_trajectory

TRACKING_FILE = os.path.join(TRACKING_FOLDER, "342843.avi.npy")

ONLY_MOVING_BACTERIA = False
MOVING_BACTERIA_MINIMAL_SPEED = 25
ONLY_MOVING_BACTERIA_BY_PERCENT = False
MOVING_BACTERIA_PERCENT = 0.05


def main() -> None:
    # Load and prepare data
    tracked_data: List[List[BoundingBox]] = np.load(
        TRACKING_FILE, allow_pickle=True).tolist()
    restructured = restructure_data(tracked_data)

    # Create trajectories
    trajectories = [Trajectory(bboxes) for bboxes in restructured]

    if ONLY_MOVING_BACTERIA_BY_PERCENT:
        sorted_trajs = sorted(
            trajectories,
            key=lambda t: t.average_speed(),
            reverse=True
        )

        keep_count = max(1, int(len(sorted_trajs) * MOVING_BACTERIA_PERCENT))
        trajectories = sorted_trajs[:keep_count]
    elif ONLY_MOVING_BACTERIA:
        trajectories = [
            t for t in trajectories 
            if t.average_speed() >= MOVING_BACTERIA_MINIMAL_SPEED
        ]

    # Analyze diffusion
    gauss_analyzer = GaussianRandomWalkTest()
    msd_analyzer = MSDLinearTest()

    gauss_results = [gauss_analyzer.analyze(
        trajectory) for trajectory in trajectories]
    msd_results = [msd_analyzer.analyze(trajectory)
                   for trajectory in trajectories]

    print(f"Gaussian fit: {sum(gauss_results)}/{len(gauss_results)}")
    print(f"Linear MSD fit: {sum(msd_results)}/{len(msd_results)}")
    
    os.makedirs(ANALYSIS_GRAPHICS_PATH, exist_ok=True)

    plots = {
        "speed_distribution": plot_speed_distribution(trajectories),
        "speed_distribution_per_trajectory": plot_speed_distribution_per_trajectory(trajectories),
        "angular_distribution": plot_angular_distribution(trajectories),
        "trajectories": plot_trajectories(trajectories)
    }

    for name, fig in plots.items():
        img_path = os.path.join(ANALYSIS_GRAPHICS_PATH, f"{name}.png")
        fig.write_image(img_path, engine="kaleido")
        print(f"Saved {name} to:\t {img_path}")
        fig.show()



if __name__ == "__main__":
    main()
