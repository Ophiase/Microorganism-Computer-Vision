import os
from typing import List
import numpy as np
from common import TRACKING_FOLDER
from logic.bounding_box import BoundingBox, restructure_data
from logic.trajectories import Trajectory
from logic.diffusion.gaussian_random_walk_analysis import GaussianRandomWalkTest
from logic.diffusion.msd_linear_analysis import MSDLinearTest
from visualization.trajectories_visualization import plot_trajectories, plot_speed_distribution

TRACKING_FILE = os.path.join(TRACKING_FOLDER, "342843.avi.npy")

def main() -> None:
    # Load and prepare data
    tracked_data: List[List[BoundingBox]] = np.load(TRACKING_FILE, allow_pickle=True).tolist()
    restructured = restructure_data(tracked_data)

    # Create trajectories
    trajectories = [Trajectory(bboxes) for bboxes in restructured]

    # Analyze diffusion
    gauss_analyzer = GaussianRandomWalkTest()
    msd_analyzer = MSDLinearTest()

    gauss_results = [gauss_analyzer.analyze(trajectory) for trajectory in trajectories]
    msd_results = [msd_analyzer.analyze(trajectory) for trajectory in trajectories]

    print(f"Gaussian fit: {sum(gauss_results)}/{len(gauss_results)}")
    print(f"Linear MSD fit: {sum(msd_results)}/{len(msd_results)}")

    # Generate plots
    fig1 = plot_trajectories(trajectories)
    fig2 = plot_speed_distribution(trajectories)
    fig1.show()
    fig2.show()

if __name__ == "__main__":
    main()
