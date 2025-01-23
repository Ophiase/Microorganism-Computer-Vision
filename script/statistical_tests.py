import os
from typing import List
import numpy as np
from common import ANALYSIS_GRAPHICS_PATH, TRACKING_FOLDER
from logic.bounding_box import BoundingBox, restructure_data
from logic.trajectories import Trajectory
from logic.diffusion.gaussian_random_walk_analysis import GaussianRandomWalkTest
from logic.diffusion.msd_linear_analysis import MSDLinearTest
from visualization.trajectories_visualization import plot_trajectories, plot_speed_distribution, plot_angular_distribution, plot_speed_distribution_per_trajectory

TRACKING_FILE = os.path.join(TRACKING_FOLDER, "342843.avi.npy")

def main() -> None:
    # Load and prepare data
    tracked_data: List[List[BoundingBox]] = np.load(
        TRACKING_FILE, allow_pickle=True).tolist()
    restructured = restructure_data(tracked_data)

    # Create trajectories
    trajectories = [Trajectory(bboxes) for bboxes in restructured]

    # Analyze diffusion
    gauss_analyzer = GaussianRandomWalkTest()
    msd_analyzer = MSDLinearTest()

    gauss_results = [gauss_analyzer.analyze(
        trajectory) for trajectory in trajectories]
    msd_results = [msd_analyzer.analyze(trajectory)
                   for trajectory in trajectories]

    print(f"Gaussian fit: {sum(gauss_results)}/{len(gauss_results)}")
    print(f"Linear MSD fit: {sum(msd_results)}/{len(msd_results)}")
    
    os.makedirs(ANALYSIS_GRAPHICS_PATH)

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



if __name__ == "__main__":
    main()
