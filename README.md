# Microorganism Computer Vision 🧫

Deep Computer Vision 🦠 - Analysis of the motion of microorganisms

- <div style="text-align: center;"> <img src="./resources/results/342843_original.gif" width="300" /> <img src="./resources/results/342843_transformed.gif" width="300" /> </div>

## Installation

```bash
# install dependencies
make pip

# download the dataset
make extract
# preprocess the videos
make transform
```

## Execution

Extract the trajectories

```bash
# extract trajectories
make detection
# create synthetic trajectories
make synthetic
```

Analyse the trajectories

```bash
# render trajectories with bbox gif
make render

# render trajectories's analysis
make analysis
# which is equivalent to:
python3 -m script.main --task analysis
# if you want to specify the video:
python3 -m script.main --task analysis --video synthetic_brownian 
```

## Results


- <div style="text-align: center;">
  <img src="./resources/results/342843_analysis/angular_distribution.png" width="290">
  <img src="./resources/results/342843_analysis/speed_distribution_per_trajectory.png" width="300">
</div>

- <div style="text-align: center;">
  <img src="./resources/results/342843_analysis/trajectories.png" width="290">
  <img src="./resources/results/342843_analysis/speed_distribution.png" width="300">
</div>

## TODO (Deadline: February 30th, 2025)

- ✅ Find videos of microorganisms
    - ✅ Download and extract them
- ✅ Optical Flow
    - ✅ Check the associated vector field
    - Plot the vector field with a quiver plot
- Cluster entities
    - ✅ Compute the image gradient and optical flow gradient for each frame
        - Deduce clusters
    - ✅ Object detection on each frame 
    - ✅ Kalman Filter to ensure tracking (bbox id)
    - ✅ Visualize the entities in the video
- Motion analysis
    - ✅ Create the time series
        - Depends on the information we can extract from the microorganisms
            - E.g., Salmonella
        - Proposition:
    - ✅ Propose diffusion hypotheses (e.g., Gaussian Random Walk)
    - ✅ Perform statistical tests

- Experimental:
    - Simulate videos to train a segmentation model
    - Simulate trajectories to verify the statistical tests