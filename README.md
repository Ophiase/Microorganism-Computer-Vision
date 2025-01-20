# Microorganism-Computer-Vision

Deep Computer Vision 🦠 - Analysis of the motion of microorganisms

## TODO (Deadline: February 30th, 2025)

- ✅ Find videos of microorganisms
    - ✅ Download and extract them
- ✅ Optical Flow
    - ✅ Check the associated vector field
    - Plot the vector field with quiver plot
- Cluster entities
    - ✅ Compute the image gradient and optical flow gradient for each frame
        - Deduce clusters
    - If a cluster's shape remains consistent (e.g., bounding box)
        - It may be an entity 
    - Visualize the entities
    - Propose alternative approaches (e.g., Kalman filter with YOLO object detection)
- Motion analysis
    - Create time series of the form (x, y, radius)
        - Depends on the information we can extract from the microorganisms
            - Eg: Salmonel
    - Propose diffusion hypotheses (e.g., Gaussian Random Walk)
    - Perform statistical tests


- Experimental:
    - Simulate videos, to train a segmentation model

