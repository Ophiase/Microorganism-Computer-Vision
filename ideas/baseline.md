# Baseline

**[fact]** 2026-11-12, v0.6.0, on the laptop: video `342843.avi`, frames 0 to
85, 672 x 512 pixels. Re-measure on the desktop before any optimization step.

| Stage | Time | Done by |
| --- | ---: | --- |
| Load and grayscale | 0.5 s | OpenCV |
| `pre_transform`, compile included | 2.4 s | numba |
| Farneback optical flow | 7.6 s | OpenCV |
| `transform_video`, compile only | 11.5 s | numba |
| `transform_video` | 34.0 s | numba and NumPy |
| Detection and Kalman tracking | 5.1 s | OpenCV, SciPy |
| Six hypotheses, 283 trajectories | 0.5 s | NumPy, SciPy |

Peak memory 3.9 GB; one run is about a minute.

## What the numbers say

- **Three quarters of the time is self-inflicted.** `transform_video` is a
  3 x 3 box blur and a forward difference. The numba blur gives the same
  result as `cv2.filter2D` (largest difference 0.0); it is the compile and
  the stacking of a float64 tensor that cost.
- **The intermediate tensor is 2.8 GB**, float64, shape
  `(85, 512, 672, 3, 4)`. The detection reads one channel of the twelve.
- **What remains is already compiled code**: Farneback is OpenCV's C++, and
  the analysis is under a second. No language change moves these numbers;
  removing the waste does.

## Defects found on the way

- **[fact]** `just run analysis` crashes on a machine without Chrome:
  `write_image` goes through kaleido 1.x, which drives a Chrome install.
- **[fact]** The tracking file of February 2025 no longer loads: it is a
  pickled object array, and the refactor of `BoundingBox` broke it. It reads
  as one trajectory, then fails with an `AttributeError`.
- **[fact]** The six hypotheses are not exclusive: on the 283 trajectories,
  180 fit a linear MSD, 202 a persistent motion and 47 a subdiffusion.
- **[fact]** No test runs the pipeline end to end, so neither failure above
  was caught.
