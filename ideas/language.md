# Python, JAX or Rust

**[sketch]** **Python stays the host; neither JAX nor Rust is the next
step.** The decision is the author's ([open](open.md)); here is the case.

## Not Rust, now

- The time goes to waste, not to the interpreter ([baseline](baseline.md)):
  the remaining stages are OpenCV's C++ and SciPy's C.
- Farneback has no mature Rust implementation; a Rust port would bind
  OpenCV anyway, and keep the same speed.
- The canon says no rewrite for elegance. Rust earns its place on one
  condition: **tracking live**, on a microscope stream, or over hours of
  video. Then the kernel to port is small — threshold, contours, Kalman,
  Hungarian, about three hundred lines — and the Python version stays as the
  test oracle, as in the point-cloud project.

## Not JAX for the pipeline

Nothing in the measurement needs a gradient: the flow, the contours and the
matching are discrete. JAX on CPU would add a compile, like numba, to work
that OpenCV already does.

## JAX, if the science becomes a model

JAX becomes the right tool the day the project **fits a generative model of
motility to the trajectories**: a run-and-tumble walk with a speed, a tumble
rate and a rotational diffusion, its likelihood or a simulation-based
inference, `vmap` over hundreds of trajectories and `grad` over the
parameters. That is the same reason Emitter-Optimizer chose JAX: the core
becomes an optimization. See [science](science.md).
