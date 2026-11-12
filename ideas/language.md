# Python, JAX or Rust

**[sketch]** **Python stays.** The pivot makes it clearer: the user of the
toolkit is an agent, and the toolkit is a set of small, readable steps with
their parameters exposed.

- **Not Rust**: the time goes to waste, not to the interpreter
  ([baseline](baseline.md)); the remaining stages are OpenCV's C++. Rust
  earns its place only for live tracking on a microscope stream.
- **Not JAX**: nothing needs a gradient; the calibration is done by the
  agent, not by a descent. JAX returns only if a motility model is someday
  fitted to the trajectories by likelihood.
- **The speed that matters** is one calibration loop: a run on a short clip
  must take seconds, so that the agent can try many configurations. That is
  step 1 of the [roadmap](roadmap.md).
