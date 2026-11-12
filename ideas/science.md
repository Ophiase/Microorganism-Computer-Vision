# The science

**[sketch]** The ambition is in the measurement, and the agent needs it more
than a person does: it can only be trusted on a truth.

## 1. A truth to measure against

Render synthetic videos from the trajectory generator: cells with known paths
and known motility, on noise and backgrounds taken from the real sources. The
**tracking accuracy** and the **classification accuracy** become measurable.
Labelled swimming videos are rare in public ([sources](../catalogue/sources.yaml)),
so the synthetic truth is the benchmark, and the real videos the check.

## 2. One verdict per trajectory, with its uncertainty

Six independent booleans that overlap are not a classification
([baseline](baseline.md)). Replace them with a model selection: fit each
candidate model, compare them, report the anomalous exponent α with an
interval, in micrometres and seconds.

## Prior art

- **[to check]** TrackMate (Fiji) and trackpy are the usual trackers; YSMR
  is a bacterial tracker in Python; Omnipose segments bacteria.
- **[to check]** The AnDi challenge benchmarks the classification of
  anomalous diffusion on trajectories with a known truth.
- **[to check]** LLM agents that tune pipelines exist for machine-learning
  pipelines (IMPROVE, 2025) and loss weights (LossAgent, 2024); none found
  that calibrates a classical image pipeline in place of training one.
