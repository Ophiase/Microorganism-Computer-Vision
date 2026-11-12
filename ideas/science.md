# Where the ambition is

**[sketch]** The project is ambitious through its measurement, not through
its language. Three steps make it a result instead of a demo.

## 1. A truth to measure against

Render synthetic videos from the trajectory generator: cells with known paths
and known motility. Then two numbers become measurable — the **tracking
accuracy** (how many true paths the tracker recovers) and the
**classification accuracy** (how often the verdict is the true motion). An
honest benchmark, as in the other projects.

## 2. One verdict per trajectory, with its uncertainty

Six independent booleans that overlap are not a classification
([baseline](baseline.md)). Replace them with a model selection: fit each
candidate model, compare them by likelihood or information criterion, report
the anomalous exponent α with an interval. The physical units — micrometres
per pixel and frames per second — come with it.

## 3. A fitted motility model

Fit a run-and-tumble model to each trajectory and report its parameters.
That is where JAX enters ([language](language.md)).

## Prior art

- **[to check]** trackpy (Crocker–Grier linking) and TrackMate are the usual
  particle trackers; Omnipose segments bacteria specifically.
- **[to check]** The AnDi challenge benchmarks the classification of
  anomalous diffusion and the estimation of α on trajectories with known
  truth; its datasets could serve step 2.
