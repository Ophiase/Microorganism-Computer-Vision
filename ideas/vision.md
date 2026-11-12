# Vision: a change of direction

**[defined]** 2026-11-12. The project changes direction.

## What it was

A student project, made by hand in two months of spare time: one pipeline of
filters, optical flow, Kalman tracking and six diffusion tests, tuned by eye
on nine videos of one study. The way to make such a pipeline advanced was
the machine-learning way: label data, train a network, ship the weights.

## What it becomes

**A toolkit an agent calibrates.** The researcher brings their own videos;
an agent — Claude or any other — reads them, chooses the filters, sets the
thresholds, the tracker's noise and the hypotheses, looks at the result and
corrects it. The knobs that a network would have learned by gradient
descent are set by an agent that can see the frames and explain each choice.

- **The library** (`libraries/motility/`) is the set of tools: small, typed,
  each with its parameters exposed and documented.
- **The test cases** are the sources of
  [catalogue/sources.yaml](../catalogue/sources.yaml): the agent is measured
  on the ones with a known truth, and serves the ones without.
- **The output** is a motility report a researcher can read and check: the
  tracks, the verdict, the parameters, and why they were chosen.

## Why

A network needs labelled data a researcher with one Petri dish does not
have, and gives no reason for its answer. A pipeline of classical filters is
cheap, runs on a laptop and is readable, but needs an expert to tune it on
each microscope. An agent is that expert, available to anyone.

The claim to measure ([calibration](calibration.md)): *on videos with a known
truth, an agent with the toolkit reaches tracking and classification close to
a tuned method, without a label and in minutes, and says why.*

## For whom

- **A researcher with videos and no annotations**: one species, one
  microscope, a question about motility.
- **A student** who wants to understand what each step does, not a weight
  file.
- **Agents**, which drive the toolkit through commands and read its results
  as data, as in the author's other projects.
