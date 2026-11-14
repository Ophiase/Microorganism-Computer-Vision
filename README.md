# Microorganism Computer Vision 🧫

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.14+](https://img.shields.io/badge/Python-3.14%2B-yellow.svg)](https://www.python.org/)
[![just](https://img.shields.io/badge/just-ready-blue.svg)](https://github.com/casey/just)
[![uv](https://img.shields.io/badge/uv-managed-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Kanon](https://img.shields.io/badge/%3C%3E-Kanon-black.svg)](.agents/kanon.md)

> [!WARNING]
> **A student project of February 2025, being turned into an agent's
> toolkit.** The 2025 version was tuned by hand on one study's videos. Since
> November 2026 Claude reworks it: the tools, the statistics and the
> calibration of every video are the agent's work in progress, measured where
> a truth exists and reviewed by eye where none does. Read the results as
> work in progress, not as a published method.

**Statistical tests of how microorganisms move, read from raw microscope
video.** Each cell is tracked, then its trajectory is tested against
hypotheses of behaviour: does it diffuse, swim, tumble, circle, turn one way,
stay caged? An agent calibrates the filters and the tracker on your videos,
reviews what they miss, and reports what the cells do, in µm/s.

<div align="center">
  <img src="docs/figures/salmonella-tracks.gif" width="95%">
  <br>
  <em>Salmonella Typhimurium near a surface, the study's video. Left: as
  filmed. Right: blue swimmers, orange diffusing, pink still cells, light
  tracks too short to judge. Calibrated by Claude.</em>
</div>

**More videos**: [the study's nine videos](docs/study/README.md), each with
its film, tracks and tests; [six synthetic scenes](docs/scenes/README.md),
each scored against its truth.

## The hypotheses, tested

The 2025 project tested six hypotheses with fixed thresholds. They are now
statistical tests, each with a null model, a p-value, and a correction for
testing hundreds of tracks at once (Benjamini-Hochberg).

```mermaid
graph LR
    track(a track) --> moves{how it moves}
    moves -->|no motion| still[still]
    moves -->|alpha = 1| brownian[Brownian]
    moves -->|alpha below 1, saturating| confined[confined]
    moves -->|alpha above 1, heading kept| swimmer[swimmer]
    swimmer -->|turns one way| circling[circling]
    swimmer -->|turns too sharply| tumble[run-and-tumble]
    swimmer -->|turns in a rhythm| reverse[run-and-reverse]
    classDef still fill:#e87ba4,stroke:#b85a80,color:#ffffff
    classDef brownian fill:#eb6834,stroke:#b34a20,color:#ffffff
    classDef confined fill:#1baf7a,stroke:#137f58,color:#ffffff
    classDef swimmer fill:#2a78d6,stroke:#1d5599,color:#ffffff
    classDef circling fill:#eda100,stroke:#b37900,color:#ffffff
    class still still
    class brownian brownian
    class confined confined
    class swimmer swimmer
    class circling circling
```

The colours are the verdicts' in every video and figure. Left out of the
drawing: the test of non-Gaussian steps, run on every track, which flags
heavy tails such as waiting times and Lévy flights.

**Every test is checked on a truth.** Synthetic cells of ten behaviours, a
hundred tracks each, with localisation noise; the share of tracks each test
calls significant (q < 0.05). On a behaviour that satisfies the null this is
the false positive rate, near 0.05 at most; elsewhere it is the power.

| True behaviour | anomalous | persistent | chiral | non-Gaussian | tumbling | confined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brownian | **0.01** | **0.00** | **0.00** | **0.00** | **0.00** | **0.02** |
| Swimmer | 1.00 | 1.00 | 0.00 | 0.25 | 0.00 | 0.00 |
| Run-and-tumble | 1.00 | 1.00 | 0.00 | 0.97 | **0.91** | 0.00 |
| Circling | 1.00 | 1.00 | **1.00** | 0.99 | 0.00 | 0.00 |
| Confined | 0.88 | 0.00 | 0.00 | 0.00 | 0.00 | **0.77** |
| CTRW, heavy waits | 0.05 | 0.00 | 0.00 | **0.95** | 0.00 | 0.03 |
| Lévy walk | 1.00 | 0.99 | 0.00 | 0.97 | 0.91 | 0.00 |

What fails, said plainly: the rhythm test does not yet find run-and-reverse
swimmers, and fractional subdiffusion (α = 0.6) over 100 frames goes
undetected. The 2025 Fourier test had the first flaw too: a reversal turns by
±π at random, which hides the rhythm in signed angles.

## The agent's tools

A pipeline gets most cells right and some wrong. The agent finds which, by
eye, and corrects them; every correction is an annotation, with its reason.

| | |
| --- | --- |
| ![sweep](docs/figures/review-sweep-threshold.png) | **`mcv sweep`**: one parameter over several values, the same view side by side. Here no threshold catches the large diagonal rods: the 2025 area filter measured bounding boxes, which grow when a rod lies diagonally. |
| ![misses](docs/figures/review-misses.png) | **`mcv misses`**: a looser detection, tracked, where no kept track goes. The agent looks; plain rods get `mcv correct ... force`. |
| ![suspects](docs/figures/review-suspects.png) | **`mcv suspects`**: jumps that look like identity swaps, broken tracks, unsure verdicts, filmed up close; `mcv correct ... split / join / drop / label`. |
| ![still](docs/figures/review-gallery-still.png) | **`mcv gallery`**: every track of one verdict at once. These are still rods, dead or sleeping, which a median background erased from every frame until the agent noticed. |

Also: `mcv probe` measures a video and reads its scale bar, `mcv inspect`
films one track, `mcv stats` compares populations, `mcv synth` renders a twin
with a known truth, `mcv bench` and `mcv search` score against it.

## Read more

- [The study's videos, case by case](docs/cases/salmonella.md): Salmonella
  near surfaces and host cells, in µm/s.
- [The calibration, measured](docs/benchmark/calibration.md): the agent
  against blind and local searches on unseen scenes.
- [The method an agent follows](docs/guide.md), printed by `mcv guide`.
- [Twenty sources of motility videos](catalogue/sources.yaml), their truth
  and licence.

## Use it

```bash
just install                 # Python 3.14 and the locked dependencies, by uv
just extract                 # the nine videos of the 2025 study, into data/
uv run mcv guide             # the method, for you or your agent
uv run mcv probe data/342843.avi
```

## Where it comes from

A student project of February 2025
([its README](https://github.com/Ophiase/Microorganism-Computer-Vision/tree/v0.5)),
its six hypothesis tests the seed of this one. The plan is in
[ideas/](ideas/README.md); the rules any contributor follows, human or agent,
in [AGENTS.md](AGENTS.md). The canon is [.agents/kanon.md](.agents/kanon.md);
`<>` is its sign.
