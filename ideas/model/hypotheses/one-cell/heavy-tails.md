# Heavy tails

**Question**: Are the steps drawn from a law with heavier tails than a Gaussian?

| | |
| --- | --- |
| $H_0$ | Gaussian steps |
| $H_1$ | [waiting times or Lévy flights](../../models.md#anomalous) |

## The test

Shapiro-Wilk on the steps of one frame, each axis standardised, pooled.

## Checked on a truth

**[fact]** 95 % power on waiting times, 97 % on Lévy walks; it also flags
swimmers and circling cells, whose steps are not Gaussian either.

## What it misses

It says *not Gaussian*, not *which* law: the model selection does that.

Back: [one cell](README.md).
