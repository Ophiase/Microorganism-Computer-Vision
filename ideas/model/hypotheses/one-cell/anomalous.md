# Anomalous

**Question**: Does the MSD grow as a power of time other than one?

| | |
| --- | --- |
| $H_0$ | [Brownian](../../models.md#passive), $\alpha = 1$ |
| $H_1$ | a power law $t^\alpha$ |

## The test

A likelihood ratio between a straight line and a power law, both with a noise
offset and fitted with Qian's variance; $\chi^2_1$ under the null.

## Checked on a truth

**[fact]** 1 % false positives on Brownian cells, 100 % power on swimmers, **0 %
on fractional subdiffusion** at $\alpha = 0.6$ over 100 frames.

## What it misses

A plain log-log regression called 85 % of Brownian cells anomalous: the noise
offset bends the MSD at short lags. Fractional motion bends it less than the
noise does at this length.

Back: [one cell](README.md).
