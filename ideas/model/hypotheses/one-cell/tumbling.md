# Tumbling

**Question**: Does a swimmer turn more sharply than its rotational diffusion
allows?

| | |
| --- | --- |
| $H_0$ | [active swimmer](../../models.md#active), turns Gaussian |
| $H_1$ | [run-and-tumble](../../models.md#active) |

## The test

The turns' robust spread (median absolute deviation) gives a Gaussian; the count
of turns beyond sixty degrees is binomial under it. Tested only on tracks found
[persistent](persistent.md).

## Checked on a truth

**[fact]** 0 % false positives, 91 % power on run-and-tumble.

## What it misses

An identity swap makes a sharp turn too: a tumble is believed only on a reviewed
track.

Back: [one cell](README.md).
