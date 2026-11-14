# Rhythmic

**Question**: Do the reversals come in a rhythm?

| | |
| --- | --- |
| $H_0$ | sharp turns at random times |
| $H_1$ | [run-and-reverse](../../models.md#active) |

## The test

Today: the periodogram's peak of the turns' sizes against shuffled sizes.

## Checked on a truth

**[fact]** fails: 0 % power on run-and-reverse.

## What it misses

Each run's length varies, so the spectrum has no sharp peak. **[sketch]** Next:
the intervals between sharp turns, more regular than a Poisson process allows
(their coefficient of variation below one).

Back: [one cell](README.md).
