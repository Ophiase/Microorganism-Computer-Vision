# Rhythmic

**Question**: Do the reversals come in a rhythm?

| | |
| --- | --- |
| $H_0$ | sharp turns at random frames, a Bernoulli sequence |
| $H_1$ | [run-and-reverse](../../models.md#active), turns that keep time |

## The test

The frames that hold a turn sharper than ninety degrees, turns closer than
two frames merged into one event. The statistic is the coefficient of
variation of the intervals between events: small when they keep time. Its
null distribution comes from shuffling which frames hold the turns, which
keeps their number and destroys any rhythm.

Not a variation of one: in discrete time, turns at random give geometric
intervals, whose coefficient of variation is $\sqrt{1 - p}$, below one.
Comparing with one would call Brownian cells rhythmic.

## Checked on a truth

**[fact]** 2026-11-14: 100 % power on run-and-reverse, 0 % false positives
on the nine other behaviours.

## What it misses

- **[fact]** The first version failed for two reasons. The 2025 test took the
  spectrum of signed headings, and a reversal turns by $\pm\pi$ at a random
  sign. And the synthetic run-and-reverse had no rhythm: it drew a new period
  at every frame. The test was right to find none.
- A rhythm of tumbles that are not sharp, below ninety degrees, is not seen.

Back: [one cell](README.md).
