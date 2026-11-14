# Hypotheses about the decor and the others

A cell's law may depend on where it is and on who is near. Every test here
compares the real tracks with [surrogates](../population.md#surrogates-the-null-of-an-interaction):
the same tracks shifted or moved, which keep each cell's law and break the
relation under test. **[question]** None is built yet.

| Hypothesis | Question | Statistic |
| --- | --- | --- |
| [Trapped](#trapped) | do swimmers linger near objects? | time within $r$ of an object |
| [Slowed](#slowed) | do cells slow down near objects? | slope of speed against distance |
| [Steered](#steered) | do cells follow the edges? | mean $\lvert\cos\rvert$ of heading to the edge |
| [Aligned](#aligned) | do neighbours swim the same way? | polar order within $r$ |
| [Gathering](#gathering) | are cells attracted or repelled? | pair correlation $g(r)$ |
| [Driven](#driven) | does one kind follow or flee another? | cross correlation over time lags |

The laws behind them are drawn in [interactions](../interactions.md).

## Trapped

Swimmers near a surface turn in circles and stay; the study's paper links
this near-surface swimming to the sites where Salmonella invades host cells
**[to check]**. The time the real swimmers spend within $r$ of the host
cells, against the same tracks moved across the frame.

## Slowed

The speed of each observation against its distance to the nearest object; the
slope, against the slopes of moved tracks.

## Steered

Near an edge, the heading against the edge's tangent: parallel swimming gives
a mean $\lvert\cos\rvert$ near one, random headings near $2/\pi$.

## Aligned

The polar order of the headings of the cells within $r$ of each other, against
the same tracks shifted in time.

## Gathering

The pair correlation $g(r)$ of the positions: above one at short range is
attraction, below one is avoidance; the null shifts tracks in time.

## Driven

For two kinds, the correlation between one's velocity and the direction to
the other, at a time lag: a positive peak at a positive lag is a chase.

Back: [the hypotheses](README.md).
