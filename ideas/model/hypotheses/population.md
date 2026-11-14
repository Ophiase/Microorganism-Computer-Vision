# Hypotheses about a population

Questions about the set of a run's cells, not about one of them. Each departs
from the [independence hypothesis](../population.md#the-independence-hypothesis).

| Hypothesis | Question | Status |
| --- | --- | --- |
| [Two kinds](#two-kinds) | are there two populations, by their speeds? | **[question]** |
| [Heterogeneous](#heterogeneous) | does one law fit every cell of a kind? | **[question]** |
| [Chirality bias](#chirality-bias) | do the circling cells prefer one sense? | **[fact]** built |
| [Ergodic](#ergodic) | do time averages equal the ensemble's? | **[question]** |

## Two kinds

$H_0$: one law of speeds; $H_1$: a mixture of two, such as motile and
non-motile. **[sketch]** The BIC of one Gaussian against two, on the log of
the speeds. The study's videos show it by eye: swimmers near 20 µm/s, still
and diffusing cells near zero.

## Heterogeneous

$H_0$: one diffusion coefficient for every cell of a kind; $H_1$: one per
cell. **[sketch]** The spread of the cells' $D$ against the spread their track
lengths alone would give.

## Chirality bias

$H_0$: each circling cell turns either way with probability one half; $H_1$:
one sense is preferred, as hydrodynamics near a surface predict. A binomial
test on the senses of the [chiral](one-cell/chiral.md) tracks, in `mcv stats`.

## Ergodic

$H_0$: a long track's time-averaged MSD equals the ensemble's; $H_1$: it does
not, as with waiting times. **[sketch]** The ergodicity breaking parameter:
the variance of the cells' time-averaged MSD over its squared mean.

Back: [the hypotheses](README.md).
