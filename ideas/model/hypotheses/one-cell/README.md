# Hypotheses about one cell

Each tests one track at a time; across a run's tracks, the false discoveries
are held at 5 % by Benjamini-Hochberg. The rates are measured on a hundred
synthetic tracks of each behaviour, 100 frames, 0.3 px of noise.

| Hypothesis | Question |
| --- | --- |
| [Moves](moves.md) | Does the cell move at all, beyond the noise of its position? |
| [Anomalous](anomalous.md) | Does the MSD grow as a power of time other than one? |
| [Persistent](persistent.md) | Does the cell keep its heading from one step to the next? |
| [Chiral](chiral.md) | Does the cell turn one way more than the other? |
| [Tumbling](tumbling.md) | Does a swimmer turn more sharply than its rotational diffusion allows? |
| [Rhythmic](rhythmic.md) | Do the reversals come in a rhythm? |
| [Confined](confined.md) | Is the cell held near a centre? |
| [Heavy tails](heavy-tails.md) | Are the steps drawn from a law with heavier tails than a Gaussian? |
| [Body-frame](body-frame.md) | Are the kicks larger along the body than across it? |
| [Gait](gait.md) | Does a wave of curvature travel along the body? |

Back: [the hypotheses](../README.md).
