# catalogue/: where the videos come from

Two files, kept by agents so that none searches the web from scratch. Add what
you learn; say who judged and when.

- [metasources.yaml](metasources.yaml): **where to look**. Each place that
  holds motility videos (Zenodo, Dataverse, Wikimedia Commons, figshare,
  Dryad, IDR, the challenges), how to search it, how to read a record, what it
  usually yields.
- [sources.yaml](sources.yaml): **what was found**. Each source with its
  truth, licence, trust, tags and, when it can be brought in, its files and
  their sizes.

A source with files comes into `data/` with `mcv fetch` (or `just extract`):
by its id, by a tag, only those with a truth, or the smallest first within a
disk budget. Without `--yes` it shows the plan and the total, and downloads
nothing.
