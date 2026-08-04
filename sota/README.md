# sota: the state-of-the-art comparison (both sides, one place)

Both sides of the comparison for **N variable circles in a unit square, maximize Σr**:

- [`theirs/`](theirs/): the Packomania `csqv` **best-known records** (N=1..100), the reference to beat
  (verified four independent ways; see `theirs/README.md`).
- [`ours/`](ours/): **our** evolved solver's results across N=2..100: the comparison table, the 98 usable
  packings, and the record-beating one (N=27) in [`ours/wins/csqv27.pck`](ours/wins/csqv27.pck).

**Headline:** 26 ties, **1 strict win (N=27)**, the rest under-searched at the sweep's per-N budget. Full
per-N table: [`ours/comparison.md`](ours/comparison.md). Win details, seed provenance, and how to verify:
[`ours/README.md`](ours/README.md).
