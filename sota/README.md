# sota: the state-of-the-art comparison (both sides, one place)

Both sides of the comparison for **N variable circles in a unit square, maximize
Σr**:

- [`packomania/`](packomania/): the Packomania `csqv` **best-known records**
  (N=1..100), the reference to beat (verified four independent ways; see
  `packomania/README.md`).
- [`nietzsche-sm-radical-v6-n27/`](nietzsche-sm-radical-v6-n27/): **our**
  evolved solver's results across N=2..100: the comparison table, the 98 usable
  packings, and the record-beating one (N=27) in
  [`nietzsche-sm-radical-v6-n27/wins/csqv27.pck`](nietzsche-sm-radical-v6-n27/wins/csqv27.pck).

**Headline:** 27 ties, **1 strict win (N=27)**, the rest under-searched at the
sweep's per-N budget. Full per-N table:
[`nietzsche-sm-radical-v6-n27/comparison.md`](nietzsche-sm-radical-v6-n27/comparison.md).
Win details, seed provenance, and how to verify:
[`nietzsche-sm-radical-v6-n27/README.md`](nietzsche-sm-radical-v6-n27/README.md).
