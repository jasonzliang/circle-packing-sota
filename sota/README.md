# sota: the state-of-the-art comparison (both sides, one place)

Both sides of the comparison for **N variable circles in a unit square, maximize
Σr**:

- [`packomania/`](packomania/): the Packomania `csqv` **best-known records**
  (N=1..100), the reference to beat (verified four independent ways; see
  `packomania/README.md`).
- [`si-v2-20260927/`](si-v2-20260927/): the best strictly feasible packing per
  N = 1..100 from the 2026-09-27 **fifteen-solver portfolio sweep**, against the
  2026-09-27 snapshot; regenerated and committed automatically while the sweep
  runs (see its README for the current WIN / tie / below counts).
- [`sm-radical-v6-n27/`](sm-radical-v6-n27/): **our** updated solver's results
  across N=2..100: the comparison table, the 98 usable packings, and the
  record-beating one (N=27) in
  [`sm-radical-v6-n27/pck/csqv27.pck`](sm-radical-v6-n27/pck/csqv27.pck).

**Headline:** 27 ties, **1 strict win (N=27)**, the rest under-searched at the
sweep's per-N budget. Full per-N table:
[`sm-radical-v6-n27/comparison.md`](sm-radical-v6-n27/comparison.md). Win
details, seed provenance, and how to verify:
[`sm-radical-v6-n27/README.md`](sm-radical-v6-n27/README.md).
