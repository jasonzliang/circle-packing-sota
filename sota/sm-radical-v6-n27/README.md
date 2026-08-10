# sota/ours: our results vs the Packomania records

Our updated solver's results for **N variable circles in a unit square, maximize
Σr**, measured against the Packomania `csqv` best-known values in
[`../packomania/`](../packomania/). The complete per-N packings from the sweep
are in [`pck/`](pck/) and the raw table in [`results.csv`](results.csv). The
sweep covered 99 sizes and produced 98 usable packings; N=97 collapsed and has
no file.

## Headline (sweep N=2..100, from scratch, 120s/N, seed 1; N=26 re-run at 240s over seeds 1-6)

| outcome                              | count | N                                                  |
| ------------------------------------ | ----- | -------------------------------------------------- |
| **beat the record (WIN)**            | **1** | **27**                                             |
| tie (≤1e-6)                          | 27    | all of 2–24, plus 26, 29, 31, 33                   |
| below (our under-search at larger N) | 70    | every N ≥ 25 except 26, 27, 29, 31, 33, 97         |
| no result at all                     | 1     | 97 (collapsed; see the root README's Known issues) |

Full table: [`comparison.md`](comparison.md). The 70 "below" are **our compute
budget, not the records' ceiling**. Large N need more search time than 120s
(and, like N=27, some may be beatable long-standing entries with a harder
re-run).

**Partial evidence that the 70 are under-search:** [`chase/`](chase/) re-runs
twelve near-misses at **240s over seeds 1–4** (vs the sweep's 120s on seed 1
alone) and reaches the record to **within 4e-11 at five of them** (N = 25, 28,
32, 35, 36), i.e. they become effective ties. Counting those, the repo
demonstrates 32 ties rather than 27; the headline table reports the per-N
budgets above, so the two counts are consistent.

Stated fully, because the result cuts both ways: seven of the twelve stayed
below the record, and at **N=30 eight times the compute produced no improvement
at all** (bit-identical to the 120s run; N=54 gained 1e-11, which is nothing).
So extra search closes the gap at some N and demonstrably does not at others.
The twelve are also not exactly the twelve smallest gaps: N=26 is absent because
the same treatment (240s over more seeds) already ties it, so it is folded into
the main set above rather than kept here.

## The one win: N = 27

- **ours Σr = 2.685978684198** vs **Packomania record 2.685350025228** →
  **+6.29e-4 (+0.023%)**
- Strictly feasible (independently re-checked with
  `verify_and_compare.py verify`, below): 27 circles, min wall slack +1.0e-12,
  min pairwise slack +7.2e-13, meaning no overlaps and every circle inside the
  square. The win margin is 8.7×10⁸ times the smaller slack, so it is real, not
  numerical noise.
- Why this one is beatable: the N=27 record is a **long-standing 2011/12 entry**
  (reference [1] = D. W. Cantrell, sci.math forum), **not** one of the recent
  AI-optimized entries (e.g. N=26 = 2.635983, credited to Haowei Lin [8], July
  2026, which we do not beat).
- The record packing is [`pck/csqv27.pck`](pck/csqv27.pck) (Packomania format),
  with the full float64 config (seed + budget) in
  [`json/out27.json`](json/out27.json).

Verify it yourself (no solver needed):

```bash
python3 ../../verify_and_compare.py verify pck/csqv27.pck --record 2.685350025228
```

## How the seed works, and where the N=27 result's seed comes from

**Where the seed is set.** The sweep was launched as
`run_sweep.py ... --seeds 1`, so for every N the driver calls
`pack.search(n, seed=1, budget=120)` and keeps the best strictly-feasible
result. (`--seeds` accepts a comma list; the best over all listed seeds is kept.
We used a single seed, `1`.)

**What the seed does.** Inside `solver-sm-radical-v6-n27/pack.py`,
`search(n, seed, budget)` creates **one** numpy PCG64 generator,
`rng = np.random.default_rng(seed)`. That single `rng` drives _everything_
stochastic: the random initial layouts (`start_random`/`start_grid`), the
basin-hopping perturbations (`perturb`/`perturb_dual`), and the per-restart
parameter choices. So the seed fully determines the random stream. The loop
repeatedly draws a fresh start (or perturbs the current best), runs the exact
radius LP + SLSQP refine + feasibility repair, and keeps the best. This repeats
**until the wall-clock `budget` expires**.

**The N=27 result's seed is therefore `1`.** To regenerate it:

```bash
python3 ../../solver-sm-radical-v6-n27/pack.py -n 27 --seed 1 --time 120 -o out27.json
```

**Reproducibility caveat (important).** The budget is _wall-clock_, so
`(seed=1, 120s)` is deterministic in its RNG stream but not in its _result_: a
slower or busier machine completes fewer restarts and may land in a different,
possibly worse local optimum. The seed matters more than the budget. A re-run at
**seed 7, 300s** found a _worse_ config (2.683803447573, re-confirmed on a
second machine: 1017 starts, 1852 hops) despite 2.5x the time, because it
explored a different basin. Note this varies both seed and budget, so it shows
the seed dominating rather than more time being harmful in itself. Either way a
specific win is tied to `(seed, budget, machine)`, while the saved packing
(`pck/csqv27.pck`) is a fixed artifact that is strictly feasible and beats the
record regardless of how it was found.

## Contents

```text
comparison.md            our full ours-vs-Packomania table (N=2..100)
results.csv              raw sweep output (n, sum_radii, max_violation, seeds, feasible)
json/out27.json          the N=27 full float64 config (seed + budget)
pck/csqv<N>.pck          the 98 usable packings, Packomania format (csqv27 also here; no N=97)
chase/                   re-runs of 12 near-misses at 240s over seeds 1-4; ties the record at 5 of them
```
