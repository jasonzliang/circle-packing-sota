# circle-packing-sota

**AI-generated circle-packing solvers**, extracted from self-improvement
(AI-Generating-Algorithms) experiments and run across a range of N, compared to
the authoritative **Packomania** records for:

> **Pack N variable-sized circles in a unit square so that no two overlap and
> all stay inside the square, maximizing the sum of the radii Σr**, the
> AlphaEvolve / ShinkaEvolve benchmark (Packomania's `csqv` table).

Records are **fetched live** from Packomania (not hardcoded) — the `csqv` table
is actively updated, so best-known values move over time and every claim below
is checked against the table _as fetched_.

**Start with the [write-up](writeup/README.md)** for the original N=27 result
(note: that write-up predates the live-record update described below). The rest
of this file is the technical record: the current standings, how to verify them,
how to reproduce, and how the solvers work.

## Result

Two AI-generated solvers live here; all standings are against the **live**
Packomania `csqv` best-known.

**The updated n54 solver holds 21 strictly-feasible packings that beat the
current best-known.** Verified against the live table (fetched 2026-08-10): N =
**50, 51, 52, 53, 54, 55, 62, 63, 66, 67, 68, 69, 71, 72, 77, 80, 82, 83, 84,
85, 87**, by margins from +2.3e-5 (N=55) to +2.6e-3 (N=51), each independently
re-verified from its `.pck` coordinates. Full per-N table:
[`sota/sm-radical-v6-n54/comparison.md`](sota/sm-radical-v6-n54/comparison.md)
(**21 wins · 45 ties · 26 below**, of 92 solved). The solver is in
[`solver-sm-radical-v6-n54/`](solver-sm-radical-v6-n54/).

**The original n27 solver set the N=27 best-known.** It reproduces 27 known
records and found N=27 Σr = **2.685978684198**, which beat the long-standing
2011/12 entry (2.685350025228, +6.3e-4; reference [1], D. W. Cantrell, sci.math
forum). The live `csqv` table has since been updated to **exactly that value** —
it matches our Σr to all 12 digits — so against the _current_ best-known, N=27
now reads as a **tie**: our result is the record, not a beat. As many other
records also rose (43 of 100 between the 2026-08-03 and 2026-08-10 snapshots),
this solver's full sweep is now **0 wins · 28 ties · 70 below** vs the live
table:
[`sota/sm-radical-v6-n27/comparison.md`](sota/sm-radical-v6-n27/comparison.md).
(It failed outright only at N=97 — see [Known issues](#known-issues).)

**Every claimed win is independently re-verifiable from its `.pck` with
`verify_and_compare.py verify`**; `verify_and_compare.py fetch` refreshes the
record table.

## Verify the results

**No search re-run needed.** Each packing is a fixed file; checking it is pure
arithmetic on its (x, y, r) triples, so nothing needs installing. Verify any
single packing — feasibility + Σr, and (with `--records`) its status against the
best-known:

```bash
python3 verify_and_compare.py verify sota/sm-radical-v6-n54/pck/csqv54.pck \
        --records sota/packomania/packomania_csqv.json
```

Expected output (exit code 0):

```text
file            : sota/sm-radical-v6-n54/pck/csqv54.pck
author          : Jason Liang
circles (N)     : 54
sum of radii Σr : 3.842635794451
min radius      : 5.000e-02  (must be > 0)
min wall slack  : 1.000e-11  (>= 0 => all inside the square)
min pair slack  : 1.775e-11  (>= 0 => no overlap)
STRICTLY FEASIBLE (tol 1e-09): True
record          : 3.841733103296
Δ (ours-record) : +9.027e-04   -> BEATS the record
```

To check **all** packings against the **current live** records at once — a WIN
must be strictly feasible AND strictly exceed the record — refresh the table
straight from Packomania and compare:

```bash
python3 verify_and_compare.py fetch                                              # -> sota/packomania/packomania_csqv.json
python3 verify_and_compare.py compare --pck-dir sota/sm-radical-v6-n54 --refresh
```

`verify_and_compare.py` is **pure standard library and shares no code with the
solver**, so it is a genuine independent check of our packings or anyone else's:
it re-derives circle count, containment, overlap and Σr from the coordinates
alone, and exits non-zero on any failure. The win slacks are small but
**positive** (strictly feasible, not feasible-within-tolerance).

**On N=27:** the earlier N=27 packing
([`sota/sm-radical-v6-n27/pck/csqv27.pck`](sota/sm-radical-v6-n27/pck/csqv27.pck),
Σr = 2.685978684198) is now the _listed_ best-known — the live record equals it
to 12 digits — so `verify … --records` reports it as a **tie**. Against the
older 2011/12 value it originally beat it still shows `BEATS`:
`verify_and_compare.py verify sota/sm-radical-v6-n27/pck/csqv27.pck --record 2.685350025228`.

### Zero-tolerance check in exact arithmetic

`verify_and_compare.py` uses float64 and a 1e-9 tolerance, which leaves a fair
objection: with slacks around 1e-12, is the margin itself float noise?
`solver-sm-radical-v6-n27/exact_check.py` settles it. Every number in the stored
config is a finite binary float, hence an exact rational, so the constraints can
be _decided_ with **zero tolerance** via `fractions.Fraction` and squared
comparisons, with no square roots and no epsilon:

```bash
python3 solver-sm-radical-v6-n27/exact_check.py sota/sm-radical-v6-n27/json/out27.json --n 27
# exact: 27 circles, all constraints decided in exact rational arithmetic with ZERO tolerance
# exact: tightest wall slack (squared) = +1.460063e-13  (ok)
# exact: tightest pair slack (d^2-s^2) = +6.269444e-13  (ok)
# exact: sum(r) = 2.685978684198
# EXACT_FEASIBLE: True
```

A configuration that passes this is feasible **as a matter of arithmetic fact,
not of tolerance**. Two things to note. `--n 27` is **required**, because it
defaults to 26 and a count mismatch is reported as `EXACT_FEASIBLE: False` (exit
1, so it is script-safe). And its slacks are **squared** (`d² - s²`), so they
are not comparable to `verify_and_compare.py`'s linear ones above; it also reads
the full-precision `.json` rather than the `.pck`.

## `.pck` format

Packomania's submission format, defined at
[packomania.com/hints.html](https://www.packomania.com/hints.html):

- line 1 = radius of the **largest** circle, as a bare number (no letters, no
  `=`);
- line 2 = author name(s), comma-separated if several;
- line 3 onward = one `x y r` per circle, whitespace-separated, **sorted by
  increasing radius**;
- the square container is fixed at **side 1, centred at (0,0)**, so coordinates
  live in `[-0.5, 0.5]²` and must be rescaled to fit. We emit 12 decimals,
  matching Packomania's own published `csqv` coordinates.

**Two conventions coexist in this repo:** `.pck` files are origin-centred as
above, while the solver's `.json` configs use the `[0, 1]²` corner convention
(see `container.vertices`); convert by subtracting 0.5.
`verify_and_compare.py verify --corner`/`--side` reads either.

## Requirements

Python 3 with **numpy** and **scipy** (`pip install -r requirements.txt`). scipy
supplies both `linprog` and `minimize`, so without it there is **no LP and no
SLSQP**: `pack.py` degrades to an iterative radius-shrinking heuristic that is
measurably worse and returns no duals. Treat scipy as required.
`verify_and_compare.py` and `exact_check.py` need only the standard library.

## Reproduce the 21 live record-beats (n54 solver — the headline)

**No search needed to check them** — the 21 wins are fixed `.pck` files.
Re-derive their status against the current live records with the same
`compare --refresh` shown in [Verify](#verify-the-results): it re-verifies all
92 committed packings and prints `WINS: 21` while the table stands, regenerating
`sota/sm-radical-v6-n54/comparison.md` in place. The count can move as the live
records do; the `.pck` never change.

**Re-run the sweeps that produced them** (stochastic wall-clock search). It took
three passes — the base sweep, a longer targeted pass for four hard near-misses,
and a still-longer phase-2 pass for three more:

_Base sweep — 13 of the 21 wins_ (`N=2..100`, 250 s/(N, seed), best over seeds
1..10; ~7 h on 10 cores; N=54's committed win is instead Run B's own 240 s run,
folded in — see notes):

```bash
cd solver-sm-radical-v6-n54                     # run_sweep.py does `import solve`, so run it here
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 run_sweep.py --nmin 2 --nmax 100 --time 250 \
          --seeds 1,2,3,4,5,6,7,8,9,10 --workers 10 --out-dir ../repro-n54
cd .. && python3 verify_and_compare.py compare --pck-dir repro-n54 --refresh --out repro-n54/comparison.md
```

_Targeted near-miss pass — the other 4 wins (N=72, 82, 84, 85)_, which needed a
longer 400 s budget:

```bash
cd solver-sm-radical-v6-n54
OMP_NUM_THREADS=1 python3 run_sweep.py --ns 72,82,84,85 --time 400 \
  --seeds 1,2,3,4,5,6,7,8,9,10 --workers 4 --out-dir ../repro-n54-targeted
```

_Phase-2 near-miss pass — 3 more wins (N=62, 66, 67)_, at a much longer 1500
s/(N, seed) over ten stubborn near-misses (seeds 1–4):

```bash
cd solver-sm-radical-v6-n54
OMP_NUM_THREADS=1 python3 run_sweep.py --ns 38,39,57,60,62,65,66,67,70,75 --time 1500 \
  --seeds 1,2,3,4 --workers 10 --out-dir ../repro-n54-phase2
```

Write to a **fresh `--out-dir`** — the default (`../sota/sm-radical-v6-n54`)
_is_ the committed reference. Both are stochastic searches under a wall-clock
budget, so a re-run **won't be bit-identical** and the exact set of live wins
can differ run to run; every emitted `.pck` is independently verifiable (see
caveats below). Each win's `json/out<N>.json` records the budget + seed it was
found at (250 s for the 13 base-sweep wins, 240 s for N=54, 400 s for the four
targeted ones, and 1500 s for the three phase-2 ones).

## Reproduce the N=27 record (n27 solver)

```bash
pip install -r requirements.txt                              # numpy + scipy
python3 solver-sm-radical-v6-n27/pack.py -n 27 --seed 1 --time 120 -o out27.json
```

That is the exact invocation behind the record: **seed 1, 120 s**. It has
reproduced the stored configuration bit-for-bit on two different machines,
though at different points in the budget (the winning value first appeared at 91
s on one and 109 s on the other), which is what you would expect from a
wall-clock budget. It is a stochastic search, so a re-run elsewhere is not
pass/fail; see the caveats below.

**Whole n27 sweep, one command** (`reproduce.sh` drives the n27 `run_sweep.py`,
then compares + independently verifies every packing into a fresh `repro/` dir —
it does **not** clobber the committed reference; full `N=2..100` takes ~25 min
on a 10-core machine):

```bash
./reproduce.sh                           # full sweep N=2..100 @120s/N, then compare, then verify
./reproduce.sh 20 30 60 4                # quick demo: N=20..30 @60s/N, 4 workers
NMIN=27 NMAX=27 TIME=120 ./reproduce.sh  # a single N
```

Or step by step:

```bash
OMP_NUM_THREADS=1 python3 solver-sm-radical-v6-n27/run_sweep.py \
  --nmin 2 --nmax 100 --time 120 --workers 8 --out-dir repro
python3 verify_and_compare.py compare --pck-dir repro --refresh --out repro/comparison.md
for f in repro/pck/csqv*.pck; do python3 verify_and_compare.py verify "$f" >/dev/null || echo "INFEASIBLE: $f"; done
```

(To re-score the _committed_ n27 sweep against the live table instead of
`repro/`, point the same `compare --refresh` at
`--pck-dir sota/sm-radical-v6-n27`; N=27 now reads as a tie — see
[Verify](#verify-the-results).)

## Reproducibility notes

- **The artifacts are fixed and verifiable; both searches are stochastic.** Each
  solver is a multi-start search under a _wall-clock_ budget, so a given N's
  best depends on seed, budget and machine speed — a re-run won't be
  bit-identical, and for n54 the exact set of live wins can shift from run to
  run. That weakens no claim: every committed `.pck` is strictly feasible and
  independently checkable with `verify_and_compare.py verify` (pure stdlib, no
  shared code), however it was found.
- **The seed dominates — vary seeds before raising `--time`.** n27: seed 1 @120
  s found the record-beating 2.685978684198, while seed 7 @300 s found a worse
  2.683803 on 2.5× the budget. n54: 13 of the 21 wins came from the `N=2..100`
  base sweep at 250 s/(N, seed), best over seeds `1..10` (winning seeds spread
  across the range — e.g. N=87 seed 4, N=55 seed 9, N=63 seed 10); N=54's
  committed packing is Run B's own 240 s iteration-10 result, folded in as it
  beats the sweep's N=54; the other 4 (N=72, 82, 84, 85) came from a **targeted
  400 s** near-miss pass; and 3 (N=62, 66, 67) from a longer **1500 s phase-2**
  pass — so both _more seeds_ and _more time per seed_ mattered, and no single
  seed carries the result. Use `--seeds` (both `run_sweep.py`) or
  `pack.py --seed`. n27 N=97 is a case where more time cannot help at all — see
  [Known issues](#known-issues).
- **What is stored where.**
  - **n54 (headline, 21 wins)** → `sota/sm-radical-v6-n54/`: `pck/csqv<N>.pck`
    (92 packings for N=2..93 — N=94..100 have no feasible packing at 250 s, see
    Known issues; 12 dp, Packomania format), `json/out<N>.json` (full float64
    config + winning seed + budget), `results.csv`, `comparison.md`, and the
    run's `sweep.log`.
  - **n27 (set the N=27 record)** → `sota/sm-radical-v6-n27/`: `pck/csqv<N>.pck`
    (98 packings, no file for N=97), `json/out27.json` (the N=27 full float64
    config with its seed and budget), `results.csv`, `comparison.md`.
  - A fresh sweep from either solver writes the same `pck/` +
    `json/out<N>.json` + `results.csv` layout to its `--out-dir`.
- **Rounding to 12 dp is safe.** Every comparison and verification runs on the
  rounded `.pck` itself, not on the unrounded solution, and the packings stay
  strictly feasible (at n27 the shave costs 3.1e-13 of Σr; the positive win
  slacks in [Verify](#verify-the-results) are all measured post-rounding). Where
  a raw solution only just touches (a 12-dp slack of ~−1e-13),
  `verify_and_compare.py repair <pck>` makes it submission-grade — a uniform
  radius shrink to a target min slack (default 1e-11), record-gated so it never
  emits a packing that no longer wins.

## Known issues

**The sweep produces no result at N=97**, so it covers 99 sizes and yields 98
packings: `results.csv` has a blank row with `feasible=0` and there is no
`pck/csqv97.pck`.

It is an **absorbing state**, not a hard instance. Replaying
`search(97, seed=1, budget=120)`: the first grid start refines to 532 coincident
centre-pairs, so `repair()`'s single uniform scale factor hits 0 and zeroes
every radius. That config violates nothing and scores 0, and since `best_s`
starts at `-1.0`, **zero counts as an improvement** and becomes the incumbent.
Every later iteration hops from it, but a hop moves only a subset of the
circles, so the unmoved ones stay stacked and one leftover zero distance
re-zeroes everything. Coincident pairs rose with every one of the run's 4
candidates.

So the budget bought one real attempt, not four, and n=97 is otherwise fine: a
different grid start refines cleanly to 5.121755, only 1.04% under the record.
**More time on seed 1 cannot help** since the state is absorbing; more seeds
can, because each seed is an independent stream with its own first candidate.

Both checkers now catch it, fixed in the harness since `pack.py` is kept
unmodified: `verify_and_compare.py` prints `DEGENERATE` and exits 1 on any zero
radius, and `run_sweep.py` discards such candidates so the N records
`feasible=0` with no `.pck` rather than a file that looks valid. The N=27 claim
is unaffected.

**The n54 sweep finds no feasible packing at N=94..100 (at 250 s).** There its
`solve()` returns `no candidate — budget too small`: the broad-multistart stage
cannot construct a feasible layout that large within 250 s, so `results.csv`
carries blank `feasible=0` rows and no `.pck` there. Unlike n27's N=97 (an
absorbing-state bug), this is purely a budget limit — those sizes need a much
larger per-N time budget. So the n54 artifacts cover **N=2..93** (92 packings).

## Layout

```text
solver-sm-radical-v6-n27/  pack.py container.py shape.py  # original AI-written solver (set the N=27 record), UNCHANGED
               exact_check.py  # zero-tolerance feasibility decision in exact rational arithmetic
               run_sweep.py    # parallel N-sweep -> pck + json + results.csv
solver-sm-radical-v6-n54/  pipeline.py slp.py packlib.py endgame.py broad.py ...  # the UPDATED solver (21 live wins)
               run_sweep.py    # its N-sweep driver
verify_and_compare.py          # fetch LIVE packomania records + independent pck verify + compare -> comparison.md
sota/          the SOTA comparison, all in one place:
  packomania/    packomania_csqv.json (live, canonical) + history/<dated> snapshots + README   # the best-known records
  sm-radical-v6-n27/  results.csv comparison.md + README; pck/ (no N=97), json/out27.json, chase/
  sm-radical-v6-n54/  results.csv comparison.md; pck/ json/   # updated solver: 21 live record-beats
writeup/       README.md + figs   # narrative explainer of the original N=27 result
reproduce.sh  requirements.txt
```

## Solver Algorithm

This repo has **two** solvers, both produced **unchanged** by the same automated
program-search / self-improvement loop (LLM-driven coding, lineage
`sm-radical-v6`), not hand-written here. They share one structural idea and
differ only in how they optimise the circle _centres_.

**Shared inner layer — radii are an exact LP for fixed centres.** For _fixed_
centres, the optimal radii are the solution of a linear program
(`r_i + r_j ≤ d_ij` for every pair, `r_i ≤ w_i`, `r_i ≥ 0`), solved exactly and
in milliseconds, so both solvers treat the problem as a search over centres with
the radii one LP away. Both implement it with scipy/HiGHS and fall back to a
numpy-only monotone fixed point when scipy is absent (n54: `packlib.max_radii`;
n27: `pack.lp_solve`).
[full algorithm](solver-sm-radical-v6-n54/README.md#algorithm)

**The updated n54 solver** (`solver-sm-radical-v6-n54/`) is behind all **21 live
record-beats**. Its one n-generic entry point `solve(n, seconds, seed)`
(`solve.py` → `pipeline.py`) runs a **broad multistart** to a good funnel, then
an **endgame** basin-hop: exact-LP radii → an **SLP-KKT primal linearization
polish** (a feasibility-preserving Sequential LP that drives any layout to a
jammed KKT point) → **threshold-accepting basin hopping** over KKT points. This
is the improvement that cracked records the n27 approach plateaued on.
[full algorithm](solver-sm-radical-v6-n54/README.md#algorithm)

**The original n27 solver** (`solver-sm-radical-v6-n27/pack.py`) set the N=27
best-known (Σr = 2.685978684198 — now the _listed_ best-known, a tie; see
[Verify](#verify-the-results)). Its entry point `search(n, seed, budget)` is a
multi-start over centres whose optimiser is **joint SLSQP over (x, y, r)** with
exact-LP radii, a **provable contact-graph reduction**, and greedy **subset
hopping**. [full algorithm](solver-sm-radical-v6-n27/README.md#algorithm)

**Novelty, stated without over-claim.** Both solvers are clean **syntheses of
published parts**, assembled and tuned autonomously by the loop, not new
algorithms: the **radii-as-LP-for-fixed-centres** formulation (Eppstein 2016,
arXiv:1607.02184); the **convex-concave / convex-feasible-set** linearisation of
the non-overlap constraint used by the SLP (DCCP, Shen–Diamond–Gu–Boyd 2016);
**monotonic basin hopping** (Addis–Locatelli–Schoen) with a
**threshold-accepting** acceptance rule (Dueck & Scheuer 1990). The value here
is the **autonomous synthesis plus independently verified live record-beats**,
not a novel method.

> **Stale internal references.** Being verbatim copies from the originating
> experiment, both solvers' docstrings still address that experiment's layout:
> `tools/*.py` paths, a `bench/` tree (`bench/bench1/best.json`,
> `bench/bench2/`, `bench/verify.py`), `bench1`…`bench5` instances,
> `iter 4`/`iter 5` stages, and logs such as `artifacts/iter3/speedup.log` and
> `artifacts/iter9_cycles_sweep.log`. None exist here. Read them as provenance,
> not instructions; every runnable entry point is
> `solver-sm-radical-v6-{n27,n54}/*.py`.
