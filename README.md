# circle-packing-sota

An **AI-written circle-packing solver**, extracted from a self-improvement (AI-Generating-Algorithms)
experiment and run across a range of N, compared to the authoritative **Packomania** records for:

> **Pack N variable-sized circles in a unit square so that no two overlap and all stay inside the square,
> maximizing the sum of the radii Σr**, the AlphaEvolve / ShinkaEvolve benchmark (Packomania's `csqv` table).

## Result

A sweep of N=2..100 (120s per N, seed 1) **reproduces 26 of the known records to within 4e-11** (worst
3.4e-11 at N=29, median 1.5e-11) and found one **strictly-feasible packing that beats a current Packomania
best-known**: N=27, Σr = 2.685978684198 vs the listed 2.685350025228 (+6.3e-4). That record is a
long-standing 2011/12 entry (reference [1], D. W. Cantrell, sci.math forum), not one of the recent
AI-optimized entries (e.g. N=26 = 2.635983, credited to Haowei Lin [8] in July 2026, which we do **not**
beat).

**The other 71 fall short, several by more than 1%**: 42 gaps exceed 0.5% and 26 exceed 1%, the worst being
N=92 at 2.29%. The sweep covered 99 sizes but produced only **98 usable packings**, failing outright at
N=97 (see [Known issues](#known-issues)). Per-N detail is in `sota/ours/comparison.md`; **every claimed win
is independently re-verifiable from its `.pck` with `solver/verify_pck.py`.**

## Write-up

A narrative explainer of this result — what the problem is, what record was beaten, how a
self-improving AI agent produced the solver, and why it matters — with figures:
**[`writeup/`](writeup/README.md)**.

## Verify the N=27 result

**The win does not depend on re-running the search.** The packing is a fixed file, and checking it is pure
arithmetic on 27 (x, y, r) triples, so nothing needs installing. From a clean checkout:

```bash
python3 solver/verify_pck.py sota/ours/wins/csqv27.pck --record 2.685350025228
```

Expected output (exit code 0):

```text
file            : sota/ours/wins/csqv27.pck
author          : Jason Liang
circles (N)     : 27
sum of radii Σr : 2.685978684198
min radius      : 6.648e-02  (must be > 0)
min wall slack  : 1.000e-12  (>= 0 => all circles inside the square)
min pair slack  : 7.198e-13  (>= 0 => no two circles overlap)
STRICTLY FEASIBLE (tol 1e-09): True
record          : 2.685350025228
Δ (ours-record) : +6.287e-04   -> BEATS the record
```

`verify_pck.py` is **pure standard library and shares no code with the solver**, so it is a genuine
independent check of our packings or anyone else's: it re-derives the circle count, containment, overlap
and Σr from the coordinates alone, and exits non-zero on any failure. A copy of the output above is stored
in [`sota/ours/wins/csqv27.verify.txt`](sota/ours/wins/csqv27.verify.txt) to diff against.

The slacks are small but **positive** (strictly feasible, not feasible-within-tolerance), and the win
margin (+6.287e-4) is 8.7×10⁸ times the smaller of them, so this is not numerical noise.

### Zero-tolerance check in exact arithmetic

`verify_pck.py` uses float64 and a 1e-9 tolerance, which leaves a fair objection: with slacks around
1e-12, is the margin itself float noise? `solver/exact_check.py` settles it. Every number in the stored
config is a finite binary float, hence an exact rational, so the constraints can be *decided* with **zero
tolerance** via `fractions.Fraction` and squared comparisons, with no square roots and no epsilon:

```bash
python3 solver/exact_check.py sota/ours/wins/csqv27.seed1.json --n 27
# exact: 27 circles, all constraints decided in exact rational arithmetic with ZERO tolerance
# exact: tightest wall slack (squared) = +1.460063e-13  (ok)
# exact: tightest pair slack (d^2-s^2) = +6.269444e-13  (ok)
# exact: sum(r) = 2.685978684198
# EXACT_FEASIBLE: True
```

A configuration that passes this is feasible **as a matter of arithmetic fact, not of tolerance**. Two
things to note. `--n 27` is **required**, because it defaults to 26 and a count mismatch is reported as
`EXACT_FEASIBLE: False` (exit 1, so it is script-safe). And its slacks are **squared** (`d² - s²`), so they
are not comparable to `verify_pck.py`'s linear ones above; it also reads the full-precision `.json` rather
than the `.pck`.

## `.pck` format

Packomania's submission format, defined at
[packomania.com/hints.html](https://www.packomania.com/hints.html):

- line 1 = radius of the **largest** circle, as a bare number (no letters, no `=`);
- line 2 = author name(s), comma-separated if several;
- line 3 onward = one `x y r` per circle, whitespace-separated, **sorted by increasing radius**;
- the square container is fixed at **side 1, centred at (0,0)**, so coordinates live in `[-0.5, 0.5]²` and
  must be rescaled to fit. We emit 12 decimals, matching Packomania's own published `csqv` coordinates.

**Two conventions coexist in this repo:** `.pck` files are origin-centred as above, while the solver's
`.json` configs use the `[0, 1]²` corner convention (see `container.vertices`); convert by subtracting
0.5. `verify_pck.py --corner`/`--side` reads either.

## Requirements

Python 3 with **numpy** and **scipy** (`pip install -r requirements.txt`). scipy supplies both `linprog` and
`minimize`, so without it there is **no LP and no SLSQP**: `pack.py` degrades to an iterative
radius-shrinking heuristic that is measurably worse and returns no duals. Treat scipy as required.
`verify_pck.py` and `exact_check.py` need only the standard library.

## Reproduce the N=27 result from scratch

```bash
pip install -r requirements.txt                              # numpy + scipy
python3 solver/pack.py -n 27 --seed 1 --time 120 -o out27.json
```

That is the exact invocation behind the win: **seed 1, 120 s**. It has reproduced the stored configuration
bit-for-bit on two different machines, though at different points in the budget (the winning value first
appeared at 91s on one and 109s on the other), which is what you would expect from a wall-clock budget. It
is a stochastic search, so a re-run elsewhere is not pass/fail; see the caveat below.

## Reproduce the whole sweep (one command)

```bash
./reproduce.sh                           # full sweep N=2..100, then compare, then verify every packing
./reproduce.sh 20 30 60 4                # quick demo: N=20..30 @ 60s/N, 4 workers
NMIN=27 NMAX=27 TIME=120 ./reproduce.sh  # a single N
```

`reproduce.sh` writes to a fresh `repro/` directory (it does **not** clobber the committed reference
results), then independently verifies every emitted packing and fails loudly if any is infeasible. Full
N=2..100 takes ~25 min on a 10-core machine.

Or step by step:
```bash
OMP_NUM_THREADS=1 python3 solver/run_sweep.py --nmin 2 --nmax 100 --time 120 --workers 8 --out-dir repro
python3 solver/compare.py --results repro/results.csv --out repro/comparison.md
for f in repro/pck/csqv*.pck; do python3 solver/verify_pck.py "$f" >/dev/null || echo "INFEASIBLE: $f"; done
```

With no `--results`/`--out`, `compare.py` regenerates the committed `sota/ours/comparison.md` from the
committed `results.csv`, which is a check in itself. It rewrites that file **in place**, so confirm with
git rather than by eye:

```bash
python3 solver/compare.py && git diff --stat sota/ours/comparison.md   # expect no output: byte-identical
```

## Reproducibility notes

- **The artifacts are fixed and verifiable; the search is stochastic.** It is multi-start over random
  layouts under a *wall-clock* budget, so a given N's best depends on seed, budget and machine speed, and
  **the seed dominates**: at N=27, seed 1 at 120s found the record-beating 2.685978684198 while seed 7 at
  300s found a worse 2.683803 on 2.5x the budget. That does not weaken the win, since the saved file is
  strictly feasible and beats the record however it was found. To make a search more repeatable, vary the
  seed (`run_sweep.py --seeds`, or `pack.py --seed`) rather than only raising `--time`; N=97 below is a case
  where more time cannot help at all.
- **What is stored where.** The sweep's 98 usable outputs are in `sota/ours/pck/csqv<N>.pck` (12 dp,
  Packomania format), with no file for N=97. For N=27, `sota/ours/wins/` also holds the full float64 config
  with its seed and budget (`csqv27.seed1.json`) and the stored verifier output (`csqv27.verify.txt`). A
  fresh sweep additionally writes `<out-dir>/json/out<N>.json`.
- **Rounding to 12 dp is safe.** It costs 3.1e-13 of Σr and the packing stays strictly feasible; the
  slacks above are measured on the rounded `.pck` itself, not on the unrounded solution.

## Known issues

**The sweep produces no result at N=97**, so it covers 99 sizes and yields 98 packings: `results.csv` has a
blank row with `feasible=0` and there is no `pck/csqv97.pck`.

It is an **absorbing state**, not a hard instance. Replaying `search(97, seed=1, budget=120)`: the first
grid start refines to 532 coincident centre-pairs, so `repair()`'s single uniform scale factor hits 0 and
zeroes every radius. That config violates nothing and scores 0, and since `best_s` starts at `-1.0`, **zero
counts as an improvement** and becomes the incumbent. Every later iteration hops from it, but a hop moves
only a subset of the circles, so the unmoved ones stay stacked and one leftover zero distance re-zeroes
everything. Coincident pairs rose with every one of the run's 4 candidates.

So the budget bought one real attempt, not four, and n=97 is otherwise fine: a different grid start refines
cleanly to 5.121755, only 1.04% under the record. **More time on seed 1 cannot help** since the state is
absorbing; more seeds can, because each seed is an independent stream with its own first candidate.

Both checkers now catch it, fixed in the harness since `pack.py` is kept unmodified: `verify_pck.py` prints
`DEGENERATE` and exits 1 on any zero radius, and `run_sweep.py` discards such candidates so the N records
`feasible=0` with no `.pck` rather than a file that looks valid. The N=27 claim is unaffected.

## Layout

```text
solver/        pack.py container.py shape.py   # the AI-written solver, copied UNCHANGED
               exact_check.py  # zero-tolerance feasibility decision in exact rational arithmetic
               run_sweep.py    # parallel N-sweep -> pck + json + results.csv
               compare.py      # ours vs the Packomania records -> comparison.md
               verify_pck.py   # independent, pure-stdlib feasibility + Σr checker for any .pck
sota/          the SOTA comparison, both sides in one place:
  theirs/        packomania_csqv_records.csv     # the best-known records (N=1..100), + README
  ours/          results.csv  comparison.md  + README
                 wins/          # the record-beating N=27 result: csqv27.pck + full-precision json + verify
                 pck/           # the 98 usable packings (csqv27 also here); no N=97, see Known issues
                 chase/         # 12 near-misses re-run at 240s over seeds 1-4 (ties the record at 5)
writeup/       README.md + fig1..3.png   # narrative explainer of the N=27 result (+ make_figs.py to regen)
reproduce.sh  requirements.txt
```

## Solver Algorithm

`solver/pack.py` (+ `container.py`, `shape.py`, `exact_check.py`) is used **unchanged** as produced by an
automated program-search / self-improvement loop (an LLM-driven coding process), not hand-written for this
repo. `n` is a parameter throughout, so the same code runs at any N.

> **Stale internal references.** Being verbatim copies, these files' docstrings still address the
> originating experiment's layout: `tools/*.py` paths, a `bench/` tree (`bench/verify.py`,
> `bench/bench1/best.json`), `bench1`...`bench5` instances, `iter 4`/`iter 5` stages, and one log
> (`artifacts/iter3/speedup.log`). None exist here. Read them as provenance, not instructions; every
> runnable entry point is `solver/*.py`.

### What the N=27 win actually used

Read this first, because it says which of the mechanisms below matter to the record claim and which are
context. The win came from the **default path only**: multi-start joint SLSQP over `(x, y, r)`, exact-LP
radii, uniform subset hopping, the LP's contact-graph reduction on, Euclidean disks in the unit square, and
a dense SLSQP pair set. So the next four sections are the ones that produced it.

It did **not** use the dual-guided hopping (`--dual`), the trust-region QP (`--sparse`), or the polytope
joint-LP path. The stored config's `method` field reads `multi-start SLSQP + exact-LP radii + basin
hopping`, and its `container` / `shape` fields record `unit_square` / `ball`; since `method` differs on the
polytope path, the file rules that path out on its own. Only `--dual` and `--sparse` go unserialized, so
those two rest on `run_sweep.py` calling `search()` with its defaults.

### The exactly solvable inner layer

**Notation.** `c_i = (x_i, y_i)` and `r_i` are circle `i`'s centre and radius; `d_ij = |c_i - c_j|`;
`w_i` is the distance from `c_i` to the nearest wall; the objective is `Σr = Σ_i r_i`.

Maximizing `Σr` over centres *and* radii is a nonlinear program, but freezing the centres leaves

```text
maximize     Σ r_i
subject to   r_i + r_j  ≤  d_ij        for every pair (i, j)
             r_i        ≤  w_i         for every i
             r_i        ≥  0
```

a **pure linear program in `r`**, solved exactly and in milliseconds, with no gradient noise and no
step-size tuning. That is what lets the search treat the problem as a search over *centres*: for any layout
the best radii are one LP away.

One implementation caveat, small but worth knowing: `refine()` computes the LP radii each round and keeps
them only when they beat what SLSQP already had, otherwise it stops and `repair()` rescales SLSQP's radii
instead. SLSQP's raw sum is higher surprisingly often (13 of 24 trial starts at n=27) because its output can
be a hair infeasible, so the scored radii are not *always* the LP optimum for the final centres. The
discrepancy is immaterial in practice: 12 of those 13 were under 1e-12, i.e. below the `1e-12` shave
`repair()` applies anyway.

### The outer search

The centres are the hard part, since the landscape is full of near-degenerate local optima. Under a
wall-clock budget, the loop spends ~65% of its iterations hopping from the incumbent and the rest on fresh
starts, using:

- **joint SLSQP over all of `(x, y, r)`** with analytic constraint Jacobians, not just over the centres, so
  the local solver can trade radius against position in one step;
- **`refine()`**, alternating SLSQP with the exact radius LP for up to 3 rounds, keeping the LP's answer
  whenever it beats SLSQP's and stopping early when it does not;
- **subset hopping** that perturbs a random 12%, 25% or 45% of circles at one of three jump scales, also
  deflating their radii to 30%, rather than restarting from scratch. The code calls this basin hopping, but
  acceptance is strictly greedy (`if s > best_s`): there is no Metropolis criterion and no temperature, so
  it is closer to iterated local search;
- a **diverse start pool** of random placements and staggered-row grids, the structure good packings
  actually have.

### Innovation: an exact contact-graph reduction

The formulation has `n(n-1)/2` pair constraints, but a real packing's contact graph is planar, so it has at
most `3n - 6` contacts. At `n = 100` that is 4950 rows modelling ~270 real ones. **This reduction applies to
the radius LP only**; SLSQP on the default path still builds the full dense Jacobian, and only `--sparse`
(rejected below) tries to shrink that.

The reduction is **provable, not heuristic**. Since `r_i + r_j ≤ d_ij` and `r_j ≥ 0`, every `j` forces
`r_i ≤ d_ij`. Therefore

```text
u_i  :=  min( w_i,  min_{j ≠ i} d_ij )        is a valid bound on r_i, in every feasible config
drop pair (i, j)   when   d_ij  ≥  u_i + u_j
```

Impose `r_i ≤ u_i` as a variable bound (valid bounds never cut off the optimum) and the dropped pairs are
**implied by those two bounds**, so deleting them loses provably nothing. Because `u_i` is roughly a
nearest-neighbour distance, the surviving rows are a superset of the true contacts, concentrated in each
circle's local neighbourhood.

Three caveats the argument depends on: `u_i` is valid only for the **centres it was computed from**, so it
goes stale the moment SLSQP moves anything; only the optimal **value** is preserved, not the duals (dropped
rows are reported with `λ = 0`, which matters for the next section); and the bound and the drop rule are one
argument, so dropping rows without imposing `r ≤ u` lets the solver inflate radii straight through the
deleted constraints. `--self-test` checks the reduced LP against a naive all-pairs reference and measures
**a worst gap of 4.4e-16 while keeping as few as 5.9% of the rows**.

### Strict feasibility by construction, not by tolerance

Nothing is trusted until `repair()` makes it strictly feasible: centres are projected into the container,
then a **single uniform radius scale** is applied, `min(1, s)` where `s` is the largest factor satisfying
every pair and wall constraint at once, and the radii are shaved by a further `1e-12` (costing `n·1e-12` of
score, 2.7e-11 at n=27). Any candidate still violating a constraint by more than `1e-9` is dropped by the
search loop, costing that iteration.

Uniformity is what makes this cheap and also what makes it brittle: **one badly placed circle can zero the
whole configuration**, since `s = 0` scales every radius to 0. That is exactly how N=97 collapsed, see
[Known issues](#known-issues). Note too that this repair is float arithmetic; the tolerance-free guarantee
comes from `exact_check.py`, run above for N=27.

### Innovation: LP duals as a search signal

The radius LP is solved for its **duals** as well as its optimum (`lam` and `mu` in the code):

```text
λ_k  =  price of pair row k = (i, j);  zero unless that contact is tight
μ_i  =  price of the variable bound r_i ≤ u_i
Σ_j λ_ij  +  μ_i  =  1        for every i with r_i > 0
```

The identity is complementary slackness (the reduced cost of `r_i` vanishing), and it holds with `≥ 1` when
`r_i = 0`. Read as a sensitivity, `λ_k` is the gain in `Σr` per unit of extra room at contact `k`, which
requires a unique optimal dual to be a true derivative; this LP is often degenerate, so treat it as one
optimal dual vector rather than a gradient.

Because the bound is `r_i ≤ u_i` and not `r_i ≤ w_i`, `μ_i` is only a *boundary* price when `u_i = w_i`,
i.e. when the wall is what caps circle `i`. In that case `1 - μ_i` is the share of its radius set by
**neighbours** rather than the boundary, and that is the signal `perturb_dual()` aims with: sample contacts
with probability proportional to `λ` (without replacement, so proportional only for the first draw) and
move both ends, spending hops on the load-bearing contacts instead of on circles whose radius is
boundary-limited *at their current position*. Opt-in via `--dual`. No measurement in this repo shows it
beats uniform hopping, and the N=27 win did not use it.

### Self-validation

`python3 solver/pack.py --self-test` checks the machinery against facts rather than against itself: the
dual identity above, a zero duality gap, the contact-graph reduction against a naive all-pairs LP, and a
**proved optimum**. For `n = k²` axis-aligned squares of half-side `r` in the unit square,
`max Σr = √n / 2` exactly (Cauchy-Schwarz on `Σ 4r_i² ≤ 1`, attained by the `k × k` grid); the search is
asserted never to exceed it and to come within 1e-4. That runs at `k = 2, 3` only, so `n = 4` and `n = 9`,
not at sweep sizes.

### Beyond the record run

Everything below is implemented and exercised by `--self-test`, but **did not contribute to the N=27
result**. Skip it unless you are interested in the solver as a solver.

#### Generality: container and packed shape are both abstracted

In the *formulation*, the container enters in **exactly one place**: the wall rows `r_i ≤ F_k(c_i)`, where
`F_k(c_i)` is the largest radius wall `k` allows at `c_i` and `w_i = min_k F_k(c_i)`. Those are linear in `r`
for any convex container, so the inner LP survives verbatim when the unit square becomes a disk or a
triangle (`--container`, three are implemented). The *code* touches the container handle on 19 lines for
projection, sampling and bounding boxes; it is the LP structure that is untouched, not the call graph.

The packed *object* is abstracted the same way. A circle becomes a homothet `c_i + r_i·K` of a
centrally-symmetric convex body `K`, which enters in just two coefficient slots:

```text
pair distance   hypot(·)  ->  γ_K(·)          the gauge (Minkowski functional) of K
wall divisor    1         ->  h_K(a_k)        support value of K in unit wall normal a_k
```

Both stay linear in `r`, so the inner LP survives a change of object too (`--shape`, four are implemented).
From here on, read `d_ij` as `γ_K(c_i - c_j)`. The two axes do **not** compose: a non-ball shape asserts a
polyhedral container, so disks-in-a-disk works and squares-in-a-disk is refused.

**For polytopal `K` this gets strictly better.** Non-overlap `γ_K(c_i - c_j) ≥ r_i + r_j` is a
*disjunction* of linear constraints, holding as soon as one facet normal separates the pair. Fix each
pair's separating facet and the whole problem, **centres and radii together**, becomes one linear program.
Iterating that is a monotone ascent, not a heuristic: every LP-feasible point is *truly* feasible (one
separating facet suffices, since `γ_K` is the max over all of them), and the incoming configuration is
itself LP-feasible, so the optimum can only improve. Each step is a global optimum **within its
combinatorial cell**, which is not global optimality; re-selecting facets and re-solving converges to a
fixed point of that selection map. In the code SLSQP is skipped entirely on this path, and `repair()` still
brackets it as a guard, with a `1e-11` stopping tolerance. Coverage is thin: the only evidence here is two
self-test instances.

#### A path that is implemented but disabled

This is a note about a development decision, not a runtime mechanism: nothing in the solver tries
alternatives and discards them while packing.

The idea was to shrink SLSQP's dense Jacobian the way the LP's rows were shrunk, by optimizing only over
each circle's local contact neighbourhood inside a trust region that bounds how far centres may move. It is
implemented, reachable via **`--sparse`**, and **off by default because it is not faster**: the extra passes
needed to re-earn the movement the trust region gave up cost about what the skipped rows saved.

Benchmarking `refine()` at `n = 49 / 64 / 100`, as a ratio of dense time to sparse time (so above 1 means
sparse won):

| source of measurement | n=49 | n=64 | n=100 |
|---|---|---|---|
| the solver's own log, not included in this repo | 0.1 | 1.0 | 1.1 |
| re-measured here, seed 0, 3 repetitions | 0.43 | 1.26 | 1.06 |

So it is roughly break-even at large `n` and clearly worse at `n = 49`, with the `n = 64` figure varying by
seed. "Not faster" means no reliable win, not a uniform loss. Two caveats on the sparse path itself: its
safety argument bounds centre motion per *coordinate* while the row-dropping margin is Euclidean, so
correctness actually rests on the cutting-plane re-check that follows each solve plus `repair()`; and the
committed repo contains no benchmark script, so the second row above is not reproducible from the repo as
it stands. The LP-side reduction above is unaffected and stays on.
