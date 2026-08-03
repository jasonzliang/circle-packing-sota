# circle-packing-sota

An **evolved circle-packing solver**, extracted from a self-improvement (AI-Generating-Algorithms)
experiment and run across a range of N, compared to the authoritative **Packomania** records for:

> **Pack N variable-sized circles in a unit square so that no two overlap and all stay inside the square,
> maximizing the sum of the radii Σr**, the AlphaEvolve / ShinkaEvolve benchmark (Packomania's `csqv` table).

## Result

Across N the solver **reproduces 26 of the known records to within 4e-11** (worst 3.4e-11 at N=29, median
1.5e-11) and lands a fraction of a percent under the heavily-optimized larger records (that gap is our
compute budget, not the record's ceiling). It also found one **strictly-feasible packing that beats a
current Packomania best-known**: N=27, Σr = 2.685978684198 vs the listed 2.685350025228 (+6.3e-4). That
record is a long-standing 2011/12 entry (reference [1], D. W. Cantrell, sci.math forum), not one of the
recent AI-optimized entries (e.g. N=26 = 2.635983, credited to Haowei Lin [8] in July 2026, which we do
**not** beat). The always-current verdict is `sota/ours/comparison.md`, and **every claimed win is
independently re-verifiable from its `.pck` with `solver/verify_pck.py`.**

## Verify the N=27 result (30 seconds, no dependencies, no solver)

**The win does not depend on re-running the search.** The packing is a fixed file, and checking it is
pure arithmetic on 27 (x, y, r) triples. From a clean checkout:

```bash
python3 solver/verify_pck.py sota/ours/wins/csqv27.pck --record 2.685350025228
```

Expected output (exit code 0):

```text
circles (N)     : 27
sum of radii Σr : 2.685978684198
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
margin (+6.29e-4) is ~6×10⁸ times larger than the smallest of them, so this is not numerical noise.

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

A configuration that passes this is feasible **as a matter of arithmetic fact, not of tolerance**. Note it
reads the full-precision `.json`, while `verify_pck.py` reads `.pck` files.

## Reproduce the N=27 result from scratch

```bash
pip install -r requirements.txt                              # numpy + scipy
python3 solver/pack.py -n 27 --seed 1 --time 120 -o out27.json
```

That is the exact invocation behind the win: **seed 1, 120 s**. On the machine used here it reproduces the
stored configuration bit-for-bit, first reaching the winning value at ~109s of the 120s budget. It is a
stochastic search, though, so a re-run elsewhere is not pass/fail; see the caveat below.

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
committed `results.csv`, which is a check in itself: the output should be byte-identical to the repo's.

## Reproducibility notes

- **The artifacts are fixed and verifiable; the search is stochastic.** The solver is multi-start over
  random initial layouts under a *wall-clock* budget, so a given N's best is seed-, time- **and
  machine-dependent**: a slower or busier machine completes fewer restarts in the same 120s and may land
  in a worse local optimum. More time is not monotonically better either: at N=27, seed 1 (120s) found
  the record-beating 2.685978684198 while seed 7 (300s) found a worse 2.683803. None of this weakens the
  win: a packing is either valid or not, and the saved file is strictly feasible and exceeds the record
  however it was found. Use more `--time`/`--seeds` for a more repeatable search.
- **What is stored where.** All 99 packings are in `sota/ours/pck/csqv<N>.pck` (12 dp, Packomania format).
  For N=27, `sota/ours/wins/` also holds the full float64 config with its seed and budget
  (`csqv27.seed1.json`) and the stored verifier output (`csqv27.verify.txt`). A fresh sweep additionally
  writes `<out-dir>/json/out<N>.json`.
- **Rounding to 12 dp is safe.** It costs 3.1e-13 of Σr and the packing stays strictly feasible; the
  slacks above are measured on the rounded `.pck` itself, not on the unrounded solution.

## Layout

```text
solver/        pack.py container.py shape.py   # the evolved solver, copied UNCHANGED
               exact_check.py  # zero-tolerance feasibility decision in exact rational arithmetic
               run_sweep.py    # parallel N-sweep -> pck + json + results.csv
               compare.py      # ours vs the Packomania records -> comparison.md
               verify_pck.py   # independent, pure-stdlib feasibility + Σr checker for any .pck
sota/          the SOTA comparison, both sides in one place:
  theirs/        packomania_csqv_records.csv     # the best-known records (N=1..100), + README
  ours/          results.csv  comparison.md  + README
                 wins/          # the record-beating N=27 result: csqv27.pck + full-precision json + verify
                 pck/           # all 99 packings (complete set; csqv27 also here)
                 chase/         # 12 near-misses re-run at 240s over seeds 1-4 (ties the record at 5)
reproduce.sh  requirements.txt
```

## Solver Algorithm

`solver/pack.py` (+ `container.py`, `shape.py`, `exact_check.py`) is used **unchanged** as produced by an
automated program-search / self-improvement loop (an LLM-driven coding process), not hand-written for this
repo. `n` is a parameter throughout, so the same code runs at any N.

> **Stale internal references.** Being verbatim copies, these files' docstrings still address the
> originating experiment's layout: `tools/*.py`, `bench1`...`bench5`, `iter 4`/`iter 5`, and one log
> (`artifacts/iter3/speedup.log`). None exist here. Read them as provenance, not instructions; every
> runnable entry point is `solver/*.py`.

**Notation.** `c_i = (x_i, y_i)` and `r_i` are circle `i`'s centre and radius; `d_ij = |c_i - c_j|`;
`w_i` is the distance from `c_i` to the nearest wall; the objective is `Σr = Σ_i r_i`.

### The exactly solvable inner layer

Maximizing `Σr` over centres *and* radii is a nonlinear program, but freezing the centres leaves

```text
maximize     Σ r_i
subject to   r_i + r_j  ≤  d_ij        for every pair (i, j)
             r_i        ≤  w_i         for every i
             r_i        ≥  0
```

a **pure linear program in `r`**, solved exactly and in milliseconds, with no gradient noise and no
step-size tuning. Every candidate gets that LP polish before it is scored, which reduces the whole problem
to a search over *centres* alone.

### The outer search

The centres are the hard part, since the landscape is full of near-degenerate local optima. Under a
wall-clock budget, the loop spends ~65% of its iterations hopping from the incumbent and the rest on fresh
starts, using:

- **joint SLSQP over all of `(x, y, r)`** with analytic constraint Jacobians, not just over the centres, so
  the local solver can trade radius against position in one step;
- **`refine()`**, alternating SLSQP with the exact radius LP for up to 3 rounds, keeping the LP's answer
  whenever it beats SLSQP's and stopping early when it does not;
- **basin hopping** that perturbs a random *subset* of circles (12%, 25% or 45%, at one of three jump
  scales) rather than restarting from scratch;
- a **diverse start pool** of random placements and staggered-row grids, the structure good packings
  actually have.

### Innovation: an exact contact-graph reduction

The formulation has `n(n-1)/2` pair constraints, but a real packing's contact graph is **sparse**, roughly
`3n` contacts, since it is essentially planar. At `n = 100` that is 4950 rows modelling ~300 real ones, and
it is the dense Jacobian, not the geometry, that makes large `n` slow.

The reduction is **provable, not heuristic**. Since `r_i + r_j ≤ d_ij` and `r_j ≥ 0`, every `j` forces
`r_i ≤ d_ij`. Therefore

```text
u_i  :=  min( w_i,  min_{j ≠ i} d_ij )        is a valid bound on r_i, in every feasible config
drop pair (i, j)   when   d_ij  ≥  u_i + u_j
```

Impose `r_i ≤ u_i` as a variable bound (valid bounds never cut off the optimum) and the dropped pairs are
**implied by those two bounds**, so deleting them loses provably nothing. Because `u_i` is roughly a
nearest-neighbour distance, the surviving rows are exactly the local neighbourhood.

The bound and the drop rule are one argument and must be used together: drop rows without imposing
`r ≤ u` and the solver inflates radii straight through the deleted constraints. `--self-test` checks the
reduced LP against a naive all-pairs reference and measures **a worst gap of 4.4e-16 while keeping as few
as 5.9% of the rows**.

### Innovation: LP duals as a search signal

The radius LP is solved for its **duals** as well as its optimum:

```text
λ_k  =  ∂(Σr) / ∂d_ij      price of pair constraint k;  zero unless that contact is tight
μ_i  =  ∂(Σr) / ∂w_i       price of circle i's wall bound
Σ_j λ_ij  +  μ_i  =  1     LP duality, for every i with r_i > 0
```

(`lam` and `mu` in the code.) That identity makes `1 - μ_i` precisely the share of circle `i`'s radius
limited by its **neighbours** rather than by the boundary. `perturb_dual()` uses it to aim: sample
contacts with probability proportional to `λ` and move both ends, targeting the load-bearing part of the
contact graph instead of wasting hops on circles a wall already caps, which no amount of sliding improves.
Opt-in via `--dual`.

### Strict feasibility by construction, not by tolerance

Nothing is trusted until `repair()` makes it strictly feasible: centres are projected into the container,
then a **single uniform radius scale** `s ≤ 1` is taken as the largest value satisfying every pair and wall
constraint at once, and the radii are shaved by a further `1e-12`. That costs ~`1e-11` of score and buys
exactness. Any configuration still violating a constraint by more than `1e-9` is discarded, not reported.

### Generality: container and packed shape are both abstracted

The container enters in **exactly one place**, the wall rows `r_i ≤ F_k(c_i)`, linear in `r` for any convex
container. The inner LP therefore survives verbatim when the unit square becomes a disk or a triangle.

The packed *object* is abstracted the same way. A circle becomes a homothet `c_i + r_i·K` of a
centrally-symmetric convex body `K`, which enters in just two coefficient slots:

```text
pair distance   hypot(·)  ->  γ_K(·)          the gauge (Minkowski functional) of K
wall divisor    1         ->  h_K(a_k)        the support value of K in wall normal a_k
```

Both stay linear in `r`, so the inner LP survives a change of object too.

**For polytopal `K` this gets strictly better.** Non-overlap `γ_K(c_i - c_j) ≥ r_i + r_j` is a
*disjunction* of linear constraints, holding as soon as one facet normal separates the pair. Fix each
pair's separating facet and the whole problem, **centres and radii together**, becomes one linear program.
Iterating that is a genuine ascent, not a heuristic, for two reasons: every LP-feasible point is *truly*
feasible (one separating facet suffices, since `γ_K` is the max over all of them), and the incoming
configuration is itself LP-feasible, so the optimum can only improve. Each step is thus a global optimum
within its combinatorial cell, needing no repair and no tolerance argument. A nonsmooth gauge then costs
the search nothing, since the kinks that break SLSQP are exactly the combinatorial choices this makes
explicit.

### What was tried and rejected

A trust-region QP restricted to the contact neighbourhood is implemented (`--sparse`) and is *correct*,
but it is **not faster**: the extra passes it needs to re-earn the movement it gave up cost as much as the
rows it saved. The source reports 0.1x / 1.0x / 1.1x at `n = 49 / 64 / 100` from a log not included in this
repo; re-measuring `refine()` here gives **0.43x / 1.26x / 1.06x** at the same sizes. Different machine,
same verdict, so it is off by default and the source explicitly declines to call it a speedup. The LP-side
reduction above, which is exact and needs no trust region, stays on.

### Self-validation

`python3 solver/pack.py --self-test` checks the machinery against facts rather than against itself: the
dual identity above, a zero duality gap, the contact-graph reduction against a naive all-pairs LP, and, at
full instance size, a **proved optimum**. For `n = k²` axis-aligned squares in the unit square,
`max Σr = √n / 2` exactly (Cauchy-Schwarz on the area bound, attained by the `k × k` grid), and the search
is asserted to reach it and never exceed it.

### What the N=27 win actually used

The default path only: **multi-start joint SLSQP + exact-LP radii + uniform basin hopping**, with the
exact contact-graph reduction on, Euclidean disks in the unit square, dense pair set. Not the dual-guided
hopping (`--dual`), not the trust-region QP (`--sparse`), and not the polytope joint-LP path. The stored
config records this in `sota/ours/wins/csqv27.seed1.json` under `method`.

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

Python 3 with **numpy** and **scipy** (`pip install -r requirements.txt`). scipy enables the SLSQP path;
`pack.py` falls back to an LP-only path without it. `verify_pck.py` needs only the standard library.
