# solver-nietzsche-sm — a second, stronger AI-written solver

A **distinct** AI-written circle-packing solver from a _different_
self-improvement run than the top-level
[`solver-nietzsche-sm-radical-v6-n27/`](../solver-nietzsche-sm-radical-v6-n27/).
Where `solver-nietzsche-sm-radical-v6-n27/` swept N=2..100 and won at **N=27**
but under-searched at large N (it falls short of the record at, e.g., N=54),
this solver **beats the Packomania `csqv` best-known exactly there**: **N=54
(+9.0e-4)** and **N=55 (+2.3e-5)**.

## Provenance

Evolved by the SI-v2 self-improvement loop (not hand-written):

- **Run:** `circle-packing-n54-240s-20260809`, values `nietzsche-sm-radical-v6`,
  **stopped at iteration 10**.
- **Source commit:** `06a8c0a` (iteration-10 solver). Verified Σr(n=54) =
  **3.8426357949378** vs record 3.841733103296 — feasible at zero tolerance
  (independent recheck).
- Different lineage and codebase from `solver-nietzsche-sm-radical-v6-n27/`
  (which uses `pack.py`/`container.py`/`shape.py` and a `.pck` sweep harness);
  this one is the n-generic **SLP-KKT pipeline**.

## Entrypoint / contract

```python
import solve
circles = solve.solve(n, seconds=None, seed=0)   # -> list of (x, y, r), n circles in the unit square
```

`solve.py` re-exports the record-beating `pipeline.solve`. Method: broad
multistart → exact LP radii → **SLP-KKT primal inner-linearization polish**
(feasible-by-construction, monotone, converges to a jammed KKT point) →
threshold-accepting basin hopping over KKT points. Needs `numpy` + `scipy`
(top-level `requirements.txt`; the polish calls HiGHS via
`scipy.optimize.linprog`).

## Files (keep FLAT — modules import each other as siblings)

`pipeline.py` (solver), `slp.py` (SLP-KKT), `packlib.py` (LP radii + core),
`endgame.py`, `broad.py`, `baseline.py`, `adversary.py`, `split_sweep.py`,
`walkcurve.py`, `verify.py` (independent feasibility+Σr checker), `solve.py`
(entrypoint shim). (Solver code only — the record packings live under
`sota/nietzsche-sm-radical-v6-n54/`.)

## Status & how it would slot into the comparison

This directory is the **solver only**. A full N-sweep of it (analogous to the
top-level `reproduce.sh` sweep of `solver-nietzsche-sm-radical-v6-n27/`) has
**not** been run here yet; when it is, its results/packings belong in a sibling
`sota/nietzsche-sm-radical-v6-n54/` so both solvers are compared against the
same `sota/packomania/` records.

**Caveats (do not publish without these):** the N=54/55 beats are feasible at
zero tolerance in _double precision_; a submission-grade claim wants an
exact-arithmetic feasibility check (like the root repo's N=27) and a re-verified
`.pck`. "Best-known" ≠ proven optimum.

## Algorithm

### The shared inner layer: radii are an exact LP for fixed centres

**Notation.** `c_i = (x_i, y_i)`, `r_i` are circle `i`'s centre and radius;
`d_ij = |c_i − c_j|`; `w_i` is the distance from `c_i` to the nearest wall; the
objective is `Σr = Σ_i r_i`.

Maximising `Σr` over centres _and_ radii is a nonlinear program, but freezing
the centres leaves

```text
maximize     Σ r_i
subject to   r_i + r_j  ≤  d_ij       for every pair (i, j)
             r_i        ≤  w_i        for every i
             r_i        ≥  0
```

a **pure linear program in `r`** — exact, no gradient noise, no step-size
tuning. Both solvers implement it with scipy/HiGHS and fall back to a numpy-only
monotone fixed point when scipy is absent, keeping whichever is better, so scipy
is an accelerator, not a hard dependency (n54: `packlib.max_radii`; n27:
`pack.lp_solve`). n27 additionally solves it for its **duals** and applies a
**provable contact-graph reduction** (both in the
[n27 solver's algorithm](../solver-nietzsche-sm-radical-v6-n27/README.md#algorithm)).

### The updated solver (`solver-nietzsche-sm-radical-v6-n54/`) — the 21 live wins

`solve(n, seconds, seed)` runs one **broad → endgame** chain (default split ≈
25% broad / 75% endgame, measured; `pipeline.py`). Every configuration either
stage emits is strictly feasible by construction.

**broad — multistart to a good funnel (`broad.py`).** Waves of _construct → Adam
→ LP screen → refine → SLP-KKT polish_, ranked by the **converged KKT value**
rather than by how far a penalty descent got:

1. **Topology-diverse construction** — six start families (uniform, jittered
   grid, hex rows, size-graded greedy, rotated hex lattice, random row
   partition), because the optimum is genuinely unequal-radius and different
   constructions land in different contact topologies.
2. **Batched quadratic-penalty Adam** over all `3n` variables of all `B` starts
   at once (`packlib.adam_run`, shape `(3, B, n)`), a coarse funnel-finder.
3. **LP screen** — exact-LP radii for each start; keep a top slice, refine those
   with more Adam.
4. **SLP-KKT polish of every kept start** to a true local optimum, then rank by
   that value. The screen is _audited_: a random sample of non-elite starts is
   polished too and the Spearman correlation between screen and KKT value is
   reported, so a misleading screen would be caught.

**SLP — the feasibility-preserving centre optimiser (`slp.py`).** The key
insight: radii are already an exact LP for fixed centres, and the centres can
**join** that LP because the only nonlinear constraint is a norm, and a norm is
convex, so its first-order expansion is a **global under-estimator**:

```text
‖a + s‖  ≥  ‖a‖ + ê·s,        ê = a / ‖a‖
```

Taking `a = c_i − c_j` and `s = δc_i − δc_j` at the current point turns each
non-overlap constraint into one linear cut that **implies** (not approximates)
the true constraint:

```text
r_i + r_j − ê·(δc_i − δc_j)  ≤  d_ij
```

The box rows are already exactly linear, so with variables `[δx, δy, r]` and a
trust box `|δx|,|δy| ≤ δ`, the LP is an **inner (restricted) model** of the true
problem. Three consequences: every LP solution is **truly feasible** (no repair
can eat the gain); `δc = 0` is feasible, so the step is **monotone** by
construction; and at a fixed point the linearisation is first-order exact, so it
lands on a **KKT point** — an actually jammed packing — in ~1 s. `δ` only bounds
how far the model is trusted and is shrunk geometrically to squeeze out the last
digits. Warm entry: `slp_polish(x, y, r)`.

**endgame — threshold-accepting basin hopping over KKT points (`endgame.py`).**
The hop loop compares local optima to local optima:

```text
perturb incumbent  →  LP radii  →  SLP-polish to KKT  →  accept-if-better
```

- **Coherent moves** (jitter, region shake, affine stretch/rotate, swirl,
  recluster) — displacing a single circle far in a jammed packing forfeits its
  whole radius, so every move deforms many circles together instead of tearing a
  hole. Because SLP is monotone and feasible from any start, it absorbs the
  perturbation with no Adam needed.
- **Threshold-accepting walk** (Dueck & Scheuer): the walker may step downhill
  by up to `T`, where `T` cycles geometrically from `T_HI = 8e-4` down to
  `T_LO = 2e-5` and resets, so exploration stays alive for the whole budget
  while the band is wide enough to reach neighbouring optima but never to buy
  out of the good basin family. A separate incumbent **`best` only ever rises**,
  and the walker is teleported back to it if it drifts more than
  `MAX_DRIFT = 3e-3` below.

Warm entry: `endgame(x, y, r, seconds)`. Both `broad_frac` and `t_hi` are
exposed but were measured to be flat-or-negative to tune (see the `pipeline.py`
/ `endgame.py` docstrings); the solver ships at the defaults.

**Repair.** `packlib.repair` guarantees strict feasibility by shrinking radii
only (centres untouched), capping each `r_i` at `w_i − 1e-12` and halving any
residual pair overlap — a per-circle shrink, so a single bad circle cannot zero
the whole configuration.
