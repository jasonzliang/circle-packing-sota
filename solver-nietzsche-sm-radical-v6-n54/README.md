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
