# circle-packing-sota

An **evolved circle-packing solver**, extracted from a self-improvement (AI-Generating-Algorithms)
experiment and run across a range of N, compared to the authoritative **Packomania** records for:

> **Pack N variable-sized circles in a unit square so that no two overlap and all stay inside the square,
> maximizing the sum of the radii Σr** — the AlphaEvolve / ShinkaEvolve benchmark (Packomania's `csqv` table).

## Result in one paragraph

Across N the solver **reproduces the known optima to ~1e-11 on small/mid N** and lands a fraction of a
percent under the heavily-optimized larger records (that gap is our compute budget, not the record's
ceiling). It also found at least one **strictly-feasible packing that beats a current Packomania
best-known** — N=27, Σr = 2.685978684198 vs the listed 2.685350025228 (+6.3e-4). That record is a
long-standing 2011/12 entry (reference [1], D. W. Cantrell, sci.math forum), not one of the recent
AI-optimized entries (e.g. N=26 = 2.635983, credited to Haowei Lin [8] in July 2026, which we do
**not** beat). The authoritative,
always-current verdict is `sota/ours/comparison.md`; **every claimed win is independently re-verifiable
from its `.pck` with `solver/verify_pck.py` (see below).**

## Reproduce (one command)

```bash
pip install -r requirements.txt          # numpy + scipy
./reproduce.sh                           # full sweep N=2..100, then compare, then verify every packing
./reproduce.sh 20 30 60 4                # quick demo: N=20..30 @ 60s/N, 4 workers
NMIN=27 NMAX=27 TIME=120 ./reproduce.sh  # a single N
```

`reproduce.sh` runs the sweep, writes `sota/ours/comparison.md`, and **independently verifies every emitted
packing** (fails loudly if any is infeasible). Full N=2..100 takes ~25 min on a 10-core machine.

Or step by step:
```bash
OMP_NUM_THREADS=1 python3 solver/run_sweep.py --nmin 2 --nmax 100 --time 120 --workers 8 --out-dir results
python3 solver/compare.py                                   # -> sota/ours/comparison.md
python3 solver/verify_pck.py sota/ours/pck/csqv27.pck --record 2.685350025228   # independent check
```

## Independent verification (the important part)

`solver/verify_pck.py` is **pure standard library and shares no code with the solver**, so it is a genuine
independent check of any packing — ours or anyone else's. It recomputes Σr and confirms strict
feasibility (exactly N circles, all inside the square, no overlaps) straight from the coordinates:

```bash
python3 solver/verify_pck.py sota/ours/pck/csqv27.pck --record 2.685350025228
# ... STRICTLY FEASIBLE (tol 1e-09): True ;  Δ (ours-record): +6.287e-04 -> BEATS the record ;  exit 0
```

## Reproducibility notes (read before trusting a "win")

- **The artifacts are fixed and verifiable; the search is stochastic.** The solver is multi-start over
  random initial layouts under a wall-clock budget, so a given N's best is **seed- and time-dependent**.
  Example: at N=27, seed 1 (120s) found the record-beating 2.685978684198, while seed 7 (300s) found a
  worse local optimum (2.683803). This does **not** weaken the win: a packing is valid or not, and the
  saved configuration is a fixed artifact that is strictly feasible and exceeds the record. Re-running the
  sweep reproduces the overall landscape (ties on easy N, small gaps on hard N) and *can* re-find a given
  win, but is not guaranteed to on one seed.
- **Every result is saved twice:** `sota/ours/pck/csqv<N>.pck` (12 dp, Packomania format) and
  `sota/ours/json/out<N>.json` (full float64, plus the seed and time budget that produced it) — so a win is
  preserved exactly and its provenance is recorded.
- **Determinism caveat:** because the budget is wall-clock, a slower or busier machine does fewer restarts
  in the same seconds and may land slightly lower. Give more `--time` (or more `--seeds`) for a stronger,
  more repeatable result. The `.pck`/`.json` artifacts remain valid and verifiable regardless.

## Layout

```
solver/        pack.py container.py shape.py exact_check.py   # the evolved solver, copied UNCHANGED
               run_sweep.py    # parallel N-sweep -> pck + json + results.csv
               compare.py      # ours vs the Packomania records -> comparison.md
               verify_pck.py   # independent, pure-stdlib feasibility + Σr checker for any .pck
sota/          the SOTA comparison, both sides in one place:
  theirs/        packomania_csqv_records.csv     # the best-known records (N=1..100), + README
  ours/          results.csv  comparison.md  pck/csqv<N>.pck (csqv27 = the N=27 win)  + README
                 chase/         # supplementary harder re-runs of the closest near-misses
reproduce.sh  requirements.txt
email_draft.md   # a drafted submission email to Packomania's maintainer (git-ignored, local only)
```

## Solver provenance

`solver/pack.py` (+ `container.py`, `shape.py`) is used **unchanged** as produced by an automated
program-search / self-improvement process (an LLM-driven coding loop); it is not hand-written for this
repo. The optimizer:

1. **holds the circle centres fixed and solves an exact linear program for the radii** — Σr is linear in r
   for a fixed layout (each radius ≤ its distance to the four walls, and for every pair r_i + r_j ≤ the
   centre distance), so the best radii for any layout are found exactly and instantly;
2. wraps that in a joint **SLSQP** search over the centres (analytic Jacobians, diverse multi-start) with
   **basin hopping** to escape local optima;
3. finishes with an exact **uniform-radius-scaling repair**, so every emitted config is strictly feasible
   in exact arithmetic. `n` is a parameter throughout, so the same code runs unchanged at any N.

The solver code was itself written with substantial help from AI (the LLM-driven self-improvement loop).

## `.pck` format

Packomania format (`hints.html`): line 1 = largest radius, line 2 = author, then one `x y r` per circle
sorted by increasing radius. Coordinates are placed in a **unit-side square centred at the origin**
(`[-0.5, 0.5]²`); `verify_pck.py --corner`/`--side` handle other conventions.

## Requirements

Python 3 with **numpy** and **scipy** (`pip install -r requirements.txt`). scipy enables the SLSQP path;
`pack.py` falls back to an LP-only path without it. `verify_pck.py` needs only the standard library.
