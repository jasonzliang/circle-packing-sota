# circle-packing-sota

An **evolved circle-packing solver**, extracted from a self-improvement (AI-Generating-Algorithms)
experiment and run across a range of N, compared to the authoritative **Packomania** records for:

> **Pack N variable-sized circles in a unit square so that no two overlap and all stay inside the square,
> maximizing the sum of the radii Σr**, the AlphaEvolve / ShinkaEvolve benchmark (Packomania's `csqv` table).

## Result in one paragraph

Across N the solver **reproduces the known optima to ~1e-11 on small/mid N** and lands a fraction of a
percent under the heavily-optimized larger records (that gap is our compute budget, not the record's
ceiling). It also found at least one **strictly-feasible packing that beats a current Packomania
best-known**: N=27, Σr = 2.685978684198 vs the listed 2.685350025228 (+6.3e-4). That record is a
long-standing 2011/12 entry (reference [1], D. W. Cantrell, sci.math forum), not one of the recent
AI-optimized entries (e.g. N=26 = 2.635983, credited to Haowei Lin [8] in July 2026, which we do
**not** beat). The authoritative, always-current verdict is `sota/ours/comparison.md`; **every claimed win
is independently re-verifiable from its `.pck` with `solver/verify_pck.py` (see below).**

## Verify the N=27 result (30 seconds, no dependencies, no solver)

**The win does not depend on re-running the search.** The packing is a fixed file, and checking it is
pure arithmetic on 27 (x, y, r) triples. From a clean checkout:

```bash
python3 solver/verify_pck.py sota/ours/wins/csqv27.pck --record 2.685350025228
```

Expected output (exit code 0):

```
circles (N)     : 27
sum of radii Σr : 2.685978684198
min wall slack  : 1.000e-12  (>= 0 => all circles inside the square)
min pair slack  : 7.198e-13  (>= 0 => no two circles overlap)
STRICTLY FEASIBLE (tol 1e-09): True
record          : 2.685350025228
Δ (ours-record) : +6.287e-04   -> BEATS the record
```

`verify_pck.py` is **pure standard library and shares no code with the solver**, so it is a genuine
independent check of our packings or anyone else's. It re-reads the coordinates and confirms from
scratch that there are exactly N circles, all inside the square, none overlapping, and what Σr really
sums to. Exits non-zero on any failure, so it works in a script. A copy of the output above is stored in
[`sota/ours/wins/csqv27.verify.txt`](sota/ours/wins/csqv27.verify.txt) to diff against.

The slacks are small but **positive** (strictly feasible, not feasible-within-tolerance), and the win
margin (+6.29e-4) is ~6×10⁸ times larger than the smallest of them, so this is not numerical noise.

## Reproduce the N=27 result from scratch

```bash
pip install -r requirements.txt                              # numpy + scipy
python3 solver/pack.py -n 27 --seed 1 --time 120 -o out27.json
```

That is the exact invocation behind the win: **seed 1, 120 s**. It is a stochastic search, so a re-run is
not pass/fail; see the caveat below.

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

With no `--results`/`--out`, `compare.py` instead regenerates the committed `sota/ours/comparison.md`
from the committed `sota/ours/results.csv`, a useful check in itself, since it should come out
byte-identical to what is in the repo.

## Reproducibility notes (read before trusting a "win")

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

```
solver/        pack.py container.py shape.py exact_check.py   # the evolved solver, copied UNCHANGED
               run_sweep.py    # parallel N-sweep -> pck + json + results.csv
               compare.py      # ours vs the Packomania records -> comparison.md
               verify_pck.py   # independent, pure-stdlib feasibility + Σr checker for any .pck
sota/          the SOTA comparison, both sides in one place:
  theirs/        packomania_csqv_records.csv     # the best-known records (N=1..100), + README
  ours/          results.csv  comparison.md  + README
                 wins/          # the record-beating N=27 result: csqv27.pck + full-precision json + verify
                 pck/           # all 99 packings (complete set; csqv27 also here)
                 chase/         # supplementary harder re-runs of the closest near-misses
reproduce.sh  requirements.txt
email_draft.md   # a drafted submission email to Packomania's maintainer (git-ignored, local only)
```

## Solver provenance

`solver/pack.py` (+ `container.py`, `shape.py`) is used **unchanged** as produced by an automated
program-search / self-improvement process (an LLM-driven coding loop); it is not hand-written for this
repo. The optimizer:

1. **holds the circle centres fixed and solves an exact linear program for the radii**: Σr is linear in r
   for a fixed layout (each radius ≤ its distance to the four walls, and for every pair r_i + r_j ≤ the
   centre distance), so the best radii for any layout are found exactly and instantly;
2. wraps that in a joint **SLSQP** search over the centres (analytic Jacobians, diverse multi-start) with
   **basin hopping** to escape local optima;
3. finishes with an exact **uniform-radius-scaling repair**, so every emitted config is strictly feasible
   in exact arithmetic. `n` is a parameter throughout, so the same code runs unchanged at any N.

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
