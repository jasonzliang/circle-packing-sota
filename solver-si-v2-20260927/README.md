# solver-si-v2-20260927: a portfolio of fifteen AI-evolved circle-packing solvers, and the sweep that runs them

This directory holds the solvers behind [`sota/si-v2-20260927/`](../sota/si-v2-20260927/): fifteen
`solver.py` files, each written by an LLM agent inside one run of the self-improvement-v2 (SI-v2)
`circle-packing` mission, copied **byte for byte** (sha256 in `portfolio.json`), plus the tooling that
sweeps them over N = 1..100 with a budget that grows with N, re-validates every packing it gets back,
and keeps the best strictly feasible packing per N.

Nothing here was written by hand except the sweep tooling. The solvers were extracted unchanged from
the run workspaces; the only thing the sweep changes at run time is each solver's *CPU allowance*
(a module-level constant such as `CPU_BUDGET`, set after import), so a solver that gave itself 100
CPU-seconds inside the mission can be handed 20 minutes here.

## Provenance

| key | origin run (SI-v2 `results/`) | values arm | brief | held-out endpoint | bytes |
|---|---|---|---|---:|---:|
| `tv14pf6` | jason-mbp `_archive/circle-packing/2026-09-15_transcendence-v14-probefix-x10/transcendence-v14-probefix-6` | universalism-v14 (`81dc2151`) | `4a1cda44` | 6.744 (aws warm re-score 6.753) | 58,346 |
| `tv16ob6` | jason-mbp `_archive/circle-packing/2026-09-15_transcendence-v16-oldbrief-x10/transcendence-v16-oldbrief-6` | transcendence-v16 (`f67e211e`) | `39f43f76` | void (backstopped; aws re-score 6.592) | 80,674 |
| `none7` | aws `circle-packing/2026-09-15_circle-6arm-v14-probefix-x10/none-7` | none-v14 (`c16b7ffb`) | `4a1cda44` | 6.355 | 80,975 |
| `tv14ob9` | jason-mbp `circle-packing/2026-09-15_transcendence-v14-oldbrief-x10/transcendence-v14-9` | universalism-v14 (`81dc2151`) | `39f43f76` | 6.287 (DEV 6.583) | 39,288 |
| `smv16_10` | aws `circle-packing/2026-08-26_circle_smframe_4x10_15it30m/explore-sm-v16-10` | explore-sm-v16 | `39f43f76` | DEV 6.905 | 82,297 |
| `deon6` | aws `circle-packing/2026-08-22_circle-register-x10-15it30m/explore-deon-6` | explore (deontic register) | `39f43f76` | 6.465 | 122,866 |
| `nalt6` | aws `circle-packing/2026-08-24_circle-figures-x10-15it30m/nietzsche-alt-6` | nietzsche-alt | `39f43f76` | n=94 record holder | 65,165 |
| `smv16_6` | jason-mbp `_archive/circle-packing/2026-08-26_circle-smframe-4x10-15it30m/explore-sm-v16-6` | explore-sm-v16 | `39f43f76` | n=80 record holder | 62,357 |
| `cvirt7` | jason-mbp `_archive/circle-packing/2026-08-22_circle-register-x10-15it30m/conservative-virt-7` | conservative (virtue register) | `39f43f76` | n=82 record holder | 65,552 |
| `anchor6` | jason-mbp `_archive/circle-packing/2026-08-19_circle-bulletcount-x10-12it30m/anchor-6` | anchor (bullet-count study) | `39f43f76` | n=96 (not reproduced) | 47,691 |
| `anchor1` | jason-mbp `_archive/circle-packing/2026-08-19_circle-bulletcount-x10-12it30m/anchor-1` | anchor | `39f43f76` | n=98 (not reproduced) | 99,940 |
| `b6x2_1` | jason-mbp `_archive/circle-packing/2026-08-19_circle-bulletcount-x10-12it30m/6bullets-2xlength-1` | 6 bullets, 2x length | `39f43f76` | n=100 record holder | 142,041 |
| `b9_5` | jason-mbp `_archive/circle-packing/2026-08-19_circle-bulletcount-x10-12it30m-b/9bullets-5` | 9 bullets | `39f43f76` | n=85 record holder | 94,981 |
| `plain1` | aws `circle-packing/2026-08-23_circle-plain-b/explore-plain-1` | explore (plain) | `39f43f76` | n=83, 89 record holder | 114,749 |
| `none3` | aws `circle-packing/2026-09-15_circle-6arm-v14-probefix-x10/none-3` | none-v14 (`c16b7ffb`) | `4a1cda44` | n=99 record holder | 60,338 |

"held-out endpoint" is the mission's score of that run's final solver on the 38 even sizes it never
saw (mean capped digits of relative gap to the frozen record table, 60 CPU-s per size, max 7); the
first four are the top four of every SI-v2 circle-packing run by that number. The remaining eleven
were included because each produced the programme's best packing at one even size during its run.
Values arms (`none`, `explore`, `conservative`, `universalism-v14`, `transcendence-v16`, ...) are the
value-statement conditions of the SI-v2 experiment the run belonged to; the md5 identifies the exact
text. `run_record` in `portfolio.json` carries the rollup row where one exists.

## Entrypoint and contract (the SI-v2 mission's)

Every solver exposes

```python
solve(evaluate, meter, rng, targets)          # some also accept cpu_allowance= / cpu_budget=
```

* `evaluate(n, packing) -> (feasible, sum_r)`: `packing` is an `(n, 3)` array of `(x, y, r)` in the
  **centred** unit square `[-0.5, 0.5]^2`, or a batch `(B, n, 3)`. It is the only sanctioned way to
  test a candidate; the harness privately tracks the best strictly feasible packing per n
  (`scorer/harness.py`, `Evaluator`), and that tracked best is what the sweep harvests. Nothing the
  solver *returns* is trusted.
* `meter`: read-only evaluation budget (`.left()`, `.used`, `.budget`).
* `rng`: `numpy.random.RandomState(seed)`, the only randomness source.
* `targets`: the list of n to solve (the sweep passes one n per process).
* The solver reads its warm starts itself from `bench/packs/csqv<n>.pck` (and neighbours n±k) and
  the target table from `bench/records.json`, both relative to the working directory; the sweep builds
  that layout in a throwaway workspace per job.
* Feasibility (`scorer/harness.py:validate`): exactly n circles, all finite, every `r > 0`, wall
  slack and pairwise slack `>= -1e-9`. The sweep additionally requires slack `>= 0` (zero tolerance)
  before a packing can be published.

`scorer/` is the mission's frozen scorer, vendored unchanged (`harness.py`, `verify.py`; pure
stdlib + numpy). It is what the solvers were evolved against and what the tools import.

## Algorithms (one line each)

All fifteen share one idea: for fixed centres the radii are an exact LP; centres move by
trust-region sequential linear programming (SLP) or a penalty method; a hopping/perturbation layer
escapes local optima; warm starts come from the stored census and from neighbouring sizes.

* `tv14pf6`: L-BFGS-B penalty explorer (mu ramp) + sparse HiGHS SLP finisher + exact radius LP;
  cross-size transfer from n±4, cold grid/random starts, threshold-accepting hops with inflate/shake/
  ruin/dilate moves; time split by remaining prize; repairs to slack >= +1e-13.
* `tv16ob6`: pure trust-region SLP (linprog + sparse LU) with stall stop; structure transfer, graft
  and crossover from donor packs; rich move bank; allocation by measured digits per second.
* `none7`: SLP (highs-ipm) to the LP optimum; hex-lattice starts over spacing and rotation, row
  families, donors n±6 with crossover, randomised trust radius, threshold-accepting hops.
* `tv14ob9`: no LP; multistart L-BFGS-B quadratic penalty with an equal-radius homotopy annealed to
  zero, three lattice column counts, neighbour transfer, shake bursts.
* `smv16_10`, `smv16_6`: SLP over a convex restriction + exact repair, basin hopping.
* `deon6`: SLP primary + SLSQP fallback, teleport hopping, worst-digits-first allocator.
* `nalt6`: SLP with ticketed excursions (quantised time slices) and half-space symmetry moves.
* `cvirt7`: SLP with margin-aware trust region and k-spot kicks.
* `anchor6`, `anchor1`: SLP + symmetry splicing (D4 maps, Chebyshev cuts), Apollonius hole finding.
* `b6x2_1`, `b9_5`: SLP with guaranteed-feasible steps, deep restarts counted as single evaluations.
* `plain1`: convex-concave procedure + augmented-Lagrangian local solver, seed queue with quotas.
* `none3`: SLP with a cumulative absolute per-n deadline schedule and deep-polish hops.

## Files

```
solvers/<key>.py     the fifteen solvers, unchanged (sha256 in portfolio.json)
portfolio.json       key -> solver path, origin run, values arm, CPU-constant names to override,
                     sha256/bytes/lines, rollup row
scorer/harness.py    frozen mission scorer: validate(), Meter, Evaluator, AST scan (vendored)
scorer/verify.py     frozen mission verifier: parse_pck + census scoring (vendored)
tools/sweep_one.py   run ONE (solver, n, seed) under the contract with a chosen budget; independent
                     re-validation (own pure-python rule AND harness.validate must agree); .pck at %.17g
tools/sweep_multi.py run a schedule of (solver, n, seed, mode, cpu) jobs through one worker pool
tools/make_schedule.py  first-round schedule: every solver x n x seed x warm mode, cpu = base + slope*n
tools/escalate.py    successive halving: keep the top-K solvers per n from earlier rounds, new seeds,
                     bigger budget, extra seeds for n still below the live record
tools/aggregate.py   re-validate every job, best per n (strict), per-solver specialisation table,
                     next round's warm directory
tools/export_sota.py write sota/<name>/{pck,json,results.csv,manifest.json} and run the repository's
                     verify_and_compare.py compare --tol 0
tools/publish.py     aggregate -> export -> README -> git commit/push when anything changed
tools/campaign.py    the 24 h driver: rounds of escalate -> sweep_multi, publish every 10 minutes
tools/sweep_all.py   single-solver convenience launcher
tools/restart_analysis.py  seeds-vs-run-length analysis from the recorded anytime traces (k restarts of B/k)
tools/focus.json     the allocation policy used from round C on: skip N<=25, one solver on 26-50, both
                     warm modes on 51-100 with one 90 s refine job and yield-weighted (Thompson-sampled)
                     neighbour-mode restarts; rounds interleave size bands so every band progresses
```

## What the campaign taught about allocation

* Refine-the-incumbent runs are deterministic polishers: the same incumbent gives the same result for
  any seed 86% of the time, and improvements arrive within seconds. One short run per size is enough.
* Neighbours-only runs are the stochastic search: seeds differ 72% of the time when a run improves, and
  the first improvement typically arrives only after several minutes, so a single long run beats
  splitting the same budget into shorter restarts.
* Yield differs by an order of magnitude across solvers (neighbour-mode incumbent-beat rates 0.9% to
  9% at N >= 51), and it is not predicted by which solver holds the most published packings, since
  polishers inherit provenance. Allocation is therefore by observed yield, recomputed every round.
* Sizes 1-50 are converged: 800+ jobs produced no improvement over the tabulated values, which every
  solver reproduces to ~1e-12.

## Running a sweep

```bash
cd solver-si-v2-20260927
export CP_CENSUS=/path/to/census        # warm_p1/csqv<n>.pck (your best-known packings) + records_live_inflated.json
python3 tools/make_schedule.py --out roundA.jsonl --solvers all --n 1-100 --seeds 0 --modes self,nbr --base 20 --slope 1
python3 tools/sweep_multi.py --schedule roundA.jsonl --out-dir roundA --jobs 60 \
        --warm $CP_CENSUS/warm_p1 --records-file $CP_CENSUS/records_live_inflated.json
python3 tools/aggregate.py --phase roundA --warm $CP_CENSUS/warm_p1 --out agg
python3 tools/escalate.py --agg agg --out roundB.jsonl --top 4 --seeds 1-2 --seeds-below 1-4 --base 60 --slope 3
```

Two warm modes: `self` gives the solver the incumbent `csqv<n>.pck` plus its neighbours (refine);
`nbr` withholds the incumbent so the solver must build n from neighbouring sizes or cold starts
(diversify). The records file handed to the solvers is the live Packomania table inflated by 5e-4
relative, so no solver stops early at "the record".

Requirements: Python 3.10+, numpy, scipy (HiGHS via `scipy.optimize.linprog`). Single-threaded BLAS is
forced per job (`OMP_NUM_THREADS=1` etc.).

## Caveats

* The CPU override is a module-constant override; a few solvers derive other constants from it at
  import time or read an environment variable (`anchor1`: `CP_SOLVER_CPU`), see `cpu_div`/`cpu_add`
  in `portfolio.json`.
* Results are deterministic for (solver bytes, warm files, seed) only when the solver ends on its own
  deadline; runs cut by the wall or CPU alarm land at a slightly different point each time.
* The solvers were evolved on Linux x86-64 and a MacBook; numerical paths (HiGHS, L-BFGS-B) can differ
  in the last bits across platforms, which is why the exported packings are re-verified from
  coordinates rather than by re-running.
