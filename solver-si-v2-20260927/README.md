# solver-si-v2-20260927: a portfolio of twenty-seven AI-evolved circle-packing solvers, and the sweep that runs them

This directory holds the solvers behind [`sota/si-v2-20260927/`](../sota/si-v2-20260927/): twenty-seven
`solver.py` files (fifteen original, twelve added in round I on 2026-09-28), each written by an LLM
agent inside one run of the self-improvement-v2 (SI-v2) `circle-packing` mission (or its
`circle-packing-v2` variant), copied **byte for byte** (sha256 in `portfolio.json`) except where
`portfolio.json` says `patched`, plus the tooling that sweeps them over N = 1..100 with a budget that
grows with N, re-validates every packing it gets back, and keeps the best strictly feasible packing per N.

Nothing here was written by hand except the sweep tooling, the two `smr*` adapters and one constant in
`deon10` (below). The solvers were extracted unchanged from the run workspaces; the only thing the
sweep changes at run time is each solver's *CPU allowance* (a module-level constant such as
`CPU_BUDGET`, set after import), so a solver that gave itself 100 CPU-seconds inside the mission can be
handed 20 minutes here.

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

Twelve more solvers were added on 2026-09-28 and screened in round I (N 51-100, `nbr` + `self`, seed 0)
and round J (all 27, Thompson-allocated). None of them produced a strict beat.

| key | origin run | arm (from the run name) | bytes | screening note |
|---|---|---|---:|---|
| `uv15pf7` | jason-mbp `_archive/2026-09-14_universalism-v15-probefix-x10/universalism-v15-probefix-7` | universalism-v15-probefix | 67,827 | |
| `tv14ob1` | jason-mbp `2026-09-15_transcendence-v14-oldbrief-x10/transcendence-v14-1` | transcendence-v14 | 75,544 | |
| `tv16pf2` | jason-mbp `_archive/2026-09-14_transcendence-v16-probefix-x10/transcendence-v16-probefix-2` | transcendence-v16-probefix | 84,249 | |
| `tv16pf1` | jason-mbp `_archive/2026-09-14_transcendence-v16-probefix-x10/transcendence-v16-probefix-1` | transcendence-v16-probefix | 85,225 | |
| `tv14pf9` | jason-mbp `_archive/2026-09-15_transcendence-v14-probefix-x10/transcendence-v14-probefix-9` | transcendence-v14-probefix | 78,332 | |
| `deon10` | aws `2026-08-22_circle-register-x10-15it30m/explore-deon-10` | explore (deontic register) | 96,279 | **patched**: the single-target budget, hard-coded to 8.7 s (`min(100, 6+2.7*1)`) in the original, reads the `CPU_BUDGET` module constant; sha256 is of the patched copy |
| `sm2` | aws `2026-08-17_circle-sm-x10-20it30m/explore-sm-2` | explore-sm | 128,327 | fast-saturating SLP; budget 100+3n s |
| `sm7` | aws `2026-08-17_circle-sm-x10-20it30m/explore-sm-7` | explore-sm | 111,686 | fast-saturating SLP; budget 100+2n s |
| `v2enh5` | aws `2026-09-16_circle-v2-enh15-x10/enhancement-v15-5` (`circle-packing-v2`) | enhancement-v15 | 63,009 | fast-saturating SLP; budget 100+2n s |
| `v2exp8` | aws `2026-09-16_circle-v2-8arm-x10/explore-8` (`circle-packing-v2`) | explore | 175,574 | spends 90% of each run in a target phase: the records table handed to solvers is inflated by 5e-4, so its `tf_open` never closes |
| `smr27` | this repository, `solver-sm-radical-v6-n27/pack.py` (`pack.search`) | adapter, not an SI-v2 solver | 1,776 | overruns its CPU allowance: the wall-clock deadline is checked only between multi-second SLSQP refines (29/100 backstop hits, 16 `no_packing`) |
| `smr54` | this repository, `solver-sm-radical-v6-n54/{endgame,pipeline}.py` | adapter, not an SI-v2 solver | 2,119 | pipeline tuned at N=54: median `nbr` gap -7.7e-3, worst -2e-2 at N 88-90 at its native 240 s |

`smr27` and `smr54` are thin adapters (33 and 43 lines) that run this repository's own solvers under the
SI-v2 `solve()` contract, converting between their corner frame `[0,1]^2` and the centred frame; the
repository copies locate `solver-sm-radical-v6-n27/` and `-n54/` relative to the adapter file. From
round K on only ten solvers receive CPU (`tv14pf6`, `tv16ob6`, `none7`, `nalt6`, `anchor6`, `anchor1`
and, of the new twelve, `v2enh5`, `sm2`, `sm7`, `deon10`); the other seventeen never tied or have found
nothing since round A.

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
* `rng`: `numpy.random.RandomState(seed)`, the only randomness source *by contract*; in practice many
  solvers ignore it (see *Seed audit* below), which is why `sweep_one.py` now mixes the job seed into
  every generator the solver constructs.
* `targets`: the list of n to solve (the sweep passes one n per process).
* The solver reads its warm starts itself from `bench/packs/csqv<n>.pck` (and neighbours n±k) and
  the target table from `bench/records.json`, both relative to the working directory; the sweep builds
  that layout in a throwaway workspace per job.
* Feasibility (`scorer/harness.py:validate`): exactly n circles, all finite, every `r > 0`, wall
  slack and pairwise slack `>= -1e-9`. The sweep additionally requires slack `>= 0` (zero tolerance)
  before a packing can be published, and the export re-checks it in exact rational arithmetic.

`scorer/` is the mission's frozen scorer, vendored unchanged (`harness.py`, `verify.py`; pure
stdlib + numpy). It is what the solvers were evolved against and what the tools import.

## Algorithms (one line each)

All the SI-v2 solvers share one idea: for fixed centres the radii are an exact LP; centres move by
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
* the ten SI-v2 solvers added in round I are not summarised here; `sm7`, `v2enh5`, `sm2` and `deon10`
  are SLP solvers that saturate within a few minutes and get shorter budgets (below).
* `smr27`: `pack.search` of `solver-sm-radical-v6-n27`, seeded from the incumbent when one is present.
* `smr54`: the `solver-sm-radical-v6-n54` endgame walk from the incumbent (warm ILS), or the full
  broad -> endgame `pipeline.solve` without one.

## Files

```
solvers/<key>.py     the twenty-seven solvers (sha256 in portfolio.json; `patched` marks the three
                     that are not byte-for-byte: deon10, smr27, smr54)
portfolio.json       key -> solver path, origin run, values arm, CPU-constant names to override,
                     sha256/bytes/lines, rollup row, `added` (round) and `patched` where applicable
scorer/harness.py    frozen mission scorer: validate(), Meter, Evaluator, AST scan (vendored)
scorer/verify.py     frozen mission verifier: parse_pck + census scoring (vendored)
tools/sweep_one.py   run ONE (solver, n, seed) under the contract with a chosen budget; independent
                     re-validation (own pure-python rule AND harness.validate must agree); .pck at %.17g;
                     --module-attrs (CPU constants), --records-file, --warm-exclude-self (nbr mode),
                     --warm-kick F (kick mode), anytime trace of the best-so-far, and seed mixing: the job
                     seed is mixed into every RandomState/default_rng/random.Random the solver constructs
                     (recorded as "seed_mixing": true in the job JSON)
tools/sweep_multi.py run a schedule of (solver, n, seed, mode, cpu) jobs through one worker pool;
                     size bands and warm modes are interleaved so every (band, mode) progresses at the same pace
tools/make_schedule.py  first-round schedule: every solver x n x seed x warm mode, cpu = base + slope*n
tools/escalate.py    successive halving: keep the top-K solvers per n from earlier rounds, new seeds,
                     bigger budget, extra seeds for n still below the live record; with focus.json:
                     Thompson-sampled allocation by observed yield, kick jobs, ruggedness bonus (rounds C-J)
tools/focus.json     the escalate.py policy used from round C to J: skip N<=25, one solver on 26-50, both
                     warm modes on 51-100 with one 90 s refine job and yield-weighted (Thompson-sampled)
                     neighbour-mode restarts
tools/size_signals.py  per-size softness signals for N 51-100 from the aggregate: verdict, increment
                     anomaly, distinct near-best nbr optima, rebuild rate, best/median gap, recent beats
tools/alloc_round.py the round-K-onward allocator: nbr only, one full-length run per seed, solver shares
                     per band, sizes weighted by softness (see below) -> schedule jsonl
tools/aggregate.py   re-validate every job, best per n (strict; deterministic tie-break keeps the earliest
                     job at equal sums to 1e-12), N <= 100, per-solver specialisation table, next round's
                     warm directory
tools/exact_verify.py  feasibility of a .pck in exact rational arithmetic (fractions.Fraction, squared
                     comparisons, zero tolerance); --repair-out shrinks radii by k*2^-50 until it passes
tools/export_sota.py write sota/<name>/{pck,json,results.csv,manifest.json}; every packing must pass
                     exact_verify (ulp-level round-off repaired by shrinking radii, factor recorded); then
                     run the repository's verify_and_compare.py compare --tol 0
tools/publish.py     aggregate -> export -> README -> git commit/push when anything changed
tools/campaign.py    the 24 h driver (rounds B-H): rounds of escalate -> sweep_multi, publish every 10 minutes
tools/finish_campaign.py  round I (screen the 12 new solvers) then round J (all 27, Thompson) to the deadline
tools/final_campaign2.py  rounds K, L: alloc_round.py allocation, to the original 24 h deadline
tools/final_campaign3.py  the running driver: attaches to round K, then plans ~3 h rounds L, M, N, ...
                     (seeds 61+100*idx, budgets x1.0 / x1.5 alternating) to the extended deadline
tools/sweep_all.py   single-solver convenience launcher
tools/restart_analysis.py  seeds-vs-run-length analysis from the recorded anytime traces (k restarts of B/k)
```

The drivers' deadlines are the recorded epoch values by default (kept in a comment); override with
`--t-end` or the environment variable `CP_T_END`.

## What the campaign taught about allocation

* Refine-the-incumbent runs are deterministic polishers: the same incumbent gives the same result for
  any seed 86% of the time, and improvements arrive within seconds. One short run per size is enough.
  Measured again on 09-28: 91% of `self` runs reach the incumbent within 60 s and nothing above 1e-9
  arrives later, so from round K on `self` is one run of <= 60 s, only for sizes whose incumbent moved.
* Neighbours-only runs are the stochastic search: seeds differ 72% of the time when a run improves, and
  the first improvement typically arrives only after several minutes, so a single long run beats
  splitting the same budget into shorter restarts. Beats arrive late (median at 0.57 of the allowance,
  P90 at 0.94), and splitting a budget into k restarts never won for beats, so every `nbr` job is one
  full-length run per seed. Budget `clip(9.4n - 154, 350, 800)` s (a fit of the 90th percentile of the
  time to re-find the record), shorter for the fast-saturating SLP solvers `sm7`/`v2enh5` (100+2n),
  `sm2` (100+3n) and `deon10` (120+4n). Rounds alternate x1.0 / x1.5 budgets (cap 1200 s) as a
  cutoff hedge.
* Seed audit: many solvers construct their own `numpy.random.RandomState(const)`, `default_rng(const)`
  or `random.Random(const)` generators and ignore the `rng` they receive, so distinct job seeds produced
  identical searches (`tv14pf6` 47% of seed pairs identical, `smv16_6` 60%, `tv16ob6` 54%, `none7` 40%).
  Since 06:50 UTC 09-28 `sweep_one.py` mixes the job seed into every generator constructed in the child
  (monkey-patched constructors; `seed_mixing: true` in the job JSON). `self` mode is unaffected
  (deterministic).
* Kick mode (iterated-local-search perturbation of 15-45% of the circles, better-only acceptance):
  0 beats in 1,318 runs (103 CPU-h); it returns to the incumbent 13% of the time. Dropped from round K on.
* Yield differs by an order of magnitude across solvers (neighbour-mode incumbent-beat rates 0.9% to
  9% at N >= 51), and it is not predicted by which solver holds the most published packings, since
  polishers inherit provenance. Allocation is therefore by observed yield, recomputed every round.
  Solver shares of `nbr` CPU from beats per CPU-h over rounds C-J: N 51-77: `tv14pf6` .33, `tv16ob6`
  .22, `v2enh5` .13, `sm2` .13, `nalt6` .09, `anchor6` .05, `deon10` .05; N 78-100: `tv14pf6` .33,
  `tv16ob6` .25, `none7` .10, `sm7` .09, `sm2` .09, `anchor1` .05, `v2enh5` .05, `nalt6` .04. The other
  seventeen solvers get no CPU from round K on (never tied, or nothing since round A).
* Sizes (`alloc_round.py` + `size_signals.py`): unbeaten sizes get 80% of the CPU, split by a softness
  score `exp(0.35 z(-log rebuild_rate) + 0.25 z(log distinct near-best nbr optima)) x
  (1 + 0.4 clip(-increment_anomaly / 2.5e-3, 0, 1))`, tempered half-uniform; already-beaten sizes share
  the remaining 20% in proportion to their recent beat counts; per-size cap 6%; at least 4 runs per
  unbeaten size; 26-50 keep one `tv14pf6` run per size.
* Sizes 1-50 are converged: 800+ jobs produced no improvement over the tabulated values, which every
  solver reproduces to ~1e-12. Sizes 1-25 are skipped entirely.
* Workers: 32 cores x 2 SMT. A thread runs 1.33x faster at 32 wide, but total work is 0.69x, so the
  pool is 62 workers.
* Yield history (beats vs the round incumbent, N >= 51): round A 150 (18 sizes), B 7, C 14, D 6, E 6,
  G 2, H 0, I 0, J 0. The marginal P(beat) per run is now <= 0.25% (every size has 155-279 runs).
* Standing result at 07:41 UTC 09-28: **19 WIN / 81 tie / 0 below** vs the live Packomania table (wins
  at N 63, 66, 78, 79, 80, 82, 83, 84, 85, 89, 92, 93, 94, 95, 96, 97, 98, 99, 100).
* Campaign: started 12:09 UTC 09-27; the deadline was extended by the operator to 12:09 UTC 09-29. The
  running driver is `final_campaign3.py`, which attaches to a running round and plans ~3 h rounds
  (L, M, N, ...) with seeds 61+100*idx.

## Exact-arithmetic gate

`export_sota.py` re-checks every exported packing with `fractions.Fraction` arithmetic at zero
tolerance (`exact_verify.py`: every double is an exact rational; containment and non-overlap are decided
with squared comparisons, no sqrt, no epsilon). A packing that fails by round-off is repaired by
shrinking every radius by `k * 2^-50` ulps (the factor is recorded in the sidecar and the manifest).
Five ties (N 7, 9, 14, 18, 26) needed the one-step repair (radius factor 1 - 2^-50); all 100 pass.

## Running a sweep

```bash
cd solver-si-v2-20260927
export CP_CENSUS=/path/to/census        # warm_p1/csqv<n>.pck (your best-known packings) + records_live_inflated.json
export CP_LIVE=/path/to/live            # latest.json, the live Packomania table (publish.py refreshes it hourly)
export CP_SWEEP_ROOT=$PWD               # rounds root: round dirs, agg/, tmp/ (default /tmp/si_tools/cp_sweep)
# CP_LIVE_RECORDS overrides the records file aggregate.py compares against (default ../sota/packomania/packomania_csqv.json)
python3 tools/make_schedule.py --out roundA.jsonl --solvers all --n 1-100 --seeds 0 --modes self,nbr --base 20 --slope 1
python3 tools/sweep_multi.py --schedule roundA.jsonl --out-dir roundA --jobs 60 \
        --warm $CP_CENSUS/warm_p1 --records-file $CP_CENSUS/records_live_inflated.json
python3 tools/aggregate.py --phase roundA --warm $CP_CENSUS/warm_p1 --out agg
python3 tools/escalate.py --agg agg --out roundB.jsonl --top 4 --seeds 1-2 --seeds-below 1-4 --base 60 --slope 3
# round K onward: softness-weighted, nbr-only allocation
python3 tools/size_signals.py tmp/size_signals.json agg
python3 tools/alloc_round.py --signals tmp/size_signals.json --warm-prev $CP_CENSUS/warm_roundJ \
        --warm $CP_CENSUS/warm_roundK --cpu-hours 140 --seed0 61 --cpu-scale 1.0 --out roundK.jsonl
# exact feasibility of any .pck, with an optional repaired copy
python3 tools/exact_verify.py agg/best/csqv63.pck [--repair-out fixed/]
# the running driver (attaches to a running round, then plans ~3 h rounds to the deadline)
CP_T_END=<epoch seconds> python3 tools/final_campaign3.py     # or --t-end; default = the recorded deadline
```

Three warm modes: `self` gives the solver the incumbent `csqv<n>.pck` plus its neighbours (refine);
`nbr` withholds the incumbent so the solver must build n from neighbouring sizes or cold starts
(diversify); `kick` (`--warm-kick F`) is an iterated-local-search perturbation: a fraction F of the
incumbent's circles are relocated at random with feasibility preserved, and the solver regrows the
packing from that new basin (deterministic in size and seed; strengths cycle over 0.15-0.45; dropped
from round K on, see above). The records file handed to the solvers is the live Packomania table
inflated by 5e-4 relative, so no solver stops early at "the record".

Requirements: Python 3.10+, numpy, scipy (HiGHS via `scipy.optimize.linprog`). Single-threaded BLAS is
forced per job (`OMP_NUM_THREADS=1` etc.).

## Caveats

* The CPU override is a module-constant override; a few solvers derive other constants from it at
  import time or read an environment variable (`anchor1`: `CP_SOLVER_CPU`), see `cpu_div`/`cpu_add`
  in `portfolio.json`.
* Results are deterministic for (solver bytes, warm files, seed) only when the solver ends on its own
  deadline; runs cut by the wall or CPU alarm land at a slightly different point each time. With seed
  mixing (jobs since 06:50 UTC 09-28), the seed that matters is the job seed; a job without
  `"seed_mixing": true` used the solver's own constant seeds.
* The solvers were evolved on Linux x86-64 and a MacBook; numerical paths (HiGHS, L-BFGS-B) can differ
  in the last bits across platforms, which is why the exported packings are re-verified from
  coordinates rather than by re-running.
* `smr27` overruns its CPU allowance (its deadline is checked only between multi-second SLSQP
  refines), `smr54` is tuned at N=54, and `v2exp8` spends most of each run chasing the inflated
  records table; none of the three receives CPU from round K on.
