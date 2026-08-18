# circle-packing csqv record beats -- SI-v2 batches of 2026-08-17

Seventeen circle packings that **strictly beat the 2026-08-14 snapshot of the Packomania `csqv`
table**, exported byte-for-byte from the runs that produced them, with a standalone re-verifier.

This directory is a **self-contained export**, added alongside the earlier `sota/` results already in
this repository; it overwrites nothing of theirs and shares no packings with them (see *Provenance and
prior claims*, point 3).

Produced by the self-improvement-v2 (SI-v2) `circle-packing` mission: an LLM agent evolves one solver
file over 20 iterations against a frozen scorer, and commits a census of packings. Nothing here is a
new packing *family*; these are numerical refinements of known configurations, at margins of 1e-4 to
1e-3 on sums near 4-5. Read *Margins, honestly* and *Provenance and prior claims* before quoting any
of it.

## What the number is

For each **n**, place n circles of **freely chosen radii** in the unit square so that no two overlap
and every circle lies entirely inside the square, maximizing

```
sum_r = r_1 + r_2 + ... + r_n
```

That is Packomania's **`csqv`** problem ("circles in a square, variable radii", the objective
Packomania tabulates as the sum of radii). It is the same benchmark AlphaEvolve and ShinkaEvolve
report on. Packomania's values are **best-known**, not proven optima, so "beating" one means finding a
strictly feasible packing whose sum of radii exceeds the tabulated value.

Coordinates in these files live in the **centered** unit square, `[-0.5, 0.5] x [-0.5, 0.5]`.

## The claim, and exactly which table it is against

> **A beat here is against ONE SNAPSHOT: `packomania.com/csqv`, retrieved 2026-08-14, the site
> reading "Last Update 13-Aug-2026".** That snapshot is vendored in this directory as
> `packomania_csqv_20260814.json` (values only, no coordinates), sha256 `1a3cd5f8a7652d0db6596d7fa13436fd708a9d4c569b9ec2c6b1a5aa07238f4e`.
> Packomania updates continuously -- 22 of the 37 values relevant here rose in the 11 days before
> this snapshot alone. **This is not a claim that these are current records, and it is not a claim on any
> later table.** Anyone re-checking against a fresh fetch may well find some or all of these absorbed
> or overtaken. Re-fetch before repeating the claim.

Against that snapshot, the best packing found anywhere in the swept batches beats the table at
**17 of the 37 n the runs attempt** (odd n, 27 to 99):

| n | Σr (this packing) | record 2026-08-14 | margin (abs) | margin (rel) | source batch / run | iter | runs beating |
|---:|---|---|---:|---:|---|---:|---:|
| 53 | 3.805568432564 | 3.804408802004 | +1.160e-03 | +3.05e-04 | `2v_x15` / `explore-1` | 12 | 6/50 |
| 63 | 4.154116740618 | 4.153848570061 | +2.682e-04 | +6.46e-05 | `6v_x5` / `overfit-sm-4` | 13 | 2/50 |
| 65 | 4.221928422662 | 4.219413302014 | +2.515e-03 | +5.96e-04 | `6v_x5` / `overfit-sm-4` | 11 | 5/50 |
| 67 | 4.289481392390 | 4.287777805457 | +1.704e-03 | +3.97e-04 | `6v_x5` / `overfit-sm-5` | 4 | 11/50 |
| 69 | 4.357977705112 | 4.356995844873 | +9.819e-04 | +2.25e-04 | `6v_x5` / `overfit-3` | 12 | 7/50 |
| 73 | 4.488782078921 | 4.488004575020 | +7.775e-04 | +1.73e-04 | `6v_x5` / `overfit-3` | 12 | 2/50 |
| 75 | 4.550081805192 | 4.548223289937 | +1.859e-03 | +4.09e-04 | `2v_x15` / `conservative-1` | 11 | 2/50 |
| 77 | 4.605527989026 | 4.605135681249 | +3.923e-04 | +8.52e-05 | `6v_x5` / `generalize-4` | 13 | 2/50 |
| 79 | 4.666630641409 | 4.664674436598 | +1.956e-03 | +4.19e-04 | `6v_x5` / `overfit-sm-1` | 19 | 2/50 |
| 83 | 4.784464220734 | 4.783919133241 | +5.451e-04 | +1.14e-04 | `2v_x15` / `explore-2` | 8 | 3/50 |
| 85 | 4.846671972827 | 4.845134378577 | +1.538e-03 | +3.17e-04 | `2v_x15` / `explore-2` | 12 | 2/50 |
| 87 | 4.907558130319 | 4.906503570705 | +1.055e-03 | +2.15e-04 | `2v_x15` / `explore-3` | 8 | 4/50 |
| 91 | 5.022578167830 | 5.021462693130 | +1.115e-03 | +2.22e-04 | `6v_x5` / `overfit-1` | 19 | 1/50 |
| 93 | 5.073650364681 | 5.071029091028 | +2.621e-03 | +5.17e-04 | `6v_x5` / `none-5` | 9 | 8/50 |
| 95 | 5.126259907290 | 5.125567045105 | +6.929e-04 | +1.35e-04 | `6v_x5` / `generalize-2` | 18 | 3/50 |
| 97 | 5.181096077359 | 5.180639279289 | +4.568e-04 | +8.82e-05 | `2v_x15` / `explore-2` | 12 | 2/50 |
| 99 | 5.235423639117 | 5.233204567399 | +2.219e-03 | +4.24e-04 | `2v_x15` / `explore-2` | 10 | 4/50 |

`iter` is the loop iteration of the run whose commit last wrote that pack file. `runs beating` counts
how many of the 50 swept runs independently produced a beating packing at that n, which is a rough
robustness signal: n = 67 was reached by 11 runs, n = 91 by exactly one.

**Not beaten.** At the other 20 attempted n nothing is claimed. Seventeen of them (27, 29, 31, 33,
35, 37, 39, 41, 43, 45, 47, 49, 55, 57, 59, 61, 89) land within 1e-10 of the tabulated value -- six of
those (27, 33, 39, 43, 55, 57) are fractionally *above* it, by 2e-13 to 4e-11, which is float noise
far under the `WIN_MARGIN` of 1e-6 and is scored a **tie**, not a beat, and is **not** exported. Only
three (51, 71, 81) are genuinely short, by 9.1e-5 to 1.4e-3. Sizes outside this
census are not attempted by these runs, so nothing is claimed there either.

## Verification

Everything is re-checked **from coordinates**, by the frozen mission scorer, at **zero** feasibility
tolerance:

```python
circ = verify.parse_pck(path)
v    = harness.validate(circ, n, tol=0.0)                 # recomputes sum_r + both slacks
beat = v["feasible"] and harness.counts_for(v["sum_r"], record)   # sum_r >= record + 1e-6, ABSOLUTE
```

* `harness.validate` is pure stdlib: exactly n circles, every `r > 0`, every coordinate finite, wall
  slack `>= -tol` and pairwise slack `>= -tol`, `sum_r` by `math.fsum`.
* **`tol = 0.0`, not the harness default `FEAS_TOL = 1e-9`.** The distinction is not cosmetic: across
  the swept batches **203 of 1850** committed packings are feasible at 1e-9 and infeasible at 0.0.
  Only packings feasible at **0.0** are exported here. Every exported pack is reported at both
  tolerances in `manifest.json`; all 17 pass both.
* The beat bar `harness.counts_for` is **absolute**, `sum_r >= record + 1e-6`. A tie does not count.
* Slacks in the exported packings run from `+0.0` to `+4.7e-9` (wall) and `+0.0` to `+1.0e-10`
  (pairwise): they are on the constraint boundary, as an optimal packing must be, but never past it.

Re-run it yourself:

```bash
python3 verify_sota.py                                  # frozen scorer at its in-repo path
python3 verify_sota.py --scorer /path/to/missions/circle-packing/scorer
```

`verify_sota.py` re-parses every `.pck`, recomputes `sum_r` and both slacks, re-reads the vendored
record table, checks the sha256 of the table and of the scorer files against `manifest.json`, and
**contradicts** the manifest where they disagree (it treats `manifest.json` as a claim to test, not as
input). It prints one `PASS`/`FAIL` line per n plus a summary, and exits non-zero on any failure. It
uses only the frozen scorer and the standard library, and runs no optimizer. Current output ends:

```text
checked 17 packings against the 2026-08-14 packomania csqv snapshot
PASS 17 / 17   FAIL 0
every claim reproduces from coordinates; absolute margins 2.682e-04 .. 2.621e-03
VERDICT: ALL CLAIMS VERIFIED
```

It was checked against a negative control: perturbing one radius in one pack by +1e-6 makes that n
report `FAIL  infeasible at tol=0.0; ...; pack sha256 != manifest` and the script exit 1.

## Margins, honestly

| | |
|---|---|
| absolute margin over the record | **2.68e-04 to 2.62e-03** |
| relative margin | **6.46e-05 to 5.96e-04** |
| the sums themselves | 3.8056 to 5.2354 |

These are **refinements in the fourth-to-sixth significant figure of a known configuration, not new
packing families.** The right description is "improves the best-known value at these n by 0.006% to
0.06% on one snapshot", not "solves circle packing". For scale, the margins are three to six orders of
magnitude above the harness's own numerical noise floor (relative 1.0e-8 to 1.9e-8), so they are real
rather than float noise -- but they are small.

There is no claim of optimality anywhere in this directory. Packomania's entries are best-known values
themselves.

## Provenance and prior claims

**Where each packing came from** is in `manifest.json` per n: batch, run arm, path inside the run,
the run's seed commit and HEAD, the commit that last wrote the pack, the loop iteration, and the
`solver_sha` of the evolved solver that produced it. Sources:

| batch | runs swept | wins contributed |
|---|---:|---|
| `results/circle_6v_x5_20it30m_20260817b` (COMPLETE, 30 runs) | 30 | 63, 65, 67, 69, 73, 77, 79, 91, 93, 95 |
| `results/circle_2v_x15_20it30m_20260817` (in flight at export) | 16 of 30 | 53, 75, 83, 85, 87, 97, 99 |
| `results/circle_sm_x10_20it30m_20260817` (in flight at export) | 4 of 10 | none |
| `results/circle-packing` (older 2026-08-01..10 runs) | -- | holds no `.pck` census at all |

The two in-flight batches were read-only at export time and are still running, so a later sweep can
only add to this set, never subtract from it.

**These are attributable to the runs, not to the seed.** Every run starts from
`missions/circle-packing/scorer/baseline_packs/`, seeded byte-identical (verified: the `bench/packs`
tree at each winning run's *first* commit is `diff -r` identical to `baseline_packs/`). That seed
census **beats nothing** on this table -- it reaches only 66% to 78% of the tabulated value (0.62 to
1.75 *below* it in absolute terms), and 0 of 37 clear the bar at either tolerance. The gap
from that seed to these packings was closed inside the runs.

**The runs were optimizing against an easier table than the one they are scored on here.** The in-run
scorer is pinned to the **2026-08-03** retrieval (`scorer/records.json`), which is at or below the
2026-08-14 snapshot at all 37 n and materially below at 22 of them, by 1.82e-4 to 1.08e-3 relative
(a 23rd, n = 43, differs by 6e-13, which is float noise). The
beats above are measured against the *later, harder* table, which makes them a strictly stronger
statement than the `# BEATS` counts the runs printed for themselves.

### What was already held before these batches -- read this before calling any n "new"

1. **Three n were already beaten by an earlier SI-v2 run.** `missions/circle-packing/README.md`
   records "the three beats standing at the time of the revert ... (n = 69/85/95)", against this same
   2026-08-14 table, at approximately +6.0e-5, +9.8e-4 and +1.3e-3. The revert commit is
   2026-08-17 03:12:57 and the 30-arm batch started 03:13:14, seventeen seconds later, so that
   earlier run predates every batch swept here. **Established:** exactly three n were pre-held, and
   for two of them (69, 85) the packings exported here are better than the quoted earlier margins.
   **Could NOT be established:** the earlier run's directory no longer exists (`results/` is
   gitignored, and no archive copy survives anywhere on this machine), so its packings, its Σr to
   full precision and its identity are unrecoverable. At **n = 95** the two-significant-figure margin
   quoted for that earlier run (+1.3e-3) is *larger* than the +6.9e-4 exported here, so the best
   n = 95 packing this programme ever produced is probably **not** the one in this directory and
   cannot be retrieved. Whether more than three n were beaten and then lost could not be settled from
   any surviving artifact; the only surviving statement is the "three" in that README.

2. **At 8 of the 17 n the record being beaten is this programme's own earlier result.** At
   n = 53, 63, 67, 69, 77, 83, 85, 87 the 2026-08-14 tabulated value equals, to within 4e-13, the Σr
   of the packing committed by an earlier (v6, 2026-08-01..10) solver from this same programme,
   published in this repository as `sota/sm-radical-v6-n54/`. Packomania's 2026-08-13 update
   absorbed those values. So at those 8 n **the batch is improving on itself**, not on an unrelated prior art.
   The remaining 9 (65, 73, 75, 79, 91, 93, 95, 97, 99) are against table entries that were already
   present, byte-identical, in the independent 2026-08-10 retrieval vendored at
   `sota/packomania/packomania_csqv.json`, so they did not arrive with the 2026-08-13 update that
   absorbed this programme's v6 results. That is as far as their attribution could be established
   here; Packomania does not name contributors in the values table.

3. **The earlier published solvers in this repo beat nothing on this table.** Re-verified here: the
   `sota/sm-radical-v6-n54/` census (92 packings) and `sota/sm-radical-v6-n27/` census beat the
   2026-08-14 snapshot at **0** n, exactly as expected once point 2 is understood -- they are now
   ties, not wins. There is no double-counting between that older export and this one.

## Files

```
packs/csqv<n>.pck                17 packings, copied byte-for-byte from the run that found them
manifest.json                    per n: Σr, record, both margins, feasibility at tol 0.0 and 1e-9,
                                 min pair slack, min wall slack, min r, pack sha256, and full
                                 provenance (batch, run, commits, iteration, solver_sha)
packomania_csqv_20260814.json    the record snapshot the claim is against, copied byte-for-byte
verify_sota.py                   standalone re-verifier (frozen scorer + stdlib, no optimizer)
README.md                        this file
```

### `.pck` format

```
0.10192148722692326          <- line 1: the largest radius in the packing
agent                        <- line 2: a free-text label (always `agent` in these files)
-0.45003230199020933 -0.17980286582529728 0.049967698009736355
 0.44958357506982849 -0.28788915779280577 0.050416424930116693
...                          <- one `x y r` triple per circle, n of them, full repr precision
```

Coordinates are in the **centered** unit square: every circle satisfies
`-0.5 <= x - r`, `x + r <= 0.5` and the same in y. `verify.parse_pck` skips the first two lines and
reads the rest as triples.

## Regenerating

The packings are fixed files; checking them needs no re-run, and re-running would not reproduce them
(the solvers are LLM-evolved per run and the search is stochastic). What can be re-derived is the
selection: the best feasible Σr per n across every committed census.

```bash
REPO=/Users/jason/Desktop/science_moonshot/self_improvement_v2

# the scorer this directory is verified with (frozen, operator-owned)
$REPO/missions/circle-packing/scorer/{harness.py,verify.py,records.json,baseline_packs/}

# the record snapshot
$REPO/missions/circle-packing/heldout/packomania_csqv_20260814.json

# the batches swept, 50 runs, 1850 committed packings, at <run>/bench/packs/csqv<n>.pck
$REPO/results/circle_6v_x5_20it30m_20260817b/{none,placebo,generalize,overfit,generalize-sm,overfit-sm}-{1..5}
$REPO/results/circle_2v_x15_20it30m_20260817/{conservative,explore}-{1..8}
$REPO/results/circle_sm_x10_20it30m_20260817/explore-sm-{1..4}
```

Sweep rule: for each n, take the committed packing with the largest Σr **among those feasible at
`tol = 0.0`**, then apply `harness.counts_for` against the snapshot. Nothing was re-optimized,
re-rounded or repaired on the way out; `manifest.json` records the sha256 of every file so a byte
comparison against the source run is a one-liner.

Mission notes, including the record-table history and why the in-run score is graded rather than a
beat count: `missions/circle-packing/README.md` in that repo.
