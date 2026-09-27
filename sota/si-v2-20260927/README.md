# si-v2-20260927: best packing per N from the SI-v2 solver portfolio sweep

**100 packings (N = 1..100)**, the best strictly feasible packing per N found by sweeping fifteen
LLM-evolved circle-packing solvers from the self-improvement-v2 (SI-v2) `circle-packing` mission, seeded
from every packing those runs ever stored. Compared against the Packomania `csqv` table
**retrieved 2026-09-27** (`sota/packomania/history/packomania_csqv_2026-09-27.json`): **18 WIN / 82 tie / 0 below**.

Sweep campaign status at 2026-09-27 21:43 UTC: 6075 jobs aggregated from phase1, roundA, roundB, roundC, roundD, roundE.

This directory is regenerated automatically by the sweep campaign; every number in it is recomputed
from the `.pck` coordinates at each update. Read *Margins, honestly* before quoting any of it.

## The problem and the rule

Place N circles of freely chosen radii in the unit square (centred, `[-0.5, 0.5]^2`), no overlaps,
all inside, maximise the sum of radii. Packomania's values are best-known, not proven optima.

A packing is exported only if it is **strictly feasible at zero tolerance** (every wall slack and every
pairwise slack `>= 0` in double precision, recomputed from the coordinates by two independent
implementations). A **WIN** means `sum_r > record + 1e-9` (the repository's `WIN_EPS`); `tie` means
`|sum_r - record| <= 1e-9`; `below` is everything else.

## Wins against the 2026-09-27 table

| N | ours Σr | record | Δ (abs) | Δ (rel) | found by |
|---:|---|---|---:|---:|---|
| 66 | 4.256749492862 | 4.255807931448 | +9.416e-04 | +2.21e-04 | `tv14pf6` nbr seed 12 |
| 78 | 4.636796920037 | 4.636377420432 | +4.195e-04 | +9.05e-05 | `tv14pf6` nbr seed 1 |
| 79 | 4.666928346066 | 4.666466802554 | +4.615e-04 | +9.89e-05 | `deon6` self seed 1 |
| 80 | 4.695876897572 | 4.695590667948 | +2.862e-04 | +6.10e-05 | `none7` self seed 2 |
| 82 | 4.756048298931 | 4.755873680213 | +1.746e-04 | +3.67e-05 | `deon6` self seed 0 |
| 83 | 4.786683838811 | 4.786393807284 | +2.900e-04 | +6.06e-05 | `deon6` self seed 0 |
| 84 | 4.816567858213 | 4.816500879756 | +6.698e-05 | +1.39e-05 | `deon6` self seed 0 |
| 85 | 4.848035445623 | 4.847351780582 | +6.837e-04 | +1.41e-04 | `deon6` self seed 1 |
| 89 | 4.967596361102 | 4.966850487024 | +7.459e-04 | +1.50e-04 | `deon6` self seed 0 |
| 92 | 5.049048324656 | 5.048761669590 | +2.867e-04 | +5.68e-05 | `deon6` self seed 5 |
| 93 | 5.076445594003 | 5.076245164998 | +2.004e-04 | +3.95e-05 | `deon6` self seed 5 |
| 94 | 5.103121333926 | 5.101975416147 | +1.146e-03 | +2.25e-04 | `deon6` self seed 1 |
| 95 | 5.129235581979 | 5.129111105835 | +1.245e-04 | +2.43e-05 | `none7` nbr seed 5 |
| 96 | 5.155743250529 | 5.155289947655 | +4.533e-04 | +8.79e-05 | `tv14pf6` nbr seed 23 |
| 97 | 5.182315239431 | 5.181609663002 | +7.056e-04 | +1.36e-04 | `deon6` self seed 5 |
| 98 | 5.209524986801 | 5.208578422017 | +9.466e-04 | +1.82e-04 | `tv14pf6` nbr seed 26 |
| 99 | 5.236903198995 | 5.236244728577 | +6.585e-04 | +1.26e-04 | `anchor1` self seed 5 |
| 100 | 5.264237427967 | 5.262787458777 | +1.450e-03 | +2.76e-04 | `anchor1` self seed 1 |

## Sizes still below the table

| N | ours Σr | record | Δ |
|---:|---|---|---:|
| - | - | - | - |

Every other N ties the tabulated value to within 1e-9.

## Margins, honestly

The wins are refinements in the fourth to sixth significant figure of known configurations, not new
packing families, at relative margins of about 1e-5 to 3e-4. The 2026-09-27 table already absorbed
many values this programme submitted in August; a row marked "stored census packing" was found by an
SI-v2 run before this sweep and merely re-verified here. Packomania updates continuously: re-fetch
before repeating any claim (`python3 verify_and_compare.py fetch`).

## Verify

From the repository root, re-verify every packing from coordinates at zero tolerance and rebuild the
comparison table:

```bash
python3 verify_and_compare.py compare --pck-dir sota/si-v2-20260927 \
  --records sota/packomania/history/packomania_csqv_2026-09-27.json --tol 0 --out /tmp/comparison-si-v2.md
```

`manifest.json` records, per N, the recomputed Σr, both minimum slacks, the sha256 of the `.pck`, and
full provenance (solver key, warm mode, seed, job path, solver sha256 and origin run).

## Solver portfolio and per-N specialisation

Fifteen solvers were swept. Each is a single `tools/solver.py` evolved by an LLM agent inside one
SI-v2 run (values arm and brief noted below); the solver files and the sweep tooling are in
`solver-si-v2-20260927/`. "packings" counts how many of the exported bests each solver produced
("census" = a stored pre-sweep packing was never beaten).

| key | origin run | values file | values md5 | held-out endpoint | solver sha256 | packings |
|---|---|---|---|---:|---|---:|
| `tv14pf6` | jason-mbp results/_archive/circle-packing/2026-09-15_transcendence-v14-probefix-x10/transcendence-v14-probefix-6 | universalism-v14.md | 81dc2151 | 6.744 | `4498fcdd1716` | 12 |
| `tv16ob6` | jason-mbp results/_archive/circle-packing/2026-09-15_transcendence-v16-oldbrief-x10/transcendence-v16-oldbrief-6 | transcendence-v16.md | f67e211e | void | `64209a4d5b52` | 0 |
| `none7` | aws results/circle-packing/2026-09-15_circle-6arm-v14-probefix-x10/none-7 | none-v14.md | c16b7ffb | 6.355 | `08e14f9cbf06` | 2 |
| `tv14ob9` | jason-mbp results/circle-packing/2026-09-15_transcendence-v14-oldbrief-x10/transcendence-v14-9 | universalism-v14.md | 81dc2151 | 6.287 | `84be19a746c5` | 12 |
| `smv16_10` | aws results/circle-packing/2026-08-26_circle_smframe_4x10_15it30m/explore-sm-v16-10 | - | - | - | `99ee426845af` | 0 |
| `deon6` | aws results/circle-packing/2026-08-22_circle-register-x10-15it30m/explore-deon-6 | - | - | - | `2ffd019e66c4` | 57 |
| `nalt6` | aws results/circle-packing/2026-08-24_circle-figures-x10-15it30m/nietzsche-alt-6 | - | - | - | `2d59f02c64cc` | 0 |
| `smv16_6` | jason-mbp _archive 2026-08-26_circle-smframe-4x10-15it30m/explore-sm-v16-6 | - | - | - | `b58b0c619a67` | 0 |
| `cvirt7` | jason-mbp _archive 2026-08-22_circle-register-x10-15it30m/conservative-virt-7 | - | - | - | `716aa29ec11f` | 0 |
| `anchor6` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m/anchor-6 | - | - | - | `bab7b669d06a` | 0 |
| `anchor1` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m/anchor-1 | - | - | - | `319de73d2ce7` | 13 |
| `b6x2_1` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m/6bullets-2xlength-1 | - | - | - | `51078242d9e3` | 0 |
| `b9_5` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m-b/9bullets-5 | - | - | - | `7ad13997ec9a` | 2 |
| `plain1` | aws results/circle-packing/2026-08-23_circle-plain-b/explore-plain-1 | - | - | - | `9a9b0f34c82a` | 0 |
| `none3` | aws results/circle-packing/2026-09-15_circle-6arm-v14-probefix-x10/none-3 | none-v14.md | c16b7ffb | 4.656 | `2d43ecf819d6` | 0 |

Per-solver sweep statistics (mean digits vs the live table by N band, unique bests, improvements):

| solver | jobs | ok | feasible | strict | mean digits vs live | n at best (±1e-9) | unique best | improved over warm | beats live | digits 1-25 | digits 26-50 | digits 51-75 | digits 76-100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor1 | 335 | 335 | 335 | 303 | 5.757 | 155 | 0 | 54 | 14 | - | 7.00 | 6.62 | 6.53 |
| anchor6 | 224 | 224 | 224 | 224 | 5.358 | 84 | 0 | 35 | 13 | - | 7.00 | 6.36 | 6.45 |
| b6x2_1 | 247 | 247 | 247 | 247 | 4.696 | 89 | 0 | 4 | 7 | - | 7.00 | 5.40 | 5.22 |
| b9_5 | 301 | 301 | 301 | 301 | 5.672 | 154 | 0 | 35 | 12 | - | 7.00 | 6.49 | 6.40 |
| cvirt7 | 161 | 161 | 161 | 161 | 4.250 | 37 | 0 | 9 | 11 | - | 7.00 | 6.48 | 6.15 |
| deon6 | 685 | 685 | 685 | 685 | 5.768 | 399 | 0 | 37 | 16 | 7.00 | 7.00 | 6.64 | 6.79 |
| nalt6 | 559 | 559 | 559 | 559 | 5.954 | 253 | 0 | 122 | 17 | 7.00 | 7.00 | 6.90 | 6.92 |
| none3 | 224 | 224 | 224 | 224 | 4.706 | 49 | 0 | 17 | 15 | - | 7.00 | 6.39 | 6.73 |
| none7 | 655 | 643 | 655 | 655 | 6.206 | 368 | 0 | 84 | 16 | 7.00 | 7.00 | 6.36 | 6.74 |
| plain1 | 339 | 337 | 339 | 339 | 5.192 | 134 | 0 | 39 | 13 | 7.00 | 7.00 | 6.49 | 6.48 |
| smv16_10 | 326 | 324 | 326 | 326 | 5.392 | 134 | 0 | 29 | 15 | 7.00 | 7.00 | 6.62 | 6.59 |
| smv16_6 | 190 | 190 | 190 | 190 | 4.314 | 0 | 0 | 8 | 12 | - | 7.00 | 6.36 | 6.26 |
| tv14ob9 | 623 | 623 | 623 | 470 | 6.289 | 366 | 9 | 35 | 10 | 7.00 | 7.00 | 6.51 | 5.97 |
| tv14pf6 | 718 | 718 | 718 | 688 | 6.514 | 471 | 3 | 147 | 18 | 7.00 | 7.00 | 7.00 | 7.00 |
| tv16ob6 | 488 | 488 | 488 | 488 | 6.263 | 243 | 0 | 135 | 17 | 7.00 | 7.00 | 6.74 | 6.86 |


## Files

```
pck/csqv<N>.pck     Packomania text format: line 1 = largest radius, line 2 = author, then N lines
                    "x y r" at %.17g (bit-exact doubles), centred square [-0.5, 0.5]^2
json/out<N>.json    sidecar in the corner frame [0,1]^2 (x+0.5, y+0.5, r) with Σr, slacks, provenance
results.csv         n, sum_radii, max_violation, record_20260927, delta, feasible
comparison.md       output of verify_and_compare.py compare --tol 0 against the 2026-09-27 snapshot
manifest.json       per-N recomputed values, sha256 of every pck, provenance, counts
survey_all_runs.md  survey of every packing ever stored by the SI-v2 runs on all three machines:
                    which stored packings beat the current table, and how many independent runs found each
README.md           this file (generated)
```
