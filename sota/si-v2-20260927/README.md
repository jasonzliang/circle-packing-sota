# si-v2-20260927: best packing per N from the SI-v2 solver portfolio sweep

**78 packings (N = 1..100)**, the best strictly feasible packing per N found by sweeping fifteen
LLM-evolved circle-packing solvers from the self-improvement-v2 (SI-v2) `circle-packing` mission, seeded
from every packing those runs ever stored. Compared against the Packomania `csqv` table
**retrieved 2026-09-27** (`sota/packomania/history/packomania_csqv_2026-09-27.json`): **17 WIN / 56 tie / 5 below**.

Sweep campaign status at 2026-09-27 12:19 UTC: 941 jobs aggregated from phase1, roundA.

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
| 78 | 4.636796919967 | 4.636377420432 | +4.195e-04 | +9.05e-05 | `tv16ob6` nbr seed 0 |
| 79 | 4.666928345989 | 4.666466802554 | +4.615e-04 | +9.89e-05 | `tv16ob6` self seed 0 |
| 80 | 4.695845998678 | 4.695590667948 | +2.553e-04 | +5.44e-05 | `deon6` self seed 0 |
| 82 | 4.756048298931 | 4.755873680213 | +1.746e-04 | +3.67e-05 | `deon6` self seed 0 |
| 83 | 4.786683838811 | 4.786393807284 | +2.900e-04 | +6.06e-05 | `deon6` self seed 0 |
| 84 | 4.816567858213 | 4.816500879756 | +6.698e-05 | +1.39e-05 | `deon6` self seed 0 |
| 85 | 4.848035445623 | 4.847351780582 | +6.837e-04 | +1.41e-04 | `deon6` self seed 0 |
| 89 | 4.967596361102 | 4.966850487024 | +7.459e-04 | +1.50e-04 | `deon6` self seed 0 |
| 92 | 5.049048324656 | 5.048761669590 | +2.867e-04 | +5.68e-05 | `deon6` self seed 0 |
| 93 | 5.076445593073 | 5.076245164998 | +2.004e-04 | +3.95e-05 | `nalt6` nbr seed 0 |
| 94 | 5.103121333925 | 5.101975416147 | +1.146e-03 | +2.25e-04 | `deon6` self seed 0 |
| 95 | 5.129114920508 | 5.129111105835 | +3.815e-06 | +7.44e-07 | `none7` nbr seed 0 |
| 96 | 5.155341141002 | 5.155289947655 | +5.119e-05 | +9.93e-06 | `nalt6` self seed 0 |
| 97 | 5.182308473600 | 5.181609663002 | +6.988e-04 | +1.35e-04 | `tv14ob9` nbr seed 0 |
| 98 | 5.209385523635 | 5.208578422017 | +8.071e-04 | +1.55e-04 | `nalt6` self seed 0 |
| 99 | 5.236903198004 | 5.236244728577 | +6.585e-04 | +1.26e-04 | `nalt6` self seed 0 |
| 100 | 5.264237427967 | 5.262787458777 | +1.450e-03 | +2.76e-04 | `deon6` self seed 0 |

## Sizes still below the table

| N | ours Σr | record | Δ |
|---:|---|---|---:|
| 62 | 4.122816770019 | 4.123334225100 | -5.175e-04 |
| 63 | 4.155192697703 | 4.155766324151 | -5.736e-04 |
| 64 | 4.189631472744 | 4.190585830354 | -9.544e-04 |
| 66 | 4.254850507574 | 4.255807931448 | -9.574e-04 |
| 68 | 4.324028521190 | 4.324531544548 | -5.030e-04 |

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
| `tv14pf6` | jason-mbp results/_archive/circle-packing/2026-09-15_transcendence-v14-probefix-x10/transcendence-v14-probefix-6 | universalism-v14.md | 81dc2151 | 6.744 | `4498fcdd1716` | 0 |
| `tv16ob6` | jason-mbp results/_archive/circle-packing/2026-09-15_transcendence-v16-oldbrief-x10/transcendence-v16-oldbrief-6 | transcendence-v16.md | f67e211e | void | `64209a4d5b52` | 2 |
| `none7` | aws results/circle-packing/2026-09-15_circle-6arm-v14-probefix-x10/none-7 | none-v14.md | c16b7ffb | 6.355 | `08e14f9cbf06` | 2 |
| `tv14ob9` | jason-mbp results/circle-packing/2026-09-15_transcendence-v14-oldbrief-x10/transcendence-v14-9 | universalism-v14.md | 81dc2151 | 6.287 | `84be19a746c5` | 1 |
| `smv16_10` | aws results/circle-packing/2026-08-26_circle_smframe_4x10_15it30m/explore-sm-v16-10 | - | - | - | `99ee426845af` | 0 |
| `deon6` | aws results/circle-packing/2026-08-22_circle-register-x10-15it30m/explore-deon-6 | - | - | - | `2ffd019e66c4` | 13 |
| `nalt6` | aws results/circle-packing/2026-08-24_circle-figures-x10-15it30m/nietzsche-alt-6 | - | - | - | `2d59f02c64cc` | 4 |
| `smv16_6` | jason-mbp _archive 2026-08-26_circle-smframe-4x10-15it30m/explore-sm-v16-6 | - | - | - | `b58b0c619a67` | 0 |
| `cvirt7` | jason-mbp _archive 2026-08-22_circle-register-x10-15it30m/conservative-virt-7 | - | - | - | `716aa29ec11f` | 0 |
| `anchor6` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m/anchor-6 | - | - | - | `bab7b669d06a` | 0 |
| `anchor1` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m/anchor-1 | - | - | - | `319de73d2ce7` | 6 |
| `b6x2_1` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m/6bullets-2xlength-1 | - | - | - | `51078242d9e3` | 0 |
| `b9_5` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m-b/9bullets-5 | - | - | - | `7ad13997ec9a` | 1 |
| `plain1` | aws results/circle-packing/2026-08-23_circle-plain-b/explore-plain-1 | - | - | - | `9a9b0f34c82a` | 0 |
| `none3` | aws results/circle-packing/2026-09-15_circle-6arm-v14-probefix-x10/none-3 | none-v14.md | c16b7ffb | 4.656 | `2d43ecf819d6` | 1 |

Per-solver sweep statistics (mean digits vs the live table by N band, unique bests, improvements):

| solver | jobs | ok | feasible | strict | mean digits vs live | n at best (±1e-9) | unique best | improved over warm | beats live | digits 1-25 | digits 26-50 | digits 51-75 | digits 76-100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor1 | 62 | 62 | 62 | 60 | 5.229 | 19 | 0 | 8 | 12 | - | - | 7.00 | 6.40 |
| anchor6 | 62 | 62 | 62 | 62 | 5.148 | 22 | 0 | 9 | 12 | - | - | 7.00 | 6.28 |
| b6x2_1 | 62 | 62 | 62 | 62 | 4.400 | 13 | 0 | 4 | 7 | - | - | 5.02 | 5.22 |
| b9_5 | 62 | 62 | 62 | 62 | 5.036 | 21 | 0 | 9 | 12 | - | - | 7.00 | 6.26 |
| cvirt7 | 62 | 62 | 62 | 62 | 4.613 | 16 | 0 | 4 | 11 | - | - | 7.00 | 6.15 |
| deon6 | 62 | 62 | 62 | 62 | 4.564 | 19 | 0 | 6 | 11 | - | - | 7.00 | 6.07 |
| nalt6 | 62 | 62 | 62 | 62 | 5.902 | 30 | 4 | 17 | 15 | - | - | 7.00 | 6.68 |
| none3 | 62 | 62 | 62 | 62 | 5.060 | 21 | 0 | 8 | 12 | - | - | 7.00 | 6.40 |
| none7 | 64 | 64 | 64 | 64 | 5.443 | 26 | 0 | 10 | 13 | - | - | 7.00 | 6.37 |
| plain1 | 62 | 62 | 62 | 62 | 5.151 | 23 | 0 | 8 | 12 | - | - | 7.00 | 6.33 |
| smv16_10 | 62 | 62 | 62 | 62 | 5.154 | 22 | 0 | 8 | 12 | - | - | 7.00 | 6.30 |
| smv16_6 | 64 | 64 | 64 | 64 | 4.881 | 0 | 0 | 6 | 12 | - | - | 7.00 | 6.26 |
| tv14ob9 | 63 | 63 | 63 | 60 | 5.405 | 14 | 1 | 7 | 9 | - | - | 6.99 | 5.83 |
| tv14pf6 | 66 | 66 | 66 | 66 | 5.693 | 28 | 0 | 11 | 13 | - | - | 7.00 | 6.33 |
| tv16ob6 | 64 | 64 | 64 | 64 | 5.865 | 32 | 2 | 13 | 13 | - | - | 7.00 | 6.42 |


## Files

```
pck/csqv<N>.pck     Packomania text format: line 1 = largest radius, line 2 = author, then N lines
                    "x y r" at %.17g (bit-exact doubles), centred square [-0.5, 0.5]^2
json/out<N>.json    sidecar in the corner frame [0,1]^2 (x+0.5, y+0.5, r) with Σr, slacks, provenance
results.csv         n, sum_radii, max_violation, record_20260927, delta, feasible
comparison.md       output of verify_and_compare.py compare --tol 0 against the 2026-09-27 snapshot
manifest.json       per-N recomputed values, sha256 of every pck, provenance, counts
README.md           this file (generated)
```
