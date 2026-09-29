# si-v2-20260927: best packing per N from the SI-v2 solver portfolio sweep

**100 packings (N = 1..100)**, the best strictly feasible packing per N found by sweeping fifteen
LLM-evolved circle-packing solvers from the self-improvement-v2 (SI-v2) `circle-packing` mission, seeded
from every packing those runs ever stored. Compared against the Packomania `csqv` table
**retrieved 2026-09-27** (`sota/packomania/history/packomania_csqv_2026-09-27.json`): **19 WIN / 81 tie / 0 below**.

Sweep campaign status at 2026-09-29 11:39 UTC: 24544 jobs aggregated from phase1, roundA, roundB, roundC, roundD, roundE, roundF, roundG, roundH, roundI, roundJ, roundK, roundL, roundS0, roundS1, roundS2, roundS3, roundS4, roundS5, roundS6, roundS7, roundT1, roundT2, roundT3, roundU1.

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
| 63 | 4.155808089351 | 4.155766324151 | +4.177e-05 | +1.00e-05 | `sm7` self seed 0 |
| 66 | 4.256749492862 | 4.255807931448 | +9.416e-04 | +2.21e-04 | `tv14pf6` nbr seed 12 |
| 78 | 4.636796919967 | 4.636377420432 | +4.195e-04 | +9.05e-05 | `tv16ob6` nbr seed 0 |
| 79 | 4.666928346063 | 4.666466802554 | +4.615e-04 | +9.89e-05 | `b9_5` self seed 1 |
| 80 | 4.695876897572 | 4.695590667948 | +2.862e-04 | +6.10e-05 | `anchor6` self seed 1 |
| 82 | 4.756048298923 | 4.755873680213 | +1.746e-04 | +3.67e-05 | `anchor6` self seed 0 |
| 83 | 4.786683838803 | 4.786393807284 | +2.900e-04 | +6.06e-05 | `anchor6` self seed 0 |
| 84 | 4.816567858205 | 4.816500879756 | +6.698e-05 | +1.39e-05 | `anchor6` self seed 0 |
| 85 | 4.848035445604 | 4.847351780582 | +6.837e-04 | +1.41e-04 | `anchor1` self seed 0 |
| 89 | 4.967596361095 | 4.966850487024 | +7.459e-04 | +1.50e-04 | `anchor6` self seed 0 |
| 92 | 5.049048324651 | 5.048761669590 | +2.867e-04 | +5.68e-05 | `b6x2_1` self seed 0 |
| 93 | 5.076445594003 | 5.076245164998 | +2.004e-04 | +3.95e-05 | `deon6` self seed 5 |
| 94 | 5.103121333917 | 5.101975416147 | +1.146e-03 | +2.25e-04 | `anchor6` self seed 0 |
| 95 | 5.129349439039 | 5.129111105835 | +2.383e-04 | +4.65e-05 | `tv14pf6` nbr seed 162 |
| 96 | 5.155841390323 | 5.155289947655 | +5.514e-04 | +1.07e-04 | `tv14pf6` nbr seed 167 |
| 97 | 5.182315239429 | 5.181609663002 | +7.056e-04 | +1.36e-04 | `b9_5` self seed 5 |
| 98 | 5.209524986801 | 5.208578422017 | +9.466e-04 | +1.82e-04 | `tv14pf6` nbr seed 20 |
| 99 | 5.236903198990 | 5.236244728577 | +6.585e-04 | +1.26e-04 | `b9_5` self seed 1 |
| 100 | 5.264237427961 | 5.262787458777 | +1.450e-03 | +2.76e-04 | `anchor1` self seed 0 |

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
| `tv14pf6` | jason-mbp results/_archive/circle-packing/2026-09-15_transcendence-v14-probefix-x10/transcendence-v14-probefix-6 | universalism-v14.md | 81dc2151 | 6.744 | `4498fcdd1716` | 6 |
| `tv16ob6` | jason-mbp results/_archive/circle-packing/2026-09-15_transcendence-v16-oldbrief-x10/transcendence-v16-oldbrief-6 | transcendence-v16.md | f67e211e | void | `64209a4d5b52` | 1 |
| `none7` | aws results/circle-packing/2026-09-15_circle-6arm-v14-probefix-x10/none-7 | none-v14.md | c16b7ffb | 6.355 | `08e14f9cbf06` | 0 |
| `tv14ob9` | jason-mbp results/circle-packing/2026-09-15_transcendence-v14-oldbrief-x10/transcendence-v14-9 | universalism-v14.md | 81dc2151 | 6.287 | `84be19a746c5` | 0 |
| `smv16_10` | aws results/circle-packing/2026-08-26_circle_smframe_4x10_15it30m/explore-sm-v16-10 | - | - | - | `99ee426845af` | 0 |
| `deon6` | aws results/circle-packing/2026-08-22_circle-register-x10-15it30m/explore-deon-6 | - | - | - | `2ffd019e66c4` | 45 |
| `nalt6` | aws results/circle-packing/2026-08-24_circle-figures-x10-15it30m/nietzsche-alt-6 | - | - | - | `2d59f02c64cc` | 0 |
| `smv16_6` | jason-mbp _archive 2026-08-26_circle-smframe-4x10-15it30m/explore-sm-v16-6 | - | - | - | `b58b0c619a67` | 0 |
| `cvirt7` | jason-mbp _archive 2026-08-22_circle-register-x10-15it30m/conservative-virt-7 | - | - | - | `716aa29ec11f` | 0 |
| `anchor6` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m/anchor-6 | - | - | - | `bab7b669d06a` | 14 |
| `anchor1` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m/anchor-1 | - | - | - | `319de73d2ce7` | 16 |
| `b6x2_1` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m/6bullets-2xlength-1 | - | - | - | `51078242d9e3` | 1 |
| `b9_5` | jason-mbp _archive 2026-08-19_circle-bulletcount-x10-12it30m-b/9bullets-5 | - | - | - | `7ad13997ec9a` | 4 |
| `plain1` | aws results/circle-packing/2026-08-23_circle-plain-b/explore-plain-1 | - | - | - | `9a9b0f34c82a` | 0 |
| `none3` | aws results/circle-packing/2026-09-15_circle-6arm-v14-probefix-x10/none-3 | none-v14.md | c16b7ffb | 4.656 | `2d43ecf819d6` | 0 |
| `uv15pf7` | universalism-v15-probefix-7 (jason-mbp _archive 2026-09-14_universalism-v15-probefix-x10) | - | - | - | `754172099953` | 0 |
| `tv14ob1` | transcendence-v14-1 (jason-mbp 2026-09-15_transcendence-v14-oldbrief-x10) | - | - | - | `03a93725a0c0` | 0 |
| `tv16pf2` | transcendence-v16-probefix-2 (jason-mbp _archive 2026-09-14_transcendence-v16-probefix-x10) | - | - | - | `0ae3994e6645` | 0 |
| `tv16pf1` | transcendence-v16-probefix-1 (jason-mbp _archive 2026-09-14_transcendence-v16-probefix-x10) | - | - | - | `66f0ddd1dad0` | 0 |
| `tv14pf9` | transcendence-v14-probefix-9 (jason-mbp _archive 2026-09-15_transcendence-v14-probefix-x10) | - | - | - | `20e1ea5e8f02` | 0 |
| `deon10` | explore-deon-10 (aws 2026-08-22_circle-register-x10-15it30m) | - | - | - | `002595ff8cbd` | 0 |
| `sm2` | explore-sm-2 (aws 2026-08-17_circle-sm-x10-20it30m) | - | - | - | `054deb4d6118` | 0 |
| `sm7` | explore-sm-7 (aws 2026-08-17_circle-sm-x10-20it30m) | - | - | - | `47ac76f6d3fd` | 2 |
| `v2enh5` | circle-packing-v2 enhancement-v15-5 (aws 2026-09-16_circle-v2-enh15-x10) | - | - | - | `b94f3cc37cbf` | 0 |
| `v2exp8` | circle-packing-v2 explore-8 (aws 2026-09-16_circle-v2-8arm-x10) | - | - | - | `38618042a9f2` | 0 |
| `smr27` | circle-packing-sota solver-sm-radical-v6-n27 pack.search (adapter) | - | - | - | `82182c73b8e2` | 0 |
| `smr54` | circle-packing-sota solver-sm-radical-v6-n54 endgame/pipeline (adapter) | - | - | - | `54d263b9567a` | 0 |
| `expl1_08276armv14` | main:2026-08-27_circle-6arm-v14-x10-15it30m/explore-1 | - | - | - | `a13b86908d6c` | 0 |
| `tranv16prob6_0914transcendembp` | _mbp:2026-09-14_transcendence-v16-probefix-x10/transcendence-v16-probefix-6 | - | - | - | `9e30ccf49e51` | 0 |
| `hint2_0818hintv10` | main:2026-08-18_circle-hint-v10-x10-12it30m/hint-2 | - | - | - | `5b81d9292514` | 0 |
| `explutil8_0822register` | main:2026-08-22_circle-register-x10-15it30m/explore-util-8 | - | - | - | `58ce5edc7ef4` | 0 |
| `expl10_08172v` | main:2026-08-17_circle-2v-x15-20it30m/explore-10 | - | - | - | `fad4b82c5ecf` | 0 |
| `mast1_08276armv14` | main:2026-08-27_circle-6arm-v14-x10-15it30m/mastery-1 | - | - | - | `8792c0b7095d` | 0 |
| `9bul2_0819bulletcoun` | main:2026-08-19_circle-bulletcount-x10-12it30m-b/9bullets-2 | - | - | - | `dd3741def88d` | 0 |
| `m_ove5_0817` | jason-mbp:2026-08-17_circle-6v-x5-20it30m-b/overfit-5 | - | - | - | `1875b6a54bc8` | 0 |
| `m_mac4_0824` | jason-mbp:2026-08-24_circle-figures-x10-15it30m/machiavelli-4 | - | - | - | `1f982cb50225` | 0 |
| `m_hin9_0818` | jason-mbp:2026-08-18_circle-hint-v10-x10-12it30m/hint-9 | - | - | - | `5f4e69e2f30f` | 0 |
| `consplai9_0823plainb` | main:2026-08-23_circle-plain-b/conservative-plain-9 | - | - | - | `ae1ad849a862` | 1 |
| `9bul6_0819bulletcoun` | main:2026-08-19_circle-bulletcount-x10-12it30m-b/9bullets-6 | - | - | - | `98d355675bc4` | 0 |
| `conssmv141_0826smframe` | main:2026-08-26_circle_smframe_4x10_15it30m/conservative-sm-v14-1 | - | - | - | `44004fa5e17d` | 0 |
| `expl6_09156armv14pro` | main:2026-09-15_circle-6arm-v14-probefix-x10/explore-6 | - | - | - | `ea126a00f444` | 0 |
| `expl7_08276armv14` | main:2026-08-27_circle-6arm-v14-x10-15it30m/explore-7 | - | - | - | `4995ce29c252` | 0 |
| `nietsmv148_0828nietzscheg` | main:2026-08-28_circle-nietzsche-gen/nietzsche-sm-v14-8 | - | - | - | `e29df4548462` | 0 |
| `univv15prob3_0914universalimbp` | _mbp:2026-09-14_universalism-v15-probefix-x10/universalism-v15-probefix-3 | - | - | - | `012351fbbf5b` | 0 |
| `m_enhv1410_0916` | jason-mbp:2026-09-16_circle-v2-8arm-x10/enhancement-v14-10 | - | - | - | `8edc94ce6a6c` | 0 |
| `m_6bu5_0819` | jason-mbp:2026-08-19_circle-bulletcount-x10-12it30m/6bullets-5 | - | - | - | `fc587f9a74e0` | 0 |
| `oversm5_08176v` | main:2026-08-17_circle-6v-x5-20it30m-b/overfit-sm-5 | - | - | - | `9f0a11d5183f` | 0 |
| `9bul1_0819bulletcoun` | main:2026-08-19_circle-bulletcount-x10-12it30m-b/9bullets-1 | - | - | - | `74754ccb8ee5` | 0 |
| `univv15prob2_0914universalimbp` | _mbp:2026-09-14_universalism-v15-probefix-x10/universalism-v15-probefix-2 | - | - | - | `6d17a62099c4` | 0 |
| `mach1_0824figures` | main:2026-08-24_circle-figures-x10-15it30m/machiavelli-1 | - | - | - | `b5a9ed673c08` | 0 |
| `expl8_08276armv14` | main:2026-08-27_circle-6arm-v14-x10-15it30m/explore-8 | - | - | - | `4e8eeda23577` | 0 |
| `nietv157_0828nietzscheg` | main:2026-08-28_circle-nietzsche-gen/nietzsche-v15-7 | - | - | - | `e766309057e9` | 0 |
| `m_6bu4_0819` | jason-mbp:2026-08-19_circle-bulletcount-x10-12it30m/6bullets-4 | - | - | - | `dc0bd8567442` | 0 |
| `m_9bu3_0819` | jason-mbp:2026-08-19_circle-bulletcount-x10-12it30m-b/9bullets-3 | - | - | - | `c0c4d421caf5` | 0 |
| `topo1` | built 2026-09-29 for the strategy rotation: defect migration + contact-graph tabu + population basin hopping | - | - | - | `70af56e29b1a` | 0 |

Per-solver sweep statistics (mean digits vs the live table by N band, unique bests, improvements):

| solver | jobs | ok | feasible | strict | mean digits vs live | n at best (±1e-9) | unique best | improved over warm | beats live | digits 1-25 | digits 26-50 | digits 51-75 | digits 76-100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9bul1_0819bulletcoun | 184 | 184 | 184 | 184 | 5.354 | 59 | 0 | 1 | 0 | - | 6.46 | 7.00 | - |
| 9bul2_0819bulletcoun | 184 | 184 | 184 | 184 | 5.625 | 67 | 0 | 1 | 0 | - | 6.59 | 7.00 | - |
| 9bul6_0819bulletcoun | 402 | 402 | 402 | 402 | 5.983 | 11 | 0 | 18 | 10 | - | 7.00 | 7.00 | 5.81 |
| anchor1 | 557 | 557 | 557 | 525 | 5.362 | 184 | 0 | 75 | 16 | - | 7.00 | 6.62 | 6.73 |
| anchor6 | 798 | 798 | 798 | 798 | 5.150 | 243 | 0 | 94 | 16 | - | 6.62 | 6.53 | 6.91 |
| b6x2_1 | 429 | 429 | 429 | 429 | 4.261 | 96 | 0 | 5 | 7 | - | 7.00 | 5.55 | 5.30 |
| b9_5 | 448 | 448 | 448 | 448 | 5.317 | 173 | 0 | 43 | 14 | - | 7.00 | 6.49 | 6.60 |
| consplai9_0823plainb | 583 | 583 | 583 | 495 | 6.023 | 281 | 0 | 90 | 19 | - | 7.00 | 7.00 | 6.63 |
| conssmv141_0826smframe | 298 | 298 | 298 | 298 | 5.614 | 116 | 0 | 1 | 0 | - | 6.82 | 7.00 | - |
| cvirt7 | 370 | 370 | 370 | 370 | 4.051 | 52 | 0 | 27 | 16 | - | 7.00 | 7.00 | 6.60 |
| deon10 | 476 | 476 | 476 | 476 | 5.798 | 225 | 0 | 38 | 19 | - | 6.86 | 7.00 | 7.00 |
| deon6 | 837 | 837 | 837 | 837 | 5.348 | 413 | 0 | 40 | 16 | 7.00 | 7.00 | 6.64 | 6.79 |
| expl10_08172v | 298 | 298 | 298 | 298 | 5.779 | 124 | 0 | 1 | 0 | - | 6.87 | 7.00 | - |
| expl1_08276armv14 | 402 | 402 | 402 | 402 | 6.157 | 200 | 0 | 19 | 11 | - | 6.97 | 7.00 | 5.93 |
| expl6_09156armv14pro | 298 | 298 | 298 | 298 | 5.694 | 107 | 0 | 2 | 0 | - | 6.72 | 7.00 | - |
| expl7_08276armv14 | 265 | 265 | 265 | 265 | 5.122 | 0 | 0 | 1 | 0 | - | 6.70 | 7.00 | - |
| expl8_08276armv14 | 265 | 265 | 265 | 265 | 5.925 | 0 | 0 | 1 | 0 | - | 6.97 | 7.00 | - |
| explutil8_0822register | 287 | 287 | 287 | 287 | 5.835 | 121 | 0 | 3 | 0 | - | 6.86 | 7.00 | - |
| hint2_0818hintv10 | 287 | 287 | 287 | 286 | 5.846 | 117 | 0 | 1 | 0 | - | 6.74 | 7.00 | - |
| m_6bu4_0819 | 223 | 223 | 223 | 223 | 5.546 | 80 | 0 | 1 | 0 | - | 6.71 | 7.00 | - |
| m_6bu5_0819 | 184 | 184 | 184 | 184 | 5.726 | 0 | 0 | 2 | 0 | - | 6.86 | 7.00 | - |
| m_9bu3_0819 | 298 | 298 | 298 | 140 | 5.503 | 118 | 0 | 1 | 0 | - | 7.00 | 7.00 | - |
| m_enhv1410_0916 | 287 | 287 | 287 | 287 | 6.034 | 134 | 0 | 2 | 0 | - | 7.00 | 7.00 | - |
| m_hin9_0818 | 265 | 265 | 265 | 265 | 5.435 | 87 | 0 | 1 | 0 | - | 6.57 | 7.00 | - |
| m_mac4_0824 | 298 | 298 | 298 | 0 | 5.741 | 127 | 0 | 2 | 0 | - | 6.82 | 7.00 | - |
| m_ove5_0817 | 412 | 412 | 412 | 412 | 5.957 | 208 | 0 | 2 | 0 | - | 6.67 | 7.00 | - |
| mach1_0824figures | 316 | 316 | 316 | 222 | 5.789 | 126 | 0 | 4 | 0 | - | 6.88 | 7.00 | - |
| mast1_08276armv14 | 287 | 287 | 287 | 287 | 5.903 | 124 | 0 | 3 | 0 | - | 6.73 | 7.00 | - |
| nalt6 | 1605 | 1605 | 1605 | 1605 | 5.843 | 630 | 0 | 308 | 19 | 7.00 | 7.00 | 7.00 | 7.00 |
| nietsmv148_0828nietzscheg | 223 | 223 | 223 | 223 | 5.537 | 81 | 0 | 2 | 0 | - | 6.75 | 7.00 | - |
| nietv157_0828nietzscheg | 223 | 223 | 223 | 223 | 5.176 | 70 | 0 | 3 | 0 | - | 6.52 | 7.00 | - |
| none3 | 453 | 453 | 453 | 453 | 4.405 | 78 | 0 | 30 | 16 | - | 7.00 | 6.52 | 6.86 |
| none7 | 952 | 940 | 952 | 952 | 5.845 | 440 | 0 | 135 | 17 | 7.00 | 7.00 | 6.36 | 6.86 |
| oversm5_08176v | 298 | 298 | 298 | 298 | 5.437 | 93 | 0 | 1 | 0 | - | 6.52 | 7.00 | - |
| plain1 | 585 | 583 | 585 | 585 | 4.855 | 179 | 0 | 47 | 13 | 7.00 | 7.00 | 6.88 | 6.48 |
| sm2 | 779 | 779 | 779 | 773 | 5.330 | 304 | 0 | 102 | 19 | - | 7.00 | 7.00 | 7.00 |
| sm7 | 278 | 278 | 278 | 278 | 5.273 | 76 | 0 | 59 | 19 | - | - | 7.00 | 7.00 |
| smr27 | 108 | 88 | 88 | 79 | 4.912 | 48 | 0 | 18 | 19 | - | - | 7.00 | 7.00 |
| smr54 | 117 | 117 | 117 | 108 | 4.631 | 49 | 0 | 18 | 19 | - | - | 7.00 | 7.00 |
| smv16_10 | 498 | 496 | 498 | 498 | 5.082 | 169 | 0 | 52 | 18 | 7.00 | 7.00 | 6.88 | 7.00 |
| smv16_6 | 404 | 404 | 404 | 404 | 3.891 | 0 | 0 | 13 | 14 | - | 7.00 | 6.36 | 6.46 |
| tranv16prob6_0914transcendembp | 265 | 265 | 265 | 265 | 5.623 | 101 | 0 | 3 | 0 | - | 6.88 | 7.00 | - |
| tv14ob1 | 121 | 121 | 121 | 121 | 5.034 | 49 | 0 | 18 | 19 | - | - | 7.00 | 7.00 |
| tv14ob9 | 1043 | 1043 | 1043 | 849 | 5.682 | 421 | 9 | 59 | 11 | 7.00 | 7.00 | 6.65 | 6.26 |
| tv14pf6 | 2878 | 2878 | 2878 | 2848 | 6.384 | 1758 | 0 | 731 | 19 | 7.00 | 7.00 | 7.00 | 7.00 |
| tv14pf9 | 116 | 116 | 116 | 110 | 5.234 | 50 | 0 | 18 | 19 | - | - | 7.00 | 7.00 |
| tv16ob6 | 1833 | 1833 | 1833 | 1833 | 6.178 | 888 | 0 | 453 | 19 | 7.00 | 7.00 | 7.00 | 7.00 |
| tv16pf1 | 109 | 109 | 109 | 108 | 5.154 | 46 | 0 | 17 | 19 | - | - | 7.00 | 7.00 |
| tv16pf2 | 121 | 121 | 121 | 120 | 5.432 | 45 | 0 | 18 | 19 | - | - | 7.00 | 7.00 |
| univv15prob2_0914universalimbp | 30 | 30 | 30 | 30 | 5.175 | 13 | 0 | 1 | 0 | - | 4.03 | 7.00 | - |
| univv15prob3_0914universalimbp | 30 | 30 | 30 | 30 | 5.379 | 14 | 0 | 1 | 0 | - | 4.09 | 7.00 | - |
| uv15pf7 | 100 | 100 | 100 | 100 | 5.531 | 47 | 0 | 18 | 19 | - | - | 7.00 | 7.00 |
| v2enh5 | 1021 | 1021 | 1021 | 1021 | 5.951 | 545 | 0 | 150 | 19 | - | 7.00 | 7.00 | 7.00 |
| v2exp8 | 116 | 116 | 116 | 0 | 5.108 | 47 | 0 | 17 | 19 | - | - | 7.00 | 7.00 |


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
