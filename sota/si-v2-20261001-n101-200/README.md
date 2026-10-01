# si-v2-20261001-n101-200: SI-v2 solver sweep extended to N = 101..200

**7 WIN** against the Packomania `csqv` table retrieved **2026-10-01 06:07 UTC** (`sota/packomania/history/packomania_csqv_2026-10-01.json`), out of 100 sizes searched.
Only WIN packings (strictly feasible in exact rational arithmetic, `sum_r > record + 1e-9`) are published here.

Extension campaign status at 2026-10-01 06:47 UTC: 53 jobs aggregated from roundE0, roundE1, roundP1.

## How this differs from `si-v2-20260927` (N <= 100)

The N <= 100 sweep was warm-started from packings our own runs had stored. Above N = 100 we had none, so every job in this
extension was warm-started from **Packomania's published coordinates** (`https://www.packomania.com/csqv/txt/csqv<N>.txt`,
12 decimals, re-inflated to strict feasibility by an exact radius LP): `nbr` jobs see the packings of the neighbouring sizes
N±1..4 and build an N-circle start from them; `self` jobs polish the published N-circle packing itself. A result that merely
re-derives the published packing (a "tie") is therefore **not ours to claim and is not exported**; the table below reports how
many sizes ended tied or below, from the sweep's own aggregate.

| status at N = 101..200 (best strict sweep result vs the live table) | sizes |
|---|---:|
| - | 88 |
| BEAT | 7 |
| tie | 5 |

Closest non-winning results (within 1e-5 of the record): 150 (-4.7e-11), 190 (-2.2e-11), 192 (-3.1e-11), 193 (-2.2e-11), 199 (-4.2e-11).

## Wins

| N | ours Σr | record | Δ (abs) | Δ (rel) | found by | record holder |
|---:|---|---|---:|---:|---|---|
| 191 | 7.317574330378 | 7.314938403871 | +2.636e-03 | +3.60e-04 | `hpolish` self seed 0 | Wilfred Heap |
| 194 | 7.376251732102 | 7.376248679483 | +3.053e-06 | +4.14e-07 | `hpolish` self seed 0 | Jean-René Denoual |
| 195 | 7.397110203899 | 7.395054318099 | +2.056e-03 | +2.78e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 196 | 7.417474619912 | 7.414752186446 | +2.722e-03 | +3.67e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 197 | 7.436323901330 | 7.435507140839 | +8.168e-04 | +1.10e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 198 | 7.455453405700 | 7.454135017058 | +1.318e-03 | +1.77e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 200 | 7.494071021855 | 7.493815304616 | +2.557e-04 | +3.41e-05 | `hpolish` self seed 0 | Jean-René Denoual |

## All records beaten, N = 1..200, against the live table (2026-10-01 06:07 UTC)

Combined with `si-v2-20260927` (N <= 100). A row means our published packing exceeds the live Packomania value by more than 1e-9.

| N | ours Σr | live record | margin | live holder |
|---:|---|---|---:|---|
| 63 | 4.155808089352 | 4.155766324151 | +4.2e-05 | Wilfred Heap |
| 66 | 4.256749492869 | 4.255807931448 | +9.4e-04 | Wilfred Heap |
| 78 | 4.636796920045 | 4.636377420432 | +4.2e-04 | Wilfred Heap |
| 82 | 4.756048298932 | 4.755873680213 | +1.7e-04 | Wilfred Heap |
| 83 | 4.786683838811 | 4.786393807284 | +2.9e-04 | Wilfred Heap |
| 94 | 5.103121333926 | 5.102335812493 | +7.9e-04 | Wilfred Heap |
| 96 | 5.155841390333 | 5.155358373110 | +4.8e-04 | Jean-René Denoual |
| 98 | 5.209524986811 | 5.209107205724 | +4.2e-04 | Jean-René Denoual |
| 99 | 5.236903198995 | 5.236421941919 | +4.8e-04 | Wilfred Heap |
| 100 | 5.264237427968 | 5.263989520982 | +2.5e-04 | Jean-René Denoual |
| 191 | 7.317574330378 | 7.314938403871 | +2.6e-03 | Wilfred Heap |
| 194 | 7.376251732102 | 7.376248679483 | +3.1e-06 | Jean-René Denoual |
| 195 | 7.397110203899 | 7.395054318099 | +2.1e-03 | Jean-René Denoual |
| 196 | 7.417474619912 | 7.414752186446 | +2.7e-03 | Jean-René Denoual |
| 197 | 7.436323901330 | 7.435507140839 | +8.2e-04 | Jean-René Denoual |
| 198 | 7.455453405700 | 7.454135017058 | +1.3e-03 | Jean-René Denoual |
| 200 | 7.494071021855 | 7.493815304616 | +2.6e-04 | Jean-René Denoual |

## Files

`pck/csqv<N>.pck` (Packomania text, centred square, %.17g), `json/out<N>.json` (corner-frame sidecar), `results.csv`, `manifest.json`,
`comparison.md` (`verify_and_compare.py compare --tol 0`). Verify any file with `python verify_and_compare.py verify pck/csqv<N>.pck`.
Sweep: 53 jobs over 3 round(s); solvers are the SI-v2 portfolio in `solver-si-v2-20260927/`.
