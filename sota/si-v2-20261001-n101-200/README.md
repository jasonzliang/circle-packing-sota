# si-v2-20261001-n101-200: SI-v2 solver sweep extended to N = 101..200

**17 WIN** against the Packomania `csqv` table retrieved **2026-10-01 07:07 UTC** (`sota/packomania/history/packomania_csqv_2026-10-01.json`), out of 100 sizes searched.
Only WIN packings (strictly feasible in exact rational arithmetic, `sum_r > record + 1e-9`) are published here.

Extension campaign status at 2026-10-01 07:08 UTC: 146 jobs aggregated from roundE0, roundE1, roundP1.

## How this differs from `si-v2-20260927` (N <= 100)

The N <= 100 sweep was warm-started from packings our own runs had stored. Above N = 100 we had none, so every job in this
extension was warm-started from **Packomania's published coordinates** (`https://www.packomania.com/csqv/txt/csqv<N>.txt`,
12 decimals, re-inflated to strict feasibility by an exact radius LP): `nbr` jobs see the packings of the neighbouring sizes
N±1..4 and build an N-circle start from them; `self` jobs polish the published N-circle packing itself. A result that merely
re-derives the published packing (a "tie") is therefore **not ours to claim and is not exported**; the table below reports how
many sizes ended tied or below, from the sweep's own aggregate.

| status at N = 101..200 (best strict sweep result vs the live table) | sizes |
|---|---:|
| - | 76 |
| BEAT | 17 |
| tie | 7 |

Closest non-winning results (within 1e-5 of the record): 150 (-4.7e-11), 178 (-3.1e-11), 181 (-4.2e-11), 184 (-3.5e-11), 186 (-2.3e-11), 187 (-2.6e-11), 199 (-4.2e-11).

## Wins

| N | ours Σr | record | Δ (abs) | Δ (rel) | found by | record holder |
|---:|---|---|---:|---:|---|---|
| 179 | 7.081774948878 | 7.080903944143 | +8.710e-04 | +1.23e-04 | `tv16ob6` self seed 7101 | Jean-René Denoual |
| 180 | 7.100584660995 | 7.099941756488 | +6.429e-04 | +9.06e-05 | `tv16ob6` self seed 7101 | Jean-René Denoual |
| 182 | 7.138717442089 | 7.138696832806 | +2.061e-05 | +2.89e-06 | `hpolish` self seed 0 | Wilfred Heap |
| 183 | 7.158914812347 | 7.158659503357 | +2.553e-04 | +3.57e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 185 | 7.198651480801 | 7.197625081950 | +1.026e-03 | +1.43e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 188 | 7.257784502262 | 7.256806779802 | +9.777e-04 | +1.35e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 189 | 7.277955287382 | 7.277259920003 | +6.954e-04 | +9.56e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 190 | 7.298004709019 | 7.297283358464 | +7.214e-04 | +9.89e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 191 | 7.317709106344 | 7.314938403871 | +2.771e-03 | +3.79e-04 | `hpolish` self seed 0 | Wilfred Heap |
| 192 | 7.337281474317 | 7.336598450128 | +6.830e-04 | +9.31e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 193 | 7.357357788547 | 7.357305320712 | +5.247e-05 | +7.13e-06 | `hpolish` self seed 0 | Jean-René Denoual |
| 194 | 7.377183819368 | 7.376248679483 | +9.351e-04 | +1.27e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 195 | 7.397295373865 | 7.395054318099 | +2.241e-03 | +3.03e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 196 | 7.417474619912 | 7.414752186446 | +2.722e-03 | +3.67e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 197 | 7.436323901330 | 7.435507140839 | +8.168e-04 | +1.10e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 198 | 7.456737445837 | 7.454135017058 | +2.602e-03 | +3.49e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 200 | 7.494071021855 | 7.493815304616 | +2.557e-04 | +3.41e-05 | `hpolish` self seed 0 | Jean-René Denoual |

## All records beaten, N = 1..200, against the live table (2026-10-01 07:07 UTC)

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
| 179 | 7.081774948878 | 7.080903944143 | +8.7e-04 | Jean-René Denoual |
| 180 | 7.100584660995 | 7.099941756488 | +6.4e-04 | Jean-René Denoual |
| 182 | 7.138717442089 | 7.138696832806 | +2.1e-05 | Wilfred Heap |
| 183 | 7.158914812347 | 7.158659503357 | +2.6e-04 | Wilfred Heap |
| 185 | 7.198651480801 | 7.197625081950 | +1.0e-03 | Jean-René Denoual |
| 188 | 7.257784502262 | 7.256806779802 | +9.8e-04 | Jean-René Denoual |
| 189 | 7.277955287382 | 7.277259920003 | +7.0e-04 | Jean-René Denoual |
| 190 | 7.298004709019 | 7.297283358464 | +7.2e-04 | Jean-René Denoual |
| 191 | 7.317709106344 | 7.314938403871 | +2.8e-03 | Wilfred Heap |
| 192 | 7.337281474317 | 7.336598450128 | +6.8e-04 | Jean-René Denoual |
| 193 | 7.357357788547 | 7.357305320712 | +5.2e-05 | Jean-René Denoual |
| 194 | 7.377183819368 | 7.376248679483 | +9.4e-04 | Jean-René Denoual |
| 195 | 7.397295373865 | 7.395054318099 | +2.2e-03 | Jean-René Denoual |
| 196 | 7.417474619912 | 7.414752186446 | +2.7e-03 | Jean-René Denoual |
| 197 | 7.436323901330 | 7.435507140839 | +8.2e-04 | Jean-René Denoual |
| 198 | 7.456737445837 | 7.454135017058 | +2.6e-03 | Jean-René Denoual |
| 200 | 7.494071021855 | 7.493815304616 | +2.6e-04 | Jean-René Denoual |

## Files

`pck/csqv<N>.pck` (Packomania text, centred square, %.17g), `json/out<N>.json` (corner-frame sidecar), `results.csv`, `manifest.json`,
`comparison.md` (`verify_and_compare.py compare --tol 0`). Verify any file with `python verify_and_compare.py verify pck/csqv<N>.pck`.
Sweep: 146 jobs over 3 round(s); solvers are the SI-v2 portfolio in `solver-si-v2-20260927/`.
