# sota — Packomania reference records (the SOTA to beat)

`packomania_csqv_records.csv` holds the best-known **sum of radii** for packing **N variable-sized circles
in a unit square** (maximize Σr), for N = 1..100.

- **Source:** https://www.packomania.com/csqv/csqv.html — the `csqv` table.
- **Maintainer:** Dr. Eckard Specht, Otto-von-Guericke-Universität Magdeburg.
- **Fetched:** 2026-08-03 (the table was last refreshed 01-Aug-2026).
- These are **best-known** values, **not proven optima**.

## Verification (why we trust these numbers)

The values were extracted via an automated fetch, so they were cross-checked four independent ways
(all passed):

1. **Monotonic + smooth:** strictly increasing in N with smoothly decreasing increments (0.2105 at N=3
   down to 0.0223 at N=91); no non-monotone entries, no increment anomalies. A mis-transcribed digit
   would almost certainly break this.
2. **Analytic exact values:** N=1 = 0.5 (one inscribed circle); N=2 = 0.585786437626 vs the exact
   2−√2 = 0.585786437627 (matches to 9e-13); N=26 = 2.635983085 matches the literature value 2.635983.
3. **Independent solver agreement:** our from-scratch solver reproduces 26 of these records to ~1e-11.
4. **Independent re-extraction:** a second fetch of 8 scattered N (3, 17, 33, 50, 66, 75, 88, 100)
   returned values matching this table to all 12 digits.

## Attribution notes (relevant to which records are beatable)

The `csqv` table mixes a recent AI-optimized frontier with older program-generated baselines:

- The strongest recent entries (e.g. **N=26 = 2.635983**) are credited to **LLM/AI-driven** work —
  AlphaEvolve (ref [4], Google DeepMind) and contributors **Yiping Wang [5] → Haowei Lin [8]** (mid–late
  July 2026). These are hard to beat.
- Other entries (e.g. **N=27 = 2.685350025228**) are credited to **reference [1] = Specht's own `csqv`
  program (2011–2026)** — baseline values that a good modern optimizer can sometimes improve (we beat
  N=27; see `../ours-sota/`).
