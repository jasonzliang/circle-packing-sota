# sota/theirs: Packomania reference records (the SOTA to beat)

`packomania_csqv_records.csv` holds the best-known **sum of radii** for packing **N variable-sized circles
in a unit square** (maximize Σr), for N = 1..100.

- **Source:** https://www.packomania.com/csqv/csqv.html (the `csqv` table).
- **Maintainer:** Dr. Eckard Specht, Otto-von-Guericke-Universität Magdeburg.
- **Fetched:** 2026-08-03 (the table was last refreshed 01-Aug-2026).
- These are **best-known** values, **not proven optima**.

## Verification (why we trust these numbers)

The values were extracted via an automated fetch, so they were cross-checked four independent ways
(all passed):

1. **Monotonic, with a decreasing increment trend:** strictly increasing in N at all 99 steps, and the
   increments trend down from 0.2105 at N=3 to 0.0223 at N=91. The increment sequence is *not* itself
   monotone (it oscillates locally, rising against the previous step at 49 of 98 points, which is
   expected since packings gain and lose symmetry as N changes), so only the trend is evidence here.
   A mis-transcribed leading digit would still almost certainly break monotonicity.
2. **Analytic exact values:** N=1 = 0.5 (one inscribed circle); N=2 = 0.585786437626 vs the exact
   2−√2 = 0.585786437627 (matches to 9e-13); N=26 = 2.635983085 matches the literature value 2.635983.
3. **Independent solver agreement:** our from-scratch solver reproduces 26 of these records to within
   4e-11 (worst 3.4e-11, median 1.5e-11), and beats one (N=27).
4. **Independent re-extraction:** a re-fetch of the live table on 2026-08-03 was reparsed from raw HTML
   and compared against **all 100 rows** of this CSV. Every value matched to all 12 digits, with zero
   mismatches.

## Attribution notes (relevant to which records are beatable)

The `csqv` table mixes a recent AI-optimized frontier with much older entries. Per the reference legend on
[csqv.html](https://www.packomania.com/csqv/csqv.html): **[1]** = D. W. Cantrell, sci.math forum 2011/12;
**[2]** = E. Specht, program `csqv`, 2011–2026; **[4]** = AlphaEvolve (Novikov et al., Google DeepMind);
**[5]** = Yiping Wang; **[7]** = Sebastian Pokutta; **[8]** = Haowei Lin, `packing_records`, mid-July 2026;
**[10]** = Everett Dutton, private communication, July 2026.

- The strongest recent entries are credited to **LLM/AI-driven and contributed** work: e.g. **N=26 =
  2.635983** is now credited to **Haowei Lin [8]** (superseding Yiping Wang [5]), and Everett Dutton [10]
  supplied ~25 improvements on 01-Aug-2026. These are hard to beat.
- Other entries (e.g. **N=27 = 2.685350025228**) still carry **reference [1] = D. W. Cantrell, sci.math
  forum 2011/12**, long-standing values that a good modern optimizer can sometimes improve (we beat
  N=27; see `../ours/`).

Note the table is actively moving: the changelog shows record churn through late July / 01-Aug-2026, so
re-scrape before claiming any win.
