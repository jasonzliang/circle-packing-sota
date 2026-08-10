# sota/packomania: Packomania reference records (the SOTA to beat)

Best-known **sum of radii** for packing **N variable-sized circles in a unit
square** (maximize Σr), from https://www.packomania.com/csqv/csqv.html
(maintainer: Dr. Eckard Specht, Otto-von-Guericke-Universität Magdeburg).
**Best-known** values, **not proven optima**.

- **`packomania_csqv.json`** — the **canonical, latest** table, refreshed live
  by `../../verify_and_compare.py fetch` (N=1..1013; N=1..100 contiguous). This
  is what `compare` reads.
- **`history/packomania_csqv_<YYYY-MM-DD>.json`** — **dated snapshots**, one per
  fetch (older ones taken from git history, the authoritative source),
  preserving the historical bar each result was measured against. Snapshots on
  record: **2026-08-03** (git commit `21e7568`, the original 4-way-verified
  fetch, N≤100, verified below) and **2026-08-10** (full N=1..1013, live). The
  `csqv` table is **actively updated**: **43 of the 100 records rose between
  these two snapshots** (all upward, up to +5.8e-3 — e.g. N=50 +3.5e-3, N=54
  +2.6e-3, N=27 2.685350→2.685979; a few differ only at 1e-12 storage rounding).
  **15 of our win-N had their bar raised in this window, and we beat the raised
  Aug-10 values** — so re-fetch before claiming any win.

_(The former `packomania_csqv_records.csv` — the 2026-08-03 fetch — is retired
into `history/packomania_csqv_2026-08-03.json`; the verification below documents
that snapshot.)_

## Verification (why we trust the 2026-08-03 snapshot)

The values were extracted via an automated fetch, so they were cross-checked
four ways (all passed). Note up front that only the first two are independent of
this repo; checks 3 and 4 are weaker than they look and are labelled as such:

1. **Monotonicity.** Strictly increasing in N at all 99 steps. Increments run
   from 0.2105 at N=3 down to 0.0318 at N=100, with the minimum 0.0223 at N=91;
   the sequence is _not_ monotone, rising against the previous step at 49 of 98
   points, which is expected as packings gain and lose symmetry with N. So the
   evidence here is monotonicity of the values, not smoothness of the
   increments.
2. **Analytic values.** N=1 = 0.5 exactly (one inscribed circle). N=2 =
   0.585786437626 against 2−√2 = 0.5857864376269, agreeing to 9.0e-13 (the
   snapshot is stored at 12 dp, so the printed strings differ in the last
   place). N=26 = 2.635983084918 matches the 2.635983 quoted in the
   AlphaEvolve/ShinkaEvolve literature to its 6 published digits.
3. **Solver agreement (not independent).** Our own solver reproduces 27 of these
   records to within 4e-11 (worst 3.4e-11, median 1.5e-11) and beats one (N=27).
   This is the artifact under review checking its own reference, so it argues
   the two agree, not that either is right.
4. **Re-extraction (same-day, so it tests the parser).** The live table was
   re-fetched on 2026-08-03, reparsed from raw HTML, and compared against **all
   100 rows** here: every value matched to 12 digits, zero mismatches. Since the
   original fetch was the same day, this rules out transcription and parsing
   error, not a stale table.

## Attribution notes (relevant to which records are beatable)

The `csqv` table mixes a recent AI-optimized frontier with much older entries.
Per the reference legend on
[csqv.html](https://www.packomania.com/csqv/csqv.html): **[1]** = D. W.
Cantrell, sci.math forum 2011/12; **[2]** = E. Specht, program `csqv`,
2011–2026; **[4]** = AlphaEvolve (Novikov et al., Google DeepMind); **[5]** =
Yiping Wang; **[7]** = Sebastian Pokutta; **[8]** = Haowei Lin,
`packing_records`, mid-July 2026; **[10]** = Everett Dutton, private
communication, July 2026.

- The strongest recent entries are credited to **LLM/AI-driven and contributed**
  work: e.g. **N=26 = 2.635983** is now credited to **Haowei Lin [8]**
  (superseding Yiping Wang [5]), and Everett Dutton [10] supplied ~25
  improvements on 01-Aug-2026. These are hard to beat.
- Other entries (e.g. **N=27 = 2.685350025228**) still carry **reference [1] =
  D. W. Cantrell, sci.math forum 2011/12**, long-standing values that a good
  modern optimizer can sometimes improve (we beat N=27; see
  `../nietzsche-sm-radical-v6-n27/`).

Note the table is actively moving: the changelog shows record churn through late
July / 01-Aug-2026, so re-scrape before claiming any win.
