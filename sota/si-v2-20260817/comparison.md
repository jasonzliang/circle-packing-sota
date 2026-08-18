# SI-v2 2026-08-17 batches vs Packomania `csqv` records (max Sigma_r, N variable circles in a unit square)

Records: https://www.packomania.com/csqv/csqv.html -- retrieved 2026-08-14, site "Last Update 13-Aug-2026"
(snapshot committed here as `packomania_csqv_20260814.json`). Every row is **independently re-verified**
**from its `.pck` coordinates** at ZERO feasibility tolerance (pure geometry, no solver); a **WIN** is
strictly feasible AND Sigma_r exceeds the record by at least WIN_MARGIN = 1e-6 absolute.

**WINS: 17** -- of which **9 beat an EXTERNAL record** and
**8 beat this programme's own earlier result** that the 2026-08-13 site update absorbed.
Wins at N = [53, 63, 65, 67, 69, 73, 75, 77, 79, 83, 85, 87, 91, 93, 95, 97, 99].

Re-run `python3 verify_sota.py` to reproduce every row. Exit 0 iff all rows verify.

| N | ours Sigma_r | record | delta (ours-rec) | rel | record held by | source run |
|---:|---:|---:|---:|---:|:--|:--|
| 53 | 3.805568432564 | 3.804408802004 | +1.160e-03 | 3.05e-04 | **us (v6, absorbed)** | {'batch': 'circle_2v_x15_20it30m_20260817',  |
| 63 | 4.154116740618 | 4.153848570061 | +2.682e-04 | 6.46e-05 | **us (v6, absorbed)** | {'batch': 'circle_6v_x5_20it30m_20260817b',  |
| 65 | 4.221928422662 | 4.219413302014 | +2.515e-03 | 5.96e-04 | external | {'batch': 'circle_6v_x5_20it30m_20260817b',  |
| 67 | 4.289481392390 | 4.287777805457 | +1.704e-03 | 3.97e-04 | **us (v6, absorbed)** | {'batch': 'circle_6v_x5_20it30m_20260817b',  |
| 69 | 4.357977705112 | 4.356995844873 | +9.819e-04 | 2.25e-04 | **us (v6, absorbed)** | {'batch': 'circle_6v_x5_20it30m_20260817b',  |
| 73 | 4.488782078921 | 4.488004575020 | +7.775e-04 | 1.73e-04 | external | {'batch': 'circle_6v_x5_20it30m_20260817b',  |
| 75 | 4.550081805192 | 4.548223289937 | +1.859e-03 | 4.09e-04 | external | {'batch': 'circle_2v_x15_20it30m_20260817',  |
| 77 | 4.605527989026 | 4.605135681249 | +3.923e-04 | 8.52e-05 | **us (v6, absorbed)** | {'batch': 'circle_6v_x5_20it30m_20260817b',  |
| 79 | 4.666630641409 | 4.664674436598 | +1.956e-03 | 4.19e-04 | external | {'batch': 'circle_6v_x5_20it30m_20260817b',  |
| 83 | 4.784464220734 | 4.783919133241 | +5.451e-04 | 1.14e-04 | **us (v6, absorbed)** | {'batch': 'circle_2v_x15_20it30m_20260817',  |
| 85 | 4.846671972827 | 4.845134378577 | +1.538e-03 | 3.17e-04 | **us (v6, absorbed)** | {'batch': 'circle_2v_x15_20it30m_20260817',  |
| 87 | 4.907558130319 | 4.906503570705 | +1.055e-03 | 2.15e-04 | **us (v6, absorbed)** | {'batch': 'circle_2v_x15_20it30m_20260817',  |
| 91 | 5.022578167830 | 5.021462693130 | +1.115e-03 | 2.22e-04 | external | {'batch': 'circle_6v_x5_20it30m_20260817b',  |
| 93 | 5.073650364681 | 5.071029091028 | +2.621e-03 | 5.17e-04 | external | {'batch': 'circle_6v_x5_20it30m_20260817b',  |
| 95 | 5.126259907290 | 5.125567045105 | +6.929e-04 | 1.35e-04 | external | {'batch': 'circle_6v_x5_20it30m_20260817b',  |
| 97 | 5.181096077359 | 5.180639279289 | +4.568e-04 | 8.82e-05 | external | {'batch': 'circle_2v_x15_20it30m_20260817',  |
| 99 | 5.235423639117 | 5.233204567399 | +2.219e-03 | 4.24e-04 | external | {'batch': 'circle_2v_x15_20it30m_20260817',  |

## What these margins are

Absolute margins run 2.7e-04 to 2.6e-03 on sums of 3.8 to 5.2, i.e. relative 6.5e-05 to 6.0e-04.
These are refinements of known configurations, not new packing families.

## Caveats that travel with any claim here

1. A beat is against the **2026-08-14 snapshot**. The site updates; this is not a permanent record claim.
2. At N = 53, 63, 67, 69, 77, 83, 85, 87 the tabulated value **is this programme's own earlier v6
   result** (identical to within 3.5e-13, verified against `sota/sm-radical-v6-n54/pck/`). There the
   improvement is over ourselves. Across all 98 v6 sizes, 19 match the live table exactly.
3. N = 95 may be a **regression**: an earlier run of this programme beat it by +1.3e-03 against the
   +6.9e-04 here, and that packing is unrecoverable (its results dir was never archived).
4. The runs optimized against the frozen **2026-08-03** table, which is easier than live at 22 of the 37
   N in this census, so these live-table beats are a stronger statement than the beats the runs
   reported for themselves.
