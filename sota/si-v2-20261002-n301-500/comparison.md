# Our solver vs Packomania `csqv` records (max Σr, N variable circles in a unit square)

Records: https://www.packomania.com/csqv/txt/sumradii.txt — retrieved 2026-10-01. Every row below is **independently re-verified from its `.pck` coordinates** (pure geometry, no solver); a **WIN** is strictly feasible AND Σr strictly exceeds the record.

**WINS: 5** · ties (≤1e-6): 0 · below: 0 · infeasible: 0 · sizes: 5.
Wins at N = [306, 309, 310, 355, 385].

| N | ours Σr | record | Δ (ours−rec) | gap % | verdict |
|---:|---:|---:|---:|---:|:--|
| 306 | 9.288501361867 | 9.284302999971 | +4.20e-03 | +0.0452 | **WIN** 🏆 |
| 309 | 9.336984289516 | 9.335666509317 | +1.32e-03 | +0.0141 | **WIN** 🏆 |
| 310 | 9.352776318902 | 9.352370720792 | +4.06e-04 | +0.0043 | **WIN** 🏆 |
| 355 | 10.014994322019 | 10.011883344382 | +3.11e-03 | +0.0311 | **WIN** 🏆 |
| 385 | 10.430051306024 | 10.424399434790 | +5.65e-03 | +0.0542 | **WIN** 🏆 |
