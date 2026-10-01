# Our solver vs Packomania `csqv` records (max Σr, N variable circles in a unit square)

Records: https://www.packomania.com/csqv/txt/sumradii.txt — retrieved 2026-10-01. Every row below is **independently re-verified from its `.pck` coordinates** (pure geometry, no solver); a **WIN** is strictly feasible AND Σr strictly exceeds the record.

**WINS: 3** · ties (≤1e-6): 0 · below: 0 · infeasible: 0 · sizes: 3.
Wins at N = [293, 295, 298].

| N | ours Σr | record | Δ (ours−rec) | gap % | verdict |
|---:|---:|---:|---:|---:|:--|
| 293 | 9.088258641565 | 9.088041361012 | +2.17e-04 | +0.0024 | **WIN** 🏆 |
| 295 | 9.117257272531 | 9.116204790462 | +1.05e-03 | +0.0115 | **WIN** 🏆 |
| 298 | 9.161869538266 | 9.161800772953 | +6.88e-05 | +0.0008 | **WIN** 🏆 |
