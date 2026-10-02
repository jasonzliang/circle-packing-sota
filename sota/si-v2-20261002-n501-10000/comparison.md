# Our solver vs Packomania `csqv` records (max Σr, N variable circles in a unit square)

Records: https://www.packomania.com/csqv/txt/sumradii.txt — retrieved 2026-10-01. Every row below is **independently re-verified from its `.pck` coordinates** (pure geometry, no solver); a **WIN** is strictly feasible AND Σr strictly exceeds the record.

**WINS: 4** · ties (≤1e-6): 0 · below: 0 · infeasible: 0 · sizes: 4.
Wins at N = [576, 676, 685, 729].

| N | ours Σr | record | Δ (ours−rec) | gap % | verdict |
|---:|---:|---:|---:|---:|:--|
| 576 | 12.775149186935 | 12.765618381692 | +9.53e-03 | +0.0747 | **WIN** 🏆 |
| 676 | 13.853532292832 | 13.843506930660 | +1.00e-02 | +0.0724 | **WIN** 🏆 |
| 685 | 13.922201173079 | 13.915210345965 | +6.99e-03 | +0.0502 | **WIN** 🏆 |
| 729 | 14.391974693301 | 14.380692110222 | +1.13e-02 | +0.0785 | **WIN** 🏆 |
