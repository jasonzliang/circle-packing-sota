#!/usr/bin/env python3
"""escalate.py -- successive-halving schedule: from the aggregate of earlier rounds, keep the top-K solvers
per n (ranked by their best sum_r over all prior jobs at that n, tie-break by improvement over the warm
baseline and by digits), and emit new (solver, n, seed, mode, cpu) jobs with a bigger budget.

  python escalate.py --agg AGG_DIR --out roundB.jsonl --top 4 --seeds 1-2 --base 60 --slope 3 \
        [--modes best|both] [--seeds-below 1-4] [--only-open]
  --seeds-below: extra seed range for n whose published best is still below/tie vs the live record
  --only-open : skip n whose best already BEATS the live record by >= 1e-6 (leave them alone)
"""
import sys, argparse, csv, json, os, collections
ap = argparse.ArgumentParser()
ap.add_argument("--agg", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=4); ap.add_argument("--seeds", default="1-2")
ap.add_argument("--seeds-below", default=None); ap.add_argument("--modes", default="best")
ap.add_argument("--base", type=float, default=60); ap.add_argument("--slope", type=float, default=3)
ap.add_argument("--only-open", action="store_true"); ap.add_argument("--n", default="1-100")
ap.add_argument("--min-solvers", type=int, default=2, help="always keep at least this many distinct solvers per n")
a = ap.parse_args()
def rng(s):
    out = set()
    for p in s.split(","):
        if "-" in p: lo, hi = p.split("-"); out |= set(range(int(lo), int(hi) + 1))
        else: out.add(int(p))
    return sorted(out)
ns = set(rng(a.n)); seeds = rng(a.seeds); seeds_below = rng(a.seeds_below) if a.seeds_below else seeds
per_n_status = {}
for r in csv.DictReader(open(os.path.join(a.agg, "per_n.csv"))):
    per_n_status[int(r["n"])] = r["verdict"]
best = collections.defaultdict(dict)   # n -> (solver, mode) -> (sum_r, d_warm)
for r in csv.DictReader(open(os.path.join(a.agg, "jobs.csv"))):
    if r["feasible"] != "1" or not r["sum_r"]: continue
    n = int(r["n"]); key = (r["solver"], r["mode"]); s = float(r["sum_r"]); dw = float(r["d_warm"]) if r["d_warm"] else 0.0
    if key not in best[n] or s > best[n][key][0]: best[n][key] = (s, dw)
rows = []; kept = collections.Counter()
for n in sorted(ns):
    if n not in best: continue
    verdict = per_n_status.get(n, "-")
    if a.only_open and verdict == "BEAT": continue
    # rank solvers by their best over modes
    per_solver = {}
    for (s, m), (v, dw) in best[n].items():
        if s not in per_solver or v > per_solver[s][0]: per_solver[s] = (v, dw, m)
    ranked = sorted(per_solver.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
    keep = ranked[:max(a.top, a.min_solvers)]
    sd = seeds_below if verdict in ("below", "tie", "beat(<1e-6)") else seeds
    cpu = round(a.base + a.slope * n, 1)
    for s, (v, dw, m) in keep:
        modes = [m] if a.modes == "best" else ["self", "nbr"]
        for seed in sd:
            for mode in modes:
                rows.append({"solver": s, "n": n, "seed": seed, "mode": mode, "cpu": cpu}); kept[s] += 1
with open(a.out, "w") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
tot = sum(r["cpu"] for r in rows)
print("%d jobs, %.0f CPU-s = %.1f CPU-h, ~%.2f h wall at 62 wide" % (len(rows), tot, tot / 3600, tot * 1.08 / 3600 / 62))
print("jobs per solver:", dict(kept.most_common()))
