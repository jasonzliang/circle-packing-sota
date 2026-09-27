#!/usr/bin/env python3
"""Emit a schedule.jsonl. cpu(n) = base + slope*n (generous, increasing with n).
   python make_schedule.py --out phase1.jsonl --solvers all --seeds 0 --modes self,nbr --base 60 --slope 3
   --restrict '{"nalt6":"60-100", ...}' limits some solvers to an n range."""
import argparse, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--portfolio", default=os.path.join(HERE, "portfolio.json"))
ap.add_argument("--solvers", default="all"); ap.add_argument("--n", default="1-100")
ap.add_argument("--seeds", default="0"); ap.add_argument("--modes", default="self,nbr")
ap.add_argument("--base", type=float, default=60); ap.add_argument("--slope", type=float, default=3)
ap.add_argument("--restrict", default=None, help='JSON {solver: "lo-hi"}')
ap.add_argument("--per-n-solvers", default=None, help="JSON {n: [solvers]} -- overrides --solvers per n")
a = ap.parse_args()
port = json.load(open(a.portfolio))
def rng(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part: lo, hi = part.split("-"); out |= set(range(int(lo), int(hi) + 1))
        else: out.add(int(part))
    return sorted(out)
ns = rng(a.n); seeds = rng(a.seeds); modes = a.modes.split(",")
solvers = list(port) if a.solvers == "all" else a.solvers.split(",")
restrict = {k: set(rng(v)) for k, v in (json.loads(a.restrict) if a.restrict else {}).items()}
per_n = {int(k): v for k, v in (json.loads(a.per_n_solvers) if a.per_n_solvers else {}).items()}
rows = []
for n in ns:
    cpu = round(a.base + a.slope * n, 1)
    for s in per_n.get(n, solvers):
        if s in restrict and n not in restrict[s]: continue
        for seed in seeds:
            for m in modes:
                rows.append({"solver": s, "n": n, "seed": seed, "mode": m, "cpu": cpu})
with open(a.out, "w") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
tot = sum(r["cpu"] for r in rows)
print("%d jobs, %.0f CPU-s = %.1f CPU-h, ~%.2f h wall at 60 wide (x1.08 overhead)" % (len(rows), tot, tot / 3600, tot * 1.08 / 3600 / 60))
