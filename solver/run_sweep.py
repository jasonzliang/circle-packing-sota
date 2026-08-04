#!/usr/bin/env python3
"""Sweep the self-improved circle-packing solver (pack.py) across N, in PARALLEL, and emit results.

Each N is an independent optimization, so the sweep is embarrassingly parallel: a process pool runs
--workers instances of pack.search() at once (pack itself is single-core). For each N it keeps the best
strictly-feasible, non-degenerate sum of radii over --seeds, writes a packomania `.pck` file, and appends a
row to results.csv. An N with no acceptable config gets a blank row with feasible=0 and no `.pck`. The solver is used UNCHANGED from n=26 (n is a parameter throughout).

    OMP_NUM_THREADS=1 python3 run_sweep.py --nmin 2 --nmax 100 --time 120 --workers 8 --out-dir ../sota/ours

Runs are from scratch (random multi-start, no warm-start), so each N is an honest independent attempt.
Set BLAS threads to 1 in the environment so the workers don't oversubscribe the cores.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import time

import numpy as np

import pack   # the self-improved solver (imports container, shape); scipy used if present


def to_pck(circles, author, center_origin=True):
    """packomania .pck text for `circles` = list of (x,y,r) in the unit square [0,1]^2.
    Line 1 = largest radius; line 2 = author(s); lines 3+ = `x y r` sorted by INCREASING radius.
    With center_origin, coordinates are shifted to a side-1 square centred at the origin ([-0.5,0.5]^2)."""
    rows = sorted(([float(x), float(y), float(r)] for x, y, r in circles), key=lambda t: t[2])
    shift = 0.5 if center_origin else 0.0
    lines = [f"{rows[-1][2]:.12f}", author]
    for x, y, r in rows:
        lines.append(f"{x - shift:.12f} {y - shift:.12f} {r:.12f}")
    return "\n".join(lines) + "\n"


def solve_n(n, budget, seeds):
    """Best strictly-feasible, NON-DEGENERATE (sum_r, circles, violation) for n circles over the seeds.

    Runs in a worker. A feasibility test alone is not enough: repair() rescales every radius by one
    uniform factor, so a single degenerate cluster (two coincident centres) drives that factor to 0 and
    zeroes the whole configuration. The result violates nothing and scores 0, so `v <= 1e-9` accepts it.
    Reject any config containing a zero radius instead, since a radius-0 circle is not a circle.
    """
    best = None
    for s in seeds:
        try:
            z, sr = pack.search(n, seed=s, budget=budget, verbose=False)
        except Exception as e:
            print(f"  n={n} seed={s} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        v = pack.max_violation(z, n)
        if v > 1e-9:
            continue
        if float(np.min(z[2 * n:])) <= 1e-12:
            print(f"  n={n} seed={s} DEGENERATE (a radius is 0), rejected", flush=True)
            continue
        if best is None or sr > best[0]:
            c = np.stack([z[:n], z[n:2 * n], z[2 * n:]], axis=1)
            best = (sr, [[float(t) for t in row] for row in c], v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmin", type=int, default=2)
    ap.add_argument("--nmax", type=int, default=100)
    ap.add_argument("--ns", default=None, help="explicit comma-separated N list; overrides --nmin/--nmax")
    ap.add_argument("--time", type=float, default=120.0, help="search seconds per (N, seed)")
    ap.add_argument("--seeds", default="1", help="comma-separated seeds; best is kept")
    ap.add_argument("--workers", type=int, default=8, help="parallel N solved at once (pack is single-core)")
    ap.add_argument("--out-dir", default="../sota/ours")
    ap.add_argument("--author", default="Jason Liang")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    pck_dir = os.path.join(a.out_dir, "pck")
    json_dir = os.path.join(a.out_dir, "json")
    os.makedirs(pck_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, "results.csv")
    Ns = [int(x) for x in a.ns.split(",") if x.strip()] if a.ns else list(range(a.nmin, a.nmax + 1))
    print(f"sweeping N={a.nmin}..{a.nmax} ({len(Ns)} sizes), {a.workers} workers, "
          f"{a.time:.0f}s/N x {len(seeds)} seed(s)", flush=True)
    t_start = time.time()
    done = 0
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "sum_radii", "max_violation", "seeds", "feasible"])
        with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(solve_n, n, a.time, seeds): n for n in Ns}
            for fut in cf.as_completed(futs):
                n = futs[fut]
                done += 1
                try:
                    best = fut.result()
                except Exception as e:
                    print(f"n={n:3d}  WORKER ERROR {type(e).__name__}: {e}", flush=True)
                    best = None
                if best is None:
                    print(f"[{done}/{len(Ns)}] n={n:3d}  NO FEASIBLE CONFIG", flush=True)
                    w.writerow([n, "", "", a.seeds, 0]); f.flush(); continue
                sr, circles, v = best
                open(os.path.join(pck_dir, f"csqv{n}.pck"), "w").write(to_pck(circles, a.author))
                json.dump({"n": n, "sum_radii": sr, "max_violation": v, "seeds": a.seeds,
                           "time_budget_s": a.time, "circles": circles},   # full float64, [0,1]^2
                          open(os.path.join(json_dir, f"out{n}.json"), "w"), indent=1)
                print(f"[{done}/{len(Ns)}] n={n:3d}  sum_r={sr:.12f}  viol={v:.1e}", flush=True)
                w.writerow([n, f"{sr:.12f}", f"{v:.2e}", a.seeds, 1]); f.flush()
    print(f"done -> {csv_path}  ({time.time()-t_start:.0f}s wall)", flush=True)


if __name__ == "__main__":
    main()
