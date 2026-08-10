#!/usr/bin/env python3
"""Sweep the n-generic SLP-KKT solver (pipeline.solve) across N, best-of-seeds.

Loop order: OUTER = seed, INNER = N. Each seed does one full parallel pass over all N (all cores),
and the per-N best-of-seeds is refined after every pass -- so a complete N=2..100 table exists after
pass 1 and only improves. Emits the same `.pck` / results.csv / json as the n27 solver's run_sweep.py,
so the repo's verify_and_compare.py applies unchanged.

Drives THIS solver's native entrypoint:  solve.solve(n, seconds, seed) -> (x, y, r, cost)

    OMP_NUM_THREADS=1 python3 run_sweep.py --nmin 2 --nmax 100 --time 250 \
        --seeds 1,2,3,4,5,6,7,8,9,10 --workers 10 --out-dir ../sota/sm-radical-v6-n54
"""
import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import time

import solve  # entrypoint shim -> pipeline.solve


def to_circles(res, n):
    if (isinstance(res, (list, tuple)) and len(res) in (3, 4)
            and hasattr(res[0], "__len__") and len(res[0]) == n):
        return [(float(a), float(b), float(c)) for a, b, c in zip(res[0], res[1], res[2])]
    if hasattr(res, "tolist"):
        res = res.tolist()
    return [(float(t[0]), float(t[1]), float(t[2])) for t in res]


def feasibility(circles, n, tol=1e-9):
    """(sum_r, max_violation) if strictly feasible + non-degenerate, else None."""
    if len(circles) != n:
        return None
    wb = wp = 0.0
    for x, y, r in circles:
        if r <= 1e-12:
            return None
        wb = max(wb, -(x - r), -(y - r), (x + r) - 1.0, (y + r) - 1.0)
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            wp = max(wp, (ri + rj) - math.hypot(xi - xj, yi - yj))
    v = max(wb, wp)
    return (sum(r for _, _, r in circles), v) if v <= tol else None


def to_pck(circles, author, center_origin=True):
    rows = sorted(([float(x), float(y), float(r)] for x, y, r in circles), key=lambda t: t[2])
    shift = 0.5 if center_origin else 0.0
    lines = [f"{rows[-1][2]:.12f}", author]
    for x, y, r in rows:
        lines.append(f"{x - shift:.12f} {y - shift:.12f} {r:.12f}")
    return "\n".join(lines) + "\n"


def solve_one(n, budget, seed):
    """One (N, seed): best strictly-feasible (sum_r, circles, v) or None."""
    try:
        res = solve.solve(n, seconds=budget, seed=seed, verbose=False)
    except Exception as e:
        print(f"  n={n} seed={seed} ERROR {type(e).__name__}: {e}", flush=True)
        return None
    fr = feasibility(to_circles(res, n), n)
    if fr is None:
        return None
    return (fr[0], to_circles(res, n), fr[1])


def write_csv(best, Ns, csv_path, seeds_str):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "sum_radii", "max_violation", "seeds", "feasible"])
        for n in Ns:
            if n in best:
                sr, _, v, _ = best[n]
                w.writerow([n, f"{sr:.12f}", f"{v:.2e}", seeds_str, 1])
            else:
                w.writerow([n, "", "", seeds_str, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmin", type=int, default=2)
    ap.add_argument("--nmax", type=int, default=100)
    ap.add_argument("--ns", default=None, help="explicit comma-separated N list; overrides --nmin/--nmax")
    ap.add_argument("--time", type=float, default=250.0, help="search seconds per (N, seed)")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10", help="comma-separated seeds; best is kept")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out-dir", default="../sota/sm-radical-v6-n54")
    ap.add_argument("--author", default="Jason Liang")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    pck_dir = os.path.join(a.out_dir, "pck")
    json_dir = os.path.join(a.out_dir, "json")
    os.makedirs(pck_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, "results.csv")
    Ns = [int(x) for x in a.ns.split(",") if x.strip()] if a.ns else list(range(a.nmin, a.nmax + 1))
    print(f"sweep N={a.nmin}..{a.nmax} ({len(Ns)} sizes), OUTER=seed INNER=N, {a.workers} workers, "
          f"{a.time:.0f}s/(N,seed), seeds={seeds} -> {a.out_dir}", flush=True)
    t0 = time.time()
    best = {}  # n -> (sum_r, circles, v, winning_seed)
    for si, s in enumerate(seeds):
        tp = time.time()
        improved = 0
        with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(solve_one, n, a.time, s): n for n in Ns}   # INNER: all N in parallel
            for fut in cf.as_completed(futs):
                n = futs[fut]
                r = fut.result()
                if r is None:
                    continue
                sr, circles, v = r
                if n not in best or sr > best[n][0]:
                    best[n] = (sr, circles, v, s)
                    improved += 1
                    open(os.path.join(pck_dir, f"csqv{n}.pck"), "w").write(to_pck(circles, a.author))
                    json.dump({"n": n, "sum_radii": sr, "max_violation": v, "seed": s,
                               "seeds_tried": seeds[:si + 1], "time_budget_s": a.time, "circles": circles},
                              open(os.path.join(json_dir, f"out{n}.json"), "w"), indent=1)
        write_csv(best, Ns, csv_path, ",".join(str(x) for x in seeds[:si + 1]))
        print(f"[seed {s}] pass {si+1}/{len(seeds)} in {time.time()-tp:.0f}s; "
              f"feasible N: {len(best)}/{len(Ns)}; improved this pass: {improved}; "
              f"elapsed {time.time()-t0:.0f}s", flush=True)
    print(f"done -> {csv_path}  ({time.time()-t0:.0f}s wall)", flush=True)


if __name__ == "__main__":
    main()
