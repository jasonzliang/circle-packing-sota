#!/usr/bin/env python3
"""The fixed CHECK anchor: a deliberately SIMPLE reproducible optimizer.

This is the bar, not the contender.  It is exactly what a competent first
attempt looks like: ONE fixed-seed random start, textbook Adam on a
quadratic-penalty objective, and greedy Gauss-Seidel radii -- no multistart, no
basin hopping, no LP, no perturbation.  It is a real, decent optimizer, so the
bar is a real bar: everything that beats it has to beat *search quality*, not a
straw man.

Written once to bench/bench1/anchor.json and then never regenerated: the bar
does not move.  Re-running with the same seed reproduces the same number.

Self-test:  python3 tools/baseline.py --self-test
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packlib as P  # noqa: E402

SEED = 0
ITERS = 20000


def run(n=P.N_DEFAULT, seed=SEED, iters=ITERS, verbose=True):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    X = np.empty((3, 1, n))
    X[0, 0] = rng.random(n)
    X[1, 0] = rng.random(n)
    X[2, 0] = 0.02
    c = P.Counters()
    c.restarts = 1
    P.adam_run(X, iters, mu0=50.0, mu1=1e5, lr0=3e-3, lr1=1e-5, counters=c)
    # greedy radii only -- the LP polish is part of the contender, not the bar
    x0, y0, r0 = P.repair(X[0, 0], X[1, 0], X[2, 0])
    x, y, r = P.repair(x0, y0, P._radii_gauss_seidel(x0, y0, r0=r0))
    wall = time.time() - t0
    if verbose:
        print(f"baseline n={n} seed={seed}: sum_r={r.sum():.12f}")
        print(f"cost: {c.as_dict(wall)}")
    return x, y, r, c.as_dict(wall)


def _self_test():
    ok = True
    x, y, r, cost = run(n=20, iters=300, verbose=False)
    p, b = P.violations(x, y, r)
    print(f"  [{'ok' if p <= 0 and b <= 0 else 'FAIL'}] baseline output feasible "
          f"pair={p:.2e} box={b:.2e}")
    ok = ok and p <= 0 and b <= 0
    x2, y2, r2, _ = run(n=20, iters=300, verbose=False)
    same = np.allclose(r, r2, atol=0, rtol=0)
    print(f"  [{'ok' if same else 'FAIL'}] baseline is deterministic for a fixed seed")
    ok = ok and same
    print(f"  [{'ok' if cost['nit'] == 300 else 'FAIL'}] telemetry recorded {cost}")
    ok = ok and cost["nit"] == 300
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write anchor.json here (refuses to overwrite)")
    # --n exists so a NEW bench (bench2 at a neighbouring n) gets an anchor from
    # the *identical* procedure as bench1: same seed, same iters, same method.
    # It cannot touch bench1, whose anchor.json is already written and whose
    # --out path refuses to be overwritten.
    ap.add_argument("--n", type=int, default=P.N_DEFAULT)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        print("baseline self-test")
        sys.exit(0 if _self_test() else 1)
    x, y, r, cost = run(n=a.n)
    if a.out:
        if os.path.exists(a.out):
            sys.exit(f"refusing to overwrite existing anchor {a.out} -- the bar is fixed")
        P.save_config(a.out, x, y, r, cost=cost,
                      meta={"kind": "anchor", "seed": SEED, "iters": ITERS, "n": a.n,
                            "method": "single-start normalised gradient descent + greedy radii"})
        print(f"wrote {a.out}")
