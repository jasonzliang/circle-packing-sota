#!/usr/bin/env python3
"""Fixed benchmark scorer for the circle-packing mission.

DELIBERATELY SELF-CONTAINED: this file imports only numpy + stdlib and shares no
code with tools/.  The scorer must stay correct even if the solver library is
rewritten, and it must not be able to fail open on a missing dependency.

Each bench/benchN/ directory holds
    spec.json    {"n": ..., "tol": ..., "note": ...}   the frozen task
    anchor.json  the fixed bar: a stored, feasible config from a simple optimizer
    best.json    the stored config being graded

`--check` grades the STORED configs.  It never runs an optimizer.

PASS for a bench  iff  best.json is strictly feasible with exactly n circles
                  AND  sum_r(best) > sum_r(anchor)
                  AND  anchor.json is itself strictly feasible with n circles
Overall CHECK: PASS iff every bench present passes.
SCORE is bench1's sum_r (the mission capability metric, n=54); an infeasible
bench1 scores 0.

Usage:  python3 bench/verify.py --check [--verbose]
        python3 bench/verify.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD_N54 = 3.841733103296  # packomania csqv best-known, for reporting only


def grade(circles, n_expected, tol):
    """Return (sum_r, feasible, detail).  Pure geometry, no tolerance games."""
    a = np.asarray(circles, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        return 0.0, False, "malformed config (need Kx3)"
    if a.shape[0] != n_expected:
        # load-bearing: sum(r) is unbounded as n grows, so the count is a
        # feasibility condition, not a formality.
        return 0.0, False, f"expected exactly {n_expected} circles, got {a.shape[0]}"
    if not np.all(np.isfinite(a)):
        return 0.0, False, "non-finite value in config"
    x, y, r = a[:, 0], a[:, 1], a[:, 2]
    if float(r.min()) < 0.0:
        return 0.0, False, f"negative radius {float(r.min()):.3e}"

    # inside the square: r <= x <= 1-r and r <= y <= 1-r
    box = float(np.max(r - np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y))))

    # pairwise non-overlap: dist >= r_i + r_j
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    np.fill_diagonal(d, np.inf)
    pair = float(np.max(r[:, None] + r[None, :] - d))

    detail = f"max_pair_overlap={pair:.3e} max_box_violation={box:.3e}"
    if pair > tol or box > tol:
        return 0.0, False, "INFEASIBLE " + detail
    return float(r.sum()), True, detail


def _load(path):
    with open(path) as fh:
        doc = json.load(fh)
    return doc["circles"]


def check(verbose=False):
    benches = sorted(
        d for d in os.listdir(HERE)
        if d.startswith("bench") and os.path.isdir(os.path.join(HERE, d))
    )
    if not benches:
        print("no bench directories found")
        print("SCORE: 0")
        print("CHECK: FAIL")
        return False

    overall = True
    headline = 0.0
    for name in benches:
        bdir = os.path.join(HERE, name)
        try:
            with open(os.path.join(bdir, "spec.json")) as fh:
                spec = json.load(fh)
            n, tol = int(spec["n"]), float(spec["tol"])
            a_sum, a_ok, a_det = grade(_load(os.path.join(bdir, "anchor.json")), n, tol)
            b_sum, b_ok, b_det = grade(_load(os.path.join(bdir, "best.json")), n, tol)
        except Exception as exc:  # missing/corrupt files must FAIL, never pass
            print(f"{name}: ERROR {exc}")
            overall = False
            continue

        ok = a_ok and b_ok and b_sum > a_sum
        overall = overall and ok
        gap = RECORD_N54 - b_sum if n == 54 else None
        line = (f"{name}: n={n} anchor={a_sum:.12f} best={b_sum:.12f} "
                f"delta={b_sum - a_sum:+.6e} -> {'PASS' if ok else 'FAIL'}")
        if gap is not None:
            line += f"  [record {RECORD_N54:.12f}, gap {gap:+.6e}]"
        print(line)
        if verbose or not ok:
            print(f"    anchor: {'feasible' if a_ok else 'INFEASIBLE'} {a_det}")
            print(f"    best  : {'feasible' if b_ok else 'INFEASIBLE'} {b_det}")
        if name == "bench1":
            headline = b_sum if b_ok else 0.0

    print(f"SCORE: {headline:.12f}")
    print(f"CHECK: {'PASS' if overall else 'FAIL'}")
    return overall


def _self_test():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")

    # four disjoint circles in the corners of the unit square
    good = [[0.25, 0.25, 0.25], [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25], [0.75, 0.75, 0.25]]
    s, f, d = grade(good, 4, 1e-9)
    chk("accepts a tight feasible config", f and abs(s - 1.0) < 1e-12, d)

    chk("rejects wrong count (too few)", not grade(good[:3], 4, 1e-9)[1])
    chk("rejects wrong count (too many)",
        not grade(good + [[0.5, 0.5, 0.0]], 4, 1e-9)[1])

    over = [row[:] for row in good]
    over[0][2] += 1e-6
    chk("rejects overlap beyond tol", not grade(over, 4, 1e-9)[1])

    out = [row[:] for row in good]
    out[0] = [0.1, 0.25, 0.25]
    chk("rejects circle poking out of the square", not grade(out, 4, 1e-9)[1])

    neg = [row[:] for row in good]
    neg[0][2] = -0.1
    chk("rejects negative radius", not grade(neg, 4, 1e-9)[1])

    nan = [row[:] for row in good]
    nan[0][0] = float("nan")
    chk("rejects NaN", not grade(nan, 4, 1e-9)[1])

    chk("infeasible scores 0, not raw sum", grade(over, 4, 1e-9)[0] == 0.0)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        print("verify self-test")
        sys.exit(0 if _self_test() else 1)
    if args.check:
        sys.exit(0 if check(args.verbose) else 1)
    ap.print_help()
