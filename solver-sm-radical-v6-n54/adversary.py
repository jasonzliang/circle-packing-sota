#!/usr/bin/env python3
"""Adversarial re-check: try to prove the stored config is NOT a real win.

bench/verify.py uses float64 numpy with a 1e-9 tolerance. At a 1e-3 gap to the
record, the obvious way to fool myself is a config that is feasible only to
float64 -- so this attacks the result from a different direction:

  1. EXACT ARITHMETIC. Re-check every pair and box constraint in 60-digit
     `decimal` with no tolerance at all (violation must be <= 0, not <= 1e-9).
     Distances are compared as squared quantities where possible to avoid
     introducing a square root, and Σr is re-summed exactly.
  2. STRUCTURAL CHEATS. Exact count, no duplicate centers, no zero/ghost
     circles, no radius outside a sane range.
  3. PERTURBATION. Nudge the config by ~1e-12 in random directions; a genuine
     interior-feasible solution stays feasible, a knife-edge one does not.
  4. ANCHOR INTEGRITY. Confirm the anchor is itself feasible and that the claimed
     improvement is many orders of magnitude larger than the checker tolerance.

Exit code 0 only if every attack fails.

  python3 tools/adversary.py bench/bench1
  python3 tools/adversary.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal, getcontext

import numpy as np

getcontext().prec = 60


def exact_report(circles):
    """Worst pair and box violation in exact decimal arithmetic (<=0 is feasible)."""
    C = [(Decimal(repr(a)), Decimal(repr(b)), Decimal(repr(c))) for a, b, c in circles]
    one = Decimal(1)
    worst_pair = Decimal(-1)
    worst_box = Decimal(-1)
    first = True
    for i, (xi, yi, ri) in enumerate(C):
        # box: r <= min(x, y, 1-x, 1-y), checked directly (no roots involved)
        b = min(xi, yi, one - xi, one - yi)
        v = ri - b
        if first or v > worst_box:
            worst_box = v
        for j in range(i + 1, len(C)):
            xj, yj, rj = C[j]
            s = ri + rj
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            # (r_i+r_j)^2 > d^2  <=>  overlap, with s >= 0; compare squares so no
            # sqrt is introduced, then convert back to a length for reporting
            gap2 = d2 - s * s
            v = -gap2 / (s + d2.sqrt() + Decimal("1e-300"))  # ~ (s - d), signed
            if first or v > worst_pair:
                worst_pair = v
            first = False
    return worst_pair, worst_box, sum(c[2] for c in C)


def attack(bdir, verbose=True):
    fails = []

    def say(name, ok, extra=""):
        if not ok:
            fails.append(name)
        if verbose:
            print(f"  [{'survived' if ok else 'BROKEN  '}] {name} {extra}")

    with open(os.path.join(bdir, "spec.json")) as fh:
        spec = json.load(fh)
    n, tol = int(spec["n"]), float(spec["tol"])
    with open(os.path.join(bdir, "best.json")) as fh:
        best = json.load(fh)["circles"]
    with open(os.path.join(bdir, "anchor.json")) as fh:
        anchor = json.load(fh)["circles"]

    say("exact count", len(best) == n, f"n={len(best)}")

    wp, wb, ssum = exact_report(best)
    say("exact-arithmetic pair constraint (no tolerance)", wp <= 0,
        f"worst={float(wp):.3e}")
    say("exact-arithmetic box constraint (no tolerance)", wb <= 0,
        f"worst={float(wb):.3e}")

    a = np.asarray(best, float)
    r = a[:, 2]
    say("no ghost / degenerate circles", float(r.min()) > 1e-6,
        f"r_min={float(r.min()):.3e}")
    say("no absurd radii", float(r.max()) < 0.5 + 1e-12, f"r_max={float(r.max()):.6f}")
    d = np.sqrt((a[:, 0][:, None] - a[:, 0][None, :]) ** 2 +
                (a[:, 1][:, None] - a[:, 1][None, :]) ** 2)
    np.fill_diagonal(d, np.inf)
    say("no duplicate centers", float(d.min()) > 1e-9, f"min_center_dist={float(d.min()):.3e}")

    # perturbation: a knife-edge "feasible" config dies here
    rng = np.random.default_rng(12345)
    worst = -np.inf
    for _ in range(200):
        p = a.copy()
        p[:, :2] += rng.normal(0, 1e-13, (n, 2))
        dd = np.sqrt((p[:, 0][:, None] - p[:, 0][None, :]) ** 2 +
                     (p[:, 1][:, None] - p[:, 1][None, :]) ** 2)
        np.fill_diagonal(dd, np.inf)
        worst = max(worst, float(np.max(p[:, 2][:, None] + p[:, 2][None, :] - dd)))
    say("stable under 1e-13 center jitter", worst <= 0, f"worst={worst:.3e}")

    wpa, wba, asum = exact_report(anchor)
    say("anchor is itself exactly feasible", wpa <= 0 and wba <= 0,
        f"pair={float(wpa):.3e} box={float(wba):.3e}")
    gain = ssum - asum
    say("gain is not a tolerance artifact", float(gain) > 1e6 * tol,
        f"gain={float(gain):.6e} vs tol={tol:.0e}")

    if verbose:
        rec = float(spec.get("record", float("nan")))
        print(f"  exact Σr(best)   = {ssum}")
        print(f"  exact Σr(anchor) = {asum}")
        print(f"  gain over anchor = {float(gain):+.9e}")
        print(f"  gap to record    = {rec - float(ssum):+.9e}")
    return fails


def _self_test():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")

    good = [[0.25, 0.25, 0.25], [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25], [0.75, 0.75, 0.25]]
    wp, wb, s = exact_report(good)
    chk("exact check accepts a tangent config", wp <= 0 and wb <= 0 and s == Decimal("1.00"),
        f"pair={float(wp):.2e} sum={s}")

    # a violation far below the 1e-9 float tolerance must still be caught exactly
    sneaky = [row[:] for row in good]
    sneaky[0][2] = 0.25 + 1e-13
    wp2, _, _ = exact_report(sneaky)
    chk("exact check catches a 1e-13 overlap that float tol would pass", wp2 > 0,
        f"pair={float(wp2):.2e}")

    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", nargs="?", default="bench/bench1")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        print("adversary self-test")
        sys.exit(0 if _self_test() else 1)
    print(f"adversarial re-check of {a.bench}")
    f = attack(a.bench)
    print("VERDICT: " + ("no attack succeeded" if not f else f"BROKEN by {f}"))
    sys.exit(0 if not f else 1)
