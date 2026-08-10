#!/usr/bin/env python3
"""Sequential Linear Programming polish for max-sum-of-radii circle packing.

THE STRUCTURE THIS EXPLOITS
---------------------------
packlib already uses the fact that radii are an exact LP for FIXED centers.
That leaves the centers to a penalty method, which is what stalls at the end.
But the centers can join the LP, because the only nonlinear constraint is a
norm -- and a norm is CONVEX, so its first-order Taylor expansion is a global
UNDER-estimator:

        || a + s ||  >=  ||a|| + e . s          with e = a / ||a||

Take a = c_i - c_j at the current point and s = (dc_i - dc_j).  Then the linear
constraint

        r_i + r_j - e . (dc_i - dc_j)  <=  d_ij                        (*)

IMPLIES the true non-overlap constraint ||c_i + dc_i - c_j - dc_j|| >= r_i + r_j.
Not approximates -- implies.  The box constraints r_i <= x_i + dx_i etc. are
already exactly linear.  So the LP

        max sum(r)   s.t. (*) for all pairs, the 4n box rows,
                          |dx|,|dy| <= delta,  r >= 0

is an *inner* (restricted) model of the true problem.  Three consequences:

  1. Every LP solution is TRULY FEASIBLE.  No trust region is needed for safety
     and no repair-after-step can eat the gain.
  2. dc = 0, r = current r is LP-feasible, so the LP optimum is never worse than
     where we started: the iteration is MONOTONE by construction.
  3. At a fixed point the linearization is first-order exact, so we land on a
     KKT point of the true problem -- an actual jammed packing, which the
     penalty descent only ever approaches.

delta only bounds how far the model is trusted; it cannot make a step unsafe.
We shrink it geometrically to squeeze out the last digits.

    python3 tools/slp.py --self-test
    python3 tools/slp.py --in bench/bench1/best.json --out /tmp/polished.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packlib as P  # noqa: E402


def have_lp():
    try:
        from scipy.optimize import linprog  # noqa: F401
        from scipy.sparse import csr_matrix  # noqa: F401
        return True
    except Exception:
        return False


class SLPCounters:
    """Telemetry for the polish: LP solves and total SLP iterations."""

    def __init__(self):
        self.nlp = 0        # number of LP solves
        self.nit = 0        # SLP iterations (accepted or not)
        self.polishes = 0   # number of slp_polish() calls


def _build_pair_rows(x, y, keep_mask=None):
    """Rows of (*) for every pair (or the subset selected by keep_mask)."""
    n = len(x)
    iu, ju = np.triu_indices(n, 1)
    if keep_mask is not None:
        iu, ju = iu[keep_mask], ju[keep_mask]
    ax = x[iu] - x[ju]
    ay = y[iu] - y[ju]
    d = np.sqrt(ax * ax + ay * ay)
    safe = np.maximum(d, 1e-15)
    ex, ey = ax / safe, ay / safe
    return iu, ju, d, ex, ey


def _solve_lp(x, y, delta, r_hi, iu, ju, d, ex, ey):
    """One LP: variables [dx (n), dy (n), r (n)].  Returns (dx, dy, r) or None."""
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    n = len(x)
    m = len(iu)

    # --- pair rows: r_i + r_j - ex*(dx_i - dx_j) - ey*(dy_i - dy_j) <= d_ij
    rp = np.repeat(np.arange(m), 6)
    cp = np.empty(6 * m, dtype=np.int64)
    vp = np.empty(6 * m, dtype=float)
    cp[0::6], vp[0::6] = 2 * n + iu, 1.0            # r_i
    cp[1::6], vp[1::6] = 2 * n + ju, 1.0            # r_j
    cp[2::6], vp[2::6] = iu, -ex                    # dx_i
    cp[3::6], vp[3::6] = ju, ex                     # dx_j
    cp[4::6], vp[4::6] = n + iu, -ey                # dy_i
    cp[5::6], vp[5::6] = n + ju, ey                 # dy_j
    bp = d

    # --- box rows (exactly linear, no approximation)
    idx = np.arange(n)
    rb, cb, vb, bb = [], [], [], []
    row0 = m
    for sgn, off, base in ((-1.0, 0, x), (1.0, 0, 1.0 - x),
                           (-1.0, n, y), (1.0, n, 1.0 - y)):
        rows = row0 + idx
        rb.append(np.repeat(rows, 2))
        c = np.empty(2 * n, dtype=np.int64)
        v = np.empty(2 * n, dtype=float)
        c[0::2], v[0::2] = 2 * n + idx, 1.0          # r_i
        c[1::2], v[1::2] = off + idx, sgn            # +/- dx_i (or dy_i)
        cb.append(c)
        vb.append(v)
        bb.append(base)
        row0 += n

    rows = np.concatenate([rp] + rb)
    cols = np.concatenate([cp] + cb)
    vals = np.concatenate([vp] + vb)
    b_ub = np.concatenate([bp] + bb)
    A = csr_matrix((vals, (rows, cols)), shape=(m + 4 * n, 3 * n))

    c_obj = np.zeros(3 * n)
    c_obj[2 * n:] = -1.0  # maximize sum(r)
    bounds = [(-delta, delta)] * (2 * n) + [(0.0, float(r_hi))] * n
    try:
        res = linprog(c=c_obj, A_ub=A, b_ub=b_ub, bounds=bounds, method="highs")
    except Exception:
        return None
    if not res.success or res.x is None:
        return None
    z = np.asarray(res.x, float)
    return z[:n], z[n:2 * n], np.maximum(z[2 * n:], 0.0)


def slp_polish(x, y, r=None, iters=60, delta0=0.02, delta_min=1e-9,
               shrink=0.7, tol=1e-13, counters=None, verbose=False,
               time_limit=None, max_stall=None):
    """Drive a config to a KKT point by sequential LP.  Monotone and feasible.

    Returns (x, y, r) strictly feasible (packlib.MARGIN), with sum(r) >= the
    sum(r) of the strictly-feasible starting point.
    """
    t0 = time.time()
    x = np.array(x, float)
    y = np.array(y, float)
    if r is None:
        x, y, r = P.max_radii(x, y)
    x, y, r = P.repair(x, y, r)
    best = (float(r.sum()), x.copy(), y.copy(), r.copy())
    if not have_lp():
        return best[1], best[2], best[3]
    if counters is not None:
        counters.polishes += 1

    delta = delta0
    stall = 0
    for k in range(iters):
        if time_limit is not None and time.time() - t0 > time_limit:
            break
        _, xc, yc, rc = best
        iu, ju, d, ex, ey = _build_pair_rows(xc, yc)
        out = _solve_lp(xc, yc, delta, 0.5, iu, ju, d, ex, ey)
        if counters is not None:
            counters.nlp += 1
            counters.nit += 1
        if out is None:
            break
        dx, dy, rn = out
        xn, yn = xc + dx, yc + dy
        # The LP is an inner model, so this is already feasible; repair() only
        # stamps on the strict MARGIN (and is a hard guarantee if HiGHS ever
        # returns a point a few ulps outside).
        xn, yn, rn = P.repair(xn, yn, rn)
        sn = float(rn.sum())
        if sn > best[0] + tol:
            gain = sn - best[0]
            best = (sn, xn, yn, rn)
            stall = 0
            if verbose:
                print(f"  slp[{k:3d}] d={delta:.2e} sum_r={sn:.12f} (+{gain:.3e})",
                      flush=True)
        else:
            # accept a non-improving-but-feasible point only if it is a true tie;
            # otherwise tighten the model and retry from the incumbent
            stall += 1
            delta *= shrink
            if delta < delta_min or (max_stall is not None and stall >= max_stall):
                break
    return best[1], best[2], best[3]


def polish_file(path_in, path_out=None, iters=80, delta0=0.02, verbose=True):
    doc = P.load_config(path_in)
    a = np.asarray(doc["circles"], float)
    x, y, r = a[:, 0], a[:, 1], a[:, 2]
    s0 = float(r.sum())
    c = SLPCounters()
    t0 = time.time()
    x2, y2, r2 = slp_polish(x, y, r, iters=iters, delta0=delta0,
                            counters=c, verbose=verbose)
    s1 = float(r2.sum())
    p, b = P.violations(x2, y2, r2)
    wall = time.time() - t0
    print(f"sum_r {s0:.12f} -> {s1:.12f}  (+{s1 - s0:.6e})")
    print(f"record {P.RECORD_N54:.12f}  gap {P.RECORD_N54 - s1:+.6e}")
    print(f"feasibility max_pair={p:.3e} max_box={b:.3e}  "
          f"[{c.nlp} LPs, {wall:.1f}s]")
    if path_out:
        cost = dict(doc.get("cost", {}))
        cost["slp_lps"] = c.nlp
        cost["slp_wall_s"] = round(wall, 3)
        P.save_config(path_out, x2, y2, r2, cost=cost,
                      meta={"kind": "best", "method": "SLP polish of " + path_in})
        print(f"wrote {path_out}")
    return s0, s1


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _self_test():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")

    if not have_lp():
        print("  [skip] scipy unavailable; slp_polish degrades to identity")
        rng = np.random.default_rng(0)
        x, y = rng.random(12), rng.random(12)
        x2, y2, r2 = slp_polish(x, y)
        p, b = P.violations(x2, y2, r2)
        chk("degraded path still feasible", p <= 0 and b <= 0)
        return ok

    # 1. the linearization really is an inner approximation: an LP step never
    #    produces a true violation BEFORE repair() is applied.
    rng = np.random.default_rng(7)
    n = 16
    x, y = rng.random(n), rng.random(n)
    x, y, r = P.max_radii(x, y)
    iu, ju, d, ex, ey = _build_pair_rows(x, y)
    dx, dy, rn = _solve_lp(x, y, 0.05, 0.5, iu, ju, d, ex, ey)
    p_raw, b_raw = P.violations(x + dx, y + dy, rn)
    chk("raw LP step is feasible without repair", p_raw <= 1e-9 and b_raw <= 1e-9,
        f"pair={p_raw:.2e} box={b_raw:.2e}")

    # 2. monotone: polish never lowers sum(r), on several random configs
    worst = 1.0
    gains = []
    for s in range(4):
        rng = np.random.default_rng(100 + s)
        n = 20
        x, y = rng.random(n), rng.random(n)
        x, y, r = P.max_radii(x, y)
        s0 = r.sum()
        x2, y2, r2 = slp_polish(x, y, r, iters=40)
        gains.append(r2.sum() - s0)
        worst = min(worst, r2.sum() - s0)
        p, b = P.violations(x2, y2, r2)
        if p > 0 or b > 0:
            worst = -1.0
    chk("monotone non-decreasing over 4 seeds", worst >= 0.0,
        f"min gain={worst:.3e}")
    chk("actually improves random configs", min(gains) > 1e-4,
        f"gains={[f'{g:.4f}' for g in gains]}")

    # 3. every output is strictly feasible (margin, not tolerance)
    rng = np.random.default_rng(5)
    x, y = rng.random(30), rng.random(30)
    x2, y2, r2 = slp_polish(x, y, iters=30)
    p, b = P.violations(x2, y2, r2)
    chk("output strictly feasible with margin", p <= -1e-13 and b <= -1e-13,
        f"pair={p:.2e} box={b:.2e}")

    # 4. n=54 sanity: polish beats plain LP-radii on the same centers
    rng = np.random.default_rng(11)
    X = P.init_batch(1, 54, rng)
    P.adam_run(X, 1500, 60.0, 3e4, 4e-3, 1e-4)
    xa, ya, ra = P.max_radii(X[0, 0], X[1, 0], X[2, 0])
    xb, yb, rb = slp_polish(xa, ya, ra, iters=50)
    chk("n=54 polish > LP-radii alone", rb.sum() > ra.sum(),
        f"{ra.sum():.9f} -> {rb.sum():.9f}")
    p, b = P.violations(xb, yb, rb)
    chk("n=54 polish output feasible", p <= 0 and b <= 0, f"pair={p:.2e}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", dest="out", default=None)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--delta0", type=float, default=0.02)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        print("slp self-test")
        sys.exit(0 if _self_test() else 1)
    if a.inp:
        polish_file(a.inp, a.out, iters=a.iters, delta0=a.delta0)
    else:
        ap.print_help()
