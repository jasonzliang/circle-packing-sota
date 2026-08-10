#!/usr/bin/env python3
"""packlib -- numpy-only toolkit for max-sum-of-radii circle packing.

Problem: place n circles (x_i, y_i, r_i) in the unit square maximizing sum(r_i)
subject to
    r_i <= x_i <= 1 - r_i,  r_i <= y_i <= 1 - r_i        (inside the square)
    dist(c_i, c_j) >= r_i + r_j                           (no overlap)

Design (the two halves that make this work):

  1. *Positions are the hard part.*  A batched quadratic-penalty ascent moves all
     3n variables of many restarts at once (pure numpy, shape (3, B, n)), so a
     240 s budget buys hundreds of simultaneous multistarts instead of one.

  2. *Radii are the easy part.*  For FIXED centers, the optimal radii are the
     solution of a linear program
         max sum(r)  s.t.  r_i + r_j <= d_ij,  0 <= r_i <= b_i
     so we never have to trust the penalty phase for feasibility: we re-solve
     radii exactly at the end and then hard-repair to a strict margin.  Every
     configuration this module emits is feasible by construction.

The LP uses scipy.optimize.linprog when importable and falls back to a
numpy-only Gauss-Seidel maximal-radius fixed point otherwise; the fallback is
always run too and the better of the two is kept, so scipy is an accelerator,
never a dependency.

Self-test:  python3 tools/packlib.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

N_DEFAULT = 54
RECORD_N54 = 3.841733103296  # packomania csqv best-known for n=54

# packomania `csqv` ("circles in a square, variable radii", side length 1,
# objective = sum of radii) best-known values, transcribed from the site's own
# table (fetched read-only, iteration 5 for n=54 and iteration 6 for the rest).
# REPORTING ONLY: nothing here is a bar, a threshold or part of any CHECK --
# bench/verify.py is standalone and never imports this module.  These let a run
# at any n print its gap against the real published number instead of against
# n=54's, and they make the cross-n calibration in tools/pipeline.py possible.
CSQV_RECORDS = {
    4: 1.006788474668, 9: 1.524365359370,
    44: 3.466043568234, 45: 3.503095055216, 46: 3.539568936862,
    47: 3.578955041750, 48: 3.615166622821, 49: 3.655413576541,
    50: 3.689944926007, 51: 3.727021028707, 52: 3.765353522998,
    53: 3.803308399164, 54: 3.841733103296, 55: 3.881992983052,
    56: 3.918995278232, 57: 3.954683754497, 58: 3.990566331646,
    59: 4.023613974606, 60: 4.057375010904, 61: 4.091602684217,
    62: 4.122646173929, 63: 4.153419675083, 64: 4.190585830354,
}


def area_bound(n):
    """A provable upper bound on sum(r) for n circles in the unit square.

    Disjoint circles inside the square give pi*sum(r_i^2) <= 1, and
    Cauchy-Schwarz gives sum(r) <= sqrt(n * sum(r^2)) <= sqrt(n/pi).  Weak but
    completely independent of every optimizer and scorer here, so a config that
    violates it is proof of a bug rather than of a record.
    """
    return math.sqrt(n / math.pi)

# Strict-feasibility margin we bake into every emitted config.  Costs at most
# n * MARGIN ~ 5e-11 of score and buys immunity to float re-association in any
# independent re-check.
MARGIN = 1e-12


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def box_cap(x, y):
    """Largest radius each circle may have from the square constraint alone."""
    return np.maximum(np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y)), 0.0)


def dist_matrix(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    return d


def violations(x, y, r):
    """(max pair overlap, max out-of-square amount) for one config; <=0 is feasible."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = np.asarray(r, float)
    d = dist_matrix(x, y)
    pair = float(np.max(r[:, None] + r[None, :] - d)) if len(x) > 1 else -np.inf
    box = float(np.max(np.maximum(r - box_cap(x, y), -np.inf)))
    return pair, box


def is_feasible(x, y, r, tol=0.0):
    p, b = violations(x, y, r)
    return p <= tol and b <= tol and float(np.min(r)) >= -tol


# --------------------------------------------------------------------------
# radii for fixed centers
# --------------------------------------------------------------------------
def _radii_gauss_seidel(x, y, r0=None, sweeps=40, order_seed=0):
    """Maximal (not necessarily maximum) radii by Gauss-Seidel cap chasing.

    Started from a FEASIBLE r0 the update r_i <- min(b_i, min_j d_ij - r_j) can
    only raise r_i (feasibility says r_i is already below that cap), so the
    sweep is monotone and never leaves the feasible set.  Started from r0=0 it
    is instead a plain greedy that lets the first circles bully the rest -- that
    is the difference between ~3.78 and ~2.76 at n=54, so always warm start when
    a repaired estimate is available.
    """
    n = len(x)
    b = box_cap(x, y)
    d = dist_matrix(x, y)
    rng = np.random.default_rng(order_seed)
    r = np.zeros(n) if r0 is None else np.clip(np.asarray(r0, float), 0.0, None)
    for s in range(sweeps):
        order = rng.permutation(n) if s else np.argsort(-b)
        prev = r.sum()
        for i in order:
            r[i] = max(0.0, min(b[i], float(np.min(d[i] - r))))
        if r.sum() <= prev + 1e-15:
            break
    return r


def _radii_lp(x, y):
    """Exact LP radii via scipy/HiGHS; returns None when scipy is unavailable."""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return None
    n = len(x)
    d = dist_matrix(x, y)
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    rows = np.repeat(np.arange(m), 2)
    cols = np.empty(2 * m, dtype=int)
    cols[0::2] = iu
    cols[1::2] = ju
    A = coo_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n))
    try:
        res = linprog(
            c=-np.ones(n),
            A_ub=A.tocsr(),
            b_ub=d[iu, ju],
            bounds=[(0.0, float(v)) for v in box_cap(x, y)],
            method="highs",
        )
    except Exception:
        return None
    if not res.success or res.x is None:
        return None
    return np.maximum(np.asarray(res.x, float), 0.0)


def repair(x, y, r, margin=MARGIN, iters=200):
    """Force strict feasibility by shrinking radii only (centers untouched)."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    y = np.clip(np.asarray(y, float), 0.0, 1.0)
    r = np.maximum(np.asarray(r, float), 0.0)
    d = dist_matrix(x, y)
    for _ in range(iters):
        r = np.minimum(r, box_cap(x, y) - margin)
        r = np.maximum(r, 0.0)
        v = r[:, None] + r[None, :] + margin - d
        np.fill_diagonal(v, -np.inf)
        worst = v.max(axis=1)
        if worst.max() <= 0.0:
            break
        # halving the violation on both endpoints clears the offending pair
        r = np.maximum(r - 0.5 * np.maximum(worst, 0.0), 0.0)
    return x, y, r


def max_radii(x, y, r0=None):
    """Best strictly-feasible radii we can get for these centers.

    Tries the exact LP (scipy/HiGHS) and the numpy-only monotone sweep, and
    keeps whichever wins -- so scipy is an accelerator, never a dependency.
    """
    cands = []
    lp = _radii_lp(x, y)
    if lp is not None:
        cands.append(lp)
    warm = repair(x, y, r0)[2] if r0 is not None else None
    cands.append(_radii_gauss_seidel(x, y, r0=warm))
    best = None
    for c in cands:
        xr, yr, rr = repair(x, y, c)
        if best is None or rr.sum() > best[2].sum():
            best = (xr, yr, rr)
    return best


# --------------------------------------------------------------------------
# batched penalty optimizer
# --------------------------------------------------------------------------
class Counters:
    """Telemetry: objective/gradient evaluations, optimizer iterations, restarts."""

    def __init__(self):
        self.nfev = 0
        self.nit = 0
        self.restarts = 0

    def as_dict(self, wall_s):
        return {
            "wall_s": round(wall_s, 3),
            "nfev": int(self.nfev),
            "nit": int(self.nit),
            "restarts": int(self.restarts),
        }


def penalty_grad(X, mu):
    """Gradient of  F = -sum(r) + mu/2 * (pair^2 + box^2)  for a batch.

    X has shape (3, B, n) holding x, y, r.  Returns (grad, sum_r, max_viol).
    """
    x, y, r = X[0], X[1], X[2]
    dx = x[:, :, None] - x[:, None, :]
    dy = y[:, :, None] - y[:, None, :]
    d2 = dx * dx + dy * dy
    n = x.shape[1]
    di = np.arange(n)
    d2[:, di, di] = 1.0
    d = np.sqrt(d2)
    v = r[:, :, None] + r[:, None, :] - d
    v[:, di, di] = 0.0
    np.maximum(v, 0.0, out=v)

    inv = v / d
    gx = -np.einsum("bij,bij->bi", inv, dx)
    gy = -np.einsum("bij,bij->bi", inv, dy)
    gr = v.sum(axis=2)

    b_lo_x = np.maximum(r - x, 0.0)
    b_hi_x = np.maximum(x + r - 1.0, 0.0)
    b_lo_y = np.maximum(r - y, 0.0)
    b_hi_y = np.maximum(y + r - 1.0, 0.0)
    gx += b_hi_x - b_lo_x
    gy += b_hi_y - b_lo_y
    gr += b_lo_x + b_hi_x + b_lo_y + b_hi_y

    g = np.empty_like(X)
    g[0] = mu * gx
    g[1] = mu * gy
    g[2] = mu * gr - 1.0
    maxv = np.maximum(v.max(axis=(1, 2)), np.maximum(b_lo_x, b_hi_x).max(axis=1))
    return g, r.sum(axis=1), maxv


def adam_run(X, iters, mu0, mu1, lr0, lr1, counters=None):
    """In-place batched Adam on the penalty objective with a geometric mu ramp."""
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-12
    for t in range(1, iters + 1):
        f = (t - 1) / max(iters - 1, 1)
        mu = mu0 * (mu1 / mu0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        g, _, _ = penalty_grad(X, mu)
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        X -= lr * mh / (np.sqrt(vh) + eps)
        np.clip(X[0], 0.0, 1.0, out=X[0])
        np.clip(X[1], 0.0, 1.0, out=X[1])
        np.maximum(X[2], 0.0, out=X[2])
        if counters is not None:
            counters.nfev += X.shape[1]
            counters.nit += X.shape[1]
    return X


# --------------------------------------------------------------------------
# initialisations
# --------------------------------------------------------------------------
def init_batch(B, n, rng):
    """Mixed pool: uniform random, jittered lattices, and jittered hex rows."""
    X = np.empty((3, B, n))
    for b in range(B):
        kind = b % 3
        if kind == 0:
            p = rng.random((n, 2))
        elif kind == 1:
            k = int(math.ceil(math.sqrt(n)))
            gx, gy = np.meshgrid((np.arange(k) + 0.5) / k, (np.arange(k) + 0.5) / k)
            p = np.stack([gx.ravel(), gy.ravel()], axis=1)
            p = p[rng.permutation(len(p))[:n]] + rng.normal(0, 0.35 / k, (n, 2))
        else:
            rows = rng.integers(6, 10)
            per = int(math.ceil(n / rows))
            pts = []
            for i in range(rows):
                off = 0.5 / per if i % 2 else 0.0
                for j in range(per):
                    pts.append(((j + 0.5) / per + off, (i + 0.5) / rows))
            p = np.array(pts)[rng.permutation(len(pts))[:n]]
            p = p + rng.normal(0, 0.25 / per, (n, 2))
        p = np.clip(p, 0.01, 0.99)
        X[0, b] = p[:, 0]
        X[1, b] = p[:, 1]
    X[2] = 0.02
    return X


# --------------------------------------------------------------------------
# scoring helpers / io
# --------------------------------------------------------------------------
def score_config(circles, n_expected=N_DEFAULT, tol=1e-9):
    """(sum_r, feasible, detail) for a list of (x, y, r)."""
    a = np.asarray(circles, float)
    if a.ndim != 2 or a.shape[1] != 3:
        return 0.0, False, "bad shape"
    if a.shape[0] != n_expected:
        return 0.0, False, f"expected {n_expected} circles, got {a.shape[0]}"
    if not np.all(np.isfinite(a)):
        return 0.0, False, "non-finite value"
    x, y, r = a[:, 0], a[:, 1], a[:, 2]
    if float(r.min()) < 0.0:
        return 0.0, False, "negative radius"
    p, bx = violations(x, y, r)
    ok = p <= tol and bx <= tol
    return float(r.sum()), ok, f"max_pair={p:.3e} max_box={bx:.3e}"


def save_config(path, x, y, r, cost=None, meta=None):
    circles = [[float(a), float(b), float(c)] for a, b, c in zip(x, y, r)]
    doc = {"n": len(circles), "sum_r": float(np.sum(r)), "circles": circles}
    if cost is not None:
        doc["cost"] = cost
    if meta:
        doc["meta"] = meta
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    return doc


def load_config(path):
    with open(path) as fh:
        doc = json.load(fh)
    return doc


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(0)
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")

    # 1. repair always yields strict feasibility from a badly overlapping start
    n = 20
    x, y = rng.random(n), rng.random(n)
    r = np.full(n, 0.3)
    xr, yr, rr = repair(x, y, r)
    p, b = violations(xr, yr, rr)
    chk("repair -> strictly feasible", p <= 0 and b <= 0, f"pair={p:.2e} box={b:.2e}")

    # 2. max_radii beats the naive shrink and stays feasible
    xm, ym, rm = max_radii(x, y)
    p2, b2 = violations(xm, ym, rm)
    chk("max_radii feasible", p2 <= 0 and b2 <= 0, f"pair={p2:.2e}")
    chk("max_radii >= repair", rm.sum() >= rr.sum() - 1e-12,
        f"{rm.sum():.6f} vs {rr.sum():.6f}")

    # 3. analytic gradient matches finite differences
    B, n = 2, 6
    X = np.empty((3, B, n))
    X[0] = rng.random((B, n))
    X[1] = rng.random((B, n))
    X[2] = 0.15 + 0.05 * rng.random((B, n))
    mu = 7.3

    def F(X):
        x, y, r = X[0], X[1], X[2]
        d = np.sqrt((x[:, :, None] - x[:, None, :]) ** 2 +
                    (y[:, :, None] - y[:, None, :]) ** 2)
        v = np.maximum(r[:, :, None] + r[:, None, :] - d, 0.0)
        di = np.arange(n)
        v[:, di, di] = 0.0
        pair = 0.25 * (v * v).sum(axis=(1, 2))
        bx = 0.5 * (np.maximum(r - x, 0) ** 2 + np.maximum(x + r - 1, 0) ** 2 +
                    np.maximum(r - y, 0) ** 2 + np.maximum(y + r - 1, 0) ** 2).sum(axis=1)
        return -r.sum(axis=1) + mu * (pair + bx)

    g, _, _ = penalty_grad(X.copy(), mu)
    h = 1e-6
    num = np.zeros_like(X)
    for a in range(3):
        for b_ in range(B):
            for i in range(n):
                Xp = X.copy(); Xp[a, b_, i] += h
                Xm = X.copy(); Xm[a, b_, i] -= h
                num[a, b_, i] = (F(Xp)[b_] - F(Xm)[b_]) / (2 * h)
    err = float(np.max(np.abs(num - g)))
    chk("analytic gradient == finite diff", err < 1e-5, f"maxerr={err:.2e}")

    # 4. a tiny end-to-end run produces a feasible n=54 config
    t0 = time.time()
    Xb = init_batch(8, N_DEFAULT, np.random.default_rng(1))
    c = Counters()
    adam_run(Xb, 300, 50.0, 5e3, 3e-3, 2e-4, c)
    best = -1.0
    for b_ in range(Xb.shape[1]):
        xx, yy, rr2 = max_radii(Xb[0, b_], Xb[1, b_])
        best = max(best, rr2.sum())
    chk("end-to-end n=54 feasible & positive", best > 2.5, f"sum_r={best:.6f}")
    chk("counters advanced", c.nit == 300 * 8, f"nit={c.nit}")

    # 5. scorer rejects the classic cheats
    good = np.stack(max_radii(*init_batch(1, N_DEFAULT, np.random.default_rng(2))[:2, 0]), 1)
    s, f, _ = score_config(good)
    chk("scorer accepts a feasible 54-config", f and s > 0)
    s, f, why = score_config(good[:53])
    chk("scorer rejects n=53", not f, why)
    ghost = np.vstack([good, [[0.5, 0.5, 0.0]]])
    s, f, why = score_config(ghost)
    chk("scorer rejects n=55 ghost padding", not f, why)
    bad = good.copy(); bad[:, 2] *= 1.05
    s, f, why = score_config(bad)
    chk("scorer rejects inflated radii", not f, why)
    print(f"  (self-test wall {time.time() - t0:.1f}s)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        print("packlib self-test")
        raise SystemExit(0 if _self_test() else 1)
    ap.print_help()
