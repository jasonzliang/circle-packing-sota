#!/usr/bin/env python3
"""tools/pack.py -- general circle-packing search: place n circles in the unit
square maximising the sum of radii.

Why it is built this way (the part worth remembering):

  * The problem is a nonlinear program, but it has an exactly solvable inner
    layer.  With the CENTRES held fixed, maximising sum(r) subject to
    r_i + r_j <= d_ij and r_i <= dist(c_i, wall) is a pure LINEAR PROGRAM in r.
    So a local optimum in the joint variables can always be certified/improved
    by an LP that costs milliseconds and is exact -- no gradient noise, no
    step-size tuning.  Every candidate gets that LP polish before it is scored.
  * The outer layer (centres) is what is actually hard: many near-degenerate
    local optima.  That is attacked with joint SLSQP over (x, y, r) with
    analytic Jacobians, from a diverse pool of starts, plus basin hopping that
    perturbs only a random subset of circles.
  * Nothing is trusted until it is repaired to STRICT feasibility by uniform
    radius scaling, which is exact arithmetic rather than a tolerance argument.

n is a parameter throughout, so the same tool serves bench1 (n=26) and any
harder bench (n=27, 28, ...) without modification.

    python3 tools/pack.py --self-test
    python3 tools/pack.py -n 26 --time 300 --seed 1 -o out.json
    python3 tools/pack.py -n 26 --time 300 --warm best.json -o out.json
"""
import argparse
import json
import math
import time

import numpy as np

import container as ctn
import shape as shp

try:
    from scipy.optimize import linprog, minimize
    HAVE_SCIPY = True
except ImportError:                                    # pragma: no cover
    HAVE_SCIPY = False


# ----------------------------------------------------- container and shape --
#
# The container enters ONLY through the wall rows r_i <= F_k(c_i).  Those are
# linear in r for any convex container, so the exact inner LP -- the structural
# fact this whole search is built on -- is unchanged when the unit square is
# swapped for a disk or a triangle.  Everything below therefore reads the
# container from one module-level handle rather than hard-coding [0,1]^2.
# Default is the unit square, so bench1/2/3 behaviour is untouched.
#
# The packed OBJECT is abstracted the same way (iter 5).  A circle becomes a
# homothet c_i + r_i*K of a symmetric convex body K, and K enters in exactly two
# coefficient slots: the pair distance becomes the gauge gamma_K instead of
# hypot, and each wall row is divided by the support value h_K(a_k) instead of
# by 1.  Both stay LINEAR IN r, so the inner LP survives again.  Default is the
# Euclidean ball, which reproduces the circle code path exactly.

CONTAINER = ctn.UNIT_SQUARE
SHAPE = shp.BALL


def set_container(ct):
    global CONTAINER
    CONTAINER = ct
    return ct


def set_shape(sh):
    global SHAPE
    SHAPE = sh
    return sh


# ---------------------------------------------------------------- geometry --

_PAIR_CACHE = {}


def pairs(n):
    p = _PAIR_CACHE.get(n)
    if p is None:
        p = _PAIR_CACHE[n] = np.triu_indices(n, k=1)
    return p


def gdist(dx, dy):
    """Distance in the shape's own gauge; hypot for the Euclidean ball."""
    return SHAPE.gauge(dx, dy)


def _wall_h():
    """h_K(a_k) for each wall row -- the divisor turning slack into max scale."""
    if SHAPE.is_ball:
        return None                                    # unit normals: h = 1
    assert hasattr(CONTAINER, "A"), (
        "a non-ball shape needs a polyhedral container (its wall rows must be "
        "half-planes); got " + CONTAINER.name)
    return SHAPE.support(CONTAINER.A[:, 0], CONTAINER.A[:, 1])


def wall_rows(x, y):
    """(F, Fx, Fy) with F[k,i] = the largest r_i that wall k allows at c_i."""
    F, Fx, Fy = CONTAINER.walls(x, y)
    h = _wall_h()
    if h is None:
        return F, Fx, Fy
    hc = h[:, None]
    return F / hc, Fx / hc, Fy / hc


def wall_cap(x, y):
    if SHAPE.is_ball:
        return CONTAINER.cap(x, y)
    return np.maximum(wall_rows(x, y)[0].min(axis=0), 0.0)


def shape_inradius():
    """Largest single homothet that fits: max_c min_k (b_k - a_k.c)/h_K(a_k)."""
    if SHAPE.is_ball:
        return CONTAINER.inradius()
    A, b, h = CONTAINER.A, CONTAINER.c, _wall_h()
    if HAVE_SCIPY:                                     # a 3-variable LP
        M = np.concatenate([A, h[:, None]], axis=1)
        res = linprog(np.array([0.0, 0.0, -1.0]), A_ub=M, b_ub=b,
                      bounds=[(None, None), (None, None), (0.0, None)],
                      method="highs")
        if res.success:
            return float(res.x[2])
    g = np.random.default_rng(0)                       # conservative fallback
    xs, ys = CONTAINER.sample(20000, g)
    return float(np.max(wall_cap(xs, ys)))


def max_violation(z, n):
    x, y, r = z[:n], z[n:2 * n], z[2 * n:]
    i, j = pairs(n)
    d = gdist(x[i] - x[j], y[i] - y[j])
    over = float(np.max(r[i] + r[j] - d)) if len(i) else -np.inf
    F, _, _ = wall_rows(x, y)
    wall = float(np.max(r[None, :] - F))
    return max(over, wall, float(-np.min(r)))


def repair(z, n):
    """Largest uniform radius scale s <= 1 giving a strictly feasible config."""
    z = z.copy()
    x, y, r = z[:n], z[n:2 * n], z[2 * n:]
    px, py = CONTAINER.project(x, y)
    x[:], y[:] = px, py
    np.maximum(r, 0.0, out=r)
    i, j = pairs(n)
    s = 1.0
    tot = r[i] + r[j]
    m = tot > 0
    if np.any(m):
        s = min(s, float(np.min(gdist(x[i] - x[j], y[i] - y[j])[m] / tot[m])))
    wall = wall_cap(x, y)
    m = r > 0
    if np.any(m):
        s = min(s, float(np.min(np.maximum(wall[m], 0.0) / r[m])))
    r *= max(0.0, min(1.0, s))
    np.maximum(r - 1e-12, 0.0, out=r)      # ~1e-11 of score buys exactness
    return z


def score(z, n):
    return float(np.sum(z[2 * n:]))


# ------------------------------------------- structure: the contact graph ----
#
# The formulation has n(n-1)/2 pair constraints, but a packing's contact graph
# is SPARSE -- roughly 3n contacts, because it is essentially planar.  At n=100
# that is 4950 constraints modelling ~300 real ones, and it is the dense
# Jacobian, not the geometry, that makes large n intractable.
#
# The reduction below is EXACT, not a heuristic.  For any feasible config,
# r_i + r_j <= d_ij with r_j >= 0 forces r_i <= d_ij for EVERY j, so
#
#     u_i := min( dist(c_i, wall), min_{j != i} d_ij )
#
# is a valid upper bound on r_i.  Impose r_i <= u_i as a variable bound -- valid
# bounds never cut off the optimum -- and every pair with d_ij >= u_i + u_j is
# then implied by those two bounds and can be DELETED with provably zero loss.
# u_i is about a nearest-neighbour distance, i.e. ~2r, so the surviving set is
# the local neighbourhood: an order of magnitude fewer rows at n=100, and the
# LP/QP still returns the identical optimum (asserted in --self-test).

def valid_caps(x, y, n):
    """u_i: a bound on r_i valid for EVERY feasible config with these centres."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    D = gdist(dx, dy)
    np.fill_diagonal(D, np.inf)
    wall = wall_cap(x, y)
    return np.minimum(wall, D.min(axis=1)), D


def live_pairs(x, y, n, margin=0.0):
    """Rows that can bind, plus the caps that make dropping the rest exact.

    Returns (i, j, cap) with cap = u + margin.  The drop rule and the cap are
    two halves of ONE argument and must be used together: a row is dropped only
    when d_ij >= cap_i + cap_j, which is safe precisely because the solver is
    also told r <= cap.  Enforce the rule without the bound and the solver will
    happily inflate radii through the rows you deleted.

    margin=0 is the exact reduction (cap = u is valid for every feasible
    config, so nothing is lost).  A positive margin buys the outer SLSQP room
    to grow radii before the linearisation has to be refreshed.
    """
    u, D = valid_caps(x, y, n)
    cap = u + margin
    i, j = pairs(n)
    keep = D[i, j] < cap[i] + cap[j]
    return i[keep], j[keep], cap


def lp_solve(x, y, n):
    """Exact optimal radii for fixed centres (a linear program), plus its DUALS.

    Returns (r, lam, mu):
      lam[k] = dual price of pair constraint r_i + r_j <= d_ij, i.e.
               d(sum r)/d(d_ij) -- how much the objective would gain per unit of
               extra room between that pair.  Zero unless the contact is tight.
      mu[i]  = dual price of the wall bound r_i <= dist(c_i, wall).
    LP duality gives sum_j lam_ij + mu_i == 1 for every circle with r_i > 0, so
    (1 - mu_i) is exactly the share of circle i's radius that is limited by its
    NEIGHBOURS rather than by the boundary.  That is the signal perturb_dual()
    uses to decide which circles are worth moving.

    Falls back to an iterative feasible assignment (no duals) without scipy.

    Rows are pruned by the exact redundancy test in live_pairs(); lam is
    returned expanded back over ALL pairs (zero on the pruned ones), so callers
    never have to know the reduction happened.
    """
    i, j = pairs(n)
    ki, kj, ub = live_pairs(x, y, n)                    # exact: no optimum lost
    d = gdist(x[ki] - x[kj], y[ki] - y[kj])
    if HAVE_SCIPY:
        A = np.zeros((len(ki), n))
        A[np.arange(len(ki)), ki] = 1.0
        A[np.arange(len(ki)), kj] = 1.0
        res = linprog(-np.ones(n), A_ub=A, b_ub=d,
                      bounds=[(0.0, float(u)) for u in ub], method="highs")
        if res.success:
            # HiGHS reports marginals for the MINIMISED objective (-sum r), so
            # they are <= 0; negate to get gains for the maximisation.
            lam_k = np.maximum(-np.asarray(res.ineqlin.marginals, dtype=float), 0.0)
            mu = np.maximum(-np.asarray(res.upper.marginals, dtype=float), 0.0)
            lam = np.zeros(len(i))
            if len(ki):
                # scatter the kept rows back into full-pair indexing
                lam[_pair_index(i, j, ki, kj, n)] = lam_k
            return np.maximum(res.x, 0.0), lam, mu
    r = ub.copy()                                       # scipy-free fallback
    dfull = gdist(x[i] - x[j], y[i] - y[j])
    for _ in range(400):
        excess = r[i] + r[j] - dfull
        bad = excess > 1e-15
        if not np.any(bad):
            break
        shrink = np.zeros(n)
        np.maximum.at(shrink, i[bad], excess[bad] / 2)
        np.maximum.at(shrink, j[bad], excess[bad] / 2)
        r = np.maximum(r - shrink, 0.0)
    return r, np.zeros(len(i)), np.zeros(n)


def _pair_index(i, j, ki, kj, n):
    """Position of pairs (ki,kj) inside the full triu ordering of pairs(n)."""
    return (ki * (2 * n - ki - 1)) // 2 + (kj - ki - 1)


def lp_radii(x, y, n):
    return lp_solve(x, y, n)[0]


# ---------------------------------------------- outer layer: joint SLSQP -----

def slsqp(z0, n, maxiter=300, pair_set=None, rmax=None, step=None):
    """Joint SLSQP over (x, y, r).

    pair_set restricts which pair constraints are modelled; rmax caps the radii
    and step caps how far a centre may move.  Those three go together -- see
    live_pairs() -- and with all three the reduced problem provably cannot
    violate a row that was left out.
    """
    if not HAVE_SCIPY:
        return z0
    i, j = pairs(n) if pair_set is None else pair_set
    npair = len(i)
    inr = shape_inradius()
    rcap = np.full(n, inr) if rmax is None else np.asarray(rmax, dtype=float)
    ar = np.arange(npair)
    nwall = wall_rows(np.zeros(1), np.zeros(1))[0].shape[0]

    def cons(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        d = gdist(x[i] - x[j], y[i] - y[j])
        F, _, _ = wall_rows(x, y)
        return np.concatenate([d - r[i] - r[j], (F - r[None, :]).ravel()])

    def cons_jac(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dx, dy = x[i] - x[j], y[i] - y[j]
        gx, gy = SHAPE.gauge_grad(dx, dy)
        J = np.zeros((npair + nwall * n, 3 * n))
        J[ar, i] = gx
        J[ar, j] = -gx
        J[ar, n + i] = gy
        J[ar, n + j] = -gy
        J[ar, 2 * n + i] = -1.0
        J[ar, 2 * n + j] = -1.0
        e = np.arange(n)
        _, Fx, Fy = wall_rows(x, y)
        for k in range(nwall):
            row = npair + k * n + e
            J[row, e] = Fx[k]
            J[row, n + e] = Fy[k]
            J[row, 2 * n + e] = -1.0
        return J

    z0 = z0.copy()
    np.minimum(z0[2 * n:], rcap, out=z0[2 * n:])       # start inside the bounds
    xlo, xhi, ylo, yhi = CONTAINER.bbox
    blo = np.concatenate([np.full(n, xlo), np.full(n, ylo)])
    bhi = np.concatenate([np.full(n, xhi), np.full(n, yhi)])
    if step is None:
        xy_bounds = [(float(a), float(b)) for a, b in zip(blo, bhi)]
    else:
        c0 = z0[:2 * n]
        lo = np.clip(c0 - step, blo, bhi)
        hi = np.clip(c0 + step, blo, bhi)
        xy_bounds = [(float(a), float(b)) for a, b in zip(lo, hi)]
    res = minimize(lambda z: -np.sum(z[2 * n:]), z0, jac=lambda z: np.concatenate(
                       [np.zeros(2 * n), -np.ones(n)]),
                   method="SLSQP",
                   bounds=xy_bounds + [(0.0, float(c)) for c in rcap],
                   constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                   options={"maxiter": maxiter, "ftol": 1e-12})
    return res.x


# MEASURED (artifacts/iter3/speedup.log): the trust-region QP is NOT faster than
# the dense one (0.1x / 1.0x / 1.1x at n=49/64/100) -- the passes it needs to
# re-earn the movement it gave up cost as much as the rows it saved.  So it is
# off by default and reachable with --sparse; the LP-side reduction above, which
# IS exact and needs no trust region, stays on.  Do not describe this as a
# speedup: it has not been shown to be one.
SPARSE_FROM = 10 ** 9


def _slsqp_cutting_plane(z, n, cuts=3, passes=3):
    """SLSQP on the local contact neighbourhood, inside a trust region.

    A dropped row is only safe if the solver cannot reach it.  Bounding radii by
    cap AND centre motion by step gives exactly that guarantee: a pair left out
    satisfies d_ij >= cap_i + cap_j + 2*step at the start, both centres move by
    at most step, so afterwards d_ij >= cap_i + cap_j >= r_i + r_j.  It cannot
    be violated -- which is the difference between a reduction and a shortcut.
    Without the trust region the solver slides excluded circles clean through
    each other and lands on coincident centres, an absorbing state the
    linearisation cannot recover from (measured: it does, immediately).

    The cutting plane is then belt-and-braces rather than the safety argument:
    after every solve all n(n-1)/2 constraints are evaluated, and anything that
    did bind is added before re-solving.  repair() still has the last word.
    """
    i, j = pairs(n)
    for _ in range(passes):
        x, y = z[:n], z[n:2 * n]
        u, _ = valid_caps(x, y, n)
        step = float(np.mean(u))
        cap = u + step
        D = gdist(x[i] - x[j], y[i] - y[j])
        extra = D < cap[i] + cap[j] + 2.0 * step
        before = float(np.sum(z[2 * n:]))
        for _ in range(cuts):
            sel = np.flatnonzero(extra)
            z = slsqp(z, n, pair_set=(i[sel], j[sel]), rmax=cap, step=step)
            x, y, r = z[:n], z[n:2 * n], z[2 * n:]
            viol = r[i] + r[j] - gdist(x[i] - x[j], y[i] - y[j]) > 1e-12
            new = viol & ~extra
            if not np.any(new):
                break
            extra |= new                          # cut off what the model missed
        if float(np.sum(z[2 * n:])) <= before + 1e-12:
            break                                 # trust region bought nothing
    return z


# ------------------------------- the inner layer, one level up (polytopes) ---

def lp_joint(z, n, rounds=40, tol=1e-11):
    """Exact global step over CENTRES AND RADII TOGETHER, for polytopal shapes.

    For a polytopal K the non-overlap condition gamma_K(c_i-c_j) >= r_i+r_j is
    a DISJUNCTION of linear constraints -- it holds as soon as one facet normal
    a_k separates the pair.  Fix the separating facet of every pair (take the
    one attaining the gauge at the current config) and the entire problem is a
    linear program in (x, y, r):

        max sum r   s.t.  a_k(ij).(c_i - c_j) >= r_i + r_j   for every pair
                          a_m.c_i + h_K(a_m) r_i <= b_m      for every wall

    Two facts make this a genuine ascent and not a heuristic:
      * every LP-feasible point is TRULY feasible -- one facet separating is
        sufficient, since gamma = max over ALL facets >= the chosen one.  So no
        repair step and no tolerance argument is involved; the output is exact.
      * the incoming config is itself LP-feasible (its own gauge-attaining
        facets are the ones selected), so the LP optimum is >= where we started.
    Re-selecting the facets and re-solving is therefore monotone, and each step
    is a global optimum inside its combinatorial cell -- strictly more than the
    radii-only LP, which can only move r.  This is also why a nonsmooth gauge
    costs the search nothing: the kinks that break SLSQP are exactly the
    combinatorial choices this makes explicit.
    """
    if not HAVE_SCIPY or SHAPE.is_ball:
        return z                                       # smooth case: SLSQP owns it
    A_s = SHAPE.A
    Ac, bc, h = CONTAINER.A, CONTAINER.c, _wall_h()
    m = len(bc)
    i, j = pairs(n)
    npair, ar, e = len(i), np.arange(len(i)), np.arange(n)
    xlo, xhi, ylo, yhi = CONTAINER.bbox
    rcap = shape_inradius()
    best = float(np.sum(z[2 * n:]))
    W = np.zeros((m * n, 3 * n))                       # wall rows never change
    for k in range(m):
        W[k * n + e, e] = Ac[k, 0]
        W[k * n + e, n + e] = Ac[k, 1]
        W[k * n + e, 2 * n + e] = h[k]
    wb = np.repeat(bc, n)
    for _ in range(rounds):
        x, y = z[:n], z[n:2 * n]
        vx, vy = x[i] - x[j], y[i] - y[j]
        a = A_s[SHAPE.active(vx, vy)]                  # separating direction
        P = np.zeros((npair, 3 * n))                   # -a.(c_i-c_j)+r_i+r_j<=0
        P[ar, i] = -a[:, 0]
        P[ar, j] = a[:, 0]
        P[ar, n + i] = -a[:, 1]
        P[ar, n + j] = a[:, 1]
        P[ar, 2 * n + i] = 1.0
        P[ar, 2 * n + j] = 1.0
        res = linprog(np.concatenate([np.zeros(2 * n), -np.ones(n)]),
                      A_ub=np.vstack([P, W]),
                      b_ub=np.concatenate([np.zeros(npair), wb]),
                      bounds=([(xlo, xhi)] * n + [(ylo, yhi)] * n
                              + [(0.0, rcap)] * n),
                      method="highs")
        if not res.success:
            break
        s = float(np.sum(res.x[2 * n:]))
        if s <= best + tol:
            if s > best:
                z, best = res.x, s
            break
        z, best = res.x, s
    return z


def refine(z, n, rounds=3, sparse=None):
    """SLSQP <-> exact-LP alternation, then strict repair."""
    if not SHAPE.is_ball:
        # polytopal shape: the joint LP dominates SLSQP (global per cell, and
        # exactly feasible), so use it and keep repair() only as a guard.
        z = repair(np.asarray(z, dtype=float), n)
        z = lp_joint(z, n)
        x, y = z[:n], z[n:2 * n]
        r_lp = lp_radii(x, y, n)
        if r_lp.sum() > z[2 * n:].sum():
            z = np.concatenate([x, y, r_lp])
        return repair(z, n)
    if sparse is None:
        sparse = n >= SPARSE_FROM
    for _ in range(rounds):
        z = _slsqp_cutting_plane(z, n) if sparse else slsqp(z, n)
        x, y = z[:n], z[n:2 * n]
        r_lp = lp_radii(x, y, n)
        if r_lp.sum() > z[2 * n:].sum():
            z = np.concatenate([x, y, r_lp])
        else:
            break
    return repair(z, n)


# --------------------------------------------------------------- starts -----

def _scale():
    xlo, xhi, ylo, yhi = CONTAINER.bbox
    return min(xhi - xlo, yhi - ylo)


def start_random(n, rng):
    x, y = CONTAINER.sample(n, rng, inset=0.03 * _scale())
    return np.concatenate([x, y, np.full(n, 0.01 * _scale())])


def start_grid(n, rng, jitter=0.03):
    """Rows of a (possibly staggered) grid -- the structure good packings have.

    The square keeps its original code path byte-for-byte (it is the one whose
    behaviour is already measured at n=26..100); non-box containers get the
    same idea over the bounding box, with points outside the container filtered
    out and the grid densified until n survive.
    """
    if CONTAINER.is_box:
        xlo, xhi, ylo, yhi = CONTAINER.bbox
        W, H = xhi - xlo, yhi - ylo
        k = int(math.ceil(math.sqrt(n)))
        rows, pts = [], []
        left = n
        while left > 0:
            rows.append(min(k, left) if rng.random() < 0.7 else min(k - 1, left))
            rows[-1] = max(1, rows[-1])
            left -= rows[-1]
        stagger = rng.random() < 0.5
        for ri, cnt in enumerate(rows):
            yy = (ri + 0.5) / len(rows)
            off = 0.5 / cnt if (stagger and ri % 2) else 0.0
            for ci in range(cnt):
                pts.append(((ci + 0.5) / cnt + off, yy))
        pts = np.array(pts[:n], dtype=float)
        pts += rng.normal(0, jitter, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        pts[:, 0] = xlo + pts[:, 0] * W
        pts[:, 1] = ylo + pts[:, 1] * H
        return np.concatenate([pts[:, 0], pts[:, 1], np.full(n, 0.01)])

    xlo, xhi, ylo, yhi = CONTAINER.bbox
    W, H, sc = xhi - xlo, yhi - ylo, _scale()
    inset = 0.02 * sc
    stagger = rng.random() < 0.5
    for mult in (1.0, 1.3, 1.7, 2.2, 3.0, 4.0):
        k = max(1, int(math.ceil(math.sqrt(n) * mult)))
        pts = []
        for ri in range(k):
            yy = ylo + (ri + 0.5) / k * H
            cnt = max(1, k - 1) if (stagger and ri % 2) else k
            off = 0.5 / cnt if (stagger and ri % 2) else 0.0
            for ci in range(cnt):
                pts.append((xlo + ((ci + 0.5) / cnt + off) * W, yy))
        P = np.array(pts, dtype=float)
        P = P[CONTAINER.cap(P[:, 0], P[:, 1]) >= inset]
        if len(P) >= n:
            P = P[rng.choice(len(P), size=n, replace=False)]
            P += rng.normal(0, jitter * sc, P.shape)
            px, py = CONTAINER.project(P[:, 0], P[:, 1], inset=0.5 * inset)
            return np.concatenate([px, py, np.full(n, 0.01 * sc)])
    return start_random(n, rng)


def _kick(z, n, idx, rng, sigma):
    z = z.copy()
    k = len(idx)
    px, py = CONTAINER.project(z[idx] + rng.normal(0, sigma, k),
                               z[n + idx] + rng.normal(0, sigma, k),
                               inset=0.02 * _scale())
    z[idx], z[n + idx] = px, py
    z[2 * n + idx] *= 0.3
    return z


def perturb(z, n, rng, frac=0.25, sigma=0.10):
    """Baseline basin hop: move a uniformly random subset of circles."""
    k = max(1, int(frac * n))
    return _kick(z, n, rng.choice(n, size=k, replace=False), rng, sigma)


def perturb_dual(z, n, rng, lam, frac=0.25, sigma=0.10):
    """Basin hop guided by the LP duals.

    lam[k] is the objective gain per unit of extra room at contact k, so the
    contacts with the largest lam are the ones actually holding sum(r) down.
    Sample contacts with probability proportional to lam and move BOTH ends of
    each -- that targets the load-bearing part of the contact graph instead of
    spending hops on circles whose radius is capped by the wall (which no amount
    of sliding will improve).  Falls back to a uniform hop if every dual is 0.
    """
    k = max(1, int(frac * n))
    w = np.asarray(lam, dtype=float)
    if w.size == 0 or not np.isfinite(w).all() or w.sum() <= 0:
        return perturb(z, n, rng, frac, sigma)
    i, j = pairs(n)
    live = np.flatnonzero(w > 0)                     # only tight contacts have a price
    if live.size == 0:
        return perturb(z, n, rng, frac, sigma)
    live = live[rng.permutation(live.size)]          # break ties without bias
    p = w[live] / w[live].sum()
    picks = rng.choice(live.size, size=min(live.size, 4 * k), replace=False, p=p)
    chosen = []
    seen = set()
    for c in live[picks]:
        for e in (int(i[c]), int(j[c])):
            if e not in seen:
                seen.add(e)
                chosen.append(e)
        if len(chosen) >= k:
            break
    return _kick(z, n, np.array(chosen[:max(k, 2)], dtype=int), rng, sigma)


# ---------------------------------------------------------------- search -----

def search(n, seed=0, budget=60.0, warm=None, verbose=True, dual=False, sparse=None):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best, best_s = None, -1.0
    best_lam = None
    if warm is not None:
        z = refine(np.asarray(warm, dtype=float), n, sparse=sparse)
        best, best_s = z, score(z, n)
        if verbose:
            print(f"  warm start: {best_s:.9f}")
    tries = hops = 0
    while time.time() - t0 < budget:
        if best is not None and rng.random() < 0.65:
            kw = dict(frac=rng.choice([0.12, 0.25, 0.45]),
                      sigma=rng.choice([0.04, 0.10, 0.20]))
            if dual:
                if best_lam is None:            # duals of the incumbent, cached
                    best_lam = lp_solve(best[:n], best[n:2 * n], n)[1]
                z0 = perturb_dual(best, n, rng, best_lam, **kw)
            else:
                z0 = perturb(best, n, rng, **kw)
            hops += 1
        else:
            z0 = start_grid(n, rng) if rng.random() < 0.5 else start_random(n, rng)
            tries += 1
        z = refine(z0, n, sparse=sparse)
        if max_violation(z, n) > 1e-9:
            continue
        s = score(z, n)
        if s > best_s:
            best, best_s = z, s
            best_lam = None                     # incumbent moved: duals are stale
            if verbose:
                print(f"  [{time.time()-t0:6.1f}s] new best {s:.9f} "
                      f"(starts={tries} hops={hops})")
    if verbose:
        print(f"  done: {tries} fresh starts, {hops} hops, best {best_s:.9f}")
    return best, best_s


# ------------------------------------------------------------- self-test -----

def self_test():
    n = 6
    rng = np.random.default_rng(0)
    # LP layer is exact: for 2 circles at distance d it must return the optimum
    x = np.array([0.25, 0.75]); y = np.array([0.5, 0.5])
    r = lp_radii(x, y, 2)
    assert abs(r.sum() - 0.5) < 1e-9, r          # both capped by the walls
    # repair() makes any junk strictly feasible
    z = np.concatenate([rng.uniform(0, 1, 2 * n), np.full(n, 0.4)])
    zr = repair(z, n)
    v = max_violation(zr, n)
    assert v <= 1e-9, v
    # LP radii never violate feasibility
    xs, ys = rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n)
    zl = np.concatenate([xs, ys, lp_radii(xs, ys, n)])
    assert max_violation(zl, n) <= 1e-9, max_violation(zl, n)
    # ---- LP DUALS -----------------------------------------------------------
    if HAVE_SCIPY:
        # two circles far from the walls: the contact carries the whole load, so
        # its dual price must be exactly 1 (one extra unit of gap -> one of sum r)
        xt = np.array([0.35, 0.65]); yt = np.array([0.5, 0.5])
        rt, lamt, mut = lp_solve(xt, yt, 2)
        assert abs(lamt[0] - 1.0) < 1e-9, lamt
        # strong duality + complementary slackness: every circle with r_i > 0
        # satisfies sum_j lam_ij + mu_i == 1 exactly.  If the sign convention
        # were wrong this identity would fail.
        xs, ys = rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n)
        rr, lam, mu = lp_solve(xs, ys, n)
        ii, jj = pairs(n)
        load = np.zeros(n)
        np.add.at(load, ii, lam)
        np.add.at(load, jj, lam)
        pos = rr > 1e-9
        err = float(np.max(np.abs(load[pos] + mu[pos] - 1.0)))
        assert err < 1e-7, f"LP dual identity violated by {err:.2e}"
        # dual objective must equal the primal objective
        d_ij = np.hypot(xs[ii] - xs[jj], ys[ii] - ys[jj])
        ub = np.maximum(np.minimum(np.minimum(xs, 1 - xs), np.minimum(ys, 1 - ys)), 0.0)
        gap = abs(float(lam @ d_ij + mu @ ub) - float(rr.sum()))
        assert gap < 1e-7, f"duality gap {gap:.2e}"
        # perturb_dual must move only real circles and keep the array shape
        zz = np.concatenate([xs, ys, rr])
        zp = perturb_dual(zz, n, rng, lam, frac=0.34, sigma=0.1)
        assert zp.shape == zz.shape and np.any(zp != zz)
        # with all-zero duals it must degrade gracefully to a uniform hop
        assert perturb_dual(zz, n, rng, np.zeros(len(ii))).shape == zz.shape

    # ---- the contact-graph reduction must be EXACT, not approximate ---------
    if HAVE_SCIPY:
        def lp_dense(xx, yy, m):
            """Deliberately naive reference: every pair, wall bounds only."""
            ii, jj = pairs(m)
            A = np.zeros((len(ii), m))
            A[np.arange(len(ii)), ii] = 1.0
            A[np.arange(len(ii)), jj] = 1.0
            b = np.hypot(xx[ii] - xx[jj], yy[ii] - yy[jj])
            w = np.maximum(np.minimum(np.minimum(xx, 1 - xx),
                                      np.minimum(yy, 1 - yy)), 0.0)
            res = linprog(-np.ones(m), A_ub=A, b_ub=b,
                          bounds=[(0.0, float(t)) for t in w], method="highs")
            assert res.success
            return float(np.sum(res.x))

        worst_gap, worst_keep = 0.0, 1.0
        for m, seed in ((12, 3), (25, 4), (40, 5)):
            g = np.random.default_rng(seed)
            for xs2, ys2 in ((g.uniform(0.05, 0.95, m), g.uniform(0.05, 0.95, m)),
                             (np.tile(np.linspace(.1, .9, 5), m // 5 + 1)[:m],
                              np.repeat(np.linspace(.1, .9, 5), 5)[:m])):
                xs2, ys2 = np.asarray(xs2, float), np.asarray(ys2, float)
                if len(ys2) < m:                       # grid arm may be short
                    continue
                ki, kj, _ = live_pairs(xs2, ys2, m)
                gap = abs(float(lp_radii(xs2, ys2, m).sum()) - lp_dense(xs2, ys2, m))
                worst_gap = max(worst_gap, gap)
                worst_keep = min(worst_keep, len(ki) / max(1, m * (m - 1) // 2))
                assert gap < 1e-9, f"reduction lost {gap:.2e} of the optimum at n={m}"
        assert worst_keep < 0.5, f"reduction kept {worst_keep:.0%} of rows -- no gain"
        # caps must be valid bounds: no optimal radius may exceed its own cap
        g = np.random.default_rng(7)
        xs3, ys3 = g.uniform(0.05, 0.95, 30), g.uniform(0.05, 0.95, 30)
        u3, _ = valid_caps(xs3, ys3, 30)
        assert np.all(lp_radii(xs3, ys3, 30) <= u3 + 1e-12), "cap u_i is not valid"
        # the sparse refine path must still emit a config feasible on ALL pairs
        zc = _slsqp_cutting_plane(np.concatenate(
            [xs3, ys3, np.full(30, 0.01)]), 30)
        assert max_violation(repair(zc, 30), 30) <= 1e-9
        print(f"  reduction exact (worst gap {worst_gap:.1e}, "
              f"kept as few as {worst_keep:.1%} of rows)")

    # ---- the SHAPE axis: homothets of a polytope -----------------------------
    if HAVE_SCIPY:
        try:
            set_shape(shp.SQUARE)                      # axis-aligned squares
            # 1. wall divisor: h_K(unit normal of the box) = 1 for L-inf, so the
            #    cap at the centre of the unit square is 0.5 (half-side), and the
            #    largest single square is the container itself.
            assert abs(shape_inradius() - 0.5) < 1e-9, shape_inradius()
            # 2. every lp_joint iterate is EXACTLY feasible -- no repair needed.
            g = np.random.default_rng(11)
            for m2 in (5, 9):
                z0 = np.concatenate([g.uniform(0.1, 0.9, 2 * m2), np.full(m2, 0.01)])
                before = float(np.sum(z0[2 * m2:]))
                zj = lp_joint(z0, m2)
                v = max_violation(zj, m2)
                assert v <= 1e-12, f"lp_joint emitted a violation {v:.2e}"
                assert float(np.sum(zj[2 * m2:])) >= before - 1e-12, "not monotone"
            # 3. PROVED optimum, at full instance size: n = k^2 axis-aligned
            #    squares in the unit square have max sum(r) = sqrt(n)/2 exactly
            #    (Cauchy-Schwarz on the area bound, attained by the k x k grid).
            #    The search must reach it -- this tests the whole shape path
            #    against a theorem rather than against another search.
            for k in (2, 3):
                zs, ss = search(k * k, seed=2, budget=3.0, verbose=False)
                opt = math.sqrt(k * k) / 2.0
                assert max_violation(zs, k * k) <= 1e-9
                assert ss <= opt + 1e-9, f"n={k*k} beat a PROVED optimum: {ss} > {opt}"
                assert ss > opt - 1e-4, f"n={k*k} got {ss:.9f}, proved optimum {opt}"
                print(f"  square shape n={k*k}: {ss:.9f} vs proved optimum {opt}")
            # 4. a non-Euclidean shape in a non-polyhedral container must be
            #    refused loudly, not silently mis-scored.
            set_container(ctn.UNIT_DISK)
            try:
                wall_cap(np.zeros(1), np.zeros(1))
            except AssertionError:
                pass
            else:
                raise AssertionError("accepted a polytopal shape in a disk")
        finally:
            set_shape(shp.BALL)
            set_container(ctn.UNIT_SQUARE)

    # a short search on n=6 must beat a naive 2x3 grid of equal circles
    z, s = search(n, seed=1, budget=4.0, verbose=False)
    assert max_violation(z, n) <= 1e-9
    naive = 6 * (1.0 / 6.0)          # 3x2 grid of touching circles, r=1/6
    assert s > naive, (s, naive)
    zd, sd = search(n, seed=1, budget=4.0, verbose=False, dual=True)
    assert max_violation(zd, n) <= 1e-9 and sd > naive, (sd, naive)
    print(f"self-test OK (scipy={HAVE_SCIPY}, n=6 sum_r={s:.6f} > naive {naive:.6f}, "
          f"violation={max_violation(z, n):.2e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=26)
    ap.add_argument("--container", default="unit_square",
                    help="unit_square | unit_disk | right_triangle")
    ap.add_argument("--shape", default="disk",
                    help="packed object: disk | square | diamond | hexagon")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", type=float, default=60.0, help="search seconds")
    ap.add_argument("--warm", help="json config to warm-start from")
    ap.add_argument("--dual", action="store_true",
                    help="choose basin-hop targets from the LP duals")
    ap.add_argument("--sparse", action="store_true",
                    help="trust-region QP on the contact neighbourhood "
                         "(exact, but measured no faster than dense -- off by default)")
    ap.add_argument("-o", "--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    ct = set_container(ctn.get(a.container))
    sh = set_shape(shp.get(a.shape))
    warm = None
    if a.warm:
        d = json.load(open(a.warm))
        c = np.array(d["circles"] if isinstance(d, dict) else d, dtype=float)
        assert len(c) == a.n, f"warm config has {len(c)} circles, need {a.n}"
        warm = np.concatenate([c[:, 0], c[:, 1], c[:, 2]])
    z, s = search(a.n, seed=a.seed, budget=a.time, warm=warm, dual=a.dual,
                  sparse=True if a.sparse else None)
    v = max_violation(z, a.n)
    print(f"sum_r = {s:.9f}  max_violation = {v:.2e}")
    assert v <= 1e-9, "refused to emit an infeasible config"
    if a.out:
        c = np.stack([z[:a.n], z[a.n:2 * a.n], z[2 * a.n:]], axis=1)
        json.dump({"source": "tools/pack.py",
                   "container": ct.spec(), "shape": sh.spec(),
                   "method": ("multi-start joint-LP ascent + exact-LP radii + "
                              "basin hopping" if not sh.is_ball else
                              "multi-start SLSQP + exact-LP radii + basin hopping"),
                   "seed": a.seed, "budget_s": a.time, "scipy": HAVE_SCIPY,
                   "n": a.n, "sum_radii": s, "max_violation": v,
                   "circles": [[float(t) for t in row] for row in c]},
                  open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
