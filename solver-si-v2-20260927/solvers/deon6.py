"""Circle-packing solver: SEQUENTIAL LP inner-approximation (primary) + SLSQP (kept as a 2nd line),
teleport basin-hopping, and a worst-digits-first CPU allocator.

WHY THIS SHAPE (iteration 2 -- the SLSQP core is replaced, not tuned)
---------------------------------------------------------------------
Iteration 1's engine was dense SLSQP on z=(x,y,r). Its Jacobian is a dense (m+4n, 3n) matrix and its
QP subproblem is dense too, so one polish costs ~32 CPU-s at n=99. The 120 s backstop then ran out
before n>=91 was ever touched. The fix is not a cheaper allocator, it is a cheaper *iteration*:

  SEQUENTIAL LINEAR PROGRAMMING AS AN INNER APPROXIMATION.
  The separation constraint  ||c_i - c_j|| >= r_i + r_j  is a CONVEX function of the centres, so it
  lies ABOVE every one of its tangent planes:  ||c_i - c_j|| >= u . (c_i - c_j)  for ANY unit u.
  Replacing the norm by u . (c_i - c_j), with u the unit direction at the current iterate, therefore
  RESTRICTS the feasible set -- it never enlarges it. The resulting subproblem

      max  sum r   s.t.   u_ij . (c_i - c_j) >= r_i + r_j,   walls linear,   |dc| <= t,  r <= r0 + t

  is a sparse LP (3n columns, ~6n rows) that HiGHS solves in milliseconds even at n=99. Consequences:
    * EVERY LP solution is a genuinely feasible packing -- no penalty, no restoration phase;
    * the current iterate is feasible for the LP, so sum r is MONOTONE NON-DECREASING. Warm-starting
      from the committed census can therefore never make an n worse;
    * a pair may be dropped from the LP whenever its slack exceeds 4t (the move can close at most
      2t of distance and open at most 2t of radius), which is a *proof*, not a heuristic cutoff --
      iteration 1's 3.2/sqrt(n) prune had no such guarantee and had to be tuned by hand;
    * ~3 ms/iteration at n=99 versus ~32 s for one dense SLSQP polish, i.e. hundreds of trust-region
      steps per n inside the same budget, which is what makes basin-hopping affordable at all.

  SLSQP is kept (`_polish`) as an independent second line of attack: it converges to different KKT
  points than SLP does, and on a converged incumbent it is used as an alternate move.

  TELEPORT MOVES.  Local optima of max-sum-radii are usually "one small circle stuck in a bad seat".
  `_teleport` finds the largest empty disc by sampling and moves the smallest circle(s) there, then
  lets SLP re-converge. This is a structured basin hop; plain gaussian jitter is kept alongside it.

ALLOCATOR.  Digits, not cost. Each target's current digit count is read from the committed census and
bench/records.json (guarded), targets are processed WORST-DIGITS-FIRST, and the slice is weighted
toward the loose ones with all unused time flowing forward. Deadlines are computed AT ENTRY, so a
second solve() call in the same process gets a full fresh budget, and len(targets)==1 gets everything.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog, minimize
from scipy.sparse import coo_matrix

LO, HI = -0.5, 0.5
CUT = 3.2           # SLSQP pair-pruning radius is CUT/sqrt(n)
CPU_BUDGET = 104.0  # process-CPU seconds this CALL may use; re-based at every entry, never global


# ------------------------------------------------------------------ feasible radii for fixed centres
def _wall(xy):
    return np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0], xy[:, 1] - LO, HI - xy[:, 1]])


def _grow(xy):
    """Cheap guaranteed-feasible cap r_i = min(wall_i, min_j d_ij/2). Used only as a cold fallback."""
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return np.maximum(np.minimum(_wall(xy), d.min(1) / 2.0), 1e-12)


def _lp_radii(xy):
    """OPTIMAL radii for these centres: max sum r s.t. r_i+r_j <= d_ij, 0 <= r_i <= wall_i.
    Pairs with d_ij >= wall_i + wall_j can never bind and are dropped."""
    n = len(xy)
    w = np.maximum(_wall(xy), 0.0)
    ii, jj = np.triu_indices(n, 1)
    dv = np.sqrt((xy[ii, 0] - xy[jj, 0]) ** 2 + (xy[ii, 1] - xy[jj, 1]) ** 2)
    keep = dv < w[ii] + w[jj]
    ii, jj, dv = ii[keep], jj[keep], dv[keep]
    m = len(dv)
    if m == 0:
        return w.copy()
    rows = np.repeat(np.arange(m), 2)
    cols = np.empty(2 * m, dtype=np.int64)
    cols[0::2], cols[1::2] = ii, jj
    A = coo_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n))
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=dv, bounds=np.stack([np.zeros(n), w], 1),
                      method="highs")
    except Exception:
        return _grow(xy)
    if res.x is None or not np.all(np.isfinite(res.x)):
        return _grow(xy)
    return np.maximum(res.x, 0.0)


def _repair(xy, r):
    """Make (xy, r) STRICTLY feasible in float arithmetic: clip centres into the square, floor the
    radii strictly positive, then scale ALL radii by 1/violation (both the wall and pair constraints
    are homogeneous in r, so one uniform scale fixes both at once)."""
    xy = np.clip(xy, LO + 1e-12, HI - 1e-12)
    n = len(xy)
    w = np.maximum(_wall(xy), 1e-12)
    r = np.clip(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0), 1e-12, None)
    v = np.max(r / w)
    if n >= 2:
        ii, jj = np.triu_indices(n, 1)
        d = np.sqrt((xy[ii, 0] - xy[jj, 0]) ** 2 + (xy[ii, 1] - xy[jj, 1]) ** 2)
        v = max(v, np.max((r[ii] + r[jj]) / np.maximum(d, 1e-300)))
    if v > 1.0:
        r = r / v
    r = r * (1.0 - 1e-13)
    return np.concatenate([xy, np.maximum(r, 1e-300)[:, None]], axis=1)


# ------------------------------------------------------- the sequential-LP engine (primary attack)
def _slp_step(xy, r, t):
    """One trust-region LP: max sum r over the tangent-plane INNER approximation of the true feasible
    set. Returns (xy', r') -- always a feasible packing -- or None if the LP failed.

    Pair (i,j) may be dropped when its slack d_ij-(r_i+r_j) exceeds 4t: the centres move at most t
    each (closing <= 2t) and the radii grow at most t each (opening <= 2t), so a dropped pair CANNOT
    become violated. That makes the sparsity exact rather than heuristic.
    """
    n = len(xy)
    x0, y0 = xy[:, 0], xy[:, 1]
    ii, jj = np.triu_indices(n, 1)
    dx, dy = x0[ii] - x0[jj], y0[ii] - y0[jj]
    d = np.sqrt(dx * dx + dy * dy)
    keep = (d - (r[ii] + r[jj])) < 4.0 * t
    keep &= d > 1e-12                                   # coincident centres have no tangent direction
    ii, jj, dx, dy, d = ii[keep], jj[keep], dx[keep], dy[keep], d[keep]
    m = len(d)
    ux, uy = dx / d, dy / d

    # rows 0..m-1 : r_i + r_j - u.(c_i - c_j) <= 0     (6 nonzeros each)
    # rows m..    : the four wall constraints           (2 nonzeros each)
    rows = np.repeat(np.arange(m), 6)
    cols = np.empty(6 * m, dtype=np.int64)
    vals = np.empty(6 * m)
    cols[0::6], vals[0::6] = ii, -ux
    cols[1::6], vals[1::6] = jj, ux
    cols[2::6], vals[2::6] = n + ii, -uy
    cols[3::6], vals[3::6] = n + jj, uy
    cols[4::6], vals[4::6] = 2 * n + ii, 1.0
    cols[5::6], vals[5::6] = 2 * n + jj, 1.0
    idx = np.arange(n)
    wrow, wcol, wval = [], [], []
    for k, (var, sgn) in enumerate(((0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0))):
        base = m + k * n
        wrow.append(base + idx); wcol.append(var * n + idx); wval.append(np.full(n, sgn))
        wrow.append(base + idx); wcol.append(2 * n + idx); wval.append(np.ones(n))
    rows = np.concatenate([rows] + wrow)
    cols = np.concatenate([cols] + wcol)
    vals = np.concatenate([vals] + wval)
    A = coo_matrix((vals, (rows, cols)), shape=(m + 4 * n, 3 * n))
    b = np.concatenate([np.zeros(m), np.full(4 * n, HI)])

    lb = np.concatenate([np.maximum(x0 - t, LO), np.maximum(y0 - t, LO), np.zeros(n)])
    ub = np.concatenate([np.minimum(x0 + t, HI), np.minimum(y0 + t, HI),
                         np.minimum(r + t, HI - LO)])
    c = np.zeros(3 * n)
    c[2 * n:] = -1.0
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=np.stack([lb, ub], 1), method="highs")
    except Exception:
        return None
    z = res.x
    if z is None or not np.all(np.isfinite(z)):
        return None
    return np.stack([z[:n], z[n:2 * n]], 1), np.maximum(z[2 * n:], 0.0)


def _slp(xy, r, deadline, t0, tmin=1e-11, max_it=100000):
    """Trust-region SLP to convergence (or the deadline). sum r is monotone non-decreasing."""
    pk = _repair(xy, r)
    xy, r = pk[:, :2], pk[:, 2]
    s = float(r.sum())
    t = float(t0)
    for _ in range(int(max_it)):
        if t < tmin or time.process_time() >= deadline:
            break
        out = _slp_step(xy, r, t)
        if out is None:
            t *= 0.5
            continue
        pk = _repair(out[0], out[1])
        s2 = float(pk[:, 2].sum())
        if s2 > s + 1e-15:
            gain = s2 - s
            xy, r, s = pk[:, :2], pk[:, 2], s2
            t = min(t * 1.6, 0.25) if gain > 0.3 * t else t * 0.7
        else:
            t *= 0.4                                    # no progress at this radius -> tighten
    return xy, r, s


# ------------------------------------------------------------------- the SLSQP polish (2nd attack)
def _polish(xy, r, cut, maxiter):
    """SLSQP on z = (x, y, r) with analytic jacobians and distance-pruned pair constraints."""
    n = len(xy)
    ai, aj = np.triu_indices(n, 1)
    d = np.sqrt((xy[ai, 0] - xy[aj, 0]) ** 2 + (xy[ai, 1] - xy[aj, 1]) ** 2)
    sel = d < cut
    ii, jj = ai[sel], aj[sel]
    m = len(ii)
    rows, idx = np.arange(m), np.arange(n)
    J = np.zeros((m + 4 * n, 3 * n))
    for k, (sx, sy) in enumerate(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))):
        b = m + k * n
        if sx:
            J[b + idx, idx] = sx
        if sy:
            J[b + idx, n + idx] = sy
        J[b + idx, 2 * n + idx] = -1.0
    g = np.zeros(3 * n)
    g[2 * n:] = -1.0

    def fun(z):
        return -float(z[2 * n:].sum())

    def jac(z):
        return g

    def cf(z):
        x, y, rr = z[:n], z[n:2 * n], z[2 * n:]
        dx, dy = x[ii] - x[jj], y[ii] - y[jj]
        s = rr[ii] + rr[jj]
        return np.concatenate([dx * dx + dy * dy - s * s,
                               x - rr - LO, HI - x - rr, y - rr - LO, HI - y - rr])

    def cj(z):
        x, y, rr = z[:n], z[n:2 * n], z[2 * n:]
        dx, dy = x[ii] - x[jj], y[ii] - y[jj]
        s = rr[ii] + rr[jj]
        J[rows, ii] = 2 * dx
        J[rows, jj] = -2 * dx
        J[rows, n + ii] = 2 * dy
        J[rows, n + jj] = -2 * dy
        J[rows, 2 * n + ii] = -2 * s
        J[rows, 2 * n + jj] = -2 * s
        return J

    z0 = np.concatenate([xy[:, 0], xy[:, 1], np.maximum(r, 0.0)])
    try:
        res = minimize(fun, z0, jac=jac, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": cf, "jac": cj}],
                       options={"maxiter": int(maxiter), "ftol": 1e-14})
        z = res.x
    except Exception:
        z = z0
    if not np.all(np.isfinite(z)):
        z = z0
    return np.stack([z[:n], z[n:2 * n]], 1), np.maximum(z[2 * n:], 0.0)


# ------------------------------------------------------------------------------ structured basin hop
def _teleport(xy, r, rng, k=1, nsamp=768):
    """Move the k smallest circles to the k largest empty discs (found by sampling). The classic
    escape from 'a small circle is stuck in a bad seat', which local moves cannot undo."""
    n = len(xy)
    k = max(1, min(int(k), n - 1))
    order = np.argsort(r)
    xy = xy.copy()
    for q in range(k):
        i = order[q]
        mask = np.ones(n, bool)
        mask[order[:q + 1]] = False                     # ignore the ones already lifted out
        others, ro = xy[mask], r[mask]
        P = rng.uniform(LO, HI, (nsamp, 2))
        free = np.minimum(_wall(P),
                          (np.sqrt(((P[:, None, :] - others[None]) ** 2).sum(-1)) - ro[None]).min(1))
        b = int(np.argmax(free))
        xy[i] = P[b] + rng.normal(0.0, 1e-3, 2)
    return np.clip(xy, LO + 1e-6, HI - 1e-6)



# ------------------------------------------------- BATCHED PENALTY-ASCENT RELAXER (iteration 3 core)
def _relax(C, R, iters, lr0=0.9, lr1=0.05, mu0=3.0, mu1=400.0, W=None, BAN=None,
           bx=0.5, by=0.5, WBAN=None):
    """Locally optimise a WHOLE BATCH of packings at once, in pure vectorised numpy.

    C:(B,n,2) centres, R:(B,n) radii.  Ascend  F = sum r - mu*[sum_{i<j} viol_ij^2 + sum_i wallviol_i^2]
    with mu ramped geometrically (so the iterate is driven onto the feasible boundary) and a
    SIGN-NORMALISED step (step = lr * g / max|g|), which makes the method scale-free: no per-n step
    tuning, and no divergence when a perturbation starts deep inside an overlap.

    This is the throughput lever.  One HiGHS LP solve costs ~3 ms and handles ONE configuration; one
    batched step here costs ~2 ms and handles B of them, so a basin costs ~10x less CPU -- and CPU per
    local optimum is the measured binding constraint on this census.  Precision is NOT its job: the
    exact monotone SLP finisher runs afterwards on the elite only.

    (bx, by) are the container HALF-EXTENTS. They default to the unit square; `_box_homotopy` calls
    the same relaxer with (bx, by) = (0.5*sqrt(e), 0.5/sqrt(e)) so an EQUAL-AREA RECTANGLE can be
    relaxed by exactly this code path.
    """
    B, n, _ = C.shape
    C = C.copy()
    R = np.maximum(R.copy(), 1e-9)
    eye = np.eye(n, dtype=bool)[None]
    # WALL BAN (iteration 14). The pair ban above deforms the feasible set at a PAIR contact; this
    # does the same at a WALL contact -- circle i must end the stage `dd` clear of side k. Iteration
    # 11 flagged that the constraint-space primitive was only half built (pairs, not walls); the
    # active-set pivot needs both, because a wall contact carries a dual multiplier exactly like a
    # pair contact does. Zero-cost when unused: DW stays None and the wall term is untouched.
    DW = None
    if WBAN is not None:
        wb, wi, wk, wd = WBAN
        DW = np.zeros((4, B, n))
        DW[wk, wb, wi] = wd
    for it in range(int(iters)):
        f = it / max(1.0, iters - 1.0)
        mu = mu0 * (mu1 / mu0) ** f
        lr = (lr0 * (lr1 / lr0) ** f) / np.sqrt(n)
        d = C[:, :, None, :] - C[:, None, :, :]                     # (B,n,n,2)
        dist = np.sqrt((d * d).sum(-1))
        np.copyto(dist, 1.0, where=eye)
        # a contact ban can shove two centres onto the SAME clipped box point; without this floor
        # u = d/dist is 0/0 and the whole row goes NaN (measured as a new RuntimeWarning when the
        # ban move was added). Floored, a coincident pair simply gets no push and its radii shrink.
        np.maximum(dist, 1e-12, out=dist)
        ov = np.maximum(R[:, :, None] + R[:, None, :] - dist, 0.0)
        np.copyto(ov, 0.0, where=eye)
        if BAN is not None:
            # CONTACT BAN (iteration 10). A deformation of the FEASIBLE SET rather than of the
            # objective: the listed pairs are required to stand `dd` further apart than tangency,
            # so a contact that is ACTIVE at the incumbent is forced OPEN and whatever contact the
            # rest of the packing can form instead gets to form. Released in stage 2 (BAN=None).
            bb, bi, bj, dd = BAN
            e = np.maximum(R[bb, bi] + R[bb, bj] + dd - dist[bb, bi, bj], 0.0)
            ov[bb, bi, bj] = e
            ov[bb, bj, bi] = e
        u = d / dist[..., None]
        gC = 2.0 * mu * (ov[..., None] * u).sum(2)                  # push overlapping pairs apart
        wl = np.stack([R - (C[:, :, 0] + bx), R - (bx - C[:, :, 0]),
                       R - (C[:, :, 1] + by), R - (by - C[:, :, 1])], 0)
        if DW is not None:
            wl = wl + DW
        wl = np.maximum(wl, 0.0)
        gC[:, :, 0] += 2.0 * mu * (wl[0] - wl[1])
        gC[:, :, 1] += 2.0 * mu * (wl[2] - wl[3])
        # WEIGHTED OBJECTIVE (iteration 8). W=None is the true objective sum r. A per-circle weight
        # vector instead ascends sum_i W_i r_i: circles with W_i > 1 grow INTO their neighbours, the
        # overlap penalty shoves those neighbours aside, and the contact graph REWIRES. Releasing the
        # weights afterwards (W=None + SLP) lands in a different basin of the TRUE objective.
        gR = (1.0 if W is None else W) - 2.0 * mu * (ov.sum(2) + wl.sum(0))
        g = np.concatenate([gC.reshape(B, 2 * n), gR], 1)
        sc = np.abs(g).max(1)
        sc = np.where(sc > 1e-30, sc, 1.0)[:, None]
        step = lr * g / sc
        C += step[:, :2 * n].reshape(B, n, 2)
        R = np.maximum(R + step[:, 2 * n:], 1e-9)
        np.clip(C[:, :, 0], -bx, bx, out=C[:, :, 0])
        np.clip(C[:, :, 1], -by, by, out=C[:, :, 1])
    return C, R


# ------------------------------------ BATCHED PRIMAL INTERIOR-POINT (log-barrier) -- iteration 4 core
def _minslack(C, R):
    """(B,) minimum over pair AND wall slacks. O(B n^2), no temporaries beyond the distance block."""
    B, n, _ = C.shape
    d = C[:, :, None, :] - C[:, None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    s = dist - (R[:, :, None] + R[:, None, :])
    s[:, np.arange(n), np.arange(n)] = np.inf
    w = np.minimum.reduce([C[:, :, 0] - LO, HI - C[:, :, 0],
                           C[:, :, 1] - LO, HI - C[:, :, 1]]) - R
    return np.minimum(s.reshape(B, -1).min(1), w.min(1))


def _barrier(C, R, iters, t0=2e-3, t1=1e-11, a0=None):
    """Optimise a WHOLE BATCH of packings JOINTLY in (centres, radii) to a genuine local optimum,
    in pure vectorised numpy -- no HiGHS, no scipy, no per-candidate Python loop.

    Ascend the log-barrier merit

        F_t = sum_i r_i + t * [ sum_{i<j} log(d_ij - r_i - r_j) + sum_{i,wall} log(wall - r_i) ]

    with t annealed geometrically t0 -> t1. Every iterate stays STRICTLY inside the feasible set (a
    step is rejected and that candidate's trust radius halved whenever min-slack would go
    non-positive), so EVERY iterate is itself a legal packing -- which is what makes the
    BEST-ITERATE TRACKER below sound: `bs/bC/bR` keep, per candidate, the highest sum r ever visited,
    so the routine is monotone by construction and a warm start can never come out worse than it
    went in. As t -> 0 the central path converges to the exact KKT point of  max sum r  s.t.
    non-overlap + containment: the same point _slp reaches, but for a whole BATCH at once and
    without a single LP solve.

    STEP = DIAGONAL NEWTON, NOT NORMALISED GRADIENT.  Measured (artifacts/probe4.py, first cut): a
    sign-normalised step `a*g/max|g|` -- which is exactly right for the crude penalty explorer
    `_relax` -- makes this DIVERGE, losing 0.08 of sum r at n=99 and getting worse with more
    iterations, because barrier gradients span ~10 orders of magnitude (1/s on a jammed contact vs
    1/s on a far pair) and one global scale cannot serve both. Dividing by the (always positive)
    diagonal barrier curvature

        h_r = t*[sum_j 1/s^2 + sum_walls 1/w^2]
        h_x = t*[sum_j (u_x^2/s^2 + (1-u_x^2)/(d s)) + 1/wL^2 + 1/wR^2]

    rescales every coordinate by its own conditioning, and the trust cap (max|step| <= a) keeps the
    step honest where the curvature vanishes.

    WHY THIS REPLACES THE HiGHS FINISHER (iteration 3's measured bottleneck).  Iteration 3 proved the
    census is limited by THROUGHPUT OF LOCAL OPTIMA, not by precision: _relax was cheap but produced
    no feasible point, so every candidate worth keeping still had to be converged one-at-a-time by
    _slp (>= 3 ms per trust-region LP, tens of LPs per basin, strictly serial in Python). That serial
    finisher, not the explorer, owned the CPU. The barrier is O(B n^2) numpy per step for B basins
    simultaneously -- so the exact converger now rides the SAME vectorised path as the explorer."""
    B, n, _ = C.shape
    C = np.array(C, dtype=float, copy=True)
    R = np.maximum(np.array(R, dtype=float, copy=True), 1e-12)
    idx = np.arange(n)
    amax = 0.25 / np.sqrt(n)
    a = np.full(B, (a0 if a0 is not None else 0.05 / np.sqrt(n)), dtype=float)
    bs = np.where(_minslack(C, R) > 0.0, R.sum(1), -np.inf)     # best FEASIBLE sum r seen, per cand
    bC, bR = C.copy(), R.copy()
    iters = int(iters)
    for it in range(iters):
        f = it / max(1.0, iters - 1.0)
        t = t0 * (t1 / t0) ** f
        d = C[:, :, None, :] - C[:, None, :, :]
        dist = np.sqrt((d * d).sum(-1))
        dist[:, idx, idx] = 1.0
        s = dist - (R[:, :, None] + R[:, None, :])
        s[:, idx, idx] = 1.0
        np.maximum(s, 1e-15, out=s)
        inv = 1.0 / s
        inv[:, idx, idx] = 0.0
        inv2 = inv * inv
        u = d / dist[..., None]
        gC = t * (inv[..., None] * u).sum(2)
        uu = u * u
        hC = t * ((inv2[..., None] * uu).sum(2)
                  + ((1.0 - uu) * (inv / dist)[..., None]).sum(2))
        wL = np.maximum(C[:, :, 0] - LO - R, 1e-15)
        wR = np.maximum(HI - C[:, :, 0] - R, 1e-15)
        wB = np.maximum(C[:, :, 1] - LO - R, 1e-15)
        wT = np.maximum(HI - C[:, :, 1] - R, 1e-15)
        iL, iR_, iB, iT = 1.0 / wL, 1.0 / wR, 1.0 / wB, 1.0 / wT
        gC[:, :, 0] += t * (iL - iR_)
        gC[:, :, 1] += t * (iB - iT)
        hC[:, :, 0] += t * (iL * iL + iR_ * iR_)
        hC[:, :, 1] += t * (iB * iB + iT * iT)
        wsq = iL * iL + iR_ * iR_ + iB * iB + iT * iT
        gR = 1.0 - t * (inv.sum(2) + iL + iR_ + iB + iT)
        hR = t * (inv2.sum(2) + wsq)
        g = np.concatenate([gC.reshape(B, 2 * n), gR], 1)
        h = np.concatenate([hC.reshape(B, 2 * n), hR], 1)
        step = g / np.maximum(h, 1e-300)
        sc = np.abs(step).max(1)
        lim = np.minimum(1.0, a / np.maximum(sc, 1e-300))[:, None]
        step = step * lim
        Ct = C + step[:, :2 * n].reshape(B, n, 2)
        Rt = np.maximum(R + step[:, 2 * n:], 1e-12)
        np.clip(Ct, LO + 1e-13, HI - 1e-13, out=Ct)
        ok = _minslack(Ct, Rt) > 1e-15
        if ok.any():
            C[ok] = Ct[ok]
            R[ok] = Rt[ok]
            sr = R.sum(1)
            up = ok & (sr > bs)
            if up.any():
                bs[up] = sr[up]
                bC[up] = C[up]
                bR[up] = R[up]
        a = np.where(ok, a * 1.3, a * 0.35)
        np.clip(a, 1e-16, amax, out=a)
    keep = np.isfinite(bs)
    C[keep], R[keep] = bC[keep], bR[keep]
    return C, R


def _feasible_seed(C, shrink=0.93):
    """Strictly-interior radii for a batch of centre sets: the screening cap, pulled in so the
    barrier has room to move on its first steps (a start ON the boundary has infinite gradient)."""
    return _grow_batch(C) * shrink


def _grow_batch(C):
    """Vectorised guaranteed-feasible radii for a batch of centre sets (the screening inflater)."""
    B, n, _ = C.shape
    d = np.sqrt(((C[:, :, None, :] - C[:, None, :, :]) ** 2).sum(-1))
    d[:, np.arange(n), np.arange(n)] = np.inf
    w = np.minimum.reduce([C[:, :, 0] - LO, HI - C[:, :, 0], C[:, :, 1] - LO, HI - C[:, :, 1]])
    return np.maximum(np.minimum(w, d.min(2) / 2.0), 1e-12)


def _repair_batch(C, R):
    out = np.empty((len(C), C.shape[1], 3))
    for b in range(len(C)):
        out[b] = _repair(C[b], R[b])
    return out


# ------------------------------------------------- D4 SYMMETRY PROJECTION (iteration 7 core)
# Every move the search has ever had is a perturbation INSIDE one basin (jitter/teleport/rotate), a
# recombination of two basins (crossover), or a change of dimension (transfer). None of them can
# impose STRUCTURE. Yet the square has a symmetry group, and a large share of the published optima
# are invariant under one of its reflections: an incumbent that is nearly symmetric but a few 1e-3
# off is a DIFFERENT basin from the exact symmetric one, and no local move finds the crossing,
# because getting there means moving every circle at once in a correlated way.
# `_symmetrize` is that correlated move: match the configuration to its own mirror image with an
# optimal assignment (Hungarian), then average each circle with the pre-image of its partner. The
# result is exactly invariant (each generator below is an involution, so the matching folds into
# orbits), lives in a HALF-DIMENSIONAL subspace, and is then handed to the ordinary relax->SLP
# finisher, which is free to break the symmetry again if that pays.
_D4 = (np.array([[-1.0, 0.0], [0.0, 1.0]]),      # reflect in the vertical axis
       np.array([[1.0, 0.0], [0.0, -1.0]]),      # reflect in the horizontal axis
       np.array([[0.0, 1.0], [1.0, 0.0]]),       # reflect in the main diagonal
       np.array([[0.0, -1.0], [-1.0, 0.0]]),     # reflect in the anti-diagonal
       np.array([[-1.0, 0.0], [0.0, -1.0]]))     # rotate by pi


def _symmetrize(xy, rng, a=None, g=None):
    """Project centres onto the nearest configuration invariant under one involutive element of D4.

    All five generators satisfy M = M^T = M^-1, so the optimal assignment pi between the image
    M*P and P is (generically) an involution and new_i = (P_i + M*P_pi(i))/2 is exactly M-invariant:
    circles pair up across the axis and any circle matched to itself lands ON the axis.
    `a` blends between the incumbent (a=0) and the exact projection (a=1) -- a partial fold is a
    weaker but safer move, so the caller samples it."""
    P = np.asarray(xy, float)
    n = len(P)
    if n < 2:
        return P.copy()
    M = _D4[int(rng.randint(len(_D4)))] if g is None else np.asarray(g, float)
    Q = P @ M                                    # M is symmetric, so rows transform as v @ M
    C = ((Q[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    try:
        _, pi = linear_sum_assignment(C)
    except Exception:
        pi = np.arange(n)
    S = 0.5 * (P + P[pi] @ M)
    if a is None:
        a = float(rng.uniform(0.4, 1.0))
    out = (1.0 - a) * P + a * S
    out = out + rng.normal(0.0, 1e-4 / np.sqrt(n), out.shape)
    return np.clip(out, LO + 1e-5, HI - 1e-5)


# --------------------------------------------------------------- spatial crossover (population move)
def _crossover(A, Bp, rng):
    """Take the circles of parent A on one side of a random line through the centre and those of
    parent Bp on the other, then fix the count back to exactly n (drop the most crowded / insert into
    the largest empty discs). Recombines two DIFFERENT converged basins -- a move no perturbation of a
    single incumbent can make."""
    n = len(A)
    th = rng.uniform(0.0, np.pi)
    u = np.array([np.cos(th), np.sin(th)])
    c = rng.uniform(-0.18, 0.18)
    pts = np.concatenate([A[A @ u > c], Bp[Bp @ u <= c]])
    if len(pts) > n:
        r = _grow(pts)
        pts = pts[np.argsort(-r)[:n]]
    while len(pts) < n:
        P = rng.uniform(LO, HI, (512, 2))
        if len(pts):
            free = np.minimum(_wall(P), np.sqrt(((P[:, None, :] - pts[None]) ** 2).sum(-1)).min(1))
        else:
            free = _wall(P)
        pts = np.concatenate([pts, P[int(np.argmax(free))][None]])
    return np.clip(pts + rng.normal(0.0, 1e-4, (n, 2)), LO + 1e-6, HI - 1e-6)

# --------------------------------------------------------------------------------------- warm starts
def _read_pack(n):
    """Centres of the committed best for this n, or None. GUARDED: any missing/odd file -> cold start."""
    p = "bench/packs/csqv%d.pck" % n
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            rows = [ln.split() for ln in f.read().splitlines()]
        trips = [(float(a), float(b), float(c)) for a, b, c in (r for r in rows if len(r) == 3)]
        if len(trips) != n:
            return None
        arr = np.asarray(trips, dtype=float)
        if not np.all(np.isfinite(arr)):
            return None
        return np.clip(arr[:, :2], LO, HI), np.maximum(arr[:, 2], 0.0)
    except Exception:
        return None


# ---------------------------------------------- LIVE CENSUS (iteration 6 core)
# The driver FREEZES bench/packs/ for the whole solve() call: every warm start and every cross-n
# transfer seed therefore reads the census as it stood BEFORE this call, and a record basin found at
# n=83 could not reach n=81/85 until the NEXT iteration (measured, iteration 5). _LIVE is the
# in-process census: every evaluate()-confirmed best is published the moment it is found, and every
# reader below goes through _best(), so improvements cascade WITHIN one call instead of once per
# iteration. It deliberately survives across solve() calls in one process -- geometry is geometry.
_LIVE = {}


def _live_put(n, xy, r, s):
    """Publish an evaluate()-CONFIRMED feasible packing to the in-process census."""
    e = _LIVE.get(int(n))
    if e is None or float(s) > e[2]:
        _LIVE[int(n)] = (np.asarray(xy, float).copy(), np.asarray(r, float).copy(), float(s))


def _best(n):
    """Census read: the better of this call's in-memory best and the committed pack, or None.
    Same guarded contract as _read_pack -- a missing/odd file simply yields the live entry or None."""
    d = _read_pack(n)
    e = _LIVE.get(int(n))
    if e is None:
        return d
    if d is not None and float(d[1].sum()) > e[2]:
        return d[0], d[1]
    return e[0], e[1]


def _records():
    try:
        with open("bench/records.json") as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("records"), dict):
            d = d["records"]
        out = {}
        for k, v in d.items():
            try:
                out[int(k)] = float(v)
            except (TypeError, ValueError):
                continue       # metadata keys ("source", "retrieved", ...) are not records
        return out
    except Exception:
        return {}


def _starts(n, rng, k):
    """A warm start (if any) followed by k-1 diverse cold starts: jittered near-square grids,
    hex rows and uniform clouds. Purely generated -- no per-n coordinate table anywhere."""
    out = []
    w = _best(n)
    if w is not None:
        out.append(w[0])
    while len(out) < max(1, k):
        t = len(out) % 3
        if t == 1:
            c = int(np.ceil(np.sqrt(n)))
            g = (np.arange(c) + 0.5) / c - 0.5
            gx, gy = np.meshgrid(g, g)
            pts = np.stack([gx.ravel(), gy.ravel()], 1)[:n]
            if len(pts) < n:
                pts = np.concatenate([pts, rng.uniform(LO + .05, HI - .05, (n - len(pts), 2))])
            out.append(np.clip(pts + rng.normal(0, 0.35 / c, pts.shape), LO + 1e-3, HI - 1e-3))
        elif t == 2:
            rows = max(int(round(np.sqrt(n / 0.866))), 2)
            per = int(np.ceil(n / rows))
            pts = [(LO + (j + 0.5) / per + 0.5 / per * (i % 2), LO + (i + 0.5) / rows)
                   for i in range(rows) for j in range(per)]
            pts = np.asarray(pts)[:n]
            out.append(np.clip(pts + rng.normal(0, 0.15 / per, pts.shape), LO + 1e-3, HI - 1e-3))
        else:
            out.append(rng.uniform(LO + 0.02, HI - 0.02, (n, 2)))
    return out


# ------------------------------------------------- CROSS-n BASIN TRANSFER (iteration 5 core)
# The census has been solved as 37 INDEPENDENT searches. It is not 37 problems: the optimal packing
# for n and for n+2 share almost all of their structure (the same rows/shells, two circles more or
# less), so a converged neighbour is a start no single-n perturbation can ever reach -- you cannot
# jitter your way from an n-circle basin to an (n+2)-circle one, because the dimension changes.
# 8 of 37 n found the record basin in iteration 4; this is the operator that lets the other 29
# borrow that structure instead of re-discovering it from a grid.
def _gap_insert(xy, r, k, rng, nsamp=1024):
    """Add k circles at the k largest empty discs -> a start for n = m + k from an m-circle pack."""
    pts, rr = xy.copy(), np.maximum(np.asarray(r, float), 0.0).copy()
    for _ in range(int(k)):
        P = rng.uniform(LO, HI, (nsamp, 2))
        free = np.minimum(_wall(P),
                          (np.sqrt(((P[:, None, :] - pts[None]) ** 2).sum(-1)) - rr[None]).min(1))
        b = int(np.argmax(free))
        pts = np.concatenate([pts, P[b][None] + rng.normal(0.0, 1e-3, (1, 2))])
        rr = np.concatenate([rr, [max(float(free[b]), 1e-5)]])
    return pts


def _drop_small(xy, r, k, rng):
    """Remove k of the (k+2) smallest circles -> a start for n = m - k from an m-circle pack.
    Sampling from k+2 rather than taking exactly the k smallest keeps the operator stochastic."""
    n, k = len(xy), int(k)
    order = np.argsort(np.asarray(r, float))
    pool = order[:max(k, min(n - 1, k + 2))]
    sel = pool[:k] if len(pool) <= k else pool[rng.choice(len(pool), size=k, replace=False)]
    keep = np.ones(n, bool)
    keep[sel] = False
    return xy[keep].copy()


def _donors(n, recs, span=8, k=4):
    """Committed neighbours of n, best-digits-first then nearest. GUARDED: missing packs are skipped,
    so an n with no neighbours at all simply gets no transfer seeds and the cold path is unchanged."""
    cands = []
    for m in range(max(2, n - span), n + span + 1):
        if m == n:
            continue
        w = _best(m)
        if w is None or len(w[0]) != m:
            continue
        rec = recs.get(m)
        d = 0.0
        if rec is not None and rec > 0:
            gap = max(0.0, (rec - float(w[1].sum())) / rec)
            d = 7.0 if gap <= 1e-7 else float(min(7.0, -np.log10(max(gap, 1e-300))))
        lv = int(m) in _LIVE and (_read_pack(m) is None
                                  or float(_LIVE[int(m)][2]) > float(_read_pack(m)[1].sum()))
        cands.append((-d, abs(m - n), m, w, bool(lv)))
    cands.sort(key=lambda e: (e[0], e[1]))
    return [(c[2], c[3], c[4]) for c in cands[:k]]


def _transfer_seeds(n, rng, recs, k=4):
    """(centre-set, tag) pairs for n built by add/remove from nearby n's best known pack.
    tag is "transfer" for a donor read off disk and "transfer*" for one improved during THIS call --
    the second kind only exists because of the live census, so provenance measures it separately.
    Never raises."""
    out = []
    try:
        for m, w, lv in _donors(n, recs, k=k):
            xy, r = w
            s = _drop_small(xy, r, m - n, rng) if m > n else _gap_insert(xy, r, n - m, rng)
            s = np.asarray(s, float)
            if s.shape == (n, 2) and np.all(np.isfinite(s)):
                out.append((np.clip(s, LO + 1e-4, HI - 1e-4), "transfer*" if lv else "transfer"))
    except Exception:
        return out
    return out


# Provenance: which move actually produced each accepted basin. Iterations 3 and 4 both ended with
# "instrument the moves" as the top open question and both spent the budget elsewhere; this is a
# counter, it costs nothing, and it is what tells the next iteration where the CPU should go.
_PROV = {}


def _prov(tag, n, gain):
    e = _PROV.setdefault(tag, [0, 0.0, set()])
    e[0] += 1
    e[1] += float(gain)
    e[2].add(int(n))


def _prov_dump(path="artifacts/prov_latest.txt"):
    try:
        if not os.path.isdir(os.path.dirname(path) or "."):
            return
        rows = sorted(_PROV.items(), key=lambda kv: -kv[1][0])
        with open(path, "w") as f:
            f.write("move            accepts   sum_gain      n_touched\n")
            for t, (c, g, ns) in rows:
                f.write("%-14s %8d %12.6f %6d  %s\n" % (t, c, g, len(ns), sorted(ns)))
    except Exception:
        pass


# ------------------------------------------------------------------------------------------- driver
def _digits_now(n, recs):
    """Digits the committed census currently holds for this n (0.0 if unknown -> highest priority)."""
    rec = recs.get(n)
    w = _best(n)
    if rec is None or w is None or rec <= 0:
        return 0.0
    gap = max(0.0, (rec - float(w[1].sum())) / rec)
    return 7.0 if gap <= 1e-7 else float(min(7.0, -np.log10(max(gap, 1e-300))))


# Batch slot schedule. Iteration 5 MEASURED gain-per-slot: teleport .097 > crossover .089 >
# transfer .080 >> jitter .052 ~ rotate .051. The two weakest moves are kept alive (they own n=55,
# 91 and n=31 respectively -- a move that is half as productive on average is not a move that never
# wins) but drop from 4 slots in 12 to 2, and transfer takes them: with the live census its donors
# are strictly better than the frozen-disk donors that produced that 0.080.
# Iteration 7 ADDS the symmetry projection without CUTTING anything: the cycle grows 12 -> 14 slots
# so every existing move keeps its absolute share (iteration 6 measured that per-slot and per-accept
# productivity rank the moves in OPPOSITE orders, so a reallocation off either table alone is not
# supported). The first 12 slots still exercise all six moves, which is what a narrow batch sees.
# ------------------------------------------- RATIO-SPACE RESEEDING (iteration 8 core)
# MEASURED (artifacts/iter8_probe*.py, this iteration): every one of the 12 stuck n is a FULLY
# converged local optimum -- a 6 s SLP from the committed pack moves sum r by ~1e-14 -- and a
# gaussian jitter of ANY amplitude up to 0.15/sqrt(n) reconverges to the SAME value to 1e-9. The
# space of CENTRES is exhausted: SLP's basin of attraction swallows every centre perturbation the
# solver can make. So open a different space.
#
# THE SPACE OF SIZE RATIOS. A packing's basin is decided by its CONTACT GRAPH, and which contacts
# form is decided by the relative SIZES the circles are trying to reach, not by where their centres
# start. `_relax` already ascends sum r; ascending sum_i W_i r_i instead makes circle i push harder,
# its neighbours give way, and the graph rewires into an arrangement no centre move can reach.
# Releasing the weights (plain relax + the exact SLP finisher on the TRUE objective) then reports
# what that rearrangement is actually worth. W is the new search space; the old five moves all live
# in centre space, so this one is orthogonal to every one of them.
def _ban_contacts(xy, r, rng, kmax=3, tol=1e-4):
    """Pick a few ACTIVE contacts of a converged packing and say how far to prise each one open.

    A converged packing is pinned by its contact graph: every operator this solver had before moved
    the CENTRES (and iteration 8 measured that the relaxer's basin swallows every such perturbation)
    or the OBJECTIVE (iteration 8's ratio weights). This picks the third object -- the CONSTRAINTS.
    Returns [(i, j, d), ...]: pair (i,j) must end stage 1 with slack >= d, which is a contact the
    incumbent CANNOT keep, so the packing has to find a different graph rather than relax back.
    """
    n = len(r)
    if n < 2:
        return []
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    sl = d - (r[:, None] + r[None, :])
    ii, jj = np.triu_indices(n, 1)
    act = np.nonzero(sl[ii, jj] <= tol * (r[ii] + r[jj]) + 1e-9)[0]
    if act.size == 0:                                    # no contact resolved as active: prise the
        act = np.argsort(sl[ii, jj])[:max(1, n // 4)]    # tightest pairs open instead
    k = 1 + int(rng.randint(min(kmax, act.size)))
    pick = rng.choice(act, size=k, replace=False)
    out = []
    for q in pick:
        a, b = int(ii[q]), int(jj[q])
        f = float(10.0 ** rng.uniform(-1.2, -0.3))       # 6% .. 50% of the pair's own scale
        out.append((a, b, f * float(r[a] + r[b])))
    return out


# =============================== THE ACTIVE-SET PIVOT (iteration 14 core) ===========================
# MEASURED THIS ITERATION (artifacts/iter14_diag.py, offline, no packs written): eight CPU-seconds of
# pure `_slp` started FROM each of the six stuck incumbents gains 1.8e-14 / 4.0e-14 / -6.3e-14 /
# 5.9e-14 / 2.8e-14 / -2.0e-13 on n = 31 / 37 / 41 / 45 / 59 / 81, against absolute record shortfalls
# of 1.07e-5 .. 2.34e-3. The incumbents are EXACT KKT POINTS. That kills a whole family of candidate
# fixes at once: no finisher that keeps the ACTIVE SET fixed -- a longer SLP, a tighter trust region,
# an exact equality-constrained KKT solve, a better interior-point -- can buy a single digit, because
# the point already satisfies those conditions to machine precision. The ONLY thing left to optimise
# is WHICH constraints are active. This is the object iterations 12 and 13 named as untried.
#
# `ban` (iteration 10) gropes at the active set from outside: it picks contacts UNIFORMLY AT RANDOM,
# releases them, and takes one shot. The pivot does the two things that makes it a search rather than
# a perturbation:
#   1. IT ASKS THE PROBLEM WHICH CONTACT TO RELEASE. At a KKT point the LP dual multiplier of an
#      active constraint IS d(sum r)/d(relaxing it) -- the exact price the packing pays for that
#      contact. Releasing the largest-multiplier contact is the steepest available combinatorial
#      move; releasing a zero-multiplier one is provably worthless. `_duals` reads them straight out
#      of HiGHS. Walls carry multipliers exactly like pairs do, which is why `_relax` grew WBAN.
#   2. IT CHAINS, AND IT ACCEPTS DOWNHILL. Every accept rule in this solver so far is elitist (the
#      pool keeps a candidate only if it beats the 5th best), so the walk can never cross a barrier:
#      a two-pivot rearrangement whose first pivot costs 1e-4 is unreachable. `_pivot_chain` is a
#      Metropolis walk over the contact graph -- one released constraint per step, re-optimised
#      exactly, accepted with prob exp(-loss/T) -- and it reports its best-ever, not its last.
def _duals(xy, r, t=None):
    """LP dual multipliers of the active constraints at a converged packing.

    Builds the SAME linearisation `_slp_step` uses, at a tiny trust radius, and reads HiGHS's row
    marginals. Returns (pi, pj, plam, wi, wk, wlam): `plam[q] >= 0` is the rate at which sum r would
    grow if pair (pi[q], pj[q]) were allowed to overlap, and `wlam[q] >= 0` the same for pushing
    circle `wi[q]` through wall `wk[q]` (0:-x 1:+x 2:-y 3:+y). Empty arrays if the LP fails.
    """
    n = len(xy)
    empt = (np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0),
            np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0))
    if n < 2:
        return empt
    if t is None:
        t = 1e-5 / np.sqrt(n)
    x0, y0 = xy[:, 0], xy[:, 1]
    ii, jj = np.triu_indices(n, 1)
    dx, dy = x0[ii] - x0[jj], y0[ii] - y0[jj]
    d = np.sqrt(dx * dx + dy * dy)
    keep = ((d - (r[ii] + r[jj])) < 4.0 * t) & (d > 1e-12)
    ii, jj, dx, dy, d = ii[keep], jj[keep], dx[keep], dy[keep], d[keep]
    m = len(d)
    if m == 0:
        return empt
    ux, uy = dx / d, dy / d
    rows = np.repeat(np.arange(m), 6)
    cols = np.empty(6 * m, dtype=np.int64)
    vals = np.empty(6 * m)
    cols[0::6], vals[0::6] = ii, -ux
    cols[1::6], vals[1::6] = jj, ux
    cols[2::6], vals[2::6] = n + ii, -uy
    cols[3::6], vals[3::6] = n + jj, uy
    cols[4::6], vals[4::6] = 2 * n + ii, 1.0
    cols[5::6], vals[5::6] = 2 * n + jj, 1.0
    idx = np.arange(n)
    wrow, wcol, wval = [], [], []
    for k, (var, sgn) in enumerate(((0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0))):
        base = m + k * n
        wrow.append(base + idx); wcol.append(var * n + idx); wval.append(np.full(n, sgn))
        wrow.append(base + idx); wcol.append(2 * n + idx); wval.append(np.ones(n))
    A = coo_matrix((np.concatenate([vals] + wval),
                    (np.concatenate([rows] + wrow), np.concatenate([cols] + wcol))),
                   shape=(m + 4 * n, 3 * n))
    b = np.concatenate([np.zeros(m), np.full(4 * n, HI)])
    lb = np.concatenate([np.maximum(x0 - t, LO), np.maximum(y0 - t, LO), np.zeros(n)])
    ub = np.concatenate([np.minimum(x0 + t, HI), np.minimum(y0 + t, HI),
                         np.minimum(r + t, HI - LO)])
    c = np.zeros(3 * n)
    c[2 * n:] = -1.0
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=np.stack([lb, ub], 1), method="highs")
        mg = np.asarray(res.ineqlin.marginals, float)
    except Exception:
        return empt
    if mg.size != m + 4 * n or not np.all(np.isfinite(mg)):
        return empt
    lam = np.maximum(-mg, 0.0)                       # d(sum r)/d(relaxing that row), >= 0
    wl = lam[m:].reshape(4, n)
    wk, wi = np.nonzero(wl > 0.0)
    return (ii, jj, lam[:m], wi.astype(np.intp), wk.astype(np.intp), wl[wk, wi])


def _pivot_picks(du, rng, kmax=2):
    """Sample 1..kmax constraints to RELEASE, with probability proportional to lambda^2 (steepest
    combinatorial move, but randomized so the chain is not a deterministic ridge-walk). Returns
    (pairs, walls) as [(i, j, dd)] and [(i, k, dd)] separation demands."""
    pi, pj, plam, wi, wk, wlam = du
    lam = np.concatenate([plam, wlam])
    if lam.size == 0 or not np.any(lam > 0):
        return [], []
    w = lam ** 2
    w = w / w.sum()
    k = 1 + int(rng.randint(min(kmax, int((w > 0).sum()))))
    pick = rng.choice(len(w), size=k, replace=False, p=w)
    np_, nw = len(plam), []
    pairs = []
    for q in pick:
        f = float(10.0 ** rng.uniform(-1.4, -0.4))
        if q < np_:
            a, b = int(pi[q]), int(pj[q])
            pairs.append((a, b, f))
        else:
            q -= np_
            nw.append((int(wi[q]), int(wk[q]), f))
    return pairs, nw


# PERSISTENT CHAIN STATE, n -> (cx, cr, cs, T)  (iteration 15)
# MEASURED (iteration 14): an 8-second uninterrupted chain from the committed n=37 incumbent reached
# 2e-12 of the record from a 1.6e-3 gap; the SAME code inside the metered run only bought 7.4e-4,
# because `_run_n` re-entered `_pivot_chain` from the pool head once per round with ~0.5-1.5 s each
# and every walk was thrown away at the end of the round. A Metropolis walk's whole value is its
# LENGTH -- a chain restarted every second is a chain that never leaves the incumbent's basin, and
# restarting it at the pool head also erases the deliberate downhill steps that are the only way
# across a contact-graph barrier. So the walk now lives in `_PIV` and SURVIVES rounds, draws and
# waves: each re-entry continues the same walk at the same temperature instead of resetting it.
_PIV = {}


def _pivot_chain(evaluate, meter, rng, n, xy, r, s, deadline, t0, K=6):
    """Metropolis walk over the CONTACT GRAPH. Each step releases 1-2 dual-selected constraints,
    re-optimises exactly (soft batched relax -> release -> SLP), and accepts the step even when it
    loses ground, with probability exp(-loss/T). Returns [(sum_r, xy, r)] of every distinct KKT point
    it visited that beat the chain's start -- plus the best-ever, so the caller never loses it.

    The walk is PERSISTENT: on re-entry it resumes `_PIV[n]` (point + temperature) whenever that
    stored point is still within a relative 1e-4 of the caller's start, so a chain accumulates
    length across rounds and draws. It cools geometrically while it runs and REHEATS on one resume
    in five, keeping both the settling line and the barrier-crossing line alive."""
    out = []
    if n < 2:
        return out
    cx, cr, cs = xy.copy(), r.copy(), float(s)
    coin = rng.rand()          # drawn UNCONDITIONALLY: the rng stream must not fork on whether a
                               # persistent state happens to exist, or a resume would silently
                               # re-index every later draw in the same walk.
    st = _PIV.get(n)
    resumed = (st is not None and st[0].shape[0] == n
               and st[2] >= s - 1e-4 * max(abs(s), 1e-9))
    if resumed:
        cx, cr, cs = st[0].copy(), st[1].copy(), float(st[2])
    T = float(10.0 ** rng.uniform(-5.6, -4.2)) * max(cs, 1e-9)
    if resumed and coin >= 0.20:                       # 4 resumes in 5 continue at the stored T
        T = float(st[3])
    best = (cs, cx, cr)
    while time.process_time() < deadline and meter.left() > K + 4:
        du = _duals(cx, cr)
        if du[2].size == 0 and du[5].size == 0:
            break
        C = np.empty((K, n, 2))
        bans, wbans = [], []
        for b in range(K):
            pr, wr = _pivot_picks(du, rng)
            if not pr and not wr:
                pr = [(int(du[0][0]), int(du[1][0]), 0.1)] if du[0].size else []
            for (a, bb, f) in pr:
                bans.append((b, a, bb, f * float(cr[a] + cr[bb])))
            for (i, k, f) in wr:
                wbans.append((b, i, k, f * float(cr[i])))
            sc = float(10.0 ** rng.uniform(-3.4, -2.2)) / np.sqrt(n)
            C[b] = np.clip(cx + rng.normal(0.0, sc, cx.shape), LO + 1e-4, HI - 1e-4)
        BAN = None
        if bans:
            BAN = (np.array([q[0] for q in bans], np.intp), np.array([q[1] for q in bans], np.intp),
                   np.array([q[2] for q in bans], np.intp), np.array([q[3] for q in bans], float))
        WBAN = None
        if wbans:
            WBAN = (np.array([q[0] for q in wbans], np.intp),
                    np.array([q[1] for q in wbans], np.intp),
                    np.array([q[2] for q in wbans], np.intp),
                    np.array([q[3] for q in wbans], float))
        it = int(max(40, min(260, 6000 // n)))
        # stage 1: the released constraints are HELD open and the graph is free to rewire (soft mu)
        C, _ = _relax(C, _grow_batch(C), it, mu0=0.3, mu1=40.0, BAN=BAN, WBAN=WBAN)
        # stage 2: released -- the TRUE feasible set decides what the new graph is worth
        C, _ = _relax(C, _grow_batch(C), it)
        cand = _repair_batch(C, _grow_batch(C))
        feas, sr = evaluate(n, cand)
        sr = np.where(feas, sr, -np.inf)
        order = np.argsort(-sr)
        fin = []
        for j in order[:2]:
            if time.process_time() >= deadline or meter.left() <= 2 or not np.isfinite(sr[j]):
                break
            xy2, r2, _ = _slp(cand[j][:, :2], _lp_radii(cand[j][:, :2]),
                              min(deadline, time.process_time() +
                                  max(0.10, (deadline - time.process_time()) * 0.25)), t0)
            fin.append(_repair(xy2, _lp_radii(xy2)))
        if not fin:
            break
        feas, sr = evaluate(n, np.stack(fin))
        sr = np.where(feas, sr, -np.inf)
        j = int(np.argmax(sr))
        if not np.isfinite(sr[j]):
            break
        ns = float(sr[j])
        nxy, nr = fin[j][:, :2].copy(), fin[j][:, 2].copy()
        if ns > best[0]:
            best = (ns, nxy, nr)
        if ns > s + 1e-12:
            out.append((ns, nxy, nr))
        # MOVE EVEN WHEN IT COSTS: the barrier between two contact graphs is what the elitist pool
        # can never cross. exp(-loss/T) with T ~ 1e-5 of sum r admits a step that loses a few 1e-6.
        if ns >= cs or rng.rand() < np.exp(-(cs - ns) / max(T, 1e-300)):
            cx, cr, cs = nxy, nr, ns
        T = max(T * 0.99, 1e-9 * max(cs, 1e-9))        # anneal: a long walk must eventually settle
        _PIV[n] = (cx.copy(), cr.copy(), cs, T)        # the walk survives this call
    if best[0] > s + 1e-12 and not any(abs(e[0] - best[0]) < 1e-12 for e in out):
        out.append(best)
    return out


# ------------------------------------------------- RUIN & RECREATE / LNS (iteration 12 core)
# MEASURED (artifacts/prov_latest.txt, iteration 11): `BEST:warm` owns the final pool for EXACTLY the
# six n still short of the cap (31, 37, 41, 45, 59, 81) -- across ~40 draws not one operator produced
# a basin better than the incumbent on any of them, and the near-miss surrogate never exceeded 4.25
# digits, so the search is not even finding NEAR-TIES around those incumbents. Every move the solver
# has is a PERTURBATION of the incumbent's arrangement: teleport lifts the k smallest circles, jitter
# and rotate displace them, symm folds them, crossover splices two whole arrangements, ban forbids one
# contact, ratio re-weights the objective. All of them keep the arrangement's LOCAL COMBINATORICS --
# which circle sits between which -- and hand it to a relaxer whose basin (iteration 8) swallows the
# displacement.
#
# Ruin & recreate is the standard large-neighbourhood answer and it is a different class of move:
# DELETE a contiguous REGION of the packing outright -- every circle in it, big and small alike -- and
# REBUILD that region from nothing by randomized greedy largest-empty-disc insertion. The rebuilt
# region is free to use a different number of circles per row, a different offset, a different
# neighbour ordering; nothing carries over except the untouched remainder, which acts as the boundary
# condition. That is precisely the "a whole row has to re-index" rearrangement no perturbation can
# reach, and it is what the six stuck n need.
def _ruin_recreate(xy, r, rng, nsamp=512):
    """Delete a region of the packing and rebuild it by randomized greedy insertion.

    Four ruins (a disc, a band, a scatter, and a band whose far side is COMPACTED inward -- the last
    one deletes a row and closes the gap, which changes the row count of the whole strip) and one
    recreate: insert the missing circles one at a time at a point sampled from the top few largest
    empty discs, so two calls on the same incumbent rebuild it differently.
    """
    P0 = np.asarray(xy, float)
    n = len(P0)
    if n < 4:
        return np.clip(P0 + rng.normal(0.0, 1e-3, P0.shape), LO + 1e-5, HI - 1e-5)
    rr0 = np.maximum(np.asarray(r, float), 0.0)
    frac = float(rng.uniform(0.07, 0.32))
    k = int(max(2, min(n - 3, round(n * frac))))
    mode = int(rng.randint(4))
    pts = P0.copy()
    if mode == 0:                                        # DISC ruin: the k circles nearest a point
        c = rng.uniform(LO + 0.05, HI - 0.05, 2)
        sel = np.argsort(((P0 - c) ** 2).sum(1))[:k]
    elif mode == 2:                                      # SCATTER ruin: k circles at random
        sel = rng.choice(n, size=k, replace=False)
    else:                                                # BAND ruin (mode 1), COMPACTED band (mode 3)
        th = float(rng.uniform(0.0, np.pi))
        u = np.array([np.cos(th), np.sin(th)])
        p = P0 @ u
        c = float(rng.uniform(float(p.min()), float(p.max())))
        sel = np.argsort(np.abs(p - c))[:k]
        if mode == 3:
            # close the deleted row: slide everything strictly beyond the band back toward it, so the
            # remainder is a packing of the square MINUS a slab and the rebuild has to re-tile it.
            w = float(np.abs(p[sel] - c).max())
            sh = w * float(rng.uniform(0.25, 0.9))
            far = p > c
            far[sel] = False
            pts = pts - np.outer(np.where(far, sh, 0.0), u)
    keep = np.ones(n, bool)
    keep[sel] = False
    pts, rr = pts[keep], rr0[keep]
    pts = np.clip(pts, LO + 1e-5, HI - 1e-5)
    for _ in range(k):                                   # RECREATE: randomized greedy insertion
        S = rng.uniform(LO, HI, (nsamp, 2))
        free = np.minimum(_wall(S),
                          (np.sqrt(((S[:, None, :] - pts[None]) ** 2).sum(-1)) - rr[None]).min(1))
        m = min(5, len(free))
        top = np.argpartition(-free, m - 1)[:m] if m > 1 else np.array([0])
        b = int(top[int(rng.randint(len(top)))])
        pts = np.concatenate([pts, S[b][None] + rng.normal(0.0, 1e-3, (1, 2))])
        rr = np.concatenate([rr, [max(float(free[b]), 1e-5)]])
    return np.clip(pts, LO + 1e-5, HI - 1e-5)


# --------------------------------- CONTAINER-DEFORMATION HOMOTOPY (iteration 13 core)
def _box_homotopy(xy, r, rng, iters=None):
    """Re-solve this packing in an EQUAL-AREA RECTANGLE and continue the container back to the square.

    Every move this solver has had -- teleport, jitter, rotate, symm, crossover, ban, ratio, ruin --
    perturbs the CONFIGURATION inside a container that is a fixed constant. Iteration 12 measured
    all ten of them scoring 0.000000 total gain on the six n still short of the cap: the search keeps
    re-finding one family of arrangements a part in ~3000 below the record. This move perturbs the
    OTHER object in the problem. It makes the container a parameter:

        box(e) = [-0.5*sqrt(e), 0.5*sqrt(e)] x [-0.5/sqrt(e), 0.5/sqrt(e)]        (area 1 for all e)

    and walks e from e0 != 1 back to e_K = 1 in K stages, re-relaxing at each. This is NOT an affine
    re-labelling of the square problem: squashing the box by sqrt(e) and the circles with it would
    turn them into ellipses, so the optimal arrangement in box(e) is a genuinely different
    combinatorial object. A rectangle re-tiles a row at a different count and a different offset, and
    the continuation carries that new contact graph back into the square, which is a basin no
    perturbation of the square incumbent can reach.

    Returns centres in the unit square (the caller's stage-1/stage-2 relax and the SLP finisher then
    judge them on the true square objective, exactly as for every other move).
    """
    n = len(xy)
    K = 2 + int(rng.randint(4))                                  # 2..5 continuation stages
    e0 = float(np.exp(rng.uniform(0.05, 0.34) * (1.0 if rng.rand() < 0.5 else -1.0)))
    it = int(iters) if iters else int(max(40, min(160, 5200 // max(n, 1))))
    C = np.asarray(xy, float)[None].copy()
    R = np.maximum(np.asarray(r, float)[None].copy(), 1e-9)
    pbx = pby = 0.5
    for k in range(K + 1):
        e = e0 ** (1.0 - k / float(K))
        bx, by = 0.5 * np.sqrt(e), 0.5 / np.sqrt(e)
        fx, fy = bx / pbx, by / pby
        C[:, :, 0] *= fx
        C[:, :, 1] *= fy
        R *= min(fx, fy)                                         # affine map keeps it non-overlapping
        # soft first, stiff last: the intermediate boxes are where the graph is allowed to rewire,
        # the final square stage has to settle onto a real arrangement.
        soft = k < K
        C, R = _relax(C, R, it, mu0=(0.4 if soft else 3.0), mu1=(60.0 if soft else 400.0),
                      bx=bx, by=by)
        pbx, pby = bx, by
    out = np.clip(C[0], LO + 1e-4, HI - 1e-4)
    return out if np.all(np.isfinite(out)) else np.clip(np.asarray(xy, float), LO + 1e-4, HI - 1e-4)


def _ratio_weights(n, rng):
    """A per-circle growth-weight vector: mostly 1, with a random subset asked to grow or shrink."""
    w = np.ones(n)
    k = max(1, int(rng.randint(1, max(2, n // 2))))
    idx = rng.choice(n, size=k, replace=False)
    sig = float(rng.uniform(0.06, 0.75))
    w[idx] = np.exp(rng.normal(0.0, sig, k))
    return np.clip(w, 0.15, 6.0)


# Iteration 10 measured that a NINTH move in a 21-slot schedule mostly dilutes the eight already
# there, so `ruin` does not simply get appended: it takes four slots, two off `transfer` (4 accepts,
# 0.000000 total gain last iteration) and one off `symm` (6 accepts, 0.000000 gain), plus one added.
# Every previous line of attack keeps at least one slot -- none is retired on one iteration's numbers.
# iteration 13: `box` takes 4 of the 22 slots -- one each off teleport / crossover / ratio / ruin,
# so every one of the nine older lines of attack keeps at least one slot and none is retired.
_SCHED = ("teleport", "crossover", "ratio", "ban", "ruin", "symm", "box", "ratio",
          "crossover", "ban", "ruin", "jitter", "teleport", "box", "crossover", "transfer",
          "ban", "rotate", "box", "ratio", "transfer", "box")


def _pool_round(n, rng, pool, nb, relax_it, xfer=()):
    """Build a batch of nb candidate centre-sets from the elite pool: perturbations of pool members
    (teleport / jitter / shell rotation), crossovers between two distinct members, and CROSS-n
    TRANSFERS (a neighbour's best known pack with circles added/removed -- from the LIVE census as
    of iteration 6, so a basin found earlier in THIS call is already a donor).
    Returns (batch, tags, W, BAN) where tags[b] names the move that produced row b, W[b] is the
    per-circle growth weight the relaxer should ascend for that row (all-ones = the true objective;
    only the `ratio` move deviates) and BAN is the flat (row, i, j, d) contact-ban list that stage 1
    must hold open (empty on every row but `ban`)."""
    outs, tags, wts, bans = [], [], [], []
    P = len(pool)
    for b in range(nb):
        mv = _SCHED[b % len(_SCHED)]
        if mv == "transfer" and not xfer:
            mv = "teleport"
        if mv == "ruin" and n < 4:
            mv = "jitter"
        if mv == "box" and n < 2:
            mv = "jitter"
        if mv == "crossover" and P <= 1:
            mv = "rotate"
        if mv == "transfer":
            q = xfer[int(rng.randint(len(xfer)))]
            tag = "transfer"
            if isinstance(q, tuple):
                q, tag = q[0], q[1]
            q = np.asarray(q, float)
            sc = float(10.0 ** rng.uniform(-3.4, -1.4)) / np.sqrt(n)
            outs.append(np.clip(q + rng.normal(0.0, sc, q.shape), LO + 1e-4, HI - 1e-4))
            tags.append(tag)
            wts.append(np.ones(n))
            continue
        i = int(rng.randint(P))
        xy, r = pool[i][1], pool[i][2]
        tags.append(mv)
        wts.append(np.ones(n))
        if mv == "box":
            # the container itself is the perturbation -- see _box_homotopy. The returned centres are
            # already square-feasible-ish, so no extra nudge and no weight deformation.
            outs.append(_box_homotopy(xy, r, rng))
        elif mv == "ruin":
            # a rebuilt region starts OVERLAPPING; stage 1's soft relax (mu0=0.3) is what lets the
            # rebuilt circles pass through each other into a new graph before the penalty closes.
            outs.append(_ruin_recreate(xy, r, rng))
        elif mv == "ban":
            # prise a few of THIS member's active contacts open, then nudge so stage 1 does not
            # start exactly on the (now infeasible) KKT point of the undeformed problem
            for (bi, bj, bd) in _ban_contacts(xy, r, rng):
                bans.append((b, bi, bj, bd))
            sc = float(10.0 ** rng.uniform(-3.2, -2.0)) / np.sqrt(n)
            outs.append(np.clip(xy + rng.normal(0.0, sc, xy.shape), LO + 1e-4, HI - 1e-4))
        elif mv == "ratio":
            # a small centre nudge so the weighted relax does not start exactly on a KKT point
            wts[-1] = _ratio_weights(n, rng)
            sc = float(10.0 ** rng.uniform(-3.2, -2.0)) / np.sqrt(n)
            outs.append(np.clip(xy + rng.normal(0.0, sc, xy.shape), LO + 1e-4, HI - 1e-4))
        elif mv == "teleport":
            outs.append(_teleport(xy, r, rng, k=1 + int(rng.randint(3))))
        elif mv == "jitter":
            sc = float(10.0 ** rng.uniform(-2.4, -0.9)) / np.sqrt(n)
            outs.append(np.clip(xy + rng.normal(0.0, sc, xy.shape), LO + 1e-4, HI - 1e-4))
        elif mv == "symm":
            outs.append(_symmetrize(xy, rng))
        elif mv == "crossover":
            j = int(rng.randint(P - 1))
            j += (j >= i)
            outs.append(_crossover(xy, pool[j][1], rng))
        else:
            # rotate the circles inside a random disc -- moves a whole shell to a new seat at once
            ctr = rng.uniform(LO + 0.15, HI - 0.15, 2)
            rad = float(rng.uniform(0.12, 0.4))
            m = np.sqrt(((xy - ctr) ** 2).sum(1)) < rad
            a = float(rng.uniform(0.15, np.pi))
            ca, sa = np.cos(a), np.sin(a)
            q = xy.copy()
            v = q[m] - ctr
            q[m] = ctr + np.stack([ca * v[:, 0] - sa * v[:, 1], sa * v[:, 0] + ca * v[:, 1]], 1)
            outs.append(np.clip(q, LO + 1e-4, HI - 1e-4))
    BAN = None
    if bans:
        BAN = (np.array([q[0] for q in bans], np.intp), np.array([q[1] for q in bans], np.intp),
               np.array([q[2] for q in bans], np.intp), np.array([q[3] for q in bans], float))
    return np.stack(outs), tags, np.stack(wts), BAN


# MEASURED ROUND COST (iteration 7). Iteration 6 measured the exact failure of the wave schedule:
# 3 waves x 37 n over 104 CPU-s hands each slice ~0.9 s, but one memetic round is not preemptible --
# it costs 1-2 s whatever deadline it was given -- so the early n in a wave OVERRAN their slices and
# five n (27, 39, 45, 47, 49) got zero CPU. `_RCOST[n]` is an EMA of what a round on this n actually
# costs and `_ROUNDS[n]` counts them, so a slice too short to pay for a round is DECLINED (its time
# flows to the n behind it) instead of being overspent.
_RCOST = {}
_ROUNDS = {}


def _round_cost(n):
    return _RCOST.get(int(n))


def _run_n(evaluate, meter, rng, n, recs, n_end, pool):
    """ONE SLICE of work on a single n, bounded by the CPU stamp n_end. `pool` is the elite pool this
    n carried out of an earlier wave ([] on its first slice), so a later wave RESUMES the search
    instead of re-seeding it. Every accepted basin is published to the live census immediately, which
    is what lets the next n in this same wave transfer from it. Returns the (possibly grown) pool."""
    t0 = 0.25 / np.sqrt(n)
    cut = CUT / np.sqrt(n)
    # _relax is O(B n^2) vectorised numpy, so per-configuration cost is nearly flat in B while the
    # exact finisher still runs on only the top 3: widening buys strictly more basins per SLP spend.
    nb = int(max(6, min(64, 2600 // n)))               # batch width for the relaxer
    relax_it = int(max(60, min(400, 9000 // n)))
    pool_cap = 5

    # CROSS-n TRANSFER SEEDS, rebuilt at the START OF EVERY SLICE off the LIVE census -- that rebuild
    # is the whole point of the wave schedule: in wave 2 the donors include basins this call found.
    xfer = _transfer_seeds(n, rng, recs)

    if not pool:
        w0 = _best(n)
        warm0 = w0 is not None
        starts = _starts(n, rng, 3)
        stags = [("warm" if (warm0 and i == 0) else "cold") for i in range(len(starts))]
        if xfer:
            starts = list(starts[:2]) + [xfer[0][0]]
            stags = list(stags[:2]) + [xfer[0][1]]
        packs = np.stack([_repair(s, _grow(s)) for s in starts])
        feas, sr = evaluate(n, packs)
        sr = np.where(feas, sr, -np.inf)
        for si, s in enumerate(starts):
            if time.process_time() >= n_end or meter.left() <= 0:
                break
            r0 = w0[1] if (si == 0 and warm0) else _lp_radii(s)
            if len(r0) != n or not np.any(r0 > 0):
                r0 = _grow(s)
            sub = min(n_end, time.process_time() + max(0.2, (n_end - time.process_time()) * 0.30))
            xy2, r2, _ = _slp(s, r0, sub, t0)
            pk = _repair(xy2, _lp_radii(xy2))
            f2, s2 = evaluate(n, pk)
            if f2:
                pool.append((float(s2), pk[:, :2].copy(), pk[:, 2].copy(), stags[si]))
                _live_put(n, pk[:, :2], pk[:, 2], s2)
        if not pool:
            j = int(np.argmax(sr))
            if not np.isfinite(sr[j]):
                return pool
            pool = [(float(sr[j]), packs[j][:, :2].copy(), packs[j][:, 2].copy(), "seedfallback")]
            _live_put(n, packs[j][:, :2], packs[j][:, 2], sr[j])
        pool.sort(key=lambda e: -e[0])
        pool = pool[:pool_cap]

    # ---- memetic rounds: cheap batched relax -> screen -> exact SLP finish on the elite ----------
    while time.process_time() < n_end and meter.left() > nb + 4:
        # DECLINE a round this slice cannot pay for: an un-preemptible 1-2 s round started with 0.3 s
        # left is exactly how iteration 6 starved the tail of every wave.
        t_r0 = time.process_time()
        est = _RCOST.get(n)
        if est is not None and t_r0 + 0.75 * est > n_end:
            break
        C, ctags, W, BAN = _pool_round(n, rng, pool, nb, relax_it, xfer)
        if BAN is not None or not np.all(W == 1.0):
            # STAGE 1 -- rearrange under the DEFORMED objective sum_i W_i r_i (rows with W==1 are
            # untouched by the deformation, so one batched pass still serves the whole schedule).
            # MEASURED (artifacts/iter8_probe4.py): the overlap stiffness of this stage is what
            # decides whether the graph rewires at all. At the relaxer's default mu0=3 -> mu1=400 the
            # deformation only strains the incumbent (n=27 gain 3e-14); SOFTENING it to 0.3 -> 40,
            # so overlapping circles may briefly pass THROUGH each other before the penalty closes,
            # took n=27 from 5.76e-4 short of the record to 2.3e-4 ABOVE it in the same 6 s. The
            # stiff setting is kept on one round in three: it is a different line of attack, not a
            # strictly worse one (it was the better of the two on n=65 in the same probe).
            stiff = rng.rand() < 0.34
            C, _ = _relax(C, _grow_batch(C), max(20, relax_it // 2),
                          mu0=(3.0 if stiff else 0.3), mu1=(400.0 if stiff else 40.0), W=W,
                          BAN=BAN)
        # STAGE 2 -- RELEASE the weights: the true objective decides what the rearrangement is worth.
        C, R = _relax(C, _grow_batch(C), relax_it)
        R = _grow_batch(C)                              # feasible radii for the relaxed centres
        cand = _repair_batch(C, R)
        feas, sr = evaluate(n, cand)
        sr = np.where(feas, sr, -np.inf)
        order = np.argsort(-sr)
        worst = pool[-1][0] if len(pool) >= pool_cap else -np.inf
        best_before = pool[0][0]
        fin, ftags = [], []
        for j in order[:3]:                             # exact finisher on the best relaxed centres
            if time.process_time() >= n_end or meter.left() <= 2:
                break
            if not np.isfinite(sr[j]):
                break
            xy = cand[j][:, :2]
            r0 = _lp_radii(xy)
            sub = min(n_end, time.process_time() + max(0.15, (n_end - time.process_time()) * 0.3))
            xy2, r2, _ = _slp(xy, r0, sub, t0)
            fin.append(_repair(xy2, _lp_radii(xy2)))
            ftags.append(ctags[int(j)])
        # keep the second line of attack alive: SLSQP on the incumbent sees a different subproblem
        if fin and time.process_time() < n_end and n <= 71 and meter.left() > len(fin) + 2:
            xy3, r3 = _polish(pool[0][1], pool[0][2], cut, 25)
            fin.append(_repair(xy3, _lp_radii(xy3)))
            ftags.append("polish")
        if not fin:
            break
        feas, sr = evaluate(n, np.stack(fin))
        sr = np.where(feas, sr, -np.inf)
        for j in range(len(fin)):
            if not np.isfinite(sr[j]) or sr[j] <= worst + 1e-15:
                continue
            xy, r = fin[j][:, :2], fin[j][:, 2]
            # diversity: a basin already in the pool (same sum r to 1e-9) does not re-enter
            if any(abs(e[0] - float(sr[j])) < 1e-9 for e in pool):
                continue
            tg = ftags[j] if j < len(ftags) else "?"
            _prov(tg, n, max(0.0, float(sr[j]) - best_before))
            pool.append((float(sr[j]), xy.copy(), r.copy(), tg))
            _live_put(n, xy, r, sr[j])                  # publish NOW: the next n can transfer from it
            pool.sort(key=lambda e: -e[0])
            pool = pool[:pool_cap]
            worst = pool[-1][0] if len(pool) >= pool_cap else -np.inf
        # ---- ACTIVE-SET PIVOT (iteration 14): the only finisher that changes WHICH constraints are
        # active. Measured this iteration: the incumbent is an exact KKT point, so every other
        # finisher in this file is provably done before it starts. Run on the pool head, on a slice
        # of what is left of the slice, once per round.
        if time.process_time() < n_end and meter.left() > 24 and n >= 2:
            # 0.45 -> 0.70 (iteration 15): `pivot` is the ONLY move class in the file with a
            # non-zero sum_gain over the last three metered runs, and its measured offline ceiling
            # is reached at ~8 s of chain. Clock spent anywhere else has bought 0.000000 digits.
            pv_end = min(n_end, time.process_time() + max(0.30, (n_end - time.process_time()) * 0.70))
            try:
                got = _pivot_chain(evaluate, meter, rng, n, pool[0][1], pool[0][2], pool[0][0],
                                   pv_end, t0)
            except Exception:
                got = []
            for (ps, pxy, pr) in got:
                if any(abs(e[0] - ps) < 1e-9 for e in pool):
                    continue
                _prov("pivot", n, max(0.0, ps - best_before))
                pool.append((ps, pxy.copy(), pr.copy(), "pivot"))
                _live_put(n, pxy, pr, ps)
                pool.sort(key=lambda e: -e[0])
                pool = pool[:pool_cap]
        dt = time.process_time() - t_r0
        _RCOST[n] = dt if est is None else 0.6 * est + 0.4 * dt
        _ROUNDS[n] = _ROUNDS.get(n, 0) + 1
    return pool


def _merge_pool(a, b, cap=5):
    """Union two elite pools, best sum r first, one entry per distinct basin, at most `cap`.

    The restart portfolio (iteration 9) throws away the SEARCH STATE of a losing draw but must never
    throw away its RESULT: the incumbent an earlier attempt found has to survive into the resuming
    wave, or an independent restart would be a regression rather than a lottery ticket."""
    seen, keep = [], []
    for e in sorted(list(a or []) + list(b or []), key=lambda e: -e[0]):
        if any(abs(e[0] - s) < 1e-12 for s in seen):
            continue
        seen.append(e[0])
        keep.append(e)
        if len(keep) >= cap:
            break
    return keep


# --------------------------------------------------------------- THE DRAW BANDIT (iteration 11 core)
# Iteration 9 measured that success is a property OF THE RNG STREAM, not of the clock spent on it
# (n=27: stream 101 beats the record in 2.3 CPU-s, stream 7 fails at 9.6 s and nine rounds). Iteration
# 10 then measured the cost of NOT acting on that: with a fixed 3-wave equal split, `BEST:warm` owned
# the final pool for 9 of the 10 active n -- ~30 draws bought exactly ONE improvement. A binary
# hit/miss reward is far too sparse to steer 30 draws, so the arm value here is a DENSE surrogate for
# the right tail of the restart distribution: how close the draw's OWN search (everything except the
# carried incumbent) came to the incumbent it had to beat, in digits.
_QS = {}          # n -> [draw quality, ...] for this call, dumped to artifacts for the next iteration


def _draw_quality(pool, base):
    """Digits of agreement between this draw's own best NON-warm basin and the incumbent `base`.

    12.0 means the draw's own search matched or beat the incumbent -- a HIT. 3.0 means it landed a
    part in a thousand short. The warm entry is excluded on purpose: it is the incumbent handed back,
    not something the draw found, and counting it would make every arm look identical."""
    if not pool or base is None or not np.isfinite(base) or base <= 0:
        return 0.0
    vals = [e[0] for e in pool if e[3] != "warm"]
    if not vals:
        return 0.0
    v = max(vals)
    if v >= base:
        return 12.0
    return float(min(12.0, -np.log10(max((base - v) / base, 1e-12))))


def _egain(dig, _W=((0.6, 1e-6), (0.3, 1e-4), (0.1, 1e-2))):
    """EXPECTED DIGITS a draw on an arm at `dig` digits can buy, under a scale mixture over how big
    an ABSOLUTE improvement the draw achieves.

    Four iterations of lessons say the old `head = 7 - dig` term is the allocator's central flaw: it
    ranks an arm by how many digits are LEFT, which structurally starves whichever arm is CLOSEST to
    its record -- and closeness is exactly what makes a digit cheap. Iteration 14: n=31 sits 1.07e-5
    below its record while 37/41/45/59/81 sit 1.6e-3..2.3e-3 below theirs -- 150x further -- and the
    bandit gave n=31 one draw of 37.

    The fix is to price the arm by the digits a REALISTIC improvement buys, not by the digits that
    remain. With relative gap g = 10^-dig, an absolute relative improvement d moves the arm
    log10(g / (g-d)) digits (capped at the 7-digit cap). Averaging that over d in {1e-6, 1e-4, 1e-2}
    with weights {0.6, 0.3, 0.1} -- small gains common, large gains rare -- is a HEDGE, not a
    re-pointing: the 1e-2 component keeps every far arm live at 0.1 of full headroom, while the 1e-6
    component pays out only for an arm already within a part in 1e5. On the 2026-08-22 census this
    lifts n=31 from ~1/13 of the leaders' priority to ~1.7x it, and no arm's score goes to zero."""
    d0 = min(max(float(dig), 0.0), 7.0)
    head = max(0.0, 7.0 - d0)
    if head <= 0.0:
        return 0.0
    g = 10.0 ** (-d0)
    tot = 0.0
    for (w, d) in _W:
        tot += w * min(head, float(np.log10(g / max(g - d, 1e-7))))
    return tot


def _arm_score(n, dig, rng):
    """Bandit priority for one n: expected DIGITS PER CPU-SECOND, Thompson-perturbed.

    headroom  -- `_egain`: the digits a REALISTIC improvement buys on this arm (an n at the cap is
                 worth nothing however easy it is; an n a hair short of its record is worth a lot)
    q^2       -- the near-miss surrogate, squared: hit probability rises sharply in q, and squaring
                 a bounded [0,12] score gives a ~5x preference for a q=7 arm over a q=3 one without
                 the runaway of an exponential.
    /cost     -- a round on n=99 costs several times a round on n=27; the objective is SCORE per
                 second, so the price of a draw belongs in the denominator.
    noise     -- lognormal(0, 0.45): every arm keeps a live chance of being picked no matter how bad
                 its record looks, so a genuinely heavy-tailed arm can never be starved out."""
    head = _egain(dig)                                 # iteration 15: expected, not remaining
    qs = _QS.get(n) or []
    if not qs:
        q = 7.0                                        # optimistic prior: an unplayed arm looks good
    else:
        q = 0.6 * max(qs) + 0.4 * (sum(qs) / len(qs))
    cost = _RCOST.get(n) or 1.0
    noise = float(np.exp(rng.normal(0.0, 0.45)))
    return noise * head * (q * q) / max(cost, 0.05)


def _draw_len(n):
    """How long ONE draw gets. A draw must pay for its seeding phase AND at least one memetic round
    or it cannot produce anything the incumbent does not already have; beyond that, iteration 9
    measured that more seconds on the same stream buy nothing, so the ceiling is deliberately low."""
    est = _RCOST.get(n)
    if est is None:
        return 2.4
    return float(min(3.6, max(1.8, 3.0 * est)))


def _qdump(path="artifacts/qstats_latest.txt"):
    try:
        with open(path, "w") as f:
            f.write("%-6s %6s %7s %7s %7s\n" % ("n", "draws", "maxq", "meanq", "hits"))
            for n in sorted(_QS):
                qs = _QS[n]
                f.write("%-6d %6d %7.2f %7.2f %7d\n"
                        % (n, len(qs), max(qs), sum(qs) / len(qs),
                           sum(1 for q in qs if q >= 12.0)))
    except Exception:
        pass


def solve(evaluate, meter, rng, targets):
    """TRANSFER WAVES over a LIVE census (iteration 6) -> RESTART PORTFOLIO (iteration 9) -> THE DRAW
    BANDIT (iteration 11).

    The census is not 37 independent searches: a converged n is a start for n+-2 that no single-n
    move can reach, and every improvement is published to _LIVE the instant evaluate() confirms it,
    so structure propagates several hops per call. Iteration 9 then measured that a CONTINUED search
    is dominated by an INDEPENDENT RESTART -- success is a property of the rng stream, not of the
    clock -- so search state is thrown away between attempts and only RESULTS are merged.

    What remained from the wave era was the ALLOCATOR: three passes, each n getting a slice sized by
    a fixed weight. Iteration 10 measured what that costs -- ~30 draws, ONE improvement, `BEST:warm`
    owning 9 of 10 active n -- because the slices are spread evenly over arms whose hit rates differ
    by an order of magnitude (n=27/51/65 fell to a single fresh draw; n=41 lost on both streams at
    both budgets). So the unit of scheduling is no longer a wave slice but a DRAW, and draws are
    handed out by a bandit:

      PHASE A -- one draw per active n, in cascade order. Forced exploration: it gives every arm an
                 observation, and it is the pass that seeds the live census so later draws on
                 neighbouring n have donors to transfer from.
      PHASE B -- draws to the arm with the best Thompson-perturbed digits-per-second, until the
                 clock ends. Two lines of attack stay alive: every third draw on an arm RESUMES its
                 merged pool (the deep search) instead of restarting (the lottery)."""
    if not targets:
        return
    t_end = time.process_time() + CPU_BUDGET           # AT ENTRY -- never a module-level absolute
    tgts = sorted(set(int(t) for t in targets))
    recs = _records()
    _QS.clear()
    carried, ndraw = {}, {}

    dig = {n: _digits_now(n, recs) for n in tgts}
    # HARD FREEZE (iteration 9): a capped n cannot yield another point of SCORE and the driver is
    # append-or-improve, so its pack survives untouched -- unless EVERY target is capped, in which
    # case there is nothing else to spend on and they come back.
    active = [m for m in tgts if dig[m] < 6.9] or list(tgts)
    good = [m for m in tgts if dig[m] >= 6.9]

    def _one_draw(n, budget, resume):
        """Run a single draw on n and record what its own search was worth."""
        n_end = time.process_time() + budget
        base = _best(n)
        base_s = float(base[1].sum()) if base is not None else None
        sub = np.random.RandomState(int(rng.randint(1, 2147483647)))
        pool_in = list(carried.get(n) or []) if resume else []
        try:
            out = _run_n(evaluate, meter, sub, n, recs, n_end, pool_in)
        except Exception:
            out = []                                   # one bad n must never end the call
        _QS.setdefault(n, []).append(_draw_quality(out, base_s))
        ndraw[n] = ndraw.get(n, 0) + 1
        # MERGE, never discard: a losing independent draw still leaves the earlier winner in place.
        carried[n] = _merge_pool(carried.get(n), out, 5)

    # ---- PHASE A: forced exploration in CASCADE ORDER (nearest a record-holder first, then worst) --
    def _rank(n):
        d = min([abs(n - m) for m in good if m != n], default=999)
        return (d, dig[n], n)

    order = sorted(active, key=_rank)
    a_end = time.process_time() + 0.55 * (t_end - time.process_time())
    for pos, n in enumerate(order):
        if meter.left() <= 0:
            break
        left = t_end - time.process_time()
        if left <= 0.15:
            break
        rem = len(order) - pos
        # never let the sweep eat the bandit's clock: each arm gets at most its equal share of PHASE
        # A's slice, and the last arm of the sweep does NOT get to swallow the remainder.
        budget = min(_draw_len(n), max(0.3, (a_end - time.process_time()) / rem))
        _one_draw(n, min(budget, left), resume=False)

    # ---- PHASE B: bandit draws -------------------------------------------------------------------
    while meter.left() > 0:
        left = t_end - time.process_time()
        if left <= 0.4:
            break
        dig = {n: _digits_now(n, recs) for n in tgts}  # live: an n that just capped leaves the pool
        cands = [m for m in active if dig[m] < 6.9] or list(active)
        n = max(cands, key=lambda m: _arm_score(m, dig[m], rng))
        # keep BOTH lines alive: two lottery restarts, then one resumed deep search, and repeat
        resume = (ndraw.get(n, 0) % 3) == 2
        _one_draw(n, min(_draw_len(n), left), resume)

    for n, pl in carried.items():                      # once per call, not once per draw
        if pl:
            _prov("BEST:" + str(pl[0][3]), n, 0.0)
    _prov_dump()
    _qdump()

# ------------------------------------------------------------------------------------------ self-test
def _self_test():
    import sys

    class _M:
        def __init__(self, b):
            self.budget, self.used = b, 0

        def left(self):
            return max(0, self.budget - self.used)

        def tick(self, k=1):
            g = max(0, min(int(k), self.budget - self.used))
            self.used += g
            return g

    def _validate(a, n):
        a = np.asarray(a, float)
        if a.shape != (n, 3) or not np.all(np.isfinite(a)):
            return False, -np.inf
        x, y, r = a[:, 0], a[:, 1], a[:, 2]
        if r.min() <= 0.0:
            return False, -np.inf
        w = np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min()
        p = np.inf
        if n >= 2:
            ii, jj = np.triu_indices(n, 1)
            p = np.min(np.hypot(x[ii] - x[jj], y[ii] - y[jj]) - (r[ii] + r[jj]))
        return bool(w >= -1e-9 and p >= -1e-9), float(r.sum())

    ok = True
    seen = {}

    def make(meter):
        def evaluate(n, packing):
            a = np.asarray(packing, float)
            single = (a.ndim == 2)
            if single:
                a = a[None]
            B = a.shape[0]
            grant = meter.tick(B)
            feas = np.zeros(B, bool)
            sr = np.full(B, -np.inf)
            for i in range(min(grant, B)):
                f, s = _validate(a[i], n)
                feas[i], sr[i] = f, s
                if f and s > seen.get(n, (-np.inf,))[0]:
                    seen[n] = (s, a[i].copy())
            return (bool(feas[0]), float(sr[0])) if single else (feas, sr)
        return evaluate

    # -- monotonicity of the SLP engine: it must never lose ground from a feasible start ------------
    rs = np.random.RandomState(7)
    xy = rs.uniform(-0.45, 0.45, (24, 2))
    r0 = _lp_radii(xy)
    s0 = float(_repair(xy, r0)[:, 2].sum())
    xy1, r1, s1 = _slp(xy, r0, time.process_time() + 3.0, 0.25 / np.sqrt(24))
    f, sv = _validate(_repair(xy1, r1), 24)
    print("slp n=24: %.6f -> %.6f feasible=%s" % (s0, sv, f))
    if not f or sv < s0 - 1e-12:
        print("FAIL: SLP is not monotone / not feasible")
        ok = False

    # -- ACTIVE-SET PIVOT (iteration 14) -----------------------------------------------------------
    # (a) the duals exist, are non-negative, and DISCRIMINATE between contacts
    pk31 = _read_pack(31)
    if pk31 is None:
        print("note: no committed n=31 pack; dual test uses a relaxed random start")
        rs2 = np.random.RandomState(11)
        q = rs2.uniform(-0.42, 0.42, (16, 2))
        qq, qr, _ = _slp(q, _lp_radii(q), time.process_time() + 2.0, 0.25 / 4.0)
        dxy, dr = qq, qr
    else:
        dxy, dr = pk31[0], pk31[1]
    du = _duals(dxy, dr)
    lam = np.concatenate([du[2], du[5]])
    if lam.size < 4 or not np.all(np.isfinite(lam)) or lam.min() < 0.0:
        print("FAIL: _duals returned no/invalid multipliers (size=%d)" % lam.size)
        ok = False
    elif lam.max() <= 1.5 * lam.mean():
        print("FAIL: duals do not discriminate (max=%.4g mean=%.4g)" % (lam.max(), lam.mean()))
        ok = False
    else:
        print("duals: %d pair + %d wall multipliers, max=%.4g mean=%.4g"
              % (du[2].size, du[5].size, lam.max(), lam.mean()))
    if _duals(np.zeros((1, 2)), np.array([0.4]))[2].size != 0:
        print("FAIL: _duals must be empty for n<2")
        ok = False

    # (b) _pivot_picks must prefer the HIGH-multiplier constraint -- that is the whole point
    du_fake = (np.array([0, 0], np.intp), np.array([1, 2], np.intp), np.array([1.0, 0.1]),
               np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0))
    rs3 = np.random.RandomState(5)
    hits = [0, 0]
    for _ in range(400):
        pr, _w = _pivot_picks(du_fake, rs3, kmax=1)
        hits[0 if pr[0][1] == 1 else 1] += 1
    if hits[0] < 350:
        print("FAIL: pivot picks ignore the dual ranking (%d/400 on the 10x arm)" % hits[0])
        ok = False
    else:
        print("pivot picks: %d/400 on the 10x-multiplier contact (loser still live: %d)"
              % (hits[0], hits[1]))
    if _pivot_picks((np.zeros(0, np.intp),) * 2 + (np.zeros(0),) + (np.zeros(0, np.intp),) * 2
                    + (np.zeros(0),), rs3) != ([], []):
        print("FAIL: _pivot_picks must return empty for an empty dual set")
        ok = False

    # (c) the WALL ban must hold a circle off its wall -- and must not touch the other batch rows
    rs4 = np.random.RandomState(3)
    C0 = rs4.uniform(-0.30, 0.30, (3, 6, 2))
    R0 = np.full((3, 6), 0.05)
    Cn, Rn = _relax(C0, R0, 120)
    WB = (np.array([1], np.intp), np.array([0], np.intp), np.array([1], np.intp),
          np.array([0.18]))                                   # row 1, circle 0, +x wall, 0.18 clear
    Cw, Rw = _relax(C0, R0, 120, WBAN=WB)
    gap_free = 0.5 - (Cn[1, 0, 0] + Rn[1, 0])
    gap_ban = 0.5 - (Cw[1, 0, 0] + Rw[1, 0])
    if not (gap_ban > gap_free + 0.05 and gap_ban > 0.10):
        print("FAIL: wall ban did not push the circle off the wall (free=%.4f banned=%.4f)"
              % (gap_free, gap_ban))
        ok = False
    else:
        print("wall ban: +x clearance %.4f -> %.4f" % (gap_free, gap_ban))
    if not (np.array_equal(Cn[0], Cw[0]) and np.array_equal(Cn[2], Cw[2])
            and np.array_equal(Rn[0], Rw[0]) and np.array_equal(Rn[2], Rw[2])):
        print("FAIL: a wall ban on row 1 perturbed rows 0/2 (must be bit-identical)")
        ok = False
    else:
        print("wall ban is row-local: rows 0,2 bit-identical to the unbanned relax")

    # (d) the chain: everything it returns is FEASIBLE and STRICTLY better than its start; it obeys
    #     its deadline; and small n do not break it
    mp = _M(500000)
    evp = make(mp)
    ps0 = float(dr.sum())
    tp = time.process_time()
    got = _pivot_chain(evp, mp, np.random.RandomState(21), len(dr), dxy, dr, ps0,
                       time.process_time() + 4.0, 0.25 / np.sqrt(len(dr)))
    el = time.process_time() - tp
    bad = [g for g in got if not _validate(np.concatenate([g[1], g[2][:, None]], 1), len(dr))[0]
           or g[0] <= ps0]
    if bad:
        print("FAIL: pivot chain returned %d infeasible/non-improving results" % len(bad))
        ok = False
    if el > 6.5:
        print("FAIL: pivot chain overran its deadline (%.1fs for a 4.0s budget)" % el)
        ok = False
    print("pivot chain n=%d: %d improvements in %.1fs, best gain %.3e"
          % (len(dr), len(got), el, (max(g[0] for g in got) - ps0) if got else 0.0))
    for nn in (2, 3, 5):
        q = np.random.RandomState(nn).uniform(-0.3, 0.3, (nn, 2))
        qr = _lp_radii(q)
        try:
            g2 = _pivot_chain(evp, mp, np.random.RandomState(nn), nn, q, qr, float(qr.sum()),
                              time.process_time() + 0.6, 0.25 / np.sqrt(nn))
        except Exception as e:
            print("FAIL: pivot chain raised at n=%d: %r" % (nn, e))
            ok = False
            g2 = []
        for g in g2:
            if not _validate(np.concatenate([g[1], g[2][:, None]], 1), nn)[0]:
                print("FAIL: pivot chain infeasible at n=%d" % nn)
                ok = False
    print("pivot chain survives n=2,3,5")

    # (e) iteration 15: the chain is PERSISTENT -- it must leave a valid state behind, that state
    #     must actually be CONSUMED on re-entry, and it must not be written for n<2.
    _PIV.clear()
    n_p = len(dr)
    _ = _pivot_chain(evp, mp, np.random.RandomState(7), n_p, dxy, dr, ps0,
                     time.process_time() + 1.2, 0.25 / np.sqrt(n_p))
    stp = _PIV.get(n_p)
    if stp is None or stp[0].shape != (n_p, 2) or stp[1].shape != (n_p,) or not (stp[3] > 0.0):
        print("FAIL: pivot chain left no usable persistent state")
        ok = False
    elif not _validate(np.concatenate([stp[0], stp[1][:, None]], 1), n_p)[0]:
        print("FAIL: the persisted chain point is not feasible")
        ok = False
    else:
        print("chain state persisted: sum r %.9f, T %.3e" % (stp[2], stp[3]))
    # the DISCRIMINATING check: a stored state must actually be CONSUMED. Seed the walk with a
    # temperature of 1.0 -- three orders of magnitude outside the fresh draw's [7e-6, 1.8e-4] range
    # -- and the persisted T after the run says which branch ran. Seed 2 takes the resume branch
    # (its first rand() is 0.436 >= 0.20), seed 7 takes the REHEAT branch (0.076 < 0.20), so this
    # pins both halves of the hedge, not just one.
    _PIV.clear()
    _PIV[n_p] = (dxy.copy(), dr.copy(), ps0, 1.0)
    _pivot_chain(evp, mp, np.random.RandomState(2), n_p, dxy, dr, ps0,
                 time.process_time() + 1.2, 0.25 / np.sqrt(n_p))
    t_res = _PIV.get(n_p)
    _PIV.clear()
    _PIV[n_p] = (dxy.copy(), dr.copy(), ps0, 1.0)
    _pivot_chain(evp, mp, np.random.RandomState(7), n_p, dxy, dr, ps0,
                 time.process_time() + 1.2, 0.25 / np.sqrt(n_p))
    t_heat = _PIV.get(n_p)
    if t_res is None or t_heat is None:
        print("FAIL: a 1.2s chain took no step at all -- the resume test cannot discriminate")
        ok = False
    elif not t_res[3] > 0.5:
        print("FAIL: a stored temperature was NOT consumed on resume (T=%.3e, seeded 1.0)"
              % t_res[3])
        ok = False
    elif not t_heat[3] < 1.0e-3:
        print("FAIL: the 1-in-5 REHEAT branch is dead (T=%.3e, expected a fresh ~1e-5 draw)"
              % t_heat[3])
        ok = False
    else:
        print("chain resume is live: seeded T 1.0 -> continued %.3f / reheated %.3e"
              % (t_res[3], t_heat[3]))
    _PIV.clear()
    _pivot_chain(evp, mp, np.random.RandomState(1), 1, dxy[:1], dr[:1], float(dr[0]),
                 time.process_time() + 0.2, 0.3)
    if _PIV:
        print("FAIL: pivot chain wrote state for n<2")
        ok = False
    _PIV.clear()

    # (f) iteration 15: the allocator must price CLOSENESS. An arm 1 part in 1e5 short of its record
    #     (dig 5.5) has less headroom left than one 5 parts in 1e4 short (dig 3.3) and the OLD score
    #     ranked it 2.4x lower for that reason; the new one must rank it ABOVE, while leaving every
    #     sub-cap arm strictly positive and a capped arm at zero.
    e_close, e_far, e_wide = _egain(5.5), _egain(3.3), _egain(1.0)
    if not (e_close > e_far > 0.0 and e_wide > 0.0):
        print("FAIL: _egain does not prefer the near-record arm (%.4f vs %.4f, wide %.4f)"
              % (e_close, e_far, e_wide))
        ok = False
    elif _egain(7.0) != 0.0 or _egain(7.4) != 0.0:
        print("FAIL: _egain must be 0 at/above the digit cap")
        ok = False
    elif not (max(0.0, 7.0 - 5.5) < max(0.0, 7.0 - 3.3)):
        print("FAIL: the old headroom term did not actually invert here -- test is vacuous")
        ok = False
    else:
        print("egain: dig5.5 %.4f > dig3.3 %.4f (old headroom 1.50 < 3.70), dig1.0 %.4f, cap 0"
              % (e_close, e_far, e_wide))
    if not all(_egain(d) > 0.0 for d in (0.0, 0.5, 2.0, 4.0, 6.0, 6.89)):
        print("FAIL: _egain starves some sub-cap arm to exactly zero")
        ok = False

    # -- the batched relaxer must produce feasible, non-degenerate packings for a WHOLE batch ------
    rs2 = np.random.RandomState(3)
    C0 = rs2.uniform(-0.45, 0.45, (6, 20, 2))
    C1, R1 = _relax(C0, _grow_batch(C0), 200)
    pk = _repair_batch(C1, _grow_batch(C1))
    for b in range(6):
        f, s = _validate(pk[b], 20)
        if not f or not np.isfinite(s) or s <= 0:
            print("FAIL: _relax batch member %d infeasible" % b)
            ok = False
    print("relax batch n=20: sums %s" % np.round(pk[:, :, 2].sum(1), 4))
    # -- the barrier engine: every iterate strictly feasible, and MONOTONE in sum r ----------------
    Cb = rs2.uniform(-0.45, 0.45, (5, 18, 2))
    Rb = _feasible_seed(Cb)
    s_in = Rb.sum(1)
    Cb2, Rb2 = _barrier(Cb, Rb, 200, t0=3e-4, t1=1e-10)
    ms = _minslack(Cb2, Rb2)
    if ms.min() <= 0.0:
        print("FAIL: _barrier left the feasible set (min slack %.3e)" % ms.min())
        ok = False
    if np.any(Rb2.sum(1) < s_in - 1e-12):
        print("FAIL: _barrier is not monotone in sum r")
        ok = False
    for b in range(5):
        f, s = _validate(_repair(Cb2[b], Rb2[b]), 18)
        if not f:
            print("FAIL: _barrier batch member %d infeasible" % b)
            ok = False
    print("barrier batch n=18: %s -> %s (min slack %.2e)"
          % (np.round(s_in, 4), np.round(Rb2.sum(1), 4), ms.min()))

    # -- CROSS-n TRANSFER: add/remove must produce exactly the target count, finite, in the box ----
    rs3 = np.random.RandomState(11)
    base = rs3.uniform(-0.42, 0.42, (30, 2))
    rb = _grow(base)
    for k in (1, 2, 5):
        up = _gap_insert(base, rb, k, rs3)
        dn = _drop_small(base, rb, k, rs3)
        if up.shape != (30 + k, 2) or dn.shape != (30 - k, 2):
            print("FAIL: transfer shape (k=%d): %s %s" % (k, up.shape, dn.shape))
            ok = False
        if not (np.all(np.isfinite(up)) and np.all(np.isfinite(dn))
                and up.min() >= LO - 1e-12 and up.max() <= HI + 1e-12):
            print("FAIL: transfer produced out-of-box / non-finite centres (k=%d)" % k)
            ok = False
        # an inserted circle must land in genuinely free space, not on top of an existing one
        add = up[30:]
        d = np.sqrt(((add[:, None, :] - base[None]) ** 2).sum(-1)) - rb[None]
        if d.min() <= 0.0:
            print("FAIL: _gap_insert placed a centre inside an existing circle (k=%d)" % k)
            ok = False
    # transfer seeds for an n whose neighbours are ALL missing must be [] and must not raise
    if _transfer_seeds(3, rs3, {}) != []:
        print("FAIL: _transfer_seeds invented a donor for an n with no committed neighbours")
        ok = False
    ts = _transfer_seeds(28, rs3, _records())
    print("transfer seeds for n=28 from the committed census: %d (shapes %s)"
          % (len(ts), [t[0].shape for t in ts]))
    for t in ts:
        if (not isinstance(t, tuple)) or t[0].shape != (28, 2) or t[1] not in ("transfer",
                                                                               "transfer*"):
            print("FAIL: transfer seed wrong shape/tag %s" % (t,))
            ok = False
    # a pool round must emit one provenance tag per batch row
    _pr = [(1.0, base[:20], rb[:20], "warm"), (0.9, base[5:25], rb[5:25], "cold")]
    NB = len(_SCHED)
    Cx, tx, Wx, BNx = _pool_round(20, rs3, _pr, NB, 20, [(base[:20], "transfer*")])
    if Cx.shape != (NB, 20, 2) or len(tx) != NB or "transfer*" not in tx:
        print("FAIL: _pool_round tags/shape %s %s" % (Cx.shape, tx))
        ok = False
    if sorted(set(tx)) != sorted(set(["teleport", "crossover", "jitter", "rotate", "symm",
                                      "ratio", "ban", "ruin", "box", "transfer*"])):
        print("FAIL: _pool_round did not exercise every move in one full cycle: %s" % (tx,))
        ok = False
    # RATIO ROWS carry a non-trivial weight vector; every other row must be the TRUE objective, or
    # the deformation would silently leak into moves that are supposed to be undeformed.
    if Wx.shape != (NB, 20) or not np.all(np.isfinite(Wx)) or Wx.min() <= 0.0:
        print("FAIL: _pool_round weights shape/positivity %s" % (Wx.shape,))
        ok = False
    for b, t in enumerate(tx):
        if t == "ratio":
            if np.all(Wx[b] == 1.0):
                print("FAIL: ratio row %d carries the trivial weight vector" % b)
                ok = False
        elif not np.all(Wx[b] == 1.0):
            print("FAIL: non-ratio row %d (%s) carries a deformed weight vector" % (b, t))
            ok = False
    # a deformed relax must ACTUALLY move the packing somewhere a plain relax does not
    Cz = np.stack([base[:20], base[:20]])
    Wz = np.stack([np.ones(20), _ratio_weights(20, rs3)])
    Ca, _ = _relax(Cz.copy(), _grow_batch(Cz), 60, W=Wz)
    Cb, _ = _relax(Cz.copy(), _grow_batch(Cz), 60)
    if not np.all(np.isfinite(Ca)):
        print("FAIL: weighted _relax produced non-finite centres")
        ok = False
    if np.abs(Ca[0] - Cb[0]).max() > 1e-12:
        print("FAIL: an all-ones weight row is not identical to the undeformed relax")
        ok = False
    if np.abs(Ca[1] - Cb[1]).max() < 1e-6:
        print("FAIL: a deformed weight row moved nowhere the plain relax did not")
        ok = False

    # -- CONTACT BAN (iteration 10): the ban list must name genuinely ACTIVE contacts, every ban
    # -- row must carry at least one, no other row may carry any, and stage 1 must actually hold
    # -- the banned pair open while leaving the rest of the batch bit-identical.
    if BNx is None:
        print("FAIL: a full schedule cycle emitted no contact bans")
        ok = False
    else:
        bb, bi, bj, bd = BNx
        if not (len(bb) == len(bi) == len(bj) == len(bd)) or len(bb) == 0:
            print("FAIL: contact-ban list is ragged/empty %s" % (BNx,))
            ok = False
        if bd.min() <= 0.0 or not np.all(np.isfinite(bd)):
            print("FAIL: contact-ban separations must be positive and finite: %s" % (bd,))
            ok = False
        if np.any(bi == bj) or bi.min() < 0 or bj.max() >= 20 or bb.max() >= NB:
            print("FAIL: contact-ban indices out of range / self-pair")
            ok = False
        rows = set(int(v) for v in bb)
        for b, t in enumerate(tx):
            if (t == "ban") != (b in rows):
                print("FAIL: ban rows and ban list disagree at row %d (%s)" % (b, t))
                ok = False
    # the pairs picked on a converged packing must really be in contact
    bl = _ban_contacts(base[:20], _lp_radii(base[:20]), rs3, kmax=3)
    r20 = _lp_radii(base[:20])
    if not bl:
        print("FAIL: _ban_contacts found no contact in a grown packing")
        ok = False
    for (a_, b_, d_) in bl:
        gap = float(np.hypot(*(base[a_] - base[b_])) - (r20[a_] + r20[b_]))
        if gap > 1e-3 * (r20[a_] + r20[b_]) + 1e-6:
            print("FAIL: _ban_contacts picked a NON-active pair (gap %.3e)" % gap)
            ok = False
    # n < 2 must not raise
    if _ban_contacts(base[:1], _grow(base[:1]), rs3) != []:
        print("FAIL: _ban_contacts on n=1 did not return []")
        ok = False
    # stage-1 effect: row 1 banned, row 0 untouched
    Cn = np.stack([base[:20], base[:20]])
    Rn = _grow_batch(Cn)
    pa, pb = int(bl[0][0]), int(bl[0][1])
    BN = (np.array([1], np.intp), np.array([pa], np.intp), np.array([pb], np.intp),
          np.array([0.4 * (r20[pa] + r20[pb])], float))
    Cd, Rd = _relax(Cn.copy(), Rn.copy(), 120, mu0=0.3, mu1=40.0, BAN=BN)
    Cp, Rp = _relax(Cn.copy(), Rn.copy(), 120, mu0=0.3, mu1=40.0)
    if np.abs(Cd[0] - Cp[0]).max() > 1e-12:
        print("FAIL: a contact ban on row 1 leaked into un-banned row 0")
        ok = False
    sep_d = float(np.hypot(*(Cd[1][pa] - Cd[1][pb])) - (Rd[1][pa] + Rd[1][pb]))
    sep_p = float(np.hypot(*(Cp[1][pa] - Cp[1][pb])) - (Rp[1][pa] + Rp[1][pb]))
    print("contact ban: banned-pair slack %.4e (plain relax %.4e)" % (sep_d, sep_p))
    if not (sep_d > sep_p + 1e-4):
        print("FAIL: the contact ban did not prise its pair open (%.3e vs %.3e)" % (sep_d, sep_p))
        ok = False
    if not np.all(np.isfinite(Cd)):
        print("FAIL: banned _relax produced non-finite centres")
        ok = False
    # -- RUIN & RECREATE (iteration 12): the rebuild must return exactly n distinct in-box centres,
    # -- must actually DESTROY a region (many circles displaced far, unlike jitter/teleport), must be
    # -- non-deterministic across draws, and must survive tiny n. Every ruin mode is exercised.
    rs5 = np.random.RandomState(4242)
    b30 = base[:30]
    r30 = _lp_radii(b30)
    moved_frac, seen_modes = [], []
    for _ in range(24):
        Q = _ruin_recreate(b30, r30, rs5)
        if Q.shape != (30, 2) or not np.all(np.isfinite(Q)):
            print("FAIL: _ruin_recreate shape/finiteness %s" % (Q.shape,))
            ok = False
            break
        if Q.min() < LO - 1e-12 or Q.max() > HI + 1e-12:
            print("FAIL: _ruin_recreate left the box")
            ok = False
            break
        # a rebuilt circle has no counterpart near its old seat: count centres of the ORIGINAL that
        # no rebuilt centre lands within a small radius of.
        D = np.sqrt(((b30[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).min(1)
        moved_frac.append(float((D > 0.02).mean()))
        seen_modes.append(round(float(np.sort(D)[-1]), 6))
    if ok:
        mf = np.array(moved_frac)
        if mf.max() < 0.05:
            print("FAIL: _ruin_recreate never displaced 5%% of the packing (max %.3f)" % mf.max())
            ok = False
        if mf.min() > 0.60:
            print("FAIL: _ruin_recreate always destroyed >60%% -- ruin fraction is not bounded")
            ok = False
        if len(set(seen_modes)) < 8:
            print("FAIL: _ruin_recreate is not varying across draws (%d distinct)" % len(set(seen_modes)))
            ok = False
        print("ruin&recreate: displaced fraction min %.3f mean %.3f max %.3f over 24 draws"
              % (mf.min(), mf.mean(), mf.max()))
    # a ruin must never collide two centres onto one point (that is the NaN the dist floor guards)
    Qc = _ruin_recreate(b30, r30, np.random.RandomState(9))
    dmin = np.sqrt(((Qc[:, None, :] - Qc[None, :, :]) ** 2).sum(-1))
    dmin[np.arange(30), np.arange(30)] = 1.0
    if dmin.min() <= 0.0:
        print("FAIL: _ruin_recreate produced coincident centres")
        ok = False
    for tiny in (2, 3, 4, 5):
        Qt = _ruin_recreate(base[:tiny], _grow(base[:tiny]), rs5)
        if Qt.shape != (tiny, 2) or not np.all(np.isfinite(Qt)):
            print("FAIL: _ruin_recreate broke at n=%d" % tiny)
            ok = False
    # the schedule must really emit ruin rows, and they must be tagged as such
    if "ruin" not in _SCHED:
        print("FAIL: ruin is not in the schedule")
        ok = False
    if NB >= len(_SCHED) and tx.count("ruin") < 1:
        print("FAIL: a full schedule cycle emitted no ruin row (%s)" % (tx,))
        ok = False
    # a ruin row must carry NO contact ban and NO deformed weight -- it is a pure centre-space rebuild
    for b, t in enumerate(tx):
        if t == "ruin" and not np.all(Wx[b] == 1.0):
            print("FAIL: ruin row %d carries a deformed weight vector" % b)
            ok = False

    # -- CONTAINER-DEFORMATION HOMOTOPY (iteration 13). Three things have to hold. (a) `_relax` must
    # -- honour a NON-SQUARE container -- that is the new primitive and everything rests on it.
    # -- (b) the homotopy must return a legal square packing. (c) it must change the CONTACT GRAPH,
    # -- not merely displace circles: that is the whole claim over jitter/teleport.
    rs6 = np.random.RandomState(20260822)
    Cb = base[:24][None].copy()
    Rb = _lp_radii(base[:24])[None].copy()
    # `_relax` is a PENALTY method -- it leaves an O(1e-3) residual violation in the square too, so
    # the honest rectangle test is (i) the residual is no larger RELATIVE TO THE BOX than the square's
    # own, and (ii) no circle exceeds the box half-short-side. (ii) is the discriminating one: a
    # relaxer that still believed the container was [-0.5,0.5]^2 would grow r up to 0.5 in box(4.0),
    # i.e. r/min(bx,by) = 2.0, so a bound of 1.02 cannot be passed by accident.
    for (ee) in (0.25, 0.62, 1.0, 1.55, 4.0):
        bxx, byy = 0.5 * np.sqrt(ee), 0.5 / np.sqrt(ee)
        Cr, Rr = _relax(Cb * np.array([np.sqrt(ee), 1.0 / np.sqrt(ee)]),
                        np.maximum(Rb * min(np.sqrt(ee), 1.0 / np.sqrt(ee)), 1e-9), 220,
                        bx=bxx, by=byy)
        if abs(4.0 * bxx * byy - 1.0) > 1e-12:
            print("FAIL: box(%.2f) is not equal-area (%.6f)" % (ee, 4.0 * bxx * byy))
            ok = False
        if Cr[:, :, 0].max() > bxx + 1e-9 or Cr[:, :, 0].min() < -bxx - 1e-9 or \
           Cr[:, :, 1].max() > byy + 1e-9 or Cr[:, :, 1].min() < -byy - 1e-9:
            print("FAIL: rectangle _relax let a centre out of box(%.2f)" % ee)
            ok = False
        ws = np.minimum.reduce([Cr[0, :, 0] + bxx, bxx - Cr[0, :, 0],
                                byy + Cr[0, :, 1], byy - Cr[0, :, 1]]) - Rr[0]
        relv = max(0.0, -float(ws.min())) / (2.0 * min(bxx, byy))
        if relv > 0.035:
            print("FAIL: box(%.2f) wall violation %.4f of the short side" % (ee, relv))
            ok = False
        if Rr[0].max() > min(bxx, byy) * 1.02:
            print("FAIL: box(%.2f) grew r=%.4f past the half-short-side %.4f"
                  % (ee, Rr[0].max(), min(bxx, byy)))
            ok = False
    # a square call must be BIT-IDENTICAL to the pre-parameterisation default path
    Ca, Ra = _relax(Cb.copy(), Rb.copy(), 60)
    Cc, Rc = _relax(Cb.copy(), Rb.copy(), 60, bx=0.5, by=0.5)
    if not (np.array_equal(Ca, Cc) and np.array_equal(Ra, Rc)):
        print("FAIL: bx/by defaults changed the square relax")
        ok = False

    def _cgraph(q):
        rq = _lp_radii(q)
        ii, jj = np.triu_indices(len(q), 1)
        dd = np.sqrt((q[ii, 0] - q[jj, 0]) ** 2 + (q[ii, 1] - q[jj, 1]) ** 2)
        m = (dd - (rq[ii] + rq[jj])) < 1e-6
        return frozenset(zip(ii[m].tolist(), jj[m].tolist()))

    g0 = _cgraph(b30)
    hs, rewired, disp = [], 0, []
    for _ in range(16):
        Qb = _box_homotopy(b30, r30, rs6)
        if Qb.shape != (30, 2) or not np.all(np.isfinite(Qb)):
            print("FAIL: _box_homotopy shape/finiteness %s" % (Qb.shape,))
            ok = False
            break
        if Qb.min() < LO - 1e-12 or Qb.max() > HI + 1e-12:
            print("FAIL: _box_homotopy left the unit square")
            ok = False
            break
        g1 = _cgraph(Qb)
        rewired += int(len(g0 ^ g1) > 0)
        hs.append(round(float(Qb.sum()), 9))
        disp.append(float(np.sqrt(((Qb - b30) ** 2).sum(1)).mean()))
    if ok:
        if len(set(hs)) < 8:
            print("FAIL: _box_homotopy is not varying across draws (%d distinct of 16)" % len(set(hs)))
            ok = False
        if rewired < 12:
            print("FAIL: _box_homotopy rewired the contact graph on only %d of 16 draws" % rewired)
            ok = False
        print("box homotopy: rewired %d/16 draws, mean centre displacement %.4f"
              % (rewired, float(np.mean(disp))))
    for tiny in (2, 3, 5):
        Qt = _box_homotopy(base[:tiny], _grow(base[:tiny]), rs6)
        if Qt.shape != (tiny, 2) or not np.all(np.isfinite(Qt)):
            print("FAIL: _box_homotopy broke at n=%d" % tiny)
            ok = False
    if "box" not in _SCHED:
        print("FAIL: box is not in the schedule")
        ok = False
    if NB >= len(_SCHED) and tx.count("box") < 1:
        print("FAIL: a full schedule cycle emitted no box row (%s)" % (tx,))
        ok = False
    for b, t in enumerate(tx):
        if t == "box" and not np.all(Wx[b] == 1.0):
            print("FAIL: box row %d carries a deformed weight vector" % b)
            ok = False
    if BNx is not None and any(tx[int(q)] == "box" for q in BNx[0]):
        print("FAIL: a box row carries a contact ban")
        ok = False
    # every older line of attack must still hold a slot -- rule 1, no line retired for the new one
    for _mv in ("teleport", "crossover", "ratio", "ban", "ruin", "symm", "jitter", "transfer",
                "rotate"):
        if _mv not in _SCHED:
            print("FAIL: line of attack %s was retired from the schedule" % _mv)
            ok = False

    # with NO transfer seeds the transfer slots must fall back, not crash or emit a transfer tag
    Cy, ty, Wy, BNy = _pool_round(20, rs3, _pr, NB, 20, ())
    if Cy.shape != (NB, 20, 2) or any(t.startswith("transfer") for t in ty):
        print("FAIL: _pool_round transfer fallback %s" % (ty,))
        ok = False

    # -- D4 SYMMETRY PROJECTION: a=1 must give an EXACTLY invariant point set, for every generator --
    rs4 = np.random.RandomState(23)
    Ps = rs4.uniform(-0.45, 0.45, (29, 2))
    for gi, M in enumerate(_D4):
        S = _symmetrize(Ps, rs4, a=1.0, g=M)
        if S.shape != (29, 2) or not np.all(np.isfinite(S)):
            print("FAIL: _symmetrize shape/finiteness for generator %d" % gi)
            ok = False
        if S.min() < LO - 1e-12 or S.max() > HI + 1e-12:
            print("FAIL: _symmetrize left the box for generator %d" % gi)
            ok = False
        # invariance: the image set must coincide with the set itself (jitter is 1e-4/sqrt(n))
        D = np.sqrt((((S @ M)[:, None, :] - S[None, :, :]) ** 2).sum(-1)).min(1).max()
        if D > 5e-4:
            print("FAIL: _symmetrize(a=1) is not invariant under generator %d (%.2e)" % (gi, D))
            ok = False
    # a=0 must be (essentially) the identity, so the blend really is a blend
    S0 = _symmetrize(Ps, rs4, a=0.0, g=_D4[0])
    if np.abs(S0 - Ps).max() > 1e-3:
        print("FAIL: _symmetrize(a=0) moved the configuration (%.2e)" % np.abs(S0 - Ps).max())
        ok = False
    # it must not raise on degenerate inputs
    for nn in (1, 2, 3):
        _symmetrize(rs4.uniform(-0.4, 0.4, (nn, 2)), rs4)
    print("symmetrize: 5 generators invariant, blend + degenerate n ok")

    # -- crossover must return exactly n centres from two different parents -----------------------
    xc = _crossover(C1[0], C1[1], rs2)
    if xc.shape != (20, 2) or not np.all(np.isfinite(xc)):
        print("FAIL: crossover shape/finiteness")
        ok = False

    # -- LIVE CENSUS: publish/read semantics, and it must never DOWNGRADE a committed pack ---------
    fake_n = 91731
    if _best(fake_n) is not None:
        print("FAIL: _best invented a pack for an n with no file")
        ok = False
    fx = rs3.uniform(-0.4, 0.4, (5, 2))
    _live_put(fake_n, fx, np.full(5, 0.01), 0.05)
    _live_put(fake_n, fx, np.full(5, 0.02), 0.10)
    _live_put(fake_n, fx, np.full(5, 0.005), 0.025)      # worse -> must be ignored
    bl = _best(fake_n)
    if bl is None or abs(float(bl[1].sum()) - 0.10) > 1e-12:
        print("FAIL: _live_put/_best did not keep the best (%s)" % (bl,))
        ok = False
    d27 = _read_pack(27)
    if d27 is not None:
        _live_put(27, d27[0], d27[1] * 0.5, float(d27[1].sum()) * 0.5)   # a WORSE live entry
        if abs(float(_best(27)[1].sum()) - float(d27[1].sum())) > 1e-12:
            print("FAIL: a worse live entry overrode the committed pack for n=27")
            ok = False
    _LIVE.clear()

    # -- RESTART PORTFOLIO: the merge must never lose a result (iteration 9) -----------------------
    ea = [(0.5, None, None, "a"), (0.3, None, None, "a")]
    eb = [(0.7, None, None, "b"), (0.5, None, None, "b"), (0.1, None, None, "b")]
    mg = _merge_pool(ea, eb, 5)
    if [round(e[0], 6) for e in mg] != [0.7, 0.5, 0.3, 0.1]:
        print("FAIL: _merge_pool did not union+sort+dedup correctly: %s" % ([e[0] for e in mg],))
        ok = False
    if _merge_pool(ea, [], 5)[0][0] != 0.5 or _merge_pool([], eb, 5)[0][0] != 0.7:
        print("FAIL: _merge_pool loses a pool when the other side is empty")
        ok = False
    if len(_merge_pool(ea, eb, 2)) != 2:
        print("FAIL: _merge_pool ignored its cap")
        ok = False
    if _merge_pool(None, None, 5) != []:
        print("FAIL: _merge_pool did not tolerate None")
        ok = False
    # an INDEPENDENT restart on an n with a committed pack must not regress below that pack: the
    # fresh pool is re-seeded from the live/committed best, so the lottery only ever adds.
    if _read_pack(29) is not None:
        _LIVE.clear()
        mR = _M(500000)
        evR = make(mR)
        pR = _run_n(evR, mR, np.random.RandomState(5), 29, _records(),
                    time.process_time() + 2.0, [])
        base29 = float(_read_pack(29)[1].sum())
        if not pR or pR[0][0] < base29 - 1e-9:
            print("FAIL: a fresh-stream restart on n=29 regressed below the committed pack")
            ok = False
        else:
            print("restart on n=29: %.9f vs committed %.9f" % (pR[0][0], base29))
        _LIVE.clear()
        seen.clear()

    # -- THE DRAW BANDIT (iteration 11) ------------------------------------------------------------
    # quality must EXCLUDE the carried incumbent, cap a hit at 12, and read digits of agreement
    if abs(_draw_quality([(1.0, None, None, "warm")], 1.0) - 0.0) > 1e-12:
        print("FAIL: _draw_quality counted the warm incumbent as something the draw found")
        ok = False
    if _draw_quality([(1.5, None, None, "teleport")], 1.0) != 12.0:
        print("FAIL: _draw_quality did not report a beat as a hit")
        ok = False
    q3 = _draw_quality([(1.0 - 1e-3, None, None, "ratio"), (1.0, None, None, "warm")], 1.0)
    if abs(q3 - 3.0) > 1e-9:
        print("FAIL: _draw_quality digits wrong (%.6f, want 3.0)" % q3)
        ok = False
    for bad in (None, -1.0, 0.0):
        if _draw_quality([(0.5, None, None, "cold")], bad) != 0.0:
            print("FAIL: _draw_quality did not tolerate base=%s" % (bad,))
            ok = False
    if _draw_quality([], 1.0) != 0.0:
        print("FAIL: _draw_quality on an empty pool")
        ok = False
    # the arm score must PREFER the near-miss arm, must zero out a capped arm, and must still give a
    # bad-looking arm a live chance (the lognormal keeps every possibility open)
    _QS.clear(); _RCOST.clear()
    _QS[41] = [3.0] * 8
    _QS[27] = [6.5] * 8
    rsb = np.random.RandomState(4)
    wins = sum(1 for _ in range(400) if _arm_score(27, 3.0, rsb) > _arm_score(41, 3.0, rsb))
    if wins < 240:
        print("FAIL: bandit preference wrong -- near-miss arm won only %d/400" % wins)
        ok = False
    # LIVENESS: the loser must keep a strictly positive score on every single draw, so no arm can be
    # starved out of the schedule by a bad early record (rule: never spend a possibility).
    if min(_arm_score(41, 3.0, rsb) for _ in range(200)) <= 0.0:
        print("FAIL: a low-quality arm was scored dead")
        ok = False
    print("bandit: q=6.5 arm beats q=3.0 arm %d/400 draws (loser stays strictly alive)" % wins)
    if _arm_score(27, 7.0, rsb) != 0.0:
        print("FAIL: a capped arm still scored above zero")
        ok = False
    _QS[59] = [0.0] * 5                                  # an arm that has produced literally nothing
    if _arm_score(59, 3.0, rsb) != 0.0 or _arm_score(59, 3.0, rsb) < 0.0:
        print("FAIL: a zero-quality arm scored badly-formed")
        ok = False
    # an UNPLAYED arm must outrank a played-and-bad one: forced exploration is priced in
    if _arm_score(77, 3.0, np.random.RandomState(0)) <= _arm_score(41, 3.0, np.random.RandomState(0)):
        print("FAIL: an unplayed arm did not get the optimistic prior")
        ok = False
    # draw length: bounded, positive, monotone in measured round cost
    _RCOST.clear()
    if not (2.0 <= _draw_len(31) <= 3.0):
        print("FAIL: default draw length out of range (%.2f)" % _draw_len(31))
        ok = False
    _RCOST[31] = 0.1
    lo = _draw_len(31)
    _RCOST[31] = 5.0
    hi = _draw_len(31)
    if not (1.8 <= lo <= hi <= 3.6):
        print("FAIL: draw length not bounded/monotone (%.2f, %.2f)" % (lo, hi))
        ok = False
    print("draw length: cheap n %.2fs, dear n %.2fs" % (lo, hi))
    _RCOST.clear(); _QS.clear()

    # -- CALL 1 -----------------------------------------------------------------------------------
    global CPU_BUDGET
    saved = CPU_BUDGET
    CPU_BUDGET = 6.0
    m1 = _M(500000)
    # both targets must be BELOW the cap or the hard freeze will (correctly) refuse to draw on them:
    # 27 was capped in iteration 11, so this pair moved to 31/37, which are still short.
    solve(make(m1), m1, np.random.RandomState(0), [31, 37])
    for n in (31, 37):
        if n not in seen:
            print("FAIL: call 1 produced nothing for n=%d" % n)
            ok = False
        if n not in _LIVE:
            print("FAIL: solve() did not publish n=%d to the live census" % n)
            ok = False
        else:
            d = _read_pack(n)
            if d is not None and _LIVE[n][2] < float(d[1].sum()) - 1e-9:
                print("FAIL: live entry for n=%d is worse than the committed pack" % n)
                ok = False
    print("live census after call 1: %s" % sorted(_LIVE))
    # PHASE B must actually have run: more draws than arms means the bandit re-allocated clock
    tot = sum(len(v) for v in _QS.values())
    if tot <= len(_QS):
        print("FAIL: bandit never got past forced exploration (%d draws, %d arms)" % (tot, len(_QS)))
        ok = False
    print("draws per arm after call 1: %s" % {k: len(v) for k, v in sorted(_QS.items())})
    # -- HARD FREEZE: a target already AT the cap must get no draws when a loose target is present,
    # -- and must come back when every target is capped (or the call would do nothing at all).
    _QS.clear()
    CPU_BUDGET = 3.0
    mF = _M(500000)
    solve(make(mF), mF, np.random.RandomState(5), [27, 31])
    if 27 in _QS:
        print("FAIL: a capped n was not frozen out of the bandit (%s)" % (sorted(_QS),))
        ok = False
    if 31 not in _QS:
        print("FAIL: the loose n of a mixed target list got no draws (%s)" % (sorted(_QS),))
        ok = False
    _QS.clear()
    mG = _M(500000)
    solve(make(mG), mG, np.random.RandomState(6), [27])
    if 27 not in _QS:
        print("FAIL: an all-capped target list was frozen down to no work at all")
        ok = False
    print("hard freeze: mixed list drew on the loose n only; all-capped list still drew")
    _QS.clear()
    CPU_BUDGET = 6.0
    # -- the round-cost governor must have MEASURED something and never invented a negative cost ---
    if not _RCOST or any((not np.isfinite(v)) or v < 0 for v in _RCOST.values()):
        print("FAIL: round-cost governor recorded nothing / a bad cost: %s" % (_RCOST,))
        ok = False
    print("measured round cost: %s" % {k: round(v, 3) for k, v in sorted(_RCOST.items())})
    # -- CALL 2 in the SAME process (the module-constant-deadline trap) ---------------------------
    before = dict(seen)
    seen.clear()
    m2 = _M(500000)
    solve(make(m2), m2, np.random.RandomState(1), [33])
    if 33 not in seen:
        print("FAIL: the SECOND solve() call in one process returned no feasible packing")
        ok = False
    else:
        f, s2 = _validate(seen[33][1], 33)
        print("call2 n=33 sum_r=%.6f feasible=%s" % (s2, f))
        ok = ok and f
    # -- a single-target call, and an n with NO committed pack (warm-start guard) ------------------
    seen.clear()
    m3 = _M(500000)
    solve(make(m3), m3, np.random.RandomState(2), [12])
    if 12 not in seen:
        print("FAIL: cold start for an n with no committed pack produced nothing")
        ok = False
    else:
        f, s2 = _validate(seen[12][1], 12)
        print("cold n=12 sum_r=%.6f feasible=%s" % (s2, f))
        ok = ok and f
    # -- an exhausted budget must not raise ---------------------------------------------------------
    seen.clear()
    m4 = _M(3)
    solve(make(m4), m4, np.random.RandomState(3), [29])
    print("exhausted-budget call survived; used=%d" % m4.used)
    CPU_BUDGET = saved
    for n, (s, a) in sorted(before.items()):
        f, s2 = _validate(a, n)
        print("call1 n=%d sum_r=%.6f feasible=%s" % (n, s2, f))
        ok = ok and f
    print("SELF-TEST: %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
