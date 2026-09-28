"""Circle-packing census solver -- maximize Sum(r) for n circles in the centered unit square.

Design (iteration 2: SLP trust region + basin hopping; iteration 1 was LP-dual ascent + SLSQP).

  THE KEY IDENTITY.  The non-overlap constraint |c_i - c_j| >= r_i + r_j is *reverse convex*, which is
  what makes the problem hard.  But the Euclidean norm is convex, so its linearization about the
  current centres is a GLOBAL UNDER-ESTIMATOR:

        u_ij . (c_i - c_j)  <=  |c_i - c_j|      for any unit u_ij      (Cauchy-Schwarz)

  so imposing  u_ij . (c_i - c_j) >= r_i + r_j  is an exact *inner* restriction: any point satisfying
  it is feasible for the TRUE problem.  With u_ij frozen at the current contact directions, the whole
  problem -- centres AND radii jointly -- becomes ONE LINEAR PROGRAM over z = (x, y, r).  The current
  point is always LP-feasible, so the LP value is monotone non-decreasing.  This replaces both of
  iteration 1's engines: it moves centres and radii together (the LP-dual ascent moved only centres,
  with radii re-solved after) and it costs ~20 ms at n=99 where SLSQP cost ~0.5 s/iteration and the
  cost model refused to run it at all for n >= 85.

  TWO CORRECTIONS make it actually converge rather than jam:
    (a) *Curvature allowance.*  At a live contact the linearization is tangent, so a tangential slide
        looks free-to-first-order and the LP sees no gain -- it stalls.  The true distance grows like
        t^2/(2 d) under a tangential slide of t, so each pair row is relaxed by kappa*(2*delta)^2/(2 d)
        with delta the trust radius.  This is a *relaxation*, so the LP point may be slightly
        infeasible -- which is fine, because of (b).
    (b) *Exact merit.*  The LP's (x, y) are only a proposal.  The radii are re-derived at the proposed
        centres by the exact fixed-centre LP (_lp_full: max sum r s.t. r_i + r_j <= d_ij over ALL
        pairs), then _feasify'd.  So the accept/reject test is on the TRUE objective at a TRUE feasible
        point; a step that overshoots the curvature allowance is simply rejected and delta halves.

  BASIN HOPPING.  With the local solve down to ~0.2 s (n=45) / ~2 s (n=99), the binding difficulty is
  no longer local convergence but the basin: measured, the committed n=63 packing was already an SLP
  fixed point 4.5% below the record.  So the outer loop shakes the incumbent and re-converges, cycling
  four move classes -- jitter-all, teleport-the-smallest-to-random, teleport-the-smallest-into-the-
  largest-hole (a max-min-distance sample: rattlers are where the wasted area is), and swap-biggest-
  with-smallest -- accepting only strict improvements.

  Kept from iteration 1 as a second line of attack: SLSQP on the full (x, y, r) NLP, now used as an
  occasional extra move on the n where its measured cost model says it fits, plus the five diverse
  cold-start generators and the guarded warm start from the committed census.

  Metering: process CPU is the binding meter (~140 of 500,000 evaluations spend 100 s).  The deadline
  is computed at ENTRY so a second solve() call in one process gets its own slice; the per-n split is
  recomputed from the time remaining and weighted by n, so it survives len(targets) == 1.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix, vstack
CPU_BUDGET = 100.0   # per-call CPU seconds for a single target (added for the sweep; original: min(100, 6+2.7*len(targets)))

LO, HI = -0.5, 0.5
FEAS_TOL = 1e-9
RMIN = 1e-7


# ------------------------------------------------------------------ feasibility / scoring utilities
def _pairdist(xy):
    d = np.hypot(xy[:, 0, None] - xy[None, :, 0], xy[:, 1, None] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return d


def _wallcap(xy):
    """Largest radius each centre may take from the four walls."""
    return np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0], xy[:, 1] - LO, HI - xy[:, 1]])


def _feasify(xy, r):
    """Multiplicatively shrink r until the packing is STRICTLY feasible (slacks >= 0, r > 0).

    Violations coming out of SLSQP/LP are ~1e-10, so the multiplicative factor is 1 - O(1e-9) and the
    lost sum is far below the 1e-7 relative-gap cap; using a factor (not a subtraction) can never push
    a small radius to zero."""
    r = np.clip(r, RMIN, 0.5)
    w = _wallcap(xy)
    r = np.minimum(r, np.maximum(w, RMIN))
    d = _pairdist(xy)
    s = r[:, None] + r[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s > 0, d / np.maximum(s, 1e-300), np.inf)
    f = min(1.0, float(np.min(ratio)))
    if f < 1.0:
        r = r * f
    r = r * (1.0 - 1e-12)
    return np.maximum(r, RMIN)


def _pack(xy, r):
    return np.concatenate([xy, r[:, None]], axis=1)


# ------------------------------------------------------------------------------ (1) LP-optimal radii
def _lp_full(xy):
    """Exact best radii for FIXED centres AND the LP duals.

    max sum(r)  s.t.  r_i + r_j <= d_ij (ALL pairs),  0 <= r_i <= wall_i.
    Returns (r, value, lam, mu) where lam[k] >= 0 is the shadow price of pair k's distance and
    mu[i] >= 0 that of circle i's wall cap -- i.e. exactly the sensitivities of the optimal Sum(r) to
    moving the centres, which is what drives the ascent step in _refine().
    A partial pair set was tried first and FAILS: SLSQP/LP grow circles straight through the pairs that
    were left out (measured n=75: raw sum 8.77 vs a true 4.37), so the full O(n^2) set is used -- HiGHS
    solves n=99 (4851 rows) in ~6 ms, which is cheap enough to sit inside the inner loop."""
    n = xy.shape[0]
    w = np.maximum(_wallcap(xy), 0.0)
    d = _pairdist(xy)
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.repeat(np.arange(m), 2)
    cols = np.empty(2 * m, dtype=int)
    cols[0::2], cols[1::2] = ii, jj
    from scipy.sparse import csr_matrix
    A = csr_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n))
    b = d[ii, jj]
    res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=np.column_stack([np.zeros(n), w]),
                  method="highs")
    if res.x is None:
        r = np.minimum(w, d.min(axis=1) / 2.0)
        return r, float(r.sum()), None, None, (ii, jj)
    r = np.asarray(res.x, dtype=float)
    lam = mu = None
    try:
        lam = -np.asarray(res.ineqlin.marginals, dtype=float)
        mu = -np.asarray(res.upper.marginals, dtype=float)
    except Exception:
        pass
    return r, float(r.sum()), lam, mu, (ii, jj)


def _lp_grad(xy, lam, mu, pairs):
    """d(LP value)/d(centres) by the envelope theorem: each active pair pushes its two circles apart
    with force lam, each wall-capped circle is pulled off its nearest wall with force mu."""
    n = xy.shape[0]
    ii, jj = pairs
    g = np.zeros((n, 2))
    if lam is not None and len(ii):
        dx = xy[ii, 0] - xy[jj, 0]
        dy = xy[ii, 1] - xy[jj, 1]
        dist = np.maximum(np.hypot(dx, dy), 1e-12)
        ux, uy = lam * dx / dist, lam * dy / dist
        np.add.at(g, (ii, 0), ux); np.add.at(g, (jj, 0), -ux)
        np.add.at(g, (ii, 1), uy); np.add.at(g, (jj, 1), -uy)
    if mu is not None:
        walls = np.stack([xy[:, 0] - LO, HI - xy[:, 0], xy[:, 1] - LO, HI - xy[:, 1]], axis=1)
        k = np.argmin(walls, axis=1)
        gx = np.where(k == 0, 1.0, np.where(k == 1, -1.0, 0.0))
        gy = np.where(k == 2, 1.0, np.where(k == 3, -1.0, 0.0))
        g[:, 0] += mu * gx
        g[:, 1] += mu * gy
    return g


# --------------------------------------------------------------------------------- (2) NLP polish
def _neighbours(xy, r, slack=0.06):
    """Pairs whose constraint is active or could become active: d_ij < r_i + r_j + margin."""
    n = xy.shape[0]
    d = _pairdist(xy)
    margin = slack + 0.75 * (r[:, None] + r[None, :])
    ii, jj = np.where(np.triu(d < (r[:, None] + r[None, :]) + margin, 1))
    return ii, jj


def _slsqp(xy0, r0, neigh, maxiter=60):
    """One SLSQP pass over z = [x, y, r] maximizing sum(r) with the given pair constraint subset."""
    n = xy0.shape[0]
    ii, jj = neigh
    z0 = np.concatenate([xy0[:, 0], xy0[:, 1], r0])

    obj_grad = np.concatenate([np.zeros(2 * n), -np.ones(n)])

    def f(z):
        return -float(z[2 * n:].sum())

    def fp(z):
        return obj_grad

    # wall constraints (linear): x - r + 0.5 >= 0, 0.5 - x - r >= 0, same in y
    Aw = np.zeros((4 * n, 3 * n))
    idx = np.arange(n)
    Aw[idx, idx] = 1.0;            Aw[idx, 2 * n + idx] = -1.0            # x - r >= -0.5
    Aw[n + idx, idx] = -1.0;       Aw[n + idx, 2 * n + idx] = -1.0        # -x - r >= -0.5
    Aw[2 * n + idx, n + idx] = 1.0;  Aw[2 * n + idx, 2 * n + idx] = -1.0
    Aw[3 * n + idx, n + idx] = -1.0; Aw[3 * n + idx, 2 * n + idx] = -1.0
    bw = np.full(4 * n, 0.5)

    def cw(z):
        return Aw @ z + bw

    def cwp(z):
        return Aw

    m = len(ii)

    def cp(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dx, dy = x[ii] - x[jj], y[ii] - y[jj]
        return np.hypot(dx, dy) - r[ii] - r[jj]

    def cpp(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dx, dy = x[ii] - x[jj], y[ii] - y[jj]
        dist = np.maximum(np.hypot(dx, dy), 1e-12)
        J = np.zeros((m, 3 * n))
        rows = np.arange(m)
        ux, uy = dx / dist, dy / dist
        np.add.at(J, (rows, ii), ux)
        np.add.at(J, (rows, jj), -ux)
        np.add.at(J, (rows, n + ii), uy)
        np.add.at(J, (rows, n + jj), -uy)
        np.add.at(J, (rows, 2 * n + ii), -1.0)
        np.add.at(J, (rows, 2 * n + jj), -1.0)
        return J

    cons = [{"type": "ineq", "fun": cw, "jac": cwp}]
    if m:
        cons.append({"type": "ineq", "fun": cp, "jac": cpp})
    bounds = [(LO, HI)] * (2 * n) + [(RMIN, 0.5)] * n
    try:
        res = minimize(f, z0, jac=fp, bounds=bounds, constraints=cons, method="SLSQP",
                       options={"maxiter": maxiter, "ftol": 1e-12})
        z = res.x
    except Exception:
        z = z0
    if not np.all(np.isfinite(z)):
        z = z0
    xy = np.column_stack([np.clip(z[:n], LO, HI), np.clip(z[n:2 * n], LO, HI)])
    return xy, np.clip(z[2 * n:], RMIN, 0.5)


def _refine(xy, r, deadline, rng=None):
    """LP-dual gradient ascent on the centres, radii re-solved EXACTLY by LP at every step.

    Every iterate is feasible by construction (LP radii + _feasify), so the incumbent can only improve.
    Step is adaptive: grow on an accepted step, halve on a rejected one, stop when it underflows."""
    xy = np.clip(xy.copy(), LO + 1e-9, HI - 1e-9)
    r, val, lam, mu, pairs = _lp_full(xy)
    best_xy, best_r = xy.copy(), _feasify(xy, r)
    best_s = float(best_r.sum())
    step = 0.02
    while time.process_time() < deadline and step > 1e-9:
        g = _lp_grad(xy, lam, mu, pairs)
        nrm = np.linalg.norm(g, axis=1).max()
        if not np.isfinite(nrm) or nrm <= 0:
            break
        cand = np.clip(xy + (step / nrm) * g, LO + 1e-9, HI - 1e-9)
        r2, val2, lam2, mu2, pairs2 = _lp_full(cand)
        if val2 > val:
            xy, r, val, lam, mu, pairs = cand, r2, val2, lam2, mu2, pairs2
            step *= 1.3
            rr = _feasify(xy, r)
            s = float(rr.sum())
            if s > best_s:
                best_xy, best_r, best_s = xy.copy(), rr, s
        else:
            step *= 0.45
    return best_xy, best_r, best_s


def _slsqp_cost(n, maxiter):
    """Measured cost model for one SLSQP pass with the FULL pair set (fit to 0.035 s/iter at n=45,
    0.137 at n=75, 0.46 at n=99): ~1.4e-7 * n^3.27 per iteration. Used to size maxiter so the polish
    fits inside the n's CPU slice instead of blowing the 120 s process backstop."""
    return 1.4e-7 * (n ** 3.27) * maxiter


def _polish(xy, r, best_s, deadline, n):
    """SLSQP on the full (x, y, r) NLP, started from the ascent's output.

    The ascent is cheap but stalls at a nonsmooth LP vertex (~1 digit); SLSQP converges harder but one
    pass costs seconds at large n and can diverge, so maxiter is sized from the cost model above and
    the result is only ACCEPTED when it beats the incumbent (divergence is then free)."""
    room = deadline - time.process_time()
    if room < 0.15:
        return None
    maxiter = int(np.clip(room * 0.8 / max(_slsqp_cost(n, 1), 1e-9), 5, 60))
    if _slsqp_cost(n, maxiter) > room:
        return None
    xy2, r2 = _slsqp(xy, r, np.triu_indices(n, 1), maxiter=maxiter)
    rr = _feasify(xy2, r2)
    s = float(rr.sum())
    if s <= best_s:
        return None
    # the LP re-seats the radii exactly for the polished centres, which SLSQP's iterate rarely does
    try:
        rl, _, _, _, _ = _lp_full(xy2)
        rr2 = _feasify(xy2, rl)
        if float(rr2.sum()) > s:
            rr, s = rr2, float(rr2.sum())
    except Exception:
        pass
    return (xy2, rr, s)



# ----------------------------------------------------------- (3) SLP trust region (the main engine)
def _slp_static(n):
    """The n-dependent parts of the SLP that never change: the 4n wall rows and the pair-row index
    pattern (6 nonzeros per pair: x_i, x_j, y_i, y_j, r_i, r_j)."""
    idx = np.arange(n)
    wr = np.concatenate([np.arange(4 * n), np.arange(4 * n)])
    wc = np.concatenate([idx, idx, n + idx, n + idx, 2 * n + idx, 2 * n + idx, 2 * n + idx, 2 * n + idx])
    wv = np.concatenate([-np.ones(n), np.ones(n), -np.ones(n), np.ones(n), np.ones(4 * n)])
    Wm = csr_matrix((wv, (wr, wc)), shape=(4 * n, 3 * n))       # r_i -+ x_i <= 0.5, r_i -+ y_i <= 0.5
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    prow = np.repeat(np.arange(m), 6)
    pcol = np.empty(6 * m, dtype=int)
    pcol[0::6], pcol[1::6] = ii, jj
    pcol[2::6], pcol[3::6] = n + ii, n + jj
    pcol[4::6], pcol[5::6] = 2 * n + ii, 2 * n + jj
    return Wm, np.full(4 * n, 0.5), ii, jj, prow, pcol


_SLP_CACHE = {}


def _slp(xy, r, deadline, delta0=0.03, kappa=1.0, static=None):
    """Trust-region SLP on the JOINT variable z = (x, y, r).

    Each iteration freezes the contact directions u_ij, solves the LP
        max sum(r)  s.t.  u_ij.(c_i - c_j) >= r_i + r_j - kappa*(2*delta)^2/(2 d_ij),
                          r_i -+ x_i <= 0.5, r_i -+ y_i <= 0.5,  |c - c0|_inf <= delta,
    then judges the proposal by the TRUE objective: exact fixed-centre radii (_lp_full) at the
    proposed centres, made strictly feasible.  Accept -> grow delta 1.4x; reject -> halve it.
    Returns the best strictly feasible (xy, r, sum) seen -- never worse than the input."""
    n = xy.shape[0]
    if static is None:
        static = _SLP_CACHE.setdefault(n, _slp_static(n))
    Wm, bw, ii, jj, prow, pcol = static
    m = len(ii)
    c = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    xy = np.clip(np.asarray(xy, float), LO, HI)
    rr = _feasify(xy, r)
    best_xy, best_r, best = xy.copy(), rr, float(rr.sum())
    delta = delta0
    while time.process_time() < deadline and delta > 1e-10:
        dx, dy = xy[ii, 0] - xy[jj, 0], xy[ii, 1] - xy[jj, 1]
        dist = np.maximum(np.hypot(dx, dy), 1e-12)
        ux, uy = dx / dist, dy / dist
        pv = np.empty(6 * m)
        pv[0::6], pv[1::6] = -ux, ux
        pv[2::6], pv[3::6] = -uy, uy
        pv[4::6] = pv[5::6] = 1.0
        A = vstack([csr_matrix((pv, (prow, pcol)), shape=(m, 3 * n)), Wm]).tocsr()
        b = np.concatenate([kappa * (2.0 * delta) ** 2 / (2.0 * dist), bw])
        lb = np.concatenate([np.maximum(LO, xy[:, 0] - delta), np.maximum(LO, xy[:, 1] - delta),
                             np.zeros(n)])
        ub = np.concatenate([np.minimum(HI, xy[:, 0] + delta), np.minimum(HI, xy[:, 1] + delta),
                             np.full(n, 0.5)])
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=np.column_stack([lb, ub]), method="highs")
        except Exception:
            break
        if res.x is None or not np.all(np.isfinite(res.x)):
            delta *= 0.5
            continue
        z = res.x
        cxy = np.column_stack([np.clip(z[:n], LO, HI), np.clip(z[n:2 * n], LO, HI)])
        rl, _, _, _, _ = _lp_full(cxy)                  # exact radii for the PROPOSED centres
        cr = _feasify(cxy, rl)
        s = float(cr.sum())
        if s > best + 1e-13:
            best_xy, best_r, best = cxy.copy(), cr, s
            xy = cxy
            delta = min(delta * 1.4, 0.15)
        else:
            delta *= 0.5
    return best_xy, best_r, best



# ------------------------------------------------- (3b) primal log-barrier Newton (iteration 3 core)
def _bar_set(xy, r, tau):
    """Pairs close enough to matter for the barrier: d_ij < r_i + r_j + tau.  Everything else is kept
    feasible by the step-length cap instead of by a log term, which is what makes the Hessian O(n)
    blocks instead of O(n^2)."""
    n = len(r)
    d = _pairdist(xy)
    s = r[:, None] + r[None, :] + tau
    ii, jj = np.nonzero(np.triu(d < s, 1))
    if len(ii) < n:                      # degenerate/spread layout: fall back to the k nearest
        k = min(n - 1, 8)
        nb = np.argsort(d, axis=1)[:, :k]
        a = np.repeat(np.arange(n), k)
        b = nb.ravel()
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = np.unique(lo * n + hi)
        ii, jj = key // n, key % n
    return ii.astype(np.intp), jj.astype(np.intp)


def _bar_newton(xy, r, deadline, mu0=3e-3, mu_min=1e-12, tau=0.06, wt=None, ret_last=False,
                free=None):
    """Maximize sum(r) by a PRIMAL LOG-BARRIER NEWTON method on z = (x, y, r) in R^{3n}.

        min  -sum(r) - mu * [ sum_k log g_k ]
        g:   x_i +- r_i walls (4n),  r_i > 0 (n),  d_ij - r_i - r_j > 0 (near pairs only)

    Why this replaces the SLP as the core engine: the SLP needed ONE HiGHS solve per step over O(n^2)
    rows and converged linearly at best, so a local solve cost 0.2 s (n=45) to 2 s (n=99).  Here the
    Hessian is analytic and its barrier set is O(n) blocks, so a step is one 3n x 3n Cholesky --
    ~1 ms at n=99 -- and Newton with the true Lagrangian curvature term (-mu/g * grad^2 g, the
    (I-uu^T)/d tangential-stiffness block) converges quadratically inside a basin.  Cheap AND deep:
    the barrier drives the KKT residual to ~mu, so the returned point is a local optimum to many more
    digits than the SLP's trust-region fixed point, and it leaves 10-50x more clock for basin hopping.

    The pairs OUTSIDE the barrier set have no log term; they are protected by the fraction-to-boundary
    step cap, which checks ALL pairs.  So the iterate is feasible for the true problem throughout.
    `wt` reweights the linear objective to  min -sum(wt_i r_i) - mu*sum(log g)  -- the homotopy move
    class of iteration 4.  With wt != 1 the SAME feasible set has a DIFFERENT optimum, so converging
    under a random wt and then re-converging under wt == 1 walks the packing through a genuinely
    different contact topology; a random jitter cannot do that because it is re-absorbed by the same
    basin.  `ret_last` returns the final iterate rather than the best-true-sum one, which is what a
    homotopy leg must hand to its unweighted leg.
    Returns (xy, r, sum) -- with ret_last=False, never worse than the input."""
    n = xy.shape[0]
    wt = np.ones(n) if wt is None else np.asarray(wt, float)
    N = 3 * n
    idx = np.arange(n)
    IX, IY, IR = idx, n + idx, 2 * n + idx
    # ---- FROZEN EXTERIOR (iteration 10).  `free` = the circle indices allowed to move; every other
    # circle is held EXACTLY where it is, so the Newton system shrinks from 3n to 3|free| and the
    # barrier only carries the constraints that a free circle can actually violate.
    if free is None:
        fmask = np.ones(n, bool)
        fi = idx
        F = np.arange(N)
        restricted = False
    else:
        fmask = np.zeros(n, bool)
        fmask[np.asarray(free, dtype=np.intp)] = True
        fi = idx[fmask]
        F = np.concatenate([fi, n + fi, 2 * n + fi])
        restricted = True
    nF = len(F)
    wall_live = np.tile(fmask, 4)          # wall slacks of frozen circles are constants -> no log

    xy = np.clip(np.asarray(xy, float), LO, HI)
    r = _feasify(xy, r)
    best_xy, best_r, best = xy.copy(), r.copy(), float(r.sum())
    last_xy = last_r = None
    last_s = -np.inf

    z = np.concatenate([xy[:, 0], xy[:, 1], r])
    # push strictly inside: the barrier needs g > 0, and _feasify leaves slacks at ~0.  Only the FREE
    # radii are nudged -- shrinking a frozen one would silently cost 1e-7 of the sum it is holding.
    z[2 * n + idx[fmask]] *= 1.0 - 1e-7
    z[IR] = np.maximum(z[IR], RMIN)

    ii, jj = _bar_set(xy, r, tau)
    if restricted:
        sel = fmask[ii] | fmask[jj]        # a frozen-frozen pair cannot change: drop its log term
        ii, jj = ii[sel], jj[sel]
    mu = mu0
    it_since_set = 0

    def slacks(z, ii, jj):
        x, y, rr = z[IX], z[IY], z[IR]
        gw = np.concatenate([x - LO - rr, HI - x - rr, y - LO - rr, HI - y - rr])
        if restricted:
            gw = np.where(wall_live, gw, 1.0)
        dx, dy = x[ii] - x[jj], y[ii] - y[jj]
        dist = np.hypot(dx, dy)
        gp = dist - rr[ii] - rr[jj]
        return gw, rr.copy(), gp, dx, dy, dist

    def allslack(z):
        x, y, rr = z[IX], z[IY], z[IR]
        if restricted:
            # Only a FREE circle can break anything, so the O(n^2) all-pairs sweep that the
            # fraction-to-boundary test runs on every backtrack becomes an O(k*n) one.  This is the
            # step that actually makes a patch solve cheap: at n=99, k=14 it is a 7x saving on the
            # dominant cost of the whole Newton.
            fx, fy, fr = x[fi], y[fi], rr[fi]
            dd = (np.hypot(fx[:, None] - x[None, :], fy[:, None] - y[None, :])
                  - (fr[:, None] + rr[None, :]))
            dd[np.arange(len(fi)), fi] = np.inf                 # a circle against itself
            ws = np.concatenate([fx - LO - fr, HI - fx - fr, fy - LO - fr, HI - fy - fr])
            return float(dd.min()), float(min(ws.min(), rr.min()))
        d = _pairdist(np.column_stack([x, y]))
        ps = d - (rr[:, None] + rr[None, :])
        ws = np.concatenate([x - LO - rr, HI - x - rr, y - LO - rr, HI - y - rr])
        return float(np.min(ps)), float(np.min(np.concatenate([ws, rr])))

    def phi(z, ii, jj, mu):
        gw, gr, gp, _, _, _ = slacks(z, ii, jj)
        if gw.min() <= 0 or gr.min() <= 0 or (len(gp) and gp.min() <= 0):
            return np.inf
        v = -float(wt @ z[IR]) - mu * (np.log(gw).sum() + np.log(gr).sum())
        if len(gp):
            v -= mu * np.log(gp).sum()
        return float(v)

    while mu > mu_min and time.process_time() < deadline:
        for _ in range(30):
            if time.process_time() >= deadline:
                break
            gw, gr, gp, dx, dy, dist = slacks(z, ii, jj)
            if gw.min() <= 0 or gr.min() <= 0 or (len(gp) and gp.min() <= 0):
                break
            # ---- gradient
            g = np.zeros(N)
            g[IR] -= wt
            iw = mu / gw
            g[IX] += -iw[0:n] + iw[n:2 * n]                     # d/dx of -mu log(x-LO-r), -mu log(HI-x-r)
            g[IY] += -iw[2 * n:3 * n] + iw[3 * n:4 * n]
            g[IR] += iw[0:n] + iw[n:2 * n] + iw[2 * n:3 * n] + iw[3 * n:4 * n]
            g[IR] += -mu / gr
            # ---- Hessian (dense 3n x 3n; barrier set keeps the pair part O(n) blocks)
            H = np.zeros((N, N))
            Hf = H.ravel()
            w = mu / gw ** 2
            wx0, wx1, wy0, wy1 = w[0:n], w[n:2 * n], w[2 * n:3 * n], w[3 * n:4 * n]
            H[IX, IX] += wx0 + wx1
            H[IY, IY] += wy0 + wy1
            H[IR, IR] += wx0 + wx1 + wy0 + wy1 + mu / gr ** 2
            H[IX, IR] += -wx0 + wx1
            H[IR, IX] += -wx0 + wx1
            H[IY, IR] += -wy0 + wy1
            H[IR, IY] += -wy0 + wy1
            if len(ii):
                dd = np.maximum(dist, 1e-12)
                ux, uy = dx / dd, dy / dd
                one = np.ones(len(ii))
                G = np.column_stack([ux, uy, -ux, -uy, -one, -one])   # grad of g_ij
                cols = np.column_stack([ii, n + ii, jj, n + jj, 2 * n + ii, 2 * n + jj])
                wp = mu / gp ** 2
                gg = (wp[:, None, None] * G[:, :, None] * G[:, None, :]).reshape(-1)
                fl = (cols[:, :, None] * N + cols[:, None, :]).reshape(-1)
                np.add.at(Hf, fl, gg)
                np.add.at(g, cols.ravel(), (-(mu / gp)[:, None] * G).ravel())
                # true Lagrangian curvature:  -mu/g * grad^2 g,  grad^2 d = [[P,-P],[-P,P]]/d
                a = mu / (gp * dd)
                pxx, pxy, pyy = 1.0 - ux * ux, -ux * uy, 1.0 - uy * uy
                cc = np.column_stack([ii, n + ii, jj, n + jj])
                blk = np.empty((len(ii), 4, 4))
                blk[:, 0, 0] = blk[:, 2, 2] = -a * pxx
                blk[:, 1, 1] = blk[:, 3, 3] = -a * pyy
                blk[:, 0, 1] = blk[:, 1, 0] = blk[:, 2, 3] = blk[:, 3, 2] = -a * pxy
                blk[:, 0, 2] = blk[:, 2, 0] = a * pxx
                blk[:, 1, 3] = blk[:, 3, 1] = a * pyy
                blk[:, 0, 3] = blk[:, 3, 0] = blk[:, 1, 2] = blk[:, 2, 1] = a * pxy
                fl2 = (cc[:, :, None] * N + cc[:, None, :]).reshape(-1)
                np.add.at(Hf, fl2, blk.reshape(-1))
            # ---- modified Newton: ridge until Cholesky succeeds
            dz = None
            HR = H if not restricted else H[np.ix_(F, F)]
            gR = g if not restricted else g[F]
            ridge = 0.0
            base = max(1e-12, 1e-9 * np.trace(HR) / nF)
            for _ in range(12):
                try:
                    cf = cho_factor(HR + ridge * np.eye(nF), lower=True, check_finite=False)
                    dzR = cho_solve(cf, -gR, check_finite=False)
                    dz = dzR
                    break
                except Exception:
                    ridge = base if ridge == 0.0 else ridge * 12.0
            if dz is None or not np.all(np.isfinite(dz)):
                break
            if float(dz @ gR) > 0:                # not a descent direction -> steepest descent
                dz = -gR
            slope_R = float(gR @ dz)
            if restricted:
                full = np.zeros(N)
                full[F] = dz
                dz = full
            # ---- fraction-to-boundary + backtracking on the barrier merit
            f0 = phi(z, ii, jj, mu)
            slope = slope_R
            step = 1.0
            nrm = float(np.abs(dz).max())
            if nrm > 0.25:
                step = 0.25 / nrm
            ok = False
            for _bt in range(40):
                zn = z + step * dz
                zn[IX] = np.clip(zn[IX], LO, HI)
                zn[IY] = np.clip(zn[IY], LO, HI)
                zn[IR] = np.maximum(zn[IR], 1e-14)
                ps, ws = allslack(zn)             # ALL pairs + walls, not just the barrier set
                if ps > 1e-13 and ws > 1e-14:
                    fn = phi(zn, ii, jj, mu)
                    if fn < f0 + 1e-4 * step * slope:
                        z, ok = zn, True
                        break
                step *= 0.5
            if not ok:
                break
            it_since_set += 1
            if it_since_set >= 8:                 # refresh which pairs are near-active
                ii, jj = _bar_set(np.column_stack([z[IX], z[IY]]), z[IR], tau)
                if restricted:
                    sel = fmask[ii] | fmask[jj]
                    ii, jj = ii[sel], jj[sel]
                it_since_set = 0
            if abs(slope) * step < 1e-14:
                break
        cxy = np.column_stack([z[IX], z[IY]])
        cr = _feasify(cxy, z[IR])
        s = float(cr.sum())
        last_xy, last_r, last_s = cxy, cr, s
        if s > best + 1e-15:
            best_xy, best_r, best = cxy.copy(), cr.copy(), s
        mu *= 0.12
        ii, jj = _bar_set(np.column_stack([z[IX], z[IY]]), z[IR], tau)
        if restricted:
            sel = fmask[ii] | fmask[jj]
            ii, jj = ii[sel], jj[sel]
        it_since_set = 0
    if ret_last and last_xy is not None:
        return last_xy, last_r, last_s
    return best_xy, best_r, best


# --------------------------------------------------------------------------- (4) basin-hopping moves
def _shake(xy, r, rng, mode):
    """One perturbation of a converged packing.  The four classes attack different stagnations:
    0 jitter-all (wrong fine arrangement), 1 teleport the smallest circles at random (rattlers stuck
    in a bad cell), 2 teleport them into the largest hole found by a max-min-distance sample (the
    wasted area is where a rattler pays), 3 (RETIRED -- see solve(): swapping two rows leaves the set of
    centres identical, so with free radii it was a geometric no-op)."""
    n = len(r)
    xy = xy.copy()
    if mode == 0:
        return np.clip(xy + rng.normal(0, 0.4 * float(np.median(r)), size=(n, 2)), LO + 1e-3, HI - 1e-3)
    if mode == 1:
        k = max(1, int(0.08 * n))
        idx = np.argsort(r)[:k]
        xy[idx] = rng.uniform(LO + 0.01, HI - 0.01, size=(k, 2))
        return xy
    if mode == 2:
        k = max(1, int(0.06 * n))
        for i in np.argsort(r)[:k]:
            cand = rng.uniform(LO + 0.01, HI - 0.01, size=(160, 2))
            oth = np.delete(np.arange(n), i)
            d = (np.hypot(cand[:, 0, None] - xy[None, oth, 0], cand[:, 1, None] - xy[None, oth, 1])
                 - r[None, oth])
            wall = np.minimum.reduce([cand[:, 0] - LO, HI - cand[:, 0], cand[:, 1] - LO, HI - cand[:, 1]])
            xy[i] = cand[int(np.argmax(np.minimum(d.min(axis=1), wall)))]
        return xy
    o = np.argsort(r)
    a, b = o[0], o[-1]
    xy[[a, b]] = xy[[b, a]]
    return xy

# ------------------------------------------------------- (4b) POPULATION move: spatial crossover
def _hole_place(xy, i, rng, m=96):
    """Move circle i to the emptiest spot (max-min distance over m random probes)."""
    n = len(xy)
    cand = rng.uniform(LO + 0.01, HI - 0.01, size=(m, 2))
    oth = np.delete(np.arange(n), i)
    d = np.hypot(cand[:, 0, None] - xy[None, oth, 0], cand[:, 1, None] - xy[None, oth, 1])
    wall = np.minimum.reduce([cand[:, 0] - LO, HI - cand[:, 0], cand[:, 1] - LO, HI - cand[:, 1]])
    xy[i] = cand[int(np.argmax(np.minimum(d.min(axis=1), wall)))]
    return xy


# ------------------------------------------------- (4c) MESO move: ruin-and-recreate a sub-region
def _ruin_pick(xy, r, rng, lo=0.08, hi=0.22, kmax=16):
    """Choose a SPATIALLY CONTIGUOUS patch of circles to demolish: a ball around a small circle, a
    strip along one wall, or a corner quadrant.  Split out of _ruin (iteration 10) so the nested
    patch optimiser can reuse exactly the same patch geometry."""
    n = len(r)
    k = int(np.clip(round(n * rng.uniform(lo, hi)), 4, kmax))
    k = min(k, n - 1)
    mode = int(rng.randint(3))
    if mode == 0:                                  # a ball: a small circle and its nearest neighbours
        wsel = 1.0 / (r + 1e-9)
        i = int(rng.choice(n, p=wsel / wsel.sum()))
        key = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
    elif mode == 1:                                # a strip along one wall (attacks a wall-locked row)
        ax = int(rng.randint(2))
        key = (xy[:, ax] - LO) if rng.rand() < 0.5 else (HI - xy[:, ax])
    else:                                          # a corner quadrant (attacks the corner motif)
        cx = HI if rng.rand() < 0.5 else LO
        cy = HI if rng.rand() < 0.5 else LO
        key = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    return np.argsort(key)[:k]


def _refill(kxy, kr, k, rng, greed, fill):
    """Drop k circles back, one at a time, into the emptiest spot of the CURRENT void, each sized by
    the room actually found there.  Survivors (kxy, kr) come back untouched and FIRST, so the
    newcomers are exactly the last k rows -- which is what lets a caller freeze the survivors."""
    cxy, cr = kxy.copy(), kr.copy()
    for _ in range(k):
        cand = rng.uniform(LO + 0.004, HI - 0.004, size=(256, 2))
        d = (np.hypot(cand[:, 0, None] - cxy[None, :, 0], cand[:, 1, None] - cxy[None, :, 1])
             - cr[None, :])
        wall = np.minimum.reduce([cand[:, 0] - LO, HI - cand[:, 0],
                                  cand[:, 1] - LO, HI - cand[:, 1]])
        room = np.minimum(d.min(axis=1), wall)
        if rng.rand() < greed:
            j = int(np.argmax(room))
        else:
            j = int(rng.choice(np.argsort(-room)[:6]))
        cxy = np.vstack([cxy, cand[j][None, :]])
        cr = np.append(cr, max(RMIN, fill * float(room[j])))
    return cxy, cr


def _ruin(xy, r, rng):
    """RUIN A CONTIGUOUS SUB-REGION AND RECREATE IT FROM SCRATCH -- the only move at the MESO scale.

    Every other move in the mixture acts at one of two extremes.  Micro: jitter, teleport the k
    smallest, reweight -- these keep the contact graph and only slide it.  Macro: crossover, migration,
    the affine container map -- these rewrite the whole arrangement at once.  Nothing acted on a
    HANDFUL OF ADJACENT circles, and that is exactly the scale a wrong basin lives at: the 16 arms
    still 1e-4 away are ~99% correct with one local motif (a square cell where a hexagonal one fits,
    a wall row one circle too long) frozen in by contacts no continuous move can break.

    Ruin removes a spatially contiguous cluster -- a ball around a small circle, a strip along one
    wall, or a corner quadrant -- and RECREATES it by dropping the members back one at a time into the
    emptiest spot of the void they left, sized by the room actually found there.  The survivors are
    untouched, so 85-92% of a good arrangement is preserved while the ruined patch is free to come
    back with a DIFFERENT local topology.  Probes are picked greedily 70% of the time and from the top
    six otherwise (and with a per-ruin greed/ambition draw), so two ruins of the same patch do not
    recreate the same thing."""
    n = len(r)
    xy, r = np.asarray(xy, float).copy(), np.asarray(r, float).copy()
    idx = _ruin_pick(xy, r, rng)
    k = len(idx)
    greed = float(rng.uniform(0.35, 0.85))     # how deterministically the void is refilled
    fill = float(rng.uniform(0.70, 0.95))      # how ambitious each newcomer is about its room
    keep = np.setdiff1d(np.arange(n), idx)
    cxy, cr = _refill(xy[keep], r[keep] * 0.995, k, rng, greed, fill)
    return cxy, _feasify(cxy, np.minimum(cr, _wallcap(cxy)))


# --------------------------------- (4e) GLOBAL COMBINATORIAL move: wall-row RECOUNT (iteration 13)
TOUCH_TOL = 0.06         # a circle is "in the wall row" if its wall slack is under this * mean(r)


def _wallrow(xy, r, rng):
    """RECOUNT ONE WALL'S CONTACT ROW -- the k circles pinned against one wall come back as k+-1, at
    FIXED n.  The only move in the mixture that changes the packing's BOUNDARY COMBINATORICS.

    Iteration 12's post-mortem named this gap precisely.  With 33 of 37 arms at the 7-digit cap, the
    four survivors (81, 73, 41, 31) resisted BOTH allocators -- i.e. they are not precision problems
    and not clock-starvation problems, they are basins.  And every move that exists conserves the
    thing that defines a basin at the boundary, namely HOW MANY circles a wall carries: jitter and the
    homotopies slide contacts, `_affine` is a continuous map (a diffeomorphism cannot change a count),
    `_cross`/`_migrate` only recombine rows that already exist, and `_ruin`/`_patch` demolish a
    contiguous blob of 4-16 circles and refill it by greedy probing -- which reproduces the incumbent
    row count almost every time, because the void it leaves is exactly the shape the old row made.

    A wall row of k equal circles is a rigid, self-locking motif: to go from k to k+1 every one of
    them must shrink by ~k/(k+1) SIMULTANEOUSLY, and any move that shrinks them one at a time is
    rejected long before it pays.  So this move does the whole substitution in one step:

      * pick a wall, collect the circles touching it (slack <= TOUCH_TOL * mean r) -- that is k;
      * rebuild that row as k+-1 EVENLY SPACED circles (over the whole wall, or over the old row's
        own extent -- both, at random, since a partial row and a full row want different spans);
      * conserve n by trading with the interior: k+1 takes the smallest interior circle INTO the row,
        k-1 hands its spare circle to the emptiest hole in the packing (`_refill`, one body);
      * choose the row's stand-off from the wall by a 1-D scan, so the new row is sized by the room
        the interior actually leaves it rather than by an assumption -- no global `_feasify` crush.

    Returns None when no wall carries a usable row (the caller then falls back to `_ruin`)."""
    n = len(r)
    xy = np.asarray(xy, float).copy()
    r = np.asarray(r, float).copy()
    tol = TOUCH_TOL * float(np.mean(r))
    row, ax, side = None, 0, 0
    for _try in range(6):
        wsel = int(rng.randint(4))
        ax, side = wsel // 2, wsel % 2
        slack = (xy[:, ax] - LO - r) if side == 0 else (HI - xy[:, ax] - r)
        cand = np.where(slack <= tol)[0]
        if 2 <= len(cand) <= n - 3:
            row = cand
            break
    if row is None:
        return None
    k = len(row)
    delta = 1 if rng.rand() < 0.5 else -1
    if not (2 <= k + delta <= n - 2):
        delta = -delta
    k2 = k + delta
    tan = 1 - ax
    wall = LO if side == 0 else HI
    sgn = 1.0 if side == 0 else -1.0

    keep = np.setdiff1d(np.arange(n), row)
    kxy, kr = xy[keep].copy(), r[keep] * 0.995
    if delta > 0 and len(kr) > 1:
        j = int(np.argmin(kr))               # the extra body comes from the interior's smallest
        kxy = np.delete(kxy, j, axis=0)
        kr = np.delete(kr, j)

    # the span the new row occupies: the whole wall, or just the old row's own extent
    if rng.rand() < 0.5:
        a, b = LO, HI
    else:
        a = max(LO, float(np.min(xy[row, tan] - r[row])))
        b = min(HI, float(np.max(xy[row, tan] + r[row])))
        if b - a < 2.0 / n:
            a, b = LO, HI
    step = (b - a) / k2
    t = a + (np.arange(k2) + 0.5) * step
    rho0 = min(0.5 * step, float(np.max(r[row])) * 1.25)

    # 1-D scan over the row's stand-off from the wall: each candidate stand-off gives every new
    # circle the radius the interior actually leaves it, and we keep the stand-off with the best sum.
    best = None
    for f in np.linspace(0.28, 1.0, 13):
        rho = rho0 * f
        nxy = np.empty((k2, 2))
        nxy[:, tan] = t
        nxy[:, ax] = wall + sgn * rho
        nr = np.full(k2, rho)
        if len(kr):
            d = (np.hypot(nxy[:, 0, None] - kxy[None, :, 0], nxy[:, 1, None] - kxy[None, :, 1])
                 - kr[None, :])
            nr = np.minimum(nr, d.min(axis=1))
        nr = np.minimum(nr, _wallcap(nxy))
        nr = np.maximum(nr, RMIN)
        tot = float(nr.sum())
        if best is None or tot > best[0]:
            best = (tot, nxy, nr)
    _tot, nxy, nr = best

    cxy = np.vstack([kxy, nxy])
    cr = np.concatenate([kr, nr])
    if delta < 0:                            # k-1 in the row: the spare body goes to the biggest hole
        cxy, cr = _refill(cxy, cr, 1, rng, float(rng.uniform(0.5, 0.95)),
                          float(rng.uniform(0.70, 0.95)))
    if len(cr) != n:
        return None
    return cxy, _feasify(cxy, np.minimum(cr, _wallcap(cxy)))


# ------------------------------- (4f) DISCRETE CONTINUATION IN n: the SIZE LADDER (iteration 14)
def _rung(xy, r, target, rng, m=384):
    """ONE step of the ladder: add or drop exactly one circle, keeping the rest strictly feasible.

    Deliberately NOT deterministic-greedy.  The insert takes a random pick from the three roomiest
    holes and the delete a random pick from the three smallest circles, so the same donor walked twice
    lands in two different basins -- the ladder is a diversity generator, not a single deterministic
    map from (donor, n) to one child."""
    k = len(r)
    if target > k:                               # INSERT into a hole
        cand = rng.uniform(LO + 0.004, HI - 0.004, size=(m, 2))
        d = (np.hypot(cand[:, 0, None] - xy[None, :, 0], cand[:, 1, None] - xy[None, :, 1])
             - r[None, :])
        wall = np.minimum.reduce([cand[:, 0] - LO, HI - cand[:, 0],
                                  cand[:, 1] - LO, HI - cand[:, 1]])
        room = np.minimum(d.min(axis=1), wall)
        top = np.argsort(room)[-3:]
        i = int(top[rng.randint(len(top))])
        return (np.vstack([xy, cand[i]]),
                np.append(r, max(RMIN, 0.92 * float(max(room[i], RMIN)))))
    if target < k:                               # DELETE one of the three smallest
        sm = np.argsort(r)[:3]
        j = int(sm[rng.randint(len(sm))])
        keep = np.arange(k) != j
        return xy[keep], r[keep]
    return xy, r


def _ladder(sxy, sr, n, rng, stop, tc):
    """Walk a donor packing of m circles to size n ONE CIRCLE AT A TIME, re-solving at every rung --
    a HOMOTOPY IN THE DISCRETE PARAMETER n, through the EVEN sizes that are not in the census.

    ITERATION 14.  `_migrate` (iteration 7) is a one-shot jump: it deletes |m-n| circles at once (the
    smallest, always), or drops them all into holes at once, shrinks everything 3% to buy slack, and
    hands the wreckage to a single Newton.  For |m-n| = 2 that is nearly a rung; for |m-n| = 6 it is a
    demolition, which is why `_pick_donor` had to be capped at span 6 -- past that the child never
    recovered, so two thirds of the census was unreachable as donor material for the three arms that
    are still short.

    Continuation instead of jumping.  Each rung changes the count by exactly ONE and is followed by a
    full barrier re-solve, so the arrangement RELAXES into its new count before the next circle moves:
    the packing is tracked along a path of optima rather than teleported off one and hoped back onto
    another.  The intermediate sizes are the even n, which are not in the census and are never scored
    -- they exist only as waypoints, and they are exactly what makes an odd-to-odd walk continuous.

    This is the only move whose parent may be an arbitrarily DISTANT arm.  n=81's neighbours 79 and 83
    both sit at the 7-digit cap; so do 71..79 and 83..89.  Under the old span-6 jump those structures
    were either unreachable or arrived destroyed.
    """
    xy, r = sxy.copy(), sr.copy()
    step = 1 if len(r) < n else -1
    rungs = abs(len(r) - n)
    if rungs == 0:
        return xy, _feasify(xy, np.minimum(r, _wallcap(xy)))
    # split what is left of the clock over the rungs, with a floor so a long ladder still converges
    per = max(0.35 * tc, (stop - time.process_time()) / max(1, rungs) * 0.9)
    while len(r) != n:
        xy, r = _rung(xy, r, len(r) + step, rng)
        r = _feasify(xy, np.minimum(r, _wallcap(xy)))
        if time.process_time() >= stop:
            continue                              # out of clock: finish the COUNT, skip the re-solves
        xy, r, _ = _bar_newton(xy, r, min(stop, time.process_time() + per), mu_min=1e-8)
    return xy, _feasify(xy, np.minimum(r, _wallcap(xy)))


def _patch(pxy, pr, rng, stop, tc):
    """NESTED PATCH OPTIMISATION (iteration 10) -- a SUB-SOLVER, not a move.

    Iteration 9 made ruin-and-recreate the mixture's only meso move, and its own post-mortem named the
    limit: one ruin bought exactly ONE refill, because the refill was scored by a FULL barrier Newton
    over all 3n variables.  At n=99 that is a 297x297 Cholesky per step, so a single lottery ticket
    cost ~0.2 s and the ten arms still 1e-4 short were being handed a handful of meso samples per
    iteration -- nowhere near enough to find one particular local motif by chance.

    The exterior of a ruined patch is already AT ITS OPTIMUM; re-solving it is pure waste.  So here
    the survivors are FROZEN EXACTLY and only the k newcomers are variables: `_bar_newton(..., free=)`
    drops the system to 3k (k <= 16 -> a 48x48 Cholesky) and drops the barrier to the constraints a
    newcomer can actually violate.  One patch solve costs O(k^3 + n) instead of O(n^3), which buys
    10-40 independent refills of the SAME void inside the clock one full refill used to take.  The
    best of them is returned as the child, and the caller's full Newton then relaxes the rim and the
    patch together -- so the expensive global solve is paid ONCE, for the winner, instead of once per
    ticket.  The restricted solve is therefore a SCREEN: ruin used to pay a full 3n Newton to find out
    that a ticket was worthless, and now it pays a 3k one.

    This is the first move in the mixture with an inner optimisation loop of its own: the patch is a
    self-contained max-sum-of-radii problem (k circles into the void left by k circles, with the rim
    as fixed obstacles), and it is now solved as one."""
    n = len(pr)
    pxy = np.asarray(pxy, float)
    pr = np.asarray(pr, float)
    tin = max(0.03, 0.14 * tc)                      # measured: a restricted solve converges in 0.023-0.031 s
    bxy = bcr = None
    bs = -np.inf
    kxy = kr = free = None
    k = 0
    tries = 0
    while tries < 40 and time.process_time() < stop - 0.6 * tin:
        # a fresh void half the time, the SAME void refilled differently the other half: breadth over
        # patches and depth within one patch, both now priced at a fifth of a full Newton
        if kxy is None or rng.rand() < 0.5:
            idx = _ruin_pick(pxy, pr, rng)
            k = len(idx)
            keep = np.setdiff1d(np.arange(n), idx)
            kxy, kr = pxy[keep].copy(), pr[keep].copy()
            free = np.arange(len(keep), n)
        greed = float(rng.uniform(0.30, 0.90))
        fill = float(rng.uniform(0.70, 0.97))
        cxy, cr = _refill(kxy, kr, k, rng, greed, fill)
        tries += 1
        xy2, r2, s2 = _bar_newton(cxy, cr, min(stop, time.process_time() + tin), free=free)
        if s2 > bs:
            bs, bxy, bcr = s2, xy2, r2
    if bxy is None:
        idx = _ruin_pick(pxy, pr, rng)
        keep = np.setdiff1d(np.arange(n), idx)
        cxy, cr = _refill(pxy[keep], pr[keep] * 0.995, len(idx), rng, 0.7, 0.85)
        return cxy, _feasify(cxy, np.minimum(cr, _wallcap(cxy)))
    return bxy, bcr


_D4 = tuple(np.array(M, dtype=float) for M in (
    ((1., 0.), (0., 1.)), ((-1., 0.), (0., 1.)), ((1., 0.), (0., -1.)), ((-1., 0.), (0., -1.)),
    ((0., 1.), (1., 0.)), ((0., -1.), (1., 0.)), ((0., 1.), (-1., 0.)), ((0., -1.), (-1., 0.))))


def _reorient(xy, rng):
    """Act with a random element of the square's symmetry group D4 (exact: maps [-.5,.5]^2 to itself).

    A no-op for the packing's own quality, but NOT for a move that reads two packings: a half-plane
    cut or a hole-fill sees a reoriented parent as a structurally different donor."""
    return xy @ _D4[int(rng.randint(8))].T


def _affine(xy, r, rng):
    """CONTAINER HOMOTOPY -- act on the WHOLE packing with an element of the affine group, then let
    the square squeeze it back.

    Every other move perturbs CIRCLES: jitter, teleport a rattler, reweight the objective, splice two
    parents, insert/delete across n.  All of them leave the packing's relation to the four WALLS
    intact, and for a square container that relation IS the structure -- which rows are wall-locked,
    which corner holds the big circle, whether the lattice runs axis-parallel or diagonal.  A rotation
    or a squeeze changes every wall contact at once, so the barrier Newton re-lands in a contact
    topology no per-circle perturbation can reach.  The 23 sub-cap arms sit 1e-3 to 1e-4 relative from
    the record -- far too coarse for polish and exactly the size of a wrong global arrangement."""
    k = int(rng.randint(4))
    if k == 0:                                   # area-preserving anisotropic squeeze
        a = float(np.exp(rng.uniform(0.06, 0.30) * (1.0 if rng.rand() < 0.5 else -1.0)))
        M = np.array([[a, 0.0], [0.0, 1.0 / a]])
    elif k == 1:                                 # rotation -- trades axis-parallel rows for diagonal
        th = float(rng.uniform(0.03, 0.40) * (1.0 if rng.rand() < 0.5 else -1.0))
        c, s = np.cos(th), np.sin(th)
        M = np.array([[c, -s], [s, c]])
    elif k == 2:                                 # shear -- slides rows past each other
        g = float(rng.uniform(0.08, 0.30) * (1.0 if rng.rand() < 0.5 else -1.0))
        M = np.array([[1.0, g], [0.0, 1.0]]) if rng.rand() < 0.5 else np.array([[1.0, 0.0], [g, 1.0]])
    else:                                        # dilation about a random point: frees a wall strip
        c = rng.uniform(LO, HI, size=2)
        s = float(rng.uniform(0.86, 1.16))
        z = c[None, :] + s * (xy - c[None, :])
        return _fitsquare(z, r * s)
    return _fitsquare(xy @ M.T, r.copy())


def _fitsquare(z, r):
    """Re-centre the image and similarity-scale it to just fill the square (the scale rides on r too,
    so what the map costs in radius is only what the container really took)."""
    lo, hi = z.min(axis=0), z.max(axis=0)
    z = z - 0.5 * (lo + hi)
    h = float(np.abs(z).max())
    if h > 1e-9:
        f = (0.5 - 1e-3) / h
        z, r = z * f, r * f
    return z, _feasify(z, np.minimum(r, _wallcap(z)))


def _cross(axy, ar, bxy, br, rng):
    """SPATIAL HALF-PLANE RECOMBINATION of two packings of the same n.

    Keep parent A's centres on one side of a random line and parent B's on the other, then repair the
    count (drop the smallest surplus / re-admit the largest unused centre) and the collisions along the
    seam (relocate one of each too-close pair into the emptiest hole).  The child inherits a CONTACT
    TOPOLOGY that neither parent has -- the one thing no perturbation of a single incumbent can make,
    because a perturbation only ever carries ONE arrangement forward."""
    n = len(ar)
    th = rng.uniform(0.0, np.pi)
    u = np.array([np.cos(th), np.sin(th)])
    c = float(rng.uniform(-0.3, 0.3))
    ma = (axy @ u) < c
    mb = (bxy @ u) >= c
    xy = np.vstack([axy[ma], bxy[mb]])
    r = np.concatenate([ar[ma], br[mb]])
    if len(r) > n:                                   # seam overfull: the smallest circles lose
        keep = np.argsort(r)[len(r) - n:]
        xy, r = xy[keep], r[keep]
    elif len(r) < n:                                 # seam underfull: re-admit the biggest leftovers
        k = n - len(r)
        pxy = np.vstack([axy[~ma], bxy[~mb]])
        pr = np.concatenate([ar[~ma], br[~mb]])
        if len(pr) >= k:
            take = np.argsort(pr)[len(pr) - k:]
            xy = np.vstack([xy, pxy[take]])
            r = np.concatenate([r, pr[take]])
        else:
            xy = np.vstack([xy, _start_random(k, rng)])
            r = np.concatenate([r, np.full(k, float(np.median(ar)))])
    xy = np.clip(xy, LO + 1e-3, HI - 1e-3)
    thr = 0.30 / np.sqrt(n)                          # seam repair: no two centres may sit on top of
    for _ in range(2):                               # each other, or _feasify collapses every radius
        bad = np.where(_pairdist(xy).min(axis=1) < thr)[0]
        if len(bad) == 0:
            break
        for i in bad[::2]:
            xy = _hole_place(xy, int(i), rng)
    return xy, _feasify(xy, np.minimum(r, _wallcap(xy)))


# ------------------------------------------------- (4c) CENSUS move: cross-arm migration (iter 7)
def _migrate(sxy, sr, n, rng):
    """Carry a packing of m circles over to a DIFFERENT size n -- the move that makes the 37 arms one
    population instead of 37 independent problems.

    Consecutive census sizes are one insert/delete apart, and their optimal arrangements are close
    relatives: an (n+2) optimum with its two smallest circles removed is a strong n candidate (the
    survivors simply grow into the vacancies), and an (n-2) optimum with two circles dropped into its
    emptiest holes is a strong n candidate the other way.  Neither is reachable from inside arm n --
    every move so far only ever recombines arrangements that arm n already found.

    It also un-retires the best arms in the census.  Seven n are at the 7-digit cap, so the scheduler
    correctly gives them ZERO clock -- but their arrangements are the best structures the run owns,
    and until now nothing could read them.  As donors they cost nothing and pay into their neighbours.
    """
    m = len(sr)
    xy, r = sxy.copy(), sr.copy()
    if m > n:                                    # DELETE: the smallest circles leave, the rest grow
        keep = np.argsort(r)[m - n:]
        xy, r = xy[keep], r[keep]
    elif m < n:                                  # INSERT: newcomers land in the emptiest holes
        for _ in range(n - m):
            cand = rng.uniform(LO + 0.005, HI - 0.005, size=(256, 2))
            d = (np.hypot(cand[:, 0, None] - xy[None, :, 0], cand[:, 1, None] - xy[None, :, 1])
                 - r[None, :])
            wall = np.minimum.reduce([cand[:, 0] - LO, HI - cand[:, 0],
                                      cand[:, 1] - LO, HI - cand[:, 1]])
            room = np.minimum(d.min(axis=1), wall)
            i = int(np.argmax(room))
            xy = np.vstack([xy, cand[i]])
            r = np.append(r, max(RMIN, 0.9 * float(room[i])))
    # a shrink gives the barrier Newton room to re-grow every radius jointly instead of jamming the
    # inherited contacts; the guest circles are the ones that need the slack.
    r = r * 0.97
    return xy, _feasify(xy, np.minimum(r, _wallcap(xy)))


def _pick_donor(w, census, rng, span=6, flat=False):
    """Choose which arm donates to arm w: near in n (the structures stay related) and high in digits
    (a capped arm is the best donor there is), with a random pool member 40% of the time so the
    donor's diversity, not just its champion, crosses over."""
    cands, wts = [], []
    for m, q in census.items():
        if m == w.n or not q.pool:
            continue
        dist = abs(m - w.n)
        if dist > span:
            continue
        dg = q.digits()
        dg = 3.5 if dg is None else dg
        cands.append(q)
        # ITERATION 14: at span 14 the old 1/(1+0.6*dist) kernel puts a donor 14 sizes away at 0.11x
        # the weight of a neighbour -- so widening the span alone changed almost nothing.  The ladder
        # is what makes distance survivable, so the slot that uses it flattens the kernel too.
        wts.append((0.5 + dg) / (1.0 + (0.15 if flat else 0.6) * dist))
    if not cands:
        return None
    p = np.asarray(wts, dtype=float)
    p /= p.sum()
    q = cands[int(rng.choice(len(cands), p=p))]
    if rng.rand() < 0.4:
        return q.pool[int(rng.randint(len(q.pool)))]
    return max(q.pool, key=lambda z: z[2])


# ------------------------------------------------------------------------------- start generators
def _start_random(n, rng):
    return rng.uniform(LO + 0.02, HI - 0.02, size=(n, 2))


def _start_grid(n, rng):
    k = int(np.ceil(np.sqrt(n)))
    g = (np.arange(k) + 0.5) / k - 0.5
    pts = np.array([(a, b) for a in g for b in g])
    sel = rng.choice(len(pts), size=n, replace=False) if len(pts) > n else np.arange(n) % len(pts)
    return np.clip(pts[sel] + rng.normal(0, 0.25 / k, size=(n, 2)), LO + 1e-3, HI - 1e-3)


def _start_hex(n, rng):
    rows = int(np.ceil(np.sqrt(n * 1.15)))
    pts = []
    for i in range(rows + 1):
        y = LO + (i + 0.5) / (rows + 1e-9)
        off = 0.5 / rows if i % 2 else 0.0
        j = 0
        while True:
            x = LO + off + (j + 0.5) / rows
            if x > HI:
                break
            pts.append((x, y))
            j += 1
    pts = np.array(pts) if pts else _start_random(n, rng)
    if len(pts) < n:
        pts = np.vstack([pts, _start_random(n - len(pts), rng)])
    sel = rng.choice(len(pts), size=n, replace=False)
    return np.clip(pts[sel] + rng.normal(0, 0.01, size=(n, 2)), LO + 1e-3, HI - 1e-3)


def _start_bigsmall(n, rng):
    """A coarse grid of big circles plus interstitial small ones -- the shape csqv optima actually take."""
    k = max(2, int(round(np.sqrt(n / 1.6))))
    g = (np.arange(k) + 0.5) / k - 0.5
    big = np.array([(a, b) for a in g for b in g])
    h = (np.arange(k - 1) + 1.0) / k - 0.5
    small = np.array([(a, b) for a in h for b in h]) if k > 1 else np.zeros((0, 2))
    pts = np.vstack([big, small]) if len(small) else big
    if len(pts) < n:
        pts = np.vstack([pts, _start_random(n - len(pts), rng)])
    sel = rng.choice(len(pts), size=n, replace=False)
    return np.clip(pts[sel] + rng.normal(0, 0.15 / k, size=(n, 2)), LO + 1e-3, HI - 1e-3)


def _start_rings(n, rng):
    pts, left, ring = [], n, 0
    while left > 0:
        t = 0.5 - (ring + 0.5) * (0.5 / max(1, int(np.sqrt(n) / 1.6)))
        t = max(t, 0.02)
        cnt = min(left, max(1, int(8 * t / (0.5 / max(1, int(np.sqrt(n) / 1.6))))))
        if ring > 3 or t <= 0.03:
            pts.extend(_start_random(left, rng))
            break
        ang = rng.uniform(0, 2 * np.pi)
        for a in np.linspace(0, 2 * np.pi, cnt, endpoint=False) + ang:
            s = max(abs(np.cos(a)), abs(np.sin(a)))
            pts.append((t * np.cos(a) / s, t * np.sin(a) / s))
        left -= cnt
        ring += 1
    return np.clip(np.array(pts[:n]), LO + 1e-3, HI - 1e-3)


_STARTS = (_start_bigsmall, _start_grid, _start_hex, _start_random, _start_rings)


def _read_pack(n):
    """Warm start from the committed census -- GUARDED: any missing/odd pack falls back to cold start."""
    p = "bench/packs/csqv%d.pck" % n
    if not os.path.exists(p):
        return None
    try:
        rows = [ln.split() for ln in open(p).read().splitlines() if ln.strip()][2:]
        arr = np.array([[float(v) for v in row[:3]] for row in rows], dtype=float)
    except Exception:
        return None
    if arr.shape != (n, 3) or not np.all(np.isfinite(arr)):
        return None
    return arr


# ------------------------------------------- (4d) PERSISTENT BASIN ARCHIVE across iterations (it 11)
#
# Until now git remembered exactly ONE packing per n (bench/packs/csqv<n>.pck), so every iteration
# re-paid the whole diversification cost from a single point: the pool of live basins an arm builds
# over 12 CPU-s of hops -- the only structural memory the search has -- was thrown away when solve()
# returned, and crossover (which needs two members) spent the first half of every iteration with a
# one-member pool and nothing to cross.  The archive makes the POPULATION persist, not just the best:
# archive/csqv<n>.pool holds up to ARCH_K basins per arm, written at the end of solve() and admitted
# into the pool at seed time.
#
# LEGITIMACY: a member is archived only after the metered evaluate() has itself confirmed it feasible
# -- the final batch shows every pool member to the harness (it silently keeps the best, so showing
# worse ones is free) and only the rows evaluate() marks feasible are written.  Nothing enters the
# archive that did not go through the meter, and nothing here writes to bench/packs/.
ARCHIVE_DIR = "archive"
ARCH_K = 6


def _arch_path(n):
    return os.path.join(ARCHIVE_DIR, "csqv%d.pool" % n)


def _feas_ok(xy, r):
    """The scorer's feasibility geometry, re-implemented locally (importing harness is banned)."""
    if r.size == 0 or xy.shape != (r.size, 2):
        return False
    if not (np.all(np.isfinite(xy)) and np.all(np.isfinite(r))):
        return False
    if float(r.min()) <= 0.0:
        return False
    if float(np.min(_wallcap(xy) - r)) < -FEAS_TOL:
        return False
    return float(np.min(_pairdist(xy) - (r[:, None] + r[None, :]))) >= -FEAS_TOL


def _read_pool(n):
    """GUARDED read of this arm's committed basin archive -- [] for any n that has none (a fresh n,
    a size outside the census, an offline re-run), so the caller always has a working cold path."""
    p = _arch_path(n)
    if not os.path.exists(p):
        return []
    out = []
    try:
        toks = open(p).read().split()
        i = 0
        while i < len(toks) and len(out) < ARCH_K:
            if toks[i] != "MEMBER":
                i += 1
                continue
            i += 1
            chunk = toks[i:i + 3 * n]
            i += 3 * n
            if len(chunk) != 3 * n:
                break
            a = np.array([float(v) for v in chunk], dtype=float).reshape(n, 3)
            xy, r = a[:, :2].copy(), a[:, 2].copy()
            if _feas_ok(xy, r):
                out.append([xy, r, float(r.sum())])
    except Exception:
        return []
    return out


def _write_pool(n, members):
    """Persist up to ARCH_K basins for arm n.  Best-first, atomic replace, never raises."""
    try:
        if not os.path.isdir(ARCHIVE_DIR):
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
        buf = ["POOL %d" % n]
        for xy, r, _s in sorted(members, key=lambda q: -q[2])[:ARCH_K]:
            buf.append("MEMBER")
            for k in range(n):
                buf.append("%.17g %.17g %.17g" % (xy[k, 0], xy[k, 1], r[k]))
        tmp = _arch_path(n) + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(buf) + "\n")
        os.replace(tmp, _arch_path(n))
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------------------ solve
def _cold(n, gi, rng):
    """A cold layout from generator gi, with conservative starting radii."""
    xy = _STARTS[gi % len(_STARTS)](n, rng)
    d = _pairdist(xy)
    r = np.maximum(np.minimum(_wallcap(xy), d.min(axis=1) / 2.0), RMIN)
    return xy, r


# ---------------------------------------------------------------- (5) the census-level scheduler
#
# ITERATION 5.  Every earlier iteration gave each n a FIXED slice of the clock, weighted by n.  That
# is the wrong control law, and the verbose scorer says so directly: three n (29, 47, 49) are already
# at the 7-digit cap -- their gap is below the scorer's resolution, so every second spent on them is
# worth EXACTLY ZERO -- while n=73/87/89 sit at 2.4 digits.  A fixed split cannot know that, because
# it is blind to the one quantity the SCORE is made of: digits.
#
# So the objective the solver optimizes is changed from "sum(r) for each n, independently" to the
# actual census score, "mean over n of digits(n)".  bench/records.json is READ (explicitly allowed)
# so the solver can see its own per-n digits, and time is allocated by a bandit over the 37 arms:
#
#   * ONE HOP IS THE ATOM.  The per-n loop is turned inside out into a persistent _Walker (incumbent,
#     best, hop counter, cold-generator index, threshold state) plus a stateless _hop(w) that advances
#     it by exactly one basin hop.  A walker can be suspended and resumed at zero cost, so the
#     scheduler is free to interleave the 37 n at hop granularity instead of visiting each once.
#   * REWARD IS DIGITS PER CPU-SECOND, the exact derivative of the SCORE with respect to the clock.
#   * ARMS AT THE CAP ARE RETIRED.  digits >= 7 (relgap <= 1e-7) -> deficit 0 -> priority 0, forever.
#   * GREEDY ON A DECAYED RATE, WITH A DEFICIT FLOOR.  rate is an EWMA of digits/s (decay 0.55); the
#     priority is (rate + FLOOR*deficit) * U(0.7,1.3).  A productive n keeps the clock for as long as
#     it keeps paying; when its rate decays into the floor it rejoins a randomized round robin that
#     is biased toward the n with the most digits still to win.  Nothing is starved and nothing that
#     cannot pay is subsidised.
#   * The threshold-accepting temperature now anneals on the GLOBAL clock rather than a per-n slice,
#     which is the only consistent schedule once the slices are gone.
#
# The fallback path matters as much as the main one: for an n with no record (an offline re-run on a
# size outside the census) there are no digits to measure, so the reward degrades to permille of
# relative sum(r) gain and the deficit to a constant -- the scheduler still runs, just uninformed.


def _load_records():
    """READ-ONLY view of bench/records.json -- guarded, and absent records are simply unknown arms."""
    try:
        with open("bench/records.json") as f:
            raw = json.load(f).get("records", {})
        return {int(k): float(v) for k, v in raw.items()}
    except Exception:
        return {}


def _digits(s, rec):
    """The scorer's own per-n metric: clamp(-log10(relgap), 0, 7).  None when the record is unknown."""
    if rec is None or not np.isfinite(rec) or rec <= 0:
        return None
    g = max(0.0, (rec - s) / rec)
    if g <= 1e-7:
        return 7.0
    return float(min(7.0, max(0.0, -np.log10(g))))


class _Walker:
    """Everything one n's basin walk needs to be suspended between hops and resumed for free."""
    __slots__ = ("n", "rec", "static", "warm", "gi", "hop", "best_xy", "best_r", "best_s",
                 "pool", "rate", "spent", "tcap", "dry")

    def __init__(self, n, rec):
        self.n = n
        self.rec = rec
        self.static = _SLP_CACHE.setdefault(n, _slp_static(n))
        self.warm = None
        self.gi = 0
        self.hop = 0
        self.best_xy = self.best_r = None
        self.best_s = -np.inf
        self.pool = []              # POPULATION of live basins for this n: [[xy, r, sum_r], ...]
        self.rate = np.inf          # untried arms are optimistic: every n gets a first hop
        self.spent = 0.0
        self.dry = 0                # consecutive hops that did not raise best_s (iteration 15)
        # nominal cost of one local solve at this n (measured: Newton is ~0.04 s at n=27, 0.25 s at
        # n=99), used to size the inner caps now that there is no per-n slice to take fractions of.
        self.tcap = 0.02 + 0.0035 * n

    def digits(self):
        return _digits(self.best_s, self.rec)

    def deficit(self):
        d = self.digits()
        if d is None:
            return 1.0
        return max(0.0, 7.0 - d)


def _bank(w, evaluate, meter, xy, r, s):
    """Route an improvement through the metered evaluate() at once: a CPU backstop that fires
    mid-hop then loses nothing."""
    if s > w.best_s:
        w.best_s, w.best_xy, w.best_r = s, xy.copy(), r.copy()
        if meter.left() > 0:
            evaluate(w.n, _pack(xy, r))
        return True
    return False


def _seed(w, evaluate, meter, rng, stop):
    """Seed from the committed census (git-as-memory), else a cold start, then one local solve.

    ITERATION 15 -- DONOR-ONLY SEEDING.  The main loop only ever hands a hop to an arm with
    `deficit() > 0`, so every second spent seeding an arm whose committed pack is already at the
    7-digit cap is spent on an arm that will not be given a single hop.  With 34 of 37 arms capped
    that is the whole of the seeding third of the clock, minus three arms' worth.  A capped arm is
    still wanted -- it is the best DONOR material the ladder has -- but being a donor needs only its
    pack and its archived pool in `census`, not a Newton and not an `evaluate()`.  So a capped arm is
    loaded and nothing else, and the clock it used to burn goes to the arms that can still pay."""
    warm = _read_pack(w.n)
    w.warm = warm
    if warm is not None:
        xy0, r0 = warm[:, :2].copy(), warm[:, 2].copy()
        rr0 = _feasify(xy0, r0)
        s0 = float(rr0.sum())
        d0 = _digits(s0, w.rec)
        if d0 is not None and d0 >= 7.0:
            w.best_xy, w.best_r, w.best_s = xy0, rr0, s0
            w.pool = [[xy0.copy(), rr0.copy(), s0]]
            for mem in _read_pool(w.n):
                _admit(w, mem)
            return
    if warm is not None:
        xy, r = warm[:, :2].copy(), warm[:, 2].copy()
    else:
        xy, r = _cold(w.n, w.gi, rng)
        w.gi += 1
    rr = _feasify(xy, r)
    _bank(w, evaluate, meter, xy, rr, float(rr.sum()))
    xy, r, s = _bar_newton(xy, rr, min(stop, time.process_time() + 4.0 * w.tcap))
    _bank(w, evaluate, meter, xy, r, s)
    w.pool = [[w.best_xy.copy(), w.best_r.copy(), w.best_s]]
    # POPULATION memory (iteration 11): the basins this arm reached in earlier iterations re-enter the
    # pool for free, so crossover/tournament start from real structural diversity instead of one point.
    for mem in _read_pool(w.n):
        _admit(w, mem)
        if mem[2] > w.best_s:
            _bank(w, evaluate, meter, mem[0], mem[1], mem[2])


DRY_TRIGGER = 5          # consecutive no-gain hops after which an arm switches to continuation
LADDER_P = 0.45          # ...and this fraction of its hops go to the size ladder (rest: full mixture)
POOL_K = 6               # live basins carried per arm (raised from 4 with the novelty reserve)
POOL_ELITE = 3           # the top-by-sum members no novelty argument may ever displace
NOVEL_TOL = 3e-4         # sorted-radius distance below which two members are literally the SAME basin
NOVEL_FLOOR = 0.03       # a novelty-slot member may sit at most 3% of sum(r) below the pool's best


def _sigdist(sa, sb):
    """Structural distance between two packings: the max gap between their SORTED radius spectra,
    relative to the mean radius.  Sorting makes it invariant to labelling and to the 8 symmetries of
    the square, so it measures the BASIN and not the pose -- which `sum_r` alone cannot do (two
    genuinely different arrangements can share a sum, and a re-solve of one basin can move the sum)."""
    m = max(1e-12, float(sa.mean()))
    return float(np.max(np.abs(sa - sb))) / m


def _admit(w, mem, keep_worst_out=True):
    """Insert a basin into the arm's population -- ELITE + NOVELTY RESERVE, not a top-K list.

    ITERATION 13.  Iteration 11 gave each arm a population and iterations 11/12 both closed with the
    same open item: `_admit` dedups on `sum_r`, which is a weak diversity proxy.  Building `_wallrow`
    turned that from an aesthetic complaint into a hard blocker.  `artifacts/iter13/probe2.py` (real
    packs, real Newton) measures what a wall-row recount actually yields on the four uncapped arms:
    10-21 STRUCTURALLY DISTINCT basins per 10 s, ~half of them carrying a wall-row count the incumbent
    does not have -- and every one of them 0.05%-0.5% BELOW the incumbent's sum.  Under a top-K-by-sum
    pool, all of them are discarded on arrival: the arm keeps four near-copies of one basin, and the
    only move that can change the boundary combinatorics feeds a filter that deletes its output.

    So the pool splits.  The best POOL_ELITE members are protected outright (the walk must never lose
    ground).  The remaining slots are a NOVELTY RESERVE: an inferior newcomer that is FURTHER from the
    rest of the population than the least-novel non-elite member takes that member's slot, provided it
    is within NOVEL_FLOOR of the pool's best.  Novelty is measured by `_sigdist`, not by sum."""
    pool = w.pool
    sm = np.sort(mem[1])
    sigs = [np.sort(q[1]) for q in pool]
    for i, q in enumerate(pool):
        if _sigdist(sm, sigs[i]) <= NOVEL_TOL:       # the same basin, re-found: keep the better copy
            if mem[2] > q[2]:
                q[0], q[1], q[2] = mem[0], mem[1], mem[2]
            return
    if len(pool) < POOL_K:
        pool.append(mem)
        return
    order = sorted(range(len(pool)), key=lambda i: -pool[i][2])
    rest = order[POOL_ELITE:]
    j = min(rest, key=lambda i: pool[i][2])
    if mem[2] > pool[j][2] or not keep_worst_out:    # the old rule still holds for a better basin
        pool[j] = mem
        return
    if mem[2] < pool[order[0]][2] - NOVEL_FLOOR * abs(pool[order[0]][2]):
        return                                       # novel, but far off the pace: not worth a slot
    cand_nov = min(_sigdist(sm, s) for s in sigs)
    worst_i, worst_nov = None, None
    for i in rest:
        nv = min(_sigdist(sigs[i], sigs[q]) for q in range(len(pool)) if q != i)
        if worst_nov is None or nv < worst_nov:
            worst_i, worst_nov = i, nv
    if worst_i is not None and cand_nov > worst_nov:
        pool[worst_i] = mem                          # a structurally new basin buys the crowded slot


def _hop(w, evaluate, meter, rng, stop, thr, reset, census=None):
    """Advance walker w by exactly ONE basin hop, taken from -- and returned to -- its POPULATION.

    ITERATION 6.  Iteration 5's walker carried a single incumbent, so an arm could hold exactly one
    arrangement at a time and every cold restart threw the previous basin away.  Its own measurement
    said that was the binding limit: the self-test, on ONE n for 8 s, beat a record the 37-n sweep
    missed, i.e. the between-run variance was larger than the remaining gap.  So the incumbent becomes
    a pool of up to POOL_K basins, selected by tournament, and the move mixture gains the move a
    population makes possible:

      hop % 9 == 0,1,2  jitter / teleport-smallest / teleport-into-hole
      hop % 9 == 3,4    objective homotopy, global and rattler-boosted
      hop % 9 == 5      CROSSOVER of two pool members of this arm (see _cross)
      hop % 9 == 6      MIGRATION from a NEIGHBOURING ARM (iteration 7; see _migrate) -- the only
                        move whose parent is a packing of a different size
      hop % 9 == 7      CONTAINER HOMOTOPY (iteration 8; see _affine) -- the only move that acts on
                        the packing as a WHOLE against the walls, rather than on its circles
      hop % 10 == 8     RUIN-AND-RECREATE (iteration 9; see _ruin) -- the only MESO move: it rewrites
                        a contiguous patch of 4-16 circles and leaves the rest of the packing alone
      hop % 10 == 9     NESTED PATCH OPTIMISATION (iteration 10; see _patch) -- the same ruin, but the
                        void is refilled and re-solved 10-40 times with the exterior FROZEN, so the
                        meso scale gets a real inner search instead of one lottery ticket
      hop % 12 == 10,11 WALL-ROW RECOUNT (iteration 13; see _wallrow) -- the only move that changes
                        the BOUNDARY COMBINATORICS: one wall's row of k circles comes back as k+-1
      hop %  5 ==  4    SIZE LADDER (iteration 14; see _ladder) -- discrete continuation in n: a
                        capped donor up to 14 sizes away is walked to n ONE circle at a time, with a
                        full re-solve at every rung, through the EVEN sizes outside the census
      hop % 11 == 10    a cold generator -- ADMITTED AS A NEW MEMBER rather than overwriting the
                        incumbent, which is what makes restart-and-keep-best free.

    A child (crossover or cold) competes for its own slot; a perturbation of member k is threshold-
    accepted back into slot k, exactly as the single-incumbent walk did."""
    n, tc = w.n, w.tcap
    pool = w.pool
    if not pool:
        return
    if reset:
        # Late in the global clock the walk goes greedy: hold only the two best basins, and make sure
        # the arm's overall best is one of them.
        if len(pool) > 2:
            pool.sort(key=lambda q: -q[2])
            del pool[2:]
        if pool[0][2] < w.best_s and w.best_xy is not None:
            pool[0] = [w.best_xy.copy(), w.best_r.copy(), w.best_s]
    # tournament-of-2 selection, with a 30% uniform draw so a weak-but-different basin still runs
    if len(pool) > 1 and rng.rand() < 0.7:
        i, j = rng.randint(len(pool)), rng.randint(len(pool))
        k = i if pool[i][2] >= pool[j][2] else j
    else:
        k = rng.randint(len(pool))
    pxy, pr, ps = pool[k]
    m = w.hop % 12
    # ITERATION 15 -- the ladder's share is no longer a fixed slot in a global schedule.  Iteration
    # 14 measured the ladder finding an n=41 child 2.3e-04 ABOVE the incumbent in 17 unmetered draws,
    # and then measured the metered run finding nothing -- with one slot in five the arm never got
    # 17 draws.  A move whose payoff is heavy-tailed needs draws, and the arms that need it are
    # exactly the ones nothing else is moving, so the share is now conditioned on STAGNATION: an arm
    # that has gone DRY_TRIGGER hops without raising best_s puts LADDER_P of its hops on continuation
    # in n.  The other 1-LADDER_P keep the whole mixture alive (rule 1: never one line of attack).
    ladder = (w.hop % 5 == 4) or (w.dry >= DRY_TRIGGER and rng.rand() < LADDER_P)
    child = False
    if w.hop % 11 == 10:
        cxy, cr0 = _cold(n, w.gi, rng)          # an independent line of attack, never the warm basin
        w.gi += 1
        cr0 = _feasify(cxy, cr0)
        child = True
    elif ladder and census:
        # ITERATION 14: the WIDE-SPAN rung of the census ladder.  Span 14 instead of 6, because
        # continuation survives a distance a one-shot jump does not -- this is the slot that makes
        # a capped arm seven sizes away donate to an arm that is still short.
        mem = _pick_donor(w, census, rng, span=14, flat=True)
        if mem is None:
            cxy = _shake(pxy, pr, rng, 2)
            cr0 = _feasify(cxy, np.minimum(pr, _wallcap(cxy)))
        else:
            dxy = _reorient(mem[0], rng) if rng.rand() < 0.5 else mem[0]
            cxy, cr0 = _ladder(dxy, mem[1], n, rng,
                               min(stop, time.process_time() + 6.0 * tc), tc)
            child = True
    elif m == 6 and census:
        mem = _pick_donor(w, census, rng)
        if mem is None:
            cxy, cr0 = _shake(pxy, pr, rng, 2), None
            cr0 = _feasify(cxy, np.minimum(pr, _wallcap(cxy)))
        else:
            dxy = _reorient(mem[0], rng) if rng.rand() < 0.5 else mem[0]
            # HEDGE (rule 1): the one-shot jump is NOT retired.  It is the cheap, high-variance
            # arm of the same idea and it is what the ladder degenerates to at |m-n| == 2, so a
            # third of the near-span slots keep taking it.
            if rng.rand() < 0.35 or abs(len(mem[1]) - n) <= 2:
                cxy, cr0 = _migrate(dxy, mem[1], n, rng)
            else:
                cxy, cr0 = _ladder(dxy, mem[1], n, rng,
                                   min(stop, time.process_time() + 5.0 * tc), tc)
            child = True
    elif m == 10 or m == 11:
        # ITERATION 13: the only move that changes the packing's BOUNDARY COMBINATORICS (k -> k+-1
        # circles along one wall, at fixed n).  Two of the twelve slots, because the four arms still
        # uncapped are basin problems and no other move in the mixture can reach this one.
        got = _wallrow(pxy, pr, rng)
        if got is None:
            got = _ruin(pxy, pr, rng)        # no usable wall row -> the meso move instead
        cxy, cr0 = got
        child = True
    elif m == 9:
        # the SAME void, refilled and re-solved 10-40 times with the exterior frozen
        cxy, cr0 = _patch(pxy, pr, rng, min(stop, time.process_time() + 3.5 * tc), tc)
        child = True
    elif m == 8:
        cxy, cr0 = _ruin(pxy, pr, rng)           # ruin a contiguous patch and refill the void it left
        child = True
    elif m == 7:
        cxy, cr0 = _affine(pxy, pr, rng)         # rotate/squeeze/shear/dilate the WHOLE arrangement
        child = True
    elif m == 5 and len(pool) >= 2:
        j = int(rng.randint(len(pool) - 1))
        j = j + 1 if j >= k else j
        bxy = _reorient(pool[j][0], rng) if rng.rand() < 0.5 else pool[j][0]
        cxy, cr0 = _cross(pxy, pr, bxy, pool[j][1], rng)
        child = True
    elif m == 3 or m == 4:
        wt = np.ones(n)
        if m == 3:
            amp = (0.08, 0.25, 0.5)[w.hop % 3]
            wt += rng.uniform(-amp, amp, n)                      # global reweighting
        else:
            sm = np.argsort(pr)[:max(1, n // 6)]                 # pay the rattlers to grow and shove
            wt[sm] = rng.uniform(1.5, 3.0)
        cxy, cr0, _ = _bar_newton(pxy, pr, min(stop, time.process_time() + 2.0 * tc),
                                  mu0=1e-3, wt=wt, ret_last=True)
    else:
        cxy = _shake(pxy, pr, rng, m % 3)
        cr0 = _feasify(cxy, np.minimum(pr, _wallcap(cxy)))
    w.hop += 1
    # a child starts far from any optimum, so it is given a longer converge cap than a perturbation
    cxy, cr, s = _bar_newton(cxy, cr0, min(stop, time.process_time() + (5.0 if child else 3.0) * tc))
    if child:
        _admit(w, [cxy.copy(), cr.copy(), s])
    elif s > ps - thr * abs(ps):
        pool[k] = [cxy.copy(), cr.copy(), s]
    if s > w.best_s:
        _bank(w, evaluate, meter, cxy, cr, s)
        # Two further lines of attack, only on a basin that already improved (so the clock is spent
        # where it is paying): the exact fixed-centre LP + trust-region SLP, and -- where the measured
        # cost model says it fits -- the SLSQP NLP.  Each result is re-Newtoned.
        if time.process_time() < stop - 0.02:
            rl, _, _, _, _ = _lp_full(cxy)
            xy2, r2, s2 = _slp(cxy, _feasify(cxy, rl), min(stop, time.process_time() + 4.0 * tc),
                               static=w.static)
            if s2 > w.best_s:
                _bank(w, evaluate, meter, xy2, r2, s2)
                xy3, r3, s3 = _bar_newton(xy2, r2, min(stop, time.process_time() + 3.0 * tc))
                _bank(w, evaluate, meter, xy3, r3, s3)
        got = _polish(w.best_xy, w.best_r, w.best_s, min(stop, time.process_time() + 4.0 * tc), n)
        if got is not None:
            _bank(w, evaluate, meter, got[0], got[1], got[2])
            xy4, r4, s4 = _bar_newton(got[0], got[1], min(stop, time.process_time() + 3.0 * tc))
            _bank(w, evaluate, meter, xy4, r4, s4)
        # the sharpened best re-enters the population (as slot k's occupant if it came from there)
        if w.best_s > pool[k][2] and not child:
            pool[k] = [w.best_xy.copy(), w.best_r.copy(), w.best_s]
        else:
            _admit(w, [w.best_xy.copy(), w.best_r.copy(), w.best_s])


RATE_DECAY = 0.55        # how fast a stale arm's measured digits/s fades back toward the floor
RATE_FLOOR = 2.5e-3      # per deficit-digit: the subsidy that keeps a dry-but-unfinished arm in play

# ---- ITERATION 12: the allocator is a TIME-SHARE CONTROLLER, not a greedy bandit ----------------
#
# Iteration 5's control law was "hand the next hop to the arm with the largest decayed digits/s".
# `artifacts/iter12/prof2.py` (unmetered, 100 CPU-s, the real solve() with the real move set) shows
# what that law actually does now that 30 of 37 arms are capped and only 7 carry any deficit:
#
#     n=81  28.0s (29.1%)   87h   digits 3.580 -> 4.714   gain +1.134
#     n=39  27.5s (28.6%)  205h   digits 3.389 -> 3.389   gain +0.000
#     n=73  18.4s (19.1%)   60h   digits 3.684 -> 4.110   gain +0.425
#     n=97   9.5s ( 9.8%)   23h   gain +0.000
#     n=41   7.7s ( 8.0%)   55h   gain +0.000
#     n=91   5.1s ( 5.3%)   10h   digits 4.309 -> 7.000   gain +2.691   <-- the whole payout
#     n=31   0.1s ( 0.1%)    1h   gain +0.000
#
# The single most productive arm of the run -- n=91, which went all the way to the 7-digit cap --
# was given 5% of the clock, while n=39 burned 28% and 205 hops for EXACTLY ZERO.  n=31 got one hop.
# That is not bad luck, it is the control law: digit gains are rare, heavy-tailed events, so the EWMA
# rate is mostly an estimate of noise, and "greedy on a noisy mean" locks onto whichever arm happened
# to twitch first and starves the rest.  Deficit is not a usable prior either -- n=39 has the LARGEST
# deficit in the census and is the one arm that provably cannot spend a clock.
#
# So the allocator stops trying to rank arms and starts CONTROLLING THEIR TIME SHARES.  Each arm has
# a target weight, and the priority is that weight divided by the seconds it has already been given:
#
#     weight_q  = sqrt(deficit_q) * (1 + BOOST * rate_q / max_rate)      # 1x .. (1+BOOST)x
#     priority  = weight_q / (spent_q + SHARE_T0)                        # -> spent_q ∝ weight_q
#
# sqrt() flattens the deficit prior on purpose (the measurement says a big deficit predicts nothing),
# so with no productive arm every unfinished n converges to a near-equal slice -- n=31 gets ~11%
# instead of 0.1%.  The rate signal is KEPT, not discarded: an arm that is demonstrably paying climbs
# to at most (1+BOOST)x its base share, which bounds hoarding at ~40% of the clock instead of the 29%
# -for-nothing the greedy law allowed.  SHARE_EXPLORE draws a uniform arm outright, so no share model
# -- including this one -- can starve an arm to zero.
SHARE_BOOST = 3.0        # how much a demonstrably productive arm may enlarge its own time share
SHARE_T0 = 0.6           # seconds of "virtual spend": bounds the first-pull priority, no inf needed
SHARE_EXPLORE = 0.10     # fraction of hops handed to a uniformly random unfinished arm


def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    t_entry = time.process_time()
    # Deadline computed AT ENTRY and scaled by how many n this call covers, so a second solve() call
    # in the same process still gets a full slice and a 1-target call is not starved.
    budget_s = float(CPU_BUDGET) if len(targets) == 1 else min(100.0, 6.0 + 2.7 * len(targets))   # sweep override
    deadline = t_entry + budget_s

    recs = _load_records()
    order = sorted(targets)
    walkers = []
    for n in order:
        if meter.left() <= 0 or time.process_time() > deadline - 0.05:
            break
        w = _Walker(n, recs.get(n))
        # Seeding is capped at a third of the clock so a 37-n sweep cannot spend it all on seeds.
        _seed(w, evaluate, meter, rng, min(deadline, t_entry + budget_s / 3.0))
        if w.best_xy is not None:
            walkers.append(w)

    census = {q.n: q for q in walkers}   # every arm's pool is a donor for every other arm
    # ---- the bandit: repeatedly hand one hop to the arm with the best (decayed rate + deficit floor)
    span = max(1e-6, deadline - t_entry)
    while walkers and meter.left() > 0 and time.process_time() < deadline - 0.03:
        live = [q for q in walkers if q.deficit() > 0.0]   # capped arms: more time is worth zero
        if not live:
            break
        if rng.rand() < SHARE_EXPLORE:
            w = live[int(rng.randint(len(live)))]          # the floor no share model can starve
        else:
            rmax = max((q.rate for q in live if np.isfinite(q.rate)), default=0.0)
            best_p, w = -1.0, live[0]
            for q in live:
                rq = q.rate if np.isfinite(q.rate) else 0.0
                mul = 1.0 + SHARE_BOOST * (rq / rmax if rmax > 0.0 else 0.0)
                p = np.sqrt(q.deficit()) * mul / (q.spent + SHARE_T0) * rng.uniform(0.85, 1.15)
                if p > best_p:
                    best_p, w = p, q
        frac = (time.process_time() - t_entry) / span
        thr = 2e-3 * max(0.0, 1.0 - 1.6 * frac)      # hot early, strictly greedy late (global clock)
        d0 = w.digits()
        s0, t0 = w.best_s, time.process_time()
        _hop(w, evaluate, meter, rng, min(deadline, time.process_time() + 12.0 * w.tcap),
             thr, frac > 0.62, census)
        dt = max(1e-4, time.process_time() - t0)
        w.spent += dt
        w.dry = 0 if w.best_s > s0 else w.dry + 1
        d1 = w.digits()
        if d0 is None or d1 is None:
            gain = 1e3 * max(0.0, w.best_s - s0) / max(abs(s0), 1e-9)   # no record: permille of sum(r)
        else:
            gain = max(0.0, d1 - d0)
        obs = gain / dt
        w.rate = obs if not np.isfinite(w.rate) else RATE_DECAY * w.rate + (1.0 - RATE_DECAY) * obs

    # ---- harvest: show the harness every live basin (it keeps the best), then ARCHIVE the ones it
    # confirmed feasible so the next iteration inherits the population instead of one point.
    for w in walkers:
        if w.best_xy is None:
            continue
        mems = [[w.best_xy, w.best_r, w.best_s]]
        for q in w.pool:
            if all(abs(q[2] - m[2]) > 1e-12 * max(1.0, abs(q[2])) for m in mems):
                mems.append(q)
        mems = mems[:ARCH_K]
        keep = []
        if meter.left() >= len(mems):
            feas, _sums = evaluate(w.n, np.stack([_pack(m[0], m[1]) for m in mems]))
            feas = np.asarray(feas).ravel()
            keep = [mems[b] for b in range(len(mems)) if bool(feas[b])]
        elif meter.left() > 0:
            ok, _ = evaluate(w.n, _pack(w.best_xy, w.best_r))
            keep = [mems[0]] if ok else []
        if keep:
            _write_pool(w.n, keep)


# --------------------------------------------------------------------------------------- self-test
def _self_test():
    """Runs solve() TWICE in one process (the second call must still return a feasible packing) and
    checks the guarded cold start on an n that has no committed pack."""
    class _Meter:
        def __init__(self, b):
            self.budget = b
            self.used = 0

        def left(self):
            return self.budget - self.used

        def tick(self, k):
            g = min(k, self.left())
            self.used += g
            return g

    best = {}

    def make_eval(meter):
        def evaluate(n, packing):
            a = np.asarray(packing, dtype=float)
            single = a.ndim == 2
            if single:
                a = a[None]
            B = a.shape[0]
            g = meter.tick(B)
            feas = np.zeros(B, bool)
            sums = np.full(B, -np.inf)
            for b in range(min(g, B)):
                x, y, r = a[b, :, 0], a[b, :, 1], a[b, :, 2]
                d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
                np.fill_diagonal(d, np.inf)
                ok = (r.min() > 0
                      and np.min([x - r - LO, HI - x - r, y - r - LO, HI - y - r]) >= -FEAS_TOL
                      and float((d - (r[:, None] + r[None, :])).min()) >= -FEAS_TOL)
                feas[b] = ok
                sums[b] = r.sum()
                if ok and sums[b] > best.get(n, (-np.inf,))[0]:
                    best[n] = (float(sums[b]), a[b].copy())
            return (bool(feas[0]), float(sums[0])) if single else (feas, sums)
        return evaluate

    rng = np.random.RandomState(0)
    ok = True
    for call, tgts in enumerate(([27, 41], [63])):
        best.clear()
        m = _Meter(500000)
        t0 = time.process_time()
        solve(make_eval(m), m, rng, tgts)
        dt = time.process_time() - t0
        for n in tgts:
            if n not in best:
                print("FAIL call %d: no feasible packing for n=%d" % (call, n))
                ok = False
            else:
                s, pk = best[n]
                print("call %d n=%2d sum_r=%.6f  cpu=%.1fs  evals=%d" % (call, n, s, dt, m.used))
                if pk.shape != (n, 3):
                    print("FAIL shape"); ok = False

    # guarded cold start: an n with no committed pack must still work
    best.clear()
    m = _Meter(500000)
    n = 12
    assert not os.path.exists("bench/packs/csqv12.pck"), "self-test assumed n=12 is uncommitted"
    solve(make_eval(m), m, np.random.RandomState(3), [n])
    if n not in best:
        print("FAIL cold start on uncommitted n=%d" % n); ok = False
    else:
        print("cold-start n=12 sum_r=%.6f (no committed pack)" % best[n][0])
    # ---- basin archive (iteration 11): round-trip, guard on an n that has none, feasibility
    if not _read_pool(999999) == []:
        print("FAIL: unguarded archive read on an n with no pool"); ok = False
    for n in (27, 41, 63, 12):
        mems = _read_pool(n)
        if not mems:
            print("FAIL: no archive written for n=%d" % n); ok = False
            continue
        if any(m[0].shape != (n, 2) or not _feas_ok(m[0], m[1]) for m in mems):
            print("FAIL: archived basin for n=%d is not strictly feasible" % n); ok = False
        else:
            print("archive n=%2d members=%d best_sum=%.9f" % (n, len(mems), max(m[2] for m in mems)))
    # ---- iteration 13a: the wall-row recount conserves n, stays feasible, and really recounts
    rr = np.random.RandomState(5)
    a = _read_pack(27)
    if a is None:
        print("FAIL: self-test needs a committed n=27"); ok = False
    else:
        wxy, wr = a[:, :2].copy(), a[:, 2].copy()

        def _rowsig(xy, r):
            out = []
            for ws in range(4):
                ax, side = ws // 2, ws % 2
                sl = (xy[:, ax] - LO - r) if side == 0 else (HI - xy[:, ax] - r)
                out.append(int((sl <= TOUCH_TOL * r.mean()).sum()))
            return out
        base = _rowsig(wxy, wr)
        tries, feas, moved = 0, 0, 0
        for _ in range(40):
            got = _wallrow(wxy, wr, rr)
            if got is None:
                continue
            tries += 1
            cxy, cr = got
            if cxy.shape == (27, 2) and cr.shape == (27,) and _feas_ok(cxy, cr):
                feas += 1
            if _rowsig(cxy, cr) != base:
                moved += 1
        if tries == 0 or feas != tries:
            print("FAIL: _wallrow produced %d/%d strictly feasible children" % (feas, tries)); ok = False
        elif moved == 0:
            print("FAIL: _wallrow never changed a wall-row count"); ok = False
        else:
            print("wallrow n=27 children=%d feasible=%d rowcount-changed=%d" % (tries, feas, moved))
        # a wall-row recount on an n with NO usable row must return None, not raise
        tiny_xy = np.array([[0.0, 0.0], [0.2, 0.2], [-0.2, -0.2]])
        _wallrow(tiny_xy, _feasify(tiny_xy, np.full(3, 0.1)), rr)

    # ---- iteration 14: the SIZE LADDER carries a donor of a DIFFERENT size to exactly n
    d27 = _read_pack(27)
    if d27 is not None:
        rr = np.random.RandomState(14)
        dxy, dr = d27[:, :2].copy(), d27[:, 2].copy()
        bad = []
        for n_t in (33, 41, 21):                      # up the ladder, far up, and DOWN it
            for _ in range(3):
                cxy, cr = _ladder(dxy, dr, n_t, rr, time.process_time() + 3.0, 0.05)
                if cxy.shape != (n_t, 2) or cr.shape != (n_t,):
                    bad.append("27->%d wrong count %d" % (n_t, len(cr)))
                elif not _feas_ok(cxy, cr):
                    bad.append("27->%d infeasible" % n_t)
        # a ladder to its OWN size is the identity, and must stay feasible
        cxy, cr = _ladder(dxy, dr, 27, rr, time.process_time() + 1.0, 0.05)
        if len(cr) != 27 or not _feas_ok(cxy, cr):
            bad.append("27->27 identity rung broken")
        # a ladder with ZERO clock left must still deliver the right COUNT (feasible, unrelaxed)
        cxy, cr = _ladder(dxy, dr, 39, rr, time.process_time() - 1.0, 0.05)
        if len(cr) != 39 or not _feas_ok(cxy, cr):
            bad.append("expired-clock ladder gave %d circles / infeasible" % len(cr))
        if bad:
            print("FAIL: _ladder --", "; ".join(bad[:4])); ok = False
        else:
            print("size ladder: 27->{33,41,21,27} x3 + expired-clock 27->39 all exact-count feasible")

    # ---- iteration 13b: the novelty reserve keeps a structurally NEW but inferior basin
    class _W:
        pass
    tw = _W()
    tw.pool = [[np.zeros((9, 2)), np.full(9, 0.10) + 1e-6 * i, 0.90 - 1e-3 * i] for i in range(POOL_K)]
    ref = 0.90 - 1e-3 * (POOL_K - 1)
    novel = [np.zeros((9, 2)), np.linspace(0.05, 0.15, 9), ref - 1e-3]   # WORSE, but far in spectrum
    _admit(tw, novel)
    if not any(q[2] == novel[2] for q in tw.pool):
        print("FAIL: novelty reserve rejected a structurally new basin"); ok = False
    dup = [np.zeros((9, 2)), np.full(9, 0.10), 0.95]                     # same spectrum, better sum
    _admit(tw, dup)
    if sum(1 for q in tw.pool if abs(q[2] - 0.95) < 1e-12) != 1 or len(tw.pool) != POOL_K:
        print("FAIL: duplicate basin did not take its own incumbent's slot"); ok = False
    junk = [np.zeros((9, 2)), np.full(9, 0.01), 0.10]                    # novel but far off the pace
    _admit(tw, junk)
    if any(q[2] == 0.10 for q in tw.pool):
        print("FAIL: novelty reserve admitted a basin below NOVEL_FLOOR"); ok = False
    if ok:
        print("novelty reserve: new basin kept, duplicate merged, off-pace basin refused (K=%d)" % POOL_K)
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(__doc__)
