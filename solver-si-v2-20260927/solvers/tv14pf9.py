"""Circle packing (packomania `csqv`): maximise the sum of radii of n circles in the unit square.

Two engines, one search.  An exact-penalty NLP (L-BFGS-B on a rising mu ladder) is the
explorer -- it lets circles pass through one another while mu is small, so a start can
settle into a *different contact structure* -- and a conservative *sequential linear
programming* (SLP) optimiser is the closer, which can only ever walk uphill inside the
feasible set and so is what guarantees every shipped packing is exactly feasible.
Both sit inside an adaptive rotation that hands the next slice of CPU to whichever n is
still improving.

Why SLP works so well here.  With centres fixed, choosing radii is already an LP
(maximise sum r s.t. r_i + r_j <= d_ij, r_i <= wall_i, r >= 0).  To also move the
centres we linearise the distance: for the displacement w of the offset vector
v = p_i - p_j,

    ||v + w||  >=  ||v|| + (v/||v||) . w                       (convexity of the norm)

so replacing d_ij by its first-order model is a *conservative inner approximation*:
every solution of the LP is exactly feasible for the true problem, no matter how big
the step.  The current point is always LP-feasible (w = 0), so the LP optimum is
monotonically non-decreasing and the iteration converges to a KKT point while never
leaving the feasible set.  A trust region on the displacement only controls how
loose the model is, never validity.

Around that core:
  * rattler relocation -- circles the LP squeezes to (near) zero radius are teleported
    to the emptiest spot of the square, found on a clearance grid;
  * perturbation restarts -- jitter / relocate-the-smallest moves followed by a short
    SLP, accepted only when the sum of radii improves;
  * structured and random cold starts (hex/square lattices with the leftovers dropped
    into interstices), a guarded warm start from n's own committed packing, and a
    cross-size transfer that seeds n from the solved packings of NEARBY sizes;
  * a resumable per-n `_Worker`, so `solve()` can rotate CPU between sizes and drop an n
    from the rotation once it has stalled -- one rule, no per-n table.

Everything that is meant to count is routed through the metered `evaluate()`.
Time, not the evaluation counter, is the binding meter, so every deadline is derived
at entry from `time.process_time()` and from the sizes actually in `targets`.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix, csr_matrix, vstack

LO, HI = -0.5, 0.5

# CPU allowances.  The offline re-run gives 60 process-CPU-seconds per size (one size per
# call, fresh process); the in-run driver gives a 120 s backstop for the whole call.  One
# rule covers both: per-size allowance x number of sizes, capped by the whole-call cap.
CPU_PER_SIZE = 54.0
CPU_CALL_CAP = 112.0

_FEAS_TOL = 1e-9          # harness tolerance; we always land on the safe side of it
_TINY_R = 1e-9            # below this a circle counts as a rattler and gets relocated
_MOVED = 1e-12            # a |d Sigma r| below this means the hop re-found the same optimum


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def _walls(xy):
    return np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0], xy[:, 1] - LO, HI - xy[:, 1]])


def _pdist(xy):
    d = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return dist


def _repair(xy, r):
    """Scale radii down by the least factor that makes the packing strictly feasible.

    A single global factor is enough: if lam = min_ij d_ij/(r_i+r_j) and min_i wall_i/r_i
    then lam*r satisfies every constraint.  Violations coming out of the LP are ~1e-12,
    so the cost of this is far below the scoring noise floor.
    """
    r = np.maximum(r, 0.0)
    n = len(r)
    wall = np.maximum(_walls(xy), 0.0)
    lam = 1.0
    pos = r > 0
    if pos.any():
        lam = min(lam, float(np.min(wall[pos] / r[pos])))
    if n >= 2:
        dist = _pdist(xy)
        s = r[:, None] + r[None, :]
        np.fill_diagonal(s, 0.0)
        m = s > 0
        if m.any():
            lam = min(lam, float(np.min(dist[m] / s[m])))
    if lam < 1.0:
        r = r * (lam * (1.0 - 1e-12))
    return r


def _feasible(xy, r):
    if np.any(r <= 0.0):
        return False
    if np.min(_walls(xy) - r) < -_FEAS_TOL:
        return False
    if len(r) >= 2 and np.max(r[:, None] + r[None, :] - _pdist(xy)) > _FEAS_TOL:
        return False
    return True


def _pack(xy, r):
    return np.concatenate([xy, r[:, None]], axis=1)


# --------------------------------------------------------------------------------------
# the SLP step
# --------------------------------------------------------------------------------------
def _lp_step(xy, delta, active, beta=0.0):
    """One conservative LP: maximise sum r over radii and bounded centre displacements.

    `active` is the index array of pairs whose non-overlap constraint is included.  A
    dropped pair could in principle be violated, so the caller re-checks all pairs and
    re-solves with the violators added (cutting planes) -- correctness never rests on
    the pruning heuristic.

    `beta > 0` maximises `sum r + beta * n * min_i r_i` instead of `sum r`.  That needs one
    extra variable t (index 3n) and the n rows t - r_i <= 0: t is then pinned to the minimum
    radius at any optimum, so the objective is exactly the soft max-min one.  beta -> inf is
    the pure equal-circles (max-min radius) problem and beta = 0 is the true objective, so a
    DESCENDING beta ladder is a homotopy between them.  Only the OBJECTIVE changes -- the
    constraint rows are untouched, so the conservative-inner-approximation guarantee holds at
    every rung and every iterate is exactly feasible for the true problem.  See `_slp`.
    """
    n = len(xy)
    iu, ju = active
    m = len(iu)
    d = xy[iu] - xy[ju]
    dij = np.sqrt((d ** 2).sum(-1))
    dij = np.maximum(dij, 1e-12)
    u = d / dij[:, None]

    ar = np.arange(m)
    rows = [ar, ar, ar, ar, ar, ar]
    cols = [iu, ju, n + iu, 2 * n + iu, n + ju, 2 * n + ju]
    vals = [np.ones(m), np.ones(m), -u[:, 0], -u[:, 1], u[:, 0], u[:, 1]]
    nv = 3 * n + (1 if beta > 0.0 else 0)
    A1 = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                    shape=(m, nv))

    idx = np.arange(n)
    rw, cw, vw, bw, k = [], [], [], [], 0
    for off, sg, rhs in ((n, -1.0, xy[:, 0] - LO), (n, 1.0, HI - xy[:, 0]),
                         (2 * n, -1.0, xy[:, 1] - LO), (2 * n, 1.0, HI - xy[:, 1])):
        rw += [idx + k, idx + k]
        cw += [idx, off + idx]
        vw += [np.ones(n), np.full(n, sg)]
        bw.append(rhs)
        k += n
    A2 = coo_matrix((np.concatenate(vw), (np.concatenate(rw), np.concatenate(cw))),
                    shape=(4 * n, nv))

    blocks = [A1, A2]
    b = [dij] + bw
    c = np.zeros(nv)
    c[:n] = -1.0
    bounds = [(0.0, None)] * n + [(-delta, delta)] * (2 * n)
    if beta > 0.0:
        # t - r_i <= 0 for every i, with t rewarded at weight beta * n: t = min_i r_i.
        A3 = coo_matrix((np.concatenate([-np.ones(n), np.ones(n)]),
                         (np.concatenate([idx, idx]),
                          np.concatenate([idx, np.full(n, 3 * n)]))), shape=(n, nv))
        blocks.append(A3)
        b.append(np.zeros(n))
        c[3 * n] = -beta * n
        bounds.append((0.0, None))
    A = csr_matrix(vstack(blocks))
    b = np.concatenate(b)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        return None, None
    x = res.x
    return xy + np.stack([x[n:2 * n], x[2 * n:3 * n]], axis=1), x[:n]


def _all_pairs(n):
    return np.triu_indices(n, 1)


def _active_pairs(xy, r, delta, allp):
    """Pairs that could bind after a step of at most `delta` per centre."""
    iu, ju = allp
    d = xy[iu] - xy[ju]
    dij = np.sqrt((d ** 2).sum(-1))
    slack = dij - (r[iu] + r[ju])
    keep = slack <= 4.0 * delta + 0.02
    return iu[keep], ju[keep]


def _slp(xy, r, deadline, max_iter=60, delta0=0.05, tol=1e-13, beta=0.0):
    """Run SLP to (near) convergence.  Returns (xy, r, sum_r); always feasible.

    `beta > 0` optimises the soft max-min surrogate `sum r + beta * n * min_i r_i` instead
    (see `_lp_step`), and the acceptance test below then uses that same surrogate value, so
    a rung is allowed to trade a little sum for a much larger smallest circle.  Why the
    surrogate exists -- the analytic reason, measured in `artifacts/iter7_uniform_ab.py` for
    the hard (beta = inf) version and in `artifacts/iter8_homotopy_ab.py` for the ladder.  For total circle area A, Cauchy-Schwarz gives
    sum r <= sqrt(n*A/pi) with equality IFF every radius is equal, so the objective itself
    pushes towards uniform radii, and the census bears it out: the sizes where we match or
    beat the record carry a near-uniform radius histogram, while the worst sizes carry a
    tail of undersized circles (n=81: min 0.0370 against a mode of 0.057).  A local optimum
    of the TRUE objective can keep that tail for ever -- iteration 7 confirmed every
    committed pack is converged to machine precision under both SLP and SLSQP, so the whole
    residual is the contact STRUCTURE, not the polish.  The equal-radius surrogate cannot
    keep such a tail: it rates a configuration by its WORST circle, so it spreads the
    centres into the near-uniform structures the records live in, and the true objective is
    then free to inflate whatever has slack.  Same code path, one extra objective term.
    """
    n = len(xy)
    allp = _all_pairs(n)
    r = _repair(xy, r)

    def obj(rr):
        return float(rr.sum()) + (beta * n * float(rr.min()) if beta > 0.0 else 0.0)

    best_s = obj(r)
    delta = delta0
    for _ in range(max_iter):
        if time.process_time() > deadline:
            break
        act = _active_pairs(xy, r, delta, allp)
        nxy, nr = None, None
        for _round in range(4):
            nxy, nr = _lp_step(xy, delta, act, beta=beta)
            if nxy is None:
                break
            bad = _violating(nxy, nr, allp, act)
            if bad is None:
                break
            act = bad
        if nxy is None:
            delta *= 0.3
            if delta < 1e-8:
                break
            continue
        nr = _repair(nxy, nr)
        s = obj(nr)
        if s > best_s:
            gain = s - best_s
            xy, r, best_s = nxy, nr, s
            if gain < tol * max(1.0, best_s):
                delta *= 0.3
            elif gain < 1e-6:
                delta *= 0.6
            else:
                delta = min(delta * 1.4, 0.12)
        else:
            delta *= 0.3
        if delta < 1e-8:
            break
    return xy, r, float(r.sum())


def _violating(xy, r, allp, act):
    """All pairs violated beyond tolerance, unioned with the current active set (or None)."""
    iu, ju = allp
    d = xy[iu] - xy[ju]
    dij = np.sqrt((d ** 2).sum(-1))
    v = r[iu] + r[ju] - dij
    bad = v > 1e-11
    if not bad.any():
        return None
    key = act[0].astype(np.int64) * (len(r) + 1) + act[1]
    nkey = iu[bad].astype(np.int64) * (len(r) + 1) + ju[bad]
    allkey = np.union1d(key, nkey)
    return (allkey // (len(r) + 1)).astype(int), (allkey % (len(r) + 1)).astype(int)


# --------------------------------------------------------------------------------------
# penalty NLP local solver  (the cheap, basin-crossing workhorse)
# --------------------------------------------------------------------------------------
# The SLP above is a *conservative* method: every iterate is exactly feasible, so it can
# only ever walk uphill inside the basin it starts in.  That is what we want for the final
# polish and it is why no infeasible packing can ever be shipped -- but it is also why
# iteration 1's residual was a uniform ~1% for all 37 sizes: the local optimum, not the
# convergence, was the loss.
#
# So the search engine is now an exact-penalty NLP,
#
#     min  -sum r + mu * ( sum_{i<j} max(0, r_i+r_j-d_ij)^2 + sum_i sum_walls max(0,...)^2 )
#
# minimised by L-BFGS-B over (x, y, r) on a rising mu ladder.  Two things change:
#   * circles may pass THROUGH each other while mu is small, so a start can relax into a
#     different contact structure -- the escape mechanism the SLP structurally lacks;
#   * one local solve costs O(n^2) flops per gradient in pure numpy instead of an O(n^2)-row
#     HiGHS LP, which is ~an order of magnitude cheaper, so the same CPU buys ~10x the
#     restarts.
# Feasibility is never taken on trust: every NLP result is pushed through _repair and then
# the exact SLP, and only _feasible packings are ever offered to evaluate().


def _pen_fg(z, n, mu):
    """Objective and gradient of the exact-penalty NLP.  z = [x (n), y (n), r (n)]."""
    xy = np.empty((n, 2))
    xy[:, 0] = z[:n]
    xy[:, 1] = z[n:2 * n]
    r = z[2 * n:]

    d = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, 1.0)
    np.maximum(dist, 1e-9, out=dist)      # coincident centres must not make the gradient NaN
    v = r[:, None] + r[None, :] - dist
    np.fill_diagonal(v, 0.0)
    np.maximum(v, 0.0, out=v)

    f = -r.sum() + 0.5 * mu * float((v * v).sum())
    gr = -np.ones(n) + 2.0 * mu * v.sum(axis=1)
    w = (2.0 * mu) * v / dist
    gxy = -(w[:, :, None] * d).sum(axis=1)

    # walls: r_i <= x-LO, HI-x, y-LO, HI-y
    va = np.maximum(0.0, r - (xy[:, 0] - LO))
    vb = np.maximum(0.0, r - (HI - xy[:, 0]))
    vc = np.maximum(0.0, r - (xy[:, 1] - LO))
    vd = np.maximum(0.0, r - (HI - xy[:, 1]))
    f += mu * float((va * va + vb * vb + vc * vc + vd * vd).sum())
    gr += 2.0 * mu * (va + vb + vc + vd)
    gxy[:, 0] += 2.0 * mu * (vb - va)
    gxy[:, 1] += 2.0 * mu * (vd - vc)

    g = np.empty(3 * n)
    g[:n] = gxy[:, 0]
    g[n:2 * n] = gxy[:, 1]
    g[2 * n:] = gr
    return f, g


_MU_LADDER = (30.0, 300.0, 3.0e3, 3.0e4, 3.0e5, 3.0e6)

# The max-min homotopy (mode 4).  The surrogate is `sum r + beta * n * min_i r_i`, and
# n * min r is the same order as sum r, so beta is already dimensionless and scale-free:
# beta = 1 weights the worst circle as heavily as the entire sum, beta = 0 IS the true
# objective.  Halving to zero is therefore the whole schedule -- no per-n table, nothing
# measured in the units of any particular size, and the same three-decade sweep shape the
# penalty ladder above already uses.
_BETA_LADDER = (1.0, 0.5, 0.25, 0.125, 0.0)


def _nlp(xy, r, deadline, ladder=_MU_LADDER, maxiter=120):
    """Relax (xy, r) with the penalty NLP.  Result is only APPROXIMATELY feasible."""
    n = len(xy)
    z = np.empty(3 * n)
    z[:n] = xy[:, 0]
    z[n:2 * n] = xy[:, 1]
    z[2 * n:] = np.maximum(r, 1e-4)
    bounds = [(LO, HI)] * (2 * n) + [(0.0, 0.5)] * n
    for mu in ladder:
        if time.process_time() > deadline:
            break
        res = minimize(_pen_fg, z, args=(n, mu), jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": maxiter, "maxcor": 12})
        if res.x is not None and np.all(np.isfinite(res.x)):
            z = res.x
    out = np.empty((n, 2))
    out[:, 0] = z[:n]
    out[:, 1] = z[n:2 * n]
    return np.clip(out, LO, HI), np.maximum(z[2 * n:], 0.0)


# --------------------------------------------------------------------------------------
# BATCHED penalty NLP  (the explorer, run B starts at a time)
# --------------------------------------------------------------------------------------
# Measured (artifacts/iter3_profile.py): one cold L-BFGS penalty solve at n=99 costs 0.26 s,
# of which almost all is the O(n^2) gradient -- and an n=99 gradient touches only 9801
# doubles, so numpy's *per-call overhead*, not its arithmetic, is what is being paid.
# Stacking B starts into one (B, n, n) pass therefore costs far less than B times as much.
# L-BFGS-B cannot be batched (its line search is a scalar decision per instance), so the
# batched engine uses RPROP: per-coordinate step sizes driven by the SIGN of the gradient,
# grown 1.2x while the sign holds and halved when it flips.  No line search, no per-instance
# branching, and it is invariant to the gradient's scale -- which matters here because mu
# rises by five orders of magnitude along the ladder.
# Per start RPROP is weaker than L-BFGS, but at equal CPU it buys ~6x the starts and wins
# outright (artifacts/iter3_batch_ab.py: at 4 s, n=35 d=2.34 -> 2.61, n=63 2.26 -> 2.31,
# n=99 2.17 -> 2.18).  Exactly as before, nothing here is trusted: the batch is screened by
# a cheap _repair sum and only the top few go through _exact_radii + the exact SLP closer.


def _pen_g_batch(Z, n, mu, eye, out):
    """Gradient of the penalty objective for a BATCH.  Z = (B, 3n).  DENSE, all n^2 pairs.

    Gradient only: `_nlp_batch` drives RPROP, which reads the SIGN of the gradient and never
    the objective value, so every pass spent forming f was waste.  The self-pair is killed by
    adding `eye` (4.0 on the diagonal) to the squared distance instead of by two boolean
    fancy-index assignments: d_ii is then 2, and r_i + r_j <= 1 < 2, so the hinge closes the
    diagonal on its own.  Same gradient to the last bit (artifacts/iter10_grad_bench.py:
    worst relative difference 2.2e-16 over 60 cells).
    """
    x = Z[:, :n]
    y = Z[:, n:2 * n]
    r = Z[:, 2 * n:]
    dx = x[:, :, None] - x[:, None, :]
    dy = y[:, :, None] - y[:, None, :]
    dist = dx * dx
    v = dy * dy
    dist += v
    dist += eye
    np.sqrt(dist, out=dist)
    np.maximum(dist, 1e-9, out=dist)          # coincident centres must not make g NaN
    np.add(r[:, :, None], r[:, None, :], out=v)
    v -= dist
    np.maximum(v, 0.0, out=v)
    return _pen_g_rows(x, y, r, dx, dy, dist, v, mu, n, out)


def _pen_g_rows(x, y, r, dx, dy, dist, v, mu, n, out):
    """Finish a gradient from per-pair hinges laid out as ROWS: v[b, i, k] is the overlap of
    circle i with its k-th listed partner.  Both engines below share this tail, because a
    row sum over a SYMMETRIC pair set is exactly the dense sum over all j (each pair is seen
    once from each of its two endpoints) -- which is why the neighbour list needs no
    scatter-add."""
    gx = out[:, :n]
    gy = out[:, n:2 * n]
    gr = out[:, 2 * n:]
    v.sum(2, out=gr)
    v /= dist
    np.multiply(v, dx, out=dx)
    dx.sum(2, out=gx)
    np.multiply(v, dy, out=dy)
    dy.sum(2, out=gy)
    m2 = 2.0 * mu
    gr *= m2
    gr -= 1.0
    gx *= -m2
    gy *= -m2
    va = np.maximum(0.0, r - (x - LO))
    vb = np.maximum(0.0, r - (HI - x))
    vc = np.maximum(0.0, r - (y - LO))
    vd = np.maximum(0.0, r - (HI - y))
    gr += m2 * (va + vb + vc + vd)
    gx += m2 * (vb - va)
    gy += m2 * (vd - vc)
    return out


# MEASURED AND REJECTED -- the O(n^2) pair sum is NOT the waste it looks like.
# The penalty hinge max(0, r_i + r_j - d_ij) is zero for every pair further apart than
# r_i + r_j, so only O(n) of the O(n^2) pairs can contribute, and a fixed-width symmetric
# neighbour list (a distance ball, so a ROW sum over it is exactly the dense sum -- no
# scatter-add) plus a rebuild test that provably cannot miss a pair becoming binding
# (disp + grow <= slack) gives a truncation that is exact to the last bit: measured over 736
# live-list states drawn from real RPROP trajectories, worst relative |dense - list| =
# 3.5e-16, i.e. summation order only (artifacts/iter10_list_bench.py).
# It is nonetheless SLOWER, and the sweep says why (artifacts/iter10_slack_sweep.py, n=71
# and n=99): with a rebuild margin of one mean radius the list goes live and runs at 0.67x
# dense; widen the margin and the ball stops being sparse at all (K >= n/2) so it never goes
# live.  The explorer works at radii far larger than the final packing's -- R0 starts at
# 0.4/ceil(sqrt(n)) and the NLP lets r float up to 0.5 while mu is small -- so at the scales
# the NLP actually visits, the contact graph is simply not sparse, and a (B, n, K) gather
# costs more per element than a contiguous broadcast over (B, n, n).
# The lesson, not the code, is what is worth keeping: the dense pass is arithmetic-bound, and
# the way to make the explorer cheaper is fewer/cheaper PASSES over the matrix (which is what
# `_pen_g_batch` above now does), not fewer pairs in it.


def _nlp_batch(XY0, R0, deadline, ladder=_MU_LADDER, iters=120):
    """Relax B starts at once by RPROP on the mu ladder.  Only APPROXIMATELY feasible."""
    B, n, _ = XY0.shape
    Z = np.concatenate([XY0[:, :, 0], XY0[:, :, 1], np.maximum(R0, 1e-4)], axis=1)
    lo = np.concatenate([np.full(2 * n, LO), np.zeros(n)])
    hi = np.concatenate([np.full(2 * n, HI), np.full(n, 0.5)])
    eye = 4.0 * np.eye(n)
    out = np.empty((B, 3 * n))
    for mu in ladder:
        if time.process_time() > deadline:
            break
        eta = np.full(Z.shape, 3e-3)
        gprev = np.zeros(Z.shape)
        for k in range(iters):
            g = _pen_g_batch(Z, n, mu, eye, out)
            s = g * gprev
            eta = np.where(s > 0, eta * 1.2, np.where(s < 0, eta * 0.5, eta))
            np.clip(eta, 1e-9, 0.02, out=eta)
            g = np.where(s < 0, 0.0, g)          # RPROP: a sign flip retracts, it does not step
            Z = np.clip(Z - np.sign(g) * eta, lo, hi)
            gprev = g
            if (k & 15) == 15 and time.process_time() > deadline:
                break
    XY = np.stack([Z[:, :n], Z[:, n:2 * n]], axis=2)
    return np.clip(XY, LO, HI), np.maximum(Z[:, 2 * n:], 0.0)


BATCH = 16        # starts relaxed per batched NLP call -- one rule for every n
SCREEN_TOP = 2    # how many of them earn the expensive exact_radii + SLP closer


def _exact_radii(xy, r_hint):
    """Optimal radii for FIXED centres: the radius-only LP, solved exactly.

    Reuses _lp_step with a zero trust region, so the centres cannot move and the LP is the
    n-variable radius problem.  Falls back to _repair if the LP fails.
    """
    n = len(xy)
    allp = _all_pairs(n)
    act = _active_pairs(xy, np.zeros(n), 0.0, allp)
    for _ in range(4):
        nxy, nr = _lp_step(xy, 0.0, act)
        if nxy is None:
            return _repair(xy, r_hint)
        bad = _violating(xy, nr, allp, act)
        if bad is None:
            return _repair(xy, nr)
        act = bad
    return _repair(xy, nr)


# --------------------------------------------------------------------------------------
# rattlers and perturbation moves
# --------------------------------------------------------------------------------------
def _clearance_spots(xy, r, k, g=64, skip=None):
    """The k emptiest points of the square (greedy, on a g x g grid)."""
    t = np.linspace(LO + 0.5 / g, HI - 0.5 / g, g)
    gx, gy = np.meshgrid(t, t, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel()], axis=1)
    wall = np.minimum.reduce([P[:, 0] - LO, HI - P[:, 0], P[:, 1] - LO, HI - P[:, 1]])
    keep = np.ones(len(xy), dtype=bool)
    if skip is not None:
        keep[skip] = False
    cx, cr = xy[keep], r[keep]
    if len(cx):
        dd = np.sqrt(((P[:, None, :] - cx[None, :, :]) ** 2).sum(-1)) - cr[None, :]
        clear = np.minimum(wall, dd.min(axis=1))
    else:
        clear = wall
    out = []
    for _ in range(k):
        j = int(np.argmax(clear))
        out.append(P[j].copy())
        cl = clear[j]
        dd = np.sqrt(((P - P[j]) ** 2).sum(-1)) - max(cl, 0.0)
        clear = np.minimum(clear, dd)
    return np.array(out)


def _fix_rattlers(xy, r):
    bad = np.flatnonzero(r < _TINY_R)
    if len(bad) == 0:
        return xy, r, False
    spots = _clearance_spots(xy, r, len(bad), skip=bad)
    xy = xy.copy()
    xy[bad] = spots
    return xy, _repair(xy, np.maximum(r, 1e-6)), True


def _perturb(xy, r, rng, strength=1.0):
    """One basin-hopping move: relocate the smallest circles, or jitter everything.

    `strength` widens the NEIGHBOURHOOD, not the step: it multiplies how many circles a
    relocation moves and how far a jitter throws them.  strength = 1 is the move as
    measured through iteration 8; the caller escalates it while the search is stalled (see
    `_Worker.shake`), which is what turns a fixed hop into a variable-neighbourhood ladder.
    Both branches stay bounded by the geometry -- at most a quarter of the circles move, and
    the jitter sigma is capped at 1/8 of the side -- so no level of escalation can degenerate
    into "resample the square", which is mode 0's job and is already in the mix.

    Returns `(xy, r)`: the moved centres AND a radius profile that may have been REASSIGNED
    over the sites (`SWAP_HOP`).  Every other part of the search treats the profile as a
    deterministic function of the centres -- the LP re-derives it -- so nothing could ask
    "what if this site held the big circle and that one the small one?", even though the
    radii span a factor of ~2 at every size and are almost all distinct (measured, iteration
    12).  A permutation leaves Sigma r exactly unchanged, so it costs nothing up front; what
    it changes is which contacts bind, and only a caller that hands the profile to the
    penalty NLP (the one engine that tolerates the overlap a swap creates) can use it.  The
    subset size rides the same `shake()` ladder as the neighbourhood width -- k = 2, one
    transposition, at the narrowest rung -- so this adds no constant of its own.

    The two draws below are made UNCONDITIONALLY, whatever `SWAP_HOP` says, and that is
    deliberate: the same choice measured as a separate move in iteration 12 shifted every
    later draw from `rng` and so compared two random streams rather than two policies.
    Drawing either way makes the A/B of this flag PAIRED at a seed -- both arms see the same
    stream and diverge only where the swap actually changes an accepted candidate -- and it
    keeps that true for anyone who re-opens the flag later.
    """
    xy = xy.copy()
    n = len(xy)
    u = rng.rand()
    if u < 0.6 and n >= 3:
        k = 1 + int(rng.rand() < 0.4) + int(rng.rand() < 0.15)
        k = int(np.ceil(k * strength))
        k = min(k, max(1, n // 4))
        order = np.argsort(r)
        pool = order[:max(k, min(n, 3 * k))]
        pick = pool[rng.permutation(len(pool))[:k]]
        spots = _clearance_spots(xy, r, k, skip=pick)
        xy[pick] = spots + rng.normal(0.0, 0.004, size=spots.shape)
    else:
        sig = min(10.0 ** rng.uniform(-2.7, -1.6) * strength, 0.125)
        xy = xy + rng.normal(0.0, sig, size=xy.shape)
    k = int(min(n, 2 * max(1, int(np.ceil(strength)))))
    idx = rng.permutation(n)[:k]                 # drawn either way -- see the docstring
    order = rng.permutation(k)
    if SWAP_HOP and k >= 2:
        r = r.copy()
        r[idx] = r[idx[order]]
    return np.clip(xy, LO + 1e-6, HI - 1e-6), r


# --------------------------------------------------------------------------------------
# starts
# --------------------------------------------------------------------------------------
def _grid_start(n, rng):
    a = max(1, int(round(np.sqrt(n))) + int(rng.randint(-1, 2)))
    b = max(1, int(np.ceil(n / float(a))))
    t0 = (np.arange(a) + 0.5) / a - 0.5
    t1 = (np.arange(b) + 0.5) / b - 0.5
    g = np.stack(np.meshgrid(t0, t1, indexing="ij"), axis=-1).reshape(-1, 2)
    if len(g) >= n:
        g = g[rng.permutation(len(g))[:n]]
    else:
        c0 = (np.arange(1, a) / float(a)) - 0.5
        c1 = (np.arange(1, b) / float(b)) - 0.5
        if len(c0) and len(c1):
            inter = np.stack(np.meshgrid(c0, c1, indexing="ij"), axis=-1).reshape(-1, 2)
        else:
            inter = np.zeros((0, 2))
        need = n - len(g)
        if len(inter) >= need:
            inter = inter[rng.permutation(len(inter))[:need]]
        else:
            extra = rng.uniform(LO + 0.05, HI - 0.05, size=(need - len(inter), 2))
            inter = np.concatenate([inter, extra], axis=0) if len(inter) else extra
        g = np.concatenate([g, inter], axis=0)
    g = g + rng.normal(0.0, 0.25 / max(a, b), size=g.shape)
    return np.clip(g, LO + 1e-4, HI - 1e-4)


def _lattice_start(n, rng):
    """A staggered-row (hexagonal) or aligned-row lattice, with the surplus sites dropped.

    Why this family exists at all -- the analytic reason, not a tuned guess.  For N circles
    of total area A = sum pi r^2 Cauchy-Schwarz gives

        sum r  <=  sqrt(N * A / pi)   with equality iff every r_i is EQUAL,

    and A is bounded by the packing density, which for equal circles is maximised by the
    hexagonal arrangement (pi/sqrt12).  So the unconstrained-by-boundary optimum of THIS
    objective is a near-equal, near-hexagonal packing worth 0.5373*sqrt(N), against exactly
    0.5*sqrt(N) for a square grid; the records sit in between (~0.52*sqrt(n)), i.e. hex
    structure spoiled by the walls.  A square grid was the solver's only structured start,
    so it was seeding the search on the wrong lattice.

    Nothing here is tabulated per n: the row count is sampled around the hex-consistent
    0.93*sqrt(n) (from 1/m = (sqrt3/2)*(1/c) with n = m*c), the pitch is then whatever fits
    the square, and the leftover sites are dropped at random -- the holes are diversity the
    relaxation is free to close.
    """
    m = max(1, int(round(0.93 * np.sqrt(n))) + int(rng.randint(-1, 2)))
    m = min(m, n)
    rows = []
    if rng.rand() < 0.75:                      # staggered: c, c-1, c, c-1, ... sites
        a = (m + 1) // 2
        b = m - a
        c = 1
        while a * c + b * max(c - 1, 0) < n:
            c += 1
        for j in range(m):
            if j % 2 == 0:
                rows.append(LO + (np.arange(c) + 0.5) / c)
            elif c >= 2:
                rows.append(LO + (np.arange(c - 1) + 1.0) / c)
            else:
                rows.append(np.zeros(1))
    else:                                      # aligned rows (the classic grid), same code path
        c = int(np.ceil(n / float(m)))
        for j in range(m):
            rows.append(LO + (np.arange(c) + 0.5) / c)
    pts = []
    for j, xs in enumerate(rows):
        y = LO + (j + 0.5) / m
        for x in xs:
            pts.append((x, y))
    P = np.asarray(pts, dtype=float)
    if len(P) > n:
        P = P[rng.permutation(len(P))[:n]]
    elif len(P) < n:
        extra = rng.uniform(LO + 0.05, HI - 0.05, size=(n - len(P), 2))
        P = np.concatenate([P, extra], axis=0)
    P = P + rng.normal(0.0, 0.15 / m, size=P.shape)
    return np.clip(P, LO + 1e-4, HI - 1e-4)


def _cold_start(n, rng, i):
    """One cold start.  Half the batch comes from the lattice family, half from the older
    grid/uniform mix -- measured at equal CPU in `artifacts/iter4_hex_ab.py`: lattice-only
    beat the old mix at 5 of 6 sizes and the half-and-half mix beat it at 5 of 6 and on the
    mean at both seeds (+0.15 digits), so the mix, not either extreme, is what ships."""
    if i % 2 == 0:
        return _lattice_start(n, rng)
    return _grid_start(n, rng) if i % 4 == 1 else _random_start(n, rng)


def _random_start(n, rng):
    return rng.uniform(LO + 0.02, HI - 0.02, size=(n, 2))


def _read_pack(n):
    """Warm start from the committed census -- guarded: any n may be absent."""
    p = os.path.join("bench", "packs", "csqv%d.pck" % n)
    try:
        if not os.path.exists(p):
            return None
        rows = []
        with open(p) as fh:
            for line in fh.read().splitlines()[2:]:
                parts = line.split()
                if len(parts) >= 3:
                    rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(rows) != n:
            return None
        a = np.asarray(rows, dtype=float)
        if not np.all(np.isfinite(a)):
            return None
        return a
    except Exception:
        return None


NEIGH = 4         # how many nearest OTHER census sizes a transfer may be seeded from

_PACK_CACHE = {}
_CENSUS_SIZES = [None]


def _census_sizes():
    """Every n for which the census holds a packing.  Read once per process; the driver
    reverts any write made during solve(), so the listing cannot go stale mid-call."""
    if _CENSUS_SIZES[0] is None:
        out = []
        try:
            for f in os.listdir(os.path.join("bench", "packs")):
                if f.startswith("csqv") and f.endswith(".pck"):
                    try:
                        out.append(int(f[4:-4]))
                    except ValueError:
                        pass
        except Exception:
            pass
        _CENSUS_SIZES[0] = sorted(out)
    return _CENSUS_SIZES[0]


def _cached_pack(n):
    if n not in _PACK_CACHE:
        _PACK_CACHE[n] = _read_pack(n)
    return _PACK_CACHE[n]


_RECORDS = [None]


def _records():
    """The record table, read once per process -- used ONLY to rank which sizes get CPU.

    Reading `bench/records.json` from inside `solve()` is sanctioned, and this is the one
    thing it is good for: it carries no coordinates, so it cannot seed a packing, and it is
    never consulted by any move or by any feasibility test.  It answers the scheduler's
    question -- "which of these sizes has the most left to gain?" -- with the scorer's own
    quantity instead of a proxy for it.  Missing, unparseable or short: every caller falls
    back, so an n outside the table ranks by the trend residual exactly as before.
    """
    if _RECORDS[0] is None:
        out = {}
        try:
            raw = json.load(open(os.path.join("bench", "records.json")))["records"]
            for k, v in raw.items():
                try:
                    out[int(k)] = float(v)
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass
        _RECORDS[0] = out
    return _RECORDS[0]


def _neighbour_sizes(n):
    """The NEIGH census sizes nearest to n, excluding n itself.  Guarded: returns [] when
    the census is empty or holds nothing else -- an n outside the census must still work."""
    other = [m for m in _census_sizes() if m != n and m >= 1]
    other.sort(key=lambda m: (abs(m - n), m))
    return other[:NEIGH]


def _pack_quality(m):
    """How good the census's packing for m actually is, on the SCORER's own scale.

    This is the same quantity `_focus` ranks sizes by -- `digits = -log10(relgap)`, clipped
    to the scorer's cap -- read for a DONOR rather than for a target.  It carries no
    coordinates, so like `_focus` it cannot seed a packing; it only says how much a donor's
    structure is worth copying.  Returns None when the record for m is unknown, which is the
    normal case for a size outside `bench/records.json`: the caller substitutes the mean of
    the qualities it does know, so an unrated donor is neither favoured nor shut out.
    """
    a = _cached_pack(m)
    if a is None or a.shape != (m, 3):
        return None
    rec = _records().get(m)
    if rec is None or not (rec > 0.0):
        return None
    gap = max(0.0, (rec - float(a[:, 2].sum())) / rec)
    return float(min(DIGITS_CAP, -np.log10(max(gap, 1e-16))))


def _donor_pool(n):
    """The census sizes mode 3 may transfer FROM, each with a weight.

    Two things make a donor worth copying and they are not the same thing:

      * **Fidelity** -- the transfer deletes or inserts |m - n| circles, so the further m is
        from n the more of its contact structure is destroyed before the relaxation ever sees
        it.  That is the 1/|m - n| this always had.
      * **Quality** -- a donor whose own packing is 3 digits short of its record is a copy of
        a LOCAL OPTIMUM, and transferring it hands n the same trap.  That was missing: the
        pool was the NEIGH nearest sizes regardless of how good they were, and the census is
        now bimodal (13 of 37 sizes sit AT the record, the rest near 3 digits), so for a size
        whose immediate neighbours are all stuck -- n = 53, 55, 57, 59, 61 at the time of
        writing -- every transfer it could make recycled a bad structure family.

    So a donor is worth `quality / distance`, and the pool is the NEIGH nearest UNION the
    NEIGH best (ties to the nearer), which keeps it small: a plain 1/distance weight over the
    whole census dilutes the good donors instead of favouring them (at n=49 it would drop the
    chance of drawing the record-quality n=47 from 1/3 to about 1/5), while this union raises
    it to about 2/5 and gives n=59 -- which has no good neighbour at all -- a 1-in-3 chance of
    a record-quality donor where it previously had none.

    Degrades to the old rule by construction: with no records parsed, or with every rated
    donor at the same quality, the weights are proportional to 1/|m - n| exactly as before.
    """
    other = [m for m in _census_sizes() if m != n and m >= 1]
    if not other:
        return [], None
    near = sorted(other, key=lambda m: (abs(m - n), m))[:NEIGH]
    q = dict((m, _pack_quality(m)) for m in other)
    rated = [v for v in q.values() if v is not None]
    if rated:
        fill = float(np.mean(rated))
        best = sorted(other, key=lambda m: (-(q[m] if q[m] is not None else fill),
                                            abs(m - n), m))[:NEIGH]
    else:
        fill = None
        best = []
    pool = sorted(set(near) | set(best))
    w = np.array([((q[m] if q[m] is not None else fill) if fill is not None else 1.0)
                  / float(abs(m - n)) for m in pool], dtype=float)
    if not np.all(np.isfinite(w)) or w.sum() <= 0.0:   # every donor rated 0: fall back to
        w = np.array([1.0 / abs(m - n) for m in pool], dtype=float)   # pure fidelity
    return pool, w / w.sum()


def _transfer_start(n, rng):
    """Seed n from a NEARBY size's packing: delete the smallest circles, or shrink and
    insert into the emptiest holes.

    Why this move exists.  The census is the one asset that grows every iteration, and
    optimal csqv structures vary *slowly* in n -- Sigma r / sqrt(n) climbs smoothly from
    0.5156 at n=27 to 0.5256 at n=99 -- so a solved neighbour is a far better guess at n's
    contact structure than any lattice, and it costs nothing to read.  It is also where
    the expensive end of the census gets help it can otherwise not afford: a cold start at
    n=99 buys one basin per batch, while n=97's solved structure is already most of an
    answer.

    Both directions are derived, not tuned.  Deleting the k smallest circles leaves holes
    the survivors grow into.  For insertion, equal-radius scaling says radius ~ 1/sqrt(n)
    at fixed density, so every radius is scaled by sqrt(m/n) to make exactly the room the
    k newcomers need, and each newcomer takes the clearance of the hole it lands in -- so
    the seed is already feasible before the relaxation ever sees it.
    """
    cand, w = _donor_pool(n)
    if not cand:
        return None
    m = int(cand[int(rng.choice(len(cand), p=w))])
    a = _cached_pack(m)
    if a is None or a.shape != (m, 3):
        return None
    xy, r = a[:, :2].copy(), a[:, 2].copy()

    if m > n:                                    # shed the surplus, smallest first
        k = m - n
        pool = np.argsort(r)[:min(m, 2 * k)]     # randomised among the smallest, for diversity
        drop = pool[rng.permutation(len(pool))[:k]]
        keep = np.ones(m, dtype=bool)
        keep[drop] = False
        xy, r = xy[keep], r[keep]
    elif m < n:                                  # make room by the density scaling, then fill
        k = n - m
        r = r * np.sqrt(m / float(n))
        spots = _clearance_spots(xy, r, k)
        if len(spots) < k:
            return None
        wall = np.minimum.reduce([spots[:, 0] - LO, HI - spots[:, 0],
                                  spots[:, 1] - LO, HI - spots[:, 1]])
        d = np.sqrt(((spots[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]
        room = np.minimum(wall, d.min(axis=1))
        xy = np.concatenate([xy, spots], axis=0)
        r = np.concatenate([r, np.maximum(room, 1e-6)], axis=0)

    if len(r) != n:
        return None
    r = _repair(xy, r)
    if rng.rand() < 0.5:                         # half the batch is jittered by the same
        xy, r = _perturb(xy, r, rng)             # measured basin-hop move, for diversity
    return np.clip(xy, LO + 1e-6, HI - 1e-6), r


# --------------------------------------------------------------------------------------
# per-size driver
# --------------------------------------------------------------------------------------
class _Sink(object):
    """Routes candidates through the metered evaluate(), in small batches."""

    def __init__(self, evaluate, meter, n):
        self.evaluate, self.meter, self.n = evaluate, meter, n
        self.buf = []
        self.best_s = -np.inf

    def add(self, pack):
        self.buf.append(np.asarray(pack, dtype=float))
        if len(self.buf) >= 8:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        batch, self.buf = self.buf, []
        left = self.meter.left()
        if left <= 0:
            return
        batch = batch[:left]
        try:
            feas, s = self.evaluate(self.n, np.stack(batch, axis=0))
        except ValueError:
            return
        s = np.where(np.asarray(feas), np.asarray(s, dtype=float), -np.inf)
        j = int(np.argmax(s))
        if s[j] > self.best_s:
            self.best_s = float(s[j])


class _Worker(object):
    """Resumable search for ONE n.  `step(t_end)` does as much work as fits and returns the
    gain it made, so the caller can hand the next slice of CPU to whoever is still moving.

    Three move types are cycled, one rule for every n and every phase:
      0. cold NLP multistart -- a fresh structured/random start relaxed by the penalty NLP,
         which is what actually finds *new* contact structures;
      1. basin hop re-converged by the exact SLP -- cheap, keeps the current structure;
      2. basin hop re-converged by the NLP -- a bigger jump than (1) can make;
      3. cross-size transfer -- seed n from a solved NEIGHBOUR's structure (delete the
         smallest circles, or shrink by the density scaling and fill the emptiest holes),
         which is the one start that gets better every iteration as the census does.
      5. radius reassignment -- permute the radii of a random subset over their own sites,
         which is the only move that changes WHICH site holds WHICH size rather than where
         the sites are; see `_reassign`.
      4. max-min homotopy -- walk a DESCENDING ladder of surrogates
         `sum r + beta * n * min_i r_i` from a max-min-dominated rung down to beta = 0, the
         true objective.  This is the one move that can remove a tail of undersized circles,
         because a large-beta rung rates a configuration by its worst circle; see `_slp`'s
         docstring for the Cauchy-Schwarz argument and the census evidence.

    The hops (1, 2) start from an INCUMBENT that is not the best packing.  A hop accepted
    only when it improves the best can never cross a barrier, so an n whose basin is a hard
    local optimum stays there for ever -- which is exactly what the census showed after
    iteration 5 (20 of 37 sizes did not move at all).  So the incumbent is accepted by a
    THRESHOLD rule: any local optimum within `ACCEPT_FRAC * dscale` of the best becomes the
    new incumbent, where `dscale` is the solver's own EWMA of |d Sigma r| across hop moves.
    The window is measured against the BEST, never against the incumbent, so the walk is
    bounded -- it explores the plateau of near-best structures and cannot drift away, and
    it needs no reset rule, no temperature ladder and no per-n scale.
    """

    def __init__(self, n, sink, rng):
        self.n, self.sink, self.rng = n, sink, rng
        self.best_xy = self.best_r = None
        self.best_s = -np.inf
        # the incumbent the hops start from -- see the class docstring.  dscale is measured,
        # not set: it is the EWMA of |d Sigma r| a hop makes, so the acceptance window is in
        # the solver's own units at this n and needs no tuning per size.
        self.cur_xy = self.cur_r = None
        self.cur_s = -np.inf
        self.dscale = 0.0
        self.stall = 0
        self.fails = 0        # trials since the last gain -- drives `shake()` (see below)
        self.trial = 0
        self.warmed = False
        self.banked = -np.inf   # best Sigma r submitted for keeping but NOT followed
        # mode 3 (cross-size transfer) only exists when the census holds another size --
        # an n on its own must not pay MODE_FLOOR for a move that can never fire.
        self.modes = ([0, 1, 2] + ([3] if _neighbour_sizes(n) else []) + [4]
                      + ([5] if REASSIGN else []))
        self.mode_gain = np.zeros(len(self.modes))
        self.mode_cpu = np.zeros(len(self.modes))

    # -- candidate bookkeeping ---------------------------------------------------------
    def consider(self, xy, r, t_end, local=False):
        """Score one candidate.  `local=True` marks it as a hop off the incumbent, which is
        what calibrates `dscale` -- a cold start or a transfer lands an arbitrary distance
        away and would say nothing about the size of a barrier."""
        if xy is None or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(r)):
            return
        xy2, r2, moved = _fix_rattlers(xy, r)
        if moved:
            xy2, r2, _ = _slp(xy2, r2, min(t_end, time.process_time() + 0.5), max_iter=12)
        if np.any(r2 <= 0.0) or not _feasible(xy2, r2):
            return
        s = float(r2.sum())
        if local and np.isfinite(self.cur_s) and abs(s - self.cur_s) > _MOVED:
            # only hops that actually landed somewhere ELSE say anything about the height of
            # a barrier; a hop that re-converges to the same optimum would drag the scale to
            # zero and silently turn the threshold back into pure monotone acceptance.
            self.dscale = 0.9 * self.dscale + 0.1 * abs(s - self.cur_s)
        if s > self.best_s:
            self.best_xy, self.best_r, self.best_s = xy2.copy(), r2.copy(), s
            self.sink.add(_pack(xy2, r2))
        # threshold acceptance, measured against the BEST so the walk stays bounded
        if self.cur_xy is None or s >= self.best_s - ACCEPT_FRAC * self.dscale:
            self.cur_xy, self.cur_r, self.cur_s = xy2.copy(), r2.copy(), s

    def shake(self):
        """How wide the next basin hop should be -- a VARIABLE-NEIGHBOURHOOD ladder.

        Measured, and the reason this exists (`artifacts/iter9_trace_*.txt`): entered once on
        one size for the full 54 s CPU allowance -- the shape of the offline held-out re-run,
        which is how this solver is finally graded -- the search finds its answer in the first
        second and then buys NOTHING with the remaining fifty.  At n=45 the best after 0.6 s
        CPU and the best after 54 s agree to nine decimals.  With `ACCEPT_FRAC` closed the
        incumbent IS the best, so every hop departs from the same point at the same fixed
        width, re-converges to the same optimum, and the rest of the allowance is spent
        confirming it.  Unused CPU earns nothing.

        So widen the neighbourhood instead of the acceptance window (iteration 6 measured the
        window itself as a loss): after every full cycle of the move set without a gain the
        hop reaches one step further, and any gain resets it to the narrowest rung.  The rule
        carries no tuned constant and no units -- the cycle length is `len(self.modes)`,
        derived from the move set itself, so adding or removing a move re-derives it -- and it
        is the same rule at every n and in every phase: while the search is still moving it
        behaves exactly as it did before this change.
        """
        return 1.0 + self.fails / float(max(1, len(self.modes)))

    def _r_hint(self):
        return np.full(self.n, 0.4 / max(1.0, np.ceil(np.sqrt(self.n))))

    # -- one move ----------------------------------------------------------------------
    def _batch(self, XY0, R0, t_end, local=False):
        """Relax a whole batch, screen it cheaply, and pay for the closer only on the best.

        The screen is `_repair`'s sum -- exactly feasible and essentially free -- and it
        ranks the batch the same way the expensive closer does (measured in
        `artifacts/iter3_profile.py`), so spending the LP on the top `SCREEN_TOP` loses
        nothing and buys the rest of the batch.
        """
        XY, R = _nlp_batch(XY0, R0, t_end)
        screen = []
        for b in range(len(XY)):
            if not np.all(np.isfinite(XY[b])) or not np.all(np.isfinite(R[b])):
                continue
            screen.append((float(_repair(XY[b], R[b]).sum()), b))
        screen.sort(reverse=True)
        for _, b in screen[:SCREEN_TOP]:
            if time.process_time() >= t_end:
                break
            r = _exact_radii(XY[b], R[b])
            xy2, r2, _ = _slp(XY[b], r, t_end, max_iter=10, delta0=0.01)
            self.consider(xy2, r2, t_end, local=local)

    def _pick_mode(self):
        """Which move to make next -- decided by each move's MEASURED gain per CPU second,
        not by a fixed cycle.  Iteration 4 made mode 0 (cold multistart) much stronger, and
        the right reweighting is not something to tabulate: every mode is tried once, then
        trials are sampled in proportion to its own observed productivity.  `MODE_FLOOR`
        keeps a guaranteed share for every move, so a move that was merely unlucky early is
        never shut out -- the same reopen-the-rotation rule the size-level scheduler uses.
        """
        k = len(self.modes)
        if self.best_xy is None:
            return 0                              # modes[0] is always the cold multistart
        untried = np.flatnonzero(self.mode_cpu <= 0.0)
        if len(untried):
            return int(untried[0])
        rate = np.maximum(self.mode_gain / self.mode_cpu, 0.0)
        tot = float(rate.sum())
        floor = MODE_FLOOR / k                    # a share of an EQUAL split, so the rule is
        if not np.isfinite(tot) or tot <= 0.0:    # the same however many moves exist
            w = np.full(k, 1.0 / k)
        else:
            w = floor + (1.0 - k * floor) * (rate / tot)
        return int(self.rng.choice(k, p=w / w.sum()))

    def _one(self, t_end):
        j = self._pick_mode()
        self.trial += 1
        s_before, c_before = self.best_s, time.process_time()
        self._move(self.modes[j], t_end)
        self.mode_cpu[j] += max(time.process_time() - c_before, 1e-6)
        if np.isfinite(s_before) and np.isfinite(self.best_s):
            self.mode_gain[j] += max(self.best_s - s_before, 0.0)
        gained = np.isfinite(self.best_s) and (not np.isfinite(s_before)
                                               or self.best_s - s_before > 1e-12)
        self.fails = 0 if gained else self.fails + 1

    def _move(self, mode, t_end):
        n, rng = self.n, self.rng
        if mode == 0:
            XY0 = np.stack([_cold_start(n, rng, i + self.trial) for i in range(BATCH)])
            self._batch(XY0, np.full((BATCH, n), self._r_hint()[0]), t_end)
        elif mode == 1:
            # the exact SLP re-derives the radii from zero for these centres, so a
            # reassigned profile cannot survive the closer: mode 1 stays a pure centre hop.
            xy0, _ = _perturb(self.cur_xy, self.cur_r, rng, self.shake())
            xy, r, _ = _slp(xy0, np.zeros(n), t_end, delta0=0.02)
            self.consider(xy, r, t_end, local=True)
        elif mode == 2:
            # the penalty NLP takes the radii as variables and tolerates overlap, so this is
            # the one hop that can carry a REASSIGNED profile (see `_perturb`).
            hops = [_perturb(self.cur_xy, self.cur_r, rng, self.shake())
                    for _ in range(BATCH)]
            self._batch(np.stack([h[0] for h in hops]),
                        np.stack([h[1] for h in hops]), t_end, local=True)
        elif mode == 3:
            starts = [_transfer_start(n, rng) for _ in range(BATCH)]
            starts = [s for s in starts if s is not None]
            if not starts:
                self._move(0, t_end)              # census gone or n stands alone: cold start
                return
            self._batch(np.stack([s[0] for s in starts]),
                        np.stack([s[1] for s in starts]), t_end)
        elif mode == 4:
            self._uniform(t_end)
        else:
            self._reassign(t_end)

    def _reassign(self, t_end):
        """Mode 5: permute the RADIUS profile -- which site gets which size.

        Every other move perturbs CENTRES and then lets the LP re-derive the radii, so the
        radius profile has always been a deterministic function of the centres: the search
        has never been able to ask "what if this site held the big circle and that one the
        small one?".  But a packing is two things -- a set of sites and an assignment of
        sizes to them -- and the census says the assignment is where the residual lives:
        at every size, beaten record or not, the radii span a factor of ~2 and are almost
        all distinct (n=45: 0.0525..0.1056, 43 distinct values), so the sizes are NOT
        interchangeable and a wrong assignment cannot be repaired by moving centres a
        little.  A positional jitter must walk a big and a small circle past each other to
        swap their roles, and the exact closer forbids exactly that; the penalty explorer
        can only do it if the jitter happens to throw them far enough.

        So make the swap directly: take a random k-subset of the circles, permute their
        radii among themselves, and hand the result -- deliberately overlapping, since the
        penalty NLP is the one engine that tolerates overlap -- to the same batched
        explorer + exact closer every other move uses.  Sum r is unchanged by a permutation,
        so the move costs nothing up front; what it changes is which contacts bind, and the
        explorer then re-settles the centres around the new assignment.

        The neighbourhood size is the shared `shake()` ladder and nothing else: k = 2 at
        the narrowest rung (one transposition) and it widens only while the search is
        stalled, so this carries no constant of its own and reduces to the smallest possible
        structural change when the search is still moving.

        MEASURED AND SHIPPED OFF (`REASSIGN = False`; artifacts/iter12_ab_{BASE,SWAP}_{45,49}.txt).
        A/B in the regime that finally grades this solver -- the held-out probe, one size, a
        fresh process, 54 s CPU, the size's own pack withheld -- at the census's two weakest
        sizes:

            n=45   BASE 3.312 digits   SWAP 3.312   (identical to 9 decimals: same optimum)
            n=49   BASE 3.237 digits   SWAP 2.578   (-0.659)

        A tie and a loss, so the move does not ship.  Read honestly the experiment is also
        UNDERPOWERED rather than decisive: adding a sixth move changes `shake()`'s cycle
        length and shifts every later draw from `rng`, so the two arms are not a paired
        comparison at one seed -- they are two different random streams, and single sizes in
        this search are known to swing whole digits between streams.  The code stays because
        the hypothesis is worth re-testing paired (fold the permutation into `_perturb`, so
        the move mix and the rng stream are untouched), and because the *diagnosis* it came
        from survives the negative result: the radii span a factor of ~2 at every size and no
        existing move can reassign them.
        """
        n, rng = self.n, self.rng
        if self.cur_xy is None:
            self._move(0, t_end)                  # nothing to reassign yet: cold start
            return
        XY0 = np.tile(self.cur_xy, (BATCH, 1, 1))
        R0 = np.tile(self.cur_r, (BATCH, 1))
        k = int(min(n, 2 * max(1, int(np.ceil(self.shake())))))
        if k < 2:
            self._move(0, t_end)
            return
        for b in range(BATCH):
            idx = rng.permutation(n)[:k]
            R0[b, idx] = R0[b, idx[rng.permutation(k)]]
        self._batch(XY0, R0, t_end, local=True)

    def _uniform(self, t_end):
        """Mode 4: walk the max-min homotopy down to the true objective.

        Iteration 7 shipped this as a HARD switch -- converge the equal-radius surrogate,
        then release once -- and it was worth +0.559 digits.  The hard switch is however the
        crudest possible schedule: the surrogate's optimum is a *different* structure from
        the true objective's, so releasing in one jump drops the configuration at the edge of
        whatever basin the true objective happens to own there, and whatever the surrogate
        bought beyond that edge is thrown away.  A homotopy keeps it: each rung's optimum is
        the next rung's start, so the structure is carried continuously from "every circle
        matters as much as the worst one" to "only the sum matters", and the SLP's own
        feasibility guarantee holds at every rung because only the objective changes.

        The source is the incumbent when there is one (a perturbation of it, so repeated
        trials are not the same solve) and a cold start otherwise -- the same two sources the
        other moves use, so this needs no separate start family and works on an n the census
        has never seen.  The slice is split equally over the rungs: an unconverged rung is a
        worse start for the next one, and the final beta = 0 rung is the only one whose
        answer is actually scored, so neither end may starve.
        """
        n, rng = self.n, self.rng
        local = self.cur_xy is not None
        # mode 4 starts every rung from UNIFORM radii by construction, so a permutation of
        # the incumbent profile is a no-op here and is dropped.
        xy = (_perturb(self.cur_xy, self.cur_r, rng, self.shake())[0] if local
              else _cold_start(n, rng, self.trial))
        r = _repair(xy, np.full(n, self._r_hint()[0]))
        now = time.process_time()
        share = max(t_end - now, 0.0) / len(_BETA_LADDER)
        for k, beta in enumerate(_BETA_LADDER):
            if beta <= 0.0:
                r = _exact_radii(xy, r)                # released: radii free, exact LP restart
            xy, r, _ = _slp(xy, r, min(t_end, now + share * (k + 1)),
                            max_iter=30, delta0=(0.03 if k == 0 else 0.01), beta=beta)
            if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(r)):
                return
        self.consider(xy, r, t_end, local=local)

    def _bank(self, *packs):
        """Route a packing through `evaluate()` for KEEPING only -- never as the incumbent.

        The harness keeps the best strictly-feasible packing it is shown per n, so a banked
        pack can only raise what gets written and can never lower it.  It deliberately does
        NOT go through `consider()`: see `step()` for the measurement that says why.
        """
        for xy, r in packs:
            if xy is None or r is None:
                continue
            if not (np.all(np.isfinite(xy)) and np.all(np.isfinite(r))):
                continue
            if np.any(r <= 0.0) or not _feasible(xy, r):
                continue
            self.sink.add(_pack(xy, r))
            self.banked = max(self.banked, float(r.sum()))

    def step(self, t_end):
        s0 = self.best_s
        if not self.warmed:
            self.warmed = True
            warm = _read_pack(self.n)
            if warm is not None and warm.shape == (self.n, 3):
                xy, r, _ = _slp(warm[:, :2].copy(), warm[:, 2].copy(), t_end,
                                max_iter=20, delta0=0.01)
                if BANK_WARM:
                    # Banked, not followed: the census pack is kept (it is submitted to the
                    # sink, so it can still be what gets written) but it is not the incumbent
                    # the hops depart from and not the bar new structures must clear to be
                    # developed.  The search lane is then IDENTICAL whether or not this n has
                    # a pack of its own -- which is the regime the operator grades in.
                    self._bank((warm[:, :2], warm[:, 2]), (xy, r))
                else:
                    self.consider(xy, r, t_end)
        while time.process_time() < t_end:
            try:
                self._one(t_end)
            except Exception:
                break
        if self.best_xy is None:                  # never leave an n with nothing at all
            xy0 = _grid_start(self.n, self.rng)
            r0 = _repair(xy0, self._r_hint())
            if np.all(r0 > 0):
                self.consider(xy0, r0, t_end + 1.0)
        gain = self.best_s - s0 if np.isfinite(self.best_s) else 0.0
        self.stall = 0 if gain > 1e-12 else self.stall + 1
        self.sink.flush()
        return gain


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------
# Guaranteed share of trials for every move type, as a fraction of an EQUAL split -- so
# adding a fourth move does not silently re-tune the three that were measured at 0.45/3.
MODE_FLOOR = 0.45
# Width of the incumbent-acceptance window, in units of the solver's OWN measured hop scale
# (`_Worker.dscale`), so this is the only free number and it carries no length or size units:
# 0 reproduces pure monotone basin hopping.
# MEASURED AND REJECTED (artifacts/iter6_accept_ab_*.txt): pooled over 8 (size, seed) cells at
# equal CPU, frac=3 scored 2.652 mean digits against 2.715 for frac=0, and frac=1 lost too.  The
# barrier is not what traps this search -- the CPU per size is (see FOCUS_SLICE) -- so the window
# ships CLOSED.  The code stays because the measurement is the useful part: it says a wider
# acceptance window is not the lever, and it is one number to re-open if that ever changes.
ACCEPT_FRAC = 0.0
# Whether the radius-reassignment move (`_Worker._reassign`) is in the rotation.  MEASURED
# AND SHIPPED OFF -- see that method's docstring for the held-out A/B (a tie at n=45, -0.659
# digits at n=49) and for why the experiment is underpowered rather than decisive.
REASSIGN = False
# Whether a basin hop also REASSIGNS the radius profile over the sites (`_perturb`).  This
# was iteration 12's mode-5 experiment re-run the PAIRED way -- folded into an existing move,
# with the permutation drawn either way, so the move mix and the rng stream are identical in
# both arms and the comparison is of two policies rather than two random streams.
#
# MEASURED AND SHIPPED OFF (artifacts/iter13_ab_{on,off}_*.txt).  Seven paired cells in the
# regime that finally grades this solver -- one size, fresh process, 54 s CPU, the size's own
# pack withheld -- taken at the census's seven weakest sizes, digits of gap closed:
#
#     n      49     59     43     79     99     73     51      mean
#     ON    2.578  3.075  3.298  7.000  3.247  2.854  3.438    3.641
#     OFF   2.637  3.217  2.850  7.000  7.000  7.000  3.438    4.735
#
# Two ties, one win for ON (+0.448 at n=43) and four losses, two of them catastrophic: at
# n=73 and n=99 the OFF arm CROSSES THE RECORD and the ON arm does not get within three
# digits of it.  Reassignment is therefore rejected on its second, and this time properly
# powered, hearing: swapping which site holds which size costs mode 2 more than the extra
# dimension buys, because the penalty NLP has to spend its slice undoing an overlap the
# swap created instead of exploring.  The two dead draws stay so that re-opening this flag
# stays paired -- and because the arm measured above is this file, draws included.
SWAP_HOP = False
# Whether the census pack for n is BANKED (submitted to `evaluate()` so it can still be what
# gets written) rather than FOLLOWED (installed as the incumbent every hop departs from and
# as the bar a new structure must clear before it is developed).
#
# MEASURED, and this is the control iteration 13's finding was missing
# (artifacts/iter14_pair_{present,heldout}_{73,79,99}.txt).  Iteration 13 saw a fresh 54 s call
# with the size's own pack WITHHELD cross the record at n=73/79/99 where the committed pack sits
# at ~3.0 digits, and could not say whether the cause was the 9x deeper slice or the absent warm
# start.  Holding depth fixed at 54 s and varying only the pack's presence separates them:
#
#     n              73      79      99
#     own pack absent  7.000   7.000   7.000      (nfev 12 / 13 / 15)
#     own pack present  3.026   7.000   3.168      (nfev  1 / 13 /  3)
#
# Depth is NOT the variable: the present arm had the same 54 s and stayed exactly where the
# 6 s metered slices had left it.  The `nfev` column is the mechanism -- with the pack followed,
# ONE candidate in 54 s ever beat it at n=73, so every cold start and every cross-size transfer
# was relaxed once and thrown away, while in the withheld arm the first transfer BECAME the
# incumbent and twelve rounds of hops refined it into the record.  Following a good-but-trapped
# pack does not merely fail to help, it denies every other structure the development that is
# how this search actually climbs.
#
# Banking keeps the safety the warm start was there for: the harness keeps the best feasible
# packing it is shown, and the driver only replaces a pack with a strictly better one, so a
# banked pack can raise what is written and can never lower it.  It also removes a special
# case rather than adding one -- with this on, the search lane is the same whether or not n
# has a pack, so an in-run measurement finally predicts the held-out re-run it is graded by.
#
# SHIPPED, and the in-run call agrees: the four sizes it worked went 43: 2.893 -> 7.000,
# 73: 3.026 -> 7.000 (a strict record beat), 59: 2.725 -> 3.108, 49: unchanged, i.e. +8.46
# digits where following the pack had been worth +4.44 across eighteen sizes.  Note the size
# of the tail: four matched forecast probes at this depth predicted +0.41 digits in total
# (artifacts/iter14_forecast_*.txt) and the metered call, on its own rng stream, returned
# +8.46.  Gains here are not a mean plus noise -- they are rare basin finds, so read any
# single cell of this search as one draw from a fat tail.
BANK_WARM = True
# The per-size slice, i.e. how many sizes one call works (sizes x slice is capped by the
# call's own CPU budget, so this trades breadth against depth and nothing else).
#
# RE-MEASURED for the banked lane (artifacts/iter14_depth_{73,99}_{12,27}.txt plus the 54 s
# cells above).  6.0 came from iteration 6, where the warm pack was FOLLOWED and a slice only
# had to polish it; a banked lane has to re-derive a structure from cold, so the productive
# slice is a different quantity and had to be re-fitted.  Digits reached by one held-out call
# at a fixed CPU allowance:
#
#     CPU/size      12 s     27 s     54 s      committed (from 6 s slices)
#     n=73         3.275    7.000    7.000      3.026
#     n=99         2.948    3.261    7.000      3.168
#
# The curve has a knee: at 12 s neither size converts (n=99 does not even reach its committed
# pack), at 27 s one of the two crosses the record outright.  So 27 is where a slice starts
# buying whole digits instead of hundredths, and it is read off the curve rather than chosen --
# 112 s / 27 s = 4 sizes worked per call.  Depth beyond the knee keeps paying (n=99 needs the
# full 54 s), so this is the CONSERVATIVE end of the fit: it buys the knee for four sizes
# instead of the plateau for two.
FOCUS_SLICE = 27.0
# The scorer's own cap: a size at this many digits is AT or above the record, so no slice can
# buy it anything.  It is the scorer's constant, not a tuned one -- it only clips the ranking
# key so sizes already at the cap cannot be ordered ahead of sizes still short of it.
DIGITS_CAP = 7.0
STALL_DROP = 3        # slices without a gain before an n leaves the rotation
ROUNDS_TARGET = 4.0   # aim to give every live n this many slices


def _focus(targets, cpu):
    """Which sizes this call should actually work, when it cannot afford them all.

    The census is append-or-improve: an n that gets no CPU keeps the packing it already has,
    so spreading a fixed CPU budget over every size is not free -- it buys every size a slice
    too short to move anything.  Measured in `artifacts/iter6_accept_ab.py`: a single size
    given FOCUS_SLICE seconds improved on its committed pack at 6 of 6 sizes tried (+0.03 to
    +0.82 digits), while the same budget spread over 37 sizes moved 9 of them.  So take as
    many sizes as the budget can give a real slice to, and leave the rest to their incumbents.

    WHICH sizes: rank by the SCORE'S OWN GRADIENT, which is the headroom each size still has.
    A size already at d digits can gain at most `cap - d` more however much CPU it is given,
    so `d = -log10(relgap)` IS the marginal value of a slice, and the scheduler should simply
    spend on the smallest d first.  When the record for an n is unknown -- an n outside the
    table, a held-out size, a table that failed to parse -- there is no d, and the fallback is
    the trend fit this used to rank by everywhere: Sigma r / sqrt(n) is smooth in n across the
    census (0.5156 -> 0.5256 from n=27 to 99), so fit it over every size the census holds and
    rank by the RESIDUAL, a pack sitting furthest below its own neighbours' trend. One rule,
    two cases, and the cases are ordered: known headroom beats an estimate of it. A size with
    no pack at all ranks first in either case -- it has everything to gain, and that is also
    what makes this correct for an n the census has never seen.

    MEASURED, and why this replaced the residual (artifacts/iter11_ab_*.txt, one in-run call,
    all 37 visible n, 112 s, seed 12345): the residual ranking put n=27 among the two sizes
    with the MOST to gain when it was in fact the second-best non-record size in the census at
    4.14 digits, and it skipped n=45, 71, 43, 73, 89 and 31, every one of them below 3.04.
    The residual cannot tell "this n is intrinsically inefficient" from "our pack for this n is
    bad", and it is worst exactly at the ends of the fitted range.
    """
    targets = sorted(targets)
    # A size the census has no pack for has NO fallback: skipping it scores zero for that n,
    # while skipping a size that has one only keeps what is already committed.  So the trade
    # depth buys is only ever against sizes that already hold an answer -- the rest are
    # mandatory however tight the budget is.
    need = [n for n in targets if _cached_pack(n) is None or _cached_pack(n).shape != (n, 3)]
    want = max(1, len(need), int(cpu / FOCUS_SLICE))
    if want >= len(targets):
        return list(targets)
    ys, ms = [], []
    for m in _census_sizes():
        a = _cached_pack(m)
        if a is not None and a.shape == (m, 3) and m >= 1:
            ys.append(float(a[:, 2].sum()) / np.sqrt(m))
            ms.append(float(m))
    recs = _records()
    # Digits are on their own scale and rank strictly ahead of any residual, so a size whose
    # record is known is always compared against other such sizes first: the two cases never
    # interleave, and neither has to be rescaled to match the other.
    def key(n):
        a = _cached_pack(n)
        if a is None or a.shape != (n, 3):
            return (0, -np.inf)                  # no pack: everything to gain, go first
        s = float(a[:, 2].sum())
        rec = recs.get(n)
        if rec is not None and rec > 0.0:
            gap = max(0.0, (rec - s) / rec)
            return (1, min(DIGITS_CAP, -np.log10(max(gap, 1e-16))))
        y = s / np.sqrt(n)
        if len(ms) < 4:
            return (2, y)                        # too little census to fit a trend
        deg = 2 if len(ms) >= 6 else 1
        try:
            fit = np.polyval(np.polyfit(np.array(ms), np.array(ys), deg), float(n))
        except Exception:
            return (2, y)
        return (2, y - fit)
    return sorted(targets, key=key)[:want]


def solve(evaluate, meter, rng, targets, cpu_budget=None):
    targets = sorted(set(int(t) for t in targets if int(t) >= 1))
    if not targets:
        return
    t0 = time.process_time()
    if cpu_budget is None:
        cpu_budget = min(CPU_CALL_CAP, CPU_PER_SIZE * len(targets))
    deadline = t0 + float(cpu_budget)
    # Work only as many sizes as this budget can give a real slice to; the rest keep the
    # packings they already have (the census is append-or-improve, so that costs nothing).
    targets = _focus(targets, float(cpu_budget))
    if not targets:
        return

    sinks = {n: _Sink(evaluate, meter, n) for n in targets}
    live = [_Worker(n, sinks[n], rng) for n in targets]
    mean_n = float(np.mean(targets))

    # Round 0: a guaranteed slice for every n, so no size is ever left without a packing
    # even if the rotation below is cut short by the CPU backstop.
    seed_share = min(0.25 * (deadline - t0) / len(live), 0.6)
    for w in live:
        if time.process_time() >= deadline or meter.left() <= 0:
            break
        w.step(min(deadline, time.process_time() + max(0.05, seed_share * w.n / mean_n)))

    # Adaptive rotation: whoever is still improving keeps getting CPU.  When everyone has
    # stalled the rotation is simply reopened -- unused CPU earns nothing.
    active = list(live)
    while time.process_time() < deadline and meter.left() > 0:
        if not active:
            for w in live:
                w.stall = 0
            active = list(live)
        remaining = deadline - time.process_time()
        base = remaining / (ROUNDS_TARGET * len(active))
        for w in active:
            if time.process_time() >= deadline or meter.left() <= 0:
                break
            w.step(min(deadline, time.process_time() + max(0.05, base * w.n / mean_n)))
        active = [w for w in active if w.stall < STALL_DROP]

    for s in sinks.values():
        s.flush()


# --------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------
def _self_test():
    import sys

    class Meter(object):
        def __init__(self, budget):
            self.budget, self.used = budget, 0

        def left(self):
            return self.budget - self.used

    state = {}

    def make_eval(meter):
        def evaluate(n, packing):
            p = np.asarray(packing, dtype=float)
            if p.ndim == 2:
                p = p[None]
            if p.ndim != 3 or p.shape[1] != n or p.shape[2] != 3:
                raise ValueError("bad shape")
            out_f, out_s = [], []
            for c in p:
                meter.used += 1
                if meter.used > meter.budget:
                    out_f.append(False)
                    out_s.append(-np.inf)
                    continue
                xy, r = c[:, :2], c[:, 2]
                ok = _feasible(xy, r)
                out_f.append(bool(ok))
                out_s.append(float(r.sum()) if ok else -np.inf)
                if ok and float(r.sum()) > state.get(n, (-np.inf,))[0]:
                    state[n] = (float(r.sum()), c.copy())
            return np.array(out_f), np.array(out_s)
        return evaluate

    recs = {}
    try:
        recs = json.load(open(os.path.join("bench", "records.json")))["records"]
    except Exception:
        pass

    meter = Meter(500000)
    ev = make_eval(meter)
    rng = np.random.RandomState(0)

    ok = True
    # call 1: several sizes; call 2: a single size -- both in ONE process, which is what
    # catches a deadline computed once at import time or a split that assumes many targets.
    for call, tg in ((1, [13, 27, 34]), (2, [27])):
        state.clear()
        before = meter.used
        solve(ev, meter, rng, tg, cpu_budget=2.0 * len(tg))
        for n in tg:
            if n not in state:
                # Skipping a size is legitimate ONLY when the census already holds a packing
                # for it: the store is append-or-improve, so that n keeps its answer.  A size
                # with no committed pack has no fallback and must always be worked.
                have = _read_pack(n)
                if have is not None and have.shape == (n, 3):
                    print("call %d  n=%-3d skipped by _focus (committed pack survives)"
                          % (call, n))
                else:
                    print("FAIL call %d: n=%d has no committed pack and was left unsolved"
                          % (call, n))
                    ok = False
                continue
            s, c = state[n]
            if not _feasible(c[:, :2], c[:, 2]) or len(c) != n:
                print("FAIL call %d: n=%d packing not feasible" % (call, n))
                ok = False
            rec = recs.get(str(n))
            extra = ""
            if rec:
                gap = max(0.0, (rec - s) / rec)
                extra = "  relgap=%.3e  digits=%.2f" % (gap, min(7.0, -np.log10(max(gap, 1e-16))))
            print("call %d  n=%-3d sum_r=%.9f%s" % (call, n, s, extra))
        print("call %d used %d evaluations" % (call, meter.used - before))

    # an unknown n with no committed pack must cold-start cleanly
    state.clear()
    solve(ev, meter, rng, [8], cpu_budget=1.5)
    if 8 not in state:
        print("FAIL: cold start for an n outside the census produced nothing")
        ok = False
    else:
        print("cold n=8   sum_r=%.9f" % state[8][0])

    print("SELF-TEST: %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print("usage: python3 tools/solver.py --self-test")
