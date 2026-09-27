"""Circle-packing solver: max sum-of-radii of n circles in the centered unit square.

ALGORITHM
=========
Local step -- CONVEX RESTRICTION (SLP / convex-concave).  ||c_i - c_j|| is convex in the centres,
so its first-order Taylor expansion is a GLOBAL UNDER-estimate; substituting it turns the joint
(centre, radius) problem into an LP whose every feasible point is feasible for the true problem.
Hence each LP step is feasible and monotone (dx = 0, r = current is always LP-feasible).

Iteration 2 -- the LP is made SPARSE by adding a RADIUS trust region.
    Iteration 1 bounded r_i only by the wall (<= wall_i + delta), so a pair could only be proved
    slack when d_ij > wall_i + wall_j + 4*delta ~ 1.0 -- i.e. essentially NO pair was ever dropped
    and every LP carried all n(n-1)/2 rows.  Adding the extra restriction
        r_i <= r_i^cur + delta_r
    (still a restriction, so the LP remains an inner approximation, and r = r^cur is still
    feasible, so monotonicity survives) lets a pair be dropped whenever
        d_ij > (r_i + delta_r) + (r_j + delta_r) + 2*sqrt(2)*delta
    because |delta_i - delta_j|_2 <= 2*sqrt(2)*delta under the |.|_inf <= delta box.  Only
    near-contact pairs survive: the row count falls from O(n^2) to O(n), and the LP gets several
    times faster -- which is the whole game, because the binding meter is the 120 s CPU cap, not
    the 500k evaluations.

Global step -- STRUCTURED basin hopping.  Iteration 1's uniform random kick changes the contact
graph only by luck.  Here the moves are topological: (a) delete the k smallest circles and
re-insert each at the centre of the largest empty disc (a grid argmax of the "largest circle that
fits here" field), (b) swap a big and a small circle, (c) jitter.  Plus CROSS-N TRANSFER: n is
seeded from the solution just found for n-2 with two circles hole-inserted.

Metering -- candidates are buffered and pushed through evaluate() in BATCHES (one vectorized
harness pass per batch), and only local optima / incumbents are submitted, so almost all CPU goes
to the LP rather than to re-validating iterates.

Re-entrancy -- the CPU deadline is recomputed from time.process_time() AT EVERY solve() ENTRY
(never a module constant), and the per-n slice is avail/len(targets) so it survives
len(targets) == 1.  Warm-start reads of bench/packs/ are os.path.exists-guarded with a cold-start
fallback that works for ANY n.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

try:                                    # HiGHS is what scipy.linprog calls anyway; going direct
    import highspy                      # skips the linprog wrapper's validation/conversion layer
    _HS_INF = highspy.kHighsInf
except Exception:                       # pragma: no cover - fallback keeps the solver portable
    highspy = None
    _HS_INF = np.inf

LO, HI = -0.5, 0.5
CPU_CAP = 120.0          # driver's process-CPU backstop
CPU_SAFETY = 9.0         # headroom for the driver's own re-validation pass
MIN_SLICE = 0.35         # never give an n less than this many CPU seconds
SQRT8 = 2.0 * np.sqrt(2.0)


# ----------------------------------------------------------------------------- geometry helpers
def _walls(c):
    return np.minimum.reduce([c[:, 0] - LO, HI - c[:, 0], c[:, 1] - LO, HI - c[:, 1]])


def _dists(c):
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return dist


def _repair(c, r):
    """Uniform shrink so the packing is EXACTLY feasible (slack >= 0) whatever the LP tolerance."""
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    wall = np.maximum(_walls(c), 0.0)
    dist = _dists(c)
    alpha = 1.0
    pos = r > 0
    if pos.any():
        alpha = min(alpha, float(np.min(wall[pos] / r[pos])))
    rr = r[:, None] + r[None, :]
    m = rr > 0
    np.fill_diagonal(m, False)
    if m.any():
        alpha = min(alpha, float(np.min(dist[m] / rr[m])))
    if not np.isfinite(alpha):
        alpha = 0.0
    r = r * min(1.0, alpha)
    # the positivity floor must never itself create a wall violation: a circle squeezed to rcap=0
    # whose centre sits ON the boundary has wall == 0, and flooring it at a flat 1e-9 left slack
    # -1e-9 (found by the iteration-10 self-test).  Floor at the room that actually exists.
    return np.maximum(r - 1e-12, np.maximum(np.minimum(1e-9, wall), 1e-15))


def _sck(base, sc):
    """Scale a COUNT-style move size by the ladder multiplier `sc`, keeping the randint upper
    bound >= 2 so `rng.randint(1, _sck(...) + 1)` always has at least one legal draw.  sc = 1.0
    reproduces the pre-iteration-12 constant exactly."""
    return max(2, int(round(base * sc)))


def _greedy_radii(c):
    """Feasible radii from centres alone: r_i = min(wall_i, min_j d_ij / 2)."""
    return _repair(c, np.minimum(np.maximum(_walls(c), 0.0), _dists(c).min(axis=1) / 2.0))


def _pack(c, r):
    return np.concatenate([c, r[:, None]], axis=1)


# ------------------------------------------------------------------- metered evaluation sink
class _Sink:
    """Buffers candidates and pushes them through the metered evaluate() in batches."""

    def __init__(self, evaluate, meter, n, batch=24):
        self.ev, self.meter, self.n, self.batch = evaluate, meter, n, batch
        self.buf = []
        self.best = -np.inf
        self.pack = None

    def add(self, pack):
        self.buf.append(np.asarray(pack, dtype=float))
        if len(self.buf) >= self.batch:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        room = self.meter.left()
        if room <= 0:
            self.buf = []
            return
        arr = np.stack(self.buf[:room])
        self.buf = []
        feas, sr = self.ev(self.n, arr)
        feas = np.atleast_1d(feas)
        sr = np.atleast_1d(sr)
        ok = np.flatnonzero(feas)
        if ok.size:
            k = ok[int(np.argmax(sr[ok]))]
            if float(sr[k]) > self.best:
                self.best = float(sr[k])
                self.pack = arr[k].copy()


# ------------------------------------------------------------------ the convex-restriction LP
def _lp_rows(c, r, delta, delta_r, rcap=None):
    """Assemble the sparse convex-restriction rows.  Returns (A_csr, b, ub, nrow).

    Iteration 5 -- the WALL rows are filtered the same way the pair rows already were.  A wall row
    reads  sign * d_axis_i + r_i <= rhs_i  with |d| <= delta and r_i <= ub_i, so its largest
    attainable left-hand side is  ub_i + delta.  When that is below rhs_i the row cannot bind for
    ANY point of the trust region, so dropping it changes nothing -- it is an exact reduction, not
    a relaxation.  Measured on the committed census only ~29 of the 4n wall rows can bind at n=63,
    so this deletes ~60 % of the constraint matrix.

    Iteration 6 -- the rows also carry an IDENTITY KEY (int64, pair (i,j) -> i*n+j, wall (t,i) ->
    n*n + t*n + i).  Consecutive CCP steps re-derive an almost identical row set, so the keys let
    the previous simplex BASIS be replayed onto the new model row-by-row: see `_hs_solve`."""
    n = c.shape[0]
    dist = _dists(c)
    ub = np.minimum(0.5, r + delta_r)
    if rcap is not None:
        ub = np.minimum(ub, np.maximum(np.asarray(rcap, dtype=float), 0.0))

    # A pair row is provably slack (hence droppable, restriction preserved) when
    #     ub_i + ub_j  <  d_ij - |delta_i - delta_j|_2  and  |delta_i - delta_j|_2 <= 2*sqrt(2)*delta
    thr = ub[:, None] + ub[None, :] + SQRT8 * delta
    iu = np.triu_indices(n, 1)
    sel = dist[iu] <= thr[iu]
    pi, pj = iu[0][sel], iu[1][sel]
    m = pi.size

    rows, cols, vals, b, keys = [], [], [], [], []
    if m:
        dvec = np.maximum(dist[pi, pj], 1e-12)   # coincident ghosts -> u = 0, row r_i+r_j<=0
        ux = (c[pi, 0] - c[pj, 0]) / dvec
        uy = (c[pi, 1] - c[pj, 1]) / dvec
        k = np.arange(m)
        rows.append(np.concatenate([k, k, k, k, k, k]))
        cols.append(np.concatenate([pi, n + pi, pj, n + pj, 2 * n + pi, 2 * n + pj]))
        vals.append(np.concatenate([-ux, -uy, ux, uy, np.ones(m), np.ones(m)]))
        b.append(dvec)
        keys.append(pi.astype(np.int64) * n + pj)

    base = m
    for tag, (axis, sign, rhs) in enumerate(((0, -1.0, c[:, 0] - LO),
                                             (0, +1.0, HI - c[:, 0]),
                                             (1, -1.0, c[:, 1] - LO),
                                             (1, +1.0, HI - c[:, 1]))):
        keep = np.flatnonzero(ub + delta >= rhs)
        if keep.size == 0:
            continue
        kk = base + np.arange(keep.size)
        rows.append(np.concatenate([kk, kk]))
        cols.append(np.concatenate([axis * n + keep, 2 * n + keep]))
        vals.append(np.concatenate([np.full(keep.size, sign), np.ones(keep.size)]))
        b.append(rhs[keep])
        keys.append(n * n + tag * n + keep.astype(np.int64))
        base += keep.size

    if not rows:
        return None, None, ub, 0, None
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(base, 3 * n)).tocsr()
    return A, np.concatenate(b), ub, base, np.concatenate(keys)


_HS_BASIC = highspy.HighsBasisStatus.kBasic if highspy is not None else None


def _hs_model(A, b, cost, cl, cu):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("presolve", "off")
    h.setOptionValue("primal_feasibility_tolerance", 1e-11)
    h.setOptionValue("dual_feasibility_tolerance", 1e-11)
    nv = cost.size
    h.addVars(nv, cl, cu)
    h.changeColsCost(nv, np.arange(nv, dtype=np.int32), cost)
    h.addRows(A.shape[0], np.full(A.shape[0], -_HS_INF), b,
              A.nnz, A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data)
    return h


def _replay_basis(state, keys):
    """Map the basis stored in `state` onto a fresh model whose rows are identified by `keys`.

    Rows that also existed in the previous LP inherit their status; rows that are NEW (the
    near-contact filter admitted a pair the previous step had dropped) enter BASIC, i.e. slack --
    the mildest guess, and one HiGHS repairs by pivoting if the basis turns out singular."""
    prev = state.get("keys_sorted")
    if prev is None:
        return None
    pos = np.searchsorted(prev, keys)
    np.clip(pos, 0, prev.size - 1, out=pos)
    hit = prev[pos] == keys
    src = state["perm"][pos]
    rs = np.where(hit, state["rows"][src], _HS_BASIC)
    bas = highspy.HighsBasis()
    bas.col_status = state["cols"]
    bas.row_status = rs.tolist()
    return bas


def _hs_solve(A, b, cost, cl, cu, keys=None, state=None):
    """Solve  min cost'z  s.t.  Az <= b,  cl <= z <= cu  by calling HiGHS directly.

    scipy.linprog re-validates, re-copies and re-presolves the whole model on every call; for the
    small (O(n) rows, 3n cols) LPs the CCP fires thousands of times that wrapper is a large part of
    the cost.  Going straight to highspy with presolve off is ~1.4x faster end to end and returns
    the SAME vertex (verified to 1e-14 on the committed census at n=35/63/99).

    Iteration 6 -- BASIS WARM START.  Building the model is free (0.06 ms at n=99); ALL of the cost
    is `run()` re-deriving the optimal basis from scratch: 135 / 245 / 438 primal simplex iterations
    at n = 35 / 63 / 99.  But consecutive CCP steps only rotate the contact normals slightly, so the
    OPTIMAL BASIS BARELY MOVES -- measured over a 40-step chain at n=63, replaying the previous
    basis solves every single step in **0 simplex iterations** (Sum r identical to 1.5e-13), for a
    6.4x end-to-end speedup.  `state` is a caller-owned dict that carries the basis along a chain;
    a warm solve that fails to reach optimality falls back to a cold solve of the same model, so
    the returned vertex is never worse than the cold path's."""
    warm = None
    if state is not None and keys is not None:
        try:
            warm = _replay_basis(state, keys)
        except Exception:
            warm = None
    h = _hs_model(A, b, cost, cl, cu)
    if warm is not None:
        h.setBasis(warm)
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        if warm is None:
            return None
        state.clear()                       # the stored basis is stale -- start this chain over
        h = _hs_model(A, b, cost, cl, cu)
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None
    if state is not None and keys is not None:
        bas = h.getBasis()
        order = np.argsort(keys, kind="stable")
        state["keys_sorted"] = keys[order]
        state["perm"] = order
        state["rows"] = np.asarray(list(bas.row_status), dtype=object)
        state["cols"] = list(bas.col_status)
    return np.asarray(h.getSolution().col_value, dtype=float)


def _ccp_lp(c, r, delta, delta_r, rcap=None, w=None, state=None):
    """One convex-concave step.  Returns (c_new, r_new) or None if the LP fails.

    z = [dx(n), dy(n), r(n)].  Maximise sum w_i r_i under the linearised (hence conservative)
    non-overlap rows, the exact wall rows, an |d|_inf <= delta centre trust region and an
    r_i <= r_i + delta_r radius trust region.  The radius bound is what makes the pair filter bite.

    Iteration 3 adds two DEFORMATION knobs used by the global search (both are further
    restrictions of the same LP, so every iterate stays exactly feasible for the true problem):
      rcap : per-circle hard radius ceiling -- setting rcap_i = 0 SQUEEZES circle i out of
             existence and lets its neighbours flow into the vacated room.
      w    : objective weights -- re-weighting tilts the LP to a DIFFERENT vertex of the same
             feasible set, i.e. a different contact graph, without any random displacement."""
    n = c.shape[0]
    A, b, ub, nrow, keys = _lp_rows(c, r, delta, delta_r, rcap=rcap)
    nv = 3 * n
    obj = np.zeros(nv)
    obj[2 * n:] = -1.0 if w is None else -np.asarray(w, dtype=float)
    cl = np.concatenate([np.full(2 * n, -delta), np.zeros(n)])
    cu = np.concatenate([np.full(2 * n, delta), ub])
    if nrow == 0:
        z = np.where(obj < 0, cu, cl)
    else:
        z = None
        if highspy is not None:
            try:
                z = _hs_solve(A, b, obj, cl, cu, keys=keys, state=state)
            except Exception:
                z = None
        if z is None:                                   # fallback: the portable scipy path
            try:
                res = linprog(obj, A_ub=A, b_ub=b, bounds=list(zip(cl, cu)), method="highs")
            except Exception:
                return None
            if not res.success or res.x is None:
                return None
            z = res.x
    c_new = np.clip(c + np.stack([z[:n], z[n:2 * n]], axis=1), LO, HI)
    return c_new, z[2 * n:]


def _ccp(c0, deadline, sink, n, max_iter=250, delta0=None, r0=None, delta_min=3e-7):
    """Run the CCP from centres c0 until convergence / deadline.  Submits only the final local
    optimum (and the running incumbent every so often) to the metered sink."""
    c = np.clip(np.asarray(c0, dtype=float), LO, HI)
    delta = float(0.30 / np.sqrt(n) if delta0 is None else delta0)
    r = _greedy_radii(c) if r0 is None else _repair(c, r0)
    cur = float(r.sum())
    best_c, best_r, best_s = c, r, cur
    stall = 0
    lp_state = {}          # iteration 6: the simplex basis is carried along this CCP chain
    for _ in range(max_iter):
        if time.process_time() > deadline or sink.meter.left() <= 0:
            break
        out = _ccp_lp(c, r, delta, delta, state=lp_state)
        if out is None:
            delta *= 0.4
            if delta < delta_min:
                break
            continue
        c_new, r_new = out
        r_new = _repair(c_new, r_new)
        s = float(r_new.sum())
        if s > cur + 1e-13:
            stall = 0 if s > cur + max(1e-10, abs(cur) * 1e-10) else stall + 1
            c, r, cur = c_new, r_new, s
            if s > best_s:
                best_c, best_r, best_s = c, r, s
        else:
            stall += 1
        if stall >= 2:
            delta *= 0.5
            stall = 0
            if delta < delta_min:
                break
    sink.add(_pack(best_c, best_r))
    return best_s, best_c, best_r



# ============================================================================================
# ITERATION 4 -- the SMOOTH ENGINE: an augmented-Lagrangian local solver that mirrors the LP
# ============================================================================================
# The LP convex restriction is exact and monotone, but it costs 5-8 ms per step (measured 7.6 ms
# at n=63) and a basin needs ~150 of them: ~1 s per basin, ~6 basins per n per iteration.  The
# 500,000-evaluation budget is 0.1 % spent -- CPU, not candidates, is the wall.
#
# So the search engine is replaced by a smooth solver on the SAME problem
#       max  sum_i w_i r_i   s.t.  r_i + r_j <= ||c_i - c_j||,  r_i <= wall_i(c_i),  0 <= r_i <= rcap_i
# handled by an AUGMENTED LAGRANGIAN (multipliers + escalating rho) whose inner problem is a
# bound-constrained L-BFGS-B with an analytic gradient over a NEIGHBOUR LIST of near-contact
# pairs.  One function+gradient is O(n) numpy work (~20 us), so a whole basin costs ~15-30 ms
# instead of ~1 s -- 30-50x more basins for the same CPU.
#
# The two LP DEFORMATION knobs from iteration 3 carry over verbatim, because they are just an
# objective weight vector `w` and a radius ceiling `rcap` -- both are bounds/coefficients here,
# so squeeze-and-teleport and objective tilt become nearly free.
#
# The LP is NOT retired: it is the only step with an exact feasibility certificate, so it stays
# as the FINISHER.  The architecture is screen-then-polish: the smooth engine explores many
# basins cheaply, and only a basin that lands within a relative threshold of the incumbent earns
# an LP polish.  _repair() still makes every submitted packing exactly feasible.


def _rlp(c, r_hint=None):
    """EXACT optimal radii for FIXED centres.  With c held, max sum r s.t. r_i+r_j <= d_ij,
    0 <= r_i <= wall_i is a pure LP in n variables -- tiny (2.5-4 ms) next to the 3n-variable
    trust-region LP, and it is what makes the smooth engine's output comparable to the CCP's:
    the augmented Lagrangian only has to get the CENTRES right, the radii are then solved
    exactly rather than recovered by a uniform shrink (which one badly-violated pair can wreck)."""
    n = c.shape[0]
    wall = np.maximum(_walls(c), 0.0)
    if n < 2:
        return _repair(c, wall)
    d = _dists(c)
    iu = np.triu_indices(n, 1)
    sel = d[iu] <= wall[iu[0]] + wall[iu[1]]      # a pair looser than that can never bind
    pi, pj = iu[0][sel], iu[1][sel]
    m = pi.size
    if m == 0:
        return _repair(c, wall)
    k = np.arange(m)
    A = coo_matrix((np.ones(2 * m), (np.concatenate([k, k]), np.concatenate([pi, pj]))),
                   shape=(m, n))
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=d[pi, pj],
                      bounds=list(zip(np.zeros(n), wall)), method="highs")
    except Exception:
        return _greedy_radii(c)
    if not res.success or res.x is None:
        return _greedy_radii(c)
    return _repair(c, res.x)


def _near_pairs(c, r, margin):
    n = c.shape[0]
    dist = _dists(c)
    thr = r[:, None] + r[None, :] + margin
    iu = np.triu_indices(n, 1)
    sel = dist[iu] <= thr[iu]
    return iu[0][sel].astype(np.int64), iu[1][sel].astype(np.int64)


def _merge_pairs(pi, pj, lam, npi, npj, n):
    """Union the neighbour list with a freshly built one, preserving existing multipliers."""
    old = pi * n + pj
    add = np.setdiff1d(npi * n + npj, old, assume_unique=False)
    if add.size:
        pi = np.concatenate([pi, add // n])
        pj = np.concatenate([pj, add % n])
        lam = np.concatenate([lam, np.zeros(add.size)])
    return pi, pj, lam


def _al_fg(z, n, pi, pj, rho, lam_p, lam_w, wobj):
    """Augmented-Lagrangian value + analytic gradient.  z = [cx(n), cy(n), r(n)]."""
    cx, cy, r = z[:n], z[n:2 * n], z[2 * n:]
    f = -float(wobj @ r)
    gcx = np.zeros(n)
    gcy = np.zeros(n)
    gr = -wobj.copy()

    if pi.size:
        dx = cx[pi] - cx[pj]
        dy = cy[pi] - cy[pj]
        d = np.sqrt(dx * dx + dy * dy) + 1e-15
        gcon = r[pi] + r[pj] - d                      # <= 0 wanted
        t = lam_p + rho * gcon
        t = np.where(t > 0.0, t, 0.0)
        f += float(t @ t) / (2.0 * rho)
        gr += np.bincount(pi, weights=t, minlength=n) + np.bincount(pj, weights=t, minlength=n)
        ux, uy = dx / d, dy / d
        gcx += np.bincount(pj, weights=t * ux, minlength=n) - np.bincount(pi, weights=t * ux,
                                                                          minlength=n)
        gcy += np.bincount(pj, weights=t * uy, minlength=n) - np.bincount(pi, weights=t * uy,
                                                                          minlength=n)

    # four LINEAR wall constraints per circle: r - (cx-LO), r - (HI-cx), r - (cy-LO), r - (HI-cy)
    h = np.stack([r - (cx - LO), r - (HI - cx), r - (cy - LO), r - (HI - cy)])
    tw = lam_w + rho * h
    tw = np.where(tw > 0.0, tw, 0.0)
    f += float((tw * tw).sum()) / (2.0 * rho)
    gr += tw.sum(axis=0)
    gcx += tw[1] - tw[0]
    gcy += tw[3] - tw[2]
    return f, np.concatenate([gcx, gcy, gr])


def _al_local(c, r, deadline, outer=7, maxiter=90, rho0=48.0, growth=3.6,
              margin=0.05, w=None, rcap=None, exact_r=True, rng=None):
    """Smooth local solve.  Returns exactly-feasible (c, r) via _repair.

    `w` (objective tilt) and `rcap` (radius ceiling, rcap_i = 0 squeezes circle i out) are the
    iteration-3 LP deformation knobs, here just an objective vector and an upper bound."""
    c = np.clip(np.asarray(c, dtype=float), LO, HI)
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    n = c.shape[0]
    wobj = np.ones(n) if w is None else np.asarray(w, dtype=float)
    hi_r = np.full(n, 0.5) if rcap is None else np.minimum(0.5, np.maximum(
        np.asarray(rcap, dtype=float), 0.0))
    r = np.minimum(r, hi_r)

    z = np.concatenate([c[:, 0], c[:, 1], r])
    bounds = [(LO, HI)] * (2 * n) + list(zip(np.zeros(n), hi_r))
    pi, pj = _near_pairs(c, r, margin)
    lam_p = np.zeros(pi.size)
    lam_w = np.zeros((4, n))
    rho = rho0

    for _ in range(outer):
        if time.process_time() > deadline:
            break
        try:
            res = minimize(_al_fg, z, args=(n, pi, pj, rho, lam_p, lam_w, wobj), jac=True,
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": maxiter, "maxcor": 12, "ftol": 1e-16,
                                    "gtol": 1e-14})
        except Exception:
            break
        z = np.asarray(res.x, dtype=float)
        cx, cy, rr = z[:n], z[n:2 * n], z[2 * n:]
        cc = np.stack([cx, cy], axis=1)
        # refresh the neighbour list (keeping multipliers) so no pair can drift in unwatched
        npi, npj = _near_pairs(cc, rr, margin)
        pi, pj, lam_p = _merge_pairs(pi, pj, lam_p, npi, npj, n)
        # multiplier update
        if pi.size:
            dxy = cc[pi] - cc[pj]
            d = np.sqrt((dxy * dxy).sum(axis=1)) + 1e-15
            lam_p = np.maximum(0.0, lam_p + rho * (rr[pi] + rr[pj] - d))
        h = np.stack([rr - (cx - LO), rr - (HI - cx), rr - (cy - LO), rr - (HI - cy)])
        lam_w = np.maximum(0.0, lam_w + rho * h)
        rho *= growth

    cc = np.clip(np.stack([z[:n], z[n:2 * n]], axis=1), LO, HI)
    cc = _unstack(cc)
    if not exact_r:
        return cc, _repair(cc, z[2 * n:])
    rr = _rlp(cc)
    if rcap is not None:
        return cc, _repair(cc, np.minimum(rr, hi_r))
    # RESCUE COLLAPSED CIRCLES.  Nothing in the smooth problem stops a circle from being driven to
    # r = 0 and then drifting into a pile with its neighbours (r >= 0 is a bound, so a null circle
    # feels no repulsion); such a pile is worth nothing and also degrades the LP polish that
    # follows.  Teleport every collapsed circle into the largest currently-empty disc instead --
    # the same hole-insert used by the structured moves -- and re-solve the radii exactly.
    med = float(np.median(rr)) if n else 0.0
    dead = np.flatnonzero(rr < 0.12 * med) if med > 0 else np.zeros(0, dtype=int)
    if dead.size and dead.size < n:
        keep = np.setdiff1d(np.arange(n), dead)
        cc2, _ = _hole_insert(cc[keep], rr[keep], int(dead.size), rng)
        if cc2.shape[0] == n:
            cc2 = _unstack(np.clip(cc2, LO, HI))
            rr2 = _rlp(cc2)
            if rr2.sum() > rr.sum():
                return cc2, rr2
    return cc, rr


def _unstack(c, eps=1e-7):
    """Deterministically separate coincident centres (zero-radius ghosts can pile up, and a
    zero distance makes the LP's contact normal undefined)."""
    d = _dists(c)
    bad = np.argwhere(d < 1e-9)
    if bad.size:
        c = c.copy()
        for i, j in bad:
            if i < j:
                ang = 2.0 * np.pi * ((i * 7919 + j * 104729) % 997) / 997.0
                c[j] = np.clip(c[j] + eps * np.array([np.cos(ang), np.sin(ang)]), LO, HI)
    return c

# ------------------------------------------------------------------------ LP DEFORMATION phase
def _lp_phase(c, r, iters, delta, rcap=None, w=None, shrink=0.88):
    """Run `iters` LP steps ACCEPTING every step -- no monotone gate.

    _ccp() only ever accepts an improvement in sum r, which is exactly why it is a trap: it has a
    huge basin of attraction and pulls any random perturbation straight back to the same fixed
    point (measured at n=63: 8 of 12 structured kicks returned the incumbent's sum to 9 digits).
    This phase deliberately optimises the WRONG objective for a few steps -- a squeezed-out circle
    (rcap_i = 0) or a re-weighted objective -- so the configuration FLOWS to a genuinely different
    part of the feasible set instead of being displaced and snapping back.  Every iterate is still
    exactly feasible (the LP is a restriction, and _repair enforces it numerically)."""
    c = np.clip(np.asarray(c, dtype=float), LO, HI)
    r = _repair(c, r)
    lp_state = {}
    for _ in range(iters):
        out = _ccp_lp(c, r, delta, delta, rcap=rcap, w=w, state=lp_state)
        if out is None:
            delta *= 0.5
            if delta < 1e-6:
                break
            continue
        c_new, r_new = out
        c, r = c_new, _repair(c_new, r_new)
        delta *= shrink
    return c, _repair(c, r)


# --------------------------------------------------------------------------- structured moves
_GRID = None


def _grid_pts(g=44):
    global _GRID
    if _GRID is None or _GRID[0] != g:
        t = np.linspace(LO, HI, g + 2)[1:-1]
        X, Y = np.meshgrid(t, t)
        _GRID = (g, np.stack([X.ravel(), Y.ravel()], axis=1))
    return _GRID[1]


def _hole_insert(c, r, k, rng=None):
    """Place k new circles greedily at the centre of the largest currently-empty disc."""
    P = _grid_pts()
    c = np.asarray(c, dtype=float)
    r = np.asarray(r, dtype=float)
    wallP = np.minimum.reduce([P[:, 0] - LO, HI - P[:, 0], P[:, 1] - LO, HI - P[:, 1]])
    out = list(c)
    rr = list(r)
    for _ in range(k):
        if out:
            A = np.asarray(out)
            R = np.asarray(rr)
            d = np.sqrt(((P[:, None, :] - A[None, :, :]) ** 2).sum(-1)) - R[None, :]
            score = np.minimum(wallP, d.min(axis=1))
        else:
            score = wallP
        j = int(np.argmax(score))
        p = P[j] + (rng.uniform(-0.004, 0.004, size=2) if rng is not None else 0.0)
        out.append(np.clip(p, LO, HI))
        rr.append(max(1e-6, float(score[j])))
    return np.asarray(out), np.asarray(rr)


def _squeeze_release(c, r, rng, n, sc=1.0):
    """SQUEEZE-AND-TELEPORT.  Cap k circles at radius 0 and let the LP re-optimise the remaining
    n-k against the true objective: the neighbours genuinely flow into the vacated room and the
    contact graph reorganises.  The k ghosts are then teleported into the largest empty discs and
    released.  Unlike a positional kick this cannot snap back, because the configuration it
    returns from is an optimum of a DIFFERENT problem."""
    k = int(rng.randint(1, _sck(n // 9, sc) + 1))
    if rng.rand() < 0.6:
        idx = np.argsort(r)[:k]
    else:
        idx = rng.choice(n, size=k, replace=False)
    rcap = np.full(n, 0.5)
    rcap[idx] = 0.0
    cc, rr = _lp_phase(c, r, 12, 0.10 / np.sqrt(n), rcap=rcap)
    keep = np.setdiff1d(np.arange(n), idx)
    out, _ = _hole_insert(cc[keep], rr[keep], k, rng)
    return out


def _reweight(c, r, rng, n, sc=1.0):
    """OBJECTIVE TILT.  Optimise sum w_i r_i for a few steps with log-normal weights, then hand the
    result back to the true (w = 1) CCP.  The tilt walks the LP to a different vertex -- a
    different set of active contacts -- with no random displacement at all."""
    w = np.exp(rng.normal(0.0, 0.5 * sc, size=n))
    cc, _ = _lp_phase(c, r, 10, 0.07 / np.sqrt(n), w=w)
    return cc


def _op_reinsert(c, r, rng, n, sc=1.0):
    """Delete the k smallest circles and re-insert them into the largest holes."""
    k = int(rng.randint(1, _sck(n // 8, sc) + 1))
    drop = np.argsort(r)[:k]
    keep = np.setdiff1d(np.arange(n), drop, assume_unique=False)
    cc, _ = _hole_insert(c[keep], r[keep], k, rng)
    return cc


def _op_swap(c, r, rng, n, sc=1.0):
    """Swap the k largest circles with the k smallest -- a pure relabelling of the radii."""
    c = c.copy()
    order = np.argsort(r)
    for t in range(max(1, int(round((n // 12) * sc)))):
        a, b = order[t], order[-1 - t]
        c[[a, b]] = c[[b, a]]
    return c


def _op_relocate(c, r, rng, n, sc=1.0):
    """Teleport a random handful of circles to uniform positions."""
    c = c.copy()
    k = max(1, int(rng.randint(1, _sck(n // 6, sc) + 1)))
    pick = rng.choice(n, size=k, replace=False)
    c[pick] = rng.uniform(LO + 0.01, HI - 0.01, size=(k, 2))
    return c


def _op_jitter(c, r, rng, n, sc=1.0):
    """Global gaussian jitter scaled to the typical radius.  The 0.35 is the iteration-2 constant
    and is now only the sc = 1.0 rung of the ladder -- see the ITERATION 12 note below."""
    return np.clip(c + rng.normal(0.0, 0.35 * sc / np.sqrt(n), size=c.shape), LO, HI)


# ============================================================================================
# ITERATION 10 -- the INFLATION channel: escape a rigid optimum by changing the OBJECTIVE
# ============================================================================================
# Diagnosis (artifacts/iter10/diag_local.py, offline, nothing metered): every one of the 28 n
# that still sits ~1e-3 short of its record is a RIGID local optimum.  Re-running the CCP from
# the committed pack for 8 CPU-s at delta_min = 1e-9, then an augmented-Lagrangian polish, then
# the CCP again, moves n = 29/33/43/49/57/79/93 by |dSum| <= 9e-13 -- machine noise.  The local
# machinery is not the problem: it converges to eleven digits (n=31 sits 1.1e-11 from its
# record).  Every existing arm is a PERTURBATION of the same max-sum-r landscape, and that
# landscape's basin walls are exactly what is holding these 28 n.
#
# So this arm does not perturb the configuration -- it changes the PROBLEM.  Freeze the radius
# profile, scale it up by lambda > 1, and minimise pure OVERLAP over the centres:
#
#       min_c  sum_{i<j} max(0, t_i + t_j - ||c_i - c_j||)^2 ,   t = lambda * r ,
#       subject to the box  |x_i| <= 1/2 - t_i  (containment is a BOUND once t is fixed)
#
# That is an unconstrained-in-spirit, bound-constrained smooth least-squares problem -- one
# L-BFGS-B solve with an analytic O(n^2) gradient, ~20 us per f+g.  Its landscape is NOT the Sum-r
# landscape: circles slide through each other's would-be contacts on the way down, so the
# configuration it lands on routinely has a DIFFERENT contact graph.  If the residual reaches 0
# the packing is feasible at lambda*r and Sum-r jumped by the full inflation in one step; if it does
# not, the rearranged centres are still handed to the CCP, which re-derives exact radii.  Either
# way the caller leaves the basin it could not climb out of.  This is the classical
# inflate-and-resolve move from the equal-circle record literature, transposed to a non-uniform
# radius profile, and it is the first arm in the set whose descent direction is not a Sum-r ascent.


def _inflate_fg(z, n, t):
    """f, grad of  sum_{i<j} max(0, t_i+t_j-d_ij)^2  in z = [x(n), y(n)]."""
    x, y = z[:n], z[n:]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    o = (t[:, None] + t[None, :]) - d
    np.fill_diagonal(o, 0.0)
    ow = np.where(o > 0.0, o, 0.0)
    f = 0.5 * float((ow * ow).sum())
    if f == 0.0:
        return 0.0, np.zeros(2 * n)
    q = ow / d                                  # d is +inf on the diagonal, so q is 0 there
    g = np.empty(2 * n)
    g[:n] = -2.0 * (q * dx).sum(axis=1)
    g[n:] = -2.0 * (q * dy).sum(axis=1)
    return f, g


def _inflate(c, t, maxiter=400):
    """Resolve the overlaps of the FIXED radius profile t by moving centres only.
    Returns (centres, residual).  residual == 0 means the packing (c_new, t) is feasible."""
    n = c.shape[0]
    t = np.minimum(np.asarray(t, dtype=float), 0.4999)
    lo, hi = LO + t, HI - t
    z0 = np.concatenate([np.clip(c[:, 0], lo, hi), np.clip(c[:, 1], lo, hi)])
    bnds = list(zip(np.concatenate([lo, lo]), np.concatenate([hi, hi])))
    try:
        res = minimize(_inflate_fg, z0, args=(n, t), jac=True, method="L-BFGS-B",
                       bounds=bnds, options={"maxiter": maxiter, "maxfun": 2 * maxiter,
                                             "ftol": 1e-18, "gtol": 1e-14})
        z = res.x
    except Exception:
        z = z0
    f, _ = _inflate_fg(z, n, t)
    return np.stack([z[:n], z[n:]], axis=1), float(f)


def _op_inflate(c, r, rng, n, sc=1.0, steps=10):
    """DEFLATE - JITTER - RE-INFLATE.  A single inflation from a rigid optimum is a no-op: the
    optimum is a local minimum of the overlap objective too, so L-BFGS-B walks straight back to it
    (measured: residual 5.8e-19, the identical packing).  What breaks the basin is a CONTINUATION.
    Shrink the profile to lo*r, where the configuration has genuine slack, jitter it, then walk the
    profile back up to lam*r in `steps` re-solves.  The circles reorganise while the room exists and
    lock in as it closes, so the contact graph the ramp ends on is generically NOT the one it
    started from -- and unlike a positional kick it is a packing, not a scramble.  Returns centres;
    the caller re-derives exact radii with the LP."""
    r = np.asarray(r, dtype=float)
    lo = 1.0 - (1.0 - float(rng.uniform(0.84, 0.965))) * sc     # sc deepens/softens the deflation
    lam = 1.0 + float(np.exp(rng.uniform(np.log(1e-4), np.log(6e-3)))) if rng.rand() < 0.5 else 1.0
    sig = float(rng.uniform(0.35, 1.8)) * sc * float(r.mean())
    prof = r * np.exp(rng.normal(0.0, 0.04, size=n)) if rng.rand() < 0.3 else r
    cc = np.clip(np.asarray(c, dtype=float) + rng.normal(0.0, sig, size=(n, 2)), LO, HI)
    for k in range(steps + 1):
        cc, _ = _inflate(cc, (lo + (lam - lo) * (k / float(steps))) * prof, maxiter=220)
    return np.clip(cc, LO, HI)


# ============================================================================================
# ITERATION 11 -- the TOPOLOGICAL channel: T1 transitions on the contact graph
# ============================================================================================
# What actually defines the basin a rigid optimum sits in is not its coordinates -- it is its
# CONTACT GRAPH.  Every arm the solver owns so far moves coordinates and hopes the graph follows:
# jitter/relocate displace, squeeze/reinsert delete-and-refill, cross grafts halves, and the
# iteration-10 inflation ramp opens room and lets the graph re-form on the way back up.  All of
# them are indirect, and all of them are DIFFUSE -- they discard, on average, a large part of a
# structure that is already correct almost everywhere (the weak n are ~1e-3 short, i.e. one or
# two cells wrong out of n).
#
# The direct move exists and is standard in the physics of jammed matter and foams: the T1
# transition.  Take the Delaunay triangulation of the centres -- a superset of the contact graph
# for any packing -- and find an interior edge (i,j) shared by triangles (i,j,k) and (i,j,l).
# The quadrilateral i-k-j-l has two possible diagonals; the packing is using (i,j).  Pull i and j
# apart along their axis while pushing k and l together along theirs, just far enough that (k,l)
# becomes the short diagonal, and the neighbour relation is REWIRED: i and j stop touching, k and
# l start.  Everything outside that one quadrilateral is untouched.
#
# That is the minimal basin hop this problem admits, and it is a different KIND of move from the
# other nine: it is combinatorial (it names the graph edge it is changing), it is local (O(1)
# circles move), and it is exhaustively enumerable (a packing has ~3n interior Delaunay edges, so
# the arm can be pointed at a specific defect rather than sampling a neighbourhood).  It also
# hands the search a structure the other arms do not have -- an explicit adjacency -- which is
# what a targeted repair operator would need next.

try:
    from scipy.spatial import Delaunay as _Delaunay
except Exception:                                  # pragma: no cover -- scipy always has qhull
    _Delaunay = None


def _flip_edges(c):
    """Interior Delaunay edges with their two opposite vertices.

    Returns (i, j, k, l) index arrays -- edge (i,j) is shared by triangles (i,j,k) and (i,j,l) --
    or None when the triangulation is unavailable or degenerate (collinear/duplicate centres)."""
    c = np.asarray(c, dtype=float)
    n = c.shape[0]
    if _Delaunay is None or n < 4:
        return None
    try:
        tri = np.asarray(_Delaunay(c).simplices)
    except Exception:
        return None
    if tri.ndim != 2 or tri.shape[0] == 0:
        return None
    a = tri[:, [0, 1, 2]].ravel()
    b = tri[:, [1, 2, 0]].ravel()
    o = tri[:, [2, 0, 1]].ravel()                  # vertex opposite the edge (a,b)
    ei, ej = np.minimum(a, b), np.maximum(a, b)
    key = ei.astype(np.int64) * n + ej
    order = np.argsort(key, kind="stable")
    key, ei, ej, o = key[order], ei[order], ej[order], o[order]
    first = np.flatnonzero(key[1:] == key[:-1])    # an interior edge appears exactly twice
    if first.size == 0:
        return None
    keep = o[first] != o[first + 1]                # paranoia: a degenerate duplicate is not a flip
    first = first[keep]
    if first.size == 0:
        return None
    return ei[first], ej[first], o[first], o[first + 1]


def _op_flip(c, r, rng, n, sc=1.0, kmax=3):
    """T1 TRANSITION.  Rewire between 1 and `kmax` Delaunay quadrilaterals, then let the overlap
    solver settle the result at a slightly deflated profile so the new adjacency is realised as a
    packing rather than a collision.  Returns centres; the caller re-derives exact radii with the
    LP, because the whole point of the move is that the radius profile the new graph supports is
    NOT the one it started from."""
    c = np.asarray(c, dtype=float)
    r = np.asarray(r, dtype=float)
    ed = _flip_edges(c)
    if ed is None:
        return None
    ei, ej, ok, ol = ed
    d = _dists(c)
    dij, dkl = d[ei, ej], d[ok, ol]
    room = dkl - dij                               # cost of swapping the diagonal
    cand = np.flatnonzero(room > 1e-9)
    if cand.size == 0:
        return None
    rbar = max(1e-12, float(r.mean()))
    slack = np.maximum(dij[cand] - (r[ei[cand]] + r[ej[cand]]), 0.0)
    # prefer edges that BIND (a contact is load-bearing, a slack edge is not holding anything)
    # and quadrilaterals that are nearly square (small `room`), where the swap is cheapest
    w = np.exp(-slack / (0.35 * rbar)) / (1.0 + room[cand] / rbar)
    w = w / w.sum()
    k = min(int(rng.randint(1, max(1, int(round(kmax * sc))) + 1)), cand.size)
    pick = cand[rng.choice(cand.size, size=k, replace=False, p=w)]
    over = float(rng.uniform(0.10, 0.60))
    defl = float(rng.uniform(0.90, 0.98))
    out = None
    # The settle below can walk the graph straight back: a T1 whose overshoot is too small is
    # undone by the very relaxation that is supposed to realise it.  So the arm VERIFIES its own
    # move -- it re-triangulates the settled centres and checks that a targeted edge really is
    # gone -- and escalates the overshoot when it is not.  (Measured from the n=49 pack: one shot
    # changes the Delaunay edge set 5/8 of the time, up to three escalating shots 7/8.)
    for attempt in range(3):
        cc = c.copy()
        fired = 0
        for e in pick:
            i, j, kk, ll = int(ei[e]), int(ej[e]), int(ok[e]), int(ol[e])
            vij, vkl = cc[j] - cc[i], cc[ll] - cc[kk]
            nij, nkl = float(np.hypot(*vij)), float(np.hypot(*vkl))
            if nij < 1e-12 or nkl < 1e-12:
                continue
            # after the move the (k,l) diagonal is shorter than (i,j) by the sampled overshoot
            s = 0.25 * ((nkl - nij) + over * (1.9 ** attempt) * rbar)
            if s <= 0.0:
                continue
            u, v = vij / nij, vkl / nkl
            cc[i] -= s * u
            cc[j] += s * u
            cc[kk] += s * v
            cc[ll] -= s * v
            fired += 1
        if fired == 0:
            return out
        # settle: resolve the collision the swap deliberately created, with the profile deflated
        # so there is room for the new adjacency to exist at all
        cc, _ = _inflate(cc, defl * r, maxiter=260)
        cc = np.clip(cc, LO, HI)
        out = cc
        ne = _flip_edges(cc)
        if ne is None:
            return out
        gone = set(zip(ei[pick].tolist(), ej[pick].tolist())) - set(
            zip(ne[0].tolist(), ne[1].tolist()))
        if gone:
            return out
    return out


def _move(c, r, rng, n):
    """The iteration-2/4 RANDOM MIXTURE of the structured perturbations, kept intact: it is the
    fallback the adaptive selector of iteration 9 always keeps on the table."""
    u = rng.rand()
    if u < 0.38:
        return _squeeze_release(c.copy(), r, rng, n)
    if u < 0.58:
        return _reweight(c, r, rng, n)
    if u < 0.75:
        return _op_reinsert(c, r, rng, n)
    if u < 0.85:
        return _op_swap(c, r, rng, n)
    if u < 0.95:
        return _op_relocate(c, r, rng, n)
    return _op_jitter(c, r, rng, n)


# ============================================================================================
# ITERATION 8 -- the chain stops being a single trajectory: POPULATION + STRUCTURAL CROSSOVER
# ============================================================================================
# The census after iteration 7 is bimodal.  Eight of the 37 n sit within 0.0015 of their record
# (four of them past it); the other 29 sit 0.003-0.017 short -- an absolute deficit far larger
# than any local-convergence slop, and one that a 60-step CCP polish of the committed pack does
# not touch.  That is not a numerics problem, it is a BASIN problem: those n are converged to the
# wrong contact structure, and every quantum the auction hands them re-perturbs the SAME
# incumbent, so the search re-samples one structure's neighbourhood over and over.
#
# So the worker stops carrying one configuration and starts carrying a POPULATION -- an archive
# of structurally DISTINCT local optima (distinctness judged on the sorted radius spectrum, which
# is invariant to the relabelling and reflection that make two copies of one basin look
# different).  Two new channels feed it, and neither exists as a perturbation of a single point:
#
#   CROSSOVER   -- cut two archive members with a random half-plane and graft A's side onto B's.
#                  The child inherits a whole sub-structure from each parent instead of a jitter,
#                  and the seam is repaired by dropping the most-overlapped circles and refilling
#                  the holes.  This is the one move whose reach grows with the archive.
#   IMMIGRATION -- the census is 37 CORRELATED problems, and until now they were solved as 37
#                  independent ones (transfer went upward only, and only from n solved earlier in
#                  the same call).  Every n now also seeds from the COMMITTED packs at n+1 / n+2
#                  by deleting its worst circles.  n=99 sits at 4.7e-6 from its record and n=97 at
#                  1.6e-3; the good structure is next door, on disk, and was never being read.


def _drop_to(c, r, n):
    """Reduce a configuration to EXACTLY n circles, discarding the circles that pay least: at each
    step the one with the largest overlap debt minus its own radius.  On a feasible input this is
    just 'delete the smallest', but on a crossover seam it deletes the collision instead."""
    c = np.asarray(c, dtype=float).copy()
    r = np.asarray(r, dtype=float).copy()
    while c.shape[0] > n:
        ov = np.maximum(0.0, (r[:, None] + r[None, :]) - _dists(c))
        np.fill_diagonal(ov, 0.0)
        j = int(np.argmax(ov.sum(axis=1) - r))
        keep = np.arange(c.shape[0]) != j
        c, r = c[keep], r[keep]
    return c, r


def _crossover(a, b, rng, n):
    """SPATIAL RECOMBINATION.  Slice the square with a random line and take one parent's circles on
    each side.  Returns centres only -- the caller's CCP re-derives the radii, which is what makes
    the seam heal instead of being uniformly shrunk away."""
    th = rng.rand() * np.pi
    u = np.array([np.cos(th), np.sin(th)])
    pa, pb = a[1] @ u, b[1] @ u
    t = float(np.quantile(pa, rng.uniform(0.30, 0.70)))
    sa, sb = pa < t, pb >= t
    c = np.concatenate([a[1][sa], b[1][sb]], axis=0)
    r = np.concatenate([a[2][sa], b[2][sb]], axis=0)
    if c.shape[0] > n:
        c, r = _drop_to(c, r, n)
    elif c.shape[0] < n:
        c, r = _hole_insert(c, r, n - c.shape[0], rng)
    if c.shape[0] != n:
        return None
    return _unstack(np.clip(c, LO, HI))



# ----------------------------------------------------------------------------- initialisations
def _read_pack(n):
    """Warm start from the committed census -- GUARDED: any n without a pack cold-starts."""
    p = "bench/packs/csqv%d.pck" % n
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        rows = []
        for ln in lines[2:]:
            parts = ln.split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(rows) != n:
            return None
        a = np.asarray(rows, dtype=float)
        if not np.all(np.isfinite(a)):
            return None
        return np.clip(a[:, :2], LO, HI), np.abs(a[:, 2])
    except Exception:
        return None


def _init_rows(n, k, stagger):
    if k < 1:
        k = 1
    counts = np.full(k, n // k, dtype=int)
    counts[:n % k] += 1
    if stagger and k > 1:
        for i in range(1, k, 2):
            if counts[i] > 1:
                counts[i] -= 1
                counts[(i - 1) % k] += 1
    if counts.sum() != n or (counts < 1).any():
        counts = np.full(k, n // k, dtype=int)
        counts[:n % k] += 1
    pts = []
    for i, cnt in enumerate(counts):
        y = LO + (i + 0.5) / k
        xs = LO + (np.arange(cnt) + 0.5) / cnt
        for x in xs:
            pts.append((x, y))
    return np.asarray(pts, dtype=float)[:n]


def _cold_starts(n, rng, k_rand=6):
    out = []
    k0 = max(1, int(round(np.sqrt(n))))
    for k in (k0, k0 + 1, k0 - 1):
        if 1 <= k <= n:
            out.append(_init_rows(n, k, True))
            out.append(_init_rows(n, k, False))
    for _ in range(k_rand):
        out.append(rng.uniform(LO + 0.02, HI - 0.02, size=(n, 2)))
    return [a for a in out if a.shape[0] == n]


# ============================================================================================
# ITERATION 7 -- the census stops being a for-loop: RESUMABLE WORKERS + a MARGINAL-GAIN AUCTION
# ============================================================================================
# Iterations 5 and 6 bought their gains by making a basin cheaper.  The stopwatch has now moved
# somewhere else: CPU is split in proportion to n, which equalises the BASIN COUNT across the 37
# sizes -- but the score is a mean of DIGITS, and a basin is worth wildly different numbers of
# digits at different n.  After iteration 6 the census runs from 2.25 digits (n=35, relgap 5.6e-3)
# to 5.33 (n=99, relgap 4.7e-6); a proportional split hands n=99 the largest slice of all while it
# sits four orders of magnitude closer to its record than n=35 does.
#
# So the allocation is no longer a formula over n -- it is MEASURED.  Each n becomes a _Worker
# that owns its sink, its seed queue and its annealed chain, and can be advanced by an arbitrary
# slice of CPU and then put down again with all of that state intact.  solve() then runs
#   ROUND 1 (probe)   -- every n gets a slice proportional to n, and reports the digits it bought,
#   ROUND 2 (auction) -- quanta go to whoever has the best optimistic recent digits-per-second.
# Nothing here can regress an n: the driver only overwrites a committed pack with a strictly
# better one, so a size the auction ignores simply keeps what it had.
#
# The worker split is also what makes the next moves possible at all -- interleaved cross-n
# transfer, per-n engine choice, early retirement of a converged size -- none of which a single
# pass over `for n in tgts` can express.

_RECORDS = None


def _records():
    """The record table, read (never imported) from bench/records.json.  Used ONLY to price the
    auction's bids -- an n missing from it is priced against a local reference instead."""
    global _RECORDS
    if _RECORDS is None:
        tbl = {}
        try:
            with open("bench/records.json") as fh:
                raw = json.load(fh)
            src = raw.get("records", raw) if isinstance(raw, dict) else {}
            for k, v in src.items():
                try:
                    tbl[int(k)] = float(v)
                except (TypeError, ValueError):
                    pass
        except Exception:
            tbl = {}
        _RECORDS = tbl
    return _RECORDS


DIGITS_CAP = 7.0


def _pack_digits(n, ref):
    """Digits the COMMITTED pack for n already scores -- read straight off disk, before any
    worker is opened.  Returns 0.0 for a missing or unparseable pack, so a fresh n is never
    mistaken for a capped one."""
    try:
        pk = _read_pack(n)                          # None for a missing/short/unparseable pack
    except Exception:
        return 0.0
    if pk is None:
        return 0.0
    r = pk[1]
    if r is None or len(r) != n:
        return 0.0
    return _digits(float(np.sum(r)), ref)


def _digits(s, ref):
    """The scorer's currency: clamp(-log10(relgap), 0, 7) against a reference value."""
    if ref <= 0 or s is None or not np.isfinite(s):
        return 0.0
    gap = (ref - s) / ref
    if gap <= 1e-7:
        return DIGITS_CAP
    return float(min(DIGITS_CAP, max(0.0, -np.log10(gap))))


# ============================================================================================
# ITERATION 9 -- WHICH MOVE, not just which n: an adaptive operator selector
# ============================================================================================
# Iteration 7 stopped guessing HOW MUCH CPU each n deserves and started measuring it (the
# marginal-gain auction).  One level down, every remaining allocation is still a hard-coded
# guess: `_move` fires its six perturbations on fixed probability thresholds (0.38 / 0.58 / 0.75
# / 0.85 / 0.95) and the smooth AL engine is switched on for a fixed AL_SHARE = 0.25 of steps.
# A cProfile of one 6 s quantum at n=35 says those guesses are expensive: `_al_local` burns
# **47 % of the CPU for 25 % of the steps** (20 203 `_al_fg` evaluations, 4.1 s of 8.7 s), while
# the whole LP path -- 3 799 CCP steps -- costs less.  Nothing in the solver has ever checked
# whether that trade pays.
#
# So the auction's own argument is applied to the MOVE SET: each operator is an arm, priced in
# exactly the currency the scorer pays in (digits per CPU-second), with the same optimistic
# prior and the same discount, so an arm that stops paying decays to a bounded floor and is
# revisited rather than deleted -- no line of attack is ever closed off.  The stats are SHARED
# by all 37 workers, because a single n's 3 s slice is far too little evidence on its own.
OPS = ("squeeze", "reweight", "reinsert", "swap", "relocate", "jitter", "inflate",
       "flip", "cross", "al")
OP_BETA = 0.75           # discount per pull: an arm that paid once holds the selector for
                         # ~30 further empty pulls, then decays back into the rotation
OP_PRIOR_G = 0.004       # optimism: digits an untested/failing arm is still credited with...
OP_PRIOR_T = 0.15        # ...over this many seconds (~one step), so nothing is starved to zero
OP_NEW_BASIN = 0.02      # shaping: a step that files a structurally NEW basin has done work...
OP_ACCEPT = 0.01         # ...as has one the annealed chain accepts, even with no record gain

# ============================================================================================
# ITERATION 13 -- the chain was being PAID TO STAND STILL, and it had learnt to
# ============================================================================================
# artifacts/iter13/diag_accept.py replays the shipped chain loop verbatim off the committed
# packs and logs where every settled step LANDS relative to its own base.  Two measurements,
# 15 CPU-s per n at n = 33 / 49 / 71:
#
#   * 60.8 % / 54.9 % / 90.3 % of all steps come back to the point they started from -- the kick
#     is inside the local solver's recovery radius, the CCP undoes it exactly, |ds| < 1e-9.
#   * Those NULL steps collect the large majority of every reward the bandits pay out:
#     3.04 vs 0.69 (n=33), 1.69 vs 0.69 (n=49), 2.23 vs 0.17 (n=71) -- because a step that went
#     nowhere trivially satisfies  s > cur - TOL_REL|cur|  and banks OP_ACCEPT for it.
#
# So the reward had a DEGENERATE OPTIMUM: the cheapest way to earn is to do nothing, credibly.
# And the selectors had found it.  Over 500 steps at n=33 the operator bandit spent 477 pulls on
# ONE arm and gave EIGHT of the ten arms zero pulls; the scale bandit put 96 % of its pulls on one
# rung and never sampled two of the four.  The optimistic prior (0.004/0.15 = 0.027) cannot
# rotate anything back in, because the incumbent keeps re-earning 0.01 per null step and holds an
# index of 0.10-0.16 forever.  That is why iteration 11's new arm only ever saw ~30 pulls.
#
# Two changes, one principle -- a move is only worth paying for if it MOVED:
#   (1) the OP_ACCEPT bonus is gated on the step having left its basin, so a null step earns
#       nothing while still being charged its CPU.  Null steps become maximally unattractive and
#       the arms are priced by their real escape yield.
#   (2) every arm becomes SELF-VERIFYING, generalising the escalation iteration 11 built for
#       `flip` alone: when the settled point is the same basin it started from, the same operator
#       is re-fired at the NEXT RUNG UP, up to ESC_MAX times.  This does not push the search to
#       big kicks -- it stops at the FIRST rung that escapes, which is exactly the "returns AND
#       escapes" regime iteration 12's ladder identified as the productive one.
NULL_TAU = 1e-9          # relative Sum-r distance below which a settled step is the SAME point
ESC_MAX = 2              # extra shots an arm gets to actually leave its basin before it gives up

# ============================================================================================
# ITERATION 14 -- the payoff is a LOTTERY, so buy tickets, do not stare at one
# ============================================================================================
# artifacts/iter14/diag_depth.py gives one n the CPU of four and logs the digits after every
# 2-CPU-s quantum.  The curve is not concave -- it is FLAT with rare cliffs:
#
#     n=33  2.603 -> 2.603 ... 2.603      (24 CPU-s, one chain: nothing, at all)
#     n=47  2.918 -> 2.918 ... 2.918      (24 CPU-s, one chain: nothing, at all)
#     n=61  3.228 -> 3.344 ......... 3.945        (two cliffs, at 4 s and at 22 s)
#     n=29  2.741 -> 7.000 ... 7.000      (the CAP, inside the FIRST 2 s)
#
# artifacts/iter14/diag_stuck.py then reruns the SAME n from the SAME committed incumbent under
# four different RNG seeds, 6 CPU-s each:
#
#     n=33  warm 2.603  ->  2.603  2.603  7.000  2.603
#     n=37  warm 3.017  ->  3.017  3.019  3.019  3.284
#     n=47  warm 2.918  ->  2.918  2.925  2.918  2.918
#
# One draw in four takes n=33 from 2.603 digits to the 7-digit CAP in SIX seconds, while a single
# chain given TWENTY-FOUR seconds moves it by nothing.  The escape is not bought with time; it is
# bought with an independent draw.  The runtime distribution is heavy-tailed, and the hazard rate
# DECAYS with chain age -- an old chain has already committed to its region (`cur` sits in one
# basin, `pop` fills with near-copies of it, and `_Bandit.pick` is a deterministic argmax, so the
# only thing that ever differed between those four seeds was the operator RNG stream).
#
# Iteration 7 made the worker RESUMABLE so an n's quanta compose into one long chain.  That is
# precisely the wrong shape for this distribution, and it is what the depth curve is measuring.
# So the chain is cut into EPISODES: each is an independent draw from the champion, and the
# cutoff follows LUBY -- 1,1,2,1,1,2,4,... units -- which is within a log factor of the best
# fixed cutoff for ANY runtime distribution and needs no estimate of one.  Short episodes harvest
# the common fast escapes, the doublings still reach the rare slow ones, and no single line of
# attack ever gets to absorb the whole budget.  Episodes are metered in the n's OWN CPU, not the
# wall/auction clock, so the schedule survives being interleaved across 37 workers.
#
# ITERATION 15 -- the ticket PRICE is derived, not guessed.
# Iteration 14 shipped EP_UNIT = 1.5 CPU-s, a constant of the same order as an auction quantum,
# and named it as the untested allocation.  artifacts/iter15/diag_epunit.py sweeps it on the real
# _Worker from the real committed incumbents, 4 n x 4 seeds x 6 CPU-s (total digits over the 16
# runs):
#
#     EP_UNIT     0.25        0.60        1.50 (shipped)
#     total      59.829      67.561      63.939
#     n=29       2/4 escapes  4/4         3/4
#     episodes   14 per run   7           3
#
# The optimum is INTERIOR, and the reason is that Luby was never getting to run.  One full Luby
# cycle is 1,1,2,1,1,2,4 = LUBY_CYCLE units; at 1.5 s an n that owns ~7 CPU-s realises only the
# 1,1,2 PREFIX -- it never reaches a doubling, so the schedule with a worst-case guarantee had
# quietly degenerated into three short fixed-cutoff draws.  At 0.25 the cycle completes twice over
# but each unit is too short to reach an escape at all.
#
# So the unit is not a constant to re-tune: it is whatever makes ONE Luby cycle fit the CPU the n
# will actually receive.  solve() divides the per-payable-n share of the budget by LUBY_CYCLE and
# hands it to the workers, so the schedule re-derives itself from the budget, the target count and
# the number of n still payable rather than from a number measured on one machine at one budget.
# In the shipped configuration (~111 CPU-s, 16 payable n) that lands on 0.58 -- the measured
# winner, arrived at from the budget instead of from the sweep -- and with len(targets) == 1 it
# widens to a single cycle over the whole call instead of 74 truncated prefixes.
EP_UNIT = 1.5            # fallback Luby unit for a _Worker built outside solve() (diagnostics)
EP_ARCH_P = 0.25         # ...and one restart in four relaunches from a DRAWN archive member
                         #    instead of the champion, so a second structure stays alive
LUBY_CYCLE = 12.0        # sum of one full Luby cycle: 1+1+2+1+1+2+4
EP_UNIT_LO, EP_UNIT_HI = 0.15, 4.0    # the derived unit is clamped into a sane band


def _luby(i):
    """Luby's universal restart sequence, 1-indexed: 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,..."""
    i = int(i)
    if i < 1:
        return 1
    k = 1
    while True:
        span = (1 << k) - 1
        if i == span:
            return 1 << (k - 1)
        if i < span:
            return _luby(i - (1 << (k - 1)) + 1)
        k += 1


def _ep_unit(avail, n_payable):
    """CPU in one Luby unit, derived so that ONE full cycle (LUBY_CYCLE units) fits the CPU a
    payable n will actually receive.  Clamped, and safe for n_payable <= 0."""
    share = float(avail) / max(1.0, float(n_payable))
    return float(min(EP_UNIT_HI, max(EP_UNIT_LO, share / LUBY_CYCLE)))


def _is_null(s, r, cs, cr):
    """Did this settled step come back to where it started?  Sum-r to NULL_TAU relative AND the
    sorted radius spectrum to POP_TAU -- the same basin signature `_pop_add` files by, so `null`
    and `structurally new` are decided on one consistent notion of identity."""
    if r is None or cr is None or cs is None or not np.isfinite(s):
        return False
    if abs(s - cs) > NULL_TAU * max(1e-12, abs(cs)):
        return False
    a, b = np.asarray(r, dtype=float), np.asarray(cr, dtype=float)
    if a.shape != b.shape:
        return False
    return float(np.abs(np.sort(a) - np.sort(b)).max()) < POP_TAU

# ============================================================================================
# ITERATION 12 -- the second bandit DIMENSION: how far a move kicks is measured, not guessed
# ============================================================================================
# Iteration 9 armed WHICH move to make; the size of that move stayed a hard-coded constant in
# every arm (jitter's 0.35/sqrt(n), reinsert's n//8, flip's kmax=3, the inflation ramp's 0.84
# floor).  artifacts/iter12/diag_scale.py measures what those constants buy.  Sweeping jitter's
# multiplier over a ladder and running the FULL CCP from each kick, 16 draws at n = 33 / 49 / 71:
#
#     mult    frac of draws recovering to the incumbent      median delta Sum-r
#     0.02        1.00 / 0.94 / 0.75                            -2e-13   (nothing happens)
#     0.10        0.75 / 0.69 / 0.25                            -2e-13   (returns AND escapes)
#     0.20        0.19 / 0.12 / 0.00                            -1e-2
#     0.35        0.00 / 0.00 / 0.00                            -4e-2    <-- the shipped constant
#
# The shipped 0.35 is the WORST rung on the ladder: at all three n not one draw in sixteen even
# gets back to where it started, and the median kick throws away 4e-2 -- ten times the entire
# remaining gap to the record.  Every such step is rejected by the annealed chain, so the arm has
# been paying full CCP price for a guaranteed reject.  At 0.10 the same arm both returns (69 % at
# n=49) and escapes UPWARD: 16 draws found +1.81e-04 over the committed incumbent in 34 ms.
#
# The fix is not a better constant -- that would be one more guess.  Scale becomes a SECOND arm
# set, priced by the same optimistic discounted index in the same currency (digits per CPU-s),
# FACTORISED against the operator: the op bandit picks what to do, the scale bandit picks how far,
# and both are credited with the reward of the step they jointly produced.  Factorising keeps the
# evidence per arm high -- 4 rungs shared by 10 operators and 37 n, rather than 40 sparse pairs --
# and the ladder CONTAINS the old constant at sc = 1.0, so the previous behaviour is a rung the
# selector can always return to rather than something this change throws away.
SCALES = (0.20, 0.35, 0.60, 1.00)   # multipliers on every arm's shipped move size


class _Bandit:
    """Optimistic discounted index over an arm set.  `pick` is the auction's index, `credit` its
    update -- deliberately the same shape, so all three allocators (n, operator, scale) behave the
    same way.  Iteration 12 instantiates it twice: once over OPS, once over SCALES."""

    def __init__(self, names=OPS, label="op"):
        self.label = label
        self.names = list(names)
        self.G = dict.fromkeys(self.names, 0.0)
        self.T = dict.fromkeys(self.names, 0.0)
        self.pulls = dict.fromkeys(self.names, 0)
        self.cpu = dict.fromkeys(self.names, 0.0)
        self.gain = dict.fromkeys(self.names, 0.0)

    def index(self, k):
        return (self.G[k] + OP_PRIOR_G) / (self.T[k] + OP_PRIOR_T)

    def pick(self, allowed):
        best, bix = None, -np.inf
        for k in allowed:
            ix = self.index(k)
            if ix > bix:
                bix, best = ix, k
        return best

    def credit(self, k, g, t):
        self.G[k] = OP_BETA * self.G[k] + max(0.0, g)
        self.T[k] = OP_BETA * self.T[k] + max(0.0, t)
        self.pulls[k] += 1
        self.cpu[k] += t
        self.gain[k] += max(0.0, g)


class _Worker:
    """Everything one n's search knows, made RESUMABLE.  run(quota) advances the SAME annealed
    chain by a slice of CPU and returns (digits bought, CPU spent) -- the bid the auction reads."""

    def __init__(self, n, evaluate, meter, rng, bandit=None, sbandit=None):
        self.n = n
        self.rng = rng
        self.bandit = bandit if bandit is not None else _Bandit()
        # the SCALE dimension: same index, same currency, factorised against the operator
        self.sbandit = sbandit if sbandit is not None else _Bandit(SCALES, label="scale")
        self.sink = _Sink(evaluate, meter, n)
        self.seeds = None            # None = not opened yet; [] = seeding done, chain running
        self.best = [-np.inf, None, None]
        self.cur = None              # the annealed chain's current base, carried across quanta
        self.pop = []                # iteration 8: archive of structurally DISTINCT local optima
        self.barren = 0
        self.ep_i = 0                # Luby index of the current restart episode (iteration 14)
        self.ep_left = 0.0           # CPU still owed to it, metered in THIS n's own time
        self.ep_unit = EP_UNIT       # iteration 15: solve() overwrites this with the DERIVED unit
        self.ep_n = 0                # episodes opened, for the self-test and the report
        self.ref = None
        self.G = 0.0                 # discounted digits bought   (auction numerator)
        self.T = 0.0                 # discounted CPU spent       (auction denominator)
        self.cpu = 0.0               # undiscounted totals, for the report
        self.gain = 0.0

    def digits(self):
        if self.ref is None or self.sink.pack is None:
            return 0.0
        return _digits(self.sink.best, self.ref)

    def _pop_add(self, s, c, r):
        """File a local optimum in the archive.  Two entries count as the SAME basin when their
        sorted radius spectra agree to POP_TAU -- a label- and reflection-invariant signature, so
        the archive holds distinct STRUCTURES rather than distinct coordinate lists."""
        if c is None or r is None or not np.isfinite(s) or s <= 0.0:
            return
        sig = np.sort(np.asarray(r, dtype=float))
        for e in self.pop:
            if np.abs(e[3] - sig).max() < POP_TAU:
                if s > e[0]:
                    e[0], e[1], e[2], e[3] = s, c, r, sig
                return
        self.pop.append([s, c, r, sig])
        if len(self.pop) > POP_MAX:
            self.pop.sort(key=lambda e: -e[0])
            self.pop = self.pop[:POP_MAX]

    def _pop_pick(self):
        """Rank-biased draw: u**2 concentrates on the best entries without ever excluding the
        worst, which is the member most likely to be a genuinely different structure."""
        if not self.pop:
            return None
        self.pop.sort(key=lambda e: -e[0])
        return self.pop[min(len(self.pop) - 1, int(len(self.pop) * self.rng.rand() ** 2))]

    def _open(self, prev):
        """First quantum only: submit a guaranteed-feasible incumbent, fix the pricing reference,
        and build the seed queue.  Seeds are POPPED across quanta, so a probe too short to finish
        seeding resumes exactly where it stopped instead of restarting."""
        n = self.n
        warm = _read_pack(n)
        if warm is not None:
            wc, wr = warm
            self.sink.add(_pack(wc, _repair(wc, wr)))
        else:
            c0 = _init_rows(n, max(1, int(round(np.sqrt(n)))), True)
            self.sink.add(_pack(c0, _greedy_radii(c0)))
        self.sink.flush()

        # Pricing reference: the record when the census has one, else a fixed optimistic bound 1 %
        # above the opening incumbent.  An n OUTSIDE the visible census therefore still produces a
        # comparable log-scale bid -- no KeyError, no special case, no per-n table.
        s0 = self.sink.best if self.sink.pack is not None else 0.0
        R = _records().get(n)
        self.ref = R if (R is not None and R > s0) else (s0 * 1.01 if s0 > 0 else 1.0)

        seeds = []
        if warm is not None:
            # an already-committed pack is a CCP fixed point -- polish it, do not re-derive it
            seeds.append((warm[0], warm[1], 60, 0.05 / np.sqrt(n), False))
        for m in (n - 2, n - 1):
            src = prev.get(m) or _read_pack(m)        # this run's result, else the committed pack
            if src is not None and 1 <= m < n:
                cc, cr = _hole_insert(src[0], src[1], n - m, self.rng)
                seeds.append((cc, cr, 200, None, False))
        # IMMIGRATION FROM ABOVE: delete the worst circles of a LARGER committed pack.  Nothing in
        # the solver read downward before, and the census's best structures live at the top end.
        for m in (n + 1, n + 2):
            src = prev.get(m) or _read_pack(m)
            if src is not None and src[0].shape[0] == m:
                cc, cr = _drop_to(src[0], src[1], n)
                seeds.append((cc, cr, 200, None, False))
        if warm is None:
            # a cold start is exactly where the smooth engine wins: AL-presolve, then CCP-finish
            for a in _cold_starts(n, self.rng):
                seeds.append((a, None, 200, 0.03 / np.sqrt(n), True))
        self.seeds = seeds

    def _apply(self, op, deadline, sc=1.0):
        """Produce one candidate with the named operator at ladder multiplier `sc`.  Returns
        (centres, radii-hint, use_al).  Every arm is total: an operator that cannot fire right now
        returns (None, None, False) and is charged for the time it spent, never silently skipped."""
        n, rng = self.n, self.rng
        c, r = self.cur[1], self.cur[2]
        if op == "squeeze":
            return _squeeze_release(c.copy(), r, rng, n, sc), None, False
        if op == "reweight":
            return _reweight(c, r, rng, n, sc), None, False
        if op == "reinsert":
            return _op_reinsert(c, r, rng, n, sc), None, False
        if op == "swap":
            return _op_swap(c, r, rng, n, sc), None, False
        if op == "relocate":
            return _op_relocate(c, r, rng, n, sc), None, False
        if op == "jitter":
            return _op_jitter(c, r, rng, n, sc), None, False
        if op == "inflate":
            ci = _op_inflate(c, r, rng, n, sc)
            # exact optimal radii for the rearranged centres -- the greedy fallback would throw
            # away most of the room the inflation just opened
            return ci, _repair(ci, _rlp(ci)), False
        if op == "flip":
            cf = _op_flip(c, r, rng, n, sc)
            if cf is None:
                return None, None, False
            return cf, _repair(cf, _rlp(cf)), False
        if op == "cross":
            pa, pb = self._pop_pick(), self._pop_pick()
            if pa is None or pb is None or pa is pb:
                return None, None, False
            return _crossover(pa, pb, rng, n), None, False
        # "al": the SMOOTH engine of iteration 4 -- the random mixture, then an augmented-
        # Lagrangian presolve before the CCP.  Isolated as its own arm precisely because the
        # profile says it is the single most expensive thing the chain does.
        return _move(c, r, rng, n), None, True

    def _restart(self):
        """Open the next restart EPISODE: an independent draw, on the Luby schedule.  The RNG
        stream is never rewound, so relaunching from the same champion is a genuinely new draw --
        which is exactly the experiment diag_stuck.py ran across four seeds."""
        self.ep_i += 1
        self.ep_n += 1
        self.ep_left = self.ep_unit * _luby(self.ep_i)
        self.barren = 0
        e = self._pop_pick() if self.rng.rand() < EP_ARCH_P else None
        if e is not None:
            self.cur = [e[0], e[1], e[2]]
        elif self.best[1] is not None:
            self.cur = list(self.best)

    def run(self, quota, prev):
        n = self.n
        t0 = time.process_time()
        deadline = t0 + quota
        if self.seeds is None:
            self._open(prev)
        d0 = self.digits()           # the opening incumbent is the BASELINE, not a gain

        while self.seeds and time.process_time() < deadline and self.sink.meter.left() > 0:
            c_s, r_s, mi, dd, al = self.seeds.pop(0)
            if al:
                c_s, r_s = _al_local(c_s, _greedy_radii(c_s), deadline,
                                     outer=7, maxiter=90, rng=self.rng)
            s, cb, rb = _ccp(c_s, deadline, self.sink, n, max_iter=mi, delta0=dd, r0=r_s)
            self._pop_add(s, cb, rb)
            if s > self.best[0]:
                self.best = [s, cb, rb]
        if not self.seeds and self.cur is None and self.best[1] is not None:
            self.cur = list(self.best)

        # the annealed chain of iterations 2/4, PERSISTENT across quanta (iteration 7) and, from
        # iteration 9, no longer stepping with a hard-coded move mixture: every step asks the
        # SHARED bandit which operator has been buying the most digits per CPU-second lately.
        TOL_REL, RESET_AFTER = 6e-4, 8
        while (not self.seeds and self.cur is not None and self.cur[1] is not None
               and time.process_time() < deadline and self.sink.meter.left() > 0):
            if self.ep_left <= 0.0:
                self._restart()
            t_op = time.process_time()
            allowed = [k for k in self.bandit.names if k != "cross" or len(self.pop) >= 2]
            op = self.bandit.pick(allowed)
            sc = self.sbandit.pick(self.sbandit.names)
            # ITERATION 13 -- the arm must SHOW it left the basin.  A shot that settles back onto
            # its own starting point is re-fired at the next rung up (ESC_MAX times at most); the
            # loop stops at the FIRST rung that escapes, and the whole bundle is charged to the
            # (op, sc) pair that opened it, so an arm that only ever returns is priced for it.
            rung = self.sbandit.names.index(sc) if sc in self.sbandit.names else 0
            s, cb, rb, fresh, gain, null = -np.inf, None, None, 0.0, 0.0, False
            fired, improved = 0, False
            for shot in range(ESC_MAX + 1):
                sck = self.sbandit.names[min(rung + shot, len(self.sbandit.names) - 1)]
                c_try, r_try, use_al = self._apply(op, deadline, sck)
                if c_try is None or c_try.shape[0] != n:
                    break
                fired += 1
                if use_al:
                    ca, ra = _al_local(c_try, _greedy_radii(c_try), deadline,
                                       outer=7, maxiter=90, rng=self.rng)
                    s, cb, rb = _ccp(ca, deadline, self.sink, n, max_iter=140,
                                     delta0=0.03 / np.sqrt(n), r0=ra)
                else:
                    s, cb, rb = _ccp(c_try, deadline, self.sink, n, max_iter=160,
                                     delta0=0.14 / np.sqrt(n), r0=r_try)
                npop = len(self.pop)
                self._pop_add(s, cb, rb)
                if len(self.pop) > npop:
                    fresh = 1.0
                # reward in the SCORER's currency: digits this step added to the running best.
                # Read off self.best, not the sink, because the sink batches and would lag it.
                gain += max(0.0, _digits(s, self.ref) - _digits(self.best[0], self.ref))
                if s > self.best[0]:
                    self.best = [s, cb, rb]
                    improved = True
                null = _is_null(s, rb, self.cur[0], self.cur[2])
                if (not null or shot == ESC_MAX or rung + shot >= len(self.sbandit.names) - 1
                        or time.process_time() >= deadline or self.sink.meter.left() <= 0):
                    break
            if fired == 0:
                # a refused move still cost time -- charge it, or a broken arm looks free
                dt = time.process_time() - t_op
                self.ep_left -= dt
                self.bandit.credit(op, 0.0, dt)
                self.sbandit.credit(sc, 0.0, dt)
                continue
            if improved:
                self.barren = 0
            else:
                self.barren += 1
            acc = 0.0
            if cb is not None and s > self.cur[0] - TOL_REL * abs(self.cur[0]):
                self.cur = [s, cb, rb]
                # ...but the bonus is only paid to a step that actually LEFT its basin.  Making it
                # proportional to digits-gained-over-cur instead was measured and is WORSE (A/B
                # total 0.38 digits vs 4.85): `cur` drifts downhill under the TOL_REL window, so
                # any arm that rebuilds from the archive scores against a sinking reference and
                # farms the bonus.  A reward reference has to be stationary; only `best` is.
                acc = 0.0 if null else 1.0
            rew, dt = gain + OP_NEW_BASIN * fresh + OP_ACCEPT * acc, time.process_time() - t_op
            self.ep_left -= dt
            self.bandit.credit(op, rew, dt)
            self.sbandit.credit(sc, rew, dt)
            if self.barren >= RESET_AFTER:
                # restart from a DRAWN archive member, not always the incumbent: an annealed walk
                # that always re-launches from the same point cannot leave that point's basin.
                e = self._pop_pick()
                self.cur = [e[0], e[1], e[2]] if e is not None else list(self.best)
                self.barren = 0

        self.sink.flush()
        if self.sink.pack is not None:
            prev[n] = (self.sink.pack[:, :2].copy(), self.sink.pack[:, 2].copy())
        elif self.best[1] is not None:
            prev[n] = (self.best[1], self.best[2])

        spent = max(1e-6, time.process_time() - t0)
        g = max(0.0, self.digits() - d0)
        self.cpu += spent
        self.gain += g
        return g, spent


# ---------------------------------------------------------------------------------------- solve
POP_MAX = 6              # archived structurally-distinct basins per n
POP_TAU = 2e-4           # sorted-radius-spectrum distance below which two optima are ONE basin
# (iteration 8's fixed CROSS_P is gone: recombination is now the bandit's "cross" arm)
PROBE_FRAC = 0.45        # share of the CPU that goes to the breadth round
AUCT_BETA = 0.6          # discount applied to an n's own (G, T) at each of its updates
AUCT_PRIOR_G = 0.03      # optimism: digits an untested/failing n is still credited with...
AUCT_PRIOR_T = 1.5       # ...over this many seconds, so nothing is starved to zero
QUANTUM_BASE, QUANTUM_REF_N = 2.0, 63.0
QUANTUM_LO, QUANTUM_HI = 0.8, 3.5


def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    t_entry = time.process_time()                 # deadline derived AT ENTRY, never a constant
    tgts = sorted(set(int(t) for t in targets))
    avail = max(MIN_SLICE * len(tgts), CPU_CAP - CPU_SAFETY - t_entry)
    end = t_entry + avail
    bandit = _Bandit()                            # SHARED move-selector: 37 n, one set of stats
    sbandit = _Bandit(SCALES, label="scale")      # SHARED move-SIZE selector (iteration 12)
    workers = {n: _Worker(n, evaluate, meter, rng, bandit, sbandit) for n in tgts}
    prev = {}                                     # n -> (centres, radii) for cross-n transfer

    # --- ROUND 1: PROBE.  Iteration 6's rule (slice proportional to n, which equalises basin
    #     count) applied to PROBE_FRAC of the budget.  It buys breadth -- every n submits its
    #     incumbent and gets real search time -- and it MEASURES what the auction needs to bid on.
    #     Iteration 12: the probe skips any n whose COMMITTED pack is already at the 7-digit cap.
    #     Round 2 has excluded those n since iteration 9 (their index is forced to -1, and every
    #     uncapped n scores > 0, so a capped n can never win a quantum) -- but round 1 was still
    #     handing them a slice proportional to n, and the capped n are the large ones: 947 of the
    #     2331 total weight, i.e. 41 % of the probe and ~18 % of the whole CPU budget spent where
    #     the scorer cannot pay.  Skipping is SAFE, not lossy: the driver is append-or-improve, so
    #     an n the solver never touches keeps its committed pack verbatim.
    recs_probe = _records()
    probe_n = [n for n in tgts
               if not (recs_probe.get(n) and _pack_digits(n, recs_probe[n]) >= DIGITS_CAP)]
    if not probe_n:                                # every target already capped -- probe them all
        probe_n = list(tgts)
    # ITERATION 15: price a restart ticket off the budget.  An n that is still payable will own
    # roughly avail/len(probe_n) CPU-s over the whole call, and one full Luby cycle is LUBY_CYCLE
    # units -- so this is the unit at which the 1,1,2,1,1,2,4 schedule actually completes instead
    # of being truncated to its short prefix.  Derived once, from quantities the call already
    # knows, and shared by every worker so the operator/scale bandits stay comparable across n.
    ep_unit = _ep_unit(avail, len(probe_n))
    for w in workers.values():
        w.ep_unit = ep_unit
    wgt = np.asarray(probe_n, dtype=float)
    share = wgt / wgt.sum()
    probe_end = t_entry + avail * PROBE_FRAC
    for idx, n in enumerate(probe_n):
        now = time.process_time()
        if meter.left() <= 0 or now >= end:
            break
        # re-derive the slice from what is LEFT of the probe budget, so an n that overruns eats
        # its own successors' time only in proportion, never wholesale.
        tail = float(share[idx:].sum())
        rem = probe_end - now
        q = (rem * float(share[idx]) / tail) if (rem > 0.0 and tail > 0.0) else 0.0
        q = min(max(q, MIN_SLICE), end - now)
        if q <= 0.0:
            break
        g, t = workers[n].run(q, prev)
        workers[n].G, workers[n].T = g, t

    # --- ROUND 2: AUCTION.  Hand the next quantum to the n with the best OPTIMISTIC recent rate
    #     (G + PRIOR_G) / (T + PRIOR_T).  Both terms are discounted at each of that n's own
    #     updates, so T saturates near QUANTUM/(1-BETA): a run of empty quanta pushes an n down to
    #     a BOUNDED floor (it gets revisited, never starved), while an n that just paid off is bid
    #     straight back up.  With every gain zero the index degenerates to round-robin by
    #     least-time-spent, which is the old behaviour -- so the auction can only add information.
    while meter.left() > 0:
        now = time.process_time()
        if now >= end:
            break
        best_n, best_ix = None, -np.inf
        for n in tgts:
            wk = workers[n]
            ix = (wk.G + AUCT_PRIOR_G) / (wk.T + AUCT_PRIOR_T)
            # iteration 9: digits are CAPPED at 7.  An n already at the cap (its pack is at or
            # past the record) cannot pay for another second no matter what the optimistic prior
            # says, so it bids last -- the CPU it was still being handed goes to n that can move.
            if wk.digits() >= DIGITS_CAP:
                ix = -1.0
            if ix > best_ix:
                best_ix, best_n = ix, n
        if best_n is None:
            break
        wk = workers[best_n]
        q = min(max(QUANTUM_LO, min(QUANTUM_HI, QUANTUM_BASE * best_n / QUANTUM_REF_N)), end - now)
        if q < 0.1:
            break
        g, t = wk.run(q, prev)
        wk.G = AUCT_BETA * wk.G + g
        wk.T = AUCT_BETA * wk.T + t

# -------------------------------------------------------------------------------------self-test
def _self_test():
    import math

    # iteration 5: the filtered-row / direct-HiGHS LP must return the SAME vertex as the
    # unfiltered scipy path (exact reduction, not a relaxation) -- checked on a real packing.
    _c5 = np.array([[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25], [0.0, 0.0]])
    _r5 = _greedy_radii(_c5)
    for _d in (0.05, 0.005):
        _o = _ccp_lp(_c5, _r5, _d, _d)
        assert _o is not None
        _A, _b, _ub, _nr, _kk = _lp_rows(_c5, _r5, _d, _d)
        _nv = 15
        _obj = np.zeros(_nv); _obj[10:] = -1.0
        _cl = np.concatenate([np.full(10, -_d), np.zeros(5)])
        _cu = np.concatenate([np.full(10, _d), _ub])
        _ref = linprog(_obj, A_ub=_A, b_ub=_b, bounds=list(zip(_cl, _cu)), method="highs")
        assert _ref.success and abs(_o[1].sum() - _ref.x[10:].sum()) < 1e-9, "LP paths disagree"
        assert np.all(_repair(_o[0], _o[1]) >= -1e-12)

    # iteration 6: a BASIS-WARM-STARTED chain must trace the same vertices as a cold chain, and
    # its point must sit INSIDE the polytope.  A basis-mapping bug shows up as a genuinely
    # suboptimal vertex (orders of magnitude past 1e-6 here, where a step moves sum r by ~1e-2).
    # The 1e-6 band is HiGHS-with-presolve-off numerics, and it is the COLD path that spends it:
    # measured on the configuration below, the cold solve returns a point 7.8e-8 OUTSIDE the rows
    # (and so a spuriously better objective) while the warm solve is exact to 4e-16.
    _c6 = np.array([[-0.30, -0.30], [0.02, -0.31], [0.31, -0.28], [-0.28, 0.03], [0.01, 0.02],
                    [0.30, 0.01], [-0.31, 0.30], [0.00, 0.29], [0.29, 0.31]])
    for _tag, _w6 in (("plain", None), ("weighted", np.linspace(0.7, 1.3, 9))):
        _c, _r = _c6.copy(), _greedy_radii(_c6)
        _st, _dd = {}, 0.06
        _obj6 = np.zeros(27)
        _obj6[18:] = -1.0 if _w6 is None else -_w6
        for _k in range(12):
            _ow = _ccp_lp(_c, _r, _dd, _dd, state=_st, w=_w6)
            _oc = _ccp_lp(_c, _r, _dd, _dd, state=None, w=_w6)
            assert _ow is not None and _oc is not None
            _vw = float(_ow[1].sum() if _w6 is None else (_w6 * _ow[1]).sum())
            _vc = float(_oc[1].sum() if _w6 is None else (_w6 * _oc[1]).sum())
            assert _vw > _vc - 1e-6, (
                "warm-basis LP is suboptimal vs cold (%s, step %d, %.3e)" % (_tag, _k, _vc - _vw))
            _A6, _b6, _u6, _n6, _k6 = _lp_rows(_c, _r, _dd, _dd)
            _z6 = np.concatenate([(_ow[0] - _c).T.ravel(), _ow[1]])
            assert float((_A6 @ _z6 - _b6).max()) <= 1e-9, "warm LP point is outside the polytope"
            _c, _r = _oc[0], _repair(_oc[0], _oc[1])
            _dd *= 0.85
        assert _st.get("keys_sorted") is not None, "basis state was never populated"
    # a stale/garbage basis must be survived, not trusted
    _bad = {"keys_sorted": np.array([0, 1, 2], dtype=np.int64), "perm": np.array([0, 1, 2]),
            "rows": np.asarray([_HS_BASIC] * 3, dtype=object), "cols": [_HS_BASIC] * 27}
    _ob = _ccp_lp(_c6, _greedy_radii(_c6), 0.06, 0.06, state=_bad)
    _og = _ccp_lp(_c6, _greedy_radii(_c6), 0.06, 0.06, state=None)
    assert _ob is not None and abs(_ob[1].sum() - _og[1].sum()) < 1e-6, "stale basis broke the LP"

    class _M:
        def __init__(self, b):
            self.budget, self.used = b, 0

        def left(self):
            return max(0, self.budget - self.used)

        def tick(self, k=1):
            g = max(0, min(int(k), self.budget - self.used))
            self.used += g
            return g

    def make(meter, store):
        def evaluate(n, packing):
            a = np.asarray(packing, dtype=float)
            single = a.ndim == 2
            if single:
                a = a[None]
            B = a.shape[0]
            grant = meter.tick(B)
            feas = np.zeros(B, dtype=bool)
            sr = np.full(B, -np.inf)
            for i in range(min(grant, B)):
                cs = [tuple(map(float, row)) for row in a[i]]
                ok = len(cs) == n and all(math.isfinite(v) for c in cs for v in c) \
                    and min(r for *_, r in cs) > 0.0
                if ok:
                    w = min(min(x - r - LO, HI - x - r, y - r - LO, HI - y - r) for x, y, r in cs)
                    pmin = math.inf
                    for p in range(n):
                        for q in range(p + 1, n):
                            pmin = min(pmin, math.hypot(cs[p][0] - cs[q][0], cs[p][1] - cs[q][1])
                                       - cs[p][2] - cs[q][2])
                    ok = w >= -1e-9 and (n < 2 or pmin >= -1e-9)
                if ok:
                    feas[i] = True
                    sr[i] = math.fsum(r for *_, r in cs)
                    if sr[i] > store.get(n, (-math.inf,))[0]:
                        store[n] = (float(sr[i]), a[i].copy())
            if single:
                return bool(feas[0]), float(sr[0])
            return feas, sr
        return evaluate

    # iteration 7: the SCHEDULER.  Three properties the auction depends on.
    #  (a) _digits reproduces the scorer's rule exactly (clamped, one-sided, 7 at/past the record).
    assert abs(_digits(0.999, 1.0) - 3.0) < 1e-12
    assert _digits(1.0, 1.0) == 7.0 and _digits(2.0, 1.0) == 7.0
    assert _digits(0.0, 1.0) == 0.0 and _digits(-1.0, 1.0) == 0.0
    #  (b) a _Worker RESUMED across many short quanta must never lose its incumbent and must
    #      report only non-negative gains -- the bid is a difference of digits, and a negative one
    #      would let a size bid its way out of the auction after a bad quantum.
    _rngw = np.random.RandomState(3)
    _mw = _M(500_000)
    _stw = {}
    _wk = _Worker(9, make(_mw, _stw), _mw, _rngw)
    _prevw, _seen = {}, -np.inf
    for _ in range(5):
        _g, _t = _wk.run(0.25, _prevw)
        assert _g >= 0.0 and _t > 0.0
        assert _wk.sink.best >= _seen, "resumed worker lost its incumbent"
        _seen = _wk.sink.best
    assert _wk.sink.pack is not None and 9 in _prevw
    assert _wk.ref is not None and _wk.ref > 0.0, "unrecorded n got no pricing reference"
    #  (c) the auction index is bounded below by the optimism prior, so an n whose gains are all
    #      zero is pushed down but NEVER to zero -- it keeps being revisited.
    _G = _T = 0.0
    for _ in range(40):
        _G, _T = AUCT_BETA * _G, AUCT_BETA * _T + QUANTUM_HI
    _floor = AUCT_PRIOR_G / (QUANTUM_HI / (1.0 - AUCT_BETA) + AUCT_PRIOR_T)
    _fresh = AUCT_PRIOR_G / AUCT_PRIOR_T
    assert abs((_G + AUCT_PRIOR_G) / (_T + AUCT_PRIOR_T) - _floor) < 1e-9, "index floor moved"
    assert _floor > 0.1 * _fresh, "a starved n decays more than 10x below a fresh one"
    print("scheduler ok: worker resumed 5x -> %.6f, floor index %.4f"
          % (_wk.sink.best, (_G + AUCT_PRIOR_G) / (_T + AUCT_PRIOR_T)))

    # ---- iteration 9: the adaptive operator selector ---------------------------------------
    #  (a) with every reward zero the index degenerates to round-robin by least-time-spent, so
    #      NO arm is ever closed off -- the property the whole hedge rests on.
    _bd = _Bandit()
    _seen = {}
    for _ in range(4 * len(OPS)):
        _k = _bd.pick(_bd.names)
        _seen[_k] = _seen.get(_k, 0) + 1
        _bd.credit(_k, 0.0, 0.05)
    assert len(_seen) == len(OPS), "a zero-reward arm was starved: %r" % (_seen,)
    assert max(_seen.values()) - min(_seen.values()) <= 1, "zero-reward rotation is not fair"
    #  (b) an arm that pays is preferred -- and still decays back toward the others when it
    #      stops paying, so a lucky early arm cannot lock the selector.
    _bd2 = _Bandit()
    for _ in range(6):
        _bd2.credit("cross", 0.30, 0.05)
        _bd2.credit("jitter", 0.0, 0.05)
    assert _bd2.pick(_bd2.names) == "cross", "a paying arm was not preferred"
    for _ in range(40):
        _bd2.credit("cross", 0.0, 0.05)
    assert _bd2.pick(_bd2.names) != "cross", "an arm that stopped paying stayed locked in"
    #  (c) every named arm is TOTAL: _apply returns either None or exactly n valid centres, and
    #      the six geometric arms must all fire from a real committed pack.
    _n = 27
    _pk = _read_pack(_n)
    if _pk is not None:
        _w2 = _Worker(_n, lambda a, b: (True, 0.0), _M(10), np.random.RandomState(3), _bd)
        _w2.cur = [float(_pk[1].sum()), _pk[0], _pk[1]]
        _w2.ref = 1.0
        _fired = 0
        # iteration 12: totality must hold at EVERY rung of the scale ladder, not just sc = 1.0 --
        # a rung that drives some arm's move count to 0 or its jitter to 0 would be a silent no-op.
        for _k in OPS:
            for _sc in SCALES:
                _c, _r, _al = _w2._apply(_k, time.process_time() + 5.0, _sc)
                if _c is None:
                    assert _k == "cross", "arm %s refused with a populated chain" % _k
                    continue
                assert _c.shape == (_n, 2), "arm %s@%.2f returned %r centres" % (_k, _sc, _c.shape)
                assert np.isfinite(_c).all() and _c.min() >= LO - 1e-12 and _c.max() <= HI + 1e-12, _k
                assert _greedy_radii(_c).min() > 0.0, "arm %s@%.2f left no room" % (_k, _sc)
                if _sc == 1.0:
                    _fired += 1
        assert _fired == len(OPS) - 1, "only %d arms fired" % _fired
        assert _w2._apply("al", time.process_time() + 5.0)[2] is True, "al arm lost its presolve"
    #  (d) an n already at the digit cap must bid LAST -- it cannot buy another digit.
    assert _digits(1.0, 1.0) == DIGITS_CAP and _digits(0.9, 1.0) < DIGITS_CAP

    # ---- iteration 12: the SCALE dimension ----------------------------------------------------
    #  (e) the ladder is a real ladder and CONTAINS the pre-iteration-12 constant, so nothing the
    #      selector learned before can become unreachable.
    assert 1.0 in SCALES and len(set(SCALES)) == len(SCALES) >= 2
    assert list(SCALES) == sorted(SCALES) and min(SCALES) > 0.0
    #  (f) sc = 1.0 must reproduce the OLD hard-coded move size EXACTLY -- same rng, same output.
    _pk1 = _read_pack(31)
    if _pk1 is not None:
        _c1, _r1 = _pk1
        for _fn in (_op_jitter, _op_relocate, _op_swap, _op_reinsert, _squeeze_release):
            _a = _fn(_c1.copy(), _r1.copy(), np.random.RandomState(5), 31)
            _b = _fn(_c1.copy(), _r1.copy(), np.random.RandomState(5), 31, 1.0)
            assert np.allclose(_a, _b), "%s: sc=1.0 is not the shipped default" % _fn.__name__
    #  (g) the scale ladder must actually CHANGE the kick size, monotonically -- the whole premise
    #      of arming it.  Measured on jitter, whose size is exactly its sigma.
    _disp = []
    for _sc in SCALES:
        _rs2 = np.random.RandomState(9)
        _c2 = np.zeros((400, 2))
        _disp.append(float(np.abs(_op_jitter(_c2, np.full(400, 0.01), _rs2, 400, _sc) - _c2).mean()))
    assert all(_disp[_i] < _disp[_i + 1] for _i in range(len(_disp) - 1)), \
        "the ladder does not order the kick size: %r" % (_disp,)
    #  (h) the two bandits are INDEPENDENT: crediting a scale must not move an operator's index.
    _sb = _Bandit(SCALES, label="scale")
    _ob = _Bandit()
    _ix_before = _ob.index("jitter")
    for _ in range(6):
        _sb.credit(SCALES[0], 0.4, 0.05)
        _sb.credit(SCALES[-1], 0.0, 0.05)
    assert _sb.pick(_sb.names) == SCALES[0], "the scale bandit does not prefer a paying rung"
    assert _ob.index("jitter") == _ix_before, "the two bandit dimensions are coupled"
    #  (i) the probe's capped-n filter: a capped pack is skipped, an uncapped and an ABSENT one
    #      are not (a fresh n must never be mistaken for a finished one).
    _rc = _records()
    if _rc:
        _capd = [n for n in _rc if _pack_digits(n, _rc[n]) >= DIGITS_CAP]
        _wk = [n for n in _rc if _pack_digits(n, _rc[n]) < DIGITS_CAP]
        assert _wk, "the probe filter would skip every target"
        assert all(_pack_digits(n, _rc[n]) < DIGITS_CAP for n in _wk)
        print("probe filter ok: %d capped n skipped, %d payable n keep the probe"
              % (len(_capd), len(_wk)))
    assert _pack_digits(12, 3.0) == 0.0, "an n with no committed pack must not read as capped"
    print("bandit ok: %d arms rotate when flat, %d fired from a real pack, "
          "%d scale rungs x %d ops all total" % (len(_seen), _fired, len(SCALES), len(OPS)))

    def _feasible(pk):
        cx, rr = pk[:, :2], pk[:, 2]
        if rr.min() <= 0.0:
            return False
        if float(np.min(_walls(cx) - rr)) < -1e-9:
            return False
        m = cx.shape[0]
        if m < 2:
            return True
        iu = np.triu_indices(m, 1)
        return float(np.min(_dists(cx)[iu] - rr[iu[0]] - rr[iu[1]])) >= -1e-9

    # iteration 10: the INFLATION arm.
    #  (a) the analytic gradient of the overlap objective matches central differences,
    #  (b) a profile with room resolves to residual 0 and the resulting packing is FEASIBLE at
    #      those radii (so the arm's premise -- overlap 0 => a real packing -- actually holds),
    #  (c) the deflate/re-inflate ramp leaves the basin it started in, which is its whole job.
    _rs = np.random.RandomState(4)
    _nn = 11
    _cg = _rs.uniform(-0.4, 0.4, size=(_nn, 2))
    _tg = np.full(_nn, 0.09)
    _z = np.concatenate([_cg[:, 0], _cg[:, 1]])
    _f0, _g = _inflate_fg(_z, _nn, _tg)
    assert _f0 > 0.0, "the gradient check needs a config that actually overlaps"
    for _i in range(2 * _nn):
        _zp, _zm = _z.copy(), _z.copy()
        _zp[_i] += 1e-7
        _zm[_i] -= 1e-7
        _fd = (_inflate_fg(_zp, _nn, _tg)[0] - _inflate_fg(_zm, _nn, _tg)[0]) / 2e-7
        assert abs(_fd - _g[_i]) <= 1e-6 * max(1.0, abs(_fd)), \
            "inflate gradient wrong at %d: %.3e vs %.3e" % (_i, _g[_i], _fd)
    _cs, _rsd = _inflate(_cg, np.full(_nn, 0.06), maxiter=600)
    assert _rsd <= 1e-18, "a profile with room did not resolve: residual %.3e" % _rsd
    _pk = np.concatenate([_cs, np.full((_nn, 1), 0.06)], axis=1)
    assert _feasible(_pk), "residual 0 did not mean a feasible packing"
    if os.path.exists("bench/packs/csqv49.pck"):
        _c9, _r9 = _read_pack(49)
        _s9 = float(_r9.sum())
        _moved = 0
        for _i in range(6):
            _ci = _op_inflate(_c9, _r9, _rs, 49)
            assert _ci.shape == (49, 2) and np.all(np.abs(_ci) <= 0.5 + 1e-12)
            _ri = _repair(_ci, _rlp(_ci))
            assert _ri.min() > 0.0 and _feasible(np.concatenate([_ci, _ri[:, None]], axis=1))
            # a DIFFERENT basin: same-basin returns reproduce sum_r to ~1e-9
            if abs(float(_ri.sum()) - _s9) > 1e-6:
                _moved += 1
        assert _moved >= 4, "the ramp stayed in its own basin %d/6 times" % (6 - _moved)
        print("inflate ok: gradient exact, residual 0 => feasible, ramp left the basin %d/6" % _moved)

    # --- iteration 11: the T1 / Delaunay-flip arm ------------------------------------------
    #  (a) _flip_edges returns well-formed interior quadrilaterals: four DISTINCT vertices, and
    #      the edge it names really is a Delaunay edge shared by two triangles.
    #  (b) the arm actually REWIRES: the Delaunay edge set of the output differs from the input's
    #      (a coordinate perturbation that leaves the graph alone is not a T1 transition).
    #  (c) from a real committed pack it returns exactly n in-square centres whose exact LP radii
    #      are strictly positive and feasible, and it leaves the basin.
    def _edgeset(cx):
        _e = _flip_edges(cx)
        if _e is None:
            return set()
        _t = np.asarray(_Delaunay(cx).simplices)
        _a = _t[:, [0, 1, 2]].ravel()
        _b = _t[:, [1, 2, 0]].ravel()
        return set(zip(np.minimum(_a, _b).tolist(), np.maximum(_a, _b).tolist()))

    _rs = np.random.RandomState(11)
    _cq = _rs.uniform(-0.45, 0.45, size=(40, 2))
    _e = _flip_edges(_cq)
    assert _e is not None, "_flip_edges failed on a generic point set"
    _ei, _ej, _ok, _ol = _e
    assert _ei.size >= 3 * 40 // 2 - 40, "suspiciously few interior edges: %d" % _ei.size
    assert (_ei < _ej).all(), "edges are not canonically ordered"
    assert (_ok != _ol).all() and (_ok != _ei).all() and (_ok != _ej).all() \
        and (_ol != _ei).all() and (_ol != _ej).all(), "a quadrilateral has a repeated vertex"
    _es = _edgeset(_cq)
    assert all(((int(_i), int(_j)) in _es) for _i, _j in zip(_ei[:50], _ej[:50])), \
        "_flip_edges named an edge the triangulation does not have"
    assert _flip_edges(np.zeros((3, 2))) is None, "_flip_edges accepted a degenerate input"

    if os.path.exists("bench/packs/csqv49.pck"):
        _c9, _r9 = _read_pack(49)
        _s9 = float(_r9.sum())
        _e0 = _edgeset(_c9)
        _rew = _mv = 0
        for _i in range(8):
            _cf = _op_flip(_c9, _r9, _rs, 49)
            assert _cf is not None, "the flip arm refused on a real pack"
            assert _cf.shape == (49, 2) and np.all(np.abs(_cf) <= 0.5 + 1e-12)
            _rf = _repair(_cf, _rlp(_cf))
            assert _rf.min() > 0.0 and _feasible(np.concatenate([_cf, _rf[:, None]], axis=1)), \
                "the flip arm returned an infeasible packing"
            if _edgeset(_cf) != _e0:
                _rew += 1
            if abs(float(_rf.sum()) - _s9) > 1e-6:
                _mv += 1
        assert _rew >= 7, "the flip arm rewired only %d/8 times" % _rew
        assert _mv >= 6, "the flip arm stayed in its own basin %d/8 times" % (8 - _mv)
        print("flip ok: %d interior edges, rewired %d/8, left the basin %d/8"
              % (_ei.size, _rew, _mv))

        # ---- iteration 13: NULL steps are recognised, unpaid, and escalated ------------------
        assert ESC_MAX >= 1 and 0.0 < NULL_TAU < 1e-6
        _c13, _r13 = _read_pack(49)
        _s13 = float(_r13.sum())
        assert _is_null(_s13, _r13, _s13, _r13), "an identical settled point read as a real move"
        # a relabelling / reflection is the SAME basin: the signature must be invariant to it
        _perm = np.random.RandomState(3).permutation(49)
        assert _is_null(_s13, _r13[_perm], _s13, _r13), "_is_null is not label-invariant"
        # a genuinely displaced point is NOT null, on either test alone
        _far = _r13.copy()
        _far[0] += 1e-3
        _far[1] -= 1e-3                     # same Sum-r, different spectrum
        assert not _is_null(float(_far.sum()), _far, _s13, _r13), "a respread spectrum read null"
        assert not _is_null(_s13 * (1 + 1e-6), _r13, _s13, _r13), "a Sum-r change read null"
        assert not _is_null(_s13, None, _s13, _r13), "_is_null accepted a missing spectrum"
        assert not _is_null(_s13, _r13[:-1], _s13, _r13), "_is_null accepted a size mismatch"
        # and the lock-in the gate exists to break: with a real pack and a real budget the chain
        # must sample MORE THAN ONE operator and MORE THAN ONE rung (pre-13 it spent >95 % of
        # both budgets on a single arm and left 8 of 10 ops and 2 of 4 rungs at zero pulls).
        class _NS13:
            class M:
                def left(self): return 10 ** 9
            meter = M()

            def __init__(self):
                self.best, self.pack = -np.inf, None

            def add(self, p):
                p = np.asarray(p, float)
                if float(p[:, 2].sum()) > self.best:
                    self.best, self.pack = float(p[:, 2].sum()), p.copy()

            def flush(self):
                pass
        _w13 = _Worker(49, None, None, np.random.RandomState(5))
        _w13.sink = _NS13()
        _w13.run(4.0, {})
        _nop = sum(1 for v in _w13.bandit.pulls.values() if v)
        _nsc = sum(1 for v in _w13.sbandit.pulls.values() if v)
        assert _nop >= 4 and _nsc >= 2, ("the chain locked in again: %d ops, %d rungs"
                                         % (_nop, _nsc))
        print("null-gate ok: %d ops and %d rungs sampled in one 4 s chain (was 2 and 2)"
              % (_nop, _nsc))


    rng = np.random.RandomState(0)
    meter = _M(500_000)
    store = {}
    solve(make(meter, store), meter, rng, [11, 13])
    assert 11 in store and 13 in store, "first call produced no feasible packing"
    print("call 1:", {k: round(v[0], 6) for k, v in store.items()},
          "cpu=%.1fs" % time.process_time())

    # THE re-entrancy check: a second solve() in the SAME process must still work (no module-level
    # absolute CPU_STOP), and must survive len(targets) == 1.
    store2 = {}
    solve(make(meter, store2), meter, rng, [12])
    assert 12 in store2 and store2[12][0] > 0, "second solve() call returned no feasible packing"
    print("call 2:", {k: round(v[0], 6) for k, v in store2.items()},
          "cpu=%.1fs" % time.process_time())

    # guarded warm start: an n with no committed pack must still cold-start
    store3 = {}
    solve(make(meter, store3), meter, rng, [7])
    assert 7 in store3, "cold start for an un-warmable n failed"

    # a third call with an n that DOES have a committed pack must not regress below it
    store4 = {}
    solve(make(meter, store4), meter, rng, [2])
    got = store4[2][0]
    print("n=2 sum_r = %.9f (opt %.9f)" % (got, 2 - math.sqrt(2)))
    assert got > (2 - math.sqrt(2)) - 1e-6, "n=2 far from the known optimum"

    # the sparse pair filter must never produce an infeasible LP step
    c = rng.uniform(LO + .05, HI - .05, size=(20, 2))
    r = _greedy_radii(c)
    for kw in ({}, {"rcap": np.where(np.arange(20) < 3, 0.0, 0.5)},
               {"w": np.linspace(0.5, 1.5, 20)}):
        out = _ccp_lp(c, r, 0.05, 0.05, **kw)
        assert out is not None, kw
        cn, rn = out
        if "rcap" in kw:
            assert rn[:3].max() <= 1e-9, "rcap did not bind"
        rn = _repair(cn, rn)
        d = _dists(cn)
        assert (d - (rn[:, None] + rn[None, :])).min() >= -1e-12, kw
        assert (_walls(cn) - rn).min() >= -1e-12, kw

    # the deformation phase must leave an exactly feasible packing and must NOT be a no-op
    c2, r2 = _lp_phase(c, r, 8, 0.05, rcap=np.where(np.arange(20) < 2, 0.0, 0.5))
    d = _dists(c2)
    assert (d - (r2[:, None] + r2[None, :])).min() >= -1e-12
    assert (_walls(c2) - r2).min() >= -1e-12
    assert np.abs(c2 - c).max() > 1e-6, "squeeze phase did not move the configuration"
    # the smooth engine must return an EXACTLY feasible packing, honour rcap, and _rlp must be
    # exact on centres whose optimal radii are known (a regular row grid)
    c3 = rng.uniform(LO + .05, HI - .05, size=(24, 2))
    ca, ra = _al_local(c3, _greedy_radii(c3), time.process_time() + 20, outer=4, maxiter=40,
                       rng=rng)
    d = _dists(ca)
    assert (d - (ra[:, None] + ra[None, :])).min() >= -1e-12, "AL pair infeasible"
    assert (_walls(ca) - ra).min() >= -1e-12, "AL wall infeasible"
    assert ra.min() > 0, "AL produced a non-positive radius"
    cap = np.where(np.arange(24) < 4, 0.0, 0.5)
    _, rc = _al_local(c3, _greedy_radii(c3), time.process_time() + 20, outer=3, maxiter=30,
                      rcap=cap, rng=rng)
    assert rc[:4].max() <= 1e-8, "AL rcap did not bind"
    # iteration 8: the population operators.  _drop_to must return EXACTLY n circles and must
    # prefer deleting a collision over deleting a well-placed circle; _crossover must return
    # exactly n usable centres from two structurally different parents, take material from BOTH,
    # and survive parents whose seam overlaps heavily.
    cA = _init_rows(16, 4, False)
    cB = _init_rows(16, 4, True)
    rA, rB = _greedy_radii(cA), _greedy_radii(cB)
    pa, pb = [float(rA.sum()), cA, rA, np.sort(rA)], [float(rB.sum()), cB, rB, np.sort(rB)]
    took_a = took_b = 0
    for _t in range(40):
        ch = _crossover(pa, pb, rng, 16)
        assert ch is not None and ch.shape == (16, 2), "crossover shape"
        assert np.isfinite(ch).all() and ch.min() >= LO - 1e-12 and ch.max() <= HI + 1e-12
        rc = _greedy_radii(ch)
        assert rc.min() > 0 and (_dists(ch) - (rc[:, None] + rc[None, :])).min() >= -1e-12
        took_a += int(np.min(np.abs(ch[:, None, :] - cA[None, :, :]).max(-1), axis=1).min() < 1e-9)
        took_b += int(np.min(np.abs(ch[:, None, :] - cB[None, :, :]).max(-1), axis=1).min() < 1e-9)
    assert took_a > 0 and took_b > 0, "crossover never grafted from one of the parents"
    # a deliberate collision: two circles stacked on one spot must be what _drop_to removes
    cD = np.concatenate([cA, np.array([[0.0, 0.0], [0.002, 0.001]])], axis=0)
    rD = np.concatenate([rA, np.array([0.12, 0.12])])
    cK, rK = _drop_to(cD, rD, 16)
    assert cK.shape[0] == 16 and rK.shape[0] == 16, "_drop_to count"
    assert np.abs(cK - np.array([0.002, 0.001])).max(axis=1).min() > 1e-9, \
        "_drop_to kept the overlapping circle"
    # deleting from a FEASIBLE pack must leave it feasible and drop the least valuable circles
    cS, rS = _drop_to(cA, rA, 13)
    assert cS.shape[0] == 13 and (_dists(cS) - (rS[:, None] + rS[None, :])).min() >= -1e-12
    assert rS.sum() >= np.sort(rA)[3:].sum() - 1e-12, "_drop_to discarded more than the worst 3"

    # the archive must DEDUPE one basin and KEEP two different ones
    class _WSt(object):
        pass
    _w = _WSt()
    _w.pop, _w.rng = [], rng
    _Worker._pop_add(_w, float(rA.sum()), cA, rA)
    _Worker._pop_add(_w, float(rA.sum()) + 1e-9, cA + 1e-9, rA + 1e-9)   # same spectrum
    assert len(_w.pop) == 1, "archive did not dedupe an identical basin"
    _Worker._pop_add(_w, float(rB.sum()), cB, rB)
    assert len(_w.pop) == 2, "archive collapsed two distinct basins"
    for _ in range(POP_MAX + 4):
        _c = rng.uniform(LO + .05, HI - .05, size=(16, 2))
        _Worker._pop_add(_w, float(_greedy_radii(_c).sum()), _c, _greedy_radii(_c))
    assert len(_w.pop) == POP_MAX, "archive exceeded POP_MAX"
    assert _Worker._pop_pick(_w) in _w.pop

    # ---- iteration 14: the Luby restart schedule and the episode it drives -----------------
    assert [_luby(i) for i in range(1, 16)] == [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8], \
        "Luby sequence is wrong"
    assert _luby(0) == 1 and max(_luby(i) for i in range(1, 64)) == 32
    # the schedule must both stay SHORT most of the time and still reach a long cutoff:
    _lb = [_luby(i) for i in range(1, 32)]
    assert sum(1 for v in _lb if v == 1) * 2 > len(_lb), "Luby lost its short episodes"
    assert sum(_lb) == 80, "Luby total over i=1..31 (12 + 12 + 8 + 12 + 12 + 8 + 16)"

    class _WEp(object):
        pass
    _we = _WEp()
    _we.pop, _we.rng = [], np.random.RandomState(4)
    _we.ep_i, _we.ep_left, _we.ep_n, _we.barren = 0, 0.0, 0, 7
    _we.ep_unit = EP_UNIT
    _we._pop_pick = lambda: _Worker._pop_pick(_we)
    _we.best = [float(rA.sum()), cA, rA]
    _we.cur = [0.0, None, None]
    _Worker._restart(_we)
    assert _we.ep_i == 1 and _we.ep_n == 1 and _we.barren == 0
    assert abs(_we.ep_left - EP_UNIT * 1.0) < 1e-12, "first episode is not one Luby unit"
    assert _we.cur[1] is cA, "an empty archive must relaunch from the champion"
    _seen = set()
    for _ in range(7):
        _we.ep_left = 0.0
        _Worker._restart(_we)
        _seen.add(round(_we.ep_left / _we.ep_unit))
    assert _seen == {1, 2, 4}, "episode cutoffs do not follow the schedule (%r)" % _seen
    # ITERATION 15 -- the unit is DERIVED, and the property it is derived for is that one full
    # Luby cycle fits the CPU a payable n owns.  Check that directly rather than checking a number.
    for _av, _np_ in ((111.0, 16), (111.0, 37), (111.0, 1), (60.0, 8), (3.0, 30)):
        _u = _ep_unit(_av, _np_)
        assert EP_UNIT_LO - 1e-12 <= _u <= EP_UNIT_HI + 1e-12, "derived unit out of band"
        if EP_UNIT_LO < _u < EP_UNIT_HI:      # unclamped: the cycle must land on the n's share
            assert abs(_u * LUBY_CYCLE - _av / _np_) < 1e-9, "a Luby cycle does not fit the share"
    assert _ep_unit(111.0, 0) > 0.0 and _ep_unit(0.0, 16) == EP_UNIT_LO, "degenerate inputs"
    # ...and it must be MONOTONE in the CPU an n owns: fewer payable n -> longer episodes
    assert _ep_unit(111.0, 37) < _ep_unit(111.0, 16) < _ep_unit(111.0, 4)
    # the shipped configuration must reproduce the sweep's winner (0.60 beat 1.50 and 0.25 at
    # 4 n x 4 seeds x 6 CPU-s: 67.561 / 63.939 / 59.829 total digits)
    assert 0.45 < _ep_unit(CPU_CAP - CPU_SAFETY, 16) < 0.75, "derivation drifted off the measured optimum"
    # the derived unit must actually REACH the worker: a solve()-built worker is not left on the
    # fallback constant.  (Checked structurally; the live check is the run itself.)
    assert _Worker(9, lambda *a: (np.array([True]), np.array([0.0])),
                   type("M", (), {"left": staticmethod(lambda: 1)})(),
                   np.random.RandomState(0)).ep_unit == EP_UNIT
    # ...and a drawn archive member is a legal relaunch point, not just the champion
    _Worker._pop_add(_we, float(rB.sum()), cB, rB)
    _Worker._pop_add(_we, float(rA.sum()), cA, rA)
    _from_pop = 0
    for _ in range(60):
        _Worker._restart(_we)
        _from_pop += int(_we.cur[1] is not _we.best[1])
    assert 0 < _from_pop < 60, "restart never (or always) drew from the archive: %d/60" % _from_pop

    grid = _init_rows(9, 3, False)                       # 3x3 grid, spacing 1/3 -> every r = 1/6
    rg = _rlp(grid)
    assert abs(rg.sum() - 9.0 / 6.0) < 1e-7, "radius LP is not exact on a known case (%r)" % rg.sum()
    assert _rlp(grid).sum() >= _greedy_radii(grid).sum() - 1e-12
    print("SELF-TEST: PASS")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
