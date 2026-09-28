"""Circle-packing solver: sequential LINEAR programming (SLP) on a convex under-estimator.

The structural fact this solver is built on. The csqv program is

    max  sum_i r_i   s.t.   r_i + r_j <= ||p_i - p_j||,   r_i <= dist(p_i, wall)

Only ONE term is nonlinear: ||p_i - p_j||. It is *convex*, so its first-order Taylor expansion is a
GLOBAL UNDER-estimator:

    ||p_i - p_j||  >=  d_ij^0 + u_ij . (dp_i - dp_j),      u_ij = (p_i^0 - p_j^0)/d_ij^0

Replacing the nonlinear constraint by that under-estimator turns the whole program into a LINEAR
program whose feasible set is an INNER approximation of the true one.  Two consequences drive
everything below:

  1. every LP solution is TRULY feasible -- no line search, no penalty, no repair-and-hope;
  2. dp = 0 is always LP-feasible, so the LP optimum is never worse than the current point.
     Iterating the LP is therefore MONOTONE by construction, and as the trust region shrinks the
     linearisation becomes exact, so it converges on a KKT point of the real NLP.

Bounding the per-step centre motion by `delta` and the per-step radius growth by `grow` also makes the
pair set exact: a pair whose slack exceeds 2*sqrt(2)*delta + 2*grow CANNOT become active this step, so
it can be dropped from the LP with a proof rather than a heuristic cutoff.  That keeps the LP at
O(n) rows and lets HiGHS solve it in milliseconds.

This replaces iteration 1's SLSQP polish, whose dense O((3n)^3) QP factorisation ate the 120 s CPU
backstop before the large n were reached.  Same exact-structure idea (LP-exact radii), pushed from the
radii onto the centres as well: iteration 1 solved an LP for r and a generic NLP for p; this solves one
LP for (p, r) jointly.

Budget: the meter (500k evaluations) is not the binding resource -- the 120 s process-CPU backstop is.
Only converged candidates go through evaluate(); time is sliced across n by remaining-time/remaining-
weight so the large n at the end of the census are never starved.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

# THE LP CALL *IS* THE ITERATION.  Profiled (artifacts/probe_iter16.py, the real restart loop at
# n=59 for 6 s): 94.5% of all process CPU is spent inside `linprog`, 6.3 LP solves per restart at
# 5.9 ms each -- everything else in this file (pdist, the move set, the matrix build, repair) is
# 5.5%.  So restart throughput, the resource the escape lottery is measured in, is set by ONE call.
# Split (artifacts/probe_iter16b.py, the identical LP, same optimum to 12 digits): scipy's linprog
# wrapper costs 4.03 ms of which HiGHS on the SAME options needs 2.71 -- a third is per-call plumbing
# (input parsing, option validation, sparse re-conversion, result post-processing), rebuilt every step.
# And a HiGHS instance that KEEPS its model between calls resolves in 0.055 ms, because the simplex
# starts from the basis it ended on instead of from scratch.
# So the LP is driven directly, through ONE reused HiGHS object per process:
#   * no wrapper plumbing per call;
#   * the previous optimal BASIS is handed back whenever the LP has the same shape -- consecutive SLP
#     steps differ only by a shrunken trust region, so the old basis is usually near-optimal.
# This is a pure THROUGHPUT change and it is not trusted on that account: the backend is verified
# against `linprog` on the same LPs (--self-test), any failure of the raw path falls back to
# `linprog` for that call, and the geometry is unchanged -- every candidate still goes through
# repair() and the metered evaluate(), which are the only things that can certify a packing.
try:                                    # scipy's own vendored HiGHS (not the banned module list)
    from scipy.optimize import _highspy as _hs
except Exception:                       # pragma: no cover - older scipy: keep the linprog path
    _hs = None

LO, HI = -0.5, 0.5
CPU_BUDGET = 106.0          # self-imposed process-CPU ceiling (driver backstop is 120 s)
FEAS_EPS = 1e-10            # margin kept inside every constraint (feasibility tol is 1e-9)
# 1e-10 is HiGHS' tightest ACCEPTED feasibility tolerance (1e-11 is rejected as an invalid option
# value and the solve returns nothing). Measured at 1e-10 the LP output violates no constraint at all,
# so repair() below only shaves off the deliberate 1e-10 epsilons (~3e-11 of sum_r).
_LP_OPT = {"presolve": True,
           "primal_feasibility_tolerance": 1e-10,
           "dual_feasibility_tolerance": 1e-10}


class _Highs(object):
    """One reused HiGHS instance behind a `linprog`-shaped call: min c.x s.t. A x <= b, lo <= x <= hi.

    Two things make this faster than `linprog` without changing a single answer:
      1. the model is passed straight to HiGHS (no wrapper parse/validate/convert/post-process);
      2. the OPTIMAL BASIS of the previous solve is restored whenever the new LP has the same shape,
         so the simplex resumes instead of restarting.  A basis is only ever a STARTING POINT -- if
         it is stale or meaningless the simplex just repairs it, so this can cost time but can never
         cost correctness.
    Returns None on anything but a verified-optimal solve, and the caller then falls back to
    `linprog`, so the raw path is an accelerator and never a second source of truth."""

    def __init__(self):
        self.h = None
        self.shape = None
        self.basis = None

    def _boot(self):
        if self.h is not None or _hs is None:
            return self.h
        try:
            h = _hs._core._Highs()
            h.setOptionValue("output_flag", False)
            # The option set is IDENTICAL to _LP_OPT, deliberately.  Turning presolve off is
            # 0.25 ms/LP faster but changes WHICH optimal vertex comes back: the max-sum LP is
            # degenerate, so a different pivot path returns the same sum_r at different
            # coordinates -- and --self-test caught exactly that (a basin-repeat fixture at n=31
            # stopped reconverging).  Equal objective is not equal behaviour, so the accelerator
            # keeps the reference solver's settings and buys its speed only from plumbing.
            h.setOptionValue("presolve", "on")
            h.setOptionValue("primal_feasibility_tolerance", 1e-10)
            h.setOptionValue("dual_feasibility_tolerance", 1e-10)
            self.h = h
        except Exception:                       # pragma: no cover
            self.h = None
        return self.h

    def solve(self, c, A, b, lo, hi):
        h = self._boot()
        if h is None:
            return None
        try:
            Ac = A.tocsc()
            nr, nc = A.shape
            lp = _hs._core.HighsLp()
            lp.num_col_ = int(nc)
            lp.num_row_ = int(nr)
            lp.col_cost_ = np.ascontiguousarray(c, dtype=float)
            lp.col_lower_ = np.ascontiguousarray(lo, dtype=float)
            lp.col_upper_ = np.ascontiguousarray(hi, dtype=float)
            lp.row_lower_ = np.full(nr, -np.inf)
            lp.row_upper_ = np.ascontiguousarray(b, dtype=float)
            lp.a_matrix_.format_ = _hs._core.MatrixFormat.kColwise
            lp.a_matrix_.start_ = np.ascontiguousarray(Ac.indptr)
            lp.a_matrix_.index_ = np.ascontiguousarray(Ac.indices)
            lp.a_matrix_.value_ = np.ascontiguousarray(Ac.data, dtype=float)
            h.clearModel()
            h.passModel(lp)
            # The basis is reused only when the LP looks like the SAME LP slightly changed --
            # same shape AND same nonzero count.  Consecutive SLP steps satisfy that (only the
            # trust region shrank); two unrelated LPs that merely happen to share a shape do not.
            # This matters beyond speed: the max-sum LP is DEGENERATE (many optimal vertices with
            # the same objective), and a foreign starting basis picks a different one -- identical
            # sum_r, different coordinates.  Keeping the reuse local keeps the search reproducible.
            if self.basis is not None and self.shape == (nr, nc, int(Ac.nnz)):
                try:
                    h.setBasis(self.basis)
                except Exception:               # pragma: no cover
                    pass
            h.run()
            if h.getModelStatus() != _hs._core.HighsModelStatus.kOptimal:
                self.basis, self.shape = None, None
                return None
            sol = h.getSolution()
            x = np.asarray(sol.col_value, dtype=float)
            if x.shape != (nc,) or not np.all(np.isfinite(x)):
                self.basis, self.shape = None, None
                return None
            try:
                self.basis, self.shape = h.getBasis(), (nr, nc, int(Ac.nnz))
            except Exception:                   # pragma: no cover
                self.basis, self.shape = None, None
            return x
        except Exception:                       # pragma: no cover
            self.basis, self.shape = None, None
            return None


_HS = _Highs()


def _lp_solve(c, A, b, lo, hi):
    """min c.x s.t. A x <= b, lo <= x <= hi.  Raw HiGHS first, `linprog` as the fallback."""
    x = _HS.solve(c, A, b, lo, hi)
    if x is not None:
        return x
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=np.stack([lo, hi], axis=1),
                      method="highs", options=_LP_OPT)
    except Exception:
        return None
    if res.x is None or not np.all(np.isfinite(res.x)):
        return None
    return np.asarray(res.x, dtype=float)


# ---------------------------------------------------------------- geometry helpers

def _walls(xy):
    return np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0], xy[:, 1] - LO, HI - xy[:, 1]])


def _pdist(xy):
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def lp_radii(xy, eps=FEAS_EPS):
    """Exact max-sum radii for FIXED centres (a pure LP).  Pairs further apart than any attainable
    r_i + r_j cannot bind, so they are dropped with a bound, not a guess."""
    n = len(xy)
    wall = np.maximum(_walls(xy) - eps, 0.0)
    dist = _pdist(xy)
    iu = np.triu_indices(n, 1)
    dv = dist[iu]
    keep = dv < (wall[iu[0]] + wall[iu[1]])          # exact: r_i <= wall_i, so d >= w_i+w_j is slack
    I, J, dv = iu[0][keep], iu[1][keep], dv[keep]
    if len(I) == 0:
        return wall
    k = len(I)
    A = coo_matrix((np.ones(2 * k), (np.concatenate([np.arange(k)] * 2), np.concatenate([I, J]))),
                   shape=(k, n))
    x = _lp_solve(-np.ones(n), A, dv - eps, np.zeros(n), wall)
    if x is None:
        return np.zeros(n)
    return np.maximum(x, 0.0)


def repair(xy, r):
    """Shrink radii until the packing is STRICTLY feasible (wall + pairwise slack > 0)."""
    r = np.minimum(np.asarray(r, float), _walls(xy) - 1e-12)
    r = np.maximum(r, 0.0)
    dist = _pdist(xy)
    np.fill_diagonal(dist, np.inf)
    for _ in range(80):
        over = (r[:, None] + r[None, :]) - dist          # >0 means overlap
        worst = over.max(axis=1)
        if worst.max() <= -1e-12:
            break
        r = np.maximum(r - np.maximum(worst + 1e-13, 0.0) * 0.5 - 1e-16, 0.0)
    # NB: never RAISE a radius to a positive floor here -- that reintroduces overlap. Degenerate
    # (LP-zero) circles get a harmless 1e-13 so every r is strictly positive.
    r = np.maximum(r - 1e-12, 0.0)
    r = np.minimum(r, np.maximum(_walls(xy) - 1e-12, 0.0))
    return np.where(r > 0.0, r, 1e-13)


# ---------------------------------------------------------------- the SLP core

def _slp_step(xy, r, delta, grow, eps=1e-10):
    """One trust-region LP over (x, y, r) jointly.  Returns (xy, r, lp_sum) or None if the LP failed.

    Pair (i,j) is INCLUDED iff its slack can reach 0 this step: centres move <= sqrt(2)*delta each and
    radii grow <= `grow` each, so slack drops by at most 2*sqrt(2)*delta + 2*grow.  Anything slacker is
    provably inactive and is dropped -- that is what keeps the LP small.
    """
    n = len(xy)
    dist = _pdist(xy)
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, 1)
    slack = dist[iu] - (r[iu[0]] + r[iu[1]])
    thr = 2.8285 * delta + 2.0 * grow
    keep = slack <= thr
    I, J = iu[0][keep], iu[1][keep]
    m = len(I)

    x0, y0 = xy[:, 0], xy[:, 1]
    idx = np.arange(n)
    one = np.ones(n)
    if m:
        dx, dy = x0[I] - x0[J], y0[I] - y0[J]
        d = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-300)
        ux, uy = dx / d, dy / d
        rows_p = np.repeat(np.arange(m), 6)
        cols_p = np.stack([I, J, n + I, n + J, 2 * n + I, 2 * n + J], axis=1).ravel()
        data_p = np.stack([-ux, ux, -uy, uy, np.ones(m), np.ones(m)], axis=1).ravel()
    else:
        rows_p = cols_p = data_p = np.zeros(0)

    # WALL ROWS, PRUNED BY THE SAME VALID BOUND AS THE PAIRS.  Row (i,k) is
    #   x_i + r_i <= .5,  -x_i + r_i <= .5,  y_i + r_i <= .5,  -y_i + r_i <= .5.
    # Its slack falls by at most delta (centre motion is bounded by delta per axis) + grow (radius
    # growth is bounded by `grow`), so a row with slack > delta + grow CANNOT bind this step and is
    # dropped with a proof.  For large n most circles are interior, so this removes most of the 4n
    # rows -- the LP shrinks from ~7n rows to ~3.5n and HiGHS gets proportionally faster.
    wslack = np.stack([HI - x0 - r, x0 - LO - r, HI - y0 - r, y0 - LO - r], axis=1)
    wi, wk = np.nonzero(wslack <= delta + grow)
    w = len(wi)
    if w:
        rows_w = np.repeat(m + np.arange(w), 2)
        cols_w = np.stack([np.where(wk < 2, wi, n + wi), 2 * n + wi], axis=1).ravel()
        sgn = np.where(wk % 2 == 0, 1.0, -1.0)
        data_w = np.stack([sgn, np.ones(w)], axis=1).ravel()
    else:
        rows_w = cols_w = data_w = np.zeros(0)

    A = coo_matrix((np.concatenate([data_p, data_w]),
                    (np.concatenate([rows_p, rows_w]), np.concatenate([cols_p, cols_w]))),
                   shape=(m + w, 3 * n)).tocsr()
    b = np.concatenate([np.full(m, -eps), np.full(w, HI - eps)])
    lo = np.concatenate([np.maximum(x0 - delta, LO), np.maximum(y0 - delta, LO), np.zeros(n)])
    hi = np.concatenate([np.minimum(x0 + delta, HI), np.minimum(y0 + delta, HI),
                         np.minimum(r + grow, HI)])
    c = np.concatenate([np.zeros(2 * n), -one])
    z = _lp_solve(c, A, b, lo, hi)
    if z is None:
        return None
    nxy = np.clip(np.stack([z[:n], z[n:2 * n]], axis=1), LO, HI)
    return nxy, np.maximum(z[2 * n:], 0.0), float(z[2 * n:].sum())


def slp_run(xy, deadline, delta0=None, maxit=400, tol=1e-12, dmin=1e-12, r0=None,
            ref=None, ref_tol=0.0, peek_it=6, stop_eps=0.0, gate=None):
    """Run SLP from `xy` until the trust region collapses or the deadline hits.  Monotone by design.
    Returns (xy, r) -- r is the RAW LP radii (feasible to HiGHS' 1e-10 primal tolerance); call
    repair() before handing it to evaluate().  Raw is returned so a coarse run can be CONTINUED by a
    finer one without a repair round-trip.

    `dmin` is where the trust region is allowed to collapse to.  Exploration only needs enough
    accuracy to RANK basins (they differ by ~1e-3 relative), and the last decade of trust-region
    collapse costs ~ log(1e-6)/log(0.35) = 13 of the ~35 LP steps -- so search runs coarse and only a
    candidate that threatens the incumbent is polished to 1e-12.

    EARLY BASIN-REPEAT ABORT (`ref`/`ref_tol`).  A reconvergence that is walking back to the
    incumbent it was perturbed from cannot win -- the incumbent is already committed, so every LP
    step after that is spent re-deriving a known answer.  It is also *decidable early*: measured
    (artifacts/probe_iter6c.py) at LP step 6 the max per-circle displacement from the incumbent is
    0.000 for a trial that returns and 0.03-0.17 for one that does not, and the smallest peek
    displacement over all WINNERS was 0.035 -- more than 10x the abort threshold.  So one O(n)
    comparison at step `peek_it` throws away the returning trials for ~1/6 of their cost and has
    never been observed to discard a winner.  Returns (xy, r, aborted).

    CONSUMER-DRIVEN STOP (`stop_eps`).  `dmin` is a PRODUCER-side stopping rule -- it asks how far
    the trust region has collapsed, not whether anyone can still read the answer.  The exploration
    tier's only reader is the comparison `sum_r > best - POLISH_GAP`, so once a step's gain drops
    below a fraction of POLISH_GAP the remaining steps cannot move that decision.  The run is
    MONOTONE, so an early stop can only UNDER-report sum_r, never over-report: the whole error is
    `s_final - s_stop >= 0`, which merely makes the polish filter very slightly stricter.  Measured
    over 132 restarts at n=33..99 (artifacts/probe_iter8b.py) that under-report is at most 3.1e-11
    against a 3e-5 threshold -- a 1e6 margin -- while cutting 38% of the LP solves per restart.
    Left at 0.0 (the default) the rule is off, which is what the polish tier wants: its consumer is
    the scorer at 7 digits, so it runs to dmin=1e-12.

    DOOMED-BASIN GATE (`gate`).  The repeat abort kills a reconvergence that returns to the
    incumbent; nothing killed one that runs AWAY from it, and measured that is where the budget goes:
    the median restart reconverges 1.3e-2 BELOW the incumbent and only ~1% land within 1e-3
    (artifacts/probe_iter18b.py, 719 candidates at n=35/59/81).  Those are run out to their own
    stopping rule and then discarded, so ~2/3 of every funded slice buys nothing.

    The run is MONOTONE, so sum_r at step k is a LOWER bound on the final sum -- which by itself
    licenses no abort.  What licenses this one is measurement: `artifacts/probe_iter18c.py` traces the
    per-step sums of 435 real restarts at n=59/81 and replays every gate rule against BOTH consumers
    of a coarse sum (the polish filter at POLISH_GAP, and drift adoption at up to DFRAC_HI of the mean
    radius).  At step 3 an absolute gate of 1e-2 kills 66-68% of candidates for 34% of the LP steps
    with **zero** false negatives on either consumer at both n; the first false negative appears only
    at 5e-3 (drift, n=59).  The shipped gate is expressed as a multiple of the widest ball any
    consumer can legally have -- GATE_MULT * best_s / n with GATE_MULT = 2.5 * DFRAC_HI -- which is
    1.5-1.7x LOOSER than the measured zero-FN point, so a candidate that any consumer could still use
    cannot be gated.  `gate` is an absolute floor on sum_r; None (the default) is an exact no-op.

    Returns (xy, r, code): code 0 = ran out, 1 = basin repeat, 2 = gated.  The two aborts are
    DISTINCT because their controller semantics are opposite -- a repeat means the jitter is too
    small (grow sig), a gated basin means it landed far (do not grow sig).  Collapsing them into one
    flag would feed the gate's own kills back into `_sig_update` as repeats and push the moves
    further out with every kill."""
    n = len(xy)
    if delta0 is None:
        delta0 = 0.20 / np.sqrt(n)
    delta = float(delta0)
    r = lp_radii(xy) if r0 is None else r0
    best_s = float(r.sum())
    # the stop is deferred past `peek_it` so it can never pre-empt the basin-repeat abort: a
    # repeat that stalled early would otherwise reach the polish filter (where it passes, being
    # the incumbent) and pay for a polish that cannot improve on a committed answer.
    stop_after = peek_it if ref is not None else 0
    for it in range(maxit):
        if time.process_time() > deadline or delta < dmin:
            break
        prev_s = best_s
        out = _slp_step(xy, r, delta, grow=delta)
        if out is None:
            delta *= 0.4
            if stop_eps > 0.0 and it >= stop_after:
                break                                  # gain 0: nothing left for the consumer
            continue
        nxy, nr, s = out
        if s > best_s + tol:
            xy, r, best_s = nxy, nr, s
            delta = min(delta * 1.6, delta0)
        else:
            if s > best_s:
                xy, r, best_s = nxy, nr, s
            delta *= 0.35
        if ref is not None and it >= peek_it and np.abs(xy - ref).max() < ref_tol:
            return xy, r, 1
        if gate is not None and it >= peek_it and best_s < gate:
            return xy, r, 2                            # doomed: monotone, and already far below
        if stop_eps > 0.0 and it >= stop_after and best_s - prev_s < stop_eps:
            break
    return xy, r, 0


def slp_optimise(xy, deadline, delta0=None, maxit=400, tol=1e-12, dmin=1e-12, r0=None):
    """Backwards-compatible 2-tuple view of slp_run (no early abort)."""
    xy, r, _ = slp_run(xy, deadline, delta0=delta0, maxit=maxit, tol=tol, dmin=dmin, r0=r0)
    return xy, r


# ---------------------------------------------------------------- census warm start

def read_pack(n, root=""):
    """Centres+radii of the committed packing for n, or None.  GUARDED: missing/bad file -> None."""
    path = os.path.join(root, "bench", "packs", "csqv%d.pck" % n)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        rows = []
        for ln in lines[2:]:
            parts = ln.split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        arr = np.asarray(rows, dtype=float)
        if arr.shape != (n, 3):
            return None
        return arr
    except Exception:
        return None


def _cold_start(n, rng, mode):
    """Structured + random cold starts.  Must work for ANY n (no census dependence)."""
    if mode == 0:
        k = int(np.ceil(np.sqrt(n)))
        g = (np.arange(k) + 0.5) / k - 0.5
        gx, gy = np.meshgrid(g, g)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)[:n]
        if len(pts) < n:
            pts = np.concatenate([pts, rng.uniform(LO + 0.05, HI - 0.05, size=(n - len(pts), 2))])
        return pts + rng.normal(0.0, 0.25 / k, size=(n, 2))
    if mode == 1:                                    # hex-ish rows
        k = int(np.ceil(np.sqrt(n * 1.1547)))
        pts = []
        for row in range(k + 2):
            y = LO + (row + 0.5) / (k + 1)
            off = 0.5 / (k + 1) if row % 2 else 0.0
            for col in range(k + 2):
                x = LO + off + (col + 0.5) / (k + 1)
                if LO < x < HI and LO < y < HI:
                    pts.append((x, y))
        pts = np.asarray(pts, dtype=float)
        if len(pts) < n:
            pts = np.concatenate([pts, rng.uniform(LO + 0.05, HI - 0.05, size=(n - len(pts), 2))])
        rng.shuffle(pts)
        return pts[:n] + rng.normal(0.0, 0.15 / (k + 1), size=(n, 2))
    return rng.uniform(LO + 0.02, HI - 0.02, size=(n, 2))


def _pockets(xy, r, kmax, res=None):
    """The k largest EMPTY pockets: points q maximising clearance(q) = min(min_j(|q-p_j|-r_j), wall(q)).

    That clearance is exactly the radius a new circle centred at q could take, so a pocket is the
    provably best place to put a circle -- unlike a uniform-random teleport, which lands inside an
    existing circle most of the time.  Pockets are found on a grid and then de-duplicated by
    excluding everything within one clearance of an accepted pocket (same pocket, not a new one).
    """
    n = len(xy)
    if res is None:
        res = int(min(80, max(24, 5.0 * np.sqrt(n))))
    g = (np.arange(res) + 0.5) / res - 0.5
    X, Y = np.meshgrid(g, g)
    q = np.stack([X.ravel(), Y.ravel()], axis=1)
    d = np.sqrt(((q[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]
    cl = np.minimum(d.min(axis=1),
                    np.minimum.reduce([q[:, 0] - LO, HI - q[:, 0], q[:, 1] - LO, HI - q[:, 1]]))
    order = np.argsort(-cl)
    out, alive = [], np.ones(len(q), bool)
    for idx in order:
        if len(out) >= kmax:
            break
        if not alive[idx] or cl[idx] <= 0.0:
            continue
        out.append(q[idx])
        alive &= ((q - q[idx]) ** 2).sum(1) > cl[idx] ** 2
    if not out:
        return np.zeros((0, 2))
    return np.asarray(out, dtype=float)


def _transfer(src, n):
    """Structural cross-n move: reshape a packing of size m into a start for size n.

    m > n : drop the (m-n) smallest circles (they contribute least to sum r).
    m < n : add (n-m) circles at the largest empty pockets, falling back to the biggest circles'
            neighbourhood if the grid finds too few.
    Returns centres (n,2), or None if it cannot be done.
    """
    if src is None or len(src) == 0:
        return None
    xy, r = src[:, :2], src[:, 2]
    m = len(xy)
    if m == n:
        return xy.copy()
    if m > n:
        return xy[np.argsort(-r)[:n]].copy()
    need = n - m
    pk = _pockets(xy, r, need)
    if len(pk) < need:
        extra = xy[np.argsort(-r)[:need - len(pk)]] * 0.97
        pk = np.concatenate([pk, extra]) if len(pk) else extra
    return np.concatenate([xy, pk[:need]], axis=0)


# A SMOOTH FIELD MOVES THE PACKING WITHOUT BREAKING IT.  Measured (artifacts/probe_iter19.py, the
# real coarse tier, matched seed/CPU/origin): iid jitter lands a median 0.9-1.3e-2 below the
# incumbent and within 1e-3 of it 0-1% of the time -- it displaces every circle INDEPENDENTLY, so it
# breaks every contact at once and the reconvergence has to rebuild the whole packing.  A smooth
# displacement field (here a traceless affine map: shear/stretch, area-preserving to first order)
# moves circles that are neighbours by nearly the same vector, so the LOCAL contact geometry is
# carried along while the packing as a whole slides against the FIXED square -- which is where a
# different structure can appear.  At n=81 the same coarse tier lands within 1e-3 on 9-16% of
# restarts instead of 0%, with a median of -5.2e-3 against jitter's -8.9e-3.
#
# The amplitude is a THRESHOLD, not a slope, and the threshold moves with n: at 0.25*sig the field
# repeats 99% of the time at n=35 and 32% at n=81, and at 2x that amplitude it is destroyed at every
# n (the doomed-basin gate kills ~100%).  So it is not a constant -- it is driven by the same
# observable, and the same law, as SIG: grow on a basin repeat (too small to leave), shrink slowly
# otherwise (drifting down until repeats reappear).  The clamp brackets the measured window on both
# sides: 0.08 is below the amplitude that repeated at every n, and 0.55 is half the amplitude at
# which the doomed-basin gate killed ~100% of restarts at every n.
AFR0, AFR_LO, AFR_HI = 0.25, 0.08, 0.55     # field amplitude as a multiple of that n's jitter scale

# ...and the SHARE of restarts the channel gets is driven by the consumer that actually reads a
# landing.  A candidate that does not beat the incumbent is read by exactly one thing: the DRIFT
# adopter, whose tolerance is dfrac * best/n (~1e-3 to 5e-3 here).  So the channel is funded out of
# its own measured ADOPT rate -- the criterion of the mechanism it exists to feed -- not out of a
# landing statistic of its own choosing.  Up on adopt, down otherwise, so it settles at an adopt
# rate of ln(DN)/(ln(DN) - ln(UP)) ~= 0.20; floored so it stays a live probe if the regime returns,
# capped at the share it was measured with so it can only ever spend LESS on itself than measured.
FIELD_P0, FIELD_LO = 0.25, 0.03
FIELD_UP, FIELD_DN = 1.06, 0.985            # -> equilibrium adopt rate ~= 0.20


# THE FIELD IS A FAMILY, NOT A MOVE.  The affine map is the wavenumber-0 member of the smooth-field
# family (a linear field: one global shear/stretch/rotation).  The rest of the family is indexed by
# WAVENUMBER -- a k-mode field moves different regions of the packing in different directions while
# still carrying neighbours together, so it reorganises the interior instead of only sliding the
# whole object against the walls.  Nothing licenses a guess about which member pays: iteration 19
# measured the Fourier field "intermediate" at ONE amplitude with NO controller, which is a
# statement about that amplitude, not about the mode.  So every member is shipped as a live probe
# under the same repayment law as the channel itself, keyed on the same consumer (the drift
# adopter): a mode that stops being adopted decays to a floor and is retired, not deleted, and
# revives if the regime returns.  The MEASURED member (affine) starts at the cap and can only lose
# weight; an unmeasured probe starts small and can at most reach parity with it.
# A wavenumber is NOT a constant: a fixed k in square units is a different move at n=27 and n=99,
# because what "smooth" means here is measured against the packing's OWN nearest-neighbour spacing
# (~1/sqrt(n)).  So each member is specified by its correlation length in SPACINGS -- the wavevector
# has magnitude sqrt(n)/L, giving a wavelength of exactly L spacings at every n -- and the measured
# smoothness ratio then comes out at ~L independently of n, which is what the self-test asserts.
FMODES = (0, 1, 2)                     # 0 = affine (k=0 gradient); 1, 2 = Fourier, L spacings below
FMODE_L = (0.0, 10.0, 5.0)             # correlation length of each Fourier member, in spacings
# ...and each member gets its OWN starting amplitude, because the threshold is a property of the
# member, not of the channel.  Measured (artifacts/probe_iter20.py, real coarse tier, n=59/81): run
# at mode 0's amplitude the Fourier members look strictly worse (0-5% near-landings, and mode 2 is
# destroyed at n=81 -- 0% repeats, 62% gated).  Run at THEIR OWN amplitude they are not: mode 1 at
# 0.12 lands within 1e-3 on 11.6% at n=81 (mode 0's own best is 12.1% at 0.25), and mode 2 at
# 0.08-0.12 is the only member that produced a candidate inside POLISH_GAP at all (2.4-2.6% at
# n=81, 1.5% at n=59, against 0% for mode 0 at every amplitude).  A single shared amplitude would
# have measured the family and concluded the family does not work.  These are starting points, not
# settings: each stays under the same repeat-rate law and the same [AFR_LO, AFR_HI] clamp.
AFR_INIT = (0.25, 0.12, 0.10)          # per-member starting amplitude (AFR_INIT[0] == AFR0)
FW0 = (1.0, 0.25, 0.25)                # initial per-mode weights (probes below the measured member)
FW_LO, FW_HI = 0.05, 1.0               # floored (retire, not delete) and capped at the measured one


def _field(xy, rng, n, amp, mode=0):
    """A smooth displacement field of RMS magnitude `amp`, applied to centres.

    mode 0 -- traceless affine (shear/stretch/rotation), the wavenumber-0 member.
    mode m>=1 -- a sum of 2 random Fourier components whose wavelength is FMODE_L[m] nearest-
    neighbour SPACINGS (|k| = sqrt(n)/L): still smooth (neighbours travel together) but no longer a
    single global slide -- different regions of the packing move in different directions.
    """
    if int(mode) <= 0:
        a, b, c = rng.normal(size=3)
        d = xy.dot(np.array([[a, c], [b, -a]]))      # x M^T with M = [[a,b],[c,-a]]
    else:
        L = FMODE_L[int(mode) % len(FMODE_L)]
        kmag = max(1.0, np.sqrt(float(n)) / max(L, 1e-9))
        d = np.zeros_like(xy)
        for _ in range(2):
            ka, ph, th = rng.uniform(0.0, 2 * np.pi, size=3)
            kv = kmag * np.array([np.cos(ka), np.sin(ka)])
            w = rng.normal()
            d += w * np.outer(np.cos(2 * np.pi * xy.dot(kv) + ph),
                              np.array([np.cos(th), np.sin(th)]))
    rms = float(np.sqrt((d ** 2).sum(1).mean()))
    if rms < 1e-12 or not np.isfinite(rms) or amp <= 0.0:
        return xy.copy()
    return xy + (float(amp) / rms) * d


def _fw_norm(fw):
    """Coerce a per-mode weight vector; a scalar means the single measured member."""
    try:
        w = [float(x) for x in fw]
    except TypeError:
        return [1.0] + [0.0] * (len(FMODES) - 1)
    if len(w) != len(FMODES) or not all(np.isfinite(x) and x >= 0.0 for x in w) or sum(w) <= 0.0:
        return [1.0] + [0.0] * (len(FMODES) - 1)
    return w


def _pick_mode(rng, fw):
    """Sample a field mode proportional to its weight.  Consumes NO rng draw when only one member
    is live, so a single-mode weight vector reproduces the shipped move set bit-for-bit."""
    w = _fw_norm(fw)
    live = [i for i, x in enumerate(w) if x > 0.0]
    if len(live) <= 1:
        return live[0] if live else 0
    c = np.cumsum(w)
    return int(np.searchsorted(c, rng.rand() * float(c[-1]), side="right"))


def _fw_update(fw, mode, credited):
    """Per-mode weight under the SAME law and rates as the channel's own share -- same consumer."""
    w = _fw_norm(fw)
    m = int(mode) % len(w)
    w[m] = float(min(FW_HI, max(FW_LO, w[m] * (FIELD_UP if credited else FIELD_DN))))
    return tuple(w)


def _afr_get(afr, mode):
    """Read the per-mode amplitude; a scalar is the same amplitude for every mode."""
    try:
        v = [float(x) for x in afr]
    except TypeError:
        return float(afr)
    if len(v) != len(FMODES):
        return float(v[0]) if v else AFR0
    return v[int(mode) % len(v)]


def _afr_set(afr, mode, val):
    try:
        v = [float(x) for x in afr]
    except TypeError:
        v = [float(afr)] * len(FMODES)
    if len(v) != len(FMODES):
        v = [AFR0] * len(FMODES)
    v[int(mode) % len(v)] = float(val)
    return tuple(v)


def _afr_update(afr, repeated):
    """Per-n field amplitude from the observed repeat rate -- same law and same rates as SIG.
    Applied PER MODE: modes differ in how much local strain they cost at equal RMS, so a shared
    amplitude would converge to a compromise that suits neither member."""
    return float(min(AFR_HI, max(AFR_LO, float(afr) * (SIG_UP if repeated else SIG_DN))))


def _field_update(p, adopted):
    """Per-n field share, driven by the ADOPT rate -- the criterion its own consumer reads."""
    return float(min(FIELD_P0, max(FIELD_LO, float(p) * (FIELD_UP if adopted else FIELD_DN))))


def _move(pack, rng, n, pool, sig=None, pxfer=None, pfield=0.0, afr=AFR0, fw=None):
    """Pick one perturbation of the incumbent.  Returns (centres, delta0, move_id, ref_ok, fmode).

    `fmode` is the field-family member that fired (-1 when the field channel did not fire), so the
    caller can charge the per-mode repayment law for exactly the member it used.

    `ref_ok` says whether the returned centres are index-aligned with `pack`, i.e. whether the
    caller may compare them to the incumbent to detect a basin repeat.  The cross-n transfer builds
    its array from a DIFFERENT n's packing, so its indices mean nothing here and it opts out.

    The mixture below is MEASURED, not assumed (artifacts/probe_iter5c.py, probe_iter5d.py: ~200
    reconverged restarts per move at n=45 and n=71, scored against the incumbent):

      * "smallest circles -> largest pockets" and "scatter a neighbourhood into pockets" won
        **0 of ~200** trials -- median -2e-2 to -5e-2, BEST case still -1.7e-2.  A converged packing
        is fully jammed (mean contact degree 5.3-5.6, zero rattlers), so its largest pocket is tiny:
        lifting a circle out of a good seat and dropping it there is a pure loss the SLP cannot undo.
        Those two moves are DELETED -- they were half of every restart.
      * the cross-n transfer was DETERMINISTIC: it recomputed one identical candidate up to 58 times
        in a single run (median == p90 == max, to the last digit).  Jittered, the same move goes from
        0 wins to a 17% win rate at n=71 with a best of +1.1e-2.  Randomness is not what proposes the
        move -- the neighbour's structure is -- it only breaks the repeat.
      * the jitter scale is NOT a constant and is no longer guessed.  Iteration 5 picked
        sigma = 0.10/sqrt(n) because it "lands within 1e-4 of the incumbent 91-97% of the time" --
        which, re-measured (artifacts/probe_iter6c.py), is not a virtue but the failure mode: those
        landings ARE the incumbent, re-derived.  At sigma = 0.10 the reconvergence walks straight
        back 67% of the time at n=45 and 95% at n=71; at sigma = 0.20 the repeat rate falls to 15%
        and the wins per second rise 5x.  The right scale also depends on n (the same sigma repeats
        67% at n=45 and 95% at n=71), so `sig` is a per-n value driven by the observed repeat rate
        (see `_sig_update`) rather than a constant tuned on one n.

    `_pockets` survives because `_transfer` genuinely needs it: adding a circle to a packing of size
    m < n IS a pocket-filling problem.  What the measurement killed is pocket moves at FIXED n.
    """
    xy = pack[:, :2].copy()
    sig = SIG0 if sig is None else float(sig)
    pxf = XFER_P0 if pxfer is None else float(pxfer)
    if pxf > 0.0 and rng.rand() < pxf:                # cross-n: reshape a neighbour, then jitter
        cand = [pool[m] for m in (n - 2, n + 2, n - 4, n + 4) if m in pool]
        if cand:
            out = _transfer(cand[rng.randint(len(cand))], n)
            if out is not None:
                out = out + rng.normal(0.0, 0.12 / np.sqrt(n), size=(n, 2))
                return np.clip(out, LO + 1e-3, HI - 1e-3), 0.35 / np.sqrt(n), 1, False, -1
    # The remaining two channels keep the mixture they were measured at, CONDITIONAL on the transfer
    # not firing: at pxf = XFER_P0 = 0.15 the unconditional shares are exactly the old 0.10 / 0.75.
    u = rng.rand()
    if u < SHAKE_SHARE:                               # local cluster shake (largest single jump seen)
        k = max(3, int(0.15 * n))
        c = xy[rng.randint(n)]
        who = np.argsort(((xy - c) ** 2).sum(1))[:k]
        xy[who] += rng.normal(0.0, 0.50 / np.sqrt(n), size=(k, 2))
        return np.clip(xy, LO + 1e-3, HI - 1e-3), 0.35 / np.sqrt(n), 2, True, -1
    pf = float(pfield)
    if pf > 0.0 and u < SHAKE_SHARE + pf:             # smooth field: slide, do not shatter
        m = _pick_mode(rng, FW0 if fw is None else fw)
        out = _field(xy, rng, n, _afr_get(afr, m) * sig, m)
        return np.clip(out, LO + 1e-3, HI - 1e-3), 0.35 / np.sqrt(n), 4, True, m
    s = sig if u < 0.85 else sig * rng.uniform(1.5, 3.0)
    xy += rng.normal(0.0, s / np.sqrt(n), size=(n, 2))
    return np.clip(xy, LO + 1e-3, HI - 1e-3), 0.35 / np.sqrt(n), 3, True, -1


# ---------------------------------------------------------------- the solve loop

# THE CROSS-N TRANSFER IS A LEVER WHOSE SIGN FLIPPED, SO IT IS DRIVEN BY ITS OWN MEASURED RATE.
# Iteration 5 measured the jittered transfer winning 17% of trials at n=71 and installed it at a
# FIXED 15% of restarts.  That constant was fitted on a LOOSE census; against a tight one it is not
# neutral but actively destructive.  Measured (artifacts/probe_iter13b.py vs probe_iter13c.py: same
# n, same three seeds, same 6 s slice, same resumed walk, the ONLY difference being whether the pool
# holds n+-2 so the branch can fire): n=33 reaches the record in 3 of 3 seeds with the branch dead
# and in 0 of 3 with it live.  That is the whole four-iteration freeze of the worst-relgap n -- it
# was getting the LARGEST focused slice and spending it on a move that cannot pay, because a
# transfer is index-misaligned (ref_ok=False) so the basin-repeat abort cannot cut it short, and an
# adopted transfer replaces the drift ORIGIN with an alien structure, resetting the trajectory that
# was about to escape.
# It is not deleted, because where iteration 5 measured it the finding was real.  It is put under a
# law of the same shape as SIG and DFRAC, but keyed on the rate that the SCORED outcome depends on
# -- how often a transfer produces a candidate that threatens the incumbent (the POLISH_GAP filter),
# not how often it is adopted.  Multiplicative up-on-threat / down-otherwise settles at a threat
# rate of ln(XFER_DN)/(ln(XFER_DN) - ln(XFER_UP)) ~= 13%, i.e. the branch keeps its full 15% share
# exactly where iteration 5's 17% win rate still holds, and decays to a 1% probe rate where it does
# not.  The floor is non-zero so a census that later loosens can revive it; the cap is iteration 5's
# own value, so this can only ever spend LESS on the move than before, never more.
XFER_P0 = 0.15              # initial (and maximum) probability of the cross-n transfer branch
XFER_LO = 0.01              # floor: never fully dead, so the branch can revive if it starts paying
XFER_UP, XFER_DN = 2.0, 0.90           # -> equilibrium threat rate ~= 0.13
SHAKE_SHARE = 0.10 / (1.0 - XFER_P0)   # cluster-shake share GIVEN no transfer; 0.10 at pxf = XFER_P0


def _xfer_update(p, threatened):
    """Per-n cross-n transfer probability, driven by its own measured threat rate."""
    return float(min(XFER_P0, max(XFER_LO, float(p) * (XFER_UP if threatened else XFER_DN))))


PASSES = 3
# EXPLORATION STOPS WHERE THE DECISION IS ALREADY MADE.  The coarse tier exists only to RANK a
# basin against the incumbent through the POLISH_GAP filter, so it needs exactly enough accuracy to
# order two numbers 3e-5 apart -- not a converged optimum.  Measured (artifacts/probe_iter7b.py, 48
# restarts over n=33/45/71/99): collapsing the trust region from 1e-3 down to 1e-6 moves sum_r by at
# most 3.3e-13 -- EIGHT orders of magnitude below the threshold it feeds -- while costing 40-60% of
# the restart (7 of ~16 LP steps).  Those steps cannot change a single decision, so they are not
# approximated, they are dropped.  Polish (the only tier whose digits are scored) still runs to
# 1e-12, resuming from the coarse point with POLISH_DELTA0.
COARSE_DMIN = 1e-3          # trust-region floor during exploration (polish still goes to 1e-12)
POLISH_DELTA0 = 3e-3        # polish resumes just above the coarse floor -- same basin, no rework
POLISH_GAP = 3e-5           # polish a coarse candidate only if it is within this of the incumbent
# STOP WHERE THE READER STOPS READING, NOT WHERE THE TRUST REGION COLLAPSES.  COARSE_DMIN is a
# producer-side rule (how small is delta?); the exploration tier's actual reader is the POLISH_GAP
# comparison above.  Measured per LP step (artifacts/probe_iter8.py, probe_iter8b.py; 132 restarts
# over n=33/45/59/71/85/99): the coarse tier spends 10.9 LP solves per restart but its sum_r stops
# moving around step 5-6, and stopping at the first step whose gain falls below STOP_EPS cuts 38%
# of those solves.  The run is monotone, so the ONLY error this can make is under-reporting sum_r --
# measured at most 3.1e-11, i.e. the polish filter effectively tightens from 3e-5 to 3.00003e-5, and
# ZERO of the 132 polish decisions flipped.  STOP_EPS is set at POLISH_GAP/10 so the rule is stated
# in the consumer's units: a tenth of the smallest difference anyone downstream can see.
STOP_EPS = 3e-6             # coarse-tier early stop; 0.0 disables it (the polish tier keeps 0.0)
CAP_RELGAP = 1e-7           # relgap at or below which an n is pinned at the 7-digit score cap

# CONCENTRATION.  Per n the search is ONE correlated trajectory (the origin, the jitter scale and the
# drift fraction all persist), and its outcome is a LOTTERY, not a convergence: measured
# (artifacts/probe_iter11b.py) at n=33 a 2.5 s trajectory reaches the record 40% of the time and stalls
# at 2.64 digits the rest, and the escape probability rises with CONTIGUOUS time -- 10 s reached the
# record in 3 of 3 seeds.  The lottery does not split: at matched total time (probe_iter11c.py)
# 4 x 2.5 s independent restarts were never better than one 10 s trajectory and were 0.15 digits WORSE
# at n=99.  So the resource that matters is contiguous seconds per n, and a sweep that hands all ~25
# live n a slice buys every one of them a sub-threshold draw.
# Spend on HEADROOM instead: rank the live n by relgap and give time only to the worst prefix, sized so
# each focused n gets about FOCUS_BOOST times the slice a uniform sweep would give it.  Everything else
# is re-offered through evaluate() and therefore cannot regress -- concentration risks zero gain, never
# a loss.  An n with no published record (an offline re-run on sizes outside the census) sorts FIRST,
# so unknown sizes are never starved by the census's own numbers.
FOCUS_BOOST = 4.0

# HEADROOM IS A PRIOR, NOT A POSTERIOR.  `_focus` ranks purely by relgap, which is a statement about
# how much an n COULD gain and says nothing about whether this move set can gain it.  The rule is
# deterministic in a state that only ratchets, so the same worst-relgap n win the largest slice every
# pass of every iteration -- and measured over the trajectory, the top of that ranking has absorbed
# the biggest slice for several iterations and returned nothing, while every n that actually improved
# sat at the TAIL of the funded prefix or entered it only after someone above it moved.  That is the
# silent-repetition failure: a search that always departs from the same point, re-bought each round.
# So the allocator gets a law of the same shape as SIG / DFRAC / XFER: drive it from the rate at which
# its own mechanism pays off.  Each n carries a CREDIT multiplier on its headroom, decayed when a
# funded slice returns no improvement and restored when one does.  Three properties keep it safe:
#   * capped at CRED0 = 1.0, so the rule can only ever move time AWAY from an unproductive n, never
#     invent priority an n did not earn from its own headroom;
#   * floored at CRED_LO > 0, so an n is retired, never deleted -- as the currently funded n decay,
#     a starved one rises back to the top on its own and the prefix rotates instead of freezing;
#   * only an n that actually RAN a slice is updated, so an unfunded n neither gains nor loses.
# The record stays monotone (everything is re-offered through evaluate()), so a rotation that funds
# the wrong n risks zero gain, never a loss.
CRED0, CRED_LO = 1.0, 0.02
CRED_UP, CRED_DN = 4.0, 0.60     # one improvement outweighs ~3 misses, i.e. one iteration of passes


def _cred_update(c, improved):
    """Per-n funding credit from its OWN measured productivity.  Capped at CRED0 (this can only
    reduce an n's share of the budget relative to raw headroom) and floored at CRED_LO (retired,
    never deleted -- a stuck n revives once the n now ahead of it have stalled too)."""
    return float(min(CRED0, max(CRED_LO, float(c) * (CRED_UP if improved else CRED_DN))))


def _relgap(n, s, records):
    """Headroom for n: (record - sum_r)/record.  Unknown record -> +inf (always worth time)."""
    rec = records.get(n)
    if rec is None or rec <= 0.0:
        return float("inf")
    if s is None:
        return float("inf")
    return max(0.0, (rec - float(s)) / rec)


NEARCAP_MULT = 10.0         # "one decade from the cap": relgap <= NEARCAP_MULT * CAP_RELGAP


def _focus(wt, best_s, records, boost=FOCUS_BOOST, cred=None):
    """Zero the weight of every live n outside the worst-relgap prefix carrying 1/boost of the total.

    NEAR-CAP ADMISSION.  relgap ranks by PRIZE (how many digits an n could still win) and says
    nothing about COST (how far it has to travel to win them), and the score is a CLAMPED function:
    an n one decade from the cap has a bounded distance left and its last digit counts exactly as
    much as anybody's first.  Ranking on prize alone therefore starves the cheapest digits on the
    board permanently -- by construction they rank LAST, so no amount of budget ever reaches them.
    So an n within NEARCAP_MULT of the cap joins the funded set outright.  This is not a licence to
    ignore repayment: it is withdrawn once that n's credit has decayed to the retirement floor, so
    a near-cap n that is funded and returns nothing still yields its slice like everyone else.

    `cred` (optional) scales each n's headroom by its own measured productivity, so an n that has
    been funded and returned nothing yields its slice to the next one down.  cred is capped at 1.0,
    so passing it can only ever REORDER within the headroom prior, never promote an n above what its
    own relgap earns; omitting it reproduces the pure-headroom rule exactly.

    Returns a NEW weight dict.  Always keeps at least one n, and is a no-op when the live set is
    already small enough that no n would gain (that is what makes an offline re-run on a handful of
    sizes behave exactly as before)."""
    live = [n for n in wt if wt[n] > 0.0]
    if len(live) <= 1 or boost <= 1.0:
        return dict(wt)
    # An n whose headroom is UNKNOWN -- no published record, or no incumbent packing at all -- is
    # never ranked and never dropped.  A missing pack scores 0, so the gain from producing any
    # feasible packing dominates every relgap; and when NOTHING is rankable (an offline re-run on
    # sizes outside the census) this makes the whole rule an exact no-op rather than an arbitrary
    # ordering by n.
    gap = {n: _relgap(n, best_s.get(n), records) for n in live}
    if cred:
        # An UNKNOWN headroom (inf) stays inf -- credit re-ranks the rankable, it never demotes an n
        # whose gain is not yet measurable at all.
        gap = {n: (g * float(min(CRED0, max(CRED_LO, cred.get(n, CRED0))))
                   if np.isfinite(g) else g) for n, g in gap.items()}
    must = [n for n in live if not np.isfinite(gap[n])]
    # cheap-digit admission, judged on the RAW relgap (distance to the cap is a fact about the
    # packing, not about credit) but retired through the SAME credit floor as every other funded n.
    near = [n for n in live
            if np.isfinite(gap[n])
            and 0.0 < _relgap(n, best_s.get(n), records) <= NEARCAP_MULT * CAP_RELGAP
            and float(cred.get(n, CRED0) if cred else CRED0) > CRED_LO]
    must = must + [n for n in near if n not in must]
    rest = sorted((n for n in live if np.isfinite(gap[n]) and n not in must),
                  key=lambda n: (-gap[n], n))
    if not rest and not must:
        return dict(wt)
    cap = sum(wt[n] for n in live) / float(boost)
    keep, cum = list(must), sum(wt[n] for n in must)
    for n in rest:
        keep.append(n)
        cum += wt[n]
        if cum >= cap:
            break
    kset = set(keep)
    return {n: (wt[n] if n in kset else 0.0) for n in wt}

# Basin-repeat control.  A restart that reconverges to the incumbent it was perturbed FROM has
# recomputed a committed answer; it is not exploration, and at the old jitter scale it was 67-95% of
# every restart.  Two mechanisms, both measured in artifacts/probe_iter6c.py:
#   (a) ABORT_TOL/PEEK_IT -- kill such a restart at LP step PEEK_IT instead of ~35 (the peek
#       displacement of a returning trial is 0.000 against >=0.035 for every winner observed);
#   (b) SIG0/SIG_UP/SIG_DN -- drive the per-n jitter scale from the observed repeat rate instead of
#       fixing it, since the scale that repeats 67% of the time at n=45 repeats 95% at n=71.
#       Multiplicative up-on-repeat / down-otherwise settles at a repeat rate of
#       ln(SIG_DN)/(ln(SIG_DN) - ln(SIG_UP)) ~= 15%.
ABORT_TOL = 0.015           # divided by sqrt(n); winners peeked >= 0.035, so ~10x of margin
# PEEK_IT is where the repeat is DECIDED, and the decision is made far earlier than iteration 6
# assumed.  Traced per step (artifacts/probe_iter7b.py, n=45/71): by LP step 3 a returning trial's
# displacement from the incumbent is 0.00000 while every trial that lands elsewhere is already at
# >= 0.027 -- a totally separated pair of distributions, against an abort threshold of ~0.002.
# Peeking at 3 instead of 6 halves what a doomed restart costs, and the shorter coarse tier makes
# that matter more: step 6 of a 9-step restart is two thirds of it.
PEEK_IT = 3

# THE GATE ON A DOOMED RECONVERGENCE.  See slp_run's docstring for the measurement.  Expressed as a
# multiple of the WIDEST drift ball a consumer may legally choose (DFRAC_HI * best/n), so the licence
# is stated in the units of the thing it must not damage: 2.5x that ball is 1.5-1.7x looser than the
# measured zero-false-negative point, and the self-test asserts the margin in BOTH directions (the
# gate must fire, and it must sit strictly outside every consumer's tolerance).
GATE_MULT = 2.5 * 0.10      # = 2.5 * DFRAC_HI (defined below; kept literal to avoid a forward ref)
SIG0, SIG_LO, SIG_HI = 0.20, 0.06, 0.50
SIG_UP, SIG_DN = 1.06, 0.99


# DRIFT: THE SEARCH ORIGIN IS NOT THE BEST PACKING.  Until iteration 9 every restart departed from
# the incumbent and a candidate was adopted only if it BEAT it -- a pure hill-climb *in basin space*,
# so an n whose better basin is not one move away from its incumbent can never be reached, no matter
# how many restarts are spent.  Measured (artifacts/probe_iter9.py): at n=33, 863 greedy restarts --
# 20x that n's whole per-iteration slice -- improved sum_r by EXACTLY 0.0, while the same loop with an
# adoption tolerance walked to the record (relgap 3e-10) in 11 s.  So the origin is decoupled from the
# best: `cur` may be adopted up to DRIFT_TOL below `best`, `best` itself is monotone and is all that is
# ever evaluated/committed.  The tolerance is ANCHORED at best (never `cur`), so the walk is a bounded
# ball around the incumbent, not a random walk that can sink away.
# The tolerance is not a constant: it is a fraction of that n's own MEAN RADIUS (best_s / n), the
# observable that sets the scale of a basin-to-basin difference, so every member of the family tunes
# itself instead of being fed a scale fitted on one n.
#
# BUT A FIXED FRACTION IS STILL A CONSTANT FITTED ON ONE MEMBER.  Measured
# (artifacts/probe_iter10b.py, the real loop instrumented for 8 s per n): at n=33 the fixed 2.5%
# fraction adopts 20% of candidates and the polish tier fires 26 times -- the walk moves and that n
# reached its record.  At n=73/99 the SAME fraction adopts 3-8% and polish fires ZERO times in 8 s:
# candidates land a median 1.2-1.4e-2 below the incumbent while the tolerance is 1.3-1.5e-3, so the
# origin is pinned at the incumbent and iteration 9's mechanism is inert exactly where the census is
# stuck.  Time was not the constraint either (probe_iter10.py: 4x the time bought 0.000 digits at
# n=47/73).  So the fraction is driven by the ONE observable that says whether the walk is moving --
# its own adopt rate -- exactly as SIG is driven by the repeat rate: widen slowly when nothing is
# adopted, narrow fast when everything is.  Equilibrium adopt rate = ln(UP)/(ln(UP) - ln(DN)) ~= 20%,
# the rate measured at the n where the walk works.
# The clamp is NOT widened to make this fit: [DFRAC_LO, DFRAC_HI] stays inside the bracket iteration
# 9 verified (above 10*POLISH_GAP, below 10% of the mean radius), so the controller can only choose a
# tolerance that was already licensed -- it cannot walk the origin out of the incumbent's ball.
DRIFT_FRAC = 0.025          # initial fraction of the mean radius (iteration 9's fixed value)
DFRAC_LO, DFRAC_HI = 0.025, 0.10
DFRAC_UP, DFRAC_DN = 1.010, 0.961      # -> equilibrium adopt rate ~= 0.20


# A TRANSFER MAY WIN THE RECORD, BUT IT MAY NOT SILENTLY RESET THE WALK.  `_adopt` uses closeness in
# sum_r as its proxy for "a different place to search FROM".  For the two LOCAL channels that proxy is
# sound -- a jittered or shaken incumbent is near the incumbent in CONFIGURATION space too, so the
# trajectory's accumulated context (sig, dfrac, and the origin's own history) still applies to it.
# For the cross-n transfer it is exactly wrong: the candidate is built from a DIFFERENT n's packing,
# so a near-equal sum_r says nothing about proximity, and adopting it teleports the origin into an
# alien structure -- which is the mechanism iteration 13 measured killing the escape (3 of 3 -> 0 of 3
# at n=33).  Gating the branch's RATE (XFER_*) bounds how often that happens; this bounds the DAMAGE
# when it does.  The transfer keeps its full path to the record -- it is still scored through
# evaluate(), and a transfer that actually BEATS the incumbent still re-anchors the walk, because
# then the walk's own destination has moved.  It only loses the right to move the origin by being
# merely CLOSE.
XFER_DRIFT = False          # may an unscored cross-n transfer become the drift ORIGIN?


def _drift_ok(kind, xfer_drift=None):
    """May a candidate from channel `kind` move the drift ORIGIN -- and drive its controller?

    One rule, both halves: a channel barred from adopting must also be barred from the
    adopt-rate controller, or each barred trial reads as a MISS and widens `dfrac`, i.e. the
    exclusion would silently retune the very drift ball it exists to protect."""
    xd = XFER_DRIFT if xfer_drift is None else bool(xfer_drift)
    return bool(xd or kind != 1)


def _drift_tol(n, best_s, frac=DRIFT_FRAC):
    """Adoption tolerance for n: a fraction of its own mean radius.  0.0 (no drift) if unknown."""
    if best_s is None or n <= 0:
        return 0.0
    b = float(best_s)
    if not np.isfinite(b) or b <= 0.0:
        return 0.0
    return float(frac) * b / float(n)


def _dfrac_update(frac, adopted):
    """Per-n drift fraction from the observed ADOPT rate: narrow when the origin keeps moving,
    widen when it is pinned.  Clamped to the bracket the drift mechanism was verified inside."""
    return float(min(DFRAC_HI, max(DFRAC_LO, frac * (DFRAC_DN if adopted else DFRAC_UP))))


def _adopt(sc, best_s, cur_s, tol):
    """Adopt a non-improving basin as the new search ORIGIN?  Two-sided: within `tol` of the best
    (the anchor -- the walk can never sink below best - tol) and within `tol` of where it is now."""
    if tol <= 0.0 or not np.isfinite(sc):
        return False
    return sc > best_s - tol and sc > cur_s - tol


def _sig_update(sig, repeated):
    """Per-n jitter scale from the observed repeat rate: grow it when the reconvergence walks back to
    the incumbent, shrink it slowly otherwise.  Clamped, so a pathological n cannot run away."""
    return float(min(SIG_HI, max(SIG_LO, sig * (SIG_UP if repeated else SIG_DN))))


# THE WALK MUST OUTLIVE THE ITERATION.  `cur` (the search ORIGIN), `sig` and `dfrac` are the whole
# state of a per-n trajectory, and measured (artifacts/probe_iter11b.py, probe_iter11c.py) the escape
# probability of that trajectory rises with its CONTIGUOUS length and does NOT split into independent
# restarts.  But solve() rebuilds all three from scratch every iteration: the origin is re-anchored at
# the committed best and the two controllers restart at their defaults, so a walk that took 8 s to
# leave the incumbent's basin is thrown away and re-bought from zero next time.  The census carries the
# monotone record across iterations; nothing carries the non-monotone part.
#
# So persist exactly the non-monotone part, in `notes/` (writable, and NOT the packs dir the driver
# snapshots).  Three properties make this safe rather than a side channel:
#   1. NOTHING IS SCORED FROM IT.  The state holds search origins, never results; every packing that
#      can reach bench/packs/ still goes through the metered evaluate().
#   2. IT CAN NEVER CARRY A BETTER PACKING THAN THE COMMITTED ONE.  On load an origin is REJECTED
#      unless its own sum of radii is <= the committed best for that n -- so an un-evaluated packing
#      cannot be laundered into the next iteration as an "origin", it can only be discarded.
#   3. THE DRIFT BALL SURVIVES THE BOUNDARY.  An origin is also rejected unless it sits inside the
#      widest tolerance the controller is licensed to pick (DFRAC_HI of the mean radius below best),
#      the same bound that keeps the walk anchored WITHIN an iteration.  A stale origin from an older,
#      worse census therefore cannot pull the walk out of the incumbent's neighbourhood.
# Anything missing, malformed, mis-shaped or out of bounds silently falls back to the current
# behaviour (anchor at best, controllers at their defaults), so an offline re-run with no state file
# -- or on sizes that never appear in it -- behaves exactly as before.
STATE_PATH = os.path.join("notes", "solver_state.json")


def read_state(path=STATE_PATH):
    """Load the persisted per-n walk state.  GUARDED: absent/malformed -> {}."""
    try:
        with open(path, "r") as fh:
            raw = json.load(fh)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in (raw.get("walk") or {}).items():
        try:
            n = int(k)
            cur = np.asarray(v["cur"], dtype=float)
            if cur.shape != (n, 3) or not np.all(np.isfinite(cur)):
                continue
            out[n] = {"cur": cur,
                      "sig": float(v.get("sig", SIG0)),
                      "dfrac": float(v.get("dfrac", DRIFT_FRAC))}
        except Exception:
            continue
    return out


def read_cred(path=STATE_PATH):
    """Load the persisted per-n funding credit.  GUARDED: absent/malformed -> {}.

    Kept separate from read_state because credit is not part of a WALK: it survives for an n whose
    origin is the incumbent (nothing to carry) and for an n at the digit cap, and it must never be
    dropped just because that n's trajectory was not worth persisting."""
    try:
        with open(path, "r") as fh:
            raw = json.load(fh)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in (raw.get("cred") or {}).items():
        try:
            c = float(v)
            if np.isfinite(c) and c > 0.0:
                out[int(k)] = float(min(CRED0, max(CRED_LO, c)))
        except Exception:
            continue
    return out


def write_state(cur, cur_s, sig, dfrac, best_s, path=STATE_PATH, cred=None):
    """Persist the walk state.  Only origins that DIFFER from the committed best are worth carrying
    (an origin equal to best is what a cold start already reconstructs).  GUARDED: never raises."""
    walk = {}
    for n, pack in cur.items():
        try:
            b = best_s.get(n)
            s = float(np.asarray(pack)[:, 2].sum())
            if b is None or not np.isfinite(s):
                continue
            if s >= float(b) - 1e-15:        # the origin is the incumbent -- nothing to carry
                continue
            walk[str(n)] = {"cur": [[float(a) for a in row] for row in np.asarray(pack)],
                            "sig": float(sig.get(n, SIG0)),
                            "dfrac": float(dfrac.get(n, DRIFT_FRAC))}
        except Exception:
            continue
    # MERGE, never truncate.  Nothing may have drifted (or there may be no origins at all -- an
    # offline run on fresh sizes), and credit may be updated for n whose walk is not worth carrying:
    # in either case the OTHER half of the file must survive, since a stale entry is re-validated on
    # load anyway and blanking it would erase a walk that took several iterations to buy.
    prev = {}
    try:
        with open(path, "r") as fh:
            prev = json.load(fh)
    except Exception:
        prev = {}
    if not isinstance(prev, dict):
        prev = {}
    keep = dict(prev.get("cred") or {})
    for n, c in (cred or {}).items():
        try:
            keep[str(int(n))] = float(min(CRED0, max(CRED_LO, float(c))))
        except Exception:
            continue
    merged = walk if walk else dict(prev.get("walk") or {})
    if not merged and not keep:
        return {}
    try:
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(path, "w") as fh:
            json.dump({"walk": merged, "cred": keep}, fh)
    except Exception:
        pass
    return walk        # what THIS call contributed -- an undrifted origin still reports nothing


def state_origin_ok(n, cur, best_s):
    """May a persisted origin be resumed for n?  Two-sided, and both sides are load-bearing:

    * `s <= best_s` -- an origin is never allowed to be BETTER than the committed best.  Origins do
      not go through evaluate(), so this is what makes the state file provably not a way to carry an
      un-metered packing forward.
    * `s > best_s - DFRAC_HI * best_s / n` -- the origin must be inside the widest drift ball the
      controller can legally choose, so resuming cannot start the walk anywhere the in-iteration rule
      would refuse to walk to."""
    if best_s is None or n <= 0:
        return False
    b = float(best_s)
    if not np.isfinite(b) or b <= 0.0:
        return False
    a = np.asarray(cur, dtype=float)
    if a.shape != (n, 3) or not np.all(np.isfinite(a)) or a[:, 2].min() <= 0.0:
        return False
    s = float(a[:, 2].sum())
    return s <= b and s > b - DFRAC_HI * b / float(n)


def read_records():
    """The published records, keyed by n -- reading bench/records.json inside solve() is explicitly
    sanctioned by the mission (warm starts / diagnostics).  GUARDED: absent or malformed -> {}."""
    try:
        with open(os.path.join("bench", "records.json"), "r") as fh:
            raw = json.load(fh)
        rec = raw.get("records", raw)
        return {int(k): float(v) for k, v in rec.items()}
    except Exception:
        return {}


def _at_cap(n, s, records):
    """True iff n is already inside the scorer's 7-digit cap, so more search on it earns nothing.
    Unknown n (offline re-run on sizes outside the census) is never capped."""
    rec = records.get(n)
    if rec is None or s is None or rec <= 0.0:
        return False
    return (rec - s) / rec <= CAP_RELGAP


def solve(evaluate, meter, rng, targets, state_path=STATE_PATH):
    """Population-coupled basin hopping.

    The 37 census packings are NOT 37 independent searches: a good packing for n is a good start for
    n+-2 once you delete the smallest circles or fill the largest pockets.  So the run keeps a pool of
    the best pack per n and sweeps the census several times, each sweep able to pull structure that a
    previous sweep discovered at a neighbouring n.
    """
    if not targets:
        return
    t_start = time.process_time()
    ns = sorted(targets)
    pool = {}
    sig = {n: SIG0 for n in ns}          # per-n jitter scale, driven by that n's own repeat rate
    dfrac = {n: DRIFT_FRAC for n in ns}  # per-n drift fraction, driven by that n's own adopt rate
    pxfer = {n: XFER_P0 for n in ns}     # per-n transfer probability, driven by its own threat rate
    pfield = {n: FIELD_P0 for n in ns}   # per-n smooth-field share, driven by its own adopt rate
    afr = {n: AFR_INIT for n in ns}      # per-n PER-MODE amplitude, by its own repeat rate
    fw = {n: FW0 for n in ns}            # per-n weight over the field family, by its adopt rate
    cred = {n: CRED0 for n in ns}        # per-n funding credit, driven by its own improvement rate

    # seed the pool from the committed census (guarded) and re-offer it so no n can regress
    for n in ns:
        warm = read_pack(n, "")
        if warm is not None:
            feas, s = evaluate(n, warm)
            if feas:
                pool[n] = warm.copy()
    best_s = {n: float(pool[n][:, 2].sum()) for n in pool}
    cur = {n: pool[n].copy() for n in pool}   # SEARCH ORIGIN per n -- may lag `pool` by up to
    cur_s = dict(best_s)                     # _drift_tol(n); `pool`/`best_s` stay monotone.
    records = read_records()

    # RESUME THE WALK.  The committed census restores the monotone half of the state; this restores
    # the half that used to die at the iteration boundary.  Every resumed origin must pass
    # state_origin_ok (not better than the committed best, and inside the widest licensed drift ball)
    # -- anything else is dropped and that n simply starts anchored at its incumbent, as before.
    # Credit resumes independently of the walk: it is a fact about how an n has REPAID funding, which
    # outlives any particular origin, and it is what stops each iteration from re-buying the same
    # unproductive prefix from scratch.
    for n, c in read_cred(state_path).items():
        if n in cred:
            cred[n] = float(min(CRED0, max(CRED_LO, c)))

    resumed = 0
    for n, st in read_state(state_path).items():
        if n in pool and state_origin_ok(n, st["cur"], best_s.get(n)):
            cur[n] = np.asarray(st["cur"], dtype=float).copy()
            cur_s[n] = float(cur[n][:, 2].sum())
            sig[n] = float(min(SIG_HI, max(SIG_LO, st["sig"])))
            dfrac[n] = float(min(DFRAC_HI, max(DFRAC_LO, st["dfrac"])))
            resumed += 1

    for p in range(PASSES):
        pass_end = t_start + CPU_BUDGET * (p + 1) / PASSES
        # Spend by MARGINAL value, not uniformly: an n already within CAP_RELGAP of its record is
        # pinned at the 7-digit cap and cannot earn another digit, so it gets no time.  Weight is n
        # (the measured cost of one restart) for every n that can still move.
        wt = {n: (0.0 if _at_cap(n, best_s.get(n), records) else float(n)) for n in ns}
        if not any(wt.values()):
            wt = {n: float(n) for n in ns}
        # ...and among those, only the worst-relgap prefix, so each focused n gets a slice long
        # enough for its trajectory to escape rather than 25 sub-threshold draws.
        wt = _focus(wt, best_s, records, cred=cred)
        wleft = float(sum(wt.values()))
        for n in ns:
            if meter.left() <= 4:
                return
            now = time.process_time()
            remain = pass_end - now
            wleft -= wt[n]
            if remain <= 0.15 or wt[n] <= 0.0:
                continue
            deadline = now + remain * (wt[n] / max(wleft + wt[n], 1e-9))

            s_in, trials = best_s.get(n, -np.inf), 0
            atol = ABORT_TOL / np.sqrt(n)
            while time.process_time() < deadline and meter.left() > 4:
                trials += 1
                kind, ref, fmode = 0, None, -1
                dtol = _drift_tol(n, best_s.get(n), dfrac[n])
                if n not in pool:                     # cold start: must work for ANY n
                    src = _transfer(pool.get(n - 2, pool.get(n + 2)), n) if (
                        (n - 2) in pool or (n + 2) in pool) else None
                    xy = src if src is not None else _cold_start(n, rng, rng.randint(3))
                    xy, d0 = np.clip(xy, LO + 1e-3, HI - 1e-3), None
                else:
                    # perturb the ORIGIN, which may be a basin adopted below the best (see DRIFT).
                    xy, d0, kind, ref_ok, fmode = _move(cur[n], rng, n, pool, sig[n], pxfer[n],
                                                        pfield[n], afr[n], fw[n])
                    if ref_ok:
                        ref = cur[n][:, :2]
                # TIER 0 -- basin-repeat abort: a reconvergence that is walking back to the
                # incumbent cannot improve on it, and is decidable at LP step PEEK_IT.
                # TIER 1 -- coarse: enough to RANK the basin and no more.  It stops at
                # STOP_EPS (a tenth of what the filter below can resolve), not at
                # convergence -- monotone, so it can only under-report, never over-report.
                bs = best_s.get(n)
                gate = (bs - GATE_MULT * bs / float(n)) if bs is not None and np.isfinite(bs) \
                    else None
                xy, r, code = slp_run(xy, deadline, delta0=d0, dmin=COARSE_DMIN,
                                      ref=ref, ref_tol=atol, peek_it=PEEK_IT,
                                      stop_eps=STOP_EPS, gate=gate)
                repeated = (code == 1) or (ref is not None and np.abs(xy - ref).max() < atol)
                if kind == 3:                         # only the jitter channel drives its own scale
                    sig[n] = _sig_update(sig[n], repeated)
                elif kind == 4:      # the field drives its own amplitude, same law, PER MODE
                    afr[n] = _afr_set(afr[n], fmode,
                                      _afr_update(_afr_get(afr[n], fmode), repeated))
                if code:
                    # A GATED candidate is a drift MISS, not a missing trial: it was rejected on the
                    # same criterion the drift controller reads, so it must still charge that
                    # controller.  Otherwise the gate would silently retune `dfrac` by removing the
                    # majority of its negative evidence -- an accelerator that moves a decision rule.
                    if code == 2 and n in cur and _drift_ok(kind):
                        dfrac[n] = _dfrac_update(dfrac[n], False)
                        if kind == 4:                 # ...and a miss for the channel's own share
                            pfield[n] = _field_update(pfield[n], False)
                            fw[n] = _fw_update(fw[n], fmode, False)   # ...and for the MEMBER used
                    continue
                sc = float(r.sum())
                threat = sc > best_s.get(n, -np.inf) - POLISH_GAP
                if kind == 1:      # the transfer channel pays for itself out of its own threat rate
                    pxfer[n] = _xfer_update(pxfer[n], threat)
                if threat:
                    # TIER 2 -- polish only a candidate that threatens the incumbent.  Continues from
                    # the coarse point (same basin), so the work already done is not repeated.
                    xy2, r2 = slp_optimise(xy, deadline, delta0=POLISH_DELTA0, dmin=1e-12, r0=r)
                    pack = np.concatenate([xy2, repair(xy2, r2)[:, None]], axis=1)
                    feas, s = evaluate(n, pack)
                    if feas and float(s) > best_s.get(n, -np.inf):
                        pool[n], best_s[n] = pack.copy(), float(s)
                        cur[n], cur_s[n] = pack.copy(), float(s)   # a new best re-anchors the walk
                        if kind == 4:   # the strongest possible evidence for the channel: it MOVED
                            pfield[n] = _field_update(pfield[n], True)
                            fw[n] = _fw_update(fw[n], fmode, True)
                        continue
                # TIER 3 -- DRIFT.  A basin that does not beat the incumbent is still worth standing
                # in if it is close: it is a different place to search FROM, and 863 restarts from
                # the one incumbent were measured to find nothing at all.  Nothing here is evaluated
                # or committed -- only the origin moves, inside a ball of radius dtol around best.
                adopted = (n in cur and _drift_ok(kind)
                           and _adopt(sc, best_s.get(n, -np.inf),
                                      cur_s.get(n, -np.inf), dtol))
                if adopted:
                    cur[n] = np.concatenate([xy, repair(xy, r)[:, None]], axis=1)
                    cur_s[n] = sc
                if n in cur and _drift_ok(kind):
                    dfrac[n] = _dfrac_update(dfrac[n], adopted)
                    if kind == 4:
                        pfield[n] = _field_update(pfield[n], adopted)
                        fw[n] = _fw_update(fw[n], fmode, adopted)
            # The slice is over: charge it to this n's credit.  Only a slice that actually RAN is
            # judged (a deadline that expired before the first restart is not evidence about n), and
            # the verdict is the one the SCORE reads -- did the committed best for n move at all.
            if trials > 0:
                cred[n] = _cred_update(cred[n], best_s.get(n, -np.inf) > s_in)
        # checkpoint the walk each pass, so a run cut short by the CPU backstop still hands the next
        # iteration whatever trajectory it had reached.
        write_state(cur, cur_s, sig, dfrac, best_s, path=state_path, cred=cred)
    write_state(cur, cur_s, sig, dfrac, best_s, path=state_path, cred=cred)


# ---------------------------------------------------------------- self-test

class _StubHarness(object):
    """A local re-implementation of the metered evaluate() -- used only by --self-test / dev probes,
    never by the driver (which supplies the authoritative one)."""

    def __init__(self, budget=500000):
        self.budget = budget
        self.used = 0
        self.best = {}

    def left(self):
        return self.budget - self.used

    def evaluate(self, n, packing):
        a = np.asarray(packing, float)
        single = a.ndim == 2
        if single:
            a = a[None]
        self.used += a.shape[0]
        x, y, r = a[..., 0], a[..., 1], a[..., 2]
        wall = np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min(axis=1)
        d = np.sqrt((x[:, :, None] - x[:, None, :]) ** 2 + (y[:, :, None] - y[:, None, :]) ** 2)
        sl = d - (r[:, :, None] + r[:, None, :])
        for k in range(a.shape[0]):
            np.fill_diagonal(sl[k], np.inf)
        feas = (r.min(axis=1) > 0) & (wall >= -1e-9) & \
               (sl.reshape(a.shape[0], -1).min(axis=1) >= -1e-9)
        s = r.sum(axis=1)
        for k in range(a.shape[0]):
            if feas[k] and s[k] > self.best.get(n, (-1.0, None))[0]:
                self.best[n] = (float(s[k]), a[k].copy())
        if single:
            return bool(feas[0]), float(s[0])
        return feas, s


def _self_test():
    import math
    import sys
    rng = np.random.RandomState(0)
    ok = True

    # 1. geometry: repair() must always produce a strictly feasible packing
    for n in (5, 17, 40):
        xy = rng.uniform(LO + 0.02, HI - 0.02, size=(n, 2))
        r = repair(xy, lp_radii(xy))
        wall = _walls(xy) - r
        dist = _pdist(xy)
        np.fill_diagonal(dist, np.inf)
        pair = (dist - (r[:, None] + r[None, :])).min()
        assert r.min() > 0, "non-positive radius"
        assert wall.min() >= -1e-9, "wall violation %g" % wall.min()
        assert pair >= -1e-9, "pair violation %g" % pair
        print("feasible n=%d sum_r=%.6f wall=%.2e pair=%.2e" % (n, r.sum(), wall.min(), pair))

    # 2. the SLP under-estimator claim: every LP step must land TRULY feasible before repair(),
    #    and the objective must never go down.
    for n in (12, 31):
        xy = np.clip(_cold_start(n, rng, 0), LO + 1e-3, HI - 1e-3)
        r = lp_radii(xy)
        s_prev = r.sum()
        delta = 0.2 / np.sqrt(n)
        for _ in range(25):
            out = _slp_step(xy, r, delta, grow=delta)
            assert out is not None, "LP failed"
            xy, r, s = out
            dd = _pdist(xy)
            np.fill_diagonal(dd, np.inf)
            v = (r[:, None] + r[None, :] - dd).max()
            w = (r - _walls(xy)).max()
            # the under-estimator claim is exact in real arithmetic; what is left is HiGHS' own
            # primal tolerance (1e-10), so the bar here is 1e-8, and strict feasibility is asserted
            # on the REPAIRED output below -- that is what the harness actually validates.
            assert v <= 1e-8, "SLP step produced overlap %g (under-estimator claim broken)" % v
            assert w <= 1e-8, "SLP step left the square by %g" % w
            assert s >= s_prev - 1e-12, "SLP not monotone: %.15f -> %.15f" % (s_prev, s)
            s_prev = s
            delta *= 0.6
        rr = repair(xy, r)
        dd = _pdist(xy)
        np.fill_diagonal(dd, np.inf)
        assert (rr[:, None] + rr[None, :] - dd).max() <= -0.0, "repaired overlap"
        assert (rr - _walls(xy)).max() <= -0.0, "repaired wall violation"
        assert s_prev - rr.sum() < 1e-9, "repair cost %.3e is too large" % (s_prev - rr.sum())
        print("slp monotone n=%d lp_sum=%.12f repaired=%.12f (loss %.1e)"
              % (n, s_prev, rr.sum(), s_prev - rr.sum()))

    # 2b. pockets: every returned pocket must have STRICTLY POSITIVE clearance (a circle of that
    #     radius fits there), and _transfer must always return exactly n centres.
    for n in (9, 31):
        xy = np.clip(_cold_start(n, rng, 0), LO + 1e-3, HI - 1e-3)
        r = repair(xy, lp_radii(xy))
        pk = _pockets(xy, r, 5)
        assert len(pk) > 0, "no pocket found for n=%d" % n
        cl = np.minimum(
            (np.sqrt(((pk[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]).min(1),
            np.minimum.reduce([pk[:, 0] - LO, HI - pk[:, 0], pk[:, 1] - LO, HI - pk[:, 1]]))
        assert cl.min() > 0, "pocket with non-positive clearance %g" % cl.min()
        print("pockets n=%d k=%d best_clearance=%.6f (max r=%.6f)" % (n, len(pk), cl.max(), r.max()))
        src_pack = np.concatenate([xy, r[:, None]], axis=1)
        for m in (n - 4, n - 1, n, n + 1, n + 5):
            t = _transfer(src_pack, m)
            assert t is not None and t.shape == (m, 2), "transfer %d->%d gave %s" % (n, m, t)
        assert _transfer(None, n) is None

    # 2o. THE LP BACKEND IS AN ACCELERATOR, NOT A SECOND SOURCE OF TRUTH.  The raw HiGHS path exists
    #     only to remove per-call plumbing and to reuse the simplex basis; if it ever returns a
    #     DIFFERENT optimum than `linprog`, the speedup is a different algorithm wearing the same
    #     name.  So assert, on the LPs the solver actually issues, that (i) both paths agree on the
    #     objective to LP tolerance, (ii) the raw path is genuinely live (a backend silently falling
    #     back every call is a mechanism never built), and (iii) killing the raw path still solves --
    #     the fallback is load-bearing on any scipy without the vendored HiGHS.
    _rsb = np.random.RandomState(101)
    _raw_live = 0
    for n in (27, 45, 71):
        _xy = _rsb.uniform(-0.45, 0.45, size=(n, 2))
        _r0 = lp_radii(_xy)
        _wall = np.maximum(_walls(_xy) - FEAS_EPS, 0.0)
        _dist = _pdist(_xy)
        _iu = np.triu_indices(n, 1)
        _dv = _dist[_iu]
        _kp = _dv < (_wall[_iu[0]] + _wall[_iu[1]])
        _I, _J, _dv = _iu[0][_kp], _iu[1][_kp], _dv[_kp]
        _k = len(_I)
        _A = coo_matrix((np.ones(2 * _k),
                         (np.concatenate([np.arange(_k)] * 2), np.concatenate([_I, _J]))),
                        shape=(_k, n))
        _c = -np.ones(n)
        _xr = _HS.solve(_c, _A, _dv - FEAS_EPS, np.zeros(n), _wall)
        _res = linprog(_c, A_ub=_A, b_ub=_dv - FEAS_EPS,
                       bounds=np.stack([np.zeros(n), _wall], axis=1),
                       method="highs", options=_LP_OPT)
        if _xr is not None:
            _raw_live += 1
            assert abs(float(_c @ _xr) - float(_c @ _res.x)) < 1e-7, \
                "n=%d: raw HiGHS and linprog disagree on the LP optimum" % n
        # (iii) with the raw path dead the solver must still produce the same optimum
        _sv = _HS.solve
        try:
            _HS.solve = lambda *a, **kw: None
            _rf = lp_radii(_xy)
        finally:
            _HS.solve = _sv
        assert abs(float(_rf.sum()) - float(_r0.sum())) < 1e-7, \
            "n=%d: the linprog fallback does not reproduce the backend's optimum" % n
        # and the SLP built on it must still be monotone and strictly repairable
        _p2, _r2 = slp_optimise(_xy, time.process_time() + 3.0, dmin=1e-6)
        _rr = repair(_p2, _r2)
        assert _rr.min() > 0.0 and float(_rr.sum()) >= float(_r0.sum()) - 1e-9, \
            "n=%d: SLP on the raw backend is not monotone/feasible" % n
    assert _raw_live == 3 or _hs is None, \
        "the raw HiGHS backend fell back to linprog on every LP -- it is not actually installed"
    # a basis is only ever a STARTING point: feeding a deliberately WRONG one must not change the
    # optimum (this is what licenses reusing a basis across a changed LP).
    if _hs is not None and _HS.basis is not None:
        _n = 33
        _xy = _rsb.uniform(-0.45, 0.45, size=(_n, 2))
        _good = float(lp_radii(_xy).sum())
        _HS.shape = None                       # force a cold solve for reference
        _cold = float(lp_radii(_xy).sum())
        assert abs(_good - _cold) < 1e-9, "a reused basis changed the LP optimum"

    # scratch dir for the state round-trip below (created here, removed at the end of 2k)
    _td = os.path.join("notes", "_selftest_tmp")
    if not os.path.isdir(_td):
        os.makedirs(_td)

    # 2k. THE PERSISTED WALK, asserted exactly where its licence is spent.  Carrying the search
    #     origin across the iteration boundary is safe only because of two rejections, so assert
    #     both, in both directions -- a later edit that drops either fails here rather than quietly
    #     turning notes/ into an unmetered side channel for packings.
    _rs = np.random.RandomState(7)
    for n in (27, 63, 99):
        _xy = _rs.uniform(-0.45, 0.45, size=(n, 2))
        _r = repair(_xy, lp_radii(_xy))
        _pk = np.concatenate([_xy, _r[:, None]], axis=1)
        _b = float(_r.sum())
        assert state_origin_ok(n, _pk, _b), "n=%d: an origin equal to best was rejected" % n
        # (i) never BETTER than the committed best -- this is what makes the file provably unable to
        #     launder an un-evaluated packing forward.
        _up = _pk.copy(); _up[0, 2] += 1e-9
        assert not state_origin_ok(n, _up, _b), \
            "n=%d: an origin BETTER than the committed best was accepted" % n
        # (ii) inside the widest drift ball the controller may pick -- so resuming can never start
        #      the walk somewhere the in-iteration adoption rule would refuse to walk to.
        _ball = DFRAC_HI * _b / float(n)
        # shrink the LARGEST circle, not circle 0: the max-sum LP is degenerate on a random point
        # set (several optimal vertices with the same sum), so which index carries a ZERO radius is
        # a property of the LP backend, not of the rule under test.  Perturbing the largest radius
        # asserts exactly the same two-sided property without depending on that choice.
        _big = int(np.argmax(_pk[:, 2]))
        assert _pk[_big, 2] > 2.5 * _ball, "n=%d: self-test needs a radius bigger than the ball" % n
        _in = _pk.copy(); _in[_big, 2] -= 0.5 * _ball
        _out = _pk.copy(); _out[_big, 2] -= 2.0 * _ball
        assert state_origin_ok(n, _in, _b), "n=%d: an origin well inside the ball was rejected" % n
        assert not state_origin_ok(n, _out, _b), \
            "n=%d: an origin outside the licensed drift ball was resumed" % n
        assert not state_origin_ok(n, _pk[:-1], _b), "n=%d: a mis-shaped origin was accepted" % n
        assert not state_origin_ok(n, _pk, None), "n=%d: an origin with no incumbent was accepted" % n
        # (iii) exact round trip through the file, and a drifted origin is what actually gets stored.
        _tmp = os.path.join(_td, "state_%d.json" % n)
        _dr = _pk.copy(); _dr[0, 2] -= 0.5 * _ball
        _w = write_state({n: _dr}, {n: float(_dr[:, 2].sum())}, {n: 0.31}, {n: 0.041},
                         {n: _b}, path=_tmp)
        assert str(n) in _w, "n=%d: a drifted origin was not persisted" % n
        _ld = read_state(_tmp)
        assert n in _ld and np.abs(_ld[n]["cur"] - _dr).max() == 0.0, \
            "n=%d: the persisted origin did not round-trip exactly" % n
        assert abs(_ld[n]["sig"] - 0.31) < 1e-15 and abs(_ld[n]["dfrac"] - 0.041) < 1e-15
        # an UNdrifted origin (== the incumbent) is not worth carrying and must not be written
        assert write_state({n: _pk}, {n: _b}, {n: 0.2}, {n: 0.025}, {n: _b}, path=_tmp) == {}, \
            "n=%d: an origin identical to the incumbent was persisted anyway" % n
    # (iv) absent / malformed / wrong-n state must degrade to "no state", never raise -- this is the
    #      offline-re-run path (a fresh checkout, or sizes that never appear in the file).
    assert read_state(os.path.join(_td, "nope.json")) == {}
    _bad = os.path.join(_td, "bad.json")
    open(_bad, "w").write("{not json")
    assert read_state(_bad) == {}
    open(_bad, "w").write('{"walk": {"31": {"cur": [[0.0, 0.0, 0.1]], "sig": 0.2}}}')
    assert read_state(_bad) == {}, "a state row whose shape contradicts its n was loaded"
    open(_bad, "w").write('{"walk": {"x": {"cur": 3}}, "junk": 1}')
    assert read_state(_bad) == {}
    for _f in os.listdir(_td):
        os.remove(os.path.join(_td, _f))
    os.rmdir(_td)
    print("persisted walk: round-trip exact, rejects better-than-best / outside-ball / mis-shaped, "
          "absent+malformed -> no state")

    # 2j. CONCENTRATION -- assert the margin in BOTH directions, so a later edit that erases it
    #     fails loudly.  Too weak a focus is a no-op (the whole point is a slice long enough for one
    #     trajectory to escape); too strong and the census stops being swept at all.
    _recs = {n: 1.0 for n in range(27, 100, 2)}
    _wt = {n: float(n) for n in range(27, 100, 2)}
    _bs = {n: 1.0 - 1e-3 * ((n % 7) + 1) for n in _wt}          # synthetic, monotone in relgap
    _f = _focus(_wt, _bs, _recs)
    _keep = [n for n in _f if _f[n] > 0]
    _drop = [n for n in _f if _f[n] == 0]
    assert _keep and _drop, "focus kept everything or nothing"
    _g = lambda n: _relgap(n, _bs.get(n), _recs)
    assert min(_g(n) for n in _keep) >= max(_g(n) for n in _drop) - 1e-15, "focus is not headroom-ordered"
    _tot, _kw = sum(_wt.values()), sum(_f[n] for n in _keep)
    _boost = _tot / _kw
    assert _boost >= 2.0, "focus boost %.2f is too weak to cross the escape threshold" % _boost
    assert _boost <= FOCUS_BOOST + 1e-9, "focus boost %.2f exceeds its own licence" % _boost
    assert len(_keep) >= 3, "focus narrowed to %d n -- the census is no longer swept" % len(_keep)
    #     boost=1 must be an exact no-op, and a size with NO record must sort first (offline re-runs).
    assert _focus(_wt, _bs, _recs, boost=1.0) == _wt, "boost=1 is not a no-op"
    _wt2 = dict(_wt); _wt2[26] = 26.0
    assert _focus(_wt2, _bs, _recs)[26] > 0, "an n with no record was starved by the census"
    #     an n with NO incumbent pack (it would score 0) must always be kept, and when nothing is
    #     rankable the rule must be an exact no-op.
    _bs4 = dict(_bs); _bs4.pop(99)
    assert _focus(_wt, _bs4, _recs)[99] > 0, "an n with no packing at all was starved"
    assert _focus(_wt, {}, {}) == _wt, "focus is not a no-op when no headroom is known"
    #     a zero-weight (capped) n can never be revived by the focus rule.
    _wt3 = dict(_wt); _wt3[99] = 0.0
    assert _focus(_wt3, _bs, _recs)[99] == 0.0, "focus revived a capped n"

    # 2q. THE DOOMED-BASIN GATE -- an ABORT is a decision to discard work, so the guard is two-sided:
    #     it must FIRE (a gate sized below every real landing is a mechanism never built), it must be
    #     an EXACT no-op when off, and it must sit strictly OUTSIDE every consumer's tolerance, so a
    #     candidate the polish filter or the drift ball could still use can never be gated.
    assert GATE_MULT > 2.0 * DFRAC_HI, (
        "GATE_MULT=%g is not strictly looser than the widest legal drift ball" % GATE_MULT)
    _gn = 33
    _gw = read_pack(_gn, "")
    if _gw is not None:
        _gb = float(_gw[:, 2].sum())
        _gv = _gb - GATE_MULT * _gb / float(_gn)
        # margin, both directions: the gate floor must be far below anything a consumer accepts...
        assert _gv < _gb - _drift_tol(_gn, _gb, DFRAC_HI) - 1e-9, "gate can kill a driftable basin"
        assert _gv < _gb - POLISH_GAP, "gate can kill a polishable candidate"
        # ...and far above nothing: a gate at or below zero would never fire at all.
        assert _gv > 0.5 * _gb, "gate floor is so low it can never fire"
        _grng = np.random.RandomState(909)
        _dl = time.process_time() + 12.0
        # (i) OFF is an exact no-op -- same code path, same answer, gate=None vs a floor no run can
        #     breach.  Equality is on the COORDINATES, not just the sum (the LP is degenerate).
        _gxy = np.clip(_gw[:, :2] + _grng.normal(0, 0.25 / np.sqrt(_gn), (_gn, 2)),
                       LO + 1e-3, HI - 1e-3)
        _a1, _r1, _c1 = slp_run(_gxy.copy(), _dl, dmin=COARSE_DMIN, peek_it=PEEK_IT,
                                stop_eps=STOP_EPS, gate=None)
        _a2, _r2, _c2 = slp_run(_gxy.copy(), _dl, dmin=COARSE_DMIN, peek_it=PEEK_IT,
                                stop_eps=STOP_EPS, gate=-1e9)
        assert _c1 == 0 and _c2 == 0, "gate fired on a floor nothing can breach (%d,%d)" % (_c1, _c2)
        assert np.abs(_a1 - _a2).max() == 0.0, "the gate parameter perturbed the no-gate path"
        # (ii) it FIRES: a floor just under the incumbent kills a landing that ends below it, and the
        #      returned code is the GATE code, never the repeat code (their controller signs differ).
        _fired = _seen = 0
        for _ in range(24):
            _q = np.clip(_gw[:, :2] + _grng.normal(0, 0.30 / np.sqrt(_gn), (_gn, 2)),
                         LO + 1e-3, HI - 1e-3)
            _, _rr, _cc = slp_run(_q.copy(), _dl, dmin=COARSE_DMIN, ref=_gw[:, :2],
                                  ref_tol=ABORT_TOL / np.sqrt(_gn), peek_it=PEEK_IT,
                                  stop_eps=STOP_EPS, gate=_gb - 1e-4)
            _seen += 1
            if _cc == 2:
                _fired += 1
                # monotone: a gated run's own sum is genuinely below the floor it was killed on
                assert float(_rr.sum()) < _gb - 1e-4 + 1e-12, "gate fired above its own floor"
        assert _fired >= _seen // 2, ("the gate fired %d/%d times on landings it must kill"
                                      % (_fired, _seen))
        # (iii) a run gated at the SHIPPED floor must never be one the polish filter would have
        #       wanted: replay each kill to convergence and check it does not reach the incumbent.
        _fn = 0
        for _ in range(12):
            _q = np.clip(_gw[:, :2] + _grng.normal(0, 0.30 / np.sqrt(_gn), (_gn, 2)),
                         LO + 1e-3, HI - 1e-3)
            _, _rr, _cc = slp_run(_q.copy(), _dl, dmin=COARSE_DMIN, peek_it=PEEK_IT,
                                  stop_eps=STOP_EPS, gate=_gv)
            if _cc != 2:
                continue
            _fx, _fr, _ = slp_run(_q.copy(), _dl, dmin=COARSE_DMIN, peek_it=PEEK_IT,
                                  stop_eps=STOP_EPS, gate=None)
            if float(_fr.sum()) > _gb - POLISH_GAP:
                _fn += 1
        assert _fn == 0, "the shipped gate discarded %d candidate(s) the polish filter wanted" % _fn
        print("doomed-basin gate: mult %.3f (> %.3f widest ball), fired %d/%d, 0 false negatives"
              % (GATE_MULT, DFRAC_HI, _fired, _seen))

    # 2p. NEAR-CAP ADMISSION -- the cheapest digits on the board rank LAST by prize, so the guard is
    #     two-sided: the rule must actually FIRE (a near-cap n outside the ranked prefix gets funded)
    #     and must still RETIRE (a near-cap n whose credit hit the floor is not re-bought forever);
    #     and with no near-cap n present it must reproduce the old rule EXACTLY.
    _bs5 = dict(_bs)
    _nc = 65
    _bs5[_nc] = 1.0 - 5.0 * CAP_RELGAP                      # one decade from the cap: relgap 5e-7
    assert 0.0 < _relgap(_nc, _bs5[_nc], _recs) <= NEARCAP_MULT * CAP_RELGAP
    assert min(_relgap(n, _bs5.get(n), _recs) for n in _wt if n != _nc) \
        > _relgap(_nc, _bs5[_nc], _recs), "fixture broken: the near-cap n is not ranked last by prize"
    # the retired arm IS the pure-prize outcome for this n, so it doubles as the baseline
    _cr = {n: CRED0 for n in _wt}; _cr[_nc] = CRED_LO
    _fr = _focus(_wt, _bs5, _recs, cred=_cr)
    assert _fr[_nc] == 0.0, "a retired near-cap n is re-bought forever"
    _fn = _focus(_wt, _bs5, _recs, cred={n: CRED0 for n in _wt})
    assert _fn[_nc] > 0.0, "near-cap admission never fires -- a mechanism too small to fire"
    #     admission spends out of the SAME 1/boost cap as everyone else: the funded weight may
    #     exceed it only by the loop's own granularity (one member), and the boost licence holds.
    _capw = sum(_wt.values()) / FOCUS_BOOST
    _kw2 = sum(v for v in _fn.values() if v > 0.0)
    assert _kw2 <= _capw + max(_wt.values()) + 1e-9, \
        "near-cap admission invented budget instead of spending the focus cap"
    assert sum(_wt.values()) / _kw2 >= 2.0, "near-cap admission diluted focus below the escape bar"
    #     exactness: no near-cap n present -> byte-identical to the pure-headroom/credit rule.
    assert min(_relgap(n, _bs.get(n), _recs) for n in _wt) > NEARCAP_MULT * CAP_RELGAP
    assert _focus(_wt, _bs, _recs, cred={n: CRED0 for n in _wt}) == _focus(_wt, _bs, _recs), \
        "near-cap admission perturbed a census with no near-cap n"
    #     an n AT the cap (relgap 0, weight already zeroed) must not be resurrected by admission.
    _bs6 = dict(_bs5); _bs6[97] = 1.0
    _wt4 = dict(_wt); _wt4[97] = 0.0
    assert _focus(_wt4, _bs6, _recs, cred={n: CRED0 for n in _wt})[97] == 0.0, \
        "near-cap admission revived an at-cap n"
    print("near-cap admission: fires, moves (not invents) budget, retires at the credit floor, "
          "exact no-op without a near-cap n")
    print("focus keep=%d drop=%d boost=%.2fx (licence %.2fx)" % (len(_keep), len(_drop), _boost, FOCUS_BOOST))

    # 2n. THE FUNDING CREDIT, asserted two-sided on exactly the licence it takes.  A law that can
    #     only shrink is a disguised deletion, and a law that can grow past its cap is a way to
    #     invent priority headroom did not earn -- so assert BOTH ends, plus the no-op.
    _c = CRED0
    for _ in range(60):
        _c = _cred_update(_c, False)
    assert abs(_c - CRED_LO) < 1e-12, "credit did not decay to its floor (%g)" % _c
    assert _c > 0.0, "credit reached zero -- an n was deleted, not retired"
    for _ in range(60):
        _c = _cred_update(_c, True)
    assert abs(_c - CRED0) < 1e-12, "credit did not recover to its cap under pure improvement"
    assert _cred_update(CRED0, True) <= CRED0 + 1e-15, "credit exceeded its own cap"
    #     (i) uniform credit -- and no credit at all -- must reproduce the pure-headroom rule EXACTLY.
    assert _focus(_wt, _bs, _recs, cred={n: CRED0 for n in _wt}) == _focus(_wt, _bs, _recs), \
        "uniform credit is not a no-op"
    assert _focus(_wt, _bs, _recs, cred={}) == _focus(_wt, _bs, _recs), "empty credit is not a no-op"
    #     (ii) the mechanism must actually FIRE: decaying the funded prefix must hand the slice to an
    #          n that headroom alone had starved (a mechanism too small to fire is one never built).
    _c0 = {n: CRED0 for n in _wt}
    for n in _keep:
        _c0[n] = CRED_LO
    _f2 = _focus(_wt, _bs, _recs, cred=_c0)
    _keep2 = set(n for n in _f2 if _f2[n] > 0)
    assert _keep2 - set(_keep), "credit decay funded nobody new -- the rotation cannot happen"
    assert set(_keep) - _keep2, "credit decay dropped nobody -- the prefix never rotates"
    #     (iii) credit may not promote an n whose headroom is UNKNOWN out of the must-keep set, and
    #           may not revive a capped (zero-weight) n.
    assert _focus(_wt2, _bs, _recs, cred={26: CRED_LO})[26] > 0, "credit starved an n with no record"
    assert _focus(_wt3, _bs, _recs, cred={99: CRED0})[99] == 0.0, "credit revived a capped n"
    #     (iv) credit round-trips through the state file, survives a call that carries NO walk, and
    #          such a call must not blank a walk that is already stored.
    _td2 = os.path.join("notes", "_selftest_cred")
    if not os.path.isdir(_td2):
        os.makedirs(_td2)
    _tmp2 = os.path.join(_td2, "s.json")
    _n = 31
    _xy2 = np.random.RandomState(3).uniform(-0.45, 0.45, size=(_n, 2))
    _r2 = repair(_xy2, lp_radii(_xy2))
    _pk2 = np.concatenate([_xy2, _r2[:, None]], axis=1)
    _b2 = float(_r2.sum())
    _dr2 = _pk2.copy(); _dr2[0, 2] -= 0.5 * DFRAC_HI * _b2 / _n
    write_state({_n: _dr2}, {_n: float(_dr2[:, 2].sum())}, {_n: 0.2}, {_n: 0.03}, {_n: _b2},
                path=_tmp2, cred={_n: 0.36})
    assert abs(read_cred(_tmp2).get(_n, -1) - 0.36) < 1e-12, "credit did not round-trip"
    write_state({_n: _pk2}, {_n: _b2}, {_n: 0.2}, {_n: 0.03}, {_n: _b2},
                path=_tmp2, cred={_n: 0.12})          # no walk to carry, credit only
    assert abs(read_cred(_tmp2).get(_n, -1) - 0.12) < 1e-12, "a walk-free call lost the credit"
    assert _n in read_state(_tmp2), "a walk-free call blanked the stored walk"
    assert read_cred(os.path.join(_td2, "nope.json")) == {}, "missing credit file did not degrade"
    open(_tmp2, "w").write('{"cred": {"x": "y", "31": -1}}')
    assert read_cred(_tmp2) == {}, "malformed credit was loaded"
    for _f in os.listdir(_td2):
        os.remove(os.path.join(_td2, _f))
    os.rmdir(_td2)
    print("funding credit: floor %.2f cap %.2f, uniform = no-op, rotation adds %d / drops %d n, "
          "round-trips and survives a walk-free write"
          % (CRED_LO, CRED0, len(_keep2 - set(_keep)), len(set(_keep) - _keep2)))

    # 2l. THE TRANSFER LAW, asserted two-sided on exactly the licence it takes.  The branch it
    #     governs was measured (probe_iter13b/c) to cost n=33 three escapes out of three, so the law
    #     must genuinely be able to switch it off -- and equally must not quietly widen it, or
    #     silently delete a move that iteration 5 measured as a real win where the census is loose.
    _p = XFER_P0
    for _ in range(60):
        _p = _xfer_update(_p, False)
    assert abs(_p - XFER_LO) < 1e-12, "transfer law does not decay to its floor under pure misses"
    assert _p > 0.0, "transfer law reached zero -- the branch could never revive"
    for _ in range(60):
        _p = _xfer_update(_p, True)
    assert abs(_p - XFER_P0) < 1e-12, "transfer law does not recover to its cap under pure threats"
    assert _xfer_update(XFER_P0, True) <= XFER_P0 + 1e-12, "transfer law widened past its own cap"
    #     ...and the gate must actually gate: p=0 never fires the branch, p=1 always does (given a
    #     pool), and at p=XFER_P0 the OTHER two channels keep the unconditional shares they were
    #     measured at (0.10 shake / 0.75 jitter), so this change touches one component and no other.
    _n = 31
    _b = np.concatenate([np.clip(_cold_start(_n, rng, 0), LO + 1e-3, HI - 1e-3),
                         np.zeros((_n, 1))], axis=1)
    _b[:, 2] = repair(_b[:, :2], lp_radii(_b[:, :2]))
    _pl = {_n: _b, _n - 2: _b[:_n - 2], _n + 2: np.concatenate([_b, _b[:2]])}
    _k = [_move(_b, rng, _n, _pl, SIG0, XFER_P0)[2] for _ in range(4000)]
    _f1, _f2, _f3 = (_k.count(1) / 4000.0, _k.count(2) / 4000.0, _k.count(3) / 4000.0)
    assert abs(_f1 - 0.15) < 0.02, "transfer share %.3f drifted from its measured 0.15" % _f1
    assert abs(_f2 - 0.10) < 0.02, "shake share %.3f drifted from its measured 0.10" % _f2
    assert abs(_f3 - 0.75) < 0.03, "jitter share %.3f drifted from its measured 0.75" % _f3
    _k0 = [_move(_b, rng, _n, _pl, SIG0, 0.0)[2] for _ in range(400)]
    assert 1 not in _k0, "pxfer=0 still fired the cross-n transfer"
    _k1 = [_move(_b, rng, _n, _pl, SIG0, 1.0)[2] for _ in range(400)]
    assert set(_k1) == {1}, "pxfer=1 did not always fire the cross-n transfer"
    #     an n with NO neighbours in the pool must fall through to the local channels, never crash --
    #     this is the offline-re-run / single-target case.
    _kn = [_move(_b, rng, _n, {_n: _b}, SIG0, 1.0)[2] for _ in range(200)]
    assert 1 not in _kn and set(_kn) <= {2, 3}, "transfer fired with no neighbour to transfer from"
    print("transfer law: shares %.3f/%.3f/%.3f  floor %.3f  cap %.3f" % (_f1, _f2, _f3, XFER_LO, XFER_P0))

    # 2m. THE ORIGIN GUARD, asserted two-sided.  A cross-n transfer builds its candidate from a
    #     DIFFERENT n's packing, so `_adopt`'s "close in sum_r" proxy for "a nearby place to search
    #     FROM" does not hold for it, and adopting one teleports the walk into an alien structure --
    #     the mechanism iteration 13 measured (3 of 3 escapes -> 0 of 3 at n=33) and iteration 14
    #     A/B'd directly (artifacts/probe_iter14.py: matched seeds/time/state/pool, n=27 escaped to
    #     the record in 1 of 7 seeds with the guard on and 0 of 7 with it off; no arm ever lost).
    #     Both directions matter: the guard must bar the transfer channel AND leave the two local
    #     channels untouched, and flipping XFER_DRIFT must restore the old behaviour exactly, so a
    #     regime where the transfer's structure IS worth standing in is one flag away, not a rewrite.
    assert _drift_ok(1, True) and not _drift_ok(1, False), "XFER_DRIFT does not gate the transfer"
    for _kind in (0, 2, 3):
        assert _drift_ok(_kind, False) and _drift_ok(_kind, True), (
            "origin guard leaked onto local channel %d" % _kind)
    assert _drift_ok(1) is _drift_ok(1, XFER_DRIFT), "module default disagrees with its own flag"
    #     ...and the guard must cost the transfer NOTHING on the path the score actually reads: the
    #     branch still fires at its full licensed rate, and a transfer that BEATS the incumbent still
    #     goes through evaluate() and still re-anchors the walk.  Asserted end-to-end on a live
    #     solve() whose pool is deliberately seeded with a WORSE incumbent at n than at n-2, so the
    #     only way n can improve is a cross-n transfer being scored and committed.
    _tn = 21
    _src = np.concatenate([np.clip(_cold_start(_tn + 2, rng, 1), LO + 1e-3, HI - 1e-3),
                           np.zeros((_tn + 2, 1))], axis=1)
    _sxy, _sr = slp_optimise(_src[:, :2], time.process_time() + 5.0, dmin=1e-10)
    _src = np.concatenate([_sxy, repair(_sxy, _sr)[:, None]], axis=1)
    _h = _StubHarness()
    _h.evaluate(_tn + 2, _src)
    _xd = os.path.join("notes", "_selftest_xfer")
    if not os.path.isdir(_xd):
        os.makedirs(_xd)
    _xsp = os.path.join(_xd, "xfer_state.json")
    _ob = globals()["CPU_BUDGET"]
    globals()["CPU_BUDGET"] = 4.0
    try:
        solve(_h.evaluate, _h, np.random.RandomState(14), [_tn, _tn + 2], state_path=_xsp)
    finally:
        globals()["CPU_BUDGET"] = _ob
    for _f in os.listdir(_xd):
        os.remove(os.path.join(_xd, _f))
    os.rmdir(_xd)
    assert _tn in _h.best and _h.best[_tn][0] > 0.0, (
        "with the origin guard on, n=%d produced no scored packing at all" % _tn)
    print("origin guard: transfer barred from drift, local channels intact, n=%d still scored %.6f"
          % (_tn, _h.best[_tn][0]))

    # 2d. the rebuilt move set: every move must return exactly n centres strictly inside the square,
    #     and NO move may be deterministic -- a move that recomputes one identical candidate forever
    #     (the old cross-n transfer did exactly that) burns restarts for nothing.
    for n in (31, 47):
        base = np.concatenate([np.clip(_cold_start(n, rng, 0), LO + 1e-3, HI - 1e-3),
                               np.zeros((n, 1))], axis=1)
        base[:, 2] = repair(base[:, :2], lp_radii(base[:, :2]))
        pool = {n: base, n - 2: base[:n - 2], n + 2: np.concatenate([base, base[:2]])}
        seen, kinds = [], set()
        for _ in range(200):
            mv, d0, kind, ref_ok, _fm = _move(base, rng, n, pool, SIG0)
            assert ref_ok in (True, False)
            assert (kind == 1) or ref_ok, "an index-aligned move opted out of the repeat check"
            assert mv.shape == (n, 2), "move %d returned %s" % (kind, mv.shape)
            assert mv.min() > LO and mv.max() < HI, "move %d left the square" % kind
            assert d0 is None or d0 > 0
            kinds.add(kind)
            seen.append(mv)
        for k in kinds:
            pass
        dup = sum(1 for i in range(1, len(seen)) if np.allclose(seen[i], seen[i - 1]))
        assert dup == 0, "move set produced %d consecutive identical candidates" % dup
        print("moves n=%d kinds=%s all-distinct over 200 draws" % (n, sorted(kinds)))

    # 2e. the basin-repeat machinery.  Two claims are asserted, because both are load-bearing:
    #     (i) a restart launched from the incumbent itself MUST be detected as a repeat and aborted;
    #     (ii) a restart that lands somewhere genuinely different MUST NOT be (a false abort would
    #     silently throw away exactly the candidates the search exists to find).
    for n in (17, 31):
        xy0 = np.clip(_cold_start(n, rng, 0), LO + 1e-3, HI - 1e-3)
        inc, r_inc = slp_optimise(xy0, time.process_time() + 6.0, dmin=1e-9)
        atol = ABORT_TOL / np.sqrt(n)
        tiny = np.clip(inc + rng.normal(0.0, 0.02 / np.sqrt(n), size=inc.shape), LO + 1e-3, HI - 1e-3)
        _, _, ab_same = slp_run(tiny, time.process_time() + 6.0, delta0=0.35 / np.sqrt(n),
                                dmin=COARSE_DMIN, ref=inc, ref_tol=atol, peek_it=PEEK_IT)
        far = np.clip(_cold_start(n, rng, 2), LO + 1e-3, HI - 1e-3)
        _, _, ab_far = slp_run(far, time.process_time() + 6.0, delta0=0.35 / np.sqrt(n),
                               dmin=COARSE_DMIN, ref=inc, ref_tol=atol, peek_it=PEEK_IT)
        assert ab_same, "n=%d: a restart from the incumbent was NOT detected as a repeat" % n
        assert not ab_far, "n=%d: an unrelated basin was falsely aborted as a repeat" % n
        # slp_run with no ref must never abort, and must agree with the slp_optimise wrapper
        a, b, ab = slp_run(tiny.copy(), time.process_time() + 6.0, delta0=0.35 / np.sqrt(n),
                           dmin=COARSE_DMIN)
        assert not ab, "aborted with ref=None"
        print("repeat-detect n=%d: from-incumbent=%s from-elsewhere=%s" % (n, ab_same, ab_far))
    # 2f. THE COARSE-TIER CLAIM, asserted in the code that depends on it.  Exploration stops at
    #     COARSE_DMIN because the decision it feeds (is this basin within POLISH_GAP of the
    #     incumbent?) is already made there.  That is only true while the residual disagreement
    #     between the coarse and a fine run stays far below POLISH_GAP -- if a future edit loosens
    #     COARSE_DMIN too far, the search starts MIS-RANKING basins silently, discarding winners
    #     instead of merely spending less.  So assert the margin, not the constant.
    for n in (23, 41):
        xy0 = np.clip(_cold_start(n, rng, 0), LO + 1e-3, HI - 1e-3)
        worst, spd = 0.0, []
        for _ in range(3):
            st = np.clip(xy0 + rng.normal(0.0, 0.20 / np.sqrt(n), size=xy0.shape), LO + 1e-3, HI - 1e-3)
            t0 = time.process_time()
            _, rc, _ = slp_run(st.copy(), time.process_time() + 20.0,
                               delta0=0.35 / np.sqrt(n), dmin=COARSE_DMIN)
            t1 = time.process_time()
            _, rf, _ = slp_run(st.copy(), time.process_time() + 20.0,
                               delta0=0.35 / np.sqrt(n), dmin=1e-6)
            t2 = time.process_time()
            worst = max(worst, abs(float(rc.sum()) - float(rf.sum())))
            spd.append((t2 - t1) / max(t1 - t0, 1e-9))
        assert worst < 0.02 * POLISH_GAP, (
            "coarse tier disagrees with a fine run by %.2e -- too close to POLISH_GAP=%.0e to rank"
            % (worst, POLISH_GAP))
        print("coarse-tier fidelity n=%d: max|dsum|=%.2e vs POLISH_GAP=%.0e (%.0fx margin), %.2fx faster"
              % (n, worst, POLISH_GAP, POLISH_GAP / max(worst, 1e-300), float(np.mean(spd))))

    # 2g. THE EARLY-STOP MARGIN, asserted in the code that depends on it.  The coarse tier now
    #     stops at STOP_EPS rather than at convergence.  Two claims are load-bearing and both are
    #     checked, in both directions:
    #     (i) monotonicity -- the stopped run may only UNDER-report sum_r (if it could over-report,
    #         a losing basin could be polished, or worse, ranked above the incumbent);
    #     (ii) the margin -- that under-report must stay far below POLISH_GAP, because the failure
    #         mode of loosening STOP_EPS is not a crash but a SILENT mis-ranking that discards
    #         winners.  A future edit that erases the margin fails here instead.
    for n in (23, 41):
        xy0 = np.clip(_cold_start(n, rng, 0), LO + 1e-3, HI - 1e-3)
        worst, spd = -np.inf, []
        for _ in range(3):
            st = np.clip(xy0 + rng.normal(0.0, 0.20 / np.sqrt(n), size=xy0.shape),
                         LO + 1e-3, HI - 1e-3)
            t0 = time.process_time()
            _, r_st, _ = slp_run(st.copy(), time.process_time() + 20.0, delta0=0.35 / np.sqrt(n),
                                 dmin=COARSE_DMIN, stop_eps=STOP_EPS)
            t1 = time.process_time()
            _, r_fu, _ = slp_run(st.copy(), time.process_time() + 20.0, delta0=0.35 / np.sqrt(n),
                                 dmin=COARSE_DMIN, stop_eps=0.0)
            t2 = time.process_time()
            worst = max(worst, float(r_fu.sum()) - float(r_st.sum()))
            spd.append((t2 - t1) / max(t1 - t0, 1e-9))
        assert worst >= -1e-12, (
            "early stop OVER-reported sum_r by %.2e -- the run is not monotone, so the stop is "
            "unsound (a losing basin could outrank the incumbent)" % (-worst))
        assert worst < 0.01 * POLISH_GAP, (
            "early stop under-reports sum_r by %.2e -- too close to POLISH_GAP=%.0e, the polish "
            "filter would start discarding winners silently" % (worst, POLISH_GAP))
        print("early-stop margin n=%d: max under-report=%.2e vs POLISH_GAP=%.0e (%.0fx margin), "
              "%.2fx faster" % (n, max(worst, 0.0), POLISH_GAP,
                                POLISH_GAP / max(worst, 1e-300), float(np.mean(spd))))
    # a repeat must STILL be caught: the early stop is deferred past PEEK_IT precisely so it can
    # never pre-empt the abort, and that ordering is what keeps repeats out of the polish tier.
    for n in (17, 31):
        xy0 = np.clip(_cold_start(n, rng, 0), LO + 1e-3, HI - 1e-3)
        inc, _ = slp_optimise(xy0, time.process_time() + 6.0, dmin=1e-9)
        atol = ABORT_TOL / np.sqrt(n)
        tiny = np.clip(inc + rng.normal(0.0, 0.02 / np.sqrt(n), size=inc.shape),
                       LO + 1e-3, HI - 1e-3)
        _, _, ab = slp_run(tiny, time.process_time() + 6.0, delta0=0.35 / np.sqrt(n),
                           dmin=COARSE_DMIN, ref=inc, ref_tol=atol, peek_it=PEEK_IT,
                           stop_eps=STOP_EPS)
        assert ab, "n=%d: the early stop pre-empted the basin-repeat abort" % n
        print("repeat-abort survives the early stop n=%d" % n)

    # 2h. THE DRIFT RULE, asserted in the code that depends on it -- both directions, because both
    #     failure modes are SILENT.  Too small and the origin never leaves the incumbent's basin
    #     (drift becomes an expensive no-op: measured, a 3e-4 tolerance adopted 259 origins at n=33
    #     and still found nothing, while 2e-3 walked to the record); too large and the origin wanders
    #     out of the neighbourhood the incumbent's structure is worth searching from.  So assert the
    #     bracket in the units that set it: the polish gap below, the mean radius above.
    for n, ssum in ((33, 2.9777), (99, 5.2182)):
        t = _drift_tol(n, ssum)
        mean_r = ssum / n
        assert t > 10.0 * POLISH_GAP, (
            "drift tolerance %.2e at n=%d is within 10x of POLISH_GAP=%.0e -- the origin can only "
            "ever adopt basins the polish filter already accepts, so drift is a silent no-op"
            % (t, n, POLISH_GAP))
        assert t < 0.10 * mean_r, (
            "drift tolerance %.2e at n=%d exceeds 10%% of the mean radius %.2e -- the origin can "
            "leave the incumbent's neighbourhood entirely" % (t, n, mean_r))
        # the walk is ANCHORED at best: anything at or below best - t must be refused, from any cur
        assert not _adopt(ssum - t, ssum, ssum, t), "adopted a basin at exactly best - tol"
        assert not _adopt(ssum - 2 * t, ssum, ssum - t, t), "the drift ball is not anchored at best"
        assert _adopt(ssum - 0.5 * t, ssum, ssum, t), "refused a basin well inside the drift ball"
        assert not _adopt(ssum - 0.5 * t, ssum, ssum, 0.0), "adopted with drift disabled (tol=0)"
        print("drift bracket n=%d: tol=%.2e in (10*POLISH_GAP=%.2e, 0.1*mean_r=%.2e), anchored"
              % (n, t, 10.0 * POLISH_GAP, 0.10 * mean_r))
    # 2i. THE ADOPT-RATE CONTROLLER, asserted where the licence it takes is spent.  It is allowed to
    #     widen the drift ball because a FIXED fraction was measured inert at large n (adopt rate
    #     3-8%, zero polishes in 8 s) -- but the whole safety argument of 2h is that the tolerance
    #     stays inside a verified bracket, so assert that the controller CANNOT leave it, in both
    #     directions, and that its equilibrium is the adopt rate it was designed for.
    v = DRIFT_FRAC
    for _ in range(3000):
        v = _dfrac_update(v, False)
    assert abs(v - DFRAC_HI) < 1e-12, "drift fraction did not saturate upward: %g" % v
    for _ in range(3000):
        v = _dfrac_update(v, True)
    assert abs(v - DFRAC_LO) < 1e-12, "drift fraction did not saturate downward: %g" % v
    mid = math.sqrt(DFRAC_LO * DFRAC_HI)          # strictly inside the clamp, so the sign shows
    assert _dfrac_update(mid, False) > mid > _dfrac_update(mid, True), \
        "the controller responds to the adopt rate with the wrong sign"
    eq = math.log(DFRAC_UP) / (math.log(DFRAC_UP) - math.log(DFRAC_DN))
    assert 0.12 < eq < 0.30, "equilibrium adopt rate %.3f is not the measured working rate" % eq
    # the clamp must stay inside 2h's bracket at EVERY n of the census -- the ceiling is the thing a
    # later edit would be tempted to raise, and raising it is exactly how the origin escapes the ball.
    for n in (27, 63, 99):
        w = read_pack(n, "")
        if w is None:
            continue
        ssum = float(w[:, 2].sum())
        # <= because DFRAC_HI sits EXACTLY on 2h's ceiling (0.10 of the mean radius): the
        # controller is licensed up to the edge of the verified bracket and no further, so any later
        # edit that raises DFRAC_HI above it fails here instead of quietly freeing the origin.
        assert _drift_tol(n, ssum, DFRAC_HI) <= 0.10 * (ssum / n) + 1e-15, (
            "DFRAC_HI=%g puts the drift tolerance outside the verified bracket at n=%d" % (DFRAC_HI, n))
        assert _drift_tol(n, ssum, DFRAC_LO) > 10.0 * POLISH_GAP, (
            "DFRAC_LO=%g makes drift a no-op at n=%d" % (DFRAC_LO, n))
    print("adopt-rate controller clamped to [%g, %g] (inside the 2h bracket), equilibrium %.0f%%"
          % (DFRAC_LO, DFRAC_HI, 100.0 * eq))

    # an n with no known best (an offline re-run on a fresh size) must fall back to NO drift, so the
    # loop degrades exactly to the greedy search rather than adopting garbage.
    assert _drift_tol(31, None) == 0.0 and _drift_tol(31, float("nan")) == 0.0
    assert _drift_tol(31, -1.0) == 0.0 and _drift_tol(0, 1.0) == 0.0

    # the sigma controller must stay in its clamp from any start and move the right way
    v = SIG0
    for _ in range(200):
        v = _sig_update(v, True)
    assert abs(v - SIG_HI) < 1e-12, "sigma did not saturate upward: %g" % v
    for _ in range(2000):
        v = _sig_update(v, False)
    assert abs(v - SIG_LO) < 1e-12, "sigma did not saturate downward: %g" % v
    assert _sig_update(SIG0, True) > SIG0 > _sig_update(SIG0, False)
    print("sigma controller clamped to [%g, %g], equilibrium repeat rate %.0f%%"
          % (SIG_LO, SIG_HI, 100 * np.log(SIG_DN) / (np.log(SIG_DN) - np.log(SIG_UP))))

    # 2c. records/cap guards: an unknown n must NEVER be treated as capped (that would silently
    #     starve every n of an offline re-run on sizes outside the census).
    recs = read_records()
    assert _at_cap(31, 1.0, {}) is False, "capped with no records table"
    assert _at_cap(10 ** 6, 1.0, recs) is False, "capped an n absent from the records"
    if recs:
        n0 = sorted(recs)[0]
        assert _at_cap(n0, recs[n0], recs) is True, "an exact-record packing is not at the cap"
        assert _at_cap(n0, recs[n0] * (1 - 1e-4), recs) is False, "a 4-digit packing read as capped"
        print("records ok: %d entries, cap guard exercised on n=%d" % (len(recs), n0))

    # 3. read_pack must be guarded: absurd n -> None, never an exception
    assert read_pack(10 ** 6, "") is None
    assert read_pack(4, "/nonexistent") is None

    # 4. end-to-end: solve() on sizes with NO committed pack must work and beat a naive grid
    h = _StubHarness(20000)
    global CPU_BUDGET
    keep, CPU_BUDGET = CPU_BUDGET, 12.0
    _sp = os.path.join("notes", "_selftest_state.json")
    solve(h.evaluate, h, rng, [6, 16, 28], state_path=_sp)
    CPU_BUDGET = keep
    if os.path.exists(_sp):
        # the walk did persist through a real solve() -- assert it, then keep the scratch out of the
        # census's own state file
        assert read_state(_sp), "solve() wrote a state file the loader cannot read back"
        print("persisted walk: solve() checkpointed %d origin(s)" % len(read_state(_sp)))
        os.remove(_sp)
    # 2r. THE SMOOTH-FIELD CHANNEL -- a new proposal channel is a claim about WHERE candidates land,
    # so the guard is two-sided: it must be an EXACT no-op when its share is zero (an offline caller
    # or a decayed channel must reproduce the shipped move set bit-for-bit, coordinates included),
    # it must FIRE and be reported as its own kind when the share is live, its amplitude window must
    # bracket the MEASURED window on both sides, and -- the load-bearing one -- the field must
    # actually be SMOOTH: at equal displacement it must disturb local neighbour distances far less
    # than iid jitter, which is the entire reason it lands closer.
    _pk = np.concatenate([_cold_start(37, np.random.RandomState(5), 0),
                          np.full((37, 1), 0.05)], axis=1)
    for _seed in (1, 2, 3):
        a = _move(_pk, np.random.RandomState(_seed), 37, {}, 0.2, 0.0)
        b = _move(_pk, np.random.RandomState(_seed), 37, {}, 0.2, 0.0, pfield=0.0, afr=AFR0)
        assert a[2] == b[2] and np.array_equal(a[0], b[0]) and a[1] == b[1], \
            "pfield=0 is not an EXACT no-op on the shipped move set"
    _kinds = set()
    _rg = np.random.RandomState(7)
    for _ in range(300):
        _kinds.add(_move(_pk, _rg, 37, {}, 0.2, 0.0, pfield=0.5, afr=AFR0)[2])
    assert 4 in _kinds and 3 in _kinds, "the field channel never fires at a live share"
    assert AFR_LO < AFR0 < AFR_HI and AFR_LO >= 0.08 and AFR_HI <= 0.55, \
        "the field amplitude clamp left the measured window"
    assert _afr_update(AFR_LO, False) == AFR_LO and _afr_update(AFR_HI, True) == AFR_HI, \
        "the amplitude controller is not clamped"
    assert FIELD_LO > 0.0 and _field_update(FIELD_P0, True) <= FIELD_P0, \
        "the field share must be floored (a live probe) and capped at its measured value"
    assert _field_update(FIELD_LO, False) == FIELD_LO, "the field share is not floored"
    # smoothness: same RMS displacement, compare the disturbance to NEIGHBOUR distances
    _xy = _pk[:, :2]
    _d0 = _pdist(_xy)
    _near = _d0 < np.percentile(_d0, 15)
    _fj, _ff = [], []
    for _seed in range(6):
        _r1 = np.random.RandomState(_seed)
        _j = _xy + _r1.normal(0.0, 1.0, size=_xy.shape)
        _j = _xy + (_j - _xy) * (0.02 / np.sqrt(((_j - _xy) ** 2).sum(1).mean()))
        _f = _field(_xy, np.random.RandomState(_seed), 37, 0.02)
        _fj.append(np.abs(_pdist(_j) - _d0)[_near].mean())
        _ff.append(np.abs(_pdist(_f) - _d0)[_near].mean())
    _mj, _mf = float(np.mean(_fj)), float(np.mean(_ff))
    assert _mf < 0.35 * _mj, ("the field is not smooth: neighbour-distance disturbance %.2e vs "
                              "jitter %.2e at equal displacement" % (_mf, _mj))
    print("field channel: no-op at share 0, fires at share>0, amp in [%.2f,%.2f], "
          "neighbour disturbance %.2e vs jitter %.2e (%.1fx smoother)"
          % (AFR_LO, AFR_HI, _mf, _mj, _mj / max(_mf, 1e-18)))

    # 2s. THE FIELD FAMILY.  Widening one measured channel into a family is a licence to spend the
    # channel's budget on members that were never measured, so it is guarded on both sides.
    # (a) COLLAPSE: with only the measured member live the family is an EXACT no-op -- same
    #     coordinates AND the same rng stream as iteration 19's single-mode field (the mode draw is
    #     skipped, not merely ignored), re-derived here from the raw stream rather than from _field.
    # (b) EXPANSION: every member must actually fire at live weights, and every member must be
    #     SMOOTH -- the whole reason the family exists is that neighbours travel together, and a
    #     high-wavenumber member is the one that could stop being smooth.
    # (c) The two controllers must be clamped: weights floored (retire, never delete) and capped at
    #     the measured member's own weight, and the amplitude clamped PER MODE inside the measured
    #     window with no cross-mode leakage.
    for _seed in (1, 2, 3, 4, 5, 6):
        _rg1 = np.random.RandomState(_seed)
        _o = _move(_pk, _rg1, 37, {}, 0.2, 0.0, pfield=1.0, afr=AFR0, fw=(1.0, 0.0, 0.0))
        _rg2 = np.random.RandomState(_seed)
        if _rg2.rand() < SHAKE_SHARE:
            continue                                  # the shake channel won this draw, not a field
        _a, _b, _c = _rg2.normal(size=3)
        _dd = _pk[:, :2].dot(np.array([[_a, _c], [_b, -_a]]))
        _ref19 = _pk[:, :2] + (AFR0 * 0.2 / np.sqrt((_dd ** 2).sum(1).mean())) * _dd
        assert _o[2] == 4 and _o[4] == 0 and np.array_equal(
            _o[0], np.clip(_ref19, LO + 1e-3, HI - 1e-3)), \
            "a single-member family is not bit-identical to the shipped single-mode field"
        assert _rg1.rand() == _rg2.rand(), "the collapsed family consumed a spurious rng draw"
    _seen = set()
    _rg = np.random.RandomState(11)
    for _ in range(600):
        _mv = _move(_pk, _rg, 37, {}, 0.2, 0.0, pfield=0.9, afr=(AFR0,) * len(FMODES), fw=FW0)
        if _mv[2] == 4:
            _seen.add(_mv[4])
    assert _seen == set(range(len(FMODES))), \
        "a field-family member never fired at live weights: %r" % (sorted(_seen),)
    assert FW_LO > 0.0 and FW_HI == max(FW0) and all(0.0 < w <= FW_HI for w in FW0), \
        "the family weights must be floored above zero and capped at the measured member"
    assert _fw_update(FW0, 0, True) == FW0, "the measured member is not capped at its own weight"
    assert _fw_update(tuple([FW_LO] * len(FMODES)), 1, False)[1] == FW_LO, \
        "a family weight is not floored -- a decayed member would be deleted, not retired"
    assert AFR_INIT[0] == AFR0 and len(AFR_INIT) == len(FMODES) and all(
        AFR_LO <= a <= AFR_HI for a in AFR_INIT), \
        "a member's starting amplitude left the measured clamp"
    _amps = (AFR_LO, AFR0, AFR_HI)
    assert _afr_get(_amps, 1) == AFR0 and _afr_get(AFR0, 2) == AFR0, \
        "the per-mode amplitude reader is not mode-selective (or scalar-compatible)"
    _set = _afr_set(_amps, 2, _afr_update(_afr_get(_amps, 2), True))
    assert _set[0] == AFR_LO and _set[1] == AFR0 and _set[2] == AFR_HI, \
        "a per-mode amplitude update leaked across modes (or left the clamp)"
    # every member must be smooth -- measured against iid jitter at EQUAL rms displacement
    _rat = []
    for _m in FMODES:
        _fm = []
        for _seed in range(8):
            _f = _field(_xy, np.random.RandomState(_seed), 37, 0.02, _m)
            assert np.abs(np.sqrt(((_f - _xy) ** 2).sum(1).mean()) - 0.02) < 1e-9, \
                "field mode %d does not honour its RMS amplitude" % _m
            _fm.append(np.abs(_pdist(_f) - _d0)[_near].mean())
        _rat.append(_mj / max(float(np.mean(_fm)), 1e-18))
    assert min(_rat) > 2.0, ("a field-family member is not smooth: neighbour-disturbance ratios "
                             "vs jitter %r" % ([round(x, 2) for x in _rat],))
    print("field family: %d members, collapse is bit-exact, all fire, smoothness vs jitter %s"
          % (len(FMODES), " ".join("%.1fx" % x for x in _rat)))

    for n in (6, 16, 28):
        if n not in h.best:
            print("FAIL: no feasible packing for n=%d" % n)
            ok = False
        else:
            k = int(np.ceil(np.sqrt(n)))
            naive = n * (0.5 / k)                      # plain sqrt(n) grid of equal circles
            print("solve n=%d sum_r=%.6f (naive grid %.6f)" % (n, h.best[n][0], naive))
            if h.best[n][0] <= naive:
                print("FAIL: n=%d did not beat the naive grid" % n)
                ok = False
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print("usage: python3 tools/solver.py --self-test")
