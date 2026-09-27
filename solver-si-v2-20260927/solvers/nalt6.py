"""Circle-packing solver: sequential-LP (convex-underestimate) polish + basin hopping.

The iteration-1 solver used exact LP radii for fixed centres plus a first-order dual-gradient
step on the centres. That step is the crude part: it anneals a scalar step size and stalls far
from a KKT point. The fix here is second order in the only sense that matters -- solve the *joint*
(x, y, r) problem, not r-then-x alternately.

Key structure: the pair constraint is r_i + r_j <= d_ij, and d_ij is a CONVEX function of the
centres. Its first-order Taylor expansion is therefore a global UNDER-estimate:

    d_ij(P + D) >= d_ij(P) + u_ij . (D_i - D_j),      u_ij = (P_i - P_j)/d_ij

so replacing d_ij by that linear form gives an *inner* approximation of the feasible set. Every
solution of the resulting LP over (dx, dy, r) is therefore EXACTLY FEASIBLE for the true problem,
and since (D=0, r=r_now) is LP-feasible the LP value never decreases. Wall constraints
(x_i - r_i >= -1/2 etc.) are already linear, so they enter exactly. The result is a
monotone, always-feasible sequential-LP (SLP) ascent whose fixed points are KKT points -- and
HiGHS solves one such sparse LP in ~1-6 ms, so a full local convergence costs 0.1-0.35 s even at
n=99, against 44 s (and divergence) for SLSQP on the same problem.

With a trust radius `delta` on every variable, a pair with slack > 4*delta cannot possibly bind
(centres move <= delta each, radii grow <= delta each), so it is dropped exactly -- that keeps the
LP at ~n rows instead of ~n^2/2.

Because a local polish is now ~0.2 s rather than the whole per-n slice, the slice buys ~10-25
polishes, so the search becomes BASIN HOPPING: converge, then perturb (shake all centres, or
teleport the smallest circles to fresh sites) and re-converge, keeping the best.

Contract notes (MISSION.md):
  * every candidate counted goes through the metered `evaluate()`; improved packings are buffered
    and flushed in small batches.
  * the CPU deadline is RELATIVE (time.process_time() + budget at entry), so a second solve() call
    in the same process still works; the per-n slice is recomputed adaptively -> len(targets)==1 ok.
  * warm starts read bench/packs/csqv<n>.pck only when it exists; every n falls back to a cold start.
  * no hard-coded coordinates: cold starts are rng-driven; every packing comes out of the search.
"""
import json
import os
import time

import numpy as np

LO, HI = -0.5, 0.5
# CPU BUDGET. The driver arms a 120 s process-CPU backstop at solve() entry (solve_driver.py:272,
# clock started at :264), and this constant is the self-imposed stop. It sat at 95.0 for eleven
# iterations -- a 25 s margin set once and never revisited -- and the driver measured cpu=95.3 s,
# i.e. the overshoot past the self-stop is 0.3 s, not 25. Iteration 14's shootouts
# (artifacts/iter14/wshoot.py, src_arm.py) measured the excursion lottery at 1 real hit in 36
# tickets at the current census, down from the ~1/3 of iteration 12: at that rate a ticket buys
# nothing but a DRAW, so expected hits are linear in ticket count and nothing else. +17 CPU-s is
# +18% tickets for free. 112.0 keeps a 7.7 s margin -- 25x the measured overshoot -- so the
# backstop still never fires (if it did the driver would still harvest every tracked best, but a
# buffered pack could be lost, hence FLUSH below).
CPU_BUDGET_S = 112.0        # relative CPU budget per solve() call, under the driver's 120 s backstop
EDGE = 1e-6                 # keep centres this far inside the wall
RMIN = 1e-9                 # smallest radius we ever emit
SLACK = 1e-11               # feasibility margin we insist on (harness tol is 1e-9)
PASS1_FRAC = 0.60           # share of the CPU budget for the ascending pass; rest = descending
FLUSH = 2                   # improved packings buffered before one batched evaluate() call
#   was 6. A whole iteration spends ~44 of the 500,000 evaluations, so batching buys nothing
#   and an unflushed buffer is the ONLY way a harvested best can be lost if the backstop fires.
# TICKET QUANTUM. An excursion used to be handed `min(slice_end, deadline)` -- the WHOLE rest of
# the slice -- and simply returned when its internal ladder stopped moving, which measured at
# 1.3 s (n=37) to 4.7 s (n=81) per ticket and ~7 tickets per n per iteration
# (artifacts/iter13/tickets.py). Iteration 13 measured what that length actually buys.
# Per-draw (tools/probe.py at --secs 0.6/1.5/3.0, artifacts/iter13/q*.json) LONGER looks better:
# the only real hits grow with length (n=61 `nx:+2,-1`: +7.8e-5 @0.6 s -> +2.2e-3 hit 3/3 @3.0 s)
# and every 0.6 s "hit" at n=47 was a +2.7e-09 no-op. But the POLICY shootout at EQUAL total
# seconds (artifacts/iter13/quantum.py: 4.5 s per n, k tickets sampled from _EXC_POOL, chained off
# the running best, scored by the max the solver would bank) inverts it:
#     k=6 (0.75 s)  mean +3.69e-4      k=3 (1.5 s)  mean +1.94e-4      k=1 (4.5 s) mean +1.26e-4
# because what a ticket buys is a DRAW from a 19-entry pool whose useful tag is per-n, not deeper
# convergence -- n=61's +2.2e-3 is reachable in 0.75 s chained and unreachable in one 4.5 s draw
# that spent itself on `sym:mx,d2` (0 real hits in 24 draws across 4 n and 3 lengths).
# So a ticket gets a FIXED quantum and the slice buys as many as it can afford. The quantum scales
# with n because ticket cost does: n=81 wanted 1.5 s (k=6/0.75 s returned 0 there, k=3 hit
# +1.2e-3) while n=61 and n=47 were fine at 0.75 s -- the same ~n^1.6 that the measured 1.3 s
# (n=37) -> 4.7 s (n=81) natural lengths trace out.
TICKET_S = 0.75             # seconds one excursion ticket gets at the n=61 calibration point
TICKET_N0 = 61.0
TICKET_POW = 1.6            # ticket cost grows ~n^1.6 (measured natural lengths, tickets.py)
TICKET_MIN = 0.40           # below this an excursion cannot leave the incumbent at all
DIGITS_CAP = 7.0            # the scorer's per-n digit cap: an n at the cap cannot gain ANY score
CAP_EPS = 0.02              # headroom below which an n is treated as capped and skipped
LP_OPTS = {"primal_feasibility_tolerance": 1e-10,   # HiGHS floor is 1e-10
           "dual_feasibility_tolerance": 1e-10}


def _pairwise(xy):
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d


def _walls(xy):
    return np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0], xy[:, 1] - LO, HI - xy[:, 1]])


def _greedy_radii(xy):
    """Guaranteed-feasible radii -- the fallback whenever an LP does not return."""
    return np.maximum(np.minimum(_walls(xy), _pairwise(xy).min(axis=1) / 2.0) - 2e-9, RMIN)


def _make_feasible(xy, r):
    """Shrink r uniformly by the smallest amount that restores strict feasibility (margin SLACK)."""
    d = _pairwise(xy)
    r = np.maximum(np.asarray(r, dtype=float), RMIN)
    pair = float((d - (r[:, None] + r[None, :])).min()) if len(r) > 1 else np.inf
    wall = float((_walls(xy) - r).min())
    delta = 0.0
    if pair < SLACK:
        delta = max(delta, (SLACK - pair) / 2.0)
    if wall < SLACK:
        delta = max(delta, SLACK - wall)
    if delta > 0.0:
        r = r - delta
    return np.maximum(r, 1e-12)


def _lp_radii(xy, linprog, coo_matrix):
    """Exact max-sum radii for FIXED centres (a small LP; optimal, not the equal-split rule)."""
    n = len(xy)
    wall = np.maximum(_walls(xy), 0.0)
    d = _pairwise(xy)
    iu = np.triu_indices(n, 1)
    dd = d[iu]
    keep = dd < (wall[iu[0]] + wall[iu[1]])
    I, J = iu[0][keep], iu[1][keep]
    m = len(I)
    A = b = None
    if m:
        rows = np.repeat(np.arange(m), 2)
        cols = np.stack([I, J], 1).ravel()
        A = coo_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n)).tocsr()
        b = dd[keep]
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b,
                      bounds=np.stack([np.zeros(n), wall], 1),
                      method="highs", options=LP_OPTS)
    except Exception:
        res = None
    if res is None or res.x is None:
        return _greedy_radii(xy)
    return np.asarray(res.x, dtype=float)


def _pval(r, p):
    """Surrogate value of the power family, written so LARGER IS ALWAYS BETTER.

    p = 1   -> sum r          (the census objective)
    0 < p<1 -> sum r^p        (concave; rewards spreading radius onto the small circles)
    p = 0   -> sum log r
    p < 0   -> -sum r^p       (r^p is decreasing, so negate; p -> -inf tends to max-min)

    In every branch dS/dr_i = |p| * r_i^(p-1) > 0, so the LP weight is the SAME expression
    r_i^(p-1) throughout and p sweeps one continuous knob from max-sum to max-min.
    """
    r = np.maximum(np.asarray(r, dtype=float), 1e-15)
    if p == 1.0:
        return float(r.sum())
    if p == 0.0:
        return float(np.log(r).sum())
    s = float((r ** p).sum())
    return s if p > 0 else -s


def _pweights(r, p, wcap=1e3):
    """LP objective weights for the power family: w_i proportional to r_i^(p-1), scaled to max 1."""
    if p == 1.0:
        return np.ones(len(r))
    r = np.maximum(np.asarray(r, dtype=float), 1e-12)
    w = (r / max(float(r.max()), 1e-12)) ** (p - 1.0)
    return np.minimum(w, wcap)


def _slp(xy, r, linprog, coo_matrix, t_end, max_it=300, delta0=None, p=1.0, hw=0.5, hh=0.5):
    """Sequential-LP ascent on (x, y, r). Monotone and always exactly feasible (see module doc).

    `p` selects the objective from the power family (see _pval): p=1 is the census objective and
    reproduces the original ascent exactly. p<1 solves a DIFFERENT problem whose KKT points are a
    different set -- that is what a p-homotopy walks along.

    (hw, hh) are the container HALF-width/half-height; (0.5, 0.5) is the census's unit square and
    reproduces the original ascent exactly. Any other aspect is a DIFFERENT container whose
    optimal contact graph is different -- that is what the aspect homotopy (_arect) walks along.
    The incoming (xy, r) must be feasible for THIS box (see _rebox).
    """
    n = len(xy)
    if n < 2:
        return xy.copy(), _make_feasible(xy, _greedy_radii(xy))
    xy = np.array(xy, dtype=float)
    r = np.array(r, dtype=float)
    iu = np.triu_indices(n, 1)
    I0, J0 = iu
    delta = float(delta0) if delta0 else 0.5 * float(r.mean())
    best = _pval(r, p)
    bxy, br = xy.copy(), r.copy()
    ar_ii = np.arange(n)
    for _ in range(max_it):
        if time.process_time() >= t_end or delta < 1e-13:
            break
        dv = np.sqrt(((xy[I0] - xy[J0]) ** 2).sum(1))
        keep = (dv - r[I0] - r[J0]) < 4.0 * delta
        I, J = I0[keep], J0[keep]
        m = int(keep.sum())
        dd = dv[keep]
        rows, cols, vals = [], [], []
        if m:
            ux = (xy[I, 0] - xy[J, 0]) / dd
            uy = (xy[I, 1] - xy[J, 1]) / dd
            ar = np.arange(m)
            rows.append(np.concatenate([ar] * 6))
            cols.append(np.concatenate([I, J, n + I, n + J, 2 * n + I, 2 * n + J]))
            vals.append(np.concatenate([-ux, ux, -uy, uy, np.ones(m), np.ones(m)]))
        b = [dd]
        base = m
        half = (hw, hh)
        for sgn, ax in ((-1.0, 0), (1.0, 0), (-1.0, 1), (1.0, 1)):
            rows.append(base + ar_ii); cols.append(ax * n + ar_ii); vals.append(np.full(n, sgn))
            rows.append(base + ar_ii); cols.append(2 * n + ar_ii); vals.append(np.ones(n))
            b.append((xy[:, ax] + half[ax]) if sgn < 0 else (half[ax] - xy[:, ax]))
            base += n
        A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(m + 4 * n, 3 * n)).tocsc()
        lo = np.concatenate([np.maximum(-delta, -hw - xy[:, 0]),
                             np.maximum(-delta, -hh - xy[:, 1]), np.zeros(n)])
        hi = np.concatenate([np.minimum(delta, hw - xy[:, 0]),
                             np.minimum(delta, hh - xy[:, 1]),
                             np.minimum(min(hw, hh), r + delta)])
        try:
            res = linprog(np.concatenate([np.zeros(2 * n), -_pweights(r, p)]),
                          A_ub=A, b_ub=np.concatenate(b),
                          bounds=np.stack([lo, hi], 1), method="highs", options=LP_OPTS)
        except Exception:
            res = None
        if res is None or res.x is None:
            delta *= 0.5
            continue
        z = res.x
        nxy = xy + np.stack([z[:n], z[n:2 * n]], 1)
        nr = z[2 * n:]
        s = _pval(nr, p)
        if s > best + 1e-16:
            gain = s - best
            best, xy, r = s, nxy, nr
            bxy, br = nxy.copy(), nr.copy()
            if gain < 1e-4 * delta:
                delta *= 0.6
        else:
            delta *= 0.5
    return bxy, br


def _slp_eq(xy, linprog, coo_matrix, t_end, max_it=400):
    """Sequential-LP for the MAX-MIN problem: maximize a COMMON radius rho over (dx, dy, rho).

    This is a DIFFERENT objective from the census's max-sum-r, so its local optima are a
    different set -- hexagonal / square-lattice arrangements that the sum-objective search never
    visits. Iterations 3-6 refuted five distinct intra-n perturbations of the sum problem (kick,
    teleport, Metropolis, remove-relax-reinsert, random objective reweighting): the monotone sum
    polish pulls every one straight back to the same KKT point. Converging under max-min instead
    moves the packing a long way through feasible space under a different gradient, and the
    configuration it lands on is combinatorially foreign -- measured (artifacts/iter6/eqprobe.py)
    at +1.4e-2 on n=99 and +8.0e-3 on n=45, the two worst n in the census.

    Same convex-underestimate construction as `_slp`: d_ij is convex in the centres, so its
    linearization is a global under-estimate and every LP solution is exactly feasible.
    """
    n = len(xy)
    if n < 2:
        return np.array(xy, dtype=float), 0.5
    xy = np.array(xy, dtype=float)
    I0, J0 = np.triu_indices(n, 1)
    rho = max(float(min(_pairwise(xy).min() / 2.0, _walls(xy).min())), 1e-6)
    delta = 0.5 * rho
    best, bxy, brho = rho, xy.copy(), rho
    ii = np.arange(n)
    for _ in range(max_it):
        if time.process_time() >= t_end or delta < 1e-12:
            break
        dv = np.sqrt(((xy[I0] - xy[J0]) ** 2).sum(1))
        keep = (dv - 2.0 * rho) < 4.0 * delta
        I, J = I0[keep], J0[keep]
        m = int(keep.sum())
        dd = dv[keep]
        rows, cols, vals = [], [], []
        if m:
            ux = (xy[I, 0] - xy[J, 0]) / dd
            uy = (xy[I, 1] - xy[J, 1]) / dd
            ar = np.arange(m)
            rows.append(np.concatenate([ar] * 5))
            cols.append(np.concatenate([I, J, n + I, n + J, np.full(m, 2 * n)]))
            vals.append(np.concatenate([-ux, ux, -uy, uy, np.full(m, 2.0)]))
        b = [dd]
        base = m
        for sgn, ax in ((-1.0, 0), (1.0, 0), (-1.0, 1), (1.0, 1)):
            rows.append(base + ii); cols.append(ax * n + ii); vals.append(np.full(n, sgn))
            rows.append(base + ii); cols.append(np.full(n, 2 * n)); vals.append(np.ones(n))
            b.append((xy[:, ax] - LO) if sgn < 0 else (HI - xy[:, ax]))
            base += n
        A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(m + 4 * n, 2 * n + 1)).tocsc()
        lo = np.concatenate([np.maximum(-delta, LO - xy[:, 0]),
                             np.maximum(-delta, LO - xy[:, 1]), [0.0]])
        hi = np.concatenate([np.minimum(delta, HI - xy[:, 0]),
                             np.minimum(delta, HI - xy[:, 1]), [min(0.5, rho + delta)]])
        c = np.zeros(2 * n + 1)
        c[2 * n] = -1.0
        try:
            res = linprog(c, A_ub=A, b_ub=np.concatenate(b),
                          bounds=np.stack([lo, hi], 1), method="highs", options=LP_OPTS)
        except Exception:
            res = None
        if res is None or res.x is None:
            delta *= 0.5
            continue
        z = res.x
        nxy = xy + np.stack([z[:n], z[n:2 * n]], 1)
        nrho = float(z[2 * n])
        if nrho > best + 1e-16:
            g = nrho - best
            best, xy, rho = nrho, nxy, nrho
            bxy, brho = nxy.copy(), nrho
            if g < 1e-4 * delta:
                delta *= 0.6
        else:
            delta *= 0.5
    return bxy, brho


def _ladder(xy, r, ps, linprog, coo_matrix, t_end, tail=0.45):
    """p-HOMOTOPY: converge under each p in `ps` in turn, warm-starting from the previous stage.

    Iteration 6 established that only a change of *problem* escapes the sum-objective's basin, and
    that the extreme change (max-min) works. This walks the whole family between the two ends
    instead of jumping: sum r^p for p rising to 1 (see _pval). Each intermediate p is a genuinely
    different optimization whose KKT set is different, and warm-starting the next p from the
    previous one TRACKS a solution path that changes contact topology at critical p -- so the
    p=1 point it arrives at is a different (measured: usually better) one than the direct jump.

    Measured offline (artifacts/iter7/phomo.py, artifacts/iter7/chain.py) against the iteration-6
    census: n=33 2.980533 -> 2.987285 (the record to 1e-9), n=99 5.226948 -> 5.232784 (a BEAT),
    n=61 +2.1e-3, n=45 +2.4e-3. Re-applying the ladder is a fixed-point map: it converges after
    1-2 rounds, so chaining is cheap and bounded.

    TIME: every stage gets a share of what is left, and the FINAL stage (which must be p=1, the
    real objective) is reserved `tail` of the remaining time -- a ladder truncated before its
    p=1 stage returns a low-sum packing and wastes the slice.
    """
    xy = np.array(xy, dtype=float)
    r = np.array(r, dtype=float)
    k = len(ps)
    for i, p in enumerate(ps):
        left = t_end - time.process_time()
        if left <= 0.01:
            break
        last = (i == k - 1)
        share = left if last else min(left * (1.0 - tail), left / float(k - i))
        xy, r = _slp(xy, r, linprog, coo_matrix, time.process_time() + share, p=p)
    return xy, r


def _rand_ladder(rng, k):
    """A p-ladder ending at 1.0. The first two are the shapes measured best offline; after that
    the start depth, the number of rungs and the spacing are all rng-drawn, so the ladder family
    is an UNBOUNDED generator of distinct fixed points (n=61: L3 and L4b converge to different
    ones) that stores no coordinates."""
    if k == 0:
        return [-1.0, 0.0, 1.0]
    if k == 1:
        return [-2.0, -0.5, 0.2, 1.0]
    p0 = -(0.4 + 3.0 * rng.rand() ** 1.5)          # start depth in [-3.4, -0.4]
    m = int(rng.randint(2, 6))                      # rungs strictly below 1.0
    u = np.sort(rng.rand(m))
    return [float(p0 + (1.0 - p0) * (t ** (0.6 + rng.rand()))) for t in u] + [1.0]


def _rebox(xy, r, hw0, hh0, hw1, hh1):
    """Map a packing feasible in box (hw0, hh0) to one feasible in box (hw1, hh1).

    Anisotropic scaling sx = hw1/hw0, sy = hh1/hh0 on the centres, and r -> r * min(sx, sy):
    the scaled centre distance is sqrt((sx dx)^2 + (sy dy)^2) >= min(sx, sy) * d, while the radius
    sum shrinks by exactly min(sx, sy), so EVERY pair and wall constraint is preserved. That
    exactness matters: the SLP ascent is only monotone-and-feasible if its starting point is
    feasible for the box it is given.
    """
    sx, sy = hw1 / hw0, hh1 / hh0
    return np.stack([xy[:, 0] * sx, xy[:, 1] * sy], 1), np.asarray(r, dtype=float) * min(sx, sy)


def _half(a):
    """Half-extents of the AREA-1 rectangle of aspect ratio a (width/height); a=1 -> the square."""
    q = np.sqrt(a)
    return 0.5 * q, 0.5 / q


def _arect(xy, r, aspects, linprog, coo_matrix, t_end, tail=0.45):
    """CONTAINER homotopy: converge in a sequence of area-1 rectangles ending at the unit square.

    The p-homotopy (_ladder) deforms the OBJECTIVE; this deforms the CONSTRAINTS. Squeezing the
    box to aspect a != 1 and letting the packing re-converge there forces rows to slide past one
    another -- a contact-graph change no perturbation of the square problem produces, because the
    square's own optimum is a strong attractor. Walking a back to 1 then TRACKS that different
    topology into a square packing (the same reason a p-ladder beats a jump: the path crosses the
    critical aspects where contacts switch, and lands on a different KKT point of the census
    problem).

    Area is held at 1 so the packing never has to grow or shrink globally; `_rebox` makes each
    stage's starting point exactly feasible for its box. The LAST aspect must be 1.0 -- anything
    else returns a rectangle packing, which is not a census answer -- and it is reserved `tail`
    of the remaining time.
    """
    xy = np.array(xy, dtype=float)
    r = np.array(r, dtype=float)
    hw, hh = 0.5, 0.5
    k = len(aspects)
    for i, a in enumerate(aspects):
        left = t_end - time.process_time()
        if left <= 0.01:
            break
        nhw, nhh = _half(a)
        xy, r = _rebox(xy, r, hw, hh, nhw, nhh)
        hw, hh = nhw, nhh
        last = (i == k - 1)
        share = left if last else min(left * (1.0 - tail), left / float(k - i))
        xy, r = _slp(xy, r, linprog, coo_matrix, time.process_time() + share, hw=hw, hh=hh)
    if (hw, hh) != (0.5, 0.5):                      # never return a non-square packing
        xy, r = _rebox(xy, r, hw, hh, 0.5, 0.5)
    return xy, r


def _rand_arect(rng, k):
    """An aspect ladder ending at 1.0. Shape 0/1 are the two measured best; after that the squeeze
    depth, its sign, the number of rungs and the spacing are rng-drawn, so -- like _rand_ladder --
    the family is an unbounded generator of distinct fixed points and stores no coordinates."""
    if k == 0:
        return [1.15, 1.07, 1.0]
    if k == 1:
        return [0.87, 0.93, 1.0]
    lg = (0.05 + 0.35 * rng.rand() ** 1.4) * (1.0 if rng.rand() < 0.5 else -1.0)
    m = int(rng.randint(1, 4))                      # rungs strictly before the square
    u = np.sort(rng.rand(m))[::-1]
    return [float(np.exp(lg * (t ** (0.6 + rng.rand())))) for t in u] + [1.0]


def _drop_idx(r, k, rng):
    """Indices to KEEP after removing k circles chosen among the 3k smallest.

    Removing the k smallest deterministically is what iterations 3-6 measured at zero: the
    largest hole left behind is exactly where the circle was, so the reinsertion undoes the
    removal. Drawing k out of the 3k smallest makes the removed SET a discrete random variable,
    so repeated excursions from one incumbent land on different contact graphs.
    """
    n = len(r)
    k = max(0, min(int(k), n - 2))
    if k == 0:
        return np.arange(n)
    pool = np.argsort(r)[:min(n - 1, 3 * k)]
    gone = pool if len(pool) <= k else rng.choice(pool, size=k, replace=False)
    keep = np.ones(n, bool)
    keep[np.asarray(gone, dtype=int)] = False
    return np.flatnonzero(keep)


def _mid_ps(rng):
    """A short p-ladder for a DETOUR stage. Iteration 9 fixed this at (-1, 1); a fixed detour
    ladder on a fixed detour packing is a deterministic map, i.e. the very fixed point the
    excursion exists to escape. Drawn shapes make repeated excursions land differently."""
    u = rng.rand()
    if u < 0.34:
        return [-1.0, 1.0]
    if u < 0.60:
        return [1.0]
    if u < 0.85:
        return [-(0.5 + 2.0 * rng.rand()), 0.0, 1.0]
    return [0.4, 1.0]


def _resize_to(xy, r, m, rng, top=1):
    """Feasibly change the circle COUNT of a packing to exactly m (insert at holes / drop small)."""
    n = len(xy)
    if m == n:
        return xy, r
    if m < n:
        if m < 2:
            return None, None
        idx = _drop_idx(r, n - m, rng)
        if len(idx) != m:
            return None, None
        return xy[idx], r[idx]
    return _insert_holes(xy, r, m - n, rng=rng, top=top)


def _nexc(xy, r, deltas, rng, linprog, coo_matrix, t_end, tail=0.55, top=3):
    """DISCRETE homotopy in the CIRCLE COUNT: n -> n+d1 -> n+d2 -> ... -> n, converging at each
    detour count on the way.

    _ladder deforms the objective and _arect deforms the container; both are CONTINUOUS, and
    iteration 8 measured that 13 of 19 live n sit on their fixed points and do not move. The
    reason is structural: a continuous deformation followed by a monotone ascent can only leave
    a basin if the deformation destroys it, and continuity is exactly the property that lets the
    basin survive. The circle COUNT is the one parameter of this problem that cannot be varied
    continuously -- packing n+1 circles is a different combinatorial problem, not a warped one,
    so its optimum has a different contact GRAPH, not merely different coordinates.

    Iteration 9 shipped the one-stage round trip `n -> n+/-k -> n` and it hit on 4 of the 5 worst
    n where a further p-ladder gained nothing. This generalizes it on the two axes iteration 9's
    own lesson named as untouched -- the MIDDLE:
      * `deltas` is a SEQUENCE, so `(+2, -1)` visits n+2 and then n-1 before returning. A
        multi-stage detour cannot be undone by any single inverse operation, so nothing about the
        path back is the reverse of the path out.
      * the detour ladder shape is drawn (`_mid_ps`) instead of fixed at (-1, 1), and insertion
        sites are drawn from the `top` largest holes instead of always the single largest, so
        repeating one excursion from one incumbent is no longer a deterministic map.

    Every stage is exactly feasible (subsets of a feasible packing are feasible; _insert_holes
    places at clearance; _make_feasible closes any residual), which the monotone SLP requires.
    """
    xy = np.array(xy, dtype=float)
    r = np.array(r, dtype=float)
    n = len(xy)
    if isinstance(deltas, (int, np.integer)):
        deltas = (int(deltas),)
    stages = []
    for d in deltas:
        m = n + int(d)
        m = max(2, min(m, 2 * n))
        if m != (stages[-1] if stages else n):
            stages.append(m)
    left = t_end - time.process_time()
    if left <= 0.05 or not stages:
        return xy, r
    mid_total = left * (1.0 - tail)
    sxy, sr = xy, r
    ns = len(stages)
    for i, m in enumerate(stages):
        gxy, gr = _resize_to(sxy, sr, m, rng, top)
        if gxy is None:
            return xy, r
        gr = _make_feasible(gxy, np.minimum(gr, _greedy_radii(gxy)))
        stage_end = min(t_end, time.process_time() + mid_total / float(ns - i))
        sxy, sr = _ladder(gxy, gr, _mid_ps(rng), linprog, coo_matrix, stage_end)
        mid_total = max(0.0, mid_total - (mid_total / float(ns - i)))
    sxy, sr = _resize_to(sxy, sr, n, rng, top)
    if sxy is None or len(sxy) != n:                  # never hand back the wrong count
        return xy, r
    sr = _make_feasible(sxy, np.minimum(sr, _greedy_radii(sxy)))
    return _ladder(sxy, sr, _rand_ladder(rng, 0), linprog, coo_matrix, t_end)


# --- SYMMETRY EXCURSION -------------------------------------------------------------------
# The four REFLECTIONS of the square's dihedral group. Each is an involution with a 1-D fixed
# line, which is what makes the orbit projection below well defined (see _symmetrize).
_REFL = {
    "mx": (lambda q: np.stack([-q[:, 0], q[:, 1]], 1),          # mirror in x -> fixed line x=0
           lambda q: np.stack([np.zeros(len(q)), q[:, 1]], 1)),
    "my": (lambda q: np.stack([q[:, 0], -q[:, 1]], 1),          # mirror in y -> fixed line y=0
           lambda q: np.stack([q[:, 0], np.zeros(len(q))], 1)),
    "d1": (lambda q: np.stack([q[:, 1], q[:, 0]], 1),           # main diagonal  -> y = x
           lambda q: np.repeat(((q[:, 0] + q[:, 1]) / 2.0)[:, None], 2, 1)),
    "d2": (lambda q: np.stack([-q[:, 1], -q[:, 0]], 1),         # anti-diagonal  -> y = -x
           lambda q: np.stack([(q[:, 0] - q[:, 1]) / 2.0, (q[:, 1] - q[:, 0]) / 2.0], 1)),
}


def _sym_pairing(xy, g, lsa):
    """Match every circle to its mirror image and return an INVOLUTION pi with pi(pi(i)) == i.

    linear_sum_assignment on the |x_i - g(x)_j|^2 cost gives the cheapest matching, but nothing
    forces it to be an involution and the orbit average is only g-symmetric when it is. So the
    assignment is reduced to disjoint pairs, cheapest pair first; whatever is left over becomes a
    fixed point (a circle that will be pushed onto the mirror LINE).
    """
    n = len(xy)
    q = g(xy)
    C = ((xy[:, None, :] - q[None, :, :]) ** 2).sum(-1)
    try:
        ri, ci = lsa(C)
    except Exception:
        return np.arange(n)
    pi = np.arange(n)
    order = np.argsort(C[ri, ci])
    used = np.zeros(n, dtype=bool)
    for k in order:
        i, j = int(ri[k]), int(ci[k])
        if i == j or used[i] or used[j]:
            continue
        pi[i], pi[j] = j, i
        used[i] = used[j] = True
    return pi


def _symmetrize(xy, r, key, lsa):
    """Project a packing onto the g-SYMMETRIC subspace (g one of the four reflections).

    Paired circles are replaced by the midpoint of the orbit {x_i, g(x_pi(i))} and the mean of the
    two radii, unpaired ones are dropped onto the mirror line. The result is EXACTLY g-symmetric:
    g maps the circle set to itself. It is generally infeasible (centres moved), so the caller
    must repair -- but it is a genuinely DISCRETE structural move, because which circles pair with
    which is a combinatorial choice, not a continuous parameter.
    """
    g, onto = _REFL[key]
    xy = np.asarray(xy, dtype=float)
    r = np.asarray(r, dtype=float)
    pi = _sym_pairing(xy, g, lsa)
    fix = pi == np.arange(len(xy))
    q = g(xy)
    nxy = 0.5 * (xy + q[pi])
    nxy[fix] = onto(xy[fix])
    nr = 0.5 * (r + r[pi])
    nxy = np.clip(nxy, LO + EDGE, HI - EDGE)
    return nxy, nr


def _proj_score(xy, r, key, lsa, linprog, coo_matrix):
    """Sigma_r a g-projection RETAINS, before any ascent. Costs one LSA + one LP -- ~free."""
    pxy, pr = _symmetrize(xy, r, key, lsa)
    pr = _make_feasible(pxy, _lp_radii(pxy, linprog, coo_matrix))
    return float(pr.sum()), pxy, pr


def _auto_key(xy, r, rng, lsa, linprog, coo_matrix, avoid=None, top=2, pw=6.0, anti=False):
    """MEASURE which reflection to project onto instead of guessing it from a frozen queue.

    Iteration 11 shipped `_symexc` with the reflection named in the start-queue tag ("sym:mx,d2"),
    a constant chosen by a per-n probe -- and iteration 11's own lesson then established that that
    probe seeded one rng stream per n and so overstated how reproducible any such constant is.
    The queue is the frozen thing: a slice affords ~3 excursions, so with 4 reflections x 2 stages
    (20 sequences) the right key for THIS n is mostly never tried.

    It does not have to be guessed. If the true optimum carries reflection g, the incumbent is a
    symmetry-broken copy of it and is therefore ALREADY nearly g-invariant -- so projecting onto g
    destroys little Sigma_r, while projecting onto a reflection the optimum does not carry collapses
    it. That loss is computable for all four reflections in ~4 LSA + 4 LP solves (milliseconds
    against a 1.5 s excursion), so the discrete choice can be made from measurement at the moment
    it is made, per n and per stage, rather than frozen in a list months of iterations ago.

    Drawn, not argmax: weight by retained-Sigma_r^pw over the `top` best and sample, so repeating
    `sym:auto` from one incumbent is not a deterministic map (iteration 10's lesson) while still
    concentrating on the reflections the measurement likes.

    *** MEASURED AND REFUTED (iteration 12, artifacts/iter12/probe.json, probe2.json). ***
    3 independent rng streams x {47, 61, 37, 81} x 1.5 s. Against the frozen `sym:mx,d2`
    (mean -1.21e-3, hit 0.08) this scores `auto` mean -3.75e-3 hit 0.00 and `auto,auto` mean
    -1.87e-3 hit 0.08; the INVERTED reading (`anti` -- project onto the reflection that costs the
    most Sigma_r, i.e. the biggest structural shake) is no better: mean -3.38e-3 / -3.50e-3,
    hit 0.00 / 0.17. So retained-Sigma_r carries NO information about which projection is
    productive, in either direction -- the reflection the incumbent is already near-invariant
    under is the one that changes nothing, and the one it is farthest from is unrecoverable.
    Kept, tested and documented at ZERO queue weight so a later iteration does not re-derive it;
    the useful part of the finding is in `_mkqueue`, not here.
    """
    cands = [k for k in ("mx", "my", "d1", "d2") if k != avoid]
    sc = []
    for k in cands:
        try:
            s, _, _ = _proj_score(xy, r, k, lsa, linprog, coo_matrix)
        except Exception:
            continue
        if np.isfinite(s):
            sc.append((s, k))
    if not sc:
        return cands[0] if cands else "mx"
    sc.sort(reverse=not anti)
    sc = sc[:max(1, top)]
    v = np.array([s for s, _ in sc], dtype=float)
    w = (v / max(v.max(), 1e-300)) ** pw
    if anti:                      # prefer the projection that COSTS the most = the biggest shake
        w = 1.0 / np.maximum(w, 1e-12)
    w = w / w.sum()
    return sc[int(rng.choice(len(sc), p=w))][1]


def _symexc(xy, r, keys, rng, linprog, coo_matrix, t_end, rounds=4, tail=0.5):
    """DISCRETE homotopy in the SYMMETRY GROUP: converge under g-symmetry, then release.

    Iteration 11's diagnostic (artifacts/iter11/polish.py) settled what the residual gap on the
    eleven live n actually is: a 6-second PURE p=1 ascent from every committed incumbent gains
    1e-9 -- every one of them is already an exact KKT point of the census objective. There is no
    convergence left to buy anywhere; the whole remaining 1e-4 is a wrong contact GRAPH.

    _nexc escapes a basin by changing the circle count. This changes the SYMMETRY instead. The
    optimal square packings overwhelmingly carry a reflection symmetry, and a near-miss local
    optimum is very often a symmetry-BROKEN copy of one: the two halves have drifted into
    different contact graphs and the monotone ascent cannot re-fuse them, because re-fusing means
    moving both halves uphill through a barrier. Projecting onto the symmetric subspace crosses
    that barrier in one step, and it is discrete for the same reason the count excursion is --
    the pairing pi is a combinatorial object, so the projection is not a deformation of anything.

    The ascent is PROJECTED, not merely started symmetric: an LP step off a symmetric point is not
    itself symmetric, so a plain ladder would leak straight back to the asymmetric basin. Each
    round runs a short SLP and re-projects, which converges inside (a neighbourhood of) the
    symmetric subspace -- the "converge at the detour" step that iteration 9 measured to be the
    entire mechanism. Only then is the symmetry RELEASED with a full ladder to p=1.

    `keys` is a SEQUENCE of reflections, so `("d1", "my")` converges g1-symmetric and then
    g2-symmetric before releasing. Iteration 10 measured that on the count axis the STAGE COUNT --
    not the randomizations -- is what carries an excursion (`+2,-1` was worth +0.50 SCORE where
    one-stage round trips had stalled), because a multi-stage detour is not undoable by any single
    inverse. The same argument is if anything stronger here: two reflections generate a larger
    subgroup, so the second stage is a projection onto a subspace the first stage does not
    contain, and the packing that comes back has been through two different contact graphs.
    """
    from scipy.optimize import linear_sum_assignment as lsa
    xy = np.array(xy, dtype=float)
    r = np.array(r, dtype=float)
    if isinstance(keys, str):
        keys = (keys,)
    keys = [k for k in keys if k in _REFL or k in ("auto", "anti")]
    if len(xy) < 3 or not keys:
        return xy, r
    left = t_end - time.process_time()
    if left <= 0.05:
        return xy, r
    mid_total = left * (1.0 - tail)
    sxy, sr = xy, r
    nk = len(keys)
    rounds = max(2, rounds - (nk - 1))
    prev_key = None
    for ki, key in enumerate(keys):
        if key == "auto":
            # resolve HERE, not at queue-build time: stage 2's landscape is the one stage 1 left.
            key = _auto_key(sxy, sr, rng, lsa, linprog, coo_matrix, avoid=prev_key)
        elif key == "anti":
            key = _auto_key(sxy, sr, rng, lsa, linprog, coo_matrix, avoid=prev_key, anti=True)
        prev_key = key
        stage_total = mid_total / float(nk - ki)
        mid_total = max(0.0, mid_total - stage_total)
        for i in range(rounds):
            pxy, pr = _symmetrize(sxy, sr, key, lsa)
            pr = _make_feasible(pxy, np.minimum(pr, _greedy_radii(pxy)))
            share = stage_total / float(rounds - i)
            stage_end = min(t_end, time.process_time() + share)
            stage_total = max(0.0, stage_total - share)
            p = 1.0 if i else float(_mid_ps(rng)[0])
            sxy, sr = _slp(pxy, pr, linprog, coo_matrix, stage_end, p=p)
            if time.process_time() >= t_end:
                break
        if time.process_time() >= t_end:
            break
    sxy, sr = _symmetrize(sxy, sr, prev_key if prev_key in _REFL else "mx", lsa)
    sr = _make_feasible(sxy, np.minimum(sr, _greedy_radii(sxy)))
    return _ladder(sxy, sr, _rand_ladder(rng, 0), linprog, coo_matrix, t_end)


# --- START-QUEUE SAMPLER ----------------------------------------------------------------------
# Excursion tags with a MEASURED nonzero hit rate, and their sampling weights. Iteration 12's
# multi-stream probe (tools/probe.py, 3 independent rng streams x 4 worst live n) replaced every
# earlier single-seed measurement with a DISTRIBUTION, and it says two things:
#   (a) every generator has a NEGATIVE mean delta and a hit rate near 1/3 -- an excursion is a
#       lottery, not an improvement operator; and
#   (b) `lad` (and by iteration 11's polish diagnostic, `ar`) hits ZERO times out of twelve and
#       returns the incumbent to 1e-9, because a converged incumbent is already an exact KKT point
#       of every continuous deformation we own.
# The solver's per-n objective is the MAX over its trials, so a loss costs one slice-second and
# nothing else: mean is the wrong statistic for the policy and hit rate is the right one. A tag
# with hit 0 is worth exactly zero and a tag with hit 1/3 is a free ticket. So the frozen 46-entry
# LIST -- of which a ~4 s slice ever reached the first three, in the same order for every n, one
# draw each -- becomes a SAMPLER WITH REPLACEMENT over the hit-bearing tags: a repeat of the same
# tag is a second independent ticket, which is precisely what iteration 11 found the generators
# to need (its `sym:mx,d2` hit at n=47 was real but one draw in three, and did not reproduce).
# Several two-stage pairs here (mx,my / d1,d2 / my,mx) were never in any queue -- with 20 ordered
# reflection pairs and ~3 excursions per slice, a list can only ever have tried a handful.
_EXC_POOL = (
    ("sym:mx,d2", 4), ("nx:+2,-1", 4), ("sym:d1,my", 3), ("nx:-3,+1", 3),
    ("sym:d2,mx", 2), ("sym:my,d1", 2), ("nx:-1,+2", 2), ("sym:mx,my", 2),
    ("sym:d1,d2", 2), ("sym:d1,mx", 1), ("sym:d2,my", 1), ("sym:my,mx", 1),
    ("sym:d2,d1", 1), ("nx:+3,-2", 1), ("nx:-2,+3", 1), ("nx:+2", 1),
    ("nx:-2", 1), ("sym:my", 1), ("sym:d1", 1),
)
# SOURCE ROTATION -- transfers only. The `eq` (max-min) source was carried from iteration 8 on
# the sole grounds that it had never been measured; iteration 14's src_arm then measured it at
# 0 hits / 12 cells and it was STILL not cut. Iteration 15 measured what it costs and what it
# returns (artifacts/iter15/eqcost.py): 0.22-1.0 CPU-s per slot -- about half a ticket -- and
# every one of 6 cells at n in {47, 37, 81} landed 5.3e-3 to 2.3e+0 BELOW the incumbent. That is
# not a near miss on a ~2e-3 residual, it is a different league, so the slot can never be banked.
# Cutting it converts one half-ticket in three into a DRAW, which iteration 14 measured to be the
# only thing a ticket buys. Transfers stay: they are how one n's gain propagates to its
# neighbours inside a call, and they cost nothing when the source has not moved.
_SRC_Q = ("xf-2", "xf2", "xf-4", "xf4", "xf-6", "xf6")


def _mkqueue(rng, nslots=48, every=3):
    """Sample a per-n start queue: `every` excursion tickets drawn WITH REPLACEMENT, then one
    start SOURCE (eq/xf), repeating. Sources stay a rotation, not a sample -- they are distinct
    origins (a max-min configuration, a neighbour's packing) rather than draws of one lottery,
    and the probe never measured them, so cutting them would be a guess."""
    tags = [t for t, _ in _EXC_POOL]
    w = np.array([float(x) for _, x in _EXC_POOL])
    w = w / w.sum()
    q = []
    si = 0
    while len(q) < nslots:
        idx = rng.choice(len(tags), size=every, p=w)
        q.extend(tags[int(i)] for i in idx)
        q.append(_SRC_Q[si % len(_SRC_Q)])
        si += 1
    return q[:nslots]


def _ticket_end(n, slice_end, deadline, now=None):
    """Deadline for ONE excursion ticket: a fixed n-scaled quantum, never past the slice.

    See the TICKET_* constants for the measurement. Always returns a time strictly in the future
    unless the slice itself is already over, so a ticket is never handed a zero budget by rounding.
    """
    now = time.process_time() if now is None else now
    q = max(TICKET_MIN, TICKET_S * (float(max(n, 1)) / TICKET_N0) ** TICKET_POW)
    return min(slice_end, deadline, now + q)


def _equal_start(seed_xy, linprog, coo_matrix, t_end):
    """Foreign-topology start: equalize `seed_xy` under max-min, hand the result to the sum polish.

    Returns None when the max-min ascent collapses (a degenerate seed with near-coincident
    centres can leave rho at ~1e-4), so the caller falls through to the next start.
    """
    exy, rho = _slp_eq(seed_xy, linprog, coo_matrix, t_end)
    if not np.isfinite(rho) or rho < 1e-4:
        return None
    return exy, _make_feasible(exy, np.minimum(np.full(len(exy), rho), _greedy_radii(exy)))


def _read_pack(n):
    """Warm start from the committed census, or None. GUARDED: missing/odd file -> cold start."""
    path = os.path.join("bench", "packs", "csqv%d.pck" % n)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        rows = [ln.split() for ln in lines[2:]]
        a = np.array([[float(v) for v in row[:3]] for row in rows], dtype=float)
    except Exception:
        return None
    if a.shape != (n, 3) or not np.isfinite(a).all():
        return None
    xy = np.clip(a[:, :2], LO + EDGE, HI - EDGE)
    return xy, np.maximum(a[:, 2], RMIN)


def _cold_start(n, rng):
    """Jittered near-square grid -- structured enough to beat pure noise, still rng-driven."""
    k = int(np.ceil(np.sqrt(n)))
    g = np.linspace(LO + 0.5 / k, HI - 0.5 / k, k)
    gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    pts = pts[rng.permutation(len(pts))[:n]]
    pts = pts + rng.normal(0.0, 0.2 / k, size=pts.shape)
    return np.clip(pts, LO + EDGE, HI - EDGE)


def _rows_start(n, R, stagger, rng, jit=0.0):
    """Structured start: n circles in R rows (counts as even as possible), optional half-row
    offset. Measured (iter 3) as the only basin move that beats the incumbent -- random kicks,
    Metropolis wandering and symmetry-manifold restarts all returned zero gain at n=61."""
    base, extra = divmod(n, R)
    cnt = [base + (1 if j < extra else 0) for j in range(R)]
    rng.shuffle(cnt)
    pts = []
    for j, k in enumerate(cnt):
        if k <= 0:
            continue
        y = LO + (j + 0.5) / R
        h = 0.5 / k
        off = 0.5 * h if (stagger and j % 2) else 0.0
        for x in np.linspace(LO + h, HI - h, k) + off:
            pts.append((x, y))
    a = np.array(pts[:n], dtype=float)
    if jit:
        a = a + rng.normal(0.0, jit, a.shape)
    return np.clip(a, LO + EDGE, HI - EDGE)


def _insert_holes(xy, r, k, g=128, rng=None, top=1):
    """Place k new circles at a LARGE HOLE (max clearance to any circle/wall).

    `top == 1` is the deterministic largest-hole rule (what `_transfer_start` wants). With
    `top > 1` the site is DRAWN from the `top` largest *spatially distinct* holes, weighted by
    clearance^3. Iteration 9 randomized the DELETION (`_drop_idx`) but left this deterministic,
    so every up-excursion from one incumbent built the SAME detour packing and re-derived the
    same fixed point -- the up slots in the start queue were paying for repeats. Making the site
    a discrete random variable is the same fix `_drop_idx` applied to the other direction.
    """
    xy = np.array(xy, dtype=float)
    r = np.array(r, dtype=float)
    ax = np.linspace(LO, HI, g)
    GX, GY = np.meshgrid(ax, ax)
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    wall = np.minimum.reduce([P[:, 0] - LO, HI - P[:, 0], P[:, 1] - LO, HI - P[:, 1]])
    for _ in range(k):
        d = np.sqrt(((P[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]
        room = np.minimum(wall, d.min(1))
        i = int(np.argmax(room))
        if rng is not None and top > 1:
            # greedily collect `top` candidate sites that are pairwise separated (otherwise the
            # neighbours of the argmax cell are all "distinct" sites for the same hole)
            order = np.argsort(-room)[:400]
            cand = []
            for j in order:
                if room[j] <= 0.0:
                    break
                pj = P[j]
                if all(float(np.hypot(*(pj - P[c]))) > float(room[j]) for c in cand):
                    cand.append(int(j))
                if len(cand) >= top:
                    break
            if len(cand) > 1:
                w = np.maximum(room[np.asarray(cand)], 0.0) ** 3
                if w.sum() > 0:
                    i = int(rng.choice(np.asarray(cand), p=w / w.sum()))
        xy = np.vstack([xy, P[i]])
        r = np.append(r, max(float(room[i]), 1e-6))
    return np.clip(xy, LO + EDGE, HI - EDGE), np.maximum(r, RMIN)


def _census_get(census, m):
    """Live best-so-far packing for m: this call's improvement if any, else the committed pack."""
    if m not in census:
        got = _read_pack(m)
        census[m] = None if got is None else [got[0], got[1], float(got[1].sum())]
    return census[m]


def _transfer_start(n, m, census):
    """Structured start for n built from the LIVE census packing of a NEIGHBOUR m.

    m > n: keep the n largest circles.  m < n: insert (n-m) circles at the largest holes.
    Neighbouring n share contact topology, so a neighbour's optimized packing is a far better
    basin than anything sampling produces -- measured at n=61, where 5 of 6 neighbours beat a
    fixed point that iteration 3's ~390 random restarts could not move.

    `census` is the IN-MEMORY best-so-far (see _census_get). solve_driver reverts any write to
    bench/packs/ made during solve(), so re-reading the file gives the STALE pre-iteration pack:
    an ascending disk-only pass had every n borrow from a neighbour as it was last committed, and
    this call's own gains never propagated. Reading the live census is what makes transfers
    compound within one iteration. GUARDED: no entry -> None.
    """
    if m < 2 or m == n:
        return None
    got = _census_get(census, m)
    if got is None:
        return None
    xy, r = got[0], got[1]
    if m > n:
        keep = np.argsort(r)[::-1][:n]
        return xy[keep].copy(), r[keep].copy()
    return _insert_holes(xy, r, n - m)


def _perturb(xy, r, rng):
    """Basin-hopping kick: either shake every centre, or teleport the smallest circles."""
    n = len(xy)
    xy = xy.copy()
    if rng.rand() < 0.5:
        sc = float(r.mean()) * (0.10 + 0.9 * rng.rand())
        xy = xy + rng.normal(0.0, sc, size=xy.shape)
    else:
        k = 1 + int(rng.randint(0, max(1, n // 12)))
        idx = np.argsort(r)[:k]
        xy[idx] = rng.uniform(LO + EDGE, HI - EDGE, size=(k, 2))
    return np.clip(xy, LO + EDGE, HI - EDGE)


def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    t0 = time.process_time()
    deadline = t0 + CPU_BUDGET_S                     # RELATIVE: safe on a 2nd call in one process

    # HEADROOM-WEIGHTED SLICE. The score is mean clamp(-log10(relgap), 0, 7) and after iteration 7
    # eighteen of the 37 visible n sit exactly AT that cap -- every CPU-second spent on them buys
    # literally zero. Read the record table and the committed census (both explicitly allowed) and
    # (a) skip capped n outright, (b) weight the rest by n (a ladder round costs ~n^2) times the
    # digits still available. That hands the 19 uncapped n ~2.3x their old slice. GUARDED at every
    # step: no records file, no pack, or an n outside the census -> full headroom and a cold start,
    # and if EVERY target is capped we fall back to the plain list rather than doing nothing.
    head = {}
    try:
        with open(os.path.join("bench", "records.json")) as fh:
            recs = json.load(fh).get("records", {})
    except Exception:
        recs = {}
    for n_ in targets:
        rv = recs.get(str(n_))
        got_ = _read_pack(n_)
        if rv is None or got_ is None or float(rv) <= 0.0:
            head[n_] = DIGITS_CAP
            continue
        rv = float(rv)
        gap = (rv - float(got_[1].sum())) / rv
        dig = DIGITS_CAP if gap <= 0.0 else min(DIGITS_CAP, -float(np.log10(max(gap, 1e-300))))
        head[n_] = max(0.0, DIGITS_CAP - dig)

    buf = {}
    census = {}          # n -> [xy, r, sum_r] live best (disk pack until this call improves it)
    tried = {}           # n -> {m: sum_r of m when we last transferred from it}

    def flush(n, pack=None, force=False):
        if pack is not None:
            buf.setdefault(n, []).append(np.asarray(pack, dtype=float))
        items = buf.get(n, [])
        if items and (force or len(items) >= FLUSH) and meter.left() > 0:
            evaluate(n, np.stack(items, 0))
            buf[n] = []

    up = sorted(n_ for n_ in targets if head[n_] > CAP_EPS) or sorted(targets)
    wt = {n_: float(n_) * max(head[n_], 0.05) for n_ in up}
    # TWO PASSES. Pass 1 ascending, pass 2 descending over the SAME n: a transfer start is only
    # as good as the neighbour it copies, and pass 1 improves neighbours as it goes, so pass 2
    # re-borrows from sources that have since moved (in either direction). Sources whose sum_r is
    # unchanged since we last used them are skipped, so pass 2 costs nothing where nothing moved.
    passes = [(up, t0 + PASS1_FRAC * CPU_BUDGET_S), (up[::-1], deadline)]
    for order, pass_end in passes:
        for idx, n in enumerate(order):
            now = time.process_time()
            remaining = pass_end - now
            if remaining <= 0.05 or meter.left() <= 0:
                break
            # Adaptive split over the n still to do, weighted by cost x headroom (see `head`
            # above): a ladder round costs ~0.7 s at n=45 and ~3 s at n=99 (chain.py), so an equal
            # split starves the large n, and an n at the digit cap must get nothing at all.
            # Correct at len(order)==1 too (one n takes the whole pass).
            wleft = float(sum(wt[m_] for m_ in order[idx:]))
            slice_end = min(pass_end, now + remaining * (wt[n] / wleft))

            cur = _census_get(census, n)
            if cur is None:
                xy = _cold_start(n, rng)
                r = _greedy_radii(xy)
            else:
                xy, r = cur[0], cur[1]
                r = _make_feasible(xy, np.minimum(r, _lp_radii(xy, linprog, coo_matrix)))

            best = -np.inf
            bxy = br = None
            first = True
            rbase = max(3, int(np.sqrt(n)) - 3)
            nrows = list(range(rbase, int(np.sqrt(n)) + 5))
            seen = tried.setdefault(n, {})
            # Start queue, best-measured-first. 'eq' = a MAX-MIN (equal-circle) configuration,
            # 'xf' = a transfer from a neighbour's census packing. The two are the only start
            # sources ever measured to beat a converged incumbent; eq seeds are interleaved with
            # transfers so a per-n slice that only affords 3-4 starts gets one of each kind.
            # 'ar' = a CONTAINER (aspect) homotopy, the constraint-side twin of 'lad'. Measured
            # (artifacts/iter8/aprobe.py, afine.py): it snaps back or loses at most n, but it hits
            # where the p-ladder has already reached its fixed point (n=61 +1.8e-3, +5.1e-4), so
            # it earns a few diversity slots BEHIND the ladders, not in front of them.
            # 'nx:...' = a DISCRETE circle-count excursion (_nexc), promoted to the front
            # after artifacts/iter9/nprobe.py measured it hitting on 4 of the 5 worst n
            # (45 +3.1e-3, 61 +3.6e-3, 81 +2.4e-3, 47 +1.9e-3) in the same seconds where a
            # further random p-ladder returned <= 0 on all five -- the continuous generators are
            # on their fixed point and the discrete one is not.
            # 'nx:<d1[,d2...]>' = a multi-stage DISCRETE count excursion: converge at n+d1,
            # then at n+d2, ... then back at n. Single-stage entries are iteration 9's measured
            # winners; the two-stage ones (iteration 10) sit behind them.
            # measured (artifacts/iter10/mprobe.py, 5 s/trial, 4 worst live n): the two-stage
            # detour '+2,-1' is the only variant that hit n=29 (+2.94e-3, the WHOLE remaining
            # gap) where iteration 9's one-stage round trip lost on every k and both directions,
            # and it also beat the best one-stage trial at n=81 (+2.81e-3 vs +1.87e-3). So it
            # leads; '-3,+1' (also a hit at 81) and the one-stage winners follow.
            # 'sym:<g1[,g2]>' = a DISCRETE SYMMETRY excursion (_symexc): converge projected onto
            # the g1-symmetric subspace, then g2, then release. Iteration 11's polish diagnostic
            # (artifacts/iter11/polish.py) showed every live incumbent is already a KKT point to
            # 1e-9, so the residual is purely a wrong contact GRAPH; measured
            # (artifacts/iter11/sprobe.py, 5 s/trial) the two-stage symmetry detour cracks exactly
            # the two n that NO generator had ever moved -- n=47 3.067 -> 7.000 ('mx,d2', the
            # whole gap and a record beat) and n=41 3.105 -> 4.033 ('d1,my') -- where nx:+2,-1
            # lost on both. Single-stage 'my' also hit n=61 (+1.28e-3). So the two-stage sym
            # entries lead alongside the count excursions, single-stage ones behind them.
            # ORDER/SELECTION is no longer frozen here: iteration 12 replaced the 46-entry list
            # with `_mkqueue`, a SAMPLER over `_EXC_POOL` (see there for the multi-stream
            # measurement that motivates it). 'lad'/'ar' are gone from the rotation -- the fixed
            # probe hit them 0/12 -- and tags now REPEAT with fresh draws.
            queue = _mkqueue(rng)
            neq = 0
            nlad = 0
            nar = 0
            trial = 0
            while True:
                now = time.process_time()
                if now >= slice_end or meter.left() <= 0:
                    break
                pxy = pr = None
                if first:
                    sxy, sr = xy, r
                    first = False
                elif queue:
                    tag = queue.pop(0)
                    if tag == "lad":
                        # chain the homotopy off the RUNNING BEST -- re-applying it is a fixed
                        # point map, so a repeat that gains nothing costs one round and stops.
                        if bxy is None:
                            continue
                        pxy, pr = _ladder(bxy, br, _rand_ladder(rng, nlad), linprog,
                                          coo_matrix, min(slice_end, deadline))
                        nlad += 1
                    elif tag[:4] == "sym:":
                        if bxy is None:
                            continue
                        pxy, pr = _symexc(bxy, br, tuple(tag[4:].split(",")),
                                          rng, linprog, coo_matrix,
                                          _ticket_end(n, slice_end, deadline))
                    elif tag[:3] == "nx:":
                        if bxy is None:
                            continue
                        pxy, pr = _nexc(bxy, br, tuple(int(s) for s in tag[3:].split(",")),
                                        rng, linprog, coo_matrix,
                                        _ticket_end(n, slice_end, deadline))
                    elif tag[0] == "a":
                        if bxy is None:
                            continue
                        pxy, pr = _arect(bxy, br, _rand_arect(rng, nar), linprog,
                                         coo_matrix, min(slice_end, deadline))
                        nar += 1
                    elif tag[0] == "e":
                        # seed the max-min ascent: first from the incumbent's own centres (the
                        # single strongest variant measured), then from structured lattices.
                        if neq == 0:
                            seed = (bxy if bxy is not None else xy).copy()
                        elif neq % 2 == 1:
                            seed = _rows_start(n, max(3, int(round(np.sqrt(n))) + (neq // 2) - 1),
                                               neq % 4 == 1, rng, 0.05 / np.sqrt(n))
                        else:
                            seed = _cold_start(n, rng)
                        neq += 1
                        got = _equal_start(seed, linprog, coo_matrix, min(slice_end, deadline))
                        if got is None:
                            continue                  # degenerate max-min ascent -> next start
                        sxy, sr = got
                    else:
                        m = n + int(tag[2:])
                        src_ = _census_get(census, m)
                        if src_ is None or seen.get(m) == src_[2]:
                            continue                  # no pack, or that source has not moved
                        seen[m] = src_[2]
                        got = _transfer_start(n, m, census)
                        if got is None:
                            continue
                        sxy, sr = got
                        sr = np.minimum(sr, _greedy_radii(sxy))
                elif trial % 2 == 0:                  # structured lattice restart
                    q = trial // 2
                    R = nrows[q % len(nrows)]
                    cyc = q // len(nrows)
                    sxy = _rows_start(n, R, cyc % 2 == 1, rng,
                                      0.0 if cyc < 2 else 0.15 / np.sqrt(n))
                    sr = _greedy_radii(sxy)
                    trial += 1
                else:                                 # kick the incumbent
                    sxy = _perturb(bxy, br, rng)
                    sr = _greedy_radii(sxy)
                    trial += 1
                if pxy is None:
                    pxy, pr = _slp(sxy, sr, linprog, coo_matrix, min(slice_end, deadline))
                pr = np.maximum(pr, _lp_radii(pxy, linprog, coo_matrix))  # exact radii, these centres
                pr = _make_feasible(pxy, pr)
                s = float(pr.sum())
                if s > best:
                    best, bxy, br = s, pxy.copy(), pr.copy()
                    flush(n, np.concatenate([pxy, pr[:, None]], 1))
            # publish to the live census so every LATER n in this call transfers from the improvement
            if bxy is not None and (census.get(n) is None or best > census[n][2]):
                census[n] = [bxy, br, best]
            flush(n, force=True)
    for n in list(buf):
        flush(n, force=True)


# ---------------------------------------------------------------------------------------------
def _ticket_selftest():
    """The ticket quantum must be finite, ordered in n, floored, and never past the slice."""
    now = 100.0
    q37 = _ticket_end(37, now + 99, now + 99, now) - now
    q61 = _ticket_end(61, now + 99, now + 99, now) - now
    q81 = _ticket_end(81, now + 99, now + 99, now) - now
    assert TICKET_MIN <= q37 < q61 < q81, (q37, q61, q81)
    assert abs(q61 - TICKET_S) < 1e-12, q61
    assert q37 >= TICKET_MIN - 1e-12 and _ticket_end(3, now + 99, now + 99, now) - now >= TICKET_MIN
    # never past the slice or the hard deadline, even for a huge n
    assert _ticket_end(999, now + 0.2, now + 99, now) == now + 0.2
    assert _ticket_end(999, now + 99, now + 0.1, now) == now + 0.1
    # an already-expired slice yields a non-future deadline (caller's `>= t_end` guards fire)
    assert _ticket_end(61, now - 1.0, now + 99, now) == now - 1.0
    print("ticket-quantum self-test OK  (q37=%.2fs q61=%.2fs q81=%.2fs)" % (q37, q61, q81))


def _queue_selftest():
    """The sampled start queue must contain ONLY live tag families: excursion tickets drawn from
    `_EXC_POOL` and `xf` transfer sources. Iteration 15 cut the `eq` source after measuring it at
    0 hits / 12 cells (iter14 src_arm) and 5.3e-3 .. 2.3e+0 BELOW the incumbent on every one of 6
    cells (artifacts/iter15/eqcost.py); this asserts the cut and that every surviving source tag
    still parses into the neighbour offset solve() reads off it."""
    pool = {t for t, _ in _EXC_POOL}
    rng = np.random.RandomState(11)
    q = _mkqueue(rng)
    assert len(q) == 48
    nsrc = 0
    for tag in q:
        if tag in pool:
            continue
        assert tag[:2] == "xf", "dead source tag in the queue: %r" % (tag,)
        assert int(tag[2:]) != 0                      # solve() does n + int(tag[2:])
        nsrc += 1
    assert nsrc >= 10, "sources vanished from the rotation (%d)" % nsrc
    assert set(_SRC_Q) == {"xf-2", "xf2", "xf-4", "xf4", "xf-6", "xf6"}
    print("queue self-test OK  (%d slots, %d transfer sources, no dead eq slots)" % (len(q), nsrc))


def _self_test():
    """Offline check: feasibility of what we emit, and that a SECOND solve() call in the same
    process still produces a feasible packing (the CPU_STOP-as-absolute-constant trap)."""
    class _Meter:
        def __init__(self, b):
            self.budget, self.used = b, 0

        def tick(self, k=1):
            g = max(0, min(int(k), self.budget - self.used))
            self.used += g
            return g

        def left(self):
            return max(0, self.budget - self.used)

    seen = {}

    def make_eval(meter):
        def ev(n, packing):
            a = np.asarray(packing, dtype=float)
            single = a.ndim == 2
            if single:
                a = a[None]
            assert a.shape[2] == 3 and a.shape[1] == n
            B = a.shape[0]
            grant = meter.tick(B)
            feas = np.zeros(B, bool)
            sr = np.full(B, -np.inf)
            for i in range(min(grant, B)):
                x, y, r = a[i, :, 0], a[i, :, 1], a[i, :, 2]
                d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
                np.fill_diagonal(d, np.inf)
                ok = bool((r > 0).all())
                ok &= float(np.minimum.reduce([x - r - LO, HI - x - r,
                                               y - r - LO, HI - y - r]).min()) >= -1e-9
                ok &= float((d - (r[:, None] + r[None, :])).min()) >= -1e-9
                feas[i] = ok
                sr[i] = r.sum()
                if ok and sr[i] > seen.get(n, (-np.inf,))[0]:
                    seen[n] = (float(sr[i]), a[i].copy())
            return (bool(feas[0]), float(sr[0])) if single else (feas, sr)
        return ev

    global CPU_BUDGET_S
    CPU_BUDGET_S = 4.0
    tg = [9, 28]
    for call in (1, 2):
        seen.clear()
        m = _Meter(20000)
        solve(make_eval(m), m, np.random.RandomState(call), tg)
        for n in tg:
            assert n in seen, "call %d: no feasible packing for n=%d" % (call, n)
            assert len(seen[n][1]) == n
            print("  call %d  n=%-3d sum_r=%.6f  (evals used %d)" % (call, n, seen[n][0], m.used))
    # single-target call must also work (per-n split must survive len(targets)==1)
    seen.clear()
    m = _Meter(20000)
    solve(make_eval(m), m, np.random.RandomState(7), [31])
    assert 31 in seen, "single-target call produced nothing"
    print("  single-target n=31 sum_r=%.6f" % seen[31][0])
    # missing-n guard: an n with no committed pack must cold-start, not raise
    seen.clear()
    m = _Meter(20000)
    solve(make_eval(m), m, np.random.RandomState(3), [12])
    assert 12 in seen, "cold start for an uncommitted n produced nothing"
    print("  cold-start n=12 sum_r=%.6f" % seen[12][0])
    # exhausted meter must not hang or raise
    seen.clear()
    m = _Meter(0)
    solve(make_eval(m), m, np.random.RandomState(5), [15])
    print("  zero-budget call returned cleanly")
    _ticket_selftest()
    _queue_selftest()
    print("SELF-TEST: PASS")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
