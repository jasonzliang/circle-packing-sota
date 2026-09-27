"""Circle-packing solver: maximise the sum of radii of n circles in the unit square.

Contract (MISSION.md): ``solve(evaluate, meter, rng, targets)``; every candidate that is to count is
routed through ``evaluate``; ``rng`` is the only randomness source; CPU deadlines are derived at entry
from ``time.process_time()`` and from the sizes actually in ``targets``.

Method -- sequential linear programming (SLP) with a *conservative* linearisation:

*   For FIXED centres, ``max sum r  s.t.  r_i + r_j <= d_ij,  0 <= r_i <= wall_i`` is an exact LP in r
    alone.  This alone is a large win over the equal-radius rule ``r_i = min(wall_i, min_j d_ij/2)``:
    the sum-of-radii optimum is genuinely *uneven* (big circles paid for by small neighbours).
*   For MOVING centres the only nonconvexity is ``r_i + r_j <= ||c_i - c_j||``.  Because
    ``||c_i - c_j|| >= u . (c_i - c_j)`` for ANY unit vector u, replacing the norm by the unit vector
    of the current iterate gives a linear *inner* approximation: every LP solution is feasible for the
    true problem, and the current iterate is itself LP-feasible, so each step is monotone.  A box
    trust region on the centres keeps the linearisation tight and is grown on a successful step; it is
    NOT shrunk on a failed one, because the LP value is nondecreasing in the box radius, so a step
    that fails to gain certifies that no smaller box can gain either (see `_slp`).

Both LPs are pruned exactly (never heuristically): a pair constraint is dropped only when a valid
upper bound on ``r_i + r_j`` after the step already implies it.  Bound used: ``r_i <= wall_i`` and
``r_i <= min_j d_ij`` (since r_j > 0), each relaxed by the trust radius.

The SLP alone is not enough: it is monotone-feasible, so it stops at the first first-order point of
whatever basin it starts in (SLSQP from the same point agrees, converging in one iteration), and those
points sit ~1e-2 relative below the records at every n.  So the workhorse is a *penalty* local solve
(``_penalty``): minimise ``-sum r + mu * sum max(0, violation)^2`` with L-BFGS-B over a ramp of mu, which
lets circles overlap by O(1/mu) so contacts can break and re-form -- and costs ~1/50th of one SLP
convergence.  ``_slp`` is then the exact finisher that turns those approximate centres into a strictly
feasible packing.  Global search is many such penalty+polish attempts: warm start from the committed
census, CROSS-SIZE TRANSFER from the committed packings of neighbouring sizes (drop the runts when
transferring down, fill the roomiest holes when transferring up), cold multi-start, then basin hopping.
The hop chain is THRESHOLD-ACCEPTING, not a ratchet: each hop is built from a chain state that moves
to any local optimum within a relative `eps` of the best (`_accept`), because the sizes still short
of the record are all certified first-order points -- their gap is a topology that is not uphill from
the incumbent, so a search that only ever rebuilds from the incumbent cannot reach it.
Half the hops are ``_inflate``: both `_penalty` and `_slp` MAXIMISE, so they can only find the local
optimum they start beside, and once the census sits at its local optima every maximising move gains
exactly 0.  `_inflate` inverts the question -- it PINS sum(r) at (1+eps) times the incumbent and
minimises overlap alone, so there is no ``-sum r`` term holding contacts together and the topology
reorganises.  It is itself deterministic, so the hop JITTERS the incumbent first (see `_solve_one`);
without that, repeating the move recomputes the same handful of points.  The other half is the screened perturbation wheel (windowed shake / ruin-and-recreate /
dilation / a fresh transfer).  Every warm start carries its own
RADII into the penalty solve, which is what keeps a hop local instead of re-melting the packing.  The
split is driven purely by the CPU clock, so the same code fills a 3 s slice and a 60 s offline call.

Self-test: ``python3 tools/solver.py --self-test`` (calls solve() TWICE in one process).
"""
import json
import os
import time

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix, vstack as spvstack

LO, HI = -0.5, 0.5

# the two wall signs, shaped (2, 1, 1) to broadcast against (2, n) coordinates: the four
# containment half-planes are sgn * coord + r <= HI for sgn in {+1, -1} and coord in {x, y}.
_WSGN = np.array([1.0, -1.0]).reshape(2, 1, 1)

# The scorer clamps each size at 7 digits of closed relative gap (MISSION.md / bench/anchor.json
# "digits_cap"), so a size's remaining PRIZE -- what the clock allocation below is weighted by --
# is bounded by this, not by its instantaneous digit gradient.
DIGITS_CAP = 7.0

# CPU allowance per solve() call, derived from the number of targets (never a module-level absolute
# deadline -- a later call in the same process must get a fresh one).  One target => the offline
# held-out regime (60 s process-CPU per size); many targets => the in-loop driver (120 s for the lot).
CPU_ONE_TARGET = 54.0
CPU_MANY_TARGETS = 104.0

_LP = dict(method="highs")


def _walls(xy):
    x, y = xy[:, 0], xy[:, 1]
    return np.minimum.reduce([x - LO, HI - x, y - LO, HI - y])


def _dist(xy):
    d = xy[:, None, :] - xy[None, :, :]
    D = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(D, np.inf)
    return D


def _repair(xy, r):
    """Shave r so the packing is strictly feasible under the scorer's geometry (slack >= -1e-9)."""
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    r = np.minimum(r, np.maximum(_walls(xy), 0.0))
    D = _dist(xy)
    v = float((r[:, None] + r[None, :] - D).max())
    if v > 0.0:
        r = r - (0.5 * v + 1e-13)
    return np.maximum(r, 1e-12)


def _radius_lp(xy):
    """Exact max-sum-radii for FIXED centres.  Returns r (n,) or None if the LP failed."""
    n = xy.shape[0]
    wall = np.maximum(_walls(xy), 0.0)
    D = _dist(xy)
    I, J = np.triu_indices(n, 1)
    dall = D[I, J]
    # Valid upper bound on r_i: r_i <= wall_i, and r_i + r_j <= d_ij with r_j > 0 gives
    # r_i <= min_j d_ij.  A pair with d_ij >= U_i + U_j therefore CANNOT bind -- an exact prune,
    # and a far tighter one than wall_i + wall_j (354 vs 1616 pairs at n=99).
    U = np.minimum(wall, D.min(1))
    keep = dall < U[I] + U[J]
    bnds = np.column_stack([np.zeros(n), wall])
    for _ in range(4):
        i, j, dij = I[keep], J[keep], dall[keep]
        m = i.size
        A = b = None
        if m:
            rows = np.repeat(np.arange(m), 2)
            cols = np.stack([i, j], 1).ravel()
            A = coo_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n)).tocsc()
            b = dij
        try:
            res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bnds, **_LP)
        except Exception:
            return None
        if not res.success or res.x is None:
            return None
        r = np.asarray(res.x, dtype=float)
        # a-posteriori check: any DROPPED pair that the solution violates is added back and the LP
        # re-solved, so the pruned LP's optimum is the full LP's optimum, never an approximation.
        bad = (~keep) & (r[I] + r[J] > dall + 1e-12)
        if not bad.any():
            return r
        keep = keep | bad
    return r


def _centre_lp(xy, r, delta):
    """One trust-region SLP step over (x, y, r).  Returns new centres (n,2) or None."""
    n = xy.shape[0]
    wall = _walls(xy)
    D = _dist(xy)
    nnd = D.min(1)
    # Valid post-step upper bound on r_i: r_i <= wall_i (grows by <= delta) and r_i <= min_j d_ij
    # (shrinks by <= 2*delta when both centres move by delta).
    R = np.maximum(np.minimum(wall + delta, nnd + 2.0 * delta), 0.0)

    i, j = np.triu_indices(n, 1)
    dij = D[i, j]
    keep = (dij - 2.0 * delta) < (R[i] + R[j])       # exact prune: dropped pairs cannot bind
    i, j, dij = i[keep], j[keep], dij[keep]
    m = i.size

    blocks, rhs = [], []
    if m:
        ux = (xy[i, 0] - xy[j, 0]) / dij
        uy = (xy[i, 1] - xy[j, 1]) / dij
        rows = np.repeat(np.arange(m), 6)
        cols = np.stack([i, n + i, j, n + j, 2 * n + i, 2 * n + j], 1).ravel()
        one = np.ones(m)
        vals = np.stack([-ux, -uy, ux, uy, one, one], 1).ravel()
        blocks.append(coo_matrix((vals, (rows, cols)), shape=(m, 3 * n)))
        rhs.append(np.zeros(m))

    # containment: +-x_k + r_k <= 0.5, +-y_k + r_k <= 0.5
    k = np.arange(n)
    wrows = np.repeat(np.arange(4 * n), 2)
    wcols = np.stack([k, 2 * n + k, k, 2 * n + k,
                      n + k, 2 * n + k, n + k, 2 * n + k], 1).reshape(4 * n, 2).ravel()
    wvals = np.tile(np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]), n).reshape(4 * n, 2).ravel()
    blocks.append(coo_matrix((wvals, (wrows, wcols)), shape=(4 * n, 3 * n)))
    rhs.append(np.full(4 * n, HI))

    A = spvstack(blocks).tocsc()
    b = np.concatenate(rhs)

    lo = np.concatenate([np.maximum(xy[:, 0] - delta, LO), np.maximum(xy[:, 1] - delta, LO),
                         np.zeros(n)])
    hi = np.concatenate([np.minimum(xy[:, 0] + delta, HI), np.minimum(xy[:, 1] + delta, HI), R])
    c = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=np.column_stack([lo, hi]), **_LP)
    except Exception:
        return None
    if not res.success or res.x is None:
        return None
    z = np.asarray(res.x, dtype=float)
    return np.clip(np.stack([z[:n], z[n:2 * n]], 1), LO, HI)


# ---------------------------------------------------------------- penalty local solve

def _penalty(xy, deadline, stages=6, it=110, r0=None):
    """Local solve that walks THROUGH slight overlap: minimise -sum r + mu * sum max(0, viol)^2.

    The feasible-path SLP is monotone, so it can only ever slide downhill inside the basin it starts
    in, and it halts at a genuine first-order point (SLSQP from the same point also reports
    convergence in one iteration).  Letting circles overlap by O(1/mu) and ramping mu lets contacts
    break and re-form, which is what crosses basins -- and one L-BFGS-B solve costs ~1/50th of one
    SLP convergence, so the same slice buys ~50x the local solves.  The result is only
    APPROXIMATELY feasible; `_slp` (exact radius LP + `_repair`) is what turns it into a packing.

    mu is derived from the problem scale, not tuned: with r_typ = 0.5/sqrt(n) the stationary overlap
    is ~1/(2 mu), so mu_lo admits ~0.1 r_typ of overlap and mu_hi ~1e-7 r_typ.

    `r0` seeds the radius block.  The default flat `0.9 r_typ` is right for a cold start and WRONG
    for every warm one: the optimum's radii span ~0.5..1.4 r_typ, so flattening them re-melts the
    whole packing and a perturbation of a good incumbent stops being a local move.  Callers that
    have radii (the incumbent, a transferred size, an insertion radius) pass them in.
    """
    n = xy.shape[0]
    I, J = np.triu_indices(n, 1)
    r_typ = 0.5 / np.sqrt(n)
    if r0 is None:
        rr0 = np.full(n, 0.9 * r_typ)
    else:
        rr0 = np.clip(np.asarray(r0, dtype=float).reshape(-1), 1e-9, 0.5)
    z = np.concatenate([xy[:, 0], xy[:, 1], rr0])
    bnds = np.array([[LO, HI]] * (2 * n) + [[0.0, 0.5]] * n)

    def fg(z, mu):
        xy2 = z[:2 * n].reshape(2, n)
        rr = z[2 * n:]
        # WALL overlap for all four half-planes in ONE expression.  The old code ran a Python loop
        # over (+x, -x, +y, -y), each iteration boolean-indexing its own active set -- ~60 of the
        # ~100 numpy calls in this gradient, on arrays of length n.  Stacking the two signs against
        # the two axes is the identical function of z (same four terms, same gradient) in ~8 calls,
        # which is what a size-33 solve, where every array is tiny and the cost is per-call
        # overhead rather than per-element, actually pays for.  It also runs unconditionally, so
        # the float accumulators it seeds are what the pair scatter below adds into.
        ow = np.maximum(rr + _WSGN * xy2[None, :, :] - HI, 0.0)
        cw = (2.0 * mu) * ow
        f = -rr.sum() + mu * float((ow * ow).sum())
        gr = cw.sum((0, 1)) - 1.0
        gw = (_WSGN * cw).sum(0)
        # PAIR overlap.  The active set is carved out with flatnonzero (only ~3n of the n(n-1)/2
        # pairs overlap at all, and a dense max(0, .) formulation measured 1.6x SLOWER at n=99
        # because its per-element work grows as n^2 -- artifacts/probe35.py), but the scatter back
        # to circles is np.bincount rather than np.add.at, which costs ~50x per element for the
        # same sum.
        dx = xy2[0][I] - xy2[0][J]
        dy = xy2[1][I] - xy2[1][J]
        d = np.sqrt(dx * dx + dy * dy) + 1e-300
        k = np.flatnonzero(rr[I] + rr[J] > d)
        ia, ja = I[k], J[k]
        o = rr[ia] + rr[ja] - d[k]
        c = (2.0 * mu) * o
        f += mu * float(o @ o)
        cd = c / d[k]
        px, py = cd * dx[k], cd * dy[k]
        gr += np.bincount(ia, c, n) + np.bincount(ja, c, n)
        gx = gw[0] + np.bincount(ja, px, n) - np.bincount(ia, px, n)
        gy = gw[1] + np.bincount(ja, py, n) - np.bincount(ia, py, n)
        return f, np.concatenate([gx, gy, gr])

    lo, hi = 1.0 / (0.2 * r_typ), 1.0 / (2e-7 * r_typ)
    for mu in np.geomspace(lo, hi, stages):
        if time.process_time() >= deadline:
            break
        try:
            res = minimize(fg, z, args=(float(mu),), jac=True, method="L-BFGS-B", bounds=bnds,
                           options={"maxiter": it, "maxcor": 20})
        except Exception:
            break
        if res.x is not None and np.isfinite(res.x).all():
            z = res.x
    return np.clip(np.stack([z[:n], z[n:2 * n]], 1), LO, HI)


def _inflate(xy, r, s_target, deadline, stages=5, it=140):
    """DEMAND a higher total radius and repair the overlap: min sum viol^2 with sum(r) PINNED to
    `s_target`.  Returns (xy, r), approximately feasible.

    Every other move here perturbs the incumbent and then hands it to `_penalty`, i.e. to
    ``min -sum r + mu viol^2``, which descends to the nearest KKT point -- the search only ever asks
    "what is the local maximum near this configuration", and once the census is at its local optima
    the answer is "the one you already have" (measured: shake/relocate/swap/radius-jitter gain
    exactly 0).  Inflation asks the opposite and far better-behaved question: fix the objective at a
    value strictly ABOVE the incumbent and look for ANY arrangement that achieves it.  That
    landscape's global minimum is 0 and is *attained* whenever a packing at that sum exists, and --
    the point -- there is no ``-sum r`` term pulling contacts together, so circles slide through one
    another while overlapping and the contact topology reorganises.  This is the move that reaches a
    different topology, which is what the previous three iterations' bottleneck asked for.

    The sum is pinned EXACTLY, not by a penalty, by the reparametrisation ``r = s_target * u/sum(u)``
    with ``u = z^2`` (so r >= 0 and sum(r) = s_target hold identically for any z).  The chain rule
    collapses to ``dF/dz = 2 z (s_target/T) (g - (g.r)/s_target)`` with ``g = dF/dr``.

    `stages` ramps a weak pull toward the even spread ``s_target/n`` from 1 down to 1e-4.  With the
    sum pinned, nothing stops the parametrisation from starving one circle to feed another; the ramp
    starts by asking for a spread the walls can actually support and then releases, so the early
    stages relocate mass between circles (a topology move) and the late ones are pure overlap
    repair.  A/B'd against pure overlap repair (artifacts/probe24.py).
    """
    n = xy.shape[0]
    I, J = np.triu_indices(n, 1)
    z0 = np.sqrt(np.maximum(np.asarray(r, dtype=float).reshape(-1), 1e-12))
    v = np.concatenate([xy[:, 0], xy[:, 1], z0])
    bnds = np.array([[LO, HI]] * (2 * n) + [[1e-5, 1.0]] * n)

    def fg(v, w):
        x, y, z = v[:n], v[n:2 * n], v[2 * n:]
        u = z * z
        T = u.sum() + 1e-300
        rr = s_target * u / T
        # walls in one expression, pairs over the flatnonzero active set -- see `_penalty.fg`.
        # Unlike there, rr is NOT bounded by 0.5 here (the reparametrisation can starve every
        # circle to feed one), so both walls of an opposing pair can be violated at once; keeping
        # all four half-planes explicit rather than folding them to |coord| is what stays exact.
        xy2 = np.stack([x, y])
        ow = np.maximum(rr + _WSGN * xy2[None, :, :] - HI, 0.0)
        cw = 2.0 * ow
        f = float((ow * ow).sum())
        gr = cw.sum((0, 1))
        gw = (_WSGN * cw).sum(0)
        dx = x[I] - x[J]
        dy = y[I] - y[J]
        d = np.sqrt(dx * dx + dy * dy) + 1e-300
        k = np.flatnonzero(rr[I] + rr[J] > d)
        ia, ja = I[k], J[k]
        o = rr[ia] + rr[ja] - d[k]
        c = 2.0 * o
        f += float(o @ o)
        cd = c / d[k]
        px, py = cd * dx[k], cd * dy[k]
        gr += np.bincount(ia, c, n) + np.bincount(ja, c, n)
        gx = gw[0] + np.bincount(ja, px, n) - np.bincount(ia, px, n)
        gy = gw[1] + np.bincount(ja, py, n) - np.bincount(ia, py, n)
        if w > 0.0:
            f += w * float(((rr - s_target / n) ** 2).sum())
            gr += 2.0 * w * (rr - s_target / n)
        gz = 2.0 * z * (s_target / T) * (gr - float(gr @ rr) / s_target)
        return f, np.concatenate([gx, gy, gz])

    for w in np.geomspace(1.0, 1e-4, stages):
        if time.process_time() >= deadline:
            break
        try:
            res = minimize(fg, v, args=(float(w),), jac=True, method="L-BFGS-B", bounds=bnds,
                           options={"maxiter": it, "maxcor": 20})
        except Exception:
            break
        if res.x is not None and np.isfinite(res.x).all():
            v = res.x
    u = v[2 * n:] ** 2
    return (np.clip(np.stack([v[:n], v[n:2 * n]], 1), LO, HI),
            np.maximum(s_target * u / (u.sum() + 1e-300), 1e-12))


def _accept(ss, best_s, eps):
    """THRESHOLD ACCEPTING for the hop chain: does a landed local optimum of sum `ss` become the
    chain's next state, given the best sum found so far and the hop's own relative step `eps`?

    Accept anything within a relative `eps` BELOW the best.  Accepting slightly-worse landings is
    the point -- `probe44.py` shows every short size in the census is a certified first-order point,
    so the topology that would close it is not uphill from the incumbent and the route there runs
    through configurations that are worse before they are better.  Measuring the band from `best_s`
    rather than from the chain's own sum is what keeps that from becoming a downhill random walk:
    the chain wanders the local optima within a relative `eps` of the best and is pulled back the
    moment the best improves, whereas a band around the chain would let ~50 hops compound their
    thresholds into a several-percent drift.
    """
    return float(ss) > float(best_s) * (1.0 - float(eps)) - 1e-15


def _keep_transferring(ss, best_s, had_incumbent):
    """Should the cross-size transfer sweep try ANOTHER neighbouring size, having just landed `ss`?

    Run while it pays.  With no incumbent at entry (`had_incumbent` False -- the cold held-out
    regime) the sweep always runs in full: a transfer is the strongest start available there, and the
    first one sets the best, so gating on "did it improve" would be circular.  With an incumbent
    already committed, stop at the first transfer that fails to improve on the best -- measured
    (artifacts/probe47.py), at all 8 sizes still short of the record every transfer from every one
    of the two nearest neighbours lands ~1 digit below the incumbent, so the rest of the sweep is
    known-worthless clock.  No threshold and no count: the rule is the same one `push` already uses.
    """
    if not had_incumbent:
        return True
    return float(ss) >= float(best_s) - 1e-15


# ---------------------------------------------------------------- starts

def _pck_path(n):
    for base in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        p = os.path.join(base, "bench", "packs", "csqv%d.pck" % n)
        if os.path.exists(p):
            return p
    return None


def _read_pck(n):
    """Read the committed census entry for n as (xy, r), if there is one.  GUARDED: any miss => None.

    The radii matter as well as the centres: the transfer move below needs to know which circles of a
    neighbouring size's packing are the runts worth dropping.
    """
    p = _pck_path(n)
    if p is None:
        return None
    try:
        with open(p, "r") as fh:
            lines = fh.read().splitlines()
        rows = []
        for ln in lines[2:]:
            parts = ln.split()
            if len(parts) >= 3:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        if len(rows) != n:
            return None
        a = np.array(rows, dtype=float)
        if not np.isfinite(a).all():
            return None
        return np.clip(a[:, :2], LO, HI), np.maximum(a[:, 2], 0.0)
    except Exception:
        return None


def _warm_start(n):
    """Centres of the committed census entry for n, or None (cold start)."""
    got = _read_pck(n)
    return None if got is None else got[0]


def _relgap(n):
    """Relative gap of the COMMITTED pack for n to the frozen record, or None if either is unreadable.

    GUARDED both ways: a fresh n, a size outside the census, or an unreadable records table all give
    None, and the caller then falls back to the size-proportional share.
    """
    got = _read_pck(n)
    if got is None:
        return None
    try:
        with open(os.path.join(os.getcwd(), "bench", "records.json"), "r") as fh:
            rec = json.load(fh).get("records", {}).get(str(int(n)))
    except Exception:
        return None
    if rec is None or not np.isfinite(float(rec)) or float(rec) <= 0.0:
        return None
    return max(0.0, (float(rec) - float(got[1].sum())) / float(rec))


def _grid_start(n, rng, rows=None):
    """Even rows of evenly spaced centres -- the generic 'grid-ish' basin, jittered."""
    if rows is None:
        rows = max(1, int(round(np.sqrt(n))))
    rows = min(rows, n)
    counts = np.full(rows, n // rows)
    counts[:n % rows] += 1
    pts = []
    for ri in range(rows):
        c = int(counts[ri])
        yy = LO + (ri + 0.5) / rows
        for ci in range(c):
            pts.append((LO + (ci + 0.5) / c, yy))
    xy = np.array(pts, dtype=float)
    xy += rng.normal(0.0, 0.25 / np.sqrt(n), size=xy.shape)
    return np.clip(xy, LO + 1e-4, HI - 1e-4)


def _random_start(n, rng):
    return rng.uniform(LO + 1e-3, HI - 1e-3, size=(n, 2))


def _hole_points(xy, r, count, rng, grid=41, top=8):
    """`count` centres at the roomiest holes of the partial packing (xy, r), WITH their radii.

    A point's insertion radius is its distance to the nearest circle's boundary, capped by the wall.
    Scoring a coarse grid by it and sampling one of the best `top` cells (plus a sub-cell jitter) is
    the generic "where does another circle still fit" move.  Each pick is appended with its own
    insertion radius before the next is chosen, so two new circles cannot land in the same hole.

    Returns ``(pts, pts_r)``: that insertion radius is also the right RADIUS SEED for the new circle,
    so `_penalty` can start from it instead of from a flat guess.
    """
    t = np.linspace(LO, HI, grid)
    gx, gy = np.meshgrid(t, t)
    cand = np.stack([gx.ravel(), gy.ravel()], 1)
    cwall = np.minimum.reduce([cand[:, 0] - LO, HI - cand[:, 0], cand[:, 1] - LO, HI - cand[:, 1]])
    P = np.asarray(xy, dtype=float).reshape(-1, 2).copy()
    R = np.maximum(np.asarray(r, dtype=float).reshape(-1).copy(), 0.0)
    out = np.empty((0, 2))
    out_r = np.empty(0)
    for _ in range(int(max(0, count))):
        if P.shape[0]:
            d = np.hypot(cand[:, 0][:, None] - P[:, 0][None, :],
                         cand[:, 1][:, None] - P[:, 1][None, :]) - R[None, :]
            ins = np.minimum(cwall, d.min(1))
        else:
            ins = cwall
        sel = np.argsort(ins)[-top:]
        idx = int(sel[rng.randint(sel.size)])
        c = np.clip(cand[idx] + rng.normal(0.0, 0.4 / grid, size=2), LO + 1e-5, HI - 1e-5)
        ri = max(1e-9, float(ins[idx]))
        out = np.vstack([out, c[None, :]])
        out_r = np.append(out_r, ri)
        P = np.vstack([P, c[None, :]])
        R = np.append(R, ri)
    return out, out_r


def _relocate(xy, r, rng, k, deadline, grid=41):
    """RUIN AND RECREATE: lift k circles, let the survivors RELAX into the room, then refill the
    holes the relaxed packing actually has.  Returns (centres, radius seed).

    A local optimum of the sum is usually spoiled by a few tiny circles wedged where they earn almost
    nothing; a gaussian shake never moves them far enough to escape.  Dropping the runts and
    re-inserting them with `_hole_points` is the move that does -- but only if the survivors move
    first.  The previous version scored the holes of the packing with the runts simply DELETED, and
    the only holes such a packing has are the ones the runts were sitting in, so `_hole_points` put
    them back essentially where they came from and the move returned (to float noise) its own input.
    That is why the screening runs kept scoring this family at exactly 0.

    The fix is one line of intent: between the ruin and the recreate, solve the (n-k)-circle problem.
    `_penalty` on the survivors lets them spread and grow into the freed room -- a genuinely
    DIFFERENT configuration, whose empty space is therefore somewhere else -- and `_radius_lp` then
    gives them honest radii, so the holes scored are real holes of a real packing.  This is the
    classic ruin-and-recreate move, and it is the only family here that changes the contact topology
    without needing an incumbent to inflate: it works identically on a cold size with no census.

    WHICH k is drawn, not fixed.  Taking literally the k smallest made the move almost
    deterministic: with k <= n/12 the same handful of runts was lifted every hop, so repeating the
    move re-explored one tiny neighbourhood and the census stalled.  Sampling k distinct circles
    without replacement with weight 1/r keeps the runts far likelier to move while leaving every
    circle reachable, which is what makes the move a neighbourhood rather than a single point.

    The sub-solve uses `_penalty`'s OWN defaults -- no second schedule is introduced for it; the
    survivors are just a smaller instance of the same problem and are solved by the same rule.
    """
    n = xy.shape[0]
    k = int(max(1, min(k, n - 1)))
    r = np.asarray(r, dtype=float).reshape(-1)
    w = 1.0 / np.maximum(r, 1e-12)
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0.0:
        order = np.argsort(r)[:k]
    else:
        order = rng.choice(n, size=k, replace=False, p=w / tot)
    keep = np.ones(n, dtype=bool)
    keep[order] = False
    sub, sub_r = xy[keep], r[keep]
    # the survivors relax into the freed room -- this is what makes the holes new
    sub = _penalty(sub, deadline, r0=sub_r)
    lp = _radius_lp(sub)
    sub_r = _repair(sub, sub_r if lp is None else lp)
    pts, pts_r = _hole_points(sub, sub_r, k, rng, grid=grid)
    new = np.vstack([sub, pts])
    new_r = np.concatenate([sub_r, pts_r])
    return np.clip(new, LO + 1e-5, HI - 1e-5), np.maximum(new_r, 1e-12)


def _neighbour_sizes(n, span=4):
    """Nearest sizes m != n that HAVE a committed pack, closest first.  Empty when none do."""
    ms = sorted((m for m in range(max(1, n - span), n + span + 1) if m != n),
                key=lambda m: (abs(m - n), m))
    return [m for m in ms if _pck_path(m) is not None]


def _transfer_start(n, m, rng):
    """Re-use the committed packing of a NEARBY size m as a start for n (cross-size transfer).

    Adjacent sizes' optima differ by ~0.5 % and share most of their contact structure, so a good
    packing of m is a far better guess for n than any lattice or random draw -- and it is the one
    piece of information this call cannot generate for itself, because it is another size's entire
    search, condensed.  m > n: drop the (m-n) smallest circles (they earn least, and removing them
    frees exactly the room the survivors want).  m < n: insert (n-m) circles at the roomiest holes.

    Returns ``(centres, radius seed)``: size m's own radii for the circles it contributes and the
    insertion radius for each circle added, so the transferred start arrives near-feasible.
    """
    got = _read_pck(m)
    if got is None:
        return None
    xy, r = got
    if m > n:
        sel = np.sort(np.argsort(r)[m - n:])
        xy, r = xy[sel], r[sel]
    elif m < n:
        pts, pts_r = _hole_points(xy, r, n - m, rng)
        xy = np.vstack([xy, pts])
        r = np.concatenate([r, pts_r])
    if xy.shape[0] != n:
        return None
    xy = np.clip(xy + rng.normal(0.0, 0.05 / np.sqrt(n), size=xy.shape), LO + 1e-5, HI - 1e-5)
    return xy, np.maximum(np.asarray(r, dtype=float), 1e-9)


# ---------------------------------------------------------------- per-n driver

def _slp(xy, deadline, max_fail=26):
    """Run SLP from `xy` until it is CERTIFIED converged, a numerical failure exhausts the retries,
    or the CPU deadline hits.  Yields (xy, r, sum_r) at every strict improvement so the caller can
    meter it.

    There is no trust-region shrink tail, and that is a theorem about this linearisation rather than
    a tuning choice.  Let ``V(delta)`` be `_centre_lp`'s optimal value with the centre box of radius
    delta at the current iterate.  Then

      (1) V is NONDECREASING in delta -- a smaller box is a subset of a larger one and the
          linearisation point is the same; and
      (2) the LP's own r is feasible at the returned centres (the linearised pair constraint
          ``r_i + r_j <= u_ij . (c_i - c_j)`` implies the true one, because ``u . v <= ||v||`` for any
          unit u), so the exact radius LP at those centres returns a sum ``>= V(delta)`` and the
          realised ``s2 >= V(delta)`` up to `_repair`'s ~1e-13 shave.

    So a step that fails to improve certifies ``V(delta) <= cur``, and by (1) ``V(delta') <= cur`` for
    EVERY smaller delta': no smaller trust region can gain anything here, and the iterate is a
    first-order point of the true problem.  Shrinking is the standard response to a model that
    OVERestimates progress; an inner approximation UNDERestimates, so the old ``delta *= 0.45`` up to
    26 times was provably dead clock -- measured at 2.3x (n=99) to 16.7x (n=33) of the whole
    finisher's cost for a bit-identical result (artifacts/probe29.py).

    The one failure that is NOT such a certificate is a NUMERICAL one -- linprog reporting failure or
    raising -- where the model said nothing at all; there shrinking is legitimate, and only those
    failures count against `max_fail`.
    """
    r = _radius_lp(xy)
    if r is None:
        r = _repair(xy, np.zeros(xy.shape[0]))
    r = _repair(xy, r)
    cur = float(r.sum())
    yield xy, r, cur
    delta = 0.45 / np.sqrt(xy.shape[0])
    fails = 0
    while time.process_time() < deadline and delta > 1e-11 and fails < max_fail:
        xy2 = _centre_lp(xy, r, delta)
        r2 = _radius_lp(xy2) if xy2 is not None else None
        if r2 is None:
            # numerical failure: no certificate either way, so retry on a smaller box
            fails += 1
            delta *= 0.45
            continue
        r2 = _repair(xy2, r2)
        s2 = float(r2.sum())
        if s2 <= cur + 1e-13:
            return          # CERTIFIED first-order: V(delta') <= cur for every delta' < delta
        xy, r, cur = xy2, r2, s2
        fails = 0
        delta = min(delta * 1.7, 0.45 / np.sqrt(xy.shape[0]))
        yield xy, r, cur


def _solve_one(n, evaluate, meter, rng, deadline):
    best_xy, best_r, best_s = None, None, -np.inf

    def push(xy, r, s):
        """Route a candidate through the metered evaluate(); returns its accepted sum_r."""
        nonlocal best_xy, best_r, best_s
        if meter.left() <= 0:
            return -np.inf
        pack = np.concatenate([xy, r[:, None]], 1)
        try:
            feas, sr = evaluate(n, pack)
        except Exception:
            return -np.inf
        if feas and sr > best_s:
            best_s = float(sr)
            best_xy = xy.copy()
            best_r = r.copy()          # the incumbent's RADII are a start in their own right
        return float(sr) if feas else -np.inf

    def polish(xy, sub_deadline):
        """Exact finisher: SLP (radius LP + trust-region centre LP) turns approximate centres into a
        strictly feasible packing, and every strict improvement on the way is metered.

        Returns the BEST POINT the SLP actually reached as ``(xy, r, sum_r)`` -- feasible, whether or
        not it beat the incumbent and so whether or not it was worth a budget unit.  The hop loop
        needs the landed local optimum itself, not merely the news that it was worse: a hop that ends
        below the incumbent is still a first-order point of a DIFFERENT topology, and that is the
        only thing the search has to walk on.
        """
        last = -np.inf
        sxy, sr, ss = None, None, -np.inf
        for cxy, cr, cs in _slp(xy, min(sub_deadline, deadline)):
            if cs > ss:
                sxy, sr, ss = cxy, cr, cs
            if cs > last + 1e-12 and cs > best_s - 1e-12:
                push(cxy, cr, cs)
                last = cs
        return sxy, sr, ss

    def attempt(xy, share, r0=None):
        """Penalty local solve (crosses basins) then the exact SLP finisher.

        `r0` is the radius seed.  Passing the start's OWN radii (the incumbent's, a neighbouring
        size's, an insertion radius) instead of the flat `0.9 r_typ` default is what makes a move
        local: the radii of the circles the move did not touch are already right, so the penalty
        solve rearranges the disturbed neighbourhood instead of re-melting the whole packing.
        Measured at a 3 s slice from the committed pack: relgap 8.5e-4 -> 4.6e-4 (n=35) and
        1.9e-3 -> 8.6e-4 (n=63) for the same clock and the same move sequence.
        """
        # The penalty ramp is self-limiting (fixed stages x iterations), so it is bounded only by the
        # slice deadline -- capping it by `share` truncated the mu ramp at small n and wasted the
        # attempt.  `share` bounds only the SLP finisher, whose step count is open-ended.
        xy = _penalty(xy, deadline, r0=r0)
        return polish(xy, min(deadline, time.process_time() + max(0.05, share)))

    def step(cand, cand_r, eps):
        """One hop: solve the perturbed start, then decide where the CHAIN stands next.

        `_accept` is the rule (threshold accepting within a relative `eps` of the best -- the same
        `eps` this hop asked for).  When the landing is outside the band the chain falls back to the
        best packing, so the chain is always either exploring a near-best plateau or restarting from
        the best.  `best_*` is untouched by any of this: only `push` (i.e. the metered `evaluate`)
        decides what counts, so a lateral chain move can cost clock but can never cost score.
        """
        nonlocal cur_xy, cur_r, cur_s
        sxy, sr, ss = attempt(cand, share, r0=cand_r)
        if sxy is not None and _accept(ss, best_s, eps):
            cur_xy, cur_r, cur_s = sxy, sr, ss
        elif best_xy is not None and best_s > cur_s:
            cur_xy, cur_r, cur_s = best_xy, best_r, best_s
        return ss

    total = max(deadline - time.process_time(), 0.0)
    # One penalty+polish attempt is cheap, so the slice is spent on MANY of them rather than on a
    # long single descent; `share` is a soft per-attempt cap, not a schedule.
    share = max(0.05, total / 24.0)

    # 1. Warm start from the committed census (guarded).  Metered immediately from ONE exact radius
    #    LP, so this call can never hand the driver a worse packing than the one already committed
    #    for n -- and that costs ~2 ms, not a whole SLP convergence.  The SLP is then given the same
    #    soft `share` cap as any other attempt rather than the whole slice: a committed pack is
    #    already a first-order point, so an uncapped re-convergence pays a full trust-region collapse
    #    (measured: 0.12 s at n=35 and 0.19 s at n=63 for exactly zero gain, 0.43 s at n=99 for
    #    5e-4) to rediscover that.  Capping it hands the remainder to the restart loop, where the
    #    same clock buys diversity -- which is what actually pays here.
    warm = _warm_start(n)
    if warm is not None:
        wr = _radius_lp(warm)
        if wr is not None:
            wr = _repair(warm, wr)
            push(warm, wr, float(wr.sum()))
        polish(warm, time.process_time() + share)

    # 1b. Cross-size structure transfer (guarded; no-op for an n with no neighbouring pack, e.g. a
    #     cold held-out size).  These are the strongest starts obtainable: the committed packing of
    #     n-2 is already within ~0.5 % of ITS record and shares most of n's contact structure, so
    #     transferring it in costs one attempt and imports another size's whole search.  Closest
    #     sizes first, and each is just another `attempt` under the same soft `share` cap.
    #     The sweep RUNS WHILE IT PAYS rather than for a fixed four sizes.  Measured
    #     (artifacts/probe47.py) at all 8 sizes still short of the record, with the committed pack
    #     already pushed: every transfer from every one of the two nearest neighbours lands ~1 digit
    #     BELOW the incumbent (mean 2.72 vs 3.72), and at 5 of the 8 the melt and a melt-free
    #     re-convergence return the SAME point -- so once an incumbent exists the sweep is not even
    #     exploring, it is re-deriving a worse packing four times for ~21 % of the slice (4 of the
    #     ~24 shares, on top of the warm polish).  So: keep going only while a transfer actually
    #     improves on the best so far.  That is the same shape as the cold-start gate below and adds
    #     no constant -- and it leaves the COLD regime byte-identical, because with no incumbent at
    #     entry the first transfer sets the best and the sweep runs in full, which is exactly where
    #     a transfer is the strongest start there is.
    nbrs = _neighbour_sizes(n)
    had_incumbent = best_xy is not None
    for m in nbrs[:4]:
        if time.process_time() >= deadline or meter.left() <= 0:
            break
        got = _transfer_start(n, m, rng)
        if got is not None:
            _, _, ss = attempt(got[0], share, r0=got[1])
            if not _keep_transferring(ss, best_s, had_incumbent):
                break

    # 2. Cold multi-start over the rest of the first part of the slice: the penalty solver needs no structured
    #    start (probed: hand-built hex/grid lattices polish no better than random), so the starts
    #    are uniform random plus the row-grid family for variety.
    # When an incumbent already exists, cold random starts are nearly worthless: at n=99 a cold
    # start polishes to ~1e-2 relative, an order of magnitude behind the committed pack, so the 45 %
    # of the slice they used to take bought nothing and starved the hop loop (~8 hops at n=99, where
    # the productive moves need tens of trials to land).  They keep the whole share when there is no
    # incumbent -- the cold held-out regime, where they are the only start there is.
    explore_until = deadline - (0.55 if best_xy is None else 0.90) * total
    k = 0
    while time.process_time() < explore_until and meter.left() > 0:
        xy = _random_start(n, rng) if k % 3 else _grid_start(n, rng)
        k += 1
        attempt(xy, share)

    # 3. Basin hopping on the incumbent: a gaussian shake (local escape) alternating with
    #    ruin-and-recreate -- lift the runts, RELAX the survivors, refill the holes that relaxation
    #    opened (the move that crosses combinatorial basins without needing an incumbent to inflate).
    # The hop CHAIN state, distinct from the best-ever packing.  Every hop used to be built from
    # `best_xy`, so a hop that landed below the incumbent was discarded whole and the next hop
    # restarted from the same point -- the search could only ever ratchet, never walk.  It has to
    # walk: `probe44.py` shows all 12 short sizes are CERTIFIED first-order points (uncapped `_slp`
    # from each committed pack gains < 4e-12 in one step), so their remaining 1e-4..8e-4 is a
    # topology the incumbent's own basin does not contain, and `probe40.py` shows the inflation hop
    # lands BELOW the incumbent essentially always.  The moves that reach a new topology are exactly
    # the ones whose output was being thrown away.
    cur_xy, cur_r, cur_s = best_xy, best_r, best_s
    scale = 0.30 / np.sqrt(n)
    hop = 0
    while time.process_time() < deadline and meter.left() > 0 and best_xy is not None:
        before = best_s
        if cur_xy is None or cur_s <= -np.inf:
            cur_xy, cur_r, cur_s = best_xy, best_r, best_s
        base_r = cur_r if cur_r is not None else _repair(cur_xy, np.zeros(n))
        # ONE relative step size per hop, drawn from the window that already bracketed the census's
        # own relative gaps: it is the inflation family's demand when that family is drawn, and it is
        # always the chain's lateral-acceptance threshold below.  Asking for +eps and accepting a
        # landing within -eps is the same number used once, not a second schedule.
        eps = 10.0 ** rng.uniform(-4.0, -2.0)
        # Half the hops go to INFLATION, the rest to the screened perturbation wheel below.  The
        # split is not a tuned constant: screened one family at a time against the committed packs
        # (artifacts/probe23.py, 9 s per family), inflation was the only family that gained anything
        # at every size tried -- n=43 +1.55e-3, n=67 +1.01e-3, n=33 +2.47e-3, n=99 +1.97e-2 (which
        # CLEARED the record) -- while dilation gained +1.8e-3 at n=99 and 0 elsewhere and the
        # windowed shake gained 0 or went backwards everywhere.  It dominates, so it gets the larger
        # half; the wheel keeps the other half because a move that only ratchets the incumbent
        # upward still needs something that changes where the incumbent IS.
        if rng.rand() < 0.5:
            # demand a relative gain of `eps` (drawn above, log-uniformly over 1e-4..1e-2): the
            # small end is a reachable ratchet, the large end asks for a reorganisation and usually
            # fails.  The window brackets the census's own relative gaps rather than being tuned.
            # START THE INFLATION FROM A JITTERED INCUMBENT, not from the incumbent itself.
            # `_inflate` is a DETERMINISTIC map: L-BFGS-B from a fixed start, no rng inside it.
            # Fed `best_xy` every time, the whole chain inflate -> _penalty -> _slp is a
            # deterministic function of eps alone, and eps -> outcome turned out to be
            # piecewise-constant with a handful of pieces -- so 50 hops recomputed ~5 points and
            # two independent seeds returned BIT-IDENTICAL totals (artifacts/probe41.py,
            # probe42.py: gain exactly 0 at n=33/45/49/81 over 22..71 hops, and unchanged by
            # widening the eps window from 1e-2 out to 1).  Half of every hop budget was spent
            # recomputing the incumbent.  The jitter is the wheel's OWN adaptive amplitude
            # `scale` -- the same displacement the shake and the dilation already use, adapted by
            # the same rule below -- so no constant is introduced and the move keeps its identity
            # (what makes this family different is the PINNED sum, not where it starts).
            start = np.clip(cur_xy + rng.normal(0.0, scale, size=cur_xy.shape),
                            LO + 1e-5, HI - 1e-5)
            cand, cand_r = _inflate(start, base_r, cur_s * (1.0 + eps), deadline)
            hop += 1
            step(cand, cand_r, eps)
            scale = min(scale * 1.3, 0.6 / np.sqrt(n)) if best_s <= before \
                else max(scale * 0.7, 0.01 / np.sqrt(n))
            continue
        kind = rng.randint(6)
        if kind == 5 and nbrs:
            # structured diversity: a fresh transfer (different m, different hole draws) instead of
            # yet another perturbation of the incumbent.  Probed: diversity is what pays here.
            got = _transfer_start(n, nbrs[rng.randint(min(4, len(nbrs)))], rng)
            cand, cand_r = got if got is not None else (_random_start(n, rng), None)
        elif kind in (1, 3) and n > 2:
            cand, cand_r = _relocate(cur_xy, base_r, rng, 1 + rng.randint(max(1, n // 12)), deadline)
        elif kind == 2:
            # DILATION: scale the centres about the middle of the square and clip.  Every other move
            # conserves the packing's overall extent, so none of them can reach the arrangements in
            # which the outer ring is pressed onto the walls -- and the walls are where a
            # sum-of-radii packing earns.  The circles pushed past the wall are squashed onto it and
            # the penalty solve re-negotiates the interior around them.  Screened move-by-move
            # against the committed packs (artifacts/probe20.py), this was the only family that
            # still beat the incumbent at n=99 by more than noise: +3.7e-3 in 26 trials, where
            # relocate, swap, radius-jitter and contraction all gained exactly 0.  The amplitude is
            # tied to the adaptive shake amplitude rather than being its own constant.
            sc = 1.0 + rng.uniform(0.5, 4.5) * scale
            cand = np.clip(cur_xy * sc, LO + 1e-5, HI - 1e-5)
            cand_r = base_r * sc
        else:
            # A WINDOWED shake: perturb only the m circles nearest a randomly chosen one, with m
            # drawn log-uniformly over 3..n.  Shaking all n at once (the old move) is the m = n end
            # of that same draw, and it re-melts the packing; the small-m end is the genuinely local
            # move that the seeded radii of iteration 5 made worth having.  One draw covers both,
            # so there is no window-size constant to tune.
            m = int(round(3.0 * (n / 3.0) ** rng.uniform(0.0, 1.0))) if n > 3 else n
            m = int(min(n, max(1, m)))
            cand = cur_xy.copy()
            if m >= n:
                cand = cand + rng.normal(0.0, scale, size=cand.shape)
            else:
                c0 = cur_xy[rng.randint(n)]
                idx = np.argsort(np.hypot(cur_xy[:, 0] - c0[0], cur_xy[:, 1] - c0[1]))[:m]
                cand[idx] += rng.normal(0.0, scale, size=(m, 2))
            cand = np.clip(cand, LO + 1e-5, HI - 1e-5)
            cand_r = base_r
        hop += 1
        step(cand, cand_r, eps)
        scale = min(scale * 1.3, 0.6 / np.sqrt(n)) if best_s <= before else max(scale * 0.7, 0.01 / np.sqrt(n))

    return best_s


def solve(evaluate, meter, rng, targets):
    t0 = time.process_time()
    tg = sorted(set(int(t) for t in targets))
    if not tg:
        return
    total = CPU_ONE_TARGET if len(tg) == 1 else CPU_MANY_TARGETS
    # Weight the slice by n: the LPs grow with n and the large sizes are the ones still far from the
    # record, so they get proportionally more of the clock.  Deadlines are cumulative, so time a small
    # n finishes early rolls forward instead of being lost.
    w = np.array([float(n) for n in tg])
    # ... but a second of clock is only worth the DIGITS it buys, and digits are
    # clamp(-log10(relgap), 0, 7).  Iteration 7 weighted each size by that score's DERIVATIVE,
    # sqrt(median_relgap / relgap) -- and the derivative 1/(ln10 * (record - sum_r)) DIVERGES as the
    # gap closes, so it hands ever more clock to sizes with ever less left to win.  The score is
    # CLAMPED, so what a size can still yield is not its gradient but its REMAINING PRIZE:
    #
    #     prize(n) = DIGITS_CAP - clamp(-log10(relgap), 0, DIGITS_CAP)
    #
    # the digits between where it stands and the cap.  That is the quantity the census score is a
    # mean of, and it is bounded: a size at relgap 5.4e-6 has 1.73 digits left FOREVER, while one at
    # 3.2e-3 has 4.50.  The gradient rule rated the first 11x the second per unit of sum_r; the
    # prize rule rates it 0.38x, and the prize is what actually lands, because ONE successful
    # inflation hop moves sum_r by ~1e-3 relative -- larger than every remaining gap in the census,
    # so a success collects the whole remaining prize rather than a slice of it.  Measured
    # (artifacts/probe25.py, 2 seeds at the committed slices): the gradient rule's two biggest
    # near-cap grants, n=29 at 12.6 s and n=67 at 11.5 s, gained EXACTLY 0.0 in every seed -- 24 s
    # of a 104 s iteration -- while n=33, n=45, n=47 and n=49 (the four weakest sizes in the census,
    # 4.20..4.50 digits still on the table) were each held to ~1.2 s.
    #
    # Weighted by prize x n (cost still grows with n), nothing else changes, and two special cases
    # DISAPPEAR rather than being coded: a size at the cap scores prize = 0 and so takes none of the
    # clock by the same formula that allocates every other size, and there is no window, exponent or
    # median to pick.  A size whose gap cannot be read -- a held-out n, a fresh n, no records table
    # -- takes the MEDIAN live prize, i.e. a neutral share of a rule it cannot be scored under, and
    # if no gap at all is readable the split stays purely cost-proportional, so the cold regime is
    # untouched.
    rg = [_relgap(int(n)) for n in tg]
    prize = [None if g is None else max(0.0, DIGITS_CAP - min(DIGITS_CAP, -np.log10(max(g, 1e-300))))
             for g in rg]
    live = [p for p in prize if p is not None and p > 0.0]
    if live:
        neutral = float(np.median(live))
        f = np.array([neutral if p is None else p for p in prize])
        if (w * f).sum() > 0.0:
            w = w * f
    w = w / w.sum()
    acc = t0
    for k, n in enumerate(tg):
        if meter.left() <= 0:
            break
        acc += total * float(w[k])
        if time.process_time() >= acc:
            continue
        try:
            _solve_one(n, evaluate, meter, rng, acc)
        except Exception:
            continue


# ---------------------------------------------------------------- self-test

def _self_test():
    import sys

    class _Meter:
        def __init__(self, budget):
            self.budget, self.used = budget, 0

        def tick(self, k=1):
            g = max(0, min(k, self.budget - self.used))
            self.used += g
            return g

        def left(self):
            return self.budget - self.used

    class _Ev:
        def __init__(self, meter):
            self.meter, self.best, self.calls = meter, {}, 0

        def evaluate(self, n, packing):
            self.calls += 1
            a = np.asarray(packing, dtype=float)
            single = a.ndim == 2
            if single:
                a = a[None]
            if a.ndim != 3 or a.shape[2] != 3:
                raise ValueError("bad shape")
            if a.shape[1] != n:
                raise ValueError("bad n")
            B = a.shape[0]
            grant = self.meter.tick(B)
            feas = np.zeros(B, dtype=bool)
            sr = np.full(B, -np.inf)
            for b in range(grant):
                x, y, r = a[b, :, 0], a[b, :, 1], a[b, :, 2]
                wall = np.minimum.reduce([x - LO - r, HI - x - r, y - LO - r, HI - y - r]).min()
                d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
                np.fill_diagonal(d, np.inf)
                pair = (d - (r[:, None] + r[None, :])).min()
                feas[b] = (r.min() > 0) and wall >= -1e-9 and pair >= -1e-9
                sr[b] = r.sum()
                if feas[b] and (n not in self.best or sr[b] > self.best[n]):
                    self.best[n] = float(sr[b])
            return (bool(feas[0]), float(sr[0])) if single else (feas, sr)

    global CPU_ONE_TARGET, CPU_MANY_TARGETS
    CPU_ONE_TARGET, CPU_MANY_TARGETS = 2.0, 3.0      # keep the self-test quick; solve() is unchanged

    ok = True
    meter = _Meter(500_000)
    ev = _Ev(meter)
    rng = np.random.RandomState(0)

    # TWO calls in ONE process: the second must still produce a feasible packing (guards against a
    # module-level absolute deadline), and a single-element `targets` must not starve.
    t = time.process_time()
    solve(ev.evaluate, meter, rng, [11])
    first = ev.best.get(11)
    d1 = time.process_time() - t
    t = time.process_time()
    solve(ev.evaluate, meter, rng, [13])
    second = ev.best.get(13)
    d2 = time.process_time() - t
    print("call1 n=11 sum_r=%r cpu=%.1fs | call2 n=13 sum_r=%r cpu=%.1fs | evals=%d"
          % (first, d1, second, d2, meter.used))
    if first is None or second is None:
        print("FAIL: a call produced no feasible packing"); ok = False
    if d2 < 0.5 * d1 - 1.0:
        print("FAIL: second call did far less work than the first (stale deadline?)"); ok = False

    # A multi-target call, including an n with no committed pack (must cold-start, not raise).
    meter2 = _Meter(500_000)
    ev2 = _Ev(meter2)
    solve(ev2.evaluate, meter2, np.random.RandomState(1), [7, 9])
    if ev2.best.get(7) is None or ev2.best.get(9) is None:
        print("FAIL: multi-target call lost a cold-start n"); ok = False

    # The cross-size transfer must map BOTH directions and must be a guarded no-op for a size with
    # no neighbouring pack at all (the cold held-out regime).
    for n in (28, 30):
        nb = _neighbour_sizes(n)
        if not nb:
            continue
        for m in nb[:3]:
            got = _transfer_start(n, m, np.random.RandomState(3))
            if got is None:
                print("FAIL: _transfer_start(%d, %d) bad" % (n, m)); ok = False
                continue
            st, sr = got
            if st.shape != (n, 2) or not np.isfinite(st).all() or st.min() < LO or st.max() > HI \
                    or sr.shape != (n,) or not np.isfinite(sr).all() or sr.min() <= 0:
                print("FAIL: _transfer_start(%d, %d) bad" % (n, m)); ok = False
    if _neighbour_sizes(100000) != [] or _transfer_start(100000, 99999, np.random.RandomState(4)) is not None:
        print("FAIL: transfer not guarded for a size with no neighbouring pack"); ok = False

    # The radius seed must be honoured (r0 shapes the penalty start) and `_relocate` must keep the
    # radii of the circles it did NOT move -- that is what makes a hop local rather than a re-melt.
    rs = np.random.RandomState(5)
    xy_t = _grid_start(16, rs)
    r_t = _repair(xy_t, _radius_lp(xy_t))
    mv, mv_r = _relocate(xy_t, r_t, rs, 3, time.process_time() + 2.0)
    # ruin-and-recreate: the survivors RELAX, so nothing is required to stay put -- what must hold
    # is that the move returns a complete, positively-radiused, in-square seed that is genuinely
    # DIFFERENT from its input (the old version's failure mode was returning its own input back).
    if mv.shape != (16, 2) or mv_r.shape != (16,) or mv_r.min() <= 0 or not np.isfinite(mv).all() \
            or mv.min() < LO or mv.max() > HI:
        print("FAIL: _relocate returned a malformed seed"); ok = False
    elif float(np.abs(np.sort(mv_r) - np.sort(r_t)).max()) < 1e-9:
        print("FAIL: _relocate did not change the packing"); ok = False
    if not np.isfinite(_penalty(xy_t, time.process_time() + 2.0, r0=r_t)).all():
        print("FAIL: _penalty rejected a radius seed"); ok = False
    pts, pts_r = _hole_points(xy_t, r_t, 2, rs)
    if pts.shape != (2, 2) or pts_r.shape != (2,) or pts_r.min() <= 0:
        print("FAIL: _hole_points radii bad"); ok = False

    # The hop chain's acceptance rule must genuinely WALK -- an invariance assertion here (e.g.
    # "the chain never worsens") is exactly what a ratchet satisfies, so assert the opposite: a
    # slightly worse local optimum IS accepted, while a far worse one is NOT (no downhill drift).
    if not _accept(1.0, 1.0, 1e-3):
        print("FAIL: _accept rejected an equal-best landing"); ok = False
    if not _accept(1.0 - 5e-4, 1.0, 1e-3):
        print("FAIL: _accept is a ratchet -- it rejected a landing inside its own band"); ok = False
    if _accept(1.0 - 5e-2, 1.0, 1e-3) or _accept(-np.inf, 1.0, 1e-2):
        print("FAIL: _accept admits a landing outside its band (downhill drift)"); ok = False
    if not _accept(1.1, 1.0, 1e-4):
        print("FAIL: _accept rejected an improvement"); ok = False

    # The transfer sweep's gate.  Assert what it must DO, not that it is harmless: with an
    # incumbent it must CUT a transfer that failed to improve (an "always True" stub passes any
    # invariance assertion here), and with no incumbent it must never cut, because a cold size's
    # first transfer is both its best and its strongest start.
    if _keep_transferring(0.99, 1.0, True):
        print("FAIL: _keep_transferring did not cut a worse transfer on a warm size"); ok = False
    if _keep_transferring(-np.inf, 1.0, True):
        print("FAIL: _keep_transferring did not cut a failed transfer"); ok = False
    if not _keep_transferring(1.0, 1.0, True) or not _keep_transferring(1.01, 1.0, True):
        print("FAIL: _keep_transferring cut a transfer that paid"); ok = False
    if not _keep_transferring(-np.inf, -np.inf, False) or not _keep_transferring(0.5, 1.0, False):
        print("FAIL: _keep_transferring gated the COLD regime (no incumbent)"); ok = False

    # Known optimum sanity: n=2 -> two circles on the diagonal, sum_r = 2 - sqrt(2) ~ 0.5857864.
    meter3 = _Meter(20_000)
    ev3 = _Ev(meter3)
    solve(ev3.evaluate, meter3, np.random.RandomState(2), [2])
    got = ev3.best.get(2)
    print("n=2 sum_r=%r (optimum %.9f)" % (got, 2 - np.sqrt(2)))
    if got is None or got < (2 - np.sqrt(2)) - 1e-6:
        print("FAIL: n=2 did not reach its known optimum"); ok = False

    print("SELF-TEST: %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
