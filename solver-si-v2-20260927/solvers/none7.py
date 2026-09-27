"""Circle-packing solver: maximize sum of radii of n circles in the unit square.

Contract (see MISSION.md): solve(evaluate, meter, rng, targets).

WHY THIS WORKS -- the one idea the whole solver rests on
--------------------------------------------------------
The problem is  max sum(r)  s.t.  ||c_i - c_j|| >= r_i + r_j  and  r_i <= wall_i.
The pairwise constraint is *reverse convex* (non-convex), but ||.|| is CONVEX, so its first-order
expansion is a global UNDER-estimator:

    ||(c_i+u_i) - (c_j+u_j)||  >=  D_ij + e_ij . (u_i - u_j),     e_ij = (c_i-c_j)/D_ij.

Replacing the true constraint by

    D_ij + e_ij . (u_i - u_j)  >=  (r_i + w_i) + (r_j + w_j)

therefore yields a CONSERVATIVE linear restriction of the feasible set: *every* solution of the LP
    max sum(w)  s.t. the linearized pair constraints + the (already linear) wall constraints
is strictly feasible for the true problem, for ANY step size.  And u = v = w = 0 is always LP-feasible,
so each LP can only improve sum(r).  Iterating it (sequential linear programming, SLP) is a monotone
ascent that converges to a locally optimal packing -- with the radii re-allocated OPTIMALLY at every
step (the LP decides which circles get big; the seed solver's "equal-ish r_i = min(wall, minD/2)" rule
cannot).  A trust region on (u,v,w) only controls step quality and lets us prune far-apart pairs; it is
never needed for feasibility.

The measured gaps are STRUCTURAL, not precision: when the SLP lands in the right basin it closes the
gap to 1e-12 (full 7 digits), and when it does not it stalls at relgap ~5e-3.  So everything around the
local ascent is about reaching more basins per second:
  * the SLP stops the instant its LP optimum is 0 -- that point is first-order stationary and, because
    the restriction is homogeneous in the trust radius t, every SMALLER t gives a provably-zero LP too.
    Shrinking t there (the old behaviour) burned 13 of 21 LPs re-proving stationarity.  Wall rows are
    pruned by the same 2t argument that already pruned pair rows;
  * structured starts over TWO lattice dimensions -- spacing and ROTATION.  The densest clipped hex
    patch in a square is often tilted, and no first-order move can rotate a lattice, so the tilt has to
    come from the start.  Plus row-count families, square grids, and a guarded census warm start;
  * a RANDOMIZED initial trust radius: t0 was a fixed 0.03 for every n, i.e. 0.14*spacing at n=27 and
    0.28*spacing at n=99 -- a hidden size-dependent choice.  Two ascents with different t0 from the
    SAME start land in DIFFERENT basins (measured), so t0 is a search dimension and is now drawn
    log-uniformly in [0.12, 1.0]*spacing per start; every other family diversifies the start, this one
    diversifies the ascent, and it costs no throughput (see _rand_t0);
  * basin hopping with THRESHOLD ACCEPTANCE over three moves (teleport a few smallest circles into the
    roomiest holes / shake one neighbourhood / shake globally).  A greedy walker re-perturbs the single
    best optimum forever; a walker that accepts small losses drifts across the plateau of near-equal
    local optima, which is where the record structures sit.  The incumbent is never lost -- but after
    STALL non-improving hops it IS abandoned for a fresh randomly-tilted lattice, because a walker that
    only ever perturbs one optimum keeps re-deriving it (measured: n=27 reaches 271 local optima in 6 s
    and still misses, so the failure there is diversification, not throughput);
  * the LP solved by interior point: it is >90% of all CPU (measured), so its cost IS the price of a
    basin, and HiGHS' IPM is ~1.4-1.7x faster than the default on these row/column shapes;
  * no CPU on a size whose committed pack is already at the 7-digit cap, and none on one that reaches
    the cap mid-search -- the freed time flows to the sizes that can still gain;
  * an exact feasibility repair (global radius scale, then monotone per-circle maximal growth) so what
    is handed to evaluate() clears the 1e-9 tolerance by construction, not by luck.

Metering.  CPU -- not the 500k evaluation budget -- is the binding meter for a deep per-n optimizer, so
the deadline is derived AT ENTRY from time.process_time() and from len(targets) (never a module-level
absolute), and time is spread over the n actually requested.  Every candidate that might be the best is
routed through evaluate() as soon as it is found, so a CPU backstop mid-SLP still leaves the best-so-far
registered and harvestable.  All randomness comes from `rng`.
"""
import json
import os
import re
import time

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

LO, HI = -0.5, 0.5

# The in-loop driver caps the WHOLE solve() call at 120 s process-CPU; the offline re-run gives 60 s per
# call with ONE n in targets.  min(CPU_TOTAL, CPU_PER_TARGET*k) must sit under BOTH without ever
# assuming the 37-size visible census.
#
# Why CPU_PER_TARGET is 48 and not 57.  A deadline is only ever TESTED at the top of a loop, so the
# call always runs past it by the cost of whatever move was already in flight.  Measured
# (artifacts/ab10_sf.txt, cpu_allowance=8.0, n = 37/49/81 x 2 seeds): the call actually burned
# 8.1-15.2 s, i.e. +0.1 to +7.2 s of OVERSHOOT -- additive, not proportional, and dominated by the one
# move that checked no clock at all (_relax's L-BFGS-B levels, now deadline-aware).  With k >= 2 that
# costs nothing, because every per-n deadline is min(end_all, ...) so the slices absorb it and the call
# still lands on CPU_TOTAL.  With k == 1 -- which is EXACTLY the offline graded case -- there is no
# later slice to absorb anything: 57 + 7 > 60, so the graded call was being CPU-KILLED, throwing away
# the final exact-growth polish and its registration.  48 s leaves >= 12 s of headroom, more than the
# worst overshoot ever measured, and unused CPU earns nothing anyway.
CPU_TOTAL = 110.0
CPU_PER_TARGET = 48.0

_EPS = 1e-13          # feasibility shrink; ~1e4x under the 1e-9 tolerance, ~1e-11 relative on sum(r)
_RTINY = 1e-12
_LPMETH = "highs-ipm"   # the LP is >90% of all CPU (measured); IPM is ~1.4-1.7x faster here
_LPZERO = 1e-11       # LP optimum below this == first-order stationary (see _lp_step)
DIGIT_CAP_RELGAP = 1e-7   # scorer caps digits at 7 == relgap 1e-7: below this an n cannot gain SCORE
START_FRAC = 0.55     # share of a target's CPU spent on structured starts vs basin hopping
ACCEPT_THR = 2e-3     # threshold-accept width, in units of the mean radius
STALL = 25            # consecutive non-improving hops before the walker abandons the incumbent
MIN_CPU_PER_N = 55.0  # a size gets at least this much CPU, or none at all (see solve()).  This is the
                      # measured KNEE of the depth-response curve, not a floor on politeness.
                      # RAISED 33 -> 55 in iteration 15 on artifacts/deep15/ (all 10 active sizes x 2
                      # seeds x 55 s, single-target = exactly what a slot sees): at 55 s n=51 reaches
                      # the RECORD in 2/2 and n=63 in 1/2, while at 33 s (artifacts/rank14/, 3 seeds)
                      # n=51 reached it in only 1/3 and n=63 in 0/3.  The same eight sizes are frozen
                      # at both allowances, so the extra 22 s is not bought from anything that moves.
                      # artifacts/dep11_summary.txt: base solver, 5 active sizes x 3 seeds, the same
                      # paired starts at four allowances, scoring only a STRICT improvement over the
                      # committed pack (>= +1e-9 on sum_r):
                      #   alloc= 10 s  summed digit gain 0.733   2/15 cells improved   0.49 per 100 s
                      #   alloc= 20 s                    0.990   3/15                  0.33 per 100 s
                      #   alloc= 33 s                    4.887   6/15                  1.03 per 100 s
                      #   alloc= 55 s                    8.899   8/15                  1.15 per 100 s
                      # Depth is not merely better in absolute terms, it is 2.1x more CPU-EFFICIENT:
                      # 10 s buys a size a warm start plus a few hops, which re-derives the pack it
                      # started from (11 of 15 cells at 10 s returned the committed value to the last
                      # digit, for four different solver variants -- artifacts/ab11_summary.txt), while
                      # 33 s is enough to reach a NEW basin (n=35 seed 2 went 3.328 -> 7.000, i.e. the
                      # record, at 33 s and never at 20 s).  With 11 active sizes and 110 s the old
                      # value of 9.0 left 11*9 = 99 < 110, so the rotation below never fired and every
                      # size got exactly the 10 s that buys nothing; 11*33 > 110 fires it, and the
                      # allowance now goes to ~3 sizes at the knee with the rest keeping their
                      # committed packs (which costs zero -- the census is append-or-improve).
BAND = 12.0           # walker may sit at most BAND*thr below the incumbent
POOL = 8              # elite local optima kept per n as crossover parents (see _xover)


# ---------------------------------------------------------------- geometry helpers

def _wall(xy):
    """Distance from each centre to the nearest wall -- the cap on that circle's radius."""
    return np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0], xy[:, 1] - LO, HI - xy[:, 1]])


def _dist(xy):
    d = xy[:, None, :] - xy[None, :, :]
    D = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(D, np.inf)
    return D


def _make_feasible(xy, r, D=None):
    """Scale radii down by the single worst violation ratio -> exactly feasible (scaling all r by
    s <= 1 preserves every constraint), with a tiny absolute shrink for float safety."""
    xy = np.clip(xy, LO + 1e-9, HI - 1e-9)
    r = np.maximum(r, _RTINY)
    if D is None:
        D = _dist(xy)
    w = _wall(xy)
    s = float(np.min(w / r))
    if len(r) > 1:
        ratio = D / (r[:, None] + r[None, :])   # D has inf on the diagonal
        np.fill_diagonal(ratio, np.inf)         # ...but inf/inf is nan, which would poison the min
        s = min(s, float(np.min(ratio)))
    if s < 1.0:
        r = r * s
    r = np.maximum(r - _EPS, _RTINY)
    return xy, r, D


def _grow(xy, r, D, order):
    """Monotone maximal growth: set r_i to its exact cap given the OTHER radii, one circle at a time.
    Touching only r_i keeps every constraint not involving i intact, and the cap enforces the ones that
    do -- so the packing is feasible after every single assignment (and sum(r) never decreases)."""
    w = _wall(xy)
    for i in order:
        cap = min(w[i], float(np.min(D[i] - r)))
        if cap > _RTINY:
            r[i] = cap - _EPS
    return r


def _polish(xy, r, rng, passes=3):
    xy, r, D = _make_feasible(xy, r)
    for _ in range(passes):
        _grow(xy, r, D, rng.permutation(len(r)))
    return xy, r


# ---------------------------------------------------------------- the SLP ascent

def _lp_step(xy, r, D, t):
    """One LP of the convex restriction at (xy, r) with trust radius t.

    Returns (z, lpval).  `lpval` is the LP optimum = the exact gain the linearized model promises; it
    is 0 EXACTLY when the current point is first-order stationary, and because the restriction is
    positively homogeneous in t (z feasible at t => a*z feasible at a*t, since b_ub >= 0), lpval == 0
    at t implies lpval == 0 for every smaller t.  So a caller that sees lpval ~ 0 must STOP, not shrink:
    shrinking re-solves a provably-zero LP.  (Measured: the old shrink-to-tmin loop spent 13 of 21 LPs
    that way.)

    Only wall rows with slack < 2t can bind (a centre moves at most t and a radius grows at most t), so
    the 4n wall rows are pruned the same way the pair rows already are -- at n=99 that is ~400 rows
    down to ~40, and HiGHS time scales with the row count.
    """
    n = len(r)
    nv = 3 * n
    x, y = xy[:, 0], xy[:, 1]
    wall = np.stack([x - r - LO, HI - x - r, y - r - LO, HI - y - r])   # (4, n) wall slacks
    ks, ki = np.nonzero(wall < 2.0 * t + 1e-12)
    nw = ks.size
    wr = np.arange(nw)
    # rows 0/1 constrain x (-u_i + w_i <= slack for the low wall, +u_i + w_i <= slack for the high one),
    # rows 2/3 the same for y.
    sgn = np.where((ks == 0) | (ks == 2), -1.0, 1.0)
    axv = np.where(ks < 2, ki, n + ki)
    row = [wr, wr]
    col = [axv, 2 * n + ki]
    val = [sgn, np.ones(nw)]
    b = [wall[ks, ki]]

    slack = D - (r[:, None] + r[None, :])
    ii, jj = np.nonzero(np.triu(slack < 4.0 * t + 1e-12, 1))
    m = ii.size
    if m:
        dd = D[ii, jj]
        ex, ey = (x[ii] - x[jj]) / dd, (y[ii] - y[jj]) / dd
        prow = nw + np.arange(m)
        row += [prow] * 6
        col += [ii, jj, n + ii, n + jj, 2 * n + ii, 2 * n + jj]
        val += [-ex, ex, -ey, ey, np.ones(m), np.ones(m)]
        b.append(slack[ii, jj])
    A = coo_matrix((np.concatenate(val), (np.concatenate(row), np.concatenate(col))),
                   shape=(nw + m, nv)).tocsr()
    cost = np.zeros(nv)
    cost[2 * n:] = -1.0                      # maximize sum(w)
    lb = np.concatenate([np.full(2 * n, -t), np.maximum(-t, -0.9 * r)])
    ub = np.concatenate([np.full(2 * n, t), np.full(n, t)])
    bnd = np.stack([lb, ub], axis=1)
    bb = np.concatenate(b)
    res = linprog(cost, A_ub=A, b_ub=bb, bounds=bnd, method=_LPMETH)
    if not res.success or res.x is None:
        res = linprog(cost, A_ub=A, b_ub=bb, bounds=bnd, method="highs")
    if not res.success or res.x is None:
        return None, 0.0
    return res.x, -float(res.fun)


def _slp(xy, r, rng, reg, deadline, t0=0.03, tmin=3e-8, max_it=300):
    """Sequential-LP ascent from (xy, r).  Returns (xy, r, sum_r) at the local optimum.
    `reg(xy, r)` registers a candidate through evaluate() (monotone, so every call is a new best)."""
    n = len(r)
    xy, r, D = _make_feasible(xy, r)
    cur = float(r.sum())
    t = t0
    for _ in range(max_it):
        if time.process_time() > deadline or t <= tmin:
            break
        z, lpval = _lp_step(xy, r, D, t)
        if z is None or lpval < _LPZERO:
            break                            # first-order stationary: no smaller t can help
        nxy = xy + np.stack([z[:n], z[n:2 * n]], axis=1)
        nr = r + z[2 * n:]
        nxy, nr, nD = _make_feasible(nxy, nr)
        s = float(nr.sum())
        if s > cur + 1e-13:
            xy, r, D, gain = nxy, nr, nD, s - cur
            cur = s
            reg(xy, r)
            if gain < 0.25 * t:              # step was throttled by curvature, not by the bound
                t *= 0.6
        else:
            t *= 0.35                        # the LP promised a gain the true geometry did not pay
    return xy, r, cur


# ---------------------------------------------------------------- starts and moves

def _init_radii(xy):
    """Equal-ish feasible radii for a fresh centre set: r_i = min(wall_i, min_j D_ij/2)."""
    D = _dist(xy)
    return np.maximum(np.minimum(_wall(xy), D.min(axis=1) / 2.0) - _EPS, _RTINY)


def _rowed(n, rows, rng, jitter=0.0):
    """n centres laid out in `rows` rows; the surplus goes to alternating rows, which makes the row
    counts differ by one and the rows interlock -- a hexagonal-flavoured lattice.  Square-ish grids
    fall out of the same code when rows divides n."""
    rows = int(max(1, min(n, rows)))
    q, rem = divmod(n, rows)
    cnt = np.full(rows, q)
    if rem:
        order = np.concatenate([np.arange(0, rows, 2), np.arange(1, rows, 2)])
        cnt[order[:rem]] += 1
    ys = np.linspace(LO + 0.5 / rows, HI - 0.5 / rows, rows) if rows > 1 else np.array([0.0])
    pts = []
    for k in range(rows):
        c = int(cnt[k])
        if c <= 0:
            continue
        xs = np.linspace(LO + 0.5 / c, HI - 0.5 / c, c) if c > 1 else np.array([0.0])
        pts.append(np.stack([xs, np.full(c, ys[k])], axis=1))
    xy = np.concatenate(pts, axis=0)
    if jitter > 0:
        xy = xy + rng.normal(0.0, jitter / max(1, int(np.sqrt(n))), size=xy.shape)
    return np.clip(xy, LO + 1e-4, HI - 1e-4)


def _hexlat(n, cols, rng, jitter=0.0):
    """A genuinely interlocked hex lattice: every row uses the SAME horizontal spacing s = 1/cols, and
    odd rows are shifted by s/2 and hold one circle fewer (_rowed instead re-spreads each row over the
    full width, which breaks the interlock).  Rows are added until n circles are placed; the last row is
    truncated.  This is the topology the csqv records actually have."""
    cols = int(max(1, min(n, cols)))
    s = 1.0 / cols
    pts = []
    k = 0
    while len(pts) < n:
        # cols == 1 has no "one fewer" odd row (c would be 0 and the loop used to break with a SINGLE
        # centre placed, i.e. a packing of the wrong size -- evaluate() raises ValueError on that and
        # the driver then ends the WHOLE call, losing every n after it).  Degenerate to a single column.
        c = cols if (k % 2 == 0 or cols == 1) else cols - 1
        if c <= 0:
            break
        x0 = LO + s / 2.0 if k % 2 == 0 else LO + s
        for i in range(c):
            pts.append((x0 + i * s, float(k)))
            if len(pts) >= n:
                break
        k += 1
    if not pts:
        return _rowed(n, 1, rng)
    a = np.array(pts, dtype=float)
    rows = a[:, 1].max() + 1.0
    a[:, 1] = LO + 0.5 / rows + a[:, 1] * (1.0 / rows) if rows > 1 else 0.0
    if jitter > 0:
        a = a + rng.normal(0.0, jitter * s, size=a.shape)
    return np.clip(a, LO + 1e-4, HI - 1e-4)


def _hexrot(n, rng, theta=0.0, off=(0.0, 0.0), jitter=0.0):
    """n centres on a hex lattice of spacing s, ROTATED by theta and shifted by off, clipped to the
    square by keeping the n most interior points.

    Why rotation is its own start dimension: _rowed/_hexlat can only produce axis-aligned rows, but the
    densest clipped hex patch in a SQUARE is often tilted -- the lattice trades a worse fit along one
    wall for a better one along the other three.  Since sum(r) for a near-monodisperse packing is set by
    how many lattice sites fit at a given spacing, a tilt that fits one more interior site at the same s
    is a strictly better basin, and no amount of local ascent will rotate a lattice into it (that is a
    global rearrangement, not a first-order move).  s starts at the value that puts exactly n hex cells
    in the unit square and shrinks until n sites actually fit inside the walls.
    """
    s = float(np.sqrt(2.0 / (np.sqrt(3.0) * max(1, n))))
    c, sn = np.cos(theta), np.sin(theta)
    for _ in range(60):
        h = s * np.sqrt(3.0) / 2.0
        K = int(2.0 / min(s, h)) + 3
        i = np.arange(-K, K + 1, dtype=float)
        I, J = np.meshgrid(i, i, indexing="ij")
        px = I.ravel() * s + J.ravel() * (s / 2.0)
        py = J.ravel() * h
        qx = c * px - sn * py + off[0]
        qy = sn * px + c * py + off[1]
        clr = np.minimum(np.minimum(HI - qx, qx - LO), np.minimum(HI - qy, qy - LO))
        keep = clr > 1e-4
        if int(keep.sum()) >= n:
            o = np.argsort(-clr[keep])[:n]     # the n most interior sites = the compact clipped patch
            a = np.stack([qx[keep][o], qy[keep][o]], axis=1)
            if jitter > 0:
                a = a + rng.normal(0.0, jitter * s, size=a.shape)
            return np.clip(a, LO + 1e-4, HI - 1e-4)
        s *= 0.96
    return _rowed(n, int(round(np.sqrt(n))), rng)



def _sym(n, rng, theta=0.0, off=(0.0, 0.0), diag=False, sfac=1.0, jitter=0.0):
    """A start that is EXACTLY mirror-symmetric, built by FOLDING a clipped lattice patch.

    Why this is its own family.  Every other start here is a lattice patch, and a lattice patch is
    mirror-symmetric in the square only for the two canonical tilts with an aligned offset -- so the
    randomized `rot` tail, which is where most of the CPU goes, samples continuous (theta, offset) and
    therefore draws an ASYMMETRIC patch with probability 1.  But the csqv optima are overwhelmingly
    symmetric: a reflection maps a packing to a packing of the same sum(r), so optima come in mirror
    orbits and the generic one is its own mirror.  Folding decouples the two: the lattice supplies the
    local hex texture at ANY tilt, and the fold supplies the global symmetry, giving a family the tail
    could not otherwise reach.  It also halves the effective search dimension, which is the whole point
    of a symmetry-restricted search -- without needing a restricted optimizer, because the SLP that
    follows is free to break the symmetry again.

    Construction: take the rotated/offset lattice sites inside the square, keep the ones strictly on the
    + side of the mirror, snap the ones sitting ON it (|d| <= s/4) to the mirror, and reflect.  Exactly
    n centres are returned: z axis sites (z parity-matched to n) plus (n - z)/2 mirrored pairs, both
    chosen most-interior-first.  Shrinks the spacing until that many sites exist.
    """
    s = float(np.sqrt(2.0 / (np.sqrt(3.0) * max(1, n)))) * float(sfac)
    c, sn = np.cos(theta), np.sin(theta)
    # mirror normal: x -> -x, or the anti-diagonal fold about y = x
    nx, ny = (np.sqrt(0.5), -np.sqrt(0.5)) if diag else (1.0, 0.0)
    for _ in range(60):
        h = s * np.sqrt(3.0) / 2.0
        K = int(2.0 / min(s, h)) + 3
        i = np.arange(-K, K + 1, dtype=float)
        I, J = np.meshgrid(i, i, indexing="ij")
        px = I.ravel() * s + J.ravel() * (s / 2.0)
        py = J.ravel() * h
        qx = c * px - sn * py + off[0]
        qy = sn * px + c * py + off[1]
        d = nx * qx + ny * qy                      # signed distance to the mirror
        onax = np.abs(d) <= 0.25 * s
        # snap the near-axis sites onto the mirror, then keep the + side
        ax = np.stack([qx[onax] - d[onax] * nx, qy[onax] - d[onax] * ny], axis=1)
        pos = d > 0.25 * s
        pp = np.stack([qx[pos], qy[pos]], axis=1)

        def _clr(a):
            return np.minimum(np.minimum(HI - a[:, 0], a[:, 0] - LO),
                              np.minimum(HI - a[:, 1], a[:, 1] - LO))
        if len(ax):
            ax = ax[_clr(ax) > 1e-4]
        if len(pp):
            pp = pp[_clr(pp) > 1e-4]
        # z = axis sites used; must match n's parity and leave (n - z)/2 pairs available
        zmax = min(len(ax), n)
        z = -1
        for cand in range(n % 2, zmax + 1, 2):
            if (n - cand) // 2 <= len(pp):
                z = cand                            # smallest parity-legal z that the pairs can cover
                break
        if z >= 0:
            m = (n - z) // 2
            a0 = ax[np.argsort(-_clr(ax))[:z]] if z else np.zeros((0, 2))
            p0 = pp[np.argsort(-_clr(pp))[:m]] if m else np.zeros((0, 2))
            mir = p0.copy()
            if len(mir):
                dd = nx * mir[:, 0] + ny * mir[:, 1]
                mir = mir - 2.0 * dd[:, None] * np.array([nx, ny])
            a = np.concatenate([a0, p0, mir], axis=0)
            if len(a) == n:
                if jitter > 0:
                    a = a + rng.normal(0.0, jitter * s, size=a.shape)
                return np.clip(a, LO + 1e-4, HI - 1e-4)
        s *= 0.95
    return _hexrot(n, rng, theta=theta, off=off, jitter=jitter)


def _pen_obj(v, n, mu):
    """f = -sum(r) + mu*(sum_{i<j} overlap_ij^2 + sum_i sum_walls violation^2), with its gradient.

    This is the one objective in the solver that is defined on INFEASIBLE configurations, which is the
    entire point: see _relax.
    """
    x, y, r = v[:n], v[n:2 * n], v[2 * n:]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    gp = np.maximum((r[:, None] + r[None, :]) - d, 0.0)   # overlap depth, 0 on the diagonal (d=inf)
    wl = np.stack([r - (x - LO), r - (HI - x), r - (y - LO), r - (HI - y)])
    wp = np.maximum(wl, 0.0)
    f = -float(r.sum()) + mu * (0.5 * float((gp * gp).sum()) + float((wp * wp).sum()))
    # d is floored before it becomes a DENOMINATOR: two centres can land exactly on top of each other
    # here (this objective is defined on infeasible points, and L-BFGS clips against the bounds), and
    # 0/0 would put a nan in the gradient -- which L-BFGS-B does not report, it just stops, silently
    # returning the point it started from.  The diagonal is already inf, so dx/inf = 0 there.
    dsafe = np.maximum(d, 1e-12)
    ux = dx / dsafe
    uy = dy / dsafe
    gx = -2.0 * mu * (gp * ux).sum(axis=1) + 2.0 * mu * (wp[1] - wp[0])
    gy = -2.0 * mu * (gp * uy).sum(axis=1) + 2.0 * mu * (wp[3] - wp[2])
    gr = -np.ones(n) + 2.0 * mu * gp.sum(axis=1) + 2.0 * mu * wp.sum(axis=0)
    return f, np.concatenate([gx, gy, gr])


def _relax(xy, r, rng, lam=1.12, swaps=0, mu0=3e2, levels=4, it=60, deadline=None):
    """INFLATE & RELAX: the only move in the solver that is allowed to pass through INFEASIBILITY.

    Why this is not just another shake.  Every other move here -- teleport, local/global shake,
    crossover -- is repaired to strict feasibility and then handed to the SLP, and the SLP's convex
    restriction keeps every iterate feasible by construction.  So the whole search only ever moves
    along feasible paths, and along a feasible path two touching circles can never swap sides and a
    small circle can never tunnel past a large one: the CONTACT TOPOLOGY can only change by a discrete
    jump (teleport / crossover seam), never by a continuous rearrangement.  Inflating every radius by
    `lam` makes the packing overlap, and minimizing the quadratic-penalty objective with the penalty
    weight ramped up from `mu0` lets the circles push through each other while they settle -- a melt
    and re-freeze.  The result is deliberately slightly infeasible; `_make_feasible` + `_polish` project
    it back exactly, and the SLP then finishes it, so the ONLY thing this move contributes is topology.

    `swaps` random radius transpositions first (a big circle dropped where a small one sat, and vice
    versa) make that rearrangement violent on purpose: after the swap the configuration is locally
    impossible, so the relaxation cannot return it unchanged.
    """
    n = len(r)
    if n < 2:
        return None
    r2 = r * float(lam)
    xy2 = xy.copy()
    for _ in range(int(swaps)):
        i, j = int(rng.randint(n)), int(rng.randint(n))
        if i != j:
            r2[i], r2[j] = r2[j], r2[i]
    v = np.concatenate([xy2[:, 0], xy2[:, 1], r2])
    lo = np.concatenate([np.full(2 * n, LO), np.full(n, _RTINY)])
    up = np.concatenate([np.full(2 * n, HI), np.full(n, 0.5)])
    bnd = list(zip(lo, up))
    mu = float(mu0)
    for lv in range(int(levels)):
        # Poll the clock BETWEEN levels.  This was the only move in the solver that checked no deadline
        # at all, and it is the most expensive one (levels x it L-BFGS-B iterations on 3n variables),
        # so it was the main source of the measured CPU overshoot past the call's allowance.  Stopping
        # early only means a coarser melt, and the melt is deliberately infeasible anyway: whatever it
        # returns is projected back by _make_feasible/_polish and finished by the SLP, so a truncated
        # relaxation is a weaker probe, never an invalid one.  The first level always runs, otherwise
        # this returns the inflated (infeasible) input unchanged.
        if lv and deadline is not None and time.process_time() > deadline:
            break
        res = minimize(_pen_obj, v, args=(n, mu), jac=True, method="L-BFGS-B",
                       bounds=bnd, options={"maxiter": int(it)})
        if res.x is None or not np.all(np.isfinite(res.x)):
            break
        v = res.x
        mu *= 10.0
    if not np.all(np.isfinite(v)):
        return None
    out_xy = np.stack([v[:n], v[n:2 * n]], axis=1)
    out_r = np.maximum(v[2 * n:], _RTINY)
    return out_xy, out_r


def _local_shake(xy, r, rng, frac=0.25):
    """Rearrange ONE neighbourhood: displace the circles nearest a random seed strongly, leave the rest
    of the packing alone.  A global shake damages good structure everywhere at once; most escapes only
    need a single region re-solved, so this move has a far better accept rate at the same amplitude."""
    n = len(r)
    k = int(max(3, min(n, round(frac * n))))
    d = ((xy - xy[rng.randint(n)]) ** 2).sum(1)
    idx = np.argsort(d)[:k]
    out = xy.copy()
    out[idx] += rng.normal(0.0, 0.6 * float(np.mean(r)), size=(k, 2))
    return np.clip(out, LO + 1e-4, HI - 1e-4), r.copy()


_PACK_CACHE = {}
_CH = []                                  # memo for _census_hash (one census per process)
_PCKRE = re.compile(r"^csqv(\d+)\.pck$")


def _read_pack(m):
    """Guarded read of ANY committed census file (returns None unless it parses as exactly m circles).
    solve() may be called for an n with no pack -- and the neighbour transfer below asks for packs at
    m != n, most of which may not exist -- so every read has to fail soft.

    Cached on (path, mtime_ns, size).  The transfer family is now sampled an UNBOUNDED number of times
    per size (see _start_stream), so the same donor file would otherwise be re-parsed hundreds of times
    per second; keying on the stat means a pack rewritten between two solve() calls in one process
    invalidates itself rather than going stale."""
    p = "bench/packs/csqv%d.pck" % m
    try:
        st = os.stat(p)
    except OSError:
        return None
    key = (p, st.st_mtime_ns, st.st_size)
    if key in _PACK_CACHE:
        hit = _PACK_CACHE[key]
        return None if hit is None else (hit[0].copy(), hit[1].copy())
    a = None
    try:
        rows = [ln.split() for ln in open(p).read().splitlines() if ln.strip()][2:]
        a = np.array([[float(v) for v in row[:3]] for row in rows], dtype=float)
    except (OSError, ValueError, IndexError):
        a = None
    if a is None or a.ndim != 2 or a.shape[0] != m or a.shape[1] < 3 or not np.all(np.isfinite(a)):
        if len(_PACK_CACHE) < 4096:
            _PACK_CACHE[key] = None
        return None
    out = (a[:, :2].copy(), a[:, 2].copy())
    if len(_PACK_CACHE) < 4096:
        _PACK_CACHE[key] = out
    return out[0].copy(), out[1].copy()


def _warm(n):
    """The census warm start for n itself."""
    return _read_pack(n)


def _insert(xy, r, k, rng, ns=2048, soft=0.0):
    """Add k circles at the roomiest empty spots (max clearance to every circle and to the walls).

    `soft` > 0 samples the hole with probability proportional to clearance**soft instead of taking the
    argmax.  The greedy pick gives ONE deterministic start per donor pack; a square packing normally
    has several near-equally roomy holes, and which of them the new circles go into is exactly the
    combinatorial choice the SLP cannot make for itself, so sampling it turns a single start into an
    unbounded family of structurally distinct ones."""
    xy, r = xy.copy(), r.copy()
    for _ in range(k):
        p = rng.uniform(LO, HI, size=(ns, 2))
        clr = np.sqrt(((p[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]
        clr = np.minimum(clr.min(axis=1), _wall(p))
        j = -1
        if soft > 0.0:
            w = np.maximum(clr, 0.0) ** soft
            tot = float(w.sum())
            if np.isfinite(tot) and tot > 0.0:
                j = int(rng.choice(ns, p=w / tot))
        if j < 0:
            j = int(np.argmax(clr))
        xy = np.concatenate([xy, p[j][None, :]], axis=0)
        r = np.concatenate([r, [max(_RTINY, 0.9 * float(clr[j]))]])
    return xy, r


def _transfer(n, m, rng, mode="greedy"):
    """NEIGHBOUR TRANSFER: build a start for n out of the committed packing for a DIFFERENT size m.

    Why this is the strongest start family available.  The census is append-or-improve, so after a few
    iterations some sizes sit AT the record (relgap ~1e-12) while their neighbours stall at ~1e-3.  The
    measured failure is basin selection, and the optimal contact graphs of n and n+-1 or n+-2 differ
    only locally -- the rest of the packing is the same near-hex arrangement.  A random lattice start
    throws that away and re-derives it; a transfer keeps it and lets the SLP repair only the seam.  It
    also flows BOTH ways, so a size that is solved well seeds its neighbours on the next iteration, and
    it needs no per-n knowledge (it reads whatever the solver itself produced earlier).

    m > n: drop m-n circles.  m < n: insert n-m circles at the roomiest holes.  Returns None whenever
    pack m is missing, so any n and any census state is safe.

    `mode` is the combinatorial choice the SLP cannot make for itself, and it is the whole point of
    this function being sampled repeatedly rather than once:
      "greedy"  -- drop exactly the m-n smallest circles / insert at the argmax hole.  One
                   deterministic start per donor; the obvious first thing to try.
      "small"   -- drop m-n circles sampled from the smallest few, weighted toward the smallest.
      "cluster" -- drop the m-n circles nearest a random point, which opens ONE large hole instead of
                   m-n scattered small ones -- a structurally different reseating of the whole
                   neighbourhood, not a perturbation of the same contact graph.
    Anything other than "greedy" also samples the insertion holes (see _insert), so each call is a
    fresh member of the family.
    """
    p = _read_pack(m)
    if p is None or m < 1:
        return None
    xy, r = p
    if m > n:
        k = m - n
        drop = None
        if mode == "cluster" and k < m:
            c = rng.uniform(LO, HI, size=2)
            drop = np.argsort(((xy - c[None, :]) ** 2).sum(1))[:k]
        elif mode == "small" and k < m:
            pool = np.argsort(r)[:min(m - 1, max(k + 1, 3 * k))]
            w = 1.0 / (r[pool] + 1e-9)
            tot = float(w.sum())
            if np.isfinite(tot) and tot > 0.0:
                drop = rng.choice(pool, size=k, replace=False, p=w / tot)
        if drop is None:
            keep = np.argsort(-r)[:n]
        else:
            mask = np.ones(m, dtype=bool)
            mask[np.asarray(drop, dtype=int)] = False
            keep = np.nonzero(mask)[0]
        xy, r = xy[keep].copy(), r[keep].copy()
    elif m < n:
        xy, r = _insert(xy, r, n - m, rng, soft=(0.0 if mode == "greedy" else 8.0))
    return xy, r


def _donors(n, recs, span=6):
    """Donor sizes for a neighbour transfer, BEST DONOR FIRST.

    The old order was "nearest size first" (n+-1, n+-2, ...).  That is the wrong key: the value of a
    donor is how close ITS OWN packing is to the record, because a record-quality donor carries the
    correct contact structure while a stalled neighbour carries the same wrong basin we are trying to
    escape.  n=37, for instance, has no capped neighbour within 2 but two of them at distance 4.  Sorts
    by (relgap of donor pack, |m-n|), so quality dominates and distance only breaks ties; a donor with
    no record entry sorts last, which reduces exactly to the old distance order when no records apply
    (the offline re-run on held-out sizes)."""
    out = []
    for d in range(1, span + 1):
        for m in (n - d, n + d):
            if m < 1:
                continue
            w = _read_pack(m)
            if w is None:
                continue
            R = recs.get(m)
            g = 1.0
            if R:
                v = float(w[1].sum())
                if np.isfinite(v):
                    g = max(0.0, (R - v) / R)
            out.append((g, d, m))
    out.sort()
    return [m for _, _, m in out]


def _xover(A, B, rng, alpha=0.72):
    """REGION CROSSOVER of two local optima for the SAME n: take A's circles inside a random region and
    B's outside it, drop the near-duplicates the seam creates, and top the count back up to n.

    Why this move and not another perturbation.  Every existing escape move (teleport / local shake /
    global shake) perturbs ONE incumbent, so the only structural information in the search is whatever
    that single packing already holds -- and the measured failure here is basin selection, not
    precision.  Two distinct local optima for the same n usually differ only over PART of the square:
    one has the better corner, the other the better interior seam, and each is locally optimal so no
    first-order move can import the other's good region.  That is exactly the situation a region
    recombination is for, and it is the same argument `_transfer` already makes for a neighbouring SIZE
    -- applied to two optima of the size we are actually solving, where the structures are far more
    compatible.

    Repair.  The union of a region of A and the complement region of B has ~n circles but the seam can
    place two centres almost on top of each other; a global feasibility scale would then crush EVERY
    radius.  So the centres are accepted greedily largest-inherited-radius first, rejecting one that
    sits closer than `alpha*(r_i+r_j)` to an accepted one (alpha < 1 keeps mild overlap, which the
    grow passes in _polish repay), and any shortfall is filled at the roomiest holes by _insert.
    Returns exactly n circles, or None if the cut was degenerate.
    """
    axy, ar = A
    bxy, br = B
    n = len(ar)
    if n < 4 or len(br) != n:
        return None
    if rng.rand() < 0.5:                       # half-plane cut at a random angle and offset
        th = rng.rand() * 2.0 * np.pi
        nv = np.array([np.cos(th), np.sin(th)])
        c = float(rng.uniform(-0.32, 0.32))
        sa = (axy @ nv) <= c
        sb = (bxy @ nv) > c
    else:                                      # disc cut: import one compact neighbourhood of A
        cc = rng.uniform(LO * 0.85, HI * 0.85, size=2)
        rad2 = (0.14 + 0.34 * rng.rand()) ** 2
        sa = ((axy - cc[None, :]) ** 2).sum(1) <= rad2
        sb = ((bxy - cc[None, :]) ** 2).sum(1) > rad2
    xy = np.concatenate([axy[sa], bxy[sb]], axis=0)
    r = np.concatenate([ar[sa], br[sb]])
    if len(r) < 2 or len(r) < n - (n // 2):
        return None                            # degenerate cut: filling it would cost more than a start
    keep = []
    for i in np.argsort(-r):
        i = int(i)
        if keep:
            k = np.asarray(keep)
            d2 = ((xy[k] - xy[i]) ** 2).sum(1)
            lim = alpha * (r[k] + r[i])
            if np.any(d2 < lim * lim):
                continue
        keep.append(i)
        if len(keep) == n:
            break
    xy, r = xy[keep].copy(), r[keep].copy()
    if len(r) < n:
        xy, r = _insert(xy, r, n - len(r), rng, soft=(0.0 if rng.rand() < 0.5 else 8.0))
    if xy.shape != (n, 2):
        return None
    return xy, r


def _teleport(xy, r, rng, k=1, ns=1536):
    """Move the k smallest circles into the roomiest empty spots (largest clearance to every other
    circle and to the walls) -- the escape move that unsticks a locally optimal packing."""
    xy, r = xy.copy(), r.copy()
    for _ in range(k):
        i = int(np.argmin(r))
        keep = np.ones(len(r), dtype=bool)
        keep[i] = False
        c, rr = xy[keep], r[keep]
        p = rng.uniform(LO, HI, size=(ns, 2))
        clr = np.sqrt(((p[:, None, :] - c[None, :, :]) ** 2).sum(-1)) - rr[None, :]
        clr = np.minimum(clr.min(axis=1), _wall(p))
        j = int(np.argmax(clr))
        xy[i] = p[j]
        r[i] = max(_RTINY, clr[j] * 0.9)
    return xy, r


def _census_hash():
    """A deterministic fingerprint of the WHOLE committed census (every pack's sum(r)).

    Guarded and cached: an unreadable directory, or no census at all (the offline re-run on held-out
    sizes), simply contributes 0.  Reads go through _read_pack, which is itself stat-cached.
    """
    if _CH:
        return _CH[0]
    h = 0
    try:
        names = sorted(os.listdir("bench/packs"))
    except OSError:
        names = []
    for nm in names:
        mo = _PCKRE.match(nm)
        if mo is None:
            continue
        w = _read_pack(int(mo.group(1)))
        if w is None:
            continue
        v = float(w[1].sum())
        if np.isfinite(v):
            h = (h * 1000003 + int(abs(v) * 1e12)) % 2147483647
    _CH.append(h)
    return h


def _seed_for(n, rng):
    """A per-n RandomState seeded from `rng`, from the committed pack for n, AND from the hash of the
    WHOLE census.

    Why this exists.  The driver hands solve() a FIXED seed (0 every iteration) and the structured-start
    family is otherwise deterministic, so without a census term every iteration re-derives the very same
    local optima for a given n -- i.e. the majority of the CPU is spent re-proving that starts which
    already failed still fail.

    Why the GLOBAL hash and not just this n's pack (measured, iteration 10).  Keying only on n's own
    pack decorrelates a size exactly when it IMPROVES -- and never when it does not.  That is backwards:
    the sizes that matter are precisely the stuck ones, whose pack is by definition unchanged, so they
    repeated a bit-identical 9 s search every iteration (n=49's pack has not moved since iteration 5,
    n=35's since iteration 4; the same determinism is why whole A/B cells came back bit-identical in
    iterations 8 and 9).  Hashing every pack means ANY improvement anywhere in the census re-randomizes
    EVERY size's stream, while reproducibility is untouched: same files + same rng => same stream.
    An n with no pack, or no census at all, contributes 0 and simply follows `rng`.
    """
    h = _census_hash()
    w = _read_pack(n)
    if w is not None:
        v = float(w[1].sum())
        if np.isfinite(v):
            h ^= int(abs(v) * 1e12) % 2147483647
    return np.random.RandomState((int(rng.randint(0, 2147483647)) ^ h) % 2147483647)


T0_LO, T0_HI = 0.12, 1.00   # trust-region start, in units of the lattice spacing (see _rand_t0)


def _rand_t0(n, rng):
    """Draw the SLP's INITIAL trust radius log-uniformly in [T0_LO, T0_HI] * lattice spacing.

    t0 was a fixed 0.03 for every n, which is not a neutral choice but a hidden, size-dependent one:
    the natural length here is the lattice spacing sp = sqrt(2/(sqrt3 n)), and 0.03 is 0.14*sp at n=27
    but 0.28*sp at n=99, so the ascent was relatively twice as mobile at the large sizes -- which is
    exactly where this census already succeeds.

    MEASURED (artifacts/exp_t0.py, matched-CPU best-of-K, n = 37/49/81 x seeds 0/1, K within +-30% of
    each other so this costs no throughput): no single t0 wins everywhere, and the differences are
    basin-level, not precision-level.  n=37 is won by the small end (0.03 == 0.17*sp and 0.15*sp both
    reach 3.156979747; 0.30*sp and above stall at 3.155858962), n=81 by the large end (0.30/0.50/1.00*sp
    reach 4.721873120, 0.15*sp does not), and n=49 only by the ends TOGETHER -- 0.15*sp, 0.50*sp and
    1.00*sp each reach 3.648307312, which is ABOVE the committed 3.647064354, while the fixed 0.03
    never gets past the committed value on either seed.

    So t0 is a SEARCH DIMENSION, not a constant: every other family here diversifies the START, none
    diversified the ASCENT, and two ascents with different t0 from the SAME start land in different
    basins.  Sampling it per start is therefore a free extra axis of diversification -- the one thing
    the notes say actually pays here (a size jumping to the cap is a lottery ticket, so more distinct
    tickets beat more depth on one).  Scale-free so it transfers to the never-scored offline sizes.
    """
    sp = float(np.sqrt(2.0 / (np.sqrt(3.0) * max(1, n))))
    u = float(rng.rand())
    return sp * (T0_LO * (T0_HI / T0_LO) ** u)


def _start_stream(n, rng):
    """An UNBOUNDED stream of starts, most promising first, then randomized for ever.

    The old version was a fixed list of ~25 starts, which at ~5 s per size consumed over half the
    allowance re-running one SLP per entry.  Two things are wrong with a fixed list: it re-tries the
    same tilts every iteration (see _seed_for), and tilt/offset are CONTINUOUS dimensions that five
    hard-coded angles barely sample.  A generator lets the start phase be bounded by TIME instead of by
    a list length, so a size with cheap LPs gets many more distinct structures for the same CPU.
    """
    yield ("warm", None)
    # neighbour transfers, BEST DONOR first (see _donors/_transfer): the census's own good packings are
    # the best structural priors available, and they cost one cached file read each.
    don = _donors(n, _records())
    for m in don:
        yield ("xfer", m)
    q = np.pi / 6.0
    base = int(round(np.sqrt(n / 0.866)))
    sp = float(np.sqrt(2.0 / (np.sqrt(3.0) * max(1, n))))
    # the two canonical hex extremes (axis-aligned and "pointy"), then the row-count families
    for th in (0.0, q):
        yield ("rot", (th, (0.0, 0.0), 0.004))
    # the two canonical FOLDED patches (axis mirror, diagonal mirror) -- see _sym
    for dg in (False, True):
        yield ("sym", (0.0, (0.0, 0.0), dg, 1.0, 0.004))
    for d in (0, 1, -1, 2, -2):
        yield ("lat", base + d)
    for d in (0, 1, -1, 2, -2, 3):
        yield ("hex", base + d)
    yield ("sq", int(round(np.sqrt(n))))
    yield ("sq", int(round(np.sqrt(n))) + 1)
    # The tail used to be lattice-only, so the family this solver calls its STRONGEST prior was sampled
    # at exactly one deterministic point per donor and then never again.  The tail now returns to the
    # donors with a RANDOMIZED removal/insertion (see _transfer modes), biased toward the best donors.
    # MEASURED, and the number matters: at a 0.34 share (taken out of the rotated-lattice family) this
    # was a clear LOSS -- mean digits over n = 31, 37, 49, 81 at 9 s fell 3.958 -> 2.940 (seed 0) and
    # 2.988 -> 2.831 (seed 1), because `rot` is where the winning basins actually come from (seed 0
    # reaches the n=31 record from a rotated lattice).  So the transfer tail is a small 0.08 SPRINKLE
    # taken out of the lat/hex share, leaving `rot` at the 0.74 it had; the prefix, where a transfer is
    # tried once per donor, is where this family earns its keep.
    while True:
        u = rng.rand()
        if don and u < 0.08:
            m = don[min(len(don) - 1, int(abs(rng.randn()) * 1.6))]
            yield ("xfer", (m, "cluster" if rng.rand() < 0.5 else "small"))
        elif u < 0.08 + 0.74:
            yield ("rot", (rng.rand() * (np.pi / 3.0),
                           (rng.rand() * sp, rng.rand() * sp * np.sqrt(3.0) / 2.0),
                           0.002 + 0.03 * rng.rand()))
        elif u < 0.08 + 0.74 + 0.10:
            # folded patches: continuous tilt/offset/spacing, but globally mirror-symmetric.  The share
            # is taken from the lat/hex slice, NOT from `rot` -- iteration 6 measured that stealing from
            # `rot` costs more than any tail family returns.
            yield ("sym", (rng.rand() * (np.pi / 3.0),
                           (rng.rand() * sp, rng.rand() * sp * np.sqrt(3.0) / 2.0),
                           rng.rand() < 0.35, 0.90 + 0.25 * rng.rand(),
                           0.002 + 0.02 * rng.rand()))
        elif u < 0.08 + 0.74 + 0.10 + 0.05:
            yield ("lat", base + int(rng.randint(-3, 4)))
        else:
            yield ("hex", base + int(rng.randint(-3, 4)))


# ---------------------------------------------------------------- driver

def _records():
    """Guarded read of the frozen record table (values only -- no coordinates).  Used ONLY to decide
    where the CPU goes; a missing/unparseable file just means no size is skipped."""
    try:
        with open("bench/records.json") as f:
            return {int(k): float(v) for k, v in json.load(f)["records"].items()}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _is_capped(n, recs):
    """True if the COMMITTED pack for n already earns the full 7 digits (relgap <= 1e-7).

    The score is `mean over all 37 visible n of clamp(-log10(relgap), 0, 7)` and the census is
    append-or-improve, so a size already at the 7-digit cap cannot yield another point of SCORE no
    matter how much CPU it is given -- while its neighbours sit 4-5 digits short.  Spending the uniform
    1/37 share on it is therefore pure waste (measured: 13 of 37 sizes, i.e. 35% of the whole CPU
    allowance).  Skipping it is safe because the driver only ever OVERWRITES a pack with a strictly
    better one: a size we do not touch keeps exactly what is committed.
    """
    if n not in recs:
        return False
    w = _read_pack(n)
    if w is None:
        return False
    s = float(w[1].sum())
    return (recs[n] - s) / recs[n] <= DIGIT_CAP_RELGAP


def _deep_order(active, recs):
    """Order the active sizes for the DEEP prefix: SMALLEST remaining relative gap first (a size with
    no committed pack at all still sorts ahead of everything).

    Only the first `total // MIN_CPU_PER_N` sizes of this list get a real slice, so this ordering IS
    the allocation.  Iteration 14 ordered by LARGEST gap on an upside-bound argument (a size at d
    digits can gain at most 7 - d).  Two rounds of per-size measurement say the upside bound is the
    wrong half of the product: what varies across sizes is not the upside, it is P(improve), and it
    varies by a factor of infinity, not by a factor of two.

      artifacts/rank14/  -- 10 active sizes x 3 seeds x 33 s:  movers n=49, n=51.
      artifacts/deep15/  -- 10 active sizes x 2 seeds x 55 s:  movers n=51 (RECORD 2/2), n=63 (1/2).
      The other eight (35, 37, 39, 41, 47, 49, 59, 81) return the committed sum_r to the last digit
      in EVERY one of their five cells across the two rounds.

    Ranked by gap, the two sizes that reach the record are the 1st and 3rd SMALLEST gaps of the ten,
    and the three largest gaps (49, 35, 59) are frozen -- so largest-gap-first points the whole
    allowance at a measured zero, and smallest-gap-first is the ordering that captures the measured
    gain.  The mechanism this is consistent with: a size already within ~1e-4 of the record holds
    essentially the right structure and a deep slice can finish it, while a size 4e-4 away holds the
    wrong structure and needs a basin no random start finds in a minute.

    Honest limits, stated because the same cells refute the stronger claim.  (i) n=41 sits at 9.3e-5 --
    the 2nd smallest gap -- and is frozen in all five of its cells, so gap does not PREDICT movement,
    it only ranks better than its inverse.  (ii) n=49 moved last round from the LARGEST gap of the ten
    (an improvement, not a record).  What both rounds actually agree on is that mover/frozen is a
    stable per-size property; gap is the best in-loop proxy for it available without spending CPU to
    measure it, and it keeps the two properties the policy needs -- it is self-rotating (a size that
    reaches the cap leaves `active` entirely) and it is defined for a never-scored held-out size.
    """
    def key(n):
        w = _read_pack(n)
        rec = recs.get(n)
        if w is None or not rec:
            return -1.0                      # never-scored size: the full 7 digits are on the table
        return max(0.0, (rec - float(w[1].sum())) / rec)
    return sorted(active, key=lambda n: (key(n), n))


def solve(evaluate, meter, rng, targets, cpu_allowance=None):
    if not targets:
        return
    t_entry = time.process_time()
    tg = sorted(targets)
    total = float(cpu_allowance) if cpu_allowance else min(CPU_TOTAL, CPU_PER_TARGET * len(tg))
    end_all = t_entry + total

    # CPU goes only to the sizes that can still gain SCORE (see _is_capped).  If every requested size
    # is already capped -- or nothing can be read -- fall back to working all of them, so the offline
    # re-run on never-scored sizes behaves exactly as before.
    recs = _records()
    active = [n for n in tg if not _is_capped(n, recs)]
    if not active:
        active = list(tg)

    # DEPTH BEATS BREADTH, and breadth here is not free coverage -- it is a choice to give every size
    # too little.  The census is append-or-improve, so a size we never touch keeps exactly what is
    # committed: skipping it costs ZERO.  Splitting the allowance evenly over every active size is
    # therefore not the safe option, it is the one that guarantees no size gets enough CPU to reach a
    # new basin.  When the even share would fall below MIN_CPU_PER_N we instead work a PREFIX deeply
    # and leave the tail untouched, rotating which prefix that is by an offset derived from the census
    # itself -- so successive iterations deep-search different sizes and every size comes round.
    if len(active) * MIN_CPU_PER_N > total and len(active) > 1:
        active = _deep_order(active, recs)

    for pos, n in enumerate(active):
        left_n = len(active) - pos
        now = time.process_time()
        if meter.left() <= 0 or now >= end_all:
            break
        deadline = min(end_all, now + max((end_all - now) / left_n, MIN_CPU_PER_N))

        srng = _seed_for(n, rng)
        donors = _donors(n, recs)
        state = {"s": -np.inf, "r": None}
        rec_n = recs.get(n)
        cap_s = rec_n * (1.0 - DIGIT_CAP_RELGAP) if rec_n else np.inf

        def reg(cxy, cr, _n=n, _st=state):
            """Route a candidate through the metered oracle; the harness keeps the best it is shown.

            The shape guard is not cosmetic: evaluate() RAISES ValueError (it does not return
            "infeasible") for a packing whose circle count is not n, the driver catches that and ends
            the whole solve() call, and every size after this one -- plus anything not yet routed
            through evaluate() -- is lost.  One cheap comparison here makes a malformed start cost a
            single wasted candidate instead of the rest of the iteration."""
            if meter.left() <= 0:
                return
            if cxy.ndim != 2 or cxy.shape != (_n, 2) or cr.shape != (_n,):
                return
            pack = np.concatenate([cxy, cr[:, None]], axis=1)
            feas, s = evaluate(_n, pack)
            if feas and s > _st["s"]:
                _st["s"] = float(s)
                _st["r"] = (cxy.copy(), cr.copy())

        best_xy = best_r = None
        best_s = -np.inf
        # ELITE POOL: the distinct local optima found for this n, best first.  It is what makes the
        # crossover move possible (see _xover); a single-incumbent walker has nothing to recombine.
        pool = []

        def _pool_add(sv, cxy, cr, _p=pool):
            if cxy is None or not np.isfinite(sv):
                return
            sig = np.sort(cr)
            for sv2, _x2, r2 in _p:
                if abs(sv - sv2) < 1e-9 and float(np.abs(sig - np.sort(r2)).sum()) < 1e-7:
                    return                      # structurally the same optimum -- not a new parent
            _p.append((float(sv), cxy.copy(), cr.copy()))
            if len(_p) > POOL:
                _p.sort(key=lambda e: -e[0])
                del _p[POOL:]
        # Spend at most START_FRAC of this n's allowance enumerating structured starts; the rest goes to
        # basin hopping from the best one found.  (Whichever phase is cut short, every improving iterate
        # was already routed through evaluate(), so the harvest is unaffected.)
        t_starts = now + START_FRAC * (deadline - now)
        for kind, arg in _start_stream(n, srng):
            if time.process_time() >= t_starts or meter.left() <= 0 or state["s"] >= cap_s:
                break
            if kind == "warm":
                w0 = _warm(n)
                if w0 is None:
                    continue
                xy0, r0 = _polish(w0[0].copy(), w0[1].copy(), srng, passes=1)
            elif kind == "xfer":
                m_d, mode = arg if isinstance(arg, tuple) else (arg, "greedy")
                tr = _transfer(n, m_d, srng, mode=mode)
                if tr is None:
                    continue
                xy0, r0 = _polish(tr[0], tr[1], srng, passes=2)
            else:
                if kind == "rot":
                    xy0 = _hexrot(n, srng, theta=arg[0], off=arg[1], jitter=arg[2])
                elif kind == "sym":
                    xy0 = _sym(n, srng, theta=arg[0], off=arg[1], diag=arg[2],
                               sfac=arg[3], jitter=arg[4])
                elif kind == "lat":
                    xy0 = _hexlat(n, arg, srng, jitter=0.01)
                else:
                    xy0 = _rowed(n, arg, srng, jitter=0.02 if kind == "hex" else 0.0)
                r0 = _init_radii(xy0)
            xy1, r1, s1 = _slp(xy0, r0, srng, reg, deadline, t0=_rand_t0(n, srng))
            _pool_add(s1, xy1, r1)
            if s1 > best_s:
                best_xy, best_r, best_s = xy1, r1, s1

        # Basin hopping with THRESHOLD ACCEPTANCE.  A strictly greedy walker (what iteration 1 did)
        # re-perturbs the one best local optimum forever, so once every move from it is rejected the
        # search is dead.  Keeping a walker that also accepts small LOSSES lets it drift across the
        # plateau of near-equal local optima -- which is where the record structures sit, since the
        # measured gaps (relgap ~5e-3) are structural, not precision.  `best` is never lost, and the
        # walker is held in a band below it so the drift cannot run away.
        wxy, wr, ws = best_xy, best_r, best_s
        mv = 0
        stall = 0
        while (best_xy is not None and time.process_time() < deadline and meter.left() > 0
               and state["s"] < cap_s):
            mv += 1
            fresh = stall >= STALL
            k3 = mv % 5
            if fresh:
                stall = 0
                # a stalled walker needs a start from a DIFFERENT structure family, and for a size with
                # a record-quality neighbour a randomized transfer is a far better one than a random
                # lattice -- same argument as the start phase, applied to the restart.
                tr = None
                if donors and srng.rand() < 0.25:
                    m_d = donors[min(len(donors) - 1, int(abs(srng.randn()) * 1.6))]
                    tr = _transfer(n, m_d, srng,
                                   mode="cluster" if srng.rand() < 0.5 else "small")
                if tr is not None:
                    pxy, pr = tr
                else:
                    sp = np.sqrt(2.0 / (np.sqrt(3.0) * max(1, n)))
                    pxy = _hexrot(n, srng, theta=srng.rand() * (np.pi / 3.0),
                                  off=(srng.rand() * sp, srng.rand() * sp * np.sqrt(3.0) / 2.0),
                                  jitter=0.01 + 0.04 * srng.rand())
                    pr = _init_radii(pxy)
            elif k3 == 0:
                pxy, pr = _teleport(wxy, wr, srng, k=1 + int(srng.randint(0, 3)))
            elif k3 == 1:
                pxy, pr = _local_shake(wxy, wr, srng, frac=0.12 + 0.3 * srng.rand())
            elif k3 == 4:
                # INFLATE & RELAX, run as a SIDE PROBE off the incumbent best (see _relax).
                #
                # Why it is a probe and not a hop.  Measured (artifacts/ab9_*): as an ordinary escape
                # slot feeding the threshold-accepting walker, this move was a clear LOSS -- summed
                # digits over n = 31/37/49/81 x 2 seeds fell 31.280 -> 25.340, and n=31 stopped reaching
                # the record on BOTH seeds.  Its eval throughput was within 6% of the baseline's, so
                # the cost was not CPU: a melt re-freezes into a DIFFERENT but usually slightly WORSE
                # optimum, which still lands inside the acceptance band, so the walker followed the
                # melts away from the productive region.  Kept off the walker's trajectory the move can
                # only ever add a basin: it updates `best`/`pool` when it wins and is discarded when it
                # does not.
                nsw = 0 if srng.rand() < 0.55 else 1 + int(srng.randint(0, 3))
                rx = _relax(best_xy, best_r, srng,
                            lam=1.03 + 0.22 * srng.rand(), swaps=nsw,
                            mu0=10.0 ** (1.8 + 1.2 * srng.rand()), deadline=deadline)
                if rx is not None:
                    qxy, qr = _polish(rx[0], rx[1], srng, passes=1)
                    qxy, qr, qs = _slp(qxy, qr, srng, reg, deadline, t0=0.02)
                    _pool_add(qs, qxy, qr)
                    if qs > best_s:
                        best_xy, best_r, best_s = qxy, qr, qs
                        stall = 0
                        wxy, wr, ws = qxy, qr, qs      # a genuine win IS worth moving the walker to
                continue
            elif k3 == 3 and len(pool) >= 2:
                # recombine two DISTINCT elites -- one of them the incumbent best half the time, so the
                # move is an import of foreign structure into the champion rather than a random pair.
                i = 0 if srng.rand() < 0.5 else int(srng.randint(len(pool)))
                j = int(srng.randint(len(pool) - 1))
                if j >= i:
                    j += 1
                cx = _xover((pool[i][1], pool[i][2]), (pool[j][1], pool[j][2]), srng)
                if cx is None:
                    pxy, pr = _teleport(wxy, wr, srng, k=1 + int(srng.randint(0, 3)))
                else:
                    pxy, pr = cx
            else:
                amp = (0.15 + 0.3 * srng.rand()) * float(np.mean(wr))
                pxy = np.clip(wxy + srng.normal(0.0, amp, size=wxy.shape), LO + 1e-4, HI - 1e-4)
                pr = wr.copy()
            pxy, pr = _polish(pxy, pr, srng, passes=1)
            xy1, r1, s1 = _slp(pxy, pr, srng, reg, deadline,
                               t0=_rand_t0(n, srng) if fresh else 0.02)
            _pool_add(s1, xy1, r1)
            if s1 > best_s:
                best_xy, best_r, best_s = xy1, r1, s1
                stall = 0
            else:
                stall += 1
            thr = ACCEPT_THR * float(np.mean(r1))
            if fresh or (s1 > ws - thr and s1 > best_s - BAND * thr):
                wxy, wr, ws = xy1, r1, s1
            elif mv % 8 == 0:
                wxy, wr, ws = best_xy, best_r, best_s      # re-anchor on the incumbent

        # final exact growth on the very best, then one last registration
        if state["r"] is not None and meter.left() > 0:
            fxy, fr = state["r"]
            fxy, fr = _polish(fxy, fr, srng, passes=4)
            reg(fxy, fr)


# ---------------------------------------------------------------- self-test

def _self_test():
    """Local mock of the metered oracle: validates feasibility exactly as harness.validate does, keeps
    the best per n, and asserts the invariants the driver relies on -- crucially that solve() works on a
    SECOND call in the same process (a module-level absolute deadline would silently no-op it) and with
    len(targets) == 1."""
    class Meter:
        def __init__(self, b):
            self.budget, self.used = b, 0

        def left(self):
            return max(0, self.budget - self.used)

    class Ev:
        def __init__(self, m):
            self.m, self.best, self.worst_slack = m, {}, np.inf

        def __call__(self, n, pack):
            pack = np.asarray(pack, dtype=float)
            single = pack.ndim == 2
            batch = pack[None] if single else pack
            assert batch.shape[1:] == (n, 3), (batch.shape, n)
            fs, ss = [], []
            for p in batch:
                if self.m.left() <= 0:
                    fs.append(False)
                    ss.append(-np.inf)
                    continue
                self.m.used += 1
                x, y, r = p[:, 0], p[:, 1], p[:, 2]
                sl = [float(np.min(r)), float(np.min(np.minimum.reduce(
                    [x - LO - r, HI - x - r, y - LO - r, HI - y - r])))]
                d = np.sqrt(((p[:, None, :2] - p[None, :, :2]) ** 2).sum(-1)) - (r[:, None] + r[None, :])
                np.fill_diagonal(d, np.inf)
                if n > 1:
                    sl.append(float(np.min(d)))
                ok = sl[0] > 0 and min(sl[1:]) >= -1e-9
                self.worst_slack = min(self.worst_slack, min(sl[1:]))
                s = float(r.sum())
                fs.append(ok)
                ss.append(s if ok else -np.inf)
                if ok and s > self.best.get(n, -np.inf):
                    self.best[n] = s
            return (fs[0], ss[0]) if single else (np.array(fs), np.array(ss))

    recs = {}
    p = "bench/records.json"
    if os.path.exists(p):
        recs = {int(k): v for k, v in json.load(open(p))["records"].items()}

    # allocation unit checks: a size whose committed pack is at the 7-digit cap must be recognised as
    # capped (no CPU), and a size with no pack must NOT be (it needs the full search).
    rc = _records()
    if rc:
        capped = [n for n in sorted(rc) if _is_capped(n, rc)]
        assert not _is_capped(14, rc), "_is_capped must be False for an n with no committed pack"
        assert all(_read_pack(n) is not None for n in capped)
        print("capped (skipped) n = %s" % capped)
    assert _transfer(14, 999999, np.random.RandomState(0)) is None, "transfer from a missing pack must fail soft"
    tr = _transfer(30, 29, np.random.RandomState(0))
    if tr is not None:
        assert tr[0].shape == (30, 2) and tr[1].shape == (30,), "transfer must resize to n"
    tr = _transfer(28, 29, np.random.RandomState(0))
    if tr is not None:
        assert tr[0].shape == (28, 2) and tr[1].shape == (28,), "transfer must resize to n"
    # every randomized transfer mode, in BOTH directions, must resize exactly and stay finite -- these
    # are now drawn an unbounded number of times per size, so a single bad draw would poison a search.
    for mode in ("greedy", "small", "cluster"):
        for (nn, mm) in ((30, 35), (35, 30), (34, 35), (36, 35)):
            t2 = _transfer(nn, mm, np.random.RandomState(7), mode=mode)
            if t2 is None:
                continue
            assert t2[0].shape == (nn, 2) and t2[1].shape == (nn,), (mode, nn, mm, t2[0].shape)
            assert np.all(np.isfinite(t2[0])) and np.all(t2[1] > 0), (mode, nn, mm)
    # donor order: best-relgap donor first, and a size with no census neighbours yields nothing
    if rc:
        dn = _donors(37, rc)
        assert dn, "n=37 must have donors in a seeded census"
        gaps = [max(0.0, (rc[m] - float(_read_pack(m)[1].sum())) / rc[m]) for m in dn if m in rc]
        assert gaps == sorted(gaps), "donors must be ordered best-relgap-first: %s" % dn
        print("donors(37) = %s" % dn[:6])
    assert _donors(10 ** 6, rc) == [], "an n with no census neighbour must yield no donors"
    # EVERY structured start must produce exactly n centres: evaluate() raises on a wrong circle count
    # and the driver then ends the whole call, so a single bad (kind, arg) costs the rest of the run.
    g = np.random.RandomState(3)
    for nn in (2, 3, 5, 14, 27, 31, 50, 99):
        for k in range(1, 12):
            for f in (_rowed, _hexlat):
                a = f(nn, k, g, jitter=0.01)
                assert a.shape == (nn, 2), (f.__name__, nn, k, a.shape)
        a = _hexrot(nn, g, theta=0.3, off=(0.01, 0.02), jitter=0.01)
        assert a.shape == (nn, 2), ("hexrot", nn, a.shape)
        # _sym must return exactly nn centres for EITHER parity and EITHER mirror, at any tilt/spacing
        # (its count is assembled as z axis sites + 2 pairs, so an off-by-one parity bug is the obvious
        # failure mode and would end the whole call via evaluate()'s ValueError) -- and it must really
        # be symmetric: the multiset of centres has to be invariant under the mirror.
        for dg in (False, True):
            for kk in range(4):
                th = float(g.rand() * (np.pi / 3.0))
                a = _sym(nn, g, theta=th, off=(g.rand() * 0.1, g.rand() * 0.1), diag=dg,
                         sfac=0.85 + 0.4 * g.rand())
                assert a.shape == (nn, 2), ("sym", nn, dg, a.shape)
                nx, ny = (np.sqrt(0.5), -np.sqrt(0.5)) if dg else (1.0, 0.0)
                d = a[:, 0] * nx + a[:, 1] * ny
                b = a - 2.0 * d[:, None] * np.array([nx, ny])
                pair = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1)
                assert float(pair.max()) < 1e-9, ("sym not mirror-invariant", nn, dg,
                                                  float(pair.max()))
    # and the randomized tail of the stream must only ever emit startable entries
    for nn in (5, 14, 31):
        st = _start_stream(nn, np.random.RandomState(1))
        for _ in range(400):
            kind, arg = next(st)
            if kind == "lat":
                assert _hexlat(nn, arg, g).shape == (nn, 2), (kind, arg, nn)
            elif kind == "hex":
                assert _rowed(nn, arg, g).shape == (nn, 2), (kind, arg, nn)
            elif kind == "sym":
                assert _sym(nn, g, theta=arg[0], off=arg[1], diag=arg[2],
                            sfac=arg[3], jitter=arg[4]).shape == (nn, 2), (kind, nn)

    # _xover is fed two local optima and its output goes straight into an SLP, so the invariant that
    # matters is the same one _sym needed: EXACTLY n circles (evaluate() raises otherwise and the driver
    # ends the whole call).  Exercise it on structurally different parents, on IDENTICAL parents (the
    # degenerate case the elite pool can still produce), and on a parent pair that overlaps heavily.
    gx = np.random.RandomState(11)
    for nn in (4, 5, 14, 27, 31, 50, 99):
        pa = _hexrot(nn, gx, theta=0.0, off=(0.0, 0.0))
        pb = _hexrot(nn, gx, theta=0.41, off=(0.013, 0.007))
        A = (pa, _init_radii(pa))
        B = (pb, _init_radii(pb))
        seen = 0
        for _ in range(40):
            for P, Q in ((A, B), (B, A), (A, A)):
                cx = _xover(P, Q, gx)
                if cx is None:
                    continue
                seen += 1
                assert cx[0].shape == (nn, 2) and cx[1].shape == (nn,), ("xover", nn, cx[0].shape)
                assert np.all(np.isfinite(cx[0])) and np.all(cx[1] > 0), ("xover finite", nn)
                fx, fr = _polish(cx[0], cx[1], gx, passes=1)
                D = _dist(fx)
                assert float(np.min(D - (fr[:, None] + fr[None, :]))) >= -1e-9, ("xover infeasible", nn)
                assert float(np.min(_wall(fx) - fr)) >= -1e-9, ("xover wall", nn)
        assert seen > 0, ("xover produced nothing at n=%d" % nn)
    # a size mismatch between the two parents must fail soft, never raise or return a wrong count
    assert _xover((_hexrot(9, gx), _init_radii(_hexrot(9, gx))),
                  (_hexrot(10, gx), _init_radii(_hexrot(10, gx))), gx) is None

    # CPU HEADROOM.  A deadline is only tested at the top of a loop, so a call always overruns its
    # allowance by the cost of the move already in flight.  That overshoot is harmless with k >= 2
    # targets (every per-n deadline is min(end_all, ...), so the later slices absorb it), but with
    # len(targets) == 1 -- the offline graded case, capped at 60 s -- nothing absorbs it.  Assert the
    # static headroom, then MEASURE the real overshoot on the most expensive shape available.
    assert CPU_PER_TARGET + 8.0 <= 60.0, "CPU_PER_TARGET leaves no room for the deadline overshoot"
    assert min(CPU_TOTAL, CPU_PER_TARGET * 37) + 8.0 <= 120.0, "CPU_TOTAL leaves no backstop room"

    m = Meter(500000)
    ev = Ev(m)
    rng = np.random.RandomState(0)
    t_ov = time.process_time()
    solve(ev, m, rng, [81], cpu_allowance=6.0)
    ov = time.process_time() - t_ov - 6.0
    assert 81 in ev.best, "single-target solve produced no feasible packing"
    print("single-target CPU overshoot at n=81 = %+.2f s (headroom is %.1f s)"
          % (ov, 60.0 - CPU_PER_TARGET))
    assert ov <= 60.0 - CPU_PER_TARGET, "overshoot exceeds the offline 60 s headroom"

    solve(ev, m, rng, [27], cpu_allowance=6.0)
    assert 27 in ev.best, "first call produced no feasible packing"
    first = ev.best[27]

    # SECOND call in the SAME process -- must still work (deadline recomputed at entry), and with a
    # single, different n (per-n split must survive len(targets) == 1).
    solve(ev, m, rng, [31], cpu_allowance=6.0)
    assert 31 in ev.best, "SECOND solve() call in the same process produced no feasible packing"

    # a never-scored n with no committed pack: the warm-start read must be guarded, not raise
    solve(ev, m, rng, [14], cpu_allowance=4.0)
    assert 14 in ev.best, "cold start failed for an n outside the census"

    # CONCENTRATION.  MIN_CPU_PER_N is the measured knee of the depth curve, so when the even share
    # would fall under it the allowance must go to a PREFIX of the active sizes at full depth and the
    # tail must be left untouched (it keeps its committed pack, which costs zero).  Three never-scored
    # sizes with 20 s cannot support even two slices at the knee, so exactly ONE of them may be worked.
    # This is the assertion that fails if MIN_CPU_PER_N is ever dropped back below total/len(active):
    # at the old 9.0 this same call spread 9 + 9 + 2 s over all three.
    m2 = Meter(500000)
    ev2 = Ev(m2)
    solve(ev2, m2, np.random.RandomState(5), [12, 16, 18], cpu_allowance=20.0)
    assert len(ev2.best) == 1, ("allowance was spread instead of concentrated", sorted(ev2.best))
    assert 20.0 < 2 * MIN_CPU_PER_N, "the concentration test no longer discriminates"

    # DEEP ORDER: the allocation is the ordering (only total//MIN_CPU_PER_N sizes get a slice), so the
    # SMALLEST remaining gap must come first among the sizes that have a pack, and a size with NO
    # committed pack must come before every one of them.  A regression to the uniform hash rotation
    # fails both; a regression to iteration 14's largest-gap-first fails the first.
    _recs = _records()
    _act = [n for n in sorted(_recs) if not _is_capped(n, _recs)]
    if len(_act) > 2:
        _o = _deep_order(_act, _recs)
        def _gap(n):
            _w = _read_pack(n)
            return max(0.0, (_recs[n] - float(_w[1].sum())) / _recs[n])
        _g = [_gap(n) for n in _o]
        assert all(_g[i] <= _g[i + 1] + 1e-18 for i in range(1, len(_g) - 1)), ("deep order not gap-sorted: %r" % (_o,))
        assert _deep_order([_act[0], 14, _act[1]], _recs)[0] == 14, "a never-scored size must sort first"
        print("  deep-order OK  (first=%d gap=%.2e, last=%d gap=%.2e)" % (_o[0], _g[0], _o[-1], _g[-1]))
    print("concentration: 20 s over 3 active sizes worked %d size(s) at depth" % len(ev2.best))

    # every requested size already capped -> must fall back to searching them all, not return empty
    if rc:
        cp = [n for n in sorted(rc) if _is_capped(n, rc)][:1]
        if cp:
            solve(ev, m, rng, cp, cpu_allowance=4.0)
            assert cp[0] in ev.best, "all-capped fallback produced no feasible packing"

    # _relax / _pen_obj invariants.  The gradient is the thing that MUST be right: L-BFGS-B does not
    # report a nan gradient, it just stops and hands back the point it started from -- a silently
    # no-op escape move.  So check it against central differences, including on configurations with
    # COINCIDENT centres (0/0) which the swap+inflate path can really produce.
    trng = np.random.RandomState(20260915)
    for nn in (2, 3, 7, 20):
        for _ in range(3):
            cxy = trng.uniform(LO, HI, size=(nn, 2))
            cr = trng.uniform(0.04, 0.26, size=nn)          # deliberately INFEASIBLE
            v = np.concatenate([cxy[:, 0], cxy[:, 1], cr])
            f0, g = _pen_obj(v, nn, 137.0)
            num = np.empty_like(v)
            for k in range(v.size):
                e = np.zeros_like(v)
                e[k] = 1e-7
                num[k] = (_pen_obj(v + e, nn, 137.0)[0] - _pen_obj(v - e, nn, 137.0)[0]) / 2e-7
            rel = float(np.max(np.abs(num - g))) / max(1.0, float(np.max(np.abs(g))))
            assert rel < 2e-5, "_pen_obj gradient disagrees with finite differences (%.2e)" % rel
    f0, g0 = _pen_obj(np.concatenate([np.zeros(8), np.full(4, 0.2)]), 4, 1e3)
    assert np.isfinite(f0) and np.all(np.isfinite(g0)), "_pen_obj is nan on coincident centres"
    for nn in (2, 4, 14, 27, 31, 49, 99):
        bxy = trng.uniform(LO + 0.05, HI - 0.05, size=(nn, 2))
        bxy, br = _polish(bxy, _init_radii(bxy), trng, passes=2)
        for sw in (0, 1, 3, 6):
            out = _relax(bxy, br, trng, lam=1.0 + 0.25 * trng.rand(), swaps=sw,
                         mu0=10.0 ** (1.8 + 1.2 * trng.rand()))
            assert out is not None, "_relax returned None for n=%d" % nn
            axy, ar = out
            assert axy.shape == (nn, 2) and ar.shape == (nn,), "_relax changed the circle count"
            assert np.all(np.isfinite(axy)) and np.all(ar > 0.0), "_relax produced a non-finite circle"
            # the melt is deliberately infeasible; one projection+polish must make it STRICTLY feasible
            axy, ar = _polish(axy, ar, trng, passes=2)
            assert float(np.min(_wall(axy) - ar)) >= -1e-9, "_relax output not wall-feasible after polish"
            if nn > 1:
                dm = _dist(axy) - (ar[:, None] + ar[None, :])
                assert float(np.min(dm)) >= -1e-9, "_relax output not pair-feasible after polish"

    assert ev.worst_slack >= -1e-9, "registered a packing violating the feasibility tolerance"
    assert m.used > 0
    # --- _census_hash / _seed_for: the seed must be REPRODUCIBLE, must survive a missing census, and
    # must change when ANY pack in the census changes (the whole point of the global term).
    for _n in (14, 27, 49, 99):
        a = _seed_for(_n, np.random.RandomState(3)).randint(0, 10 ** 9)
        b = _seed_for(_n, np.random.RandomState(3)).randint(0, 10 ** 9)
        assert a == b, "seed_for not reproducible at n=%d" % _n
    _h0 = _census_hash()
    del _CH[:]
    _save = dict(_PACK_CACHE)
    _PACK_CACHE.clear()
    assert _census_hash() == _h0, "census hash not stable across a cache flush"
    # perturb ONE pack's contents in the cache and check every size's seed moves
    _seeds0 = [_seed_for(k, np.random.RandomState(5)).randint(0, 10 ** 9) for k in (27, 49, 99)]
    _victim = None
    for _k, _v in list(_PACK_CACHE.items()):
        if _v is not None and _k[0].endswith("csqv27.pck"):
            _victim = (_k, _v)
            _PACK_CACHE[_k] = (_v[0], _v[1] + 1e-6)
            break
    del _CH[:]
    if _victim is not None:
        _seeds1 = [_seed_for(k, np.random.RandomState(5)).randint(0, 10 ** 9) for k in (27, 49, 99)]
        assert all(x != y for x, y in zip(_seeds0, _seeds1)), \
            "a changed pack did not decorrelate every size: %r vs %r" % (_seeds0, _seeds1)
        _PACK_CACHE[_victim[0]] = _victim[1]
    _PACK_CACHE.clear()
    _PACK_CACHE.update(_save)
    del _CH[:]
    print("  census-hash/seed invariants OK  (hash=%d, decorrelated=%s)"
          % (_h0, _victim is not None))
    # _rand_t0: the new ASCENT axis.  It must be (a) inside its declared band for any n, (b) genuinely
    # SCALE-FREE -- the same rng state must give the same t0/sp at every n, which is what makes it
    # transfer to never-scored offline sizes -- and (c) actually spread, since a draw that collapses to
    # one value would silently restore the fixed-t0 behaviour this change exists to remove.
    for _n in (2, 7, 27, 37, 49, 81, 99, 250):
        _sp = float(np.sqrt(2.0 / (np.sqrt(3.0) * _n)))
        _v = np.array([_rand_t0(_n, np.random.RandomState(_k)) for _k in range(64)]) / _sp
        assert _v.min() >= T0_LO - 1e-12 and _v.max() <= T0_HI + 1e-12, (_n, _v.min(), _v.max())
        _w = np.array([_rand_t0(37, np.random.RandomState(_k)) for _k in range(64)])
        _w = _w / float(np.sqrt(2.0 / (np.sqrt(3.0) * 37)))
        assert np.allclose(_v, _w, rtol=1e-12), ("t0 is not scale-free at n=%d" % _n)
    _v = np.array([_rand_t0(49, _r0) for _r0 in [np.random.RandomState(11)] * 1 for _ in range(400)])
    assert _v.max() / _v.min() > 4.0, ("t0 draw collapsed: %r" % (_v.max() / _v.min()))
    assert _rand_t0(49, np.random.RandomState(3)) == _rand_t0(49, np.random.RandomState(3))
    print("  rand-t0 invariants OK  (band=[%.2f, %.2f]*sp, spread=%.1fx)"
          % (T0_LO, T0_HI, _v.max() / _v.min()))
    print("self-test OK  evals=%d  worst_slack=%.2e" % (m.used, ev.worst_slack))
    for n, s in sorted(ev.best.items()):
        if n in recs:
            gap = max(0.0, (recs[n] - s) / recs[n])
            print("  n=%2d sum_r=%.9f  record=%.9f  relgap=%.3e  digits=%.2f"
                  % (n, s, recs[n], gap, min(7.0, -np.log10(gap)) if gap > 0 else 7.0))
        else:
            print("  n=%2d sum_r=%.9f  (no record)" % (n, s))


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
