"""SLP circle-packing solver for the packomania csqv census (max Sigma r, n circles in a square).

WHAT THIS REPLACED.  Iteration 1 stopped guessing the RADII: for fixed centres the optimal radii are a
linear program, and it solved that LP exactly.  But it still guessed how to move the CENTRES -- gradient
ascent on the LP's duals with an adaptively halved trust step.  Those duals are the subgradient of a
piecewise-linear max, so the ascent creeps along kinks and stalls near a 5e-2 relative gap.  The exact
radius solver was bolted to a heuristic centre solver, and the heuristic half was the binding one.

THE IDEA.  Solve for the centres AND the radii together, exactly, in one LP per step -- by exploiting the
fact that the hard constraint is a CONVEX function on the wrong side:

        ||c_i - c_j||  >=  r_i + r_j .

Because the Euclidean norm is convex, its linearisation about the current centres is a GLOBAL
under-estimator:  ||c_i + dx_i - c_j - dx_j|| >= d_ij + u_ij . (dx_i - dx_j)  everywhere, with
u_ij = (c_i - c_j)/d_ij.  So replacing the norm by that linearisation turns the non-convex packing
problem into an LP that is a **convex RESTRICTION** of it, not a relaxation:

        max sum_i r_i   s.t.  r_i + r_j - u_ij.(dx_i - dx_j) <= d_ij,   r_i +- (dx or dy)_i <= wall_i,
                              |dx_i|,|dy_i| <= D,   0 <= r_i <= r_i^cur + RHO .

Three consequences, and they are the whole design:
  * **Every LP solution is exactly feasible for the true problem** -- the linearisation only ever
    under-states the true separation.  There is no line search, no penalty, no repair loop, no risk of
    the iterate blowing up (which is exactly what SLSQP on the same NLP does at n >= 61).
  * dx = dy = 0, r = r^cur is always LP-feasible, so the step is **monotone**: Sigma r never decreases.
  * The trust radii (D, RHO) make the pair set an **exact reduction**: a pair whose current slack exceeds
    2D + 2RHO cannot possibly collide after the step, so dropping it is a proof, not a truncation.  Near
    convergence D -> 0 and the LP shrinks to the contact graph itself (n=99: ~4900 pairs -> ~220 rows).

Each step is therefore an exactly-solved subproblem whose solution needs no validation, and the sequence
converges to a KKT point of the full NLP.  Around it sits basin hopping over a LADDER of perturbation
scales -- hedging across kick sizes instead of committing to one -- because a converged SLP is finished
and any CPU left in that n's slice is worth more spent in a neighbouring basin.

THE CENSUS IS ONE POPULATION, NOT 37 PROBLEMS.  The optimal packing for n and for n +- 2 share almost all
of their combinatorics, so a basin discovered at one size is evidence about every nearby size.  A call is
therefore three phases: DIFFUSE (every n imports its neighbours' structures, resized by an exact
subproblem -- drop the smallest circles, or insert at the exact largest empty pocket), then per-n local
search, then DIFFUSE again so what this call found propagates.  Sweeps alternate direction over a LIVE
pool, so one improvement can travel the length of the census inside a single call.

Pipeline per n:  (1) a cheap fully-batched numpy pre-relaxation over many random restarts, scored with a
maximal-growth radius rule, giving an immediate feasible pack and well-spread starting centres; (2) SLP to
convergence from the live census entry (warm start, guarded) and from the best fresh start; (3)
perturb-and-repolish basin hopping over the kick ladder for the rest of the slice; (4) a final exact LP
over radii at the winning centres plus a strict-feasibility repair with a tiny safety margin.

Budgeting: every deadline is derived from time.process_time() AT ENTRY (no module-level absolute stop), so
solve() may be called repeatedly in one process; the per-n slice is a rolling n-weighted split of the time
that is LEFT, so it is correct for len(targets) == 1 as well as 37.  meter.left() is polled before every
evaluate() so an exhausted evaluation budget is never mistaken for infeasibility.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

LO, HI = -0.5, 0.5
CPU_BUDGET = 104.0          # seconds of process CPU this call may use (driver backstop is 120 s)
EPS_IN = 1e-9               # keep centres strictly inside the square
SAFETY = 3e-11              # uniform radius shave after the exact repair (relgap cost ~ 1e-9)
LAT_FRAC = 0.50             # share of a COLD n's CPU slice spent enumerating the structured lattice family
LAT_FRAC_WARM = 0.0         # ... but the family is dominated by any committed pack (probe_alt), so a warm
                            #     n is not charged for it at all: a dominated stage is a budget, not a fix
WARM_SLP_FRAC = 0.12        # a committed pack is already SLP-RIGID (probe_conv), so re-converging it is a
                            #     formality, and the random-cloud restart it used to race is dominated too
SLOT_C = 0.0016             # one VISIT costs SLOT_C*n^2 CPU-s: the MEASURED ladder-completion time, see _slot
SLOT_LO, SLOT_HI = 2.6, 16.0
DRY_HOPS = 40               # ... and once the ladder is spent, this many dry hops end the visit early.
                            # MEASURED, not guessed (artifacts/probe_op.py + probe_dry.py): 6 was
                            # calibrated when a ladder pass was 18 rungs, so "6 dry hops" meant "a
                            # whole pass plus some".  Screening (iter 12) cut the plan to 3 rungs and
                            # the SAME constant silently became "quit after 3 rungs + 3 stochastic
                            # hops": probe_op measured EVERY visit at the five neediest sizes ending
                            # at hop 6 having spent 15-20% of its slot, and _rounds then cycling back
                            # to re-run the byte-identical deterministic ladder from the unchanged
                            # pack.  Spending the slot instead finds what the exit was cutting off:
                            # n=37 1.611e-3 -> 8.689e-4 (a kick at hop>6), n=53 9.207e-4 -> 2.370e-4
                            # (a ruin at hop>6); the early-exiting visit found ZERO at all five.
SLP_FLOOR = 1e-6            # trust-region floor at which the SLP stops -- see _slp / probe_tail
SLP_FLOOR_FINAL = 1e-12     # ... except for the once-per-visit polish of the pack actually committed
SCREEN_STEPS = 4            # LP solves that RANK a ladder rung (probe_screen); it takes ~50 to answer it
SCREEN_KEEP = 3             # how many screened rungs are then converged: the true best is inside 3
CAP_RELGAP = 1e-7           # a size at or under this already scores the full 7 digits: it cannot raise SCORE
KICKS = (0.30, 0.45, 0.18, 0.65, 0.10)   # continuous-kick ladder, in units of the mean radius
RUIN = (0.06, 0.14, 0.03, 0.22, 0.09)    # ruin-and-recreate ladder: fraction of the n circles ejected
TRANS_PRE = 0.42            # share of the call spent DIFFUSING structure across n before the per-n loop
TRANS_POST = 0.28           # ... and again after it, so what this call discovered propagates too
DONORS = (-4, -2, 2, 4)     # which neighbouring sizes may donate a structure to n
FAR_DONORS = (-8, -6, 6, 8)  # long-range donors, tried only once the near diffusion has gone dry
SLICE_MIN = 2.5             # CPU-s floor on ONE transfer attempt -- MEASURED: below ~1 CPU-s inside the
SLICE_MAX = 4.0             # morph the resize has not re-formed the contact graph.  Ceiling: no size
MORPH_FRAC = 0.35           # eats a phase.  A visit spends this share of its slice on the resize itself.


# ------------------------------------------------------------------ cheap batched radius assignment
def _walls(xy):
    x, y = xy[..., 0], xy[..., 1]
    return np.minimum.reduce([x - LO, HI - x, y - LO, HI - y])


def _grow_batch(xy, rounds=24):
    """Feasible radii for a BATCH of centre sets xy:(B,n,2) -> packs (B,n,3), fully vectorized.

    Start from the equal-split radius (always feasible) and then repeatedly hand each circle HALF of its
    remaining slack.  Halving keeps the step feasible even when both circles of a pair grow at once, and
    the iteration converges geometrically to a MAXIMAL packing -- strictly better than the equal split,
    and far cheaper than an LP, which is what makes it usable inside a batched restart loop."""
    B, n, _ = xy.shape
    wall = _walls(xy)
    if n < 2:
        return np.concatenate([xy, np.maximum(wall - 2e-9, 1e-7)[..., None]], axis=-1)
    d = xy[:, :, None, :] - xy[:, None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    di = np.arange(n)
    dist[:, di, di] = np.inf
    r = np.maximum(np.minimum(wall, dist.min(axis=2) / 2.0), 0.0)
    for _ in range(rounds):
        slack = dist - (r[:, :, None] + r[:, None, :])
        slack[:, di, di] = np.inf
        room = np.minimum(wall - r, slack.min(axis=2))
        r = r + 0.5 * np.maximum(room, 0.0)
    r = np.maximum(r - 2e-9, 1e-7)
    return np.concatenate([xy, r[..., None]], axis=-1)


def _prerelax(rng, n, B, steps):
    """B random restarts pushed apart by a repulsion + wall-push flow; returns the centre batch."""
    xy = rng.uniform(LO + 0.08, HI - 0.08, size=(B, n, 2))
    if n < 2:
        return xy
    di = np.arange(n)
    lr = 0.06
    for _ in range(steps):
        d = xy[:, :, None, :] - xy[:, None, :, :]
        dist2 = (d ** 2).sum(-1) + 1e-9
        dist2[:, di, di] = np.inf
        f = (d / (dist2[..., None] ** 1.5)).sum(axis=2) * 0.01
        x, y = xy[..., 0], xy[..., 1]
        for wall, gx, gy in ((x - LO, 1.0, 0.0), (HI - x, -1.0, 0.0),
                             (y - LO, 0.0, 1.0), (HI - y, 0.0, -1.0)):
            push = 0.002 / (np.maximum(wall, 1e-3) ** 2)
            f[..., 0] += push * gx
            f[..., 1] += push * gy
        nrm = np.sqrt((f ** 2).sum(-1))[..., None] + 1e-12
        xy = np.clip(xy + lr * f / nrm, LO + 1e-4, HI - 1e-4)
        lr *= 0.985
    return xy


# ------------------------------------------------------- the exact LP over radii at FIXED centres
def _lp_radii(xy):
    """Optimal radii for FIXED centres.  Returns (value, r) or None if the LP fails.

    Only pairs with d_ij <= wall_i + wall_j can ever bind (r_i + r_j <= wall_i + wall_j always), so
    dropping the rest is an EXACT reduction.  Used once at the end: SLP's radii are optimal only
    *locally* in the last trust box, and this can still find a better vertex at the same centres."""
    n = xy.shape[0]
    w = np.maximum(_walls(xy), 0.0)
    I, J = np.triu_indices(n, 1)
    dv = np.sqrt(((xy[I] - xy[J]) ** 2).sum(-1))
    keep = dv <= (w[I] + w[J])
    I, J, dv = I[keep], J[keep], dv[keep]
    m = I.size
    if m:
        rows = np.repeat(np.arange(m), 2)
        cols = np.empty(2 * m, dtype=np.int64)
        cols[0::2], cols[1::2] = I, J
        A = coo_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n)).tocsr()
    else:
        A, dv = None, None
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=dv,
                      bounds=np.stack([np.zeros(n), w], axis=1), method="highs",
                      options={"primal_feasibility_tolerance": 1e-10,
                               "dual_feasibility_tolerance": 1e-10})
    except Exception:                                   # noqa: BLE001
        return None
    if not res.success or res.x is None:
        return None
    r = np.maximum(np.asarray(res.x, dtype=float), 0.0)
    return float(r.sum()), r


# ------------------------------------------- one SLP step: the convex restriction over (dx, dy, r)
def _slp_step(x, y, r, D, RHO):
    """Solve the trust-region LP restriction; returns (x', y', r') or None.

    Exactness of the reduction: a pair whose slack exceeds 2D+2RHO keeps slack > 0 after any step in the
    box (each centre moves <= D, each radius grows <= RHO), and a wall whose slack exceeds D+RHO likewise;
    those rows are therefore PROVABLY redundant and are left out.  r_i >= 0 together with the retained
    wall rows also keeps every centre inside the square, so no explicit position bound is needed."""
    n = x.size
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    iu = np.triu_indices(n, 1)
    sel = (d[iu] - (r[iu[0]] + r[iu[1]])) < (2.0 * D + 2.0 * RHO)
    I, J = iu[0][sel], iu[1][sel]
    dv = d[I, J]
    inv = 1.0 / np.maximum(dv, 1e-15)
    ux, uy = (x[I] - x[J]) * inv, (y[I] - y[J]) * inv
    mp = I.size
    rows, cols, vals, rhs = [], [], [], []
    if mp:
        rk = np.arange(mp)
        rows.append(np.tile(rk, 6))
        cols.append(np.concatenate([2 * n + I, 2 * n + J, I, J, n + I, n + J]))
        vals.append(np.concatenate([np.ones(mp), np.ones(mp), -ux, ux, -uy, uy]))
        rhs.append(dv)
    wsl = np.stack([x - r - LO, HI - x - r, y - r - LO, HI - y - r], axis=1)
    WI, WW = np.nonzero(wsl < (D + RHO))
    mw = WI.size
    if mw:
        rk = mp + np.arange(mw)
        sgn = np.where((WW == 0) | (WW == 2), -1.0, 1.0)
        var = np.where(WW < 2, WI, n + WI)
        rows.append(np.tile(rk, 2))
        cols.append(np.concatenate([2 * n + WI, var]))
        vals.append(np.concatenate([np.ones(mw), sgn]))
        rhs.append(np.where(WW == 0, x[WI] - LO,
                   np.where(WW == 1, HI - x[WI],
                   np.where(WW == 2, y[WI] - LO, HI - y[WI]))))
    m = mp + mw
    if m:
        A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(m, 3 * n)).tocsr()
        b = np.concatenate(rhs)
    else:
        A, b = None, None
    lb = np.concatenate([np.full(2 * n, -D), np.zeros(n)])
    ub = np.concatenate([np.full(2 * n, D), np.minimum(0.5, r + RHO)])
    try:
        res = linprog(np.concatenate([np.zeros(2 * n), -np.ones(n)]), A_ub=A, b_ub=b,
                      bounds=np.stack([lb, ub], axis=1), method="highs",
                      options={"primal_feasibility_tolerance": 1e-10,
                               "dual_feasibility_tolerance": 1e-10})
    except Exception:                                   # noqa: BLE001
        return None
    if not res.success or res.x is None or not np.isfinite(res.x).all():
        return None
    z = np.asarray(res.x, dtype=float)
    return x + z[:n], y + z[n:2 * n], np.maximum(z[2 * n:], 0.0)


def _slp(x, y, r, deadline, d0=0.30, steps=None, floor=SLP_FLOOR):
    """SLP to convergence from (x, y, r).  Returns (x, y, r, sum_r) of the best iterate.

    The trust radius grows on an accepted step and halves on a stalled one; since every step is feasible
    and monotone, 'best' is simply the last accepted iterate -- there is nothing to reject.  `steps` caps
    the number of LP solves: a truncated run is the SCREENING oracle for the lattice enumeration, and
    because every iterate is feasible and monotone a truncated run is still a valid packing, just a
    less-converged one -- so screening never has to reject anything either."""
    rm = max(float(r.mean()), 1e-12)
    D = d0 * rm
    cap = 0.6 * rm
    best = float(r.sum())
    it = 0
    while D > floor and time.process_time() < deadline:
        if steps is not None and it >= steps:
            break
        it += 1
        out = _slp_step(x, y, r, D, D)
        if out is None:
            break
        nx, ny, nr = out
        v = float(nr.sum())
        if v > best + 1e-14:
            x, y, r, best = nx, ny, nr, v
            D = min(D * 1.7, cap)
        else:
            D *= 0.5
    return x, y, r, best


# ------------------------------------------------- the exact subproblem behind the basin move:
#                                                    WHERE does an ejected circle want to go?
def _clearance(px, py, x, y, r):
    """f(p) = min( wall(p), min_i(||p-c_i|| - r_i) ) for a batch of query points p:(M,).

    f is the radius of the largest circle that may be centred at p without overlapping anything, and
    it is 1-Lipschitz in p -- which is what makes a grid search over it a BOUND, not a guess."""
    out = np.minimum.reduce([px - LO, HI - px, py - LO, HI - py])
    if x.size:
        step = max(1, 2000000 // max(x.size, 1))
        for s in range(0, px.size, step):
            dx = px[s:s + step, None] - x[None, :]
            dy = py[s:s + step, None] - y[None, :]
            np.minimum(out[s:s + step],
                       (np.sqrt(dx * dx + dy * dy) - r[None, :]).min(axis=1),
                       out=out[s:s + step])
    return out


def _largest_empty(x, y, r, G=96):
    """argmax of _clearance: the largest empty pocket in the packing.  Returns (px, py, rho).

    Because f is 1-Lipschitz, a grid of spacing h certifies max f - max_grid f <= h/sqrt(2); three
    nested refinements (each 4x finer over the winning cell) drive the residual to ~1e-5 of a radius,
    and the SLP that follows the insertion is the exact refiner anyway.  The returned rho is f itself,
    so a circle placed there is feasible BY CONSTRUCTION -- nothing to repair, nothing to reject."""
    g = (np.arange(G) + 0.5) / G + LO
    PX, PY = np.meshgrid(g, g, indexing="ij")
    PX, PY = PX.ravel(), PY.ravel()
    f = _clearance(PX, PY, x, y, r)
    k = int(np.argmax(f))
    cx, cy = float(PX[k]), float(PY[k])
    h = 1.0 / G
    for _ in range(3):
        off = np.linspace(-h, h, 9)
        QX, QY = np.meshgrid(np.clip(cx + off, LO, HI), np.clip(cy + off, LO, HI), indexing="ij")
        QX, QY = QX.ravel(), QY.ravel()
        f2 = _clearance(QX, QY, x, y, r)
        k2 = int(np.argmax(f2))
        cx, cy = float(QX[k2]), float(QY[k2])
        h *= 0.25
    rho = float(_clearance(np.array([cx]), np.array([cy]), x, y, r)[0])
    return cx, cy, max(rho, 0.0)


# ------------------------------------------- the exact subproblem behind the SYMMETRY move:
#                                             WHICH circle is which circle's mirror image?
# The full symmetry group of the square is D4 (order 8).  Iteration 8 shipped only its four
# REFLECTIONS; `artifacts/probe_group.py` measured the two rotations it was missing and they are not
# redundant -- at n=37 the 90-degree rotation reaches 1.61e-3 where the best reflection stalls at
# 3.10e-3, and at n=63 3.02e-4 against 5.08e-4.  A rotation is not an involution, so the projection
# has to average over the whole ORBIT of the cyclic group it generates, not over a pair.
GROUP = (
    ("mirx",   ((-1.0, 0.0), (0.0, 1.0)),  2),
    ("miry",   ((1.0, 0.0), (0.0, -1.0)),  2),
    ("diag",   ((0.0, 1.0), (1.0, 0.0)),   2),
    ("adia",   ((0.0, -1.0), (-1.0, 0.0)), 2),
    ("rot180", ((-1.0, 0.0), (0.0, -1.0)), 2),
    ("rot90",  ((0.0, -1.0), (1.0, 0.0)),  4),
)
# How much of the packing to move onto the fixed subspace.  1.0 is the full projection iteration 8
# shipped.  It is often TOO VIOLENT: a csqv optimum is frequently symmetric only in part (a symmetric
# shell around an asymmetric core, or a few rattlers), so forcing all n circles onto the fixed
# subspace destroys the part that was already right, the repair shrinks everything, and the size does
# not move at all.  `artifacts/probe_partial.py`: at frac 0.8 the sizes that were DEAD under every
# full projection come loose -- n=35 3.775e-3 -> 2.246e-3, n=81 2.012e-3 -> 1.357e-3, and n=63
# 1.561e-3 -> 0 (past the record).  The quantity is discrete and low-dimensional, so it is enumerated.
FRACS = (0.80, 1.00, 0.55)
# ... and behind that discrete quantity sat one more, exactly where the last one did.  `frac` says HOW
# MUCH of the packing to hand to the group; it never said WHICH part, because the answer looked
# obvious: move the circles that are already nearly symmetric and leave the asymmetric core alone.
# That is the LOW-break direction, and `artifacts/probe_sel.py` measured it dry -- from the committed
# census the entire 18-rung ladder returns the pack it was given at all five stuck sizes tested.  The
# COMPLEMENT -- leave the well-matched shell exactly where it is and force the WORST-matched circles
# onto the fixed subspace -- breaks the part of the state that is currently wrong instead of the part
# that is already right, and it is not dry: n=51 1.800e-3 -> 1.286e-3, n=69 7.000e-4 -> 4.076e-4,
# n=81 1.5185e-3 -> 1.5159e-3, in LESS CPU than the shell rungs cost.  Both are partial projections
# and only one was in the ladder; a new rung is additive, so both are, and the max is kept.
SELS = (0, 1)               # 0 = project the best-matched frac*n (shell);  1 = the worst-matched (core)
# One ladder entry = one (group element, fraction, selection) triple, tried at FULL strength in
# sequence: the deadline truncates the ladder rather than the slice being split across it.  At
# frac = 1.0 every circle moves, so the two selections are the same rung and only one is listed.
LADDER = tuple((gi, f, sel) for f in FRACS for sel in (SELS if f < 1.0 else (0,))
               for gi in range(len(GROUP)))


def _project(x, y, r, gi, frac, sel=0):
    """Project a packing onto (part of) the fixed subspace of one cyclic subgroup of the square's D4.

    Every committed pack is a RIGID local optimum -- `artifacts/probe_conv.py`: 0 circles with fewer
    than three contacts, and a 3 CPU-s SLP moves Sigma r by <2e-9 at every size.  Jitter and ruin both
    fall back into the same basin, and the donor channel is saturated (`artifacts/probe_far.py`).  What
    is left is to move the COMBINATORICS, and the square's own symmetry group is a structured direction
    to move it in.

    The projection contains a guess that is really a solve.  To average a circle with its image under
    the transform you must first know WHICH circle is its image -- an assignment problem over the n
    images, which `linear_sum_assignment` answers exactly in one call rather than by nearest-neighbour
    greedy (greedy is not a permutation, so it collapses circles onto each other and the pack dies).
    The matching gives a permutation pi with T(p_i) ~ p_{pi[i]}; averaging p over the orbit
    {T^-j p_{pi^j(i)}} lands exactly on the nearest <T>-invariant configuration.  `frac` then keeps
    only the best-matched share of the circles on that subspace and leaves the rest where they are,
    so an almost-symmetric packing is symmetrised without its asymmetric core being flattened.
    """
    n = x.size
    if n < 2:
        return None
    T = np.array(GROUP[gi][1], dtype=float)
    m = GROUP[gi][2]
    P = np.stack([x, y], axis=1)
    Q = P @ T.T
    cost = ((Q[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    try:
        ii, jj = linear_sum_assignment(cost)
    except Exception:                                   # noqa: BLE001
        return None
    pi = np.empty(n, dtype=int)
    pi[ii] = jj
    Ti = T.T                                            # every element of D4 is orthogonal
    acc = np.zeros((n, 2))
    accr = np.zeros(n)
    idx = np.arange(n)
    M = np.eye(2)
    for _ in range(m):
        acc += P[idx] @ M.T
        accr += r[idx]
        idx = pi[idx]
        M = M @ Ti
    acc /= m
    accr /= m
    if frac < 1.0:
        move = np.linalg.norm(acc - P, axis=1)          # how far this circle must travel to be symmetric
        order = np.argsort(move)
        k = int(round(frac * n))
        # WHICH part of the state goes to the group.  sel=0 keeps the well-matched shell on the fixed
        # subspace and leaves the asymmetric core untouched (the gentle break).  sel=1 is its exact
        # complement: the shell stays exactly where it already is and the core -- the part that is
        # currently NOT symmetric, and so the part most likely to be the wrong combinatorics -- is the
        # part forced onto the subspace.  See SELS: from a ladder-exhausted pack only sel=1 still pays.
        stay = order[k:] if sel == 0 else order[:n - k]
        acc[stay] = P[stay]
        accr[stay] = r[stay]
    acc = np.clip(acc, LO + EPS_IN, HI - EPS_IN)
    # the average may overlap; shrink to strict feasibility so the SLP starts from a real packing.
    pack = _repair(acc, accr)
    return pack[:, 0], pack[:, 1], np.maximum(pack[:, 2], 1e-12)


def _ladder_plan(best, rng, deadline, keep=SCREEN_KEEP):
    """SCREEN the whole enumerated ladder, then converge only the survivors.

    The ladder asks two different questions of the SLP and they have two different floors.  "Which of
    the eighteen (D4 element x fraction) rungs is worth taking?" is a RANKING question; "how good is
    this rung exactly?" is an ANSWERING one.  Iterations 8-11 paid the answering price for all
    eighteen -- ~50 LP solves each -- because the ranking floor had never been measured.
    `artifacts/probe_screen.py` measures it: at SCREEN_STEPS = 4 LP solves the truncated value puts the
    TRULY best rung inside its own top three at n = 37, 51 and 81, with a loss of 0, 0 and 1.7e-10
    against converging all eighteen -- three to four hundred times below the 1e-7 relgap the scorer can
    see.  Screening plus three conversions costs ~0.5 CPU-s at n=51 where the full ladder costs 3.05.

    Note the truncated iterate is NOT thrown away: it is a feasible, monotone, partly-converged packing,
    so the survivors resume from it and the four screening solves are spent, not wasted.

    This is not a split of one attempt N ways -- the thing my hedging rule forbids.  The rule's real
    content is that an operator has a floor BELOW WHICH IT RETURNS NOISE, and the floor is a property of
    the QUESTION: ranking is cheap, answering is not.  Measure the two separately and the branches that
    would have starved never have to be attempted at full strength at all.
    """
    x0, y0, r0 = best[:, 0], best[:, 1], best[:, 2]
    cands = []
    for gi, fr, sel in LADDER:
        if time.process_time() >= deadline:
            break
        p = _project(x0, y0, r0, gi, fr, sel)
        if p is None:
            continue
        px, py, pr, v = _slp(p[0], p[1], p[2], deadline, steps=SCREEN_STEPS)
        cands.append((v, px, py, pr))
    cands.sort(key=lambda t: -t[0])
    return [c[1:] for c in cands[:keep]]


def _ruin(x, y, r, rng, k, mode):
    """Pick k circles to eject.  Three ruin operators, hedged over rather than committed to:
    0 = the k smallest (they contribute least to Sigma r), 1 = k uniformly at random,
    2 = a spatial cluster (the k nearest a random point) -- the only one that can free a whole
    REGION and so let the contact graph re-form at a different scale."""
    n = x.size
    if mode == 0:
        idx = np.argsort(r)[:k]
    elif mode == 1:
        idx = rng.choice(n, size=k, replace=False)
    else:
        px, py = rng.uniform(LO, HI), rng.uniform(LO, HI)
        idx = np.argsort((x - px) ** 2 + (y - py) ** 2)[:k]
    keep = np.setdiff1d(np.arange(n), np.asarray(idx, dtype=np.int64))
    return keep


def _recreate(x, y, r, k, deadline):
    """Greedily re-insert k circles, each at the exact largest empty pocket of the CURRENT partial
    packing.  Feasible after every insertion, by construction."""
    for _ in range(k):
        cx, cy, rho = _largest_empty(x, y, r)
        x = np.append(x, cx)
        y = np.append(y, cy)
        r = np.append(r, max(rho * (1.0 - 1e-9), 1e-9))
        if time.process_time() >= deadline:
            break
    return x, y, r


# ---------------------------------------------- the structured seed family, ENUMERATED not sampled
def _row_lattice(n, k, shift):
    """One structured seed: n circles in k rows, row counts as equal as possible with the remainder
    handed to alternating rows; `shift` offsets the odd rows by half a spacing (hexagonal) instead of
    leaving them aligned (square grid).  Returns (n,2) centres, or None if k rows cannot hold n."""
    base, rem = divmod(n, k)
    if base < 1:
        return None
    counts = np.full(k, base, dtype=np.int64)
    order = np.concatenate([np.arange(0, k, 2), np.arange(1, k, 2)])
    counts[order[:rem]] += 1
    pts = np.empty((n, 2))
    p = 0
    for j in range(k):
        c = int(counts[j])
        off = (0.5 / c) if (shift and (j % 2)) else 0.0
        pts[p:p + c, 0] = LO + (np.arange(c) + 0.5) / c + off
        pts[p:p + c, 1] = LO + (j + 0.5) / k
        p += c
    pts[:, 0] = np.clip(pts[:, 0], LO + 1e-6, HI - 1e-6)
    return pts


def _lattice_seeds(n):
    """The whole family: every row count k that can hold n, in both the aligned and the offset variant.

    This is the move that replaces the last GUESS about where a packing comes from.  The seed used to be
    a random cloud pushed apart by a repulsion flow, and which basin it landed in was luck.  But the
    thing that actually distinguishes a good csqv basin from a bad one is COMBINATORIAL -- how many rows,
    how the surplus circles are spread over them, and whether adjacent rows interlock -- and that variable
    ranges over a set small enough to ENUMERATE (~2.4*sqrt(n) values of k, times two interlock parities).
    So the continuous part is left to the SLP, which solves it exactly, and the discrete part is not
    sampled at all: it is exhausted."""
    out = []
    for k in range(2, int(2.4 * np.sqrt(max(n, 1))) + 2):
        a = _row_lattice(n, k, 0)
        if a is None:
            break
        out.append(a)
        b = _row_lattice(n, k, 1)
        if b is not None:
            out.append(b)
    return out


def _lattice_stage(n, evaluate, meter, deadline):
    """Enumerate the structured family, screen it with truncated SLP, converge the survivors.

    Screening is a tournament in three rounds with a shrinking field (all seeds at 3 LP solves ->
    the best 8 at 12 -> the best 2 to convergence) because a truncated SLP is a far better predictor of
    the converged value than the raw seed is: at n=99 the raw maximal-growth radii rank the winning
    lattice 15th, while three LP solves already rank it 1st.  Every intermediate state is a feasible
    packing, so the rounds can be cut off by the clock at any point without losing anything."""
    seeds = _lattice_seeds(n)
    if not seeds:
        return None
    packs = _grow_batch(np.stack(seeds))
    if meter.left() >= packs.shape[0]:
        evaluate(n, packs)                     # already feasible by construction -- score them for free
    t0 = time.process_time()
    span = max(deadline - t0, 0.0)
    if span <= 0.0:
        return None
    d1, d2 = t0 + 0.45 * span, t0 + 0.72 * span
    vals, states = [], []
    for i in range(packs.shape[0]):
        if time.process_time() >= d1:
            break
        x, y, r, v = _slp(packs[i, :, 0].copy(), packs[i, :, 1].copy(), packs[i, :, 2].copy(),
                          d1, steps=3)
        vals.append(v)
        states.append((x, y, r))
    if not states:
        return None
    rank = list(np.argsort(-np.asarray(vals)))
    round2 = []
    for i in rank[:8]:
        if time.process_time() >= d2:
            break
        x, y, r = states[i]
        x, y, r, v = _slp(x, y, r, d2, steps=12)
        states[i] = (x, y, r)
        round2.append((v, int(i)))
    finals = [i for _, i in sorted(round2, reverse=True)[:2]] or [int(rank[0])]
    best, best_v = None, -np.inf
    for i in finals:
        x, y, r = states[i]
        x, y, r, _ = _slp(x, y, r, deadline)
        pack = _finish(x, y, r)
        v = float(pack[:, 2].sum())
        if v > best_v:
            best_v, best = v, pack
        if time.process_time() >= deadline:
            break
    return None if best is None else (best, best_v)


# ---------------------------------------- the census is ONE population: structure transfer across n
def _morph(x, y, r, n, deadline):
    """Re-size an m-circle packing into an n-circle one.  Returns (x, y, r) of size n, or None.

    The census was 37 independent problems; it is really one family, because the optimal packing for n
    and for n +- 2 share almost all of their combinatorics.  Both directions of the resize are already
    exactly-solved subproblems in this file:

      m > n : drop the (m - n) SMALLEST circles -- they contribute least to Sigma r -- and let the SLP
              re-form the contact graph of the survivors in the space that frees up.
      m < n : relax FIRST (an unrelaxed packing has no pocket bigger than the hole you just made), then
              insert (n - m) circles at the exact argmax of the clearance function, feasible by
              construction.

    Every intermediate state is a valid packing, so a deadline may cut this at any point."""
    m = x.size
    t = time.process_time()
    span = max(deadline - t, 0.0)
    if m == n:
        return x.copy(), y.copy(), r.copy()
    if m > n:
        keep = np.argsort(r)[m - n:]
        x, y, r = x[keep].copy(), y[keep].copy(), r[keep].copy()
        x, y, r, _ = _slp(x, y, r, t + 0.55 * span)
    else:
        x, y, r, _ = _slp(x.copy(), y.copy(), r.copy(), t + 0.30 * span)
        x, y, r = _recreate(x, y, r, n - m, t + 0.75 * span)
    if x.size != n or not np.isfinite(r).all():
        return None
    return x, y, r


_RECORDS = None


def _quality(n, v):
    """A packing's value made COMPARABLE ACROSS SIZES, so donors of different n can be ranked.

    With the record table (reading it is explicitly sanctioned) this is the fraction of the record
    attained.  Without it -- an offline re-run on an un-censused n -- Sigma r grows like c*sqrt(n) for
    near-optimal packings, so v/sqrt(n) is the scale-free stand-in.  Only used to ORDER work."""
    global _RECORDS
    if _RECORDS is None:
        _RECORDS = {}
        try:
            with open("bench/records.json") as f:
                d = json.load(f)
            src = d.get("records", d)
            _RECORDS = {int(k): float(x) for k, x in src.items()}
        except Exception:
            _RECORDS = {}
    rec = _RECORDS.get(int(n))
    if rec:
        return v / rec
    return v / max(np.sqrt(float(n)), 1e-12)


def _attempt(n, pool, evaluate, meter, deadline, donors, tried):
    """ONE transfer attempt at n: carry the single best UNTRIED donor for the whole slice.

    The previous schedule split half the slice evenly over all four donors, and that was measured to be
    the binding mistake: `artifacts/probe_donor.py` shows n=35's only useful donor (m=33) needs ~1 CPU-s
    *inside the morph* before the resized contact graph has re-formed at all -- at the ~0.15 s a flat
    split could afford, every donor comes back looking equally bad and the ranking is noise.  Handing
    the freed time to the post-morph SLP instead does not help: it polishes a state the resize never
    finished building.

    So the hedge moves from WITHIN a visit to ACROSS visits.  One visit = one donor, carried for the
    full measured unit; `_quality` (the donor's own fraction of its record) picks which one, and the
    work-list brings n back for the next-best donor later, when it is still the neediest size.  A size
    whose donors are all spent is dropped from the work-list -- and re-armed the moment a neighbour
    improves, because the donor it already tried is now a different packing.

    Returns True if n's entry improved, False if the attempt lost, None if n has no donor left."""
    now = time.process_time()
    span = deadline - now
    if span <= 0.0:
        return False
    seen = tried.setdefault(n, set())
    srcs = []
    for off in donors:
        if off in seen:
            continue
        m = n + off
        if m < 2:
            continue
        d = pool.get(m)
        if d is None:
            d = _read_pack(m)
            if d is not None:
                pool[m] = d
        if d is not None:
            srcs.append((_quality(m, float(d[2].sum())), off, d))
    if not srcs:
        return None
    srcs.sort(key=lambda s: -s[0])
    _q, off, d = srcs[0]
    seen.add(off)
    out = _morph(d[0], d[1], d[2], n, min(now + MORPH_FRAC * span, deadline))
    if out is None:
        return False
    x, y, r, _ = _slp(out[0], out[1], out[2], deadline)
    pack = _finish(x, y, r)
    v = float(pack[:, 2].sum())
    cur = pool.get(n)
    if cur is not None and v <= float(cur[2].sum()):
        return False
    pool[n] = (pack[:, 0].copy(), pack[:, 1].copy(), pack[:, 2].copy())
    if meter.left() > 0:
        evaluate(n, pack)
    return True


def _diffuse(ns, pool, evaluate, meter, deadline, donors=DONORS):
    """Diffusion as a FIXPOINT over a work-list, not a fixed number of passes.

    A sweep visits all 37 sizes whether or not anything upstream of them moved, so it spends most of
    the phase re-screening pairs that settled two sweeps ago -- and it pays for that breadth out of the
    only thing that matters, the CPU each attempt gets.  Measured: an attempt needs ~1-3 CPU-s to rank
    its donors correctly; at the ~0.25 s a 37-wide double sweep could afford, n=35's winning donor was
    not even selected, and one second on it alone cut that size's gap by a third.

    So: keep a work-list.  Visit only sizes whose neighbourhood has actually moved, ordered by how far
    the best neighbour out-qualifies them; when a size improves, WAKE its neighbours.  An empty
    work-list means the census has finished diffusing -- the phase ends early and hands the rest of its
    CPU to whatever runs next, instead of burning it on attempts that cannot pay."""
    nsset = set(int(v) for v in ns)
    todo = set(nsset)
    tried = {}
    while todo:
        now = time.process_time()
        if now >= deadline or meter.left() <= 0:
            break
        pick, best_p = None, None
        for n in todo:
            own = pool.get(n)
            q0 = _quality(n, float(own[2].sum())) if own is not None else -1e9
            p = -1e9
            for off in donors:
                d = pool.get(n + off)
                if d is not None:
                    p = max(p, _quality(n + off, float(d[2].sum())) - q0)
            if best_p is None or p > best_p:
                pick, best_p = n, p
        span = min(max((deadline - now) / float(len(todo) + 1), SLICE_MIN), SLICE_MAX)
        got = _attempt(pick, pool, evaluate, meter, min(now + span, deadline), donors, tried)
        if got is None:
            todo.discard(pick)                  # every donor of `pick` is spent -- rest until re-armed
            continue
        if got:
            for off in donors:
                if pick + off in nsset:
                    todo.add(pick + off)
                    tried.pop(pick + off, None)  # a moved neighbour is new material: re-arm its donors


def _repair(xy, r):
    """Make (xy, r) STRICTLY feasible with a minimal local shave, then a uniform safety margin.

    Violations arriving here are ~1e-11 (LP tolerances), so shaving only the circles that actually
    participate in a violation -- rather than every circle -- keeps the digit loss negligible."""
    n = xy.shape[0]
    r = np.minimum(np.maximum(r, 0.0), np.maximum(_walls(xy), 0.0))
    if n >= 2:
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        di = np.arange(n)
        d[di, di] = np.inf
        for _ in range(40):
            exc = (r[:, None] + r[None, :]) - d
            exc[di, di] = -np.inf
            e = exc.max(axis=1)
            if e.max() <= 0.0:
                break
            r = np.maximum(r - 0.5 * np.maximum(e, 0.0), 0.0)
    r = np.maximum(r - SAFETY, 1e-12)
    return np.concatenate([xy, r[:, None]], axis=1)


def _finish(x, y, r):
    """Best radii for these centres (the LP is exact there), repaired to strict feasibility."""
    xy = np.stack([np.clip(x, LO + EPS_IN, HI - EPS_IN), np.clip(y, LO + EPS_IN, HI - EPS_IN)], axis=1)
    lp = _lp_radii(xy)
    if lp is not None and lp[0] > float(np.sum(r)):
        r = lp[1]
    return _repair(xy, r)


# ------------------------------------------------------------------------------- census warm start
def _read_pack(n):
    """(x, y, r) from the committed census pack for n, or None.  GUARDED: solve() is routinely called with
    an n that has no pack (fresh n, offline re-run), and an unguarded open() would abort the whole call."""
    p = "bench/packs/csqv%d.pck" % n
    try:
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            rows = [ln.split() for ln in fh.read().strip().splitlines()[2:]]
        a = np.array([[float(v) for v in row[:3]] for row in rows if len(row) >= 3], dtype=float)
        if a.shape[0] != n or not np.isfinite(a).all():
            return None
        return (np.clip(a[:, 0], LO, HI), np.clip(a[:, 1], LO, HI), np.maximum(a[:, 2], 0.0))
    except Exception:                                   # noqa: BLE001
        return None


# =============================================================================================== main
def _slot(n):
    """CPU-s one VISIT to size n is worth: the measured time for its 18-entry ladder to complete.

    `artifacts/probe_slice.py` ran the real hop loop from every committed pack at 1.2 / 5.0 / 12.0
    CPU-s.  The gain is a STEP, not a curve: at 1.2 s -- what an even 37-way split of the phase affords
    -- three of four sizes moved by exactly nothing, at 5.0 s n=37 went 3.726e-3 -> 1.611e-3, n=49
    2.893e-3 -> 1.379e-3, n=61 1.715e-3 -> 1.108e-3, and 5.0 -> 12.0 s then bought NOTHING more at any
    of them.  The step sits exactly where `lad` reaches 18, i.e. where the ladder finishes; past that
    kick and ruin are dead (probe_conv: every committed pack is rigid).  So a visit has a floor AND a
    ceiling and the ladder's own completion time is both.  It grows about quadratically in n (n=37
    ~2.4 s, n=49 ~3.5 s, n=61 ~5 s, n=81 ~11 s): one LP solve is superlinear in n, and the ladder is
    eighteen of them with an SLP each."""
    return float(min(max(SLOT_C * float(n) * float(n), SLOT_LO), SLOT_HI))


def _relgap(n, v):
    """Distance to the record, or None when this n is not in the census (offline re-run on a fresh n)."""
    _quality(n, v)                                      # populates _RECORDS on first use
    rec = _RECORDS.get(int(n))
    if not rec:
        return None
    return max(0.0, (rec - float(v)) / rec)


def _worklist(ns, pool):
    """Which sizes this call VISITS, neediest first.

    Two exclusions, both measured rather than assumed.  (a) A size already inside CAP_RELGAP scores the
    full seven digits, so every further CPU-s spent on it raises SCORE by exactly zero -- and eleven of
    the thirty-seven were in that state, holding 35% of this phase's CPU under the n-proportional split
    it used to get.  (b) Since a visit costs a whole _slot(n) the phase can only afford a handful of
    sizes, so they are ordered by distance from the record: the neediest have the most digits available
    and are the ones the ladder has not yet been given a full pass on.  A size skipped by this call
    keeps its committed pack untouched (the driver restores its snapshot of bench/packs) and still
    receives both diffusion sweeps, so skipping it costs nothing but the visit.

    A target with no record entry -- an offline re-run on an un-censused n -- sorts FIRST, never out."""
    scored = []
    for n in ns:
        d = pool.get(n)
        g = _relgap(n, float(d[2].sum())) if d is not None else None
        if g is None:
            scored.append((1e9, n))                     # un-censused or cold: everything to gain
        elif g > CAP_RELGAP:
            scored.append((g, n))
    if not scored:                                      # every target is capped already -- visit anyway
        scored = [(0.0, n) for n in ns]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [n for _g, n in scored]


def _rounds(ns, pool, main_end):
    """The visit schedule: the work-list CYCLED, not walked once.

    `artifacts/probe_diff.py` measured the two things this replaces.  (a) The diffusion phases held 56%
    of the budget and, run for 26 CPU-s against the committed census, spent **zero evaluations and
    improved nothing** -- the census has finished diffusing, so that budget is confiscated below.
    (b) Where it goes: the same 26 CPU-s of VISITS bought +0.455 digits, and all three sizes that moved
    (35, 49, 81) had already been visited to ladder-exhaustion the previous iteration.  So a visit's
    ceiling is per-PACK, not per-SIZE: once the ladder lands an improvement the pack is different
    material and the same eighteen rungs climb somewhere new.  Walking the list once therefore leaves
    the confiscated CPU with nowhere to go; cycling it spends every second on the neediest size
    *currently* in the census.

    The order is recomputed each pass from the live `pool`, so a size that just improved falls back in
    the queue behind whatever is now furthest from its record."""
    while time.process_time() < main_end:
        wl = _worklist(ns, pool)
        if not wl:
            return
        for n in wl:
            if time.process_time() >= main_end:
                return
            yield n


def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    t0 = time.process_time()
    t_end = t0 + CPU_BUDGET                             # ALWAYS relative to entry (re-entrant safe)
    ns = sorted(int(t) for t in targets)

    # The live census: best (x, y, r) known for each size, seeded from the committed packs and updated
    # in place.  This is the state the diffusion sweeps travel over.
    pool = {}
    for n in ns:
        d = _read_pack(n)
        if d is not None:
            pool[n] = d

    # -- (A) PRE-DIFFUSION, but ONLY when the census it would travel over is not already saturated.
    #        Diffusion earned its 56% while the census was young and neighbouring sizes still held
    #        structure each other had never seen.  `artifacts/probe_diff.py` re-measured it on the
    #        CURRENT census: 26 CPU-s of pre- plus far-diffusion spent **0 evaluations** and improved
    #        **0 of 37** sizes -- every donor is spent, `_attempt` refuses every pair, and the phase
    #        returns having done nothing.  A stage that cannot pay is not a stage to improve, it is a
    #        budget to take: the same 26 CPU-s of visits bought +0.455 digits in the same probe.
    #        It is confiscated CONDITIONALLY, not deleted -- on a COLD n (an offline re-run, a fresh
    #        size) there is no committed pack to be saturated and transfer is the whole point, so the
    #        phase runs at full strength exactly when some target has nothing of its own.
    #        ... and it is UN-confiscated again, because that reading EXPIRED.  `artifacts/probe_reopen.py`
    #        re-runs this exact stage on the census as it is today: 45 CPU-s takes n=65 from 4.201 to
    #        7.000 digits and n=77 from 3.387 to 7.000 (+0.173 SCORE), and lifts four already-capped
    #        sizes further past the record.  Nothing about the stage changed -- the CENSUS did.  When
    #        iteration 11 judged it, almost no size held a record-quality packing, so every donor was
    #        as stuck as its recipient and `_attempt` correctly refused them all.  There are now
    #        SIXTEEN sizes at the cap, eleven strictly past the record, and each is donor material that
    #        did not exist at the moment the stage was measured dead.  Against the same census the
    #        per-n visit loop is the one returning zero: `artifacts/probe_op1x.py` at the REAL slot
    #        finds every operator -- ladder, kick and all three ruin modes -- gaining 0.000e+00 at
    #        n=37, 81, 59, 53, 49 over 27 CPU-s.  So the shares swap.  It stays a FIXPOINT, not a fixed
    #        phase: an empty work-list ends it early and hands the remainder straight to the visits, so
    #        this is additive -- the visit loop can only gain CPU by diffusion going dry.
    _diffuse(ns, pool, evaluate, meter, min(t0 + CPU_BUDGET * TRANS_PRE, t_end))
    main_end = t0 + CPU_BUDGET * (1.0 - TRANS_POST)
    # -- (B) THE VISIT LIST.  Not "every target, one slice each": that split is below the operator's
    #        minimum viable unit (see _slot) and was measured buying zero at three of four sizes.
    #        Neediest first, each at the ladder's own measured unit, for as many sizes as fit.
    for n in _rounds(ns, pool, main_end):
        now = time.process_time()
        if now >= main_end or meter.left() <= 0:
            break
        deadline = min(now + _slot(n), main_end)

        # -- (0) STRUCTURED LATTICE ENUMERATION.  The discrete part of the problem (row count and
        #        interlock parity) is exhausted rather than sampled, and the SLP solves the continuous
        #        part exactly on each candidate.  But `artifacts/probe_alt.py` measured this whole
        #        family -- old rows and the mixed-size alternating rows both -- landing WORSE than the
        #        already-committed pack at every censused n: against a warm start it is a DOMINATED
        #        stage, and half the slice was going to it.  It keeps the full share only where it is
        #        not dominated, i.e. on a COLD n with no committed pack, where it is the only
        #        structured seed there is.  The rest is donated to the projection ladder below.
        best, best_v = None, -np.inf
        warm = pool.get(n)
        lf = LAT_FRAC if warm is None else LAT_FRAC_WARM
        if lf > 0.0:
            lat = _lattice_stage(n, evaluate, meter, now + (deadline - now) * lf)
            if lat is not None:
                best, best_v = lat
                if meter.left() > 0:
                    evaluate(n, best)
            now = time.process_time()

        # -- (1) cheap batched restarts: an immediate feasible pack for this n, and good starting centres
        B = int(max(8, min(32, 1500 // max(n, 1))))
        steps = int(max(24, min(120, 2500 // max(n, 1))))
        seed = None
        if warm is None and meter.left() >= B:
            cand = _prerelax(rng, n, B, steps)
            packs = _grow_batch(cand)
            feas, s = evaluate(n, packs)
            s = np.where(np.asarray(feas), np.asarray(s), -np.inf)
            k = int(np.argmax(s))
            if np.isfinite(s[k]):
                seed = (packs[k, :, 0].copy(), packs[k, :, 1].copy(), packs[k, :, 2].copy())
        if seed is None and warm is None:
            c = np.clip(rng.uniform(LO + 0.1, HI - 0.1, size=(n, 2)), LO + EPS_IN, HI - EPS_IN)
            seed = (c[:, 0], c[:, 1], np.full(n, 0.3 / max(np.sqrt(n), 1.0)))

        # A warm n does NOT race a random cloud.  The committed pack is an SLP-rigid local optimum
        # (probe_conv: 3 CPU-s of SLP moves Sigma r by <2e-9 at every size) and the cloud has never once
        # beaten it -- yet the old race handed the cloud 55% of the slice and the warm start 27%.  Both
        # shares are confiscated: the warm start gets a short formality re-converge and the projection
        # ladder below gets the rest.  A COLD n keeps the full race; there the seed is all there is.
        starts = [warm] if warm is not None else [seed]
        share = WARM_SLP_FRAC if warm is not None else 0.55

        # -- (2) SLP to convergence from every start, sharing this n's CPU slice
        for k, (sx, sy, sr) in enumerate(starts):
            sub = now + (deadline - now) * share * (k + 1) / len(starts)
            px, py, pr, _ = _slp(sx.copy(), sy.copy(), sr.copy(), min(sub, deadline))
            pack = _finish(px, py, pr)
            v = float(pack[:, 2].sum())
            if v > best_v:
                best_v, best = v, pack
            if time.process_time() >= deadline:
                break
        if best is None:
            continue
        if meter.left() > 0:
            evaluate(n, best)

        # -- (3) basin hopping over the kick ladder: a converged SLP is FINISHED, so the rest of the
        #        slice is worth more spent in neighbouring basins than idling.  Hedging across several
        #        kick scales beats committing to one -- which scale pays depends on n and on the basin.
        hop = 0
        plan = None                                     # screened survivors of the projection ladder
        passes = 0                                      # ladder passes taken from the CURRENT best
        dry = 0                                         # hops since the last improvement
        while time.process_time() < deadline and meter.left() > 0:
            # The slot has a CEILING as well as a floor: probe_slice measured 5.0 -> 12.0 CPU-s buying
            # nothing once the ladder was spent.  So when the ladder is spent and the stochastic
            # operators have gone dry, END the visit and hand the remainder to the next size.
            if plan is not None and not plan and dry >= DRY_HOPS:
                break
            if plan is None:
                # A ladder pass is now cheap enough to REPEAT within one visit.  Iteration 11 measured
                # that the ceiling is per-PACK, not per-size: once the pack moves, the same eighteen
                # rungs climb somewhere new.  It acted on that by cycling the work-list; screening makes
                # a pass ~4x cheaper, so the same fact can now be exploited INSIDE the visit as well --
                # every accepted improvement resets `plan` and the ladder is re-screened from the new
                # pack.  Bounded by the slot, which is what hands the remainder to the next size.
                plan = _ladder_plan(best, rng, deadline)
                passes += 1
            rm = max(float(best[:, 2].mean()), 1e-12)
            # The ladder leads, but every 7th hop is handed back to the stochastic operators: a new
            # stage is additive, never substitutive, so kick and ruin keep competing for the max.
            if plan and hop % 7 != 6:
                # SYMMETRY PROJECTION FIRST, over the whole enumerated (D4 element x fraction) ladder.
                # It is a different KIND of move from kick or ruin: it displaces every circle at once,
                # but along the square's own symmetry, so it lands in a structured neighbour of the
                # basin instead of a random one.  Measured against the committed census: the four
                # reflections alone took n=47 to the 7-digit cap (iteration 8); adding the two
                # rotations takes n=37 to 1.61e-3 where no reflection passes 3.10e-3, and dropping to
                # frac 0.8 takes n=63 past the record and unsticks n=35 and n=81, which NO full
                # projection could move.  Each entry is tried at full strength against the remaining
                # deadline -- the ladder is truncated by time, never split across it.
                sy_ = plan.pop(0)
                px, py, pr, _ = _slp(sy_[0].copy(), sy_[1].copy(), sy_[2].copy(), deadline)
            elif hop % 4 == 3:
                # keep the old continuous kick alive as one operator among several: it moves ALL
                # circles a little, which is a different neighbourhood from ejecting a few entirely.
                sig = KICKS[(hop // 4) % len(KICKS)] * rm
                jx = np.clip(best[:, 0] + rng.normal(0.0, sig, size=n), LO, HI)
                jy = np.clip(best[:, 1] + rng.normal(0.0, sig, size=n), LO, HI)
                px, py, pr, _ = _slp(jx, jy, best[:, 2] * 0.9, deadline)
            else:
                # RUIN & RECREATE.  Eject k circles, let SLP relax the survivors into the freed space
                # (this is what makes it a real combinatorial move -- re-inserting into an unrelaxed
                # hole just refills the hole it came from), then put k circles back at the exact
                # largest empty pockets and re-converge.  The recreate step is an exactly solved
                # subproblem, and its answer is feasible by construction.
                k = int(np.clip(RUIN[hop % len(RUIN)] * n, 1, max(1, n // 3)))
                keep = _ruin(best[:, 0], best[:, 1], best[:, 2], rng, k, hop % 3)
                mid = time.process_time() + 0.45 * max(deadline - time.process_time(), 0.0)
                hx, hy, hr, _ = _slp(best[keep, 0].copy(), best[keep, 1].copy(),
                                     best[keep, 2].copy(), min(mid, deadline))
                hx, hy, hr = _recreate(hx, hy, hr, n - keep.size, deadline)
                if hx.size != n:
                    hop += 1
                    continue
                px, py, pr, _ = _slp(hx, hy, hr, deadline)
            hop += 1
            pack = _finish(px, py, pr)
            v = float(pack[:, 2].sum())
            if v > best_v:
                best_v, best, dry = v, pack, 0
                plan = None                             # the pack moved: the ladder is worth re-screening
                if meter.left() > 0:
                    evaluate(n, pack)
            else:
                dry += 1

        # -- (4) the ONE place the terminal trust-region squeeze is worth paying for.  Every SLP above
        #        stops at SLP_FLOOR: probe_tail measured the last ~25 of ~50 LP solves buying <1.5e-9 of
        #        Sigma r, i.e. relgap ~3e-10 against a 1e-7 cap -- invisible to the scorer and 2x the
        #        hops when confiscated.  But it is not free forever: it is paid back exactly once, on
        #        the single pack this visit actually commits.
        if best is not None and time.process_time() < deadline + 0.5 and meter.left() > 0:
            px, py, pr, _ = _slp(best[:, 0].copy(), best[:, 1].copy(), best[:, 2].copy(),
                                 min(deadline + 0.5, main_end), d0=0.02, floor=SLP_FLOOR_FINAL)
            pack = _finish(px, py, pr)
            v = float(pack[:, 2].sum())
            if v > best_v:
                best_v, best = v, pack
                evaluate(n, pack)

        cur = pool.get(n)
        if best is not None and (cur is None or best_v > float(cur[2].sum())):
            pool[n] = (best[:, 0].copy(), best[:, 1].copy(), best[:, 2].copy())

    # -- (C) POST-DIFFUSION.  Whatever the per-n loop discovered is now the census's property: sweep it
    #        back down and up so a basin found at one size is tried at every size that can host it.
    _diffuse(ns, pool, evaluate, meter, t_end)
    # ... and once the NEAR census has gone dry, whatever CPU that handed back buys the long-range
    # hops the sweep never had room for: +-6 and +-8 donors over the same fixpoint machinery.
    _diffuse(ns, pool, evaluate, meter, t_end, donors=FAR_DONORS)


# ==================================================================================== self-test
def _self_test():
    class _M:
        def __init__(self, b):
            self.budget, self.used = b, 0

        def left(self):
            return max(0, self.budget - self.used)

        def tick(self, k=1):
            g = max(0, min(int(k), self.budget - self.used))
            self.used += g
            return g

    recs = {}
    p = "bench/records.json"
    if os.path.exists(p):
        with open(p) as fh:
            recs = {int(k): float(v) for k, v in json.load(fh)["records"].items()}

    def make(meter, store):
        def ev(n, packing):
            a = np.asarray(packing, dtype=float)
            single = a.ndim == 2
            if single:
                a = a[None]
            B = a.shape[0]
            g = meter.tick(B)
            feas = np.zeros(B, bool)
            sr = np.full(B, -np.inf)
            if g:
                x, y, r = a[:g, :, 0], a[:g, :, 1], a[:g, :, 2]
                wall = np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min(1)
                dx = x[:, :, None] - x[:, None, :]
                dy = y[:, :, None] - y[:, None, :]
                sl = np.sqrt(dx * dx + dy * dy) - (r[:, :, None] + r[:, None, :])
                sl[:, np.eye(a.shape[1], dtype=bool)] = np.inf
                pair = sl.reshape(g, -1).min(1)
                f = (r.min(1) > 0) & (wall >= -1e-9) & (pair >= -1e-9)
                feas[:g], sr[:g] = f, r.sum(1)
                for i in range(g):
                    if f[i] and sr[i] > store.get(n, (-np.inf,))[0]:
                        store[n] = (float(sr[i]), a[i].copy())
            return (bool(feas[0]), float(sr[0])) if single else (feas, sr)
        return ev

    def _gap(n, v):
        return "" if n not in recs else "  relgap=%.3e" % max(0.0, (recs[n] - v) / recs[n])

    ok = True
    global CPU_BUDGET
    real_budget, CPU_BUDGET = CPU_BUDGET, 8.0   # self-test only: shrink the CPU slice so the suite is
    #                                             quick. The METERED run always uses the real budget.
    # CALL solve() TWICE IN ONE PROCESS (MISSION.md): a module-level absolute CPU stop would make the
    # second call a no-op, and a single call would never reveal it.
    for call in (1, 2):
        store = {}
        m = _M(20000)
        t = time.process_time()
        solve(make(m, store), m, np.random.RandomState(7 + call), [29, 40])
        dt = time.process_time() - t
        for n in (29, 40):
            if n not in store:
                print("FAIL call %d: no feasible packing for n=%d" % (call, n))
                ok = False
                continue
            sr, pk = store[n]
            assert pk.shape == (n, 3)
            print("  call %d n=%2d  sum_r=%.9f%s  (%.1fs cpu, %d evals)"
                  % (call, n, sr, _gap(n, sr), dt, m.used))
    # len(targets) == 1 must still work (per-n budget split must not divide by a stale count)
    store = {}
    m = _M(5000)
    solve(make(m, store), m, np.random.RandomState(3), [31])
    if 31 not in store:
        print("FAIL: single-target call produced nothing")
        ok = False
    else:
        print("  single-target n=31 sum_r=%.9f%s" % (store[31][0], _gap(31, store[31][0])))
    # an n with NO committed pack must cold-start cleanly (guarded warm-start read)
    store = {}
    m = _M(4000)
    solve(make(m, store), m, np.random.RandomState(5), [12])
    if 12 not in store:
        print("FAIL: un-censused n=12 produced nothing")
        ok = False
    else:
        print("  cold n=12 sum_r=%.9f" % store[12][0])
    # an exhausted meter must not raise
    m = _M(0)
    solve(make(m, {}), m, np.random.RandomState(1), [27])
    print("  exhausted-meter call returned cleanly")
    CPU_BUDGET = real_budget
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
