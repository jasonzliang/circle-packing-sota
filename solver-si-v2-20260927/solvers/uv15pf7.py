"""Circle-packing solver for the packomania `csqv` family: n circles in the unit square, maximize Sigma r.

Contract (MISSION.md): `solve(evaluate, meter, rng, targets)`. `evaluate(n, packing) -> (feasible, sum_r)`
is the only sanctioned candidate test; the driver writes exactly the strictly-feasible bests routed
through it.

WHAT THIS DOES (and why it replaces the seed solver's equal-radius descent)
--------------------------------------------------------------------------
The seed solver gave every circle the "safe" radius r_i = min(wall_i, min_j d_ij / 2). That rule is
optimal only when all radii are equal, and the csqv optima are strongly *unequal*, so the seed left
~26% of every record on the table. This solver instead treats the problem as the nonlinear program it
is -- variables (x_i, y_i, r_i), objective Sigma r_i, constraints r_i + r_j <= d_ij and r_i <= wall --
and attacks it with:

  1. a quadratic-penalty relaxation of all constraints, minimized by L-BFGS-B with an analytic,
     fully vectorized gradient, over an escalating penalty schedule mu = 1e1 .. 1e6 (`_penalty_solve`);
  2. an exact LP for the radii once the centres are fixed -- max Sigma r s.t. r_i + r_j <= d_ij,
     0 <= r_i <= wall_i is a linear program, so `_lp_radii` returns the *best possible* radii for those
     centres (HiGHS), not a heuristic approximation;
  3. `_repair`, a one-pass, provably sufficient shrink that turns any radius vector into a strictly
     feasible one (see its docstring), so every candidate handed to evaluate() is legal by construction.

The EXPLORE/EXPLOIT split is adaptive, not fixed (`_COLD_PROBE_ROUNDS`): instrumenting a 16 s depth
run on the iteration-9 census showed it screening 128 cold starts at n=57 (202 at n=49, 88 at n=81)
without one of them beating the committed packing, so on a warm size most of the exploration budget
was buying candidates that cannot win. The restart phase now PROBES with two halving batches and
abandons exploration only when both came back below the warm incumbent; a cold size (no committed
pack) has warm_sum = -inf, so nothing there changes and restarts keep the whole slice.

Search = multi-start, then basin hopping, whose steps are of TWO kinds: a Gaussian shake of every
centre (continuous -- almost always relaxes back into the same contact graph) and, half the time, a
RESEAT (`_reseat_move`: lift the k smallest circles out and re-drop them into sampled holes), which is
discrete and can change the contact graph. Measured at matched CPU (artifacts/exp_reseat*.py): a wash
at 6 s/size, where the depth loop barely runs (5 sizes x 2 seeds, 3.7799 vs 3.7811 mean digits), and
+0.127 mean digits at 15 s/size (3 sizes x 1 seed: +0.376 at n=43, +0.004 at n=49, tie at n=57, no
loss) -- i.e. it pays in the deep-per-size regime the offline held-out re-run occupies, and costs
nothing in the shallow in-run one. Restarts are run
under SUCCESSIVE HALVING (`_halving_round`): a partially converged penalty solve already ranks starts
at Spearman 0.83-0.99 against their converged Sigma r, so every start in a batch gets only the cheap
low-mu stages (~26% of a restart) and is scored, and only the best two of each batch of eight pay for
the rest. At matched CPU that is worth +0.086 mean digits at 3 s/size and +0.246 at 10 s/size, and it
lost at no size at either budget. The START DISTRIBUTION is what dominates the result, and there are two good ones, alternated on every other
restart (`_cold_init`):

  * jittered HEXAGONAL lattices with randomized row count / stagger / shrink / jitter (`_hex_init`),
    which at matched CPU beat a jittered square grid by ~0.4-0.5 digits and uniform-random starts by
    ~0.8-1.2 digits per size;
  * CROSS-SIZE STRUCTURAL TRANSFER (`_transfer_init`): take a NEIGHBOURING size's solved packing and
    drop its smallest circles (m > n) or fill its largest holes (m < n). Sizes n and n+-2 share almost
    the same contact graph, so this reaches basins hex restarts find only by luck -- measured at
    matched CPU it moved n=45 from 2.38 to 3.26 digits and n=87 from 4.52 past the record, and tied at
    n=95. Neighbours come from the committed census AND from this run's own bests, so an improvement at
    one size propagates to the next. It degrades to a hex start when no neighbour packing exists, which
    is the case for an n far from the census.

Warm starts from the committed census for n ITSELF are used when present and are always re-scored, so a
committed packing can never be lost.

KNOWN LIMIT, measured not assumed (notes/init-distribution-experiment.md): the packings this produces are already KKT
points -- an active-set SLSQP polish of a committed packing moved Sigma r by <2e-5. For the LAGGING sizes,
whose gap is ~3e-3, the remaining gap is therefore entirely COMBINATORIAL (the wrong contact graph) and
the next gain must come from the start distribution or a structure-changing move. That same <2e-5 is
decisive, though, for a size already within 1e-5 of the record, which is why `polish` is now run on the
incumbent (it took n=31 and n=35 from 6.09/6.06 digits to the 7.00 cap).

METERING. Two meters bind: 500k evaluations and process CPU. This solver is CPU-bound by a wide margin
(a restart costs ~0.05 s at n=27 and ~0.25 s at n=99 but only ONE evaluation), so the allocation
decision that matters is CPU, not evaluations. Deadlines are computed from `time.process_time()` AT
ENTRY and from `len(targets)` -- never as module constants -- so a second call in the same process and a
one-size-per-call offline re-run both behave correctly. Work is done in two passes (a fast sweep over
every target that has no committed packing to fall back on, then depth), so an early CPU backstop still
leaves every target with a feasible packing already routed through evaluate(). The depth pass is
a SCREEN, not a proportional split (`_depth_order`, `_solve_one(probe=...)`): a size already at the
7-digit cap gets nothing; every size with digits left is put in a sampled order and given a probe of
~18 s (scaled by n) to beat its own committed packing; one that does keeps the CPU up to
`_CPU_PER_SIZE_S`, one that does not is abandoned on the spot and the next size gets the seconds.
Measured on ten lagging sizes at 52 s each x 2 seeds (artifacts/exp_curve_*.log), the census splits
cleanly into sizes that improve within ~14 s (37, 39, 41, 53, 61 -- 8 of 9 runs, four of them all the
way to the cap) and sizes that produce nothing at all in 52 s (43, 45, 49, 57, 73 -- 0 of 10), and the
GAP does not tell them apart. For a single-size call the ordering is inert and a size with no
committed packing can never be abandoned.

`python3 tools/solver.py --self-test` exercises all of this, including two solve() calls in one process.
"""
import json
import os
import time

import numpy as np

LO, HI = -0.5, 0.5

# CPU allowance. The in-run driver caps a whole solve() at 120 s of process CPU for all 37 visible n;
# the offline re-run caps each *single-size* call at 60 s. Both are derived here from len(targets) with
# margin, never assumed. A call is never allowed to exceed ~52 s per size or ~104 s in total.
_CPU_PER_SIZE_S = 52.0
_CPU_TOTAL_CAP_S = 104.0

# Penalty schedule for the relaxation. Low mu lets circles slide past each other (global rearrangement);
# high mu drives the constraint violation to ~1e-7 so the repair step costs almost nothing.
_MU_SCHEDULE = (10.0, 1e2, 1e3, 1e4, 1e5, 1e6)

# Cold-start mix, one entry consumed per restart: 6 hex lattices to 1 square grid to 1 uniform. Hex won
# by ~0.5-1.0 digits per size in a matched-CPU sweep (see `_hex_init`); the other two are insurance for
# a size whose optimum is not row-structured, which is why they are not dropped entirely.
_MODE_CYCLE = (0, 0, 0, 1, 0, 0, 0, 2)

# Successive halving over restarts. `_CUT` = how many mu stages every start gets before it is
# ranked; the surviving `_KEEP` of each batch of `_BATCH` get the rest. mu = 10, 100 is ~26% of a
# restart's CPU and already ranks starts at Spearman 0.83-0.99 against their fully converged Sigma r
# (artifacts/exp_halving.py), so this buys ~2x the basins per CPU second for a small ranking risk.
_CUT, _BATCH, _KEEP = 2, 8, 2

# Share of the CPU allowance spent on the pass-1 insurance sweep, counted only over the targets that
# have no committed packing to fall back on (see `solve`).
_SWEEP_FRAC = 0.12


# ----------------------------------------------------------------------------- feasibility primitives
def walls(xy):
    """Distance from each centre to the nearest side of the centered unit square."""
    x, y = xy[..., 0], xy[..., 1]
    return np.minimum.reduce([x - LO, HI - x, y - LO, HI - y])


def repair(xy, r):
    """Shrink `r` to a STRICTLY feasible radius vector for centres `xy`. One pass is provably enough.

    After clipping r_i <= wall_i, let delta_i = max_j max(0, r_i + r_j - d_ij) >= 0 be circle i's worst
    overlap. Setting r_i <- r_i - delta_i/2 makes every pair slack non-negative, because for any pair
    (i, j):  (r_i - delta_i/2) + (r_j - delta_j/2) - d_ij <= (r_i + r_j - d_ij) - (delta_i + delta_j)/2
    and both delta_i and delta_j are >= (r_i + r_j - d_ij). Radii only ever decrease, so the wall clip
    survives. A 1e-12 haircut keeps the result strictly inside the harness's 1e-9 tolerance (its cost,
    ~1e-10 on Sigma r, is 4 orders below the 1e-7 relative gap that already earns full marks).
    """
    n = xy.shape[0]
    r = np.minimum(r, walls(xy))
    if n >= 2:
        d = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt((d * d).sum(-1))
        di = np.arange(n)
        dist[di, di] = np.inf
        delta = np.maximum(r[:, None] + r[None, :] - dist, 0.0).max(axis=1)
        r = r - 0.5 * delta
    return np.maximum(r - 1e-12, 1e-12)


def _lp_radii(xy):
    """EXACT optimal radii for fixed centres: max Sigma r s.t. r_i + r_j <= d_ij, 0 <= r_i <= wall_i.

    That is a linear program, so this is not a heuristic -- it is the best Sigma r those centres admit.
    Only pairs with d_ij < wall_i + wall_j can ever bind, which cuts the constraint count several-fold.
    Returns None if scipy/HiGHS declines, so callers keep their fallback radii.
    """
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:                                        # pragma: no cover - scipy is a dependency
        return None
    n = xy.shape[0]
    w = np.maximum(walls(xy), 0.0)
    if n < 2:
        return w
    d = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    I, J = np.triu_indices(n, 1)
    b = dist[I, J]
    keep = b < (w[I] + w[J])
    I, J, b = I[keep], J[keep], b[keep]
    k = I.size
    if k == 0:
        return w
    rows = np.concatenate([np.arange(k), np.arange(k)])
    A = coo_matrix((np.ones(2 * k), (rows, np.concatenate([I, J]))), shape=(k, n))
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b,
                      bounds=np.stack([np.zeros(n), w], axis=1), method="highs")
    except Exception:                                        # pragma: no cover - solver hiccup
        return None
    if res.x is None:
        return None
    return np.asarray(res.x, dtype=float)


def _grow(xy):
    """Fallback radii when the LP is unavailable: the equal-radius safe rule (what the seed used)."""
    n = xy.shape[0]
    w = walls(xy)
    if n < 2:
        return np.maximum(w - 1e-12, 1e-12)
    d = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    di = np.arange(n)
    dist[di, di] = np.inf
    return np.maximum(np.minimum(w, dist.min(axis=1) / 2.0) - 1e-12, 1e-12)


def best_radii(xy):
    """Strictly feasible radii for `xy`: LP-optimal where possible, safe-rule otherwise."""
    r = _lp_radii(xy)
    if r is None or not np.all(np.isfinite(r)):
        r = _grow(xy)
    return repair(xy, r)


# ------------------------------------------------------------------------------ exact-constraint polish
# Near-active band for `polish`: a pair whose slack exceeds this cannot become binding inside the
# ~1e-5 move an exact-constraint polish makes, so dropping it keeps the SQP small without changing
# the answer. `maxiter` is the SLSQP iteration cap (it costs 0.05-0.3 s per call at n = 27..99).
_POLISH_BAND, _POLISH_ITERS = 0.02, 120


def polish(xy, r):
    """Active-set SQP on (x, y, r) under the EXACT constraints -- the last digit the penalty cannot buy.

    The penalty relaxation stops at mu = 1e6, so its centres sit ~1e-7..1e-5 off the true KKT point;
    `_lp_radii` then gives the best radii for THOSE centres, which is still short of the best radii
    for the nearby exact optimum. Iteration 3 measured this polish at <2e-5 on Sigma r and concluded
    it was irrelevant -- true for the lagging sizes, whose gap is ~3e-3 and combinatorial, but NOT for
    a size already within 1e-5 of the record: measured on the committed census (artifacts/exp_polish.py)
    it moved n=31 from 6.086 digits to the 7.000 cap and n=35 from 6.056 to 7.000, i.e. past the record.

    It is a LOCAL refinement and it can diverge (at n=33 on the committed pack it returned a packing
    a factor of 4 worse). That costs nothing: the result is routed through evaluate() like any other
    candidate and only kept if it is actually better. Returns None if scipy declines.
    """
    try:
        from scipy.optimize import minimize
    except Exception:                                        # pragma: no cover - scipy is a dependency
        return None
    xy = np.asarray(xy, float)
    r = np.asarray(r, float)
    n = xy.shape[0]
    if n < 2:
        return None
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    I, J = np.triu_indices(n, 1)
    keep = (d[I, J] - (r[I] + r[J])) < _POLISH_BAND
    I, J = I[keep], J[keep]
    k = I.size
    di = np.arange(n)
    z0 = np.concatenate([xy.ravel(), r])

    def fun(z):
        return -z[2 * n:].sum()

    def fjac(z):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    def cons(z):
        pt, rr = z[:2 * n].reshape(n, 2), z[2 * n:]
        dx, dy = pt[I, 0] - pt[J, 0], pt[I, 1] - pt[J, 1]
        pair = np.sqrt(dx * dx + dy * dy) - (rr[I] + rr[J])
        wall = np.concatenate([pt[:, 0] - LO - rr, HI - pt[:, 0] - rr,
                               pt[:, 1] - LO - rr, HI - pt[:, 1] - rr])
        return np.concatenate([pair, wall])

    def cjac(z):
        pt = z[:2 * n].reshape(n, 2)
        M = np.zeros((k + 4 * n, 3 * n))
        dx, dy = pt[I, 0] - pt[J, 0], pt[I, 1] - pt[J, 1]
        dist = np.sqrt(dx * dx + dy * dy) + 1e-300
        ux, uy = dx / dist, dy / dist
        rows = np.arange(k)
        M[rows, 2 * I] = ux
        M[rows, 2 * J] = -ux
        M[rows, 2 * I + 1] = uy
        M[rows, 2 * J + 1] = -uy
        M[rows, 2 * n + I] -= 1.0
        M[rows, 2 * n + J] -= 1.0
        for b, (col, sgn) in enumerate(((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0))):
            rw = k + b * n + di
            M[rw, 2 * di + col] = sgn
            M[rw, 2 * n + di] = -1.0
        return M

    try:
        res = minimize(fun, z0, jac=fjac, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                       bounds=[(LO, HI)] * (2 * n) + [(1e-9, 0.5)] * n,
                       options={"maxiter": _POLISH_ITERS, "ftol": 1e-14})
    except Exception:                                        # pragma: no cover - solver hiccup
        return None
    z = np.asarray(res.x, dtype=float)
    if z.shape != z0.shape or not np.all(np.isfinite(z)):
        return None
    return np.clip(z[:2 * n].reshape(n, 2), LO, HI)


# ------------------------------------------------------------------------- penalty NLP (the workhorse)
def _make_fg(n, mu):
    """Value+gradient of  -Sigma r + mu * (pairwise overlap^2 + wall overshoot^2), vectorized over the
    n(n-1)/2 pairs. np.bincount does the scatter-add (measurably faster than np.add.at or an (n,n,2)
    tensor at these sizes: ~170 us per call at n=99)."""
    I, J = np.triu_indices(n, 1)

    def fg(z):
        xy = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        dx = xy[I, 0] - xy[J, 0]
        dy = xy[I, 1] - xy[J, 1]
        dist = np.sqrt(dx * dx + dy * dy) + 1e-15
        v = np.maximum(r[I] + r[J] - dist, 0.0)
        u1 = np.maximum(r - (xy[:, 0] - LO), 0.0)
        u2 = np.maximum(r - (HI - xy[:, 0]), 0.0)
        u3 = np.maximum(r - (xy[:, 1] - LO), 0.0)
        u4 = np.maximum(r - (HI - xy[:, 1]), 0.0)
        f = -r.sum() + mu * ((v * v).sum() + (u1 * u1 + u2 * u2 + u3 * u3 + u4 * u4).sum())
        c = 2.0 * mu * v
        gr = np.bincount(I, c, n) + np.bincount(J, c, n) - 1.0 + 2.0 * mu * (u1 + u2 + u3 + u4)
        cx, cy = c * dx / dist, c * dy / dist
        gx = np.bincount(J, cx, n) - np.bincount(I, cx, n) + 2.0 * mu * (u2 - u1)
        gy = np.bincount(J, cy, n) - np.bincount(I, cy, n) + 2.0 * mu * (u4 - u3)
        return f, np.concatenate([np.stack([gx, gy], axis=1).ravel(), gr])

    return fg


def _penalty_stages(n, z, mus, maxiter=300):
    """Run PART of the escalating-mu schedule from state `z`, returning the new state.

    Split out from `_penalty_solve` so a restart can be stopped after the cheap low-mu stages,
    ranked, and only resumed if it is worth the expensive ones (see `_halving_round`).
    """
    from scipy.optimize import minimize
    bounds = [(LO, HI)] * (2 * n) + [(1e-9, 0.5)] * n
    for mu in mus:
        res = minimize(_make_fg(n, mu), z, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter})
        z = res.x
    return z


def _z0(n, xy0, r0):
    return np.concatenate([np.asarray(xy0, float).ravel(), np.asarray(r0, float)])


def _penalty_solve(n, xy0, r0, maxiter=300):
    """Escalating-mu quadratic penalty, each stage an L-BFGS-B run warm-started from the last."""
    z = _penalty_stages(n, _z0(n, xy0, r0), _MU_SCHEDULE, maxiter)
    return z[:2 * n].reshape(n, 2)


# ------------------------------------------------------------------------------------- census reading
def _pack_path(n):
    rel = os.path.join("bench", "packs", "csqv%d.pck" % n)
    if os.path.exists(rel):
        return rel
    alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "bench", "packs", "csqv%d.pck" % n)
    return alt if os.path.exists(alt) else None


def read_pack(n):
    """Warm start from the committed census, or None. GUARDED: any missing/odd file means a cold start
    (solve() is called with sizes that have no pack -- a fresh n, or the offline held-out re-run)."""
    p = _pack_path(n)
    if p is None:
        return None
    try:
        with open(p, "r") as fh:
            rows = [ln.split() for ln in fh.read().splitlines()[2:] if ln.strip()]
        a = np.array([[float(v) for v in row[:3]] for row in rows if len(row) >= 3], dtype=float)
    except Exception:
        return None
    if a.shape[0] != n or a.shape[1] != 3 or not np.all(np.isfinite(a)):
        return None
    return a


# --------------------------------------------------------------- cross-size structural transfer
def _greedy_holes(xy, r, k, rng, grid=64, q=None):
    """Insert `k` circles at the `k` largest empty spots of the packing (xy, r).

    The largest empty spot is argmax over the square of m(p) = min(wall(p), min_j |p-c_j| - r_j) --
    the radius of the biggest circle that fits at p. It is evaluated on a `grid`x`grid` lattice
    (a coarse argmax is enough: the penalty relaxation that follows moves everything anyway) and
    applied greedily, each placed circle joining the set the next one must avoid.

    `q` switches the pick from argmax to a SAMPLE with probability proportional to max(m, 0)**q --
    still concentrated on the big holes (and on the big ones' larger footprint in grid points), but
    able to choose a different hole on a different draw. The reseat move needs that: re-placing a
    circle in the hole it just vacated is a no-op, so a strictly greedy pick cannot change structure.
    """
    g = (np.arange(grid) + 0.5) / grid + LO
    P = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    xy, r = np.asarray(xy, float).copy(), np.asarray(r, float).copy()
    for _ in range(int(k)):
        d = np.sqrt(((P[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]
        m = np.minimum(d.min(1), walls(P))
        if q is None:
            i = int(np.argmax(m))
        else:
            w = np.maximum(m, 0.0) ** float(q)
            tot = w.sum()
            i = int(np.argmax(m)) if not np.isfinite(tot) or tot <= 0 else \
                int(rng.choice(P.shape[0], p=w / tot))
        xy = np.vstack([xy, P[i] + rng.normal(0.0, 0.25 / grid, 2)])
        r = np.append(r, max(float(m[i]), 1e-4))
    return np.clip(xy, LO + 1e-3, HI - 1e-3), r


def _pool_get(pool, m):
    """Best known packing for size m: this run's own incumbent if we have one, else the committed
    census on disk, else None. Disk misses are cached so a size with no pack is read at most once."""
    if m in pool:
        return pool[m]
    a = read_pack(m)
    pool[m] = None if a is None else (a[:, :2], a[:, 2])
    return pool[m]


def _transfer_init(n, rng, pool, span=6):
    """Cold start for n built from a NEIGHBOURING size's packing -- the highest-value start after hex.

    Structure at n and at n +/- 1,2,... is nearly the same contact graph with a few circles added or
    dropped, so a neighbour's solved packing lands the relaxation in a basin that hex restarts reach
    only by luck. Measured at matched CPU (5 s/size, notes/init-distribution-experiment.md): best-of-run
    digits went 2.38 -> 3.26 (n=45), 2.47 -> 2.73 (n=59) and 4.52 -> 7.00 (n=87, i.e. past the record),
    with n=95 a tie -- so this is mixed with hex starts, never a replacement for them.

      m > n : drop the (m-n) SMALLEST circles (they carry the least Sigma r and their space is
              redistributed by the relaxation).
      m < n : add (n-m) circles at the largest empty spots (`_greedy_holes`).

    Returns None when no neighbour packing exists -- an n far from the census, or a fresh workspace --
    and the caller falls back to a cold hex start, which must work on its own for any n.
    """
    cands = [m for m in range(max(1, n - span), n + span + 1) if m != n]
    rng.shuffle(cands)                                   # break ties between the two nearest sizes
    cands.sort(key=lambda m: abs(m - n))
    for m in cands[:2 * span]:
        got = _pool_get(pool, m)
        if got is None:
            continue
        xy, r = got[0].copy(), got[1].copy()
        if m > n:
            keep = np.argsort(r)[m - n:]
            xy, r = xy[keep], r[keep]
        elif m < n:
            xy, r = _greedy_holes(xy, r, n - m, rng)
        jit = float(rng.uniform(0.0, 0.15)) / np.sqrt(n)
        return np.clip(xy + rng.normal(0.0, jit, xy.shape), LO + 1e-3, HI - 1e-3)
    return None



# ------------------------------------------------------------------------- structural move: reseat
# How often a basin-hopping step is a RESEAT rather than a Gaussian shake of every centre, and how
# many circles at most it moves. Both tuned in artifacts/exp_reseat.py against _RESEAT_P = 0.
_RESEAT_P, _RESEAT_KMAX, _RESEAT_GRID = 0.5, 3, 48


def _reseat_move(xy, r, rng, kmax=_RESEAT_KMAX, grid=_RESEAT_GRID):
    """Basin-hopping step that CHANGES THE CONTACT GRAPH: pull the k smallest circles out and drop
    them into holes elsewhere.

    The Gaussian shake this alternates with is a continuous perturbation: it moves every centre a
    little, so the relaxation almost always falls back into the same contact graph, and the census
    profile says that graph is exactly what the lagging sizes have wrong (their gap is ~1e-3 while an
    exact-constraint polish moves Sigma r by <2e-5, i.e. they are converged KKT points of the WRONG
    combinatorial structure). Removing a circle and re-placing it in a different hole is a discrete
    move that no amount of jitter reaches.

    The smallest circles are chosen because they contribute least to Sigma r and are the ones most
    often stranded in a bad spot; the remaining centres get a light jitter so the vacated hole is not
    an exact copy of itself, and the hole pick is SAMPLED (`q`) rather than greedy so the circle can
    land somewhere it was not.
    """
    xy = np.asarray(xy, float)
    r = np.asarray(r, float)
    n = xy.shape[0]
    if n < 4:
        return xy
    k = min(1 + int(rng.randint(max(1, int(kmax)))), max(1, n // 5))
    keep = np.argsort(r)[k:]
    xy2, r2 = xy[keep].copy(), r[keep].copy()
    jit = float(rng.uniform(0.0, 0.5)) / np.sqrt(n)
    xy2 = np.clip(xy2 + rng.normal(0.0, jit, xy2.shape), LO + 1e-3, HI - 1e-3)
    out, _ = _greedy_holes(xy2, r2, k, rng, grid=grid, q=8.0)
    return out


# Gaussian-kick scale for the depth loop, in units of the MEAN RADIUS rbar = Sigma r / n rather
# than absolute. Measured at n=49/73 on the committed census (artifacts/exp_kick.py, 12 re-solves per
# rung): every kick with sigma <= 0.30*rbar re-converges into the incumbent's own basin (12/12 within
# 1e-4 of it), while sigma = 0.50*rbar lands 2e-3..1.8e-2 BELOW it -- as bad as a cold restart, i.e.
# a guaranteed reject. The window is therefore roughly [0, 0.35]*rbar and it is NARROW.
#
# The rule this replaces, sigma ~ U(0.01, 0.08) ABSOLUTE, is not scale-free, so how much of it falls
# inside that window depends on n: rbar ~ Sigma r / n runs from ~0.113 at n=27 to ~0.052 at n=99, so
# the same constants span sigma/rbar = 0.09..0.71 at n=27 but 0.19..1.55 at n=99. At the large sizes
# well over half of every shake was spent on candidates too destroyed to win, and the waste grew with
# n -- precisely the sizes with the most circles to place. Expressing the kick in rbar makes it the
# same move at every n, which is also what the offline held-out re-run (sizes this loop never scored)
# needs.
_KICK_LO, _KICK_HI = 0.05, 0.35


def _kick_scale(n, r, rng):
    """Sigma of one Gaussian kick: a fraction of the incumbent's mean radius (see above). Falls back
    to the equal-radius estimate 0.5/sqrt(n) when there are no radii yet."""
    rbar = float(np.mean(r)) if r is not None and len(r) else 0.5 / np.sqrt(n)
    return max(1e-6, rbar) * float(rng.uniform(_KICK_LO, _KICK_HI))


# Adaptive cold-restart share. Instrumented at 16 s/size on the iteration-9 census
# (artifacts/exp_where.py): a depth run at n=57 SCREENS 128 COLD STARTS and not one of them beats the
# committed packing, while the loop that refines the incumbent gets ~30 steps -- and the same at n=49
# (202 starts) and n=81 (88). Nine iterations of accumulation have made the committed packing far
# better than anything a fresh 16 s multistart reaches, so on a warm size most of the exploration
# budget is spent on candidates that cannot win. The fix is not a smaller constant (that would break
# the COLD case, where restarts are the only thing there is) but a PROBE: always run
# `_COLD_PROBE_ROUNDS` halving batches, and keep exploring only while some batch's best still reaches
# the warm incumbent. When it does not, the rest of the restart slice -- and most of the depth phase's
# own cold-restart share -- goes to perturbing the incumbent instead.
# One round, not two. The probe's job is to answer "can a fresh start still beat the incumbent?",
# and at a WARM size the answer has been no in every measurement this loop has taken:
# artifacts/exp_where.py screened 128 cold starts at n=57, 202 at n=49 and 88 at n=81 without one
# of them reaching the committed packing, and artifacts/exp_repair.py now adds the other half of the
# restart mix -- a TRANSFER start from a neighbouring size lands 2.48 mean digits against the warm
# incumbent's 3.40, i.e. ~0.9 digits BELOW it, at 14 of 14 lagging sizes and under all four solve
# schedules tried. A second confirming round therefore buys no information and costs `_BATCH` more
# restarts (~n^2 each) out of every screened size's probe window, which is the scarcest CPU here: at
# 104 s and _probe_s(n) = 10..29 s per size, a pass screens only four or five sizes.
# A COLD size has warm_sum = -inf, so this break never fires there and the exploration phase of the
# offline held-out re-run is untouched -- which is why the constant is cut rather than the phase.
_COLD_PROBE_ROUNDS = 1
_PCOLD_LIVE, _PCOLD_DEAD = 0.35, 0.10


# ------------------------------------------------------------------------------------- the search loop
class _Best(object):
    """Tracks the incumbent for one n and routes every candidate through the metered evaluate()."""

    def __init__(self, n, evaluate, meter):
        self.n, self.evaluate, self.meter = n, evaluate, meter
        self.xy, self.r, self.sum_r = None, None, -np.inf
        self.raw_max = -np.inf        # best UNPOLISHED candidate seen (see _trial's record polish)
        self.seen_max = -np.inf                      # best Sigma r SHOWN since the last reset, even
                                                     # when it did not beat the incumbent

    def offer(self, xy, r=None):
        if self.meter.left() <= 0:
            return -np.inf
        xy = np.clip(np.asarray(xy, float), LO, HI)
        r = repair(xy, r) if r is not None else best_radii(xy)
        feas, s = self.evaluate(self.n, np.concatenate([xy, r[:, None]], axis=1))
        s = float(s)
        if bool(feas):
            if s > self.seen_max:
                self.seen_max = s
            if s > self.sum_r:
                self.xy, self.r, self.sum_r = xy.copy(), r.copy(), s
        return s

    def offer_polished(self):
        """Re-offer the incumbent after an exact-constraint polish. Pure upside: a diverged polish is
        simply not better, so `offer` discards it (one evaluation), and a good one can be worth the
        last digit at a size already near the record."""
        if self.xy is None or self.meter.left() <= 0:
            return -np.inf
        p = polish(self.xy, best_radii(self.xy))
        return -np.inf if p is None else self.offer(p)


def _hex_init(n, rng):
    """Jittered HEXAGONAL-lattice cold start -- the single highest-value ingredient in this solver.

    Measured (notes/init-distribution-experiment.md; 10-25 restarts per size, same CPU): best-of-batch
    digits at n=41/71/99 were 2.12/1.96/1.93 from uniform-random starts, 2.59/2.27/2.25 from the
    jittered SQUARE grid this replaces, and 3.31/>4/3.46 from this hex start. The reason is structural,
    not tuning luck: the densest circle arrangement in the plane is hexagonal, so a hex lattice already
    sits in (or beside) the basin of the row-structured optima, whereas a square grid has to break its
    own symmetry first and usually relaxes into a worse contact graph.

    The lattice parameters are RANDOMIZED per restart rather than fixed, because which of them wins is
    strongly n-dependent and no single setting dominated in the sweep:
      rows   -- round(sqrt(n / 0.866)) + d, d in {-2..2}: the hex row count for n points, perturbed.
                (0.866 = sqrt(3)/2, the hex row spacing, so this balances rows against row length.)
      offs   -- the odd-row shift as a fraction of the column pitch (0.25 or 0.5).
      shrink -- 0.92..1.0, pulling the lattice off the walls so the relaxation has room to spread.
      jit    -- Gaussian jitter, sigma = f / max(rows, per) with f ~ 0.08..0.35. LOW jitter clearly won:
                the point is to break ties and let the relaxation choose, not to randomize the start.
    """
    rows = max(1, int(round(np.sqrt(n / 0.866))) + int(rng.randint(-2, 3)))
    per = int(np.ceil(n / float(rows)))
    offs = 0.25 if rng.rand() < 0.5 else 0.5
    i = np.arange(rows)[:, None]
    j = np.arange(per)[None, :]
    y = np.broadcast_to((i + 0.5) / rows + LO, (rows, per))
    x = (j + 0.5 + offs * (i % 2)) / per + LO
    pts = np.stack([x.ravel(), y.ravel()], axis=1)[:n] * float(rng.uniform(0.92, 1.0))
    if rng.rand() < 0.5:                                  # the transpose is a different lattice for n
        pts = pts[:, ::-1].copy()                         # whose rows and columns do not divide evenly
    jit = float(rng.uniform(0.08, 0.35)) / max(rows, per)
    return np.clip(pts + rng.normal(0.0, jit, size=(n, 2)), LO + 1e-3, HI - 1e-3)


def _init_centres(n, rng, mode):
    """Cold starts, drawn from `rng` only. `mode` 0/1/2 = hex lattice / jittered square grid / uniform.
    The caller weights these heavily towards hex (see `_MODE_CYCLE`); the other two are kept purely for
    basin diversity, since hex is a bad prior for the few small n whose optimum is not row-structured."""
    if mode == 0:
        return _hex_init(n, rng)
    if mode == 1:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / float(cols)))
        gx, gy = np.meshgrid((np.arange(cols) + 0.5) / cols + LO,
                             (np.arange(rows) + 0.5) / rows + LO)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)[:n]
        jit = 0.35 / max(cols, rows)
        return np.clip(pts + rng.normal(0.0, jit, size=(n, 2)), LO + 1e-3, HI - 1e-3)
    return rng.uniform(LO + 0.05, HI - 0.05, size=(n, 2))


def _trial(n, best, xy0, record_polish=False):
    """One penalty solve from `xy0`, scored through evaluate(). Returns the score.

    `record_polish` fixes a SELECTION BIAS measured in artifacts/exp_slop.py. The incumbent has had
    `offer_polished()` applied to it; a fresh candidate has not, and the scoring round trip
    (penalty solve -> _lp_radii -> repair) stops ~3e-6 (n=49) to ~5e-5 (n=57) short of the exact KKT
    point it is converging to. So the loop compares a POLISHED incumbent against UNPOLISHED
    challengers, and a candidate must be that much better STRUCTURALLY merely to tie. Evidence: 480
    perturbation trials over 4 move types at n=49/57 (artifacts/exp_cluster2.py) produced zero wins
    and every arm's ceiling sat at the SAME -3e-6 -- the signature of the round trip, not of
    structure; polishing those same candidates lifts 77 of 80 to within 5e-11 of the incumbent
    (artifacts/exp_slop.py). The bias is the size of the whole remaining gap at a near-record size.

    Polishing EVERY candidate would cost it: polish runs 0.22-0.30 s against the solve's 0.075-0.088 s,
    i.e. ~4x fewer trials per second. So only a candidate that sets a new raw record pays for one --
    records are the running max of a bounded sample, so they thin out as ~log(trials) (~2 s of a 52 s
    slice) while the challenger that matters, the best one, always gets polished before it is judged.
    """
    try:
        xy = _penalty_solve(n, xy0, np.full(n, 0.5 / np.sqrt(n)))
    except Exception:
        xy = xy0
    xy = np.clip(np.asarray(xy, float), LO, HI)
    s = best.offer(xy)
    if record_polish and np.isfinite(s) and s > best.raw_max:
        best.raw_max = s
        p = polish(xy, best_radii(xy))
        if p is not None:
            best.offer(np.asarray(p)[:, :2])
    return s


def _halving_round(n, best, rng, pool, k, deadline, batch=_BATCH, keep=_KEEP):
    """One SUCCESSIVE-HALVING batch of cold restarts. Returns the number of starts consumed.

    Why not just run every start to convergence: this solver is CPU-starved, not evaluation-starved
    (a whole in-run pass gives each size only a couple of seconds, i.e. a handful of full restarts),
    and most starts are decided long before the schedule ends. So every start gets only the cheap
    mu = 10, 100 stages, is scored, and only the best `keep` of the batch pay for mu = 1e3..1e6.

    Nothing is thrown away and nothing is risked by the cut: each partial result is itself offered
    through evaluate() (repaired, hence strictly feasible), so it is ranked and kept for free -- the
    LP score used to rank it is the same number evaluate() returns. Measured against a matched-CPU
    full-schedule multistart (artifacts/exp_arms.py, n = 33/49/63/95 x 2 seeds): +0.086 mean digits
    at 3 s/size and +0.246 at 10 s/size, with no size losing at either budget.

    The deadline is checked before each start, so a batch truncated by the CPU cap still leaves every
    candidate it began scored. The top candidate is always given its high-mu stages even if the
    deadline has just passed -- a start stopped at mu = 100 is a genuinely worse packing, and the
    overrun is bounded by one restart.
    """
    zs, sc = [], []
    r0 = np.full(n, 0.5 / np.sqrt(n))
    for _ in range(max(1, int(batch))):
        if best.meter.left() <= 0 or (zs and time.process_time() >= deadline):
            break
        x0 = _cold_init(n, rng, pool, k)
        k += 1
        try:
            z = _penalty_stages(n, _z0(n, x0, r0), _MU_SCHEDULE[:_CUT])
        except Exception:
            continue
        zs.append(z)
        sc.append(best.offer(z[:2 * n].reshape(n, 2)))
    if not zs:
        return 0
    for rank, i in enumerate(np.argsort(-np.asarray(sc, float))[:max(1, int(keep))]):
        if best.meter.left() <= 0:
            break
        if rank > 0 and time.process_time() >= deadline:
            break
        try:
            z = _penalty_stages(n, zs[int(i)], _MU_SCHEDULE[_CUT:])
        except Exception:
            continue
        best.offer(z[:2 * n].reshape(n, 2))
    return len(zs)


def _cold_init(n, rng, pool, k):
    """One cold start, alternating structural TRANSFER from a neighbour size with a hex/grid/uniform
    lattice. Transfer is tried on every other restart and silently degrades to `_MODE_CYCLE` when no
    neighbour packing exists, so a size with no census around it loses nothing."""
    if pool is not None and k % 2 == 1:
        x0 = _transfer_init(n, rng, pool)
        if x0 is not None:
            return x0
    return _init_centres(n, rng, _MODE_CYCLE[(k // 2) % len(_MODE_CYCLE)])


def _solve_one(n, deadline, evaluate, meter, rng, pool=None, restart_frac=0.45, batch=True,
               probe=None):
    """Search n until `deadline` (process CPU). Returns the incumbent Sigma r.

    `batch` selects successive halving for the cold-restart phase. It is turned OFF for the pass-1
    sweep, whose slice is far too short for a batch: there the job is to get ONE decent feasible
    packing per target on the board quickly, not to explore basins efficiently.

    Two ways this returns before `deadline`, both of which hand the unused seconds back to the
    caller (`solve` gives them to the next size):

    * `probe` -- a CPU time by which the size must have beaten its own committed packing. Measured
      (see `_DEPTH_PROBE_S`), a size that can move shows it within ~14 s and a size that cannot
      produces nothing in 52 s, so screening is worth far more than persisting. A size with no
      committed packing has warm_sum = -inf and can never be abandoned this way.
    * the DIGIT CAP -- `verify.py` clamps a size's score at `_DIGITS_CAP` digits, i.e. at a relative
      gap of 1e-7, so once the incumbent is that close to the record every further second here is
      worth exactly zero to the census and is owed to a size that still has digits left. Silently
      disabled (never fires) when records.json is unreadable.
    """
    best = _Best(n, evaluate, meter)
    R = _record(n)
    cap_sum = R * (1.0 - 10.0 ** -_DIGITS_CAP) if R is not None else np.inf

    warm = read_pack(n)
    if warm is not None:
        best.offer(warm[:, :2], warm[:, 2])          # never lose the committed packing
        best.offer(warm[:, :2])                      # ... and try LP-optimal radii on the same centres
        best.offer_polished()                        # ... and an exact-constraint polish of it
        if time.process_time() < deadline:
            _trial(n, best, warm[:, :2])             # ... and a penalty polish of it

    split = time.process_time() + restart_frac * max(0.0, deadline - time.process_time())
    warm_sum = best.sum_r                    # -inf when n has no committed packing (a cold size)
    cold_live = True                         # are fresh restarts still competitive with it?
    k, rounds = 0, 0
    while time.process_time() < split and meter.left() > 0 and best.sum_r < cap_sum:
        if batch:
            best.seen_max = -np.inf
            k += max(1, _halving_round(n, best, rng, pool, k, split))
            rounds += 1
            # Stop exploring only after a real probe, and only when the probe FAILED: every batch so
            # far came back below the packing we already had. A cold size has warm_sum = -inf, so this
            # never fires there and the exploration phase is untouched.
            if rounds >= _COLD_PROBE_ROUNDS and best.seen_max < warm_sum:
                cold_live = False
                break
        else:
            _trial(n, best, _cold_init(n, rng, pool, k))
            k += 1

    while time.process_time() < deadline and meter.left() > 0 and best.sum_r < cap_sum:
        if probe is not None and time.process_time() >= probe and best.sum_r <= warm_sum:
            break                                # screened out: it has not moved, the next size gets
                                                 # these seconds (warm_sum is -inf for a cold size)
        if best.xy is None:
            _trial(n, best, _init_centres(n, rng, 0))
            continue
        # keep sampling fresh basins, but at a rate that reflects whether they are still paying:
        # _PCOLD_DEAD, not _PCOLD_LIVE, once the probe showed restarts cannot match the warm start.
        if rng.rand() < (_PCOLD_LIVE if cold_live else _PCOLD_DEAD):
            _trial(n, best, _cold_init(n, rng, pool, int(rng.randint(1 << 20))))
            continue
        if best.r is not None and rng.rand() < _RESEAT_P:
            _trial(n, best, _reseat_move(best.xy, best.r, rng), record_polish=True)
            continue
        _trial(n, best, np.clip(best.xy + rng.normal(0.0, _kick_scale(n, best.r, rng),
                                                     size=(n, 2)), LO + 1e-3, HI - 1e-3),
               record_polish=True)
    best.offer_polished()                            # final exact-constraint polish of the incumbent
    if pool is not None and best.xy is not None:
        # feed this run's own result back, so later sizes transfer from it rather than from disk
        pool[n] = (best.xy.copy(), repair(best.xy, best_radii(best.xy)))
    return best.sum_r


def _relgap(n):
    """Relative gap of the COMMITTED packing for n against the published record, or None when that
    cannot be determined (no records.json -- the offline re-run's workspace may not have one -- no
    entry for n, no committed pack). 0.0 means the committed packing already meets/exceeds the record,
    so further CPU on n cannot raise the score by even one digit. Reading bench/records.json is
    explicitly allowed by MISSION.md; every failure path returns None, i.e. "no information".
    """
    try:
        p = os.path.join("bench", "records.json")
        if not os.path.exists(p):
            return None
        with open(p, "r") as fh:
            rec = json.load(fh)
        rec = rec.get("records", rec)
        if str(n) not in rec:
            return None
        a = read_pack(n)
        if a is None:
            return None
        R = float(rec[str(n)])
        if not (R > 0.0):
            return None
        return max(0.0, (R - float(a[:, 2].sum())) / R)
    except Exception:
        return None


_DIGITS_CAP = 7.0                    # verify.py clamps -log10(relgap) here: relgap <= 1e-7 is full marks
_RECORD_CACHE = {}


def _record(n):
    """Published record for n, or None when it cannot be determined (no bench/records.json -- the
    offline re-run's workspace may not have one -- or no entry for n). Cached: records.json is
    static, and this is read once per size per depth slice."""
    if n in _RECORD_CACHE:
        return _RECORD_CACHE[n]
    val = None
    try:
        p = os.path.join("bench", "records.json")
        if os.path.exists(p):
            with open(p, "r") as fh:
                rec = json.load(fh)
            rec = rec.get("records", rec)
            if str(n) in rec and float(rec[str(n)]) > 0.0:
                val = float(rec[str(n)])
    except Exception:
        val = None
    _RECORD_CACHE[n] = val
    return val


def _deficit(n):
    """DIGITS still available at n: `_DIGITS_CAP` minus what the committed packing already scores.

    This -- not the raw relative gap -- is the quantity the scorer pays for, and the two rank sizes
    very differently. The score is -log10(relgap), so gap magnitude enters only logarithmically: the
    widest gap in the census (n=49, 1.1e-3) is 5x the narrowest lagging one (n=39, 2.9e-4) but is
    worth just 1.2x as many digits. Weighting by gap therefore skews the depth budget ~5x towards a
    size that is worth 1.2x more, and measured (artifacts/exp_curve_*.log) gap does not predict
    whether a size can be moved at all. None = no information (a fresh n, no record, no pack), which
    callers treat as maximally needy.
    """
    g = _relgap(n)
    if g is None:
        return None
    if g <= 0.0:
        return 0.0
    return max(0.0, _DIGITS_CAP - min(_DIGITS_CAP, -np.log10(g)))


# THE DEPTH PHASE IS A SCREEN OVER SIZES, NOT A PROPORTIONAL SPLIT.
#
# Measured this iteration (artifacts/exp_full.py, artifacts/exp_curve_*.log): ten lagging census
# sizes, each given ONE single-size 52 s call, two seeds apiece. The result is a clean dichotomy --
#
#     moved: 37 (1/1 seeds), 39 (1/2), 41 (2/2), 53 (2/2), 61 (2/2)   -> 8 of 9 runs improved
#     dead : 43, 45, 49, 57, 73                                       -> 0 of 10 runs improved,
#            not by one part in 1e9, in 52 s each
#
# and four of the movers (39, 41, 53, 61) ran all the way to the 7-digit cap. Two things follow.
#
# (1) The gap does NOT say which. The live set's gaps (2.9e-4 .. 1.0e-3) and the dead set's
#     (4.4e-4 .. 1.1e-3) are the same range; n=37 and n=49 are both ~1e-3 and one moves, one does
#     not. The old rule sampled proportional to the gap, which spends ~half the budget on the five
#     widest -- 49, 37, 81, 57, 41 -- of which four are dead. Sizes are now ordered by DIGIT
#     DEFICIT (`_deficit`), the scorer's own units, which is nearly uniform over the lagging census
#     and so stops pretending the gap is evidence it is not.
#
# (2) Whether a size can move shows up FAST or not at all. First-improvement CPU times at the live
#     sizes were 6.8, 8.4, 8.6, 11.1, 12.7, 13.6, 23.8, 34.9 s; the dead sizes produced no
#     improvement in 52 s. So the depth budget is spent like the restart phase already spends its
#     own: PROBE, then commit. Every size gets `_DEPTH_PROBE_S` (scaled by n, since a restart costs
#     ~n^2 and the old rule weighted by n for the same reason); a size that has not beaten its own
#     committed packing by then is abandoned and the clock moves to the next size, while one that
#     has keeps the CPU up to `_CPU_PER_SIZE_S`. How many sizes a pass funds is now emergent rather
#     than a constant `_FOCUS_K` guessed in advance.
#
# A size with NO committed packing has warm_sum = -inf, so it can never be abandoned by the probe --
# the offline held-out re-run, which is one size per call, is untouched by all of this.
_DEPTH_PROBE_S = 18.0
_PROBE_REF_N = 50.0


def _probe_s(n):
    """CPU seconds a size gets to show it can move, scaled by n (a restart costs ~n^2)."""
    return _DEPTH_PROBE_S * float(n) / _PROBE_REF_N


def _depth_order(targets, rng=None):
    """The order in which pass 2 screens `targets`; a size already at the digit cap is dropped.

    Three rules:
      * a size whose committed packing already scores the full `_DIGITS_CAP` gets nothing: no amount
        of CPU there can raise the SCORE by so much as a thousandth of a digit, and that CPU is
        worth strictly more on a size that still has digits left;
      * a size with NO gap information (no record entry, no committed pack -- a fresh n, the offline
        held-out re-run) is maximally needy and goes first: it may have no packing at all;
      * the rest are drawn WITHOUT REPLACEMENT with probability proportional to their digit deficit
        (see `_deficit`). With `rng=None` the order is deterministic (deficit-descending), so the
        rule stays testable without randomness. Sampling rather than sorting matters because the
        census is monotone and the key is what our own work reduces: a fixed order re-spends the
        budget on the same sizes every iteration, and a size stays near the top of it precisely by
        refusing to move (iteration 11's lesson, kept).
    """
    targets = list(targets)
    defs = {n: _deficit(n) for n in targets}
    unknown = [n for n in targets if defs[n] is None]
    known = [n for n in targets if defs[n] is not None and defs[n] > 0.0]
    known.sort(key=lambda n: -defs[n])
    if rng is not None and len(known) > 1:
        w = np.array([defs[n] for n in known], dtype=float)
        if w.sum() > 0.0:
            known = [int(n) for n in rng.choice(known, size=len(known), replace=False,
                                                p=w / w.sum())]
        unknown = [int(n) for n in rng.permutation(unknown)] if unknown else unknown
    return list(unknown) + list(known)


def solve(evaluate, meter, rng, targets):
    """Drive the search across `targets` under a CPU allowance derived AT ENTRY (so repeated calls in
    one process each get a fresh allowance) and from `len(targets)` (so a one-size offline re-run gets
    the full per-size allowance instead of 1/37th of it)."""
    targets = sorted(set(int(t) for t in targets))
    if not targets:
        return
    t0 = time.process_time()
    deadline = t0 + min(_CPU_TOTAL_CAP_S, _CPU_PER_SIZE_S * len(targets))
    pool = {}                                   # size -> (centres, radii) known packings, disk + ours

    # Pass 1 -- insurance, sized to the targets that actually need it. Its job is to get a feasible
    # packing routed through evaluate() for every target BEFORE the CPU backstop can fire. A target
    # that already has a committed packing needs no search for that: `_solve_one` re-offers it for
    # ~zero CPU, and a target pass 2 never reaches keeps its committed .pck (the driver restores
    # bench/packs/ and then writes only the bests it tracked). A target with NO pack -- a fresh n,
    # the offline held-out re-run -- has nothing to fall back on, so it still gets the full sweep
    # slice. So the sweep is given `_SWEEP_FRAC` of the budget scaled by the FRACTION of targets that
    # are cold, and spent only on those: 0 s for the 37-size census, the same 12% as before for a
    # one-size call on an n this loop never scored.
    cold = [n for n in targets if read_pack(n) is None]
    sweep_end = t0 + _SWEEP_FRAC * (deadline - t0) * len(cold) / float(len(targets))
    for i, n in enumerate(cold):
        if meter.left() <= 0:
            return
        left = len(cold) - i
        stop = min(sweep_end, time.process_time() + max(0.0, sweep_end - time.process_time()) / left)
        _solve_one(n, stop, evaluate, meter, rng, pool=pool, restart_frac=1.0, batch=False)

    # Pass 1b -- one evaluation each, for insurance. The depth screen below deliberately never
    # reaches a size that is already at the digit cap, and the driver is documented to restore
    # bench/packs/ and then write only the bests it tracked -- so such a size keeps its committed
    # packing either way. Re-offering it costs one evaluation and ~0 CPU, and makes that guarantee
    # depend on this solver rather than on my reading of the driver.
    for n in targets:
        if meter.left() <= 0:
            return
        w = read_pack(n)
        if w is not None:
            _Best(n, evaluate, meter).offer(w[:, :2], w[:, 2])

    # Pass 2 -- depth, run as a SCREEN over sizes (see `_depth_order`). Every size still holding
    # digits is put in a sampled order; each gets a probe of `_probe_s(n)` CPU seconds to beat its
    # own committed packing, keeps the CPU up to `_CPU_PER_SIZE_S` if it does, and is dropped
    # immediately if it does not. Both early exits return the unused seconds here, so the number of
    # sizes a pass reaches is emergent: a census of dead sizes is screened quickly, a live one is
    # worked deeply. Sizes never reached lose nothing -- the driver restores bench/packs/ and writes
    # only tracked improvements, so the census is monotone either way.
    order = _depth_order(targets, rng)
    if not order:                               # every target already at the digit cap -- share the
        order = list(targets)                   # remaining time evenly rather than waste it
    for n in order:
        if meter.left() <= 0:
            return
        now = time.process_time()
        if now >= deadline:
            return
        stop = min(deadline, now + _CPU_PER_SIZE_S)
        _solve_one(n, stop, evaluate, meter, rng, pool=pool,
                   probe=min(stop, now + _probe_s(n)))


# ------------------------------------------------------------------------------------------ self-test
def _self_test():
    import numpy as _np

    class _Meter(object):
        def __init__(self, b):
            self.budget, self.used = b, 0

        def left(self):
            return max(0, self.budget - self.used)

        def tick(self, k=1):
            g = max(0, min(int(k), self.budget - self.used))
            self.used += g
            return g

    def make_eval(meter, seen):
        def evaluate(n, packing):
            a = _np.asarray(packing, float)
            single = (a.ndim == 2)
            if single:
                a = a[None]
            if a.shape[2] != 3 or a.shape[1] != n:
                raise ValueError("bad packing shape %r for n=%d" % (a.shape, n))
            B = a.shape[0]
            g = meter.tick(B)
            x, y, r = a[..., 0], a[..., 1], a[..., 2]
            wall = _np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min(1)
            d = a[:, :, None, :2] - a[:, None, :, :2]
            dist = _np.sqrt((d * d).sum(-1))
            di = _np.arange(n)
            dist[:, di, di] = _np.inf
            pair = (dist - (r[:, :, None] + r[:, None, :])).min((1, 2))
            feas = (r.min(1) > 0) & (wall >= -1e-9) & (pair >= -1e-9)
            s = r.sum(1)
            if g < B:
                feas[g:] = False
                s[g:] = -_np.inf
            for i in range(min(g, B)):
                if feas[i] and s[i] > seen.get(n, (-_np.inf,))[0]:
                    seen[n] = (float(s[i]), a[i].copy())
            return (bool(feas[0]), float(s[0])) if single else (feas, s)
        return evaluate

    # 1. repair() always yields a strictly feasible packing, even from absurd radii.
    rs = _np.random.RandomState(7)
    for n in (1, 2, 5, 31):
        xy = rs.uniform(LO, HI, size=(n, 2))
        r = repair(xy, rs.uniform(0.0, 5.0, size=n))
        assert (r > 0).all(), "repair produced a non-positive radius"
        assert (walls(xy) - r).min() >= -1e-9, "repair violated a wall"
        if n >= 2:
            d = _np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
            _np.fill_diagonal(d, _np.inf)
            assert (d - (r[:, None] + r[None, :])).min() >= -1e-9, "repair left an overlap"
    # 2. the LP beats the equal-radius safe rule it replaces (the whole point of this solver).
    xy = rs.uniform(LO + 0.05, HI - 0.05, size=(40, 2))
    assert best_radii(xy).sum() > _grow(xy).sum(), "LP radii should dominate the safe rule"
    # 3. read_pack must be safe for an n with no committed pack.
    assert read_pack(10 ** 6) is None, "read_pack must return None for a missing size"
    # 3b. cross-size transfer: exact circle count both ways, and a graceful None far from the census.
    pool = {}
    for n in (28, 44, 100):
        x0 = _transfer_init(n, rs, pool)
        assert x0 is not None and x0.shape == (n, 2), "transfer start must have exactly n centres"
        assert _np.isfinite(x0).all() and (x0 >= LO).all() and (x0 <= HI).all(), "transfer out of box"
    assert _transfer_init(10 ** 6, rs, pool) is None, "transfer must return None with no neighbour"
    hx, hr = _greedy_holes(_np.zeros((1, 2)), _np.array([0.1]), 3, rs)
    assert hx.shape == (4, 2) and hr.shape == (4,) and (hr > 0).all(), "_greedy_holes count/radii"
    # 3c. _cold_init must always return n centres -- both on its lattice restarts and on its transfer
    #     restarts, including n=5, which is far enough from the census that no neighbour pack exists.
    for n in (5, 33):
        for k in (0, 1):
            assert _cold_init(n, rs, {}, k).shape == (n, 2), "_cold_init must return n centres"
    # 3d. one successive-halving round must score every start it begins (nothing is dropped by the
    #     cut) and must respect an already-expired deadline by still finishing exactly one candidate.
    class _B(object):
        def __init__(self, n, ev, m):
            self.n, self.evaluate, self.meter, self.calls = n, ev, m, 0
            self.xy, self.sum_r, self.seen_max = None, -_np.inf, -_np.inf
            self.r, self.raw_max = None, -_np.inf
        offer = _Best.offer
        offer_polished = _Best.offer_polished

    for dl, lo, hi in ((time.process_time() + 60.0, 8, 10), (0.0, 1, 3)):
        meter = _Meter(500)
        seen = {}
        b = _B(21, make_eval(meter, seen), meter)
        got = _halving_round(21, b, rs, {}, 0, dl)
        if not (lo <= meter.used <= hi and b.sum_r > 0 and 21 in seen):
            print("FAIL _halving_round: starts=%d evals=%d best=%s" % (got, meter.used, b.sum_r))
            raise AssertionError("_halving_round")

    # 3d2. `polish` returns exactly n centres inside the box, or None -- never a malformed packing.
    for n in (2, 17):
        xy = rs.uniform(LO + 0.1, HI - 0.1, size=(n, 2))
        got = polish(xy, best_radii(xy))
        assert got is not None and got.shape == (n, 2), "polish must return n centres"
        assert _np.isfinite(got).all() and (got >= LO).all() and (got <= HI).all(), "polish out of box"
    assert polish(_np.zeros((1, 2)), _np.array([0.4])) is None, "polish must decline n<2"
    #      and on a committed near-record pack it must not LOSE: offer_polished only keeps improvements.
    warm31 = read_pack(31)
    if warm31 is not None:
        meter = _Meter(50)
        seen = {}
        b = _B(31, make_eval(meter, seen), meter)
        b.offer(warm31[:, :2], warm31[:, 2])
        base = b.sum_r
        b.offer_polished()
        assert b.sum_r >= base, "offer_polished must never lower the incumbent"
        print("  polish n=31: %.9f -> %.9f" % (base, b.sum_r))

    # 3c2. the RECORD POLISH in _trial (artifacts/exp_slop.py): a raw-record challenger must pay for
    #      exactly one polish and must never lower the incumbent, the polish must fire at most once
    #      per new raw record (it is the cost control -- polish is ~3x a solve), and record_polish
    #      must default OFF so the cold-restart path is untouched.
    meter = _Meter(400)
    seen = {}
    b = _B(20, make_eval(meter, seen), meter)
    rs = _np.random.RandomState(4)
    start = _init_centres(20, rs, 0)
    _trial(20, b, start)
    assert b.raw_max == -_np.inf, "record_polish must default OFF"
    used0 = meter.used
    s1 = _trial(20, b, start, record_polish=True)
    assert b.raw_max == s1 and _np.isfinite(s1), "a first challenger always sets the raw record"
    assert 1 <= meter.used - used0 <= 2, "a record challenger costs one solve plus at most one polish"
    inc = b.sum_r
    used1 = meter.used
    _trial(20, b, start, record_polish=True)          # same start -> same score, NOT a new record
    assert meter.used - used1 == 1, "a non-record challenger must not pay for a polish"
    assert b.sum_r >= inc, "the record polish must never lower the incumbent"
    print("  record polish n=20: raw_max=%.9f incumbent=%.9f" % (b.raw_max, b.sum_r))

    # 3d3. the depth SCREEN's order: a size already at the digit cap is dropped entirely, a size
    #      with no gap information is always first (it may have no packing at all), and with
    #      rng=None the order is deterministic and sorted by digit deficit.
    census = list(range(27, 100, 2))
    order = _depth_order(census)
    assert len(set(order)) == len(order), "the depth order must not repeat a size"
    for n in census:
        d = _deficit(n)
        if d == 0.0:
            assert n not in order, "a size at the digit cap must get no depth CPU"
        elif d is not None and d > 0.0:
            assert n in order, "a size with digits left must be in the depth order"
    ds = [_deficit(n) for n in order]
    assert all(a >= b - 1e-12 for a, b in zip(ds, ds[1:])), "rng=None must sort by digit deficit"
    print("  depth order (deterministic, first 6): %s" % order[:6])
    mixed = [44, 10 ** 6] + census[:3]          # two sizes with no pack and no record entry
    om = _depth_order(mixed)
    assert om[:2] == [44, 10 ** 6] or set(om[:2]) == {44, 10 ** 6}, \
        "an unknown-gap size must be screened first"
    #      _deficit is the SCORER's units, not the raw gap: a size 10x closer to the record is worth
    #      exactly one digit more, not 10x more.
    assert _deficit(10 ** 6) is None, "no record entry -> no gap information"

    # 3d4. the order is SAMPLED: it still drops capped sizes, still contains every size with digits
    #      left, and genuinely varies across seeds (a size that refuses to move must not lock out
    #      the rest, iteration 11's lesson), while staying tilted towards the widest deficits.
    firsts, tilt = [], 0
    lagging = [n for n in census if (_deficit(n) or 0.0) > 0.0]
    top = set(sorted(lagging, key=lambda n: -(_deficit(n) or 0.0))[:5])
    for sd in range(8):
        o = _depth_order(census, _np.random.RandomState(sd))
        assert sorted(o) == sorted(lagging), "the sampled order must cover exactly the lagging sizes"
        assert all(_deficit(n) != 0.0 for n in o), "a capped size must never be sampled"
        firsts.append(tuple(o[:3]))
        tilt += len(set(o[:5]) & top)
    if len(lagging) > 5:
        assert len(set(firsts)) > 1, "the sampled depth order must vary across seeds"
        print("  sampled order: %d distinct heads over 8 seeds, %d/40 of the first-5 in the top-5"
              % (len(set(firsts)), tilt))

    # 3d5. the depth probe scales with n and a size that cannot move is ABANDONED, handing its
    #      seconds back: a probe deadline already in the past must end the search almost at once,
    #      while the same call with probe=None runs to its deadline.
    assert _probe_s(100) > _probe_s(27) > 0.0, "the probe must scale with n"
    if read_pack(31) is not None:
        meter = _Meter(4000)
        seen = {}
        t = time.process_time()
        _solve_one(31, time.process_time() + 6.0, make_eval(meter, seen), meter,
                   _np.random.RandomState(5), probe=0.0)
        el = time.process_time() - t
        assert 31 in seen, "an abandoned size must still have its warm packing on the board"
        print("  abandon-on-probe: %.2fs of a 6.0s slice used" % el)

    # 9. the depth kick must be SCALE-FREE: the same multiple of the mean radius at every n, always
    #    inside the measured productive window [0, 0.35]*rbar (artifacts/exp_kick.py), and it must
    #    survive having no radii yet (r=None, before the first feasible candidate).
    rs9 = _np.random.RandomState(4)
    for n9 in (27, 49, 99):
        rb = 0.5 / _np.sqrt(n9) * 1.1
        rr = _np.full(n9, rb)
        sc = _np.array([_kick_scale(n9, rr, rs9) for _ in range(400)]) / rb
        assert sc.min() >= _KICK_LO - 1e-12 and sc.max() <= _KICK_HI + 1e-12, \
            "kick escaped [%g,%g]*rbar at n=%d" % (_KICK_LO, _KICK_HI, n9)
        assert _KICK_HI <= 0.35 + 1e-12, "kick window widened past the measured productive range"
        assert _kick_scale(n9, None, rs9) > 0.0, "kick must work before any radii exist"
        assert _kick_scale(n9, _np.zeros(n9), rs9) > 0.0, "kick must survive degenerate radii"
    # ... and it must NOT be the absolute rule it replaced: the n=27 and n=99 kicks differ in scale.
    rs9 = _np.random.RandomState(4)
    a27 = _np.mean([_kick_scale(27, _np.full(27, 0.5 / _np.sqrt(27)), rs9) for _ in range(400)])
    rs9 = _np.random.RandomState(4)
    a99 = _np.mean([_kick_scale(99, _np.full(99, 0.5 / _np.sqrt(99)), rs9) for _ in range(400)])
    assert a27 > 1.5 * a99, "kick should shrink with n (it is a multiple of the mean radius)"
    print("  kick scale/rbar in [%.2f,%.2f]; mean sigma n=27 %.4f vs n=99 %.4f" % (_KICK_LO, _KICK_HI, a27, a99))

    # 3e. the pass-1 sweep is sized by the COLD targets only: a call whose targets all have committed
    #     packings must still put every one of them on the board (via the warm offer in `_solve_one`),
    #     and a call on a size with no pack must still get its full sweep slice.
    meter = _Meter(4000)
    seen = {}
    warm_targets = [n for n in (27, 29, 31) if read_pack(n) is not None]
    if warm_targets:
        solve(make_eval(meter, seen), meter, _np.random.RandomState(3), warm_targets)
        assert all(n in seen for n in warm_targets), "an all-warm call must score every target"

    # 4. solve() TWICE in one process, second call with len(targets) == 1 -- the deadline-at-entry and
    #    per-call-budget-split contracts a single call cannot catch.
    ok = True
    for call, tg in enumerate(([29, 44], [37])):
        meter = _Meter(150)
        seen = {}
        t = time.process_time()
        solve(make_eval(meter, seen), meter, _np.random.RandomState(call), tg)
        el = time.process_time() - t
        for n in tg:
            if n not in seen:
                ok = False
                print("FAIL call %d: no feasible packing for n=%d" % (call, n))
            else:
                print("  call %d n=%-3d sum_r=%.6f  (%d evals, %.1fs cpu)"
                      % (call, n, seen[n][0], meter.used, el))
        if el > _CPU_PER_SIZE_S * len(tg) + 5.0:
            ok = False
            print("FAIL call %d: overran its CPU allowance (%.1fs)" % (call, el))
    # `seen_max` must track what was SHOWN, not only what won -- the explore/exploit probe reads it
    # to decide whether cold restarts are still competitive, so a losing candidate must move it.
    meter = _Meter(50)
    seen = {}
    b = _Best(9, make_eval(meter, seen), meter)
    if b.seen_max != -_np.inf:
        ok = False
        print("FAIL seen_max not initialised to -inf")
    b.offer(_init_centres(9, _np.random.RandomState(3), 0))
    good = b.sum_r
    if not (b.seen_max == good > 0.0):
        ok = False
        print("FAIL seen_max did not follow a winning offer")
    b.seen_max = -_np.inf
    b.offer(_np.zeros((9, 2)))                   # all centres coincident: feasible, but far worse
    if not (-_np.inf < b.seen_max < b.sum_r == good):
        ok = False
        print("FAIL a losing offer must move seen_max and not the incumbent (%r, %r)"
              % (b.seen_max, b.sum_r))
    # The cold probe must stay a REAL probe: at least one full halving batch has to be run and
    # scored before exploration can be declared dead, or a warm size would abandon restarts having
    # looked at nothing. (Iteration 14 cut this 2 -> 1; 0 would disable the phase outright.)
    if not _COLD_PROBE_ROUNDS >= 1:
        ok = False
        print("FAIL _COLD_PROBE_ROUNDS must be >= 1, got %r" % (_COLD_PROBE_ROUNDS,))

    print("SELF-TEST: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
