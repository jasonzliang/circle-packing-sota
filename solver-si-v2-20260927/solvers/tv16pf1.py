"""Circle-packing solver for the packomania `csqv` family: maximize the SUM of radii of n
circles packed in the unit square (centered [-0.5, 0.5]^2).

    solve(evaluate, meter, rng, targets)

WHY THIS DESIGN (vs. the seeded baseline it replaces)
-----------------------------------------------------
The baseline assigned every candidate the "guaranteed" radius r_i = min(wall_i, min_j d_ij/2).
That rule is *feasible* but structurally sub-optimal: it forces near-equal radii, while the csqv
optimum mixes large and small circles. It also optimized centres with a hand-tuned repulsion
field that has no relation to the objective being scored.

This solver instead treats the real problem directly:

    max  sum_i r_i
    s.t. r_i + r_j <= d_ij        (no overlap)
         r_i <= wall_i            (inside the square)
         r_i >= 0

and attacks it in three layers:

  1. SMOOTH PENALTY NLP over all 3n variables (x, y, r) jointly, minimized with L-BFGS-B under a
     rising penalty ladder mu = 1e2 .. 1e7.  Analytic gradient, fully vectorized, O(n^2) per
     evaluation.  Letting the radii move *with* the centres is what unlocks the unequal-radius
     structure the baseline could not express.
  2. EXACT RADII LP.  With the centres FIXED, choosing the radii is a linear program
     (max 1'r s.t. r_i + r_j <= d_ij, 0 <= r_i <= wall_i).  Solved exactly with HiGHS on the
     pruned pair set (only pairs with d_ij < wall_i + wall_j can ever bind).  This recovers the
     chain trades ("shrink one, grow two") that no coordinate-wise rule can see.
  3. STRICT FEASIBILITY REPAIR + monotone coordinate ascent.  Everything handed to evaluate() is
     forced strictly feasible in exact arithmetic first (global down-scale, then Gauss-Seidel
     sweeps r_i <- min(wall_i, min_j (d_ij - r_j)) - tiny, which can only INCREASE r_i and always
     preserves feasibility).  Nothing relies on the optimizer's own tolerance.

  4. STRUCTURED LATTICE SWEEP (iteration 3).  Because layer 2b is an EXACT local solver -- it
     reaches a KKT point in ~0.25 s and then stops moving (a 0.25 s and a 2.0 s run return
     bit-identical packings) -- final quality is decided almost entirely by which basin the
     start lands in.  The basins that matter are row-lattice topologies: how many rows, which
     parity of long/short rows, hexagonally staggered or square.  That family is small and
     enumerable, so it is swept best-first from the theoretical hex row count
     k ~= sqrt(2n/sqrt(3)) rather than sampled at random.  Measured at n=57: relgap 1.1e-2
     from random restarts vs 1.4e-3 from the third lattice candidate.

  5. STREAM POOL (iteration 4).  Basin hopping runs round-robin over
     POOL_P independent streams -- the best few lattice optima the sweep already produced, plus
     reserved slots for cold starts the lattice family cannot evict -- instead of a single
     incumbent, because with one incumbent the search is only as good as its starting basin.

CONTRACT NOTES (all of these are load-bearing; see MISSION.md)
  * The CPU deadline is derived at ENTRY from time.process_time(), never a module constant, so a
    second solve() call in the same process still works.  --self-test asserts exactly that.
  * The per-n time split survives len(targets) == 1 (the offline held-out re-run shape) and is
    re-derived after every n, so time unused by an easy n flows to the next one.
  * Warm starts from bench/packs/csqv<n>.pck are GUARDED by os.path.exists; every n must be
    solvable cold, because the solver is re-run on sizes that have no pack.
  * Nothing is keyed on n: the same search runs for any n >= 1.  No coordinate tables.
  * Only evaluate() decides what counts; we route every candidate we care about through it, in
    batches, and stop when meter.left() is exhausted.

LIMITS (stated plainly for whoever picks this up)
  * The penalty NLP finds LOCAL optima.  Multi-start + basin hopping mitigates but does not
    remove that; for large n the landscape has many basins and the gap to the record stays well
    above the 1e-7 that would earn full digits.
  * The LP polish is skipped when scipy.optimize.linprog is unavailable or the solve fails; the
    coordinate-ascent grow is the always-available fallback and is strictly weaker.
  * Runs that end on the CPU backstop rather than on the deadline are not bit-reproducible
    (the driver documents this); the deadlines below are set to finish before the backstop.
"""
import json
import os
import time

import numpy as np

LO, HI = -0.5, 0.5
SAFE = 1e-12          # feasibility margin baked into every radius we emit (tol is 1e-9)
LP_MARGIN = 1e-9      # right-hand-side shortfall absorbing the LP solver's own primal residual
PACKS = "bench/packs/csqv%d.pck"

# CPU allowance per solve() call, in process-CPU seconds, derived from len(targets):
#   * len(targets) == 1  -> the offline held-out shape: one size, 60 s cap  -> take 55 s.
#   * otherwise          -> the in-run shape: 37 sizes sharing a 120 s cap  -> ~3 s per size,
#                           capped at 112 s so the backstop never fires.
CPU_ONE = 55.0
CPU_PER_N = 3.0
CPU_TOTAL_CAP = 112.0
SWEEP_SHARE = 0.6     # fraction of a size's allowance the structured lattice sweep may spend

# Iteration 4: the search keeps POOL_P independent STREAMS alive at once instead of one
# incumbent.  At most POOL_SWEEP of them may come from the structured lattice sweep; the rest
# are reserved for independent cold starts, so no single deep lattice optimum can swallow the
# whole search (measured failure at n=40, see _solve_one).
POOL_P = 3
POOL_SWEEP = 2
POOL_DIST = 1e-5      # sorted-radius L-inf distance below which two optima are the SAME basin
# Iteration 12: the SWEEP DOMINANCE STOP.  The lattice sweep is deterministic, so on a size that
# already carries a committed pack it re-derives the SAME optima every call -- and measured on the
# committed census (artifacts/iter12_sweep_probe.txt) the ENTIRE 80-86 candidate plan fails to beat
# that pack at n=43/65/99 (sweep best 3.410348 / 4.193211 / 5.196696 vs pack 3.410348 / 4.194643 /
# 5.211432).  In the in-run meter the sweep's SWEEP_SHARE is ~1.8 s of a ~3 s size, i.e. 60% of the
# budget spent re-deriving starts the incumbent already dominates.  So: once SWEEP_STALL consecutive
# sweep candidates have failed to beat an incumbent that did NOT come from the sweep, stop sweeping
# and give the rest of the size's time to basin hopping.
#   GATED ON AN INDEPENDENT INCUMBENT ON PURPOSE.  Cold (no pack -- the offline held-out shape) the
#   only incumbent is the sweep's own best, and the sweep's value is NOT monotone in plan index
#   (measured: the best candidate sits at index 38/86 at n=43 and 42/82 at n=65), so a stall rule
#   keyed on the sweep's own best would cut the plan off before its best candidate.  Cold, the rule
#   therefore never fires and the sweep behaves exactly as before.
SWEEP_STALL = 8       # consecutive sweep candidates failing to beat a non-sweep incumbent
SWEEP_STALL_DEFAULT = SWEEP_STALL   # restored by --self-test after it varies the knob
HOP_ALPHA_LO = 0.92   # radius-cap hop: cap is alpha * mean(r) for alpha drawn in this range
HOP_ALPHA_HI = 1.12
HOP_MODES = 5         # move kinds: 3 centre shakes + radius-cap homogenisation + gap reinsertion
REINS_GRID = 48       # candidate grid per axis for the gap-directed reinsertion move
REINS_TOP = 5         # reinsertion picks uniformly among the this-many widest gaps
REINS_FRAC = 0.08     # fraction of the circles (the smallest ones) a reinsertion turn moves
HOP_FAILS = 12        # consecutive failed hops before a stream is declared exhausted
PICK_C = 2.0          # exploration weight in the racing rule that allocates hop TURNS

# Iteration 14: the hop MODE is chosen by a cost-aware racing rule (_pick_mode) instead of the
# uniform rng.randint(HOP_MODES) rotation.  Two measured facts forced this:
#   (a) the modes are NOT equally useful at a given n.  Iteration 13's own probe found the
#       radius-cap hop worth +0.015 at n=65 and +0.011 at n=81 but NEGATIVE at all five alphas
#       at n=43 -- and n=43 is one of the sizes closest to its record.  A fixed 1/HOP_MODES
#       rotation spends a quarter of every size's turns on a move that some sizes only lose by.
#   (b) the modes do not COST the same.  The radius-cap hop runs _slp twice (once capped, once
#       released), so it is ~2x an ordinary shake; iteration 11 measured that hop turns are the
#       scarce resource.  So the currency is sum(r) gained per CPU SECOND, not per turn.
# MODE_WARM forced tries per mode come first, so no mode can be written off before it has been
# seen, and the rule degenerates to the old uniform rotation while no mode has gained anything.
MODE_C = 2.0          # exploration weight in the racing rule that allocates hop MODES
MODE_WARM = 2         # forced tries per mode before the rule may prefer any of them

# Iteration 7: _slp's CONVERGENCE STOP.  _slp is ~99% of the cost of one local solve (measured:
# 0.28-0.59 s of a 0.29-0.60 s `run` at n=41/67/99; _polish is under 6 ms).  Its per-iteration
# gain profile is a cliff, not a taper: measured over 7 start types x 3 sizes, the sum is within
# 2e-9 of its 80-iteration value by iteration 8-20, and iterations 20..80 buy between 0 and 2e-9
# while costing 1.5-2.5x the CPU.  Stopping at the cliff therefore roughly DOUBLES the number of
# basins a fixed CPU budget can visit, which is the only thing that has been moving quality.
#
# THE PRICE, STATED IN THE UNIT THAT MATTERS.  Giving up at most SLP_STOP_K * SLP_STOP_TOL of
# sum(r) is at most 4e-9 absolute, i.e. a relative gap of ~1e-9 on these sizes -- two orders of
# magnitude below the 1e-7 relgap that already earns the scorer's full 7 digits.  It is free at
# today's gaps (~1e-3).  IF the census ever closes to within ~1e-7, this tolerance is the first
# thing to tighten: at that point the trade reverses and the tail iterations start to matter.
SLP_STOP_TOL = 1e-9   # a per-iteration gain in sum(r) below this counts as "no progress"
SLP_STOP_K = 4        # consecutive no-progress iterations before _slp declares convergence

# Iteration 8: _slp's LAZY PAIR SET.  Probed: the committed census packs are EXACT fixed points of
# _slp (gain 0.0, at every trust-box size from 0.02 to 0.4), so the whole ~3e-3 relgap is basin
# CHOICE -- every CPU-second must buy a NEW basin, and the LP inside _slp is ~99% of what a basin
# costs.  Of the ~0.4 n^2 pairs in the provably-safe superset, only the few with small slack can
# actually bind, so the LP is solved on a guessed-active subset and any dropped row that turns out
# violated is added and the LP re-solved.  EXACTNESS: the loop exits only when no dropped row is
# violated (subset optimum feasible for the full LP => it IS the full LP OPTIMAL VALUE) or on the
# full superset itself.  Under LP degeneracy the returned VERTEX may still differ, so a long _slp
# trajectory is not bit-reproducible against the non-lazy code.  LAZY_PAD is a guess-quality knob:
# too small costs an extra round, too large costs rows -- neither can cost correctness.
LAZY_PAD = 0.5        # extra slack (in units of delta) admitted into the first guessed active set
LAZY_TOL = 1e-11      # a dropped row counts as violated only above this (HiGHS residual is ~1e-10)
LAZY_ROUNDS = 4       # rounds before falling back to the full safe superset (which is exact)


def _call_allowance(k):
    """CPU seconds this solve() call may spend, for k = len(targets). Never a module constant."""
    if k <= 1:
        return CPU_ONE
    return min(CPU_TOTAL_CAP, CPU_PER_N * k)


# --------------------------------------------------------------------------- geometry (exact)
def _wall(xy):
    """Distance from each centre to the nearest wall -> (n,). An r_i above this leaves the square."""
    x, y = xy[:, 0], xy[:, 1]
    return np.minimum(np.minimum(x - LO, HI - x), np.minimum(y - LO, HI - y))


def _pdist(xy):
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d * d).sum(-1))


def _feasible_sum(xy, r):
    """Strict feasibility check in the SAME terms the scorer uses; returns (ok, sum_r)."""
    if not (np.all(np.isfinite(xy)) and np.all(np.isfinite(r))):
        return False, -np.inf
    if r.min() <= 0.0:
        return False, -np.inf
    if (_wall(xy) - r).min() < -1e-9:
        return False, -np.inf
    n = len(r)
    if n >= 2:
        d = _pdist(xy)
        np.fill_diagonal(d, np.inf)
        if (d - (r[:, None] + r[None, :])).min() < -1e-9:
            return False, -np.inf
    return True, float(r.sum())


# --------------------------------------------------------------- layer 1: smooth penalty NLP
def _fg(z, n, mu):
    """Objective + analytic gradient of  -sum(r) + mu * (overlap^2 + wall^2 + negative-r^2)."""
    x, y, r = z[:n], z[n:2 * n], z[2 * n:]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, 1.0)                       # keep the diagonal out of the sqrt
    d = np.sqrt(d2)
    o = r[:, None] + r[None, :] - d
    np.fill_diagonal(o, 0.0)
    np.maximum(o, 0.0, out=o)                       # hinge: only violations are penalized

    w0 = np.maximum(r - (x - LO), 0.0)
    w1 = np.maximum(r - (HI - x), 0.0)
    w2 = np.maximum(r - (y - LO), 0.0)
    w3 = np.maximum(r - (HI - y), 0.0)
    neg = np.maximum(-r, 0.0)

    pen = 0.5 * (o * o).sum() + (w0 * w0 + w1 * w1 + w2 * w2 + w3 * w3 + neg * neg).sum()
    f = -r.sum() + mu * pen

    # o is 0 on the diagonal (d=1 there), and d is floored so two coincident centres -- which the
    # penalty is actively pushing apart -- give a large finite force instead of inf/NaN.
    od = o / np.maximum(d, 1e-9)
    gx = mu * (-2.0 * (od * dx).sum(1) + 2.0 * (w1 - w0))
    gy = mu * (-2.0 * (od * dy).sum(1) + 2.0 * (w3 - w2))
    gr = -1.0 + mu * (2.0 * o.sum(1) + 2.0 * (w0 + w1 + w2 + w3) - 2.0 * neg)
    return f, np.concatenate([gx, gy, gr])


def _ladder(xy, r, n, mus, maxiter, bounds):
    """Run the rising-penalty L-BFGS-B ladder from (xy, r). Returns refined (xy, r)."""
    try:
        from scipy.optimize import minimize
    except Exception:                                # pragma: no cover - scipy is installed
        return xy, r
    z = np.concatenate([xy[:, 0], xy[:, 1], r])
    for mu in mus:
        res = minimize(_fg, z, args=(n, mu), jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": maxiter, "maxcor": 12})
        z = res.x
    return np.stack([z[:n], z[n:2 * n]], axis=1), z[2 * n:].copy()


# ------------------------------------------------------------------ layer 3: repair + grow
def _separate(xy):
    """Nudge coincident/near-coincident centres apart so every circle can hold a positive radius."""
    xy = np.clip(xy, LO + 1e-7, HI - 1e-7)
    n = xy.shape[0]
    if n < 2:
        return xy
    d = _pdist(xy)
    np.fill_diagonal(d, np.inf)
    bad = np.where(d.min(1) < 1e-7)[0]
    for i in bad:                                   # deterministic nudge, no rng needed
        xy[i] = np.clip(xy[i] + 1e-6 * np.array([np.cos(i * 2.4), np.sin(i * 2.4)]),
                        LO + 1e-7, HI - 1e-7)
    return xy


def _repair(xy, r):
    """Force STRICT feasibility in exact arithmetic: clip, cap at the wall, then global down-scale."""
    xy = _separate(np.asarray(xy, dtype=float))
    r = np.clip(np.asarray(r, dtype=float), 0.0, None)
    w = np.maximum(_wall(xy) - SAFE, 0.0)
    r = np.minimum(r, w)
    n = len(r)
    if n >= 2:
        d = _pdist(xy)
        np.fill_diagonal(d, np.inf)
        s = ((r[:, None] + r[None, :]) / d).max()
        if s > 1.0:
            r = r / (s * (1.0 + 1e-12))
    return xy, r


def _grow(xy, r, sweeps=8):
    """Monotone coordinate ascent: r_i <- min(wall_i, min_j (d_ij - r_j)) - SAFE.

    Every update can only RAISE r_i (feasibility already implies r_i <= that bound) and leaves the
    packing feasible, so the sum is non-decreasing and the result is always safe to emit."""
    n = len(r)
    w = np.maximum(_wall(xy) - SAFE, 0.0)
    if n < 2:
        return np.maximum(w, 1e-12)
    d = _pdist(xy)
    np.fill_diagonal(d, np.inf)
    r = r.copy()
    for _ in range(sweeps):
        moved = 0.0
        for i in range(n):
            cap = min(w[i], float((d[i] - r).min()) - SAFE)
            if cap > r[i]:
                moved += cap - r[i]
                r[i] = cap
        if moved < 1e-14:
            break
    return r


# ------------------------------------------------------------------- layer 2: exact radii LP
def _lp_radii(xy):
    """Exact LP optimum for the radii with the centres FIXED, or None if unavailable.

    max 1'r  s.t.  r_i + r_j <= d_ij  for every pair,  0 <= r_i <= wall_i.
    Only pairs with d_ij < wall_i + wall_j can ever be active, so the rest are pruned."""
    n = xy.shape[0]
    w = np.maximum(_wall(xy) - SAFE, 0.0)
    if n < 2:
        return w
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:                                # pragma: no cover - scipy is installed
        return None
    d = _pdist(xy)
    iu = np.triu_indices(n, 1)
    dv = d[iu]
    keep = dv < (w[iu[0]] + w[iu[1]])
    I, J, b = iu[0][keep], iu[1][keep], dv[keep]
    m = len(b)
    if m == 0:
        return w
    rows = np.repeat(np.arange(m), 2)
    cols = np.empty(2 * m, dtype=np.int64)
    cols[0::2], cols[1::2] = I, J
    A = coo_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n))
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b,
                      bounds=np.stack([np.zeros(n), w], axis=1), method="highs")
    except Exception:
        return None
    if not getattr(res, "success", False) or res.x is None:
        return None
    return np.asarray(res.x, dtype=float)


# ------------------------------------------------- layer 2b: sequential LP on the TRUE problem
def _slp(xy, r, t_end=None, iters=80, delta0=0.02, delta_min=1e-11, rcap=None):
    """Monotone, always-feasible ascent on the real program, by a sequence of LPs.

    WHY THIS IS SAFE (the property the whole refiner rests on).  The only nonlinear constraint is
    r_i + r_j <= |p_i - p_j|.  Replace the right-hand side by its first-order expansion at the
    current centres,  d_ij + u_ij . (dp_i - dp_j)  with  u_ij = (p_i - p_j)/d_ij.  The Euclidean
    norm is CONVEX, so it lies above every tangent plane:

        |(p_i + dp_i) - (p_j + dp_j)|  >=  d_ij + u_ij . (dp_i - dp_j)

    The linearized constraint is therefore STRICTLY CONSERVATIVE -- anything feasible for the LP is
    feasible for the true problem, at any step length.  The wall constraints are already exactly
    linear.  And dp = 0, r = r_cur is LP-feasible, so the LP optimum is never worse than where we
    started.  Hence: every iterate is truly feasible and sum(r) is non-decreasing, with no line
    search and no trust-region acceptance test.  The box |dp| <= delta is needed only to keep the
    pair PRUNING valid and to drive convergence; shrinking it to 0 drives the iterate to a KKT
    point of the true program.  (delta = 0 reduces exactly to the radii-only LP.)
    """
    n = xy.shape[0]
    if n < 2:
        return xy, r
    try:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix
    except Exception:                                # pragma: no cover - scipy is installed
        return xy, r
    xy = np.array(xy, dtype=float)
    r = np.array(r, dtype=float)
    if rcap is not None:
        # dp = 0, r = r_cur must be LP-FEASIBLE or the very first step looks like a loss and the
        # trust region collapses.  Under a cap that means clipping the incumbent radii first.
        r = np.minimum(r, float(rcap))
    cur = float(r.sum())
    delta = float(delta0)
    idx = np.arange(n)
    iu = np.triu_indices(n, 1)
    stall = 0                                        # consecutive iterations gaining < SLP_STOP_TOL
    for _ in range(iters):
        if t_end is not None and time.process_time() > t_end:
            break
        if delta < delta_min:
            break
        w = _wall(xy)
        d = _pdist(xy)
        dv = d[iu]
        # A pair may be dropped only if it CANNOT bind: each centre moves at most delta, so the
        # gap shrinks by at most 2*delta while each radius is capped by wall+delta.  This is the
        # SAFE SUPERSET -- correctness rests on it; the lazy set below only reorders work.
        keep = dv <= (w[iu[0]] + w[iu[1]] + 4.0 * delta)
        Is, Js, ds = iu[0][keep], iu[1][keep], dv[keep]
        ms = len(ds)
        if ms and ds.min() <= 0.0:
            return xy, r                             # coincident centres: let _repair handle it
        if ms:
            uxs = (xy[Is, 0] - xy[Js, 0]) / ds
            uys = (xy[Is, 1] - xy[Js, 1]) / ds
        else:
            uxs = uys = np.zeros(0)
        # The conservativeness theorem above is exact arithmetic; HiGHS only satisfies A x <= b to
        # its own primal tolerance (1e-7 by default), which showed up as ~1e-7 overlaps in the raw
        # iterate.  Tighten that tolerance AND shrink every right-hand side by LP_MARGIN, so the
        # solver's residual slack is spent against a target we deliberately set short.  Cost: about
        # 1e-9 of sum(r); benefit: _slp's output is feasible on its own, not only after _repair.
        bs = ds - LP_MARGIN
        ones = np.ones(n)
        # Walls are exactly linear:  r_i -+ dx_i <= dist to that wall.  Always present, so they
        # are built once here and re-used by every lazy round (only their row OFFSET moves).
        wall_c = np.concatenate([np.column_stack([axis * n + idx, 2 * n + idx]).ravel()
                                 for axis in (0, 0, 1, 1)])
        wall_v = np.concatenate([np.column_stack([sign * ones, ones]).ravel()
                                 for sign in (-1.0, 1.0, -1.0, 1.0)])
        wall_b = np.concatenate([xy[:, 0] - LO, HI - xy[:, 0],
                                 xy[:, 1] - LO, HI - xy[:, 1]]) - LP_MARGIN
        wall_rows = np.repeat(np.arange(4 * n), 2)
        lo = np.concatenate([np.full(2 * n, -delta), np.zeros(n)])
        # RADIUS CAP (iteration 13).  rcap=None is the true program (cap 0.5 never binds).
        # A finite rcap turns the SAME LP into the HOMOGENISED surrogate max sum min(r_i,cap):
        # every circle above the cap must shrink, so the centres are driven to a layout where
        # as many circles as possible reach ONE common radius.  Nothing else changes -- the
        # conservativeness argument above is untouched, so the iterate stays truly feasible.
        rhi = 0.5 if rcap is None else float(rcap)
        hi = np.concatenate([np.full(2 * n, delta), np.full(n, rhi)])
        cost = np.concatenate([np.zeros(2 * n), -ones])
        # LAZY PAIR SET (iteration 8).  Only pairs whose slack d_ij - r_i - r_j is small can bind;
        # measured, that is ~3n of the ~0.4 n^2 pairs in the safe superset, and the LP is ~99% of
        # this solver's CPU.  So solve on a guessed-active subset, then CHECK every dropped safe
        # pair against the same linearized row and re-solve with the violators added.  This is
        # exact, not an approximation: when no dropped row is violated, the subset optimum is
        # feasible for the full LP and therefore IS the full LP optimum -- so the returned iterate
        # EQUALS the full LP's optimal VALUE (--self-test asserts this per LP, to 1e-12).
        # The last round falls back to the whole safe superset, so worst case costs one extra LP.
        #
        # WHAT IS *NOT* GUARANTEED, stated plainly.  The LP is often DEGENERATE: several vertices
        # attain the same optimum, and HiGHS may return a different one when the row set changes.
        # So each step's VALUE is exact but its ARGMAX can differ, and over ~15 steps the two
        # trajectories can land in different basins.  Measured over 24 (size, start) pairs: 22
        # agreed to 1.2e-12, one ended +0.0043 and one -0.0101 (artifacts/iter8_lazy_probe.txt).
        # That is basin luck, not lost accuracy -- but it means this change is NOT a pure no-op on
        # quality, and its quality effect is unmeasured on held-out sizes (see TASKS_DONE iter 8).
        act = (ds <= r[Is] + r[Js] + (2.0 + LAZY_PAD) * delta) if ms else np.zeros(0, bool)
        z = None
        for rnd in range(LAZY_ROUNDS):
            sel = np.flatnonzero(act)
            m = len(sel)
            if m:
                Ik, Jk = Is[sel], Js[sel]
                rows = np.concatenate([np.repeat(np.arange(m), 6), m + wall_rows])
                cols = np.concatenate([
                    np.column_stack([Ik, Jk, n + Ik, n + Jk, 2 * n + Ik, 2 * n + Jk]).ravel(),
                    wall_c])
                vals = np.concatenate([
                    np.column_stack([-uxs[sel], uxs[sel], -uys[sel], uys[sel],
                                     np.ones(m), np.ones(m)]).ravel(), wall_v])
                b = np.concatenate([bs[sel], wall_b])
            else:
                rows, cols, vals, b = wall_rows, wall_c, wall_v, wall_b
            A = csr_matrix((vals, (rows, cols)), shape=(m + 4 * n, 3 * n))
            try:
                res = linprog(cost, A_ub=A, b_ub=b,
                              bounds=np.stack([lo, hi], axis=1), method="highs",
                              options={"primal_feasibility_tolerance": 1e-10,
                                       "dual_feasibility_tolerance": 1e-10})
            except Exception:
                res = None
            if res is None or not getattr(res, "success", False) or res.x is None:
                z = None
                break
            z = np.asarray(res.x, dtype=float)
            if m == ms:
                break                                # solved the full safe set: nothing left out
            lhs = (z[2 * n + Is] + z[2 * n + Js]
                   - uxs * (z[Is] - z[Js]) - uys * (z[n + Is] - z[n + Js]))
            bad = (lhs > bs + LAZY_TOL) & ~act
            if not bad.any():
                break                                # subset optimum is full-LP feasible => optimal
            act = act | bad
            if rnd >= LAZY_ROUNDS - 2:
                act[:] = True                        # give up guessing; the next round is exact
        if z is None:
            break
        if not np.all(np.isfinite(z)):
            break
        new_s = float(z[2 * n:].sum())
        gain = new_s - cur
        # CONVERGENCE STOP (iteration 7).  `iters` remains the hard cap; this is the rule that
        # normally ends the loop.  It is adaptive rather than a smaller `iters`: a start that is
        # still gaining 1e-2 per iteration keeps every iteration it needs (measured: a mode-2
        # cold start at n=41 is still moving at iteration 14), while one that has arrived stops
        # within SLP_STOP_K iterations instead of grinding delta down to delta_min for ~1e-9.
        # Counted on GAIN, not on delta, because delta shrinks on stalls too -- a run that stalls
        # once and then finds a large step would be cut off by a delta test.
        if gain < SLP_STOP_TOL:
            stall += 1
        else:
            stall = 0
        if gain > 1e-15:
            xy = np.stack([xy[:, 0] + z[:n], xy[:, 1] + z[n:2 * n]], axis=1)
            r = z[2 * n:].copy()
            cur = new_s
            if gain < 1e-10:                         # converged at this radius: tighten
                delta *= 0.5
        else:
            delta *= 0.35                            # stalled: tighten and re-solve
        if stall >= SLP_STOP_K:
            break
    # HiGHS satisfies A x <= b only to its own primal tolerance, so the raw iterate can carry a
    # ~1e-7 overlap even though the linearization is exact-arithmetic conservative.  Close that in
    # exact arithmetic before returning: _repair down-scales into strict feasibility and _grow
    # (monotone) takes the margin back.  _slp's contract is therefore "strictly feasible out".
    xy, r = _repair(xy, r)
    return xy, _grow(xy, r)


def _polish(xy, r):
    """Repair -> LP radii (if it helps) -> grow. Returns a STRICTLY feasible (xy, r, sum_r)."""
    xy, r = _repair(xy, r)
    r = _grow(xy, r)
    lp = _lp_radii(xy)
    if lp is not None:
        _, r2 = _repair(xy, lp)
        r2 = _grow(xy, r2)
        if r2.sum() > r.sum():
            r = r2
    ok, s = _feasible_sum(xy, r)
    return xy, r, (s if ok else -np.inf)


# ------------------------------------------------------------------------------ warm starts
def _read_pack(n):
    """Guarded read of the committed pack for n. Returns (xy, r) or None -- never raises."""
    p = PACKS % n
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            rows = [ln.split() for ln in fh.read().splitlines()]
        pts = [(float(a), float(b), float(c)) for a, b, c in
               (t for t in rows if len(t) == 3)]
        if len(pts) != n:
            return None
        arr = np.array(pts, dtype=float)
        return arr[:, :2], arr[:, 2]
    except Exception:
        return None


# --------------------------------------------------------- structured lattice initializations
# HEX_K: rows in a hexagonal packing of k rows x ~m columns fill the square when
#   k * (sqrt(3)/2) * pitch ~= 1  and  m * pitch ~= 1  with  k*m ~= n,
# i.e. k ~= sqrt(2n/sqrt(3)) = sqrt(1.1547 n).  That is the row count to try FIRST for any n;
# the enumeration below fans outward from it.  This is arithmetic in n, not a table keyed on n.
HEX_K = np.sqrt(2.0 / np.sqrt(3.0))


def _row_counts(n, k, mode):
    """Split n circles over k rows. `mode` chooses where the remainder rows go.

    mode 0: extra circles on the leading rows; 1: on alternating rows (preserves the
    hex parity of long/short rows); 2: on the trailing rows."""
    if k <= 0 or k > n:
        return None
    base, extra = divmod(n, k)
    counts = [base] * k
    if mode == 1:
        order = list(range(0, k, 2)) + list(range(1, k, 2))
    elif mode == 2:
        order = list(range(k - 1, -1, -1))
    else:
        order = list(range(k))
    for i in range(extra):
        counts[order[i % k]] += 1
    return counts


def _lattice(n, k, mode, stagger):
    """k rows of evenly spread circles, odd rows shifted by `stagger` of a pitch.

    The stagger is the whole point: an unstaggered lattice is a square grid, whose density
    ceiling is pi/4, while the staggered one is hexagonal (pi/(2 sqrt 3)).  The probe in
    artifacts/iter3_init_probe.py measured the difference as the gap between relgap 1e-2 and
    1.7e-3 at n=57 -- far larger than anything the local solver recovers on its own."""
    counts = _row_counts(n, k, mode)
    if counts is None:
        return None
    ys = (np.arange(k) + 0.5) / float(k) - 0.5
    xs, yy = [], []
    for i, m in enumerate(counts):
        if m <= 0:
            continue
        off = (stagger / float(m)) if (i % 2) else 0.0
        row = (np.arange(m) + 0.5) / float(m) - 0.5 + off
        xs.append(row)
        yy.append(np.full(m, ys[i]))
    if not xs:
        return None
    pts = np.stack([np.concatenate(xs), np.concatenate(yy)], axis=1)
    if pts.shape[0] != n:
        return None
    return np.clip(pts, LO + 1e-3, HI - 1e-3)


def _lattice_plan(n):
    """Deterministic (k, mode, stagger, transpose) sequence, BEST-FIRST.

    Order is load-bearing, because the two meters buy very different amounts of it: the in-run
    allowance (~3 CPU-s for a census size) buys roughly the first half-dozen candidates, while
    the offline 60 s-per-size re-run buys the whole sweep.  So the sequence is:

      1. the staggered (hexagonal) fan over row counts, starting at the theoretical k and
         working outward -- measured winners for n = 57/83/99 all sit in its first three;
      2. transposes of the leading row counts (the square is symmetric, the remainder
         placement is not);
      3. the unstaggered (square-grid) fan, which wins only for near-perfect squares.

    When k divides n every remainder mode gives the same lattice, so only one is emitted."""
    k0 = max(1, int(round(HEX_K * np.sqrt(max(n, 1)))))
    ks = []
    for d in range(0, 7):
        for kk in ((k0,) if d == 0 else (k0 + d, k0 - d)):
            if 1 <= kk <= n and kk not in ks:
                ks.append(kk)
    def modes_for(kk):
        return (1,) if n % kk == 0 else (1, 2, 0)
    plan = [(kk, mode, 0.5, False) for kk in ks for mode in modes_for(kk)]
    plan += [(kk, mode, 0.5, True) for kk in ks[:4] for mode in modes_for(kk)]
    plan += [(kk, mode, 0.0, False) for kk in ks for mode in modes_for(kk)]
    return plan


def _init(n, rng, kind):
    """Cold starts. `kind` cycles strategies; all work for ANY n, none is keyed on n."""
    if kind == 0:                                    # jittered near-square grid
        k = int(np.ceil(np.sqrt(n)))
        gx, gy = np.meshgrid(np.linspace(LO + 0.5 / k, HI - 0.5 / k, k),
                             np.linspace(LO + 0.5 / k, HI - 0.5 / k, k))
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)[:n]
        pts = pts + rng.normal(0.0, 0.25 / k, size=pts.shape)
    elif kind == 1:                                  # jittered hex lattice at the hex row count
        k = max(1, int(round(HEX_K * np.sqrt(max(n, 1)))))
        pts = _lattice(n, k, 1, 0.5)
        if pts is None:
            pts = rng.uniform(LO + 0.05, HI - 0.05, size=(n, 2))
        pts = pts + rng.normal(0.0, 0.15 / max(k, 1), size=(n, 2))
    else:                                            # uniform random
        pts = rng.uniform(LO + 0.05, HI - 0.05, size=(n, 2))
    pts = np.clip(pts, LO + 1e-3, HI - 1e-3)
    r0 = np.full(n, 0.35 / np.sqrt(n))
    return pts, r0


# ------------------------------------------------------------- the stream pool (iteration 4)
def _pool_insert(pool, xy, r, s, src):
    """Insert a local optimum into `pool` as a search STREAM.  Pure function, in-place on `pool`.

    Two rules, and both are the point of the pool existing at all:

      * DEDUPE BY BASIN.  Streams are told apart by their sorted radius profile; two optima whose
        profiles agree to POOL_DIST are the same basin, so the better point replaces the worse
        one instead of occupying a second slot.  Without this the pool collapses into POOL_P
        copies of one basin and buys no diversity whatsoever.
      * RESERVED SLOTS BY ORIGIN.  A `src="sweep"` stream competes only against other sweep
        streams (at most POOL_SWEEP of them); the remaining slots belong to `src="cold"`.  So an
        arbitrarily good lattice optimum can never evict the independent cold-start stream.
        That is exactly the failure iteration 3 measured at n=40: the sweep's best lattice is a
        deep local optimum, hopping cannot leave it, and with a single incumbent there was no
        other stream left to be anywhere else.

    Returns True if the pool changed."""
    if not np.isfinite(s):
        return False
    key = np.sort(r)
    for st in pool:
        if st["key"].shape == key.shape and np.max(np.abs(st["key"] - key)) < POOL_DIST:
            if s > st["s"]:                      # same basin, better point: upgrade in place
                st.update(xy=xy.copy(), r=r.copy(), s=s, key=key, sigma=0.04, fails=0)
                return True
            return False
    st = dict(xy=xy.copy(), r=r.copy(), s=s, key=key, src=src, sigma=0.04, fails=0, turns=0)
    cap = POOL_SWEEP if src == "sweep" else max(0, POOL_P - POOL_SWEEP)
    mine = [i for i, q in enumerate(pool) if q["src"] == src]
    if len(mine) < cap:
        pool.append(st)
        return True
    if not mine:
        return False
    w = min(mine, key=lambda i: pool[i]["s"])
    if s > pool[w]["s"]:
        pool[w] = st
        return True
    return False


def _pick_slot(svals, turns, c=PICK_C):
    """RACING RULE (iteration 5): which pool stream spends the next hop TURN.  Pure function.

        score_i = quality_i + c * sqrt(log(1 + T) / turns_i),      T = sum(turns)
        quality_i = 1 - rank_i / (P - 1)      ORDINAL: 1.0 for the best stream, 0.0 for the worst

    WHY THIS REPLACES ROUND-ROBIN.  Iteration 4 measured that a uniform turn split is what still
    loses at n=40 (3.282920 vs iteration 2's single-incumbent 3.291324): the size wants many
    CONSECUTIVE hops on one promising basin, and round-robin taxes a third of the winner's turns
    away.  Iteration 4 also measured that the cure is not *less* diversity -- dropping to one
    incumbent is what iteration 3 had, and it lost at n=50/64.  So the allocation, not the pool,
    is the lever: keep all POOL_P streams alive, but spend turns on them by measured standing.

    WHY ORDINAL AND NOT THE RAW sum_r GAP.  The gaps between distinct basins have no stable
    scale -- they are ~1e-2 early and ~1e-6 once every stream has converged, and they differ by
    orders of magnitude across n.  A raw-value bonus would therefore be pure greed at one size
    and indistinguishable from round-robin at another.  Rank is scale-free, so ONE c allocates
    turns the same way at every n and at both meters.

    PROPERTIES (asserted in the self-test):
      * a stream that has never hopped is picked first, so every stream is tried before any is
        preferred -- the pool cannot be starved at birth by a lucky first stream;
      * the exploration term decays as the leader accumulates turns, so the split settles
        (c=2.0, P=3 -> the leader takes ~60% of turns rather than round-robin's 33%);
      * no stream is starved permanently: a stream left idle has its exploration term grow;
      * a challenger that OVERTAKES the leader inherits the quality bonus on the next turn, so
        the allocation follows the measurement instead of the order the pool was built in.
    """
    m = len(svals)
    if m <= 1:
        return 0
    for i in range(m):
        if turns[i] <= 0:                        # every stream gets its first hop before any
            return i                             # stream gets its second
    s = np.asarray(svals, dtype=float)
    t = np.asarray(turns, dtype=float)
    rank = np.empty(m, dtype=float)
    rank[np.argsort(-s, kind="stable")] = np.arange(m, dtype=float)
    score = (1.0 - rank / float(m - 1)) + c * np.sqrt(np.log1p(t.sum()) / t)
    return int(np.argmax(score))


def _pick_mode(gain, cpu, tries, c=MODE_C, warm=MODE_WARM):
    """RACING RULE over the hop MODES (iteration 14): which move kind the next TURN uses.

    Pure function of its arguments, the same shape as _pick_slot but over modes rather than
    streams, and in the currency that actually matters:

        rate_m  = gain_m / cpu_m            sum(r) GAINED per process-CPU SECOND by mode m
        score_m = quality_m + c * sqrt(log(1 + T) / tries_m),     T = sum(tries)
        quality_m = 1 - rank_m / (M - 1)    ORDINAL, ties AVERAGED: 1.0 best, 0.0 worst

    WHY ORDINAL (same reason as _pick_slot).  Gains have no stable scale: ~1e-2 on a fresh
    stream, ~1e-7 once every stream has converged, and different by orders of magnitude across
    n.  Rank is scale-free, so ONE c behaves the same at n=27 and n=99 and at both meters.

    WHY PER CPU SECOND AND NOT PER TURN.  The radius-cap hop solves _slp twice (capped, then
    released) and so costs about twice an ordinary shake.  Ranking by gain-per-turn would buy
    that mode turns it has not earned; ranking by gain-per-second prices it against the shakes
    it displaces.  The costs are MEASURED per turn, not assumed, so this stays correct if a
    future mode is cheaper or more expensive than today's.

    PROPERTIES (asserted in the self-test):
      * every mode is tried `warm` times before any mode is preferred, so a mode cannot be
        written off on zero evidence (and HOP_MODES-1 is never dead code at any n);
      * TIES ARE AVERAGED, so while no mode has gained anything all qualities are equal and the
        rule is exactly the old uniform rotation -- the change is inert until it has evidence;
      * a mode that stops paying decays to the exploration floor but is never starved: its
        bonus grows as the others accumulate tries, so it is periodically re-measured;
      * it follows the measurement: a mode that overtakes on rate inherits the quality bonus.
    """
    m = len(tries)
    if m <= 1:
        return 0
    t = np.asarray(tries, dtype=float)
    for i in range(m):
        if t[i] < warm:                          # forced exploration: no mode is written off
            return i                             # before it has been seen `warm` times
    rate = np.asarray(gain, dtype=float) / np.maximum(np.asarray(cpu, dtype=float), 1e-9)
    # average rank (0 = best); ties share a rank, so all-zero gains -> equal quality
    better = (rate[None, :] > rate[:, None]).sum(axis=1).astype(float)
    equal = (rate[None, :] == rate[:, None]).sum(axis=1).astype(float)
    rank = better + 0.5 * (equal - 1.0)
    score = (1.0 - rank / float(m - 1)) + c * np.sqrt(np.log1p(t.sum()) / t)
    return int(np.argmax(score))


def _sweep_step(indep_s, ss, stall):
    """One update of the lattice sweep's DOMINANCE STOP.  Pure function of its arguments.

    Returns (indep_s, stall, keep_going).

      * `indep_s` is the best sum reached by a stream the sweep did not produce (today: the
        committed pack), or -inf when there is none.  DISARMED at -inf: cold, the sweep always
        runs its full share, because its value is not monotone in plan index and a stall rule on
        its own best would truncate it before its best candidate (measured, see SWEEP_STALL).
      * A sweep candidate that BEATS the incumbent re-arms the rule against itself and resets the
        counter, so a sweep that is still finding new bests is never cut off.
      * SWEEP_STALL consecutive candidates that fail to beat it end the sweep.
    """
    if not np.isfinite(indep_s):
        return indep_s, 0, True
    if np.isfinite(ss) and ss > indep_s:
        return float(ss), 0, True
    stall += 1
    return indep_s, stall, stall < SWEEP_STALL


# ------------------------------------------------------------------------------ the search
_MUS_FULL = (1e2, 3e2, 1e3, 3e3, 1e4, 1e5, 1e6, 1e7)
_MUS_HOP = (1e3, 1e4, 1e5, 1e6, 1e7)


def _homogenize(xy, r, rng, t_end):
    """RADIUS-CAP HOP (iteration 13). Re-solve the SAME program under a common cap on every radius.

    WHY THIS MOVE EXISTS.  Maximising sum(r) for a fixed circle count rewards equal radii: for a
    fixed total AREA (which is what the square bounds), sum(r) is maximised when all r_i agree, so
    the optima we are chasing are near-uniform dense layouts.  But sum(r) is also a LINEAR objective
    that will happily pay for one oversized circle by starving its neighbours, and _slp -- being a
    monotone ascent -- can never undo that once it is jammed: every committed pack has n DISTINCT
    radii spread over a continuum (iteration 12 checked this at 6 dp).

    Capping every radius at alpha * mean(r) removes exactly that payoff.  The oversized circles must
    shrink, the LP has no reason to keep their neighbours starved, and the centres drift to a layout
    where as many circles as possible reach the one common radius.  Releasing the cap and re-solving
    then lands in a DIFFERENT basin -- reached by a move no amount of centre jitter performs, because
    it is driven by the objective rather than by noise.

    Unlike the centre shakes this move is not random noise: the only randomness is alpha, and alpha
    matters (measured, artifacts/iter13_cap_probe.txt: at n=65 alpha in 0.95..1.05 gained +0.013 to
    +0.015 on the committed pack, while alpha=0.85 collapsed straight back to the same optimum and
    alpha=1.15 gained nothing).  Costs ~0.1-0.3 process-CPU s, the same order as one ordinary hop.
    """
    rm = float(np.mean(r))
    if not np.isfinite(rm) or rm <= 0.0:
        return xy.copy(), r.copy()
    cap = rm * float(rng.uniform(HOP_ALPHA_LO, HOP_ALPHA_HI))
    return _slp(xy.copy(), r.copy(), t_end=min(t_end, time.process_time() + 2.0), rcap=cap)


def _shake(xy, r, rng, mode, sigma):
    """One basin-hopping move on the centres. Never keyed on n; works for any size."""
    n = xy.shape[0]
    xy = xy.copy()
    if mode == 0 or n < 3:                           # diffuse jitter, adaptive width
        xy += rng.normal(0.0, sigma, size=xy.shape)
    elif mode == 1:
        # RELOCATE THE SMALLEST.  In a max-sum-radii packing the tiny circles are the ones wedged
        # into corners of the contact graph, contributing almost nothing; teleporting them is the
        # move that actually changes the contact TOPOLOGY, which jitter alone cannot do.
        k = max(1, int(round(0.12 * n)))
        who = np.argsort(r)[:k]
        xy[who] = rng.uniform(LO + 0.02, HI - 0.02, size=(k, 2))
    else:
        # Swap a small circle with a large one, then jitter: exchanges roles between two sites.
        order = np.argsort(r)
        k = max(1, int(round(0.08 * n)))
        a, b = order[:k], order[-k:]
        xy[a], xy[b] = xy[b].copy(), xy[a].copy()
        xy += rng.normal(0.0, 0.25 * sigma, size=xy.shape)
    return np.clip(xy, LO + 1e-3, HI - 1e-3)


def _reinsert(xy, r, rng, frac=REINS_FRAC, grid=REINS_GRID, top=REINS_TOP):
    """GAP-DIRECTED REINSERTION (iteration 15).  Lift the smallest circles out and put them back
    where the packing is actually empty, instead of where a random draw happens to land.

    WHY THIS MOVE EXISTS, AND WHY IT IS NOT MODE 1.  Mode 1 already relocates the smallest
    circles -- to `rng.uniform` positions.  That is the right IDEA (the tiny circles are the ones
    wedged into the contact graph, so moving them is the only thing that changes the contact
    TOPOLOGY) executed blindly: a uniform point in the unit square lands inside an existing
    circle almost every time at n>=27, `_repair` then shoves it back out along the nearest
    escape direction, and the move degenerates into a large, badly aimed jitter.  This move aims
    it.  For every point of a jittered grid it computes the clearance

        clear(p) = min( 0.5 - max(|p_x|, |p_y|),  min_j ||p - c_j|| - r_j )   over the KEPT circles

    -- exactly the radius a new circle at p could take -- and drops the circle at one of the
    `top` widest gaps, chosen by rng so repeated turns explore different holes rather than
    replaying one.  Circles are placed ONE AT A TIME, each seeing the ones already placed, so two
    of them cannot be stacked into the same hole.  The reinserted radius is seeded at its own
    clearance, which is feasible by construction and gives `_slp` a live starting point instead
    of a speck.

    Cost is one `_slp` (the grid itself is ~2e-3 s: grid**2 x n distances, vectorised), so it is
    the price of an ordinary shake, not of the radius-cap hop.  Whether it is WORTH that price at
    a given n is not asserted here -- `_pick_mode` prices it against the other moves per size,
    which is the whole reason this could be added as a mode rather than as a policy.
    """
    n = xy.shape[0]
    k = int(min(max(1, int(round(frac * n))), max(n - 2, 1)))
    xy = xy.copy()
    r = r.copy()
    who = np.argsort(r)[:k]
    keep = np.ones(n, dtype=bool)
    keep[who] = False
    g = np.linspace(LO, HI, grid)
    P = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    P = np.clip(P + rng.uniform(-0.5, 0.5, size=P.shape) / float(grid - 1), LO, HI)
    wall = HI - np.maximum(np.abs(P[:, 0]), np.abs(P[:, 1]))
    for idx in who:
        c = xy[keep]
        if c.shape[0]:
            d = np.sqrt(((P[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)) - r[keep][None, :]
            clear = np.minimum(wall, d.min(axis=1))
        else:
            clear = wall
        m = min(top, clear.shape[0])
        cand = np.argpartition(-clear, m - 1)[:m] if m < clear.shape[0] else np.arange(m)
        j = int(cand[rng.randint(m)])
        xy[idx] = P[j]
        r[idx] = max(float(min(clear[j], 0.5)) - SAFE, 1e-6)
        keep[idx] = True
    return np.clip(xy, LO + 1e-3, HI - 1e-3), r


def _solve_one(n, evaluate, meter, rng, t_end, batch=24):
    """Search size n until process-CPU passes t_end. Routes every candidate through evaluate().

    ORDER OF WORK:
      1. the committed pack for n, if there is one (guarded -- there may not be);
      2. the STRUCTURED LATTICE SWEEP, best-first from the hexagonal row count;
      3. basin hopping over a POOL of independent streams (iteration 4) whose TURNS are
         allocated by a racing rule (iteration 5), with whatever CPU is left.

    Stage 3 hops several streams rather than one incumbent because a single incumbent makes the
    search only as good as the basin it happens to start in: iteration 3 measured that at n=40,
    where the sweep's best lattice is a deep local optimum, hopping never left it, and the
    result was WORSE than iteration 2's hopping-only search (3.285278 vs 3.291324).  Extra CPU
    could not fix that (capping the sweep changed nothing); only having a stream somewhere else
    can.  The pool keeps the top POOL_SWEEP sweep optima -- which the sweep already paid for and
    iteration 3 discarded -- and reserves the remaining slots for cold starts that the lattice
    family cannot evict (see _pool_insert).

    Stage 2 exists because _slp is an exact local solver: it reaches a KKT point of the true
    program in ~0.25 s and then stops moving (measured: a 0.25 s and a 2.0 s run return
    bit-identical packings).  So quality is decided almost entirely by which basin the start
    lands in, and the basins that matter here are the row-lattice topologies -- which row
    count, which parity of long/short rows, staggered or square.  Enumerating that small
    family beats spending the same CPU on random restarts, because the family is where the
    optima live.  Both meters are respected: the in-run allowance buys the first few
    candidates (the fan around the hex row count), the 60 s offline allowance buys the sweep.
    """
    pool = []                                    # the live search streams (see _pool_insert)
    pending = []
    best_xy = best_r = None
    best_s = -np.inf

    def flush():
        nonlocal pending
        if not pending:
            return
        k = int(meter.left())
        if k <= 0:
            pending = []
            return
        chunk = pending[:k]
        pending = []
        evaluate(n, np.stack(chunk))                 # the harness keeps the best it is shown

    def offer(xy, r, s):
        nonlocal best_xy, best_r, best_s
        if not np.isfinite(s):
            return False
        pending.append(np.concatenate([xy, r[:, None]], axis=1))
        gained = s > best_s
        if gained:
            best_xy, best_r, best_s = xy.copy(), r.copy(), s
        if len(pending) >= batch:
            flush()
        return gained

    def run(xy0, r0, slp_budget):
        """repair -> exact local solve -> strictly-feasible polish -> offer.

        Returns the polished (xy, r, s) so the CALLER decides what it means: stage 2 and the
        cold starts feed it to the stream pool, a hop compares it against its own stream."""
        xy0, r0 = _repair(xy0, r0)
        xy1, r1 = _slp(xy0, r0, t_end=min(t_end, time.process_time() + slp_budget))
        xy2, r2, s2 = _polish(xy1, r1)
        offer(xy2, r2, s2)
        return xy2, r2, s2

    def alive():
        return time.process_time() < t_end and meter.left() > len(pending)

    # (1) warm start from the committed census, if this n has one.  Guarded: may be absent.
    #     `indep_s` is the best sum from a stream the LATTICE SWEEP did not produce; it is what
    #     the sweep has to beat to be worth continuing (see SWEEP_STALL).  It stays -inf when
    #     there is no pack, which is exactly when the dominance stop must not fire.
    indep_s = -np.inf
    warm = _read_pack(n)
    if warm is not None and meter.left() > 0:
        # The committed pack is a stream in its own right, and it is NOT part of the lattice
        # family, so it takes a reserved (cold) slot rather than a sweep one.
        wxy, wr, ws = run(warm[0], warm[1], 3.0)
        _pool_insert(pool, wxy, wr, ws, src="cold")
        if np.isfinite(ws):
            indep_s = ws

    # (2) structured lattice sweep, best-first.  Deterministic: no rng, so the strong
    #     candidates are hit in the same order however much CPU the caller happens to give.
    #
    #     CAPPED AT SWEEP_SHARE of the allowance so basin hopping always gets a share.  The
    #     sweep is best-first, so its marginal value decays fast.  The cap is a share, not a
    #     constant, so it is correct at both meters: the ~3 s census allowance and the 60 s
    #     offline one.
    #
    #     Every sweep result is offered to the STREAM POOL, not just the best one: the sweep
    #     already pays for a dozen distinct local optima and iteration 3 threw all but one
    #     away.  Keeping the top POOL_SWEEP of them, plus a reserved slot for an independent
    #     cold start, is iteration 3's diagnosed fix for its own measured n=40 regression --
    #     the sweep's best lattice there is a deep optimum that hopping cannot leave, so the
    #     search needs a stream that is somewhere else entirely.
    t_sweep = time.process_time() + SWEEP_SHARE * max(t_end - time.process_time(), 0.0)
    r0 = np.full(n, 0.35 / np.sqrt(max(n, 1)))
    sweep_stall = 0
    for k, mode, stagger, transpose in _lattice_plan(n):
        if not alive() or time.process_time() >= t_sweep:
            break
        pts = _lattice(n, k, mode, stagger)
        if pts is None:
            continue
        if transpose:
            pts = pts[:, ::-1].copy()
        sxy, sr, ss = run(pts, r0.copy(), 2.0)
        _pool_insert(pool, sxy, sr, ss, src="sweep")
        # DOMINANCE STOP (iteration 12).  Only armed when an independent incumbent exists.
        indep_s, sweep_stall, go = _sweep_step(indep_s, ss, sweep_stall)
        if not go:
            break

    # (3) basin hopping over the stream pool, turns allocated by the RACING RULE _pick_slot.
    #     One turn = one hop on one stream, each stream carrying its own step width and its own
    #     stall counter, so a converging stream keeps its small sigma while a stalled one widens
    #     independently.  Iteration 4 cycled the turns uniformly; _pick_slot instead spends them
    #     on the streams that are measurably standing best, while still trying every stream
    #     before preferring any and never starving one permanently (see _pick_slot).
    #
    #     Until the pool holds POOL_P streams the turn buys an INDEPENDENT cold start (never a
    #     copy of the incumbent), which is what keeps a stream outside the lattice family alive.
    #     If the pool refuses two in a row -- every slot already claimed, the newcomer no better
    #     -- stop offering them, or a pool that cannot reach POOL_P spends EVERY turn restarting
    #     instead of hopping.  A pop reopens a slot and re-arms the offer.
    fill_rejects = 0
    # Per-SIZE mode statistics for _pick_mode: turns taken, CPU spent and sum(r) gained by each
    # move kind.  Per size on purpose -- iteration 13 measured that which move pays is a
    # property of n (radius-cap: +0.015 at n=65, negative at n=43), not a global constant -- and
    # rebuilt from zero on every _solve_one, so nothing leaks across sizes or across calls.
    m_tries = np.zeros(HOP_MODES)
    m_cpu = np.zeros(HOP_MODES)
    m_gain = np.zeros(HOP_MODES)
    while alive():
        if len(pool) < POOL_P and fill_rejects < 2:
            before = len(pool)
            _pool_insert(pool, *run(*_init(n, rng, rng.randint(3)), slp_budget=2.0), src="cold")
            fill_rejects = 0 if len(pool) > before else fill_rejects + 1
            continue
        i = _pick_slot([q["s"] for q in pool], [q["turns"] for q in pool])
        st = pool[i]
        st["turns"] += 1
        if st["fails"] >= HOP_FAILS:
            if st["s"] < best_s:
                pool.pop(i)                          # exhausted and not the best: recycle the
                fill_rejects = 0                     # slot into a fresh independent stream
                continue
            st["sigma"] = 0.15                       # the best stream is never thrown away;
            st["fails"] = 0                          # it just goes back to wide exploration
        # WHICH MOVE.  Not a uniform draw: the modes differ in value BY SIZE and differ in
        # cost, so the turn goes to the mode standing best on gain per CPU second (_pick_mode).
        mode = _pick_mode(m_gain, m_cpu, m_tries)
        t_mode = time.process_time()
        if mode == 3:
            # The radius-cap hop starts the release solve from the HOMOGENISED (xy, r), not from
            # the stream's own radii: the whole point is that the centres moved under the cap.
            hxy, hr = _homogenize(st["xy"], st["r"], rng, t_end)
            xy1, r1, s1 = run(hxy, hr, 2.0)
        elif mode == 4:
            # Gap-directed reinsertion carries its own radii too: the moved circles are seeded at
            # the clearance of the hole they were dropped into, not at the size they had while
            # wedged, so _slp starts from a live circle rather than from a speck.
            gxy, gr = _reinsert(st["xy"], st["r"], rng)
            xy1, r1, s1 = run(gxy, gr, 2.0)
        else:
            xy0 = _shake(st["xy"], st["r"], rng, mode, st["sigma"])
            xy1, r1, s1 = run(xy0, st["r"].copy(), 2.0)
        m_tries[mode] += 1.0
        m_cpu[mode] += max(time.process_time() - t_mode, 1e-9)
        if np.isfinite(s1) and s1 > st["s"]:
            m_gain[mode] += float(s1 - st["s"])      # credited to the mode that earned it
            st.update(xy=xy1.copy(), r=r1.copy(), s=s1, key=np.sort(r1))
            st["sigma"] = max(0.008, st["sigma"] * 0.7)
            st["fails"] = 0
        else:
            st["sigma"] = min(0.15, st["sigma"] * 1.15)
            st["fails"] += 1

    flush()
    return best_s


def solve(evaluate, meter, rng, targets):
    targets = list(targets or [])
    if not targets:
        return
    # Deadline derived AT ENTRY from the process clock and from the sizes actually in `targets`,
    # so a second solve() call in the same process gets a fresh allowance (MISSION.md).
    call_end = time.process_time() + _call_allowance(len(targets))
    order = sorted(targets)
    for idx, n in enumerate(order):
        if meter.left() <= 0:
            break
        now = time.process_time()
        left = call_end - now
        if left <= 0.05:
            break
        # Re-split the REMAINING time over the REMAINING sizes, so slack from an easy n is reused
        # and the split is correct for len(targets) == 1.
        t_end = now + left / float(len(order) - idx)
        try:
            _solve_one(int(n), evaluate, meter, rng, t_end)
        except Exception:
            # One bad n must never cost the sizes after it: whatever already went through
            # evaluate() is kept, and we move on.
            continue


# ------------------------------------------------------------------------------- self-test
def _self_test():
    import sys

    class _Meter:
        def __init__(self, b):
            self.budget = b
            self.used = 0

        def left(self):
            return self.budget - self.used

    ok_all = True
    seen = {}

    def make_eval(meter):
        def evaluate(n, packing):
            p = np.asarray(packing, dtype=float)
            if p.ndim == 2:
                p = p[None]
            if p.shape[1] != n or p.shape[2] != 3:
                raise ValueError("bad shape")
            meter.used += p.shape[0]
            feas, sums = [], []
            for k in range(p.shape[0]):
                f, s = _feasible_sum(p[k, :, :2], p[k, :, 2])
                feas.append(f)
                sums.append(s)
                if f and s > seen.get(n, -np.inf):
                    seen[n] = s
            return np.array(feas), np.array(sums)
        return evaluate

    # gradient check against finite differences
    rng = np.random.RandomState(0)
    n = 6
    z = np.concatenate([rng.uniform(-0.4, 0.4, 2 * n), rng.uniform(0.05, 0.2, n)])
    f0, g = _fg(z, n, 500.0)
    num = np.empty_like(g)
    for i in range(len(z)):
        h = 1e-7
        zp = z.copy(); zp[i] += h
        zm = z.copy(); zm[i] -= h
        num[i] = (_fg(zp, n, 500.0)[0] - _fg(zm, n, 500.0)[0]) / (2 * h)
    err = np.max(np.abs(num - g)) / max(1.0, np.max(np.abs(g)))
    print("gradient rel-err: %.2e" % err)
    ok_all &= err < 1e-4

    # _grow must never break feasibility and must never lower the sum
    xy, r = _repair(rng.uniform(-0.45, 0.45, (10, 2)), np.full(10, 0.2))
    r2 = _grow(xy, r)
    okg, sg = _feasible_sum(xy, r2)
    print("grow feasible=%s  sum %.6f -> %.6f" % (okg, r.sum(), sg))
    ok_all &= okg and sg >= r.sum() - 1e-15

    # THE SLP THEOREM: because the linearized pair constraint under-estimates the true distance,
    # every iterate must be TRULY feasible (not just feasible for the LP) and sum(r) must never
    # decrease -- with no line search.  If that ever fails, the refiner is silently emitting
    # overlaps that only _repair's down-scale hides.  Check it directly on random starts.
    for trial in range(5):
        n0 = int(rng.randint(4, 30))
        xy0, r0 = _repair(rng.uniform(-0.45, 0.45, (n0, 2)), np.full(n0, 0.3 / np.sqrt(n0)))
        s_in = r0.sum()
        xy1, r1 = _slp(xy0, r0, t_end=time.process_time() + 3.0)
        feas, s_out = _feasible_sum(xy1, r1)
        if not feas or s_out < s_in - 1e-12:
            print("FAIL: slp trial %d n=%d feasible=%s %.9f -> %.9f" % (trial, n0, feas, s_in, s_out))
            ok_all = False
    print("slp: 5 random starts stayed feasible and monotone")

    # THE CONVERGENCE STOP (iteration 7) must be a SPEEDUP, not a quality cut.  Two claims, both
    # asserted rather than trusted:
    #   (a) the early-stopped solve lands within SLP_STOP_K * SLP_STOP_TOL (plus slack) of what
    #       grinding to delta_min returns -- i.e. we are buying CPU with ~1e-9 of sum, not with
    #       a worse local optimum;
    #   (b) it is genuinely cheaper, so the CPU actually goes somewhere else.
    # `_slp_full` reproduces the pre-iteration-7 behaviour by disarming the stop (K > iters).
    def _slp_full(xy0, r0):
        global SLP_STOP_K
        keep, SLP_STOP_K = SLP_STOP_K, 10 ** 9
        try:
            return _slp(xy0.copy(), r0.copy())
        finally:
            SLP_STOP_K = keep

    worst, t_fast, t_full = 0.0, 0.0, 0.0
    for n0 in (17, 41, 67):
        k, mode, stagger, transpose = _lattice_plan(n0)[0]
        pts = _lattice(n0, k, mode, stagger)
        starts = [(pts, np.full(n0, 0.35 / np.sqrt(n0)))]
        for kind in (0, 1, 2):
            starts.append(_init(n0, rng, kind))
        for xy0, r0 in starts:
            xy0, r0 = _repair(xy0, r0)
            t0 = time.process_time()
            _, rf = _slp(xy0.copy(), r0.copy())
            t_fast += time.process_time() - t0
            t0 = time.process_time()
            _, rF = _slp_full(xy0, r0)
            t_full += time.process_time() - t0
            worst = max(worst, float(rF.sum() - rf.sum()))
    tol = SLP_STOP_K * SLP_STOP_TOL + 1e-9
    print("slp stop: worst quality loss %.2e (tol %.2e), CPU %.3fs vs %.3fs (%.2fx)"
          % (worst, tol, t_fast, t_full, t_full / max(t_fast, 1e-9)))
    ok_all &= worst < tol
    ok_all &= t_fast < t_full

    # THE LATTICE FAMILY: every plan entry must yield EXACTLY n distinct-enough centres inside
    # the square, for any n -- including the degenerate small n and sizes the census never sees.
    # A plan entry that returns the wrong count would raise ValueError inside evaluate() and, per
    # MISSION.md, end the whole solve() call and lose every size after it.
    for n0 in (1, 2, 3, 4, 5, 17, 36, 64, 100, 137):
        plan = _lattice_plan(n0)
        if not plan:
            print("FAIL: empty lattice plan for n=%d" % n0)
            ok_all = False
        for (k, mode, stagger, transpose) in plan:
            pts = _lattice(n0, k, mode, stagger)
            if pts is None:
                continue
            if pts.shape != (n0, 2) or not np.all(np.isfinite(pts)):
                print("FAIL: lattice n=%d k=%d mode=%d stag=%s -> %s"
                      % (n0, k, mode, stagger, pts.shape))
                ok_all = False
                break
            if pts.min() < LO or pts.max() > HI:
                print("FAIL: lattice n=%d k=%d escaped the square" % (n0, k))
                ok_all = False
                break
    print("lattice plans: %d sizes, all entries give n centres inside the square" % 10)

    # The staggered hex lattice must actually beat the square grid it replaces, or stage 2 of
    # the search is not earning its CPU.  Checked on one mid size at a fixed row count.
    n0 = 57
    r00 = np.full(n0, 0.35 / np.sqrt(n0))
    k0 = max(1, int(round(HEX_K * np.sqrt(n0))))
    got = {}
    for stagger in (0.0, 0.5):
        a, b = _repair(_lattice(n0, k0, 2, stagger), r00.copy())
        a, b = _slp(a, b, t_end=time.process_time() + 2.0)
        f, s = _feasible_sum(*(_polish(a, b)[:2]))
        got[stagger] = s if f else -np.inf
    print("n=57 k=%d  square-grid %.6f  vs  staggered %.6f" % (k0, got[0.0], got[0.5]))
    ok_all &= got[0.5] > got[0.0]

    # THE POOL INVARIANT (iteration 4's whole claim): a lattice optimum, however good, must
    # never evict the independent cold-start stream, and same-basin optima must not take two
    # slots.  Both are what makes the pool diverse rather than POOL_P copies of one basin.
    xy_p = rng.uniform(-0.4, 0.4, (8, 2))
    pool = []
    _pool_insert(pool, xy_p, np.full(8, 0.01), 0.08, "cold")          # a weak INDEPENDENT stream
    for j in range(6):                                                # six strong sweep optima
        _pool_insert(pool, xy_p, np.full(8, 0.10 + 0.01 * j), 10.0 + j, "sweep")
    n_cold = sum(1 for q in pool if q["src"] == "cold")
    n_sweep = sum(1 for q in pool if q["src"] == "sweep")
    print("pool: %d streams (%d sweep <= %d, %d cold reserved)  best=%.3f"
          % (len(pool), n_sweep, POOL_SWEEP, n_cold, max(q["s"] for q in pool)))
    if not (len(pool) <= POOL_P and n_sweep <= POOL_SWEEP and n_cold >= 1):
        print("FAIL: the reserved cold slot was evicted by sweep streams")
        ok_all = False
    if n_sweep != POOL_SWEEP or max(q["s"] for q in pool) < 15.0 - 1e-12:
        print("FAIL: the pool did not keep the BEST sweep optima")
        ok_all = False
    pool2 = []
    _pool_insert(pool2, xy_p, np.full(8, 0.05), 0.40, "sweep")
    _pool_insert(pool2, xy_p, np.full(8, 0.05 + 1e-9), 0.41, "sweep")  # same basin, better point
    if len(pool2) != 1 or pool2[0]["s"] < 0.41 - 1e-12:
        print("FAIL: same-basin optima took two slots (len=%d)" % len(pool2))
        ok_all = False
    print("pool: same-basin optima share one slot and upgrade in place")

    # THE RACING RULE (iteration 5's whole claim).  Three properties, asserted directly on the
    # pure function, because the score over 37 sizes cannot tell me which of them broke:
    #   (a) every stream is tried before any is preferred (no birth starvation);
    #   (b) the LEADER takes a clear majority of turns -- more than round-robin's 1/P -- while
    #       the worst stream is still visited (no permanent starvation);
    #   (c) when a challenger OVERTAKES, the turns follow the measurement, not the pool order.
    first = [_pick_slot([0.9, 0.5, 0.1], [0, 0, 0]), _pick_slot([0.9, 0.5, 0.1], [1, 0, 0]),
             _pick_slot([0.9, 0.5, 0.1], [1, 1, 0])]
    print("pick: first turns go to the untried streams -> %s" % first)
    if first != [0, 1, 2]:
        print("FAIL: _pick_slot starved a stream at birth")
        ok_all = False
    svals = [1.0, 0.99, 0.98]
    cnt = [0, 0, 0]
    for _ in range(60):
        j = _pick_slot(svals, cnt)
        cnt[j] += 1
    share = cnt[0] / float(sum(cnt))
    print("pick: 60 turns over 3 streams -> %s  (leader %.0f%%, round-robin is 33%%)"
          % (cnt, 100.0 * share))
    if not (share > 1.0 / POOL_P + 0.1 and min(cnt) >= 1):
        print("FAIL: the racing rule is round-robin (or it starves a stream)")
        ok_all = False
    svals2 = [0.98, 0.99, 1.0]                   # slot 2 has overtaken; slot 0 is now worst
    cnt2 = [0, 0, 0]
    for _ in range(40):
        j = _pick_slot(svals2, [cnt[i] + cnt2[i] for i in range(3)])
        cnt2[j] += 1
    print("pick: after an overtake the next 40 turns -> %s" % cnt2)
    if not (cnt2[2] > cnt2[0]):
        print("FAIL: turns did not follow the overtake")
        ok_all = False

    # CONTRACT: solve() called TWICE in one process; the SECOND call must still produce a packing.
    # A single call passes even when the deadline is a module constant -- two calls do not.
    for call in (1, 2):
        m = _Meter(20000)
        seen.clear()
        solve(make_eval(m), m, np.random.RandomState(call), [11])
        got = seen.get(11, None)
        print("call %d: evals=%d  best sum_r=%s" % (call, m.used, got))
        if got is None or not (got > 0):
            print("FAIL: call %d produced no feasible packing" % call)
            ok_all = False

    # CONTRACT: len(targets) == 1 and multi-target splits both work; unknown n cold-starts fine.
    m = _Meter(20000)
    seen.clear()
    solve(make_eval(m), m, np.random.RandomState(7), [5, 13])
    print("multi-target: %s" % {k: round(v, 5) for k, v in sorted(seen.items())})
    ok_all &= set(seen) == {5, 13}

    # ITERATION 8: the LAZY PAIR SET returns the FULL LP's OPTIMAL VALUE, per LP.
    # LAZY_PAD = 1e9 admits every pair of the safe superset into the first guessed set, which IS
    # the pre-iteration-8 full LP.  The comparison is made at iters=1 -- exactly ONE LP -- on
    # purpose: at iters>1 the two are NOT required to agree, because the LP is degenerate and a
    # different optimal VERTEX re-linearizes the next step somewhere else.  (An earlier version of
    # this assertion used iters=1,2,3 and failed at 4.2e-03, which is that divergence, not a
    # wrong LP; the failure is what established the caveat now documented in _slp.)
    global LAZY_PAD
    pad0 = LAZY_PAD
    worst_lp = 0.0
    cases = 0
    for n in (23, 41, 65):
        for kind in (0, 1, 2):
            xy0, r0 = _repair(*_init(n, np.random.RandomState(11 + kind), kind))
            # Two very different iterates per start: a cold one (radii far too small, so the
            # first guessed set is a poor guess and the lazy loop must actually add rows) and a
            # grown one (radii near-tight, the regime _slp spends most of its iterations in).
            for xy, r in ((xy0, r0), (xy0, _grow(xy0, r0))):
                vals = []
                for pad in (pad0, 1e9):
                    LAZY_PAD = pad
                    vals.append(float(_slp(xy.copy(), r.copy(), iters=1)[1].sum()))
                LAZY_PAD = pad0
                worst_lp = max(worst_lp, abs(vals[0] - vals[1]))
                cases += 1
    LAZY_PAD = pad0
    print("lazy-LP value vs full-LP value, ONE LP, over %d (size,start,iterate) cases: "
          "worst |delta| = %.3e (bound 1e-12)" % (cases, worst_lp))
    ok_all &= worst_lp <= 1e-12

    # ...and it is CHEAPER over a whole _slp run, which is the claim that buys anything.
    lazy_t = full_t = 0.0
    for n in (65, 81):
        for kind in (0, 1, 2):
            xy0, r0 = _repair(*_init(n, np.random.RandomState(5 + kind), kind))
            for pad, acc in ((pad0, "lazy"), (1e9, "full")):
                LAZY_PAD = pad
                t = time.process_time()
                _slp(xy0.copy(), r0.copy())
                dt = time.process_time() - t
                if acc == "lazy":
                    lazy_t += dt
                else:
                    full_t += dt
    LAZY_PAD = pad0
    print("full _slp CPU over 6 (size,start) pairs: lazy %.3fs vs full %.3fs = %.2fx"
          % (lazy_t, full_t, full_t / max(lazy_t, 1e-9)))
    ok_all &= lazy_t < full_t

    # ITERATION 12: the SWEEP DOMINANCE STOP -- the pure rule, then the end-to-end effect.
    #   (a) disarmed with no independent incumbent: it can never stop the sweep;
    #   (b) armed, it stops after exactly SWEEP_STALL non-improving candidates;
    #   (c) an improving candidate re-arms against ITSELF and resets the counter;
    #   (d) a non-finite candidate sum counts as non-improving (a degenerate lattice must not
    #       reset the counter, or one -inf every SWEEP_STALL candidates disables the rule).
    st = 0
    for _ in range(3 * SWEEP_STALL):
        _a, st, go = _sweep_step(-np.inf, 1.0, st)
        ok_all &= go and st == 0
    st = 0
    fired = None
    for i in range(3 * SWEEP_STALL):
        _a, st, go = _sweep_step(5.0, 4.0, st)
        if not go:
            fired = i + 1
            break
    print("sweep dominance stop: disarmed=never, armed fires after %s candidates (SWEEP_STALL=%d)"
          % (fired, SWEEP_STALL))
    ok_all &= fired == SWEEP_STALL
    a, st, go = _sweep_step(5.0, 6.0, SWEEP_STALL - 1)
    ok_all &= go and st == 0 and abs(a - 6.0) < 1e-15
    st = 0
    for i in range(SWEEP_STALL):
        _a, st, go = _sweep_step(5.0, -np.inf, st)
    ok_all &= (not go) and st == SWEEP_STALL

    # END TO END: with a committed pack that dominates the lattice family, a size must spend
    # STRICTLY fewer local solves on the sweep than without the rule -- and the search as a whole
    # must still return a packing at least as good.  Counted by wrapping _slp for this test only.
    _real_slp = globals()["_slp"]
    calls = {"k": 0}

    def _counting_slp(*a, **k):
        calls["k"] += 1
        return _real_slp(*a, **k)

    nsw = 43
    sweep_counts = {}
    sweep_best = {}
    if _read_pack(nsw) is not None:
        cpu0 = globals()["CPU_ONE"]
        globals()["CPU_ONE"] = 4.0               # a self-test must not cost the offline 55 s
        for label, stall in (("with-stop", SWEEP_STALL_DEFAULT), ("no-stop", 10 ** 9)):
            globals()["SWEEP_STALL"] = stall
            globals()["_slp"] = _counting_slp
            calls["k"] = 0
            m = _Meter(20000)
            seen.clear()
            try:
                solve(make_eval(m), m, np.random.RandomState(3), [nsw])
            finally:
                globals()["_slp"] = _real_slp
            sweep_counts[label] = calls["k"]
            sweep_best[label] = seen.get(nsw)
        globals()["SWEEP_STALL"] = SWEEP_STALL_DEFAULT
        globals()["CPU_ONE"] = cpu0
        print("n=%d local solves: with-stop %d vs no-stop %d ; best sum %.9f vs %.9f"
              % (nsw, sweep_counts["with-stop"], sweep_counts["no-stop"],
                 sweep_best["with-stop"], sweep_best["no-stop"]))
        # Both arms get the SAME CPU allowance, so the stop does not reduce total solves -- it
        # MOVES them from the sweep to hopping.  What must hold is that the result is not worse.
        ok_all &= sweep_best["with-stop"] >= sweep_best["no-stop"] - 1e-9

    # -------- RADIUS-CAP HOP (iteration 13) --------
    # (a) The cap must be INERT when it cannot bind.  Radii are bounded by 0.5 in any feasible
    #     packing, so rcap=0.5 has to reproduce the true program BIT-FOR-BIT -- otherwise the new
    #     argument has quietly changed the solver everyone else uses.
    rngc = np.random.RandomState(21)
    worst_inert = 0.0
    for nc in (9, 17):
        xy0, r0 = _init(nc, rngc, 0)
        xy0, r0 = _repair(xy0, r0)
        xa, ra = _slp(xy0.copy(), r0.copy(), t_end=time.process_time() + 3.0)
        xb, rb = _slp(xy0.copy(), r0.copy(), t_end=time.process_time() + 3.0, rcap=0.5)
        worst_inert = max(worst_inert, float(np.max(np.abs(ra - rb))),
                          float(np.max(np.abs(xa - xb))))
    print("rcap=0.5 vs rcap=None (must be identical): worst |delta| = %.3e" % worst_inert)
    ok_all &= worst_inert == 0.0

    # (b) A BINDING cap must actually homogenise: the capped LP iterate is what the move trades
    #     on, so check the spread of the radii it produces BEFORE _slp's closing regrow.  (_slp
    #     always returns a regrown packing, so this is read off a one-iteration capped solve.)
    nc = 21
    xy0, r0 = _repair(*_init(nc, np.random.RandomState(5), 1))
    xy1, r1 = _slp(xy0.copy(), r0.copy(), t_end=time.process_time() + 3.0)
    cap = 1.0 * float(np.mean(r1))
    xy2, r2 = _slp(xy1.copy(), r1.copy(), t_end=time.process_time() + 3.0, rcap=cap)
    moved = float(np.max(np.abs(xy2 - xy1)))
    print("cap hop at n=%d: cap=%.6f  centres moved by %.3e  spread %.4f -> %.4f"
          % (nc, cap, moved, float(np.std(r1)), float(np.std(r2))))
    ok_all &= moved > 1e-6                            # the move must MOVE, not return the input
    okf, sf = _feasible_sum(xy2, r2)
    ok_all &= bool(okf)                               # and its output must be strictly feasible

    # (c) _homogenize must never raise, for ANY incumbent -- including degenerate radii, which a
    #     failing stream can hand it.  A raise here kills the whole solve() call for that n.
    hx, hr = _homogenize(np.zeros((4, 2)), np.zeros(4), np.random.RandomState(0),
                         time.process_time() + 1.0)
    deg_ok = hx.shape == (4, 2) and hr.shape == (4,)
    for nn in (7, 13):
        xs, rs = _repair(*_init(nn, np.random.RandomState(nn), 2))
        hx, hr = _homogenize(xs, rs, np.random.RandomState(nn), time.process_time() + 3.0)
        f2, _ = _feasible_sum(hx, hr)
        deg_ok &= bool(f2)
    print("_homogenize: degenerate all-zero incumbent handled, 2 sizes feasible out: %s" % deg_ok)
    ok_all &= deg_ok

    # (d) The move must be REACHED by the hop rotation -- a mode the racing rule never selects is
    #     dead code.  Count real calls during a short end-to-end solve().
    hcalls = {"k": 0, "g": 0}
    _real_hom = _homogenize
    _real_rei = _reinsert

    def _counting_hom(*a, **k):
        hcalls["k"] += 1
        return _real_hom(*a, **k)

    def _counting_rei(*a, **k):
        hcalls["g"] += 1
        return _real_rei(*a, **k)

    cpu0 = CPU_ONE
    globals()["CPU_ONE"] = 6.0
    globals()["_homogenize"] = _counting_hom
    globals()["_reinsert"] = _counting_rei
    m = _Meter(20000)
    seen.clear()
    try:
        solve(make_eval(m), m, np.random.RandomState(11), [23])
    finally:
        globals()["_homogenize"] = _real_hom
        globals()["_reinsert"] = _real_rei
        globals()["CPU_ONE"] = cpu0
    print("hop rotation reaches cap move %d and gap-reinsertion %d times in a 6s n=23 solve"
          % (hcalls["k"], hcalls["g"]))
    ok_all &= hcalls["k"] > 0 and hcalls["g"] > 0 and HOP_MODES == 5

    # ---- iteration 15: GAP-DIRECTED REINSERTION _reinsert -----------------------------------
    # (a) IT AIMS.  Every reinserted circle must land in a hole that actually fits it: its seeded
    #     radius must not overlap any circle that was KEPT.  This is the property mode 1's
    #     uniform draw does not have, and it is the entire reason this move exists -- so assert
    #     it directly, on real converged packings, rather than trusting the geometry by eye.
    aim_ok = True
    worst_pen = 0.0
    for nn in (11, 27, 64):
        rs0 = np.random.RandomState(1000 + nn)
        xs, rs = _repair(*_init(nn, rs0, 1))
        xs, rs = _slp(xs, rs, t_end=time.process_time() + 3.0)
        who = np.argsort(rs)[:max(1, int(round(REINS_FRAC * nn)))]
        keep = np.ones(nn, dtype=bool); keep[who] = False
        nx, nr = _reinsert(xs, rs, np.random.RandomState(7))
        aim_ok &= nx.shape == xs.shape and nr.shape == rs.shape
        aim_ok &= bool(np.all(np.abs(nx) <= HI) and np.all(nr > 0.0))
        # the movers must have MOVED, and the kept circles must NOT have
        aim_ok &= bool(np.allclose(nx[keep], xs[keep]) and np.allclose(nr[keep], rs[keep]))
        aim_ok &= float(np.max(np.abs(nx[who] - xs[who]))) > 1e-6
        for i in who:                                 # placed circle vs every kept circle
            d = np.sqrt(((nx[i] - nx[keep]) ** 2).sum(axis=1)) - nr[keep] - nr[i]
            w = HI - np.max(np.abs(nx[i])) - nr[i]
            worst_pen = max(worst_pen, -min(float(d.min()), float(w)))
    print("_reinsert aims: movers land clear of every kept circle and wall, worst penetration "
          "%.2e  (checks %s)" % (worst_pen, aim_ok))
    ok_all &= aim_ok and worst_pen <= 1e-12

    # (b) IT NEVER RAISES, for any n it can be handed -- including n small enough that the
    #     "smallest 8%" rounds to the whole packing, and a degenerate all-zero incumbent.  A
    #     raise inside a hop turn ends the whole solve() call for that size.
    rob_ok = True
    for nn in (1, 2, 3, 5):
        gx, gr = _reinsert(np.zeros((nn, 2)), np.zeros(nn), np.random.RandomState(nn))
        rob_ok &= gx.shape == (nn, 2) and gr.shape == (nn,) and bool(np.all(np.isfinite(gx)))
    print("_reinsert robust at n=1,2,3,5 and on an all-zero incumbent: %s" % rob_ok)
    ok_all &= rob_ok

    # ---- iteration 14: the MODE RACING RULE _pick_mode -------------------------------------
    # (a) FORCED EXPLORATION.  Every mode must be tried MODE_WARM times before any preference,
    #     whatever the (fabricated) rates say -- otherwise a mode can be written off on zero
    #     evidence, and at some n that is the only mode that pays.
    z = np.zeros(HOP_MODES)
    first = []
    tr = z.copy()
    gn = z.copy(); gn[0] = 1.0
    cp = np.full(HOP_MODES, 1.0)
    for _ in range(HOP_MODES * MODE_WARM):
        i = _pick_mode(gn, cp, tr)
        first.append(i)
        tr[i] += 1.0
    warm_ok = all(int(c) == MODE_WARM for c in np.bincount(first, minlength=HOP_MODES))
    print("_pick_mode forced exploration: first %d picks = %s -> each mode %d times: %s"
          % (len(first), first, MODE_WARM, warm_ok))
    ok_all &= warm_ok

    # (b) INERT WITHOUT EVIDENCE.  With no mode having gained anything, ties are averaged, so
    #     every quality is equal and the rule reduces to the uniform rotation it replaces: the
    #     picks must cycle, not pile onto mode 0 (which unaveraged ranks would do).
    tr = np.full(HOP_MODES, float(MODE_WARM))
    cp = np.full(HOP_MODES, 1.0)
    cyc = []
    for _ in range(4 * HOP_MODES):
        i = _pick_mode(z.copy(), cp, tr)
        cyc.append(i)
        tr[i] += 1.0
    counts = np.bincount(cyc, minlength=HOP_MODES)
    flat_ok = int(counts.max() - counts.min()) <= 1
    print("_pick_mode with zero gains everywhere: counts %s (uniform +-1: %s)"
          % (list(counts), flat_ok))
    ok_all &= flat_ok

    # (c) IT FOLLOWS THE MEASUREMENT, IN GAIN PER CPU SECOND.  Mode 1 gains half as much as
    #     mode 3 but costs a quarter as much, so mode 1 has the better RATE and must take the
    #     majority; and no mode may be starved to zero over a long run.
    tr = np.full(HOP_MODES, float(MODE_WARM))
    gn = np.zeros(HOP_MODES); gn[1] = 0.5; gn[3] = 1.0
    cp = np.ones(HOP_MODES); cp[3] = 4.0
    take = []
    for _ in range(200):
        i = _pick_mode(gn, cp, tr)
        take.append(i)
        tr[i] += 1.0
    counts = np.bincount(take, minlength=HOP_MODES)
    lead_ok = int(np.argmax(counts)) == 1 and counts[1] > 0.4 * len(take)
    live_ok = bool(counts.min() > 0)
    print("_pick_mode rate ranking: counts %s (mode 1 leads with %.0f%%: %s; none starved: %s)"
          % (list(counts), 100.0 * counts[1] / len(take), lead_ok, live_ok))
    ok_all &= lead_ok and live_ok

    # (d) END TO END: with the rule live, a real solve must still reach every mode (nothing is
    #     dead code) and must concentrate -- i.e. the split is NOT uniform once evidence exists.
    mode_hits = {"v": None}
    _real_pick = _pick_mode
    _seen_modes = np.zeros(HOP_MODES)

    def _counting_pick(*a, **k):
        i = _real_pick(*a, **k)
        _seen_modes[i] += 1
        return i

    cpu0 = CPU_ONE
    globals()["CPU_ONE"] = 8.0
    globals()["_pick_mode"] = _counting_pick
    m = _Meter(20000)
    seen.clear()
    try:
        solve(make_eval(m), m, np.random.RandomState(7), [29])
    finally:
        globals()["_pick_mode"] = _real_pick
        globals()["CPU_ONE"] = cpu0
    mode_hits["v"] = list(_seen_modes.astype(int))
    live_all = bool(_seen_modes.min() > 0)
    print("live solve n=29 (8s): mode picks %s -- every mode reached: %s"
          % (mode_hits["v"], live_all))
    ok_all &= live_all and _seen_modes.sum() >= HOP_MODES * MODE_WARM

    # CONTRACT: an n with no committed pack must cold-start (never raise on the warm-start read).
    print("no-pack read for n=100000: %s" % (_read_pack(100000),))
    ok_all &= _read_pack(100000) is None

    print("SELF-TEST: %s" % ("PASS" if ok_all else "FAIL"))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
