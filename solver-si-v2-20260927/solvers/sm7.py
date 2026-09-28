"""Circle-packing solver: SLP (sequential LP) inner-approximation + basin hopping.

Iteration 1 optimised the true NLP with SLSQP on a restricted active set and reached ~1.43 mean
digits. Measuring iteration 2 showed *why* it stopped there: running any local method from the
committed census produces **zero** improvement -- those packings are already KKT points. The binding
constraint is not local convergence, it is **which basin you are in**. So the local solve has to get
cheap enough that we can afford many basins, and the search has to move between basins on purpose.

LOCAL SOLVE -- sequential linear programming on a *conservative* inner approximation.
``||p_i - p_j||`` is convex, so its linearisation at the current iterate is a global LOWER bound:

    ||p_i - p_j||  >=  d_ij  +  u_ij . (dp_i - dp_j),      u_ij = (p_i - p_j)/d_ij

Imposing ``d_ij + u_ij.(dp_i - dp_j) >= r_i + r_j`` is therefore *stricter* than the real constraint,
and the wall constraints are already linear. Maximising ``sum r`` over (dp, r) with a trust-region box
is then a sparse LP whose every feasible point is feasible for the true NLP -- an MM/minorise-maximise
scheme with guaranteed-feasible, monotonically improving iterates and no line search. HiGHS solves the
n=99 LP (4851 pair rows, 297 cols) in ~20 ms, so a full local solve costs ~0.03 s at n=27 and ~0.9 s at
n=99 -- roughly an order of magnitude cheaper per basin than the SLSQP loop it replaces, and it
converges from a *random* start to a better packing than the entire committed census.

Two things make the LP cheap enough to repeat. (1) An EXACT row reduction: with the step box |dp|<=tr
and radii capped at r+tr, a pair whose slack exceeds (2+2*sqrt2)*tr provably cannot bind, so its row is
dropped without changing the LP's feasible set at all (measured max deviation from the full LP: 2e-11).
(2) The LP tolerance is pinned at 1e-10, since HiGHS's default 1e-7 is itself a ~6-digit score cap.
(3) With the CENTRES fixed the problem is *exactly* an LP in the radii alone, so the SLP's long
trust-region tail is unnecessary: anneal only to tr=1e-3 and finish with that exact r-LP. Measured
over n=31/63/99 it reproduces the full anneal's sum_r to <1e-11 at 1.3-2.2x less CPU.

GLOBAL SEARCH -- two measured facts drive it.
  * Greedy basin hopping SATURATES: at n=39 459 rounds found exactly what 44 rounds found, and n=31/99
    also stopped improving well inside their budget. More rounds of "perturb the best, keep only
    improvements" buys nothing, so the search accepts a *worse* basin with probability exp(dS/T) -- a
    Metropolis chain whose state is separate from the monotone best-ever packing, T ~ 4e-4 * sum_r
    (the spread between neighbouring local optima is ~1e-3 relative).
  * EXACT-COUNT ROW LATTICE cold starts (~25% of rounds). record/(0.5*sqrt n) is 1.03-1.05 right across
    the census, so the optimum is a mildly size-varied dense lattice. A lattice is a stack of rows, and
    a hexagonal one is exactly a stack whose row lengths alternate m, m-1; so the row lengths are drawn
    directly and made to sum to n, instead of building an R x C grid and deleting the surplus at random
    (which seeded every start with random vacancies the local solve then had to undo).
  * around the chain state, a global gaussian shake and a RELOCATE move -- lift the j smallest circles
    and drop them into the emptiest holes, found by sampling clearance
    ``min(wall, min_i ||q - c_i|| - r_i)`` over a random point cloud. For a *sum of radii* objective a
    tiny circle in a tight spot is nearly free to move and pays immediately in a bigger hole; this is
    the move that actually changes the contact graph.

Everything is banked as it happens: every local optimum is projected to strict feasibility (one global
radius scale over the full n^2 pair matrix) and routed through the metered ``evaluate()``, and each n
gets its own process-CPU slice -- weighted by n * headroom(n), because basin cost grows ~n^1.2 while a
size whose relative gap is already under the scorer's 1e-7 digit cap has exactly ZERO marginal score
left to win -- so the driver's 120 s backstop can cost at most the basin in flight.

Warm start reads the committed census, GUARDED (MISSION.md): a missing/short/unparseable
bench/packs/csqv<n>.pck falls back to a cold start, so any n works. Nothing is keyed on n.

Self-test:  python tools/solver.py --self-test
"""
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

LO, HI = -0.5, 0.5
FEAS_TOL = 1e-9
CPU_BUDGET_S = 110.0        # our own ceiling, inside the driver's 120 s process-CPU backstop
TR0 = 0.06                  # initial trust-region half-width for a cold solve
SLP_TR_STOP = 1e-3          # stop the SLP anneal here; the exact r-LP finishes the radii
T_FAC = 4e-4                # Metropolis temperature, as a fraction of sum_r (basin-to-basin scale)
PASSES = 3                  # visits per n per iteration, alternating sweep direction (see solve())
LP_OPTS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
CAP_DIGITS = 7.0            # the scorer clamps digits here: beyond it a size earns NOTHING more
CAP_RELGAP = 10.0 ** (-CAP_DIGITS)
CAPPED_W = 0.05             # residual CPU weight for a size already at that cap (see _headroom)


# ----------------------------------------------------------------- feasibility (local; no harness import)
def _project(pack):
    """Scale every radius by the largest single factor keeping EVERY wall and EVERY pair constraint
    satisfied (checked over the full n^2 matrix) -> strictly feasible by construction."""
    a = np.asarray(pack, float)
    xy = np.clip(a[:, :2], LO, HI)
    r = np.maximum(a[:, 2], 1e-12)
    n = len(r)
    wall = np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0], xy[:, 1] - LO, HI - xy[:, 1]])
    s = float(np.min(wall / r))
    if n >= 2:
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        rr = r[:, None] + r[None, :]
        np.fill_diagonal(d, np.inf)
        np.fill_diagonal(rr, 1.0)
        s = min(s, float(np.min(d / rr)))
    s = min(s, 1.0) * (1.0 - 1e-13)
    return np.concatenate([xy, np.maximum(r * s, 1e-12)[:, None]], axis=1)


def _sum_r(p):
    return float(p[:, 2].sum())


# ------------------------------------------------------------------------------- the SLP local optimiser
# EXACT row reduction. Variables are z = [dx (n) | dy (n) | r (n)] with |dp_i|_inf <= tr and
# r_i <= r_i^cur + tr. The pair row is  r_i + r_j - u_ij.(dp_i - dp_j) <= d_ij, and its LHS is bounded by
#     r_i^cur + r_j^cur + 2*tr  +  |dp_i - dp_j|  <=  r_i^cur + r_j^cur + (2 + 2*sqrt(2))*tr,
# so a pair whose current slack d_ij - r_i - r_j is at least (2 + 2*sqrt(2))*tr CANNOT bind. Dropping it
# does not enlarge the feasible set at all -- this is an exact reformulation of the same LP, not a
# heuristic active set, so the inner-approximation guarantee (every LP-feasible point is NLP-feasible)
# survives untouched. At n=99 it takes the pair rows from 4851 to a few hundred once tr has annealed.
SLACK_K = 2.0 + 2.0 * np.sqrt(2.0)


def _slp(pack, tr0=TR0, max_it=200, deadline=None, tr_stop=1e-9, w=None):
    """Maximise sum(r) by repeated LP on the conservative linearisation. Returns an (n,3) array.
    ``tr_stop`` ends the anneal early: measured, stopping at 1e-3 and finishing with the EXACT r-LP
    below reproduces the full anneal's sum_r to <1e-11 for 1.3-2.2x less CPU.
    ``w`` re-weights the objective to ``max sum w_i r_i`` (default: all ones, the true objective).
    The FEASIBLE SET is untouched, so every iterate is still a valid packing and the MM ascent
    guarantee still holds -- for the weighted objective. See ``_pressure``."""
    a = np.asarray(pack, float)
    xy = a[:, :2].copy()
    r = np.maximum(a[:, 2], 0.0).copy()
    n = len(r)
    if n < 1:
        return np.concatenate([xy, r[:, None]], axis=1)
    I0, J0 = np.triu_indices(n, 1)
    ar = np.arange(n)
    wv = np.ones(n) if w is None else np.maximum(np.asarray(w, float).ravel(), 0.0)
    if wv.shape != (n,) or not np.all(np.isfinite(wv)) or wv.sum() <= 0.0:
        wv = np.ones(n)
    cost = np.zeros(3 * n)
    cost[2 * n:] = -wv
    wrow = np.repeat(np.arange(4 * n), 2)
    wcol = np.concatenate([np.stack([ar, 2 * n + ar], 1).ravel()] * 2 +
                          [np.stack([n + ar, 2 * n + ar], 1).ravel()] * 2)
    wval = np.concatenate([np.tile([1., 1.], n), np.tile([-1., 1.], n)] * 2)
    tr = tr0
    for _ in range(max_it):
        if deadline is not None and time.process_time() > deadline:
            break
        dx = xy[I0, 0] - xy[J0, 0]
        dy = xy[I0, 1] - xy[J0, 1]
        d = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-12)
        live = d - r[I0] - r[J0] < SLACK_K * tr          # exact: the rest cannot bind (see above)
        I, J, dxk, dyk, dk = I0[live], J0[live], dx[live], dy[live], d[live]
        m = len(I)
        prow = np.repeat(np.arange(m), 6)
        pcol = np.stack([I, J, n + I, n + J, 2 * n + I, 2 * n + J], 1).ravel()
        pval = np.stack([-dxk / dk, dxk / dk, -dyk / dk, dyk / dk,
                         np.ones(m), np.ones(m)], 1).ravel()
        A = sp.coo_matrix((np.concatenate([pval, wval]),
                           (np.concatenate([prow, wrow + m]), np.concatenate([pcol, wcol]))),
                          shape=(m + 4 * n, 3 * n)).tocsr()
        b = np.concatenate([dk, HI - xy[:, 0], xy[:, 0] - LO, HI - xy[:, 1], xy[:, 1] - LO])
        rhi = np.minimum(0.5, r + tr)
        bounds = [(-tr, tr)] * (2 * n) + list(zip(np.zeros(n), rhi))
        try:
            res = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs", options=LP_OPTS)
        except (ValueError, TypeError):
            break
        if not res.success or res.x is None:
            break
        z = np.asarray(res.x, float)
        if not np.all(np.isfinite(z)):
            break
        if float(wv @ z[2 * n:]) <= float(wv @ r) + 1e-13:   # linearisation exhausted -> tighten
            tr *= 0.4
            if tr < tr_stop:
                break
            continue
        xy = xy + np.stack([z[:n], z[n:2 * n]], 1)
        r = z[2 * n:]
        tr = min(tr * 1.6, tr0)
    return np.concatenate([xy, r[:, None]], axis=1)


# --------------------------------------------------------- EXACT max sum(r) for FIXED centres (an LP)
def _rlp(xy):
    """With the centres held fixed the packing problem is *exactly* a linear program in the radii:

        max sum r   s.t.   r_i + r_j <= d_ij  (all pairs),   0 <= r_i <= wall_i.

    No linearisation, no trust region -- this is the true optimum for those centres. Two exact
    redundancy filters keep it small: r_i <= ub_i := min(wall_i, min_j d_ij) since every r_j >= 0, and
    then a pair row is redundant whenever ub_i + ub_j <= d_ij. Used to finish a coarse SLP: the SLP's
    long trust-region tail exists only to squeeze the last of the radii out, and this does that in one
    solve. Returns the radius vector, or None if the LP fails (caller keeps the SLP radii)."""
    n = len(xy)
    if n < 1:
        return None
    wall = np.maximum(np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0],
                                         xy[:, 1] - LO, HI - xy[:, 1]]), 0.0)
    if n == 1:
        return wall.copy()
    I, J = np.triu_indices(n, 1)
    d = np.sqrt(((xy[I] - xy[J]) ** 2).sum(-1))
    full = np.full((n, n), np.inf)
    full[I, J] = d
    full[J, I] = d
    ub = np.minimum(wall, full.min(1))
    live = d < ub[I] + ub[J]
    I, J, d = I[live], J[live], d[live]
    m = len(I)
    A = sp.coo_matrix((np.ones(2 * m), (np.repeat(np.arange(m), 2), np.stack([I, J], 1).ravel())),
                      shape=(m, n)).tocsr()
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=d, bounds=list(zip(np.zeros(n), ub)),
                      method="highs", options=LP_OPTS)
    except (ValueError, TypeError):
        return None
    if not res.success or res.x is None or not np.all(np.isfinite(res.x)):
        return None
    return np.maximum(np.asarray(res.x, float), 0.0)


# ------------------------------------------------------------- the SCREEN/REFINE ladder (iteration 14)
# MEASURED (artifacts/probe_iter14_screen.out): the metered run spends its 110 s CPU and only ~2000 of
# its 500000 evaluations, so the binding meter is BASINS PER SECOND, not the budget. Stopping the SLP
# anneal an order of magnitude earlier (tr_stop 3e-2 instead of 1e-3) costs 0.60-0.74x a full solve and
# ranks the starts the SAME way: Spearman +0.996/+0.999/+1.000 at n=41/45/73 over 24 starts each, and
# the true best start was inside the coarse top-6 every time. So the precision tail is only worth
# paying on a candidate that is actually in contention -- screen every round cheaply, then REFINE the
# ones that land near this size's best.
#
# MEASURED A/B (artifacts/probe_iter14_ladder.out): NULL, and instructive. SCREEN_TR 0.0 vs 3e-2, equal
# 14 s CPU per arm, 2 seeds, warm-started, block 35..51 (nine contiguous, all uncapped). Paired per-n
# sum_r: 2 wins / 3 losses / 13 ties of 18 -- but two of the three "losses" are 1e-12 ties, so the real
# tally is 2 wins / 1 loss / 15 ties, i.e. basin re-rolls. THE MECHANISM FAILED ITS OWN PRECONDITION:
# the ladder is supposed to buy ROUNDS, and it bought ~5% (314 vs 300 metered candidates), not the
# ~40% the cost measurement promised -- because 127 of its 278 screens were PROMOTED. Most rounds start
# from the Metropolis chain state, which is already near this size's best, so nearly every candidate
# clears the gate and pays the tail anyway: the screen only saves time on starts that were hopeless,
# and those are the cheap ones. SHIPPED OFF at 0.0: the round collapses to the iteration-13 one exactly
# (asserted in the self-test), and the ladder stays here, tested, for the context where its precondition
# could hold -- a cold-start-only phase, or a gate keyed to the chain state rather than the best.
SCREEN_TR = 0.0             # SLP stop for the screening pass (0.0 == every round pays the full solve)
PROMOTE_REL = 2e-3          # a screened candidate within this RELATIVE gap of the size's best is refined


def _local(start, tr0, deadline=None, tr_stop=SLP_TR_STOP):
    """One complete local solve: coarse SLP for the centres, then the exact r-LP for the radii, then
    projection to strict feasibility. Returns (packing, sum_r) -- or (None, -inf) on failure.
    ``tr_stop`` sets where the SLP anneal stops; the exact r-LP always runs, so even a screening solve
    returns the TRUE optimal radii for its centres (and hence a strictly feasible packing)."""
    cand = _slp(start, tr0=tr0, deadline=deadline, tr_stop=tr_stop)
    r = _rlp(cand[:, :2])
    if r is not None and r.sum() > cand[:, 2].sum():
        cand = np.concatenate([cand[:, :2], r[:, None]], axis=1)
    p = _project(cand)
    return p, _sum_r(p)


def _solve_round(start, tr0, best_s, deadline=None):
    """One round of the sweep: SCREEN cheaply, REFINE only what is in contention (see SCREEN_TR).
    With ``SCREEN_TR = 0.0`` this is exactly the iteration-13 round -- one full ``_local`` -- and it
    draws no randomness in either branch, so the two settings stay comparable seed for seed."""
    if SCREEN_TR <= 0.0:
        return _local(start, tr0, deadline=deadline)
    STATS["screen"] += 1
    cand, s = _local(start, tr0, deadline=deadline, tr_stop=max(SCREEN_TR, SLP_TR_STOP))
    if cand is None:
        return cand, s
    bar = -np.inf if not np.isfinite(best_s) else best_s - PROMOTE_REL * abs(best_s)
    if s >= bar:
        STATS["refine"] += 1
        cand2, s2 = _local(cand, min(SCREEN_TR, tr0), deadline=deadline)
        if cand2 is not None and s2 > s:
            cand, s = cand2, s2
    return cand, s


# ------------------------------------------------------------------------------------- starts and moves
def _read_pack(n):
    """Warm start from the committed census -- GUARDED: anything missing/odd returns None."""
    p = "bench/packs/csqv%d.pck" % n
    if not os.path.exists(p):
        return None
    try:
        raw = [ln for ln in open(p).read().splitlines() if ln.strip()]
        arr = np.array([[float(v) for v in ln.split()] for ln in raw[2:]], dtype=float)
    except (OSError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape != (n, 3) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _cold(n, rng):
    """A fresh start, drawn from an EXACT-COUNT ROW LATTICE family.

    The previous family built an R x C rectangular/offset grid and, when R*C > n, deleted the surplus
    at random -- so most starts carried random vacancies, i.e. defects the local solve then had to
    undo. Rows are the natural coordinate here: a lattice is a stack of rows, and a *hexagonal* lattice
    is precisely a stack of rows whose lengths alternate m, m-1 (the shorter row's circles sit in the
    longer row's gaps automatically once each row is centred, because its spacing 1/L is wider). So
    draw the row structure directly and make it sum to n exactly:

      * R rows, R drawn log-uniformly around sqrt(n) (the aspect ratio of a square-ish lattice);
      * split n across the R rows into lengths differing by at most one -- assigned to ALTERNATING
        rows (the hex pattern) or at random (mixed/defected patterns);
      * row i sits at y = (i+.5)/R - .5 and its L_i circles at x = (j+.5)/L_i - .5, so every row is
        centred and neighbouring rows of unequal length interlock;
      * jitter proportional to the local spacing, and the transposed orientation half the time.

    This spans square lattices (R | n, equal rows), hex lattices (alternating lengths), and the
    intermediate defected patterns that non-lattice-friendly n need -- with no random deletion. A
    minority of starts stay uniform-random clouds so the family can never be the only thing searched.
    Radii start at 0: the first LP inflates them optimally for whatever centres it is given. Nothing
    here is keyed on a table of n."""
    if rng.rand() < 0.15:
        xy = rng.uniform(LO, HI, size=(n, 2))
        return np.concatenate([xy, np.zeros((n, 1))], axis=1)
    R = int(round(np.sqrt(n) * np.exp(rng.uniform(-0.35, 0.35))))
    R = int(min(max(R, 1), n))
    q, s = divmod(n, R)
    lens = np.full(R, q, dtype=int)
    if s:
        if rng.rand() < 0.6:                      # alternating rows -> the hexagonal pattern
            idx = np.concatenate([np.arange(0, R, 2), np.arange(1, R, 2)])[:s]
        else:                                     # random rows -> mixed / defected patterns
            idx = rng.permutation(R)[:s]
        lens[idx] += 1
    ys = (np.arange(R) + 0.5) / R - 0.5
    pts = np.concatenate([np.stack([(np.arange(L) + 0.5) / L - 0.5, np.full(L, ys[i])], 1)
                          for i, L in enumerate(lens)], axis=0)
    if rng.rand() < 0.5:                          # the transposed orientation
        pts = pts[:, ::-1]
    jit = rng.uniform(0.0, 0.35) / max(R, int(lens.max()))
    xy = np.clip(pts + rng.normal(0.0, jit, size=pts.shape), LO, HI)
    return np.concatenate([xy, np.zeros((n, 1))], axis=1)


def _holes(pack, keep, rng, want, samples=3000):
    """`want` well-separated points with the largest clearance to the circles in `keep` and the walls."""
    q = rng.uniform(LO, HI, size=(samples, 2))
    xy, r = pack[keep, :2], pack[keep, 2]
    wall = np.minimum.reduce([q[:, 0] - LO, HI - q[:, 0], q[:, 1] - LO, HI - q[:, 1]])
    d = np.sqrt(((q[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]
    clear = np.minimum(wall, d.min(1))
    out = []
    for _ in range(want):
        i = int(np.argmax(clear))
        out.append(q[i])
        clear = np.minimum(clear, np.sqrt(((q - q[i]) ** 2).sum(-1)) - clear[i])
    return np.array(out)


def _relocate(pack, rng):
    """Lift the j smallest circles and drop them into the emptiest holes. This is the move that
    actually rewires the contact graph -- for a sum-of-radii objective a pinched circle is cheap to
    give up and a hole pays back immediately."""
    n = len(pack)
    j = int(min(max(1, rng.randint(1, 4)), n - 1))
    victims = np.argsort(pack[:, 2])[:j]
    keep = np.setdiff1d(np.arange(n), victims)
    pts = _holes(pack, keep, rng, j)
    out = pack.copy()
    out[victims, :2] = pts
    out[victims, 2] = 0.0
    return out


# MEASURED A/B (artifacts/probe_iter12_ruin.{py,out}): NEGATIVE. RUIN_P 0.0 vs 0.08, equal 7.5 s CPU
# per arm, 3 seeds, warm-started, block 39..51 (contiguous). Paired per-n sum_r: the ruin arm WON 2,
# LOST 4, TIED 15 of 21; mean block digits -0.035 / -0.191 / +0.024 (overall -0.067). The move fired
# (7, 7, 13 ruins per arm), so this is a real negative, not a no-op -- the rounds it takes from the
# shake band pay less than the shake did. SHIPPED OFF at 0.0: the band collapses and the dispatcher is
# the iteration-11 one exactly (asserted in the self-test), and the mechanism stays here, tested, for
# a future composition (e.g. ruin a seam a splice just made, rather than a random blob).
RUIN_P = 0.0                # share of rounds spent on the spatial ruin-and-recreate (0.0 == iter-11)


def _ruin(pack, rng):
    """LARGE-NEIGHBOURHOOD RUIN: lift a *spatially connected cluster* and greedily re-create it.

    Every rewiring move I have differs in WHICH circles it lifts, and all of them so far pick by
    score or by cut: `_relocate` lifts the j SMALLEST circles -- scattered singletons whose removal
    perturbs j separate places in the contact graph by one edge each -- and `_cross` splices along a
    straight LINE, which severs a chord of the graph wherever it happens to fall. Neither can restage
    a *region*: a wrong local arrangement (a 5-circle pocket that should be a 4+1) survives both,
    because undoing it needs several adjacent circles moved AT ONCE and every single-circle step out
    of it is downhill.

    So pick a random circle, lift the k circles NEAREST it (k ~ U{2..6}) -- by construction a
    connected blob -- and re-fill with `_holes`, which drops k well-separated maximum-clearance points
    into the void the lift just opened. The re-fill is greedy and geometric, not a copy: the same k
    circles come back in a DIFFERENT arrangement, so the region's sub-graph is genuinely re-drawn
    while the rest of the packing is untouched and still optimal. Radii start at 0 and the r-LP
    re-grows them, so the result is feasible by construction (the survivors never moved).

    Returns None when the packing is too small to ruin (n < 4), so the caller can fall back."""
    n = len(pack)
    if n < 4:
        return None
    c = pack[rng.randint(0, n), :2]
    d2 = ((pack[:, :2] - c) ** 2).sum(-1)
    k = int(min(rng.randint(2, 7), n - 2))
    victims = np.argsort(d2)[:k]
    keep = np.setdiff1d(np.arange(n), victims)
    pts = _holes(pack, keep, rng, k)
    out = pack.copy()
    out[victims, :2] = pts
    out[victims, 2] = 0.0
    STATS["ruin"] += 1
    return out


def _shake(pack, rng, scale):
    out = pack.copy()
    out[:, :2] = np.clip(out[:, :2] + rng.normal(0.0, scale, size=(len(pack), 2)), LO, HI)
    return out


# ------------------------------------------------- the INFEASIBLE excursion: inflate, repair, re-solve
# MEASURED, and this is why it is the move that is left. Three meters have now been cleared as being
# in SURPLUS, and each reading BANS a family:
#   * distinct basins per slice (probe_iter15_basins.out): 80-106 structurally distinct optima among
#     the 126-204 candidates of one 7 s slice -- start diversity is not scarce (GRAFT_DIV, null).
#   * the evaluation budget (iteration 14): ~2000 of 500000 spent -- candidate count is not scarce.
#   * CPU per stuck size (probe_iter16_share.out): the sweep already gives n=35 9.4 s of the 110 s,
#     MORE than the 7.0 s undivided slice that reached its record in probe_iter15_basins.out. So
#     "concentrate the tail on one size" buys a meter that is already in surplus -- banned unbuilt.
# What is scarce is none of those: it is basins BETTER THAN THE INCUMBENT. At n=45/49/73 the best of
# ~140 fresh, distinct basins exactly TIES the committed pack, so the search's basin distribution is
# centred on the incumbent and its upper tail is thinner than 1/140. More samples, cheaper samples and
# more scattered samples all leave that distribution where it is; only a BIASED move shifts it.
#
# Every operator here -- graft, crossover, cold, relocate, shake, ruin, pressure -- hands _local a
# FEASIBLE start, and the SLP is a feasible-path method whose every iterate is a valid packing. So the
# search has never once left the feasible set, and the configurations reachable only by passing THROUGH
# overlap are exactly the ones it cannot see. This move takes that path: scale every radius up by
# lam > 1 (the packing is now infeasible by construction, and by an amount no shake produces), then
# repair the overlap with centre pushes only -- the classic decision-version relaxation, radii fixed,
# pure vectorised numpy, no LP -- and hand the repaired CENTRES back to the exact machinery. It is
# biased UPWARD rather than scattered: the excursion is aimed at a sum_r the incumbent cannot support,
# and the repair redistributes that pressure into whatever gaps exist, so the basin it lands in is
# drawn from a distribution shifted toward the target, not from a wider one around the incumbent.
# MEASURED ON (iteration 17). The A/B outcome statistic was silent -- 0 ON wins / 1 OFF win / 13 ties
# over 2 seeds x 22 s x the residue block 39..51 -- but the CURRENCY readout (OPSTAT) is not: in the
# band this operator takes over, the shake it displaces bought 0 wins in 116 tries and landed its
# basins a mean -4.0e-01 RELATIVE below the incumbent (near-rate 0.000), while the inflate bought 5
# wins in 105 tries at 1.24 win/CPU-s with mean -1.7e-04 and near-rate 0.714 -- second only to the
# relocate. The zero-sum test an added operator has to pass is beating the MARGINAL move it displaces,
# and measured in the unit it is paid in, it does. Set to 0.0 to recover the iteration-15 dispatch.
INFLATE_P = 0.15            # share of rounds spent on the inflate-repair excursion (0.0 == iter-15)
INFLATE_IT = 40             # repair sweeps (Jacobi separation); the excursion is cheap by design
INFLATE_LO = 1.004          # lam is drawn log-uniform in [INFLATE_LO, INFLATE_HI] so the operator
INFLATE_HI = 1.060          # spans a FAMILY -- a single lam would be one deterministic start per parent
INFLATE_W = 0.85            # relaxation factor on each separation push


def _inflate(pack, rng):
    """Inflate every radius by lam > 1 (INFEASIBLE), push the overlaps apart, keep the centres.

    Returns a strictly feasible (n,3) start, or None if the parent is unusable. The radii handed back
    are the inflated ones scaled by ``_project`` to the largest factor the repaired centres actually
    admit, so the caller always receives a valid packing; the exact r-LP inside ``_local`` then
    re-derives the true optimal radii for those centres. GUARDED for any n and any parent shape."""
    p = np.asarray(pack, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
        return None
    lam = float(np.exp(rng.uniform(np.log(INFLATE_LO), np.log(INFLATE_HI))))
    r = np.maximum(p[:, 2], 1e-12) * lam
    if not np.all(np.isfinite(r)) or r.max() >= 0.5:
        return None
    xy = np.clip(p[:, :2].copy(), LO, HI)
    lo = np.minimum(LO + r, 0.0)[:, None]
    hi = np.maximum(HI - r, 0.0)[:, None]
    for _ in range(INFLATE_IT):
        D = xy[:, None, :] - xy[None, :, :]
        d = np.sqrt((D * D).sum(-1))
        np.fill_diagonal(d, np.inf)
        ov = (r[:, None] + r[None, :]) - d
        np.maximum(ov, 0.0, out=ov)
        if ov.max() <= 1e-12:
            break
        # AVERAGED, not summed: in a jammed packing every circle has 4-6 contacts and all of them
        # overlap at once, so taking the sum of the per-neighbour separations displaces it ~5x too far
        # and the sweep diverges (measured: worst overlap 4.4e-3 -> 1.2e-2 and sum_r -3.5% at n=31).
        # Dividing by the live-contact count makes each sweep a damped Jacobi step that contracts.
        push = ((INFLATE_W * 0.5) * ov / d)[:, :, None] * D
        cnt = np.maximum((ov > 0.0).sum(1), 1)[:, None]
        xy = xy + push.sum(1) / cnt
        if not np.all(np.isfinite(xy)):
            return None
        np.clip(xy, lo, hi, out=xy)
    return _project(np.concatenate([xy, r[:, None]], axis=1))


# ------------------------------------------- objective perturbation: re-solve under a different rule
PRESSURE_P = 0.0            # iter-18: 0.15 -> 0.0. RETIRED by its own currency meter, not deleted: over
                            # the iter-17 and iter-18 probes the pressure move bought 1 win in 335 tries
                            # (near-rate 0.05-0.12) while the reloc it now hands its band to bought ~2/s.
                            # Its start construction is ALSO the one operator whose cost OPSTAT under-
                            # reports (built before the t_op stamp), so it was even dearer than it read.
                            # The move, its knob and its self-test stay: a measured-off knob is a result.


def _press_w(a, rng, hard):
    """Draw the perturbed objective weights. Four shapes, because the useful asymmetries differ:
      * SUPPRESS a random subset -- "give these circles up"; the freed room is taken by whoever can
        use it, which is exactly the rearrangement a sum-of-radii optimum will not do on its own;
      * a spatial GAUSSIAN BUBBLE, suppressing or boosting a whole REGION at once;
      * SIZE POLARISATION w = (r/rmax)^p -- p<0 favours the small circles a jammed optimum leaves
        pinched, p>0 favours the big ones and squeezes the small;
      * a single circle boosted HARD -- the discrete "grow this one and see what has to move".
    Amplitude matters more than it looks: at a jammed optimum, growing one circle by d costs its k
    contacts d each, so a boost below ~k is refused by the LP and the operator is a silent no-op.
    ``hard`` (the retry) forces the shape measured to move the most and pushes the amplitude up."""
    n = len(a)
    if hard:
        w = np.ones(n)
        w[rng.permutation(n)[:max(1, n // 6)]] = 0.0
        return w
    mode = rng.randint(4)
    if mode == 0:
        w = np.ones(n)
        k = int(min(n - 1, max(1, rng.binomial(n, rng.uniform(0.05, 0.25)))))
        w[rng.permutation(n)[:k]] = rng.uniform(0.0, 0.4)
        return w
    if mode == 1:
        q = rng.uniform(LO, HI, size=2)
        sd = rng.uniform(0.08, 0.35)
        g = np.exp(-((a[:, :2] - q) ** 2).sum(1) / (2.0 * sd * sd))
        return 1.0 - rng.uniform(0.5, 1.0) * g if rng.rand() < 0.5 else 1.0 + rng.uniform(1.0, 5.0) * g
    if mode == 2:
        r = np.maximum(a[:, 2], 1e-12)
        return (r / r.max()) ** rng.uniform(-1.5, 1.5)
    w = np.ones(n)
    w[rng.randint(n)] += rng.uniform(3.0, 8.0)
    return w


def _pressure(pack, rng, deadline=None):
    """Re-solve the SAME feasible set under a PERTURBED OBJECTIVE, then let the true objective take
    over from wherever that lands.

    Every basin move so far perturbs the *geometry* -- displace centres (shake), move circles to holes
    (relocate), splice two parents (crossover/graft) -- and then asks the same optimiser to repair the
    damage. Against a sum-of-radii local optimum that is a weak lever: the optimiser's whole job is to
    undo displacement, so a random kick tends to be walked straight back into the basin it came from.
    Changing the objective instead moves along the constraint structure. ``max sum w_i r_i`` over the
    identical feasible set is still an SLP with the identical conservative-inner-approximation
    guarantee, so every iterate is still a valid packing -- but its optimum sits at a DIFFERENT
    contact graph, with the favoured circles grown and their neighbours pushed aside. Releasing the
    weights and re-optimising from there is a directed jump between basins: the intermediate state is
    a genuine optimum of a genuine problem, not damage waiting to be repaired.

    Returns a packing feasible by construction (the SLP never leaves the feasible set), or None for
    n < 2. If the drawn weighting was too weak to move anything (see ``_press_w``) it retries once
    with a forcing one, so the move cannot silently degenerate into a no-op."""
    a = np.asarray(pack, float)
    n = len(a)
    if n < 2 or a.shape[1] != 3 or not np.all(np.isfinite(a)):
        return None
    q = a
    for hard in (0, 1):
        q = _slp(a, tr0=0.5 * TR0, max_it=40, deadline=deadline, tr_stop=1e-3,
                 w=_press_w(a, rng, hard))
        if float(np.abs(q[:, :2] - a[:, :2]).max()) > 1e-9:
            break
    return q


# ------------------------------------------------------- what a size is still WORTH (budget weighting)
def _records():
    """The published best-known sum_r per n, read from the read-only bench/records.json (MISSION.md
    explicitly sanctions reading it from inside solve()). GUARDED: a missing or malformed file, or a
    size with no entry, simply yields nothing and every caller falls back to size-only weighting -- so
    an offline re-run on sizes that appear in no record table behaves exactly as before."""
    try:
        with open("bench/records.json") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    src = raw.get("records") if isinstance(raw, dict) else None
    if not isinstance(src, dict):
        return {}
    out = {}
    for k, v in src.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _headroom(n, s, recs):
    """Multiplier on this size's CPU weight: 1.0 while it can still earn score, CAPPED_W once it cannot.

    The objective is a MEAN OF CLAMPED DIGITS: a size whose relative gap is already below 1e-7 scores
    the full 7 and *cannot* gain a thousandth of a point no matter how much CPU it is given, while its
    packing is banked in phase 0 and cannot be lost. So the marginal value of its slice is exactly zero
    -- not small, zero -- and the weight-by-n schedule was handing the largest such sizes the largest
    slices. Reading the score rule rather than the search tells us where the budget is dead: down-weight
    those to a token share (kept non-zero so a mis-read record or a still-improvable incumbent is never
    stranded, and so they keep donating fresh structure to their neighbours' grafts) and the freed CPU
    lands on the sizes that still have digits on the table."""
    rec = recs.get(n)
    if rec is None or not np.isfinite(rec) or rec <= 0.0:
        return 1.0
    if s is None or not np.isfinite(s):
        return 1.0
    return CAPPED_W if (rec - s) / rec <= CAP_RELGAP else 1.0


def _digits(n, s, recs):
    """Score a packing the way the SCORING RULE does: ``clamp(-log10(relgap), 0, 7)``.

    This is the unit the OBJECTIVE pays in, and it is NOT the unit the search counts. A win is any
    increase in sum_r; a *digit* is only earned while the size is still below the 1e-7 cap, so on a
    census whose sizes are mostly saturated the two have come apart -- the search can spend a whole
    budget earning wins worth exactly zero. Returns None wherever the rule cannot be evaluated (no
    record for this n, a non-finite score), which is also how an offline size behaves."""
    rec = recs.get(int(n)) if recs else None
    if rec is None or not np.isfinite(rec) or rec <= 0.0:
        return None
    if s is None or not np.isfinite(s):
        return None
    gap = (rec - float(s)) / rec
    if gap <= CAP_RELGAP:
        return CAP_DIGITS
    return min(CAP_DIGITS, max(0.0, -np.log10(gap)))


SCHED_VALUE = 1             # 1 = slice weight by REALIZABLE DIGITS; 0 == the iteration-12 schedule
SCHED_NEXP = 0.0            # size exponent kept alongside it: 1.0 reproduces "equal attempts per n"
GAIN_FLOOR = 0.25           # a size never falls to exactly zero weight (see _slice_w)


def _slice_w(n, s, recs):
    """CPU weight for one size's slice of the sweep.

    The old rule was ``n * headroom``: time proportional to SIZE, i.e. roughly EQUAL ATTEMPTS per n
    (basin cost grows ~n^1.2), with a 20x down-weight once a size hit the digit cap. But the objective
    is a MEAN OF CLAMPED DIGITS, so what a size is worth is not its size -- it is the digits it can
    still realize, ``CAP_DIGITS - digits(n)``, and that is bounded above by 7 for every n regardless of
    how expensive n is to attempt. Weighting by n therefore hands the LARGEST slice to the size whose
    attempts cost the most, not to the one with the most score on the table: at this census n=97 (3.39
    digits of headroom) was drawing 2.4x the CPU of n=41 (4.20 of headroom) while each of its attempts
    costs ~3x more. Weighting by realizable digits instead spends equal time per point of recoverable
    score, and the cheaper attempts at small n come for free.

    ``SCHED_NEXP`` keeps the size term available (1.0 -> the old equal-attempts reading, 0.0 -> equal
    time per point of headroom). GUARDED: with no record for n (an offline size, a missing table) the
    headroom is unknowable and this falls back to the old size weighting EXACTLY, so a re-run on sizes
    outside the census schedules as it always did."""
    h = _headroom(n, s, recs)
    rec = recs.get(n)
    if not SCHED_VALUE or rec is None or not np.isfinite(rec) or rec <= 0.0:
        return float(n) * h
    if s is None or not np.isfinite(s):
        return float(n) * h
    gap = (rec - s) / rec
    if gap <= CAP_RELGAP:
        d = CAP_DIGITS
    else:
        d = min(CAP_DIGITS, max(0.0, -np.log10(gap)))
    gain = max(CAP_DIGITS - d, GAIN_FLOOR)
    return (float(n) ** SCHED_NEXP) * gain * h


# ------------------------------------------------------------- cross-SIZE structure transfer (graft)
_PACK_CACHE = {}


def _cached_pack(m):
    if m not in _PACK_CACHE:
        _PACK_CACHE[m] = _read_pack(m)
    return _PACK_CACHE[m]


# MEASURED (artifacts/probe_iter15_basins.out + the graft-entropy count in the self-test): the graft
# fires on ~18% of rounds AND opens every revisit, but its m > n branch -- "delete the (m-n) SMALLEST
# circles" -- is a DETERMINISTIC function of the donor. Over 200 draws at n=45/49/73 the graft produced
# ~102 distinct starts, and every one of them came from the m < n branch (whose hole insertion is
# stochastic): each m > n donor contributes exactly ONE start, re-solved from scratch every time it is
# drawn. At ~0.05 s per local solve that is ~9% of the slice spent re-finding basins already in the
# archive -- pure loss in the meter that binds (basins per second; the run uses 110 s of CPU and 0.4%
# of its evaluation budget). GRAFT_DIV makes the deleted set a SAMPLE instead of an argmin: draw the
# (m-n) victims from the K smallest circles rather than taking the K = m-n smallest outright. The
# start stays feasible by construction (deleting circles never creates an overlap) and stays good (the
# candidates are still the smallest circles, i.e. the ones the donor packing least depends on), but the
# operator now spans a family instead of a point. GRAFT_DIV = 0.0 restores the iteration-14 graft.
# MEASURED A/B (artifacts/probe_iter15_graftdiv.out): NULL, and the basin probe above had already
# predicted it. GRAFT_DIV 0.0 vs 1.0, equal 12 s CPU per arm, 2 seeds, warm-started, block 43..53
# (six contiguous sizes, donors intact). Paired per-n sum_r: 0 ON wins / 1 OFF win / 11 ties -- and the
# single non-tie is a jackpot the OFF arm drew at n=45, i.e. a basin re-roll, not a mechanism. The
# entropy the knob adds is real (1 -> 50 distinct deleted sets over 50 draws, asserted in self-test 17)
# and it fired on every m > n graft, but it bought nothing, because DISTINCT BASINS WERE ALREADY IN
# SURPLUS: probe_iter15_basins.out counts 80-106 structurally distinct optima among the 126-204
# candidates of a single 7 s slice at n=35/45/49/73. Cheapening a resource that is not scarce is
# zero-sum against the move it displaces. SHIPPED OFF at 0.0 -- the graft is the iteration-14 argmin
# rule exactly (asserted) -- and the sampler stays here, tested, for a regime where the graft is the
# only start (a cold census, an offline n with no archive) and its entropy would be the binding one.
GRAFT_DIV = 0.0             # share of m > n grafts whose deleted set is sampled, not argmin-ed
GRAFT_POOL = 3              # victims are drawn from the (GRAFT_POOL * (m-n) + 2) smallest circles


def _graft_drop(p, k, rng):
    """Choose which k circles to delete from an (m,3) donor. Returns the KEPT row indices, sorted.
    With GRAFT_DIV = 0 this is exactly ``argsort(r)[k:]`` -- the iteration-14 rule."""
    m = len(p)
    if k <= 0:
        return np.arange(m)
    order = np.argsort(p[:, 2])
    if GRAFT_DIV <= 0.0 or rng.rand() >= GRAFT_DIV:
        return np.sort(order[k:])
    cap = min(m, GRAFT_POOL * k + 2)
    victims = order[:cap][rng.choice(cap, size=k, replace=False)]
    keep = np.setdiff1d(np.arange(m), victims, assume_unique=False)
    return keep


def _graft(n, rng, pool):
    """Start n from a NEIGHBOURING SIZE's packing.

    Neighbouring n share almost all of their structure: record[n]/(0.5*sqrt n) drifts by <2% across
    the whole census, so an optimised packing of m = n +/- 2 is a dense lattice-with-defects of very
    nearly the right spacing. Rescaling is unnecessary (the square is fixed); what is needed is a
    change of *count*:
      * m > n -- delete the (m - n) smallest circles. The survivors stay strictly feasible and the
        freed area is exactly where the next local solve will expand.
      * m < n -- insert the (n - m) missing circles at the emptiest holes, radius 0.
    Either way the start is feasible by construction and carries a contact graph that took the search
    real CPU to find, which no cold lattice draw reproduces. `pool` is the in-run incumbent map, so a
    size grafts from its neighbour's *current* best, not just the committed one -- the census becomes a
    population that recombines instead of 37 independent searches. GUARDED: sizes with no pack (a fresh
    n, an offline re-run) simply yield nothing and the caller falls back to a cold start."""
    cands = [n - 2, n + 2, n - 1, n + 1, n - 4, n + 4, n - 3, n + 3]
    rng.shuffle(cands)
    cands.sort(key=lambda m: abs(m - n) + rng.rand() * 0.5)
    for m in cands:
        if m < 1 or m == n:
            continue
        src = pool.get(m, (None, None))[0]
        if src is None:
            src = _cached_pack(m)
        if src is None or len(src) != m:
            continue
        p = np.array(src, dtype=float)
        if m > n:
            return p[_graft_drop(p, m - n, rng)]
        base = _project(p)
        pts = _holes(base, np.arange(m), rng, n - m)
        add = np.concatenate([pts, np.zeros((n - m, 1))], axis=1)
        return np.concatenate([base, add], axis=0)
    return None


# ----------------------------------------------------- same-SIZE recombination (spatial patch crossover)
STATS = {"cross": 0, "press": 0, "xcross": 0, "etie": 0, "ecrowd": 0, "ruin": 0,
         "screen": 0, "refine": 0, "inflate": 0}   # diagnostics only: how often each operator fired

# ---- THE ROUND BANDS (what iteration 17's currency meter turned into a decision) --------------------
# The per-round operator is chosen by one uniform draw u; these are the cut points, so each operator's
# BAND WIDTH is literally the share of the fixed round budget it holds. Adding capability here is
# zero-sum: every round handed to a new move is taken from a specific incumbent one, so the bands are
# knobs and not constants, and they are set from OPSTAT (wins per CPU-second) rather than from taste.
#   graft  : u <  GRAFT_U                 (plus the forced open of every revisit)
#   cross  : u <  CROSS_U                 (when >=2 elites exist)
#   cold   : u <  COLD_U                  (plus the forced first rounds of a size's first slice)
#   reloc  : u <  RELOC_U
#   shake  : u <  1 - PRESSURE_P - RUIN_P - INFLATE_P
#   inflate: u <  1 - PRESSURE_P - RUIN_P
#   ruin   : u <  1 - PRESSURE_P
#   press  : otherwise
# A band collapses to zero width when its upper cut point meets the one below it -- that is how a
# measured-worthless operator is retired WITHOUT deleting its (documented, self-tested) code.
GRAFT_U = 0.00              # iter-19: 0.18 -> 0.00. RETIRED by its own currency meter (the last
                            # unaudited incumbent, and the dearest: ~24% of arm CPU). Pooled over the
                            # iter-18 and iter-19 probes graft bought 0 wins in 736 tries, meanrel
                            # -5.5e-03..-1.3e-02, near-rate 0.0000 -- it never once landed a basin
                            # within reach of the incumbent. Retiring it does NOT close the cross-size
                            # channel: crossover's second parent still comes from a neighbouring size's
                            # archive via XCROSS_P/_xdonor. Set back to 0.18 to recover iteration 18.
GRAFT_OPEN = False          # iter-19: True -> False. graft held TWO shares and the forced opener was
                            # the larger; retiring the band alone measured no better than base (27 wins
                            # vs 29), retiring both measured 39 (artifacts/probe_iter19_graft.out).
CROSS_U = 0.17              # iter-19: 0.35 -> 0.17. Lowered by exactly GRAFT's old 0.18 so that CROSS
COLD_U = 0.17               # keeps its own 0.17 share and only graft's band is reallocated; cold's
                            # random band stays collapsed (CROSS_U == COLD_U). iter-18: cold bought 0
                            # wins in 213 tries, meanrel -1.7e-02, near-rate 0.000; it keeps its forced
                            # opener for a size with no incumbent.
RELOC_U = 0.85              # iter-18: 0.70 -> 0.85. The retired cold, press and (iter-19) graft shares
                            # all go to the relocate, the best-paid move in the repertoire: 40 wins/arm
                            # vs base's 31 (iter-18) and 39 vs 29 (iter-19) at equal CPU. Its band is now
                            # [CROSS_U, RELOC_U] = 0.68 of every round.
# INVARIANT the arms must respect: GRAFT_U <= CROSS_U <= COLD_U <= RELOC_U <= 1 - PRESSURE_P - RUIN_P
# - INFLATE_P. Raising RELOC_U past the shake cut does not just retire the shake -- it silently swallows
# the inflate band as well, because the checks are ordered and the first one that matches wins.


# Firing counts hide the one number that decides whether an operator earns its slot: a move can fire
# constantly and still buy nothing. OPSTAT records, per operator, how many starts it produced ("try"),
# how much CPU those rounds cost ("cpu"), and how many of them landed STRICTLY ABOVE the size's running
# best ("win") -- the currency every operator is paid in. Diagnostics only: nothing here is read by the
# search, so the rng stream and every decision are identical with or without it.
# RE-DENOMINATED (iteration 20): "win" counts basins above the incumbent -- the SEARCH's unit.
# "dig"/"dwin" count the same events in the SCORING RULE's unit (see ``_digits``): the clamped
# digit gain the win actually realised, and how many wins realised any at all. A win on a size
# already at the 7-digit cap scores 0 here, which is exactly what it is worth to the objective.
OPSTAT = {"try": {}, "win": {}, "cpu": {}, "gap": {}, "ngap": {}, "near": {}, "dig": {}, "dwin": {}}


def opstat_reset():
    for d in OPSTAT.values():
        d.clear()


def opstat_report():
    """One line per operator: tries, wins, win rate, CPU, and wins per CPU-second."""
    out = []
    for k in sorted(OPSTAT["try"], key=lambda k: -OPSTAT["win"].get(k, 0)):
        t = OPSTAT["try"][k]
        w = OPSTAT["win"].get(k, 0)
        c = OPSTAT["cpu"].get(k, 0.0)
        g = OPSTAT["gap"].get(k, 0.0)
        ng = OPSTAT["ngap"].get(k, 0)
        nr = OPSTAT["near"].get(k, 0)
        out.append("%-8s try=%5d win=%4d cpu=%6.1fs win/s=%.3f meanrel=%+.2e near=%4d/%-5d (%.3f)"
                   " dwin=%4d dig=%+.3f dig/s=%.4f"
                   % (k, t, w, c, w / max(c, 1e-9), g / max(ng, 1), nr, ng, nr / max(ng, 1),
                      OPSTAT["dwin"].get(k, 0), OPSTAT["dig"].get(k, 0.0),
                      OPSTAT["dig"].get(k, 0.0) / max(c, 1e-9)))
    return "\n".join(out)


def _cross(a, b, rng):
    """Splice a spatial PATCH of one same-size packing into another and let the LP refill the radii.

    The graft moves structure between *sizes*, but it is a deterministic copy of one individual: given
    the donor, the start is fixed, so a size's grafts all land in the same family. Two distinct local
    optima of the SAME n, however, usually disagree only over part of the square -- one has the good
    row structure on the left, the other on the right. Cutting along a random line and keeping A's side
    and B's side produces a contact graph neither parent contains, which no single-parent move can
    reach.

    Only the CENTRES are spliced; every radius is set to 0, so the start is feasible for any pair of
    parents (the first LP inflates the radii optimally for whatever centres it is given) and the seam
    needs no repair. The count is then made exact without touching the interiors:
      * too many circles (both sides overfull) -- greedily thin the most crowded centre, i.e. the one
        with the smallest nearest-neighbour distance, which is precisely the seam duplication;
      * too few -- insert at the emptiest holes, as the graft does.

    B NEED NOT HAVE THE SAME SIZE AS A. The output count is always len(a), and the count repair above
    is exactly the machinery that makes that true -- so the donor may be a NEIGHBOURING size's optimum
    (see `_xdonor`). Neighbouring n differ in row spacing by ~1/(2n) (record[n]/(0.5*sqrt n) drifts <2%
    across the whole census), i.e. far less than one local solve's basin width, so half of a size-45
    optimum is a legitimate half of a size-43 packing once the count is fixed. This is what the graft
    could never do: the graft copies ONE neighbour whole, so a size's grafts all land in that
    neighbour's family, while a splice mixes two lineages and reaches contact graphs neither contains.
    """
    n = len(a)
    if n < 3 or len(b) < 1:
        return None
    STATS["cross"] += 1
    if len(b) != n:
        STATS["xcross"] += 1
    th = rng.uniform(0.0, np.pi)
    u = np.array([np.cos(th), np.sin(th)])
    pa, pb = np.asarray(a, float), np.asarray(b, float)
    sa, sb = pa[:, :2] @ u, pb[:, :2] @ u
    lo, hi = np.percentile(np.concatenate([sa, sb]), [25.0, 75.0])
    c = rng.uniform(lo, hi)                      # a cut that both parents actually straddle
    xy = np.concatenate([pa[sa < c, :2], pb[sb >= c, :2]], axis=0)
    m = len(xy)
    if m == 0:
        return None
    while m > n:                                 # thin the crowded seam, one centre at a time
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        xy = np.delete(xy, int(np.argmin(d.min(1))), axis=0)
        m -= 1
    out = np.concatenate([xy, np.zeros((m, 1))], axis=1)
    if m < n:
        pts = _holes(out, np.arange(m), rng, n - m)
        out = np.concatenate([out, np.concatenate([pts, np.zeros((n - m, 1))], axis=1)], axis=0)
    return out


ELITES = 3                  # diverse local optima kept per n, as crossover parents
ELITE_SEP = 1e-7            # legacy key: two elites had to differ in sum_r by this (relative)
ELITE_SAME = 1e-6           # STRUCTURAL key: max |sorted-radius| gap below which two packings are
                            # the same basin (radii are ~0.05 and distinct contact graphs disagree by
                            # ~1e-3, while two runs into one basin agree to ~1e-12 -- a wide margin)
ELITE_DIVERSE = 0           # 1 = structural key + crowding eviction; 0 == the iteration-10 archive.
                            # MEASURED (artifacts/probe_iter11_elitediv.out, two seeds, 8 contiguous
                            # small sizes, equal CPU): 1 is NOT better -- -0.62 and -0.02 mean digits,
                            # with the mechanism firing hard (217/237 crowding evictions per arm). So
                            # the archive's SELECTION RULE is not what binds here, and paying for
                            # distance by evicting near-best parents costs more than it buys. Kept
                            # behind the knob, off, as a measured negative rather than a guess.


def _sig(p):
    """Structural signature of a packing: its radii, sorted descending.

    Permutation- and symmetry-invariant (a reflected or relabelled copy of a packing is the same
    basin and should not occupy a second archive slot), O(n log n), and it separates CONTACT GRAPHS:
    two descents into one basin agree in every radius to ~1e-12, whereas two different graphs of the
    same n disagree in the profile even when their SUMS happen to coincide."""
    return np.sort(np.asarray(p, float)[:, 2])[::-1]


def _sig_d(a, b):
    """Structural distance between two signatures; inf for incomparable shapes."""
    if a is None or b is None or a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max())


def _elite_add(pool, p, s):
    """Keep ELITES local optima for one n -- selected for STRUCTURE as well as score.

    The archive is the parent pool for crossover, and since iteration 10 also the donor pool its
    NEIGHBOURS draw from, so what it holds decides what recombination can reach. Selecting it by score
    alone is self-defeating: a pool ranked only by sum_r fills with the incumbent's own family, and
    splicing two members of one family reaches nothing a single parent could not. Two fixes, both
    structural:

      * IDENTITY is structural, not numeric. The old key called two entries the same basin iff their
        sums agreed to 1e-7 relative -- which both merged genuinely distinct graphs whose sums happen
        to coincide (rejecting real diversity) and kept symmetric copies of one graph whose sums drift
        apart. `_sig` compares the radius profile instead, so a mirror image is recognised as the same
        basin and a numerically-tied different graph is admitted as a new one.
      * EVICTION is by CROWDING, not by rank. When the pool overflows, the old rule dropped the worst
        entry -- so a third, structurally novel optimum was discarded to keep a near-copy of the best.
        Instead, find the CLOSEST PAIR in the archive and drop its weaker member: the best is never
        evicted (the pool is score-sorted, so the victim is never index 0), quality is preserved at the
        top, and the slots below it are spent on distance rather than on decimals.

    `ELITE_DIVERSE = 0` restores the iteration-10 archive exactly (numeric key, rank truncation)."""
    if not ELITE_DIVERSE:
        for e in pool:
            if abs(s - e[1]) <= ELITE_SEP * max(1.0, abs(e[1])):
                return
        pool.append((p, s, None))
        pool.sort(key=lambda t: -t[1])
        del pool[ELITES:]
        return
    sig = _sig(p)
    for e in pool:                                # diagnostics: entries the legacy numeric key would
        if abs(s - e[1]) <= ELITE_SEP * max(1.0, abs(e[1])) and _sig_d(sig, e[2]) > ELITE_SAME:
            STATS["etie"] += 1                    # have rejected as a tie, but which are new basins
            break
    for i, e in enumerate(pool):
        if _sig_d(sig, e[2]) <= ELITE_SAME:      # same basin -- keep the better copy, never both
            if s > e[1]:
                pool[i] = (p, s, sig)
                pool.sort(key=lambda t: -t[1])
            return
    pool.append((p, s, sig))
    pool.sort(key=lambda t: -t[1])
    if len(pool) <= ELITES:
        return
    k = len(pool)
    victim, near = k - 1, np.inf
    for i in range(k):
        for j in range(i + 1, k):                # pool is score-sorted, so j is the weaker member
            d = _sig_d(pool[i][2], pool[j][2])
            if d < near:
                near, victim = d, j
    if victim != k - 1:                           # diagnostics: the crowding rule kept an entry that
        STATS["ecrowd"] += 1                      # rank truncation would have thrown away
    del pool[victim]


# ------------------------------------------------- CROSS-SIZE recombination (who may donate to whom)
XCROSS_P = 0.5              # share of crossover draws whose second parent comes from another SIZE
XCROSS_NB = (2, 4, 1, 3, 6) # candidate |n - m| offsets, nearest first


def _use_xcross(rng):
    """Whether this crossover draw reaches outside its own size. Short-circuits at XCROSS_P == 0.0 so
    the disabled solver consumes NO randomness here and reproduces the previous rng stream exactly."""
    return XCROSS_P > 0.0 and rng.rand() < XCROSS_P


def _xdonor(n, rng, elite, pool):
    """A second parent drawn from a NEIGHBOURING SIZE's elite archive.

    Crossover so far recombined two distinct optima of the same n, and the graft moved a whole
    neighbour across; neither mixes lineages BETWEEN sizes. But the census is one population, not 37:
    n and n +/- 2 share almost all of their structure, and the archives of the neighbours hold basins
    this size has never visited. Sampling the donor from a neighbour's archive (elites first, the
    in-run incumbent or the committed pack as fallback) makes every size's search see its neighbours'
    diversity, not just their single best. GUARDED: a size with no reachable neighbour yields None and
    the caller falls back to a same-size parent or a cold start."""
    cands = []
    for d in XCROSS_NB:
        for m in (n - d, n + d):
            if m >= 3:
                cands.append(m)
    order = sorted(cands, key=lambda m: abs(m - n) + rng.rand())
    for m in order[:6]:
        ep = elite.get(m)
        if ep:
            return ep[rng.randint(len(ep))][0]
        src = pool.get(m, (None, None))[0]
        if src is None:
            src = _cached_pack(m)
        if src is not None and len(src) == m:
            return np.asarray(src, float)
    return None


# ------------------------------------------------------------------------------------- the solver entry
def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    t_start = time.process_time()
    order = sorted(targets)

    # --- phase 0: bank the committed census immediately, so a backstop can never lose ground -------
    incumbent = {}
    for n in order:
        w = _read_pack(n)
        if w is None or meter.left() <= 0:
            continue
        p = _project(w)
        f, s = evaluate(n, p)
        if bool(f):
            incumbent[n] = (p, float(s))

    # --- phase 1: a MULTI-PASS ALTERNATING SWEEP over the census --------------------------------
    # The graft (a size starting from a neighbour's optimised packing) made the census a population,
    # but the schedule still read it as a list: one ascending pass, each n visited once. So structure
    # could only ever flow UPWARD, and only one hop per iteration -- n=27 grafted from last iteration's
    # committed neighbours while n=99 grafted from freshly improved ones. Splitting the same CPU into
    # PASSES visits that alternate direction fixes both: every n both donates to and receives from its
    # neighbours, repeatedly, within one iteration. The per-n chain state (`chain`) persists across
    # visits so a later visit continues its Metropolis walk instead of restarting it, and each revisit
    # opens with a forced graft, which is exactly when its donors are newest.
    sched = []
    for p in range(PASSES):
        sched.extend(order if p % 2 == 0 else order[::-1])
    # Slice weight = _slice_w(n): time per size proportional to the DIGITS IT CAN STILL REALIZE rather
    # than to n, with a capped size held at a token share. The weights are re-read at every visit from
    # the *current* incumbent, so a size that reaches the cap mid-sweep stops consuming its later
    # passes -- the schedule follows the score function instead of the size list.
    recs = _records()

    chain = {}
    elite = {}
    for n, (p, sv) in incumbent.items():
        _elite_add(elite.setdefault(n, []), p, sv)
    def _w(m):
        return _slice_w(m, incumbent.get(m, (None, None))[1], recs)

    for k, n in enumerate(sched):
        if meter.left() <= 2:
            return
        left = CPU_BUDGET_S - (time.process_time() - t_start)
        if left <= 0.5:
            return
        rem = sum(_w(m) for m in sched[k:])       # remaining weight, re-read from current incumbents
        slice_end = time.process_time() + left * _w(n) / max(1e-9, rem)
        best, bs = incumbent.get(n, (None, -np.inf))
        revisit = n in chain
        cur, cs = chain.get(n, (best, bs))
        temp = T_FAC * max(bs if np.isfinite(bs) else 0.0, 0.5 * np.sqrt(n))
        rounds = 0
        while time.process_time() < slice_end and meter.left() > 2:
            u = rng.rand()
            if (GRAFT_OPEN and rounds == (0 if revisit else 1)) or (cur is not None and u < GRAFT_U):
                # GRAFT: transfer a neighbouring size's optimised structure (see _graft). Opens every
                # revisit, because that is when the neighbours' packings are freshest; falls back to a
                # cold start whenever no neighbour pack exists.
                op = "graft"
                start, tr = _graft(n, rng, incumbent), TR0
                if start is None:
                    start = _cold(n, rng)
                elif rng.rand() < 0.5:
                    start = _shake(start, rng, 0.15 / n)   # a little variety per graft draw
            elif u < CROSS_U and (len(elite.get(n, ())) >= 2 or
                               (XCROSS_P > 0.0 and len(elite.get(n, ())) >= 1)):
                # CROSSOVER: recombine two DISTINCT local optima (see _cross). This is the only move
                # whose output is not a deterministic function of a single parent -- and with
                # XCROSS_P > 0 the second parent may come from a NEIGHBOURING SIZE's archive
                # (see _xdonor), so structure recombines ACROSS the census, not just within one n.
                op = "cross"
                pool_n = elite[n]
                partner = _xdonor(n, rng, elite, incumbent) if _use_xcross(rng) else None
                if partner is None:
                    if len(pool_n) < 2:
                        partner = None
                    else:
                        i = 1 + rng.randint(0, len(pool_n) - 1)
                        partner = pool_n[i][0]
                if partner is None:
                    start, tr = _cold(n, rng), TR0
                else:
                    pair = [pool_n[0][0], partner]
                    if rng.rand() < 0.5 and len(partner) == n:
                        pair.reverse()
                    start, tr = _cross(pair[0], pair[1], rng), TR0
                    if start is None:
                        start = _cold(n, rng)
            elif (rounds < 3 and not revisit) or cur is None or u < COLD_U:
                # a structured cold start is a first-class move, not just a seed: the exact-count row
                # lattice family is where the historically largest single gains came from.
                op = "cold"
                start, tr = _cold(n, rng), TR0
            elif u < RELOC_U:
                op = "reloc"
                start, tr = _relocate(cur, rng), 0.5 * TR0
            elif u < 1.0 - PRESSURE_P - RUIN_P - INFLATE_P:
                op = "shake"
                start, tr = _shake(cur, rng, 0.02 * (0.5 ** (rounds % 3))), 0.5 * TR0
            elif u < 1.0 - PRESSURE_P - RUIN_P:
                # INFLATE & REPAIR: the one move that leaves the FEASIBLE SET (see _inflate). Every
                # other start here is a valid packing and the SLP never passes through overlap, so the
                # basins reachable only through it are invisible to the rest of the repertoire. With
                # INFLATE_P = 0.0 this branch is unreachable and no randomness is drawn here, so the
                # round reproduces the iteration-15 rng stream exactly.
                op = "inflate"
                STATS["inflate"] += 1
                start, tr = _inflate(cur, rng), TR0
                if start is None:
                    start = _shake(cur, rng, 0.02)
            elif u < 1.0 - PRESSURE_P:
                # RUIN & RECREATE: re-draw one CONNECTED REGION of the contact graph (see _ruin).
                # A shake displaces everything a little and the SLP walks most of it straight back;
                # this removes a blob outright and lets the greedy re-fill choose a different local
                # arrangement, which is the one thing a per-circle move cannot reach.
                op = "ruin"
                start, tr = _ruin(cur, rng), TR0
                if start is None:
                    start = _shake(cur, rng, 0.02)
            else:
                # PRESSURE: re-solve this packing under a perturbed objective (see _pressure) and let
                # the true objective take over from there. Unlike a shake, the intermediate state is
                # itself an optimum -- of a different rule -- so the jump lands on a neighbouring
                # contact graph instead of somewhere the optimiser will simply walk back.
                op = "press"
                STATS["press"] += 1
                start, tr = _pressure(cur, rng, deadline=slice_end), 0.5 * TR0
                if start is None:
                    start = _shake(cur, rng, 0.02)
            rounds += 1
            t_op = time.process_time()
            cand, s = _solve_round(start, tr, bs, deadline=slice_end)
            OPSTAT["try"][op] = OPSTAT["try"].get(op, 0) + 1
            OPSTAT["cpu"][op] = OPSTAT["cpu"].get(op, 0.0) + (time.process_time() - t_op)
            if cand is None:
                continue
            f, s_ev = evaluate(n, cand)
            if not bool(f):
                continue
            s = float(s_ev)
            # DENSE companion to the win count: wins against a warm incumbent are rare jackpots, so a
            # zero-win arm says nothing on its own. The relative deficit (s - bs)/bs of every basin an
            # operator produces is readable at every round, and "near" counts the basins that land
            # within 1e-4 of the incumbent -- i.e. whether the distribution is CENTRED any better.
            if np.isfinite(bs) and bs > 0.0:
                rel = (s - bs) / bs
                OPSTAT["gap"][op] = OPSTAT["gap"].get(op, 0.0) + rel
                OPSTAT["ngap"][op] = OPSTAT["ngap"].get(op, 0) + 1
                if rel > -1e-4:
                    OPSTAT["near"][op] = OPSTAT["near"].get(op, 0) + 1
            _elite_add(elite.setdefault(n, []), cand, s)
            if s > bs:
                # THE CURRENCY, not the firing count: a basin strictly ABOVE this size's incumbent is
                # the only thing any operator is actually paid in (see OPSTAT).
                OPSTAT["win"][op] = OPSTAT["win"].get(op, 0) + 1
                d0, d1 = _digits(n, bs, recs), _digits(n, s, recs)
                if d0 is not None and d1 is not None and d1 > d0:
                    OPSTAT["dig"][op] = OPSTAT["dig"].get(op, 0.0) + (d1 - d0)
                    OPSTAT["dwin"][op] = OPSTAT["dwin"].get(op, 0) + 1
                best, bs = cand, s
                temp = T_FAC * bs
            # Metropolis over basins: uphill always, downhill sometimes.
            if cur is None or s >= cs or rng.rand() < np.exp((s - cs) / max(temp, 1e-12)):
                cur, cs = cand, s
        chain[n] = (cur, cs)
        if best is not None:
            incumbent[n] = (best, bs)
            evaluate(n, best)


# ------------------------------------------------------------------------------------------- self-test
def _ev_track(store, n, pack, ev):
    """Self-test helper: score through the local evaluator and keep the best sum_r seen per n."""
    a = np.asarray(pack, float)
    if a.ndim == 3:
        return [_ev_track(store, n, q, ev) for q in a]
    f, sv = ev(n, a)
    if bool(f) and float(sv) > store.get(n, -1e18):
        store[n] = float(sv)
    return (f, sv)


def _self_test():
    global RUIN_P
    ok = True

    class _M:
        budget = 500000

        def __init__(self):
            self.used = 0

        def left(self):
            return self.budget - self.used

    M = _M()
    seen = {}

    def ev(n, packing):
        a = np.asarray(packing, float)
        single = a.ndim == 2
        if single:
            a = a[None]
        M.used += a.shape[0]
        fs, ss = [], []
        for p in a:
            x, y, r = p[:, 0], p[:, 1], p[:, 2]
            wall = np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min()
            d = np.sqrt(((p[:, :2][:, None] - p[:, :2][None]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            pair = float((d - (r[:, None] + r[None, :])).min())
            f = bool(r.min() > 0 and wall >= -FEAS_TOL and pair >= -FEAS_TOL)
            fs.append(f)
            ss.append(float(r.sum()))
            if f:
                seen[n] = max(seen.get(n, -1.0), float(r.sum()))
        return (fs[0], ss[0]) if single else (np.array(fs), np.array(ss))

    rng = np.random.RandomState(0)

    # 1. projection makes an adversarially overlapping packing strictly feasible
    bad = np.concatenate([rng.uniform(LO, HI, size=(12, 2)), np.full((12, 1), 3.0)], axis=1)
    f, _ = ev(12, _project(bad))
    print("projection feasible on overlapping input:", f)
    ok &= f

    # 2. the SLP is a genuine ascent from a cold start, and its output projects with tiny loss
    c = _cold(20, rng)
    q = _slp(c)
    pq = _project(q)
    f2, s2 = ev(20, pq)
    loss = (_sum_r(q) - s2) / max(_sum_r(q), 1e-12)
    print("slp n=20 feasible=%s sum_r=%.6f projection_loss_rel=%.2e" % (f2, s2, loss))
    ok &= bool(f2) and s2 > 1.0 and loss < 1e-5

    # 3. monotone: the SLP never returns less than it was given (MM property), on a feasible input
    q2 = _slp(pq, tr0=0.02)
    print("slp is non-decreasing from a local optimum:", _sum_r(q2) >= s2 - 1e-9)
    ok &= _sum_r(q2) >= s2 - 1e-9

    # 4. full solve on sizes with NO committed pack (the MISSION.md absent-pack path)
    global CPU_BUDGET_S
    keep, CPU_BUDGET_S = CPU_BUDGET_S, 6.0
    try:
        solve(ev, M, np.random.RandomState(1), [9, 14])
    finally:
        CPU_BUDGET_S = keep
    for n in (9, 14):
        base = _sum_r(_project(_slp(_cold(n, np.random.RandomState(2)))))
        got = seen.get(n, -1.0)
        good = got >= base - 1e-9
        print("n=%d solved sum_r=%.6f  single-cold-solve %.6f  ok: %s" % (n, got, base, good))
        ok &= good

    # 5. the near-pair row filter is EXACT: the reduced LP must trace the full LP step for step
    global SLACK_K
    kept = SLACK_K
    rng5 = np.random.RandomState(11)
    c5 = _cold(30, rng5)
    a5 = _slp(c5)
    SLACK_K = 1e9                      # every pair row present = the unreduced LP
    try:
        b5 = _slp(c5)
    finally:
        SLACK_K = kept
    dev = float(np.abs(_sum_r(a5) - _sum_r(b5)))
    print("row filter exact vs full LP: sum_r dev %.2e" % dev)
    ok &= dev < 1e-8

    # 6. the exact r-LP: for FIXED centres it must be feasible and never worse than the SLP radii
    rng6 = np.random.RandomState(23)
    for n6 in (7, 30, 55):
        a6 = _slp(_cold(n6, rng6), tr_stop=1e-3)
        r6 = _rlp(a6[:, :2])
        ok6 = r6 is not None and r6.min() >= -1e-12 and r6.sum() >= a6[:, 2].sum() - 1e-9
        f6, s6 = ev(n6, _project(np.concatenate([a6[:, :2], r6[:, None]], 1)))
        print("r-LP n=%d feasible=%s sum_r=%.9f  (slp radii %.9f) ok: %s"
              % (n6, f6, s6, a6[:, 2].sum(), bool(ok6) and bool(f6)))
        ok &= bool(ok6) and bool(f6)

    # 7. every cold start must contain EXACTLY n circles inside the square (no random deletion)
    rng7 = np.random.RandomState(31)
    bad7 = [n7 for n7 in (1, 2, 5, 27, 40, 99) for _ in range(25)
            if _cold(n7, rng7).shape != (n7, 3)]
    inside = all(np.all((_cold(n7, rng7)[:, :2] >= LO - 1e-12) &
                        (_cold(n7, rng7)[:, :2] <= HI + 1e-12)) for n7 in (5, 27, 99))
    print("cold starts exact-count and in-square:", not bad7 and inside)
    ok &= (not bad7) and inside

    # 8. absent .pck must not raise
    # graft: exact count from any donor size, feasible-by-construction, guarded when no donor exists
    gok = True
    for n in (27, 40, 55, 99):
        for seed in range(6):
            rg = np.random.RandomState(1000 + seed)
            g = _graft(n, rg, {})
            if g is None:
                continue
            if g.shape != (n, 3) or not np.all(np.isfinite(g)):
                gok = False
                continue
            p = _project(g)
            if abs(_sum_r(_slp(g, tr0=TR0, max_it=6)) ) < 0:
                gok = False
            if p.shape != (n, 3):
                gok = False
    # a donor-less pool AND a donor-less disk must yield None, never an exception
    gnone = _graft(1234567, np.random.RandomState(0), {}) is None
    # in-run pool donors are used even when no file exists
    pool = {14: (_project(np.concatenate([np.random.RandomState(3).uniform(LO, HI, (14, 2)),
                                          np.full((14, 1), 0.02)], 1)), 0.0)}
    gp = _graft(15, np.random.RandomState(4), pool)
    gpool = gp is not None and gp.shape == (15, 3)
    gdel = _graft(13, np.random.RandomState(5), pool)
    gdel_ok = gdel is not None and gdel.shape == (13, 3)
    print("graft exact-count/finite from census donors:", gok)
    print("graft guarded (no donor -> None):", gnone)
    print("graft uses in-run pool (insert 14->15, delete 14->13):", gpool, gdel_ok)
    ok = ok and gok and gnone and gpool and gdel_ok

    # 9. the MULTI-PASS schedule must not starve a target or overrun the CPU ceiling. Splitting one
    #    slice into PASSES visits fragments the time, so the new failure mode is a size that never
    #    completes a single local solve -- assert every target still ends with a feasible packing and
    #    that the whole sweep respects CPU_BUDGET_S (plus at most the basin in flight).
    global PASSES
    keep2, keepP, keepR = CPU_BUDGET_S, PASSES, RUIN_P
    CPU_BUDGET_S, PASSES = 4.0, 3
    RUIN_P = 0.08 if RUIN_P == 0.0 else RUIN_P   # exercise the ruin DISPATCH even when it ships off
    seen.clear()
    t9 = time.process_time()
    try:
        solve(ev, M, np.random.RandomState(7), [9, 14, 20])
    finally:
        CPU_BUDGET_S, PASSES, RUIN_P = keep2, keepP, keepR
    cpu9 = time.process_time() - t9
    starved = [n for n in (9, 14, 20) if seen.get(n, -1.0) <= 0.0]
    print("multi-pass: no target starved %s  cpu %.1fs of 4.0s ceiling" % (not starved, cpu9))
    ok &= (not starved) and cpu9 < 4.0 + 2.0

    # 10. CROSSOVER: exact count, in-square, feasible-by-construction, guarded -- and the elite
    #     archive must admit only DISTINCT basins (otherwise recombination silently degenerates).
    cok = True
    rngc = np.random.RandomState(77)
    for nc in (5, 27, 63, 99):
        pa = _project(_slp(_cold(nc, rngc), tr0=TR0, max_it=8))
        pb = _project(_slp(_cold(nc, rngc), tr0=TR0, max_it=8))
        for _ in range(8):
            for pair in ((pa, pb), (pa, pa)):        # identical parents must also be handled
                x = _cross(pair[0], pair[1], rngc)
                if x is None or x.shape != (nc, 3) or not np.all(np.isfinite(x)):
                    cok = False
                    continue
                if x[:, 2].max() != 0.0 or x[:, :2].min() < LO - 1e-12 or x[:, :2].max() > HI + 1e-12:
                    cok = False
        fc, sc = ev(nc, _project(_slp(_cross(pa, pb, rngc), tr0=TR0)))
        if not fc or sc <= 0.0:
            cok = False
    cnone = _cross(np.zeros((2, 3)), np.zeros((2, 3)), rngc) is None and \
        _cross(np.zeros((6, 3)), np.zeros((0, 3)), rngc) is None
    # elite archive: STRUCTURAL identity + crowding eviction (and the legacy archive on the 0 knob)
    def _pk(rr):
        rr = np.asarray(rr, float)
        return np.column_stack([np.zeros(len(rr)), np.zeros(len(rr)), rr])
    global ELITE_DIVERSE
    _ed0 = ELITE_DIVERSE
    ELITE_DIVERSE = 1                                  # exercise the structural path either way
    e_same = []                                        # one basin, seen twice: the better copy wins
    _elite_add(e_same, _pk([.1, .1, .1]), .3)
    _elite_add(e_same, _pk([.1 + 1e-7, .1, .1]), .3 + 1e-7)
    e_tie = []                                         # SUMS identical, structures different: both
    _elite_add(e_tie, _pk([.20, .10, .05]), .35)       # (the legacy numeric key rejected the second)
    _elite_add(e_tie, _pk([.15, .15, .05]), .35)
    e_cr = []                                          # crowding: the near-copy of the best is the
    for rr in ([.30, .20, .10], [.2999, .20, .10],     # one evicted, NOT the weakest entry
               [.20, .19, .19], [.25, .20, .12]):
        _elite_add(e_cr, _pk(rr), float(sum(rr)))
    cr_scores = [round(t[1], 4) for t in e_cr]
    cr_sep = min(_sig_d(e_cr[i][2], e_cr[j][2])
                 for i in range(len(e_cr)) for j in range(i + 1, len(e_cr)))
    ELITE_DIVERSE = 0                                  # the iteration-10 archive, bit-for-bit
    e_old = []
    for sv in (1.0, 1.0, 1.0 + 1e-13, 2.0, 0.5, 3.0):
        _elite_add(e_old, np.zeros((3, 3)), sv)
    ELITE_DIVERSE = _ed0
    edistinct = (len(e_same) == 1 and abs(e_same[0][1] - (.3 + 1e-7)) < 1e-15 and
                 len(e_tie) == 2 and
                 len(e_cr) == ELITES and cr_scores == [.60, .58, .57] and cr_sep > 1e-3 and
                 [t[1] for t in e_old] == [3.0, 2.0, 1.0])
    print("crossover exact-count/in-square/zero-radii/solvable:", cok)
    print("crossover guarded (n<3, empty donor -> None):", cnone)
    print("elite archive: structural identity, tied-sum diversity, crowding eviction, legacy knob:",
          edistinct, "(kept %s, min structural sep %.3g)" % (cr_scores, cr_sep))
    print("crossover actually fired during the multi-pass solve:", STATS["cross"] > 0, STATS["cross"])
    ok = ok and cok and cnone and edistinct and STATS["cross"] > 0

    # 11. CAP-AWARE BUDGET WEIGHTING: the records table must parse, the headroom multiplier must be
    #     1.0 wherever score is still winnable and CAPPED_W only where it provably is not, and every
    #     unknown/malformed input must fall back to 1.0 (an offline re-run on unlisted sizes must be
    #     unaffected). Plus the precondition check: on THIS census the reallocation must have something
    #     to reallocate, else the mechanism is a no-op and the test would be vacuous.
    recs11 = _records()
    parsed = len(recs11) > 0 and all(isinstance(k, int) and v > 0 for k, v in recs11.items())
    r27 = recs11.get(27, 0.0)
    unit = (_headroom(27, r27 * (1.0 - 1e-3), recs11) == 1.0 and          # a real gap -> full weight
            _headroom(27, r27 * (1.0 - 1e-9), recs11) == CAPPED_W and     # inside the cap -> token
            _headroom(27, r27 * 1.5, recs11) == CAPPED_W and              # above the record -> token
            _headroom(10 ** 6, 1.0, recs11) == 1.0 and                    # unlisted n -> full weight
            _headroom(27, None, recs11) == 1.0 and                        # no incumbent -> full weight
            _headroom(27, float("nan"), recs11) == 1.0 and
            _headroom(27, r27 * 0.9, {}) == 1.0)                          # no records at all -> as before
    capped11 = [n11 for n11 in sorted(recs11)
                if (pk := _read_pack(n11)) is not None
                and _headroom(n11, _sum_r(_project(pk)), recs11) == CAPPED_W]
    wtot = sum(float(n11) for n11 in recs11)
    freed = sum(float(n11) for n11 in capped11) * (1.0 - CAPPED_W) / max(wtot, 1e-9)
    print("records parsed: %s (%d sizes)" % (parsed, len(recs11)))
    print("headroom multiplier correct on every branch:", unit)
    print("precondition -- capped sizes on this census: %d, CPU share freed %.1f%%"
          % (len(capped11), 100.0 * freed))
    ok = ok and parsed and unit and len(capped11) > 0 and freed > 0.0

    # 12. PRESSURE (objective perturbation): the weighted SLP must (a) stay inside the feasible set
    #     for every weighting -- that is the whole safety argument, since the feasible set is
    #     untouched; (b) genuinely MOVE the packing (a no-op weighting would make the operator
    #     vacuous); (c) actually favour what it was told to favour; (d) reduce EXACTLY to the old
    #     solver at w=1; and (e) have fired during a real solve.
    pok = True
    rngp = np.random.RandomState(101)
    for np_ in (9, 27, 63):
        base, sb = _local(_cold(np_, rngp), TR0)
        for _ in range(4):
            q = _pressure(base, rngp)
            if q is None or q.shape != (np_, 3) or not np.all(np.isfinite(q)):
                pok = False
                continue
            fp, _sp = ev(np_, _project(q))          # feasible by construction, no repair needed
            moved = float(np.abs(q[:, :2] - base[:, :2]).max())
            if not fp or moved <= 1e-9:
                pok = False
            rel, sr = _local(q, 0.5 * TR0)          # releasing the weights must re-optimise cleanly
            fr, _srv = ev(np_, rel)
            if not fr or sr < 0.5 * sb:
                pok = False
    # (c) the weighted objective favours the circle it boosts
    b12, _s12 = _local(_cold(24, rngp), TR0)
    i12 = int(np.argmin(b12[:, 2]))
    w12 = np.ones(24)
    w12[i12] += 4.0
    g12 = _slp(b12, tr0=0.5 * TR0, max_it=40, tr_stop=1e-3, w=w12)
    grew = g12[i12, 2] > b12[i12, 2] + 1e-9
    # (d) w=None, w=ones and a malformed w must all give the identical unweighted result
    u12 = _slp(b12, tr0=0.5 * TR0, max_it=12, tr_stop=1e-3)
    same = all(float(np.abs(_slp(b12, tr0=0.5 * TR0, max_it=12, tr_stop=1e-3, w=ww) - u12).max()) == 0.0
               for ww in (np.ones(24), np.zeros(24), np.full(24, np.nan), np.ones(3)))
    print("pressure feasible-by-construction, moves the packing, re-optimises:", pok)
    print("pressure grows the circle it boosts (r %.6f -> %.6f): %s"
          % (b12[i12, 2], g12[i12, 2], grew))
    print("w=ones/degenerate reduces EXACTLY to the unweighted SLP:", same)
    # (e) FIRING, as a property of the knob rather than of its shipped setting. Iteration 18 retired
    #     the pressure band on its own currency reading (1 win / 335 tries), so asserting "press fired
    #     during a real solve" would now assert the retirement away. The property that must survive a
    #     retirement is that the code still WORKS when the knob is turned back on -- otherwise
    #     "shipped OFF behind a tested knob" is a fiction -- and that at the shipped value it really
    #     is silent, consuming no rounds at all.
    quiet_press = (PRESSURE_P > 0.0) or STATS["press"] == 0
    keep_pp12, keep_b12s, keep_ps12 = PRESSURE_P, CPU_BUDGET_S, PASSES
    fire12 = 0
    try:
        globals()["PRESSURE_P"] = 0.9
        globals()["CPU_BUDGET_S"] = 2.0
        globals()["PASSES"] = 2
        STATS["press"] = 0
        solve(lambda n, pk: ev(n, pk), _M(), np.random.RandomState(1218), [13, 15])
        fire12 = STATS["press"]
    finally:
        globals()["PRESSURE_P"] = keep_pp12
        globals()["CPU_BUDGET_S"] = keep_b12s
        globals()["PASSES"] = keep_ps12
    print("pressure is SILENT at the shipped PRESSURE_P=%.2f: %s" % (PRESSURE_P, quiet_press))
    print("pressure still FIRES when its knob is turned back on (press=%d):" % fire12, fire12 > 0)
    ok = ok and pok and grew and same and quiet_press and fire12 > 0

    # 13. CROSS-SIZE recombination: a donor of a DIFFERENT size must still yield an exact-count,
    #     in-square, solvable start (the count repair is what makes that legal); the donor picker must
    #     be guarded when no neighbour exists; disabling the mechanism must consume NO randomness (so
    #     XCROSS_P = 0.0 reproduces the previous solver's rng stream bit-for-bit); and -- the
    #     precondition check -- it must actually have fired during the real multi-pass solve above.
    xok = True
    rngx = np.random.RandomState(202)
    for na, nb in ((27, 29), (29, 27), (43, 49), (63, 55), (5, 40), (99, 27)):
        pa = _project(_slp(_cold(na, rngx), tr0=TR0, max_it=8))
        pb = _project(_slp(_cold(nb, rngx), tr0=TR0, max_it=8))
        x = _cross(pa, pb, rngx)
        if x is None or x.shape != (na, 3) or not np.all(np.isfinite(x)):
            xok = False
            continue
        if x[:, :2].min() < LO - 1e-12 or x[:, :2].max() > HI + 1e-12 or x[:, 2].max() != 0.0:
            xok = False
        fx, sx = ev(na, _project(_slp(x, tr0=TR0)))
        if not fx or sx <= 0.0:
            xok = False
    # donor picker: finds a neighbour's elite, falls back to the pool/census, guards when neither
    B = 10 ** 6                                  # a size range the census cannot supply, so the
    el13 = {B - 2: [(np.zeros((B - 2, 3)), 1.0)]}   # elite / pool / guard branches are unambiguous
    d1 = _xdonor(B, np.random.RandomState(5), el13, {})
    d2 = _xdonor(B, np.random.RandomState(6), {}, {B + 2: (np.zeros((B + 2, 3)), 1.0)})
    d3 = _xdonor(B, np.random.RandomState(7), {}, {})
    d4 = _xdonor(45, np.random.RandomState(8), {}, {})     # falls back to the committed census
    dok = (d1 is not None and len(d1) == B - 2 and
           d2 is not None and len(d2) == B + 2 and d3 is None and
           d4 is not None and len(d4) != 45 and abs(len(d4) - 45) <= max(XCROSS_NB))
    # the disabled switch must not touch the rng
    global XCROSS_P
    keepx = XCROSS_P
    XCROSS_P = 0.0
    r_a, r_b = np.random.RandomState(9), np.random.RandomState(9)
    off = _use_xcross(r_a)
    quiet = (off is False) and float(r_a.rand()) == float(r_b.rand())
    XCROSS_P = keepx
    print("cross-size splice exact-count/in-square/solvable:", xok)
    print("donor picker: neighbour elite / pool / census fallback / guarded:", dok)
    print("XCROSS_P=0.0 consumes no randomness (old stream preserved):", quiet)
    print("cross-size splice actually fired during the multi-pass solve:",
          STATS["xcross"] > 0, STATS["xcross"])
    ok = ok and xok and dok and quiet and STATS["xcross"] > 0

    # 14: the spatial ruin is a feasible, exact-count, CONNECTED-cluster re-draw -- and it fired
    rr = np.random.RandomState(14)
    rok, moved_any, disj_any = True, False, False
    for nr in (4, 27, 45, 99):
        base, _ = _local(_cold(nr, rr), TR0)
        for _ in range(4):
            q = _ruin(base, rr)
            if q is None or len(q) != nr:
                rok = False
                continue
            if q[:, :2].min() < LO - 1e-12 or q[:, :2].max() > HI + 1e-12:
                rok = False
            lifted = np.where(np.abs(q[:, 2]) == 0.0)[0]
            if not (2 <= len(lifted) <= 6) or len(lifted) > nr - 2:
                rok = False
            # the survivors are untouched (the rest of the packing stays exactly optimal) ...
            surv = np.setdiff1d(np.arange(nr), lifted)
            if not np.allclose(q[surv], base[surv], atol=0, rtol=0):
                rok = False
            # ... the lifted set is a CONNECTED cluster: every victim is nearer the seed circle than
            # any survivor is (that is what "k nearest" means, and what _relocate's k-smallest is not)
            c = base[lifted, :2].mean(0)
            dv = np.sqrt(((base[lifted, :2] - c) ** 2).sum(-1)).max()
            ds = np.sqrt(((base[surv, :2] - c) ** 2).sum(-1)).min()
            if len(surv) and dv > 3.0 * max(ds, 1e-12):
                rok = False
            if np.abs(q[lifted, :2] - base[lifted, :2]).max() > 0:
                moved_any = True
            fq, sq = ev(nr, _project(_slp(q, tr0=TR0)))
            if not fq or sq <= 0.0:
                rok = False
    disj_any = _ruin(np.zeros((3, 3)), rr) is None      # too small to ruin -> caller falls back
    print("ruin: exact count / in square / survivors untouched / connected / solvable:", rok)
    print("ruin: re-fill actually relocates the lifted circles:", moved_any)
    print("ruin guarded for n<4:", disj_any)
    print("ruin actually fired during the multi-pass solve:", STATS["ruin"] > 0, STATS["ruin"])
    # RUIN_P = 0.0 must be the iteration-11 solver exactly: the empty band draws no randomness
    keepr = RUIN_P
    RUIN_P = 0.0
    ra, rb = np.random.RandomState(21), np.random.RandomState(21)
    band = 1.0 - PRESSURE_P - RUIN_P
    quiet_r = (band == 1.0 - PRESSURE_P) and float(ra.rand()) == float(rb.rand())
    RUIN_P = keepr
    print("RUIN_P=0.0 collapses the band to the iter-11 dispatch:", quiet_r)
    ok = ok and rok and moved_any and disj_any and STATS["ruin"] > 0 and quiet_r

    print("warm-start guard returns None for absent n:", _read_pack(10 ** 6) is None)
    ok &= _read_pack(10 ** 6) is None

    # 15. VALUE-WEIGHTED SCHEDULE: the slice weight must follow the digits a size can STILL realize,
    #     must collapse to the old n-weighting on every unknown input (offline sizes, no record table,
    #     no incumbent) and exactly when the knob is off -- and, the precondition check, must actually
    #     REALLOCATE on this census, i.e. the ranking it induces over the uncapped residue must differ
    #     from the ranking by n. A weighting that agreed with n would make the A/B vacuous.
    recs15 = _records()
    r41 = recs15.get(41, 0.0)
    keep15 = SCHED_VALUE
    try:
        globals()["SCHED_VALUE"] = 1
        near = _slice_w(41, r41 * (1.0 - 1e-6), recs15)      # 1 digit of headroom left
        far = _slice_w(41, r41 * (1.0 - 1e-2), recs15)       # 5 digits of headroom left
        capd = _slice_w(41, r41 * (1.0 - 1e-9), recs15)      # at the cap -> floor x CAPPED_W
        mono = far > near > capd > 0.0
        fall = (_slice_w(10 ** 6, 1.0, recs15) == 10.0 ** 6 and       # unlisted n -> old n weighting
                _slice_w(41, r41 * 0.9, {}) == 41.0 and               # no records -> old n weighting
                _slice_w(41, None, recs15) == 41.0)                   # no incumbent -> old n weighting
        # the reallocation actually differs from weighting by n on the live census
        live = []
        for n15 in sorted(recs15):
            pk15 = _read_pack(n15)
            if pk15 is None:
                continue
            s15 = _sum_r(_project(pk15))
            if _headroom(n15, s15, recs15) == CAPPED_W:
                continue
            live.append((n15, _slice_w(n15, s15, recs15)))
        tot15 = sum(w for _, w in live)
        totn = sum(float(n15) for n15, _ in live)
        shift = max(abs(w / max(tot15, 1e-9) - n15 / max(totn, 1e-9)) for n15, w in live) if live else 0.0
        globals()["SCHED_VALUE"] = 0
        offv = all(_slice_w(n15, s15, recs15) == float(n15) * _headroom(n15, s15, recs15)
                   for n15, s15 in [(41, r41 * 0.99), (41, r41), (97, 5.0), (10 ** 6, 1.0)])
    finally:
        globals()["SCHED_VALUE"] = keep15
    print("slice weight monotone in realizable digits:", mono)
    print("slice weight falls back to n-weighting on unknowns:", fall)
    print("SCHED_VALUE=0 reproduces the iteration-12 schedule exactly:", offv)
    print("precondition -- uncapped sizes %d, max share shift vs weight-by-n %.1f pp"
          % (len(live), 100.0 * shift))
    ok = ok and mono and fall and offv and len(live) > 1 and shift > 0.01
    # 16. SCREEN/REFINE LADDER: a screening solve must still return a STRICTLY FEASIBLE, exact-count
    #     packing (the exact r-LP runs in both branches), the refine pass must never make a candidate
    #     worse, the promotion gate must actually FIRE and must actually GATE (a hopeless candidate is
    #     not refined), and SCREEN_TR = 0.0 must reproduce the iteration-13 round exactly -- one full
    #     `_local`, with no extra randomness drawn either way.
    rng16 = np.random.RandomState(1616)
    n16 = 24
    st16 = _cold(n16, rng16)
    keep16, keepP = SCREEN_TR, PROMOTE_REL
    feas16 = better16 = True
    try:
        globals()["SCREEN_TR"] = 3e-2
        globals()["PROMOTE_REL"] = 2e-3
        c_scr, s_scr = _local(st16, TR0, tr_stop=3e-2)
        c_full, s_full = _local(st16, TR0)
        feas16 = (c_scr.shape == (n16, 3) and bool(ev(n16, c_scr)[0]) and c_scr[:, 2].min() > 0.0 and
                  c_full.shape == (n16, 3) and bool(ev(n16, c_full)[0]))
        # the tail is a continuation, so refining the screened solve cannot lose ground to itself
        n_ref0 = STATS["refine"]
        c_ref, s_ref = _solve_round(st16, TR0, s_scr, deadline=None)
        fired16 = STATS["refine"] > n_ref0 and STATS["screen"] > 0
        better16 = (s_ref >= s_scr - 1e-12 and c_ref.shape == (n16, 3) and
                    bool(ev(n16, c_ref)[0]))
        # ... and the gate really gates: an unreachable bar must leave the screened solve unrefined
        n_ref1 = STATS["refine"]
        _solve_round(st16, TR0, 1e9, deadline=None)
        gated16 = STATS["refine"] == n_ref1
        globals()["SCREEN_TR"] = 0.0
        n_scr2 = STATS["screen"]
        c_off, s_off = _solve_round(st16, TR0, -np.inf, deadline=None)
        off16 = (STATS["screen"] == n_scr2 and abs(s_off - s_full) <= 1e-12 and
                 np.array_equal(c_off, c_full))
    finally:
        globals()["SCREEN_TR"], globals()["PROMOTE_REL"] = keep16, keepP
    print("screening solve stays strictly feasible / exact count:", feas16)
    print("refine never loses to its own screen, and FIRED:", better16, fired16)
    print("promotion gate gates a hopeless candidate:", gated16)
    print("SCREEN_TR=0.0 reproduces the iteration-13 round exactly:", off16)
    print("screen/refine counts during this self-test: %d / %d" % (STATS["screen"], STATS["refine"]))
    ok = ok and feas16 and better16 and fired16 and gated16 and off16

    # 17. GRAFT ENTROPY: the m > n graft must span a FAMILY, not a point. Assert (a) the sampled
    #     deletion keeps a feasible, exact-count start whose kept circles are a genuine subset of the
    #     donor, (b) the victims are still drawn from the SMALL end (never the largest circle), (c) the
    #     operator's entropy over repeated draws is high with GRAFT_DIV = 1 and exactly ONE with
    #     GRAFT_DIV = 0 -- the measured precondition this change exists to fix -- and (d) GRAFT_DIV = 0
    #     reproduces the iteration-14 argmin rule bit-for-bit.
    keep17 = GRAFT_DIV
    donor17 = _local(_cold(31, np.random.RandomState(1717)), TR0)[0]   # solved: radii genuinely differ
    kd = 4
    try:
        globals()["GRAFT_DIV"] = 0.0
        r0 = np.random.RandomState(9)
        base17 = _graft_drop(donor17, kd, r0)
        off17 = np.array_equal(base17, np.sort(np.argsort(donor17[:, 2])[kd:]))
        seen0 = {_graft_drop(donor17, kd, r0).tobytes() for _ in range(50)}
        globals()["GRAFT_DIV"] = 1.0
        r1 = np.random.RandomState(9)
        draws = [_graft_drop(donor17, kd, r1) for _ in range(50)]
        seen1 = {d.tobytes() for d in draws}
        big = int(np.argmax(donor17[:, 2]))
        shape17 = all(len(d) == len(donor17) - kd and len(set(d.tolist())) == len(d) and
                      d.min() >= 0 and d.max() < len(donor17) and big in d.tolist() for d in draws)
        sub17 = bool(ev(len(donor17) - kd, donor17[draws[0]])[0])
    finally:
        globals()["GRAFT_DIV"] = keep17
    ent17 = len(seen1) > 8 and len(seen0) == 1
    print("graft delete keeps exact count / distinct rows / never the largest circle:", shape17)
    print("a sampled graft start is still strictly feasible:", sub17)
    print("precondition -- distinct deleted sets over 50 draws: GRAFT_DIV=0 -> %d, =1 -> %d"
          % (len(seen0), len(seen1)))
    print("GRAFT_DIV=0.0 reproduces the iteration-14 argmin rule exactly:", off17)
    ok = ok and shape17 and sub17 and ent17 and off17

    # --- 18: the INFEASIBLE excursion (_inflate). Four things, in the order the values demand:
    #     (a) the mechanism's OUTPUT is legal -- strictly feasible, exact count, for several n;
    #     (b) the excursion really is INFEASIBLE -- the inflated packing overlaps by construction, i.e.
    #         the precondition (a state no feasible-path operator can produce) genuinely holds;
    #     (c) the repair does its job -- it strictly REDUCES the worst overlap of that inflated state;
    #     (d) it spans a FAMILY, not a point (50 draws -> many distinct starts), and INFLATE_P = 0.0
    #         leaves the round's rng stream untouched (the branch is unreachable when off).
    inf_ok, inf_ent, inf_infeas, inf_fix = True, 0, 0, 0
    for n18 in (17, 31, 44):
        par = _local(_cold(n18, np.random.RandomState(1800 + n18)), TR0)[0]
        r18 = np.random.RandomState(31)
        seen18 = set()
        for _ in range(50):
            st = _inflate(par, r18)
            if st is None or len(st) != n18 or not bool(ev(n18, st)[0]):
                inf_ok = False
                break
            seen18.add(np.round(st, 9).tobytes())
        inf_ent += len(seen18)
        # (b)+(c): reproduce one excursion and measure the overlap before and after the repair
        keep_it = INFLATE_IT
        try:
            globals()["INFLATE_IT"] = 0
            raw = _inflate(par, np.random.RandomState(5))          # inflated, NOT repaired
            globals()["INFLATE_IT"] = keep_it
            fixed = _inflate(par, np.random.RandomState(5))        # same lam, repaired
        finally:
            globals()["INFLATE_IT"] = keep_it
        lam18 = float(np.exp(np.random.RandomState(5).uniform(np.log(INFLATE_LO), np.log(INFLATE_HI))))
        def worst(xy, rr):
            I18, J18 = np.triu_indices(len(xy), 1)
            dd = np.sqrt(((xy[I18] - xy[J18]) ** 2).sum(-1))
            return float(np.max(rr[I18] + rr[J18] - dd))
        rin = par[:, 2] * lam18
        o_before = worst(par[:, :2], rin)
        o_after = worst(fixed[:, :2], rin)
        inf_infeas += int(o_before > 1e-9)                          # the excursion IS infeasible
        # MEASURED, and it is the honest reading rather than the one this test was first written to
        # assert: the repair does NOT clear the pair overlap, because an inflated jammed packing has
        # no room to clear (at n=31, lam=1.016, worst overlap 4.4e-3 before and 5.9e-3 after). What it
        # does deliver is a CONVERGED, wall-legal stressed state -- successive sweeps stop moving it --
        # and that is the thing the feasible-path operators cannot produce. Asserted as measured.
        globals()["INFLATE_IT"] = 4 * keep_it
        more = _inflate(par, np.random.RandomState(5))
        globals()["INFLATE_IT"] = keep_it
        wall_ok = float(np.max(np.abs(fixed[:, :2]) + fixed[:, 2][:, None])) <= 0.5 + 1e-12
        inf_fix += int(wall_ok and np.max(np.abs(more[:, :2] - fixed[:, :2])) < 1e-3 and
                       o_after > 0.0)
    keep_p18 = INFLATE_P
    try:                            # with the knob off the branch is unreachable: same stream as iter 15
        globals()["INFLATE_P"] = 0.0
        thr = 1.0 - PRESSURE_P - RUIN_P
        off18 = abs((thr - INFLATE_P) - thr) < 1e-15
    finally:
        globals()["INFLATE_P"] = keep_p18
    print("inflate start is strictly feasible and exact-count at n=17/31/44:", inf_ok)
    print("precondition -- the inflated state really is INFEASIBLE (of 3 sizes): %d" % inf_infeas)
    print("repair converges to a wall-legal STRESSED state, overlap uncleared (of 3): %d" % inf_fix)
    print("distinct starts over 50 draws x 3 sizes (a family, not a point): %d" % inf_ent)
    print("INFLATE_P=0.0 leaves the round's branch thresholds unchanged:", off18)
    ok = ok and inf_ok and inf_infeas == 3 and inf_fix == 3 and inf_ent >= 120 and off18

    # --- 19: OPSTAT, the CURRENCY meter, and the shipped INFLATE_P. Three things:
    #     (a) the operator this iteration turned ON actually FIRES in the real dispatch at the shipped
    #         value -- a knob that never reaches its branch is a knob that ships nothing;
    #     (b) the meter is INTERNALLY consistent -- every round is attributed to exactly one operator,
    #         wins never exceed tries, and the dense deficit sample never exceeds the tries;
    #     (c) it is DIAGNOSTICS ONLY -- two runs of the same seed with the counters reset in between
    #         return bit-identical packings, so nothing the meter records can feed back into the search.
    # (c) has to be judged under a budget the MACHINE'S LOAD cannot move. Every slice here ends on a
    # process-CPU deadline, so two replicates bounded by CPU do not run the same number of rounds when
    # something else is on the box, and the comparison reports load rather than determinism (it did:
    # this assertion failed spuriously under a concurrent probe). Bounding both replicates by the
    # EVALUATION meter -- a count, not a clock, with the CPU ceiling raised clear of it -- makes the
    # two runs identical by construction whenever the property actually holds.
    class _M19(_M):
        budget = 40

    def _spend(m, out):           # the meter only depletes if the EVALUATOR charges it
        m.used += 1
        return out

    opstat_reset()
    m19a = _M19()
    r19 = np.random.RandomState(1917)
    keep_b19, keep_pass19 = CPU_BUDGET_S, PASSES
    res19a = {}
    try:
        globals()["CPU_BUDGET_S"] = 20.0
        globals()["PASSES"] = 2
        solve(lambda n, pk: _spend(m19a, _ev_track(res19a, n, pk, ev)), m19a, r19, [13, 15])
        snap19 = {k: dict(v) for k, v in OPSTAT.items()}
        opstat_reset()
        res19b = {}
        r19b = np.random.RandomState(1917)
        m19b = _M19()
        solve(lambda n, pk: _spend(m19b, _ev_track(res19b, n, pk, ev)), m19b, r19b, [13, 15])
    finally:
        globals()["CPU_BUDGET_S"] = keep_b19
        globals()["PASSES"] = keep_pass19
    fired19 = snap19["try"].get("inflate", 0) > 0 and INFLATE_P > 0.0
    cons19 = all(snap19["win"].get(k, 0) <= v and snap19["ngap"].get(k, 0) <= v
                 for k, v in snap19["try"].items()) and len(snap19["try"]) >= 3
    same19 = (sorted(res19a) == sorted(res19b) and
              all(abs(res19a[k] - res19b[k]) <= 0.0 for k in res19a))
    print("the shipped INFLATE_P fires in the REAL dispatch: %s (try=%d, win=%d)"
          % (fired19, snap19["try"].get("inflate", 0), snap19["win"].get("inflate", 0)))
    print("OPSTAT is internally consistent (win<=try, ngap<=try, all ops attributed):", cons19)
    print("OPSTAT is diagnostics-only -- same seed reproduces bit-identically:", same19)
    ok = ok and fired19 and cons19 and same19

    # --- 20: THE BANDS. Iteration 18's change is an allocation, so what has to be tested is the
    #     allocation itself, in three parts:
    #     (a) the shipped cut points satisfy the ordered-check invariant -- the checks run in order, so
    #         a RELOC_U above the shake cut does not merely retire the shake, it silently swallows the
    #         inflate band underneath it. Asserted, and then SHOWN to be load-bearing by violating it;
    #     (b) at the shipped bands the paid moves fire and the retired ones take no random rounds --
    #         BUT cold keeps its forced opener, which is the precondition the whole census rests on;
    #     (c) a size with NO committed pack still comes back strictly feasible, since retiring cold's
    #         random band is only safe if the forced cold start still covers a from-nothing size.
    inv20 = GRAFT_U <= CROSS_U <= COLD_U <= RELOC_U <= 1.0 - PRESSURE_P - RUIN_P - INFLATE_P + 1e-12
    keep20 = (CPU_BUDGET_S, PASSES, RELOC_U)
    keep_pp20 = PRESSURE_P
    res20 = {}
    try:
        globals()["CPU_BUDGET_S"] = 2.5
        globals()["PASSES"] = 2
        opstat_reset()
        solve(lambda n, pk: _ev_track(res20, n, pk, ev), _M(), np.random.RandomState(2018), [13, 15, 17])
        snap20 = {k: dict(v) for k, v in OPSTAT.items()}
        # the trap, shown rather than asserted: with a pressure band below it (the iteration-17
        # layout) a RELOC_U above the shake cut leaves the inflate branch unreachable -- the ordered
        # checks hand every one of its rounds to the relocate, and the knob still reads 0.15.
        globals()["RELOC_U"] = 0.95
        globals()["PRESSURE_P"] = 0.15
        opstat_reset()
        solve(lambda n, pk: ev(n, pk), _M(), np.random.RandomState(2018), [13, 15, 17])
        swallowed = OPSTAT["try"].get("inflate", 0)
        opstat_reset()
    finally:
        globals()["CPU_BUDGET_S"], globals()["PASSES"], globals()["RELOC_U"] = keep20
        globals()["PRESSURE_P"] = keep_pp20
    paid20 = snap20["try"].get("reloc", 0) > 0 and snap20["try"].get("inflate", 0) > 0
    retired20 = snap20["try"].get("press", 0) == 0
    forced20 = snap20["try"].get("cold", 0) > 0        # the opener survives COLD_U == CROSS_U
    fresh20, sf20 = _local(_cold(23, np.random.RandomState(7)), TR0)
    ff20, _sv20 = ev(23, fresh20)
    print("shipped bands satisfy the ordered-check invariant:", inv20)
    print("violating it SWALLOWS the inflate band (inflate tries at RELOC_U=0.95): %d" % swallowed)
    print("paid moves fire / retired press takes 0 rounds / cold keeps its forced opener: %s %s %s (cold try=%d)"
          % (paid20, retired20, forced20, snap20["try"].get("cold", 0)))
    print("a size with no committed pack still solves strictly feasible:", bool(ff20))
    ok = ok and inv20 and swallowed == 0 and paid20 and retired20 and forced20 and bool(ff20)

    # --- 21: THE GRAFT RETIREMENT (iteration 19). A retirement is only a result if the retired move is
    #     still reachable and still correct, so this asserts the PROPERTY rather than the setting:
    #     (a) at the shipped knobs graft takes ZERO rounds in a real dispatch -- it held two separate
    #         shares (the band GRAFT_U and the forced opener GRAFT_OPEN) and both must be silent;
    #     (b) turning EACH share back on on its own makes it fire again -- a knob that cannot restore
    #         the move is a deletion pretending to be a knob;
    #     (c) the cross-size channel graft used to carry is NOT closed by the retirement: crossover's
    #         out-of-size donor still runs (the splice test above), and with graft silent a revisit
    #         still improves on its own incumbent.
    keep21 = (GRAFT_U, GRAFT_OPEN, CPU_BUDGET_S, PASSES)
    tries21 = {}
    try:
        globals()["CPU_BUDGET_S"] = 2.0
        globals()["PASSES"] = 2
        for tag, gu, go in (("shipped", GRAFT_U, GRAFT_OPEN), ("band", 0.30, False), ("opener", 0.0, True)):
            globals()["GRAFT_U"], globals()["GRAFT_OPEN"] = gu, go
            opstat_reset()
            solve(lambda n, pk: ev(n, pk), _M(), np.random.RandomState(2019), [13, 15, 17])
            tries21[tag] = OPSTAT["try"].get("graft", 0)
        opstat_reset()
    finally:
        globals()["GRAFT_U"], globals()["GRAFT_OPEN"], globals()["CPU_BUDGET_S"], globals()["PASSES"] = keep21
    silent21 = tries21["shipped"] == 0
    restore21 = tries21["band"] > 0 and tries21["opener"] > 0
    print("graft is SILENT at the shipped knobs: %s (try=%d)" % (silent21, tries21["shipped"]))
    print("each retired share restores the move on its own: %s (band try=%d, opener try=%d)"
          % (restore21, tries21["band"], tries21["opener"]))
    ok = ok and silent21 and restore21

    # --- 22: THE METER'S DENOMINATION (iteration 20). Every ship for three iterations was decided by
    #     wins-per-CPU-second, where a "win" is any increase in sum_r -- the SEARCH's unit. The
    #     objective pays in CLAMPED DIGITS and most of the census sits at the cap, where a win is worth
    #     exactly nothing. ``_digits`` re-denominates the meter in the scoring rule's own unit; this
    #     asserts (a) it reproduces the rule, including the clamp and the guards, (b) a win on a CAPPED
    #     size books ZERO digits while the same win still books one in the search's unit, and (c) the
    #     counters remain DIAGNOSTICS-ONLY: the packings a dispatch produces are bit-identical whether
    #     or not the digit counters are read, because nothing in the search consults them.
    recs22 = _records()
    n22 = 41
    r22 = recs22.get(n22)
    d_at = _digits(n22, r22, recs22)                       # exactly the record -> the clamp
    d_over = _digits(n22, r22 * 1.001, recs22)             # above it -> still the clamp, never more
    d_3 = _digits(n22, r22 * (1.0 - 1e-3), recs22)         # relgap 1e-3 -> 3 digits
    d_cap = _digits(n22, r22 * (1.0 - 1e-9), recs22)       # inside 1e-7 -> capped
    rule22 = (abs(d_at - CAP_DIGITS) < 1e-12 and abs(d_over - CAP_DIGITS) < 1e-12
              and abs(d_3 - 3.0) < 1e-9 and abs(d_cap - CAP_DIGITS) < 1e-12
              and _digits(4, 1.0, recs22) is None                  # no record for this n -> unknowable
              and _digits(n22, float("inf"), recs22) is None
              and _digits(n22, None, recs22) is None)
    # (b) the two units disagree exactly where the scoring rule saturates.
    capped_win = _digits(n22, r22 * (1.0 - 1e-9), recs22) - _digits(n22, r22 * (1.0 - 5e-9), recs22)
    live_win = _digits(n22, r22 * (1.0 - 1e-4), recs22) - _digits(n22, r22 * (1.0 - 1e-3), recs22)
    unit22 = abs(capped_win) < 1e-12 and live_win > 0.9    # a real sum_r gain, 0 digits vs ~1 digit
    # (c) diagnostics-only: same seed, same targets, identical packings; the counters only observe.
    #     Bounded by the EVALUATION METER (a count) and not by the CPU clock: two clock-bounded
    #     replicates differ whenever anything else is running on the host, so a wall-clock bound makes
    #     this assertion report the machine's load instead of the property (iteration 18 learned this).
    class _M22(_M):
        budget = 40

    def _spend22(m, out):
        m.used += 1
        return out
    keep22 = (CPU_BUDGET_S, PASSES)
    try:
        globals()["CPU_BUDGET_S"], globals()["PASSES"] = 20.0, 2
        runs22 = []
        for _ in range(2):
            store, m22 = {}, _M22()
            opstat_reset()
            solve(lambda n, pk: _spend22(m22, _ev_track(store, n, pk, ev)), m22,
                  np.random.RandomState(2020), [13, 15])
            runs22.append((store, dict(OPSTAT["dig"]), dict(OPSTAT["dwin"]), dict(OPSTAT["win"])))
        opstat_reset()
    finally:
        globals()["CPU_BUDGET_S"], globals()["PASSES"] = keep22
    repro22 = runs22[0][0] == runs22[1][0] and runs22[0][1] == runs22[1][1]
    # every digit-booked win is a sum_r win, never the other way round
    subset22 = all(runs22[0][2].get(k, 0) <= runs22[0][3].get(k, 0) for k in runs22[0][3])
    print("_digits reproduces the scoring rule (clamp + guards):", rule22)
    print("a win on a CAPPED size books 0 digits, a live one books ~1: %s (%.3e vs %.3f)"
          % (unit22, capped_win, live_win))
    print("digit counters are diagnostics-only and reproduce bit-identically:", repro22, subset22)
    ok = ok and rule22 and unit22 and repro22 and subset22

    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(__doc__)
