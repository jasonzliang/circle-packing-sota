"""circle-packing-v2 solver: maximize sum of radii of n circles in the unit square.

Iteration 3: the outer search became a POPULATION, the local solve became CHEAP, and CPU is now
allocated by MARGINAL VALUE.  Measured facts that forced it: a hop's SLP reaches 9 correct digits in
~5 LPs and then burns ~30 more shrinking the trust box for ZERO gain (85% of every hop wasted), and at
2.35 s/n that bought only ~12 hops per n per iteration -- a hill-climb far too short to change a
structure.  Three coupled changes:

  (a) STALL-BASED EARLY EXIT + INCUMBENT BAIL-OUT.  _slp_refine stops when the objective stops moving
      instead of when the trust box hits 1e-11, and a hop that has converged BELOW the incumbent is
      abandoned the moment that is provable-enough.  Only a candidate that actually beats the
      incumbent earns a DEEP polish (the old full schedule) to recover the last 1e-13 digits, so
      precision is paid for exactly where the log-scale score can use it.
  (b) POOL, NOT A SINGLE INCUMBENT.  Hops are drawn from a pool of the best few STRUCTURALLY DISTINCT
      packings, not always from the champion.  A hill-climb on one point cannot leave its basin; a
      pool can hold a worse-but-different structure long enough for a second move to redeem it.
  (c) DIRECTED MOVE: RESEAT INTO THE LARGEST HOLE.  The wasted circles are the small ones; the useful
      place is the point of maximum free radius.  A grid scan finds it in ~1 ms, so a reseat move is
      a *targeted* teleport rather than a uniform-random one.
  (d) CPU BY HEADROOM.  n=31 is already at the 7-digit cap and can gain NOTHING, yet it was taking a
      full 1/10 of the CPU.  solve() reads records.json (allowed) + the census and starves any n whose
      relgap is already below the cap, rolling that CPU into the n that can still move.

Iteration 2's engine is unchanged underneath: the local optimizer is a CONSERVATIVE SEQUENTIAL LP (SLP),
which replaced both of iteration 1's inner engines (Adam penalty ascent as the refiner, and the
fixed-centres radii LP).

  THE IDEA.  Around a feasible packing, linearize each pair constraint  ||c_i - c_j|| >= r_i + r_j
  in the step variables (dx, dy, dr).  Because the Euclidean norm is CONVEX, its linearization is an
  UNDER-estimate:  ||c+dc|| >= ||c|| + u . dc.  So the linearized constraint is *stricter* than the
  true one, and any point feasible for the LP is feasible for the REAL problem.  The walls are linear
  already, hence exact.  Maximizing  sum dr  over that polytope, inside a trust box |dx|,|dy| <= delta,
  dr <= rho, is therefore a small sparse LP whose every solution is a strictly feasible packing with a
  provably larger sum of radii.  Iterate, shrinking the trust box on failure -> a KKT point of the true
  NLP, reached in ~30 LPs (~0.1 s at n=45) with residual violation at the 1e-13 level.

  WHY IT IS A STRICT UPGRADE.  Penalty ascent only ever approaches feasibility from outside and stalls
  at ~1e-3 relative error; the fixed-centres LP moves radii but never centres.  SLP moves centres AND
  radii together and converges to machine precision.  Measured against SLSQP on the identical NLP it
  finds the SAME local optima ~400x faster (0.1 s vs 40-100 s per n), which is what makes the outer
  basin search affordable at all.

  TRUST-REGION ACTIVE SET (also provably safe).  A pair with slack >= 2*delta + 2*rho cannot bind:
  the step moves each centre by <= delta (distance drops by <= 2 delta) and each radius by <= rho.  So
  those rows are DROPPED.  Pruning by a heuristic margin instead is catastrophic -- one missing pair
  and the LP inflates two circles through each other -- so the bound is tied to the trust box, never
  guessed.  Wall rows are pruned on the same bound.

  OUTER SEARCH.  The SLP only finds the optimum of the basin it starts in, and the census packings are
  already basin-optimal, so the remaining gap is STRUCTURAL.  Basin hopping therefore applies a
  structural move (relocate the k smallest / k random circles; or shake all centres and deflate the
  radii) and re-solves exactly.  Every hop is a full exact local solve, so an accepted hop is a real
  improvement in the true problem, not a convergence artifact.

  COLD START (any n, including sizes never scored in-run).  Structured k x k grid + interstitial seeds,
  the census warm start when one exists, and uniform random starts are pushed through a short batched
  Adam penalty ascent -- kept from iteration 1, where it is genuinely good: cheap, vectorized basin
  *generation* -- then snapped feasible and handed to the SLP for exact convergence.

Metering: candidates are ranked locally, so evaluate() is spent only on strictly feasible, SLP-polished
packings (a few hundred units of 135k) -- CPU is the binding meter, as MISSION says.  Every deadline is
derived from time.process_time() AT ENTRY, so a second solve() call in the same process gets its own
full allowance.  All randomness comes from `rng`.
"""
import json
import os
import time

# BLAS threads, pinned BEFORE numpy loads.  Iteration 7 measured a 163x163 np.linalg.solve costing
# 22 s of process_time on this 64-core host, because process_time SUMS across threads -- a solver
# whose budget is a CPU clock is destroyed by thread thrash.  The driver already pins these, so in
# the metered run this is a no-op; it matters when solver.py is run directly (--self-test) or by any
# harness that does not pin, and it means the self-test's timings are the ones the real run sees.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

LO, HI = -0.5, 0.5
FEAS_TOL = 1e-9
SAFE_TOL = 1e-12          # our own, stricter than the harness tolerance

# Per-call process-CPU allowance.  The in-run driver gives 32 s for ALL targets; the offline held-out
# re-run gives 60 s for a single size.  Both are computed at entry, never as a module constant.
CPU_TOTAL_MULTI = 27.0
CPU_SINGLE = 52.0

# HiGHS clamps its feasibility tolerances at 1e-10; the default 1e-7 is far too loose here -- it lets
# the LP return steps that violate a pair by ~1e-7, which reads as a spurious gain and poisons the
# incumbent.  The strict re-check in _slp_refine is the real guard; this just wastes fewer steps.
_LPOPT = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}

# solve() must never let one bad n kill the others, but a blanket `except` also HIDES real bugs (it hid
# a broadcasting bug in iteration 1 and an UnboundLocalError in iteration 2).  So every swallowed
# exception is recorded here and the self-test asserts the list is empty.
LAST_ERRORS = []


# --------------------------------------------------------------------------- geometry helpers
def _walls(xy):
    """(..., n, 2) -> (..., n) distance from each centre to the nearest wall."""
    x, y = xy[..., 0], xy[..., 1]
    return np.minimum(np.minimum(x - LO, HI - x), np.minimum(y - LO, HI - y))


def _dists(xy):
    """(B, n, 2) -> (B, n, n) pairwise centre distances, diagonal = +inf."""
    n = xy.shape[-2]
    d = xy[:, :, None, :] - xy[:, None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    di = np.arange(n)
    dist[:, di, di] = np.inf
    return dist


def _slack(p):
    """Worst constraint slack of ONE pack (n,3): >= 0 means strictly feasible."""
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    v = min((x - r - LO).min(), (HI - x - r).min(), (y - r - LO).min(), (HI - y - r).min())
    if len(p) >= 2:
        d = np.sqrt(((p[:, None, :2] - p[None, :, :2]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        v = min(v, (d - r[:, None] - r[None, :]).min())
    return float(v)


def _feasible_sumr(packs):
    """Local mirror of the harness feasibility rule (so we only spend evals on winners)."""
    x, y, r = packs[..., 0], packs[..., 1], packs[..., 2]
    n = packs.shape[1]
    sum_r = r.sum(axis=1)
    ok = (r.min(axis=1) > 0.0)
    wall = np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min(axis=1)
    ok &= wall >= -FEAS_TOL
    if n >= 2:
        dist = _dists(packs[..., :2])
        ok &= (dist - r[:, :, None] - r[:, None, :]).min(axis=(1, 2)) >= -FEAS_TOL
    return ok, sum_r


def _retract(xy, r):
    """Make a BATCH (B,n,2)+(B,n) strictly feasible, then greedily regrow -> packs (B,n,3)."""
    B, n, _ = xy.shape
    w = _walls(xy)
    r = np.clip(r, 1e-9, None)
    r = np.minimum(r, np.maximum(w - 1e-12, 1e-9))
    if n >= 2:
        dist = _dists(xy)
        s = r[:, :, None] + r[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(s > 0, dist / np.maximum(s, 1e-300), np.inf)
        f = np.minimum(1.0, ratio.min(axis=(1, 2)))
        r = r * f[:, None] * (1.0 - 1e-13)
        for _ in range(6):
            grew = False
            for i in range(n):
                cap = np.minimum(w[:, i], (dist[:, i, :] - r).min(axis=1))
                gain = cap - r[:, i]
                pos = gain > 1e-15
                if pos.any():
                    r[pos, i] = cap[pos] * (1.0 - 1e-13)
                    grew = True
            if not grew:
                break
    r = np.maximum(r, 1e-12)
    return np.concatenate([xy, r[..., None]], axis=-1)


# --------------------------------------------------------------------------- exact radius repair
# Iteration 12.  _retract is the ONE stage every graph-space move ends in (hop, scan, beam, chain
# all finish `q -> clip -> _retract`), and it repairs an overlapping candidate by a GLOBAL
# multiplicative shrink: one deeply overlapping pair scales down all n radii, and the greedy regrow
# that follows only claws back what a per-circle sweep can see.  Iteration 11 measured the
# consequence -- a tabu walk whose nodes sat at a MEDIAN 40% sum_r deficit -- and blamed "the
# realisation".  artifacts/proto_real.py splits that blame three ways and the answer is narrower and
# more fixable than the diagnosis: over 60 swaps of each of n=35/37/39 the 7-step KKT Newton
# converges to residual ~9e-16 on the median swap (it is NOT broken), but on the tail where it does
# not (residual up to 9e-1) the global shrink destroys the pack.
#
# The fix is a change of representation for the repair.  With the CENTRES FROZEN, every constraint
# is LINEAR in r -- r_i + r_j <= d_ij and 0 <= r_i <= wall_i -- so "the best radii these centres
# admit" is not a heuristic question at all, it is a small LP that HiGHS answers optimally.  The
# greedy sweep was solving by hand a problem with an exact solver sitting in scipy.
def _lp_radii(xy):
    """Exact  max sum r  s.t. r_i + r_j <= d_ij, 0 <= r_i <= wall_i,  for FIXED centres (n,2).

    Returns r (n,) or None (no scipy / LP failure).  Pairs with d_ij >= wall_i + wall_j can never
    bind (the box bounds already imply them) and are dropped, which is most of the n^2/2 rows."""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return None
    n = len(xy)
    w = np.clip(_walls(xy[None])[0], 0.0, None)
    if n < 2:
        return w.copy()
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    iu, ju = np.triu_indices(n, 1)
    dv = d[iu, ju]
    keep = dv < (w[iu] + w[ju])
    iu, ju, dv = iu[keep], ju[keep], dv[keep]
    m = len(iu)
    try:
        if m:
            A = coo_matrix((np.ones(2 * m), (np.concatenate([np.arange(m), np.arange(m)]),
                                             np.concatenate([iu, ju]))), shape=(m, n))
            res = linprog(-np.ones(n), A_ub=A, b_ub=dv,
                          bounds=np.stack([np.zeros(n), w], 1), method="highs")
        else:
            return w.copy()
    except Exception:
        return None
    if not res.success or res.x is None:
        return None
    r = np.asarray(res.x, float)
    if not np.isfinite(r).all():
        return None
    return np.clip(r, 0.0, None)


LP_GATE = 2e-3            # relative sum_r a greedy repair may cost before the ~10 ms LP earns itself


def _repair(q, ref=None, deadline=None):
    """Turn a RAW graph-solve output (n,3) into a strictly feasible pack, LP-repairing the tail.

    Gated on purpose.  proto_real.py measured the two repairs to agree to ~1e-9 on the swaps the
    Newton actually solved -- there the pack is already near-tangent and there is nothing for an LP
    to recover -- while on the collapsing tail the LP is worth a MEAN 7e-3 (n=37) to 2e-1 (n=39) of
    sum_r per swap.  So run the cheap one first and only pay for the LP when the cheap one visibly
    collapsed, which keeps the throughput of the scan and buys the chain its manifold back.
    Never returns something worse than the greedy repair."""
    xy = np.ascontiguousarray(q[:, :2], dtype=float)
    np.clip(xy, LO + 1e-9, HI - 1e-9, out=xy)
    r0 = np.abs(np.asarray(q[:, 2], float)) + 1e-9
    z = _retract(xy[None].copy(), r0[None].copy())[0]
    s = float(z[:, 2].sum())
    tgt = float(np.abs(q[:, 2]).sum())
    if ref is not None:
        tgt = min(tgt, float(ref))
    if s >= tgt * (1.0 - LP_GATE):
        return z
    if deadline is not None and time.process_time() > deadline:
        return z
    r = _lp_radii(xy)
    if r is None:
        return z
    z2 = _retract(xy[None].copy(), r[None].copy())[0]
    return z2 if float(z2[:, 2].sum()) > s else z


# --------------------------------------------------------------------------- conservative SLP
def _slp_step(p, delta, rho, IU, JU, M=None, wv=None):
    """One trust-region LP step.  Returns a pack (n,3) or None.

    `wv` (length n, mean ~1) reweights the objective to  max sum(wv_i * dr_i)  instead of
    max sum(dr_i).  The CONSTRAINTS are untouched, so every weighted step is still a strictly
    feasible packing -- only the direction of ascent changes.  That is what makes objective-space
    continuation free: the whole conservative-SLP guarantee carries over unchanged.

    Every constraint handed to the LP is an UNDER-estimate of the true one (norm linearization) or
    exact (walls), and the active set is pruned only by the bound `slack >= 2*delta + 2*rho`, which
    no step inside the trust box can close.  So the LP optimum is a genuinely feasible packing."""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return None
    n = len(p)
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    lim = 2.0 * delta + 2.0 * rho
    rows, cols, vals, rhs = [], [], [], []
    off = 0
    if len(IU):
        dx = x[IU] - x[JU]
        dy = y[IU] - y[JU]
        d = np.sqrt(dx * dx + dy * dy)
        sl = d - r[IU] - r[JU]
        keep = sl < lim
        iu, ju = IU[keep], JU[keep]
        m = len(iu)
        if m:
            dd = np.maximum(d[keep], 1e-12)
            ux, uy = (x[iu] - x[ju]) / dd, (y[iu] - y[ju]) / dd
            k = np.arange(m)
            rows.append(np.tile(k, 6))
            cols.append(np.concatenate([iu, ju, n + iu, n + ju, 2 * n + iu, 2 * n + ju]))
            vals.append(np.concatenate([-ux, ux, -uy, uy, np.ones(m), np.ones(m)]))
            rhs.append(sl[keep])
            off = m
    for sgn, ax, b in ((-1.0, 0, x - r - LO), (1.0, 0, HI - x - r),
                       (-1.0, 1, y - r - LO), (1.0, 1, HI - y - r)):
        sel = np.nonzero(b < lim)[0]
        q = len(sel)
        if not q:
            continue
        k = off + np.arange(q)
        rows.append(np.tile(k, 2))
        cols.append(np.concatenate([ax * n + sel, 2 * n + sel]))
        vals.append(np.concatenate([np.full(q, sgn), np.ones(q)]))
        rhs.append(b[sel])
        off += q
    A = None
    if off:
        A = coo_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))), shape=(off, 3 * n))
    if M is None:
        lo = np.concatenate([np.maximum(-delta, LO - x), np.maximum(-delta, LO - y),
                             np.maximum(-rho, 1e-9 - r)])
        hi = np.concatenate([np.minimum(delta, HI - x), np.minimum(delta, HI - y),
                             np.full(n, rho)])
        c = np.zeros(3 * n)
        c[2 * n:] = -1.0 if wv is None else -wv
    else:
        # SYMMETRY-REDUCED step: the displacement is forced to z_full = Msp @ z_red, so every
        # iterate stays exactly inside the G-invariant subspace.  Composing the SAME linearized
        # constraint rows with Msp keeps the step conservative (A_red = A_full @ Msp is exact).
        Msp, kinds, memb, cred = M
        k = Msp.shape[1]
        lo = np.where(kinds == 0, -delta, np.maximum(-rho, 1e-9 - r[memb]))
        hi = np.where(kinds == 0, delta, rho)
        c = cred if wv is None else -np.asarray(Msp[2 * n:, :].T.dot(wv)).ravel()
        if A is not None:
            A = (A.tocsr() @ Msp)
    bnds = np.stack([lo, hi], axis=1)
    try:
        if A is None:
            res = linprog(c, bounds=bnds, method="highs", options=_LPOPT)
        else:
            res = linprog(c, A_ub=A, b_ub=np.concatenate(rhs), bounds=bnds,
                          method="highs", options=_LPOPT)
    except Exception:
        return None
    if not getattr(res, "success", False) or res.x is None:
        return None
    z = res.x
    if M is not None:
        z = np.asarray(M[0] @ z).ravel()
    return np.stack([x + z[:n], y + z[n:2 * n], r + z[2 * n:]], axis=1)


def _slp_refine(p, iters=90, delta=0.05, rho=0.05, dmin=1e-11, deadline=None,
                target=None, patience=4, gtol=1e-12, M=None, w=None):
    """Exact local solve by conservative SLP.  Input need NOT be optimal, but MUST be feasible.

    Only strictly feasible, strictly improving steps are accepted, so the returned pack is feasible
    by construction and its sum_r is an honest lower bound on the basin optimum.

    TWO EXITS ON TOP OF THE TRUST-BOX FLOOR, both measured rather than guessed:
      * STALL.  The trajectory reaches 9 correct digits in ~5 LPs and then spends ~30 more halving
        delta for nothing.  `patience` consecutive rounds gaining < `gtol` ends it.  That is a ~6x
        throughput win on every hop, and the cost is only the last few 1e-13 digits -- which the
        caller buys back with a DEEP refine (large `patience`, small `gtol`) on winners only.
      * TARGET.  `target` is the incumbent to beat.  Once the objective has stalled at all (2 rounds)
        and is still below it, this basin is decided: abandon it now rather than polish a loser."""
    n = len(p)
    IU, JU = np.triu_indices(n, 1) if n >= 2 else (np.empty(0, int), np.empty(0, int))
    best = np.array(p, dtype=float, copy=True)
    # With `w` the ASCENT is on the weighted objective w.r (a surrogate); monotonicity/stall/target
    # are all measured on whatever objective is actually being climbed, and the returned score is
    # always the TRUE sum of radii so callers never have to know which mode ran.
    wv = None if w is None else np.asarray(w, dtype=float).ravel()
    bs = float(best[:, 2].sum()) if wv is None else float(wv @ best[:, 2])
    stall = 0
    for _ in range(iters):
        if deadline is not None and time.process_time() > deadline:
            break
        q = _slp_step(best, delta, rho, IU, JU, M, wv)
        gain = 0.0
        if q is None:
            delta *= 0.5
            rho *= 0.5
        else:
            s = float(q[:, 2].sum()) if wv is None else float(wv @ q[:, 2])
            if s > bs + 1e-16 and q[:, 2].min() > 0.0 and _slack(q) >= -SAFE_TOL:
                gain = s - bs
                best, bs = q, s
                if gain < 1e-12:
                    delta *= 0.6
                    rho *= 0.6
                else:
                    delta = min(0.05, delta * 1.2)
                    rho = min(0.05, rho * 1.2)
            else:
                delta *= 0.45
                rho *= 0.45
        stall = stall + 1 if gain < gtol else 0
        if stall >= patience:
            break
        if target is not None and wv is None and stall >= 2 and bs <= target:
            break                      # converged, and converged short: this basin is a loser
        if delta < dmin:
            break
    return best, float(best[:, 2].sum())


def _slp_deep(p, deadline=None):
    """Full iteration-2 schedule: run a winner all the way down to the 1e-13 residual level."""
    return _slp_refine(p, iters=90, deadline=deadline, patience=14, gtol=1e-15)


# --------------------------------------------------------------------------- hard-target feasibility
# Iteration 13.  Every move above ascends sum_r from a FEASIBLE point, and all of them warm-start
# from the committed incumbent.  On n=35/37/39 that had produced exactly zero improvement for eight
# iterations, and artifacts/proto_infl3.py says why: the incumbent is not a nearly-right structure a
# local edit can fix, it is a TRAP, and every family in this file was chained to it.
#
# This family inverts the problem.  Fix only the TOTAL sum of radii to a target T and ask a
# FEASIBILITY question instead of an optimisation one:
#
#     minimise over (xy, a)   E = sum_{i<j} max(0, r_i+r_j-d_ij)^2 + sum_i sum_walls max(0, r_i-w)^2
#     where                   r = T * softmax(a)        (so sum r == T exactly, and r > 0 always)
#
# E == 0 at target T IS a feasible packing with sum_r = T.  Three things make it different in kind
# from everything else here, and the third is the one that pays:
#   * it travels through INFEASIBLE space, where circles slide past one another, so the contact
#     topology reorganises wholesale rather than one swapped edge at a time;
#   * the radius PROFILE is a free variable (n extra dof), not inherited from the incumbent;
#   * T is an INPUT, so the search can be pointed at a target and started COLD -- which is the only
#     configuration of this mechanism that has ever beaten these incumbents.
#
# Measured (proto_infl3.py, 6 CPU-s per mode): cold multi-start at T=record hit 3.161498916 on n=37
# and cold over-pressure continuation hit 3.074036364 on n=35 -- both the packomania record to 9
# digits -- while the WARM variants (continuation from the incumbent at every kappa in 0.02/0.08/0.25,
# 4 seeds each) returned the incumbent to 1e-13 every single time.  So this family is COLD ONLY on
# purpose; warming it is what kills it.
TF_KAPPA = 0.25           # over-pressure: start the continuation this far ABOVE T, where E cannot be 0
TF_STEPS = 14             # levels on the way back down to T
TF_ITER_C = 400           # L-BFGS iterations per continuation level
TF_ITER_M = 600           # ... and per single-shot multi-start solve
TF_SHARE = 0.90           # share of an OPEN n's CPU slice.  proto_tfrate.py measures the shipped
#                           phase as a LOTTERY -- 1 of 6 seeds hits the exact n=35 record in 6.2 s,
#                           0 of 6 on n=37 -- so tickets are the currency, and the warm mix it
#                           takes the share from has measured ZERO yield on these n for three
#                           iterations (10, 11, 12).  The incumbent is never at risk: it is re-read
#                           and SLP-refined at the top of _solve_one before this phase runs.


def _tf_eg(z, n, T):
    """E and its gradient w.r.t. z = [xy.ravel(), a], with r = T*softmax(a).

    Both signs in here were wrong on the first attempt (the pair term and the wall term), and both
    times the optimiser silently did nothing rather than failing loudly -- the first prototype
    reported a clean set of numbers that were entirely an artefact of cos(analytic, numeric) = -1.
    The self-test finite-differences this function for that reason."""
    xy = z[:2 * n].reshape(n, 2)
    a = z[2 * n:]
    e = np.exp(a - a.max())
    r = T * (e / e.sum())
    d = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    g = (r[:, None] + r[None, :]) - dist
    np.maximum(g, 0.0, out=g)
    np.fill_diagonal(g, 0.0)
    E = 0.5 * float((g * g).sum())
    c = np.where(g > 0, 2.0 * g / dist, 0.0)
    np.fill_diagonal(c, 0.0)
    gxy = -(c[:, :, None] * d).sum(1)                 # dE/dxy_i = -2 sum_j g_ij u_ij
    dEdr = 2.0 * g.sum(1)
    t = np.stack([r - (xy[:, 0] - LO), r - (HI - xy[:, 0]),
                  r - (xy[:, 1] - LO), r - (HI - xy[:, 1])], 1)
    np.maximum(t, 0.0, out=t)
    E += float((t * t).sum())
    gxy[:, 0] += 2.0 * (t[:, 1] - t[:, 0])
    gxy[:, 1] += 2.0 * (t[:, 3] - t[:, 2])
    dEdr += 2.0 * t.sum(1)
    ga = r * dEdr - (r / T) * float(dEdr @ r)         # softmax chain rule
    return E, np.concatenate([gxy.ravel(), ga])


def _tf_solve(n, T, z0, maxiter):
    """Descend E at a FIXED target T.  Returns (z, E) -- z is reusable as the next level's start."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return z0, np.inf
    res = minimize(_tf_eg, z0, args=(n, T), jac=True, method="L-BFGS-B",
                   options={"maxiter": int(maxiter), "maxcor": 20, "ftol": 1e-22, "gtol": 1e-16})
    z = np.asarray(res.x, float)
    if not np.isfinite(z).all():
        return z0, np.inf
    return z, float(res.fun)


def _tf_cold_z(n, rng, mode):
    """A COLD start for the target solve.  Never reads the incumbent -- that is the whole point."""
    if mode == "grid":
        k = max(1, int(np.ceil(np.sqrt(n))))
        ax = np.linspace(LO + 0.5 / k, HI - 0.5 / k, k)
        gx, gy = np.meshgrid(ax, ax)
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        if len(pts) < n:                                # k*k >= n always, but never trust it
            pts = np.concatenate([pts, (rng.rand(n - len(pts), 2) - 0.5) * 0.9], 0)
        xy = pts[:n] + (rng.rand(n, 2) - 0.5) * (0.6 / k)
        a = rng.randn(n) * 0.1
    else:
        xy = (rng.rand(n, 2) - 0.5) * 0.9
        a = np.zeros(n) if mode == "unif" else rng.randn(n) * 0.25
    return np.concatenate([np.clip(xy, LO + 1e-6, HI - 1e-6).ravel(), a])


def _tf_harvest(n, z, deadline):
    """Target-solve output -> strictly feasible pack at its exact local optimum, or (None, -inf).

    The descent lands NEAR-feasible (E ~ 1e-7), never exactly on the manifold, so the radii are
    re-derived at those centres by the iteration-12 LP -- which is exact there -- and the pack is
    then polished by SLP.  Feasibility is re-checked locally before anything is emitted."""
    xy = np.clip(z[:2 * n].reshape(n, 2), LO + 1e-9, HI - 1e-9)
    r = _lp_radii(xy)
    if r is None:
        return None, -np.inf
    q = _retract(np.ascontiguousarray(xy)[None].copy(), r[None].copy())[0]
    if q[:, 2].min() <= 0.0 or _slack(q) < -SAFE_TOL:
        return None, -np.inf
    p, s = _slp_refine(q, deadline=deadline, patience=8, gtol=1e-15)
    ok, sv = _feasible_sumr(p[None])
    if not ok[0]:
        return None, -np.inf
    return p, float(sv[0])


def _tf_target(n, cur_s):
    """The target T to aim the feasibility question at.

    For a size in the visible census the record is a published SCALAR and the mission states
    plainly that solve() may read bench/records.json -- so the question asked is exactly "is the
    record realisable".  For any other n (a held-out size, a fresh n) there is no entry, and a
    coordinate table would be memorisation and would not generalise anyway; instead the visible
    records are fitted by least squares to the two-parameter law  sum_r ~ c*sqrt(n) + b  (the
    asymptotically correct shape: n circles of radius ~ c/sqrt(n)), which extrapolates to any n.
    Whatever the source, T is never allowed below the incumbent -- a target under what we already
    hold asks a question whose answer we have."""
    T = None
    try:
        with open("bench/records.json") as f:
            recs = json.load(f).get("records", {})
        if str(n) in recs:
            T = float(recs[str(n)])
        elif len(recs) >= 3:
            ns = np.array(sorted(float(k) for k in recs), dtype=float)
            vs = np.array([float(recs[str(int(k))]) for k in ns], dtype=float)
            A = np.stack([np.sqrt(ns), np.ones_like(ns)], 1)
            c, b = np.linalg.lstsq(A, vs, rcond=None)[0]
            T = float(c * np.sqrt(n) + b)
    except Exception as exc:
        LAST_ERRORS.append("tf_target:%r" % (exc,))
        T = None
    if T is None or not np.isfinite(T) or T <= 0:
        T = 0.53 * np.sqrt(max(1, n))          # crude fallback: n equal circles, hexagonal-ish
    if np.isfinite(cur_s):
        T = max(T, float(cur_s) * (1.0 + 1e-9))
    return float(T)


# --- BASIN HOPPING ON E (iteration 14) ----------------------------------------------------------
# proto_e0.py settled the question iteration 13 left open.  The cold descent lands at E ~ 1e-5 and
# that is a TRUE LOCAL MINIMUM of E, not an unconverged one: 6000 extra L-BFGS iterations move it by
# ZERO, and a Gauss-Newton `least_squares` on the residual VECTOR (the correct tool for a sum of
# squares) moves it by zero as well, on 6 of 6 cold starts across n=35 and n=37.  E ~ 1e-5 harvests
# to relgap ~ 4e-3; only E < 1e-9 harvests to the record.  So the entire distance between a miss and
# a hit is STRUCTURAL, and the KICK -- not the descent -- is the search.
#
# That makes independent multi-start (what iteration 13 shipped) the weakest possible global
# strategy on this landscape: it throws away a converged E ~ 1e-6 structure to draw a fresh E ~ 1e-5
# one.  This hops instead, and `proto_bh2.py` measures the combination at the shipped 6.2 s slice
# over n=35/37/39: mean relgap 1.37e-3 against 1.72e-3 for monotone hopping and 1.9e-3 for the
# shipped multi-start, with 14x fewer harvests.  Two details carry that margin:
#   * THRESHOLD acceptance (TF_BAND).  Monotone-on-E stalled at 1e-6 and burned its slice on
#     restarts (6-9 per slice); the band lets the walk drift uphill and it restarts ~once.
#   * A CHEAP harvest on every new global best -- LP radii + retract, no SLP, ~2 ms against ~50 ms.
#     A structure that lands near but under the record used to be discarded; now it is EMITTED, and
#     it is the structurally-distinct pool parent that eight iterations of warm local search could
#     not manufacture.  The full harvest is reserved for E < TF_BH_EPS and for the deadline.
TF_BAND = 0.05            # accept a kick whose E is up to 5% WORSE: the walk must keep moving
TF_PATIENCE = 14          # non-improving kicks before a cold restart
TF_BH_EPS = 1e-8          # E below this is worth the full ~50 ms harvest immediately
TF_BH_SLICE = 0.35        # share of the REMAINING tf slice one hop run may take (never all of it)
TF_KICKS = ("tele", "jolt", "prof", "swap", "tele", "jolt")


def _tf_kick(z, n, T, rng, kind):
    """A STRUCTURAL perturbation of an infeasible state, in the z coordinates of _tf_eg.

    All four kinds are reachable only in infeasible space -- circles slide THROUGH one another on the
    way -- which is what the feasible-space moves elsewhere in this file cannot do."""
    z = z.copy()
    xy = z[:2 * n].reshape(n, 2)
    a = z[2 * n:]
    e = np.exp(a - a.max())
    r = T * e / e.sum()
    d = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    g = np.maximum((r[:, None] + r[None, :]) - dist, 0.0)
    np.fill_diagonal(g, 0.0)
    wv = np.maximum(r[:, None] - np.stack([xy[:, 0] - LO, HI - xy[:, 0],
                                           xy[:, 1] - LO, HI - xy[:, 1]], 1), 0.0)
    v = g.sum(1) + wv.sum(1)                       # per-circle violation: who is in the way
    if kind == "tele":                             # teleport the m WORST circles somewhere random
        m = int(rng.randint(1, 4))
        xy[np.argsort(v)[::-1][:m]] = (rng.rand(m, 2) - 0.5) * 0.92
    elif kind == "jolt":                           # displace everything: reorganise globally
        xy = xy + rng.randn(n, 2) * (0.45 * r.mean())
    elif kind == "prof":                           # perturb the RADIUS PROFILE, centres untouched
        a = a + rng.randn(n) * 0.25
    else:                                          # exchange two circles' places
        i, j = int(rng.randint(0, n)), int(rng.randint(0, n))
        xy[[i, j]] = xy[[j, i]]
        xy = xy + rng.randn(n, 2) * (0.05 * r.mean())
    return np.concatenate([np.clip(xy, LO + 1e-6, HI - 1e-6).ravel(), a])


def _tf_cheap(n, z):
    """LP radii at these centres + retract, and NO SLP.  ~2 ms: the price of keeping a ticket."""
    xy = np.clip(z[:2 * n].reshape(n, 2), LO + 1e-9, HI - 1e-9)
    r = _lp_radii(xy)
    if r is None:
        return None, -np.inf
    q = _retract(np.ascontiguousarray(xy)[None].copy(), r[None].copy())[0]
    if q[:, 2].min() <= 0.0 or _slack(q) < -SAFE_TOL:
        return None, -np.inf
    return q, float(q[:, 2].sum())


def _tf_bh(n, T, rng, end, emit):
    """Basin-hop on E at fixed target T.  Returns (kicks, restarts, best E reached)."""
    cz, cE = None, np.inf
    gz, gE = None, np.inf
    nk = nres = 0
    fail = 0
    while time.process_time() < end:
        if cz is None:
            cz, cE = _tf_solve(n, T, _tf_cold_z(n, rng, "grid" if nres % 2 else "unif"), TF_ITER_M)
            nres += 1
            fail = 0
        else:
            z2, E2 = _tf_solve(n, T, _tf_kick(cz, n, T, rng, TF_KICKS[nk % len(TF_KICKS)]),
                               TF_ITER_C)
            nk += 1
            if E2 < cE * (1.0 + TF_BAND):
                fail = fail + 1 if E2 >= cE else 0
                cz, cE = z2, E2
            else:
                fail += 1
            if fail >= TF_PATIENCE:
                cz, cE = None, np.inf
                continue
        if cE < gE:
            gz, gE = cz.copy(), cE
            q, s = _tf_cheap(n, gz)                # keep the ticket
            if q is not None:
                emit(q, s)
            if gE < TF_BH_EPS:                     # close enough that the full harvest pays
                q, s = _tf_harvest(n, gz, end)
                if q is not None:
                    emit(q, s)
    if gz is not None:                             # the deadline best always gets a full harvest
        q, s = _tf_harvest(n, gz, time.process_time() + 1.5)
        if q is not None:
            emit(q, s)
    return nk, nres, gE


# --- A POPULATION ON E, WITH Z-SPACE CROSSOVER (iteration 15) -------------------------------------
# Iteration 14 left the front: "_tf_bh is a population of one".  One full descent costs 0.07-0.14
# CPU-s, so a real population is affordable, and it unlocks a move that nothing else in this file
# can make -- RECOMBINATION BETWEEN TWO CONVERGED *INFEASIBLE* STRUCTURES.  _crossover does the same
# cut on the feasible side, but there the two inherited halves must not overlap across the seam, so
# it has to delete circles and regrow them; here circles are allowed to pass through one another and
# the seam is simply handed to the descent.  The z parametrisation gives the child one property for
# free that the feasible-space child has to be repaired into: r = T*softmax(a) renormalises ANY
# inherited radius profile back onto sum r == T exactly, so a child is always a legal candidate
# answer to the same feasibility question.
#
# proto_zx.py, shipped slice 6.2 CPU-s, n=35/37/39 x seeds 11/23, against the shipped hopper on the
# same seeds: mean relgap 1.46e-3 (pop) vs 1.54e-3 (hop), better on 4 of 6 draws, and the best E
# reached was LOWER on 5 of 6 (e.g. n=35 seed 11: 1.6e-6 vs 7.5e-6).  Neither arm hit a record in
# those 6 draws, so this is added as a MODE, not a replacement: the hopper and the multi-start
# lottery (which owns the only two record hits ever observed here) both keep their slots.
TF_POP = 5                # members; 5 fits ~2 generations of turnover into the shipped slice
TF_PCROSS = 0.5           # crossover vs kick per generation -- the two moves hedge each other
TF_POP_SEP = 1e-6         # two members within this RELATIVE E are the same structure: refuse both


def _tf_zcross(za, zb, n, T, rng):
    """Recombine two z states: cut the square with a random line, take A's circles on one side and
    B's on the other.  The radius log-weights ride along with the circles that own them, and the
    softmax renormalises them, so the child satisfies sum r == T with no repair."""
    xa, aa = za[:2 * n].reshape(n, 2), za[2 * n:]
    xb, ab = zb[:2 * n].reshape(n, 2), zb[2 * n:]
    th = rng.uniform(0.0, np.pi)
    dv = np.array([np.cos(th), np.sin(th)])
    sa, sb = xa @ dv, xb @ dv
    t = float(np.sort(np.concatenate([sa, sb]))[int(rng.uniform(0.3, 0.7) * 2 * n)])
    ma, mb = sa <= t, sb > t
    X = np.concatenate([xa[ma], xb[mb]], 0)
    A = np.concatenate([aa[ma], ab[mb]], 0)
    if len(X) > n:                                 # surplus: drop whoever sits deepest in the seam
        w = np.exp(A - A.max())
        r = T * w / w.sum()
        d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        keep = np.argsort(np.maximum(0.0, r[:, None] + r[None] - d).sum(1))[:n]
        X, A = X[keep], A[keep]
    while len(X) < n:                              # deficit: a fresh point at the mean weight
        X = np.concatenate([X, (rng.rand(1, 2) - 0.5) * 0.9], 0)
        A = np.concatenate([A, [A.mean() if len(A) else 0.0]])
    return np.concatenate([np.clip(X, LO + 1e-6, HI - 1e-6).ravel(), A])


def _tf_pop(n, T, rng, end, emit):
    """Evolve TF_POP converged structures on E at fixed T.  Returns (kicks, crossovers, best E)."""
    pop = []                                       # [(E, z)], every member a converged descent
    nk = nx = 0
    gE = np.inf
    while time.process_time() < end:
        if len(pop) < TF_POP:
            z, E = _tf_solve(n, T, _tf_cold_z(n, rng, "grid" if len(pop) % 2 else "unif"),
                             TF_ITER_M)
            pop.append((E, z))
        else:
            if rng.rand() < TF_PCROSS:
                i, j = rng.choice(len(pop), 2, replace=False)
                z0 = _tf_zcross(pop[i][1], pop[j][1], n, T, rng)
                nx += 1
            else:
                i = int(rng.randint(0, len(pop)))
                z0 = _tf_kick(pop[i][1], n, T, rng, TF_KICKS[nk % len(TF_KICKS)])
                nk += 1
            z, E = _tf_solve(n, T, z0, TF_ITER_C)
            w = int(np.argmax([e for e, _ in pop]))
            if E < pop[w][0] and min(abs(E - e) / max(e, 1e-30) for e, _ in pop) > TF_POP_SEP:
                pop[w] = (E, z)                    # elitist, and never two copies of one structure
        b = int(np.argmin([e for e, _ in pop]))
        if pop[b][0] < gE:                         # keep the ticket, exactly as the hopper does
            gE = pop[b][0]
            q, s = _tf_cheap(n, pop[b][1])
            if q is not None:
                emit(q, s)
            if gE < TF_BH_EPS:
                q, s = _tf_harvest(n, pop[b][1], end)
                if q is not None:
                    emit(q, s)
    if pop:
        b = int(np.argmin([e for e, _ in pop]))
        q, s = _tf_harvest(n, pop[b][1], time.process_time() + 1.5)
        if q is not None:
            emit(q, s)
    return nk, nx, gE


def _tf_phase(n, rng, end, emit, cur_s=-np.inf):
    """COLD target-feasibility search.  Returns the number of starts it managed.

    Two cold modes alternate because the measurement gave one record to each and neither dominates:
    a single-shot multi-start at T (cheap, ~0.2 s, many independent draws) and an OVER-PRESSURE
    CONTINUATION that starts at T*(1+TF_KAPPA) -- a target no arrangement can hold, so the whole
    system is under pressure and rearranges -- and then releases T down to the real target."""
    T = _tf_target(n, cur_s)
    # "pop" takes the SECOND hop slot, not a multi-start slot: the two cold single-shot draws and
    # the two continuations are the arms with the only record hits ever observed, and iteration 14
    # already flagged halving them as the first thing to suspect.
    modes = ("unif", "cont", "bh", "grid", "cont", "pop")
    k = 0
    while time.process_time() < end:
        mode = modes[k % len(modes)]
        k += 1
        try:
            if mode in ("bh", "pop"):
                # a hop run is a LONG move; it may never take the whole remaining slice, so the
                # multi-start lottery -- the arm that has actually hit a record -- keeps its draws.
                now = time.process_time()
                sub = min(end, now + max(1.2, TF_BH_SLICE * (end - now)))
                (_tf_bh if mode == "bh" else _tf_pop)(n, T, rng, sub, emit)
                continue
            z = _tf_cold_z(n, rng, "grid" if k % 2 else "unif")
            if mode == "cont":
                bq, bs = None, -np.inf
                for Tl in np.linspace(T * (1.0 + TF_KAPPA), T, TF_STEPS):
                    if time.process_time() > end:
                        break
                    z, _E = _tf_solve(n, Tl, z, TF_ITER_C)
                    q, s = _tf_harvest(n, z, end)
                    if q is not None and s > bs:
                        bq, bs = q, s
                if bq is not None:
                    emit(bq, bs)
            else:
                z = _tf_cold_z(n, rng, mode)
                z, _E = _tf_solve(n, T, z, TF_ITER_M)
                q, s = _tf_harvest(n, z, end)
                if q is not None:
                    emit(q, s)
        except Exception as exc:
            LAST_ERRORS.append("tf_phase:%r" % (exc,))
    return k


# --------------------------------------------------------------------------- batched penalty ascent
def _ascend(xy, r, iters, mu0, mu1, lr0, lr1):
    """Batched Adam ascent on sum(r) - mu*(overlap^2 + wall^2): cheap BASIN GENERATION for cold starts.

    It is a poor refiner (it stalls at ~1e-3 relative error) but a good, vectorized way to turn many
    raw seeds into many distinct near-feasible structures at once, which is exactly what the SLP needs
    handed to it."""
    xy = xy.copy()
    r = r.copy()
    B, n, _ = xy.shape
    di = np.arange(n)
    m_xy = np.zeros_like(xy); v_xy = np.zeros_like(xy)
    m_r = np.zeros_like(r); v_r = np.zeros_like(r)
    b1, b2, eps = 0.9, 0.999, 1e-12
    for t in range(iters):
        frac = t / max(1, iters - 1)
        mu = mu0 * (mu1 / mu0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        gxy = np.zeros_like(xy)
        gr = np.ones_like(r)
        if n >= 2:
            dvec = xy[:, :, None, :] - xy[:, None, :, :]
            d2 = (dvec ** 2).sum(-1)
            d2[:, di, di] = 1.0
            d = np.sqrt(np.maximum(d2, 1e-24))
            v = r[:, :, None] + r[:, None, :] - d
            v[:, di, di] = 0.0
            np.maximum(v, 0.0, out=v)
            c = 2.0 * mu * v
            gr -= c.sum(axis=2)
            gxy += ((c / d)[..., None] * dvec).sum(axis=2)
        x, y = xy[..., 0], xy[..., 1]
        for slack, ax, sgn in ((x - LO, 0, +1.0), (HI - x, 0, -1.0),
                               (y - LO, 1, +1.0), (HI - y, 1, -1.0)):
            u = np.maximum(r - slack, 0.0)
            cu = 2.0 * mu * u
            gr -= cu
            gxy[..., ax] += sgn * cu
        m_xy = b1 * m_xy + (1 - b1) * gxy
        v_xy = b2 * v_xy + (1 - b2) * gxy * gxy
        m_r = b1 * m_r + (1 - b1) * gr
        v_r = b2 * v_r + (1 - b2) * gr * gr
        bc1 = 1 - b1 ** (t + 1)
        bc2 = 1 - b2 ** (t + 1)
        xy += lr * (m_xy / bc1) / (np.sqrt(v_xy / bc2) + eps)
        r += lr * (m_r / bc1) / (np.sqrt(v_r / bc2) + eps)
        np.clip(xy, LO + 1e-6, HI - 1e-6, out=xy)
        np.clip(r, 1e-7, 0.5, out=r)
    return xy, r


# --------------------------------------------------------------------------- seeds
def _grid_seed(n, k, rng):
    """k x k grid of centres plus interstitial fills, trimmed/padded to exactly n centres."""
    g = (np.arange(k) + 0.5) / k - 0.5
    gx, gy = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    if k >= 2:
        h = (np.arange(k - 1) + 1.0) / k - 0.5
        hx, hy = np.meshgrid(h, h, indexing="ij")
        inter = np.stack([hx.ravel(), hy.ravel()], axis=1)
        rng.shuffle(inter)
        pts = np.concatenate([pts, inter], axis=0)
    if len(pts) < n:
        extra = rng.uniform(LO + 0.02, HI - 0.02, size=(n - len(pts), 2))
        pts = np.concatenate([pts, extra], axis=0)
    return pts[:n].copy()


def _make_seeds(n, B, rng, warm):
    """(B,n,2) centres: warm start, structured grids, then uniform random."""
    seeds = []
    if warm is not None:
        seeds.append(warm.copy())
        seeds.append(np.clip(warm + rng.normal(0, 0.006, size=warm.shape), LO + 1e-4, HI - 1e-4))
    root = int(round(np.sqrt(max(1, n))))
    for k in (root - 1, root, root + 1, root + 2):
        if k < 2:
            continue
        for jitter in (0.0, 0.01, 0.025):
            if len(seeds) >= B:
                break
            p = _grid_seed(n, k, rng)
            if jitter:
                p = p + rng.normal(0, jitter, size=p.shape)
            seeds.append(np.clip(p, LO + 1e-4, HI - 1e-4))
    while len(seeds) < B:
        seeds.append(rng.uniform(LO + 0.03, HI - 0.03, size=(n, 2)))
    return np.stack(seeds[:B], axis=0)


# --------------------------------------------------------------------------- structural moves
def _free_field(sub, m=48):
    """Grid scan of the FREE-RADIUS field  f(g) = min( wall(g), min_i ||g - c_i|| - r_i ).

    Returns (P, f) over an m x m grid.  Costs ~1 ms at m=48, n<=64.  A circle of radius f(g) centred
    at g is feasible against `sub` BY CONSTRUCTION, which is what makes insertion (below) exact."""
    g = (np.arange(m) + 0.5) / m - 0.5
    gx, gy = np.meshgrid(g, g, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel()], axis=1)
    f = np.minimum.reduce([P[:, 0] - LO, HI - P[:, 0], P[:, 1] - LO, HI - P[:, 1]])
    if len(sub):
        d = np.sqrt(((P[:, None, :] - sub[None, :, :2]) ** 2).sum(-1)) - sub[None, :, 2]
        f = np.minimum(f, d.min(axis=1))
    return P, f


def _free_point(sub, m=48):
    """The single LARGEST hole.  This is what makes a reseat move DIRECTED -- a uniform-random
    teleport lands in a hole of typical size, this one lands in the biggest."""
    P, f = _free_field(sub, m)
    j = int(np.argmax(f))
    return P[j], float(f[j])


def _free_point_rand(sub, rng, m=32, cand=10):
    """One of the `cand` largest holes, drawn rank-biased.  The greedy argmax is one construction;
    sampling the top few turns insertion into a BASIN GENERATOR that can build a different structure
    every time it is called."""
    P, f = _free_field(sub, m)
    cand = max(1, min(cand, len(f)))
    top = np.argpartition(-f, cand - 1)[:cand]
    top = top[np.argsort(-f[top])]
    w = 1.0 / (1.0 + np.arange(len(top)))
    j = int(top[int(np.searchsorted(np.cumsum(w / w.sum()), rng.rand()))
              if len(top) > 1 else 0])
    return P[j], float(f[j])


# --------------------------------------------------------------------------- n-changing moves
# The census is 10 INDEPENDENT sizes only if you insist on it.  A packing of n and a packing of n+1
# share almost all of their contact graph, so `insert into the largest hole` and `delete the smallest
# circle` are cheap bridges between sizes -- and a bridge is a move no fixed-n perturbation can make.
def _grow(p, rng=None, m=32):
    """(k,3) -> (k+1,3): drop one circle into a large hole.  Feasible by construction: the inserted
    radius is exactly the free radius at that point, so it touches nothing."""
    if rng is None:
        pt, f = _free_point(p, 48)
    else:
        pt, f = _free_point_rand(p, rng, m)
    if not np.isfinite(f) or f <= 1e-12:
        pt, f = np.zeros(2), 1e-9
    c = np.array([[pt[0], pt[1], max(1e-9, f * (1.0 - 1e-9))]])
    return np.concatenate([p, c], axis=0) if len(p) else c


def _shrink(p, rng=None):
    """(k,3) -> (k-1,3): remove the smallest circle (or, sometimes, a random small one).  Always
    feasible -- deleting a circle cannot create a violation -- and it FREES space the SLP can then
    redistribute, which is the whole point."""
    k = len(p)
    if k <= 1:
        return p[:0]
    if rng is not None and rng.rand() < 0.3:
        j = int(np.argsort(p[:, 2])[int(rng.randint(0, min(4, k)))])
    else:
        j = int(np.argmin(p[:, 2]))
    return np.delete(p, j, axis=0)


def _ladder(n, rng, deadline, seed=None, every=3):
    """CONSTRUCTIVE cold start for ANY n: grow a packing one circle at a time, relaxing every `every`
    insertions.  A random-then-Adam seed has to discover a whole structure at once; a ladder inherits
    a relaxed (n-1)-structure and only has to decide where the n-th circle goes, so the structures it
    reaches are qualitatively different from -- and usually better than -- a flat random start."""
    p = np.zeros((0, 3)) if seed is None else np.array(seed, dtype=float, copy=True)
    since = 0
    while len(p) < n:
        p = _grow(p, rng)
        since += 1
        if since >= every and len(p) >= 2 and len(p) < n:
            if deadline is not None and time.process_time() > deadline:
                break
            p, _ = _slp_refine(p, iters=10, patience=2, gtol=1e-10, deadline=deadline)
            since = 0
    while len(p) < n:                       # deadline hit mid-ladder: finish the shape, skip relaxing
        p = _grow(p, rng)
    if len(p) > n:
        p = p[:n]
    return _retract(p[None, :, :2].copy(), p[None, :, 2].copy())[0]


def _bridge(n, src, rng, deadline):
    """Carry a packing of a DIFFERENT size to n: delete down / insert up, relaxing between steps.

    This is how one solved size pays for its neighbours.  n=31 has been exactly at the record since
    iteration 2 while 29 and 33 sat ~2e-3 short; a fixed-n move can never transport that structure,
    a bridge can."""
    p = np.array(src, dtype=float, copy=True)
    guard = 0
    while len(p) != n and guard < 64:
        guard += 1
        p = _shrink(p, rng) if len(p) > n else _grow(p, rng)
        if len(p) < 2:
            continue
        if deadline is not None and time.process_time() > deadline:
            break
        p, _ = _slp_refine(p, iters=12, patience=2, gtol=1e-11, deadline=deadline)
    while len(p) < n:
        p = _grow(p, rng)
    if len(p) != n:
        return None
    return _retract(p[None, :, :2].copy(), p[None, :, 2].copy())[0]


def _srr(n, cur, rng, k, deadline):
    """SHRINK - RELAX - REGROW, the move the bridges make possible at FIXED n.

    _reseat teleports k circles while the other n-k stay pinned at their old optimum, so the holes it
    aims at are the holes of the OLD structure.  Here the k smallest are deleted and the remaining
    n-k are re-converged into the space that frees up -- the whole packing breathes outward -- and
    only then are k circles inserted into the holes of the NEW, relaxed structure.  That is a genuine
    change of contact graph, not a relocation inside a fixed one."""
    k = max(1, min(k, n - 1))
    q = cur.copy()
    for _ in range(k):
        q = _shrink(q, rng)
    if len(q) >= 2:
        q, _ = _slp_refine(q, iters=16, patience=2, gtol=1e-11, deadline=deadline)
    while len(q) < n:
        q = _grow(q, rng)
    return _retract(q[None, :, :2].copy(), q[None, :, 2].copy())[0]


def _reseat(n, cur, rng, k):
    """Pull the k smallest (or k random) circles out and DROP THEM INTO THE LARGEST HOLES, one at a
    time, recomputing the hole map after each placement so they do not all pile into the same gap."""
    q = cur.copy()
    if rng.rand() < 0.75:
        idx = np.argsort(q[:, 2])[:k]
    else:
        idx = rng.choice(n, k, replace=False)
    mask = np.ones(n, bool)
    mask[idx] = False
    sub = q[mask]
    for i in idx:
        pt, f = _free_point(sub)
        q[i, :2] = np.clip(pt + rng.normal(0.0, 0.008, size=2), LO + 1e-4, HI - 1e-4)
        q[i, 2] = max(1e-7, 0.9 * f)
        sub = np.concatenate([sub, q[i:i + 1]], axis=0)
    return _retract(q[None, :, :2].copy(), q[None, :, 2].copy())[0]


# --------------------------------------------------------------------------- symmetry-reduced search
# A NEW REPRESENTATION, not a new move: instead of searching 3n free numbers, search the G-invariant
# subspace of the square's symmetry group.  Most packomania optima carry one of these symmetries, and
# inside the subspace the LP has ~3n/|G| columns, so a hop is both cheaper AND lands in a much smaller
# space that is dense in the structures we are missing.  Every symmetric result is then RELEASED into
# the full space and re-refined, so this can only ever add structures -- it never constrains the answer.
_I2 = np.eye(2)
_MX = np.array([[-1.0, 0.0], [0.0, 1.0]])
_MY = np.array([[1.0, 0.0], [0.0, -1.0]])
_R2 = -_I2
_D1 = np.array([[0.0, 1.0], [1.0, 0.0]])
_D2 = -_D1
_R4 = np.array([[0.0, -1.0], [1.0, 0.0]])
_R4B = _R4.T
GROUPS = {
    "mx": [_I2, _MX],
    "my": [_I2, _MY],
    "d1": [_I2, _D1],
    "d2": [_I2, _D2],
    "r2": [_I2, _R2],
    "mxy": [_I2, _MX, _MY, _R2],
    "dd": [_I2, _D1, _D2, _R2],
    "c4": [_I2, _R4, _R2, _R4B],
    "d4": [_I2, _MX, _MY, _R2, _D1, _D2, _R4, _R4B],
}
GROUP_NAMES = list(GROUPS)


def _sym_map(p, G, axis_tol=0.05):
    """Partition p's circles into G-orbits and return (symmetrised pack, reduced-space map).

    Returns None when no orbit of size > 1 could be formed (the config has nothing to symmetrise).
    Circles that cannot be paired stay FREE SINGLETONS with their own 3 variables, so the reduced
    space always contains the original one -- the representation degrades gracefully, never fails."""
    try:
        from scipy.sparse import coo_matrix
    except Exception:
        return None
    n = len(p)
    c = p[:, :2]
    free = list(range(n))
    rows, cols, vals, kinds, memb = [], [], [], [], []
    xy = np.zeros((n, 2))
    rr = np.zeros(n)
    symmask = np.zeros(n, dtype=bool)
    k = 0
    nsym = 0
    while free:
        i = free[0]
        u0 = c[i]
        H = [g for g in G if np.hypot(*(g @ u0 - u0)) < axis_tol]
        P = np.mean(np.stack(H), axis=0)
        up = P @ u0
        slots = []
        for g in G:
            B = g @ P
            t = B @ up
            if any(np.hypot(*(t - s[1])) < 1e-7 for s in slots):
                continue
            slots.append((B, t))
        if len(slots) > len(free):
            # Not enough circles left for a full orbit.  First try snapping the circle onto the
            # group's own fixed subspace (a mirror line / the centre): that keeps the configuration
            # EXACTLY invariant.  Only if even that does not fit does it become a free singleton --
            # a partially symmetric configuration, which is still a representation worth searching.
            Pf = np.mean(np.stack(G), axis=0)
            uf = Pf @ u0
            s2 = []
            for g in G:
                B = g @ Pf
                t = B @ uf
                if any(np.hypot(*(t - q[1])) < 1e-7 for q in s2):
                    continue
                s2.append((B, t))
            if len(s2) <= len(free):
                slots = s2
            else:
                slots = [(_I2.copy(), c[i])]
        claimed = []
        avail = list(free)
        for B, t in slots:
            dj = [np.hypot(*(c[j] - t)) for j in avail]
            j = avail[int(np.argmin(dj))]
            avail.remove(j)
            claimed.append((B, j))
        A2 = sum(B.T @ B for B, _ in claimed)
        b2 = sum(B.T @ c[j] for B, j in claimed)
        try:
            u = np.linalg.lstsq(A2, b2, rcond=None)[0]
        except Exception:
            u = up
        rv = min(p[j, 2] for _, j in claimed)
        for B, j in claimed:
            xy[j] = B @ u
            rr[j] = rv
            rows += [j, j, n + j, n + j, 2 * n + j]
            cols += [k, k + 1, k, k + 1, k + 2]
            vals += [B[0, 0], B[0, 1], B[1, 0], B[1, 1], 1.0]
            free.remove(j)
        inv = len(claimed) > 1 or not np.allclose(claimed[0][0], _I2)
        if inv:
            nsym += 1
            for _, j in claimed:
                symmask[j] = True
        kinds += [0, 0, 1]
        memb += [claimed[0][1]] * 3
        k += 3
    if nsym == 0:
        return None
    Msp = coo_matrix((vals, (rows, cols)), shape=(3 * n, k)).tocsr()
    kinds = np.asarray(kinds, dtype=int)
    memb = np.asarray(memb, dtype=int)
    cred = -np.asarray(Msp[2 * n:, :].sum(axis=0)).ravel()
    # make the symmetrised centres feasible WITHOUT breaking symmetry: the wall map and the pair
    # graph are both G-invariant, so a per-orbit wall clip and ONE global radius scale stay exact.
    w = _walls(xy)
    rr = np.minimum(np.maximum(rr, 1e-9), np.maximum(w - 1e-12, 1e-9))
    if n >= 2:
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        ssum = rr[:, None] + rr[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.min(np.where(ssum > 0, d / np.maximum(ssum, 1e-300), np.inf))
        rr = rr * min(1.0, float(f)) * (1.0 - 1e-13)
    rr = np.maximum(rr, 1e-12)
    q = np.concatenate([xy, rr[:, None]], axis=1)
    if _slack(q) < -SAFE_TOL:
        return None
    return q, (Msp, kinds, memb, cred), symmask


def _sym_hop(p, rng, deadline, name=None):
    """One symmetry hop: project onto a G-invariant subspace, solve THERE, then release into the
    full space.  The release is the hedge -- a symmetric structure that is only nearly optimal is
    still a fresh basin for the asymmetric refiner, which is exactly what the pool wants."""
    name = name or GROUP_NAMES[int(rng.randint(0, len(GROUP_NAMES)))]
    res = _sym_map(p, GROUPS[name])
    if res is None:
        return None
    q, M, _ = res
    q, _ = _slp_refine(q, iters=70, deadline=deadline, patience=5, gtol=1e-13, M=M)
    if _slack(q) < -SAFE_TOL or q[:, 2].min() <= 0:
        return None
    return q


# ------------------------------------------------- OBJECTIVE-SPACE CONTINUATION (iteration 6)
# Every move up to now left the feasible manifold: teleport a circle, deflate radii, shake centres --
# then the SLP walks back.  A basin-optimal packing is a KKT point of  max sum(r), so NO feasible
# direction improves it and the only way out was to break feasibility and hope to land elsewhere.
#
# There is a second way out that never leaves the manifold: CHANGE THE OBJECTIVE.  A KKT point of
# max sum(r) is generically NOT a KKT point of  max sum(r^p), so re-optimizing with p != 1 slides the
# packing CONTINUOUSLY along the feasible set to a different structure -- contacts open and close on
# the way -- and then continuing p back to 1 lands in a genuinely different basin of the true problem.
# p < 1 rewards the SMALL circles (the wasted ones), pushing the packing toward the well-structured
# equal-circle regime; p > 1 rewards the big ones.  Because the constraints are untouched, every
# intermediate iterate is a strictly feasible packing, so a continuation hop cannot fail feasibly --
# unlike a teleport, which usually lands somewhere the refiner has to dig out of.


def _weights(r, p, eps=1e-9):
    """Ascent weights for  max sum(r_i^p):  grad_i = p * r_i^(p-1).  Normalised to mean 1 so the
    gain/stall thresholds inside _slp_refine keep the same meaning for every p."""
    w = np.maximum(np.asarray(r, dtype=float), eps) ** (p - 1.0)
    m = float(w.mean())
    return w / (m if m > 1e-300 else 1.0)


def _homotopy(cur, rng, deadline, M=None, p0=None, stages=None):
    """Continuation hop: deform to the max-sum(r^p0) optimum, then walk p back to 1.

    The first stage is the deformation and gets the iterations; the later stages only have to track
    a moving optimum, so they are cheap.  Weights are recomputed from the CURRENT radii at every
    stage, which makes each stage a sequential-linearization step on the true concave/convex power
    objective rather than a one-shot reweighting."""
    q = np.array(cur, dtype=float, copy=True)
    if q[:, 2].min() <= 0:
        return None
    p0 = float(rng.uniform(-0.7, 0.6)) if p0 is None else float(p0)
    stages = int(rng.randint(3, 6)) if stages is None else int(stages)
    for t in range(stages):
        if time.process_time() > deadline:
            break
        pe = p0 + (1.0 - p0) * (t / float(max(1, stages - 1)))
        it, pat = (22, 3) if t == 0 else (6, 2)
        q2, _ = _slp_refine(q, iters=it, deadline=deadline, patience=pat, gtol=1e-11,
                            M=M, w=_weights(q[:, 2], pe))
        if q2[:, 2].min() <= 0 or _slack(q2) < -SAFE_TOL:
            return None
        q = q2
    return q


def _wshake(cur, rng, deadline, M=None):
    """Objective SHAKE: one random reweighting, re-optimized.  The undirected member of the
    objective-space family -- _homotopy follows a principled path, this one just picks a random
    direction in weight space, which is the hedge against the path being the wrong one."""
    n = len(cur)
    sig = 10.0 ** rng.uniform(-0.9, -0.1)
    w = np.exp(sig * rng.normal(size=n))
    w /= max(float(w.mean()), 1e-300)
    q, _ = _slp_refine(cur, iters=26, deadline=deadline, patience=3, gtol=1e-11, M=M, w=w)
    if q[:, 2].min() <= 0 or _slack(q) < -SAFE_TOL:
        return None
    return q


def _manifold_hop(cur, rng, deadline):
    """One objective-space hop, optionally performed INSIDE a symmetry-reduced subspace.

    Composing the two representations is free: the reduced SLP already accepts any objective, so a
    continuation run under a group G traces a path through G-invariant structures only -- a far
    smaller, denser path than the full-space one -- and the result is still released into the full
    space by the caller's true-objective refine."""
    M = None
    if rng.rand() < 0.35:
        res = _sym_map(cur, GROUPS[GROUP_NAMES[int(rng.randint(0, len(GROUP_NAMES)))]])
        if res is not None:
            cur, M = res[0], res[1]
    if rng.rand() < 0.6:
        return _homotopy(cur, rng, deadline, M=M)
    return _wshake(cur, rng, deadline, M=M)


# --------------------------------------------------------------------------- CONTACT-GRAPH SPACE
# Iteration 7.  A THIRD mechanism.  The first two were "leave the feasible set and re-converge"
# (teleport/reseat/srr/bridge) and "stay on it but change the objective" (homotopy/wshake).  Both
# act on CONTINUOUS coordinates and only hope the contact graph rewires.  But the incumbents are
# measurably ISOSTATIC KKT points -- artifacts/proto_kkt.py recovers, for n=27/35/45, an active set
# of m = 82/105/135 against 3n = 81/105/135 with EVERY multiplier strictly positive (min 5.7e-2)
# and ||grad(sum r) + J^T lam|| ~ 1e-15.  m ~ 3n with all contacts load-bearing means the packing is
# *determined* by its contact graph: the real search space is the discrete set of graphs, and a
# basin IS a graph.  So the move that was missing is the one that edits the graph directly and then
# solves for the packing that realises it.


def _act_set(p, tol=1e-7, addtol=None):
    """Active contact set of a pack.  Returns (pairs (m1,2), walls (m2,2) as (circle, side)),
    plus the raw pair slacks and the (n,4) wall slack table so a caller can pick near-contacts."""
    n = len(p)
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    iu, ju = np.triu_indices(n, 1) if n >= 2 else (np.empty(0, int), np.empty(0, int))
    if len(iu):
        dx, dy = x[iu] - x[ju], y[iu] - y[ju]
        sl = np.sqrt(dx * dx + dy * dy) - r[iu] - r[ju]
    else:
        sl = np.empty(0)
    k = sl < tol
    pairs = np.stack([iu[k], ju[k]], 1) if len(iu) else np.empty((0, 2), int)
    wb = np.stack([x - r - LO, HI - x - r, y - r - LO, HI - y - r], 1)
    wi, ws = np.nonzero(wb < tol)
    return pairs, np.stack([wi, ws], 1), (iu, ju, sl), wb


def _gJ(p, pairs, walls):
    """Constraint values g (m,) and Jacobian J (m, 3n) for a GIVEN contact graph.

    Pair (i,j):  g = |c_i - c_j| - r_i - r_j.   Wall side s of i:  g = (+-)(coord) - r_i - LO/HI.
    Ordering of the 3n variables is (x, y, r), matching _slp_step."""
    n = len(p)
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    npa, nw = len(pairs), len(walls)
    J = np.zeros((npa + nw, 3 * n))
    g = np.empty(npa + nw)
    if npa:
        i, j = pairs[:, 0], pairs[:, 1]
        dx, dy = x[i] - x[j], y[i] - y[j]
        dd = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-14)
        ux, uy = dx / dd, dy / dd
        g[:npa] = dd - r[i] - r[j]
        k = np.arange(npa)
        np.add.at(J, (k, i), ux);          np.add.at(J, (k, j), -ux)
        np.add.at(J, (k, n + i), uy);      np.add.at(J, (k, n + j), -uy)
        np.add.at(J, (k, 2 * n + i), -1.0)
        np.add.at(J, (k, 2 * n + j), -1.0)
    if nw:
        wi, ws = walls[:, 0], walls[:, 1]
        k = npa + np.arange(nw)
        g[npa:] = np.where(ws == 0, x[wi] - r[wi] - LO,
                  np.where(ws == 1, HI - x[wi] - r[wi],
                  np.where(ws == 2, y[wi] - r[wi] - LO, HI - y[wi] - r[wi])))
        np.add.at(J, (k, np.where(ws < 2, 0, n) + wi), np.where(ws % 2 == 0, 1.0, -1.0))
        np.add.at(J, (k, 2 * n + wi), -1.0)
    return g, J


def _lag_hess(p, pairs, lam):
    """W = sum_k lam_k * Hess(g_k).  Walls are LINEAR, so only pair contacts contribute:
    Hess of |c_i - c_j| is the (I - u u^T)/d block form."""
    n = len(p)
    W = np.zeros((3 * n, 3 * n))
    if not len(pairs):
        return W
    x, y = p[:, 0], p[:, 1]
    i, j = pairs[:, 0], pairs[:, 1]
    dx, dy = x[i] - x[j], y[i] - y[j]
    dd = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-14)
    ux, uy = dx / dd, dy / dd
    lp = np.asarray(lam)[:len(pairs)] / dd
    b00, b01, b11 = lp * (1.0 - ux * ux), lp * (-ux * uy), lp * (1.0 - uy * uy)
    for a, b, sg in ((i, i, 1.0), (j, j, 1.0), (i, j, -1.0), (j, i, -1.0)):
        np.add.at(W, (a, b), sg * b00)
        np.add.at(W, (a, n + b), sg * b01)
        np.add.at(W, (n + a, b), sg * b01)
        np.add.at(W, (n + a, n + b), sg * b11)
    return W


def _duals(p, pairs, walls):
    """Least-squares multipliers of  grad(sum r) + J^T lam = 0.  lam_k is how load-bearing contact
    k is; a small lam_k is a contact the objective would happily give up, which is exactly the
    contact a structural edit should drop."""
    n = len(p)
    gf = np.zeros(3 * n)
    gf[2 * n:] = 1.0
    J = _gJ(p, pairs, walls)[1]
    if J.shape[0] == 0:
        return np.empty(0), float(np.linalg.norm(gf))
    lam = np.linalg.lstsq(J.T, -gf, rcond=None)[0]
    return lam, float(np.linalg.norm(gf + J.T @ lam))


def _clean(z, n):
    np.clip(z[:2 * n], LO, HI, out=z[:2 * n])
    z[2 * n:] = np.maximum(z[2 * n:], 1e-7)
    return z


def _gn_graph(p, pairs, walls, steps=14, cap=0.06, deadline=None):
    """REALISE a contact graph: damped Gauss-Newton on g(z) = 0, backtracking on ||g||.

    No multipliers and no Hessian -- with m ~ 3n the system is nearly square, so "make exactly
    these contacts hold" is a well-posed geometric question, and the SLP then optimises inside
    whatever basin that lands in."""
    n = len(p)
    z = np.concatenate([p[:, 0], p[:, 1], p[:, 2]])

    def gof(zz):
        return _gJ(np.stack([zz[:n], zz[n:2 * n], zz[2 * n:]], 1), pairs, walls)

    g, J = gof(z)
    nb = float(np.linalg.norm(g))
    for _ in range(steps):
        if nb < 1e-13 or (deadline is not None and time.process_time() > deadline):
            break
        try:
            d = np.linalg.lstsq(J, -g, rcond=None)[0]
        except Exception:
            break
        sc = min(1.0, cap / max(float(np.abs(d).max()), 1e-300))
        for _ls in range(6):
            zt = _clean(z + sc * d, n)
            gt, Jt = gof(zt)
            nt = float(np.linalg.norm(gt))
            if nt < nb:
                z, g, J, nb = zt, gt, Jt, nt
                break
            sc *= 0.35
        else:
            break
    return np.stack([z[:n], z[n:2 * n], z[2 * n:]], 1), nb


def _kkt_newton(p, pairs, walls, lam=None, steps=14, cap=0.06, deadline=None):
    """Solve the full KKT system of the EDITED graph: F(z, lam) = (grad f + J^T lam, g) = 0.

    Damped Newton with the exact Lagrangian Hessian and backtracking on ||F||.  Unlike _gn_graph
    this targets the graph's *optimum*, not merely a point realising it, so it lands on the real
    KKT point of the new structure when the edit is consistent."""
    n = len(p)
    m = len(pairs) + len(walls)
    gf = np.zeros(3 * n)
    gf[2 * n:] = 1.0
    z = np.concatenate([p[:, 0], p[:, 1], p[:, 2]])
    lam = np.zeros(m) if lam is None or len(lam) != m else np.asarray(lam, float).copy()
    dgi = np.arange(3 * n + m)
    K = np.zeros((3 * n + m, 3 * n + m))

    def Fof(zz, ll):
        pp = np.stack([zz[:n], zz[n:2 * n], zz[2 * n:]], 1)
        g, J = _gJ(pp, pairs, walls)
        return np.concatenate([gf + J.T @ ll, g]), J, pp

    F, J, pp = Fof(z, lam)
    nb = float(np.linalg.norm(F))
    for _ in range(steps):
        if nb < 1e-13 or (deadline is not None and time.process_time() > deadline):
            break
        K[:] = 0.0
        K[:3 * n, :3 * n] = _lag_hess(pp, pairs, lam)
        K[:3 * n, 3 * n:] = J.T
        K[3 * n:, :3 * n] = J
        K[dgi, dgi] += 1e-10
        try:
            d = np.linalg.solve(K, -F)
        except Exception:
            try:
                d = np.linalg.lstsq(K, -F, rcond=None)[0]
            except Exception:
                break
        sc = min(1.0, cap / max(float(np.abs(d[:3 * n]).max()), 1e-300))
        for _ls in range(6):
            zt = _clean(z + sc * d[:3 * n], n)
            lt = lam + sc * d[3 * n:]
            Ft, Jt, ppt = Fof(zt, lt)
            nt = float(np.linalg.norm(Ft))
            if nt < nb:
                z, lam, F, J, pp, nb = zt, lt, Ft, Jt, ppt, nt
                break
            sc *= 0.35
        else:
            break
    return np.stack([z[:n], z[n:2 * n], z[2 * n:]], 1), lam, nb


def _graph_edit(p, rng, drop=1, add=1, addtol=0.15):
    """One random edit of the contact graph: DROP `drop` active contacts (biased toward the least
    load-bearing, by dual) and ADD `add` near-contacts (biased toward the closest non-contacts --
    those are one nudge from forming).  Returns (pairs, walls, lam0) for the edited graph."""
    n = len(p)
    pairs, walls, (iu, ju, sl), wb = _act_set(p)
    m0 = len(pairs) + len(walls)
    if m0 == 0:
        return pairs, walls, np.empty(0)
    lam, _ = _duals(p, pairs, walls)
    keep = np.ones(m0, bool)
    d = min(int(drop), max(0, m0 - 2))
    if d > 0:
        w = 1.0 / np.maximum(lam, 1e-3)
        s = float(w.sum())
        w = w / s if s > 0 else np.full(m0, 1.0 / m0)
        keep[rng.choice(m0, size=d, replace=False, p=w)] = False
    kp, kw = keep[:len(pairs)], keep[len(pairs):]
    P = [tuple(int(v) for v in e) for e in pairs[kp]]
    Wl = [tuple(int(v) for v in e) for e in walls[kw]]
    L = list(lam[keep])
    lm = max(float(np.mean(lam)), 1e-3)
    cp = np.nonzero((sl >= 1e-7) & (sl < addtol))[0]
    cwi, cws = np.nonzero((wb >= 1e-7) & (wb < addtol))
    for _ in range(int(add)):
        if len(cp) and (len(cwi) == 0 or rng.rand() < 0.75):
            pr = 1.0 / (sl[cp] + 1e-3)
            pr /= pr.sum()
            q = cp[rng.choice(len(cp), p=pr)]
            e = (int(iu[q]), int(ju[q]))
            if e not in P:
                P.append(e)
                L.append(lm)
        elif len(cwi):
            q = int(rng.randint(0, len(cwi)))
            e = (int(cwi[q]), int(cws[q]))
            if e not in Wl:
                Wl.append(e)
                L.append(lm)
    return (np.array(P, int).reshape(-1, 2), np.array(Wl, int).reshape(-1, 2),
            np.array(L, float))


# --------------------------------------------------------------------------- systematic graph scan
# Iteration 9.  The measurement in artifacts/proto_thru.py: on the four unsolved n the hop loop
# screens only ~15 BASINS PER CPU-SECOND, and roughly half of that second goes into GENERATING the
# move, not into judging it -- _manifold_hop costs 56-98 ms/draw and _sym_hop 41-48 ms, while a
# whole graph hop costs 9-13 ms.  So the loop spends most of its budget on its most expensive moves
# and draws the single-swap neighbourhood of the incumbent's contact graph ~18 times per iteration
# out of the ~m*k ~ 1000 swaps it contains: a 2% random sample of a neighbourhood that is cheap
# enough to enumerate.  Sampling was the bottleneck, not reachability.
#
# _graph_scan turns that random draw into a real LOCAL SEARCH in graph space (the 2-opt of this
# problem): enumerate the single-swap neighbourhood ordered by the KKT duals, realise each swap
# with a short Newton, and keep the best.  It is RESUMABLE -- every swap it has already judged is
# remembered per structure, so successive calls advance the frontier instead of re-drawing the same
# prefix, and a structure whose neighbourhood is exhausted says so and steps aside.
def _swap_candidates(p, lam, pairs, walls, sl_tab, wb, kmax=14, addtol=0.30):
    """The single-swap neighbourhood of a contact graph, ORDERED by expected cheapness.

    A swap is (drop one active contact, add one near-contact).  With m ~ 3n the graph is isostatic
    (iteration 7), so drop-one/add-one is the edit that keeps the system square.  Order by
    lam_drop * slack_add: a contact with a small dual is one the objective would happily give up,
    and a non-contact with a small slack is one nudge from forming, so the product ranks the swaps
    the geometry is least unwilling to make."""
    iu, ju, sl = sl_tab
    m = len(pairs) + len(walls)
    if m < 3:
        return []
    adds = []
    cp = np.nonzero((sl >= 1e-7) & (sl < addtol))[0]
    if len(cp):
        for q in cp[np.argsort(sl[cp])][:kmax]:
            adds.append((0, int(iu[q]), int(ju[q]), float(sl[q])))
    cwi, cws = np.nonzero((wb >= 1e-7) & (wb < addtol))
    if len(cwi):
        wv = wb[cwi, cws]
        for q in np.argsort(wv)[:max(2, kmax // 3)]:
            adds.append((1, int(cwi[q]), int(cws[q]), float(wv[q])))
    if not adds:
        return []
    have = set(tuple(int(v) for v in e) for e in pairs)
    havew = set(tuple(int(v) for v in e) for e in walls)
    out = []
    for k in range(m):
        lk = float(lam[k]) if k < len(lam) else 1.0
        for a in adds:
            if a[0] == 0 and (a[1], a[2]) in have:
                continue
            if a[0] == 1 and (a[1], a[2]) in havew:
                continue
            out.append((abs(lk) * (a[3] + 1e-4), k, a))
    out.sort(key=lambda t: t[0])
    return out


def _apply_swap(pairs, walls, lam, k, a):
    """Build the edited (pairs, walls, lam0) for one swap: drop constraint k, add candidate a."""
    npa = len(pairs)
    P = [tuple(int(v) for v in e) for e in pairs]
    Wl = [tuple(int(v) for v in e) for e in walls]
    L = list(lam) if len(lam) == npa + len(walls) else [1.0] * (npa + len(walls))
    if k < npa:
        P.pop(k)
    else:
        Wl.pop(k - npa)
    L.pop(k)
    lm = max(float(np.mean(lam)) if len(lam) else 1.0, 1e-3)
    if a[0] == 0:
        P.append((a[1], a[2]))
    else:
        Wl.append((a[1], a[2]))
    L.append(lm)
    return (np.array(P, int).reshape(-1, 2), np.array(Wl, int).reshape(-1, 2),
            np.array(L, float))


# The slice is short because the scan is RESUMABLE: total neighbourhood coverage is proportional
# to total scan time, not to how long any one call runs, so a short slice costs nothing but keeps
# the scan from monopolising a hop loop whose other draws cost ~30 ms.
SCAN_SLICE = 0.15
SCAN_MAX = 14


def _graph_scan(cur, rng, deadline, seen=None, kmax=14):
    """Systematic local search over the single-swap neighbourhood of cur's contact graph.

    Returns the BEST feasible neighbour it judged in its slice (steepest ascent over the part of
    the neighbourhood it could afford), or None when the
    neighbourhood is exhausted, is empty, or yielded nothing usable in this slice -- in which case
    the caller falls through to an undirected move, so an exhausted structure never silently eats
    its share of the draws.

    MEASURED COST (self-test block 27, on pinned BLAS): ~424 swaps per CPU-second, so the full
    1134-swap neighbourhood of an n=21 pack costs ~2.7 s -- genuinely enumerable inside one n's
    share of the budget, which is exactly why this is worth doing instead of sampling 2% of it.
    (The same block measured 30 ms/swap before BLAS was pinned: that was thread thrash, not cost.)"""
    n = len(cur)
    if n < 3:
        return None
    pairs, walls, sl_tab, wb = _act_set(cur)
    if len(pairs) + len(walls) < 3:
        return None
    lam, _ = _duals(cur, pairs, walls)
    cand = _swap_candidates(cur, lam, pairs, walls, sl_tab, wb, kmax=kmax)
    if not cand:
        return None
    s0 = float(cur[:, 2].sum())
    key0 = _fp_safe(cur)
    tried = seen.setdefault(key0, set()) if seen is not None else set()
    # identity-keyed, so the frontier survives relabelling-free re-visits of the same structure
    def ident(k, a):
        npa = len(pairs)
        d = ('p',) + tuple(int(v) for v in pairs[k]) if k < npa else ('w',) + tuple(
            int(v) for v in walls[k - npa])
        return (d, a[0], a[1], a[2])

    end = min(deadline, time.process_time() + SCAN_SLICE)
    best, bs = None, -np.inf
    judged = 0
    for _score, k, a in cand:
        if judged >= SCAN_MAX or time.process_time() > end:
            break
        idk = ident(k, a)
        if idk in tried:
            continue
        tried.add(idk)
        P, Wl, L = _apply_swap(pairs, walls, lam, k, a)
        if len(P) + len(Wl) == 0:
            continue
        q, _lam2, _nb = _kkt_newton(cur, P, Wl, L, steps=7, deadline=end)
        judged += 1
        if not np.isfinite(q).all():
            continue
        z = _repair(q, ref=s0, deadline=end)
        s = float(z[:, 2].sum())
        if s > bs and z[:, 2].min() > 0.0:
            best, bs = z, s
    return best


# --------------------------------------------------------------------------- graph-space BEAM
# Iteration 10.  Three measurements this iteration (artifacts/proto_lns.py, artifacts/proto_ruin.py)
# agree on one thing: from the incumbent of n=35/37/39, EIGHT CPU-seconds of the whole existing mix,
# of large-neighbourhood destroy-and-repair, of ruin-and-recreate with a drifting anchor, and of the
# cold-start generator all accept NOTHING -- 0 of 63-91 ruin draws land within 9e-3 of the anchor,
# and a from-scratch search finishes 2e-3 to 1e-2 BELOW the incumbent.  So the missing structure is
# not far away and it is not reachable by a bigger or wilder move: it is a few contact edits away,
# behind an edit whose FIRST step is not an improvement.
#
# Every graph move so far is a ONE-STEP hill-climb: _graph_hop samples one edit, _graph_scan
# enumerates the single-swap neighbourhood and returns its steepest ascent.  Both judge a swap by
# whether the packing it realises is better, so a swap that must be paid for before it pays out is
# discarded -- and a 2-swap whose halves are each downhill is unreachable by any sequence of them.
# The beam keeps the LOSERS: it expands the best few children of each level regardless of whether
# they beat the root, so depth-2 structures behind a downhill first step come into reach.  Non-
# improving intermediates are the whole mechanism; the width/depth/per knobs are the new dimension.
def _swap_level(node, per, end, tried, kmax=14):
    """Realise up to `per` not-yet-judged single swaps of `node`'s contact graph.

    Returns [(sum_r, pack)] for the strictly feasible packings they retract to.  Factored out of
    _graph_scan so the beam and the scan judge a swap the SAME way -- same ordering, same 7-step
    KKT Newton, same retraction -- and differ only in what they do with the result."""
    if len(node) < 3:
        return []
    pairs, walls, sl_tab, wb = _act_set(node)
    if len(pairs) + len(walls) < 3:
        return []
    lam, _ = _duals(node, pairs, walls)
    cand = _swap_candidates(node, lam, pairs, walls, sl_tab, wb, kmax=kmax)
    if not cand:
        return []
    npa = len(pairs)
    s0 = float(node[:, 2].sum())
    out, judged = [], 0
    for _sc, k, a in cand:
        if judged >= per or time.process_time() > end:
            break
        d = (('p',) + tuple(int(v) for v in pairs[k]) if k < npa
             else ('w',) + tuple(int(v) for v in walls[k - npa]))
        idk = (d, a[0], a[1], a[2])
        if tried is not None:
            if idk in tried:
                continue
            tried.add(idk)
        P, Wl, L = _apply_swap(pairs, walls, lam, k, a)
        if len(P) + len(Wl) == 0:
            continue
        q, _l2, _nb = _kkt_newton(node, P, Wl, L, steps=7, deadline=end)
        if not np.isfinite(q).all():
            judged += 1
            continue
        z = _repair(q, ref=s0, deadline=end)
        judged += 1
        if z[:, 2].min() > 0.0:
            out.append((float(z[:, 2].sum()), z))
    return out


BEAM_SLICE = 0.30         # one call's CPU slice; the root frontier is resumable across calls
BEAM_DEPTH = 2
BEAM_WIDTH = 3
BEAM_PER = 7


def _graph_beam(cur, rng, deadline, seen=None, depth=BEAM_DEPTH, width=BEAM_WIDTH, per=BEAM_PER):
    """BEAM SEARCH over contact-graph edits, depth `depth`, keeping NON-IMPROVING intermediates.

    Level 0 expands the root's single-swap neighbourhood (resumably: the root's judged swaps are
    remembered in `seen`, so successive calls advance the frontier instead of re-judging a prefix).
    Every later level expands the best `width-1` children of the previous one PLUS one random
    other child -- none of which has to beat the root.  Returns the best strictly feasible packing
    seen at ANY level, or None when the root's neighbourhood is exhausted/empty, in which case the
    caller falls through to an undirected move exactly as with _graph_scan."""
    if len(cur) < 3:
        return None
    end = min(deadline, time.process_time() + BEAM_SLICE)
    tried = seen.setdefault(_fp_safe(cur), set()) if seen is not None else set()
    frontier = [cur]
    best, bs = None, -np.inf
    for d in range(max(1, int(depth))):
        kids = []
        for node in frontier:
            if time.process_time() > end:
                break
            # only the ROOT's frontier persists between calls; deeper nodes are call-local
            kids.extend(_swap_level(node, per, end, tried if d == 0 else set()))
        if not kids:
            break
        for s, z in kids:
            if s > bs:
                best, bs = z, s
        kids.sort(key=lambda t: -t[0])
        keep = max(1, int(width) - 1)
        frontier = [z for _s, z in kids[:keep]]
        if len(kids) > keep and rng.rand() < 0.8:      # one deliberate long shot per level
            frontier.append(kids[int(rng.randint(keep, len(kids)))][1])
        if time.process_time() > end:
            break
    return best


# --------------------------------------------------------------------------- graph-space TABU CHAIN
# Iteration 11.  The beam bought DEPTH but paid for it with BRANCHING: depth d width w costs w^d
# realisations, so depth 2 x width 3 is already 21 KKT solves for a horizon of two edits, and
# iteration 10's own caveat was that this samples a vanishing fraction of the 2-swap neighbourhood
# and that "depth may need to mean depth 3-4".  Branching is the wrong way to buy depth here: with
# five mechanisms measured at ZERO yield on these converged basins, what is missing is not a better
# survey of the nearby graphs but a search that can be MANY edits from the incumbent at all.
#
# So: no branching.  One node, one step at a time, ALWAYS move -- to the best non-tabu neighbour
# even when it is worse than where we stand -- and forbid undoing the last CHAIN_TABU edits so the
# walk cannot oscillate back into the basin it just left.  That is classic tabu search, and its cost
# is LINEAR in depth (CHAIN_PER realisations per edit), so the same CPU that bought the beam two
# edits of horizon buys the chain hundreds.  The chain is PERSISTENT: its node, its tabu list and
# its trace live in the caller's `state` dict, so successive draws continue ONE walk instead of
# restarting a short one, and the depth it reaches is proportional to its total share of the budget.
# The walk's node is the raw retracted KKT pack, never the SLP-refined one: refining would pull the
# node straight back uphill into the incumbent's basin, which is exactly what has to be escaped.
CHAIN_SLICE = 1.2         # one call's CPU slice -- a chain is a LONG move, not a single edit
CHAIN_PER = 5             # realisations per step: the whole cost of one edit of depth
CHAIN_KMAX = 10
CHAIN_TABU = 14           # edits whose reversal is forbidden (2 identities pushed per step)
CHAIN_PATIENCE = 45       # accepted steps with no chain-best gain before re-intensifying on `cur`
# ACCEPTANCE BAND, measured (artifacts/proto_chain.py).  The first version of the chain accepted the
# best child unconditionally, and on the converged n=35/37/39 incumbents its nodes had a MEDIAN sum_r
# deficit of 1.31-1.38 against an incumbent of ~3.07 -- the walk was not exploring graph space, it
# was collapsing out of it: one bad realisation shrinks the pack, and every step after that wanders
# in rubble, so 500 steps of depth bought nothing.  "Always move" has to mean "always move, but stay
# on the manifold": a step is accepted only if it lands within CHAIN_BAND (relative) of the chain's
# own best, and a step that does not is a dead end that re-intensifies on the incumbent.  The band
# is what makes downhill acceptance a SEARCH rather than a leak -- the same measurement says good
# children exist (the best raw node sat 2.3e-09 from the incumbent), they were just being outvoted.
CHAIN_BAND = 0.02


def _chain_step(node, tabu, per, end, kmax=CHAIN_KMAX):
    """One tabu step: realise up to `per` non-tabu single swaps of `node`, return the best CHILD
    (even when it is strictly WORSE than `node`) as (pack, sum_r, (dropped_id, added_id)), or None
    when every candidate is tabu / nothing realises."""
    if len(node) < 3:
        return None
    pairs, walls, sl_tab, wb = _act_set(node)
    npa = len(pairs)
    if npa + len(walls) < 3:
        return None
    lam, _ = _duals(node, pairs, walls)
    cand = _swap_candidates(node, lam, pairs, walls, sl_tab, wb, kmax=kmax)
    if not cand:
        return None
    s0 = float(node[:, 2].sum())
    bz, bs, bedit = None, -np.inf, None
    judged = 0
    for _sc, k, a in cand:
        if judged >= per or time.process_time() > end:
            break
        dk = (('p',) + tuple(int(v) for v in pairs[k]) if k < npa
              else ('w',) + tuple(int(v) for v in walls[k - npa]))
        ak = ('p', int(a[1]), int(a[2])) if a[0] == 0 else ('w', int(a[1]), int(a[2]))
        if dk in tabu or ak in tabu:          # forbidden: this would undo a recent edit
            continue
        P, Wl, L = _apply_swap(pairs, walls, lam, k, a)
        if len(P) + len(Wl) == 0:
            continue
        q, _l2, _nb = _kkt_newton(node, P, Wl, L, steps=7, deadline=end)
        judged += 1
        if not np.isfinite(q).all():
            continue
        z = _repair(q, ref=s0, deadline=end)
        if z[:, 2].min() <= 0.0:
            continue
        s = float(z[:, 2].sum())
        if s > bs:
            bz, bs, bedit = z, s, (dk, ak)
    if bz is None:
        return None
    return bz, bs, bedit


def _graph_chain(cur, rng, deadline, state, slice_s=None):
    """Walk the contact-graph space by tabu search, continuing the walk held in `state`.

    Returns the best strictly feasible pack visited in THIS call (which the caller retracts, SLP-
    refines and offers to the pool exactly as for the scan and the beam), or None if the walk could
    not take a single step.  `state` is per-n and persists across calls: {"node", "tabu", "since",
    "bs", "steps", "trace"}."""
    if len(cur) < 3:
        return None
    end = min(deadline, time.process_time() + (CHAIN_SLICE if slice_s is None else float(slice_s)))
    if state.get("node") is None or len(state["node"]) != len(cur):
        state["node"] = cur.copy()
        state["tabu"] = []
        state["since"] = 0
        state["bs"] = float(cur[:, 2].sum())
        state["steps"] = 0
        state["trace"] = []
        state["sd"] = set()               # contact identities toggled since the walk's origin ...
        state["dmax"] = 0                 # ... and the deepest the walk has ever been from it
    best, bsr = None, -np.inf
    fails = 0
    while time.process_time() < end and fails < 3:
        res = _chain_step(state["node"], set(state["tabu"]), CHAIN_PER, end)
        if res is None:                       # dead end -- re-intensify on the caller's incumbent
            fails += 1
            state["node"] = cur.copy()
            state["tabu"] = []
            state["sd"] = set()
            continue
        z, s, (dk, ak) = res
        if s < state["bs"] * (1.0 - CHAIN_BAND):   # off the manifold: do not follow it down
            fails += 1
            state["node"] = cur.copy()
            state["tabu"] = []
            state["sd"] = set()
            continue
        state["node"] = z                     # ALWAYS move, uphill or not: that is the mechanism
        state["steps"] += 1
        sd = state["sd"]                      # edit-distance bookkeeping: toggling is exact here
        for idk in (dk, ak):                  # because a swap drops one identity and adds one
            sd.discard(idk) if idk in sd else sd.add(idk)
        if len(sd) > state["dmax"]:
            state["dmax"] = len(sd)
        tb = state["tabu"]
        tb.append(dk)                         # do not re-add the contact we just dropped ...
        tb.append(ak)                         # ... and do not drop the one we just added
        if len(tb) > 2 * CHAIN_TABU:
            del tb[:len(tb) - 2 * CHAIN_TABU]
        tr = state["trace"]
        tr.append(s)
        if len(tr) > 256:
            del tr[:len(tr) - 256]
        if s > bsr:
            best, bsr = z, s
        if s > state["bs"] + 1e-12:
            state["bs"] = s
            state["since"] = 0
        else:
            state["since"] += 1
            if state["since"] >= CHAIN_PATIENCE:
                state["node"] = cur.copy()    # keep the tabu list: do not retrace the same prefix
                state["since"] = 0
                state["sd"] = set()
    return best


def _graph_hop(cur, rng, deadline):
    """STRUCTURE-SPACE hop: edit the contact graph, solve for the packing that realises it, hand
    the result back for retraction + the exact SLP.

    Both solvers stay in the mix on purpose: the full KKT Newton aims at the new graph's optimum,
    the Gauss-Newton only realises the graph and leaves the optimising to the SLP.  Directed and
    undirected members of the same family, exactly like _homotopy/_wshake."""
    n = len(cur)
    if n < 3:
        return None
    P, Wl, L = _graph_edit(cur, rng, drop=int(rng.randint(1, 4)), add=int(rng.randint(1, 4)),
                          addtol=10.0 ** rng.uniform(-1.4, -0.6))
    if len(P) + len(Wl) == 0:
        return None
    if rng.rand() < 0.5:
        q, _, _ = _kkt_newton(cur, P, Wl, L, deadline=deadline)
    else:
        q, _ = _gn_graph(cur, P, Wl, deadline=deadline)
    if not np.isfinite(q).all():
        return None
    return _repair(q, ref=float(cur[:, 2].sum()), deadline=deadline)


# --------------------------------------------------------------------------- recombination
# Iteration 8.  Enumerated by MECHANISM (iteration 6's lesson), every move the solver owned was
# UNARY: one packing in, one packing out -- teleport/reseat/srr/bridge leave the feasible set and
# re-converge, homotopy/wshake stay on it and bend the objective, graph_hop rewires the contact
# graph.  All three take a single incumbent and deform it, so the only structures they can reach are
# the ones connected to THAT structure by their own deformation.  A binary operator is a different
# class of reachability: a child can carry a sub-structure from each of two parents at once, and the
# intermediate states on the way there are local to neither parent.
def _fingerprint(p, tol=1e-7):
    """A CANONICAL, relabelling- and symmetry-invariant signature of a packing's CONTACT GRAPH.

    POOL_SEP used sum_r as a basin proxy: two packs agreeing to 1e-7 were declared the same basin and
    one was thrown away.  The isostatic diagnosis of iteration 7 (m ~ 3n, all lambda > 0) says the
    graph IS the state, so compare graphs instead -- the sorted multiset of per-circle
    (pair-degree, wall-degree) plus the two edge counts.  Invariant under relabelling AND under every
    symmetry of the square, which is exactly the equivalence we want to quotient by."""
    n = len(p)
    pairs, walls, _, _ = _act_set(p, tol=tol)
    dp = np.zeros(n, int)
    dw = np.zeros(n, int)
    if len(pairs):
        np.add.at(dp, pairs[:, 0], 1)
        np.add.at(dp, pairs[:, 1], 1)
    if len(walls):
        np.add.at(dw, walls[:, 0], 1)
    return (int(len(pairs)), int(len(walls)), tuple(sorted(zip(dp.tolist(), dw.tolist()))))


def _fp_safe(p):
    try:
        return _fingerprint(p)
    except Exception:
        return None


def _deflate(p, sweeps=3):
    """Shrink radii (never centres) until the pack is feasible, touching ONLY what actually conflicts.

    _retract repairs a batch by dividing EVERY radius by the worst overlap ratio -- right for a
    randomly seeded pack, destructive for a recombined one, where a single bad seam would crush two
    perfectly good half-structures.  Here each circle takes its own factor
    f_i = min(1, min_j d_ij/(r_i+r_j)), so a circle in no overlap keeps its radius EXACTLY, and one
    sweep already clears every pair: r_i f_i + r_j f_j <= (r_i+r_j) * d_ij/(r_i+r_j) = d_ij."""
    xy = p[:, :2]
    r = np.clip(p[:, 2], 1e-12, None).astype(float).copy()
    r = np.minimum(r, np.maximum(_walls(xy[None])[0], 1e-12))   # exact wall contacts stay exact
    if len(p) >= 2:
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        for _ in range(max(1, int(sweeps))):
            s = r[:, None] + r[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(s > 0, d / np.maximum(s, 1e-300), np.inf)
            f = np.minimum(1.0, ratio.min(axis=1))
            act = f < 1.0 - 1e-12                # an exact contact reads as ratio 1-1e-16: leave it
            if not act.any():
                break
            r = np.where(act, np.maximum(r * f * (1.0 - 1e-13), 1e-12), r)
    return np.concatenate([xy, r[:, None]], axis=1)


def _crossover(pa, pb, rng, deadline=None):
    """RECOMBINATION: the solver's first BINARY move.  Cut the square with a random line, keep `pa`'s
    circles on one side and `pb`'s on the other, repair the seam and the circle count, hand the
    hybrid to the exact SLP.

    The repair is what makes it a structural move rather than noise: the seam is deflated locally
    (so both inherited halves survive intact), surplus circles are dropped by how much they conflict
    across the cut, and any deficit is filled by inserting into the largest holes of the child -- the
    holes of a structure that never existed in either parent."""
    n = len(pa)
    if pb is None or len(pb) != n or n < 4:
        return None
    th = rng.uniform(0.0, np.pi)
    dvec = np.array([np.cos(th), np.sin(th)])
    sa, sb = pa[:, :2] @ dvec, pb[:, :2] @ dvec
    srt = np.sort(np.concatenate([sa, sb]))
    t = float(srt[int(rng.uniform(0.25, 0.75) * len(srt))])
    A, B = pa[sa <= t], pb[sb > t]
    if len(A) == 0 or len(B) == 0:
        return None
    q = np.concatenate([A, B], axis=0)
    guard = 0
    while len(q) > n and guard < 4 * n + 8:
        guard += 1
        dd = np.sqrt(((q[:, None, :2] - q[None, :, :2]) ** 2).sum(-1))
        np.fill_diagonal(dd, np.inf)
        ov = np.maximum(0.0, q[:, 2][:, None] + q[:, 2][None, :] - dd).sum(axis=1)
        ov = ov + np.maximum(0.0, q[:, 2] - _walls(q[None, :, :2])[0])
        j = int(np.argmax(ov)) if float(ov.max()) > 1e-12 else int(np.argmin(q[:, 2]))
        q = np.delete(q, j, axis=0)
    if len(q) > n:
        q = q[np.argsort(-q[:, 2])[:n]]
    q = _deflate(q)
    while len(q) < n:
        if deadline is not None and time.process_time() > deadline and len(q) >= 1:
            q = np.concatenate([q, np.array([[0.0, 0.0, 1e-9]])], axis=0)
            continue
        q = _grow(q, rng)
    return _retract(q[None, :, :2].copy(), q[None, :, 2].copy())[0]


def _perturb(n, cur, rng, deadline=None):
    """A STRUCTURAL move: change which circles touch which, then let the SLP re-converge exactly.

    Small centre jiggles are useless once the incumbent is basin-optimal -- the SLP just walks back
    to the same KKT point -- so every move here either teleports circles or deflates radii enough to
    let the contact graph rewire.  Three families stay alive at once, deliberately: SHRINK-RELAX-
    REGROW (iteration 4, the only one that re-converges at a smaller n before regrowing), the DIRECTED
    reseat, and the undirected teleports/shakes.  Directed is greedier but blinder; undirected is the
    hedge.  None of them is allowed to become the only one."""
    k = min(n - 1 if n > 1 else 1, 1 + int(rng.randint(0, 3)))
    mode = rng.rand()
    if mode < 0.30 and n >= 3:
        return _srr(n, cur, rng, k, deadline)
    if mode < 0.60 and n >= 2:
        return _reseat(n, cur, rng, k)
    q = cur.copy()
    if mode < 0.65:                                   # the smallest circles are the wasted ones
        idx = np.argsort(q[:, 2])[:k]
    elif mode < 0.80:
        idx = rng.choice(n, k, replace=False)
    else:
        idx = None
    if idx is not None:
        q[idx, :2] = rng.uniform(LO + 0.01, HI - 0.01, size=(k, 2))
        q[idx, 2] = 1e-7
    else:                                             # global shake + deflate
        q[:, :2] += rng.normal(0.0, 10.0 ** rng.uniform(-2.4, -1.3), size=(n, 2))
        np.clip(q[:, :2], LO + 1e-7, HI - 1e-7, out=q[:, :2])
        q[:, 2] *= rng.uniform(0.3, 0.7)
    return _retract(q[None, :, :2].copy(), q[None, :, 2].copy())[0]


# --------------------------------------------------------------------------- warm start
def _read_pack(n):
    """Full committed census pack for n, or None.  GUARDED: any missing/odd file -> None."""
    p = "bench/packs/csqv%d.pck" % n
    try:
        if not os.path.exists(p):
            return None
        with open(p) as f:
            raw = [ln for ln in f.read().splitlines() if ln.strip()]
        rows = [[float(v) for v in ln.split()] for ln in raw[2:]]
        if len(rows) != n or any(len(rw) != 3 for rw in rows):
            return None
        a = np.asarray(rows, dtype=float)
        if not np.isfinite(a).all() or a[:, 2].min() <= 0:
            return None
        return a
    except Exception:
        return None


# --------------------------------------------------------------------------- per-n search
# Share of hop draws for each hop family.  Measured, not guessed: on the same seed and the same
# 3.29 s/n as the real run, the objective-space family was worth +0.31 mean digits on {27,39,45}
# (artifacts/proto_obj.py).  Both shares stay well under 1 so the undirected perturbations keep a
# solid fraction of the draws -- no family is allowed to become the only one.
# Iteration 9 re-weights these DOWN on a measurement, not a hunch (artifacts/proto_thru.py): on the
# four unsolved n, _manifold_hop costs 56-98 ms/draw and _sym_hop 41-48 ms while a graph hop or a
# perturbation costs 9-17 ms, so at 0.28/0.24 the two dearest families were eating ~75% of all move
# time for ~52% of the draws.  The search is DRAW-limited here (the same 5 s of the mix hit n=29's
# exact record on one seed and missed it on another, 81 vs 86 draws), so buying draws with the same
# CPU is the lever.  Both families stay well clear of zero -- no family is retired on one measurement.
P_MANIFOLD = 0.16
P_SYM = 0.20
# Iteration 7: the contact-graph family.  Measured, not guessed (artifacts/proto_graph3.py): from
# the n=45 incumbent, 4 s of graph hops found a basin at 3.500347 that 4 s of the whole existing
# mix (manifold+sym+perturb) never reached, and 3 of 47 draws hit it.  It takes its share from the
# undirected remainder, which still keeps ~26% of draws -- no family becomes the only one.
P_GRAPH = 0.20

POOL_MAX = 6
POOL_SEP = 1e-7           # two packs closer than this in sum_r are the same basin for our purposes


# Recombination's share.  It takes its slice from the undirected remainder, and it is small on
# purpose: a binary move needs two structurally distinct parents, so it returns None (and falls
# through to the undirected _perturb) whenever the pool has not yet found a second structure.  No
# family is allowed to become the only one.
P_CROSS = 0.12


# Iteration 9: the systematic single-swap scan.  Measured (artifacts/proto_scan.py): from the n=35
# incumbent it found the 3.072617 basin on BOTH seeds it was given, where the existing mix found it
# on one of two -- it is the more RELIABLE of the two on the structure it can reach, and it missed
# the n=29 basin the mix found, so the two are complements and both stay in the mix.  Its share is
# small because one call is a ~0.15 s slice against ~0.03 s for every other draw.
P_SCAN = 0.07


# Iteration 10: the graph-space BEAM and the STALL ESCALATION.  Four head-to-head measurements this
# iteration (proto_lns.py, proto_ruin.py, proto_beam.py) all returned ZERO improvements from the
# converged incumbents of n=35/37/39: the existing mix, LNS destroy-and-repair, ruin-and-recreate
# with a drifting anchor, a from-scratch cold search, AND the beam itself.  What that says about the
# ALLOCATION is sharper than what it says about any one move: once a basin is converged, the cheap
# one-step families have a measured yield of zero, so paying 190 ms for a depth-2 beam draw costs
# nothing real.  So the beam's share is not a constant -- it is 0.10 while the search is still
# accepting moves and 0.40 once the incumbent has survived STALL_DRAWS draws unchanged, and the
# other five families keep their RELATIVE weights inside whatever is left (the undirected remainder
# is re-normalised, never zeroed).  A fresh or cold n never stalls, so it never pays the escalation.
P_BEAM = 0.08
P_BEAM_STALL = 0.20
STALL_DRAWS = 35
# Iteration 11 adds the TABU CHAIN alongside the beam and gives it the larger half of the deep
# share.  Both are "depth through non-improving states"; they differ in how they pay for it, and
# that is precisely why both stay: the beam SURVEYS two edits in every direction (breadth-first,
# w^d cost, complete to its horizon), the chain COMMITS to one direction and rides it for hundreds
# of edits (linear cost, no horizon, no completeness).  Neither dominates the other -- a structure
# two edits away behind a downhill step is the beam's, a structure fifty edits away is only the
# chain's -- so the deep share is split rather than handed to the new mechanism.  Stalled: 0.20 beam
# + 0.40 chain = 0.60 of draws to the deep families, which the measured-zero yield of the cheap mix
# on a converged basin makes free rather than reckless; unstalled (a fresh/cold n): 0.08 + 0.10.
P_CHAIN = 0.10
P_CHAIN_STALL = 0.40


class _Pool:
    """The POPULATION of the per-n search: the best few STRUCTURALLY DISTINCT packings.

    Two changes from the sum_r-only pool of iterations 3-7: membership is keyed on the contact-graph
    FINGERPRINT (so two different structures that happen to score the same are both kept, instead of
    one silently evicting the other), and the trim to POOL_MAX prefers keeping one representative of
    each structure over keeping the top POOL_MAX scores.  That is what makes recombination worth
    having -- a pool of near-clones has nothing to recombine."""

    def __init__(self, rng, cap=None, sep=None):
        self.rng = rng
        self.cap = POOL_MAX if cap is None else int(cap)
        self.sep = POOL_SEP if sep is None else float(sep)
        self.e = []                              # [(sum_r, pack, fingerprint)] sorted descending

    def __len__(self):
        return len(self.e)

    def add(self, p, s):
        if p is None:
            return
        fp = _fp_safe(p)
        for j, (s2, _, f2) in enumerate(self.e):
            if abs(s2 - s) < self.sep and (fp is None or f2 is None or f2 == fp):
                if s > s2:
                    self.e[j] = (float(s), p.copy(), fp)
                    self.e.sort(key=lambda t: -t[0])
                return
        self.e.append((float(s), p.copy(), fp))
        self.e.sort(key=lambda t: -t[0])
        if len(self.e) > self.cap:               # DIVERSITY-PRESERVING trim, not a top-k truncation
            seen, keep, rest = set(), [], []
            for ent in self.e:
                if ent[2] is not None and ent[2] in seen:
                    rest.append(ent)
                else:
                    seen.add(ent[2])
                    keep.append(ent)
            keep = (keep + rest)[:self.cap]
            keep.sort(key=lambda t: -t[0])
            self.e = keep

    def pick(self, fallback=None):
        """Rank-biased draw: the champion most of the time, a challenger often enough to matter."""
        if not self.e:
            return fallback
        if len(self.e) == 1 or self.rng.rand() < 0.55:
            return self.e[0][1]
        return self.e[1 + int(self.rng.randint(0, len(self.e) - 1))][1]

    def pick2(self):
        """Two parents for recombination, preferring a STRUCTURALLY different second one."""
        if len(self.e) < 2:
            return None, None
        i = 0 if self.rng.rand() < 0.6 else int(self.rng.randint(0, len(self.e)))
        alt = [j for j in range(len(self.e))
               if j != i and (self.e[j][2] is None or self.e[j][2] != self.e[i][2])]
        if not alt:
            alt = [j for j in range(len(self.e)) if j != i]
        return self.e[i][1], self.e[alt[int(self.rng.randint(0, len(alt)))]][1]


def _solve_one(n, evaluate, meter, rng, deadline):
    best_pack, best_s = None, -np.inf
    pending = []
    pool = _Pool(rng)               # the POPULATION: best few structurally DISTINCT packings

    def record(p, s=None):
        """Track locally; flush to the metered evaluate() only when it beats the incumbent."""
        nonlocal best_pack, best_s
        if p is None:
            return False
        ok, sv = _feasible_sumr(p[None])
        if not ok[0] or float(sv[0]) <= best_s:
            return False
        best_pack, best_s = p.copy(), float(sv[0])
        pending.append(p.copy())
        return True

    def pool_add(p, s):
        pool.add(p, s)

    def pool_pick():
        return pool.pick(best_pack)

    def flush():
        nonlocal pending, best_pack, best_s
        if not pending or meter.left() <= 0:
            pending = []
            return
        sub = np.stack(pending[-min(len(pending), int(meter.left())):], axis=0)
        pending = []
        feas, val = evaluate(n, sub)
        feas = np.atleast_1d(feas)
        val = np.atleast_1d(val)
        for k in range(len(sub)):
            if feas[k] and float(val[k]) > best_s:
                best_s = float(val[k])
                best_pack = sub[k].copy()

    warm = _read_pack(n)
    if warm is not None:
        p, s = _slp_refine(_retract(warm[None, :, :2].copy(), warm[None, :, 2].copy())[0],
                           deadline=deadline)
        if record(p):
            pool_add(p, s)

    # ---- COLD TARGET-FEASIBILITY: the one phase that is forbidden to look at the incumbent -------
    # Iteration 13, and it is the phase that finally moved n=35 and n=37.  Everything else in
    # _solve_one starts from `warm` or from something descended from it; proto_infl3.py measured
    # that this incumbent is a TRAP on the open n (warm continuation at three pressures x four
    # seeds returned it to 1e-13 every time) while the SAME mechanism started COLD reached the
    # published record on both.  So this phase gets its slice FIRST and takes a deliberately cold
    # start, and it runs only on an n that is actually open: an n already at its record has nothing
    # to gain and keeps its CPU for the polish.
    tf_open = True
    if best_pack is not None:
        try:
            with open("bench/records.json") as _f:
                _r = json.load(_f).get("records", {})
            if str(n) in _r:
                tf_open = best_s < float(_r[str(n)]) * (1.0 - 1e-8)
        except Exception as _exc:
            LAST_ERRORS.append("tf_open:%r" % (_exc,))
    if tf_open and time.process_time() < deadline:
        tf_end = time.process_time() + TF_SHARE * max(0.0, deadline - time.process_time())

        def _tf_emit(p_, s_):
            if record(p_, s_):
                q_, s2_ = _slp_deep(p_, deadline=deadline)
                record(q_, s2_)
                flush()
            pool_add(p_, s_)

        _tf_phase(n, rng, min(deadline, tf_end), _tf_emit, best_s)
        flush()

    # ---- CROSS-n TRANSFER: the census is a coupled system, not 10 independent searches ------------
    # A packing of n+-1/+-2 shares almost its whole contact graph with a packing of n, so a neighbour
    # that is closer to ITS record is a free structural hypothesis for this n -- one no fixed-n move
    # could ever propose.  Guarded on every read; a size with no neighbours simply skips this.
    # Its share is 30% for an n with nothing (a fresh/held-out size, where a neighbour's structure is
    # the best hypothesis available) but only 8% for an n that already holds a converged incumbent:
    # iteration 9 measured the hop loop as DRAW-limited on exactly those n, and a bridge from a
    # neighbour has never once beaten a warm incumbent that was already at 1e-4.
    if time.process_time() < deadline:
        share = 0.30 if best_pack is None else 0.08
        tb_end = time.process_time() + share * max(0.0, deadline - time.process_time())
        for dn in (-2, 2, -1, 1, -3, 3):
            if time.process_time() > tb_end:
                break
            src_pack = _read_pack(n + dn)
            if src_pack is None:
                continue
            b = _bridge(n, src_pack, rng, min(deadline, tb_end))
            if b is None:
                continue
            ok, _ = _feasible_sumr(b[None])
            if not ok[0]:
                continue
            p, s = _slp_refine(b, deadline=deadline)
            record(p)
            pool_add(p, s)

    # ---- cold start: batched basin generation + CONSTRUCTIVE LADDERS, then exact SLP --------------
    if best_pack is None or warm is None:
        B = 16 if n > 60 else 24
        gen_until = time.process_time() + 0.45 * max(0.0, deadline - time.process_time())
        cost = 0.0
        turn = 0
        while time.process_time() + cost < gen_until:
            t = time.process_time()
            if turn % 2 == 0 or n < 3:
                xy = _make_seeds(n, B, rng, warm[:, :2] if warm is not None else None)
                r0 = np.full((B, n), 0.35 / max(1.0, np.sqrt(n)))
                xy, r = _ascend(xy, r0, 380, 30.0, 6e3, 8e-3, 3e-5)
                packs = _retract(xy, r)
                ok, s = _feasible_sumr(packs)
                order = np.argsort(np.where(ok, s, -np.inf))[::-1]
                cands = [packs[j] for j in order[:3] if ok[j]]
            else:                                # a ladder builds the structure one circle at a time
                seed = None
                if len(pool) and rng.rand() < 0.5:
                    m0 = max(2, int(n * rng.uniform(0.4, 0.8)))
                    seed = pool.e[int(rng.randint(0, len(pool)))][1][:m0]
                cands = [_ladder(n, rng, min(deadline, gen_until), seed=seed)]
            turn += 1
            for c in cands:
                p, sv = _slp_refine(c, deadline=deadline)
                record(p)
                pool_add(p, sv)
                # A cold start has no structure to inherit, so give it the one regime that IS
                # well-structured for every n: continue from near-equal radii (p0=0) down to p=1.
                if rng.rand() < 0.4 and time.process_time() < gen_until:
                    h = _homotopy(p, rng, min(deadline, gen_until), p0=0.0, stages=4)
                    if h is not None:
                        p2, s2 = _slp_refine(h, deadline=deadline)
                        record(p2)
                        pool_add(p2, s2)
            cost = max(cost, time.process_time() - t)
            if time.process_time() > gen_until:
                break
        if best_pack is None:                    # last resort: never leave an n with nothing
            xy = _make_seeds(n, 1, rng, None)
            record(_retract(xy, np.full((1, n), 0.25 / max(1.0, np.sqrt(n))))[0])
    flush()
    if best_pack is None:
        return
    if not len(pool):
        pool_add(best_pack, best_s)

    # ---- pool hopping: CHEAP refine for triage, DEEP refine only for winners ----------------------
    # The cheap refine settles a basin to ~1e-9 in ~5 LPs; the deep one costs ~6x more and is worth it
    # only when the basin actually wins, because the score is logarithmic in the residual gap.
    hop_cost = 0.0
    scan_seen = {}                  # per-structure frontier, so the scan never re-judges a swap
    beam_seen = {}                  # the beam keeps its OWN frontier: two independent systematic
    chain = {}                      # searches over the same neighbourhood, neither blinding the other
    since = 0                       # the CHAIN's persistent walk: ONE node, carried across draws
    while time.process_time() + hop_cost < deadline and meter.left() > 0:
        t = time.process_time()
        base = pool_pick()
        # ESCALATION: a converged basin gets the deep systematic search, not more cheap draws
        stalled = since >= STALL_DRAWS
        pb = P_BEAM_STALL if stalled else P_BEAM
        pc = P_CHAIN_STALL if stalled else P_CHAIN
        since += 1
        u = rng.rand()
        if u < pb + pc:
            if u < pc:
                # TABU CHAIN: continue ONE walk through the graph space, always moving, uphill or
                # not, with the last CHAIN_TABU edits' reversals forbidden (iteration 11)
                q = _graph_chain(base, rng, deadline, chain)
            else:
                # depth-2 BEAM over contact-graph edits, through NON-IMPROVING intermediates
                q = _graph_beam(base, rng, deadline, beam_seen)
            if q is None:
                q = _perturb(n, base, rng, deadline)
            ok, _ = _feasible_sumr(q[None])
            if ok[0]:
                p, s = _slp_refine(q, target=best_s, deadline=deadline)
                if s > best_s:
                    p, s = _slp_deep(p, deadline=deadline)
                    if record(p):
                        flush()
                        since = 0
                pool_add(p, s)
            hop_cost = max(hop_cost, time.process_time() - t)
            continue
        # re-normalise the draw so the other five families keep their RELATIVE shares
        u = (u - pb - pc) / max(1.0 - pb - pc, 1e-9)
        if u < P_MANIFOLD:
            # OBJECTIVE-SPACE hop: never leaves the feasible manifold (iteration 6)
            q = _manifold_hop(base, rng, deadline)
        elif u < P_MANIFOLD + P_SYM:
            # symmetry-reduced hop: solve inside a G-invariant subspace, then release to full space
            q = _sym_hop(base, rng, deadline)
        elif u < P_MANIFOLD + P_SYM + P_GRAPH:
            # STRUCTURE-SPACE hop: edit the contact graph and re-solve for it (iteration 7)
            q = _graph_hop(base, rng, deadline)
        elif u < P_MANIFOLD + P_SYM + P_GRAPH + P_CROSS:
            # RECOMBINATION: the one BINARY move -- two parents, one hybrid child (iteration 8)
            pa, pb = pool.pick2()
            q = _crossover(pa, pb, rng, deadline) if pa is not None else None
        elif u < P_MANIFOLD + P_SYM + P_GRAPH + P_CROSS + P_SCAN:
            # SYSTEMATIC graph-space local search: enumerate the single-swap neighbourhood rather
            # than sampling 2% of it, resuming where the last call stopped (iteration 9)
            q = _graph_scan(base, rng, deadline, scan_seen)
        else:
            q = None
        if q is None:
            q = _perturb(n, base, rng, deadline)
        ok, _ = _feasible_sumr(q[None])
        if ok[0]:
            # bail out of this basin as soon as it has converged short of the incumbent
            p, s = _slp_refine(q, target=best_s, deadline=deadline)
            if s > best_s:
                p, s = _slp_deep(p, deadline=deadline)
                if record(p):
                    flush()
                    since = 0
            pool_add(p, s)
        hop_cost = max(hop_cost, time.process_time() - t)
    # final insurance: the champion gets the full-precision schedule before it is submitted
    if best_pack is not None:
        p, s = _slp_deep(best_pack)
        record(p)
    flush()


W_SOLVED = 0.05           # an n at the 7-digit cap can gain nothing
W_OPEN = 0.8              # an open n that is NOT this call's focus -- alive, but not the bet
W_FOCUS = 5.0             # the focus n: ~6x the even split, and the regime the offline re-run uses


def _headroom(targets):
    """CPU weight per n -- CONCENTRATED on one open n, not split evenly over all of them.

    An n already at the 7-digit cap can gain NOTHING, so it is starved (W_SOLVED) and its CPU rolls
    into an n that can still move.  Among the n that CAN move, iteration 10 measured the marginal
    value of an even split directly: eight CPU-seconds of the whole mix, of LNS, of ruin-and-
    recreate, of a cold restart and of the beam all returned ZERO improvements on each of the three
    open n.  Three shares each worth zero are worth zero together, so splitting evenly is provably
    the wrong allocation and the untested regime is pouring the open budget into ONE n.  It also
    makes the in-run regime match the one that is actually GRADED: the offline re-run calls solve()
    with a single n and 60 CPU-seconds, which no in-run n has ever had.

    Iteration 13 changes WHICH n the rolled-up CPU goes to, and the reason is a measurement rather
    than a preference.  Iterations 11-12 concentrated on ONE open n (the smallest relative gap, on
    the theory that a long graph-space walk would reach the nearest structure) and it returned zero
    twice, at 19 CPU-s.  Meanwhile proto_infl3.py found the published record on n=35 AND n=37 from
    SIX CPU-seconds each of the cold target-feasibility phase.  So the binding resource is not
    "maximum CPU on one n", it is "enough CPU on EVERY open n to run the phase that works": every
    open n now gets W_FOCUS.  Three open n at W_FOCUS against seven starved solved ones gives each
    open n ~8.8 of the 27 in-run CPU-seconds, hence ~6 s of cold target search at TF_SHARE -- which
    is exactly the slice the measurement won on.  Nothing is ever cut to zero.

    Reading records.json/the census from solve() is sanctioned; both reads are GUARDED, so an
    unknown n (offline re-run, fresh size) falls back to FOCUS weight -- a size we know nothing
    about is exactly where the CPU belongs."""
    try:
        with open("bench/records.json") as f:
            recs = json.load(f).get("records", {})
    except Exception:
        recs = {}
    w, gaps = {}, {}
    for n in targets:
        rec = recs.get(str(n))
        a = _read_pack(n)
        if rec is None or a is None or not rec:
            w[n] = W_FOCUS                       # unknown size: it gets the focus weight outright
            continue
        gap = max(0.0, (float(rec) - float(a[:, 2].sum())) / float(rec))
        if gap <= 1.2e-7:
            w[n] = W_SOLVED
        else:
            w[n] = W_FOCUS                   # EVERY open n gets the focus weight (iteration 13)
            gaps[n] = gap
    return w


def solve(evaluate, meter, rng, targets, _cpu_budget=None):
    t0 = time.process_time()                     # ALWAYS at entry: solve() may be called repeatedly
    targets = [int(t) for t in (targets or [])]
    if not targets:
        return
    if _cpu_budget is not None:                  # self-test hook only; the driver never passes it
        total = float(_cpu_budget)
    elif len(targets) == 1:
        total = CPU_SINGLE
    else:
        total = CPU_TOTAL_MULTI
    w = _headroom(targets)
    wsum = sum(w.values()) or 1.0
    # cheap n first, so whatever they leave unspent rolls forward to the n that can use it
    order = sorted(targets, key=lambda n: (w[n], n))
    acc = 0.0
    for n in order:
        if meter.left() <= 0:
            break
        acc += total * w[n] / wsum
        dl = min(t0 + total, t0 + acc)
        now = time.process_time()
        if now >= t0 + total:
            break
        if dl - now < 0.05:
            dl = now + 0.05
        try:
            _solve_one(n, evaluate, meter, rng, dl)
        except Exception as exc:                 # one bad n must never cost the remaining targets
            LAST_ERRORS.append("n=%d: %r" % (n, exc))
            continue


# --------------------------------------------------------------------------- self-test
def _self_test():
    class _M:
        def __init__(self, b):
            self.budget = b; self.used = 0

        def left(self):
            return max(0, self.budget - self.used)

        def tick(self, k=1):
            g = max(0, min(int(k), self.budget - self.used)); self.used += g; return g

    best = {}

    def make_eval(meter):
        def ev(n, packing):
            a = np.asarray(packing, dtype=float)
            single = (a.ndim == 2)
            if single:
                a = a[None]
            if a.ndim != 3 or a.shape[2] != 3:
                raise ValueError("bad shape")
            if a.shape[1] != n:
                raise ValueError("wrong n")
            grant = meter.tick(a.shape[0])
            ok, s = _feasible_sumr(a)
            ok = ok.copy(); s = s.copy()
            if grant < a.shape[0]:
                ok[grant:] = False; s[grant:] = -np.inf
            for i in range(a.shape[0]):
                if ok[i] and s[i] > best.get(n, (-np.inf,))[0]:
                    best[n] = (float(s[i]), a[i].copy())
            return (bool(ok[0]), float(s[0])) if single else (ok, s)
        return ev

    fails = []
    del LAST_ERRORS[:]
    # (1) the conservative SLP must never hand back an infeasible or worse pack.
    rng0 = np.random.RandomState(3)
    for n in (7, 19):
        xy = rng0.uniform(LO + 0.05, HI - 0.05, size=(1, n, 2))
        p0 = _retract(xy, np.full((1, n), 0.01))[0]
        p1, s1 = _slp_refine(p0)
        if s1 < p0[:, 2].sum() - 1e-15:
            fails.append("SLP lost ground at n=%d" % n)
        if _slack(p1) < -SAFE_TOL or p1[:, 2].min() <= 0:
            fails.append("SLP returned an infeasible pack at n=%d (slack %.2e)" % (n, _slack(p1)))
        print("  slp n=%d: %.6f -> %.6f  slack %.1e" % (n, p0[:, 2].sum(), s1, _slack(p1)))
    # (2) two calls in ONE process: the second must still produce a feasible packing.
    m = _M(135000)
    rng = np.random.RandomState(0)
    for call in range(2):
        best.clear()
        t = time.process_time()
        solve(make_eval(m), m, rng, [11], _cpu_budget=3.0)
        el = time.process_time() - t
        if 11 not in best:
            fails.append("call %d produced no feasible packing for n=11" % call)
        else:
            print("  call %d: n=11 sum_r=%.6f cpu=%.1fs used=%d" % (call, best[11][0], el, m.used))
    # (3) len(targets)==1 and a multi-target split both work, incl. an n with no committed pack.
    m2 = _M(20000)
    best.clear()
    solve(make_eval(m2), m2, np.random.RandomState(1), [5, 27], _cpu_budget=5.0)
    for n in (5, 27):
        if n not in best:
            fails.append("multi-target: no feasible packing for n=%d" % n)
    print("  multi-target: " + ", ".join("n=%d sum_r=%.6f" % (n, best[n][0]) for n in sorted(best)))
    # (4) strict independent recheck of every pack the harness kept.
    for n, (s, pk) in list(best.items()):
        if pk[:, 2].min() <= 0:
            fails.append("n=%d has a non-positive radius" % n)
        if _slack(pk) < -FEAS_TOL:
            fails.append("n=%d infeasible: slack=%.3e" % (n, _slack(pk)))
    # (5) budget exhaustion must not crash or overspend.
    m3 = _M(5)
    best.clear()
    solve(make_eval(m3), m3, np.random.RandomState(2), [9], _cpu_budget=2.0)
    print("  exhausted-budget call: used=%d of 5" % m3.used)
    if m3.used > 5:
        fails.append("overspent the budget")
    # (6) warm-start read must be guarded for an n with no pack.
    if _read_pack(9999) is not None:
        fails.append("_read_pack did not guard a missing pack")
    # (7) NOTHING may have been swallowed by solve()'s per-n guard.
    if LAST_ERRORS:
        fails.append("solve() swallowed exceptions: " + "; ".join(LAST_ERRORS[:4]))
    # (8) the basin-hopping loop must actually RUN, not die on the first flush: a warm n given real
    #     CPU has to submit MORE than the single cold/warm-start candidate.
    m4 = _M(135000)
    best.clear()
    del LAST_ERRORS[:]
    solve(make_eval(m4), m4, np.random.RandomState(5), [27], _cpu_budget=4.0)
    print("  hop-loop check: n=27 submitted %d candidates" % m4.used)
    if m4.used < 2:
        fails.append("hop loop submitted only %d candidate(s) -- it is not running" % m4.used)
    if LAST_ERRORS:
        fails.append("solve() swallowed exceptions: " + "; ".join(LAST_ERRORS[:4]))
    # (9) the DIRECTED reseat must produce a strictly feasible pack, and _free_point must name a hole
    #     whose claimed free radius is really free.
    rng9 = np.random.RandomState(7)
    p9 = _retract(rng9.uniform(LO + 0.05, HI - 0.05, size=(1, 17, 2)), np.full((1, 17), 0.02))[0]
    pt, f = _free_point(p9)
    d9 = np.sqrt(((pt[None, :] - p9[:, :2]) ** 2).sum(-1)) - p9[:, 2]
    wall9 = min(pt[0] - LO, HI - pt[0], pt[1] - LO, HI - pt[1])
    if f > min(float(d9.min()), wall9) + 1e-12 or f <= 0:
        fails.append("_free_point overstated the hole: claimed %.3e, actual %.3e" % (f, d9.min()))
    q9 = _reseat(17, p9, rng9, 3)
    if _slack(q9) < -SAFE_TOL or q9[:, 2].min() <= 0 or len(q9) != 17:
        fails.append("_reseat returned an infeasible pack (slack %.2e)" % _slack(q9))
    print("  free-point hole r=%.4f ; reseat slack %.1e" % (f, _slack(q9)))
    # (10) the cheap/early exits must never break feasibility, and the DEEP refine must not lose to
    #      the cheap one -- it is the cheap one's own output, polished.
    p10 = _retract(np.random.RandomState(11).uniform(LO + 0.05, HI - 0.05, size=(1, 23, 2)),
                   np.full((1, 23), 0.01))[0]
    c10, sc = _slp_refine(p10)
    d10, sd = _slp_deep(c10)
    if _slack(c10) < -SAFE_TOL or _slack(d10) < -SAFE_TOL:
        fails.append("refine/deep produced an infeasible pack")
    if sd < sc - 1e-15:
        fails.append("deep refine LOST ground vs cheap refine (%.12f -> %.12f)" % (sc, sd))
    # a target the basin cannot reach must cut the refine short but still return a feasible pack.
    c11, s11 = _slp_refine(p10, target=sc + 1.0)
    if _slack(c11) < -SAFE_TOL or s11 < float(p10[:, 2].sum()) - 1e-15:
        fails.append("target bail-out returned an infeasible or worse pack")
    print("  cheap %.9f -> deep %.9f ; bail-out %.9f" % (sc, sd, s11))
    # (11) CPU allocation must be guarded: an n with no pack/record gets the MAXIMAL weight (it is
    #      the size we know least about, so it is where the CPU belongs), and nobody is zeroed.
    w11 = _headroom([9999, 27])
    if (w11.get(9999) != W_FOCUS or w11[9999] < max(w11.values())
            or min(w11.values()) <= 0.0):
        fails.append("_headroom mis-weighted an unknown n: %r" % (w11,))
    # (12) n-CHANGING MOVES.  _grow must insert without creating a violation, _shrink must never
    #      create one, and both must keep the count exact.
    rng12 = np.random.RandomState(13)
    p12 = _retract(rng12.uniform(LO + 0.05, HI - 0.05, size=(1, 15, 2)), np.full((1, 15), 0.03))[0]
    g12 = _grow(p12, rng12)
    if len(g12) != 16 or g12[:, 2].min() <= 0 or _slack(g12) < -SAFE_TOL:
        fails.append("_grow produced an infeasible pack (slack %.2e, k=%d)" % (_slack(g12), len(g12)))
    s12 = _shrink(g12, rng12)
    if len(s12) != 15 or _slack(s12) < -SAFE_TOL:
        fails.append("_shrink produced an infeasible pack (slack %.2e)" % _slack(s12))
    e12 = _grow(np.zeros((0, 3)), rng12)         # growing from NOTHING must work (fresh cold start)
    if len(e12) != 1 or e12[0, 2] <= 0 or _slack(e12) < -SAFE_TOL:
        fails.append("_grow failed on an empty pack")
    # (13) LADDER: a constructive cold start must hand back exactly n strictly feasible circles for
    #      an n that has no census pack at all, and must beat a trivial packing.
    l13 = _ladder(21, np.random.RandomState(17), time.process_time() + 2.0)
    if len(l13) != 21 or l13[:, 2].min() <= 0 or _slack(l13) < -SAFE_TOL:
        fails.append("_ladder returned an infeasible pack (n=%d, slack %.2e)" % (len(l13), _slack(l13)))
    print("  ladder n=21 sum_r=%.6f slack %.1e" % (l13[:, 2].sum(), _slack(l13)))
    if float(l13[:, 2].sum()) < 1.0:
        fails.append("_ladder produced a degenerate packing: sum_r=%.4f" % l13[:, 2].sum())
    # (14) BRIDGE: carrying a pack across sizes must land on EXACTLY n, feasible, in both directions,
    #      and must survive a source that does not exist.
    for tgt in (19, 23):
        b14 = _bridge(tgt, l13, np.random.RandomState(19), time.process_time() + 2.0)
        if b14 is None or len(b14) != tgt or b14[:, 2].min() <= 0 or _slack(b14) < -SAFE_TOL:
            fails.append("_bridge 21->%d failed (%r)" % (tgt, None if b14 is None else _slack(b14)))
        else:
            print("  bridge 21->%d sum_r=%.6f slack %.1e" % (tgt, b14[:, 2].sum(), _slack(b14)))
    # (15) SHRINK-RELAX-REGROW must return exactly n strictly feasible circles and must not be a
    #      no-op: the relax step has to actually move the structure.
    p15 = _slp_refine(l13)[0]
    q15 = _srr(21, p15, np.random.RandomState(23), 2, time.process_time() + 2.0)
    if len(q15) != 21 or q15[:, 2].min() <= 0 or _slack(q15) < -SAFE_TOL:
        fails.append("_srr returned an infeasible pack (n=%d, slack %.2e)" % (len(q15), _slack(q15)))
    if float(np.abs(np.sort(q15[:, 2]) - np.sort(p15[:, 2])).max()) < 1e-9:
        fails.append("_srr was a no-op -- the relax step did nothing")
    print("  srr: %.6f -> %.6f (pre-refine), slack %.1e"
          % (p15[:, 2].sum(), q15[:, 2].sum(), _slack(q15)))
    # (16) SYMMETRY-REDUCED SEARCH.  Three things must hold or the representation is a lie:
    #      (a) the reduced map must reproduce the symmetrised pack exactly -- every orbit member is
    #          B@u, so the pack must be EXACTLY G-invariant (this is the assertion that catches a
    #          bad orbit assignment); (b) the reduced SLP may never leave the subspace and may never
    #          return an infeasible pack; (c) releasing into the full space may never LOSE ground.
    for gname in ("mx", "d1", "mxy", "c4", "d4"):
        res16 = _sym_map(p15, GROUPS[gname])
        if res16 is None:
            continue
        q16, M16, sm16 = res16
        if len(q16) != 21 or q16[:, 2].min() <= 0 or _slack(q16) < -SAFE_TOL:
            fails.append("_sym_map[%s] returned an infeasible pack (slack %.2e)" % (gname, _slack(q16)))
            continue
        # (a) exact G-invariance of the symmetrised pack
        sub16 = q16[sm16]
        dev = 0.0
        for g in GROUPS[gname]:
            img = sub16[:, :2] @ g.T
            d16 = np.sqrt(((img[:, None, :] - sub16[None, :, :2]) ** 2).sum(-1)).min(axis=1)
            dev = max(dev, float(d16.max()))
        if dev > 1e-9:
            fails.append("_sym_map[%s] pack is not G-invariant (dev %.2e)" % (gname, dev))
        r16, s16 = _slp_refine(q16, iters=40, M=M16)
        # (b) still feasible, still symmetric, and never worse than the start
        if r16[:, 2].min() <= 0 or _slack(r16) < -SAFE_TOL:
            fails.append("reduced SLP[%s] broke feasibility (slack %.2e)" % (gname, _slack(r16)))
        sub2 = r16[sm16]
        dev2 = 0.0
        for g in GROUPS[gname]:
            img = sub2[:, :2] @ g.T
            d16 = np.sqrt(((img[:, None, :] - sub2[None, :, :2]) ** 2).sum(-1)).min(axis=1)
            dev2 = max(dev2, float(d16.max()))
        if dev2 > 1e-7:
            fails.append("reduced SLP[%s] left the invariant subspace (dev %.2e)" % (gname, dev2))
        if s16 < float(q16[:, 2].sum()) - 1e-12:
            fails.append("reduced SLP[%s] lost ground" % gname)
        # (c) release into the full space is monotone
        f16, fs16 = _slp_refine(r16, iters=40)
        if fs16 < s16 - 1e-12:
            fails.append("release[%s] lost ground (%.9f -> %.9f)" % (gname, s16, fs16))
        print("  sym[%s]: orbits=%d sym-circles=%d/%d  %.6f -> %.6f -> %.6f (released)"
              % (gname, M16[0].shape[1] // 3, int(sm16.sum()), len(q16),
                 q16[:, 2].sum(), s16, fs16))
    # (17) _sym_hop must survive a size with no symmetry to find and must return n feasible circles.
    h17 = _sym_hop(p15, np.random.RandomState(31), time.process_time() + 3.0, name="mxy")
    if h17 is not None and (len(h17) != 21 or h17[:, 2].min() <= 0 or _slack(h17) < -SAFE_TOL):
        fails.append("_sym_hop returned an infeasible pack")
    # (18) WEIGHTED SLP.  The whole objective-space family rests on one claim: reweighting the
    #      objective changes the DIRECTION of ascent and nothing else.  So (a) a weighted step must
    #      still be strictly feasible; (b) it must ascend the WEIGHTED objective monotonically; and
    #      (c) since p15 is already a KKT point of max sum(r), a non-uniform weight must actually
    #      MOVE it -- if it does not, the weights are being ignored and every continuation hop is
    #      silently a no-op.  (c) is the assertion that would catch a dropped `wv` argument.
    w18 = _weights(p15[:, 2], 0.0)
    r18, s18 = _slp_refine(p15, iters=40, w=w18)
    if r18[:, 2].min() <= 0 or _slack(r18) < -SAFE_TOL:
        fails.append("weighted SLP broke feasibility (slack %.2e)" % _slack(r18))
    if float(w18 @ r18[:, 2]) < float(w18 @ p15[:, 2]) - 1e-12:
        fails.append("weighted SLP lost ground on its OWN objective")
    mv18 = float(np.abs(r18 - p15).max())
    if mv18 < 1e-6:
        fails.append("weighted SLP did not move a sum(r)-KKT point -- weights ignored (%.2e)" % mv18)
    print("  weighted SLP (p=0): moved %.3e, w.r %.6f -> %.6f, sum_r %.6f -> %.6f"
          % (mv18, w18 @ p15[:, 2], w18 @ r18[:, 2], p15[:, 2].sum(), s18))
    # (19) CONTINUATION.  A homotopy hop must (a) return exactly n strictly feasible circles,
    #      (b) end at a DIFFERENT point than it started (it is a structural move, not a polish), and
    #      (c) after the caller's true-objective refine, be no worse than a plain deep refine of the
    #      start -- i.e. the move may explore, but the pipeline around it may never lose ground.
    base19, bs19 = _slp_deep(p15)
    h19 = _homotopy(base19, np.random.RandomState(41), time.process_time() + 6.0)
    if h19 is None or len(h19) != 21 or h19[:, 2].min() <= 0 or _slack(h19) < -SAFE_TOL:
        fails.append("_homotopy returned an infeasible pack (%r)" % (None if h19 is None else _slack(h19),))
    else:
        if float(np.abs(h19 - base19).max()) < 1e-7:
            fails.append("_homotopy was a no-op")
        f19, fs19 = _slp_deep(h19)
        if f19[:, 2].min() <= 0 or _slack(f19) < -SAFE_TOL:
            fails.append("_homotopy release broke feasibility")
        print("  homotopy: %.6f -> %.6f (deformed) -> %.6f (released)"
              % (bs19, h19[:, 2].sum(), fs19))
    # (20) _wshake and _manifold_hop (which may run the continuation inside a reduced subspace)
    #      must both return exactly n strictly feasible circles or None -- never a broken pack.
    for nm20, fn20 in (("_wshake", lambda: _wshake(base19, np.random.RandomState(43),
                                                   time.process_time() + 4.0)),
                       ("_manifold_hop", lambda: _manifold_hop(base19, np.random.RandomState(44),
                                                               time.process_time() + 6.0))):
        q20 = fn20()
        if q20 is not None and (len(q20) != 21 or q20[:, 2].min() <= 0 or _slack(q20) < -SAFE_TOL):
            fails.append("%s returned an infeasible pack (slack %.2e)" % (nm20, _slack(q20)))
        else:
            print("  %s: %s" % (nm20, "None" if q20 is None else "sum_r=%.6f slack %.1e"
                                % (q20[:, 2].sum(), _slack(q20))))
    # (21) CONTACT-GRAPH MECHANISM (iteration 7).  Three separate claims, each of which would
    #      otherwise fail silently:
    #      (a) an SLP-converged pack really IS a KKT point of its own active set -- the multipliers
    #          from _duals must satisfy grad(sum r) + J^T lam = 0 to ~1e-10 and be non-negative.
    #          If this fails the whole contact-graph view is wrong and every hop is noise.
    #      (b) _kkt_newton on the UNEDITED active set must stay put (it is already a solution) and
    #          must drive ||F|| down, not up -- the check that catches a sign or ordering error in
    #          _gJ / _lag_hess, which a merely "returns something feasible" test would not.
    #      (c) _gn_graph on an EDITED graph must reduce ||g|| for that graph, i.e. it really does
    #          move toward realising the requested structure rather than wandering.
    pk21, sk21 = _slp_deep(p15)
    pr21, wl21, _, _ = _act_set(pk21)
    m21 = len(pr21) + len(wl21)
    lam21, res21 = _duals(pk21, pr21, wl21)
    if m21 == 0:
        fails.append("_act_set found no active contacts at an SLP optimum")
    elif res21 > 1e-9:
        fails.append("stationarity residual %.2e at an SLP optimum: the KKT view is wrong" % res21)
    elif lam21.min() < -1e-7:
        fails.append("_duals gave a negative multiplier %.2e" % lam21.min())
    else:
        print("  duals: m=%d (3n=%d) ||grad f + J^T lam||=%.2e lam in [%.3f, %.3f]"
              % (m21, 3 * len(pk21), res21, lam21.min(), lam21.max()))
    q21, l21, nF21 = _kkt_newton(pk21, pr21, wl21, lam21, steps=4)
    if nF21 > 1e-9 or float(np.abs(q21 - pk21).max()) > 1e-6:
        fails.append("_kkt_newton moved a KKT point of its own graph (||F||=%.2e, moved %.2e)"
                     % (nF21, float(np.abs(q21 - pk21).max())))
    else:
        print("  kkt_newton(unedited): ||F||=%.2e moved %.2e" % (nF21, float(np.abs(q21 - pk21).max())))
    rg21 = np.random.RandomState(47)
    P21, W21, L21 = _graph_edit(pk21, rg21, drop=2, add=2)
    if len(P21) + len(W21) == 0:
        fails.append("_graph_edit returned an empty graph")
    else:
        g0 = float(np.linalg.norm(_gJ(pk21, P21, W21)[0]))
        s21, gn21 = _gn_graph(pk21, P21, W21, steps=14)
        if gn21 > max(1e-12, 0.9 * g0):
            fails.append("_gn_graph did not realise the edited graph (||g|| %.2e -> %.2e)" % (g0, gn21))
        else:
            print("  gn_graph(edited m=%d): ||g|| %.2e -> %.2e" % (len(P21) + len(W21), g0, gn21))
    # (22) _graph_hop must return exactly n strictly feasible circles or None, never a broken pack,
    #      and over several draws must actually MOVE (a structure-space no-op is the failure mode).
    moved22 = 0.0
    for s22 in (51, 52, 53, 54):
        h22 = _graph_hop(pk21, np.random.RandomState(s22), time.process_time() + 4.0)
        if h22 is None:
            continue
        if len(h22) != 21 or h22[:, 2].min() <= 0 or _slack(h22) < -SAFE_TOL:
            fails.append("_graph_hop returned an infeasible pack (slack %.2e)" % _slack(h22))
            break
        moved22 = max(moved22, float(np.abs(h22 - pk21).max()))
        r22, v22 = _slp_refine(h22, patience=3, gtol=1e-11)
        if r22[:, 2].min() <= 0 or _slack(r22) < -SAFE_TOL:
            fails.append("_graph_hop result broke feasibility under the SLP")
            break
    else:
        if moved22 < 1e-6:
            fails.append("_graph_hop was a no-op over 4 draws (max move %.2e)" % moved22)
        else:
            print("  graph_hop: max structural move %.3e over 4 draws" % moved22)

    # (23) FINGERPRINT must be a genuine invariant of the CONTACT GRAPH: unchanged by relabelling
    #      the circles and by every symmetry of the square, and it must actually DISCRIMINATE -- a
    #      fingerprint that never changes would silently turn the diversity pool back into a top-k
    #      list, and nothing else in the solver would notice.
    fp0 = _fingerprint(pk21)
    perm = np.random.RandomState(9).permutation(len(pk21))
    if _fingerprint(pk21[perm]) != fp0:
        fails.append("_fingerprint is not relabelling-invariant")
    refl = pk21.copy(); refl[:, 0] *= -1.0
    rot = pk21.copy(); rot[:, [0, 1]] = np.stack([-pk21[:, 1], pk21[:, 0]], 1)
    if _fingerprint(refl) != fp0 or _fingerprint(rot) != fp0:
        fails.append("_fingerprint is not invariant under the square's symmetries")
    fps23 = {fp0}
    for s23 in (61, 62, 63, 64, 65, 66):
        h23 = _graph_hop(pk21, np.random.RandomState(s23), time.process_time() + 3.0)
        if h23 is None:
            continue
        rf23, _ = _slp_refine(h23, patience=3, gtol=1e-11)
        fps23.add(_fingerprint(rf23))
    if len(fps23) < 2:
        fails.append("_fingerprint never changed across 6 structure-space hops (not discriminating)")
    else:
        print("  fingerprint: m=(%d pair,%d wall), %d distinct structures over 6 graph hops"
              % (fp0[0], fp0[1], len(fps23)))

    # (24) The POOL must keep two packings that have the SAME sum_r but DIFFERENT structure.  This is
    #      the whole point of the change and the old sum_r-only rule failed it, so it is asserted on
    #      a hand-built pair (equal radii, one touching the walls, one floating free) rather than
    #      hoped for.
    def _sq(c, rr):
        return np.array([[c, c, rr], [-c, c, rr], [c, -c, rr], [-c, -c, rr]])
    pA, pB = _sq(0.30, 0.20), _sq(0.25, 0.20)     # identical sum_r = 0.8, 8 wall contacts vs 0.
    if abs(pA[:, 2].sum() - pB[:, 2].sum()) > 1e-15 or _slack(pA) < -SAFE_TOL or _slack(pB) < -SAFE_TOL:
        fails.append("self-test 24 fixture is malformed")
    if _fingerprint(pA) == _fingerprint(pB):
        fails.append("_fingerprint cannot tell a wall-contacting square from a free one")
    pl24 = _Pool(np.random.RandomState(3))
    pl24.add(pA, float(pA[:, 2].sum()))
    pl24.add(pB, float(pB[:, 2].sum()))
    if len(pl24) != 2:
        fails.append("_Pool merged two DISTINCT structures with equal sum_r (kept %d)" % len(pl24))
    else:
        a24, b24 = pl24.pick2()
        if a24 is None or float(np.abs(a24 - b24).max()) < 1e-9:
            fails.append("_Pool.pick2 did not return two different parents")
        else:
            print("  pool: kept 2 structures at identical sum_r %.3f; pick2 gave distinct parents"
                  % pA[:, 2].sum())
    pl24.add(pA * 1.0, float(pA[:, 2].sum()))      # a re-add of the SAME structure must not grow it
    if len(pl24) != 2:
        fails.append("_Pool duplicated an identical structure (%d entries)" % len(pl24))

    # (25) RECOMBINATION: a child of two n-circle parents must be exactly n strictly feasible
    #      circles, must survive the SLP, and must be a real hybrid -- different from BOTH parents
    #      (a crossover that quietly returns a copy of one parent is the silent failure here).
    pa25 = pk21
    pb25 = None
    for s25 in (71, 72, 73, 74, 75, 76, 77, 78):
        h25 = _graph_hop(pk21, np.random.RandomState(s25), time.process_time() + 3.0)
        if h25 is None:
            continue
        c25, _ = _slp_refine(h25, patience=3, gtol=1e-11)
        if _fingerprint(c25) != fp0 and _slack(c25) >= -SAFE_TOL:
            pb25 = c25
            break
    if pb25 is None:
        pb25 = _slp_refine(_perturb(21, pk21, np.random.RandomState(79),
                                    time.process_time() + 3.0), patience=3, gtol=1e-11)[0]
    nhyb = 0
    for s25 in (81, 82, 83, 84, 85, 86):
        ch = _crossover(pa25, pb25, np.random.RandomState(s25), time.process_time() + 4.0)
        if ch is None:
            continue
        if len(ch) != 21 or ch[:, 2].min() <= 0 or _slack(ch) < -SAFE_TOL:
            fails.append("_crossover returned an infeasible child (slack %.2e)" % _slack(ch))
            break
        rc, _ = _slp_refine(ch, patience=3, gtol=1e-11)
        if len(rc) != 21 or rc[:, 2].min() <= 0 or _slack(rc) < -SAFE_TOL:
            fails.append("_crossover child broke feasibility under the SLP")
            break
        if float(np.abs(ch - pa25).max()) > 1e-6 and float(np.abs(ch - pb25).max()) > 1e-6:
            nhyb += 1
    else:
        if nhyb == 0:
            fails.append("_crossover never produced a child distinct from both parents")
        else:
            print("  crossover: %d/6 draws gave a feasible hybrid distinct from both parents" % nhyb)

    # (26) _deflate must repair overlaps by shrinking ONLY what conflicts -- the property that makes
    #      a recombined seam survivable.  A globally rescaling repair would fail the second check.
    bad26 = pk21.copy()
    bad26[0, 2] *= 2.5                               # one circle blown up into its neighbours
    d26 = _deflate(bad26)
    dd26 = np.sqrt(((pk21[:, None, :2] - pk21[None, :, :2]) ** 2).sum(-1))[0]
    hit26 = dd26 < bad26[0, 2] + pk21[:, 2] + 1e-12  # the circles the blown-up one actually reaches
    hit26[0] = True
    if _slack(d26) < -SAFE_TOL:
        fails.append("_deflate left an infeasible pack (slack %.2e)" % _slack(d26))
    elif float(np.abs(d26[~hit26, 2] - pk21[~hit26, 2]).max(initial=0.0)) > 1e-12:
        fails.append("_deflate shrank a circle that was in no conflict (global rescale)")
    else:
        print("  deflate: repaired a 2.5x overlap, %d/%d radii untouched (%d in conflict)"
              % (int((~hit26).sum()), len(pk21), int(hit26.sum())))

    # (27) The scan must be SYSTEMATIC and RESUMABLE, which is the whole claim over the random
    #      _graph_edit.  Three things here fail SILENTLY (the scan would still return packings, it
    #      would just be a slower random sampler): (a) it must never judge the same swap twice, so a
    #      second call on the same structure advances the frontier instead of re-scanning the prefix;
    #      (b) it must EXHAUST and return None rather than spin forever on a finished neighbourhood;
    #      (c) the swap ordering must be the dual/slack one, not input order.
    pk27 = _slp_refine(pk21, patience=6, gtol=1e-13)[0]
    pr27, wl27, st27, wb27 = _act_set(pk27)
    lm27 = _duals(pk27, pr27, wl27)[0]
    cnd1 = _swap_candidates(pk27, lm27, pr27, wl27, st27, wb27, kmax=1)
    seen27 = {}
    rng27 = np.random.RandomState(909)
    outs27, tot27, calls27 = [], 0, 0
    t27 = time.process_time()
    for _ in range(400):
        q27 = _graph_scan(pk27, rng27, time.process_time() + 30.0, seen27, kmax=1)
        calls27 += 1
        sz = sum(len(v) for v in seen27.values())
        if sz == tot27:                       # no new swap judged: the frontier is exhausted
            if q27 is not None:
                fails.append("_graph_scan judged 0 new swaps but still returned a pack")
            break
        tot27 = sz
        if q27 is None:
            continue                          # slice found nothing usable; the frontier still moved
        if len(q27) != 21 or q27[:, 2].min() <= 0 or _slack(q27) < -SAFE_TOL:
            fails.append("_graph_scan returned an infeasible neighbour (slack %.2e)" % _slack(q27))
            break
        outs27.append(q27)
    else:
        fails.append("_graph_scan never exhausted its neighbourhood in 400 calls")
    dt27 = time.process_time() - t27
    if not fails:
        cnd14 = _swap_candidates(pk27, lm27, pr27, wl27, st27, wb27, kmax=14)
        nd27 = len(set(tuple(o.ravel().round(9)) for o in outs27))
        if len(cnd1) < 10 or len(cnd14) <= len(cnd1):
            fails.append("_swap_candidates enumerated %d/%d swaps (kmax 1/14)"
                         % (len(cnd1), len(cnd14)))
        elif not all(cnd1[k][0] <= cnd1[k + 1][0] + 1e-15 for k in range(len(cnd1) - 1)):
            fails.append("_swap_candidates is not ordered by lam_drop * slack_add")
        elif tot27 != len(cnd1):
            # the frontier must cover the neighbourhood EXACTLY once: no repeats, nothing skipped
            fails.append("_graph_scan covered %d of %d swaps (repeat or skip)" % (tot27, len(cnd1)))
        elif nd27 < 2:
            fails.append("_graph_scan returned the same neighbour every call")
        else:
            print("  graph scan: %d swaps (kmax=1; %d at kmax=14), dual-ordered, each judged "
                  "EXACTLY once over %d resumed calls -> %d distinct neighbours, %.0f swaps/cpu-s"
                  % (len(cnd1), len(cnd14), calls27, nd27, tot27 / max(dt27, 1e-9)))

    # (28) solve() must survive being called TWICE in one process (the deadline-at-entry contract)
    #      with the new CPU split, and must still work when len(targets) == 1.
    class _M28:
        budget = 4000
        used = 0

        def left(self):
            return max(0, self.budget - self.used)

        def tick(self, k=1):
            g = max(0, min(int(k), self.budget - self.used)); self.used += g; return g

    got28 = {}

    def _ev28(n, pack):
        pack = np.asarray(pack, float)
        b = pack[None] if pack.ndim == 2 else pack
        m28.tick(len(b))
        ok, s = _feasible_sumr(b)
        for k in range(len(b)):
            if ok[k] and s[k] > got28.get(n, (-np.inf,))[0]:
                got28[n] = (float(s[k]), b[k].copy())
        return (ok, s) if pack.ndim == 3 else (bool(ok[0]), float(s[0]))

    m28 = _M28()
    for _call in (1, 2):
        got28.clear()
        solve(_ev28, m28, np.random.RandomState(31), [13], _cpu_budget=1.2)
        if 13 not in got28 or _slack(got28[13][1]) < -SAFE_TOL or len(got28[13][1]) != 13:
            fails.append("solve() call #%d returned no feasible n=13 packing" % _call)
            break
    else:
        print("  solve(): two calls in one process, single-target, both feasible (sum_r %.6f)"
              % got28[13][0])

    # ---- (29) the BEAM: feasible output, resumable ROOT frontier, and depth-2 REACHABILITY -------
    # The reachability assertion is the entire claim of the mechanism over _graph_scan: a depth-2
    # node is a structure NO single swap of the root can reach.  If it failed, the beam would still
    # return packings -- it would just be a slower one-step scan -- so nothing else here would
    # notice.  (The other three prototypes this iteration measured the existing families at ZERO
    # improvements from a converged incumbent, which is why this mechanism has to be the new one.)
    rb = np.random.RandomState(7)
    pb29, _s29 = _slp_refine(_ladder(18, rb, time.process_time() + 4.0))
    seen29 = {}
    q1 = _graph_beam(pb29, rb, time.process_time() + 3.0, seen29, depth=2, width=3, per=4)
    n1 = sum(len(v) for v in seen29.values())
    if q1 is None or len(q1) != 18 or _slack(q1) < -SAFE_TOL or q1[:, 2].min() <= 0.0:
        fails.append("beam: returned %s, not a strictly feasible 18-circle pack" % type(q1))
    else:
        q2 = _graph_beam(pb29, rb, time.process_time() + 3.0, seen29, depth=2, width=3, per=4)
        n2 = sum(len(v) for v in seen29.values())
        if n2 <= n1 or n1 == 0:
            fails.append("beam: root frontier not resumable (%d -> %d swaps judged)" % (n1, n2))
        # depth-2 REACHABILITY: a child's neighbourhood must contain a structure that is neither
        # the root nor any single swap of it.
        end29 = time.process_time() + 6.0
        lvl1 = _swap_level(pb29, 40, end29, set())
        fp_root = {_fp_safe(pb29)} | {_fp_safe(z) for _s, z in lvl1}
        reach = 0
        for _s, child in sorted(lvl1, key=lambda t: -t[0])[:4]:
            for _s2, g in _swap_level(child, 12, end29, set()):
                if _fp_safe(g) not in fp_root:
                    reach += 1
        if not lvl1:
            fails.append("beam: single-swap level of an n=18 ladder pack is empty")
        elif reach == 0:
            fails.append("beam: depth 2 reached NOTHING outside the root's 1-swap neighbourhood")
        else:
            print("  beam: feasible, root frontier resumable (%d -> %d swaps), depth-2 reached %d "
                  "structures outside the root's %d-node 1-swap neighbourhood"
                  % (n1, n2, reach, len(lvl1)))

    # ---- (30) the TABU CHAIN: non-improving acceptance, unbounded depth, persistence ------------
    # Three assertions, and each one is a claim the mechanism would lose SILENTLY: if the walk
    # refused downhill steps it would be a slow _graph_scan; if it did not persist across calls it
    # would be a short beam with width 1; if it did not get many edits from the root it would buy no
    # more depth than the beam already had.  All three would still return feasible packings.
    rc = np.random.RandomState(5)
    pc30, _s30 = _slp_refine(_ladder(18, rc, time.process_time() + 4.0))
    st30 = {}
    t30 = time.process_time()
    c1 = _graph_chain(pc30, rc, time.process_time() + 3.0, st30, slice_s=1.5)
    s1 = st30.get("steps", 0)
    c2 = _graph_chain(pc30, rc, time.process_time() + 3.0, st30, slice_s=1.5)
    rate = st30.get("steps", 0) / max(1e-9, time.process_time() - t30)
    if c1 is None or len(c1) != 18 or _slack(c1) < -SAFE_TOL or c1[:, 2].min() <= 0.0:
        fails.append("chain: returned %s, not a strictly feasible 18-circle pack" % type(c1))
    elif c2 is None or _slack(c2) < -SAFE_TOL:
        fails.append("chain: second call returned no strictly feasible pack")
    elif st30["steps"] <= s1 or s1 == 0:
        fails.append("chain: walk not persistent across calls (%d -> %d steps)"
                     % (s1, st30["steps"]))
    else:
        tr = st30["trace"]
        down = sum(1 for i in range(1, len(tr)) if tr[i] < tr[i - 1] - 1e-12)
        # depth: the FURTHEST the walk ever got from its origin, in contact-graph edits.  It is the
        # max and not the final node on purpose -- a walk that reaches distance 30 and then
        # re-intensifies back on the incumbent has still searched at distance 30, which is the claim.
        d30 = st30["dmax"]
        lo30 = st30["bs"] * (1.0 - CHAIN_BAND)
        if any(v < lo30 - 1e-12 for v in tr):
            fails.append("chain: %d of %d visited nodes fell outside the acceptance band -- the "
                         "walk is leaking off the manifold"
                         % (sum(1 for v in tr if v < lo30 - 1e-12), len(tr)))
        elif down == 0:
            fails.append("chain: accepted NO downhill step in %d -- it is a hill-climber" % len(tr))
        elif st30["steps"] < 10:
            fails.append("chain: only %d steps in 3.0 CPU-s -- too slow to buy depth"
                         % st30["steps"])
        elif d30 <= 4:
            fails.append("chain: the walk never got further than %d contact identities from its "
                         "origin -- no deeper than the depth-2 beam" % d30)
        elif len(st30["tabu"]) > 2 * CHAIN_TABU:
            fails.append("chain: tabu list unbounded (%d > %d)"
                         % (len(st30["tabu"]), 2 * CHAIN_TABU))
        else:
            print("  chain: %d steps over two calls (%.0f steps/CPU-s), %d of them DOWNHILL, "
                  "reached depth %d contact identities from the origin, tabu %d/%d"
                  % (st30["steps"], rate, down, d30, len(st30["tabu"]), 2 * CHAIN_TABU))

    # ---- (31) CPU ALLOCATION: EVERY open n gets focus weight, the solved ones stay starved ------
    # Iteration 11 asserted "exactly one focus n"; iteration 13 deliberately changed that contract
    # (see _headroom) after single-n concentration returned zero twice at 19 CPU-s while six
    # CPU-seconds of the cold target phase reached the record on two n.  The assertion is REPLACED,
    # not dropped, and the replacement is tighter than the original on the parts that still hold:
    # it pins the weight of EVERY n in the call (the old one only counted the focus set), and it
    # still requires the solved n starved, nobody zeroed, and an unknown n at focus weight.
    hw = _headroom([27, 35, 37, 39, 45])
    open_n = [35, 37, 39]
    nfoc = sorted(n for n, v in hw.items() if v == W_FOCUS)
    if nfoc != open_n:
        fails.append("headroom: focus set %r, expected every open n %r (%r)" % (nfoc, open_n, hw))
    elif hw[27] != W_SOLVED or hw[45] != W_SOLVED:
        fails.append("headroom: a solved n was not starved (%r)" % hw)
    elif min(hw.values()) <= 0.0:
        fails.append("headroom: an n was cut to zero (%r)" % hw)
    elif not all(hw[n] >= 20.0 * hw[27] for n in open_n):
        fails.append("headroom: an open n is not decisively favoured over a solved one (%r)" % hw)
    else:
        hw2 = _headroom([200001])            # an n with no record and no pack: GUARDED fallback
        # the slice an open n actually gets must be big enough for the phase that won
        frac = W_FOCUS / (3 * W_FOCUS + 7 * W_SOLVED)
        tf_s = TF_SHARE * frac * CPU_TOTAL_MULTI
        if hw2.get(200001) != W_FOCUS:
            fails.append("headroom: unknown n did not fall back to focus weight (%r)" % hw2)
        elif tf_s < 5.0:
            fails.append("headroom: an open n gets only %.1f CPU-s of cold target search, under "
                         "the 6 s the record hits were measured at" % tf_s)
        else:
            print("  headroom: all %d open n at %.1f, solved %.2f, unknown n -> focus; each open n "
                  "gets %.1f s of which %.1f s is cold target search"
                  % (len(nfoc), W_FOCUS, W_SOLVED, frac * CPU_TOTAL_MULTI, tf_s))

    # ---- (32) EXACT RADIUS REPAIR: the LP recovers a collapsed pack the greedy shrink cannot ----
    # Necessary conditions only (iteration 11's lesson), but this one constrains the QUALITY of the
    # repair, not just that it returns something: a collapsed candidate must come back on-manifold.
    rs32 = np.random.RandomState(32)
    n32 = 24
    p32 = _read_pack(n32)
    if p32 is None:                               # any n: build one through the cold path
        p32 = _retract(_grid_seed(n32, 5, rs32)[None], np.full((1, n32), 0.05))[0]
    n32 = len(p32)
    base32 = float(p32[:, 2].sum())
    # a RAW graph-solve output that collapses: one pair driven into deep overlap, the rest exact
    # (a lone overlapping pair is not the failure mode -- the greedy regrow sweep handles that.
    #  A DIVERGED Newton returns every centre displaced at once, which is what is built here.)
    q32 = p32.copy()
    q32[:, :2] = np.clip(q32[:, :2] + rs32.randn(n32, 2) * float(p32[:, 2].mean()) * 0.45,
                         LO + 1e-9, HI - 1e-9)
    zg = _retract(q32[None, :, :2].copy(), np.abs(q32[None, :, 2]).copy() + 1e-9)[0]
    zl = _repair(q32, ref=base32)
    sg, sl32 = float(zg[:, 2].sum()), float(zl[:, 2].sum())
    rlp = _lp_radii(np.ascontiguousarray(p32[:, :2]))
    if _slack(zl) < -FEAS_TOL or zl[:, 2].min() <= 0.0:
        fails.append("repair: LP repair returned an INFEASIBLE pack (slack %.3e)" % _slack(zl))
    elif rlp is None:
        fails.append("repair: _lp_radii returned None on a plain incumbent (scipy missing?)")
    elif float(rlp.sum()) < base32 * (1.0 - 1e-7):
        fails.append("repair: the LP optimum for the incumbent's own centres (%.9f) is BELOW the "
                     "incumbent's sum_r (%.9f) -- the LP is not solving the right problem"
                     % (float(rlp.sum()), base32))
    elif sl32 <= sg * (1.0 + 1e-9):
        fails.append("repair: on a deliberately collapsed candidate the LP recovered %.9f, no more "
                     "than the greedy global shrink (%.9f) -- the whole mechanism is inert"
                     % (sl32, sg))
    else:
        # GATE: a candidate the Newton actually solved must NOT pay for an LP (~10 ms vs ~0.1 ms)
        t0 = time.process_time()
        for _ in range(20):
            zq = _repair(p32, ref=base32)
        tg = (time.process_time() - t0) / 20.0
        t0 = time.process_time()
        for _ in range(3):
            _lp_radii(np.ascontiguousarray(p32[:, :2]))
        tl = (time.process_time() - t0) / 3.0
        if tg > 0.5 * tl:
            fails.append("repair: the gate is not gating -- a clean pack costs %.1f ms, an LP "
                         "%.1f ms" % (1e3 * tg, 1e3 * tl))
        elif _slack(zq) < -FEAS_TOL:
            fails.append("repair: gated (greedy) path returned an infeasible pack")
        else:
            print("  repair: collapsed candidate  greedy %.6f -> LP %.6f (+%.3e, base %.6f); "
                  "gate keeps a clean pack at %.2f ms vs %.2f ms for the LP"
                  % (sg, sl32, sl32 - sg, base32, 1e3 * tg, 1e3 * tl))

    # ---- (33) HARD-TARGET FEASIBILITY: the gradient, the oracle, and the cold-only rule ---------
    # This block exists because the FIRST two versions of _tf_eg had sign errors -- one in the pair
    # term, one in the wall term -- and neither failed loudly.  A wrong-sign gradient makes L-BFGS's
    # line search fail on step 1, so the optimiser returns its own input and every downstream number
    # looks plausible: proto_infl.py produced a full table of "measurements" of a no-op.  So the
    # first assertion here is a FINITE-DIFFERENCE check, which is the only kind of assertion that
    # could have caught that, and it is a statement about CORRECTNESS, not about counts.
    print("(33) hard-target feasibility")
    rs33 = np.random.RandomState(0)
    n33, T33 = 7, 0.9
    z33 = np.concatenate([(rs33.rand(n33, 2) - 0.5).ravel() * 0.9, rs33.randn(n33) * 0.3])
    E33, g33 = _tf_eg(z33, n33, T33)
    num33 = np.zeros_like(g33)
    for kk in range(len(z33)):
        hh = 1e-7
        za, zb = z33.copy(), z33.copy()
        za[kk] += hh
        zb[kk] -= hh
        num33[kk] = (_tf_eg(za, n33, T33)[0] - _tf_eg(zb, n33, T33)[0]) / (2 * hh)
    cos33 = float(g33 @ num33 / max(1e-300, np.linalg.norm(g33) * np.linalg.norm(num33)))
    err33 = float(np.abs(g33 - num33).max())
    if not (cos33 > 1 - 1e-7 and err33 < 1e-5):
        fails.append("tf: gradient disagrees with finite differences (cos=%.9f maxerr=%.2e)"
                     % (cos33, err33))
    else:
        print("  gradient: cos(analytic, numeric) = %.12f, maxerr %.2e (E=%.4e)"
              % (cos33, err33, E33))
    # the softmax constraint is the formulation's whole point: sum r == T for ANY a
    r33 = T33 * np.exp(z33[2 * n33:] - z33[2 * n33:].max())
    r33 = r33 / r33.sum() * T33 / T33
    r33 = T33 * (np.exp(z33[2 * n33:] - z33[2 * n33:].max())
                 / np.exp(z33[2 * n33:] - z33[2 * n33:].max()).sum())
    if abs(float(r33.sum()) - T33) > 1e-12 or r33.min() <= 0:
        fails.append("tf: the parametrisation does not hold sum r == T with r > 0")
    # THE ORACLE: E ~ 0 at target T must mean a feasible pack whose sum_r is ~ T.  Asserted on a
    # target we KNOW is realisable -- an incumbent's own total -- so a failure is the mechanism's,
    # not the target's.
    n33b = 24
    rs33b = np.random.RandomState(4)
    xy33 = (rs33b.rand(n33b, 2) - 0.5) * 0.8
    rr33 = _lp_radii(xy33)
    base33 = float(_retract(xy33[None].copy(), rr33[None].copy())[0][:, 2].sum())
    zt = np.concatenate([xy33.ravel(), np.log(np.clip(rr33, 1e-12, None) / max(rr33.sum(), 1e-12))])
    zt, Et = _tf_solve(n33b, base33, zt, 600)
    qt, st = _tf_harvest(n33b, zt, time.process_time() + 3.0)
    if qt is None or _slack(qt) < -FEAS_TOL or qt[:, 2].min() <= 0:
        fails.append("tf: harvest of a realisable target returned an infeasible pack")
    elif st < base33 - 1e-9:
        fails.append("tf: harvest at a realisable target LOST value (%.9f -> %.9f)" % (base33, st))
    else:
        print("  oracle: realisable target %.6f -> E=%.2e, harvest %.9f (%+.2e), slack %.2e"
              % (base33, Et, st, st - base33, _slack(qt)))
    # TARGETS: a visible n gets the published scalar; an n with no entry gets the sqrt-law
    # extrapolation and NOT a crash -- this is the held-out-size path, so it is the one that must
    # not raise.  And T is never below what we already hold.
    t35 = _tf_target(35, -np.inf)
    t100 = _tf_target(100, -np.inf)
    t35hi = _tf_target(35, 9.0)
    if abs(t35 - 3.074036363728) > 1e-9:
        fails.append("tf_target: visible n=35 did not return the published scalar (%.9f)" % t35)
    elif not (4.0 < t100 < 7.0):
        fails.append("tf_target: held-out n=100 extrapolated to an absurd %.4f" % t100)
    elif t35hi <= 9.0:
        fails.append("tf_target: a target below the incumbent was not lifted (%.6f)" % t35hi)
    else:
        print("  target: n=35 -> %.9f (published), n=100 -> %.6f (sqrt-law, no record entry), "
              "floor at incumbent honoured" % (t35, t100))
    # COLD-ONLY: _tf_cold_z must not depend on the census.  Assert it directly -- a start that
    # correlates with the incumbent is the failure mode the measurement says kills this family.
    pk33 = _read_pack(35)
    if pk33 is not None:
        zc = _tf_cold_z(35, np.random.RandomState(1), "grid")
        dmean = float(np.abs(zc[:70].reshape(35, 2) - pk33[:, :2]).mean())
        if dmean < 0.05:
            fails.append("tf: cold start is suspiciously close to the incumbent (mean |dx|=%.4f)"
                         % dmean)
        else:
            print("  cold: grid start sits mean |dx|=%.3f from the incumbent centres" % dmean)
    # DEADLINE: a phase given no time must do nothing, emit nothing, and not raise (this is what
    # protects every later n in a multi-target call).
    got = []
    k33 = _tf_phase(29, np.random.RandomState(2), time.process_time() - 1.0,
                    lambda q, v: got.append(v), -np.inf)
    if k33 != 0 or got:
        fails.append("tf_phase: ran %d starts and emitted %d packs past an expired deadline"
                     % (k33, len(got)))
    else:
        print("  deadline: an expired slice runs 0 starts and emits nothing")
    # and one real slice on a small n: whatever it emits must be strictly feasible
    got2 = []
    _tf_phase(12, np.random.RandomState(5), time.process_time() + 2.0,
              lambda q, v: got2.append((q, v)), -np.inf)
    bad = [1 for q, v in got2 if _slack(q) < -FEAS_TOL or q[:, 2].min() <= 0 or len(q) != 12]
    if bad:
        fails.append("tf_phase: %d of %d emitted packs were not strictly feasible"
                     % (len(bad), len(got2)))
    else:
        print("  emissions: %d packs on n=12, all strictly feasible, best %.6f"
              % (len(got2), max([v for _q, v in got2], default=float("nan"))))

    # (34) BASIN HOPPING ON E.  proto_e0.py proved the cold descent's E ~ 1e-5 is a TRUE local
    #      minimum (6000 extra L-BFGS iterations and a Gauss-Newton least_squares both move it by
    #      zero), so the KICK is the search and these are the properties that make it one.
    print("basin hopping on E (block 34)")
    n34, rng34 = 24, np.random.RandomState(31)
    T34 = _tf_target(n34, -np.inf)
    z34 = _tf_cold_z(n34, rng34, "unif")
    #  (a) every kick must be a REAL structural move -- not a no-op -- and must stay in the box and
    #      on the simplex.  A kick that silently returns its input is exactly the failure mode that
    #      wasted iteration 13's first prototype, and it would leave the hopper looking like it ran.
    for kind in TF_KICKS[:4] + ("swap",):
        z2 = _tf_kick(z34, n34, T34, rng34, kind)
        xy2 = z2[:2 * n34].reshape(n34, 2)
        e2 = np.exp(z34[2 * n34:] - z34[2 * n34:].max())
        d34 = float(np.abs(z2 - z34).max())
        if d34 < 1e-9:
            fails.append("_tf_kick(%s) was a NO-OP" % kind)
        if xy2.min() < LO - 1e-12 or xy2.max() > HI + 1e-12:
            fails.append("_tf_kick(%s) left the box" % kind)
        if not np.isfinite(z2).all():
            fails.append("_tf_kick(%s) produced non-finite state" % kind)
        print("  kick %-5s max|dz|=%.4f" % (kind, d34))
    #  (b) the cheap harvest must be STRICTLY FEASIBLE (it is the one that skips the SLP, so nothing
    #      downstream re-repairs it) and must really be cheap relative to the full one.
    z35, _E35 = _tf_solve(n34, T34, z34, 400)
    t35 = time.process_time()
    qc, sc34 = _tf_cheap(n34, z35)
    tc = time.process_time() - t35
    t35 = time.process_time()
    qf, sf34 = _tf_harvest(n34, z35, time.process_time() + 5.0)
    tf34 = time.process_time() - t35
    if qc is None:
        fails.append("_tf_cheap returned nothing on a converged descent")
    elif _slack(qc) < -FEAS_TOL or qc[:, 2].min() <= 0 or len(qc) != n34:
        fails.append("_tf_cheap emitted an INFEASIBLE pack (slack %.2e)" % _slack(qc))
    if qf is not None and sf34 < sc34 - 1e-12:
        fails.append("the full harvest LOST to the cheap one (%.9f -> %.9f)" % (sc34, sf34))
    if tc > tf34:
        fails.append("the cheap harvest is not cheaper (%.1f ms vs %.1f ms)"
                     % (1e3 * tc, 1e3 * tf34))
    print("  cheap %.6f in %.1f ms ; full %.6f in %.1f ms" % (sc34, 1e3 * tc, sf34, 1e3 * tf34))
    #  (c) the hopper must actually HOP -- kicks > restarts, i.e. the threshold acceptance keeps a
    #      walk alive rather than degenerating into multi-start -- and every pack it emits must be
    #      strictly feasible, and its best E must beat a single cold descent's.
    got34 = []
    nk34, nr34, gE34 = _tf_bh(n34, T34, np.random.RandomState(37),
                              time.process_time() + 3.0, lambda q, v: got34.append((q, v)))
    if nk34 < 5:
        fails.append("_tf_bh ran only %d kicks in 3 s -- it is not hopping" % nk34)
    if nk34 <= nr34:
        fails.append("_tf_bh degenerated into multi-start (%d kicks, %d restarts)" % (nk34, nr34))
    if not got34:
        fails.append("_tf_bh emitted nothing -- the tickets are still being thrown away")
    badb = [1 for q, v in got34 if _slack(q) < -FEAS_TOL or q[:, 2].min() <= 0 or len(q) != n34]
    if badb:
        fails.append("_tf_bh emitted %d of %d infeasible packs" % (len(badb), len(got34)))
    _z0, E0 = _tf_solve(n34, T34, _tf_cold_z(n34, np.random.RandomState(41), "unif"), TF_ITER_M)
    if not (gE34 <= E0):
        fails.append("_tf_bh (E=%.2e) did not beat one cold descent (E=%.2e)" % (gE34, E0))
    print("  bh: %d kicks / %d restarts, bestE=%.2e vs cold %.2e, %d feasible emissions, best %.6f"
          % (nk34, nr34, gE34, E0, len(got34), max([v for _q, v in got34], default=float("nan"))))
    #  (d) an expired slice must hop zero times and emit nothing (this guards every later n).
    got35 = []
    nk35, nr35, _ = _tf_bh(n34, T34, np.random.RandomState(43), time.process_time() - 1.0,
                           lambda q, v: got35.append(v))
    if nk35 or nr35 or got35:
        fails.append("_tf_bh past an expired deadline: %d kicks, %d restarts, %d emits"
                     % (nk35, nr35, len(got35)))
    else:
        print("  deadline: an expired hop slice does nothing")

    # (35) A POPULATION ON E WITH Z-SPACE CROSSOVER.  The properties asserted here are the ones
    #      whose absence would let the mode LOOK like it ran while doing nothing new: a child that
    #      is really just a copy of a parent, a child that silently leaves the simplex (so it is no
    #      longer an answer to the same feasibility question), and a population that collapses onto
    #      one structure and degenerates into the hopper it was added to hedge.
    print("population on E with z-space crossover (block 35)")
    n35, rng35 = 22, np.random.RandomState(71)
    T35 = _tf_target(n35, -np.inf)
    zpa, Epa = _tf_solve(n35, T35, _tf_cold_z(n35, rng35, "unif"), 400)
    zpb, Epb = _tf_solve(n35, T35, _tf_cold_z(n35, rng35, "grid"), 400)
    #  (a) the child must MIX -- inherit circles from BOTH parents -- and be distinct from each.
    mixed = 0
    for trial in range(12):
        ch = _tf_zcross(zpa, zpb, n35, T35, np.random.RandomState(100 + trial))
        if not np.isfinite(ch).all():
            fails.append("_tf_zcross produced a non-finite child")
            break
        cx = ch[:2 * n35].reshape(n35, 2)
        if cx.min() < LO - 1e-12 or cx.max() > HI + 1e-12:
            fails.append("_tf_zcross put a centre outside the box")
            break
        #  the z-space property: sum r == T EXACTLY, for free, with every r > 0
        w = np.exp(ch[2 * n35:] - ch[2 * n35:].max())
        rr = T35 * w / w.sum()
        if rr.min() <= 0.0 or abs(rr.sum() - T35) > 1e-12 * T35:
            fails.append("_tf_zcross child is off the simplex (sum_r - T = %.2e, min r = %.2e)"
                         % (rr.sum() - T35, rr.min()))
            break
        da = np.abs(cx[:, None] - zpa[:2 * n35].reshape(n35, 2)[None]).sum(-1).min(1)
        db = np.abs(cx[:, None] - zpb[:2 * n35].reshape(n35, 2)[None]).sum(-1).min(1)
        if (da < 1e-12).any() and (db < 1e-12).any():
            mixed += 1
        if float(np.abs(ch - zpa).max()) < 1e-9 or float(np.abs(ch - zpb).max()) < 1e-9:
            fails.append("_tf_zcross returned a PARENT -- the recombination is a no-op")
            break
    else:
        if mixed < 8:
            fails.append("_tf_zcross mixed both parents in only %d of 12 trials" % mixed)
        else:
            print("  crossover: %d/12 children inherit from BOTH parents, all on the simplex"
                  % mixed)
    #  (b) the population must actually recombine, stay diverse, and beat one cold descent on E.
    got36 = []
    nk36, nx36, gE36 = _tf_pop(n35, T35, np.random.RandomState(73),
                               time.process_time() + 3.0, lambda q, v: got36.append(q))
    if nx36 < 1:
        fails.append("_tf_pop ran %d crossovers in 3 s -- it is not a population" % nx36)
    elif gE36 >= min(Epa, Epb):
        fails.append("_tf_pop (E=%.2e) did not beat one cold descent (E=%.2e)"
                     % (gE36, min(Epa, Epb)))
    elif not got36:
        fails.append("_tf_pop emitted nothing")
    else:
        badp = [1 for q in got36 if _slack(q) < -FEAS_TOL or q[:, 2].min() <= 0 or len(q) != n35]
        if badp:
            fails.append("_tf_pop emitted %d of %d infeasible packs" % (len(badp), len(got36)))
        else:
            print("  pop: %d kicks / %d crossovers, E %.2e -> %.2e, %d feasible emissions"
                  % (nk36, nx36, min(Epa, Epb), gE36, len(got36)))
    #  (c) an expired slice must do nothing at all (this is what protects every later n).
    got37 = []
    nk37, nx37, _ = _tf_pop(n35, T35, np.random.RandomState(79), time.process_time() - 1.0,
                            lambda q, v: got37.append(v))
    if nk37 or nx37 or got37:
        fails.append("_tf_pop past an expired deadline: %d kicks, %d crossovers, %d emits"
                     % (nk37, nx37, len(got37)))
    else:
        print("  deadline: an expired population slice does nothing")

    if fails:
        print("SELF-TEST FAIL:")
        for f in fails:
            print("  - " + f)
        return 1
    print("SELF-TEST PASS")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(__doc__)
