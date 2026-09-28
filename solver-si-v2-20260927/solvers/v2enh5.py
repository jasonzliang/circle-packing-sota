"""Circle-packing solver for the packomania `csqv` family: pack n circles into the unit square
maximising the sum of radii.

Method -- an always-feasible sequential convex program (SCP) wrapped in basin hopping.

The problem is

    max  sum_i r_i   s.t.   ||p_i - p_j|| >= r_i + r_j,   r_i <= dist(p_i, wall),   r_i >= 0.

The only non-convexity is ||p_i - p_j||. For ANY unit vector u,  ||v|| >= u . v, so replacing the
pair constraint by

    u_ij . (p_i - p_j) >= r_i + r_j     with  u_ij = (p_i - p_j)/||p_i - p_j||  at the CURRENT point

gives a LINEAR program that is an *inner* (conservative) restriction of the feasible set: every LP
solution is feasible for the true problem, and the current point is itself LP-feasible, so the LP
objective is monotonically non-decreasing. Iterating the LP (with a trust region on the centres so the
linearisation directions stay fresh) is a convex-concave procedure that climbs to a local optimum while
never leaving the feasible set -- which is exactly what this scorer wants, since only STRICTLY feasible
packings count.

On top of that local solver sits basin hopping: perturb the incumbent (teleport the smallest circles,
rattle every centre) and re-run the SCP, keeping the global best; periodically restart from a fresh
jittered hexagonal-ish lattice (the csqv optimum is a near-equal-circle packing, so a lattice is a good
basin to start in). Warm starts read the committed census when it exists, so quality ratchets across
iterations, and STRUCTURE TRANSFER seeds n from the committed packs of nearby sizes (delete the
smallest circles when m > n, drop circles into the largest empty gaps when m < n), so a size that
reaches the record also becomes a seed for its neighbours.

Every candidate is made exactly feasible before it is scored: radii are recomputed by an exact
radius-only LP at the final centres, uniformly scaled down if any true constraint is violated, then
greedily re-grown, then shaved by a tiny margin so the scorer's pure-Python geometry agrees.

Contract notes: all CPU deadlines are computed at CALL entry (solve() may be called repeatedly in one
process); every warm-start read is guarded so any n -- including sizes with no committed pack -- cold
starts cleanly; randomness comes only from `rng`; every packing that should count is routed through the
metered `evaluate()`.
"""
import json
import math
import os
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

try:                                  # the LP engine scipy.optimize.linprog itself wraps
    from scipy.optimize._highspy import _core as _hs
except Exception:                     # pragma: no cover -- any scipy without the vendored HiGHS
    _hs = None

LO, HI = -0.5, 0.5
DIGITS_CAP = 7.0       # the scorer's per-n digit cap (re-implemented locally, never imported)
SHAVE = 1e-11          # radius margin so the scorer's math.hypot geometry agrees with numpy's
PACK_DIR = os.path.join("bench", "packs")

# CPU allowances, computed fresh at every solve() entry (never module-level absolute times).
CPU_ONE_TARGET = 56.0  # offline held-out re-run: ONE size per call under a 60 s process-CPU cap
CPU_MANY = 30.0        # in-run: all visible n in one call under the driver's 32 s CPU backstop
FOCUS_K = 6            # how many n one multi-target call CONSIDERS (see _focus/_rounds)
HALVING_W = 2.0        # successive halving: each round's CPU share grows by this factor
ARM_FLOOR = 3          # how many arms the LAST round keeps (see _rounds)
MIN_ROUNDS = 2         # least rounds a multi-target call is split into (see _rounds)
                       # 1 = no-op: halving alone decides. With SKIP_CAPPED dropping the arm count
                       # below ARM_FLOOR, halving buys ZERO rounds, so this is what keeps the
                       # probe-then-resume shape the live arms want (iteration 15).


# --------------------------------------------------------------------------- geometry helpers
def _pairs(n):
    return np.triu_indices(n, 1)


def _wall(xy):
    return np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0], xy[:, 1] - LO, HI - xy[:, 1]])


def _dists(xy, I, J):
    d = xy[I] - xy[J]
    return np.sqrt((d * d).sum(1))


def _active_pairs(xy, I, J, dist, keep):
    """Pair rows to put in the LP. For moderate n every pair is cheap and exact; for large n keep only
    the near pairs (plus any that a previous LP violated, passed in via `keep`) -- violations are
    detected against ALL pairs afterwards, so correctness never depends on the filter."""
    if len(I) <= 6000:
        return np.ones(len(I), dtype=bool)
    cut = 6.0 / np.sqrt(len(xy))
    return (dist < cut) | keep


# --------------------------------------------------------------------------- the SCP's LP engine
class _Lp:
    """The linearised SCP LP for a fixed n: columns = (x, y, r); rows = a SELECTED subset of the
    n(n-1)/2 pair rows and the 4n wall rows.

    Two measured facts shape this class (n=45, one LP of a converged packing):

    * `scipy.optimize.linprog` re-validates its options and rebuilds the model on every call, which
      costs far more than the simplex solve itself at this size. Going straight to the vendored
      HiGHS and handing back the PREVIOUS BASIS turns a ~24 ms call into ~1.8 ms.
    * Of those 1.8 ms almost all is simplex work on 1170 rows for a problem with only 135 columns,
      whose active set is ~3n rows. Dropping the provably non-binding rows leaves ~136 rows and
      0.27 ms (6.5x); a COLD basis on the same small LP costs 1.5 ms, so the row set must stay
      fixed for as many consecutive solves as possible.

    So the row set is chosen by `set_rows()` and then held: `_scp` only rebuilds it when a row it
    dropped could become binding inside the current trust region. Dropping is EXACT, not heuristic
    -- with centres confined to +-`delta` and radii to `+rho`, a pair with slack >= 2*delta + 2*rho
    and a wall with slack >= delta + rho cannot be violated, so the filtered LP has exactly the
    same optimum as the full one over that box.

    Falls back to linprog on the current row set if the private HiGHS module is unavailable."""

    def __init__(self, n):
        self.n = n
        I, J = _pairs(n)
        self.I, self.J = I, J
        self.npair = len(I)
        self.nv = 3 * n
        self.c = np.zeros(self.nv)
        self.c[2 * n:] = -1.0
        self.hi = None
        self.basis = None
        self.pmask = None
        self.wmask = None
        if _hs is not None:
            try:
                self.hi = _hs._Highs()
                self.hi.setOptionValue("output_flag", False)
                self.hi.setOptionValue("presolve", "off")
                # Two per-solve costs that buy nothing on THIS LP. `passModel` re-runs HiGHS's
                # matrix EQUILIBRATION on every call, and the SCP calls passModel once per LP --
                # but these rows are already equilibrated by construction (pair rows are unit
                # normals plus two 1.0s; wall rows are +-1 and 1.0), so scaling is pure overhead.
                # And the default simplex strategy resolves to DUAL, which re-solves from scratch
                # after the bound changes the trust region makes every iteration; the warm basis we
                # hand back is PRIMAL-feasible (the previous optimum is feasible for the new, only
                # bound-shifted, LP) so primal simplex restarts from it in a few pivots.
                # MEASURED (artifacts/prof14g.py, medians of 3 interleaved blocks after a warm-up
                # block, 20 perturbed proposals of the committed pack per block, full _scp climbs):
                #   n=35  7.33 -> 6.29 ms/proposal   n=41 12.95 -> 11.20   n=45 14.02 -> 11.30
                # i.e. 14-19% more basin-hopping proposals per CPU second, and the LP optima are
                # BIT-IDENTICAL (sum of 20 climb objectives agrees to 1e-10 across all four option
                # combinations at every n) -- this buys tickets, it does not change the search.
                # Individually: scaling off is worth ~5%, primal ~9%; both are kept.
                # Set in their OWN guard: an option name this HiGHS build does not know must not
                # cost us the fast backend entirely (the outer except drops to the 5x slower
                # linprog path, which would be a far bigger loss than the tuning is a gain).
                for _k, _v in (("simplex_scale_strategy", 0), ("simplex_strategy", 4)):
                    try:
                        self.hi.setOptionValue(_k, _v)
                    except Exception:
                        pass
            except Exception:
                self.hi = None
        self.set_rows(np.ones(self.npair, dtype=bool), np.ones(4 * n, dtype=bool))

    # -- row selection -------------------------------------------------------
    def set_rows(self, pmask, wmask):
        """Fix the LP's rows to the selected pair/wall rows and rebuild the CSC pattern.

        Costs ~0.07 ms at n=45 (a lexsort over ~6m+2w entries) and invalidates the warm basis, so
        the caller keeps a selection for as long as it stays provably sufficient."""
        n = self.n
        self.pmask = pmask
        self.wmask = wmask
        pi = np.flatnonzero(pmask)
        wj = np.flatnonzero(wmask)
        self.pi = pi
        m = len(pi)
        nw = len(wj)
        self.m = m
        self.nrow = m + nw
        Ia, Ja = self.I[pi], self.J[pi]
        self.Ia, self.Ja = Ia, Ja
        prow = np.repeat(np.arange(m), 6)
        pcol = np.stack([Ia, Ja, n + Ia, n + Ja, 2 * n + Ia, 2 * n + Ja], 1).ravel()
        kind, wi = wj // n, wj % n
        wrow = np.repeat(np.arange(nw), 2) + m
        wcol = np.empty(2 * nw, dtype=np.int64)
        wcol[0::2] = np.where(kind < 2, wi, n + wi)
        wcol[1::2] = 2 * n + wi
        self.wdat = np.empty(2 * nw)
        self.wdat[0::2] = np.where(kind % 2 == 0, 1.0, -1.0)
        self.wdat[1::2] = 1.0
        self.rows = np.concatenate([prow, wrow])
        self.cols = np.concatenate([pcol, wcol])
        self.b = np.concatenate([np.zeros(m), np.full(nw, 0.5)])
        self.basis = None
        if self.hi is not None:
            order = np.lexsort((self.rows, self.cols))
            self.order = order
            self.indices = self.rows[order].astype(np.int32)
            self.indptr = np.zeros(self.nv + 1, dtype=np.int32)
            np.cumsum(np.bincount(self.cols, minlength=self.nv), out=self.indptr[1:])
            self.lhs = np.full(self.nrow, -_hs.kHighsInf)

    def values(self, xy, dist):
        """The flat row-ordered coefficient array for the linearisation at `xy` (selected rows)."""
        d = np.maximum(dist[self.pi], 1e-15)
        u = (xy[self.Ia] - xy[self.Ja]) / d[:, None]
        one = np.ones(self.m)
        pd = np.stack([-u[:, 0], u[:, 0], -u[:, 1], u[:, 1], one, one], 1).ravel()
        return np.concatenate([pd, self.wdat])

    # -- the solve -----------------------------------------------------------
    def solve(self, vals, lb, ub):
        """max sum r over the linearised set. Returns the solution vector or None."""
        if self.hi is not None:
            try:
                lp = _hs.HighsLp()
                lp.num_col_ = self.nv
                lp.num_row_ = self.nrow
                lp.a_matrix_.num_col_ = self.nv
                lp.a_matrix_.num_row_ = self.nrow
                lp.a_matrix_.format_ = _hs.MatrixFormat.kColwise
                lp.col_cost_ = self.c
                lp.col_lower_ = lb
                lp.col_upper_ = ub
                lp.row_lower_ = self.lhs
                lp.row_upper_ = self.b
                lp.a_matrix_.start_ = self.indptr
                lp.a_matrix_.index_ = self.indices
                lp.a_matrix_.value_ = vals[self.order]
                if self.hi.passModel(lp) == _hs.HighsStatus.kError:
                    return None
                if self.basis is not None:
                    try:
                        self.hi.setBasis(self.basis)
                    except Exception:
                        self.basis = None
                if self.hi.run() == _hs.HighsStatus.kError:
                    return None
                if self.hi.getModelStatus() != _hs.HighsModelStatus.kOptimal:
                    return None
                z = np.asarray(self.hi.getSolution().col_value, dtype=float)
                if z.size != self.nv or not np.all(np.isfinite(z)):
                    return None
                try:
                    self.basis = self.hi.getBasis()
                except Exception:
                    self.basis = None
                return z
            except Exception:
                self.hi = None                            # never retry a broken backend
        A = coo_matrix((vals, (self.rows, self.cols)), shape=(self.nrow, self.nv))
        try:
            res = linprog(self.c, A_ub=A, b_ub=self.b,
                          bounds=np.stack([lb, ub], 1), method="highs")
        except Exception:
            return None
        if not res.success or res.x is None:
            return None
        return np.asarray(res.x, dtype=float)


_LP_CACHE = {}


def _lp_for(n):
    lp = _LP_CACHE.get(n)
    if lp is None:
        if len(_LP_CACHE) > 24:
            _LP_CACHE.clear()
        lp = _Lp(n)
        _LP_CACHE[n] = lp
    return lp


# --------------------------------------------------------------------------- local solver (SCP)
SQ2 = math.sqrt(2.0)
RHO_MIN = 0.02         # radius trust region: r_i <= r_i + max(RHO_MIN, RHO_K * delta)
RHO_K = 0.0
DELTA_MAX = 0.05       # cap on the centre trust region (also caps the LP row-selection margin)
ROW_PAD = 0.02         # extra slack kept in the LP row set so a selection survives several solves
# Early abandonment of hopeless basins. Measured (artifacts/traj6.py, n in {37,45}, 6 s chains):
# a basin needs ~13 LPs to converge, but after SCREEN_K = 4 LPs EVERY basin that went on to
# improve the incumbent was already within 0.08% of it, while ~55% of the basins that never
# improved were already > 0.5% below. So the first 4 LPs are a near-lossless classifier and the
# remaining ~9 are wasted on a coin already landed.
SCREEN_K = 6           # LP solves after which a basin is judged
SCREEN_REL = 0.002     # ... and abandoned if its incumbent is this far (relative) below the reference
# WHICH reference. The screen was written when the chain was a strict hill-climb, so "the best" and
# "where the chain is standing" were the same point. Under threshold acceptance (see ACCEPT_T) they
# are not, and a floor pinned to the GLOBAL best silently caps how far the walk can descend: a
# proposal perturbed from a `cur` that has drifted more than SCREEN_REL below `best` climbs back to
# roughly cur's level, lands under the floor, and is abandoned at SCREEN_K LPs -- never finalized,
# never evaluated, never accepted. So the excursion depth was bounded by SCREEN_REL = 0.2% no matter
# how long a horizon STALL_MAX allowed. Anchoring the floor to the CURRENT point instead restores
# the screen's actual job (abort climbs worse than what the chain would accept anyway: 0.2% below
# cur is looser than ACCEPT_T = 0.06% below cur, so nothing acceptable is screened) and lets the
# walk go as deep as the horizon permits. See _search_n.
SCREEN_ON_CUR = True   # False = the old global-best anchor


def _greedy_radii(xy):
    """A feasible radius vector at these centres: r_i = min(wall_i, min_j d_ij / 2)."""
    n = len(xy)
    w = np.maximum(_wall(xy), 0.0)
    if n < 2:
        return w
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return np.minimum(w, 0.5 * D.min(1))


def _scp(xy, iters, delta0, deadline, stall_max=6, screen_floor=None, screen_k=SCREEN_K):
    """Climb from centres `xy` by the always-feasible linearised LP. Returns (xy, r, sum_r).

    `screen_floor` (an absolute sum-of-radii) abandons the climb once `screen_k` LPs have been
    solved and the incumbent is still below it -- see SCREEN_REL. The climb is monotone in
    best_s and the floor is fixed for the call, so testing once at screen_k is the whole test.

    Each LP carries a trust region on the centres (+-delta) AND on the radii (+rho); together they
    bound how far any constraint's slack can close in one step, so every pair with slack
    >= 2*delta + 2*rho and every wall row with slack >= delta + rho is provably non-binding and is
    dropped from the LP. That is an exact reformulation (same optimum over the box), and it is what
    makes the LP cheap: ~3n rows instead of n(n-1)/2 + 4n."""
    n = len(xy)
    lp = _lp_for(n)
    I, J = lp.I, lp.J
    cur = xy.copy()
    cur_r = _greedy_radii(cur)
    best_xy, best_r, best_s = cur, cur_r, float(cur_r.sum())
    delta = delta0
    stall = 0
    nlp = 0
    for _ in range(iters):
        if time.process_time() > deadline:
            break
        if nlp == screen_k and screen_floor is not None and best_s < screen_floor:
            break                      # hopeless basin: stop paying for the remaining ~9 LPs
        dist = _dists(cur, I, J)
        rho = max(RHO_MIN, RHO_K * delta)
        pslack = dist - (cur_r[I] + cur_r[J])
        wslack = np.concatenate([HI - cur[:, 0], cur[:, 0] - LO,
                                 HI - cur[:, 1], cur[:, 1] - LO]) - np.tile(cur_r, 4)
        # the centre trust region is a BOX (+-delta per COORDINATE), so a pair's linearised LHS
        # u_ij.(p_i - p_j) can move by 2*delta*(|ux|+|uy|) <= 2*SQ2*delta -- the sqrt(2) is what
        # makes the dropping exact rather than almost-exact (a uniform-random n=31 point in
        # --self-test falsified the 2*delta version). Wall rows are axis-aligned: delta is right.
        pcut, wcut = 2.0 * (SQ2 * delta + rho), delta + rho
        need_p, need_w = pslack < pcut, wslack < wcut
        # rebuild when a row we dropped could now bind, or when the kept set has gone stale-fat
        if (lp.pmask is None or np.any(need_p & ~lp.pmask) or np.any(need_w & ~lp.wmask)
                or lp.nrow > 2 * (int(need_p.sum()) + int(need_w.sum())) + 24):
            lp.set_rows(pslack < pcut + ROW_PAD, wslack < wcut + ROW_PAD)
        vals = lp.values(cur, dist)
        lb = np.concatenate([np.maximum(LO, cur[:, 0] - delta),
                             np.maximum(LO, cur[:, 1] - delta), np.zeros(n)])
        ub = np.concatenate([np.minimum(HI, cur[:, 0] + delta),
                             np.minimum(HI, cur[:, 1] + delta),
                             np.minimum(0.5, cur_r + rho)])
        z = lp.solve(vals, lb, ub)
        nlp += 1
        if z is None:
            delta *= 0.4
            if delta < 1e-11:
                break
            continue
        nxy = np.stack([z[:n], z[n:2 * n]], 1)
        nr = z[2 * n:]
        # the linearisation is an INNER restriction, so nr is feasible up to LP tolerance only;
        # rescale on any true violation so `best_r` is never worse than feasible.
        if len(I):
            dv = _dists(nxy, I, J)
            sc = float(np.max((nr[I] + nr[J]) / np.maximum(dv, 1e-15)))
            if sc > 1.0:
                nr = nr / sc
        s = nr.sum()
        if s > best_s + 1e-13:
            best_xy, best_r, best_s = nxy, nr, s
            delta = min(delta * 1.4, DELTA_MAX)
            stall = 0
        else:
            delta *= 0.5
            stall += 1
            if stall >= stall_max or delta < 1e-11:
                cur = nxy
                break
        cur = nxy
        cur_r = np.maximum(nr, 0.0)
    return best_xy, best_r, best_s


# --------------------------------------------------------------------------- exact radii + repair
def _radii_lp(xy, deadline):
    """Exact optimal radii for FIXED centres: max sum r s.t. r_i + r_j <= d_ij, 0 <= r_i <= wall_i."""
    n = len(xy)
    I, J = _pairs(n)
    dist = _dists(xy, I, J)
    w = np.maximum(_wall(xy), 0.0)
    if len(I) == 0:
        return w
    mask = _active_pairs(xy, I, J, dist, np.zeros(len(I), dtype=bool))
    Ia, Ja, da = I[mask], J[mask], dist[mask]
    m = len(Ia)
    rows = np.repeat(np.arange(m), 2)
    cols = np.stack([Ia, Ja], 1).ravel()
    A = coo_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n))
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=da,
                      bounds=np.stack([np.zeros(n), w], 1), method="highs")
    except Exception:
        return None
    if not res.success or res.x is None:
        return None
    return np.asarray(res.x)


def _make_feasible(xy, r):
    """Scale + shave + greedily re-grow so the packing is STRICTLY feasible under the scorer's rules."""
    n = len(xy)
    I, J = _pairs(n)
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    w = np.maximum(_wall(xy), 1e-12)
    if len(I):
        d = _dists(xy, I, J)
        s = max(float(np.max((r[I] + r[J]) / np.maximum(d, 1e-15))),
                float(np.max(r / w)))
    else:
        s = float(np.max(r / w))
    if s > 1.0:
        r = r / s
    r = np.minimum(r, w)
    # greedy re-grow: r_i <- min(wall_i, min_j d_ij - r_j); monotone non-decreasing in sum
    if len(I):
        D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        for _ in range(3):
            for i in range(n):
                r[i] = max(0.0, min(w[i], float(np.min(D[i] - r))))
    r = np.maximum(r - SHAVE, 1e-9)
    return r


def _finalize(xy, r, deadline):
    """Best strictly-feasible packing at these centres: try the exact radius LP, fall back to `r`."""
    best = None
    cands = []
    if time.process_time() < deadline:
        rl = _radii_lp(xy, deadline)
        if rl is not None:
            cands.append(rl)
    cands.append(np.asarray(r, dtype=float))
    for rc in cands:
        rf = _make_feasible(xy, rc)
        s = float(rf.sum())
        if best is None or s > best[1]:
            best = (rf, s)
    return best


# --------------------------------------------------------------------------- starts
def _lattice_start(n, rng):
    """A jittered near-hexagonal / rectangular lattice of n sites in the square."""
    ar = 0.866 if rng.rand() < 0.75 else 1.0        # hex rows are sqrt(3)/2 apart
    rows = int(round(np.sqrt(n / ar)))
    rows = max(1, rows + int(rng.randint(-1, 2)))
    per = int(np.ceil(n / rows))
    ii, jj = np.meshgrid(np.arange(rows), np.arange(per), indexing="ij")
    off = 0.5 * (ii % 2) if ar < 1.0 else 0.0
    px = (jj + 0.5 + off) / (per + (0.5 if ar < 1.0 else 0.0))
    py = (ii + 0.5) / rows
    pts = np.stack([px.ravel(), py.ravel()], 1) - 0.5
    idx = rng.permutation(len(pts))[:n]
    pts = pts[idx]
    if len(pts) < n:                                 # pathological rows/per rounding
        pts = np.concatenate([pts, rng.uniform(LO + 0.02, HI - 0.02, size=(n - len(pts), 2))])
    return np.clip(pts + rng.normal(0.0, 0.012, pts.shape), LO + 1e-3, HI - 1e-3)


def _read_pack(n):
    """Warm start from the committed census if present -- GUARDED; returns centres (n,2) or None."""
    p = os.path.join(PACK_DIR, "csqv%d.pck" % n)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            raw = [ln for ln in fh.read().splitlines() if ln.strip()]
        rows = [[float(v) for v in ln.split()] for ln in raw[2:]]
        rows = [t for t in rows if len(t) >= 3]
        if len(rows) != n:
            return None
        a = np.asarray(rows, dtype=float)[:, :2]
        if not np.all(np.isfinite(a)):
            return None
        return np.clip(a, LO, HI)
    except (OSError, ValueError, IndexError):
        return None


# --------------------------------------------------------------------------- structure transfer
USE_TRANSFER = True    # seed n from the committed packs of NEARBY sizes
GAP_TELEPORT_P = 0.0   # basin-hopping probability of "re-insert the smallest at the largest gaps"
# THRESHOLD ACCEPTANCE. The chain perturbs a "current" packing, not the global best: a candidate is
# accepted as the new current whenever it is no worse than ACCEPT_T (RELATIVE to sum_r) below the
# current one, so a run of small losses can walk the chain out of the basin its warm start sits in,
# while `best` still ratchets and is what gets evaluated/kept. ACCEPT_T = 0 reproduces the old
# strict hill-climb exactly.
ACCEPT_T = 0.0006
# How many CONSECUTIVE non-improving proposals the drifting chain is allowed before it is yanked
# back: at `STALL_MAX` the chain re-anchors `cur` to `best` and tries a fresh lattice/random start.
# This is the HORIZON of the threshold-acceptance walk -- a downhill excursion longer than this
# many steps can never happen, however permissive ACCEPT_T is, so for a size whose `best` never
# improves the chain only ever explores a ball of STALL_MAX drift steps around its warm start.
# MEASURED (iteration 11, artifacts/out_ab11_*.txt, solve()-level at CPU 30 s over 8 seeds): the
# old horizon of 14 was the binding limit on iteration 9's threshold-acceptance walk -- at 14 the
# three sub-cap sizes 35/39/41 did not move in ANY of 4 seeds (mean 5.9430 = the committed census
# exactly), and the mean rises monotonically with the horizon: 14 -> 5.9430, 35 -> 5.9523,
# 80 -> 5.9579, 120 -> 6.0324, 200 -> 6.0381, 500 -> 6.1051 (n=35 and n=41 reached the 7-digit cap
# for the first time). A restart also RE-ANCHORS cur to best, so it does not merely add a fresh
# basin -- it DELETES the accumulated excursion, which is the only mechanism that has moved these
# sizes. 500 is ~3 restarts in a 56 s single-target call, so the cold path keeps the escape hatch.
STALL_MAX = 500
# THE LEASH. ACCEPT_T is measured against `cur`, so the drift it allows COMPOUNDS: each accepted
# loss becomes the reference for the next one, and over a STALL_MAX-long horizon the chain can walk
# arbitrarily far below `best` (500 steps x 0.06% is a 26% descent) with no force pulling it back --
# nothing in the walk references `best` at all once iteration 12 pointed the basin screen at `cur`.
# ACCEPT_FLOOR bounds the TOTAL descent: a drifting step is refused if it would put `cur` more than
# this far (relative) below `best`. 0.0 = no leash (the iteration-12 behaviour exactly).
# MEASURED (iteration 13, artifacts/out_ab13_f*_{a,b}.txt, solve()-level at CPU 30 s over the 10
# visible n, 8 seeds in two blocks, all three arms run CONCURRENTLY so they share the same host
# load): pooled mean 0.0 -> 6.0813, 0.004 -> 6.1917, 0.015 -> 6.0827. 0.004 wins both blocks
# (6.2327 vs 6.2132; 6.1507 vs 5.9494). What matters more for the board than the mean is the tail:
# an iteration only improves the census if the ONE sanctioned draw improves some n, and the
# unleashed arm drew the committed 5.9430 EXACTLY on 3 of 8 seeds while 0.004 drew a null on 0 of 8
# and capped a live size (35/39/41 reaching 7 digits) on 5 seeds against the unleashed arm's 3.
# Too long a leash is as bad as none (0.015 is indistinguishable from 0.0): the mechanism is a
# bounded excursion, not a bigger one.
ACCEPT_FLOOR = 0.004


def _read_pack_xyr(m):
    """(m,3) array of x y r from the committed pack for m, or None -- GUARDED for ANY m."""
    p = os.path.join(PACK_DIR, "csqv%d.pck" % m)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            raw = [ln for ln in fh.read().splitlines() if ln.strip()]
        rows = [[float(v) for v in ln.split()] for ln in raw[2:]]
        rows = [t for t in rows if len(t) >= 3]
        if len(rows) != m:
            return None
        a = np.asarray(rows, dtype=float)[:, :3]
        if not np.all(np.isfinite(a)) or a.shape != (m, 3):
            return None
        a[:, :2] = np.clip(a[:, :2], LO, HI)
        a[:, 2] = np.maximum(a[:, 2], 0.0)
        return a
    except (OSError, ValueError, IndexError):
        return None


def _largest_gaps(xy, r, k, rng, grid=96, jitter=0.0):
    """k centres at the largest empty discs of the packing (xy, r): greedy
    argmax_p min(wall(p), min_j ||p-p_j|| - r_j), grid scan then a shrinking random refine.
    Each accepted point is added as a circle of its own clearance so the next one goes elsewhere."""
    cx = np.asarray(xy, dtype=float).reshape(-1, 2).copy()
    cr = np.maximum(np.asarray(r, dtype=float).ravel(), 0.0).copy()
    g = (np.arange(grid) + 0.5) / grid - 0.5
    GX, GY = np.meshgrid(g, g, indexing="ij")
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    wallP = np.minimum.reduce([P[:, 0] - LO, HI - P[:, 0], P[:, 1] - LO, HI - P[:, 1]])
    out = []
    for _ in range(k):
        if len(cx):
            d = np.sqrt(((P[:, None, :] - cx[None, :, :]) ** 2).sum(-1)) - cr[None, :]
            f = np.minimum(wallP, d.min(1))
        else:
            f = wallP
        i = int(np.argmax(f))
        p = P[i].copy()
        rad = float(f[i])
        step = 1.0 / grid
        for _ in range(12):
            q = np.clip(p + rng.normal(0.0, step, (8, 2)), LO, HI)
            wq = np.minimum.reduce([q[:, 0] - LO, HI - q[:, 0], q[:, 1] - LO, HI - q[:, 1]])
            if len(cx):
                dq = np.sqrt(((q[:, None, :] - cx[None, :, :]) ** 2).sum(-1)) - cr[None, :]
                fq = np.minimum(wq, dq.min(1))
            else:
                fq = wq
            j = int(np.argmax(fq))
            if fq[j] > rad:
                p, rad = q[j], float(fq[j])
            step *= 0.7
        if jitter > 0.0:
            p = np.clip(p + rng.normal(0.0, jitter, 2), LO + 1e-4, HI - 1e-4)
        out.append(p)
        cx = np.concatenate([cx, p[None, :]], 0)
        cr = np.concatenate([cr, [max(rad, 1e-4)]])
    return np.asarray(out, dtype=float).reshape(k, 2)


def _transfer_starts(n, rng, limit=8):
    """Starts for n derived from the committed packs of NEARBY sizes.

    A csqv optimum at m is a strong basin for n = m +- a few: delete the smallest circles
    (m > n) or drop new circles into the largest empty gaps (m < n) and re-run the SCP. Nearby
    sizes already at the record are exactly the seeds worth transferring, so this compounds --
    every n that reaches the cap becomes a seed for its neighbours. Fully guarded: an n with no
    committed neighbour simply gets no transfer starts and the cold search is unaffected."""
    if not USE_TRANSFER:
        return []
    cands = []
    for dm in (1, 2, 3, 4, 5, 6):
        for m in (n - dm, n + dm):
            if m < 1 or m == n:
                continue
            a = _read_pack_xyr(m)
            if a is None:
                continue
            d = _banked_digits(m)
            cands.append((dm - 0.25 * (0.0 if d is None else d), m, a))
    cands.sort(key=lambda t: t[0])
    seeds = []
    for _, m, a in cands[:4]:
        xy, r = a[:, :2], a[:, 2]
        for rep in (0, 1):
            if m > n:
                k = m - n
                if rep == 0:
                    who = np.argsort(r)[:k]
                else:
                    pool = np.argsort(r)[:min(m, k + 3)]
                    who = rng.permutation(pool)[:k]
                s = np.delete(xy, who, axis=0).copy()
            else:
                k = n - m
                s = np.concatenate([xy, _largest_gaps(xy, r, k, rng,
                                                      jitter=(0.0 if rep == 0 else 0.01))], 0)
            if len(s) != n:
                continue
            if rep:
                s = s + rng.normal(0.0, 0.004, s.shape)
            seeds.append(np.clip(s, LO + 1e-4, HI - 1e-4))
            if len(seeds) >= limit:
                return seeds
    return seeds


# --------------------------------------------------------------------------- per-n search
def _search_n(n, evaluate, meter, rng, deadline, state=None):
    """One basin-hopping chain for n, run until `deadline` (process CPU seconds).

    `state` RESUMES a previous chain for the same n: the round-based scheduler in solve() revisits
    an n several times, and re-seeding from the COMMITTED pack each visit would throw away
    everything the earlier visits found (the census on disk is not updated mid-solve). Returns the
    state to pass back in; `state=None` is a cold visit and does the full seeding phase."""
    best_xy = None if state is None else state["xy"]
    best_r = None if state is None else state["r"]
    best_s = -np.inf if state is None else state["s"]
    best_raw = -np.inf if state is None else state["raw"]   # best PRE-finalize SCP sum ever seen
    # the raw level OF the best point, and of the accepted point the chain is standing on: these are
    # the screen's reference (see SCREEN_ON_CUR), distinct from best_raw, which is a high-water mark
    # over all climbs including ones that finalized worse or infeasible.
    bp_raw = -np.inf if state is None else state.get("braw", best_raw)
    # the ACCEPTED point the chain perturbs (threshold acceptance); never below `cur` by more than
    # ACCEPT_T relative per step, and identical to best when ACCEPT_T == 0.
    cur_xy = best_xy if state is None else state.get("cxy", best_xy)
    cur_r = best_r if state is None else state.get("cr", best_r)
    cur_s = best_s if state is None else state.get("cs", best_s)
    cur_raw = bp_raw if state is None else state.get("craw", bp_raw)

    def try_start(xy, delta0, iters=70, screen=False):
        nonlocal best_xy, best_r, best_s, best_raw, bp_raw, cur_xy, cur_r, cur_s, cur_raw
        if meter.left() <= 0 or time.process_time() > deadline:
            return False
        # the basin screen's reference level: where the chain is standing, not its high-water mark.
        ref = cur_raw if (SCREEN_ON_CUR and cur_raw > 0.0) else best_raw
        floor = None
        if screen and ref > 0.0:
            floor = ref * (1.0 - SCREEN_REL)
        sxy, sr, raw = _scp(np.clip(xy, LO, HI), iters, delta0, deadline, screen_floor=floor)
        if raw > best_raw:
            best_raw = raw
        if floor is not None and raw < floor:
            return False               # abandoned: skip the radius LP, the repair and the eval
        fin = _finalize(sxy, sr, deadline)
        if fin is None:
            return False
        rf, s = fin
        pack = np.concatenate([sxy, rf[:, None]], 1)
        if meter.left() <= 0:
            return False
        feas, sv = evaluate(n, pack)
        feas = bool(np.all(feas))
        sv = float(np.ravel(sv)[0]) if np.ndim(sv) else float(sv)
        if not feas:
            return False
        if sv > best_s:
            best_xy, best_r, best_s = sxy.copy(), rf.copy(), sv
            bp_raw = raw
            cur_xy, cur_r, cur_s, cur_raw = best_xy, best_r, best_s, raw
            return True
        # threshold acceptance: a small loss still moves the chain, so it can leave this basin
        if ACCEPT_T > 0.0 and cur_s > -np.inf and sv >= cur_s - ACCEPT_T * abs(cur_s):
            # ... but on a leash: the compounding per-step allowance is bounded relative to `best`,
            # which is the only quantity that says how far the walk has actually wandered off.
            if not (ACCEPT_FLOOR > 0.0 and best_s > -np.inf
                    and sv < best_s - ACCEPT_FLOOR * abs(best_s)):
                cur_xy, cur_r, cur_s, cur_raw = sxy.copy(), rf.copy(), sv, raw
        return False

    if state is None:                  # cold visit: warm start, transfer seeds, first lattice
        warm = _read_pack(n)
        if warm is not None:
            try_start(warm, 0.02)
        for s0 in _transfer_starts(n, rng):
            if time.process_time() > deadline or meter.left() <= 0:
                break
            try_start(s0, 0.05)
        try_start(_lattice_start(n, rng), 0.10)

    stall = 0
    while time.process_time() < deadline and meter.left() > 0:
        if cur_xy is None or stall >= STALL_MAX:
            xy = _lattice_start(n, rng) if rng.rand() < 0.7 else \
                rng.uniform(LO + 0.03, HI - 0.03, size=(n, 2))
            d0 = 0.10
            stall = 0
            # a restart re-anchors the chain -- and with it the screen's reference level
            cur_xy, cur_r, cur_s, cur_raw = best_xy, best_r, best_s, bp_raw
        else:
            xy = cur_xy.copy()
            u = rng.rand()
            if u < 0.45:                              # relocate the smallest circles
                k = 1 + int(rng.randint(0, min(3, n)))
                who = np.argsort(cur_r)[:k]
                if rng.rand() < GAP_TELEPORT_P:       # ... into the largest empty gaps
                    keep = np.setdiff1d(np.arange(n), who)
                    xy[who] = _largest_gaps(xy[keep], cur_r[keep], k, rng, grid=64,
                                            jitter=0.01)
                else:
                    xy[who] = rng.uniform(LO + 0.02, HI - 0.02, size=(k, 2))
                d0 = 0.06
            elif u < 0.80:                            # rattle every centre
                xy = xy + rng.normal(0.0, 0.006 + 0.03 * rng.rand(), xy.shape)
                d0 = 0.04
            else:                                     # teleport random circles
                k = 1 + int(rng.randint(0, min(3, n)))
                who = rng.permutation(n)[:k]
                xy[who] = rng.uniform(LO + 0.02, HI - 0.02, size=(k, 2))
                d0 = 0.06
            xy = np.clip(xy, LO + 1e-4, HI - 1e-4)
        stall = 0 if try_start(xy, d0, screen=True) else stall + 1
    return {"xy": best_xy, "r": best_r, "s": best_s, "raw": best_raw, "braw": bp_raw,
            "cxy": cur_xy, "cr": cur_r, "cs": cur_s, "craw": cur_raw}


# --------------------------------------------------------------------------- entry point
_REC_CACHE = {}


def _record(n):
    """record[n] from bench/records.json, or None when it is missing/unreadable/non-positive."""
    if n not in _REC_CACHE:
        v = None
        try:
            with open(os.path.join("bench", "records.json")) as fh:
                v = float(json.load(fh)["records"][str(n)])
            if not (v > 0.0):
                v = None
        except (OSError, ValueError, KeyError, TypeError):
            v = None
        _REC_CACHE[n] = v
    return _REC_CACHE[n]


def _digits_of(n, s):
    """The DIGITS the scorer would award n for sum_r = s; 0.0 when the record is unknown."""
    rec = _record(n)
    if rec is None or not np.isfinite(s) or s <= 0.0:
        return 0.0
    gap = max(0.0, (rec - float(s)) / rec)
    return 7.0 if gap <= 0.0 else max(0.0, min(7.0, -math.log10(gap)))


def _banked_digits(n):
    """Digits ALREADY banked for n by the committed census, or None if unknown.

    Fully guarded: any missing/oversized/unparseable file, or an n outside records.json, returns
    None so the caller treats that n as maximum headroom and cold-starts normally."""
    try:
        rec = _record(n)
        path = os.path.join(PACK_DIR, "csqv%d.pck" % n)
        if rec is None or not os.path.exists(path):
            return None
        with open(path) as fh:
            rows = [ln.split() for ln in fh.read().splitlines()[2:] if ln.strip()]
        if len(rows) != n:
            return None
        tot = math.fsum(float(t[2]) for t in rows)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    gap = max(0.0, (rec - tot) / rec)
    return 7.0 if gap <= 0.0 else min(7.0, -math.log10(gap))


# MEASURED AND REJECTED AT 3 LIVE SIZES (iteration 10), THEN MEASURED AND ACCEPTED AT 2
# (iteration 15). Dropping the capped arms is only half a change: it also shrinks the arm count,
# and _rounds halves down to ARM_FLOOR = 3, so at 3 live arms SKIP_CAPPED = True collapsed the
# schedule to ONE round and lost (base 5.9944 vs 5.9132 over 5 seeds, artifacts/out_ab10_s*.txt)
# -- the live arms were paying for their extra CPU with the probe-then-resume shape, and it is the
# resumed chain, not the seeding phase, that moves these sizes. Iteration 10 also tried
# MIN_ROUNDS = 2/3 to buy the shape back and still lost (6.0133 vs 5.9181 / 5.9214,
# artifacts/out_ab10b_*.txt) -- at 3 live arms each probe is ~1/9 of 30 s, too short to seed.
# With 8 of 10 sizes now capped there are only 2 live arms, each probe is ~1/6 of 30 s, and the
# pairing WINS: over 12 seeds in two concurrently-run blocks (artifacts/out_ab15{,b}_*.txt) base
# 6.4469 vs 6.5507, with 3/12 null draws against 7/12 and n=35 capped on 7/12 seeds against 4/12.
# The two constants are one change: SKIP_CAPPED = True alone scored 6.4192 (block a, 4/6 null).
SKIP_CAPPED = True     # True = drop sizes whose COMMITTED pack is already at the digit cap


def _live(ts):
    """The subset of ts whose committed pack still has digit headroom.

    An n already at the 7-digit cap cannot raise the SCORE however much CPU it is given, so it is
    not a candidate at all. Fully guarded: `_banked_digits` returns None for an n with no pack or
    no record (a fresh n, the offline held-out sizes), and None means "maximum headroom", so those
    are always live. If EVERY target is capped the filter is voided and all of ts is returned --
    `solve()` must never be handed an empty arm list (and a capped n is still the right thing to
    work on when there is nothing else to do). SKIP_CAPPED = False reproduces the old behaviour."""
    if not SKIP_CAPPED:
        return list(ts)
    live = [n for n in ts
            if (lambda d: d is None or d < DIGITS_CAP - 1e-9)(_banked_digits(n))]
    return live if live else list(ts)


def _focus(ts, rng):
    """Pick the sizes this call should actually spend CPU on: the live ones, at most FOCUS_K.

    Measured: one long basin-hopping chain beats several short ones. At ~2.9 s per n (29 s spread
    over 10 visible n) n=29 reached 2.95 digits; the SAME chain given 9.7 s found the GLOBAL basin
    and hit the 7-digit cap. Score is the mean over ALL 10 n and the census is append-or-improve,
    so an n left alone simply keeps its committed packing -- concentrating buys those jackpots for
    free.

    Sizes already AT the cap are removed by `_live` BEFORE the top-FOCUS_K cut, not merely ranked
    last by headroom: once more than (10 - FOCUS_K) sizes are capped, a pure ranking still has to
    fill the FOCUS_K slots and starts admitting zero-headroom arms. With 7 of 10 capped that was
    3 of the 6 arms -- half of round one's CPU spent where the SCORE cannot move, and the three
    live sizes each losing their seeding phase to a 1.8 s probe. The small random tiebreak makes
    the selection rotate across iterations once headrooms even out."""
    cand = _live(ts)
    if len(cand) <= FOCUS_K:
        return sorted(cand)
    key = []
    for n in cand:
        d = _banked_digits(n)
        key.append((7.0 if d is None else max(0.0, 7.0 - d)) + 0.15 * float(rng.rand()))
    order = sorted(range(len(cand)), key=lambda i: -key[i])[:FOCUS_K]
    return sorted(cand[i] for i in order)


def _rounds(k):
    """SUCCESSIVE HALVING over k candidate sizes: [(arms kept, share of the CPU), ...].

    Quality here is a lottery over basins and CPU is the binding meter, so the allocation across
    the visible n is itself a bandit: a short probe on every candidate is cheap, and the sizes that
    actually MOVE in it are where the long chain belongs. Arms halve each round while the round's
    CPU share grows by HALVING_W, so per-arm time roughly doubles and the survivor gets the
    largest single uninterrupted chain -- which is what wins jackpots (iteration 4). Because
    _search_n resumes from its own state, nothing a dropped or revisited arm found is lost.
    k == 1 degenerates to ONE call with the whole allowance, so the offline held-out re-run
    (one size per call) is bit-for-bit the old behaviour."""
    seq = []
    a = int(k)
    floor = max(1, min(int(ARM_FLOOR), a))
    while a > floor:
        seq.append(a)
        a = max(floor, a // 2)
    seq.append(floor)
    # A call with k <= ARM_FLOOR halves zero times, so it would be ONE round: every arm gets a
    # single uninterrupted visit. MIN_ROUNDS forces at least that many rounds by repeating the
    # arm count, which keeps every arm alive but splits its allowance into a short probe and a
    # longer resumed chain. k == 1 is exempt: the offline held-out re-run must stay one full call.
    while int(k) > 1 and len(seq) < int(MIN_ROUNDS):
        seq.append(seq[-1])
    w = [HALVING_W ** i for i in range(len(seq))]
    tot = math.fsum(w)
    return [(seq[i], w[i] / tot) for i in range(len(seq))]


def solve(evaluate, meter, rng, targets):
    t0 = time.process_time()
    ts = sorted(set(int(t) for t in targets))
    if not ts:
        return
    total = CPU_ONE_TARGET if len(ts) == 1 else CPU_MANY
    ts = _focus(ts, rng)
    state = {}
    arms = list(ts)
    cum = 0.0
    for keep, frac in _rounds(len(ts)):
        arms = arms[:keep]
        if not arms or meter.left() <= 0:
            break
        # harder (larger) n get proportionally more of the round; survives len(arms) == 1
        wts = np.asarray([float(n) for n in arms])
        wts = wts / wts.sum()
        for n, w in zip(arms, wts):
            if meter.left() <= 0:
                break
            cum += total * frac * float(w)
            state[n] = _search_n(n, evaluate, meter, rng, t0 + cum, state.get(n))

        # Rank the survivors by MEASURED movement (digits gained over the committed census) and
        # retire any size that has reached the 7-digit cap: it has no headroom left, so further
        # CPU on it cannot move the SCORE however well it just performed.
        key = {}
        for n in arms:
            d = _digits_of(n, state[n]["s"])
            key[n] = None if d >= DIGITS_CAP - 1e-9 else \
                d - (_banked_digits(n) or 0.0) + 0.02 * float(rng.rand())
        arms = sorted([n for n in arms if key[n] is not None], key=lambda m: -key[m])


# --------------------------------------------------------------------------- self-test
def _self_test():
    import math
    try:                                   # what the committed census looks like RIGHT NOW
        with open(os.path.join("bench", "records.json")) as _fh:
            _RECORDS_FOR_TEST = {int(k): v for k, v in json.load(_fh)["records"].items()}
    except (OSError, ValueError, KeyError, TypeError):
        _RECORDS_FOR_TEST = {}
    _CAPPED_FOR_TEST = [n for n in sorted(_RECORDS_FOR_TEST)
                        if (_banked_digits(n) or 0.0) >= DIGITS_CAP - 1e-9]
    budget = [4000]

    class M:
        budget = 4000

        @property
        def used(self):
            return M.budget - budget[0]

        def left(self):
            return budget[0]

        def tick(self, k=1):
            g = max(0, min(int(k), budget[0]))
            budget[0] -= g
            return g

    seen = {}

    def ev(n, pack):
        p = np.asarray(pack, dtype=float)
        single = (p.ndim == 2)
        if single:
            p = p[None]
        if p.ndim != 3 or p.shape[2] != 3 or p.shape[1] != n:
            raise ValueError("bad shape")
        feas = np.zeros(len(p), dtype=bool)
        sums = np.full(len(p), -np.inf)
        for b in range(len(p)):
            if budget[0] <= 0:
                continue
            budget[0] -= 1
            c = p[b]
            ok = bool(np.all(np.isfinite(c))) and float(c[:, 2].min()) > 0.0
            if ok:
                w = min(min(x - c[i, 2] - LO, HI - x - c[i, 2], y - c[i, 2] - LO, HI - y - c[i, 2])
                        for i, (x, y) in enumerate(c[:, :2]))
                pr = math.inf
                for i in range(n):
                    for j in range(i + 1, n):
                        pr = min(pr, math.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
                                 - (c[i, 2] + c[j, 2]))
                ok = w >= -1e-9 and (n < 2 or pr >= -1e-9)
            if ok:
                feas[b] = True
                sums[b] = math.fsum(c[:, 2])
                if sums[b] > seen.get(n, -np.inf):
                    seen[n] = sums[b]
        if single:
            return bool(feas[0]), float(sums[0])
        return feas, sums

    # CONTRACT: solve() must work when called TWICE in the SAME process, and with one target.
    global CPU_ONE_TARGET, CPU_MANY
    CPU_ONE_TARGET = CPU_MANY = 3.0
    rng = np.random.RandomState(0)

    def run(ts):                       # the driver builds a FRESH meter per invocation; mirror that
        budget[0] = M.budget
        solve(ev, M(), rng, ts)

    run([12])
    first = seen.get(12)
    assert first is not None and first > 0, "first call produced no feasible packing"
    run([12])
    assert seen.get(12, -1) >= first, "second call in the same process produced nothing"
    print("self-test: two calls in one process OK, n=12 best sum_r = %.6f" % seen[12])

    # multi-target call, including an n with no committed pack
    run([5, 7])
    assert seen.get(5, 0) > 0 and seen.get(7, 0) > 0, "multi-target call missed an n"
    # n=1: the single circle must fill the square
    run([1])
    assert abs(seen.get(1, 0) - 0.5) < 1e-6, "n=1 should reach r=0.5, got %r" % (seen.get(1),)
    # n=2: known optimum sum_r = 2 - sqrt(2) ~= 0.585786
    run([2])
    assert seen.get(2, 0) > 0.5857, "n=2 below the known optimum: %r" % (seen.get(2),)
    print("self-test: n in {1,2,5,7} OK  (n=1 %.6f, n=2 %.6f)" % (seen[1], seen[2]))

    # warm-start reader must not raise on a missing pack
    assert _read_pack(10 ** 6) is None
    # structure transfer must be a TOTAL function: an n with no committed neighbour gets no seeds,
    # and every seed it does return must be a valid (n,2) start inside the square.
    assert _read_pack_xyr(10 ** 6) is None
    assert _transfer_starts(10 ** 6, rng) == []
    for tn in (31, 36, 46):
        for s0 in _transfer_starts(tn, rng):
            assert s0.shape == (tn, 2) and np.all(np.isfinite(s0))
            assert s0.min() >= LO and s0.max() <= HI
    # largest-empty-disc insertion: the returned centres must be inside the square and, for a
    # packing with room, strictly clear of the existing circles.
    gxy = np.array([[-0.25, -0.25], [0.25, 0.25]])
    gr = np.array([0.1, 0.1])
    gp = _largest_gaps(gxy, gr, 3, rng, grid=48)
    assert gp.shape == (3, 2) and gp.min() >= LO and gp.max() <= HI
    assert np.min(np.sqrt(((gp[:, None, :] - gxy[None, :, :]) ** 2).sum(-1)) - gr[None, :]) > 0.0
    assert _largest_gaps(np.zeros((0, 2)), np.zeros(0), 2, rng, grid=32).shape == (2, 2)
    print("self-test: structure transfer total on unknown n; gap insertion clear of all circles")
    # focus/headroom readers must be total functions on unknown n and must never drop a lone target
    assert _banked_digits(10 ** 6) is None
    assert _focus([13], rng) == [13]
    assert _focus([13, 17], rng) == [13, 17]
    big = list(range(20, 40))
    f = _focus(big, rng)
    assert len(f) == FOCUS_K and set(f) <= set(big) and f == sorted(f), f
    d27 = _banked_digits(27)
    assert d27 is None or 0.0 <= d27 <= 7.0
    # CAPPED FILTER: a size at the digit cap is not a candidate at all, but the filter must never
    # empty the arm list, and must never drop an n it knows nothing about (held-out sizes).
    _live_visible = _live(sorted(int(k) for k in (_RECORDS_FOR_TEST or {27: 1})))
    assert _live_visible, "the capped filter emptied the visible arm list"
    assert _live(_CAPPED_FOR_TEST) == _CAPPED_FOR_TEST or not _CAPPED_FOR_TEST, \
        "an all-capped target list must be returned intact, not emptied"
    assert set(_live([10 ** 6, 10 ** 6 + 1])) == {10 ** 6, 10 ** 6 + 1}, \
        "unknown n must count as live (max headroom), not be filtered out"
    assert _live([]) == []
    for _n in _CAPPED_FOR_TEST:                 # each capped n is dropped when others are live
        if _live_visible and _n not in _live_visible:
            assert _n not in _focus(sorted(set(_live_visible) | {_n}), rng), _n
    global SKIP_CAPPED
    _saved_skip = SKIP_CAPPED
    try:                                        # SKIP_CAPPED = False is the old pure ranking
        SKIP_CAPPED = False
        assert _live([27, 10 ** 6]) == [27, 10 ** 6]
    finally:
        SKIP_CAPPED = _saved_skip
    # digit accounting used by the scheduler's ranking
    assert _digits_of(10 ** 6, 1.0) == 0.0 and _digits_of(27, -np.inf) == 0.0
    r27 = _record(27)
    assert r27 is not None and r27 > 0.0
    assert _digits_of(27, r27) == 7.0 and _digits_of(27, 2.0 * r27) == 7.0
    assert 0.9 < _digits_of(27, r27 * (1 - 0.1)) < 1.1
    assert _digits_of(27, r27 * 0.99) < _digits_of(27, r27 * 0.999)
    # SUCCESSIVE-HALVING SCHEDULE: shares sum to 1, arms shrink to a single survivor, and a lone
    # target degenerates to ONE full-allowance call (the offline held-out re-run's shape).
    global ARM_FLOOR, MIN_ROUNDS
    _saved_floor, _saved_minr = ARM_FLOOR, MIN_ROUNDS
    try:
        for _mr in (1, 2, 3, MIN_ROUNDS):
            MIN_ROUNDS = _mr
            for fl in (1, 2, 3, 4, ARM_FLOOR):
                ARM_FLOOR = fl
                # MIN_ROUNDS must actually buy rounds when halving alone would not
                assert len(_rounds(4)) >= min(_mr, 4), (fl, _mr, _rounds(4))
                # a LONE target degenerates to one full-allowance call for EVERY floor and every
                # MIN_ROUNDS -- that is the offline held-out re-run's shape and must not depend
                # on either knob.
                assert _rounds(1) == [(1, 1.0)], (fl, _mr)
                for k in (1, 2, 3, 5, 6, 10):
                    rd = _rounds(k)
                    assert abs(math.fsum(f for _, f in rd) - 1.0) < 1e-12, (fl, _mr, rd)
                    assert rd[0][0] == k, (fl, _mr, rd)
                    assert rd[-1][0] == min(fl, k), (fl, _mr, rd)
                    assert all(rd[i][0] >= rd[i + 1][0]
                               for i in range(len(rd) - 1)), (fl, _mr, rd)
                    assert all(rd[i][1] < rd[i + 1][1]
                               for i in range(len(rd) - 1)) or len(rd) == 1, (fl, _mr, rd)
                    # per-arm time must GROW round over round (that is the point of halving)
                    per = [f / a for a, f in rd]
                    assert all(per[i] < per[i + 1]
                               for i in range(len(per) - 1)) or len(rd) == 1, (fl, _mr, rd)
    finally:
        ARM_FLOOR, MIN_ROUNDS = _saved_floor, _saved_minr
    # RESUME: a second visit to the same n must continue from the first visit's incumbent, never
    # restart from the committed pack -- otherwise the scheduler's rounds throw away their work.
    budget[0] = M.budget
    m1 = M()
    st = _search_n(14, ev, m1, np.random.RandomState(7), time.process_time() + 1.0)
    assert st["xy"] is not None and st["xy"].shape == (14, 2) and st["r"].shape == (14,)
    s1 = st["s"]
    assert s1 > 0.0
    st2 = _search_n(14, ev, m1, np.random.RandomState(8), time.process_time() + 1.0, st)
    assert st2["s"] >= s1 - 1e-15, (s1, st2["s"])
    assert st2["raw"] >= st["raw"] - 1e-15
    # a resumed visit must not re-run the seeding phase: with a deadline already past, it returns
    # the incumbent untouched
    st3 = _search_n(14, ev, m1, np.random.RandomState(9), time.process_time() - 1.0, st2)
    assert st3["s"] == st2["s"] and np.array_equal(st3["xy"], st2["xy"])
    print("self-test: rounds/digit accounting OK; _search_n resume keeps its incumbent (n=14 "
          "%.6f -> %.6f)" % (s1, st2["s"]))
    # _make_feasible must produce a strictly feasible packing from deliberately oversized radii
    xy = rng.uniform(LO + 0.1, HI - 0.1, size=(9, 2))
    r = _make_feasible(xy, np.full(9, 5.0))
    I, J = _pairs(9)
    assert r.min() > 0
    assert (_dists(xy, I, J) - (r[I] + r[J])).min() >= 0
    assert (_wall(xy) - r).min() >= 0
    # ROW FILTERING MUST BE EXACT: over the same trust-region box, the LP restricted to the rows
    # that can bind must have the SAME optimum as the LP that keeps every row.
    for nt in (7, 18, 31):
        c = rng.uniform(LO + 0.06, HI - 0.06, size=(nt, 2))
        cr = _greedy_radii(c)
        I, J = _pairs(nt)
        dist = _dists(c, I, J)
        delta, rho = 0.03, 0.02
        lb = np.concatenate([np.maximum(LO, c[:, 0] - delta),
                             np.maximum(LO, c[:, 1] - delta), np.zeros(nt)])
        ub = np.concatenate([np.minimum(HI, c[:, 0] + delta),
                             np.minimum(HI, c[:, 1] + delta), np.minimum(0.5, cr + rho)])
        pslack = dist - (cr[I] + cr[J])
        wslack = np.concatenate([HI - c[:, 0], c[:, 0] - LO,
                                 HI - c[:, 1], c[:, 1] - LO]) - np.tile(cr, 4)
        lp = _Lp(nt)
        zfull = lp.solve(lp.values(c, dist), lb, ub)
        nfull = lp.nrow
        lp.set_rows(pslack < 2.0 * (SQ2 * delta + rho), wslack < delta + rho)
        zcut = lp.solve(lp.values(c, dist), lb, ub)
        assert zfull is not None and zcut is not None
        assert lp.nrow <= nfull
        assert abs(zfull[2 * nt:].sum() - zcut[2 * nt:].sum()) < 1e-9, (nt, lp.nrow, nfull)
    print("self-test: row-filtered LP matches the full LP objective (n=7,18,31)")
    # BASIN SCREEN: an unreachable floor must abandon the climb early (few LPs) yet still return a
    # usable (xy, r, sum); floor=None must be identical to the pre-screen behaviour; and a floor
    # that is trivially reachable must not cut the climb short.
    nlp = [0]
    _orig_solve = _Lp.solve
    def _counted(self, vals, lb, ub):
        nlp[0] += 1
        return _orig_solve(self, vals, lb, ub)
    _Lp.solve = _counted
    try:
        for nt in (12, 24):
            c = rng.uniform(LO + 0.06, HI - 0.06, size=(nt, 2))
            dl = time.process_time() + 5.0
            nlp[0] = 0
            xf, rf, sf = _scp(c.copy(), 70, 0.10, dl)
            n_free = nlp[0]
            nlp[0] = 0
            xc, rc, sc = _scp(c.copy(), 70, 0.10, dl, screen_floor=1e9)
            n_cut = nlp[0]
            assert n_cut <= SCREEN_K + 1, (nt, n_cut, SCREEN_K)
            assert n_cut < n_free, (nt, n_cut, n_free)
            assert xc.shape == (nt, 2) and rc.shape == (nt,) and np.all(np.isfinite(rc))
            assert sc > 0.0 and abs(float(rc.sum()) - sc) < 1e-12
            nlp[0] = 0
            xr, rr, sr = _scp(c.copy(), 70, 0.10, dl, screen_floor=-1e9)
            assert nlp[0] == n_free and abs(sr - sf) < 1e-12, (nt, nlp[0], n_free)
    finally:
        _Lp.solve = _orig_solve
    print("self-test: basin screen aborts on an unreachable floor (<=%d LPs) and is inert otherwise"
          % (SCREEN_K + 1))
    # SCREEN ANCHOR: under the OLD global-best anchor the floor handed to _scp is monotone
    # non-decreasing (best_raw is a high-water mark), so it can never follow a drifting chain down
    # and it CAPS the excursion depth at SCREEN_REL. Under the cur anchor the floor must be able to
    # dip. Monotonicity under False is structural; the dip under True is made deterministic by
    # accepting almost everything, so the chain is certain to descend.
    global ACCEPT_T
    _orig_accept, _orig_anchor = ACCEPT_T, SCREEN_ON_CUR
    _orig_scp = globals()["_scp"]
    seen_floors = []

    def _spy(xy, iters, delta0, deadline, stall_max=6, screen_floor=None, screen_k=SCREEN_K):
        if screen_floor is not None:
            seen_floors.append(float(screen_floor))
        return _orig_scp(xy, iters, delta0, deadline, stall_max, screen_floor, screen_k)

    def _dips(fl):
        run, k = -np.inf, 0
        for f in fl:
            if f < run - 1e-12:
                k += 1
            run = max(run, f)
        return k

    try:
        globals()["_scp"] = _spy
        ACCEPT_T = 0.05                # accept nearly everything, so the chain is sure to drift
        out = {}
        for anchor in (False, True):
            globals()["SCREEN_ON_CUR"] = anchor
            del seen_floors[:]
            budget[0] = M.budget
            st = _search_n(22, ev, M(), np.random.RandomState(7), time.process_time() + 2.0)
            out[anchor] = (list(seen_floors), st)
            assert "braw" in st and "craw" in st, st.keys()
            # the resume state must round-trip both new fields
            budget[0] = M.budget
            st2 = _search_n(22, ev, M(), np.random.RandomState(8),
                            time.process_time() + 0.3, st)
            assert st2["s"] >= st["s"] - 1e-12, (anchor, st2["s"], st["s"])
            assert np.isfinite(st2["braw"]) and np.isfinite(st2["craw"])
        f_best, f_cur = out[False][0], out[True][0]
        assert len(f_best) > 20 and len(f_cur) > 20, (len(f_best), len(f_cur))
        assert _dips(f_best) == 0, "global-best anchor floor must be monotone, %d dips" % \
            _dips(f_best)
        assert _dips(f_cur) > 0, "cur anchor floor never followed the chain down"
    finally:
        globals()["_scp"] = _orig_scp
        ACCEPT_T, globals()["SCREEN_ON_CUR"] = _orig_accept, _orig_anchor
    print("self-test: screen floor is monotone under the best anchor (0/%d dips) and tracks the "
          "chain under the cur anchor (%d/%d dips); braw/craw round-trip through resume"
          % (len(f_best), _dips(f_cur), len(f_cur)))
    # ---- THE LEASH bounds the chain's TOTAL descent below `best`. Contract: with ACCEPT_T large
    # (so nearly every proposal is accepted and the chain is certain to drift), the accepted point
    # `cs` must stay within ACCEPT_FLOOR (relative) of `best` on EVERY call and across a resume;
    # and -- so the test cannot pass vacuously -- the SAME walk with the leash off must breach that
    # band on at least one of the seeds. Both are checked over 4 seeds, since a walk that happens
    # to end on an improvement has cs == s and shows no drift on its own.
    global ACCEPT_FLOOR
    _o_accept, _o_floor = ACCEPT_T, ACCEPT_FLOOR
    try:
        ACCEPT_T = 0.05
        FLOOR = 0.001
        drops = {0.0: [], FLOOR: []}
        for fl in (0.0, FLOOR):
            ACCEPT_FLOOR = fl
            for sd in (11, 12, 13, 14):
                budget[0] = M.budget
                st = _search_n(22, ev, M(), np.random.RandomState(sd),
                               time.process_time() + 0.8)
                budget[0] = M.budget
                st2 = _search_n(22, ev, M(), np.random.RandomState(sd + 100),
                                time.process_time() + 0.4, st)
                for t in (st, st2):
                    drops[fl].append(1.0 - float(t["cs"]) / float(t["s"]))
        assert max(drops[FLOOR]) <= FLOOR + 1e-9, ("leash breached", max(drops[FLOOR]))
        assert max(drops[0.0]) > FLOOR, ("no drift to leash -- test is vacuous", max(drops[0.0]))
    finally:
        ACCEPT_T, ACCEPT_FLOOR = _o_accept, _o_floor
    print("self-test: leash holds cur within %.3f%% of best over 8 calls (worst %.4f%%) and is not "
          "vacuous (worst unleashed drop %.3f%%)"
          % (100 * FLOOR, 100 * max(drops[FLOOR]), 100 * max(drops[0.0])))
    print("self-test: repair from oversized radii is strictly feasible; ALL CHECKS PASS")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
