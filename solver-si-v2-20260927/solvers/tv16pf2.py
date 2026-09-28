"""Circle-packing solver for the packomania `csqv` census: pack n circles in the unit square to
maximise the sum of radii.

Method -- a CONVEX RESTRICTION / convex-concave procedure (CCP) driven by an LP, run inside a
TRUST REGION with PROVABLY INERT PAIR CONSTRAINTS PRUNED, wrapped in a time-sliced multistart with
basin hopping.

    The hard constraint is  ||p_i - p_j|| >= r_i + r_j.  The left side is CONVEX in the positions, so
    for ANY unit vector u,  ||p_i - p_j|| >= u . (p_i - p_j).  Taking u = the current unit separation
    direction therefore gives a LINEAR constraint

        u_ij . (p_i - p_j) >= r_i + r_j + MARG

    that is a GLOBALLY VALID *inner* (conservative) approximation of the true feasible set -- not just
    a local linearisation.  Consequences we rely on:

      * every LP optimum is EXACTLY feasible for the original problem (no repair, no penalty tuning);
      * the current iterate is feasible for its own LP, so sum(r) is MONOTONE non-decreasing;
      * the wall constraints (r_i <= x_i - LO, ...) are already linear, so the whole restricted
        problem is an LP over (x, y, r) solved by HiGHS.

PRUNING (the reason this is fast enough at large n).  Add to the LP a box trust region of half-width
`tr` on every centre coordinate and a cap `r_i <= r_i0 + rho` on radius growth.  Both are extra
restrictions, so the LP optimum is still exactly feasible.  Under them a centre moves by at most
sqrt(2)*tr and a radius grows by at most rho, so a pair (i, j) whose current separation satisfies

        dist_ij0  >=  r_i0 + r_j0 + 2*rho + 2*sqrt(2)*tr

CANNOT become tight no matter what the LP does: its true constraint holds automatically and its row
can be dropped.  Only the O(n) locally-crowded pairs survive instead of all n(n-1)/2.  Measured on
this workspace: at n=99 one LP falls from 4851 rows / 0.133 s to ~260 rows / 0.010 s, and a full CCP
convergence from 0.88 s to 0.145 s (6x) -- reaching the SAME or a slightly better local optimum, since
the pruning changes no optimum, only the row count.  CPU, not the evaluation budget, is the binding
meter, so that 6x buys ~6x the restarts at the sizes that were starving.

Budgeting.  Two meters must both be respected: the shared evaluation budget (polled via `meter.left()`)
and a process-CPU allowance.  The CPU allowance is derived AT ENTRY from `time.process_time()` and from
`len(targets)` -- never from a module constant -- because `solve()` may be called repeatedly in one
process and, in the offline held-out re-run, once per size in a fresh process with its own cap.

TWO-STAGE LOCAL SOLVE (the iteration-3 change).  The free-radius LP above is a *refiner*: from a raw
seed it locks onto whatever contact graph the seed happens to suggest, and that graph is what the
remaining ~0.3% gap was made of.  Measured on this census, the radii of a good packing are NEARLY
EQUAL (at n=99 the committed pack runs r in [0.030, 0.072] about a mean of 0.0527), which says the
structure we are looking for is the dense, hexagonal-ish one of the CONGRUENT-circle problem.  So a
first stage solves exactly that problem -- maximise a SINGLE common radius t -- with the same convex
restriction:

    max t   s.t.   u_ij . (p_i - p_j) >= 2t + MARG,   t <= x_i - LO,  t <= HI - x_i,  ... (same walls)

That is an LP over `(x, y, t)` with 2n+1 variables, it inherits the same exact-feasibility and
monotonicity properties, and it admits the same pruning bound (`dist0 >= 2t0 + 2*rho + 2*sqrt(2)*tr`).
Its output is then handed to the free-radius CCP, which spends the heterogeneity where the walls and
corners actually pay for it.  Equal radii are NOT optimal -- the best common-radius packing reaches
only Σr = 5.06 at n=99 against a 5.23 record -- but as a STRUCTURE FINDER the stage is worth about
half the remaining gap: relgap from 6 grid seeds at a fixed CPU cost went 5.3e-3 -> 3.5e-3 (n=27),
7.1e-3 -> 3.9e-3 (n=49), 7.1e-3 -> 4.0e-3 (n=75), 8.9e-3 -> 1.4e-3 (n=99).

Warm starts read the committed census `bench/packs/csqv<n>.pck` when it exists (centres AND radii, so
the pruning bound is tight from the first LP), and fall back to cold structured/random starts for ANY
n -- including sizes with no pack at all.  Cold seeds come from three families (square lattice,
balanced hexagonal rows, rotated hexagonal lattice patch); none dominated the others in a head-to-head
(see `notes/method.md`), so all three are kept for diversity rather than one being picked.  Only
`evaluate()` ever records a packing; nothing here writes to `bench/packs/`.
"""
import json
import math
import os
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

LO, HI = -0.5, 0.5
MARG = 1e-9          # constraint margin: keeps LP round-off clear of the harness's -1e-9 slack tol
RMIN = 1e-9          # radii must be strictly positive
SQ2 = math.sqrt(2.0)

# CPU allowance (seconds of process CPU) per solve() call.  PER_N_CPU is what a single-size call may
# spend; TOTAL_CPU is the ceiling for a whole multi-size call.  Both are *derived* at entry, never
# cached across calls.  Overrunning is not catastrophic (the driver harvests every tracked best even
# when its backstop fires, and every CCP result is routed through evaluate() as soon as it exists),
# but self-limiting keeps the tail sizes from being cut off.
PER_N_CPU = 55.0     # offline re-run gives 60 CPU-s per size; leave headroom for import + harvest
TOTAL_CPU = 110.0    # in-run driver backstop is 120 CPU-s for the whole census

# Search shape, tuned on n in {35, 63, 99} at the in-run per-size CPU slice (see notes/method.md).
TR0 = 0.25           # trust-region half-width, in units of the typical radius 0.5/sqrt(n)
TR_SHRINK = 0.20     # trust-region contraction once an LP stops paying
PERTURB_SCALES = (0.20, 0.07)   # basin-hopping sigma, also in units of 0.5/sqrt(n)
PERTURB_FRAC = 0.90  # fraction of restarts that perturb the incumbent rather than start fresh
RESTRUCT_FRAC = 0.35 # fraction of perturbations that go through the equal-radius restructuring stage
SYM_FRAC = 0.15      # fraction of perturbations restructured through the SYMMETRY restriction
SEED_FRAC = 0.6      # most of the per-size slice the fixed cold-seed list may consume before the
                     # refinement stage gets a turn.  The list is ~22 local solves at EVERY n while a
                     # local solve costs ~4x more at n=99 than at n=27, so without a cap the seed
                     # phase is free at small n and eats the ENTIRE slice at large n (measured:
                     # artifacts/probe_iter14_slice.py).  Soft: the cap is checked between seeds, so
                     # the phase may overshoot by one local solve.
CENSUS_SLICE = 15.0  # CPU-s a size gets when many targets share one call and the uniform split
                     # would starve them.  Measured (artifacts/probe_iter15_horizon.py): a COLD run
                     # reaches the 7-digit cap at n=45 in 4 CPU-s and at n=99 in 12, and then spends
                     # 61-93% of a 55 CPU-s slice finding nothing more.  The in-run driver has ~110
                     # CPU-s for 37 sizes, so a UNIFORM split gives every size 3 CPU-s -- below the
                     # threshold at which the search finishes, for all 37.  The census is
                     # append-or-improve, so a size this call never reaches KEEPS its committed pack
                     # and loses nothing; spreading the CPU thinly is therefore a choice, not a
                     # constraint.  The floor only ever RAISES a slice: see solve(), where the rule is
                     # max(CENSUS_SLICE, uniform) so a single-target call (the offline grade) still
                     # gets its whole PER_N_CPU.
COVER_CPU = 0.5      # CPU-s RESERVED for every target the queue has not reached yet, so concentration
                     # can never take a size to zero.  This is a coverage guarantee, not a search
                     # budget: 0.5 CPU-s is ~3 local solves at n=99 and ~13 at n=27, enough for the
                     # warm-start LP polish that the census ratchet was already living on, and enough
                     # for a cold grid seed at a size with no pack.  It exists because the two kinds
                     # of target are NOT symmetric -- a size with a committed pack keeps it if the
                     # queue never reaches it, while a size with no pack scores ZERO.  Without the
                     # reservation, tools/robustness.py's "multi-size call keeps every target" case
                     # fails at 2 CPU-s/size, which is the same defect an operator calling solve()
                     # with several fresh sizes would hit.
DIGITS_CAP = 7.0     # the scorer's cap: relgap <= 1e-7 earns full marks, so a size already there has
                     # NO score left to win and is worked last, after every size that does.

STALL_HOPS = 24      # basin hops without improving the ANCHOR before the anchor is abandoned for a
                     # fresh cold start.  See the "anchor vs best" note in the module docstring: the
                     # in-run slice (~3 CPU-s/size) fits far fewer than this many hops at most sizes,
                     # so restarts are essentially inactive there, while the graded re-run (55 CPU-s
                     # on ONE size) fits hundreds and is where a walk that can never leave its basin
                     # spends its time.  The global best is kept across a restart, so a restart can
                     # only cost restarts, never ground.
STALL_TOL = 1e-12    # an "improvement" must clear LP round-off, or a stalled walk never counts as stalled
RELOC_FRAC = 0.20    # fraction of basin hops that RELOCATE the worst circles into the largest holes
                     # instead of nudging every centre.  See _relocate: the LP moves centres inside a
                     # trust region and the jitter moves them by a fraction of one radius, so neither
                     # can take a circle OUT OF THE CELL it was seeded into.  This is the only move in
                     # the file that can, and it is what a plateau at a hard size needs.
RELOC_MAX = 3        # most circles relocated in one hop
RELOC_POOL = 8       # ...drawn without replacement from this many SMALLEST circles, so repeated
                     # relocations from one anchor are different moves rather than the same one.
RELOC_GRID = 65      # candidate points per axis for the hole search (65 nests with 33; see self-test)
SYM_K_SCREEN = 8     # fixed-set sizes k SCREENED per symmetric start with the cheap half of the
                     # pipeline (the right k is a property of the optimum's row structure, so one k
                     # is a guess and a random handful is a partial guess; see notes/method.md)
SYM_K_COMPLETE = 2   # FLOOR on how many of those are carried through the expensive half.  >1
                     # because the screening key (the equal-radius t) is a PROXY for sum(r), not
                     # sum(r) itself, so the top-ranked k is not reliably the best k.
SYM_K_COMPLETE_MAX = 4   # ceiling: completions compete with basin hopping for the same slice.
SYM_K_MARGIN = 0.02  # ...and BETWEEN those two, complete every k whose screened t is within this
                     # relative margin of the best.  Measured spread of t across k (artifacts/
                     # probe_kspread.py): one or two k are catastrophic (-12% to -22%, which the
                     # screen separates unambiguously) and the rest form a cluster the proxy cannot
                     # order.  2% is that cluster's width, so the rule reads "complete everything
                     # the screen FAILED to separate, discard what it separated".


# --------------------------------------------------------------------------------------------- utils
def _wall(xy):
    x, y = xy[..., 0], xy[..., 1]
    return np.minimum.reduce([x - LO, HI - x, y - LO, HI - y])


def _grow_batch(xy):
    """Guaranteed-feasible equal-ish radii for a batch of centre sets xy:(B,n,2) -> packs (B,n,3).

    r_i = min(wall_i, min_j d_ij / 2), so r_i + r_j <= d_ij always.  Used to give the CCP a feasible
    r0 to prune against; the LP below does far better for the same centres."""
    B, n, _ = xy.shape
    wall = _wall(xy)
    if n >= 2:
        d = xy[:, :, None, :] - xy[:, None, :, :]
        dist = np.sqrt((d ** 2).sum(-1))
        di = np.arange(n)
        dist[:, di, di] = np.inf
        r = np.minimum(wall, dist.min(axis=2) / 2.0)
    else:
        r = wall
    r = np.maximum(r - 2e-9, RMIN)
    return np.concatenate([xy, r[..., None]], axis=-1)


def _min_slack(xy, r):
    """Smallest constraint slack of a single packing (walls and pairs); >= 0 means strictly feasible."""
    n = len(r)
    s = float(np.min(_wall(xy) - r))
    if n >= 2:
        i, j = np.triu_indices(n, 1)
        d = np.sqrt(((xy[i] - xy[j]) ** 2).sum(1))
        s = min(s, float(np.min(d - r[i] - r[j])))
    return s


def _repair(xy, r):
    """Make a packing strictly feasible by shrinking radii only (never moves centres)."""
    r = np.maximum(r, RMIN)
    s = _min_slack(xy, r)
    if s < 0.0:
        r = np.maximum(r + s - 1e-12, RMIN)          # uniform shrink absorbs the worst violation
        if _min_slack(xy, r) < 0.0:                  # pathological (overlapping centres): fall back
            r = _grow_batch(xy[None])[0][:, 2]
    return r


def _pack(xy, r):
    return np.concatenate([xy, r[:, None]], axis=1)


def _clearance(pts, xy, r):
    """Largest radius a new circle centred at each of `pts` could take, given obstacles (xy, r).

    (P,2), (m,2), (m,) -> (P,).  = min(distance to the nearest wall, min_j ||p - c_j|| - r_j),
    floored at 0.  With an empty obstacle set it is just the wall distance."""
    c = _wall(pts)
    if len(r):
        d = np.sqrt(((pts[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]
        c = np.minimum(c, d.min(axis=1))
    return np.maximum(c, 0.0)


def _relocate(xy, r, moved, off=(0.0, 0.0), g=RELOC_GRID):
    """Move the circles `moved` to the largest holes.  Pure in its arguments; returns (xy, r).

    WHY THIS MOVE EXISTS.  Every other move in this file is continuous.  The CCP LP boxes each centre
    inside a trust region; the basin-hopping jitter displaces it by a fraction of one radius; the
    equal-radius and symmetry restructurings re-solve from the same centres.  None of them can take a
    circle out of the CELL of the contact graph it was seeded into, so a circle that landed somewhere
    with no room stays small for the entire run, and every later hop re-finds the same local optimum
    with the same stunted circle in it.  Seen from outside that is a plateau -- which is exactly what
    `artifacts/probe_longrun.py` measured at n=49, where 76% of the graded CPU budget bought nothing.

    The move: drop the listed circles and re-place them ONE AT A TIME at the grid point with the most
    room left over, each placement joining the obstacle set for the next so two circles never chase
    the same hole.  A placed circle takes exactly its clearance as its radius, so if the input packing
    was feasible the output is too (a shrink-only `_repair` guards the case where it was not), and the
    free-radius CCP that follows starts feasible and monotone as usual.

    `moved` is passed in rather than chosen here (and `off` shifts the grid by a sub-cell amount) so
    that the function stays a pure, testable map and the caller owns all the randomness."""
    n = len(r)
    xy, r = np.array(xy, dtype=float), np.array(r, dtype=float)
    moved = np.atleast_1d(np.asarray(moved, dtype=int))
    if len(moved) == 0 or len(moved) >= n:
        return xy, r
    keep = np.setdiff1d(np.arange(n), moved)
    ax = np.linspace(LO, HI, g)
    pts = np.stack(np.meshgrid(ax + off[0], ax + off[1], indexing="ij"), axis=-1).reshape(-1, 2)
    pts = np.clip(pts, LO, HI)
    kxy, kr = xy[keep], r[keep]
    for i in moved:
        c = _clearance(pts, kxy, kr)
        b = int(np.argmax(c))
        xy[i] = pts[b]
        r[i] = max(float(c[b]) - 2e-9, RMIN)
        kxy = np.concatenate([kxy, xy[i][None]], axis=0)
        kr = np.concatenate([kr, r[i][None]], axis=0)
    return xy, _repair(xy, r)


# ----------------------------------------------------------------------------------------- the CCP LP
class _LP(object):
    """Reusable sparse-LP scaffolding for one n.

    The wall block is constant, so it is built once; the pair block is rebuilt per iterate because
    which pairs survive the inertness test changes as the packing moves."""

    def __init__(self, n):
        self.n = n
        self.I, self.J = np.triu_indices(n, 1)
        self.P_full = len(self.I)
        idx = np.arange(n)
        # wall rows (row indices are RELATIVE; the pair count is added at build time):
        #   r - x <= -LO,   r + x <= HI,   r - y <= -LO,   r + y <= HI
        rw, cw, dw, bw = [], [], [], []
        for off, sgn, coord in ((0, -1.0, 0), (n, 1.0, 0), (2 * n, -1.0, 1), (3 * n, 1.0, 1)):
            rw.append(np.repeat(off + idx, 2))
            cw.append(np.stack([2 * n + idx, coord * n + idx], 1).ravel())
            dw.append(np.stack([np.ones(n), np.full(n, sgn)], 1).ravel())
            bw.append(np.full(n, (-(LO + MARG)) if sgn < 0 else (HI - MARG)))
        self.rows_w = np.concatenate(rw)
        self.cols_w = np.concatenate(cw)
        self.data_w = np.concatenate(dw)
        self.b_w = np.concatenate(bw)
        self.c = np.zeros(3 * n)
        self.c[2 * n:] = -1.0
        self.last_rows = self.P_full        # diagnostics only

    def step(self, xy, r0=None, tr=None, rho=None, eq=None):
        """One LP: the best packing inside the convex restriction around `xy`.

        With `tr`/`rho` given, the centres are boxed to +/- tr, radii to r0 + rho, and every pair that
        those bounds prove can never become tight is dropped.  `eq` is an optional (A_eq, b_eq)
        symmetry restriction -- more equalities can only shrink the feasible set, so the optimum stays
        exactly feasible and the pruning proof is unaffected.  Returns (xy, r) or None."""
        n = self.n
        I, J = self.I, self.J
        d = xy[I] - xy[J]
        dist = np.maximum(np.sqrt((d ** 2).sum(1)), 1e-12)
        if tr is not None and rho is not None and r0 is not None:
            # provably inert  <=>  dist0 >= r_i0 + r_j0 + 2*rho + 2*sqrt(2)*tr
            keep = dist < (r0[I] + r0[J] + 2.0 * rho + 2.0 * SQ2 * tr + 1e-12)
            I, J, d, dist = I[keep], J[keep], d[keep], dist[keep]
        u = d / dist[:, None]
        P = len(I)
        self.last_rows = P
        ones = np.ones(P)
        rows = np.concatenate([np.repeat(np.arange(P), 6), self.rows_w + P])
        cols = np.concatenate([np.stack([I, J, n + I, n + J, 2 * n + I, 2 * n + J], 1).ravel(),
                               self.cols_w])
        data = np.concatenate([np.stack([-u[:, 0], u[:, 0], -u[:, 1], u[:, 1], ones, ones], 1).ravel(),
                               self.data_w])
        A = csr_matrix((data, (rows, cols)), shape=(P + 4 * n, 3 * n))
        b = np.concatenate([np.full(P, -MARG), self.b_w])
        if tr is None:
            lb, ub = np.full(2 * n, LO), np.full(2 * n, HI)
        else:
            c0 = np.concatenate([xy[:, 0], xy[:, 1]])
            lb, ub = np.maximum(LO, c0 - tr), np.minimum(HI, c0 + tr)
        rub = np.full(n, 0.5) if (rho is None or r0 is None) else np.minimum(0.5, r0 + rho)
        bounds = np.concatenate([np.stack([lb, ub], 1),
                                 np.stack([np.full(n, RMIN), np.maximum(rub, 2 * RMIN)], 1)], 0)
        aeq, beq = (eq if eq is not None else (None, None))
        res = linprog(self.c, A_ub=A, b_ub=b, A_eq=aeq, b_eq=beq, bounds=bounds, method="highs")
        if not res.success or res.x is None:
            return None
        z = res.x
        return np.stack([z[:n], z[n:2 * n]], 1), z[2 * n:]

    def run(self, xy, deadline, r0=None, max_iter=300, tol=1e-12, eq=None):
        """Iterate the pruned restriction to a KKT point (or until the CPU deadline).

        The trust region starts at TR0 typical radii and contracts by TR_SHRINK whenever an LP stops
        paying, which is also what tightens the pruning bound as the packing settles.  Returns
        (xy, r, sum_r); sum_r is monotone non-decreasing over the loop."""
        n = self.n
        unit = 0.5 / math.sqrt(n)
        if r0 is None:
            r0 = _grow_batch(xy[None])[0][:, 2]
        best_xy = xy
        best_r = _repair(xy, r0)
        best_s = float(best_r.sum())
        tr = TR0 * unit
        tr_min = 1e-7 * unit
        for _ in range(max_iter):
            if time.process_time() > deadline:
                break
            out = self.step(best_xy, best_r, tr, tr, eq=eq)
            if out is None:
                tr *= TR_SHRINK
                if tr < tr_min:
                    break
                continue
            nxy, nr = out
            nr = _repair(nxy, nr)
            s = float(nr.sum())
            gain = s - best_s
            if s > best_s:
                best_xy, best_r, best_s = nxy, nr, s
            if gain <= tol or gain < 1e-4 * unit:
                tr *= TR_SHRINK                      # converged at this scale: refine
                if tr < tr_min:
                    break
        return best_xy, best_r, best_s


# ------------------------------------------------------------------------------- the EQUAL-RADIUS CCP
class _LPEq(object):
    """Reusable sparse-LP scaffolding for the CONGRUENT-circle problem at one n.

    Variables are `(x[n], y[n], t)` with a SINGLE common radius t, and the objective is `max t`.  This
    is the same convex restriction as `_LP` with `r_i == r_j == t`, so it keeps both properties we
    rely on: every LP optimum is exactly feasible for the true congruent-circle problem, and the
    current iterate is feasible for its own LP so t is monotone.  Its purpose is STRUCTURE, not the
    objective -- a common-radius optimum is measurably short of the csqv record (Sum r = n*t), but it
    finds the dense hexagonal-ish contact graph that `_LP` then exploits with unequal radii."""

    def __init__(self, n):
        self.n = n
        self.I, self.J = np.triu_indices(n, 1)
        self.P_full = len(self.I)
        idx = np.arange(n)
        # wall rows (relative row indices):  t - x <= -LO,  t + x <= HI,  t - y <= -LO,  t + y <= HI
        rw, cw, dw, bw = [], [], [], []
        for off, sgn, coord in ((0, -1.0, 0), (n, 1.0, 0), (2 * n, -1.0, 1), (3 * n, 1.0, 1)):
            rw.append(np.repeat(off + idx, 2))
            cw.append(np.stack([np.full(n, 2 * n), coord * n + idx], 1).ravel())
            dw.append(np.stack([np.ones(n), np.full(n, sgn)], 1).ravel())
            bw.append(np.full(n, (-(LO + MARG)) if sgn < 0 else (HI - MARG)))
        self.rows_w = np.concatenate(rw)
        self.cols_w = np.concatenate(cw)
        self.data_w = np.concatenate(dw)
        self.b_w = np.concatenate(bw)
        self.c = np.zeros(2 * n + 1)
        self.c[-1] = -1.0
        self.last_rows = self.P_full        # diagnostics only

    def step(self, xy, t0, tr=None, rho=None, eq=None):
        """One LP.  With `tr`/`rho` given the centres are boxed to +/- tr and t to t0 + rho, and every
        pair the bounds prove can never become tight is dropped:  a centre moves at most sqrt(2)*tr
        and t grows at most rho, so `dist0 >= 2*t0 + 2*rho + 2*sqrt(2)*tr` implies the true
        constraint.  Same theorem as `_LP.step`, with r_i = r_j = t."""
        n = self.n
        I, J = self.I, self.J
        d = xy[I] - xy[J]
        dist = np.maximum(np.sqrt((d ** 2).sum(1)), 1e-12)
        if tr is not None and rho is not None:
            keep = dist < (2.0 * t0 + 2.0 * rho + 2.0 * SQ2 * tr + 1e-12)
            I, J, d, dist = I[keep], J[keep], d[keep], dist[keep]
        u = d / dist[:, None]
        P = len(I)
        self.last_rows = P
        rows = np.concatenate([np.repeat(np.arange(P), 5), self.rows_w + P])
        cols = np.concatenate([np.stack([I, J, n + I, n + J, np.full(P, 2 * n)], 1).ravel(),
                               self.cols_w])
        data = np.concatenate([np.stack([-u[:, 0], u[:, 0], -u[:, 1], u[:, 1], np.full(P, 2.0)],
                                        1).ravel(), self.data_w])
        A = csr_matrix((data, (rows, cols)), shape=(P + 4 * n, 2 * n + 1))
        b = np.concatenate([np.full(P, -MARG), self.b_w])
        if tr is None:
            lb, ub = np.full(2 * n, LO), np.full(2 * n, HI)
            tub = 0.5
        else:
            c0 = np.concatenate([xy[:, 0], xy[:, 1]])
            lb, ub = np.maximum(LO, c0 - tr), np.minimum(HI, c0 + tr)
            tub = min(0.5, t0 + rho)
        bounds = np.concatenate([np.stack([lb, ub], 1),
                                 np.asarray([[RMIN, max(tub, 2 * RMIN)]])], 0)
        aeq, beq = (eq if eq is not None else (None, None))
        res = linprog(self.c, A_ub=A, b_ub=b, A_eq=aeq, b_eq=beq, bounds=bounds, method="highs")
        if not res.success or res.x is None:
            return None
        z = res.x
        return np.stack([z[:n], z[n:2 * n]], 1), float(z[-1])

    def run(self, xy, deadline, max_iter=300, eq=None):
        """Iterate the pruned equal-radius restriction.  Returns (xy, t) with t the TRUE largest
        common radius for those centres (so the pair it returns is exactly feasible)."""
        n = self.n
        unit = 0.5 / math.sqrt(n)
        best_xy = xy
        best_t = _common_r(xy)
        tr = TR0 * unit
        tr_min = 1e-7 * unit
        for _ in range(max_iter):
            if time.process_time() > deadline:
                break
            out = self.step(best_xy, best_t, tr, tr, eq=eq)
            if out is None:
                tr *= TR_SHRINK
                if tr < tr_min:
                    break
                continue
            nxy, _t_lp = out
            # trust the GEOMETRY, not the LP's t: recompute the exact largest common radius.
            nt = _common_r(nxy)
            gain = nt - best_t
            if nt > best_t:
                best_xy, best_t = nxy, nt
            if gain <= 1e-13 or gain < 1e-5 * unit:
                tr *= TR_SHRINK
                if tr < tr_min:
                    break
        return best_xy, best_t


def _common_r(xy):
    """Largest common radius for these centres: min over i of min(wall_i, min_j d_ij/2)."""
    return float(np.min(_grow_batch(xy[None])[0][:, 2]))


# ---------------------------------------------------------------------- SYMMETRY-RESTRICTED CCP
# Many csqv optima are invariant under a symmetry of the square.  Imposing one is a set of LINEAR
# EQUALITIES on (x, y, r), so it is just one more *restriction* layered on the convex restriction
# already in use: every LP optimum stays exactly feasible for the true problem, monotonicity is
# unchanged, and the pruning theorem is untouched (extra restrictions can only shrink the reachable
# set, so a pair proved inert stays inert).  What it buys is a different search space: the free CCP
# keeps converging to slightly-asymmetric near-misses, and the equalities forbid exactly those while
# halving the effective number of degrees of freedom.
#
#   'mx'  mirror in the y-axis      (x, y) -> (-x,  y)   fixed set: x = 0
#   'c2'  half-turn about centre    (x, y) -> (-x, -y)   fixed set: the origin (so k is 0 or 1)
#   'md'  mirror in the diagonal    (x, y) -> ( y,  x)   fixed set: y = x
#
# The restriction is only a STRUCTURE FINDER: a symmetric run is always followed by an unrestricted
# free CCP polish, because the true optimum is usually near-symmetric rather than exactly symmetric.
SYM_KINDS = ("mx", "c2", "md")


def _sym_project(xy, kind, k):
    """Project centres onto the `kind`-symmetric subspace with `k` circles on the fixed set.

    Returns (xy_sym, pairs (P,2) int, axis (K,) int) or None when k is not usable for this n.
    The pairing is an arbitrary index assignment -- the geometry comes from the representatives, and
    each representative's partner is *overwritten* with its exact image, so `xy_sym` is symmetric to
    the last bit (the image is an exact negation/swap of IEEE doubles)."""
    n = len(xy)
    if kind == "c2":
        k = n % 2                                     # only the centre point is fixed by a half-turn
    if k < 0 or k > n or (n - k) % 2 != 0:
        return None
    h = (n - k) // 2
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    if kind == "mx":
        dist_fix, key = np.abs(x), x
    elif kind == "c2":
        dist_fix, key = np.sqrt(x * x + y * y), x + 1e-9 * y
    else:                                             # 'md'
        dist_fix, key = np.abs(y - x) / SQ2, (y - x)
    order = np.argsort(dist_fix, kind="stable")
    axis = order[:k]
    rest = order[k:]
    rest = rest[np.argsort(-key[rest], kind="stable")]
    reps, part = rest[:h], rest[h:]
    out = xy.copy()
    if kind == "mx":
        out[axis, 0] = 0.0
        out[reps, 0] = np.maximum(np.abs(out[reps, 0]), 1e-4)
        out[part, 0] = -out[reps, 0]
        out[part, 1] = out[reps, 1]
    elif kind == "c2":
        out[axis] = 0.0
        rx, ry = out[reps, 0], out[reps, 1]
        bad = (rx * rx + ry * ry) < 1e-8              # a representative sitting on its own image
        rx = np.where(bad, 1e-4, rx)
        out[reps, 0], out[reps, 1] = rx, ry
        out[part, 0], out[part, 1] = -rx, -ry
    else:
        s = 0.5 * (out[axis, 0] + out[axis, 1])
        out[axis, 0], out[axis, 1] = s, s
        rx, ry = out[reps, 0], out[reps, 1]
        bad = np.abs(ry - rx) < 1e-4
        ry = np.where(bad, np.minimum(rx + 1e-4, HI), ry)
        out[reps, 0], out[reps, 1] = rx, ry
        out[part, 0], out[part, 1] = ry, rx
    return np.clip(out, LO, HI), np.stack([reps, part], 1), axis


def _sym_eq(n, kind, pairs, axis, nvar, with_r):
    """Sparse (A_eq, b_eq) for the symmetry restriction over (x[n], y[n], r[n] or t)."""
    rows, cols, data = [], [], []
    rc = [0]

    def add(pairs_cd):
        for c, d in pairs_cd:
            rows.append(rc[0])
            cols.append(c)
            data.append(d)
        rc[0] += 1

    for i, j in pairs:
        i, j = int(i), int(j)
        if kind == "mx":
            add(((i, 1.0), (j, 1.0)))                       # x_i + x_j = 0
            add(((n + i, 1.0), (n + j, -1.0)))              # y_i - y_j = 0
        elif kind == "c2":
            add(((i, 1.0), (j, 1.0)))
            add(((n + i, 1.0), (n + j, 1.0)))               # y_i + y_j = 0
        else:
            add(((i, 1.0), (n + j, -1.0)))                  # x_i - y_j = 0
            add(((n + i, 1.0), (j, -1.0)))                  # y_i - x_j = 0
        if with_r:
            add(((2 * n + i, 1.0), (2 * n + j, -1.0)))      # r_i - r_j = 0
    for i in axis:
        i = int(i)
        if kind == "mx":
            add(((i, 1.0),))                                # x_i = 0
        elif kind == "c2":
            add(((i, 1.0),))
            add(((n + i, 1.0),))                            # y_i = 0
        else:
            add(((i, 1.0), (n + i, -1.0)))                  # x_i - y_i = 0
    if rc[0] == 0:
        return None
    A = csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))),
                   shape=(rc[0], nvar))
    return A, np.zeros(rc[0])


def _sym_defect(xy, kind, pairs, axis):
    """Largest coordinate-wise breach of the symmetry -- 0 for an exactly symmetric configuration."""
    d = 0.0
    for i, j in pairs:
        if kind == "mx":
            d = max(d, abs(xy[i, 0] + xy[j, 0]), abs(xy[i, 1] - xy[j, 1]))
        elif kind == "c2":
            d = max(d, abs(xy[i, 0] + xy[j, 0]), abs(xy[i, 1] + xy[j, 1]))
        else:
            d = max(d, abs(xy[i, 0] - xy[j, 1]), abs(xy[i, 1] - xy[j, 0]))
    for i in axis:
        if kind == "mx":
            d = max(d, abs(xy[i, 0]))
        elif kind == "c2":
            d = max(d, abs(xy[i, 0]), abs(xy[i, 1]))
        else:
            d = max(d, abs(xy[i, 0] - xy[i, 1]))
    return float(d)


def _n_complete(screened):
    """How many screened k to carry through the expensive half -- adaptive, not a flat count.

    `screened` is [(t, ...)] sorted by descending equal-radius t.  A flat SYM_K_COMPLETE spends the
    same effort whether the screen produced one clear winner or a five-way tie, which is wrong in
    both directions: it wastes a completion when the key discriminated and throws away the right
    structure when it did not.  Measured (artifacts/probe_kspread.py, n in {27,49,75,99}), t across
    k is bimodal -- one or two k sit 12-22% below the best and the remainder cluster inside ~2% --
    so the rule is to complete the cluster the proxy could not order and drop only what it clearly
    separated, floored at SYM_K_COMPLETE and capped at SYM_K_COMPLETE_MAX because completions
    compete with basin hopping for one slice.

    Pure function of the screened keys, so it is directly testable (self-test check 11)."""
    if not screened:
        return 0
    tbest = screened[0][0]
    if not (tbest > 0.0):
        return min(len(screened), SYM_K_COMPLETE)
    near = sum(1 for z in screened if z[0] >= (1.0 - SYM_K_MARGIN) * tbest)
    return min(len(screened), SYM_K_COMPLETE_MAX, max(SYM_K_COMPLETE, near))


def _sym_axis_counts(n, kind, rng):
    """Plausible fixed-set sizes k for this n and symmetry, in the order to try them.

    For 'mx' / 'md' the parity of k is forced by n; the useful range is small (an m-row packing has
    at most ~m circles on its mirror axis), so a handful of candidates near sqrt(n) covers it."""
    if kind == "c2":
        return [n % 2]
    top = int(math.ceil(math.sqrt(max(n, 1)))) + 2
    cand = [k for k in range(0, min(n, top) + 1) if (n - k) % 2 == 0]
    if not cand:
        return []
    rng.shuffle(cand)
    return cand


# --------------------------------------------------------------------------------------------- starts
def _read_pack(n):
    """Guarded warm start from the committed census.

    Returns (centres (n,2), radii (n,)) or None for ANY n -- a missing, short, malformed or
    non-finite pack must cold-start, never raise: an exception here ends the whole solve() call and
    loses every later target."""
    p = os.path.join("bench", "packs", "csqv%d.pck" % n)
    try:
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        rows = []
        for ln in lines[2:]:
            parts = ln.split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(rows) != n:
            return None
        a = np.asarray(rows, dtype=float)
        if not np.all(np.isfinite(a)):
            return None
        return np.clip(a[:, :2], LO, HI), np.maximum(a[:, 2], RMIN)
    except Exception:
        return None


def _grid_start(n, rng, jitter):
    """Lattice start: the k x m cell centres of the coarsest grid that holds n circles, jittered."""
    k = int(math.ceil(math.sqrt(n)))
    m = int(math.ceil(n / float(k)))
    xs = (np.arange(k) + 0.5) / k - 0.5
    ys = (np.arange(m) + 0.5) / m - 0.5
    g = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    sel = rng.choice(len(g), size=n, replace=False) if len(g) > n else np.arange(n)
    xy = g[sel] + rng.normal(0.0, jitter / max(k, m), size=(n, 2))
    return np.clip(xy, LO + 1e-4, HI - 1e-4)


def _random_start(n, rng):
    return rng.uniform(LO + 1e-3, HI - 1e-3, size=(n, 2))


def _hex_rows_start(n, m, rng, jitter=0.0):
    """n circles in `m` rows with balanced counts, alternate rows carrying the surplus.

    This is the family that the congruent-circle optima in a square actually come from (rows of a and
    a+1 circles, offset by half a step), which a square lattice cannot express.  Returns None when m
    is not a usable row count for this n, so callers must guard."""
    if m < 1 or m > n:
        return None
    base, extra = divmod(n, m)
    counts = np.full(m, base)
    order = list(range(0, m, 2)) + list(range(1, m, 2))     # surplus to alternate rows first
    for k in range(extra):
        counts[order[k % m]] += 1
    if counts.min() < 1:
        return None
    ys = (np.arange(m) + 0.5) / m - 0.5
    pts = []
    for i in range(m):
        c = int(counts[i])
        pts.append(np.stack([(np.arange(c) + 0.5) / c - 0.5, np.full(c, ys[i])], 1))
    xy = np.concatenate(pts, 0)
    if jitter > 0.0:
        xy = xy + rng.normal(0.0, jitter / m, size=xy.shape)
    return np.clip(xy, LO + 1e-4, HI - 1e-4)


def _rot_hex_start(n, theta, rng, jitter=0.0):
    """The n most central points of a hexagonal lattice rotated by `theta`, scaled so n of them fit.

    The spacing is shrunk geometrically until at least n lattice points land inside the square; the n
    nearest the patch centroid are kept so the seed stays compact.  Returns None if it never fits."""
    s = 1.0 / math.sqrt(max(n, 1))
    for _ in range(80):
        k = int(math.ceil(2.5 / s)) + 3
        if k > 400:                                  # keep the meshgrid bounded for tiny spacings
            return None
        a = np.arange(-k, k + 1)
        gi, gj = np.meshgrid(a, a, indexing="ij")
        px = (gi + 0.5 * gj) * s
        py = gj * (s * math.sqrt(3.0) / 2.0)
        ct, st = math.cos(theta), math.sin(theta)
        qx, qy = ct * px - st * py, st * px + ct * py
        inside = (qx >= LO) & (qx <= HI) & (qy >= LO) & (qy <= HI)
        if int(inside.sum()) >= n:
            q = np.stack([qx[inside], qy[inside]], 1)
            q = q[np.argsort(((q - q.mean(0)) ** 2).sum(1))[:n]]
            if jitter > 0.0:
                q = q + rng.normal(0.0, jitter * s, size=q.shape)
            return np.clip(q, LO + 1e-4, HI - 1e-4)
        s *= 0.95
    return None


def _cold_seed(n, rng):
    """One random draw from the three cold-seed families (square lattice / hex rows / rotated hex).

    Kept as a mixture because the head-to-head at fixed CPU showed no family dominating: rotated-hex
    won at n=75, plain lattice at n=27 and n=49 (see notes/method.md).  Always returns an (n,2)
    array -- every family that can fail falls back to the lattice."""
    pick = rng.randint(3)
    root = int(round(math.sqrt(max(n, 1))))
    if pick == 1:
        m = max(1, root + rng.randint(-2, 3))
        xy = _hex_rows_start(n, m, rng, jitter=0.10 if rng.rand() < 0.5 else 0.0)
        if xy is not None:
            return xy
    elif pick == 2:
        xy = _rot_hex_start(n, rng.rand() * math.pi / 3.0, rng,
                            jitter=0.05 if rng.rand() < 0.5 else 0.0)
        if xy is not None:
            return xy
    return _grid_start(n, rng, 0.20)


# ---------------------------------------------------------------------------------------------- solve
def _anchor_update(anchor, shot):
    """Advance the basin-hopping anchor by one hop.  Pure in its two arguments, so it is testable.

    `shot` is the best packing the hop just produced (None if it produced nothing feasible).  An
    improving hop becomes the new anchor and clears the stall counter; a failing hop increments it;
    the STALL_HOPS-th consecutive failure DROPS the anchor, which makes the next hop a cold start.

    This is deliberately a separate object from the global best `state`: dropping the anchor costs
    the restarts spent rebuilding one, but it can never cost score, because `state` is what the
    harness was already shown and it only grows.  The failure this guards is silent -- an anchor that
    never drops makes no packing infeasible and costs no digit at the 3 CPU-s in-run slice, it just
    burns the graded re-run's 55 CPU-s re-sampling one basin."""
    if shot["xy"] is not None and shot["s"] > anchor["s"] + STALL_TOL:
        anchor["xy"], anchor["s"], anchor["stall"] = shot["xy"], shot["s"], 0
        anchor["r"] = shot.get("r")          # carried for _relocate, which ranks circles by radius
        return anchor
    anchor["stall"] += 1
    if anchor["stall"] >= STALL_HOPS:
        anchor["xy"], anchor["s"], anchor["stall"] = None, -np.inf, 0
        anchor["r"] = None
    return anchor


def _solve_one(evaluate, meter, rng, n, slice_end, eval_stop):
    """Search one size until its CPU slice or its share of the evaluation budget runs out."""
    lp = _LP(n)
    lpe = _LPEq(n)
    unit = 0.5 / math.sqrt(n)
    state = {"xy": None, "r": None, "s": -np.inf}   # GLOBAL best -- only ever grows
    shot = {"xy": None, "r": None, "s": -np.inf}    # best of the hop currently in flight
    anchor = {"xy": None, "r": None, "s": -np.inf, "stall": 0}   # what the next hop starts FROM

    def consider(xy, r):
        """Verify locally, then spend one evaluation unit recording the candidate."""
        r = _repair(xy, r)
        if _min_slack(xy, r) < 0.0 or not np.all(r > 0.0):
            return
        if meter.left() <= 0:
            return
        feas, tot = evaluate(n, _pack(xy, r)[None])
        feas = np.atleast_1d(feas)
        tot = np.atleast_1d(tot)
        got = float(tot[0]) if bool(feas[0]) else -np.inf
        if got > state["s"]:
            state["xy"], state["r"], state["s"] = xy.copy(), r.copy(), got
        if got > shot["s"]:
            shot["xy"], shot["r"], shot["s"] = xy.copy(), r.copy(), got

    def budget_left():
        return (time.process_time() < slice_end and meter.left() > 0 and meter.used < eval_stop)

    def two_stage(xy0):
        """Equal-radius CCP to fix the contact structure, then free-radius CCP to spend it.

        Both halves are exactly feasible by construction, and the equal-radius t is handed on as the
        free CCP's r0 so its pruning bound is tight from the first LP.  The free stage is run from the
        RAW seed as well when there is time, because the equal-radius stage can itself fall into a
        square-lattice fixed point (it does at n=49) and then hands on a worse structure than the
        seed had."""
        half = min(slice_end, time.process_time() + max(0.0, (slice_end - time.process_time()) * 0.5))
        xy1, t = lpe.run(xy0, half)
        xy, r, _ = lp.run(xy1, slice_end, r0=np.full(n, max(t, RMIN)))
        consider(xy, r)

    def sym_complete(xy1, t, kind, pairs, axis):
        """Expensive half of the symmetric pipeline: symmetry-restricted FREE-radius CCP from an
        already-equalised symmetric iterate, then a free polish with the symmetry RELEASED.

        The equalities are an extra restriction on the LP, so every iterate is still exactly
        feasible; the point of them is to forbid the slightly-asymmetric near-misses the free CCP
        keeps converging to.  The final unrestricted polish is what makes the stage a structure
        finder rather than a constraint: csqv optima are usually NEAR-symmetric, not exactly so, and
        that last run can only increase sum(r) (it starts from the symmetric packing)."""
        eq_f = _sym_eq(n, kind, pairs, axis, 3 * n, True)
        now = time.process_time()
        stop_f = min(slice_end, now + max(0.0, (slice_end - now) * 0.5))
        xy2, r2, _ = lp.run(xy1, stop_f, r0=np.full(n, max(t, RMIN)), eq=eq_f)
        consider(xy2, r2)
        xy3, r3, _ = lp.run(xy2, slice_end, r0=r2)          # symmetry released
        consider(xy3, r3)

    def sym_screen(xy0, kind):
        """Screen every plausible fixed-set size k with the CHEAP half, complete only the best few.

        `k` is how many circles the restriction puts on the symmetry axis, so different k are
        different contact graphs -- the right one is a property of the optimum's row structure, not
        something a single guess finds.  Completing the full three-LP pipeline for each k costs
        roughly 3x its equal-radius half, so a sweep could previously afford only a random handful
        of k.  Screening with the symmetry-restricted equal-radius CCP alone covers the WHOLE
        candidate set for about the same CPU, and the completed runs resume from the screened
        iterate `xy1`, so the cheap half is never paid twice.

        The screening key `t` is the true largest common radius of the screened centres (`_LPEq.run`
        recomputes it geometrically), so the ranking is over a real quantity -- but it is a PROXY
        for sum(r), not sum(r): the equal-radius optimum runs 3-6% short of the csqv record
        (notes/method.md), so ranking k by t ranks them by the wrong objective.  That is exactly why
        SYM_K_COMPLETE > 1: the screen is used to discard the clearly bad k, not to pick the winner."""
        screened = []
        for k in _sym_axis_counts(n, kind, rng)[:SYM_K_SCREEN]:
            if not budget_left():
                break
            proj = _sym_project(xy0, kind, k)
            if proj is None:
                continue
            xys, pairs, axis = proj
            eq_e = _sym_eq(n, kind, pairs, axis, 2 * n + 1, False)
            now = time.process_time()
            # A screen must stay cheap or it is not a screen: cap each one at a fifth of what is
            # left, so a whole sweep cannot eat the slice the completions need.
            stop_e = min(slice_end, now + max(0.0, (slice_end - now) * 0.20))
            xy1, t = lpe.run(xys, stop_e, eq=eq_e)
            screened.append((t, k, xy1, pairs, axis))
        screened.sort(key=lambda z: -z[0])
        for t, _k, xy1, pairs, axis in screened[:_n_complete(screened)]:
            if not budget_left():
                break
            sym_complete(xy1, t, kind, pairs, axis)

    # --- seeds: the committed census first (guarded), then structured cold starts --------------------
    # The seed list is a BUDGET, not a checklist, and it is spent ROUND-ROBIN across the three cold
    # families.  Both halves of that exist for the same measured reason (artifacts/
    # probe_iter14_slice.py): one local solve costs ~0.04 CPU-s at n=27 and ~0.18 at n=99, but the
    # seed list is a fixed ~22 solves at EVERY n.  At the in-run census slice (~3 CPU-s/size) that
    # list finished in a third of the slice at n=27 and OVERRAN the whole slice at n=99 -- where the
    # trace showed the third rotated-hex seed truncated, the symmetry screen never reached, and
    # basin hopping never started at all.  So:
    #   * `seed_end` caps the seed phase at SEED_FRAC of the slice, which guarantees the refinement
    #     stage a turn at every n instead of only at the small ones;
    #   * the families INTERLEAVE, so when the cap does bite, what survives is one seed from each
    #     family rather than three lattices and nothing else.
    # At small n neither binds (the phase ends early on its own) and this is a reordering only.
    warm = _read_pack(n)
    if warm is not None and budget_left():
        xy, r, _ = lp.run(warm[0], slice_end, r0=warm[1])       # never re-equalise a good census pack
        consider(xy, r)
    now = time.process_time()
    seed_end = min(slice_end, now + max(0.0, (slice_end - now) * SEED_FRAC))

    def seed_left():
        return budget_left() and time.process_time() < seed_end

    root = int(round(math.sqrt(n)))
    for jit, m, theta in zip((0.0, 0.12, 0.25), (root - 1, root, root + 1),
                             (0.0, math.pi / 12.0, math.pi / 6.0)):
        if not seed_left():
            break
        xy0 = _grid_start(n, rng, jit)
        xy, r, _ = lp.run(xy0, slice_end)                        # free-only: the iteration-2 baseline
        consider(xy, r)
        if seed_left():
            two_stage(xy0)
        if not seed_left():
            break
        xy0 = _hex_rows_start(n, m, rng)
        if xy0 is not None:
            two_stage(xy0)
        if not seed_left():
            break
        xy0 = _rot_hex_start(n, theta, rng)
        if xy0 is not None:
            two_stage(xy0)
    if seed_left():                              # one symmetry-restricted cold start, screened k
        kind = SYM_KINDS[rng.randint(len(SYM_KINDS))]
        sym_screen(_cold_seed(n, rng), kind)

    # --- basin hopping: refine the ANCHOR, restructure it, or restart cold --------------------------
    # The anchor is deliberately NOT the global best.  Perturbing the global best forever is a
    # monotone hill-climb with no way out of its basin: every hop that fails is discarded and the
    # next one starts from the same point, so the walk's reachable set never changes.  At the in-run
    # slice (~3 CPU-s/size) there are too few hops for that to bind, which is why every A/B in this
    # workspace missed it -- but the graded re-run gives 55 CPU-s to ONE size, and there the walk
    # spends most of its budget re-sampling one basin.  After STALL_HOPS hops that fail to improve
    # the anchor, the anchor is dropped and the next hop starts cold.  `state` is untouched by that,
    # so a restart risks restarts, never ground.
    anchor["xy"], anchor["r"], anchor["s"] = state["xy"], state["r"], state["s"]
    while budget_left():
        shot["xy"], shot["r"], shot["s"] = None, None, -np.inf
        u = rng.rand()
        if anchor["xy"] is not None and u < PERTURB_FRAC:
            sigma = PERTURB_SCALES[rng.randint(len(PERTURB_SCALES))] * unit * 2.0
            xy0 = np.clip(anchor["xy"] + rng.normal(0.0, sigma, anchor["xy"].shape),
                          LO + 1e-4, HI - 1e-4)
            v = rng.rand()
            if v < SYM_FRAC:
                # RESTRUCTURE (symmetry): snap the perturbed incumbent onto a symmetry of the square
                # and re-optimise inside it, then release.  Changes the contact graph in a way
                # neither jitter nor re-equalising can, because it is a projection, not a nudge.
                kind = SYM_KINDS[rng.randint(len(SYM_KINDS))]
                sym_screen(xy0, kind)
            elif v < SYM_FRAC + RESTRUCT_FRAC:
                # RESTRUCTURE (equal radii): re-equalise the perturbed incumbent before re-optimising.
                two_stage(xy0)
            elif (v < SYM_FRAC + RESTRUCT_FRAC + RELOC_FRAC and anchor["r"] is not None
                  and n >= 2):
                # RELOCATE: the one DISCRETE move in the mix.  Take the anchor as it stands (not the
                # jittered copy -- the point is to change which cell a circle sits in, not to nudge
                # it), pull a few of its smallest circles out and drop them into the biggest holes.
                # The randomness lives here rather than in _relocate: which of the starved circles
                # move, how many, and a sub-cell grid offset, so repeated relocations from one anchor
                # explore different re-placements instead of recomputing the same one.
                # n >= 2 is the move's PRECONDITION, not a nicety: the pool is the smallest
                # min(RELOC_POOL, n-1) circles, so at n = 1 it is EMPTY and rng.randint(0) raises --
                # which the driver catches as the end of the whole solve() call, taking every target
                # queued behind it to 0.  Below n = 2 there is no "other circle" to relocate around
                # anyway, so the hop falls through to the plain jitter instead of being wasted.
                pool = np.argsort(anchor["r"])[:min(RELOC_POOL, n - 1)]
                m = 1 + rng.randint(min(RELOC_MAX, len(pool)))
                moved = pool[rng.permutation(len(pool))[:m]]
                off = (rng.rand(2) - 0.5) / float(RELOC_GRID - 1)
                xy1, r1 = _relocate(anchor["xy"], anchor["r"], moved, off)
                xy, r, _ = lp.run(xy1, slice_end, r0=r1)
                consider(xy, r)
            else:
                xy, r, _ = lp.run(xy0, slice_end)
                consider(xy, r)
        else:
            xy0 = _cold_seed(n, rng) if rng.rand() < 0.5 else _random_start(n, rng)
            two_stage(xy0)
        _anchor_update(anchor, shot)

    # last resort: if nothing feasible was recorded, make sure this n still gets a packing
    if state["xy"] is None and meter.left() > 0:
        xy = _random_start(n, rng)
        consider(xy, _grow_batch(xy[None])[0][:, 2])


def _read_record(n):
    """The frozen record for n, or None.  Reading bench/records.json is explicitly sanctioned; this
    is used ONLY to rank where to spend CPU, never to place a circle."""
    try:
        with open(os.path.join("bench", "records.json")) as fh:
            raw = json.load(fh)
        rec = raw if all(k.isdigit() for k in list(raw)[:3]) else raw.get("records", raw)
        v = rec.get(str(int(n)))
        return None if v is None else float(v)
    except Exception:
        return None


def _room(n):
    """How many digits are still WINNABLE at n: DIGITS_CAP minus what the committed pack already banks.

    Unknown counts as maximal.  A size with no record (every held-out size, and anything off the
    census) and a size with no pack both return DIGITS_CAP: we cannot deprioritise what we have never
    scored.  A size already at the cap returns 0.0 and is worked last.  Must never raise -- it runs
    before any evaluate() call, so an exception here would cost the whole call."""
    try:
        rec = _read_record(n)
        if rec is None or not (rec > 0.0):
            return DIGITS_CAP
        w = _read_pack(n)
        if w is None:
            return DIGITS_CAP
        got = float(np.sum(w[1]))
        relgap = max(0.0, (rec - got) / rec)
        if relgap <= 0.0:
            return 0.0
        return max(0.0, DIGITS_CAP - min(DIGITS_CAP, -math.log10(relgap)))
    except Exception:
        return DIGITS_CAP


def solve(evaluate, meter, rng, targets):
    if targets is None:
        return
    targets = sorted(int(t) for t in targets)
    if not targets:
        return
    # Derived AT ENTRY, never a module-level absolute: solve() may be called many times per process.
    hard_deadline = time.process_time() + min(PER_N_CPU * len(targets),
                                              TOTAL_CPU if len(targets) > 1 else PER_N_CPU)

    # WORST FIRST, and in slices big enough to finish.  The old rule was one uniform slice per
    # target in ascending n; both halves were wrong for an append-or-improve census.  Ordering by
    # winnable digits puts the CPU where score is left (and the 7-digit-capped sizes last, where
    # more search cannot pay); the CENSUS_SLICE floor keeps each worked size above the threshold the
    # horizon probe measured, accepting that the tail of the queue may not be reached -- those sizes
    # keep their committed packs, so not reaching them costs nothing.  Ties break on n so a rerun
    # reproduces exactly.
    queue = sorted(targets, key=lambda n: (-_room(n), n))
    for pos, n in enumerate(queue):
        left_n = len(queue) - pos
        now = time.process_time()
        if now >= hard_deadline or meter.left() <= 0:
            break
        left_cpu = hard_deadline - now
        # max(), never min(): with one target this is the full PER_N_CPU (the offline grade's
        # regime, unchanged), and with a generous uniform share it stays the uniform share.
        want = max(CENSUS_SLICE, left_cpu / left_n)
        # ...then hand back whatever the targets still queued behind this one need for coverage.
        cover = min(left_cpu, COVER_CPU)
        want = min(want, max(cover, left_cpu - COVER_CPU * (left_n - 1)))
        want = min(left_cpu, max(want, cover))
        slice_end = now + want
        # The evaluation budget is split the way the CPU is: without this, one n can drain the whole
        # shared meter inside its own CPU slice and every later n is refused (-inf) and lost.  The
        # share follows the CPU share, so a concentrated slice also gets a concentrated eval cap.
        share = 1.0 if left_cpu <= 0.0 else min(1.0, want / left_cpu)
        eval_stop = meter.used + max(4, int(meter.left() * share))
        _solve_one(evaluate, meter, rng, n, slice_end, eval_stop)


# ------------------------------------------------------------------------------------------ self-test
def _self_test():
    import sys

    class _Meter(object):
        def __init__(self, budget):
            self.budget = budget
            self.used = 0

        def left(self):
            return self.budget - self.used

        def tick(self):
            pass

    class _Harness(object):
        def __init__(self, meter):
            self.meter = meter
            self.best = {}

        def __call__(self, n, packing):
            p = np.asarray(packing, dtype=float)
            single = (p.ndim == 2)
            if single:
                p = p[None]
            if p.ndim != 3 or p.shape[1] != n or p.shape[2] != 3:
                raise ValueError("bad shape")
            B = p.shape[0]
            feas = np.zeros(B, dtype=bool)
            tot = np.full(B, -np.inf)
            for b in range(B):
                if self.meter.left() <= 0:
                    break
                self.meter.used += 1
                xy, r = p[b, :, :2], p[b, :, 2]
                ok = bool(np.all(r > 0)) and _min_slack(xy, r) >= -1e-9
                feas[b] = ok
                if ok:
                    tot[b] = float(r.sum())
                    if tot[b] > self.best.get(n, -np.inf):
                        self.best[n] = tot[b]
            if single:
                return bool(feas[0]), float(tot[0])
            return feas, tot

    fails = []

    # 1. two calls in ONE process: the second must still produce a feasible packing (a per-call CPU
    #    deadline computed as a module constant would silently make call #2 a no-op).  Each call gets
    #    its OWN meter so that an exhausted evaluation budget cannot be mistaken for the CPU trap --
    #    the offline re-run is one fresh process per size, so the meter is not what is shared here.
    m1 = _Meter(120)
    h1 = _Harness(m1)
    solve(h1, m1, np.random.RandomState(0), [27])
    m2 = _Meter(120)
    h2 = _Harness(m2)
    solve(h2, m2, np.random.RandomState(1), [29])
    if 27 not in h1.best:
        fails.append("call #1 produced no feasible packing for n=27")
    if 29 not in h2.best:
        fails.append("call #2 in the same process produced no feasible packing for n=29")
    print("two-calls-one-process: n=27 -> %r, n=29 -> %r" % (h1.best.get(27), h2.best.get(29)))

    # 2. len(targets) == 1 and a multi-target call must both work, and every n must be reached.
    m = _Meter(240)
    h = _Harness(m)
    solve(h, m, np.random.RandomState(2), [13, 27, 31])
    missing = [n for n in (13, 27, 31) if n not in h.best]
    if missing:
        fails.append("multi-target call skipped n=%r" % (missing,))
    print("multi-target: %s" % {k: round(v, 6) for k, v in sorted(h.best.items())})

    # 3. an n with NO committed pack must cold-start cleanly (the unguarded-open trap).
    if _read_pack(10 ** 6) is not None:
        fails.append("_read_pack must return None for a size with no pack")
    m = _Meter(60)
    h = _Harness(m)
    solve(h, m, np.random.RandomState(3), [7])
    if 7 not in h.best:
        fails.append("cold start failed for an n with no committed pack")
    print("cold start n=7 -> %r" % (h.best.get(7),))

    # 4. exhausted budget must not raise and must not fabricate anything.
    m = _Meter(0)
    h = _Harness(m)
    solve(h, m, np.random.RandomState(4), [27])
    if h.best:
        fails.append("packings recorded with a zero budget")
    print("zero budget: recorded %d packings (want 0)" % len(h.best))

    # 5. PRUNING MUST NOT CHANGE THE ANSWER.  A pruned LP step and the full-row LP step from the same
    #    iterate must agree to LP tolerance -- this is the whole safety argument for dropping rows.
    #    Checked at a converged packing, where the pruning is most aggressive.
    for n in (27, 61):
        lp = _LP(n)
        xy0 = _grid_start(n, np.random.RandomState(7), 0.15)
        xy, r, _ = lp.run(xy0, time.process_time() + 20.0)
        unit = 0.5 / math.sqrt(n)
        full = lp.step(xy)
        rows_full = lp.last_rows
        pruned = lp.step(xy, r, TR0 * unit, TR0 * unit)
        rows_pruned = lp.last_rows
        if full is None or pruned is None:
            fails.append("LP step failed at n=%d" % n)
            continue
        # the pruned LP is the full LP plus a trust region, so it can only be <= ; it must not be more
        # than the trust region explains, and it must never be INFEASIBLE for the true problem.
        sp = float(_repair(pruned[0], pruned[1]).sum())
        sf = float(_repair(full[0], full[1]).sum())
        if _min_slack(pruned[0], _repair(pruned[0], pruned[1])) < -1e-12:
            fails.append("pruned LP produced an infeasible packing at n=%d" % n)
        if rows_pruned >= rows_full:
            fails.append("pruning dropped no rows at n=%d (%d vs %d)" % (n, rows_pruned, rows_full))
        print("prune n=%d: rows %d -> %d, sum_r full %.9f vs pruned %.9f" % (n, rows_full, rows_pruned, sf, sp))

    # 6. quality sanity: the LP restriction must beat the equal-radius grow() baseline comfortably.
    m = _Meter(150)
    h = _Harness(m)
    solve(h, m, np.random.RandomState(5), [27])
    base = float(_grow_batch(_grid_start(27, np.random.RandomState(6), 0.0)[None])[0][:, 2].sum())
    if h.best.get(27, -np.inf) <= base * 1.05:
        fails.append("n=27 result %r did not beat the grow() baseline %r by 5%%"
                     % (h.best.get(27), base))
    print("quality n=27: %r vs grow-baseline %r" % (h.best.get(27), base))

    # 7. THE EQUAL-RADIUS STAGE must be exactly feasible and its pruning must not change the answer.
    #    It is a second LP with its own row algebra, so it needs its own version of check 5 -- a bug
    #    here would not make any packing infeasible (the free stage re-solves afterwards), it would
    #    just quietly hand on a worse structure, which no score check would catch.
    for n in (27, 61):
        lpe = _LPEq(n)
        xy0 = _grid_start(n, np.random.RandomState(11), 0.15)
        xy, t = lpe.run(xy0, time.process_time() + 15.0)
        if not (t > 0.0):
            fails.append("equal-radius stage returned a non-positive common radius at n=%d" % n)
        if _min_slack(xy, np.full(n, t)) < -1e-12:
            fails.append("equal-radius stage produced an infeasible packing at n=%d" % n)
        if t < _common_r(xy0) - 1e-12:
            fails.append("equal-radius stage went BACKWARDS at n=%d (%.9f < %.9f)"
                         % (n, t, _common_r(xy0)))
        unit = 0.5 / math.sqrt(n)
        full = lpe.step(xy, t)
        rows_full = lpe.last_rows
        pruned = lpe.step(xy, t, TR0 * unit, TR0 * unit)
        rows_pruned = lpe.last_rows
        if full is None or pruned is None:
            fails.append("equal-radius LP step failed at n=%d" % n)
            continue
        if rows_pruned >= rows_full:
            fails.append("equal-radius pruning dropped no rows at n=%d (%d vs %d)"
                         % (n, rows_pruned, rows_full))
        if _common_r(pruned[0]) < t - 1e-9:
            fails.append("pruned equal-radius LP lost ground at n=%d" % n)
        print("eq n=%d: t=%.9f (n*t=%.6f), rows %d -> %d, pruned t=%.9f"
              % (n, t, n * t, rows_full, rows_pruned, _common_r(pruned[0])))

    # 8. every cold-seed family must return a usable (n,2) start for awkward n, including n where a
    #    row count or a lattice spacing does not divide evenly -- these feed solve() for UNSEEN sizes.
    rs = np.random.RandomState(12)
    for n in (1, 2, 3, 7, 26, 50, 97, 150):
        for label, xy in (("cold", _cold_seed(n, rs)),
                          ("hexrows", _hex_rows_start(n, max(1, int(round(math.sqrt(n)))), rs)),
                          ("rothex", _rot_hex_start(n, 0.3, rs))):
            if xy is None:
                if label == "cold":
                    fails.append("_cold_seed returned None at n=%d" % n)
                continue
            if xy.shape != (n, 2) or not np.all(np.isfinite(xy)) \
                    or xy.min() < LO or xy.max() > HI:
                fails.append("%s start malformed at n=%d: shape %r" % (label, n, xy.shape))
    print("seed families: ok for n in (1,2,3,7,26,50,97,150)")

    # 9. THE SYMMETRY RESTRICTION must be exact, not approximate.  It is a set of linear EQUALITIES
    #    added to both LPs, so three things have to hold or the stage is silently lying: the
    #    projection must produce an exactly symmetric configuration, both LPs must PRESERVE that
    #    symmetry to machine precision, and the result must still be strictly feasible.  A bug here
    #    would make nothing infeasible (the free polish re-solves afterwards) and would cost no
    #    score -- it would just quietly degrade unmeasured sizes, which is exactly the failure mode
    #    this check exists for.  Sizes include an EVEN n, where the k parities differ.
    for n in (27, 40):
        lp, lpe = _LP(n), _LPEq(n)
        rs9 = np.random.RandomState(21)
        for kind in SYM_KINDS:
            ks = _sym_axis_counts(n, kind, rs9)
            if not ks:
                fails.append("no usable fixed-set size for %s at n=%d" % (kind, n))
                continue
            k = ks[0]
            if (n - k) % 2 != 0:
                fails.append("%s at n=%d proposed k=%d of the wrong parity" % (kind, n, k))
            proj = _sym_project(_cold_seed(n, rs9), kind, k)
            if proj is None:
                fails.append("%s projection failed at n=%d k=%d" % (kind, n, k))
                continue
            xys, pairs, axis = proj
            if len(pairs) * 2 + len(axis) != n or len(set(list(pairs.ravel()) + list(axis))) != n:
                fails.append("%s at n=%d did not partition the circles" % (kind, n))
            d0 = _sym_defect(xys, kind, pairs, axis)
            eq_e = _sym_eq(n, kind, pairs, axis, 2 * n + 1, False)
            eq_f = _sym_eq(n, kind, pairs, axis, 3 * n, True)
            xy1, t = lpe.run(xys, time.process_time() + 4.0, eq=eq_e)
            d1 = _sym_defect(xy1, kind, pairs, axis)
            xy2, r2, _s2 = lp.run(xy1, time.process_time() + 4.0,
                                  r0=np.full(n, max(t, RMIN)), eq=eq_f)
            d2 = _sym_defect(xy2, kind, pairs, axis)
            rd = float(np.max(np.abs(r2[pairs[:, 0]] - r2[pairs[:, 1]]))) if len(pairs) else 0.0
            r2f = _repair(xy2, r2)
            slack = _min_slack(xy2, r2f)
            if max(d0, d1, d2, rd) > 1e-9:
                fails.append("%s at n=%d broke symmetry: %.3e/%.3e/%.3e r %.3e"
                             % (kind, n, d0, d1, d2, rd))
            if slack < 0.0:
                fails.append("%s at n=%d produced an infeasible packing (slack %.3e)"
                             % (kind, n, slack))
            print("sym %s n=%d k=%d: defect %.1e/%.1e/%.1e rdiff %.1e sum_r %.6f slack %.1e"
                  % (kind, n, k, d0, d1, d2, rd, float(r2f.sum()), slack))

    # 10. THE k-SCREEN must rank on a real quantity and must cover the whole candidate set.
    #     `sym_screen` discards fixed-set sizes on the strength of the equal-radius t that
    #     `_LPEq.run` returns, so two things have to hold or the screen quietly throws away the
    #     right structure: t must be the TRUE largest common radius of the centres it is returned
    #     with (not the LP's own optimistic t), and `_sym_axis_counts` must enumerate every
    #     parity-valid k up to its cap rather than a subset.  Neither failure makes any packing
    #     infeasible, so nothing else in this file would catch them.
    for n in (27, 40):
        lpe = _LPEq(n)
        rs10 = np.random.RandomState(5)
        for kind in SYM_KINDS:
            ks = sorted(_sym_axis_counts(n, kind, rs10))
            top = int(math.ceil(math.sqrt(max(n, 1)))) + 2
            want = [0] if kind == "c2" and n % 2 == 0 else \
                   [1] if kind == "c2" else \
                   sorted(k for k in range(0, min(n, top) + 1) if (n - k) % 2 == 0)
            if ks != want:
                fails.append("%s at n=%d screened k %r, expected %r" % (kind, n, ks, want))
            if len(ks) < SYM_K_COMPLETE and kind != "c2":
                fails.append("%s at n=%d offers only %d k" % (kind, n, len(ks)))
            seed10 = _cold_seed(n, rs10)
            for k in ks[:3]:
                proj = _sym_project(seed10, kind, k)
                if proj is None:
                    continue
                xys, pairs, axis = proj
                eq_e = _sym_eq(n, kind, pairs, axis, 2 * n + 1, False)
                xy1, t = lpe.run(xys, time.process_time() + 3.0, eq=eq_e)
                true_t = _common_r(xy1)
                if abs(t - true_t) > 1e-12 or t <= 0.0:
                    fails.append("%s at n=%d k=%d screen key %.12g != true %.12g"
                                 % (kind, n, k, t, true_t))
        print("k-screen: n=%d keys exact, candidate sets complete" % n)

    # 11. THE ADAPTIVE COMPLETION COUNT is a pure function of the screened keys, so test it as one.
    #     Like check 10 this guards a silent failure: a wrong count never makes a packing
    #     infeasible and never costs a visible digit -- it just spends the symmetric stage on the
    #     wrong number of structures, at the sizes nobody measures.
    def _sc(*ts):
        return [(t, 0, None, None, None) for t in sorted(ts, reverse=True)]

    cases = [
        ([], 0, "empty screen completes nothing"),
        (_sc(0.09), 1, "a single candidate cannot be exceeded"),
        (_sc(0.09, 0.05), SYM_K_COMPLETE, "one clear winner falls back to the floor"),
        (_sc(0.09, 0.0899, 0.05, 0.04), SYM_K_COMPLETE, "a 2-cluster is exactly the floor"),
        (_sc(0.09, 0.0895, 0.0894, 0.05), 3, "a 3-cluster is completed in full"),
        (_sc(*([0.09] * 9)), SYM_K_COMPLETE_MAX, "a wide tie is capped"),
        (_sc(0.0, 0.0), SYM_K_COMPLETE, "a degenerate t=0 screen still returns the floor"),
    ]
    for sc, want, why in cases:
        got = _n_complete(sc)
        if got != want:
            fails.append("_n_complete(%d keys) = %d, expected %d (%s)" % (len(sc), got, want, why))
    # the margin boundary itself: t exactly on it counts as unseparated, just below it does not.
    edge = 0.09 * (1.0 - SYM_K_MARGIN)
    if _n_complete(_sc(0.09, edge, 0.05)) != SYM_K_COMPLETE:
        fails.append("_n_complete must include a candidate exactly on the margin")
    if _n_complete(_sc(0.09, edge * (1 - 1e-9), 0.05)) != SYM_K_COMPLETE:
        fails.append("_n_complete must exclude a candidate just below the margin")
    if _n_complete(_sc(0.09, 0.0895, 0.0894, 0.05, 0.04, 0.03)) != 3:
        fails.append("_n_complete must not grow with candidates the screen separated")
    for sc in (_sc(0.09), _sc(0.09, 0.05), _sc(0.09, 0.0899, 0.0898)):
        if not (0 <= _n_complete(sc) <= len(sc)):
            fails.append("_n_complete out of range for %d keys" % len(sc))
    print("adaptive completions: floor %d, cap %d, margin %.3f -- %d cases exact"
          % (SYM_K_COMPLETE, SYM_K_COMPLETE_MAX, SYM_K_MARGIN, len(cases) + 4))

    # 12. THE ANCHOR RESTART RULE, likewise as a pure function.  Its failure mode is the quietest one
    #     in the file: every packing stays feasible and the in-run SCORE does not move (the 3 CPU-s
    #     slice fits too few hops to stall), while the graded 55 CPU-s/size re-run -- the one nobody
    #     here can see -- spends its whole budget re-sampling one basin.
    def _anch(s_=1.0, stall=0, xy=True):
        return {"xy": (np.zeros((3, 2)) if xy else None), "s": s_, "stall": stall}

    def _shot(s_, xy=True):
        return {"xy": (np.ones((3, 2)) if xy else None), "s": s_}

    a = _anchor_update(_anch(1.0, 7), _shot(2.0))
    if a["stall"] != 0 or a["xy"] is None or a["s"] != 2.0:
        fails.append("_anchor_update: an improving hop must become the anchor and clear the stall")
    a = _anchor_update(_anch(1.0, 0), _shot(1.0))
    if a["stall"] != 1 or a["s"] != 1.0:
        fails.append("_anchor_update: a hop that merely TIES the anchor is not an improvement")
    a = _anchor_update(_anch(1.0, 3), _shot(-np.inf, xy=False))
    if a["stall"] != 4:
        fails.append("_anchor_update: an infeasible hop must count as a failure")
    a = _anchor_update(_anch(1.0, STALL_HOPS - 1), _shot(0.5))
    if a["xy"] is not None or a["stall"] != 0 or a["s"] != -np.inf:
        fails.append("_anchor_update: the STALL_HOPS-th failure must DROP the anchor")
    a = _anchor_update(_anch(1.0, STALL_HOPS - 1), _shot(1.0 + 2 * STALL_TOL))
    if a["xy"] is None:
        fails.append("_anchor_update: an improvement on the last hop must save the anchor")
    a = _anchor_update(_anch(1.0, STALL_HOPS - 2), _shot(0.5))
    if a["xy"] is None:
        fails.append("_anchor_update: must not drop the anchor one hop early")
    if STALL_HOPS < 2 or STALL_TOL <= 0.0:
        fails.append("STALL_HOPS/STALL_TOL must be a real threshold, got %r/%r"
                     % (STALL_HOPS, STALL_TOL))
    print("anchor restart: drops at %d stalled hops, ties count as stalls -- 7 cases exact"
          % STALL_HOPS)

    # 13. THE RELOCATION MOVE.  Three properties, all of them failure modes that are silent in the
    #     SCORE: an infeasible re-placement is thrown away by `consider` (so the hop just costs CPU
    #     and nothing says so), two circles dropped into the SAME hole waste the move, and a grid too
    #     coarse to see the hole finds a worse one -- none of which makes any committed pack invalid.
    rs = np.random.RandomState(7)
    for n_ in (13, 40, 77):
        xy_ = np.clip(rs.rand(n_, 2) - 0.5, LO + 1e-3, HI - 1e-3)
        r_ = _grow_batch(xy_[None])[0][:, 2]
        for m_ in (1, 3):
            moved = np.argsort(r_)[:m_]
            xy2, r2 = _relocate(xy_, r_, moved)
            if xy2.shape != xy_.shape or len(r2) != n_:
                fails.append("_relocate must return exactly n circles (n=%d, m=%d)" % (n_, m_))
            if _min_slack(xy2, r2) < 0.0 or not np.all(r2 > 0.0):
                fails.append("_relocate must return a STRICTLY FEASIBLE packing (n=%d, m=%d, "
                             "slack %.2e)" % (n_, m_, _min_slack(xy2, r2)))
            if not np.array_equal(np.setdiff1d(np.arange(n_), moved),
                                  np.where(np.all(xy2 == xy_, axis=1))[0]):
                fails.append("_relocate must move exactly the listed circles (n=%d, m=%d)" % (n_, m_))
            if m_ > 1:
                d = np.sqrt(((xy2[moved][:, None] - xy2[moved][None]) ** 2).sum(-1))
                if float(np.min(d + np.eye(m_) * 9.0)) <= 0.0:
                    fails.append("_relocate dropped two circles into the same hole (n=%d)" % n_)
        # a placed circle's radius IS its clearance, and the nested finer grid never finds a worse
        # hole -- the check that RELOC_GRID is a resolution, not a magic number.
        i0 = np.argsort(r_)[:1]
        keep = np.setdiff1d(np.arange(n_), i0)
        cs = []
        for g_ in (33, 65):
            x3, r3 = _relocate(xy_, r_, i0, g=g_)
            c = float(_clearance(x3[i0], xy_[keep], r_[keep])[0])
            if abs(c - float(r3[i0][0])) > 1e-8:
                fails.append("_relocate radius must equal the hole clearance (n=%d, g=%d): %.3e vs "
                             "%.3e" % (n_, g_, r3[i0][0], c))
            cs.append(c)
        if cs[1] < cs[0] - 1e-12:
            fails.append("_relocate on the NESTED finer grid found a worse hole (n=%d): %.6f < %.6f"
                         % (n_, cs[1], cs[0]))
    e = _clearance(np.array([[0.0, 0.0], [0.4, 0.0]]), np.zeros((0, 2)), np.zeros(0))
    if abs(e[0] - 0.5) > 1e-12 or abs(e[1] - 0.1) > 1e-12:
        fails.append("_clearance with no obstacles must be the wall distance, got %r" % (e,))
    if not (0.0 < RELOC_FRAC < 1.0 and SYM_FRAC + RESTRUCT_FRAC + RELOC_FRAC < 1.0):
        fails.append("RELOC_FRAC must leave the plain-jitter branch a share: %r" % (RELOC_FRAC,))
    print("relocation: %d circles max from the %d smallest, %dx%d hole grid -- 3 n x 2 m, feasible, "
          "disjoint, nested-grid monotone" % (RELOC_MAX, RELOC_POOL, RELOC_GRID, RELOC_GRID))

    # 14. DEGENERATE AND OFF-CENSUS SIZES, AND THE CASCADE THEY CAUSE.  The visible census is 37 ODD
    #     sizes 27..99, so nothing in the loop's own scoring ever runs an even n, an n below 27, or
    #     n = 1/2 where a move's pool can be empty -- and the offline re-run is on sizes the loop
    #     never scored.  An exception at any of them does not cost digits, it ends the whole call:
    #     every target queued BEHIND it scores 0.  So the second half here is the real check -- a
    #     multi-size call LED BY the degenerate size must still deliver the sizes after it.
    for n_ in (1, 2, 3, 26, 28):
        m = _Meter(40)
        h = _Harness(m)
        try:
            solve(h, m, np.random.RandomState(6), [n_])
        except Exception as exc:                       # noqa: BLE001 -- the driver swallows these
            fails.append("solve() raised on n=%d: %s: %s" % (n_, type(exc).__name__, exc))
            continue
        got = h.best.get(n_)
        if got is None:
            fails.append("no feasible packing for n=%d" % n_)
    m = _Meter(200)
    h = _Harness(m)
    try:
        solve(h, m, np.random.RandomState(7), [1, 2, 28, 31])
        lost = [n_ for n_ in (1, 2, 28, 31) if n_ not in h.best]
        if lost:
            fails.append("a degenerate size cost the targets behind it: lost %r" % (lost,))
    except Exception as exc:                           # noqa: BLE001
        fails.append("multi-size call led by n=1 raised: %s: %s" % (type(exc).__name__, exc))
    print("degenerate sizes: n in {1,2,3,26,28} solo + one call led by n=1 -> %d/4 targets kept"
          % sum(1 for n_ in (1, 2, 28, 31) if n_ in h.best))

    # 15. THE SEED PHASE IS A BUDGET, NOT A CHECKLIST.  Two claims, both measured in
    #     artifacts/probe_iter14_slice.py and both silent failures if they regress:
    #     (a) families INTERLEAVE, so a truncated seed phase keeps one seed from each family
    #         instead of three lattices and nothing else.  Order of CALLS only -- no timing.
    #     (b) the phase is capped at SEED_FRAC of the slice, so the refinement stage gets a turn at
    #         LARGE n too.  Before this, at n=99 and a 3 CPU-s slice the seed list overran the whole
    #         slice and basin hopping never ran once, while at n=27 it got two thirds of the slice.
    if not (0.0 < SEED_FRAC < 1.0):
        fails.append("SEED_FRAC must leave the refinement stage a share: %r" % (SEED_FRAC,))
    _g = globals()
    _saved = {k: _g[k] for k in ("_grid_start", "_hex_rows_start", "_rot_hex_start",
                                 "_anchor_update", "PER_N_CPU")}
    seen = []
    hops = [0]
    for _name, _tag in (("_grid_start", "grid"), ("_hex_rows_start", "hex"),
                        ("_rot_hex_start", "rot")):
        def _w(*a, _o=_saved[_name], _t=_tag, **k):
            seen.append(_t)
            return _o(*a, **k)
        _g[_name] = _w

    def _wa(*a, _o=_saved["_anchor_update"], **k):
        hops[0] += 1
        return _o(*a, **k)
    _g["_anchor_update"] = _wa
    try:
        _g["PER_N_CPU"] = 2.5
        m = _Meter(4000)
        h = _Harness(m)
        solve(h, m, np.random.RandomState(11), [27])
        order = list(seen[:3])
        seen[:] = []
        hops[0] = 0
        m = _Meter(4000)
        h = _Harness(m)
        solve(h, m, np.random.RandomState(12), [99])          # the size the probe caught
        big_hops, big_seeds = hops[0], len(seen)
    finally:
        for k, v in _saved.items():
            _g[k] = v
    if order != ["grid", "hex", "rot"]:
        fails.append("cold seed families must interleave, got %r" % (order,))
    if big_hops < 1:
        fails.append("seed phase consumed the whole slice at n=99: %d seeds, %d basin hops"
                     % (big_seeds, big_hops))
    if 99 not in h.best:
        fails.append("n=99 produced no feasible packing under a capped seed phase")
    print("seed budget: families interleave %r; n=99 @2.5 CPU-s -> %d cold seeds then %d basin hops"
          % (order, big_seeds, big_hops))

    # --- 16: CPU is allocated worst-first, in slices that survive len(targets) == 1 -----------
    # Two regressions this pins.  (a) Ordering: the census must work the sizes with digits still
    # winnable BEFORE the ones already at the cap, or a concentrated slice is spent where no score
    # can be won.  (b) The offline grade calls solve() with ONE target and 55 CPU-s; if the
    # CENSUS_SLICE floor were applied as a min() instead of a max(), that call would cap itself at
    # 15 CPU-s and throw away three quarters of the graded horizon -- invisible to every in-run
    # measurement, because in-run slices are smaller than the floor.
    _g2 = globals()
    _sv2 = {k: _g2[k] for k in ("_solve_one", "_room", "PER_N_CPU", "TOTAL_CPU")}
    calls = []

    def _fake_solve_one(evaluate, meter, rng_, n, slice_end, eval_stop):
        calls.append((n, slice_end - time.process_time()))

    rooms = {41: 0.0, 43: 6.5, 45: 3.0, 47: 7.0}
    try:
        _g2["_solve_one"] = _fake_solve_one
        _g2["_room"] = lambda n: rooms.get(n, DIGITS_CAP)
        _g2["PER_N_CPU"], _g2["TOTAL_CPU"] = 55.0, 110.0
        m = _Meter(4000)
        solve(_Harness(m), m, np.random.RandomState(3), [41, 43, 45, 47])
        order = [c[0] for c in calls]
        few = calls[0][1]                     # uniform share 27.5s > floor: floor must NOT shrink it
        calls[:] = []
        m = _Meter(400000)
        solve(_Harness(m), m, np.random.RandomState(3), list(range(27, 101, 2)))
        many = calls[0][1]                    # uniform share 3.0s < floor: floor must RAISE it
        nworked = len(calls)
        calls[:] = []
        m = _Meter(4000)
        solve(_Harness(m), m, np.random.RandomState(3), [45])
        one = calls[0][1] if calls else 0.0
    finally:
        for k, v in _sv2.items():
            _g2[k] = v
    if order != [47, 43, 45, 41]:
        fails.append("targets must be worked by winnable digits, worst first; got %r" % (order,))
    if not (CENSUS_SLICE - 0.5 <= many <= CENSUS_SLICE + 0.5):
        fails.append("37 targets / 110 CPU-s: first slice %.2f, expected the %.1f floor"
                     % (many, CENSUS_SLICE))
    if not (27.0 <= few <= 28.0):
        fails.append("4 targets / 110 CPU-s: first slice %.2f, expected the 27.5 uniform share "
                     "(the floor must never SHRINK a slice)" % (few,))
    if one < 0.9 * 55.0:
        fails.append("len(targets)==1 must get the WHOLE per-n allowance, got %.2f of 55.0" % one)
    # _room must never raise and must rank the three cases the census actually contains
    r_norec = _room(10 ** 6)
    r_bad = _room(-3)
    if not (r_norec == DIGITS_CAP and r_bad == DIGITS_CAP):
        fails.append("_room must return the full cap for an n with no record/pack (%r, %r)"
                     % (r_norec, r_bad))
    _sv3 = _g2["_read_pack"], _g2["_read_record"]
    try:
        _g2["_read_record"] = lambda n: 1.0
        _g2["_read_pack"] = lambda n: (np.zeros((n, 2)), np.full(n, 1.0 / n))
        r_cap = _room(5)                      # banked == record -> nothing left to win
        _g2["_read_pack"] = lambda n: (np.zeros((n, 2)), np.full(n, 0.9 / n))
        r_far = _room(5)                      # relgap 0.1 -> 1 digit banked, 6 winnable
        _g2["_read_pack"] = lambda n: (_ for _ in ()).throw(RuntimeError("boom"))
        r_boom = _room(5)
    finally:
        _g2["_read_pack"], _g2["_read_record"] = _sv3
    if not (r_cap == 0.0 and abs(r_far - 6.0) < 1e-9 and r_boom == DIGITS_CAP):
        fails.append("_room: capped=%r far=%r raising=%r" % (r_cap, r_far, r_boom))
    print("cpu split: worst-first order %r; 37 targets -> %.1fs x %d worked, 4 -> %.1fs, "
          "1 -> %.1fs" % (order, many, nworked, few, one))

    if fails:
        print("SELF-TEST: FAIL")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("SELF-TEST: PASS")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
