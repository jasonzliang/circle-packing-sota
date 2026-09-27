"""Circle-packing solver: sum-of-radii maximization in the centered unit square.

STRATEGY (why this shape)
------------------------
For FIXED centers, `max sum(r) s.t. r_i+r_j <= d_ij, r_i <= wall_i(x_i,y_i)` is a *linear
program*.  Letting the centers move too, the only nonlinearity is `d_ij(p)`, which is a CONVEX
function of the centers.  Linearizing it,

    d_ij(p+dp) >= d_ij(p) + u_ij . (dp_i - dp_j),      u_ij = (p_i-p_j)/d_ij

is therefore a *conservative* (inner) approximation: any point satisfying the linearized
constraint satisfies the true one.  So each SLP iteration solves an LP in (dx, dy, r) over a
trust region |dp| <= delta and returns a point that is genuinely feasible, never worse than the
incumbent (dp=0, r=r_cur is always LP-feasible).  That gives a *monotone, always-feasible* local
optimizer (`_local_opt`) that converges to a KKT point of the real problem as delta shrinks.

Iteration-1 measurement: precision is NOT the bottleneck -- SLP reaches ~1e-10 positional
accuracy while the gap to the packomania record is ~5e-3 relative.  The bottleneck is *which
local optimum*, and the number of basin-hopping moves affordable inside the CPU cap is small
(~8 per n at 100 CPU-s over 37 n).  So iteration 2 spends the budget differently:

  * the census warm start is the incumbent, and when it is an EXACT warm start the cold restarts
    are dropped entirely -- a cold start lands ~1.7 digits and can no longer beat a committed
    ~2.3-digit packing, so those local optimizations were pure waste;
  * the trust region stops at 1e-8 instead of 1e-11 (1e-8 in position is ~1e-8 in sum_r, i.e. two
    orders below the 1e-7 relgap that already earns the 7-digit cap), which removes ~6 of ~30 LP
    solves per local optimization;
  * the move set is structure-preserving and richer (single-smallest-to-best-hole, k-smallest to
    holes, small global jitter, local cluster shake, large/small position swap, random teleport),
    with threshold accepting on a `current` state that is reset to `best` after a stall;
  * an n with NO committed pack of its own is seeded by STRUCTURE TRANSFER from the nearest
    committed n (drop the smallest circles, or insert extra ones into the largest holes).  That
    is what an offline re-run on unseen sizes gets, so it is the path that must not be weak.

Iteration-3 measurement: profiling one SLP (`artifacts/prof1.py`) showed the objective stops
moving at step 1-3 after a basin hop, while the loop kept solving ~15 more LPs purely to grind the
trust region down to the 1e-8 floor -- ~2/3 of every local optimization bought < 1e-11 of sum_r.
So `_local_opt` now stops on a STALL (`MAX_FAIL` consecutive non-improving steps) as well as on
the floor, and the pair-pruning margin is the geometrically correct `2.2*delta` instead of
`4.0*delta`.  Together: 2.4x more basin hops per CPU-second, and at equal CPU
(`artifacts/prof3.py`, n=33/55/73/91) mean digits 2.212 -> 2.341.

CONTRACT NOTES
--------------
* All deadlines are computed from `time.process_time()` AT ENTRY to `solve()`, so a second call
  in the same process gets its own full budget (module-level absolute constants would be spent by
  the first call).  `--self-test` calls `solve()` twice and checks the second call still delivers.
* Every read of `bench/packs/` is guarded by `os.path.exists`; every cold-start path works on its
  own for any n, including n with no committed pack and n outside the census.
* Nothing is ever written to `bench/packs/`; every packing that should count is routed through
  `evaluate()`.
* Randomness comes only from the supplied `rng`.
* The per-n budget split is a fraction of the whole, so it is correct for `len(targets) == 1`.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

LO, HI = -0.5, 0.5

# Total process-CPU this solver allows itself per solve() call.  The driver's backstop is 120 s;
# stopping at 100 s leaves room for the driver's own re-validation of every pack and for the
# evaluate() calls, so solve() returns cleanly instead of being interrupted.
CPU_BUDGET_S = 100.0
# --- allocation (iteration 4).  There is no per-n time slice any more; see solve().
DIGITS_CAP = 7.0        # the scorer's per-n cap: relgap <= 1e-7 already earns full marks
# Optimism prior for an n that has delivered nothing yet, in (digits, CPU-seconds).  The ratio
# 0.02/0.5 = 0.04 digits/s is deliberately set near the measured whole-census rate (iteration 3
# closed ~2.8 mean digits in 100 CPU-s over 37 n), so a fresh n neither swamps nor is swamped by
# an n with a real track record.
PRIOR_GAIN = 0.02
PRIOR_CPU = 0.5
# Delivered digits are decayed each time an n is picked, so the estimate tracks RECENT
# productivity: 0.85 gives an effective window of ~7 hops.
GAIN_DECAY = 0.85
# Hard ceiling on one basin hop, so a pathological SLP cannot swallow the remaining budget
# before the allocator gets to re-decide.
HOP_CPU_CAP = 4.0
# Ceiling on the seeding pass's share of the remaining CPU for any single n.
SEED_FRAC = 0.25
# Fallback scaling-law coefficients for ref(n) = a*sqrt(n)+b, used only if bench/records.json is
# unreadable.  Refit from whatever records ARE readable at every call (see _ref_curve).
_REF_FIT = (0.53496472, -0.08964591)
SHRINK = 1e-12          # radius safety margin (feas tol is 1e-9; this costs <1e-10 of sum_r)
DELTA_MIN = 1e-8        # trust-region floor: 1e-8 in position is far below the 1e-7 relgap cap
DELTA0 = 0.06           # initial trust region
# A center moves at most `delta` per SLP step, so a pair's separation shrinks by at most
# 2*delta; keeping pairs out to 2.2*delta + 1e-3 therefore covers every constraint that can
# become active (the +1e-3 absorbs radius growth).  The old 4.0*delta kept ~2.4x as many pair
# rows at the large deltas where the LP is most expensive, for no measured quality gain.
PRUNE_K = 2.2
# The same geometric margin applied to the 4n WALL rows: `wall_i - r_i` shrinks by at most
# `delta` from the move plus whatever r_i gains, so a wall row further away than
# `PRUNE_K*delta + 1e-3` cannot bind this step.  Measured (artifacts/prof9.py): 5-8% fewer CPU
# per hop at n=35..99 with the objective identical to 9 decimals.  As with the pair prune, a
# wrong prune can only cost quality -- _repair() restores exact feasibility unconditionally.
# HiGHS algorithm choice.  Measured head-to-head over 6 hops at n=27..99 (artifacts/prof8.py),
# same objective to 9 decimals in every case, CPU ratio default/ipm:
#     n   27    35    43    51    59    67    75    83    91    99
#   x     1.09  0.95  1.10  1.17  1.32  1.60  1.61  2.06  2.03  2.31
# Interior point wins by a widening margin as the LP grows and is a wash below n~50, so switch
# on LP size (3n variables) rather than on n itself -- the rule then carries to any n an offline
# re-run hands us.  `highs` (dual simplex) stays the fallback if IPM ever fails to converge.
IPM_MIN_VARS = 150
# Stop an SLP once it has stalled: 4 consecutive non-improving steps means the trust region has
# already shrunk 175x with nothing to show.  Measured (artifacts/prof1.py): after a basin hop the
# objective stops moving at step 1-3 but the old delta-floor-only loop ground out 15 more LP
# solves, i.e. ~2/3 of every local optimization was spent below 1e-12 of sum_r.
MAX_FAIL = 4
# Consecutive non-improving jitter hops after which the chain switches to the structural
# inflate-relax escape move (see _inflate_relax).  Measured in iteration 5: as one of the random
# moves it is a net LOSS (it dilutes the cheap jitter moves that do still pay); fired only once
# the jitter moves have demonstrably stopped working, it is free.
ESCAPE_STALL = 3
# CHAIN ACCEPTANCE (iteration 14).  Threshold accepting: a hop whose optimum is within `band`
# (relative) below the chain's current point is followed anyway, so the chain can cross a ridge.
# Iteration 13 left the band at a hard-coded 2e-4, and `artifacts/prof23_step.py` measured why
# that pins the chain: over 700+ hops at the stuck n 35/45/81, across five perturbation scales
# spanning 100x, a hop either returns to the IDENTICAL optimum (~40%) or lands in a basin whose
# median relative loss is 1e-3..1e-2 -- i.e. 5-50x OUTSIDE a 2e-4 band.  So essentially every
# hop that actually moved was refused by the chain, `stall` hit the snapback every 8 hops, and
# 200 hops per n degenerated into 200 independent restarts from the one incumbent.  The band is
# now ADAPTED per n toward ACCEPT_TARGET chain-acceptance (standard adaptive threshold accepting)
# instead of guessed: it starts at the old value, so nothing is loosened a priori, and it is
# capped so the chain can never wander into a region it cannot come back from.  `best` is never
# touched by this, so a wider band cannot regress a committed packing -- only the walk changes.
ACCEPT_BAND0 = 2e-4     # initial/minimum band == iteration 13's fixed value
ACCEPT_BAND_MAX = 2e-2  # cap: ~20x the worst remaining relgap, well inside the cold-start regime
# Iteration 15, MEASURED (artifacts/prof24_accept.py, 7 stuck n, paired seeds, 26 CPU-s/config):
# 0.5 was inherited from Metropolis practice, not measured, and it was still the binding gate --
# iteration 14's bands converged to 5e-4..3.5e-3 because a 50% follow-rate stopped them there,
# while the alternative optima a hop actually reaches sit 1e-3..1e-2 below the incumbent.  At
# target 0.9 the same machinery took n=39 and n=57 to the 7-digit cap inside the probe (+6.69
# digits, vs +0.31 at 0.5); 0.75 and 0.97 won nothing on their draws, and a SECOND paired seed
# won nothing at ANY target -- so these gains are rare events and this constant is chosen on ONE
# favourable draw out of two.  It is safe to take that bet in only one direction: `best` is never
# touched by the band, and the driver keeps the committed pack unless it is strictly beaten, so
# the worst case of a loose gate is wasted CPU, never a regression.
ACCEPT_TARGET = 0.9     # target fraction of hops the chain follows
ACCEPT_ETA = 0.25       # multiplicative adaptation rate per hop
# Non-improving hops before the chain is snapped back to this n's best.  `stall` counts hops that
# did not beat BEST, which for a walking chain is almost all of them -- at 8 the chain was hauled
# back to the incumbent before it could travel anywhere.  A separate, longer horizon lets an
# excursion run while ESCAPE_STALL keeps its short horizon for switching move class.
SNAPBACK = 30
# GRAFT (iteration 7).  How far away a donor size may be when grafting another n's structure onto
# this n.  The census is odd-n only, so offsets 1..8 reach n+-2, n+-4, n+-6, n+-8; on an offline
# re-run over consecutive sizes it reaches n+-1.. as well.  Donors further than this differ by
# enough circles that the adaptation (drop/insert) destroys the structure being borrowed.
GRAFT_MAX_OFF = 8
# Share of the graft escapes that use the half-plane PATCH crossover rather than a whole-donor
# graft (iteration 8).  See _graft_patch.
PATCH_FRAC = 0.5
# SELF-CROSSOVER (iteration 9).  Both graft moves borrow from a DIFFERENT size, so every
# borrowed circle has to be adapted (drop-smallest / insert-into-holes) to fix the count, and
# iteration 7's honest limit (b) was that this adaptation destroys much of what is borrowed.  Two
# distinct local optima of the SAME n splice with ZERO adaptation -- a rank-based half-plane cut
# takes k circles from one parent and exactly n-k from the other -- so the only damage is the
# seam.  It also needs no donor at all, which is why it is tried even when the census has no
# neighbour for this n (a fresh size, or an offline re-run on sizes outside the census).
XOVER_ARCH = 6          # how many distinct local optima per n are kept as crossover parents
XOVER_WINDOW = 0.02     # a parent must be within this RELATIVE distance of this n's best
XOVER_DISTINCT = 1e-7   # two parents whose sum_r agree to this (relative) are the same basin
XOVER_FRAC = 0.5        # share of the graft escapes that use self-crossover instead (LEGACY:
                        # kept only so artifacts/prof13_xover.py still reproduces; the live
                        # escape mix is now chosen by _MoveBank, not by this constant)
# ESCAPE BANDIT (iteration 10).  The escape mix used to be three hand-picked constants -- an
# odd/even alternation between inflate-relax and the grafts, then PATCH_FRAC, then XOVER_FRAC --
# and re-reading the run traces showed what that cost.  Digits actually delivered per CPU-second,
# summed over the iteration-7/8/9 metered runs (artifacts/movebank_diag.txt, derived from
# artifacts/alloc_trace.jsonl):
#
#     graft  35.8 cpu-s -> 27.60 digits   0.772 d/s      <-- the productive escape
#     jitter 151.9      -> 24.57          0.162
#     inflate 83.4      ->  4.51          0.054
#     patch  15.0       ->  0.00          0.000
#     xover  10.5       ->  0.00          0.000
#
# Iterations 8 and 9 each took HALF of graft's share for a new move, so the best-measured escape
# went 19.2 -> 11.2 -> 5.4 CPU-s while 25.5 CPU-s went to two moves that have never once produced
# a digit.  The fix is not a fourth hand-picked split: it is to spend the escape budget where it
# is OBSERVED to pay, measured inside the run itself, so the mix also adapts on sizes I never
# score (no donor -> graft/patch unavailable -> the bank concentrates on what is left).
# EQUALIZE (iteration 11).  Cauchy-Schwarz on the area bound: sum(pi*r_i^2) <= 1 forces
# sum(r_i) <= sqrt(n/pi), with EQUALITY only when every radius is equal -- so the objective the
# scorer pays for is maximized on the equal-radius manifold, and any tail of small circles is
# pure loss.  Every committed pack still short of its record has such a tail (n=81: ten circles
# at 0.036-0.050 against a body at 0.057; n=59: nine at 0.046-0.053 against 0.065-0.070).  None
# of the four escapes above can retire a tail: they all optimize sum(r), which is happy to keep a
# small circle wedged where it is.  `_equalize` re-solves a few trust-region LPs under the
# weighted objective sum(r_i^-p * r_i) instead, which grows the smallest circles at the big ones'
# expense and so MOVES CENTERS; the ordinary sum(r) optimizer then re-maximizes from the new
# contact graph.  Measured (artifacts/prof15_equalize.py, 3.0 CPU-s/n, seeds 61/62/63 over the
# stuck core 33/37/47/49/59/81 + 51): mean digits 3.10 -> 3.79 and 3.44 -> 3.98, 4 win / 4 loss /
# 10 tie -- and the wins are not marginal: n=33 (stuck at 2.646 for four iterations) and n=47
# reached the 7-digit cap, with the equalize hop itself credited 1.095 and 3.901 digits.  The four
# losses are 0.02-0.17 digits of rng divergence, not regressions: in a metered run the committed
# pack is the floor, since `offer` only ever replaces a strict improvement.
EQ_STEPS = 3            # weighted LP steps before handing back to the sum(r) optimizer
EQ_DELTA = 0.03         # their trust region: half DELTA0, so the layout deforms without tearing
EQ_P = (0.5, 3.0)       # sampled exponent range.  p->0 is the ordinary objective (no move at
                        # all); p large is essentially max-min radius.  Sampled, not fixed, so
                        # the move spans gentle de-tailing to a full equalization.
# --- iteration 12: the EXACT max-min sibling of _equalize.
# _equalize pushes the radius tail up with an r^-p weight, which is a SMOOTH surrogate for
# "grow the smallest circles": every radius keeps some weight, so the LP can buy a cheap gain on
# a mid-sized circle instead of paying for the genuinely smallest one, and the tail it retires is
# whichever part of the tail is cheapest, not necessarily the binding one.  The exact objective
# is one LP column away (`t <= r_i` for all i, maximize `sum(r) + lam*n*t`): then ONLY the true
# minimum pays, and the LP must move whatever neighbour is pinning it.  `lam` is sampled rather
# than fixed for the same reason EQ_P is: lam -> 0 is the ordinary sum(r) step (no move at all),
# lam large is pure max-min and can cost a lot of sum(r) for one circle; the useful escapes are
# spread across the range and which one a given layout wants is not predictable from n.
MM_STEPS = 3            # max-min LP steps before handing back to the sum(r) optimizer
MM_DELTA = 0.03         # same trust region as EQ_DELTA: deform without tearing the layout
MM_LAM = (0.3, 3.0)     # sampled weight on the min-radius term, relative to "every radius at once"
# CONTACT-GRAPH PIVOT (iteration 13).  Measured at the start of this iteration: every one of the
# 37 committed packings is EXACTLY ISOSTATIC -- the number of binding constraints (circle-circle
# contacts plus wall contacts) equals 3n in all 37 cases, and the minimum contact degree is 4.
# That is the whole explanation for the ten stuck n.  At an isostatic vertex the KKT conditions
# read  grad(sum r) = -sum_k lambda_k grad(g_k)  with lambda >= 0, so releasing any single
# contact e moves sum(r) DOWN at first-order rate lambda_e: there is no first-order escape, and
# every move in this file (LP under any objective, jitter, graft, crossover) only ever sees first
# order.  Hence 200-400 jitter hops per stuck n returning to the same three or four sum_r values.
# The escape has to be COMBINATORIAL: release one contact, follow the resulting 1-DOF mechanism
# until a DIFFERENT contact forms, and re-optimize at the new vertex.  That is a pivot in
# contact-graph space, and it is what the file has never had.
CT_TOL = 1e-7           # a constraint counts as binding when its slack is below this
PV_STEPS = 10           # continuation micro-steps along the released mechanism
PV_CHAIN = 7            # up to this many chained releases per pivot (1 + randint(0, PV_CHAIN)).
                        # One link is provably useless (the start vertex is a local max, so every
                        # neighbour is worse); several links are what carries the layout out of
                        # the cell the LP can descend back into.
PV_SIGMA = 0.05         # Newton continuation step, in units of the mean radius.  Measured: a new
                        # contact forms after only 1-3 of these (0.05-0.15 mean radii of opening),
                        # so PV_STEPS is a safety cap, not the usual stopping rule.
PV_RANKQ = 2.5          # release-candidate sampling: rank^PV_RANKQ over contacts sorted by dual
                        # price ascending, so the CHEAPEST contact to release is tried most often
                        # but every contact stays reachable.
ESCAPE_CLASSES = ("inflate", "graft", "patch", "xover", "equal", "maximin")
MV_PRIOR_GAIN = 0.15    # optimistic prior digits, so an unmeasured class is tried early
MV_PRIOR_CPU = 1.0      # ... against this much prior CPU (a class needs ~1 s to move its own est.)
MV_FLOOR = 0.06         # every AVAILABLE class keeps at least this sampling share, always
MV_DECAY = 0.995        # a class's credited gain decays this much per pick of it (slow forgetting:
                        # marginal yield falls as a class harvests the n it can help)
# An "improvement" below this is numerical noise, not progress (relative, so it scales with n).
IMPROVE_TOL = 1e-12


# --------------------------------------------------------------------------- census reading
def _read_pck(path):
    """Parse a Packomania .pck into an (n,3) array of (x,y,r); None if unusable."""
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()
        rows = []
        for ln in lines[2:]:
            parts = ln.split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if not rows:
            return None
        return np.asarray(rows, dtype=float)
    except Exception:
        return None


# --------------------------------------------------------------------------- feasibility repair
def _repair(xy, r):
    """Make (xy, r) STRICTLY feasible by only *lowering* radii (exact float arithmetic).

    Walls first, then pairwise: for each i subtract half its worst overlap.  One pass already
    clears every pair (both members of a violating pair each lose >= viol/2), the loop just
    mops up float residue.  Monotone in r, so it can never turn an accepted improvement bad.
    """
    x, y = xy[:, 0], xy[:, 1]
    wall = np.minimum.reduce([x - LO, HI - x, y - LO, HI - y])
    r = np.minimum(r, wall)
    n = r.shape[0]
    if n >= 2:
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        for _ in range(64):
            worst = (r[:, None] + r[None, :] - d).max(axis=1)
            if worst.max() <= 0.0:
                break
            r = r - np.maximum(worst, 0.0) / 2.0
    return np.maximum(r - SHRINK, 1e-13)


# --------------------------------------------------------------------------- one SLP step
def _slp_step(xy, r, delta, w=None, mm=0.0, sep=None):
    """One trust-region LP: maximize sum(w*r) (+ mm*n*t) over |dp| <= delta, linearized separation.

    Pairs far apart (relative to the trust region) cannot bind and are pruned -- and even if the
    prune were wrong, _repair() fixes it exactly, so pruning only ever costs quality, never
    feasibility.

    `w` (default: all ones, i.e. the scorer's own objective sum(r)) re-weights the radii in the
    OBJECTIVE only; the geometry rows are untouched, so any w still returns a layout that
    _repair() makes strictly feasible.  _equalize() uses a non-uniform w as a structural escape.

    `sep = (i, j, margin)` FORBIDS the contact (i, j): its separation row's right-hand side is
    reduced by `margin`, so the LP must keep circles i and j at least `margin` apart instead of
    letting them touch.  That is the only way to stop the ordinary sum(r) optimizer from simply
    re-closing a contact that _pivot() has just released -- without it the LP walks straight back
    to the vertex it came from (measured: 10 of 10 pivots at n=27..81 returned the identical
    sum_r).  A margin only ever SHRINKS the feasible set, so the layout stays repairable.

    `mm > 0` additionally appends ONE scalar column t with the n rows `t <= r_i`, and puts t in
    the objective at weight mm*n -- i.e. maximize `sum(r) + mm*n*min_i r_i`, the EXACT max-min
    radius objective (as mm -> inf) that _equalize only approximates with an r^-p weight.  The
    geometry rows are again untouched, so the layout that comes back is still repairable; t is
    dropped from the return.  _maximin() uses it.
    """
    n = r.shape[0]
    x, y = xy[:, 0], xy[:, 1]
    iu, ju = np.triu_indices(n, 1)
    dif = xy[iu] - xy[ju]
    dd = np.sqrt((dif ** 2).sum(-1))
    keep = dd <= (r[iu] + r[ju] + PRUNE_K * delta + 1e-3)
    iu, ju, dif, dd = iu[keep], ju[keep], dif[keep], dd[keep]
    dd = np.maximum(dd, 1e-12)
    ux, uy = dif[:, 0] / dd, dif[:, 1] / dd

    m = iu.shape[0]
    ar = np.arange(m)
    an = np.arange(n)
    rows = [ar, ar, ar, ar, ar, ar]
    cols = [2 * n + iu, 2 * n + ju, iu, ju, n + iu, n + ju]
    vals = [np.ones(m), np.ones(m), -ux, ux, -uy, uy]
    ub_pair = dd.copy()
    if sep is not None:
        si, sj, smarg = int(sep[0]), int(sep[1]), float(sep[2])
        hit = np.nonzero(((iu == si) & (ju == sj)) | ((iu == sj) & (ju == si)))[0]
        if hit.shape[0]:
            ub_pair[hit[0]] -= smarg
    ub = [ub_pair]
    off = m
    wmarg = PRUNE_K * delta + 1e-3
    for sgn, coord, rhs in ((1.0, 0, 0.5 - x), (-1.0, 0, 0.5 + x),
                            (1.0, 1, 0.5 - y), (-1.0, 1, 0.5 + y)):
        sel = an[(rhs - r) <= wmarg]
        k = sel.shape[0]
        if k == 0:
            continue
        a = np.arange(off, off + k)
        rows += [a, a]
        cols += [2 * n + sel, coord * n + sel]
        vals += [np.ones(k), sgn * np.ones(k)]
        ub.append(rhs[sel])
        off += k
    cr_ = [np.zeros(2 * n), -np.ones(n) if w is None else -np.asarray(w, dtype=float)]
    bounds = [(-delta, delta)] * (2 * n) + [(0.0, 0.5)] * n
    nv = 3 * n
    if mm > 0.0:
        a = np.arange(off, off + n)
        rows += [a, a]
        cols += [np.full(n, 3 * n), 2 * n + an]     # +t  -r_i  <= 0
        vals += [np.ones(n), -np.ones(n)]
        ub.append(np.zeros(n))
        off += n
        nv += 1
        cr_.append(np.array([-mm * n]))
        bounds.append((0.0, 0.5))
    A = csr_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))), shape=(off, nv))
    c = np.concatenate(cr_)
    b_ub = np.concatenate(ub)
    methods = ("highs-ipm", "highs") if nv >= IPM_MIN_VARS else ("highs",)
    res = None
    for meth in methods:
        try:
            res = linprog(c, A_ub=A, b_ub=b_ub, bounds=bounds, method=meth)
        except Exception:
            res = None
        if res is not None and res.success and res.x is not None:
            break
        res = None
    if res is None:
        return None
    z = res.x
    nxy = np.clip(xy + np.stack([z[:n], z[n:2 * n]], axis=1), LO, HI)
    return nxy, z[2 * n:3 * n]


def _local_opt(xy, r, cpu_stop, delta0=DELTA0, delta_min=DELTA_MIN, max_it=240,
               max_fail=MAX_FAIL, sep=None):
    """SLP to convergence (or until `cpu_stop` process-CPU). Returns (xy, r, sum_r).

    Termination is by STALL, not only by the trust-region floor.  A failed step shrinks delta by
    0.35 and a later, smaller step can still succeed (the linearization gets more accurate), so a
    single failure must be tolerated -- but `max_fail` consecutive failures mean delta has already
    shrunk ~175x with no gain, and grinding it to the 1e-8 floor from there buys < 1e-11 of sum_r.
    Measured over 30 hops at n=27..99: 1.46x faster on its own, worst objective loss 3.0e-12
    (relgap ~6e-13, four orders below the 1e-7 that already earns the full 7 digits), and combined
    with PRUNE_K it is 2.4x more basin hops per CPU-second.
    """
    r = _repair(xy, r)
    best = float(r.sum())
    delta = delta0
    it = 0
    fails = 0
    while it < max_it and time.process_time() < cpu_stop:
        out = _slp_step(xy, r, delta, None, 0.0, sep)
        it += 1
        if out is None:
            delta *= 0.35
            fails += 1
        else:
            nxy, nr = out
            nr = _repair(nxy, nr)
            s = float(nr.sum())
            if s > best + IMPROVE_TOL * max(best, 1.0):
                xy, r, best = nxy, nr, s
                delta = min(delta * 1.7, 0.12)
                fails = 0
            else:
                delta *= 0.35
                fails += 1
        if delta < delta_min or fails >= max_fail:
            break
    return xy, r, best


# --------------------------------------------------------------------------- start layouts
def _start(n, rng, kind):
    """A cold starting center set. Every kind works for ANY n (no per-n tables)."""
    if kind == 0:                                     # jittered square grid
        k = int(np.ceil(np.sqrt(n)))
        g = (np.arange(k) + 0.5) / k - 0.5
        pts = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        pts = pts[rng.permutation(pts.shape[0])[:n]]
        return np.clip(pts + rng.normal(0, 0.15 / k, pts.shape), LO + 1e-3, HI - 1e-3)
    if kind == 1:                                     # jittered hexagonal rows
        rows = max(1, int(round(np.sqrt(n * 2.0 / np.sqrt(3.0)))))
        per = int(np.ceil(n / rows))
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        pts = []
        for i, yy in enumerate(ys):
            xs = (np.arange(per) + 0.5 + 0.25 * (i % 2)) / per - 0.5
            for xx in xs:
                pts.append((xx, yy))
        pts = np.asarray(pts)[:n] if len(pts) >= n else np.asarray(pts)
        while pts.shape[0] < n:
            pts = np.vstack([pts, rng.uniform(LO + 0.05, HI - 0.05, size=(n - pts.shape[0], 2))])
        return np.clip(pts + rng.normal(0, 0.15 / max(rows, per), pts.shape),
                       LO + 1e-3, HI - 1e-3)
    return rng.uniform(LO + 0.02, HI - 0.02, size=(n, 2))     # uniform random


# --------------------------------------------------------------------------- holes
def _hole_points(xy, r, rng, k, jitter=0.01, grid=25, greedy=False):
    """k points in the emptiest places: grid samples ranked by clearance to the nearest disc.

    `greedy=True` picks the k best holes *sequentially*, blanking out a neighbourhood after each
    pick, so two inserted circles never land on top of each other (which they otherwise do, since
    the top-k cells of one hole are all in that same hole).
    """
    ax = np.linspace(LO + 0.02, HI - 0.02, grid)
    pts = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
    d = np.sqrt(((pts[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]
    clr = d.min(axis=1)
    if greedy:
        out = []
        c = clr.copy()
        for _ in range(k):
            i = int(np.argmax(c))
            out.append(pts[i])
            far = np.sqrt(((pts - pts[i]) ** 2).sum(-1))
            c = np.where(far < max(2.0 * max(c[i], 1e-3), 1.0 / grid), -np.inf, c)
            if not np.isfinite(c).any():
                c = clr.copy()
        sel = np.asarray(out)
    else:
        idx = np.argsort(-clr)[:max(k * 3, k)]
        pick = idx[rng.permutation(idx.shape[0])[:k]]
        sel = pts[pick]
    return np.clip(sel + rng.normal(0, jitter, size=sel.shape), LO + 1e-3, HI - 1e-3)


# --------------------------------------------------------------------------- structure transfer
def _census_seed(n, rng, max_off=8):
    """Seed n from the nearest committed packing.  Returns (xy, r, exact) or None.

    `exact` is True only when the pack for this very n exists (the ordinary warm start).  For a
    neighbouring m the packing is adapted: m > n drops the (m-n) smallest circles, m < n inserts
    (n-m) tiny circles into the largest holes.  This is the path an offline re-run on a size that
    was never scored in-run takes, so it must work for ANY n, and it must never raise.
    """
    order = [0]
    for o in range(1, max_off + 1):
        order += [o, -o]
    for off in order:
        m = n + off
        if m < 1:
            continue
        p = os.path.join("bench", "packs", "csqv%d.pck" % m)
        if not os.path.exists(p):
            continue
        a = _read_pck(p)
        if a is None or a.shape[0] != m:
            continue
        xy, r = a[:, :2].copy(), a[:, 2].copy()
        if m == n:
            return xy, r, True
        if m > n:
            keep = np.argsort(-r)[:n]
            return xy[keep], r[keep], False
        add = n - m
        h = _hole_points(xy, r, rng, add, jitter=0.005, greedy=True)
        return (np.vstack([xy, h]),
                np.concatenate([r, np.full(add, 1e-3)]), False)
    return None


# --------------------------------------------------------------------------- perturbations
def _inflate_relax(xy, r, rng, n, gamma, iters=6, damp=0.9):
    """INFLATE-RELAX: grow every radius by `gamma`, then push centers apart until the overlaps
    are gone.  Returns centers only (the SLP re-derives radii from them).

    This is a different CLASS of move from the six center-jitter moves below, and iteration 5's
    trace says a different class is exactly what is needed: over a 100 CPU-s probe, 29 of 37 n
    gained < 1e-3 digits from 11-105 center-jitter hops each, i.e. the marginal value of another
    jitter hop is ~0 at almost every n.  The reason is structural.  A jitter move displaces a few
    circles and the monotone SLP pulls them back into the SAME contact graph, so the chain never
    leaves its basin.  Inflating instead makes EVERY circle overlap its neighbours at once and
    lets the relaxation decide who yields: the displacement field is generated by the packing's
    own geometry (dense regions push hardest), so the whole layout shifts coherently and the
    contact graph can re-form with a different topology.  It is the classic inflate/relax step of
    dense-packing heuristics, adapted to heterogeneous radii.

    Cost is O(iters * n^2) vectorized -- ~2 orders below one LP solve, so a hop that uses it is
    no more expensive than a hop that does not.
    """
    c = xy.copy()
    rr = r * gamma
    for _ in range(iters):
        diff = c[:, None, :] - c[None, :, :]
        d2 = (diff ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        ov = (rr[:, None] + rr[None, :]) - d          # >0 where the inflated disks overlap
        np.fill_diagonal(ov, 0.0)
        act = ov > 0
        if not act.any():
            break
        # Each overlapping pair contributes half its overlap to each center, along their axis.
        u = diff / np.maximum(d, 1e-12)[:, :, None]
        w = np.where(act, ov, 0.0) * 0.5 * damp
        c = c + (u * w[:, :, None]).sum(1)
        # Walls: an inflated disk that pokes out is pushed back in (its own radius, not rr, would
        # under-correct, so use rr and let the SLP reclaim the slack).
        c[:, 0] = np.clip(c[:, 0], LO + rr, HI - rr)
        c[:, 1] = np.clip(c[:, 1], LO + rr, HI - rr)
        # A disk too big for the box after inflation would give an empty clip range; re-center it.
        bad = rr > 0.5 - 1e-6
        if bad.any():
            c[bad] = 0.0
    return np.clip(c, LO + 1e-4, HI - 1e-4)


def _equalize(xy, r, rng, steps=EQ_STEPS, delta=EQ_DELTA, p=None):
    """Structural escape: a few trust-region LPs run under a SIZE-EQUALIZING objective.

    See the EQ_* block above for why.  The weight is r_i^-p normalized by its max, so the
    smallest circle in the layout has weight 1 and the largest has (r_min/r_max)^p -- the LP then
    spends its trust region pushing neighbours aside to grow the tail.  The result is NOT offered
    to evaluate(): it is a worse sum(r) by construction, and only a starting point.  Needs no
    donor, no archive and no records, so it is available for ANY n -- including a fresh size or an
    offline re-run outside the census.  Never raises and never returns None: on an LP failure it
    returns whatever it has deformed so far (at worst the input, which the caller can still use).
    """
    if p is None:
        p = EQ_P[0] + (EQ_P[1] - EQ_P[0]) * float(rng.rand())
    cxy, cr = xy.copy(), r.copy()
    for _ in range(steps):
        w = np.maximum(cr, 1e-9) ** (-p)
        w = w / w.max()
        try:
            out = _slp_step(cxy, cr, delta, w)
        except Exception:
            break
        if out is None:
            break
        cxy = np.clip(out[0], LO, HI)
        cr = _repair(cxy, out[1])
    return cxy, cr


def _maximin(xy, r, rng, steps=MM_STEPS, delta=MM_DELTA, lam=None):
    """Structural escape: a few trust-region LPs under the EXACT max-min-radius objective.

    See the MM_* block above.  Same shape as _equalize -- deform under a non-sum(r) objective,
    then hand the layout back to the ordinary sum(r) optimizer, whose contact graph is now one
    the sum(r) chain could not have reached -- but the objective is `sum(r) + lam*n*min_i r_i`
    rather than a weighted sum, so the LP is forced to unpin the single binding smallest circle
    instead of taking the cheapest tail gain available.  The result is NOT offered to evaluate():
    it is a worse sum(r) by construction and only a starting point.  Needs no donor, no archive
    and no records, so it is available for ANY n -- a fresh size or an offline re-run outside the
    census included.  Never raises and never returns None: on an LP failure it returns whatever
    it has deformed so far (at worst the input, which the caller can still use).
    """
    if lam is None:
        lam = MM_LAM[0] + (MM_LAM[1] - MM_LAM[0]) * float(rng.rand())
    cxy, cr = xy.copy(), r.copy()
    for _ in range(steps):
        try:
            out = _slp_step(cxy, cr, delta, None, lam)
        except Exception:
            break
        if out is None:
            break
        cxy = np.clip(out[0], LO, HI)
        cr = _repair(cxy, out[1])
    return cxy, cr


# --------------------------------------------------------------------------- contact graph
def _contacts(xy, r, tol=CT_TOL):
    """The binding constraint set at a (locally optimal) layout.

    Returns `(pi, pj, wi, ws)`: `pi[k], pj[k]` are the circle-circle contacts and `wi[k]` is the
    circle of the k-th WALL contact with side `ws[k]` in 0..3 = (x=+.5, x=-.5, y=+.5, y=-.5).
    `tol` is absolute; the slacks at a converged local optimum are ~1e-13, so any tol in
    1e-9..1e-6 gives the same set (checked over all 37 committed packs).
    """
    n = r.shape[0]
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    sl = d - (r[:, None] + r[None, :])
    iu, ju = np.triu_indices(n, 1)
    keep = sl[iu, ju] < tol
    pi, pj = iu[keep], ju[keep]
    wi, ws = [], []
    for side, g in enumerate((0.5 - xy[:, 0] - r, 0.5 + xy[:, 0] - r,
                              0.5 - xy[:, 1] - r, 0.5 + xy[:, 1] - r)):
        sel = np.nonzero(g < tol)[0]
        wi.append(sel)
        ws.append(np.full(sel.shape[0], side))
    return pi, pj, np.concatenate(wi), np.concatenate(ws)


def _ct_residual(xy, r, pi, pj, wi, ws):
    """The slack of every row of a fixed contact set: 0 at the vertex it was read from."""
    dif = xy[pi] - xy[pj]
    dd = np.sqrt((dif ** 2).sum(-1))
    gp = dd - (r[pi] + r[pj])
    x, y = xy[wi, 0], xy[wi, 1]
    rw = r[wi]
    gw = np.where(ws == 0, 0.5 - x - rw,
                  np.where(ws == 1, 0.5 + x - rw,
                           np.where(ws == 2, 0.5 - y - rw, 0.5 + y - rw)))
    return np.concatenate([gp, gw]), dd


def _ct_jac(xy, r, pi, pj, wi, ws, dd):
    """Sparse Jacobian of that residual w.r.t. z = (x, y, r), in the _slp_step variable order."""
    n = r.shape[0]
    m = pi.shape[0]
    k = wi.shape[0]
    dif = xy[pi] - xy[pj]
    ux = dif[:, 0] / np.maximum(dd, 1e-15)
    uy = dif[:, 1] / np.maximum(dd, 1e-15)
    ar = np.arange(m)
    rows = [ar, ar, ar, ar, ar, ar]
    cols = [pi, pj, n + pi, n + pj, 2 * n + pi, 2 * n + pj]
    vals = [ux, -ux, uy, -uy, -np.ones(m), -np.ones(m)]
    aw = m + np.arange(k)
    sgn = np.where(ws == 0, -1.0, np.where(ws == 1, 1.0,
                   np.where(ws == 2, -1.0, 1.0)))
    coord = np.where(ws < 2, 0, 1)
    rows += [aw, aw]
    cols += [coord * n + wi, 2 * n + wi]
    vals += [sgn, -np.ones(k)]
    return csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(m + k, 3 * n)).tocsc()


def _ct_duals(J):
    """KKT multipliers of `max sum(r)` at an isostatic vertex: solve J^T lam = -c, c = d(sum r)/dz.

    From  grad f + sum_k lam_k grad g_k = 0  with g_k >= 0 and lam_k >= 0, so `lam_e` is exactly
    the first-order PRICE of releasing contact e.  Returns None if J is not square/invertible
    (a hyperstatic or under-constrained layout, which no converged local optimum here has been).
    """
    nv = J.shape[1]
    if J.shape[0] != nv:
        return None
    n = nv // 3
    c = np.zeros(nv)
    c[2 * n:] = 1.0
    try:
        lu = splu(J.T.tocsc())
        lam = lu.solve(-c)
    except Exception:
        return None
    if not np.all(np.isfinite(lam)):
        return None
    return lam


def _pivot(xy, r, rng, cpu_stop=None, steps=PV_STEPS, chain=PV_CHAIN, rankq=PV_RANKQ):
    """CONTACT-GRAPH PIVOT: a CHAIN of contact releases, each walked to the next collision.

    See the CT_/PV_ block above for why this is the one escape in the file that is not
    first-order.  One link: release a binding circle-circle contact `e`, hold the index set
    fixed, and drive row e to a growing positive slack by Newton on the square 3n x 3n system
    {g_k = 0 for k != e, g_e = t} -- one sparse solve per micro-step.  The walk stops the instant
    some pair OUTSIDE the contact set touches: that is the new contact `f`, so the layout is now
    at the neighbouring vertex of the contact graph (e out, f in).

    A SINGLE link cannot help and is not meant to: the vertex we start from is a constrained
    local maximum, so its neighbours are all worse (measured: releasing the cheapest contacts at
    n=27/45/81 costs only 4e-5..4e-4 of sum_r, and the ordinary optimizer then walks back to the
    identical sum_r in 10 of 10 tries).  What pays is CHAINING: `chain` links change the graph in
    several places at once, each for ~1e-4 of sum_r, so the layout ends up in a cell the
    single-step LP descent cannot return from.  The last released contact is then held open by a
    margin row (`_slp_step(sep=...)`) for one restricted optimization, so the layout handed back
    is that cell's own optimum rather than the raw collision point.

    Needs nothing but this layout -- no donor, no archive, no records -- so it is available for
    ANY n, including a fresh size and an offline re-run outside the census.  Returns None (a
    clean decline, so the bank spends the hop on another class) when the layout is not isostatic
    or the linear algebra fails; never raises.

    MEASURED VERDICT (iteration 13): a NULL, and deliberately NOT registered in ESCAPE_CLASSES.
    Over 150 pivots at the ten stuck n (artifacts/prof18-22), a pivot followed by the ordinary
    optimizer returned the IDENTICAL sum_r (to 1e-9) in 139 cases and never once improved on the
    committed vertex; only 11 landed on a different contact graph at all.  The reason is
    quantitative: one link costs 4e-5..4e-4 of sum_r and displaces centers by ~0.1 mean radii,
    while the LP's own trust region starts at DELTA0 = 0.06 (~0.7 mean radii), so the whole
    <=PV_CHAIN-swap neighbourhood lies INSIDE one LP step's basin of attraction.  Kept in the
    file because the machinery it is built on (_contacts/_ct_duals, see tools/rigidity.py) is
    what proved the census locally optimal, and because the null is itself the finding: the ten
    stuck n are not a few contact swaps from a better vertex.  Registering it would have put a
    third probe-measured-null arm behind MV_FLOOR, which is the mistake iteration 12 recorded.
    """
    n = r.shape[0]
    if cpu_stop is None:
        cpu_stop = time.process_time() + 1.0
    try:
        cxy, cr = xy.copy(), r.copy()
        iu, ju = np.triu_indices(n, 1)
        key = iu.astype(np.int64) * n + ju
        links = 1 + int(rng.randint(0, max(1, chain)))
        last = None
        for _ in range(links):
            pi, pj, wi, ws = _contacts(cxy, cr)
            npair = pi.shape[0]
            if npair < 1 or npair + wi.shape[0] != 3 * n:
                break
            g, dd = _ct_residual(cxy, cr, pi, pj, wi, ws)
            lam = _ct_duals(_ct_jac(cxy, cr, pi, pj, wi, ws, dd))
            if lam is None:
                break
            # Cheapest contact to release first, but sampled so every contact stays reachable
            # and the move is a FAMILY of deformations rather than one deterministic step.
            order = np.argsort(np.maximum(lam[:npair], 0.0))
            e = int(order[min(npair - 1, int(npair * float(rng.rand()) ** rankq))])
            mask = ~np.isin(key, pi.astype(np.int64) * n + pj)
            fi, fj = iu[mask], ju[mask]
            sigma = float(cr.mean()) * PV_SIGMA
            hit = False
            for m in range(1, steps + 1):
                g, dd = _ct_residual(cxy, cr, pi, pj, wi, ws)
                tgt = np.zeros(g.shape[0])
                tgt[e] = m * sigma
                try:
                    dz = splu(_ct_jac(cxy, cr, pi, pj, wi, ws, dd)).solve(tgt - g)
                except Exception:
                    break
                if not np.all(np.isfinite(dz)) or np.abs(dz).max() > 0.5:
                    break
                nxy = cxy + np.stack([dz[:n], dz[n:2 * n]], axis=1)
                nr = cr + dz[2 * n:3 * n]
                if nr.min() <= 1e-6 or np.abs(nxy).max() > HI + 1e-6:
                    break
                cxy, cr = nxy, nr
                if fi.shape[0] and (np.sqrt(((cxy[fi] - cxy[fj]) ** 2).sum(-1))
                                    - (cr[fi] + cr[fj])).min() < 0.0:
                    hit = True
                    break
            if hit:
                last = (int(pi[e]), int(pj[e]))
            if time.process_time() >= cpu_stop:
                break
        if last is not None:
            marg = float(np.sqrt(((cxy[last[0]] - cxy[last[1]]) ** 2).sum())
                         - (cr[last[0]] + cr[last[1]]))
            if marg > 1e-9 and time.process_time() < cpu_stop:
                cxy, cr, _ = _local_opt(cxy, cr, cpu_stop, max_fail=2,
                                        sep=(last[0], last[1], marg))
        return np.clip(cxy, LO, HI), np.maximum(cr, 1e-9)
    except Exception:
        return None


def _graft_donors(n, states, cache, max_off=GRAFT_MAX_OFF):
    """Every OTHER size with a usable packing, nearest first: (m, xy, r).

    Prefers this run's own best for m (so an improvement found at m propagates to m+-2 inside the
    same call) and falls back to the committed pack on disk.  Every disk read is guarded and
    cached, so a size with no pack -- an offline re-run on unseen n -- simply yields fewer donors
    and the caller falls back to the other move classes.
    """
    inrun = {}
    for st in states:
        if st.n != n and st.best_xy is not None:
            inrun[st.n] = (st.best_xy, st.best_r)
    out = []
    for off in range(1, max_off + 1):
        for m in (n - off, n + off):
            if m < 2:
                continue
            if m in inrun:
                out.append((m, inrun[m][0], inrun[m][1]))
                continue
            if m not in cache:
                p = os.path.join("bench", "packs", "csqv%d.pck" % m)
                a = _read_pck(p) if os.path.exists(p) else None
                cache[m] = a if (a is not None and a.shape[0] == m) else None
            if cache[m] is not None:
                out.append((m, cache[m][:, :2], cache[m][:, 2]))
    return out


def _graft(n, rng, donors):
    """GRAFT: rebuild n's layout from a DIFFERENT size's good packing.  Returns (xy, r) or None.

    Every other move in this file is a displacement of n's own incumbent, so the chain can only
    ever reach structures reachable from where it already is -- and iteration 5's trace showed
    that at 29 of 37 n that neighbourhood is exhausted.  A neighbouring size's packing is the one
    source of a *genuinely different* contact graph that is also known to be good: csqv layouts at
    n and n+-2 share their grid/ring/hex family, so the adapted donor lands in a different basin
    of the same quality class rather than in the ~1.7-digit soup a random restart gives.  It is
    also the move that makes the census compound: every n that improves becomes a better donor for
    its neighbours, within this call and across iterations.

    Adaptation is the same geometry `_census_seed` already uses for an uncommitted n -- drop the
    smallest circles when the donor has too many, insert into the largest holes when it has too
    few -- except that WHICH small circles are dropped is randomized over the smallest 2k, so
    repeated grafts from one donor explore different variants instead of repeating one point.
    """
    if not donors:
        return None
    w = np.array([1.0 / (1.0 + abs(m - n)) ** 2 for m, _, _ in donors])
    w /= w.sum()
    m, dxy, dr = donors[int(rng.choice(len(donors), p=w))]
    xy, r = dxy.copy(), dr.copy()
    if m > n:
        k = m - n
        order = np.argsort(r)
        pool = order[:min(m, 2 * k)]
        if pool.shape[0] > k:
            drop = pool[rng.permutation(pool.shape[0])[:k]]
        else:
            drop = order[:k]
        keep = np.setdiff1d(np.arange(m), drop)
        xy, r = xy[keep], r[keep]
    elif m < n:
        k = n - m
        h = _hole_points(xy, r, rng, k, jitter=0.005, greedy=True)
        xy = np.vstack([xy, h])
        r = np.concatenate([r, np.full(k, 1e-3)])
    if xy.shape[0] != n:
        return None
    return np.clip(xy, LO + 1e-4, HI - 1e-4), r


def _graft_patch(n, rng, donors, xy, r):
    """PATCH GRAFT (iteration 8): splice one HALF-PLANE of a donor size into this n's incumbent.

    `_graft` replaces the whole layout with an adapted copy of a neighbouring size, so everything
    this n has already earned is thrown away and the (m-n) drops / (n-m) inserts are spread over
    the entire square -- the further the donor, the more of the borrowed structure the adaptation
    destroys (iteration 7's honest limit (b)).  A patch keeps the incumbent on one side of a
    random cut and takes the donor on the other, so it imports a genuinely different contact graph
    LOCALLY while preserving the half of this n that is already good, and the count fix-up is a
    handful of circles instead of |m-n| of them.  It is the crossover operator the whole-donor
    graft approximates with one parent.

    Returns (xy, r) or None (no donor / count could not be fixed) -- the caller then falls
    through to the other escape classes, so an n with no neighbours in the census is unaffected.
    """
    if not donors or xy is None:
        return None
    w = np.array([1.0 / (1.0 + abs(m - n)) ** 2 for m, _, _ in donors])
    w /= w.sum()
    m, dxy, dr = donors[int(rng.choice(len(donors), p=w))]
    th = float(rng.rand()) * 2.0 * np.pi
    u = np.array([np.cos(th), np.sin(th)])
    f = 0.25 + 0.35 * float(rng.rand())               # donor's share of the square
    pi_, pd = xy @ u, dxy @ u
    c = float(np.quantile(pi_, 1.0 - f))              # cut placed by the INCUMBENT's own spread
    keep = pi_ <= c
    take = pd > c
    cxy = np.vstack([xy[keep], dxy[take]])
    cr = np.concatenate([r[keep], dr[take]])
    k = cxy.shape[0] - n
    if k > 0:                                         # too many: drop smallest, randomized
        order = np.argsort(cr)
        pool = order[:min(order.shape[0], 2 * k)]
        drop = pool[rng.permutation(pool.shape[0])[:k]] if pool.shape[0] > k else order[:k]
        sel = np.setdiff1d(np.arange(cxy.shape[0]), drop)
        cxy, cr = cxy[sel], cr[sel]
    elif k < 0:                                       # too few: fill the emptiest holes
        h = _hole_points(cxy, cr, rng, -k, jitter=0.005, greedy=True)
        cxy = np.vstack([cxy, h])
        cr = np.concatenate([cr, np.full(-k, 1e-3)])
    if cxy.shape[0] != n:
        return None
    return np.clip(cxy, LO + 1e-4, HI - 1e-4), cr


def _crossover_self(xy, r, arch, cur_s, rng):
    """SELF-CROSSOVER (iteration 9): splice two distinct local optima of the SAME n.

    `_graft`/`_graft_patch` import structure from a neighbouring SIZE, so |m-n| circles have to be
    invented or discarded and the borrowed contact graph is damaged in proportion to the distance
    to the donor.  Here both parents pack exactly n circles in the same square, so a RANK-based
    half-plane cut -- the k circles of this n's incumbent with the lowest projection on a random
    direction, plus the (n-k) circles of a parent with the HIGHEST projection -- yields exactly n
    circles with no adaptation whatsoever.  The only damage is the seam, which `_repair` shrinks
    and the SLP then re-grows.

    Needs no donor size, so unlike either graft it is available for ANY n (including a size the
    census has never seen) as soon as the chain has found two distinct local optima.

    Returns (xy, r) or None when the archive holds no parent structurally distinct from the
    incumbent -- the caller then falls through to the graft escapes.
    """
    if xy is None or not arch:
        return None
    n = xy.shape[0]
    tol = XOVER_DISTINCT * max(abs(cur_s), 1e-9)
    cand = [a for a in arch if a[0] == n and abs(a[3] - cur_s) > tol]
    if not cand:
        return None
    _, mxy, mr, _ = cand[int(rng.randint(len(cand)))]
    th = float(rng.rand()) * 2.0 * np.pi
    u = np.array([np.cos(th), np.sin(th)])
    f = 0.30 + 0.40 * float(rng.rand())               # the PARENT's share of the square
    k = int(max(1, min(n - 1, round((1.0 - f) * n))))
    ai = np.argsort(xy @ u)[:k]                       # incumbent keeps the near side
    am = np.argsort(-(mxy @ u))[:n - k]               # parent supplies the far side
    cxy = np.vstack([xy[ai], mxy[am]])
    cr = np.concatenate([r[ai], mr[am]])
    if cxy.shape[0] != n:
        return None
    return np.clip(cxy, LO + 1e-4, HI - 1e-4), cr


def _arch_add(arch, xy, r, s, best_s):
    """Maintain a small archive of DISTINCT good local optima as crossover parents.

    Distinctness is keyed on sum_r at XOVER_DISTINCT relative tolerance: two runs that land in the
    same basin agree far closer than that, and two different contact graphs essentially never
    agree to 1e-7 relative.  It is a cheap proxy for structural distinctness -- the alternative,
    an assignment distance between layouts, costs an O(n^3) match per hop for a diversity test.
    Entries further than XOVER_WINDOW (relative) below the best are dropped: a parent from a much
    worse basin contributes a half-square this n has already rejected.
    """
    if xy is None or not np.isfinite(s):
        return
    tol = XOVER_DISTINCT * max(abs(s), 1e-9)
    for a in arch:
        if a[0] == xy.shape[0] and abs(a[3] - s) <= tol:
            return
    arch.append((xy.shape[0], xy.copy(), r.copy(), float(s)))
    lo = best_s - XOVER_WINDOW * max(abs(best_s), 1e-9)
    arch[:] = sorted([a for a in arch if a[3] >= lo], key=lambda a: -a[3])[:XOVER_ARCH]


def _perturb(xy, r, rng, n):
    """One basin-hopping move.  Structure-preserving moves dominate: with only ~10-30 hops
    affordable per n, a move that destroys the whole layout is almost never accepted."""
    c = xy.copy()
    mr = float(r.mean())
    m = int(rng.randint(0, 6))
    if m == 0:                                   # the single smallest circle -> the best hole
        idx = np.argsort(r)[:1]
        c[idx] = _hole_points(xy, r, rng, 1, jitter=0.02, greedy=True)
    elif m == 1:                                 # a few smallest circles -> distinct big holes
        k = min(n, 2 + int(rng.randint(0, 3)))
        idx = np.argsort(r)[:k]
        c[idx] = _hole_points(xy, r, rng, k, jitter=0.02, greedy=True)
    elif m == 2:                                 # gentle global jitter
        c = c + rng.normal(0, 0.25 * mr, c.shape)
    elif m == 3:                                 # shake one local cluster
        i = int(rng.randint(n))
        k = min(n, 2 + int(rng.randint(2, 8)))
        idx = np.argsort(((xy - xy[i]) ** 2).sum(1))[:k]
        c[idx] = c[idx] + rng.normal(0, 0.7 * mr, (k, 2))
    elif m == 4:                                 # swap a small circle's spot with a large one's
        o = np.argsort(r)
        lo_k = max(1, n // 3)
        i = int(o[int(rng.randint(0, lo_k))])
        j = int(o[n - 1 - int(rng.randint(0, lo_k))])
        c[i], c[j] = xy[j].copy(), xy[i].copy()
    else:                                        # teleport a random handful anywhere
        k = max(1, int(rng.randint(1, max(2, n // 8))))
        idx = rng.choice(n, k, replace=False)
        c[idx] = rng.uniform(LO + 0.02, HI - 0.02, size=(k, 2))
    return np.clip(c, LO + 1e-4, HI - 1e-4)


# --------------------------------------------------------------------------- driver-facing API
def _load_records():
    """Read bench/records.json if present.  Guarded: absent/garbled -> {} (never raises)."""
    p = os.path.join("bench", "records.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as fh:
            d = json.load(fh)
        recs = d["records"] if isinstance(d, dict) and "records" in d else d
        return {int(k): float(v) for k, v in recs.items()}
    except Exception:
        return {}


def _ref_curve(records):
    """Return ref(n) -> an ESTIMATE of the attainable sum_r, defined for ANY n.

    Optimal Sigma r for csqv grows smoothly like a*sqrt(n)+b: least-squares over the 37 known
    odd-n records fits to within 0.18% relative (artifacts/prof4.py logs the residual), which is
    the same order as the gaps we are actually closing.  This is a two-parameter SCALING LAW, not
    a table: it carries no per-n information and is used ONLY to rank which n deserves the next
    basin hop.  Reported scores always come from the harness, never from here.  For an n whose
    true record IS known the true record is used instead; the fit only covers unseen sizes.
    """
    a, b = _REF_FIT
    if len(records) >= 4:
        ns = np.array(sorted(records), dtype=float)
        vs = np.array([records[int(k)] for k in ns])
        A = np.stack([np.sqrt(ns), np.ones_like(ns)], axis=1)
        try:
            sol, *_ = np.linalg.lstsq(A, vs, rcond=None)
            if np.all(np.isfinite(sol)):
                a, b = float(sol[0]), float(sol[1])
        except Exception:
            pass

    def ref(n):
        if n in records:
            return records[n], True
        return max(a * np.sqrt(n) + b, 1e-6), False
    return ref


def _digits(s, ref_v):
    """The scorer's per-n metric: clamp(-log10(relgap), 0, 7), relgap measured against `ref_v`."""
    if s <= 0 or not np.isfinite(s):
        return 0.0
    relgap = max(0.0, (ref_v - s) / ref_v)
    if relgap <= 1e-7:
        return DIGITS_CAP
    return float(min(DIGITS_CAP, max(0.0, -np.log10(relgap))))


class _MoveBank(object):
    """Which escape class to spend the next escape hop on -- measured, not hand-picked.

    Exactly the estimator `_NState.priority` already uses to choose an n, applied one level down
    to choose a MOVE: value = (digits credited + MV_PRIOR_GAIN) / (CPU spent + MV_PRIOR_CPU).
    Sampling (rather than argmax) is deliberate -- the reward here is very sparse (~10 positive
    hops in a 100 CPU-s run), so an argmax would lock onto whichever class happened to win first.
    Three properties make it safe:

    * **Optimistic prior.**  A class with no measurement yet is valued at 0.15 d/s, above every
      measured rate except graft's, so it gets tried before it can be dismissed.
    * **Floor.**  Every class that is AVAILABLE this hop keeps >= MV_FLOOR of the probability, so
      no class can ever be locked out by early bad luck -- the bank can only skew the mix, never
      delete a move.  This is what keeps the estimate honest: a starved arm stops being measured.
    * **Availability.**  Asked per hop: the grafts need a donor pack, crossover needs two distinct
      local optima in the archive.  An n with neither (a fresh size, an offline re-run outside the
      census) samples only over what its own state actually supports.

    Pooled across n on purpose.  Per-n it would see ~15 escape hops and learn nothing; the
    question "which escape class pays" is about the move, and pooling gives it the whole run.
    """

    __slots__ = ("gain", "cpu", "picks")

    def __init__(self):
        self.gain = dict((c, 0.0) for c in ESCAPE_CLASSES)
        self.cpu = dict((c, 0.0) for c in ESCAPE_CLASSES)
        self.picks = dict((c, 0) for c in ESCAPE_CLASSES)

    def value(self, c):
        return (self.gain[c] + MV_PRIOR_GAIN) / (self.cpu[c] + MV_PRIOR_CPU)

    def order(self, avail, rng):
        """A sampled PREFERENCE ORDER over `avail`, best-valued class most likely first.

        An order, not a single pick, because a class can still decline the hop (a graft whose
        adaptation fails returns None).  The caller walks the order and takes the first move that
        actually produces a candidate, so a declined class costs no hop.
        """
        rest = [c for c in avail if c in self.gain]
        out = []
        while rest:
            w = np.array([self.value(c) for c in rest], dtype=float)
            tot = float(w.sum())
            k = len(rest)
            if tot <= 0 or not np.isfinite(tot):
                p = np.full(k, 1.0 / k)
            else:
                p = w / tot
                if k * MV_FLOOR < 1.0:          # blend toward uniform to hold the floor
                    p = (1.0 - k * MV_FLOOR) * p + MV_FLOOR
                p = p / float(p.sum())
            i = int(np.searchsorted(np.cumsum(p), float(rng.rand()), side="right"))
            i = min(max(i, 0), k - 1)
            out.append(rest.pop(i))
        return out

    def credit(self, c, dd, cpu):
        """Charge a hop's CPU to the class that made the move, and credit any digits it won."""
        if c not in self.gain:
            return
        self.picks[c] += 1
        self.gain[c] *= MV_DECAY
        self.cpu[c] += max(0.0, float(cpu))
        if dd > 0:
            self.gain[c] += float(dd)

    def report(self):
        return dict((c, {"picks": self.picks[c], "cpu": round(self.cpu[c], 3),
                         "gain": round(self.gain[c], 5), "val": round(self.value(c), 5)})
                    for c in ESCAPE_CLASSES)


class _NState(object):
    """Per-n search state, kept alive across the whole call so hops can be interleaved."""

    __slots__ = ("n", "best_xy", "best_r", "best_s", "cur_xy", "cur_r", "cur_s",
                 "stall", "cpu", "gain", "digits", "ref_v", "known", "seeded",
                 # --- iteration 14: the adaptive threshold-accepting walk (see ACCEPT_BAND0)
                 "since_best", "band", "chain_acc",
                 # --- observability (iteration 5): the allocator's own decisions, logged so a
                 # later iteration can tell a STARVED n from a WORKED-AND-STALLED one.
                 "hops", "acc", "seed_cpu", "seed_digits", "d_in",
                 # --- iteration 9: crossover parents (see _arch_add / _crossover_self)
                 "arch")

    def __init__(self, n, ref_v, known):
        self.n = n
        self.best_xy = self.best_r = None
        self.best_s = -np.inf
        self.cur_xy = self.cur_r = None
        self.cur_s = -np.inf
        self.stall = 0
        self.since_best = 0  # hops since this n's BEST last improved (snapback horizon)
        self.band = ACCEPT_BAND0  # adaptive threshold-accepting band, relative
        self.chain_acc = 0   # hops the chain followed (diagnostic)
        self.cpu = 0.0          # cumulative CPU this n has been given
        self.gain = 0.0         # recency-decayed digits this n has actually delivered
        self.digits = 0.0
        self.ref_v = ref_v
        self.known = known      # True iff ref_v is the published record, not the fitted curve
        self.seeded = False
        self.hops = 0           # basin hops spent on this n in phase B
        self.acc = 0            # hops that strictly improved this n's best
        self.seed_cpu = 0.0     # CPU spent in the phase-A seeding pass
        self.seed_digits = 0.0  # digits after seeding (i.e. what phase B started from)
        self.d_in = 0.0         # digits of the committed pack at entry (0 if none)
        self.arch = []          # up to XOVER_ARCH distinct local optima of THIS n, best first

    def priority(self):
        """Estimated marginal digits per CPU-second, with an optimistic prior.

        (gain + PRIOR_GAIN) / (cpu + PRIOR_CPU).  `cpu` is CUMULATIVE, so an n that stops paying
        off decays toward PRIOR_GAIN/cpu and the allocator moves on -- and because the decay is
        monotone in cpu, no n can ever starve: whoever has been given the least time eventually
        becomes the argmax.  `gain` is decayed 0.85x each time this n is picked, so the estimate
        tracks RECENT productivity rather than a lucky first hop.
        """
        if self.known and self.digits >= DIGITS_CAP - 0.1:
            return -1.0         # already at the scoring cap: another digit here is worth nothing
        return (self.gain + PRIOR_GAIN) / (self.cpu + PRIOR_CPU)


class _Trace(object):
    """Append-only JSONL log of the allocator's decisions, for the NEXT iteration to read.

    Iteration 4's allocator improved the score but was unobservable: 18 of 37 n gained nothing
    and there was no way to tell whether they were STARVED (given almost no CPU) or WORKED AND
    STALLED (given CPU and found nothing).  Those two diagnoses call for opposite fixes, so the
    distinction is worth logging.  Entirely optional and fully guarded: if `artifacts/` does not
    exist (an offline re-run elsewhere) or any write fails, the solver runs exactly as before.
    """

    __slots__ = ("fh",)

    def __init__(self, path=os.path.join("artifacts", "alloc_trace.jsonl")):
        self.fh = None
        try:
            if os.path.isdir(os.path.dirname(path)):
                self.fh = open(path, "a")
        except Exception:
            self.fh = None

    def emit(self, rec):
        if self.fh is None:
            return
        try:
            self.fh.write(json.dumps(rec, sort_keys=True) + "\n")
        except Exception:
            self.fh = None

    def close(self):
        if self.fh is not None:
            try:
                self.fh.flush()
                self.fh.close()
            except Exception:
                pass
            self.fh = None


def solve(evaluate, meter, rng, targets, cpu_budget=None):
    """`cpu_budget` is an optional override used only by --self-test (so the test is quick).
    The driver never passes it, so a metered run always gets the full CPU_BUDGET_S.

    ALLOCATION (iteration 4).  Time is no longer sliced per-n up front.  Iterations 1-3 gave n a
    share proportional to n**1.6, which equalizes the NUMBER of basin hops across n; that constant
    was calibrated when every n was cold-started and is wrong now that every n is warm-started
    from the census.  What the scorer pays for is digits, and digits-per-CPU-second is not equal
    across n and cannot be predicted from n: it depends on how much slack that particular
    committed packing still has.  So the allocator MEASURES it -- a seeding pass gives every n a
    guaranteed local optimization, then every subsequent hop goes to the n with the highest
    observed marginal digits-per-second (see `_NState.priority`).
    """
    if not targets:
        return
    targets = sorted(set(int(t) for t in targets))
    t_entry = time.process_time()                     # every deadline is relative to ENTRY
    cpu_stop_all = t_entry + (CPU_BUDGET_S if cpu_budget is None else float(cpu_budget))

    ref = _ref_curve(_load_records())
    states = []
    for n in targets:
        rv, known = ref(n)
        states.append(_NState(n, rv, known))
    trace = _Trace()
    pck_cache = {}                                    # size -> donor pack (or None); disk read once
    bank = _MoveBank()                                # escape-class allocator (iteration 10)
    run_tag = "%d.%06d" % (os.getpid(), int((t_entry * 1e6) % 1e6))

    def make_offer(st):
        def offer(xy, r, s):
            """Route a candidate through the metered evaluate(); keep it if it really counts."""
            if s <= st.best_s or meter.left() <= 0:
                return
            pack = np.concatenate([xy, r[:, None]], axis=1)
            feas, got = evaluate(st.n, pack)
            feas = bool(np.all(feas))
            got = float(np.ravel(got)[0]) if np.ndim(got) else float(got)
            if feas and got > st.best_s:
                st.best_xy, st.best_r, st.best_s = xy.copy(), r.copy(), got
                d = _digits(got, st.ref_v)
                st.gain += max(0.0, d - st.digits)
                st.digits = d
        return offer

    # ---- Phase A: seeding pass.  Every n gets one guaranteed local optimization from the census
    #      (or cold starts when it has no pack of its own), so no n can be starved by the
    #      allocator before it has any feasible packing at all.
    for pos, st in enumerate(states):
        if meter.left() <= 0 or time.process_time() >= cpu_stop_all:
            break
        n = st.n
        offer = make_offer(st)
        t0 = time.process_time()
        left = cpu_stop_all - t0
        # Cap so one expensive n cannot eat the pass, but keep the cap generous (3x its even
        # share) -- an n with no seed needs real time to reach feasibility at all.
        n_stop = t0 + max(0.25, min(left * SEED_FRAC,
                                    left * 3.0 / max(1, len(states) - pos)))

        seed = _census_seed(n, rng)
        exact = bool(seed is not None and seed[2])
        if exact:
            st.d_in = _digits(float(seed[1].sum()), st.ref_v)
        if seed is not None:
            xy, r, s = _local_opt(seed[0], seed[1], n_stop)
            offer(xy, r, s)
            _arch_add(st.arch, xy, r, s, st.best_s)

        # Cold starts.  Skipped when the warm start is exact: a fresh local optimum lands ~1.7
        # digits and cannot beat an already-committed packing, so the CPU is worth far more spent
        # on basin hops.  Always run when there is no seed.
        kinds = () if exact else ((0, 1) if seed is not None else (0, 1, 2))
        for kind in kinds:
            if time.process_time() >= n_stop or meter.left() <= 0:
                break
            xy0 = _start(n, rng, kind)
            xy, r, s = _local_opt(xy0, np.full(n, 0.01), n_stop)
            offer(xy, r, s)
            _arch_add(st.arch, xy, r, s, st.best_s)

        st.cur_xy, st.cur_r, st.cur_s = st.best_xy, st.best_r, st.best_s
        st.seed_cpu = time.process_time() - t0
        st.cpu += st.seed_cpu
        st.seed_digits = st.digits
        st.seeded = True
        # The seeding local optimization is bookkeeping, not evidence about future productivity:
        # it measures the slack the PREVIOUS iteration left behind, not what a hop will buy now.
        st.gain = 0.0

    # ---- Phase B: adaptive basin hopping.  One hop at a time, always to the current argmax of
    #      estimated marginal digits per CPU-second.
    while time.process_time() < cpu_stop_all and meter.left() > 0:
        st = None
        best_p = 0.0
        for cand in states:
            if not cand.seeded:
                continue
            p = cand.priority()
            if p > best_p:
                st, best_p = cand, p
        if st is None:
            break

        n = st.n
        offer = make_offer(st)
        t0 = time.process_time()
        hop_stop = min(cpu_stop_all, t0 + HOP_CPU_CAP)
        st.gain *= GAIN_DECAY
        st.hops += 1
        d_before = st.digits

        mv = "cold"
        if st.cur_xy is None:                         # nothing feasible yet: keep cold-starting
            xy0 = _start(n, rng, 2)
            xy, r, s = _local_opt(xy0, np.full(n, 0.01), hop_stop)
            offer(xy, r, s)
            _arch_add(st.arch, xy, r, s, st.best_s)
            st.cur_xy, st.cur_r, st.cur_s = st.best_xy, st.best_r, st.best_s
        else:
            cxy = cr = None
            if st.stall >= ESCAPE_STALL:
                # The jitter moves have failed ESCAPE_STALL times in a row: the chain is pinned in
                # one contact graph, so switch move CLASS instead of re-rolling the same dice.
                # WHICH class is chosen by the measured bank (iteration 10) rather than by the
                # odd/even alternation plus two 0.5 splits it replaces: those three constants sent
                # 25.5 CPU-s over two runs to patch+xover, which have produced 0.00 digits, while
                # graft -- 0.772 d/s, the best-measured move in the file -- was cut to 5.4 s.
                # Ask each class whether it CAN move first, so a hop is never wasted on a decline.
                dn = _graft_donors(n, states, pck_cache)
                # "inflate", "equal" and "maximin" need nothing but this n's own current
                # layout, so they are the ones that are always available -- including on a size with no census
                # neighbour and an empty archive.
                avail = ["inflate", "equal", "maximin"]
                if dn:
                    avail += ["graft", "patch"]
                if len(st.arch) >= 2:
                    avail.append("xover")
                for mv_try in bank.order(avail, rng):
                    gr = None
                    if mv_try == "xover":
                        gr = _crossover_self(st.cur_xy, st.cur_r, st.arch, st.cur_s, rng)
                    elif mv_try == "patch":
                        gr = _graft_patch(n, rng, dn, st.cur_xy, st.cur_r)
                    elif mv_try == "graft":
                        gr = _graft(n, rng, dn)
                    elif mv_try == "equal":
                        gr = _equalize(st.cur_xy, st.cur_r, rng)
                    elif mv_try == "maximin":
                        gr = _maximin(st.cur_xy, st.cur_r, rng)
                    else:
                        # Inflate-relax: blow the layout up by `g` and let the SLP relax it back.
                        # Strength grows with how long the chain has been stuck.
                        g = 1.0 + 0.02 * (1 + (st.stall - ESCAPE_STALL)) + 0.05 * float(rng.rand())
                        gr = (_inflate_relax(st.cur_xy, st.cur_r, rng, n, min(g, 1.45)),
                              st.cur_r.copy())
                    if gr is not None:
                        cxy, cr = gr
                        mv = mv_try
                        break
            if cxy is None:
                cxy = _perturb(st.cur_xy, st.cur_r, rng, n)
                cr = st.cur_r.copy()
                mv = "jitter"
            xy, r, s = _local_opt(cxy, cr, hop_stop)
            _arch_add(st.arch, xy, r, s, max(st.best_s, s))
            if s > st.best_s:
                offer(xy, r, s)
                st.stall = 0
                st.since_best = 0
            else:
                st.stall += 1
                st.since_best += 1
            # Threshold accepting: follow a slightly worse basin so the chain can cross a ridge,
            # but keep `best` untouched.  The band ADAPTS toward ACCEPT_TARGET (iteration 14): it
            # widens while the chain is refusing everything it is shown and narrows once it is
            # following half of them, so it self-calibrates to this n's own basin spacing rather
            # than to a constant measured on some other n.
            took = bool(s > st.cur_s - st.band * max(st.cur_s, 1e-9))
            if took:
                st.cur_xy, st.cur_r, st.cur_s = xy, r, s
                st.chain_acc += 1
            st.band = float(min(ACCEPT_BAND_MAX, max(
                ACCEPT_BAND0,
                st.band * np.exp(ACCEPT_ETA * (ACCEPT_TARGET - (1.0 if took else 0.0))))))
            # Snap back to the incumbent only after a long excursion has failed to beat it.
            if st.since_best >= SNAPBACK and st.best_xy is not None:
                st.cur_xy = st.best_xy.copy()
                st.cur_r = st.best_r.copy()
                st.cur_s = st.best_s
                st.since_best = 0

        hop_cpu = time.process_time() - t0
        st.cpu += hop_cpu
        bank.credit(mv, st.digits - d_before, hop_cpu)
        if st.digits > d_before:
            st.acc += 1
        trace.emit({"run": run_tag, "ev": "hop", "n": n, "hop": st.hops, "mv": mv,
                    "cpu_hop": round(time.process_time() - t0, 4),
                    "cpu_n": round(st.cpu, 3),
                    "t": round(time.process_time() - t_entry, 3),
                    "d": round(st.digits, 5), "dd": round(st.digits - d_before, 6),
                    "band": float("%.3e" % st.band), "prio": round(best_p, 5)})

    # ---- Per-n summary: the one artifact a later iteration actually reads.
    for st in states:
        trace.emit({"run": run_tag, "ev": "n", "n": st.n, "known": bool(st.known),
                    "cpu": round(st.cpu, 3), "seed_cpu": round(st.seed_cpu, 3),
                    "hops": st.hops, "acc": st.acc, "chain_acc": st.chain_acc,
                    "band": float("%.3e" % st.band),
                    "d_in": round(st.d_in, 5), "d_seed": round(st.seed_digits, 5),
                    "d_out": round(st.digits, 5),
                    "gain_seed": round(st.seed_digits - st.d_in, 5),
                    "gain_hops": round(st.digits - st.seed_digits, 5)})
    trace.emit({"run": run_tag, "ev": "end", "cpu_total": round(time.process_time() - t_entry, 3),
                "n_targets": len(states), "evals_left": int(meter.left()),
                "bank": bank.report()})
    trace.close()
    return


# --------------------------------------------------------------------------- self-test
def _self_test():
    import sys

    class _Meter:
        def __init__(self, budget):
            self.budget = budget
            self.used = 0

        def left(self):
            return self.budget - self.used

        def tick(self):
            return self.used

    recs_path = os.path.join("bench", "records.json")
    records = {}
    if os.path.exists(recs_path):
        records = {int(k): float(v) for k, v in
                   json.load(open(recs_path))["records"].items()}

    def make(meter, store):
        def evaluate(n, packing):
            p = np.asarray(packing, dtype=float)
            single = (p.ndim == 2)
            if single:
                p = p[None]
            B = p.shape[0]
            meter.used += B
            xy, r = p[..., :2], p[..., 2]
            wall = np.minimum.reduce([xy[..., 0] - LO, HI - xy[..., 0],
                                      xy[..., 1] - LO, HI - xy[..., 1]]) - r
            d = np.sqrt(((xy[:, :, None, :] - xy[:, None, :, :]) ** 2).sum(-1))
            di = np.arange(n)
            d[:, di, di] = np.inf
            pair = d - (r[:, :, None] + r[:, None, :])
            feas = ((r > 0).all(axis=1) & (wall.min(axis=1) >= -1e-9)
                    & (pair.reshape(B, -1).min(axis=1) >= -1e-9))
            s = np.where(feas, r.sum(axis=1), -np.inf)
            for b in range(B):
                if feas[b] and s[b] > store.get(n, (-np.inf,))[0]:
                    store[n] = (float(s[b]), p[b].copy())
            if single:
                return bool(feas[0]), float(s[0])
            return feas, s
        return evaluate

    def check(call, n, s, pack):
        xy, r = pack[:, :2], pack[:, 2]
        wall = min(np.min(xy[:, 0] - r - LO), np.min(HI - xy[:, 0] - r),
                   np.min(xy[:, 1] - r - LO), np.min(HI - xy[:, 1] - r))
        dm = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dm, np.inf)
        ps = float((dm - (r[:, None] + r[None, :])).min())
        good = (pack.shape[0] == n and r.min() > 0 and wall >= -1e-9 and ps >= -1e-9)
        frac = s / records[n] if n in records else float("nan")
        print("%-22s n=%-3d sum_r=%.9f frac_of_record=%.6f wall=%.2e pair=%.2e %s"
              % (call, n, s, frac, wall, ps, "OK" if good else "INFEASIBLE"))
        return good

    ok = True
    # Two calls in ONE process: the second must still produce feasible packings (this is the
    # "absolute module deadline" trap from MISSION.md).
    for call in (1, 2):
        meter = _Meter(500_000)
        store = {}
        tgt = [27, 45] if call == 1 else [99]          # also exercises len(targets)==1
        t0 = time.process_time()
        solve(make(meter, store), meter, np.random.RandomState(call), tgt, cpu_budget=4.0)
        cpu = time.process_time() - t0
        for n in tgt:
            if n not in store:
                print("FAIL call %d: no feasible packing for n=%d" % (call, n))
                ok = False
                continue
            ok = check("call %d" % call, n, store[n][0], store[n][1]) and ok
        print("  call %d cpu=%.1fs evals=%d" % (call, cpu, meter.used))

    # n with NO committed pack must work: even n (structure transfer from n-1/n+1) and a size
    # far outside the census (pure cold start).  Both are what the offline re-run exercises.
    for n_new, budget_s in ((8, 1.5), (28, 2.0), (140, 3.0)):
        meter = _Meter(20_000)
        store = {}
        solve(make(meter, store), meter, np.random.RandomState(7), [n_new], cpu_budget=budget_s)
        if n_new in store:
            ok = check("uncommitted-n", n_new, store[n_new][0], store[n_new][1]) and ok
        else:
            print("FAIL: no packing for n=%d (no committed pack -> must still solve)" % n_new)
            ok = False

    # meter exhaustion must not raise and must not corrupt anything.
    meter = _Meter(3)
    solve(make(meter, {}), meter, np.random.RandomState(3), [31], cpu_budget=2.0)
    print("exhausted-meter call survived, used=%d" % meter.used)

    print("SELF-TEST: %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print("usage: python3 tools/solver.py --self-test")
