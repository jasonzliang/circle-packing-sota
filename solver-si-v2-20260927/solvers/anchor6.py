"""SLP circle-packing solver -- sequential linear programming on (radii, centre displacements).

WHY THIS SHAPE (the transformation away from the seed solver)
------------------------------------------------------------
The seed solver treated the packing as a physics problem: repel points, then hand every circle the
*guaranteed* radius r_i = min(wall_i, min_j d_ij/2).  That cap forces near-equal radii, which is exactly
the wrong shape for csqv -- the record packings are strongly heterogeneous (a few big circles plus a
graded cascade of fillers).  The seed therefore plateaus ~0.6 digits from the records.

This solver changes the representation.  For FIXED centres, choosing the radii is a *linear program*

    max  sum_i r_i     s.t.   r_i + r_j <= d_ij  (all pairs),   0 <= r_i <= wall_i,

whose optimum is generally very unequal -- the LP finds the big/small alternation the min-rule cannot.
Moving the centres is then folded into the SAME LP by linearising the pair distance around the current
centres with a trust region |dx| <= delta:

    r_i + r_j <= d_ij + u_ij . (dx_i - dx_j),      u_ij = (p_i - p_j)/d_ij.

Because the Euclidean norm is convex, ||(p_i-p_j) + (dx_i-dx_j)|| >= d_ij + u_ij.(dx_i-dx_j), so the
linearised constraint is a CONSERVATIVE under-estimate: every LP solution is feasible for the true
problem no matter how big delta is.  The wall constraints are linear in the displacement already, so
they are exact.  dx = 0, r = r_current is always LP-feasible, hence each step is monotone non-decreasing
in sum r.  An ADAPTIVE trust region (stay while productive, contract x0.20 on a stall) drives the packing into a
genuine local optimum in ~17 LPs instead of the ~120 a fixed geometric decay cost.  A pack read
back from the census is already AT that fixed point, so warm starts re-polish at delta=0.004 and spend
the slice on basin hopping instead (measured: a full re-anneal of csqv99 costs 1.7 CPU-s for 0 digits).

Then basin hopping escapes that local optimum: kick the configuration (relocate the smallest circles,
or jitter every centre), re-anneal, keep the result only if it is better.  Several kick kinds are
rotated so the search never commits to one line of attack.

Everything that is meant to count is routed through the metered `evaluate()`; nothing is written by
this file.  Warm starts read `bench/packs/csqv<n>.pck` when it exists and fall back to a cold random
start otherwise (guarded -- `targets` may contain n with no committed pack).

Budgeting: the CPU backstop, not the 500k evaluation budget, binds a per-n optimiser, so the CPU
deadline is computed at ENTRY (`time.process_time() + budget`) and the per-n slice is derived from
`len(targets)`, so repeated calls in one process and `len(targets) == 1` both behave.
"""
import json
import os
import time

import numpy as np

try:
    from scipy.optimize import linprog, linear_sum_assignment
    from scipy.sparse import coo_matrix
    _HAVE_SCIPY = True
except Exception:                                       # pragma: no cover - scipy is declared present
    _HAVE_SCIPY = False

LO, HI = -0.5, 0.5
CPU_TOTAL_S = 114.0          # the backstop ends only THIS call; the driver still harvests, so run close to it
PACK_DIR = os.path.join("bench", "packs")


# ---------------------------------------------------------------- geometry helpers (local, no harness)
def _walls(xy):
    x, y = xy[:, 0], xy[:, 1]
    return np.minimum.reduce([x - LO, HI - x, y - LO, HI - y])


def _dmat(xy):
    d = xy[:, None, :] - xy[None, :, :]
    D = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return D


def _repair(xy, r):
    """Make (xy, r) STRICTLY feasible and then locally maximal in r (monotone coordinate ascent).

    The LP is solved to a finite tolerance, so its radii can overshoot by ~1e-7; a single global
    down-scale restores feasibility, after which raising each r_i to its own exact maximum given the
    others only ever increases sum r and never breaks feasibility."""
    w = _walls(xy)
    r = np.minimum(np.clip(r, 0.0, None), w)
    D = _dmat(xy)
    s = ((r[:, None] + r[None, :]) / D).max()
    if s > 1.0:
        r = r / s
    for _ in range(2):
        for i in range(len(r)):
            r[i] = min(w[i], (D[i] - r).min())
    return np.maximum(r - 1e-13, 1e-12)


# ---------------------------------------------------------------- one SLP step
def _lp(xy, r, delta, iu, ju, dij, ux, uy):
    """The trust-region LP over z = [r (n), dx (n), dy (n)] restricted to the pair set (iu, ju)."""
    n = len(xy)
    P = len(iu)
    rows, cols, vals, b = [], [], [], []
    rb = 0
    ar = np.arange(n)
    one = np.ones(n)
    for sgn, blk, rhs in ((-1.0, 1, xy[:, 0] - LO), (1.0, 1, HI - xy[:, 0]),
                          (-1.0, 2, xy[:, 1] - LO), (1.0, 2, HI - xy[:, 1])):
        rows.append(rb + ar); cols.append(ar); vals.append(one)
        rows.append(rb + ar); cols.append(blk * n + ar); vals.append(sgn * one)
        b.append(rhs); rb += n
    ap = np.arange(P)
    onep = np.ones(P)
    rows.append(rb + ap); cols.append(iu); vals.append(onep)
    rows.append(rb + ap); cols.append(ju); vals.append(onep)
    rows.append(rb + ap); cols.append(n + iu); vals.append(-ux)
    rows.append(rb + ap); cols.append(n + ju); vals.append(ux)
    rows.append(rb + ap); cols.append(2 * n + iu); vals.append(-uy)
    rows.append(rb + ap); cols.append(2 * n + ju); vals.append(uy)
    b.append(dij); rb += P

    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(rb, 3 * n)).tocsr()
    c = np.zeros(3 * n)
    c[:n] = -1.0
    bounds = [(0.0, 0.5)] * n + [(-delta, delta)] * (2 * n)
    try:
        return linprog(c, A_ub=A, b_ub=np.concatenate(b), bounds=bounds, method="highs")
    except Exception:
        return None


def _slp_step(xy, r, delta, iu0, ju0):
    """One trust-region LP step, solved by LAZY CONSTRAINT GENERATION.

    Only pairs that are *near-active* enter the first LP; the solution is then checked against every
    pair in the conservative candidate set and violated ones are added and the LP re-solved.  On exit
    no linearised pair constraint is violated, so the answer is identical to the full LP (verified: it
    matches the dense build to the last digit on n=27/55/99 at delta 3e-2 .. 5e-4) -- but the LP itself
    carries ~250 rows instead of ~4900 at n=99.  Returns (new centres, new radii)."""
    n = len(xy)
    D = _dmat(xy)
    w = _walls(xy)
    dall = D[iu0, ju0]
    # Conservative candidate prune ONLY: a pair can bind only if it can be reached at all. Radii are
    # bounded by the wall distance, which the trust region can grow by at most delta -- anything looser
    # than this silently drops binding constraints and the repair down-scale eats the gain (measured).
    cand = dall <= w[iu0] + w[ju0] + 6.0 * delta
    IU, JU, DIJ = iu0[cand], ju0[cand], dall[cand]
    if len(IU) == 0:
        return None, None
    UX = (xy[IU, 0] - xy[JU, 0]) / DIJ
    UY = (xy[IU, 1] - xy[JU, 1]) / DIJ
    act = DIJ - (r[IU] + r[JU]) <= 4.0 * delta + 1e-9      # near-active working set
    if not act.any():
        act[np.argmin(DIJ - (r[IU] + r[JU]))] = True
    dx = dy = rr = None
    for _ in range(8):
        res = _lp(xy, r, delta, IU[act], JU[act], DIJ[act], UX[act], UY[act])
        if res is None or not res.success or res.x is None:
            return None, None
        z = res.x
        rr, dx, dy = z[:n], z[n:2 * n], z[2 * n:3 * n]
        viol = rr[IU] + rr[JU] - DIJ - UX * (dx[IU] - dx[JU]) - UY * (dy[IU] - dy[JU])
        bad = (viol > 1e-11) & (~act)
        if not bad.any():
            break
        idx = np.where(bad)[0]
        if len(idx) > 4 * n:                               # add the worst violators first
            idx = idx[np.argsort(-viol[idx])[:4 * n]]
        act[idx] = True
    return xy + np.stack([dx, dy], 1), rr


SHRINK = 0.20                # trust-region contraction once a radius stalls
STALL_TOL = 0.02             # a step of radius d can add O(n*d); below tol*d the radius is spent
MAX_REP = 6                  # cap on consecutive LPs at one radius
TAIL_EPS = 1e-11             # a radius level gaining less than this in sum_r did no work at all
TAIL_K = 2                   # stop the anneal after this many CONSECUTIVE such levels (0 = never)


def _anneal(xy, r, iu0, ju0, d0, tstop):
    """Anneal the trust region down from d0; monotone in sum r. Returns (xy, r, sum_r).

    ADAPTIVE schedule (iter 3).  The old rule shrank delta by a fixed 0.85 after EVERY LP, so
    0.03 -> 1e-10 cost a fixed ~120 LPs whether or not any of them were doing work.  Since the
    linearisation is conservative, every LP is a true (never-rejected) improvement -- the only reason
    to shrink is that a large radius over-penalises fine adjustment.  So instead: STAY at a radius
    while it is still productive, and contract HARD (x0.20) the moment a step stalls.  The stall
    threshold scales with delta because a step of radius delta can add at most O(n*delta) to sum r.
    Same deep small-delta tail, ~17 LPs per anneal instead of ~120.

    Measured (8 CPU-s of the kick loop from the committed census, digits gained):
        n=27  fixed +0.579  ->  adaptive +0.721   (25 kicks -> 186)
        n=55  fixed +0.139  ->  adaptive +0.728   (14 kicks ->  87)
        n=99  fixed +0.168  ->  adaptive +0.708   ( 6 kicks ->  40)
    This does NOT contradict the iter-2 "never truncate the anneal" result: the tail is still run to
    1e-10, it is simply reached in fewer LPs."""
    best = float(r.sum())
    d = d0
    barren = 0
    while time.process_time() < tstop and d > 1e-10:
        lvl_in = best
        for _ in range(MAX_REP):
            if time.process_time() >= tstop:
                return xy, r, best
            nxy, nr = _slp_step(xy, r, d, iu0, ju0)
            if nxy is None:
                return xy, r, best
            nr = _repair(nxy, nr)
            s = float(nr.sum())
            gain = s - best
            if s > best:
                best, xy, r = s, nxy, nr
            if gain <= STALL_TOL * d:                  # this radius is spent -- contract
                break
        if TAIL_K:
            # TAIL STOP (iter 9).  The loop above ran to d <= 1e-10 whatever happened, and a traced
            # profile of 146 real kicks says the whole sub-1e-4 tail is DEAD WEIGHT: it costs 41% of
            # the anneal's CPU and adds 1.6e-13 to sum_r in total (every counted digit is made at
            # d in [6e-3, 1.2e-3]).  Each dead level still pays for one LP.  So instead of a fixed
            # floor -- which would be a truncation, and iter 2 measured those as losers -- stop on
            # EVIDENCE: after TAIL_K consecutive levels that gained essentially nothing, this basin is
            # at its fixed point and the remaining levels cannot find it anything.  A level that does
            # real work always resets the counter, so a deep tail is still run wherever it pays.
            if best - lvl_in <= TAIL_EPS:
                barren += 1
                if barren >= TAIL_K:
                    return xy, r, best
            else:
                barren = 0
        d *= SHRINK
    return xy, r, best


# ---------------------------------------------------------------- starts
def _read_pack(n):
    """Warm start from the committed census -- GUARDED: any missing/odd file falls back to cold."""
    p = os.path.join(PACK_DIR, "csqv%d.pck" % n)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            lines = [l.split() for l in fh.read().split("\n")[2:] if l.strip()]
        a = np.array([[float(v) for v in row[:3]] for row in lines], dtype=float)
    except Exception:
        return None
    if a.shape != (n, 3):
        return None
    return a


def _records():
    """The read-only record table (sanctioned reading, not a coordinate table). {} if unreadable."""
    try:
        with open(os.path.join("bench", "records.json")) as fh:
            d = json.load(fh)
        if isinstance(d, dict) and "records" in d:
            d = d["records"]
        return {int(k): float(v) for k, v in d.items()}
    except Exception:
        return {}


def _at_cap(n, _cache={}):
    """True iff the COMMITTED pack for n is already within 1e-7 relative of the record (7-digit cap)."""
    if not _cache:
        _cache.update(_records() or {-1: 0.0})
    rec = _cache.get(int(n))
    if not rec:
        return False
    a = _read_pack(int(n))
    if a is None or a.shape != (int(n), 3):
        return False
    return (rec - float(a[:, 2].sum())) / rec <= 1e-7


def _cold_start(n, rng):
    """Cold start that works for ANY n: a jittered near-square lattice inside the box."""
    k = int(np.ceil(np.sqrt(n)))
    g = (np.arange(k) + 0.5) / k - 0.5
    gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)[:n]
    pts = pts + rng.normal(0.0, 0.25 / k, size=pts.shape)
    return np.clip(pts, LO + 1e-3, HI - 1e-3)


_PACK_CACHE = {}


def _pack_cached(n):
    """Cached `_read_pack` -- the census cannot change during a solve() (writes are reverted), and the
    transfer kick would otherwise re-read a neighbour's file on every kick."""
    if n not in _PACK_CACHE:
        _PACK_CACHE[n] = _read_pack(n)
    return _PACK_CACHE[n]


def _symmetrize(xy, rng):
    """SYMMETRY-PROJECTION move (iter 7): snap the arrangement onto a random reflection axis.

    The published csqv packings are largely symmetric, and NO move in kinds 0-6 can turn an asymmetric
    contact graph into a symmetric one: jitter/mirror/cluster all preserve or destroy structure at
    random.  Here the whole configuration is mirrored about one of the four natural axes (x=0, y=0,
    y=x, y=-x), each circle is matched to the mirror of some circle by an OPTIMAL assignment (the cost
    matrix is symmetric because a reflection is an isometry, so the matching is essentially an
    involution), and every centre is moved a fraction t toward the midpoint of its matched pair.  At
    t=1 the result is exactly symmetric; a self-matched circle lands exactly on the axis.  The
    re-anneal then re-optimises inside that new topology."""
    k = int(rng.randint(4))
    if k == 0:
        M = np.stack([-xy[:, 0], xy[:, 1]], 1)
    elif k == 1:
        M = np.stack([xy[:, 0], -xy[:, 1]], 1)
    elif k == 2:
        M = xy[:, ::-1].copy()
    else:
        M = -xy[:, ::-1]
    C = ((xy[:, None, :] - M[None, :, :]) ** 2).sum(-1)
    try:
        _, ci = linear_sum_assignment(C)
    except Exception:
        return xy + rng.normal(0.0, 0.01, size=xy.shape)
    t = rng.uniform(0.5, 1.0)
    return xy + t * (0.5 * (xy + M[ci]) - xy)


# iter 10: donor set WIDENED -2,+-4 -> +-2,+-4,+-6.  This does NOT re-weight the kick draw (iter 8
# measured any tilt of that draw as a loss in BOTH directions); it buys variety INSIDE kind 8's
# fixed slot, which is exactly what the iter-8 mechanism says pays.  Paired arm probe
# (artifacts/probe_iter10_offsets.py, 3 n x 8 seeds x 1.2 CPU-s, digits gained):
#     +-2,+-4 (iter 7-9)  +0.8797     +-2..+-8  +0.9866     +-2..+-6  +1.1152
# Both widenings beat the shipped set, so the DIRECTION is consistent (the size of the gap is
# inside the heavy tail's noise -- 0 jackpots in 24 runs at this slice length).
_TRANSFER_OFF = (-2, 2, -4, 4, -6, 6)


def _transfer(xy, r, rng):
    """CROSS-n TRANSFER move (iter 7): import a neighbouring n's committed topology.

    The census is 37 independent searches that never talk to each other, yet csqv(n) and csqv(n+-2)
    are structurally close.  This move takes a neighbour's committed pack, deletes its smallest
    circles (m > n) or appends random fillers (m < n) to reach exactly n centres, and jitters.  It is
    the only move whose proposal comes from OUTSIDE this n's own search history, so it can reach a
    topology this walker could never have generated.  Falls back to a jitter when no neighbour is
    committed (targets may contain fresh n)."""
    n = len(xy)
    cands = []
    for off in _TRANSFER_OFF:
        m = n + off
        if m >= 2 and _pack_cached(m) is not None:
            cands.append(m)
    if not cands:
        return xy + rng.normal(0.0, 0.02, size=xy.shape)
    m = int(cands[int(rng.randint(len(cands)))])
    P = _pack_cached(m)
    cx, rr = P[:, :2].copy(), P[:, 2]
    if m > n:
        cx = cx[np.argsort(rr)[m - n:]]
    elif m < n:
        # iter 8: fillers go into the DEEPEST HOLES, not uniform-random spots (see _insert_centres).
        cx = _insert_centres(cx, rr, n - m, rng)
    return cx + rng.normal(0.0, 0.004, size=cx.shape)


DEEP_INSERT = False          # deepest-hole insertion for new centres -- MEASURED AND REJECTED (iter 8);
                             # uniform-random fillers beat it, see solve() for the numbers
_DEEP_POOL = 384             # candidate points sampled per insertion round


def _insert_centres(cx, cr, k, rng):
    """Place k NEW centres among circles (cx, cr) -- at the DEEPEST HOLES when DEEP_INSERT.

    Both census-recombination moves (transfer, crossover) have to top up a short configuration to
    exactly n centres, and iter 7 shipped the crudest possible filler: uniform-random points, which
    land on top of existing circles as often as not and are immediately shrunk to nothing by the
    repair.  A hole-seeking filler instead scores a random pool of candidate points by their
    CLEARANCE -- distance to the nearest existing circle's *boundary*, capped by distance to the
    nearest wall -- and takes the best, then re-scores against the point it just placed.  This is
    NOT the iter-2 "hole-targeted relocation" negative result: that moved existing circles inside an
    already-converged packing (and the anneal simply undid it); this decides where a genuinely new,
    radius-less circle enters a configuration that is about to be re-annealed anyway."""
    if k <= 0:
        return cx
    if not DEEP_INSERT or len(cx) == 0:
        return np.concatenate([cx, rng.uniform(LO + 0.01, HI - 0.01, size=(k, 2))], 0)
    pts = np.empty((k, 2))
    ax, ar = cx, cr
    for i in range(k):
        q = rng.uniform(LO + 0.005, HI - 0.005, size=(_DEEP_POOL, 2))
        d = np.sqrt(((q[:, None, :] - ax[None, :, :]) ** 2).sum(-1)) - ar[None, :]
        clear = d.min(1)
        wall = np.minimum(q - LO, HI - q).min(1)
        best = int(np.argmax(np.minimum(clear, wall)))
        pts[i] = q[best]
        ax = np.concatenate([ax, q[best][None, :]], 0)
        ar = np.concatenate([ar, [0.0]])
    return np.concatenate([cx, pts], 0)


def _crowded_drop(cx, cr, k, rng):
    """Drop the k most CROWDED circles -- the ones with the least room of their own.

    When a recombination hands back more than n centres, something has to go.  Dropping the k
    smallest radii (what `_transfer` does) is wrong for a spliced configuration, because the radii
    come from two different packings and are not comparable; what matters is which centres are
    fighting for the same space.  Own-room = min over the others of (centre distance - the other's
    radius), capped by the wall distance; the k smallest are removed."""
    m = len(cx)
    if k <= 0 or m - k < 1:
        return cx, cr
    D = np.sqrt(((cx[:, None, :] - cx[None, :, :]) ** 2).sum(-1)) - cr[None, :]
    np.fill_diagonal(D, np.inf)
    room = np.minimum(D.min(1), np.minimum(cx - LO, HI - cx).min(1))
    keep = np.argsort(room)[k:]
    return cx[keep], cr[keep]


def _crossover(xy, r, rng):
    """CENSUS CROSSOVER move (iter 8): splice a half-plane of a neighbouring n's pack into this one.

    Iter 7 measured that the single largest gain of the run came from letting one n read another n's
    committed pack (kind 8, cross-n transfer) -- but transfer REPLACES the incumbent wholesale, so
    everything this n's own search has learned is thrown away and only the neighbour's topology
    survives.  Crossover is the recombination that transfer is the degenerate case of: cut the square
    with a random line, keep the incumbent's circles on one side and the neighbour's on the other,
    then reconcile the seam -- drop the most crowded centres if the splice came out long, insert into
    the deepest holes if it came out short.  The child inherits a *different* contact graph from
    either parent along the seam, which is exactly the topology change a within-n kick cannot reach,
    and unlike transfer it is a mixture, so the census behaves as a POPULATION rather than a store.
    Falls back to a jitter when no neighbour is committed."""
    n = len(xy)
    cands = [n + off for off in _TRANSFER_OFF
             if n + off >= 2 and _pack_cached(n + off) is not None]
    if not cands:
        return xy + rng.normal(0.0, 0.02, size=xy.shape)
    P = _pack_cached(int(cands[int(rng.randint(len(cands)))]))
    oxy, orr = P[:, :2], P[:, 2]
    th = rng.uniform(0.0, 2.0 * np.pi)
    u = np.array([np.cos(th), np.sin(th)])
    ps, po = xy @ u, oxy @ u
    # Cut at a random quantile of the incumbent's own projection, so neither parent is degenerate.
    c = float(np.quantile(ps, rng.uniform(0.25, 0.75)))
    ks, ko = ps <= c, po > c
    if ks.sum() < 1 or ko.sum() < 1:
        return xy + rng.normal(0.0, 0.02, size=xy.shape)
    cx = np.concatenate([xy[ks], oxy[ko]], 0)
    cr = np.concatenate([r[ks], orr[ko]], 0)
    if len(cx) > n:
        cx, cr = _crowded_drop(cx, cr, len(cx) - n, rng)
    elif len(cx) < n:
        cx = _insert_centres(cx, cr, n - len(cx), rng)
    cx = cx + rng.normal(0.0, 0.003, size=cx.shape)
    # Two parents can hold EXACTLY coincident centres (a pack produced by a transfer from this very
    # neighbour keeps its coordinates verbatim), which puts a 0 in the distance matrix and NaNs the LP.
    if len(cx) > 1 and _dmat(cx).min() <= 0.0:
        cx = cx + rng.normal(0.0, 1e-4, size=cx.shape)
    return cx


# ---------------------------------------------------------------- D4 SHELL CROSSOVER (iter 12)
# Standing lead 1(d) after iter 11: "compose a D4 element with the cross-n transfer so a donor arrives
# in a different orientation RELATIVE to the fillers".  Composing D4 with `_transfer` as shipped is a
# no-op -- transfer replaces the incumbent wholesale, and a D4 image of the whole donor is congruent to
# it, so the anneal sees the same problem.  The composition only bites when the donor meets something
# of the incumbent's along a seam, and iter 11 measured which seam shape is free: the CHEBYSHEV shell,
# the D4-invariant region shape of a square.  So: keep the incumbent on one side of a Chebyshev
# threshold, take the DONOR's other side, and apply a random D4 element to the donor's part before
# splicing.  Both parts keep every wall distance and every within-part mutual distance exactly (a D4
# element maps the square, and the Chebyshev region, onto itself); only the annulus seam is new.
# This is the crossover of iter 8 with the free seam of iter 11 -- and unlike iter 8's random-line
# crossover it cannot rotate a part out of the square or change its density.
SHELLX_P = 0.0               # P(kind 8's slot performs a D4 shell crossover) -- probe sets this


def _d4_apply(pts, g):
    """Apply element g (0 = identity, 1-7 = the non-identity elements) of the square's group D4."""
    x, y = pts[:, 0], pts[:, 1]
    if g == 0:
        return pts
    if g == 1:
        nx, ny = -y, x
    elif g == 2:
        nx, ny = -x, -y
    elif g == 3:
        nx, ny = y, -x
    elif g == 4:
        nx, ny = x, -y
    elif g == 5:
        nx, ny = -x, y
    elif g == 6:
        nx, ny = y, x
    else:
        nx, ny = -y, -x
    return np.stack([nx, ny], 1)


def _shell_crossover(xy, r, rng):
    """Splice a neighbouring n's pack across a CHEBYSHEV seam, with the donor re-oriented by D4.

    Iter 8's crossover cut the square with a random LINE, which is not a shape the square's symmetry
    respects: the two parents' pieces meet along an arbitrary chord, and neither piece can be moved
    without breaking its wall distances.  A Chebyshev threshold `max(|x|,|y|) = t` cuts the square into
    a CORE and a RING, both of which every element of D4 maps onto itself -- so the donor's piece may
    arrive rotated or reflected and still touch the same walls at the same distances, with all of its
    internal contacts intact.  The child is the incumbent's core plus the donor's (re-oriented) ring,
    or the flip; only the annulus seam is a genuinely new piece of contact graph, and the eight
    orientations give eight different seams from one donor.  Falls back to a jitter with no donor."""
    n = len(xy)
    cands = [n + off for off in _TRANSFER_OFF
             if n + off >= 2 and _pack_cached(n + off) is not None]
    if not cands:
        return xy + rng.normal(0.0, 0.02, size=xy.shape)
    P = _pack_cached(int(cands[int(rng.randint(len(cands)))]))
    oxy = _d4_apply(P[:, :2].copy(), int(rng.randint(8)))
    orr = P[:, 2]
    chs = np.maximum(np.abs(xy[:, 0]), np.abs(xy[:, 1]))
    cho = np.maximum(np.abs(oxy[:, 0]), np.abs(oxy[:, 1]))
    t = float(np.quantile(chs, rng.uniform(0.25, 0.75)))
    if rng.uniform() < 0.5:
        ks, ko = chs <= t, cho > t          # incumbent CORE + donor RING
    else:
        ks, ko = chs > t, cho <= t          # incumbent RING + donor CORE
    if ks.sum() < 1 or ko.sum() < 1:
        return xy + rng.normal(0.0, 0.02, size=xy.shape)
    cx = np.concatenate([xy[ks], oxy[ko]], 0)
    cr = np.concatenate([r[ks], orr[ko]], 0)
    if len(cx) > n:
        cx, cr = _crowded_drop(cx, cr, len(cx) - n, rng)
    elif len(cx) < n:
        cx = _insert_centres(cx, cr, n - len(cx), rng)
    cx = cx + rng.normal(0.0, 0.003, size=cx.shape)
    # Two parents can hold EXACTLY coincident centres (a pack produced by a transfer from this very
    # neighbour keeps its coordinates verbatim), which puts a 0 in the distance matrix and NaNs the LP.
    if len(cx) > 1 and _dmat(cx).min() <= 0.0:
        cx = cx + rng.normal(0.0, 1e-4, size=cx.shape)
    return cx


# ---------------------------------------------------------------- D4 SHELL move (iter 11)
# SHIPPED at 0.5 (iter 11).  Two independent paired probes (artifacts/probe_iter11_shell.py, 24 paired
# runs each, digits gained) put the shared-slot arm ahead of the shipped set BOTH times:
#     n=55/73/91 x 8 seeds x 1.2s:  A shipped +0.0690   B share kind 5 +0.1214   C 10th kind +0.0192
#     n=33/47/49/81/89/97 x 4:      A shipped +1.0378   B share kind 5 +1.8591   C 10th kind +5.1190
# B is the only arm that beats A in both; C's win in the second probe is one jackpot in 24 runs and it
# is LAST in the first, which is exactly the iter-8 finding (a tenth kind dilutes a nine-kind draw).
# Sharing a slot leaves the draw over kinds untouched and buys variety INSIDE it -- iter 10's lever.
SHELL_P = 0.5                # P(kind 5's slot performs a D4 shell move instead of a sub-rectangle mirror)


def _d4_shell(xy, rng):
    """D4 SHELL move: apply one non-identity symmetry of the SQUARE to a Chebyshev shell of centres.

    Every one of the nine shipped kicks either perturbs centres locally (0-3, 6), contracts them (4),
    mirrors a sub-rectangle about its OWN axis (5), symmetrises the whole configuration (7), or imports
    a neighbour's pack (8).  None of them is an isometry of the UNIT SQUARE applied to a subset -- and
    that is the one rigid motion that is free here: a rotation by 90/180/270 degrees or a reflection in
    an axis/diagonal maps the square onto itself, so every moved circle keeps its four wall distances
    AND all of its distances to the other moved circles EXACTLY.  Only the contacts ACROSS the seam
    change, so the move rewrites the contact graph at the shell boundary while leaving the packing's
    interior structure -- which took the whole run to find -- untouched.

    The subset is a shell in the Chebyshev (L-infinity) norm, a <= max(|x|,|y|) <= b, because that is
    the D4-invariant region shape of a square: a shell maps onto ITSELF under the group, so the moved
    circles stay in the same annulus and local density is preserved.  Falls back to a small jitter when
    the shell is empty or holds everything (a whole-configuration isometry is a no-op)."""
    n = len(xy)
    ch = np.maximum(np.abs(xy[:, 0]), np.abs(xy[:, 1]))
    a = rng.uniform(0.0, 0.35)
    b = a + rng.uniform(0.06, 0.50)
    m = (ch >= a) & (ch <= b)
    k = int(m.sum())
    if k < 2 or k == n:
        return xy + rng.normal(0.0, 0.01, size=xy.shape)
    sx, sy = xy[m, 0], xy[m, 1]
    g = int(rng.randint(7))
    if g == 0:                                           # rot 90
        nx, ny = -sy, sx
    elif g == 1:                                         # rot 180
        nx, ny = -sx, -sy
    elif g == 2:                                         # rot 270
        nx, ny = sy, -sx
    elif g == 3:                                         # reflect in the x axis
        nx, ny = sx, -sy
    elif g == 4:                                         # reflect in the y axis
        nx, ny = -sx, sy
    elif g == 5:                                         # reflect in y = x
        nx, ny = sy, sx
    else:                                                # reflect in y = -x
        nx, ny = -sy, -sx
    out = xy.copy()
    out[m, 0], out[m, 1] = nx, ny
    return out


def _kick(xy, r, rng, kind):
    """Basin-hopping move -- SEVEN kinds, kinds 4-6 STRUCTURAL (iter 4).

    Kinds 0-3 nudge centres; they leave the contact graph's topology largely intact, and the notes'
    measurement was that greedy hopping on them saturates a plateau at small n with hundreds of kicks
    still unspent.  Kinds 4-6 change the topology itself:
      4 compress    -- pull every centre toward the centroid, so EVERY circle overlaps; the repair
                       shrinks them and the re-anneal re-expands into a different contact graph.
      5 mirror      -- mirror the circles inside a random sub-rectangle about that rectangle's own
                       axis: a large rearrangement whose local density is unchanged.
      6 cluster     -- re-randomise every centre inside a random disk (a LOCAL re-seed; distinct from
                       kind 0, which scatters the k smallest circles across the whole square).
    Selection is UNIFORM-RANDOM over the seven, not a cyclic rotation -- measured, see solve()."""
    n = len(xy)
    cx = xy.copy()
    if kind == 0:                                        # relocate the smallest circles
        k = max(1, n // 10)
        idx = np.argsort(r)[:k]
        cx[idx] = rng.uniform(LO + 0.01, HI - 0.01, size=(k, 2))
    elif kind == 1:                                      # global jitter
        cx = cx + rng.normal(0.0, 0.02, size=cx.shape)
    elif kind == 2:                                      # jitter scaled per circle (small ones move more)
        sc = 0.04 * (1.0 - r / (r.max() + 1e-12)) + 0.004
        cx = cx + rng.normal(0.0, 1.0, size=cx.shape) * sc[:, None]
    elif kind == 3:                                      # swap one small circle into a random spot + jitter
        idx = int(np.argmin(r))
        cx[idx] = rng.uniform(LO + 0.01, HI - 0.01, size=2)
        cx = cx + rng.normal(0.0, 0.006, size=cx.shape)
    elif kind == 4:                                      # compress toward the centroid
        c = cx.mean(0)
        cx = c + (cx - c) * (1.0 - rng.uniform(0.01, 0.06))
    elif kind == 5:                                      # mirror a sub-rectangle / D4 shell move
        if rng.uniform() < SHELL_P:
            return np.clip(_d4_shell(cx, rng), LO + 1e-4, HI - 1e-4)
        lo = rng.uniform(LO, HI - 0.25, size=2)
        hi = np.minimum(lo + rng.uniform(0.25, 0.6, size=2), HI)
        m = ((cx[:, 0] >= lo[0]) & (cx[:, 0] <= hi[0])
             & (cx[:, 1] >= lo[1]) & (cx[:, 1] <= hi[1]))
        if m.sum() < 2:
            cx = cx + rng.normal(0.0, 0.01, size=cx.shape)
        else:
            ax = int(rng.randint(2))
            cx[m, ax] = lo[ax] + hi[ax] - cx[m, ax]
    elif kind == 7:                                      # symmetry projection (iter 7)
        cx = _symmetrize(cx, rng)
    elif kind == 8:                                      # census RECOMBINATION slot (iter 7/8)
        # Both recombination moves share ONE slot rather than occupying a kind each: a probe measured
        # that giving crossover its own kind (10 kinds instead of 9) HURTS, because it dilutes the
        # draw rate of the transfer arm that produces the jackpots -- see solve() for the numbers.
        if rng.uniform() < SHELLX_P:                 # iter 12: D4-composed Chebyshev-seam crossover
            cx = _shell_crossover(cx, r, rng)
        else:
            cx = _crossover(cx, r, rng) if rng.uniform() < CROSS_P else _transfer(cx, r, rng)
    elif kind == 9:                                      # D4 shell move as its own kind (arm C)
        cx = _d4_shell(cx, rng)
    else:                                                # re-seed a random disk
        c = cx[int(rng.randint(n))]
        R = rng.uniform(0.12, 0.30)
        m = ((cx - c) ** 2).sum(1) <= R * R
        if m.sum() < 2:
            cx = cx + rng.normal(0.0, 0.01, size=cx.shape)
        else:
            k = int(m.sum())
            th = rng.uniform(0.0, 2.0 * np.pi, k)
            rad = R * np.sqrt(rng.uniform(0.0, 1.0, k))
            cx[m] = c + np.stack([rad * np.cos(th), rad * np.sin(th)], 1)
    return np.clip(cx, LO + 1e-4, HI - 1e-4)


N_KICKS = 9                  # 0-6 iter-4, 7 symmetry, 8 census recombination (transfer / crossover)
CROSS_P = 0.0                # P(the recombination slot splices instead of transferring wholesale)
KICK_W = None                # per-kind draw weights; None = uniform (the iter-4 measured default)


def _pick_kick(rng):
    """Draw a kick kind -- uniformly unless KICK_W re-weights the draw.

    Uniform-random selection is a measured result (iter 4: it beat both a cyclic rotation and a UCB1
    bandit), but "uniform" was never the claim -- "not adaptive, not cyclic" was.  A FIXED tilt toward
    an arm whose dominance has been measured twice is a different object from a bandit that re-estimates
    it online from a heavy-tailed reward."""
    if KICK_W is None:
        return int(rng.randint(N_KICKS))
    w = np.asarray(KICK_W[:N_KICKS], dtype=float)
    return int(np.searchsorted(np.cumsum(w / w.sum()), rng.uniform()))
BARREN_K = 15                # consecutive kicks with no new global best before the regime switches
DRIFT_EPS = 1.5e-3           # relative depth the walker may drift below its own value while plateaued


# ---------------------------------------------------------------- per-n driver (RESUMABLE, iter 6)
def _digits_gain(rec, s_in, s_out):
    """Digits (the score's own currency) gained by moving sum_r from s_in to s_out for this n.

    digits = -log10(relgap), so a gain is log10(gap_in / gap_out).  This -- not "did the incumbent
    improve" -- is the reward the allocator maximises: iter 4 lost a whole iteration to a bandit whose
    reward was improvement FREQUENCY, and the lesson recorded there is that a heavy-tailed payoff must
    be scored by MAGNITUDE."""
    if rec is None or rec <= 0.0:                       # no record for this n: raw relative gain
        return max(0.0, (s_out - s_in) / max(abs(s_out), 1e-12)) * 1e3
    fl = 1e-7 * rec                                     # the 7-digit cap: no credit past it
    gi = max(rec - s_in, fl)
    go = max(rec - s_out, fl)
    return max(0.0, float(np.log10(gi / go)))


class _NState(object):
    """Everything a per-n basin-hopping search needs to be SUSPENDED and RESUMED.

    Iter 1-5 ran each n exactly once, for a fixed slice, in a single `_solve_one` call, so the
    schedule had to be decided before a single measurement existed.  Splitting the state out lets the
    allocator interleave visits and re-decide after every one."""
    __slots__ = ("n", "iu0", "ju0", "best", "bx", "br", "ws", "gx", "gr", "gs",
                 "barren", "eps", "started", "spent", "rate", "gain")

    def __init__(self, n):
        self.n = int(n)
        self.iu0, self.ju0 = np.triu_indices(self.n, 1)
        self.best = [-np.inf, None]                      # [sum_r, (n,3) pack] -- metered bests only
        self.started = False
        self.spent = 0.0                                 # CPU-s this call has given to this n
        self.gain = 0.0                                  # digits gained this call
        self.rate = np.inf                               # digits/CPU-s on the LAST visit (inf = unvisited)
        self.bx = self.br = self.gx = self.gr = None
        self.ws = self.gs = -np.inf
        self.barren, self.eps = 0, 0.0


def _visit(st, evaluate, meter, rng, tstop):
    """Run n's search until `tstop`, starting it on the first visit and resuming it after that."""
    n = st.n
    t0 = time.process_time()
    if t0 >= tstop or meter.left() <= 0:
        return

    def offer(xy, r):
        """Route a candidate through the metered oracle; only a metered pack can ever be written."""
        s = float(r.sum())
        if s <= st.best[0] or meter.left() <= 0:
            return
        pack = np.concatenate([xy, r[:, None]], axis=1)
        feas, sr = evaluate(n, pack)
        if feas and float(sr) > st.best[0]:
            st.best[0] = float(sr)
            st.best[1] = pack

    if not st.started:
        st.started = True
        warm = _read_pack(n)
        if warm is not None:
            # A committed pack is ALREADY at this SLP's fixed point (measured: re-annealing
            # csqv27/55/99 from delta=0.05 costs 0.3/0.6/1.7 CPU-s and gains exactly 0.000 digits), so
            # a warm start gets only a short small-delta re-polish and spends the slice on hopping.
            xy, r = warm[:, :2].copy(), _repair(warm[:, :2].copy(), warm[:, 2].copy())
            d_open = 0.004
        else:
            xy = _cold_start(n, rng)
            r = _repair(xy, np.full(n, 1e-6))
            d_open = 0.05
        offer(xy, r)
        xy, r, s0 = _anneal(xy, r, st.iu0, st.ju0, d_open, tstop)
        offer(xy, r)
        st.bx, st.br, st.ws = xy, r, s0
        st.gx, st.gr, st.gs = xy, r, s0

    s_in = st.gs
    # UNIFORM-RANDOM kick selection (iter 4) with REGIME-SWITCHED acceptance (iter 5): the walker
    # (bx, br) is separate from the global best (gx, gr, gs) -- the only thing ever offered -- so no
    # acceptance rule can lose score.  Greedy while kicks keep setting new global bests; after
    # BARREN_K barren kicks the walker may drift down by DRIFT_EPS relative; after BARREN_K more it
    # snaps back to the global best and greedy resumes.  All of this now survives suspension.
    while time.process_time() < tstop and meter.left() > 0:
        cx = _kick(st.bx, st.br, rng, _pick_kick(rng))
        cr = _repair(cx, st.br.copy())
        cx, cr, s = _anneal(cx, cr, st.iu0, st.ju0, 0.03, tstop)
        if s > st.gs:
            offer(cx, cr)
            st.gx, st.gr, st.gs = cx, cr, s
            st.barren, st.eps = 0, 0.0                   # productive again -> back to greedy
        else:
            st.barren += 1
        if s > st.ws * (1.0 - st.eps):
            st.bx, st.br, st.ws = cx, cr, s
        if st.barren >= BARREN_K:
            if st.eps == 0.0:
                st.eps = DRIFT_EPS                       # plateau detected -> allow downhill drift
            else:
                st.bx, st.br, st.ws = st.gx, st.gr, st.gs
                st.eps = 0.0
            st.barren = 0
    dt = max(time.process_time() - t0, 1e-9)
    st.spent += dt
    g = _digits_gain(_RECS.get(n), s_in, st.gs) if np.isfinite(s_in) else 0.0
    st.gain += g
    st.rate = g / dt                                     # productivity of the LAST visit only


_RECS = {}
ROUNDS = 1.8                 # slice length used by the (rejected) auction policy -- see the probe below


def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    t_entry = time.process_time()
    budget = max(2.0, CPU_TOTAL_S - t_entry)             # survives repeated calls in ONE process
    deadline = t_entry + budget
    if not _HAVE_SCIPY:
        return _fallback(evaluate, meter, rng, targets, deadline)

    ts = sorted(int(t) for t in targets)
    # SKIP n already at the digit CAP (iter 5).  The score is the mean of clamp(-log10(relgap), 0, 7),
    # so an n whose committed pack is already inside 1e-7 relative of the record cannot gain anything
    # from its slice.  Reading bench/records.json is explicitly sanctioned; if it is unreadable, or if
    # every target is capped, fall back to attempting them all.
    _RECS.clear()
    _RECS.update(_records())
    live = [n for n in ts if not _at_cap(n)]
    if live:
        ts = live

    # EQUAL time per live n -- and iter 6 TESTED the obvious upgrade and it LOST, which is why the
    # flat rule survives with a measurement behind it instead of a prior.
    #
    # The upgrade tried: give every n one base slice, then auction every later slice to whichever n
    # paid the most DIGITS PER CPU-SECOND on its last visit (magnitude reward, per the iter-4 lesson),
    # ties to the least-served.  Head-to-head, 40 CPU-s / seed 1 / ten n (45..99), same solver
    # internals, only the policy differing (`artifacts/probe_iter6_alloc.py`):
    #     flat      +1.1715 digits total        adaptive  +0.8654 digits total
    # The reason is visible in the per-n split: returns are CONCAVE in the slice, not convex.  The
    # auction gave n=67 6.7s (+0.173) and n=91 6.6s (+0.158) where flat's 4.0s already bought
    # +0.118 / +0.203 -- so the extra seconds bought almost nothing, while the n it starved to 2.2s
    # (55, 61, 83, 89) lost the +0.150 / +0.005 / +0.099 / +0.094 that flat collected from them.
    # A zero-gain visit is NOT evidence of a low expected rate: the payoff is heavy-tailed in the kick
    # stream (iter 4), so one sample of "gained nothing" is mostly noise, and acting on it starves an
    # n that was about to pay.  Concave returns + a noisy rate estimate = spread, do not concentrate.
    #
    # The per-n state is still RESUMABLE (_NState / _visit), which the flat policy does not need but
    # every future schedule does -- suspending and resuming a walker is now free.
    states = [_NState(n) for n in ts]
    acc = t_entry
    for st in states:
        acc += budget / len(states)
        if meter.left() <= 0 or time.process_time() >= deadline:
            break
        _visit(st, evaluate, meter, rng, min(acc, deadline))


def _fallback(evaluate, meter, rng, targets, deadline):
    """scipy-free safety net: random lattice starts scored with the guaranteed-radius rule."""
    for n in sorted(int(t) for t in targets):
        if meter.left() <= 0 or time.process_time() >= deadline:
            break
        for _ in range(20):
            xy = _cold_start(n, rng)
            r = _repair(xy, np.full(n, 1e-6))
            if meter.left() <= 0:
                break
            evaluate(n, np.concatenate([xy, r[:, None]], axis=1))


# ---------------------------------------------------------------- self-test
def _self_test():
    class _M:
        def __init__(self, b):
            self.budget, self.used = int(b), 0

        def tick(self, k=1):
            g = max(0, min(int(k), self.budget - self.used))
            self.used += g
            return g

        def left(self):
            return max(0, self.budget - self.used)

    def make(meter):
        seen = {}

        def ev(n, packing):
            a = np.asarray(packing, dtype=float)
            single = a.ndim == 2
            if single:
                a = a[None]
            assert a.shape[1] == n and a.shape[2] == 3
            B = a.shape[0]
            grant = meter.tick(B)
            feas = np.zeros(B, bool)
            sr = np.full(B, -np.inf)
            for i in range(grant):
                x, y, r = a[i, :, 0], a[i, :, 1], a[i, :, 2]
                D = _dmat(a[i, :, :2])
                ok = (r.min() > 0
                      and (np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min()
                           >= -1e-9)
                      and (D - (r[:, None] + r[None, :])).min() >= -1e-9)
                feas[i] = ok
                sr[i] = r.sum()
                if ok and sr[i] > seen.get(n, (-np.inf,))[0]:
                    seen[n] = (float(sr[i]), a[i].copy())
            return (bool(feas[0]), float(sr[0])) if single else (feas, sr)
        return ev, seen

    global CPU_TOTAL_S
    CPU_TOTAL_S = 12.0                                   # keep the self-test quick; solve() reads it live
    rng = np.random.RandomState(0)
    # 1) two calls in ONE process: the second must still return a feasible packing (the CPU-deadline
    #    trap the mission calls out -- an absolute module-constant deadline would make call 2 a no-op).
    for call in (1, 2):
        m = _M(400)
        ev, seen = make(m)
        solve(ev, m, rng, [27])
        assert 27 in seen, "call %d produced no feasible packing" % call
        assert seen[27][0] > 2.0, "call %d sum_r too low: %r" % (call, seen[27][0])
        print("  call %d: n=27 sum_r=%.9f (evals used %d)" % (call, seen[27][0], m.used))
    # 2) a target with NO committed pack must cold-start rather than raise.
    m = _M(300)
    ev, seen = make(m)
    solve(ev, m, rng, [8])
    assert 8 in seen and seen[8][0] > 0, "cold start for an uncommitted n failed"
    print("  cold start n=8 sum_r=%.9f" % seen[8][0])
    # 3) len(targets) == 1 and an exhausted meter must both be no-crash.
    m = _M(0)
    ev, seen = make(m)
    solve(ev, m, rng, [31])
    print("  exhausted-meter call ok (no packs: %s)" % (not seen))
    # 4) the cap filter: a committed pack at the record is skipped, an uncapped one is kept, and an
    #    all-capped target list falls back to attempting them anyway (never an empty schedule).
    assert _at_cap(27) and not _at_cap(91) and not _at_cap(8)
    print("  cap filter ok (27 capped, 91/8 live)")
    # 5) repair really yields strictly feasible geometry.
    xy = np.random.RandomState(1).uniform(LO + 0.05, HI - 0.05, size=(20, 2))
    r = _repair(xy, np.full(20, 0.4))
    D = _dmat(xy)
    assert r.min() > 0
    assert (D - (r[:, None] + r[None, :])).min() >= -1e-9
    assert (_walls(xy) - r).min() >= -1e-9
    print("  repair feasibility ok")
    # 6) the D4 shell move (iter 11) is an ISOMETRY OF THE SQUARE on the circles it moves: every moved
    #    centre must keep all four of its wall distances and all of its distances to the other moved
    #    centres, and nothing must leave the square.  That invariant IS the move's whole justification,
    #    so it is asserted rather than assumed.
    rs = np.random.RandomState(7)
    moved_any = False
    for _ in range(200):
        xy = rs.uniform(LO + 0.02, HI - 0.02, size=(40, 2))
        out = _d4_shell(xy, rs)
        m = (np.abs(out - xy) > 1e-12).any(1)
        if not m.any() or m.all():
            continue
        moved_any = True
        assert out.min() >= LO - 1e-12 and out.max() <= HI + 1e-12
        a = np.sort(_dmat(xy[m])[np.triu_indices(int(m.sum()), 1)])
        b = np.sort(_dmat(out[m])[np.triu_indices(int(m.sum()), 1)])
        assert np.allclose(a, b, atol=1e-12)
        assert np.allclose(np.sort(_walls(xy[m])), np.sort(_walls(out[m])), atol=1e-12)
        assert np.allclose(out[~m], xy[~m])
    assert moved_any
    print("  D4 shell move is a square isometry on its subset ok")
    print("SELF-TEST: PASS")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
