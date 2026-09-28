"""Circle-packing solver: maximize the sum of radii of n circles in the unit square.

The program is

    max  sum_i r_i
    s.t. ||c_i - c_j|| >= r_i + r_j            (i<j)
         r_i <= 0.5 - |x_i| , r_i <= 0.5 - |y_i|
         r_i > 0

Iterations 1-2 drove it with SLSQP. Iteration 3 replaces that with **sequential linear
programming (SLP)**, because the one nonlinearity here has a property worth exploiting:
`||c_i - c_j||` is CONVEX, so for any unit vector u,

    ||c_i - c_j||  >=  u . (c_i - c_j)                        (convexity / Cauchy-Schwarz)

Linearising the pair constraint about the current contact direction therefore gives an
**inner** approximation: any point satisfying `u . (c_i - c_j) >= r_i + r_j` satisfies the TRUE
constraint too. The wall constraints are already linear. So the whole LP is a restriction of the
true feasible set, the current point is feasible for its own LP, and hence

    every LP optimum is exactly feasible for the original problem, and never worse than where it
    started -- monotone improvement, with no repair step and no infeasible excursion.

Add a trust region `|c - c0| <= T`, `r <= r0 + T` (as plain variable bounds) and this converges to a
KKT point. Two consequences, both measured:

* **Speed.** HiGHS solves the sparse LP (6 nonzeros per pair row, 2 per wall row) far faster than
  SLSQP factors its dense Jacobian: a full convergence at n=99 costs ~0.25 CPU-s against ~2.8 s for
  one SLSQP pass. The trust region also bounds which pairs can matter -- a pair with slack >= 4T
  cannot be violated after a step, since centres move at most T and radii grow at most T -- so the
  active set is DERIVED from T rather than guessed as "k nearest", which is what made the old k<=6
  active set unsafe.
* **Basin search becomes the affordable thing.** The budget buys tens of perturb-and-reoptimize
  cycles per n instead of one polish, and the local optimum -- not the polish -- is what the
  remaining gap was made of.

Iteration 4 removes the T *schedule*, because the trust region has one more property worth reading.
The LP's feasible set is **NESTED in T**: the linearised pair and wall rows are fixed at the current
point, and only the box `|c - c0| <= T`, `r <= r0 + T` moves, so

    T' < T  =>  feasible(T') subset of feasible(T)  =>  value(T') <= value(T).

So if the LP at radius T cannot improve the current point, **no smaller T can either** -- the classic
"shrink the region and retry" reflex is provably guaranteed to buy zero here. Measured
(`artifacts/prof2_iter4.py`, n=61 from its committed pack): 29 of 31 LP solves were that shrink
cascade, walking T from 4.7e-2 down to 1.9e-13 for no gain. The rule is now simply: a non-improving
LP **ends** the convergence. Expanding T instead was tried and is worse (`proto_tr2_iter4.py`: the
enlarged active set costs more than the extra reach buys), so T is one value with no schedule -- the
shrink factor, the T floor and the step tolerance are all deleted. Cost of a convergence fell ~4x
(n=99: 19 basin cycles per 1.5 CPU-s against 10; n=27: 101 against 39).

Scheduling (kept from iteration 2, which fixed a starved tail): breadth first -- every n gets its
starts converged against an ABSOLUTE checkpoint -- then depth, allocated by measured digit yield per
CPU-second. Iteration 2's notes flagged that ranking starts by their raw pre-optimization sum_r made
a bad committed pack an ATTRACTOR (n=65/73/81/91 never moved). SLP is cheap enough that the ranking
can simply be deleted: breadth converges EVERY start and keeps the best, which is the same one rule
for every n, measured at the only moment that means anything.

Budget: process CPU is the binding meter, not the 500k evaluations. The deadline is computed at
ENTRY to solve() -- never as a module constant -- so a second call in the same process gets a full
budget, and the per-n split is recomputed from what is actually left, so it works for len(targets)==1.
"""
import json
import itertools
import math
import os
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

LO, HI = -0.5, 0.5
CPU_BUDGET = 100.0      # seconds of PROCESS CPU this call may use (driver backstop is 120 s)
PACKS = "bench/packs/csqv%d.pck"
RECORDS = "bench/records.json"
BREADTH_FRAC = 0.5      # share of the call spent making sure EVERY n gets its starts converged
COST_EXP = 1.5          # pass cost ~ C * n**COST_EXP; C is FITTED online, never hard-coded
SLP_ITERS = 400         # hard cap; the nesting termination normally ends it in <10
_PASS_COST = {}         # n -> measured seconds of ONE full convergence (EMA); survives across calls
_LP_CALLS = [0]         # LP solves since the last _lp_reset(); diagnostics + the self-test's guard


def _lp_reset():
    _LP_CALLS[0] = 0


def _pass_cost(n):
    """Predicted CPU seconds of ONE SLP convergence at this n: the measurement if we have it, else
    the single power law fitted through every measurement we do have (geometric mean of C).

    Measured directly, not as (guessed iteration count) x (per-iteration cost) -- the scheduler wants
    the cost of a convergence, and after the nesting termination the iteration count per convergence
    is no longer a constant worth pretending to know."""
    if n in _PASS_COST:
        return _PASS_COST[n]
    if not _PASS_COST:
        return 6e-5 * float(n) ** COST_EXP     # only used before the first pass ever calibrates
    logs = [math.log(v) - COST_EXP * math.log(m) for m, v in _PASS_COST.items()]
    return math.exp(sum(logs) / len(logs)) * float(n) ** COST_EXP


def _note_cost(n, secs):
    if secs > 0:
        _PASS_COST[n] = secs if n not in _PASS_COST else 0.5 * _PASS_COST[n] + 0.5 * secs


# ----------------------------------------------------------------------------- feasibility helpers
def _pair_dists(xy):
    d = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return dist


def _walls(xy):
    return np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0], xy[:, 1] - LO, HI - xy[:, 1]])


def _repair(xy, r):
    """Largest global shrink s*r that is strictly feasible for these centres (costs ~1e-12 of sum_r).
    SLP output is already feasible in exact arithmetic; this only absorbs LP round-off."""
    xy = np.clip(xy, LO, HI)
    r = np.maximum(r, 1e-12)
    w = np.maximum(_walls(xy), 0.0)
    dist = _pair_dists(xy)
    rr = r[:, None] + r[None, :]
    s = min(float(np.min(w / r)), float(np.min(dist / np.maximum(rr, 1e-300))))
    s = min(s, 1.0) * (1.0 - 1e-12)
    return xy, np.maximum(r * s, 1e-12)


def _grow(xy, r, rounds=3):
    """Coordinate-wise maximal re-expansion: r_i <- min(wall_i, min_j d_ij - r_j). Never infeasible."""
    w = np.maximum(_walls(xy), 0.0)
    dist = _pair_dists(xy)
    n = len(r)
    for _ in range(rounds):
        for i in range(n):
            cap = min(w[i], float(np.min(dist[i] - r)))
            if cap > r[i]:
                r[i] = cap
    return np.maximum(r * (1.0 - 1e-13), 1e-12)


def _finish(xy, r):
    xy, r = _repair(xy, r)
    r = _grow(xy, r)
    return np.concatenate([xy, r[:, None]], axis=1)


# ------------------------------------------------------------------------------- the SLP itself
def _wall_block(n):
    """The 4n wall rows -- constant in (row, col, val) form, so they are built once per n.
        -x_i + r_i <= 0.5 ;  x_i + r_i <= 0.5 ;  and the same two in y."""
    idx = np.arange(n)
    row = np.repeat(np.arange(4 * n), 2)
    col = np.empty(8 * n, dtype=np.int64)
    val = np.empty(8 * n)
    for b, (base, sgn) in enumerate([(0, -1.0), (0, 1.0), (n, -1.0), (n, 1.0)]):
        sl = slice(2 * b * n, 2 * (b + 1) * n)
        col[sl][0::2] = base + idx
        col[sl][1::2] = 2 * n + idx
        val[sl][0::2] = sgn
        val[sl][1::2] = 1.0
    return row, col, val


def _optimize(n, xy, deadline, r0=None, T0=None, iters=SLP_ITERS):
    """Trust-region SLP from these centres; returns the best strictly-feasible packing found.

    `r0` (the radii a warm start came with) is kept when given -- the unequal radii ARE the
    structure worth warm-starting from. Every iterate is feasible, so the call can be cut off at
    `deadline` at any point and still return a usable packing.

    Termination: the LP's feasible set is nested in T, so an LP that cannot improve the current
    point proves that no smaller T could either -- the convergence simply ENDS there (see the module
    docstring). T is shrunk only when the LP itself fails numerically, which is a different event.
    `gtol` is the floor below which a reported gain is HiGHS's own tolerance rather than progress; it
    is set 3 orders below the 1e-7 relative gap that earns full credit, so it can never gate a digit
    that counts."""
    xy = np.clip(np.asarray(xy, dtype=float).copy(), LO, HI)
    if r0 is None:
        r = np.maximum(np.minimum(_walls(xy), _pair_dists(xy).min(axis=1) / 2.0), 1e-9)
    else:
        r = np.maximum(np.asarray(r0, dtype=float).copy(), 1e-12)
    xy, r = _repair(xy, r)
    r = _grow(xy, r)
    best_xy, best_r, bs = xy.copy(), r.copy(), float(r.sum())
    T = float(T0) if T0 else max(float(r.mean()) * 0.7, 1e-9)
    gtol = 1e-10 * max(bs, 1.0)

    wrow, wcol, wval = _wall_block(n)
    wb = np.full(4 * n, 0.5)
    c = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    iu = np.arange(n)
    nit = 0
    t0 = time.process_time()
    while nit < iters and time.process_time() < deadline:
        nit += 1
        dist = _pair_dists(xy)
        slack = dist - (r[:, None] + r[None, :])
        # A pair with slack >= 4T cannot be violated by a step of this trust region (each centre
        # moves <= T, each radius grows <= T), so dropping it from the LP is exact, not a heuristic.
        ii, jj = np.where((slack < 4.0 * T + 1e-12) & (iu[:, None] < iu[None, :]))
        m = len(ii)
        dx = xy[ii, 0] - xy[jj, 0]
        dy = xy[ii, 1] - xy[jj, 1]
        dd = np.sqrt(dx * dx + dy * dy)
        dd = np.where(dd < 1e-14, 1e-14, dd)
        ux, uy = dx / dd, dy / dd
        prow = np.repeat(np.arange(m), 6)
        pcol = np.empty(6 * m, dtype=np.int64)
        pval = np.empty(6 * m)
        pcol[0::6] = ii
        pval[0::6] = -ux
        pcol[1::6] = jj
        pval[1::6] = ux
        pcol[2::6] = n + ii
        pval[2::6] = -uy
        pcol[3::6] = n + jj
        pval[3::6] = uy
        pcol[4::6] = 2 * n + ii
        pval[4::6] = 1.0
        pcol[5::6] = 2 * n + jj
        pval[5::6] = 1.0
        A = csr_matrix((np.concatenate([pval, wval]),
                        (np.concatenate([prow, wrow + m]), np.concatenate([pcol, wcol]))),
                       shape=(m + 4 * n, 3 * n))
        b = np.concatenate([np.zeros(m), wb])
        lb = np.concatenate([np.maximum(LO, xy[:, 0] - T), np.maximum(LO, xy[:, 1] - T), np.zeros(n)])
        ub = np.concatenate([np.minimum(HI, xy[:, 0] + T), np.minimum(HI, xy[:, 1] + T), r + T])
        try:
            _LP_CALLS[0] += 1
            res = linprog(c, A_ub=A, b_ub=b, bounds=np.stack([lb, ub], axis=1), method="highs")
        except (ValueError, TypeError):
            break
        if not res.success or res.x is None or not np.all(np.isfinite(res.x)):
            T *= 0.35                              # numerical failure, not stagnation: retry smaller
            if T < 1e-13:
                break
            continue
        z = np.asarray(res.x, dtype=float)
        nxy = np.stack([z[:n], z[n:2 * n]], axis=1)
        nr = np.maximum(z[2 * n:], 1e-12)
        s = float(nr.sum())
        if s <= bs + gtol:
            break                                  # nesting: no smaller T can do better either
        xy, r = nxy, nr
        bs, best_xy, best_r = s, nxy.copy(), nr.copy()
    _note_cost(n, time.process_time() - t0)
    return _finish(best_xy, best_r)


# ------------------------------------------------------------------------------ initial guesses
def _warm_pack(n):
    """(centres, radii) of the committed census pack for n, or None -- the radii matter as much as
    the centres, since the unequal radii ARE what a good csqv packing looks like.
    GUARDED: any n may be asked for, including one with no committed pack."""
    p = PACKS % n
    if not os.path.exists(p):
        return None
    try:
        raw = [ln for ln in open(p).read().splitlines() if ln.strip()]
        rows = [[float(v) for v in ln.split()] for ln in raw[2:]]
    except (OSError, ValueError):
        return None
    if len(rows) != n or any(len(row) != 3 for row in rows):
        return None
    a = np.array(rows, dtype=float)
    return a[:, :2], a[:, 2]


def _row_ys(n, rows, anchor):
    """The `rows` row CENTRES in y -- the family's third axis, and the only one that touches the
    top and bottom walls.

    `anchor=False` is the pinned rule this family has always used: centres at `(i+0.5)/rows`, i.e.
    pitch `p = 1/rows` and a wall margin of exactly `p/2`. That margin is a *constant fraction of
    the pitch*, and nothing in the geometry says it should be. A staggered row layout with in-row
    spacing `a` and pitch `p` puts the nearest circle of the next row at `sqrt(a^2/4 + p^2)`, so the
    radius the layout can support is
        `r(a, p) = min(a/2, sqrt(a^2/4 + p^2)/2)`,
    and the top and bottom rows can only use it if the wall margin `m` is at least `r`. For the
    equal-circle hexagonal lattice `p = sqrt(3) a / 2` and then `m = r = a/2 = p/sqrt(3)`, i.e.
    `m/p = 0.577`, NOT 0.5 -- and the row family's winning counts are far from square (n=91 wants 18
    rows of 5, cell aspect 3.6:1), where `p << a`, `r ~ p/2 ... p` and the pinned `m = p/2` jams the
    boundary rows against the wall at a fraction of the radius they could carry. Σr is a boundary
    quantity (the bulk is Cauchy-Schwarz-bounded by sqrt(n) with equality iff the density is uniform,
    and the records sit 2-4% under that with the shortfall shrinking as n grows), so this is exactly
    the kind of decision the remaining deficit is made of.

    `anchor=True` therefore does not pick a better fraction -- it drops the fraction and solves for
    the layout that MAXIMISES the radius the rows can support:
        maximise  min(a/2, sqrt(a^2/4 + p^2)/2, m)   s.t.  2m + (rows-1) p = 1,
    with `a = 1/(n/rows + 0.5)` the family's own in-row spacing. The first two terms fall as `m`
    grows (`p` shrinks) and the third rises, so the maximum is the single crossing point and one
    bisection on `f(m) = m - min(a/2, sqrt(a^2/4+p^2)/2)` finds it. No new constant: `anchor` is a
    bit and every number here is `a`, `rows` or `sqrt(3)`.

    This REPLACES the pinned fraction rather than being swept beside it, and that is the measured
    shape, not the assumed one (`artifacts/proto_iter15b.py`, the row family alone, 10 parked n x 2
    seeds x 1 CPU-s per cell-arm, four arms paired cell-for-cell):
      * solved margin ALWAYS (shipped): 2.4143, **6 W / 2 L / 12 T** vs the pinned rule;
      * solved margin DRAWN per start (period 2): 2.3985, 10 W / 8 L -- a wash;
      * solved margin as a THIRD CURSOR AXIS (period 4): 2.3488, 7 W / 8 L / 5 T -- a LOSS, because
        it halves how many row COUNTS the sweep reaches, which is worth more than the coverage.
    Iteration 14's swap of one *long-row assignment* for another was a wash (10 W / 11 L) and had to
    be swept; this swap is not, and the reason is that `m = p/2` is not a rival design but a
    DOMINATED one -- when the walls do not bind, the solve returns `m/p = 0.506` at n=45/7 rows and
    0.511 at n=75/9, i.e. the old rule back to three digits, and it only departs from it where the
    old rule was leaving radius on the table (n=91 at its winning 18 rows: `m` 0.0278 -> 0.0521,
    `m/p` 0.50 -> 0.99). A swap whose two ends agree wherever the pinned value was right does not
    need a sweep to protect it."""
    rows = int(rows)
    if rows <= 1:
        return np.zeros(1)
    if not anchor:
        return (np.arange(rows) + 0.5) / rows - 0.5
    a = 1.0 / (float(n) / rows + 0.5)
    R = float(rows - 1)

    def f(m):
        p = (1.0 - 2.0 * m) / R
        return m - min(0.5 * a, 0.5 * math.sqrt(0.25 * a * a + p * p))

    lo, hi = 0.0, 0.5
    if f(hi) <= 0.0:                        # no crossing: the walls are not what binds
        return (np.arange(rows) + 0.5) / rows - 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    m = 0.5 * (lo + hi)
    p = (1.0 - 2.0 * m) / R
    if not (p > 0.0 and m > 0.0):
        return (np.arange(rows) + 0.5) / rows - 0.5
    return m + p * np.arange(rows) - 0.5


def _rows_start(n, rows, rng, spread=False, anchor=True):
    """n circles in `rows` staggered rows -- `rows` is the knob, and WHICH rows are long is DRAWN.

    Varying `rows` walks a whole family of paradigms with one rule -- `rows ~ sqrt(n)` is the square
    grid, fewer rows is wide/columnar, more rows is the tall hexagonal-ish layouts -- so no per-n
    table is needed to decide which paradigm an n wants. Jitter is scaled by the row pitch, so the
    same rule behaves the same at every n.

    The top and bottom rows meet the wall at a margin that is SOLVED, not pinned to half the pitch
    -- see `_row_ys`, which is the boundary half of this family and a swap measured at 6 W / 2 L / 12 T.

    `n` is almost never a multiple of `rows`, so `n % rows` rows hold one circle more than the rest,
    and until now those long rows were always the FIRST ones: one fixed assignment out of
    `C(rows, n % rows)`. That pinned the only part of this family that touches the boundary. Σr is
    a boundary quantity here -- in the bulk a circle of a locally uniform density rho has r ~ 1/sqrt(rho),
    so Σr ~ integral of sqrt(rho), which Cauchy-Schwarz bounds by sqrt(n) with equality IFF rho is
    uniform; the records sit 2-4% under that hexagonal bulk value with the shortfall shrinking as n
    grows, i.e. the whole deficit is how the lattice meets the four walls. Which rows are long is
    exactly that: it decides which rows reach both side walls and how the long/short rows alternate.
    The hexagonal alternation (long, short, long, short) was unreachable at EVERY row count.

    So WHICH rows are long becomes the family's SECOND axis, swept exactly like the first rather
    than fixed: `spread=False` is the old front-loaded block, `spread=True` spreads the `rem` long
    rows as EVENLY as possible over the `rows` rows --
    `long(i) iff ((i+1)*rem) % rows < rem`, the standard even-distribution rule -- instead of being
    packed at the front. At `rem ~ rows/2` that IS the alternation, at `rem = 1` it is a single
    long row and at `rem = rows-1` a single short one, and it degenerates to the old rule exactly
    when `rem` divides `rows`. One rule for every (n, rows), no new constant and nothing drawn: a
    uniform draw over the `C(rows, rem)` assignments was measured first and LOST
    (`artifacts/proto_iter14.py`: 8 wins / 14 losses, 2.487 -> 2.393 mean digits over 11 parked n
    x 2 seeds), because almost every random subset is incoherent -- what the family was missing was
    one specific structured assignment, not variety. Replacing the front-loaded rule by the even one
    was then measured as a wash (10 wins / 11 losses), which is what a zero-sum swap of one fixed
    point for another should look like: neither is right for every (n, rows), so the sweep enumerates
    both and `_optimize` decides. No new constant -- `spread` is a bit, not a tuned number."""
    rows = max(1, min(int(rows), n))
    ys = _row_ys(n, rows, anchor)
    rem = n % rows
    pts = []
    for i, yy in enumerate(ys):
        cnt = n // rows + (1 if (((i + 1) * rem) % rows < rem if spread else i < rem) else 0)
        off = 0.5 * (i % 2)
        for j in range(cnt):
            pts.append(((j + 0.5 + off) / (cnt + 0.5) - 0.5, yy))
    p = np.array(pts, dtype=float)
    # Clipped into the square: with the margin SOLVED rather than pinned at p/2 it can now be as
    # small as the supported radius, so a pitch-scaled jitter can carry a boundary row's centre
    # past the wall. `_repair` clipped it anyway (a centre outside the square just gets zero wall
    # slack), so this changes no packing -- it makes the start honest at the point it is built.
    return np.clip(p + rng.normal(0.0, 0.03 / rows, size=p.shape), LO, HI)


def _row_order(n):
    """Every row count 1..n, SQUAREST FIRST -- ordered by |log(rows^2 / n)|, i.e. by how far the
    cell aspect ratio is from 1. There is no cutoff constant: the sweep simply walks outward from
    the square layout and the CPU deadline decides how far it gets, so the range is set by what the
    budget can afford rather than by a number I picked."""
    return sorted(range(1, n + 1), key=lambda k: (abs(math.log(k * k / float(n))), k))


def _vdc(k):
    """The van der Corput (base 2) point k: 0, 1/2, 1/4, 3/4, 1/8, ... -- a sequence that REFINES
    [0,1) progressively, so a sweep driven by it is well spread after any number of terms and needs
    no count chosen in advance. The CPU deadline decides how fine the sweep gets."""
    f, v = 0.5, 0.0
    while k:
        v += f * (k & 1)
        k >>= 1
        f *= 0.5
    return v


def _lattice_start(n, theta, rng):
    """n circles on a triangular lattice ROTATED by `theta` -- the start family the row sweep cannot
    reach, because `_rows_start` is axis-aligned by construction (theta = 0).

    The records sit at `record/sqrt(n)` = 0.516..0.526 against the hexagonal equal-circle asymptote
    0.5373, and the packs that reached the cap are a near-equal BULK (0.7-0.85 of rmax) plus a
    handful of edge defects -- i.e. a lattice patch cut by the square. For equal circles in a square
    the best known patches are frequently TILTED, and theta is exactly the degree of freedom the row
    family lacks.

    The spacing is not a knob: it is bisected so the patch holds exactly n points, bracketed around
    the density identity `s = sqrt(2 / (sqrt(3) n))` (n points of a triangular lattice per unit
    area). The lattice PHASE is drawn from rng, so the same theta gives a different cut of the
    square each time and the family stays a source of diversity rather than one point per angle.
    Surplus points are dropped nearest-the-wall first, since those are the ones the square truncates.
    Returns None if the square cannot be made to hold n points, so the caller can move on."""
    ca, sa = math.cos(theta), math.sin(theta)
    ox, oy = rng.rand(), rng.rand()

    def pts(s):
        R = int(math.ceil(1.8 / s)) + 2
        i, j = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1))
        i = i.ravel().astype(float)
        j = j.ravel().astype(float)
        X = s * (i + 0.5 * j + ox)
        Y = s * (math.sqrt(3.0) / 2.0) * (j + oy)
        x = ca * X - sa * Y
        y = sa * X + ca * Y
        m = (x >= LO) & (x <= HI) & (y >= LO) & (y <= HI)
        return np.stack([x[m], y[m]], axis=1)

    s0 = math.sqrt(2.0 / (math.sqrt(3.0) * float(n)))    # density identity, not a fitted constant
    lo, hi = 0.5 * s0, 2.5 * s0
    for _ in range(4):                                   # widen only if the identity under-shot
        if len(pts(lo)) >= n:
            break
        hi, lo = lo, 0.5 * lo
    else:
        if len(pts(lo)) < n:
            return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if len(pts(mid)) >= n:
            lo = mid
        else:
            hi = mid
    p = pts(lo)
    if len(p) < n:
        return None
    if len(p) > n:
        w = np.minimum.reduce([p[:, 0] - LO, HI - p[:, 0], p[:, 1] - LO, HI - p[:, 1]])
        p = p[np.argsort(w)[::-1][:n]]
    return p


def _transfer_start(n, xy, r, rng):
    """Re-use a NEIGHBOURING size's packing as this n's start -- the census transferred across n.

    Consecutive csqv optima are near-neighbours in structure: the contact graph of the best n+2
    packing is the best n packing plus two circles, far more often than it is anything a fresh
    lattice produces. So:
      * source LARGER than n -- drop the `m-n` SMALLEST circles. Deleting circles can never break
        feasibility, so the survivors are already an exactly feasible packing of n circles sitting
        in the source's basin, and `_grow` re-expands them into the vacated room.
      * source SMALLER than n -- insert the missing circles one at a time at the point of largest
        clearance (a scan on a grid whose resolution is tied to sqrt(n), so the same rule resolves
        the same fraction of a circle at every size), with the radius that clearance allows.
    Either way the result is feasible before the SLP starts, and it carries the source's UNEQUAL
    radii, which is the part of a good csqv packing a lattice start cannot guess.

    Both directions pay, and this compounds across iterations: every n that reaches its record
    becomes a better start for its neighbours than they can find on their own."""
    m = len(r)
    if m >= n:
        # WHICH circles to drop is drawn, not fixed. Weighted sampling without replacement
        # (Efraimidis-Spirakis: key_i = Exp(1) / w_i, smallest keys drop first) with w_i = 1/AREA,
        # so the circle claiming least room is the likely casualty -- the deterministic "drop the
        # m-n smallest" is this rule's argmax -- but the same source now yields a DIFFERENT start
        # every visit instead of the one fixed point it used to. No constant: the weight is the
        # area the circle occupies, which is what dropping it actually frees.
        key = rng.standard_exponential(m) * np.maximum(r, 1e-300) ** 2
        keep = np.argsort(key)[::-1][:n]
        return xy[keep].copy(), r[keep].copy()
    xy, r = xy.copy(), r.copy()
    g = np.linspace(LO, HI, 4 * int(math.sqrt(n)) + 3)
    gx, gy = np.meshgrid(g, g)
    p = np.stack([gx.ravel(), gy.ravel()], axis=1)
    wall = np.minimum.reduce([p[:, 0] - LO, HI - p[:, 0], p[:, 1] - LO, HI - p[:, 1]])
    for _ in range(n - m):
        d = np.sqrt(((p[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]
        cl = np.minimum(d.min(axis=1), wall)
        # Same story on the insertion side: the site is SAMPLED in proportion to its clearance
        # rather than taken by argmax, so a source smaller than n also yields a fresh start each
        # visit. The weight is the radius the circle would get -- no constant, and the argmax
        # remains the most likely single outcome. Falls back to the argmax when nothing fits.
        w = np.maximum(cl, 0.0)
        tw = float(w.sum())
        if tw > 0.0:
            i = int(np.searchsorted(np.cumsum(w), rng.rand() * tw, side="right"))
            i = min(i, len(w) - 1)
        else:
            i = int(np.argmax(cl))
        xy = np.concatenate([xy, p[i][None]])
        r = np.concatenate([r, [max(float(cl[i]), 1e-9)]])
    return xy, r


def _splice_start(n, axy, ar, bxy, br, rng):
    """RECOMBINE two packings across a straight seam: everything on one side of a random line from
    A, everything on the other side from B, then the count fixed by `_transfer_start`.

    The transfer family already re-uses a neighbour's packing WHOLESALE. What it cannot produce is
    a packing whose structure changes PART WAY ACROSS THE SQUARE -- and that is exactly what
    separates two local optima here: measured this iteration (`artifacts/proto2_iter9.py`), at 5 of
    6 parked n no cold start from any of the three families beats the committed incumbent, and the
    incumbent sits ~1e-3 below the record, i.e. one region of it is laid out wrong rather than all
    of it. A seam splice keeps the region A gets right and swaps in B's answer for the rest, and
    the grain boundary it creates is placed somewhere no single-source start can put one.

    Feasibility is kept by construction, so the SLP starts inside the feasible set: A's side is
    untouched, and a B circle is admitted only if it overlaps neither the wall-clipped A side nor a
    B circle already admitted. Dropping circles can never create an overlap, so whatever survives is
    a valid packing of however many circles; `_transfer_start` then takes it to exactly n.
    The seam (direction and offset) is the only choice, and it is drawn from rng -- no constant."""
    th = rng.uniform(0.0, math.pi)
    u = np.array([math.cos(th), math.sin(th)])
    # Offset drawn among the projections themselves, so the seam always has circles on both sides
    # whatever the two sources look like -- a fixed offset range would cut some pairs into a no-op.
    proj = np.concatenate([axy @ u, bxy @ u])
    t = float(rng.uniform(proj.min(), proj.max()))
    ka = (axy @ u) < t
    kb = (bxy @ u) >= t
    xy = axy[ka]
    r = ar[ka]
    for c, rad in zip(bxy[kb], br[kb]):
        if len(xy):
            if np.min(np.sqrt(((xy - c) ** 2).sum(1)) - r) < rad:
                continue
        xy = np.concatenate([xy, c[None]])
        r = np.concatenate([r, [rad]])
    if len(r) == 0:
        return None
    return _transfer_start(n, xy, r, rng)


def _source_order(n, have):
    """The sizes to transfer INTO n from, NEAREST FIRST. No range constant and no per-n table: the
    list is just every size we hold a packing for, ordered by |src - n| (larger src first on a tie,
    since dropping circles is exactly feasible while inserting them is a guess), and the CPU
    deadline decides how far down it gets -- the same shape as the row sweep."""
    return sorted((m for m in have if m != n), key=lambda m: (abs(m - n), -m))


def _arch_add(arch, pack):
    """Insert a converged optimum into this n's POPULATION of distinct local optima.

    Every iteration so far has thrown this information away: `_optimize` returns a local optimum,
    `submit()` keeps it only if it beats the running best, and the dozens of OTHER optima a slice
    converges are discarded. Iteration 9 measured that the incumbent already beats every cold start
    at 5 of 6 parked n, and that splicing it with a NEIGHBOURING size pays -- so the runners-up at
    n ITSELF, which are optima of the very problem being solved, are the obvious second parent.

    Two optima count as the SAME when their sorted radius vectors agree to `FEAS_TOL`: a kick that
    re-converges to the basin it came from -- which is most of them -- must not occupy a slot, or
    the population is one packing listed eight times. The list is the `ARCH_MAX` best by sum_r; the
    sorted radii are cached with the entry so an insert costs one sort, not `ARCH_MAX` of them."""
    sr = np.sort(pack[:, 2])
    for _, sr2, _p in arch:
        if len(sr2) == len(sr) and np.max(np.abs(sr2 - sr)) <= FEAS_TOL:
            return False
    arch.append((float(sr.sum()), sr, pack))
    arch.sort(key=lambda t: -t[0])
    if len(arch) > ARCH_MAX:
        del arch[_crowded(arch)]
    return True


def _crowded(arch):
    """The index to EVICT from an over-full population: the nearer member of the CLOSEST pair,
    never the best member. Returns an index > 0.

    Truncating by sum_r (`del arch[ARCH_MAX:]`) is the obvious rule and it is the wrong one. What
    iterations 10 and 11 measured is that a population pays through the DIFFERENCE between its
    members -- uniform draws beat elite-anchored ones at both the splice (3.435 vs 3.210) and the
    kick (3.941 vs 3.737). But a depth loop converges mostly into the basins it is already in, so
    dropping the worst member every time fills the list with eight near-copies of the incumbent:
    the population stays full, its diversity collapses, and the uniform draw degenerates back into
    the elitist one the measurements rejected.

    So evict for CROWDING instead: the pair of members whose sorted radius vectors are closest in
    L-inf is the pair carrying the least difference, and the worse of the two is the redundant one.
    That keeps a spread of basins at the same memory bound, and it introduces no threshold -- there
    is no distance I had to pick, only "closest", so the rule reads the same at every n. `arch` is
    sorted best-first, so the later index of the closest pair is the worse member and index 0 is
    never returned."""
    m = len(arch)
    worst, wi = None, m - 1
    for i in range(m):
        for j in range(i + 1, m):
            a, b = arch[i][1], arch[j][1]
            if len(a) != len(b):
                continue
            d = float(np.max(np.abs(a - b)))
            if worst is None or d < worst:
                worst, wi = d, j
    return wi


def _arch_splice(n, arch, rng):
    """Seam-splice two UNIFORMLY drawn distinct members of n's own population, or None if the
    population is not yet two deep.

    Uniform, not elite-anchored: the probe (`artifacts/proto_iter10.py`, 6 parked n x 3 seeds,
    6 CPU-s) measured 3.435 mean digits for a uniform pair against 3.222 for the shipped mix, while
    anchoring parent A to the incumbent and drawing only B scored 3.210 -- i.e. flat. What the
    recombination is buying is the DIFFERENCE between two optima, so forcing one parent to be the
    best one throws away exactly the half of the population that carries it."""
    if len(arch) < 2:
        return None
    i, j = rng.choice(len(arch), size=2, replace=False)
    a, b = arch[i][2], arch[j][2]
    st = _splice_start(n, a[:, :2], a[:, 2], b[:, :2], b[:, 2], rng)
    if st is not None:
        _ARCH_USES[0] += 1
    return st


def _arch_parent(n, arch, fallback, rng):
    """The packing the next KICK perturbs: a UNIFORMLY drawn member of n's population, or
    `fallback` (the incumbent) when the population is not yet two deep.

    Iteration 10 built the population but the depth loop only read it for the seam-splice move; the
    two kicks still perturbed `best[n]`, so three moves in four were a (1+1) hill-climb that could
    only ever leave the incumbent's basin and fall back into it. Drawing the parent uniformly makes
    the kick explore the neighbourhood of the WHOLE population instead. The same evidence backs it
    as backed the uniform splice: anchoring to the best member measured FLAT there (iteration 10,
    3.210 vs 3.222), because what a population is worth is the members that are not the best one.

    Measured (`artifacts/proto_iter11.py`, 6 parked n x 6 seeds x 6 CPU-s, identical rng per cell):
    base 3.737 mean digits, this draw 3.941 -- and it reached the 7-digit cap in 5 of the 36
    cells against 3 for the incumbent-only kick, in both independent seed blocks. Guarded: an empty or single-member population simply returns `fallback`,
    so an n on its first convergence behaves exactly as before.
    """
    if len(arch) < 2:
        return fallback
    _ARCH_KICKS[0] += 1
    return arch[rng.randint(len(arch))][2]


_ARCH_USES = [0]        # archive splices actually handed to the SLP; diagnostics + the self-test
_ARCH_KICKS = [0]       # kicks whose parent came from the population rather than the incumbent


# ------------------------------------------------------------------------------------- the driver
DIGCAP = 7.0            # the scorer caps a single n at 7 digits; nothing is gained past it
ARCH_MAX = 8            # distinct local optima kept per n (a memory bound on the population)
FEAS_TOL = 1e-9         # the feasibility tolerance the scorer uses; also what makes two optima SAME


def _records():
    """The record table, if it is there. GUARDED: solve() may be called for n it does not cover."""
    try:
        with open(RECORDS) as fh:
            d = json.load(fh)
        rec = d.get("records", d)
        return {int(k): float(v) for k, v in rec.items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _dig(n, s, recs):
    """This n's contribution to SCORE: digits of the relative gap to the record it has closed."""
    R = recs.get(n)
    if R is None or R <= 0 or s <= 0:
        return 0.0
    gap = (R - s) / R
    if gap <= 0:
        return DIGCAP
    return max(0.0, min(DIGCAP, -math.log10(gap)))


def _headroom(n, cur, recs):
    """What n can still add to SCORE: the digits between where it is and the scorer's cap.

    Zero once n is capped -- the scorer pays nothing past `DIGCAP`, so a capped n is worth no clock
    in either phase. `DIGCAP` when no record is known (an n outside the census, or an offline re-run
    on fresh sizes), so an unknown n is treated as maximally worth working on rather than skipped.
    One rule for every n; no cap, no list of sizes, and no phase special-cased."""
    return DIGCAP - _dig(n, cur, recs)


def _score_gain(n, old, new, recs):
    """What moving this n from `old` to `new` is worth in SCORE. Where no record is known the same
    quantity is estimated scale-free from the relative improvement, so an n outside the census is
    still comparable to one inside it -- one rule, no special case."""
    if new <= old:
        return 0.0
    if n in recs and recs[n] > 0:
        return max(0.0, _dig(n, new, recs) - _dig(n, old, recs))
    return 0.4342944819 * (new - old) / max(old, 1e-12)


def _kick(n, pack, rng):
    """One perturbation of a converged packing, for the next basin.

    Two moves, chosen by coin flip, both scaled by the packing's own mean radius so the SAME rule
    behaves the same at n=27 and n=99:
      * TELEPORT -- lift the k smallest circles out and drop them at random. A max-sum-of-radii
        optimum is mostly determined by where the SMALL circles sit; jitter alone can never move one
        across the square, and this is what actually escaped the stuck basins in testing.
      * JITTER -- gaussian shake of every centre, for local re-seating of the contact graph.
    """
    xy = pack[:, :2].copy()
    r = pack[:, 2].copy()
    mr = float(r.mean())
    if rng.rand() < 0.5:
        k = 1 + rng.randint(max(1, n // 10))
        small = np.argsort(r)[:k]
        xy[small] = rng.uniform(LO + 1e-3, HI - 1e-3, size=(k, 2))
        r[small] = 1e-9
    else:
        xy = xy + rng.normal(0.0, mr * rng.uniform(0.05, 0.4), size=(n, 2))
    return xy, r


def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    t0 = time.process_time()                         # at ENTRY: a 2nd call gets a full budget
    deadline = t0 + CPU_BUDGET
    order = sorted({int(n) for n in targets})
    recs = _records()
    best = {}                                        # n -> (sum_r as evaluate() scored it, pack)
    arch = {}                                        # n -> population of distinct local optima

    def submit(n, pack):
        """Route a candidate through the metered evaluate() as soon as it beats this call's best,
        so a CPU backstop can only cost the work done since the last improvement.

        Every pack that passes through here is first offered to n's POPULATION -- including the
        ones that do not beat the running best, which is most of them and all of which used to be
        dropped on the floor. `_optimize` output is feasible by construction (the SLP's linearised
        pair rows are an INNER approximation, asserted in the self-test), so archiving costs no
        evaluation budget; only a candidate that could become the committed pack is metered."""
        _arch_add(arch.setdefault(n, []), pack)
        s = float(pack[:, 2].sum())
        if (n in best and s <= best[n][0]) or meter.left() <= 0:
            return 0.0
        feas, sr = evaluate(n, pack)
        sr = float(sr)
        if not feas or not np.isfinite(sr):
            return 0.0
        old = best[n][0] if n in best else 0.0
        if n in best and sr <= old:
            return 0.0
        best[n] = (sr, pack)
        return _score_gain(n, old, sr, recs) if old > 0 else 0.0

    # ---- phase A: BREADTH. Every n gets the committed pack re-converged plus as much of its ROW
    # SWEEP as its ABSOLUTE checkpoint affords. Iteration 4's notes called the parked n
    # basin-limited, and the iteration-5 probe confirmed the mechanism: a warm start is an
    # ATTRACTOR that no kick escapes, while a cold row layout at the right row count lands in a
    # different basin entirely (n=91: warm 2.04 digits, kicks never beat it; rows=18 -> 7.00).
    # So the sweep is the breadth work, and every start is converged and judged by what it is
    # worth AFTER optimization -- one rule for every n, no paradigm chosen per n by hand.
    # Every n's cold starts, as ONE interleaved list: the row sweep (squarest first) and the
    # TRANSFER sweep (nearest neighbouring size first), alternating so neither starves the other.
    # Iteration 5 bought its digits from the row sweep; the iteration-6 probe
    # (`artifacts/proto2_iter6.py`) showed transfers beat this n's own warm incumbent on 6 of 12
    # parked n -- n=85 from n=87 went 3.71 digits to the 7.00 cap -- so both belong in the sweep and
    # which one an n wants is left to the measurement, not to me.
    # Every n's cold starts as ONE round-robin over THREE families -- transfer source (nearest
    # size first), row count (squarest first), lattice ANGLE (van der Corput over [0, 60deg)).
    # One rule, no weights: which family an n wants is settled by which start survives `_optimize`,
    # not by a number I picked. The first two lists are finite and are skipped once spent; the angle
    # sweep never runs out, so the CPU deadline is what ends the sweep at every n.
    warm = {n: _warm_pack(n) for n in order}          # read once: phase A starts AND the headroom
    warm_sum = {n: float(w[1].sum()) for n, w in warm.items() if w is not None}
    have = sorted(set(recs) | set(order))
    srcs = {n: _source_order(n, have) for n in order}
    rowo = {n: _row_order(n) for n in order}
    cursor = {n: 0 for n in order}

    def pack_of(m):
        """(centres, radii) of the best packing we hold for size m -- this call's own improvement
        if it has one, else the committed census pack -- or None. GUARDED: m may have neither."""
        b = best.get(m)
        if b is not None:
            return b[1][:, :2], b[1][:, 2]
        return _warm_pack(m)

    def take(n):
        """The next cold start for n as (centres, radii-or-None), or None if the families all
        refuse. GUARDED: a transfer source with no packing yet, and an angle whose lattice cannot
        hold n points, are simply skipped."""
        skips = 0
        while skips < 64:
            i = cursor[n]
            cursor[n] += 1
            fam, k = i % 4, i // 4
            if fam == 0:
                if not srcs[n]:
                    skips += 1
                    continue
                # The source list WRAPS. It used to be spent after one pass, which was the right
                # rule while `_transfer_start` was deterministic (a second visit to a source
                # rebuilt the identical start). Now that dropping and inserting are both drawn,
                # re-visiting a source is a new start, so this family stops running out -- the
                # same shape the lattice angle sweep already had, with the CPU deadline as the
                # only thing that ends it.
                src = pack_of(srcs[n][k % len(srcs[n])])
                if src is None:
                    skips += 1
                    continue
                return _transfer_start(n, src[0], src[1], rng)
            if fam == 1:
                # The row-count list WRAPS, for the same reason the source list does: now that the
                # long rows are drawn, revisiting a row count is a NEW start rather than a rebuild
                # of the identical one, so the family stops being a finite list of fixed points and
                # the CPU deadline is what ends it.
                # Two axes, interleaved on one cursor: the row count (squarest first, wrapping)
                # and WHICH rows are long. Both assignments get swept at every row count. The
                # WALL MARGIN is not a third axis -- it is SOLVED inside `_rows_start` (see
                # `_row_ys`), which is why the cursor period stays 2: interleaving it as a third
                # axis was measured and it LOSES (`artifacts/proto_iter15b.py`, cursor period 4:
                # 7 W / 8 L / 5 T, 2.3983 -> 2.3488) because halving the row COUNTS the sweep
                # reaches costs more than covering both boundary rules buys.
                return _rows_start(n, rowo[n][(k // 2) % len(rowo[n])], rng, bool(k & 1)), None
            if fam == 2:
                pts = _lattice_start(n, _vdc(k) * math.pi / 3.0, rng)
                if pts is None:
                    skips += 1
                    continue
                return pts, None
            # SPLICE: this n's own incumbent recombined with a neighbouring size's packing. A
            # is the best packing we hold FOR n -- the object the measurements say is already
            # better than any cold start -- so the family edits what is working instead of
            # starting over. Guarded at both ends: with no incumbent, or no source, it is skipped.
            a = pack_of(n)
            if a is None or not srcs[n]:
                skips += 1
                continue
            b = pack_of(srcs[n][k % len(srcs[n])])
            if b is None:
                skips += 1
                continue
            st = _splice_start(n, a[0], a[1], b[0], b[1], rng)
            if st is None:
                skips += 1
                continue
            return st
        return None

    # The clock is allocated by ACHIEVABLE DIGITS, not by size. `_head(n)` is what this n can still
    # add to SCORE -- `DIGCAP - digits it already has` -- so an n already at the 7-digit cap weighs
    # ZERO and an n with no record (outside the census) weighs the full cap. Measured on the
    # iteration-7 census: 13 of 37 n are capped and they were taking **39.2%** of phase A's clock
    # re-deriving packings the scorer cannot pay for, while the lattice family at the 24 parked n is
    # budget-starved (`artifacts/proto_iter7.py`: 5 CPU-s of lattice alone capped n=45 and n=95,
    # which the full run reached only 3.31 and 3.76 on). Same rule both phases, no list of n and no
    # cap special-cased: value the work at n by the score it can still buy.
    def _head(n):
        return _headroom(n, best[n][0] if n in best else warm_sum.get(n, 0.0), recs)

    a_end = t0 + BREADTH_FRAC * (deadline - t0)
    wt = [max(_head(n), 0.0) * float(n) ** 1.3 for n in order]
    tot = sum(wt)
    if tot <= 0.0:                       # every target is already capped -- fall back to by-size,
        wt = [float(n) ** 1.3 for n in order]   # so a fully-capped call still uses its budget
        tot = sum(wt)
    cum = 0.0
    for pos, n in enumerate(order):
        if meter.left() <= 0:
            break
        cum += wt[pos]
        n_end = min(deadline, t0 + (a_end - t0) * cum / tot)
        # A zero-weight (capped) n gets a zero-length slice, so it still re-converges its committed
        # pack once -- breadth keeps covering EVERY target, it just stops sweeping where the sweep
        # cannot pay -- and the "always converge at least one start" rule below ends it there.
        w = warm[n]
        if w is not None:
            submit(n, _optimize(n, w[0], n_end, r0=w[1]))   # centres AND radii are the structure
        while meter.left() > 0:
            if (n in best or cursor[n]) and time.process_time() >= n_end:
                break                                # always converge at least one start
            st = take(n)
            if st is None:
                break
            submit(n, _optimize(n, st[0], n_end, r0=st[1]))

    # ---- phase B: DEPTH = basin hopping, allocated by measured SCORE yield. A digit counts the same
    # at every n, so the n to work on next is simply the one whose last slice bought the most digits
    # per CPU-second. The prior before any slice is measured is the same formula: digits still
    # missing / cost of a pass -- worst-and-cheapest first, fitted rather than tabulated.
    rate = {n: (DIGCAP - _dig(n, best.get(n, (0.0,))[0], recs)) / max(_pass_cost(n), 1e-6)
            for n in order}
    visits = {n: 0 for n in order}
    while meter.left() > 0:
        now = time.process_time()
        if now >= deadline:
            break
        # Only n that can still buy digits are candidates. Without this the `-visits` tiebreak hands
        # a slice to every capped n as soon as the measured rates flatten to zero, and an n that
        # reaches the cap mid-run keeps its just-earned high rate and is picked again for nothing.
        # Recomputed live, so an n drops out the moment it caps, and `or order` keeps the fallback
        # honest: if nothing has headroom left, the budget is still spent rather than abandoned.
        cand = [m for m in order if _head(m) > 0.0] or order
        n = max(cand, key=lambda m: (rate[m], -visits[m], m))
        left = deadline - now
        span = min(left, max(6.0 * _pass_cost(n), left / 12.0))
        end = now + span
        gain = 0.0
        while time.process_time() < end and meter.left() > 0:
            b = best.get(n)
            if b is None:
                st = take(n)
                if st is None:
                    break
                submit(n, _optimize(n, st[0], end, r0=st[1]))
                break
            # The two perturbations of the incumbent, plus CONTINUING
            # THE COLD SWEEP (next row count or next transfer source). The sweep does not stop when
            # breadth's clock runs out -- the bandit above already allocates slices by measured
            # digit yield, so the n where a fresh paradigm pays keep getting handed new starts, and
            # the n where it does not fall back to kicks on their own measured rate. By depth time
            # the transfer sources are this call's own improved packings, so the census propagates
            # sideways within the call as well as across iterations.
            # Four moves now, drawn uniformly: the two kicks, the cold sweep, and RECOMBINING two
            # members of n's own population of distinct local optima. The kicks edit one packing and
            # the sweep replaces it wholesale; this is the only move whose output is built out of
            # two things that both already converged at THIS n.
            draw = rng.randint(4)
            st = take(n) if draw == 2 else (_arch_splice(n, arch.get(n, ()), rng) if draw == 3
                                            else None)
            if st is not None:
                xy0, r0 = st
            else:
                # The kick's parent is drawn from n's POPULATION, not fixed to the incumbent:
                # perturbing only the best packing re-explores one basin's neighbourhood forever.
                xy0, r0 = _kick(n, _arch_parent(n, arch.get(n, ()), b[1], rng), rng)
            gain += submit(n, _optimize(n, xy0, end, r0=r0))
        spent = max(time.process_time() - now, 1e-6)
        rate[n] = gain / spent
        visits[n] += 1


# ------------------------------------------------------------------------------------- self-test
def _self_test():
    class _M:
        def __init__(self, b):
            self.budget, self.used = b, 0

        def tick(self, k):
            g = max(0, min(int(k), self.budget - self.used))
            self.used += g
            return g

        def left(self):
            return max(0, self.budget - self.used)

    meter = _M(500000)
    seen = {}
    calls = {}

    def ev(n, packing):
        calls[n] = calls.get(n, 0) + (1 if np.asarray(packing).ndim == 2 else len(packing))
        a = np.asarray(packing, dtype=float)
        single = a.ndim == 2
        if single:
            a = a[None]
        g = meter.tick(a.shape[0])
        feas = np.zeros(len(a), dtype=bool)
        sr = np.full(len(a), -np.inf)
        for i in range(g):
            x, y, r = a[i, :, 0], a[i, :, 1], a[i, :, 2]
            d = np.sqrt(((a[i, :, None, :2] - a[i, None, :, :2]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            wall = np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min()
            pair = float((d - (r[:, None] + r[None, :])).min())
            ok = bool(r.min() > 0 and wall >= -1e-9 and pair >= -1e-9)
            feas[i] = ok
            sr[i] = r.sum()
            if ok and sr[i] > seen.get(n, (-np.inf,))[0]:
                seen[n] = (float(sr[i]), a[i].copy())
        return (bool(feas[0]), float(sr[0])) if single else (feas, sr)

    global CPU_BUDGET
    CPU_BUDGET = 9.0
    rng = np.random.RandomState(0)
    # BREADTH is the property iteration 2 added: on a budget this small a greedy schedule spends
    # everything on the first n, so assert EVERY target came back feasible -- including the
    # expensive one, which is the end that used to be starved.
    solve(ev, meter, rng, [27, 61, 99])
    assert set(seen) == {27, 61, 99}, "breadth pass starved some n: got %s" % sorted(seen)
    first = seen[27][0]
    # a SECOND call in the SAME process must still work (no module-constant CPU deadline), and a
    # single-n targets list must not break the per-n split
    seen.clear()
    solve(ev, meter, rng, [29])
    assert 29 in seen, "second call in the same process returned no feasible packing"
    # an n with NO committed pack must cold-start cleanly, not raise
    seen.clear()
    solve(ev, meter, rng, [28])
    assert 28 in seen, "unseeded n did not cold-start"
    # every SLP iterate is feasible BY CONSTRUCTION (the linearisation is an inner approximation);
    # check that directly on a raw, unrepaired convergence so a regression cannot hide behind _grow.
    xy = _rows_start(40, 6, rng)
    pk = _optimize(40, xy, time.process_time() + 3.0)
    d = _pair_dists(pk[:, :2]) - (pk[:, 2][:, None] + pk[:, 2][None, :])
    assert pk[:, 2].min() > 0, "SLP produced a non-positive radius"
    assert d.min() >= -1e-9, "SLP produced overlapping circles (pair slack %.3g)" % d.min()
    assert (_walls(pk[:, :2]) - pk[:, 2]).min() >= -1e-9, "SLP produced a circle out of bounds"
    # the nesting termination: re-converging an ALREADY converged packing must cost a couple of LP
    # solves, not a shrink cascade down to T=1e-13 (that was 29 of 31 solves before iteration 4).
    _lp_reset()
    pk2 = _optimize(40, pk[:, :2], time.process_time() + 3.0, r0=pk[:, 2])
    lp = _LP_CALLS[0]
    assert lp <= 6, "converged restart cost %d LP solves -- the shrink cascade is back" % lp
    assert pk2[:, 2].sum() >= pk[:, 2].sum() - 1e-9, "re-convergence lost ground"
    # the row sweep is this iteration's change: every row count must be reachable (it is a
    # permutation of 1..n, squarest first) and NO row may be a stub -- the old generator truncated
    # a full lattice and left the last row holding a single circle, which is the bad start that
    # made the parked n parked.
    for nn in (40, 91):
        order = _row_order(nn)
        assert sorted(order) == list(range(1, nn + 1)), "row sweep is not a permutation of 1..n"
        assert abs(order[0] - round(math.sqrt(nn))) <= 1, "row sweep does not start at the square"
        for rws in (1, 3, 7, 18, nn):
            for sp in (False, True):
                pts = _rows_start(nn, rws, rng, sp)
                assert pts.shape == (nn, 2), "row start produced %s points" % (pts.shape,)
                ymid = _row_ys(nn, min(rws, nn), True)
                who = np.argmin(np.abs(pts[:, 1][:, None] - ymid[None, :]), axis=1)
                cnt = [int(np.sum(who == i)) for i in range(len(ymid))]
                assert max(cnt) - min(cnt) <= 1, "rows %d at n=%d unbalanced: %s" % (rws, nn, cnt)
                assert pts[:, 1].min() >= LO and pts[:, 1].max() <= HI, "a row start left the square"
    # THIS iteration's change: WHICH rows are long is the family's second axis, and both values of
    # it are really swept. Three things are pinned, because an implementation that quietly ignored
    # `spread` would pass every assertion above.
    #  (i) the two assignments genuinely DIFFER -- and differ in the structured way claimed: when
    #      `rem ~ rows/2` the spread rule must ALTERNATE long/short, which front-loading never does.
    for nn, rws in ((27, 5), (49, 9), (91, 6)):
        rem = nn % rws
        front = [1 if i < rem else 0 for i in range(rws)]
        spr = [1 if ((i + 1) * rem) % rws < rem else 0 for i in range(rws)]
        assert sum(front) == sum(spr) == rem, "an assignment changed the circle count"
        if 0 < rem < rws and rem % rws != 0 and front != spr:
            fr_runs = max(len(list(g)) for _, g in itertools.groupby(front))
            sp_runs = max(len(list(g)) for _, g in itertools.groupby(spr))
            assert sp_runs <= fr_runs, "the spread rule is not more spread than front-loading"
    assert [1 if ((i + 1) * 2) % 5 < 2 else 0 for i in range(5)].count(1) == 2
    assert [1 if ((i + 1) * 3) % 6 < 3 else 0 for i in range(6)] == [0, 1, 0, 1, 0, 1], \
        "rem = rows/2 must give the alternating (hexagonal) pattern"
    #  (ii) the row counts really come out different, so `spread` is not decorative.
    for nn, rws in ((27, 5), (49, 9)):
        a = _rows_start(nn, rws, np.random.RandomState(3), False)
        b = _rows_start(nn, rws, np.random.RandomState(3), True)
        ym = _row_ys(nn, rws, True)
        ca = [int(np.sum(np.argmin(np.abs(a[:, 1][:, None] - ym[None, :]), axis=1) == i)) for i in range(rws)]
        cb = [int(np.sum(np.argmin(np.abs(b[:, 1][:, None] - ym[None, :]), axis=1) == i)) for i in range(rws)]
        assert ca != cb, "spread=True gave the same row counts as spread=False at n=%d" % nn
    # THIS iteration's change: the top/bottom wall margin is SOLVED, not pinned at half the pitch.
    # An implementation that kept `m = p/2` passes every row assertion above, so three things are
    # pinned here, and (iii) is the one that fails for the old rule.
    for nn in (39, 45, 49, 75, 81, 91, 50):
        for rws in (2, 5, 7, 9, 18):
            if rws > nn:
                continue
            ys = _row_ys(nn, rws, True)
            assert len(ys) == rws and np.all(np.diff(ys) > 0), "row centres are not increasing"
            p = float(ys[1] - ys[0])
            m = float(ys[0] - LO)
            #  (i) the layout still FILLS the square symmetrically: margin equal top and bottom,
            #      one common pitch, everything strictly inside.
            assert abs((HI - ys[-1]) - m) < 1e-12, "the two wall margins differ"
            assert np.allclose(np.diff(ys), p, atol=1e-12), "the pitch is not constant"
            assert m > 0.0 and p > 0.0, "n=%d rows=%d gave m=%.3g p=%.3g" % (nn, rws, m, p)
            #  (ii) the margin IS the radius the layout supports -- the defining equation, not a
            #       fraction: m == min(a/2, sqrt(a^2/4 + p^2)/2), unless the walls never bind, in
            #       which case the solve must hand back the old uniform layout exactly.
            a = 1.0 / (float(nn) / rws + 0.5)
            rr = min(0.5 * a, 0.5 * math.sqrt(0.25 * a * a + p * p))
            uni = abs(m - 0.5 / rws) < 1e-12
            assert uni or abs(m - rr) < 1e-9, \
                "n=%d rows=%d: margin %.6g is not the supported radius %.6g" % (nn, rws, m, rr)
    #  (iii) it DEPARTS from p/2 exactly where the pinned rule was leaving radius on the table, and
    #        reproduces it to three digits where it was not. n=91 at its measured winning row count
    #        (18 rows of 5, cell aspect 3.6:1) is the case the old rule jammed against the wall.
    _y91 = _row_ys(91, 18, True)
    assert (_y91[0] - LO) / float(_y91[1] - _y91[0]) > 0.9, \
        "the tall layout's wall margin was not freed (m/p = %.3f)" % ((_y91[0] - LO) / float(_y91[1] - _y91[0]))
    for _n, _r in ((45, 7), (75, 9)):
        _y = _row_ys(_n, _r, True)
        assert abs((_y[0] - LO) / float(_y[1] - _y[0]) - 0.5) < 0.02, \
            "n=%d rows=%d moved a margin the pinned rule already had right" % (_n, _r)
    assert np.allclose(_row_ys(40, 1, True), [0.0]), "a single row must sit at the centre"
    # the TRANSFER family is this iteration's change. It must produce exactly n circles from a
    # source of ANY size, keep the LARGEST ones when it shrinks (dropping the small ones is what
    # makes the survivors an already-feasible packing in the source's basin), and the circles it
    # INSERTS must not overlap anything -- an inserted circle wider than its clearance would hand
    # the SLP an infeasible start and silently cost the whole transfer its structure.
    src = _warm_pack(61)
    for nn in (55, 61, 70):
        txy, tr = _transfer_start(nn, src[0], src[1], rng)
        assert txy.shape == (nn, 2) and tr.shape == (nn,), "transfer to n=%d gave %s" % (nn, tr.shape)
        assert tr.min() > 0, "transfer produced a non-positive radius"
        if nn <= 61:
            # ITERATION 13: the surviving set is now DRAWN, not fixed to the n largest, so what is
            # pinned is (a) it is still a genuine SUBSET of the source -- coordinates and radii
            # carried over untouched, which is what makes the survivors exactly feasible -- and
            # (b) the draw is really weighted toward dropping small circles rather than uniform.
            sr = np.sort(src[1])[::-1]
            assert all(np.min(np.abs(sr - v)) < 1e-12 for v in tr), \
                "shrinking transfer invented a radius instead of keeping a subset"
            drops = [int(np.argmin(np.abs(sr - v))) for v in
                     sorted(set(np.round(src[1], 12)) - set(np.round(tr, 12)))]
            assert len(set(np.round(tr, 12))) == len(tr), "shrinking transfer kept a duplicate"
        dd = _pair_dists(txy) - (tr[:, None] + tr[None, :])
        assert dd.min() >= -1e-9, "transfer start overlaps (slack %.3g)" % dd.min()
        assert (_walls(txy) - tr).min() >= -1e-9, "transfer start leaves the square"
    # ITERATION 13: `_transfer_start` used to IGNORE its `rng` -- every visit to a source rebuilt
    # the identical start, so the family was a finite list of |srcs| fixed points. Both directions
    # are now drawn. Pin that the draw is real (a deterministic implementation passes every
    # assertion above) and that it is WEIGHTED, not uniform: over many draws the smallest circle
    # must be dropped far more often than the largest, or the "drop the small ones" structure that
    # makes a shrinking transfer land in the source's basin is gone.
    s61 = _warm_pack(61)
    rg = np.random.RandomState(7)
    keep_sets = set()
    rank = np.argsort(np.argsort(-s61[1]))          # 0 = largest circle, 60 = smallest
    dropped = np.zeros(61)
    for _ in range(60):
        txy, tr = _transfer_start(57, s61[0], s61[1], rg)
        idx = frozenset(int(np.argmin(np.abs(s61[1] - v))) for v in tr)
        keep_sets.add(idx)
        for i in range(61):
            if i not in idx:
                dropped[i] += 1.0
    assert len(keep_sets) >= 30, \
        "shrinking transfer is still deterministic: %d distinct subsets in 60 draws" % len(keep_sets)
    big = dropped[rank < 10].sum()
    small = dropped[rank >= 51].sum()
    # The bar is what the rule IMPLIES, not a number picked to pass: with w = 1/r**2 the odds of
    # dropping a given circle scale as r**-2, and n=61's pack spans rmax/rmin = 2.12, so the ten
    # smallest should go ~(2.12)**2 = 4.5x as often as the ten largest. 2.5x is that with margin;
    # uniform (a broken draw) would give 1.0x.
    assert small > 2.5 * big, \
        "the drop is not weighted toward small circles (small %.0f vs large %.0f)" % (small, big)
    ins = set()
    for _ in range(40):
        txy, tr = _transfer_start(65, s61[0], s61[1], rg)
        assert txy.shape == (65, 2)
        dd = _pair_dists(txy) - (tr[:, None] + tr[None, :])
        assert dd.min() >= -1e-9, "sampled insertion overlaps (slack %.3g)" % dd.min()
        assert (_walls(txy) - tr).min() >= -1e-9, "sampled insertion leaves the square"
        ins.add(tuple(np.round(txy[61:].ravel(), 9)))
    assert len(ins) >= 20, \
        "growing transfer is still deterministic: %d distinct insertions in 40 draws" % len(ins)

    # the SEAM SPLICE is this iteration's change. It must be a FEASIBLE packing of exactly n
    # circles for any pair of sources and any seam draw -- an overlapping hybrid would hand the SLP
    # a start it cannot repair without collapsing radii -- and it must actually MIX: a splice that
    # silently returns one source unchanged is the round-robin slot spent on nothing. Both source
    # orders are checked, since the seam admits A wholesale and B only where it fits.
    a61 = _warm_pack(61)
    for (na, nb) in ((61, 59), (59, 61), (61, 99)):
        A, B = _warm_pack(na), _warm_pack(nb)
        mixed = 0
        for seed in range(12):
            rr = np.random.RandomState(1000 + seed)
            for nn in (57, 61, 64):
                st = _splice_start(nn, A[0], A[1], B[0], B[1], rr)
                assert st is not None, "splice %d+%d -> %d refused" % (na, nb, nn)
                sxy, sr = st
                assert sxy.shape == (nn, 2) and sr.shape == (nn,), \
                    "splice to n=%d gave %s" % (nn, sr.shape)
                assert sr.min() > 0, "splice produced a non-positive radius"
                dd = _pair_dists(sxy) - (sr[:, None] + sr[None, :])
                assert dd.min() >= -1e-9, "splice start overlaps (slack %.3g)" % dd.min()
                assert (_walls(sxy) - sr).min() >= -1e-9, "splice start leaves the square"
                if nn == 61 and na == 61:
                    same = np.abs(np.sort(sr) - np.sort(A[1])).max() < 1e-12
                    mixed += 0 if same else 1
        if na == 61 and nb != 61:
            assert mixed > 0, "splice %d+%d never mixed -- it just returned A" % (na, nb)
    # the ROTATED LATTICE family is this iteration's change. It must return exactly n points for
    # any angle, all strictly inside the square, with no two coincident (a duplicated lattice point
    # would hand the SLP a zero-radius pair it can never separate) -- and the angle sweep must
    # actually ROTATE, i.e. theta=0 and theta=30deg must give visibly different layouts, which is
    # the whole reason this family exists beside the axis-aligned row sweep.
    for nn in (27, 40, 91):
        for th in (0.0, math.pi / 12.0, math.pi / 6.0, math.pi / 3.0 - 1e-9):
            q = _lattice_start(nn, th, rng)
            assert q is not None and q.shape == (nn, 2), \
                "lattice start at n=%d theta=%.3f gave %s" % (nn, th, None if q is None else q.shape)
            assert q.min() >= LO - 1e-12 and q.max() <= HI + 1e-12, "lattice point left the square"
            assert _pair_dists(q).min() > 1e-9, "lattice start has coincident points"
    q0 = _lattice_start(64, 0.0, np.random.RandomState(1))
    q1 = _lattice_start(64, math.pi / 6.0, np.random.RandomState(1))
    assert np.abs(np.sort(q0[:, 1]) - np.sort(q1[:, 1])).max() > 1e-3, \
        "the angle sweep is not rotating -- theta is being ignored"
    # van der Corput must stay in [0,1) and never repeat, or the sweep would re-try one angle
    vs = [_vdc(k) for k in range(32)]
    assert all(0.0 <= v < 1.0 for v in vs) and len(set(vs)) == 32, "vdc sweep is degenerate"
    # THE CLOCK IS ALLOCATED BY ACHIEVABLE DIGITS -- this iteration's change. The rule itself:
    # capped -> 0, unknown n -> the full cap, and monotone in between.
    _r = {61: 2.0}
    assert _headroom(61, 2.0, _r) == 0.0, "a packing AT the record still claims headroom"
    assert _headroom(61, 2.5, _r) == 0.0, "a packing PAST the record still claims headroom"
    assert _headroom(70, 1.0, _r) == DIGCAP, "an n with no record is not treated as fully open"
    assert 0.0 < _headroom(61, 1.9, _r) < _headroom(61, 1.0, _r), "headroom is not monotone"
    # ...and end to end: a capped n must still be COVERED by breadth (one converged pack, so the
    # census keeps every n) but must receive strictly fewer candidates than a parked n in the same
    # call, or the 39% of the clock this change redirects would quietly leak back.
    recs0 = _records()
    cap_n, par_n = 61, 49
    for nn, want in ((cap_n, True), (par_n, False)):
        w = _warm_pack(nn)
        assert w is not None, "self-test needs a committed pack at n=%d" % nn
        capped = _headroom(nn, float(w[1].sum()), recs0) <= 0.0
        assert capped is want, "n=%d capped=%s -- the allocation test needs one of each" % (nn, capped)
    calls.clear(); seen.clear()
    CPU_BUDGET = 9.0
    solve(ev, meter, np.random.RandomState(3), [par_n, cap_n])
    assert cap_n in seen, "breadth stopped covering the capped n"
    assert calls.get(par_n, 0) > calls.get(cap_n, 0), \
        "the capped n took as many candidates as the parked one: %s" % calls
    # ---- the POPULATION of distinct local optima (iteration 10)
    # (a) distinctness: re-inserting the same optimum must not take a second slot, a genuinely
    #     different one must, and the list is the best ARCH_MAX by sum_r.
    _a = []
    _w31 = _warm_pack(31)
    _p0 = _optimize(31, _w31[0], time.process_time() + 5.0, r0=_w31[1])
    assert _arch_add(_a, _p0) and not _arch_add(_a, _p0.copy()), \
        "the population counted the same optimum twice -- a kick that returns to its own basin " \
        "would fill every slot with one packing"
    assert len(_a) == 1
    _rg = np.random.RandomState(11)
    for _ in range(3 * ARCH_MAX):
        _xy, _r0 = _kick(31, _p0, _rg)
        _arch_add(_a, _optimize(31, _xy, time.process_time() + 5.0, r0=_r0))
    assert len(_a) <= ARCH_MAX, "the population is unbounded: %d members" % len(_a)
    assert len(_a) >= 2, "kicks produced no second distinct optimum at n=31"
    assert _a[0][0] == max(t[0] for t in _a), "the population is not best-first"
    assert all(_a[i][0] >= _a[i + 1][0] for i in range(len(_a) - 1)), "population not sorted"
    # (b) the recombination is feasible, has exactly n circles, and actually MIXES -- a seam that
    #     handed back a parent unchanged would spend a quarter of the depth moves on nothing.
    _mix = 0
    for _sd in range(12):
        _st = _arch_splice(31, _a, np.random.RandomState(100 + _sd))
        assert _st is not None, "the population refused to recombine with %d members" % len(_a)
        _sxy, _sr = _st
        assert _sxy.shape == (31, 2) and _sr.shape == (31,), "archive splice returned %d circles" % len(_sr)
        assert _sr.min() > 0, "archive splice produced a non-positive radius"
        _dd = _pair_dists(_sxy) - (_sr[:, None] + _sr[None, :])
        assert _dd.min() >= -1e-9, "archive splice overlaps (slack %.3g)" % _dd.min()
        assert (_walls(_sxy) - _sr).min() >= -1e-9, "archive splice leaves the square"
        if min(np.abs(np.sort(_sr) - t[1]).max() for t in _a) > 1e-9:
            _mix += 1
    assert _mix >= 6, "archive splice handed back a parent unchanged on %d of 12 seeds" % (12 - _mix)
    assert _arch_splice(31, _a[:1], np.random.RandomState(0)) is None, \
        "a one-member population must refuse, not crash"
    # (b1) CROWDING eviction (iteration 12): an over-full population must drop the REDUNDANT
    #      member, not the worst one. Truncating by sum_r keeps the list full of near-copies of the
    #      incumbent -- the probe measured mean pairwise spread 6.8e-3 for truncation against
    #      2.5e-2 for crowding on 18 of 18 cells -- which turns the uniform draw of iterations
    #      10-11 back into the elitist draw those iterations measured as flat.
    _c = [(3.0, np.array([1.0, 3.0]), None), (2.0, np.array([1.0, 2.0]), None),
          (1.0, np.array([1.0, 1.99]), None)]
    assert _crowded(_c) == 2, "crowding kept the redundant member: %d" % _crowded(_c)
    _c2 = [(3.0, np.array([1.0, 1.0]), None), (2.0, np.array([1.0, 1.0]), None),
           (1.0, np.array([1.0, 9.0]), None)]
    assert _crowded(_c2) == 1, "crowding must evict the WORSE of the closest pair: %d" % _crowded(_c2)
    assert all(_crowded(_c[:k]) > 0 for k in (2, 3)), "crowding would evict the best member"
    # and it must actually WIDEN a real population: the same 3*ARCH_MAX kicks at n=31, archived
    # under each rule, must leave crowding with the strictly larger mean pairwise spread.
    def _spread(a):
        return float(np.mean([np.abs(a[i][1] - a[j][1]).max()
                             for i in range(len(a)) for j in range(i + 1, len(a))]))
    _at, _ac = [], []
    _rg2 = np.random.RandomState(11)
    for _ in range(3 * ARCH_MAX):
        _xy, _r0 = _kick(31, _p0, _rg2)
        _q = _optimize(31, _xy, time.process_time() + 5.0, r0=_r0)
        _arch_add(_ac, _q)
        _sq = np.sort(_q[:, 2])                       # the OLD rule: truncate by sum_r
        if not any(len(t[1]) == len(_sq) and np.abs(t[1] - _sq).max() <= FEAS_TOL for t in _at):
            _at.append((float(_sq.sum()), _sq, _q))
            _at.sort(key=lambda t: -t[0])
            del _at[ARCH_MAX:]
    assert len(_ac) == len(_at) == ARCH_MAX, "population not full: %d/%d" % (len(_ac), len(_at))
    assert _spread(_ac) > _spread(_at), \
        "crowding did not widen the population (%.3g vs truncation %.3g)" % (_spread(_ac), _spread(_at))
    # (b2) the KICK's parent is drawn from the population, not pinned to the incumbent. A draw that
    #      always returned the best member would pass every feasibility check above while leaving
    #      three of the four depth moves the (1+1) hill-climb they were before.
    _fb = _p0
    assert _arch_parent(31, [], _fb, np.random.RandomState(0)) is _fb, \
        "an empty population must fall back to the incumbent"
    assert _arch_parent(31, _a[:1], _fb, np.random.RandomState(0)) is _fb, \
        "a one-member population must fall back to the incumbent"
    _rgp = np.random.RandomState(5)
    _par = [_arch_parent(31, _a, _fb, _rgp) for _ in range(40)]
    assert all(any(q is t[2] for t in _a) for q in _par), "kick parent came from outside the population"
    assert sum(1 for q in _par if q is not _a[0][2]) >= 10, \
        "the kick parent is effectively pinned to the best member -- the population is decorative"
    # (c) end to end: a real solve() over a parked n must ACTUALLY reach this move, or the wiring
    #     is dead code that the unit checks above would still pass.
    _u0, _k0 = _ARCH_USES[0], _ARCH_KICKS[0]
    calls.clear(); seen.clear()
    CPU_BUDGET = 12.0
    solve(ev, meter, np.random.RandomState(7), [49])
    assert 49 in seen, "solve() returned nothing for a single parked target"
    assert _ARCH_USES[0] > _u0, "phase B never drew an archive splice -- the population is unused"
    assert _ARCH_KICKS[0] > _k0, "phase B never kicked a population member -- the draw is dead code"
    # the source sweep must be nearest-first and must never offer n itself
    so = _source_order(61, [27, 59, 61, 63, 99])
    assert so[0] == 63 and so[1] == 59 and 61 not in so, "source order is not nearest-first: %s" % so
    recs = json.load(open("bench/records.json"))["records"]
    print("self-test OK: breadth covered 27/61/99 ; n=27 sum_r=%.6f (record %.6f, %.2f digits) ; "
          "n=28/29 reruns feasible ; raw SLP output feasible ; converged restart = %d LP solves"
          % (first, float(recs["27"]),
             -math.log10(max((float(recs["27"]) - first) / float(recs["27"]), 1e-12)), lp))
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(__doc__)
