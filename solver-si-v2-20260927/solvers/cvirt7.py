"""Circle-packing solver: sequential linear programming (SLP) with multistart + teleport kicks.

Contract (MISSION.md): ``solve(evaluate, meter, rng, targets)`` packs n circles into the centred unit
square ``[-0.5,0.5]^2`` maximising ``sum r_i``.  Every candidate that should count is routed through the
metered ``evaluate()``.

Why SLP.  With the centres ``p_i`` fixed the problem is a plain LP in the radii
(``r_i + r_j <= d_ij``, ``r_i <= wall_i``, ``r_i >= 0``).  Letting the centres move too, the only
nonlinear constraint is ``||p_i - p_j|| >= r_i + r_j``.  The Euclidean norm is CONVEX, so for any step
``delta`` we have the exact inequality

    ||v + delta|| >= ||v|| + u . delta        with u = v/||v||,

i.e. the first-order model of the separation is a *lower bound* on the true separation.  Therefore an LP
that enforces ``d_ij + u_ij . (dp_i - dp_j) >= r_i + r_j`` inside a trust region produces a step whose
image is GUARANTEED feasible for the true problem -- no line search, no infeasible iterate, and
``delta = 0`` is always in the LP's feasible set so the objective never decreases.  Shrinking the trust
region on a non-improving step drives it to a local optimum.  Repairs afterwards are only float-rounding
sized.

CPU is the binding meter (120 s process-CPU backstop vs 500k evaluations), so the deadline is computed at
ENTRY as ``time.process_time() + CPU_BUDGET`` -- never as a module constant -- which keeps a second
``solve()`` call in the same process working.  The per-n time slice is recomputed from the targets still
outstanding, so it is correct for ``len(targets) == 1``.

``CPU_BUDGET`` sizes itself against that backstop.  The driver arms ITIMER_PROF for
``DRIVER_BACKSTOP_S`` of process CPU *immediately before* calling ``solve()`` and disarms it before the
harvest, so the whole cap is ours to spend; the only cost of overrunning it is that the timer lands a
few percent late and the run stops being reproducible.  ``solve()`` can overshoot ``CPU_BUDGET`` by at
most one LP iteration plus one ``evaluate`` (measured <0.1 s at n=99, the largest visible size), because
the deadline is re-checked at the top of every SLP iteration and every kick.  ``CPU_MARGIN_S`` keeps 6 s
-- ~60x that worst case -- in reserve, and the self-test pins BOTH the margin and the overshoot so the
constant cannot drift into the backstop.  Iterations 1-4 all ran at 100 s and reported ``cpu=100.0s
backstop=no``, i.e. they left a sixth of the sanctioned CPU unspent; since throughput (kicks per slice)
is what iterations 3 and 4 showed converts into digits, that headroom is worth taking.

Warm starts read ``bench/packs/csqv<n>.pck`` when it exists (guarded; a missing pack just means a cold
start).  Nothing is ever written from inside ``solve()``.

Escaping SLP local optima.  The committed census is made of fully converged SLP stationary points -- an
extra SLP pass from one buys nothing -- so all of the per-n slice goes into KICKS that reseed the search.
The kick is a *teleport*: the k smallest circles jump to fresh random spots and the pack is re-converged,
which preserves the hard-won macro layout of the big circles while reshuffling the fillers.  Measured on
an equal 3 s/n slice against a global Gaussian basin hop, cold restarts and a radius-scaled jitter, the
teleport won at every size tried (n=99: relgap 0.0050 vs 0.0135 for the Gaussian hop).

Where the kick LANDS.  A uniform destination drops a filler wherever it falls, usually on top of an
already-tight region, and the SLP then has to squeeze it back out -- which is the shape of an n that
resists the kick at ever-rising kick counts rather than one starved of samples.  ``GAP_SAMPLES`` draws
m uniform candidates per moved circle and keeps the one with the largest CLEARANCE (distance to the
nearest wall or staying circle's boundary, and to the spots already chosen in the same kick, so the k
movers spread out).  It is a ladder whose bottom rung is exactly the proven uniform teleport -- m=1
draws the identical numbers in the identical order, asserted in the self-test -- so raising it changes
one knob rather than the move.  Measured (iteration 6, artifacts/diag11.py) at a faithful ~2.7 CPU-s/n
slice over n=27,35,45,63,75,85,99 x 3 seeds, paired per cell: m=64 beat m=1 on ALL THREE seeds and won
13 of 21 cells to 5 (+0.079 mean digits).  m=8 was rejected despite a higher headline mean -- it lost
two seeds of three and 9 cells to 7, its mean coming from a single cell that happened to cap at 7
digits.  The extra draws are free next to the LP: m=64 fitted MORE kicks per slice than m=1, not fewer
(517 vs 452 evaluations at n=27..99, seed 0).

The destination is now SETTLED at m=64/argmax -- both ways off that rung were measured and both lose.
(i) MORE draws (iteration 7, artifacts/diag12.py): m=128 -> -0.029 and m=512 -> -0.058 mean digits
against m=64, each losing the per-cell view too.  The 1->64 curve turns over, so the ladder is walked
out, not merely un-extended.  Diagnostically the interesting part is that m=128/512 lost while fitting
MORE kicks per slice (530-600 evaluations vs 499), so it is not a throughput cost -- the extra draws buy
a worse move, not fewer moves.  (ii) That suggested a mechanism -- as m grows, argmax-of-m converges on
the single emptiest point of a fixed incumbent, so successive kicks re-test one configuration and the
kick stops being a search -- and ``SPOT_TOPQ`` tests it directly by picking uniformly among the top-q
draws by clearance instead of the strict argmax.  It was REFUTED (iteration 7, artifacts/diag13.py):
q=4 -> -0.030 and q=16 -> -0.043 mean digits, and q=4 lost a seed outright despite winning cells 6-4.
So the loss at large m is NOT lost diversity; clearance simply saturates as a criterion.  ``SPOT_TOPQ``
is kept at rung 1, where it is a PROVABLE no-op -- it takes the argmax and makes no rng call, so the
downstream stream is untouched, and the self-test asserts both halves of that.

Which circles MOVE is the last kick component to be measured, and it too keeps its original setting:
``MOVE_MODE`` is the k SMALLEST radii at rung 0, where it is bit-exactly the ``np.argsort(r)[:k]`` that
predates the knob and makes no rng call (both halves asserted in the self-test).  The alternative tested
was a contiguous SPATIAL cluster -- rearranging one local patch as a unit rather than k scattered
fillers -- anchored either on the smallest circle (mode 1) or on a uniformly random one (mode 2).
Both LOSE (iteration 8, artifacts/diag14.py, the same 7 n x 3 seeds at a faithful ~2.7 CPU-s slice):
mode 1 -> -0.017 mean digits, losing cells 6-5 and all three seeds; mode 2 -> -0.016, and it is the
trap again -- it WINS the per-cell view 6-4 while losing the mean and two seeds of three.  So the
smallest-radii selection is not an unexamined default any more; it is the measured best of three, and
what the cluster costs is visible in the throughput too (504-518 evaluations per slice vs 521-539),
since re-converging a whole patch is a harder LP than reseating loose fillers.

Where the slice actually goes.  Each SLP run ends in a long non-improving tail (tau is only shrunk
by 0.6 per failed iteration, so ~37 further LP solves separate "stopped improving" from the 1e-10
stop).  ``KICK_STALL`` cuts that tail, which cannot change what a run returns except by giving up an
improvement it was not finding, and buys far more kicks per slice.  The cutoff ladder has now been
walked end to end on an equal warm-started slice: 16 > 8 > 4 (iteration 3, n=29..99) and then
4 > 3 > 2 > 1 (iteration 4, n=29,35,45,67,85,99 x 2 seeds), monotone the whole way and NEVER worse at
any cell, so it is pinned at its floor of 1 -- a kick now ends the instant an SLP iteration fails to
improve, spending the whole slice on fresh kicks instead of on tau refinement.  A final full-tail
polish of the incumbent was measured alongside and REJECTED: it never once improved a pack and only
cost kicks, confirming that what a kick gives up in the tail was never there to find.

How BIG each SLP step is.  ``KICK_TAU`` is the initial trust-region radius a kick re-converges under,
and because ``KICK_STALL == 1`` cuts the run on the first non-improving iteration, the 0.6 shrink is
reached at most once and never compounds -- so this one scalar is effectively the FIXED step size of
the entire kick.  It sat at 0.03 from iteration 1 by choice, not by measurement; iteration 10 laddered
it (artifacts/diag17-19, the diag15/16 protocol: 9 n x 3 seeds at a faithful ~2.7 CPU-s/n slice, paired
per cell) and moved it to 0.015.  The ladder is bracketed on both sides: 0.12 loses (-0.014), 0.06
looked like a winner on seeds 0-2 (+0.027) and was REFUTED on fresh seeds 3-5 (-0.011), and 0.0075 is
the trap this file has now seen four times -- a higher headline mean (+0.018) with a LOSING per-cell
view (4-6) and a seed lost outright, so it is rejected on the same bar that rejected m=8, q=4 and
mode 2.  Only 0.015 repeats: +0.026 on seeds 0-2, +0.050 on the disjoint seeds 3-5, winning the mean,
the per-cell view (7-1 and 7-2) and SIX of six seeds.  The mechanism is throughput at constant quality:
halving tau costs nothing per kick (440-470 evaluations per slice against the incumbent's 440-474) while
0.06 and 0.0075 both fall to ~350, i.e. 0.015 sits at the point where the LP is still cheap and the step
no longer overshoots the basin the kick landed in.

How MANY circles a kick moves.  ``k`` is drawn ``U{1 .. max(2, n // KICK_KDIV) - 1}``, and the divisor
sat at 8 from iteration 1 by choice, not by measurement -- the last unmeasured component of the kick.
Rung 8 is a PROVABLE no-op (same rng call with the same arguments, so the k stream is bit-exact and the
downstream stream is untouched; the self-test asserts it).  Iteration 11 laddered it on the diag15-19
protocol (9 n x 3 seeds at a faithful ~2.7 CPU-s/n slice, paired per cell) and moved it to 16.
BIGGER kicks lose decisively and monotonically (artifacts/diag20.py): kdiv=4 -> -0.039 mean digits,
losing the per-cell view 3-8, and kdiv=2 -> -0.046, losing it 1-8.  kdiv=2 is the clearest negative
this file has produced -- at seeds 0 and 1 it reproduced the warm census EXACTLY, i.e. not one kick in
27 CPU-s beat the incumbent.  SMALLER kicks win, and unlike the traps this ladder REPEATED on a
disjoint seed set (artifacts/diag21.py, seeds 3-5): kdiv=16 gave +0.053 with a per-cell view of 5-5 on
seeds 0-2, and then +0.081 with 8-5 on seeds 3-5 -- both means positive, the per-cell view improving
rather than flipping, 4 of 6 seeds won.  Contrast tau=0.06, whose +0.027 flipped to -0.011 on fresh
seeds; that flip is what refutation looks like here and this is not it.  The mechanism is again
throughput at constant quality: a smaller kick is a cheaper LP to re-converge, so kdiv=16 fits 510-559
evaluations into the slice against the incumbent's 457-477, while kdiv=2 falls to 345-352.

SETTLED as of iteration 12, and the ladder is now bracketed on BOTH sides.  Iteration 11 left the rung
above 16 open: kdiv=32 had posted the LARGEST headline in diag21 (+0.183 over rung 8) but on one seed
set carried by a single capped cell (n=99, seed 3: 7.000 against a 2.53-2.89 baseline), so it was
rejected pending a repeat.  artifacts/diag22.py ran that repeat on seeds 6,7,8 -- disjoint from both
diag20 (0,1,2) and diag21 (3,4,5) -- against the NEW incumbent 16, and added kdiv=64, which for every
visible n (n <= 99) gives max(2, n//64) == 2 and hence k == 1 ALWAYS: the k=1 limit the ladder had been
walking toward, and the point past which this knob cannot move.  The result REFUTES the rung:

    kdiv=16 (incumbent)  all9 3.9155  core7 3.0342   evals 641-661   --
    kdiv=32              all9 3.9267  core7 3.0486   evals 644-651   win 2 / tie 24 / loss 1, +0.0112
    kdiv=64 (k == 1)     all9 3.9086  core7 3.0253   evals 644-674   win 1 / tie 24 / loss 2, -0.0069

+0.0112 is INSIDE the ~0.015-0.03 noise floor iteration 10 measured, and it moves only 3 of 27 cells;
the +0.183 was the capped cell and nothing else.  64 is negative outright.  Note what the throughput
mechanism does here that it did NOT do below 16: evaluations per slice are flat across 16/32/64
(641-674), where every winning rung so far bought its gain by fitting MORE kicks into the slice.  The
kick is already down to k in {1,2,3} at kdiv=16 for most visible n, so shrinking the draw further buys
no throughput and there is nothing left to win.  16 is adopted and the knob is closed: 4 and 2 lose
from below, 32 is indistinguishable and 64 loses from above.

Two things NOT to change, measured in iteration 13 and left standing.  (i) The step-(0) warm-start SLP
pass (artifacts/diag23.py) re-polishes an already-converged committed pack before the first kick, and
it is a NUMERICAL NO-OP: at all 37 visible n it recovers exactly the ~1e-12-per-circle that ``_repair``
shed off the file's radii and moves no centre (max |dx| <= 6.9e-13), for a net sum change of ~1e-13.
It costs 1.61 s over 37 n = 1.4% of the budget -- so removing it is a throughput change FOUR TIMES
SMALLER than the ~0.02-digit noise floor, i.e. unmeasurable, while keeping it guarantees the kick loop
starts from the committed value rather than n*1e-12 below it.  Kept; both halves are now self-tested.
(ii) The real allocation picture (artifacts/diag24.py), which the last three blocks kept naming: at
SLICE_POW=1.0 the eleven n 27,29,31,33,35,39,43,45,49,55,67 take 29.4 s = 25.8% of every 114 s budget
and have produced ZERO improvements across the six metered runs of iterations 7-12, while the nine n
with >=3 improvements take a comparable 31.9 s and return one improvement per 5.5 CPU-s.  Three of the
dead eleven (29, 31, 83-adjacent) are simply CAPPED and already cost ~0 s via SLICE_GAP_FLOOR; the
other nine are loose but UNRESPONSIVE, and that -- not the weighting arithmetic, which the self-test
now pins as exactly relgap**SLICE_POW and strictly monotone -- is where the next headroom is.

Also measured here, and worth more than the knob: re-running an IDENTICAL configuration at IDENTICAL
seeds in a fresh process does NOT reproduce (0.015 at seeds 0/1/2 gave 3.7894/3.8540/3.8099 in diag17
and 3.7824/3.8152/3.8140 in diag19).  Nothing is wrong -- the per-n slice is a wall-of-CPU deadline, so
the kick COUNT varies with host load and the rng stream diverges from there.  The practical consequence
is that the cross-run noise floor on a 3-seed mean is ~0.015-0.03 digits, which is the same size as
every rung this ladder has ever adopted, and it is why a fresh-seed repeat -- not a bigger single run --
is the thing that separates a real effect from a lucky one.
"""
import json
import math
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

LO, HI = -0.5, 0.5

DRIVER_BACKSTOP_S = 120.0  # the frozen driver's SIGPROF process-CPU cap, armed immediately before solve()
CPU_BUDGET = 114.0       # process-CPU seconds this solve() call may use; see CPU_MARGIN_S below
CPU_MARGIN_S = DRIVER_BACKSTOP_S - CPU_BUDGET   # 6 s of headroom against a measured <0.1 s overshoot
SLP_MAX_IT = 400         # hard cap on SLP iterations per run (it normally converges far sooner)
SEED_RUNS = 2            # cold random restarts, used only when there is no warm pack to kick
KICK_STALL = 1           # stop an SLP run after this many consecutive non-improving iterations
COLD_STALL = 25          # a cold start needs a long runway, so it gets a much more generous one
GAP_SAMPLES = 64         # teleport destinations: best-clearance of this many uniform draws (1 = uniform)
SPOT_TOPQ = 1            # pick uniformly among the top-q of those draws by clearance (1 = argmax)
MOVE_MODE = 0            # which circles a kick moves: 0 = k smallest radii (incumbent), 1/2 = cluster
SLICE_POW = 1.0          # time-slice weight exponent: w_n = relgap_n ** SLICE_POW (0 = flat share)
KICK_TAU = 0.015         # initial SLP trust-region radius for a kick's re-convergence (was 0.03)
KICK_KDIV = 16           # kick size: k ~ U{1 .. max(2, n//KICK_KDIV)-1}; was 8 (the iteration-1 default)
SLICE_GAP_FLOOR = 1e-7   # relgap below this is a closed gap (7 digits) -- floors the weight, never 0
SLICE_ABANDON = 8        # end a slice after this many CONSECUTIVE non-improving kicks (0 = never; see
                         # solve() step (2)).  The freed CPU flows to the n after it, because every
                         # later slice_end is recomputed from the time actually LEFT.

MARGIN = 1e-10           # LP right-hand-side margin (feasibility tol is 1e-9)


# ----------------------------------------------------------------------------- geometry helpers

def _walls(xy):
    """Per-circle distance to the nearest wall (clipped at 0)."""
    return np.maximum(np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0],
                                         xy[:, 1] - LO, HI - xy[:, 1]]), 0.0)


def _pdist(xy):
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d


def _repair(xy, r):
    """Make (xy, r) STRICTLY feasible by shrinking radii only.

    Subtracting the full violation from *both* endpoints of a violated pair is conservative, so one pass
    suffices; the violations this has to absorb are float-rounding sized (~1e-12)."""
    n = len(r)
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    wall = _walls(xy)
    over = np.maximum(0.0, r - wall)
    if n >= 2:
        d = _pdist(xy)
        over = np.maximum(over, np.maximum(0.0, (r[:, None] + r[None, :]) - d).max(1))
    return np.maximum(r - over - 1e-12, 1e-9)


def _pack(xy, r):
    return np.concatenate([xy, r[:, None]], axis=1)


# ----------------------------------------------------------------------------- the SLP step

def _slp(xy, r, tau, deadline, max_it=SLP_MAX_IT, stall=SLP_MAX_IT):
    """Trust-region SLP ascent on sum(r).  Returns (best_sum, best_xy, best_r); always feasible.

    ``stall`` bounds the non-improving TAIL: once tau stops buying anything it is only shrunk by
    0.6 per iteration, so reaching the 1e-10 stop costs ~37 further LP solves that by construction
    cannot change the returned answer by more than the improvement they fail to find.  Cutting the
    tail can only ever return an EARLIER best-so-far, which is already feasible and already >= the
    input, so the cutoff is safe by construction -- the only question is whether the tail pays, and
    measured over n=29..99 it does not, at any rung of the ladder: relative gap reached is monotone
    in the cutoff over 16, 8, 4, 3, 2, 1 and stall=1 was never worse than any larger cutoff at any
    (n, seed) cell, because it fits ~1.5x more kicks into the same slice than stall=4 does."""
    n = len(r)
    best_s, best_xy, best_r = float(r.sum()), xy.copy(), r.copy()
    ar = np.arange(n)
    c = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    bad = 0
    for _ in range(max_it):
        if time.process_time() > deadline or tau < 1e-10 or bad >= stall:
            break
        wall = _walls(xy)
        d = _pdist(xy)
        iu = np.triu_indices(n, 1)
        dv = d[iu]
        # Provably valid upper bound on the POST-step radius of circle i:
        #   wall row      ->  r_i <= wall_i + dx_i        <= wall_i + tau
        #   pair row (j)  ->  r_i <= d_ij + u.(dp_i-dp_j) - r_j <= d_ij + 2*tau   (since r_j >= 0)
        # so rcap_i = min(wall_i + tau, nn_i + 2*tau) with nn_i the nearest-neighbour distance.
        # A pair is provably slack after the step when d_ij - 2*tau > rcap_i + rcap_j -> dropped.
        # (The old bound used rcap_i = wall_i + tau alone; adding the nn term keeps every binding row
        # but cuts the LP by ~4x on the large n.  Verified row-for-row equal trajectories.)
        rcap = np.minimum(wall + tau, d.min(1) + 2.0 * tau)
        keep = dv < (rcap[iu[0]] + rcap[iu[1]] + 2.0 * tau)
        I, J, b = iu[0][keep], iu[1][keep], dv[keep]
        m = len(I)
        ux = (xy[I, 0] - xy[J, 0]) / b
        uy = (xy[I, 1] - xy[J, 1]) / b
        rows = [np.repeat(np.arange(m), 6)]
        cols = [np.stack([I, I + n, I + 2 * n, J, J + n, J + 2 * n], 1).ravel()]
        vals = [np.stack([-ux, -uy, np.ones(m), ux, uy, np.ones(m)], 1).ravel()]
        rhs = [b]
        for k, (sgn, off, base) in enumerate(((-1.0, 0, xy[:, 0] - LO), (1.0, 0, HI - xy[:, 0]),
                                              (-1.0, n, xy[:, 1] - LO), (1.0, n, HI - xy[:, 1]))):
            rows.append(np.repeat(m + k * n + ar, 2))
            cols.append(np.stack([ar + off, ar + 2 * n], 1).ravel())
            vals.append(np.stack([np.full(n, sgn), np.ones(n)], 1).ravel())
            rhs.append(base)
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(m + 4 * n, 3 * n)).tocsr()
        bounds = np.stack([np.concatenate([np.full(2 * n, -tau), np.zeros(n)]),
                           np.concatenate([np.full(2 * n, tau), np.full(n, 0.5)])], 1)
        try:
            res = linprog(c, A_ub=A, b_ub=np.concatenate(rhs) - MARGIN, bounds=bounds, method="highs")
        except (ValueError, MemoryError):
            res = None
        if res is None or not res.success or res.x is None:
            tau *= 0.5
            bad += 1
            continue
        z = res.x
        xy = xy + np.stack([z[:n], z[n:2 * n]], 1)
        np.clip(xy, LO, HI, out=xy)
        r = _repair(xy, z[2 * n:])
        s = float(r.sum())
        if s > best_s + 1e-14:
            best_s, best_xy, best_r, bad = s, xy.copy(), r.copy(), 0
        else:
            tau *= 0.6
            bad += 1
    return best_s, best_xy, best_r


# ----------------------------------------------------------------------------- which circles move

def _kick_movers(rng, xy, r, k, mode=0):
    """The indices of the k circles a kick teleports.

    mode 0 (INCUMBENT): the k smallest radii -- the loose fillers, so the expensive macro layout of the
        big circles survives the kick.  Bit-exactly the ``np.argsort(r)[:k]`` it replaces, and like it
        makes NO rng call, so rung 0 leaves the whole downstream stream untouched.
    mode 1: a contiguous SPATIAL cluster -- the smallest circle plus its k-1 nearest neighbours by centre
        distance -- so one local patch is rearranged as a unit instead of k scattered fillers.  Also
        makes no rng call.
    mode 2: the same cluster, anchored on a uniformly random circle instead of the smallest (one draw).
    """
    n = len(r)
    k = min(int(k), n)
    if mode <= 0:
        return np.argsort(r)[:k]
    a = int(np.argmin(r)) if mode == 1 else int(rng.randint(n))
    return np.argsort(((xy - xy[a]) ** 2).sum(1))[:k]


# ----------------------------------------------------------------------------- teleport destinations

def _kick_spots(rng, xy, r, moving, m, q=1):
    """Fresh spots for the circles indexed by ``moving``: for each, the EMPTIEST of ``m`` uniform draws.

    Clearance of a candidate point p is ``min(wall(p), min_j ||p - p_j|| - r_j)`` over the circles that
    are staying put, plus the spots already chosen in this same kick (so the k movers spread out instead
    of piling into one hole).  ``m == 1`` draws the identical numbers in the identical order as the plain
    uniform teleport it generalises, so the ladder has the proven incumbent as its bottom rung."""
    k = len(moving)
    cand = rng.uniform(LO + 0.02, HI - 0.02, size=(k, m, 2))
    if m <= 1:
        return cand[:, 0, :]
    keep = np.ones(len(r), dtype=bool)
    keep[moving] = False
    fx, fr = xy[keep], r[keep]
    out = np.empty((k, 2))
    for t in range(k):
        c = cand[t]
        clr = _walls(c)
        if len(fr):
            clr = np.minimum(clr, (np.sqrt(((c[:, None, :] - fx[None, :, :]) ** 2).sum(-1))
                                   - fr[None, :]).min(1))
        if t:
            clr = np.minimum(clr, np.sqrt(((c[:, None, :] - out[None, :t, :]) ** 2).sum(-1)).min(1))
        if q <= 1:
            out[t] = c[int(np.argmax(clr))]      # rung 1: no rng call, so the stream is untouched
        else:
            qq = min(int(q), len(clr))
            top = np.argpartition(-clr, qq - 1)[:qq]
            out[t] = c[int(top[rng.randint(qq)])]
    return out


# ----------------------------------------------------------------------------- starting points

def _read_pack(n):
    """Warm start from the committed census, or None.  GUARDED: a missing/odd pack is not an error."""
    path = os.path.join("bench", "packs", "csqv%d.pck" % n)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        rows = [ln.split() for ln in lines[2:]]
        a = np.array([[float(v) for v in row[:3]] for row in rows], dtype=float)
    except (OSError, ValueError, IndexError):
        return None
    if a.ndim != 2 or a.shape != (n, 3) or not np.isfinite(a).all():
        return None
    return a


def _records():
    """The read-only record table as ``{n: record}``, or ``{}``.  GUARDED: never raises."""
    try:
        with open(os.path.join("bench", "records.json")) as fh:
            recs = json.load(fh)["records"]
        return dict((int(k), float(v)) for k, v in recs.items() if float(v) > 0.0)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def _slice_weights(todo):
    """Relative CPU-slice weight per target: ``w = relgap ** SLICE_POW``, read off the committed census.

    ``SLICE_POW == 0`` SHORT-CIRCUITS to exactly ``1.0`` per n, which makes the slice arithmetic in
    ``solve()`` bit-identical to the flat ``(deadline - now) / left_n`` share it has used since
    iteration 1 -- rung 0 of this ladder is a provable no-op, and it makes no ``rng`` call at any rung.
    Every read is GUARDED: an n with no record, no pack, or an unusable pack cannot raise and is given
    the most generous known gap (it has the most to win, not the least), and if nothing at all can be
    read the flat share is returned.  The gap is floored at ``SLICE_GAP_FLOOR`` so no n gets weight 0.
    """
    w = np.ones(len(todo), dtype=float)
    if SLICE_POW == 0.0 or not len(todo):
        return w
    recs = _records()
    gaps = np.full(len(todo), np.nan)
    for i, n in enumerate(todo):
        rec = recs.get(int(n))
        if rec is None:
            continue
        a = _read_pack(int(n))
        if a is None:
            continue
        g = (rec - float(a[:, 2].sum())) / rec
        if np.isfinite(g):
            gaps[i] = min(max(g, SLICE_GAP_FLOOR), 1.0)
    known = np.isfinite(gaps)
    if not known.any():
        return w
    gaps[~known] = gaps[known].max()
    w = gaps ** float(SLICE_POW)
    if not np.all(np.isfinite(w)) or not np.all(w > 0.0):
        return np.ones(len(todo), dtype=float)
    return w


def _abandon(misses, has_next):
    """Should a slice end early after ``misses`` consecutive non-improving kicks?

    ``SLICE_ABANDON == 0`` short-circuits to exactly ``False`` for every argument, which is the
    pre-knob behaviour bit-for-bit.  ``has_next`` is False on the LAST target of a call: there is
    nowhere for the freed CPU to go, so the final slice is never abandoned.
    """
    return bool(SLICE_ABANDON) and bool(has_next) and misses >= SLICE_ABANDON


def _cold(rng, n):
    xy = rng.uniform(LO + 0.02, HI - 0.02, size=(n, 2))
    return xy, _repair(xy, np.full(n, 1e-9))


# ----------------------------------------------------------------------------- entry point

def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    deadline = time.process_time() + CPU_BUDGET          # computed at ENTRY -- see module docstring
    todo = sorted(set(int(n) for n in targets if int(n) >= 1))
    wts = _slice_weights(todo)                           # all exactly 1.0 at SLICE_POW = 0

    for idx_n, n in enumerate(todo):
        now = time.process_time()
        if now >= deadline or meter.left() <= 0:
            break
        w_rest = float(wts[idx_n:].sum())                # == len(todo) - idx_n when the weights are flat
        if not (w_rest > 0.0):
            w_rest = float(len(todo) - idx_n)
        # weighted share of what is actually LEFT; with w == 1 this is bit-exactly (deadline-now)/left_n
        slice_end = now + (deadline - now) * wts[idx_n] / w_rest

        best_s, best_xy, best_r = -np.inf, None, None

        def offer(s, xy, r):
            """Route a candidate through the metered oracle and keep the local incumbent.

            Returns True iff the candidate RAISED the incumbent -- step (2) counts the misses.
            """
            nonlocal best_s, best_xy, best_r
            if meter.left() <= 0:
                return False
            feas, val = evaluate(n, _pack(xy, r))
            if feas and float(val) > best_s:
                best_s, best_xy, best_r = float(val), xy.copy(), r.copy()
                return True
            return False

        # (0) warm start from the committed census, if there is one.
        warm = _read_pack(n)
        if warm is not None:
            wxy = np.clip(warm[:, :2], LO, HI)
            s, x2, r2 = _slp(wxy, _repair(wxy, warm[:, 2]), 0.05, min(slice_end, deadline),
                             stall=KICK_STALL)
            offer(s, x2, r2)

        # (1) cold random restarts -- needed only when there is no incumbent yet, so a fresh n never
        #     depends on the census.  With a warm pack the kick loop below is a strictly better use of
        #     the slice (measured iteration 2: cold restarts were the WEAKEST of four moves at every n).
        runs = 0
        while best_xy is None and runs < SEED_RUNS and time.process_time() < slice_end and meter.left() > 0:
            xy, r = _cold(rng, n)
            s, x2, r2 = _slp(xy, r, 0.05, min(slice_end, deadline), stall=COLD_STALL)
            offer(s, x2, r2)
            runs += 1

        # (2) TELEPORT kicks around the incumbent for whatever time is left in this n's slice.
        #     Move only the k SMALLEST circles to fresh random spots and re-converge: the macro layout
        #     of the big circles -- the expensive part -- survives, and only the fillers are reshuffled.
        #     Measured against the previous global-Gaussian basin hop at an equal 3 s slice per n:
        #       n=35 relgap 0.00220 vs 0.00275 | n=67 0.00692 vs 0.00832 | n=99 0.00500 vs 0.01352.
        #
        #     ABANDONMENT (iteration 14).  diag24 measured that 25.8% of every budget goes to n that
        #     returned ZERO improvements over six metered runs, so looseness (what SLICE_POW weights
        #     by) is the wrong proxy for responsiveness.  Count CONSECUTIVE kicks that fail to raise
        #     the incumbent and, once that reaches SLICE_ABANDON, end this n's slice: the time is
        #     handed to the n after it by the slice_end recomputation at the top of the loop.  Only
        #     ever when there IS a later target -- abandoning the final slice would simply discard the
        #     CPU.  SLICE_ABANDON == 0 disables the whole test, which is the pre-knob behaviour
        #     bit-for-bit (the rng stream is untouched either way).
        misses, has_next = 0, idx_n < len(todo) - 1
        while best_xy is not None and time.process_time() < slice_end and meter.left() > 0:
            k = 1 if n < 16 else int(rng.randint(1, max(2, n // KICK_KDIV)))
            idx = _kick_movers(rng, best_xy, best_r, k, MOVE_MODE)
            xy = best_xy.copy()
            xy[idx] = _kick_spots(rng, best_xy, best_r, idx, GAP_SAMPLES, SPOT_TOPQ)
            s, x2, r2 = _slp(xy, _repair(xy, best_r), KICK_TAU, min(slice_end, deadline),
                             stall=KICK_STALL)
            if offer(s, x2, r2):
                misses = 0
            else:
                misses += 1
                if _abandon(misses, has_next):
                    break


# ----------------------------------------------------------------------------- self-test

def _self_test():
    import sys

    class _Meter(object):
        def __init__(self, budget):
            self.budget, self.used = int(budget), 0

        def tick(self, k=1):
            g = max(0, min(int(k), self.budget - self.used))
            self.used += g
            return g

        def left(self):
            return max(0, self.budget - self.used)

    class _Ev(object):
        """Local re-implementation of the harness feasibility geometry (importing harness is banned)."""

        def __init__(self, meter, tol=1e-9):
            self.meter, self.tol, self.best, self.calls = meter, tol, {}, 0

        def evaluate(self, n, packing):
            self.calls += 1
            a = np.asarray(packing, dtype=float)
            single = a.ndim == 2
            if single:
                a = a[None]
            assert a.shape[1] == n and a.shape[2] == 3
            B = a.shape[0]
            grant = self.meter.tick(B)
            feas = np.zeros(B, dtype=bool)
            sr = np.full(B, -np.inf)
            if grant:
                x, y, r = a[:grant, :, 0], a[:grant, :, 1], a[:grant, :, 2]
                wall = np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min(1)
                dx = x[:, :, None] - x[:, None, :]
                dy = y[:, :, None] - y[:, None, :]
                slack = np.sqrt(dx * dx + dy * dy) - (r[:, :, None] + r[:, None, :])
                slack[:, np.eye(n, dtype=bool)] = np.inf
                pair = slack.reshape(grant, -1).min(1)
                ok = (r.min(1) > 0.0) & (wall >= -self.tol) & (pair >= -self.tol)
                feas[:grant], sr[:grant] = ok, r.sum(1)
                for i in range(grant):
                    if ok[i] and (n not in self.best or sr[i] > self.best[n][0]):
                        self.best[n] = (float(sr[i]), a[i].copy())
            return (bool(feas[0]), float(sr[0])) if single else (feas, sr)

    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # --- geometry: _repair always yields a strictly feasible pack, even from a wild overlap.
    rng = np.random.RandomState(7)
    for n in (1, 2, 5, 23):
        xy = rng.uniform(LO, HI, size=(n, 2))
        r = _repair(xy, np.full(n, 0.9))
        d = _pdist(xy)
        ok = (r > 0).all() and (_walls(xy) - r).min() >= -1e-9
        if n >= 2:
            ok = ok and (d - (r[:, None] + r[None, :])).min() >= -1e-9
        check(ok, "_repair feasible for n=%d" % n)

    # --- _slp must return a STRICTLY feasible pack (the tightened LP row filter must not break this),
    #     both with the generous cold-start runway and with the tight KICK_STALL tail cutoff.
    for n in (2, 9, 24):
        for stall in (COLD_STALL, 4, KICK_STALL):
            xy0, r0 = _cold(np.random.RandomState(n), n)
            s, x1, r1 = _slp(xy0, r0, 0.05, time.process_time() + 1.5, stall=stall)
            d = _pdist(x1)
            ok = ((r1 > 0).all() and (_walls(x1) - r1).min() >= -1e-9
                  and (d - (r1[:, None] + r1[None, :])).min() >= -1e-9
                  and abs(s - r1.sum()) < 1e-12 and s >= r0.sum() - 1e-12)
            check(ok, "_slp feasible and non-decreasing, n=%d stall=%d (sum=%.6f)" % (n, stall, s))

    # --- the tail cutoff must never LOSE ground: from the same start, stall=4 must return at least
    #     the objective the input had, and the full-tail run must not be worse than it either way
    #     (both are lower bounds on the same monotone ascent, so neither may return < input).
    xy0, r0 = _cold(np.random.RandomState(5), 18)
    lim = time.process_time() + 3.0
    sums = {}
    for stall in (KICK_STALL, 2, 4, SLP_MAX_IT):
        sums[stall], sx, sr = _slp(xy0.copy(), r0.copy(), 0.05, lim, stall=stall)
        d = _pdist(sx)
        feas = ((sr > 0).all() and (_walls(sx) - sr).min() >= -1e-9
                and (d - (sr[:, None] + sr[None, :])).min() >= -1e-9)
        check(feas and sums[stall] >= r0.sum() - 1e-12,
              "tail cutoff stall=%d: feasible and never below its input (%.6f vs in=%.6f)"
              % (stall, sums[stall], r0.sum()))
    # KICK_STALL is pinned at the FLOOR of the ladder: a cutoff of 0 would disable the ascent
    # entirely (the loop would break before its first LP), so guard the constant itself.
    check(KICK_STALL >= 1, "KICK_STALL is at least 1 (a cutoff of 0 would do no LP work at all)")
    s0, x0b, r0b = _slp(xy0.copy(), r0.copy(), 0.05, lim, stall=0)
    check(abs(s0 - r0.sum()) < 1e-12, "stall=0 degenerates to a no-op (why the floor is 1)")

    # --- TELEPORT DESTINATIONS.  The ladder is only a safe generalisation of the proven uniform kick
    #     if its bottom rung IS that kick: m=1 must draw the identical numbers in the identical order.
    for n, k in ((27, 1), (35, 4), (99, 12)):
        g1, g2 = np.random.RandomState(4), np.random.RandomState(4)
        xy = g1.uniform(LO, HI, size=(n, 2))
        r = np.abs(g1.normal(0.02, 0.005, n))
        mv = np.argsort(r)[:k]
        got = _kick_spots(g1, xy, r, mv, 1)
        g2.uniform(LO, HI, size=(n, 2)); np.abs(g2.normal(0.02, 0.005, n))
        want = g2.uniform(LO + 0.02, HI - 0.02, size=(k, 2))
        check(np.array_equal(got, want), "GAP_SAMPLES=1 is bit-exactly the uniform teleport (n=%d)" % n)
        # and for m>1: every spot is inside the box, and each is the ARGMAX of clearance over its own
        # m draws (recomputed here independently of the chooser's own bookkeeping).
        g3 = np.random.RandomState(11)
        spots = _kick_spots(g3, xy, r, mv, GAP_SAMPLES)
        g4 = np.random.RandomState(11)
        cand = g4.uniform(LO + 0.02, HI - 0.02, size=(k, GAP_SAMPLES, 2))
        keep = np.ones(n, dtype=bool); keep[mv] = False
        ok = spots.shape == (k, 2) and (spots >= LO).all() and (spots <= HI).all()
        for t in range(k):
            c = cand[t]
            clr = np.minimum(_walls(c), (np.sqrt(((c[:, None, :] - xy[keep][None, :, :]) ** 2).sum(-1))
                                         - r[keep][None, :]).min(1))
            if t:
                clr = np.minimum(clr, np.sqrt(((c[:, None, :] - spots[None, :t, :]) ** 2).sum(-1)).min(1))
            ok = ok and np.array_equal(spots[t], c[int(np.argmax(clr))])
        check(ok, "GAP_SAMPLES=%d picks the max-clearance draw, in-box, n=%d" % (GAP_SAMPLES, n))
    check(GAP_SAMPLES >= 1, "GAP_SAMPLES is at least 1 (0 draws would leave the movers nowhere to go)")

    # --- SPOT_TOPQ: the diversity rung.  q=1 must be a PROVABLE no-op -- same spots as the argmax path
    #     AND no rng call at all, so the whole downstream stream is untouched (measured negative,
    #     diag13: q=4/16 both lose the mean, so the default stays at the rung that changes nothing).
    for n, k in ((35, 4), (99, 12)):
        ga, gb = np.random.RandomState(7), np.random.RandomState(7)
        xy = ga.uniform(LO, HI, size=(n, 2))
        r = np.abs(ga.normal(0.02, 0.005, n))
        gb.uniform(LO, HI, size=(n, 2)); np.abs(gb.normal(0.02, 0.005, n))
        a = _kick_spots(ga, xy, r, np.argsort(r)[:k], GAP_SAMPLES, 1)
        b = _kick_spots(gb, xy, r, np.argsort(r)[:k], GAP_SAMPLES)
        check(np.array_equal(a, b), "SPOT_TOPQ=1 equals the plain argmax destination (n=%d)" % n)
        check(ga.tomaxint() == gb.tomaxint(), "SPOT_TOPQ=1 consumes no extra rng draw (n=%d)" % n)
        # q>1: every spot must still be in-box and be one of the top-q draws by clearance, recomputed
        # here from a replayed rng independently of the chooser.
        q = 4
        gc, gd = np.random.RandomState(13), np.random.RandomState(13)
        mv = np.argsort(r)[:k]
        spots = _kick_spots(gc, xy, r, mv, GAP_SAMPLES, q)
        cand = gd.uniform(LO + 0.02, HI - 0.02, size=(k, GAP_SAMPLES, 2))
        keep = np.ones(n, dtype=bool); keep[mv] = False
        ok = spots.shape == (k, 2) and (spots >= LO).all() and (spots <= HI).all()
        for t in range(k):
            c = cand[t]
            clr = np.minimum(_walls(c), (np.sqrt(((c[:, None, :] - xy[keep][None, :, :]) ** 2).sum(-1))
                                         - r[keep][None, :]).min(1))
            if t:
                clr = np.minimum(clr, np.sqrt(((c[:, None, :] - spots[None, :t, :]) ** 2).sum(-1)).min(1))
            top = set(np.argsort(-clr)[:q].tolist())
            ok = ok and any(np.array_equal(spots[t], c[i]) for i in top)
        check(ok, "SPOT_TOPQ=%d picks from the top-%d by clearance, in-box, n=%d" % (q, q, n))
    check(SPOT_TOPQ >= 1, "SPOT_TOPQ is at least 1")

    # --- WHICH CIRCLES MOVE.  MOVE_MODE=0 must be a PROVABLE no-op: bit-exactly the argsort(r)[:k] the
    #     kick loop used before the knob existed, AND no rng call, so no later draw can shift
    #     (measured negative, diag14: cluster modes 1/2 both lose the mean, so the default stays here).
    for n, k in ((27, 3), (35, 4), (99, 12)):
        g = np.random.RandomState(19)
        xy = g.uniform(LO, HI, size=(n, 2))
        r = np.abs(g.normal(0.02, 0.005, n))
        ga, gb = np.random.RandomState(23), np.random.RandomState(23)
        got = _kick_movers(ga, xy, r, k, 0)
        check(np.array_equal(got, np.argsort(r)[:k]),
              "MOVE_MODE=0 is bit-exactly the k smallest radii (n=%d)" % n)
        check(ga.tomaxint() == gb.tomaxint(), "MOVE_MODE=0 consumes no rng draw (n=%d)" % n)
        # mode 1: the smallest circle plus its k-1 nearest neighbours -- also rng-free, and recomputed
        # here independently of the selector.
        gc, gd = np.random.RandomState(29), np.random.RandomState(29)
        cl = _kick_movers(gc, xy, r, k, 1)
        want = np.argsort(((xy - xy[int(np.argmin(r))]) ** 2).sum(1))[:k]
        check(np.array_equal(cl, want) and int(np.argmin(r)) in set(cl.tolist()),
              "MOVE_MODE=1 is the smallest circle's k-nearest cluster (n=%d)" % n)
        check(gc.tomaxint() == gd.tomaxint(), "MOVE_MODE=1 consumes no rng draw (n=%d)" % n)
        # mode 2: same cluster around a uniformly random anchor -- exactly ONE draw, and the anchor is
        # the circle that draw names.
        ge, gf = np.random.RandomState(31), np.random.RandomState(31)
        cl2 = _kick_movers(ge, xy, r, k, 2)
        a = int(gf.randint(n))
        check(np.array_equal(cl2, np.argsort(((xy - xy[a]) ** 2).sum(1))[:k]),
              "MOVE_MODE=2 clusters around the drawn anchor (n=%d)" % n)
        check(ge.tomaxint() == gf.tomaxint(), "MOVE_MODE=2 consumes exactly one rng draw (n=%d)" % n)
        # every mode must return k DISTINCT in-range indices -- a repeat would send two circles to one
        # spot and a stale index would corrupt the pack.
        for mode in (0, 1, 2):
            s = _kick_movers(np.random.RandomState(3), xy, r, k, mode)
            check(len(set(s.tolist())) == k and s.min() >= 0 and s.max() < n,
                  "MOVE_MODE=%d returns %d distinct in-range indices (n=%d)" % (mode, k, n))
        # k > n must clamp rather than fabricate indices (solve() never does this, but the guard is free)
        check(len(_kick_movers(np.random.RandomState(3), xy, r, n + 5, 0)) == n,
              "_kick_movers clamps k to n (n=%d)" % n)
    check(MOVE_MODE in (0, 1, 2), "MOVE_MODE is a known rung")

    # --- missing pack must NOT raise (unguarded warm-start reads kill a whole solve() call).
    check(_read_pack(10 ** 6) is None, "_read_pack returns None for an absent n (guarded warm start)")

    # --- solve() called TWICE in ONE process: the second call must still deliver.
    global CPU_BUDGET
    saved, CPU_BUDGET = CPU_BUDGET, 4.0
    try:
        meter = _Meter(500000)
        ev = _Ev(meter)
        r0 = np.random.RandomState(0)
        solve(ev.evaluate, meter, r0, [13, 27])
        first = dict(ev.best)
        check(set(first) == {13, 27}, "call 1 produced a feasible pack for every target")
        # second call, SAME process, and with a SINGLE target (per-n split must survive len==1)
        solve(ev.evaluate, meter, r0, [27])
        check(27 in ev.best, "call 2 (single target, same process) still returned a feasible packing")
        check(ev.best[27][0] >= first.get(27, (0.0,))[0],
              "call 2 did not regress the incumbent for n=27")
        # a target with no committed pack must cold-start cleanly
        solve(ev.evaluate, meter, r0, [8])
        check(8 in ev.best, "cold start works for an n with no committed pack")
        # quality guard: SLP must get within 5% of the published record where we have one.
        try:
            recs = json.load(open(os.path.join("bench", "records.json")))["records"]
        except OSError:
            recs = {}
        if "27" in recs:
            gap = (recs["27"] - ev.best[27][0]) / recs["27"]
            check(gap < 0.05, "n=27 within 5%% of record (relgap=%.4f)" % gap)
    finally:
        CPU_BUDGET = saved

    # --- CPU HEADROOM: the budget must stay strictly under the driver's SIGPROF backstop by a margin
    #     large enough to absorb the overshoot, and the OVERSHOOT ITSELF must be measured, not assumed.
    #     (Overrunning the backstop does not lose the harvest, but it does make the run irreproducible.)
    check(0.0 < CPU_BUDGET < DRIVER_BACKSTOP_S and CPU_MARGIN_S >= 5.0,
          "CPU_BUDGET %.1f leaves a %.1f s margin under the %.1f s backstop"
          % (CPU_BUDGET, CPU_MARGIN_S, DRIVER_BACKSTOP_S))
    saved3, CPU_BUDGET = CPU_BUDGET, 3.0
    try:
        m3 = _Meter(500000)
        ev3 = _Ev(m3)
        t3 = time.process_time()
        solve(ev3.evaluate, m3, np.random.RandomState(1), [99])   # the largest visible n = costliest LP
        over = time.process_time() - t3 - CPU_BUDGET
    finally:
        CPU_BUDGET = saved3
    check(over < saved3 and over < CPU_MARGIN_S,
          "solve() overshoots its CPU_BUDGET by %.3f s at n=99, well inside the %.1f s margin"
          % (over, CPU_MARGIN_S))

    # --- SLICE ALLOCATION (iteration 9).  Rung 0 must stay a provable no-op, and every read guarded.
    global SLICE_POW
    sp_saved = SLICE_POW
    try:
        SLICE_POW = 0.0
        for tl in ([27, 29, 99], list(range(27, 100, 2)), [11], [4, 6, 123456]):
            w0 = _slice_weights(tl)
            check(w0.shape == (len(tl),) and np.all(w0 == 1.0),
                  "SLICE_POW=0 gives exactly the flat weight for %d target(s)" % len(tl))
        # the arithmetic identity the no-op rests on: (d * 1.0) / L is bit-exact with d / L.
        bad = [(d, L) for d in (114.0, 113.999999, 1e-3, 27.7, 0.0)
               for L in range(1, 38)
               if (d * np.float64(1.0)) / np.float64(L) != d / L]
        check(not bad, "flat slice arithmetic is bit-exact against the pre-knob (deadline-now)/left_n")

        SLICE_POW = 1.0
        vis = list(range(27, 100, 2))
        w = _slice_weights(vis)
        check(np.all(np.isfinite(w)) and np.all(w > 0.0),
              "adopted rung: every visible n gets a finite, strictly positive weight (min %.2e)"
              % w.min())
        # weights must track the gap they are derived from, recomputed here independently.
        recs = _records()
        gap = {}
        for n in vis:
            a = _read_pack(n)
            if a is not None and n in recs:
                gap[n] = min(max((recs[n] - float(a[:, 2].sum())) / recs[n], SLICE_GAP_FLOOR), 1.0)
        check(len(gap) == len(vis), "every visible n has a readable record and pack (%d)" % len(gap))
        order_ok = all((gap[vis[i]] > gap[vis[j]]) == (w[i] > w[j])
                       for i in range(len(vis)) for j in range(len(vis)))
        check(order_ok, "weight order matches relgap order at pow=1 (loosest n gets the most CPU)")
        capped = [n for n in vis if gap[n] <= SLICE_GAP_FLOOR]
        check(all(w[vis.index(n)] <= w.min() * (1.0 + 1e-12) for n in capped),
              "the %d already-capped n get the minimum weight, not an equal share" % len(capped))
        # GUARDED: n with no pack / no record must not raise and must not be starved.
        wm = _slice_weights([4, 6, 123456])
        check(wm.shape == (3,) and np.all(np.isfinite(wm)) and np.all(wm > 0.0),
              "unknown n: no exception, all weights finite and positive")
        wx = _slice_weights([29, 4])
        check(np.isfinite(wx).all() and wx[1] >= wx.max() - 1e-15,
              "an n with no pack gets the MOST generous weight, never zero")
        # and solve() itself must still work at the adopted rung for the awkward target sets.
        cb_saved, globals()["CPU_BUDGET"] = CPU_BUDGET, 2.0
        try:
            for tl in ([35], [12]):
                mS = _Meter(500000)
                eS = _Ev(mS)
                solve(eS.evaluate, mS, np.random.RandomState(7), list(tl))
                check(tl[0] in eS.best and eS.best[tl[0]][0] > 0.0,
                      "pow=1 + targets=%s: a feasible packing is still produced" % tl)
        finally:
            globals()["CPU_BUDGET"] = cb_saved
    finally:
        SLICE_POW = sp_saved

    # --- KICK TRUST REGION (iteration 10).  The adopted rung must stay sane and stay the step size.
    check(isinstance(KICK_TAU, float) and np.isfinite(KICK_TAU) and 0.0 < KICK_TAU <= 0.5,
          "KICK_TAU=%g is finite, positive and no larger than the square it moves inside" % KICK_TAU)
    # the property the ladder rests on: with KICK_STALL==1 the 0.6 shrink cannot compound, so the
    # initial tau IS the step size.  Counting LP iterations proves it rather than asserting it.
    check(KICK_STALL == 1, "KICK_STALL is at its floor, so a kick's tau is shrunk at most once")
    for n_t in (27, 35):
        rngT = np.random.RandomState(5)
        xyT = rngT.uniform(LO + 0.05, HI - 0.05, size=(n_t, 2))
        rT = _repair(xyT, np.full(n_t, 0.05))
        s_in = float(rT.sum())
        seen = []
        for tau_t in (KICK_TAU, 0.03, 0.06, 0.0075):
            sT, xT, rrT = _slp(xyT.copy(), rT.copy(), tau_t, time.process_time() + 5.0,
                               stall=KICK_STALL)
            dT = _pdist(xT)
            feasT = ((rrT > 0.0).all() and (_walls(xT) - rrT).min() >= -1e-9
                     and (dT - (rrT[:, None] + rrT[None, :])).min() >= -1e-9)
            check(feasT and sT >= s_in - 1e-15 and abs(sT - float(rrT.sum())) < 1e-12,
                  "n=%d tau=%g: _slp returns a strictly feasible pack, never below its input "
                  "(%.6f -> %.6f)" % (n_t, tau_t, s_in, sT))
            seen.append(sT)
        check(len(seen) == 4 and all(np.isfinite(v) for v in seen),
              "n=%d: all four rungs of the tau ladder ran and returned finite sums" % n_t)
    # a kick at the adopted tau must still be a real kick: solve() produces a feasible pack for a
    # single target, for a fresh n with no pack, and twice in one process.
    cb_saved2, globals()["CPU_BUDGET"] = CPU_BUDGET, 2.0
    try:
        mK = _Meter(500000)
        eK = _Ev(mK)
        for tl in ([35], [12]):
            solve(eK.evaluate, mK, np.random.RandomState(11), list(tl))
        check(35 in eK.best and 12 in eK.best and eK.best[35][0] > 0.0 and eK.best[12][0] > 0.0,
              "KICK_TAU=%g: solve() twice in one process still packs both a census n and a fresh n"
              % KICK_TAU)
    finally:
        globals()["CPU_BUDGET"] = cb_saved2

    # --- KICK_KDIV (iteration 11): the kick SIZE.  Pin the knob rather than assert it in prose.
    check(isinstance(KICK_KDIV, int) and KICK_KDIV >= 1,
          "KICK_KDIV is a positive integer (%r)" % (KICK_KDIV,))
    vis = list(range(27, 100, 2))
    # (i) rung 8 is the PROVABLE no-op: the bound is bit-exactly the pre-knob max(2, n // 8) at every
    #     visible n, so adopting a different rung is the only thing that can change the k stream.
    _bound = lambda nn, kd: max(2, nn // kd)          # the expression the kick loop evaluates
    check(all(_bound(nn, 8) == max(2, nn // 8) for nn in vis) and _bound(99, 8) == 12,
          "KICK_KDIV=8 reproduces the pre-knob bound max(2, n//8) at every visible n")
    # (ii) the k the kick loop draws must be a legal mover count at EVERY n solve() can be handed --
    #      at least 1, and strictly fewer than n, so a kick never teleports the whole pack.
    bad_k = []
    for n in list(range(16, 128)) + vis:
        hi = max(2, n // KICK_KDIV)
        g = np.random.RandomState(n)
        for _ in range(64):
            k = 1 if n < 16 else int(g.randint(1, hi))
            if not (1 <= k < n):
                bad_k.append((n, k))
    check(not bad_k, "KICK_KDIV=%d: drawn k is always in [1, n) for n=16..127 (%d bad)"
          % (KICK_KDIV, len(bad_k)))
    # (iii) the adopted rung must actually BE a change -- a divisor that silently reproduced rung 8
    #       everywhere would make the diag20/21 ladder meaningless.
    check(any(max(2, n // KICK_KDIV) != max(2, n // 8) for n in vis) or KICK_KDIV == 8,
          "KICK_KDIV=%d differs from the rung-8 default on at least one visible n" % KICK_KDIV)
    # (iii-b) iteration 12: the docstring's claim that rung 64 IS the k=1 limit must be a property of
    #         the code, not prose -- at every visible n the bound collapses to 2, so randint(1,2) == 1.
    check(all(max(2, nn // 64) == 2 for nn in vis),
          "KICK_KDIV=64 is the k=1 limit: max(2, n//64) == 2 at every visible n")
    lim = set()
    for nn in vis:
        g = np.random.RandomState(nn)
        for _ in range(32):
            lim.add(int(g.randint(1, max(2, nn // 64))))
    check(lim == {1}, "KICK_KDIV=64 draws k == 1 and only 1 over the visible n (drew %s)"
          % sorted(lim))
    # (iii-c) the adopted rung must sit strictly INSIDE the measured bracket (4/2 below, 64 above),
    #         so a later edit cannot silently push it past a rung diag20/diag22 already refuted.
    check(4 < KICK_KDIV < 64,
          "KICK_KDIV=%d is strictly inside the bracket the ladder measured (4 .. 64)" % KICK_KDIV)
    # (iv) every laddered rung must still yield a strictly feasible kick+re-converge at two sizes.
    for n_k in (27, 85):
        pk = _read_pack(n_k)
        if pk is None:
            continue
        for kd in (2, 4, 8, 16, 32, 64):
            g = np.random.RandomState(7)
            xy0 = np.clip(pk[:, :2], LO, HI)
            r0 = _repair(xy0, pk[:, 2])
            k = 1 if n_k < 16 else int(g.randint(1, max(2, n_k // kd)))
            mv = _kick_movers(g, xy0, r0, k, MOVE_MODE)
            xy1 = xy0.copy()
            xy1[mv] = _kick_spots(g, xy0, r0, mv, GAP_SAMPLES, SPOT_TOPQ)
            sK, xK, rK = _slp(xy1, _repair(xy1, r0), KICK_TAU, time.process_time() + 2.0,
                              stall=KICK_STALL)
            okK = ((rK > 0).all() and (_walls(xK) - rK).min() >= -1e-9
                   and (_pdist(xK) - (rK[:, None] + rK[None, :])).min() >= -1e-9
                   and abs(sK - rK.sum()) < 1e-12 and np.isfinite(sK))
            check(okK, "n=%d kdiv=%d: a kick re-converges to a strictly feasible pack (k=%d)"
                  % (n_k, kd, k))
    # (v) at the ADOPTED rung, solve() twice in one process still packs a census n and a fresh n.
    cb3, globals()["CPU_BUDGET"] = CPU_BUDGET, 2.0
    try:
        mD = _Meter(500000)
        eD = _Ev(mD)
        for tl in ([45], [14]):
            solve(eD.evaluate, mD, np.random.RandomState(5), list(tl))
        check(45 in eD.best and 14 in eD.best and eD.best[45][0] > 0.0 and eD.best[14][0] > 0.0,
              "KICK_KDIV=%d: solve() twice in one process still packs a census n and a fresh n"
              % KICK_KDIV)
    finally:
        globals()["CPU_BUDGET"] = cb3

    # --- iteration 13: the step-(0) warm-start SLP pass, and the slice weights, pinned to what
    #     artifacts/diag23.py and artifacts/diag24.py measured.
    # (a) reading a committed pack and _repair-ing it may only give up float-rounding-sized sum: the
    #     repair subtracts 1e-12 per circle, so the warm incumbent is never measurably below what is
    #     committed.  If this ever grew, a warm start would silently regress the census.
    # (b) on a converged pack the step-(0) SLP is a NUMERICAL NO-OP -- it recovers exactly that repair
    #     loss and moves no centre.  diag23 measured centre motion <= 6.9e-13 and a net sum change of
    #     ~1e-13 at all 37 visible n, at a cost of 1.61 s = 1.4% of the budget.  That 1.4% is why the
    #     pass is KEPT (skipping it is below the ~0.02-digit noise floor either way) and this check is
    #     what would catch the pass turning into something that actually moves the pack.
    for n_w in (27, 45, 99):
        pw = _read_pack(n_w)
        if pw is None:
            continue
        xw = np.clip(pw[:, :2], LO, HI)
        raw = float(pw[:, 2].sum())
        rw = _repair(xw, pw[:, 2])
        check(0.0 <= raw - float(rw.sum()) <= 2e-12 * n_w,
              "n=%d: _repair of the committed pack costs at most 2e-12 per circle (%.2e)"
              % (n_w, raw - float(rw.sum())))
        sW, xW, rW = _slp(xw, rw, 0.05, time.process_time() + 5.0, stall=KICK_STALL)
        okW = ((rW > 0).all() and (_walls(xW) - rW).min() >= -1e-9
               and (_pdist(xW) - (rW[:, None] + rW[None, :])).min() >= -1e-9)
        check(okW, "n=%d: the step-(0) warm pass returns a strictly feasible pack" % n_w)
        check(sW >= float(rw.sum()) - 1e-15 and abs(sW - raw) < 1e-9,
              "n=%d: the step-(0) warm pass is a no-op on a converged pack (net %+.2e)"
              % (n_w, sW - raw))
        check(float(np.abs(xW - xw).max()) < 1e-6,
              "n=%d: the step-(0) warm pass moves no centre (max |dx| = %.2e)"
              % (n_w, float(np.abs(xW - xw).max())))
    # (c) the slice weights must be a strictly MONOTONE function of relgap, and must floor a closed
    #     gap rather than zero it.  diag24's reading -- 25.8% of every budget goes to n that have not
    #     improved in six runs -- is a statement about the CENSUS, not about a weighting bug, and it
    #     is only meaningful while this monotonicity holds.
    recs_w = _records()
    vis_w = sorted(recs_w)
    if vis_w:
        ww = _slice_weights(vis_w)
        gg = []
        for n_g in vis_w:
            ag = _read_pack(n_g)
            g = 1.0 if ag is None else (recs_w[n_g] - float(ag[:, 2].sum())) / recs_w[n_g]
            gg.append(min(max(g, SLICE_GAP_FLOOR), 1.0))
        order_w = [vis_w[i] for i in np.argsort(np.asarray(ww), kind="stable")]
        order_g = [vis_w[i] for i in np.argsort(np.asarray(gg), kind="stable")]
        check(order_w == order_g,
              "_slice_weights ranks the visible n exactly by relgap (SLICE_POW=%g)" % SLICE_POW)
        capped = [i for i, g in enumerate(gg) if g <= SLICE_GAP_FLOOR]
        check(all(ww[i] > 0.0 for i in capped) and all(w > 0.0 for w in ww),
              "_slice_weights floors a closed gap to a positive weight, never 0 (%d capped n)"
              % len(capped))
        check(float(np.abs(np.asarray(ww) - np.asarray(gg) ** SLICE_POW).max()) < 1e-15,
              "_slice_weights is exactly relgap**SLICE_POW elementwise (SLICE_POW=%g)" % SLICE_POW)

    # --- SLICE ABANDONMENT (iteration 14).  diag24 located 25.8% of every budget in n that return
    #     nothing; _abandon is the reallocation test, and rung 0 must be a PROVABLE no-op.
    check(isinstance(SLICE_ABANDON, int) and not isinstance(SLICE_ABANDON, bool)
          and SLICE_ABANDON >= 0,
          "SLICE_ABANDON is a non-negative int (%r)" % (SLICE_ABANDON,))
    ab_saved = SLICE_ABANDON
    try:
        globals()["SLICE_ABANDON"] = 0
        check(not any(_abandon(m, h) for m in range(0, 2001) for h in (True, False)),
              "rung 0 is a provable no-op: _abandon is False for 0..2000 misses, both has_next")
        for rung in (1, 8, 32):
            globals()["SLICE_ABANDON"] = rung
            check(not any(_abandon(m, True) for m in range(0, rung)),
                  "rung %d: never abandons before %d consecutive misses" % (rung, rung))
            check(all(_abandon(m, True) for m in range(rung, rung + 50)),
                  "rung %d: abandons at %d misses and stays abandoned" % (rung, rung))
            check(not any(_abandon(m, False) for m in range(0, rung + 50)),
                  "rung %d: the LAST slice is never abandoned (no later n to take the CPU)" % rung)
            seq = [_abandon(m, True) for m in range(0, rung + 50)]
            check(all(b <= c for b, c in zip(seq, seq[1:])),
                  "rung %d: _abandon is monotone in the miss count" % rung)
        # the knob must survive the awkward target sets the MISSION names, at its most aggressive
        # rung: one target (has_next is False throughout), a fresh n with no pack, and TWO calls in
        # one process.  An abandoned slice may never leave an n unpacked or below its warm start.
        globals()["SLICE_ABANDON"] = 1
        cbA, globals()["CPU_BUDGET"] = CPU_BUDGET, 2.0
        try:
            mA = _Meter(500000)
            eA = _Ev(mA)
            for tl in ([35], [12], [33, 45, 71]):
                solve(eA.evaluate, mA, np.random.RandomState(19), list(tl))
            check(all(t in eA.best and eA.best[t][0] > 0.0 for t in (35, 12, 33, 45, 71)),
                  "SLICE_ABANDON=1: solve() called 3x in one process packs every target "
                  "(single n, fresh n, and a 3-n set)")
            for n_a in (35, 33, 45, 71):
                pa = _read_pack(n_a)
                if pa is None:
                    continue
                check(eA.best[n_a][0] >= float(pa[:, 2].sum()) - 1e-9,
                      "n=%d: an abandoned slice never returns below the committed warm start "
                      "(%.9f >= %.9f)" % (n_a, eA.best[n_a][0], float(pa[:, 2].sum())))
                aa = eA.best[n_a][1]
                ra = aa[:, 2]
                check((ra > 0).all() and (_walls(aa[:, :2]) - ra).min() >= -1e-9
                      and (_pdist(aa[:, :2]) - (ra[:, None] + ra[None, :])).min() >= -1e-9,
                      "n=%d: the pack kept under abandonment is strictly feasible" % n_a)
            # a SINGLE target can never be abandoned, so it must still spend a real slice: with a 2 s
            # budget the kick loop runs many times more than the one call rung 1 would allow.
            mB = _Meter(500000)
            eB = _Ev(mB)
            solve(eB.evaluate, mB, np.random.RandomState(21), [35])
            check(eB.calls > 5,
                  "SLICE_ABANDON=1 with len(targets)==1: the slice still runs (%d evaluations, "
                  "not 2)" % eB.calls)
        finally:
            globals()["CPU_BUDGET"] = cbA
    finally:
        globals()["SLICE_ABANDON"] = ab_saved

    # --- GENERALIZATION off the census (iteration 15).  MISSION re-runs this solver OFFLINE on sizes
    #     it never scored in-run, and every diag 1-26 measured it only on VISIBLE n -- i.e. only on n
    #     that HAVE a committed warm pack.  diag27 measures the other path (no pack -> _read_pack
    #     returns None -> _cold + SEED_RUNS carry the whole run) and scores it against a reference
    #     computable for ANY n: the ceil(sqrt(n))^2 equal-circle grid, sum_r = n / (2*ceil(sqrt(n))).
    #     A memorising or degenerate solver cannot clear it; the search does, at every size measured.
    def _grid_ref(n):
        k = int(math.ceil(math.sqrt(n)))
        return n / (2.0 * k)

    check(abs(_grid_ref(9) - 9 / 6.0) < 1e-12 and abs(_grid_ref(28) - 28 / 12.0) < 1e-12,
          "_grid_ref is the exact ceil(sqrt(n))-grid sum_r (a valid feasible packing for any n)")
    gen_sizes = [n_g for n_g in (28, 101) if not os.path.exists("bench/packs/csqv%d.pck" % n_g)]
    check(len(gen_sizes) == 2, "the generalization sizes 28/101 are genuinely pack-free (cold path)")
    if gen_sizes:
        cbG, globals()["CPU_BUDGET"] = CPU_BUDGET, 4.0
        try:
            mG = _Meter(500000)
            eG = _Ev(mG)
            solve(eG.evaluate, mG, np.random.RandomState(29), list(gen_sizes))
        finally:
            globals()["CPU_BUDGET"] = cbG
        for n_g in gen_sizes:
            check(n_g in eG.best, "cold n=%d (no pack): solve() returns a packing" % n_g)
            if n_g not in eG.best:
                continue
            s_g, a_g = eG.best[n_g]
            r_g = a_g[:, 2]
            check(a_g.shape == (n_g, 3) and (r_g > 0).all()
                  and (_walls(a_g[:, :2]) - r_g).min() >= -1e-9
                  and (_pdist(a_g[:, :2]) - (r_g[:, None] + r_g[None, :])).min() >= -1e-9,
                  "cold n=%d: the returned packing is strictly feasible (walls and pairs "
                  "re-derived here)" % n_g)
            check(s_g > _grid_ref(n_g),
                  "cold n=%d: sum_r %.6f beats the trivial grid reference %.6f (ratio %.3f) -- the "
                  "solver SEARCHES on sizes it has never seen" % (n_g, s_g, _grid_ref(n_g),
                                                                  s_g / _grid_ref(n_g)))

    # --- an exhausted meter must not raise and must not fabricate anything.
    dead = _Meter(0)
    ev2 = _Ev(dead)
    CPU_BUDGET, saved2 = 2.0, CPU_BUDGET
    try:
        solve(ev2.evaluate, dead, np.random.RandomState(3), [11])
    finally:
        CPU_BUDGET = saved2
    check(ev2.best == {}, "exhausted budget: nothing tracked, no exception")

    print("SELF-TEST: %s (%d failure(s))" % ("PASS" if not fails else "FAIL", len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
