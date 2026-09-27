"""Circle packing (packomania csqv): maximize the sum of radii of n circles in the unit square.

Method -- SLP (sequential linear programming) with a conservative linearization, wrapped in a
basin-hopping restart loop.

The problem is

    max  sum_i r_i   s.t.   r_i + r_j <= ||p_i - p_j||   and   r_i <= wall_i(p_i),  r_i > 0.

The wall constraints are ALREADY linear in (x, y, r): x_i + r_i <= 0.5, -x_i + r_i <= 0.5, ...
The pair constraints are the only nonlinear ones, and ||.|| is CONVEX, so its first-order Taylor
expansion is a global UNDER-estimate:

    ||p_i - p_j|| >= d0 + u . (dp_i - dp_j),     u = (p_i - p_j)/d0.

Replacing the true distance by that under-estimate therefore gives an INNER (conservative)
approximation: any point feasible for the linearized program is feasible for the true one. So every
SLP iterate is genuinely feasible and the objective improves monotonically -- no penalty weights, no
constraint-violation bookkeeping. A trust region |dx|,|dy|,|dr| <= t keeps the linearization honest;
it grows on success and shrinks on stall, which is what drives the last digits.

Constraint pruning is exact rather than heuristic: inside a trust region of size t a pair's slack
d_ij - r_i - r_j can shrink by at most 4t (2t from the two centres moving, 2t from the two radii
growing), so pairs with slack > 4t can be DROPPED with no loss of feasibility. At small t only the
~6 contacts per circle survive, so the LPs stay tiny and fast all the way down.

Globalization: SLP only finds a local optimum, so the outer loop restarts. The primary start family
is the EXACT-COUNT ROW FAMILY (`_row_patterns` / `_row_start`): every (rows, a, b) whose alternating
row counts sum to exactly n, each polished by SLP and offered. Landing SLP in the right contact
topology is worth far more than polishing the wrong one -- measured on the committed census, one
member jumped n=45 from 2.22 to 4.06 digits and n=33 from 2.02 to 3.74, while 3 CPU-s of basin hops
on an already-converged incumbent moved n=99 by 0.007. The fallback family is a JITTERED LATTICE
(square or hex, row/column counts sampled around sqrt(n)) rather than
uniformly at random: near-optimal csqv packings are lattice-like over most of the square, and SLP,
being local, inherits whatever cell topology its start has. Measured cold at 4 s/size, lattice starts
beat uniform ones at every n tried (n=75: 4.5185 vs 4.4546; n=99: 5.1856 vs 5.1575). The third move is TRANSFER (`_transfer` + `_sources`): re-seed n from the committed pack of another
size, dropping its smallest circles or filling its gaps -- sources ranked by their own measured
digits minus a distance penalty, so the best packs in the census reach sizes up to 12 counts away. The last two moves both perturb the
INCUMBENT: "relocate the worst circles" (`_relocate` -- the smallest circles are the ones trapped in
bad cells; drop them into the emptiest spots and re-polish) and "shake" (`_shake` -- jitter every
centre a little and let _repair shrink the overlaps, so the whole contact graph can re-form). They
are opposite ends of the same axis, a big change to few circles vs a small change to all, and they
win different n; shake is what moved n=99 past its record.

Budgeting: the CPU cap -- not the evaluation count -- is the binding meter here (one SLP polish is
tens of LPs, one evaluate() call). Every deadline is derived AT ENTRY from time.process_time() and
from len(targets), so solve() is safe to call repeatedly in one process and safe with a single n.
The per-n schedule is CUMULATIVE and absolute (deadline_k = entry + total * cum_w[k]/sum_w): an n
that stops early donates the remainder to the n after it, and one that overruns cannot borrow from
them, so the tail of `targets` can never be starved. A final pass spends anything left over on
whichever n is still furthest from its record.

Warm starts from bench/packs/ are guarded: any n without a committed pack simply cold-starts.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

LO, HI = -0.5, 0.5

# --- CPU budgeting -------------------------------------------------------------------------------
# In-run the driver drives all 37 visible n under a 120 s process-CPU backstop; offline the operator
# re-runs one size per call under 60 s. Both are covered by "min(whole-call cap, per-size cap x sizes)".
CALL_CPU_CAP = 112.0        # leave headroom under the 120 s in-run backstop
PER_SIZE_CPU = 56.0         # leave headroom under the 60 s offline per-size cap
DIGITS_CAP = 7.0            # the scorer clamps -log10(relgap) here: relgap <= 1e-7 already scores full
CAP_GAP = 10.0 ** (-DIGITS_CAP)

# --- SLP early exit ("screen shallow, deepen the winner") -----------------------------------------
# artifacts/slpdepth.py traced the sum-vs-LP-index curve of 39 polishes of FRESH starts: every one
# banked its entire final sum in the first 3-14 LPs of 24-36, then spent ~65% of its CPU shrinking
# the trust region from ~1e-3 to the 1e-11 exit floor for a gain of 0. artifacts/stallrule.py then
# swept the stopping rule over 42 more traces on a different seed: stopping after SCREEN_STALL
# consecutive LPs that each gained <= SCREEN_GAINTOL keeps 0.53 of the CPU (x1.89 polishes/second)
# and loses at most 1.3e-12 of sum -- 0 of 42 traces lost even 1e-9. (A `t >= tmin` floor was the
# obvious alternative and is far worse: only x1.3-1.6, because t decays geometrically and the LPs it
# cuts are the cheap small-t ones.) The screen is used for RANKING starts; anything that comes back
# within DEEP_MARGIN of the incumbent is re-polished at FULL depth before it is offered, so the loss
# never reaches a committed pack.
SCREEN_STALL = 3            # consecutive non-gaining LPs that end a screening polish
SCREEN_GAINTOL = 1e-11      # a per-LP gain at or below this counts as "no gain"
DEEP_MARGIN = 1e-9          # re-polish at full depth when the screen lands this close to the best
DEEP_CPU_S = 1.5            # ceiling for that re-polish (a full polish is <= 0.65 s at n=99)


# --- feasibility geometry (re-implemented locally; importing bench/harness.py is forbidden) --------
def _pairdist(xy):
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d


def _wall(xy):
    return np.minimum.reduce([xy[:, 0] - LO, HI - xy[:, 0], xy[:, 1] - LO, HI - xy[:, 1]])


def _repair(xy, r):
    """Shrink radii (never move centres) until the packing is STRICTLY feasible.

    Pure shrinking, so it can only lose objective -- and it loses ~1e-12 in practice, because the LP
    already returns a point that violates nothing by more than its own solver tolerance."""
    r = np.asarray(r, float).copy()
    wall = _wall(xy)
    d = _pairdist(xy)
    for _ in range(8):
        r = np.minimum(r, wall)
        r = np.minimum(r, (d - r[None, :]).min(axis=1))
        v = max(0.0, float(np.max(r[:, None] + r[None, :] - d)), float(np.max(r - wall)))
        if v <= 0.0:
            break
        r -= 0.5 * v + 1e-15
    r = np.maximum(r - 1e-12, 1e-9)
    return r


def _sum_ok(xy, r):
    """(strictly_feasible, sum_r) -- the same test the scorer applies, computed locally."""
    if not (np.all(np.isfinite(xy)) and np.all(np.isfinite(r))) or np.any(r <= 0):
        return False, -np.inf
    if float(np.min(_wall(xy) - r)) < -1e-9:
        return False, -np.inf
    d = _pairdist(xy)
    if float(np.min(d - r[:, None] - r[None, :])) < -1e-9:
        return False, -np.inf
    return True, float(r.sum())


def _pack(xy, r):
    return np.concatenate([xy, r[:, None]], axis=1)


def _wall_rows(xy, r, t):
    """The LP's wall block, PRUNED by the same exact trust-region bound the pair block uses.

    The four rows of circle i read  +-dx_i + dr_i <= slack  with slack = the gap from that wall.
    Inside the trust region |dx_i| <= t and dr_i <= t, so the LHS can never exceed 2t: a row whose
    slack exceeds 2t is non-binding at EVERY point of the region and dropping it leaves the LP
    optimum identical (artifacts/wallprune_ab.py: |dsum| <= 5.5e-13 vs the unpruned solver over 15
    polishes, i.e. LP tolerance). It is worth doing because the block was never pruned before:
    measured (artifacts/hopcost.py) it was 53-61% of all rows while the pair block was already
    pruned, since nearly every wall row belongs to an interior circle ~0.3 from the nearest wall
    against t <= 0.05. Returns (row, col, val, b) for a COO matrix of len(b) rows.

    Block order matches the b vector [HI-x-r, HI+x-r, HI-y-r, HI+y-r], so global row g = block*n + i.
    """
    n = len(r)
    wall_all = np.concatenate([HI - xy[:, 0] - r, HI + xy[:, 0] - r,
                               HI - xy[:, 1] - r, HI + xy[:, 1] - r])
    widx = np.flatnonzero(wall_all < 2.0 * t + 1e-12)
    nw = len(widx)
    kk = widx // n                      # which of the 4 wall blocks the kept row came from
    ii = widx - kk * n                  # which circle
    wr = np.repeat(np.arange(nw), 2)
    wc = np.empty(2 * nw, dtype=np.int64)
    wv = np.empty(2 * nw)
    wc[0::2] = ii + n * (kk // 2)       # blocks 0,1 -> dx_i ; blocks 2,3 -> dy_i
    wv[0::2] = np.where(kk % 2 == 0, 1.0, -1.0)
    wc[1::2] = ii + 2 * n               # dr_i, coefficient +1 in all four blocks
    wv[1::2] = 1.0
    return wr, wc, wv, wall_all[widx]


_SLP_STATS = {"lps": 0}        # LPs spent by the most recent _slp call (diagnostic/self-test)


# --- the LP call: HiGHS directly, not through scipy.optimize.linprog ------------------------------
# artifacts/lpsplit.py profiled a polish after the row- and LP-pruning of iterations 12/13 and found
# the remaining cost is NOT inside HiGHS: `_clean_inputs`, `_parse_linprog`, `_format_A_constraints`
# and the sparse `vstack` scipy runs on EVERY linprog() call together outweigh the pivoting, because
# these LPs are tiny (<= 3n = 297 columns, a few hundred pruned rows) and are re-solved ~10 times per
# polish. scipy re-validates and re-marshals the model each time; HiGHS itself does not need any of
# it. Building the HighsLp by hand and calling run() is the IDENTICAL optimization problem -- same
# matrix, same bounds, same default dual-simplex -- so it returns the same vertex, not an
# approximation of it: artifacts/lpdirect_ab.py checks the sums agree to 0.0 over real polishes.
# Measured x3.7 on the call and x1.9 end to end.
#
# scipy's private module is a version detail, so EVERY use is wrapped: if the import, the attribute
# layout, or any single solve raises, `_LP_DIRECT` latches off and the solver finishes that polish
# (and every later one) on the public linprog() path with no change in behaviour.
_LP_DIRECT = True
_HS = None
try:
    from scipy.optimize._highspy import _core as _hc
except Exception:                                  # pragma: no cover - old/newer scipy layout
    _hc = None
    _LP_DIRECT = False


def _lp(c, A, b, lo, hi):
    """min c.x  s.t.  A x <= b,  lo <= x <= hi.  Returns (success, x)."""
    global _LP_DIRECT, _HS
    if _LP_DIRECT and A is not None and A.shape[0]:
        try:
            lp = _hc.HighsLp()
            m, ncol = A.shape
            lp.num_col_ = ncol
            lp.num_row_ = m
            lp.col_cost_ = c
            lp.col_lower_ = lo
            lp.col_upper_ = hi
            lp.row_lower_ = np.full(m, -1e30)       # one-sided rows: A x <= b
            lp.row_upper_ = b
            lp.a_matrix_.format_ = _hc.MatrixFormat.kRowwise
            lp.a_matrix_.num_col_ = ncol
            lp.a_matrix_.num_row_ = m
            lp.a_matrix_.start_ = A.indptr.astype(np.int32)
            lp.a_matrix_.index_ = A.indices.astype(np.int32)
            lp.a_matrix_.value_ = A.data
            if _HS is None:
                _HS = _hc._Highs()
                _HS.setOptionValue("output_flag", False)
            _HS.passModel(lp)
            _HS.run()
            if _HS.getModelStatus() != _hc.HighsModelStatus.kOptimal:
                return False, None
            return True, np.asarray(_HS.getSolution().col_value, dtype=float)
        except Exception:                           # pragma: no cover - latch off, never fail a run
            _LP_DIRECT = False
            _HS = None
    bounds = np.stack([lo, hi], axis=1)
    res = (linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs") if A is not None and A.shape[0]
           else linprog(c, bounds=bounds, method="highs"))
    return bool(res.success), (res.x if res.success else None)


# --- the SLP core --------------------------------------------------------------------------------
def _slp(xy, r, deadline, t0=0.02, max_iter=400, tmax=0.05, stall=None, gaintol=SCREEN_GAINTOL):
    """Locally maximize sum(r) from a feasible start. Returns (xy, r, sum_r).

    `stall`: stop after that many consecutive LPs gaining <= `gaintol` (None = run to the 1e-11
    trust-region floor, the original behaviour). See SCREEN_STALL for the measurement behind it.
    """
    n = len(r)
    xy = np.asarray(xy, float).copy()
    r = _repair(xy, r)
    best_xy, best_r = xy.copy(), r.copy()
    best_s = float(r.sum())
    t = t0
    iu, ju = np.triu_indices(n, 1)

    c = np.zeros(3 * n)
    c[2 * n:] = -1.0
    run = 0                                  # consecutive LPs that gained <= gaintol
    _SLP_STATS["lps"] = 0                    # diagnostic only (asserted by --self-test)

    for _ in range(max_iter):
        if time.process_time() > deadline or t < 1e-11:
            break
        d = _pairdist(xy)
        slack_all = d[iu, ju] - r[iu] - r[ju]
        m = slack_all < 4.0 * t + 1e-12          # exact pruning bound: slack can shrink by <= 4t
        ai, aj, sl = iu[m], ju[m], slack_all[m]
        na = len(ai)

        # PRUNE the 4n wall rows by the same exact trust-region bound the pair rows use above.
        # Row (i, wall) reads  +-dx_i + dr_i <= slack, and |dx_i| <= t, dr_i <= t, so its LHS can
        # never exceed 2t: a row with slack > 2t is non-binding for EVERY point of the trust region
        # and dropping it leaves the LP optimum identical. Measured (artifacts/hopcost.py) the wall
        # block was 53-61% of all rows while the pair block was already pruned; nearly every wall
        # row belongs to an interior circle sitting ~0.3 from the nearest wall, against t <= 0.05.
        wr, wc, wv, b_wall = _wall_rows(xy, r, t)
        nw = len(b_wall)
        if na:
            dv = xy[ai] - xy[aj]
            # Two centres can land exactly on top of each other (the post-LP clip pushes squeezed
            # 1e-9 circles into the same corner). dd == 0 makes u NaN, and linprog then raises
            # "A_ub must not contain nan", which kills the WHOLE metered solve() call. Floor dd and
            # give a coincident pair an arbitrary unit direction: the constraint is still a valid
            # under-estimate (||p_i-p_j|| >= u.(dp_i-dp_j) holds for any unit u when d0 = 0).
            dd = np.sqrt((dv ** 2).sum(1))
            bad = dd < 1e-12
            if bad.any():
                dv = dv.copy()
                dv[bad] = np.array([1.0, 0.0])
                dd = np.where(bad, 1.0, dd)
            u = dv / dd[:, None]
            rows = nw + np.repeat(np.arange(na), 6)
            cols = np.stack([ai, ai + n, aj, aj + n, ai + 2 * n, aj + 2 * n], axis=1).ravel()
            vals = np.stack([-u[:, 0], -u[:, 1], u[:, 0], u[:, 1],
                             np.ones(na), np.ones(na)], axis=1).ravel()
            A = coo_matrix((np.concatenate([wv, vals]),
                            (np.concatenate([wr, rows]), np.concatenate([wc, cols]))),
                           shape=(nw + na, 3 * n)).tocsr()
            b = np.concatenate([b_wall, sl])
        else:
            A = coo_matrix((wv, (wr, wc)), shape=(nw, 3 * n)).tocsr()
            b = b_wall

        lo = np.concatenate([np.full(2 * n, -t), np.maximum(-t, 1e-9 - r)])
        hi = np.full(3 * n, t)
        # Both blocks can prune to nothing at a tiny t (every circle strictly interior and every
        # pair strictly separated by more than the trust region can close). A 0-row A_ub is not
        # something highs accepts, and the LP is then just the box -- pass no constraint matrix.
        ok, z = _lp(c, A if nw + na else None, b if nw + na else None, lo, hi)
        _SLP_STATS["lps"] += 1
        if not ok:
            t *= 0.4
            run += 1
            if stall is not None and run >= stall:
                break
            continue

        nxy = xy + np.stack([z[:n], z[n:2 * n]], axis=1)
        np.clip(nxy, LO, HI, out=nxy)
        nr = _repair(nxy, r + z[2 * n:])
        s = float(nr.sum())
        if s > best_s + 1e-15:
            gain = s - best_s
            xy, r, best_s = nxy, nr, s
            best_xy, best_r = nxy.copy(), nr.copy()
            t = min(tmax, t * 1.6) if gain > 0.3 * t else t * 0.85
            run = 0 if gain > gaintol else run + 1
        else:
            t *= 0.35
            run += 1
        if stall is not None and run >= stall:
            break
    return best_xy, best_r, best_s


# --- starts --------------------------------------------------------------------------------------
def _grow(xy):
    """Radii at their guaranteed cap for these centres: min(wall, half the nearest-centre distance)."""
    return _repair(xy, np.minimum(_wall(xy), _pairdist(xy).min(axis=1) / 2.0))


def _cold(n, rng):
    """A feasible random start: uniform centres, radii at their guaranteed cap."""
    return rng.uniform(LO + 1e-3, HI - 1e-3, size=(n, 2)), None


def _lattice(n, rng):
    """A structured start: a jittered square or hexagonal lattice, trimmed/padded to exactly n.

    Uniform random centres give a start whose Voronoi cells are wildly uneven, and SLP -- being a
    local method -- inherits that topology. Near-optimal csqv packings are lattice-like over most of
    the square, so seeding from a lattice puts SLP in a far better basin. The row/column counts are
    drawn around sqrt(n) rather than fixed, so this is a *family* of starts to sample, not a formula
    keyed on n."""
    c = int(round(np.sqrt(n)))
    cols = max(2, c + int(rng.randint(-1, 2)))
    rows = max(2, int(np.ceil(n / float(cols))) + int(rng.randint(0, 2)))
    hexed = rng.rand() < 0.5
    gx = (np.arange(cols) + 0.5) / cols - 0.5
    gy = (np.arange(rows) + 0.5) / rows - 0.5
    px = np.repeat(gx[None, :], rows, axis=0).copy()
    if hexed:
        px[1::2] += 0.5 / cols                       # offset every other row -> hex-like
    py = np.repeat(gy[:, None], cols, axis=1)
    pts = np.stack([px.ravel(), py.ravel()], axis=1)
    pts = pts + rng.normal(0.0, 0.25 / max(rows, cols), size=pts.shape)
    if len(pts) > n:                                  # trim: drop a random subset
        pts = pts[rng.permutation(len(pts))[:n]]
    elif len(pts) < n:                                # pad: uniform extras
        pts = np.concatenate([pts, rng.uniform(LO, HI, size=(n - len(pts), 2))], axis=0)
    np.clip(pts, LO + 1e-4, HI - 1e-4, out=pts)
    return pts, None


def _row_patterns(n):
    """Every (rows, a, b) whose alternating row counts sum to EXACTLY n.

    A start whose rows hold exact integer counts keeps its row topology through the first SLP
    steps; the previous `_lattice` built a rows x cols grid and then randomly trimmed or padded it
    to reach n, which shreds precisely that topology. This is an enumerated *family* -- every
    member is tried and scored through evaluate(), and which one wins is decided by measurement,
    not by any formula keyed on n. Row counts differ by at most 2 so the pattern stays lattice-like.
    """
    out = []
    for rows in range(1, int(2.2 * np.sqrt(n)) + 3):
        no, ne = (rows + 1) // 2, rows // 2
        for a in range(1, n // no + 1):
            num = n - no * a
            if num < 0:
                break
            if ne == 0:
                if num == 0:
                    out.append((rows, a, a))
                continue
            if num % ne == 0:
                b = num // ne
                if b >= 1 and abs(a - b) <= 2:
                    out.append((rows, a, b))
    return out


def _row_start(n, rows, a, b, stagger, rng=None, jitter=0.0):
    """Centres for one member of the row family: `rows` rows, alternating counts a / b."""
    ys = (np.arange(rows) + 0.5) / rows - 0.5
    pts = []
    for j in range(rows):
        c = a if j % 2 == 0 else b
        if stagger and j % 2:
            xs = (np.arange(c) + 1.0) / (c + 1.0) - 0.5      # half-pitch offset -> hex-like
        else:
            xs = (np.arange(c) + 0.5) / c - 0.5
        pts.append(np.stack([xs, np.full(c, ys[j])], axis=1))
    p = np.concatenate(pts, axis=0)
    if jitter > 0.0 and rng is not None:
        p = p + rng.normal(0.0, jitter / np.sqrt(n), size=p.shape)
    np.clip(p, LO + 1e-4, HI - 1e-4, out=p)
    return p



def _lat_pts(rows, a, b, stag):
    """A regular lattice: `rows` rows of a / b circles, SHARED pitch 1/max(a,b), centred.

    Unlike `_row_start` (which spreads each row uniformly over the full width, so a short row sits at
    the wrong offsets) every row here is on one common pitch, and a staggered row is shifted by half a
    pitch -- i.e. a genuine hexagonal lattice, which is the structure the best csqv packings actually
    have. Row spacing is the hex value sqrt(3)/2 * pitch, clipped so `rows` rows still fit."""
    m = max(a, b)
    p = 1.0 / m
    if rows > 1:
        h = min(np.sqrt(3.0) / 2.0 * p if stag else p, (1.0 - p) / (rows - 1))
    else:
        h = 0.0
    ys = (np.arange(rows) - (rows - 1) / 2.0) * h
    pts = []
    for j in range(rows):
        c = a if j % 2 == 0 else b
        xs = (np.arange(c) - (c - 1) / 2.0) * p
        if stag and j % 2 and c == a:
            xs = xs + 0.5 * p
        pts.append(np.stack([xs, np.full(c, ys[j])], axis=1))
    return np.concatenate(pts, axis=0)


def _fill_gaps(pts, k, rng, probe=1024):
    """Append k centres, each dropped one at a time into the emptiest remaining spot.

    INCREMENTAL. The obvious version redraws `probe` candidates per inserted circle and rebuilds the
    whole (probe x |cur|) distance matrix each time -- O(k * probe * n). But "distance to the nearest
    centre" only ever DECREASES when a centre is added, so one shared candidate set needs exactly one
    O(probe * n) pass plus an O(probe) running minimum per insertion. Measured (artifacts/
    start_cost.py) this call was 11-25x the cost of the _grow() prescreen it feeds, and the prescreen
    is the solver's start factory -- so this is CPU converted straight into extra draws, which
    artifacts/slp_curve.py showed is the only currency left once each polish is converged and flat.
    A shared candidate set does not clump: once a centre lands on candidate j, the running minimum
    drives the room at j (and at its neighbours) to ~0, so the next argmax goes elsewhere."""
    if k <= 0:
        return pts
    q = rng.uniform(LO + 1e-3, HI - 1e-3, size=(probe, 2))
    gap = np.sqrt(((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(axis=1)
    wall = _wall(q)
    out = [pts]
    for _ in range(k):
        j = int(np.argmax(np.minimum(gap / 2.0, wall)))
        out.append(q[j:j + 1])
        gap = np.minimum(gap, np.sqrt(((q - q[j]) ** 2).sum(-1)))
    return np.concatenate(out, axis=0)


def _lattice_patterns(n, slack=6):
    """Every (rows, a, b, stag, k) lattice whose point count k lands in [n - slack, n].

    The exact-count row family can only express patterns that use ALL n circles in near-full rows.
    But the n that already MATCH the record have a radius histogram that is one tight cluster plus a
    few markedly smaller circles, while the stuck n have radii spread ~2:1 -- the signature of an
    uneven local optimum. So the structure to search is "regular lattice of k <= n circles, the
    remaining n - k dropped into the gaps", which needs k to be free to fall SHORT of n."""
    out = []
    for rows in range(2, int(2.4 * np.sqrt(n)) + 3):
        no, ne = (rows + 1) // 2, rows // 2
        for a in range(2, n // no + 1):
            for b in (a, a - 1):
                if b < 1:
                    continue
                k = no * a + ne * b
                if k > n:
                    continue
                if k >= n - slack:
                    for stag in ((False, True) if b == a else (True,)):
                        out.append((rows, a, b, stag, k))
    return out


def _fill_start(n, pat, rng, jitter=0.0):
    """Centres for one member of the fill-lattice family."""
    rows, a, b, stag, k = pat
    q = _lat_pts(rows, a, b, stag)
    if k < n:
        q = _fill_gaps(q, n - k, rng)
    if jitter > 0.0:
        q = q + rng.normal(0.0, jitter / np.sqrt(n), size=q.shape)
    np.clip(q, LO + 1e-4, HI - 1e-4, out=q)
    return q


def _fill_room(xy, r, k, rng, probe=1024):
    """Append k circles, each at the roomiest probed spot -- RADIUS-AWARE, unlike _fill_gaps.

    _fill_gaps measures room as half the distance to the nearest CENTRE, which is right only when the
    neighbours are all the same size. A transferred pack is not: its circles already have their final,
    uneven radii, so room here is min(wall, min_j(|q - p_j| - r_j)) and the new circle takes it.

    INCREMENTAL for the same reason as _fill_gaps, and the saving is far larger here: this is the
    m < n arm of _transfer, where k = n - m now reaches 12 (iteration 8 widened the source range), so
    the naive form rebuilt a (1024 x 87) matrix twelve times for ONE start. The clearance
    min_j(|q - p_j| - r_j) is again monotone decreasing in the circles placed so far, so one pass
    plus a running minimum is exact -- a new circle only ever tightens it by |q - p_new| - r_new."""
    xy = xy.copy()
    r = r.copy()
    if k <= 0:
        return xy, _repair(xy, r)
    q = rng.uniform(LO + 1e-4, HI - 1e-4, size=(probe, 2))
    d = (np.sqrt(((q[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]).min(axis=1)
    wall = _wall(q)
    nxy = np.empty((k, 2))
    nr = np.empty(k)
    for i in range(k):
        room = np.minimum(d, wall)
        j = int(np.argmax(room))
        nxy[i] = q[j]
        nr[i] = max(1e-6, room[j])
        d = np.minimum(d, np.sqrt(((q - q[j]) ** 2).sum(-1)) - nr[i])
    xy = np.concatenate([xy, nxy], axis=0)
    r = np.concatenate([r, nr])
    return xy, _repair(xy, r)


def _transfer(n, m, rng, jitter=0.0, slop=0):
    """A start for n built from the committed pack of a DIFFERENT size m. GUARDED -> None if absent.

    The lattice families enumerate structures a FORMULA can write down. The census holds something
    they cannot express: 36 other genuinely optimized packings, several of them at or past the
    record. Their contact topology is the expensive part of the answer, and it is nearly right for a
    neighbouring count -- so re-use it. m > n: drop the (m - n) smallest circles (they are the ones
    trapped in leftover gaps) and let _grow re-inflate everything into the freed room. m < n: drop
    the (n - m) extra circles into the roomiest spots. `slop` randomizes WHICH of the small circles
    are dropped so the move is a family, not a single point.

    This is a search operator over the census, not a table: it stores no coordinates and is keyed on
    no n -- an n whose neighbours have no pack (a held-out offline re-run) simply gets None here and
    falls through to the lattice families below."""
    if m == n or m < 1:
        return None
    p = _read_pack(m)
    if p is None:
        return None
    xy, r = p
    if m > n:
        k = m - n
        order = np.argsort(r)
        drop = order[:k + slop]
        if slop:
            drop = drop[rng.permutation(len(drop))[:k]]
        keep = np.setdiff1d(np.arange(m), drop)
        xy = xy[keep]
    elif m < n:
        xy, r = _fill_room(xy, r, n - m, rng)
    if jitter > 0.0:
        xy = xy + rng.normal(0.0, jitter / np.sqrt(n), size=xy.shape)
    xy = np.clip(xy, LO + 1e-4, HI - 1e-4)
    return xy, _grow(xy)


def _shake(xy, r, sigma, rng):
    """Basin hop: jitter EVERY centre by sigma and let _repair shrink whatever now overlaps.

    `_relocate` makes a BIG change to a FEW circles (the k smallest move to the roomiest gap) and
    leaves the global contact graph intact -- so it can only fix a mis-placed rattler, never a pack
    whose whole cell structure is one contact off. That is exactly the standing failure: 25 of the 37
    visible n sit in a 2.5-4.2 digit band, ~1e-3 relative short, which is a topology error spread
    over the pack rather than one bad circle.

    So perturb the other way: a SMALL change to ALL circles. Shrinking is what makes it safe -- the
    jittered centres are re-radiused by `_repair`, which only ever shrinks, so the start is feasible
    by construction and SLP (monotone from any feasible point) can re-grow into whatever contact
    graph the displaced centres now favour. The pack pays ~1e-2 of objective up front and either
    re-converges to where it came from or lands somewhere else.

    Measured against `_relocate` on the committed census (artifacts/shake_ab.py, 4 CPU-s per n per
    arm, paired seeds, accept-if-better both sides): +0.449 mean digits over 8 live n vs +0.022 for
    `_relocate`. The two win DIFFERENT n -- shake takes 49 and 99, relocate takes 81 -- so the hop
    loop keeps drawing from both.

    sigma is passed in relative to the mean radius by the caller; it stores no coordinates and is
    keyed on no n, so a held-out size gets the identical move."""
    q = xy + rng.normal(0.0, sigma, size=xy.shape)
    np.clip(q, LO + 1e-9, HI - 1e-9, out=q)
    rr = _repair(q, r)
    if _sum_ok(q, rr)[0]:
        return q, rr
    # A jitter large enough to stack two centres on top of each other beats _repair: it can only
    # shrink, and it floors at 1e-9, so a coincident pair stays 2e-9 overlapped -- past the 1e-9
    # tolerance. Fall back to the half-nearest-distance cap (feasible for ANY centres by
    # construction), and if even that degenerates, return the incumbent untouched rather than hand
    # SLP a start that breaks its feasible-by-induction guarantee. In the hop loop sigma is at most
    # 0.3 x mean(r), so neither branch fires there; this just makes the move total for any sigma.
    rr = _repair(q, np.minimum(rr, _pairdist(q).min(axis=1) / 2.0))
    if _sum_ok(q, rr)[0]:
        return q, rr
    return xy.copy(), r.copy()


def _relocate(xy, r, k, rng):
    """Basin hop: drop the k smallest circles into the emptiest spots found by a random probe."""
    xy = xy.copy()
    r = r.copy()
    n = len(r)
    order = np.argsort(r)[:k]
    for i in order:
        keep = np.ones(n, bool)
        keep[order] = False
        p = rng.uniform(LO, HI, size=(256, 2))
        if keep.any():
            gap = np.sqrt(((p[:, None, :] - xy[None, keep, :]) ** 2).sum(-1)) - r[None, keep]
            room = np.minimum(gap.min(axis=1), _wall(p))
        else:
            room = _wall(p)
        j = int(np.argmax(room + rng.uniform(0, 0.01, size=256)))
        xy[i] = p[j]
        keep[i] = True
    return xy, _repair(xy, np.minimum(r, _wall(xy)))


_PACK_CACHE = {}          # n -> (xy, r) or None; CLEARED at every solve() entry (see below)


def _read_pack(n):
    """Warm start from the committed census -- GUARDED: any n without a pack cold-starts.

    MEMOIZED per solve() call. The census is read far more often than it looks: _capped() reads all
    37, _sources() reads up to 12 per n, and every _transfer draw re-parsed its source from text.
    The driver snapshots bench/packs/ across solve(), so the files cannot change underneath one call;
    the cache is cleared at entry so a LATER call in the same process (the offline re-run writes
    packs between calls) still sees fresh disk. Callers mutate what they get, so hand out copies."""
    if n in _PACK_CACHE:
        c = _PACK_CACHE[n]
        return None if c is None else (c[0].copy(), c[1].copy())
    v = _load_pack(n)
    _PACK_CACHE[n] = v
    return None if v is None else (v[0].copy(), v[1].copy())


def _load_pack(n):
    p = os.path.join("bench", "packs", "csqv%d.pck" % n)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            rows = [ln.split() for ln in fh.read().splitlines()[2:] if ln.strip()]
        a = np.array([[float(v) for v in row[:3]] for row in rows], float)
        if a.shape != (n, 3):
            return None
        return a[:, :2].copy(), a[:, 2].copy()
    except Exception:
        return None


# --- the contract --------------------------------------------------------------------------------
def _read_records():
    """The frozen record table, if present -- used ONLY to rank which n still needs the most work.

    Reading bench/records.json is explicitly sanctioned; it is not keyed to any packing. Guarded, so a
    held-out n (or a missing file) simply falls back to round-robin ranking."""
    try:
        with open(os.path.join("bench", "records.json")) as fh:
            return {int(k): float(v) for k, v in json.load(fh)["records"].items()}
    except Exception:
        return {}


def _capped(n, records):
    """True iff n's COMMITTED pack already scores the full 7 digits, so more CPU on it cannot help.

    The scorer clamps digits at DIGITS_CAP, i.e. any relgap <= 1e-7 scores 7.0 and no further gain on
    that n moves the mean at all. Seven of the 37 visible n sit there, and under a weight ~ n^1.2
    schedule they were still drawing ~19% of the whole CPU budget for a guaranteed zero return. This is
    keyed on the MEASURED gap of whatever pack is on disk, not on n: a fresh n (no pack) or a size with
    no record entry returns False and takes a full slice, which is exactly the held-out offline path."""
    if not records or n not in records or records[n] <= 0:
        return False
    p = _read_pack(n)
    if p is None:
        return False
    return (records[n] - float(p[1].sum())) / records[n] <= CAP_GAP


# --- transfer-source ranking ---------------------------------------------------------------------
# Iteration 6 fixed the transfer family at |offset| <= 6 and swept it in |offset| order. Two things
# changed since: the census now holds TEN packs at the 7-digit cap (they are the best topologies in
# the library by a wide margin), and the three sizes furthest from any of them -- 95, 97, 99 -- are
# all still stuck. So rank sources by MEASURED quality, not by proximity, and let the range reach the
# good packs. A 7-digit source 12 counts away outranks a 3-digit source 2 counts away; between two
# equally good sources the nearer one still wins.
XFER_OFFSETS = (-2, 2, -4, 4, -6, 6, -8, 8, -10, 10, -12, 12)
XFER_DIST_PENALTY = 0.25     # digits forfeited per unit of |m - n|; see the trade-off above
XFER_TOPK = 6                # keep the sweep the same width it was, just made of better members


def _source_digits(m, records):
    """Digits scored by the COMMITTED pack at size m, or None if there is no such pack.

    Measured from disk, keyed on no n: a size with no pack (the held-out offline path) gives None and
    a size with no record entry gives a neutral score -- it is still a genuinely optimized topology."""
    p = _read_pack(m)
    if p is None:
        return None
    if not records or m not in records or records[m] <= 0:
        return 3.0
    g = (records[m] - float(p[1].sum())) / records[m]
    if g <= CAP_GAP:
        return DIGITS_CAP
    return float(min(DIGITS_CAP, -np.log10(g)))


def _sources(n, records, k=XFER_TOPK):
    """The k committed sizes worth transferring INTO n, best first. Empty when none is committed."""
    out = []
    for off in XFER_OFFSETS:
        m = n + off
        if m < 1:
            continue
        d = _source_digits(m, records)
        if d is not None:
            out.append((d - XFER_DIST_PENALTY * abs(off), m))
    out.sort(key=lambda z: -z[0])
    return [m for _s, m in out[:k]]


def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    targets = sorted(targets)
    nt = len(targets)
    t_entry = time.process_time()
    _PACK_CACHE.clear()          # disk may have changed since the previous call in this process
    # Whole-call cap AND per-size cap, both derived AT ENTRY (so a 2nd call in this process re-derives
    # them) and from len(targets) (so len(targets) == 1 -- the offline re-run -- takes the full slice).
    total = min(CALL_CPU_CAP, PER_SIZE_CPU * nt)
    end = t_entry + total

    # CUMULATIVE schedule. Iteration 1's bug: each n got `1.6 x share` measured from *its own* start,
    # with no running cap, so early n ate the whole call and the last 13 of 37 never ran at all. An
    # absolute cumulative deadline bounds the tail by construction -- an n that stops early donates its
    # remainder to the ones after it automatically, and one that overruns cannot borrow from them.
    # Weights ~ n^1.2: an LP at n=99 has 3x the variables of one at n=27 and costs superlinearly more.
    records = _read_records()
    w = np.array([float(n) ** 1.2 for n in targets])
    # ...times a HEADROOM factor: an n whose committed pack is already at the 7-digit cap cannot raise
    # the mean by one more millisecond of work, so it keeps only a token share and donates the rest to
    # the n that are still 3-4 digits short. Normalisation means a call where EVERY target is capped
    # (or none is) is unaffected -- only the mix matters.
    w = w * np.array([0.06 if _capped(n, records) else 1.0 for n in targets])
    if w.sum() <= 0:
        w = np.ones(nt)
    frac = np.cumsum(w) / w.sum()

    best = {}                                    # n -> [xy, r, sum_r]
    _src_cache = {}

    def srcs(n):
        """Ranked transfer sources for n, computed once per n (each call re-reads up to 12 packs)."""
        if n not in _src_cache:
            _src_cache[n] = _sources(n, records)
        return _src_cache[n]

    def offer(n, xy, r):
        """Route a candidate through the metered evaluate() -- the ONLY sanctioned test."""
        ok, s = _sum_ok(xy, r)
        cur = best.get(n)
        if not ok or (cur is not None and s <= cur[2]):
            return False
        if meter.left() <= 0:
            return False
        feas, es = evaluate(n, _pack(xy, r))
        feas = bool(np.atleast_1d(feas)[0])
        es = float(np.atleast_1d(es)[0])
        if feas and (cur is None or es > cur[2]):
            best[n] = [xy.copy(), r.copy(), es]
            return True
        return False

    def polish(n, sxy, sr, dl):
        """SCREEN a start with a stall-limited SLP, DEEPEN it only if it might win, then offer it.

        The screen is x1.89 cheaper (see SCREEN_STALL) and, measured over 42 traces, never loses
        more than 1.3e-12 -- but "never in 42" is not "never", so nothing gets committed on the
        screen's word alone: a result within DEEP_MARGIN of the incumbent is re-polished to the full
        1e-11 trust-region floor first, and the deeper answer is kept only if it is actually better.
        Losing starts -- the overwhelming majority -- are rejected on the cheap polish, which is
        where all the CPU was going."""
        xy, r, sc = _slp(sxy, sr, dl, stall=SCREEN_STALL)
        cur = best.get(n)
        if cur is None or sc > cur[2] - DEEP_MARGIN:
            dxy, dr, ds = _slp(xy, r, min(end, time.process_time() + DEEP_CPU_S))
            if ds > sc:
                xy, r = dxy, dr
        return offer(n, xy, r)

    def hop(n, deadline, slice_s):
        """One basin hop (or the warm/cold seed) on n, polished by SLP, then offered. True on gain."""
        cur = best.get(n)
        u = rng.rand()
        if cur is not None and u < 0.22:
            sxy, sr = _relocate(cur[0], cur[1], int(rng.randint(1, 4)), rng)
        elif cur is not None and u < 0.45:
            # the OTHER half of the old relocate share (see _shake): same incumbent, opposite move.
            sxy, sr = _shake(cur[0], cur[1],
                             (0.02 + 0.28 * rng.rand()) * float(cur[1].mean()), rng)
        elif u < 0.62:
            # a JITTERED member of the NEIGHBOUR-TRANSFER family (see _transfer). Guarded: if the
            # drawn neighbour has no committed pack this falls through to the fill-lattice draw.
            ss = srcs(n)
            st = None
            if ss:
                m = int(ss[int(rng.randint(min(len(ss), 4)))])   # biased to the top of the ranking
                st = _transfer(n, m, rng, 0.05 + 0.2 * rng.rand(), int(rng.randint(0, 3)))
            if st is not None:
                sxy, sr = st
            else:
                pats = _lattice_patterns(n, slack=int(rng.randint(2, 9)))
                sxy = (_fill_start(n, pats[rng.randint(len(pats))], rng, 0.1 + 0.4 * rng.rand())
                       if pats else _lattice(n, rng)[0])
                sr = _grow(sxy)
        elif u < 0.80:
            # a JITTERED member of the FILL-LATTICE family: the deterministic sweep above is
            # exhausted after one iteration, so the randomized variant keeps paying out later.
            pats = _lattice_patterns(n, slack=int(rng.randint(2, 9)))
            if pats:
                sxy = _fill_start(n, pats[rng.randint(len(pats))], rng, 0.10 + 0.4 * rng.rand())
            else:
                sxy, _u2 = _lattice(n, rng)
            sr = _grow(sxy)
        elif u < 0.87:
            # a JITTERED member of the exact-count row family (iteration 3's start set)
            pats = _row_patterns(n)
            rows, a, b = pats[rng.randint(len(pats))] if pats else (1, n, n)
            sxy = _row_start(n, rows, a, b, bool(rng.randint(2)), rng, 0.12 + 0.5 * rng.rand())
            sr = _grow(sxy)
        else:
            sxy, sr = _lattice(n, rng) if rng.rand() < 0.85 else _cold(n, rng)
            sr = _grow(sxy)
        return polish(n, sxy, sr,
                      min(deadline, time.process_time() + max(0.25, 0.6 * slice_s)))

    # ---- pass 1: every n gets its slice, in order, under an absolute cumulative deadline ----------
    for k, n in enumerate(targets):
        if meter.left() <= 0 or time.process_time() >= end:
            break
        deadline = t_entry + total * float(frac[k])
        slice_s = max(0.05, deadline - time.process_time())

        warm = _read_pack(n)                      # GUARDED: a fresh n just cold-starts below
        if warm is not None:
            # OFFER IT DIRECTLY -- no SLP. Iteration 5 measured (artifacts/diag5.py) that a committed
            # pack re-polishes to ITSELF (dsum ~ -1e-13), and artifacts/slp_curve.py now shows why:
            # one SLP polish is fully converged in < 0.1 s and dead flat thereafter, so the census is
            # converged by construction. The re-polish still cost ~20 LPs per n (the trust region has
            # to shrink 0.02 -> 1e-11 before the loop exits) for a guaranteed zero. We still need the
            # incumbent registered -- offer() is what seeds best[n] for the _relocate hops -- so pay
            # the one evaluate() and skip the LPs. _repair() guards the .pck's printed precision.
            offer(n, warm[0], _repair(warm[0], warm[1]))
        # ---- NEIGHBOUR-TRANSFER sweep (the census re-used as a start family) ----------------
        # Measured at 1.2 s/n over 12 n (artifacts/nbr2_ab.py): +0.166 mean digits over the committed
        # packs -- more than iteration 5's full-slice fill-lattice sweep (+0.110) or hop loop
        # (+0.052) bought for 2.8 s. It is also the only family that gets BETTER every iteration,
        # because its inputs are the packs the previous iteration improved. Offsets are visited in a
        # random permutation for the same reason the prescreen is (iteration 5): the loop is
        # time-capped, and truncation in a fixed order is a frozen subset, not a sample.
        xf_end = time.process_time() + 0.35 * max(0.0, deadline - time.process_time())
        ms = srcs(n)                                  # GUARDED -> [] when nothing nearby is committed
        ms = rng.permutation(np.array(ms)) if ms else []
        for oi, m in enumerate(ms):
            if time.process_time() >= xf_end or meter.left() <= 0:
                break
            st = _transfer(n, int(m), rng)
            if st is None:
                continue
            per_x = max(0.08, (xf_end - time.process_time()) / max(1, len(ms) - oi))
            polish(n, st[0], st[1], min(xf_end, time.process_time() + per_x))

        # Sweep the FILL-LATTICE family. Iteration 3's exact-count row sweep is deterministic, so
        # re-running it only reproduces the committed incumbent; this family is strictly wider (the
        # lattice may use k < n circles, the rest dropped into the gaps) and measured far better at
        # the same CPU: at ~3 s/size it moved n=73 by +0.96, n=87 by +1.18 and n=91 by +1.16 digits
        # over the committed packs, where the row sweep re-derived them exactly.
        # The family is too big to SLP in full, so it is PRE-SCREENED by _grow() -- the sum of radii
        # of the raw start, one O(n^2) pass, no LP -- and only the top few are polished. Measured on
        # n=49/73/87/91 the top-8-of-~50 prescreen kept 0.96 of the 1.13 digits of the full sweep for
        # a fifth of the CPU.
        pats = _lattice_patterns(n)
        if pats:
            scored = []
            # The sweep gets a BOUNDED share of the slice, so the basin-hop loop below always runs.
            # Measured at the real in-run slice (~2.8 s/size, artifacts/alloc_ab.py + hybrid_ab.py):
            # a sweep-only slice and a hop-only slice gain +0.110 and +0.052 mean digits, but they
            # win DIFFERENT n (sweep: 27/63/71/81; hops: 55/81) -- the per-n gain is a heavy-tailed
            # lottery, so drawing from both families beats spending the whole slice on either.
            sweep_end = time.process_time() + 0.5 * max(0.0, deadline - time.process_time())
            # Prescreen in a RANDOM PERMUTATION of the family, not in index order. The prescreen is
            # time-capped, and at n >= 81 it never reaches the end of the family (n=99: the sweep ate
            # the whole slice and left 0 hops). In index order that truncation always keeps the same
            # low-`rows` PREFIX, so those n re-screened an identical subset every iteration and could
            # never see the rest of the family. A permutation makes the truncation an unbiased sample
            # instead, so successive iterations cover the whole family.
            pre_end = time.process_time() + 0.5 * max(0.0, sweep_end - time.process_time())
            for i in rng.permutation(len(pats)):
                if time.process_time() >= pre_end:
                    break
                q = _fill_start(n, pats[i], rng)
                scored.append((float(_grow(q).sum()), q))
            scored.sort(key=lambda z: -z[0])
            top = scored[:4]
            per = max(0.08, (sweep_end - time.process_time()) / max(1, len(top)))
            for _g, q in top:
                if time.process_time() >= sweep_end or meter.left() <= 0:
                    break
                polish(n, q, _grow(q), min(sweep_end, time.process_time() + per))

        if best.get(n) is None:
            sxy, _u = _lattice(n, rng)
            polish(n, sxy, _grow(sxy), min(deadline, time.process_time() + 0.5 * slice_s))

        stall = 0
        while time.process_time() < deadline and meter.left() > 0:
            stall = 0 if hop(n, deadline, slice_s) else stall + 1
            # Donate to the n after us once this basin is clearly exhausted -- but never when we are
            # the last target (offline: one n per call, and unused CPU earns nothing).
            if stall >= 6 and k < len(targets) - 1:
                break

    # ---- pass 2: spend everything left on whichever n is furthest from its record ----------------
    rr = 0
    while time.process_time() < end and meter.left() > 0 and best:
        live = [n for n in targets if n in best]
        if not live:
            break
        if records:
            n = max(live, key=lambda m: (records[m] - best[m][2]) / records[m]
                    if m in records else 1.0)
        else:
            n = live[rr % len(live)]
            rr += 1
        hop(n, min(end, time.process_time() + 1.0), 1.0)


# --- self-test -----------------------------------------------------------------------------------
def _self_test():
    class M:
        def __init__(self, b):
            self.budget = b
            self.used = 0

        def left(self):
            return self.budget - self.used

        def tick(self):
            pass

    seen = {}

    def ev(n, pack):
        a = np.asarray(pack, float)
        single = a.ndim == 2
        if single:
            a = a[None]
        if a.shape[1] != n or a.shape[2] != 3:
            raise ValueError("bad shape")
        meter.used += a.shape[0]
        fs, ss = [], []
        for p in a:
            ok, s = _sum_ok(p[:, :2], p[:, 2])
            fs.append(ok)
            ss.append(s if ok else -np.inf)
            if ok and s > seen.get(n, -np.inf):
                seen[n] = s
        if single:
            return fs[0], ss[0]
        return np.array(fs), np.array(ss)

    global CALL_CPU_CAP, PER_SIZE_CPU
    CALL_CPU_CAP, PER_SIZE_CPU = 6.0, 3.0
    meter = M(100000)
    rng = np.random.RandomState(0)

    solve(ev, meter, rng, [11, 13])
    assert set(seen) == {11, 13}, seen
    first = dict(seen)

    # solve() MUST work when called AGAIN in the same process (deadlines are relative, not absolute)
    seen.clear()
    solve(ev, meter, rng, [13])                       # ... and with a single-n target list
    assert 13 in seen, "second call in the same process produced nothing"
    assert seen[13] > 0

    # guarded warm start: an n with no committed pack must still work
    seen.clear()
    solve(ev, meter, rng, [7])
    assert 7 in seen

    # empty targets must be a no-op, not an error
    solve(ev, meter, rng, [])

    # the headroom filter must be GUARDED: no pack / no record entry -> not capped -> full slice.
    assert _capped(7, _read_records()) is False
    assert _capped(29, {}) is False

    # the transfer-source ranking must be GUARDED the same way: a size with no committed neighbour
    # anywhere (the held-out offline path) gets an EMPTY source list, never an exception.
    _recs = _read_records()
    assert _sources(7, _recs) == [] and _sources(13, _recs) == []
    assert _source_digits(7, _recs) is None
    # the memoized census reader must cache a MISS as a miss, and must never hand out an aliased
    # array a caller can mutate back into the cache.
    assert _read_pack(7) is None and _read_pack(7) is None
    _a = _read_pack(63)
    assert _a is not None
    _a[0][:] = 0.0
    assert not np.allclose(_read_pack(63)[0], 0.0), "cache handed out an aliased array"
    # ... and with no record table at all it still ranks, on proximity alone.
    assert _sources(63, {}) and all(_read_pack(m) is not None for m in _sources(63, {}))

    # --- the INCREMENTAL gap fillers -------------------------------------------------------------
    # Both now share ONE candidate set across all k insertions, so the failure they could introduce
    # is CLUMPING: the same (or a near-duplicate) candidate winning the argmax twice, which would
    # hand SLP two coincident centres -- a start that _grow() collapses to r ~ 0. Assert the
    # insertions are distinct and genuinely spread, not just that the count came out right.
    _rg = np.random.RandomState(3)
    _base = _lat_pts(8, 8, 8, True)
    for _k in (0, 1, 3, 12):
        _q = _fill_gaps(_base, _k, _rg)
        assert _q.shape == (len(_base) + _k, 2), (_q.shape, _k)
        assert np.all(_q >= LO) and np.all(_q <= HI)
        if _k:
            _new = _q[len(_base):]
            assert _pairdist(_q).min() > 1e-6, "coincident centres from a shared candidate set"
            assert len(np.unique(_new, axis=0)) == _k, "a candidate was inserted twice"
            # every insertion must stay clear of the lattice too, not merely of its own siblings
            # (a shared candidate set makes "two insertions, one hole" the realistic failure).
            assert np.sqrt(((_new[:, None, :] - _base[None, :, :]) ** 2).sum(-1)).min() > 1e-3
    assert _fill_gaps(_base, 0, _rg) is _base                 # k <= 0 must be a pure no-op

    _pk = _read_pack(63)
    for _k in (0, 1, 4, 12):
        _xy, _r = _fill_room(_pk[0], _pk[1], _k, _rg)
        assert _xy.shape == (63 + _k, 2) and _r.shape == (63 + _k,)
        assert np.all(_r > 0) and _sum_ok(_xy, _r)[0], "fill_room returned an infeasible pack"
        if _k:
            assert len(np.unique(_xy[63:], axis=0)) == _k, "a candidate was inserted twice"
    # the k <= 0 branch must still copy, never alias the caller's arrays
    _xy0, _r0 = _fill_room(_pk[0], _pk[1], 0, _rg)
    _xy0[:] = 0.0
    assert not np.allclose(_pk[0], 0.0), "_fill_room aliased its input at k=0"

    # --- the SHAKE hop ---------------------------------------------------------------------------
    # _shake hands SLP a start built by MOVING every centre, so the failure it could introduce is an
    # INFEASIBLE start (SLP's whole guarantee is that it begins feasible) or centres pushed outside
    # the square by the jitter. Assert both, across sigma from tiny to larger than a circle, and
    # assert it neither aliases nor mutates the incumbent the hop loop keeps re-using.
    _sxy, _sr = _read_pack(63)
    _sr = _repair(_sxy, _sr)
    _keep = _sxy.copy()
    for _sig in (0.0, 1e-9, 1e-3, 0.02, 0.5):
        _qxy, _qr = _shake(_sxy, _sr, _sig, _rg)
        assert _qxy.shape == _sxy.shape and _qr.shape == _sr.shape
        assert np.all(_qxy >= LO) and np.all(_qxy <= HI), "shake left the square"
        assert np.all(_qr > 0) and _sum_ok(_qxy, _qr)[0], "shake returned an infeasible start"
        assert _qxy is not _sxy and _qr is not _sr, "shake aliased its input"
        _qxy[:] = 0.0
        assert np.allclose(_sxy, _keep), "shake mutated the incumbent"
    # sigma = 0 must be the identity on the centres (and so lose nothing): the hop loop draws sigma
    # from a continuous range and a degenerate draw must not corrupt the incumbent it came from.
    _z0, _zr0 = _shake(_sxy, _sr, 0.0, _rg)
    assert np.allclose(_z0, _sxy) and abs(float(_zr0.sum()) - float(_sr.sum())) < 1e-9

    # --- the PRUNED wall block -------------------------------------------------------------------
    # The failure this can introduce is a mis-mapped row/column, which would silently constrain the
    # wrong circle or the wrong coordinate. Pin it against the dense 4n block it replaced: at a t
    # large enough that nothing is pruned, the two must be the SAME matrix, row for row.
    _wxy, _wr2 = _read_pack(45)
    _wr2 = _repair(_wxy, _wr2)
    _nn = len(_wr2)
    _idx = np.arange(_nn)
    _dr = np.zeros((4 * _nn, 3 * _nn))
    for _k, (_var, _sgn) in enumerate(((_idx, 1.0), (_idx, -1.0), (_idx + _nn, 1.0), (_idx + _nn, -1.0))):
        _dr[np.arange(_k * _nn, (_k + 1) * _nn), _var] = _sgn
        _dr[np.arange(_k * _nn, (_k + 1) * _nn), _idx + 2 * _nn] = 1.0
    _db = np.concatenate([HI - _wxy[:, 0] - _wr2, HI + _wxy[:, 0] - _wr2,
                          HI - _wxy[:, 1] - _wr2, HI + _wxy[:, 1] - _wr2])
    _rw, _cw, _vw, _bw = _wall_rows(_wxy, _wr2, 10.0)          # 2t = 20 > any slack: nothing pruned
    assert len(_bw) == 4 * _nn, "nothing should prune at t=10"
    _dense = np.zeros((len(_bw), 3 * _nn))
    _dense[_rw, _cw] = _vw
    assert np.array_equal(_dense, _dr) and np.allclose(_bw, _db), "pruned wall block != dense block"
    # ...and at a working t the kept rows must be exactly those with slack <= 2t, with their own
    # rows of the dense matrix -- so the ones dropped are provably non-binding, not merely few.
    for _t in (0.05, 0.02, 1e-3, 1e-9):
        _rw, _cw, _vw, _bw = _wall_rows(_wxy, _wr2, _t)
        _keepi = np.flatnonzero(_db < 2.0 * _t + 1e-12)
        assert len(_bw) == len(_keepi) and np.allclose(_bw, _db[_keepi])
        assert float(np.min(_db[np.setdiff1d(np.arange(4 * _nn), _keepi)], initial=np.inf)) > 2.0 * _t
        _dense = np.zeros((len(_bw), 3 * _nn))
        _dense[_rw, _cw] = _vw
        assert np.array_equal(_dense, _dr[_keepi]), "pruned wall rows lost their row/col mapping"

    # --- the SLP STALL early exit ----------------------------------------------------------------
    # The failure this can introduce is a screen that stops before the polish has converged -- a
    # start ranked below its true value, or worse, a pack committed short of its own optimum. Assert
    # BOTH halves on real starts: the stall run must spend strictly fewer LPs (the rule fires at all)
    # AND its sum must match the full-depth run to 1e-9 (it costs nothing measurable).
    _rg2 = np.random.RandomState(5)
    _cut = []
    for _n in (27, 45, 63):
        _pats = _row_patterns(_n)
        for _j in range(3):
            _rows, _a, _b = _pats[_rg2.randint(len(_pats))] if _pats else (1, _n, _n)
            _q = _row_start(_n, _rows, _a, _b, bool(_rg2.randint(2)), _rg2, 0.2)
            _dl = time.process_time() + 20.0
            _fx, _fr, _fs = _slp(_q, _grow(_q), _dl)                       # full depth
            _full_lps = _SLP_STATS["lps"]
            _sx, _sr2, _ss = _slp(_q, _grow(_q), _dl, stall=SCREEN_STALL)  # screened
            _scr_lps = _SLP_STATS["lps"]
            assert _sum_ok(_fx, _fr)[0] and _sum_ok(_sx, _sr2)[0], "SLP returned an infeasible pack"
            assert _scr_lps < _full_lps, ("the stall rule never fired", _n, _scr_lps, _full_lps)
            assert _fs - _ss < 1e-9, ("the screen lost real objective", _n, _fs - _ss)
            _cut.append(1.0 - _scr_lps / float(_full_lps))
    assert np.mean(_cut) > 0.15, ("the stall rule cut almost nothing", float(np.mean(_cut)))
    # stall=None must still be exhaustive: on a start it cannot improve (a committed pack) it has to
    # spend the whole ~20-LP shrink to the 1e-11 floor, where the screened run quits immediately.
    _px, _pr = _read_pack(45)
    _pr = _repair(_px, _pr)
    _slp(_px, _pr, time.process_time() + 20.0)
    _deep = _SLP_STATS["lps"]
    _slp(_px, _pr, time.process_time() + 20.0, stall=SCREEN_STALL)
    assert _SLP_STATS["lps"] <= SCREEN_STALL + 1 < _deep, (_SLP_STATS["lps"], _deep)
    print("stall rule: cuts %.0f%% of LPs; a committed pack re-polishes in %d LPs"
          % (100 * float(np.mean(_cut)), _deep))

    # --- the direct HiGHS call must be the SAME LP, and must degrade gracefully ------------------
    # Both halves matter: (a) bypassing scipy's wrapper returns the identical vertex, so no polish
    # can silently drift; (b) if the private scipy module ever moves, _lp falls back to linprog and
    # the solver still solves -- a solver that only works on today's scipy is not a solver.
    global _LP_DIRECT
    _rs = np.random.RandomState(3)
    for _n2 in (9, 17, 31):
        _q = _lattice(_n2, _rs)[0]
        _qr = _repair(_q, _grow(_q))
        _LP_DIRECT = _hc is not None
        _dx, _dr, _ds = _slp(_q, _qr, time.process_time() + 10.0, stall=SCREEN_STALL)
        _dlp = _SLP_STATS["lps"]
        _LP_DIRECT = False                            # the public linprog path, as iters 1-14 ran
        _lx, _lr, _ls = _slp(_q, _qr, time.process_time() + 10.0, stall=SCREEN_STALL)
        assert _ds == _ls, ("direct HiGHS changed the answer", _n2, _ds - _ls)
        assert _dlp == _SLP_STATS["lps"], ("direct HiGHS changed the LP path", _n2)
    _LP_DIRECT = False                                # fallback must produce a feasible packing
    _q19 = _lattice(19, _rs)[0]
    _fx, _fr, _fs2 = _slp(_q19, _grow(_q19), time.process_time() + 10.0)
    assert _sum_ok(_fx, _fr)[0] and _fs2 > 0
    _LP_DIRECT = _hc is not None
    print("LP path: direct HiGHS available=%s, identical vertex on 3 sizes, linprog fallback OK"
          % (_hc is not None))

    print("self-test OK  n=11 sum_r=%.9f  n=13 sum_r=%.9f  evals_used=%d"
          % (first[11], first[13], meter.used))


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print("usage: python tools/solver.py --self-test")
