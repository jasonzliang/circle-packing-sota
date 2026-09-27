"""Circle-packing solver for the csqv census: maximize the sum of radii of n circles in the unit square.

STRATEGY -- sequential LP over a *convex restriction* (this replaces the baseline's random
repulsion descent entirely).

For fixed centres the problem is a linear program in the radii; the only nonconvexity is the
pairwise constraint  r_i + r_j <= ||c_i - c_j||.  Linearising the RHS at the current centres,

    ||c_i - c_j||  >=  u_ij . (c_i - c_j),     u_ij = (c0_i - c0_j)/||c0_i - c0_j||   (|u|=1),

is a *global under-estimate* (Cauchy-Schwarz), so the LP

    max  sum_i r_i
    s.t. r_i + r_j - u_ij.(c_i - c_j) <= 0            (near pairs)
         r_i +/- x_i <= 1/2 ,  r_i +/- y_i <= 1/2     (walls, exact)
         |c - c0| <= trust ,  0 <= r <= 1/2

is an inner approximation of the true feasible set: **every LP solution is a genuinely feasible
packing**, and c0 itself is feasible for it, so the objective is monotone non-decreasing. Iterating
it (shrinking the trust region when it stalls) converges to a KKT point, typically in ~25 LPs and a
fraction of a CPU second even at n=99. Pairs are pruned to those that can become active inside the
trust region, and an exact O(n^2) repair pass afterwards absorbs HiGHS' 1e-7 primal tolerance so
the packing clears the harness' 1e-9 feasibility test with margin.

Around that core: a portfolio of starts (the committed census pack = warm start, square grids of
several pitches, random), then monotone basin hopping with two kernels -- Gaussian jitter, and
relocating the smallest circles into the largest empty holes (a contact-topology move that jitter
cannot make).

LP THROUGHPUT.  Profiling puts 92-99% of CPU inside the LP call, so the LP *is* the meter.
Iteration 3 moved off `scipy.optimize.linprog` onto raw HiGHS and carried the simplex basis by hand
across `passModel`.  Iteration 4's probe showed that basis was almost never used where it mattered:
any growth of the pair set nulled it, and after a perturbation the pair set grows on nearly every
step, so a *screen* LP still cost 11.4 ms at n=99 against 0.7 ms on an unchanged model.  The model is
now built ONCE per SLP run and mutated in place -- `changeColsBounds` (trust region), `changeCoeff`
(only the directions that drifted past _UTOL; a stale u is still a valid restriction), `addRows`
(newly-near pairs) -- so HiGHS keeps its own basis and factorisation and never sees a cold solve.
Screen LP at n=99: 11.4 -> 4.0 ms; hop throughput 10.6 -> 28.9 hops/s (n=35: 88 -> 181).
scipy remains the fallback if highspy is unavailable.

Budgets: the CPU cap, not the 500k evaluation meter, is binding, so every deadline is computed from
`time.process_time()` AT ENTRY (never a module constant) and the remaining CPU is re-split over the
remaining targets each n -- so a second solve() call in the same process, or a single-n call, both
get a full budget. Warm-start reads are existence-guarded and every start path works cold.
"""
import json
import math
import os
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

LO, HI = -0.5, 0.5
_TICK_FRAC = 0.12        # share of a slice spent on cross-n topology tickets.  Measured on my
                         # own run, not guessed: of the 8 below-cap n in iteration 9, the ticket
                         # chains cracked 6 to the 7-digit cap and the incumbent chain cracked 0.
                         # The incumbent's own basin is the one thing 9 iterations of hopping have
                         # already exhausted, so it is the wrong place for the marginal second.
_TICK_MIN = 1.5          # CPU-s floor for one ticket chain (below this a chain
                         # cannot hop at all, so buy fewer, longer draws).  Iteration 9's cracks
                         # all happened at ~1.8 CPU-s/chain, so this floor is known sufficient --
                         # which is what licenses spending a wider donor pool at the SAME floor
                         # rather than fewer, longer draws.
_TGT_FRAC = 0.10         # share of a slice given to the TARGET-FOLLOWING source, taken back
                         # from the ticket share.  It is the only source here that does not
                         # maximise sum(r), so it is the only one whose draws are not from the
                         # basin-optimum distribution the other two share.
_SYM_FRAC = 0.05         # share of a slice given to the SYMMETRY-PROJECTION source.  Taken from
                         # the two sum(r)-maximising sources, not from the target follower: both
                         # of those search the same amorphous cloud, and this one does not search
                         # it at all -- it lives on a measure-zero subspace they never enter.
                         # Cut 0.18 -> 0.10 in iteration 13: `probe_symdepth` measured the best
                         # symmetric local optimum for the one n still below cap at 0.02-0.09
                         # BELOW its incumbent, so on that n the slice is measured-dead -- but the
                         # source stays funded, because the census says 5 of 36 record packings
                         # (n=49 most of all) ARE symmetric and no other source can reach them.
_LAD_FRAC = 0.06         # share of a slice given to the CONTINUATION LADDER.  Target following
                         # jumps straight to sum(r) = record and solves ONE phase-1; the ladder
                         # walks the target up in rungs and returns to a FEASIBLE local optimum
                         # between rungs, so it tracks a homotopy path instead of taking a single
                         # leap into a violation basin it cannot see out of.
_SQZ_FRAC = 0.45         # share of a slice given to the SQUEEZE source -- the largest single
                         # share in the file, and the only frac here set from a LIVE-PATH
                         # measurement rather than an argument.  Iteration 14 built the kernel and
                         # metered nothing, because in the live path squeeze was 14% of the
                         # kernels of a ~41 s residual chain (~5.7 CPU-s) while its own probe
                         # needed ~10 CPU-s of memoryless draws to reach past the incumbent.  A
                         # source inherits the RATE at which its probe was funded or the probe
                         # does not transfer.  The four fracs above were cut to pay for it: all
                         # four are measured at exactly zero on the one n still below the cap
                         # (transfer iter 10, symmetry iter 12, ladder iters 13-14, target
                         # following decayed to 0.03 digits), so this is not a hedge being
                         # dropped, it is a slice moved off measured-dead sources onto the only
                         # source that has moved that n since iteration 11.  They stay non-zero
                         # because they are what generalises to the other 36 n and to an offline
                         # re-run on sizes the census has never seen.
_SQZ_MIN = 0.05          # CPU-s a squeeze draw needs (probe: 616 draws in 24 CPU-s = 0.039 s)
_DONOR_WIN = 10          # +/- window of census sizes a topology may be transferred from
CPU_BUDGET = 112.0        # process-CPU seconds this call may use (driver backstop is 120s)
PACK_DIR = os.path.join("bench", "packs")
_SQZ_DRAWS = 0           # diagnostic ONLY: how many squeeze draws the dedicated slice actually
                         # took.  The change this counter exists to police is a FUNDING change,
                         # and an under-funded source is indistinguishable from a funded one by
                         # looking at outputs -- both just fail to improve.  The self-test asserts
                         # a real draw count, so a slice that silently breaks out on its first
                         # pass cannot pass as the 4x funding it claims to be.


# ---------------------------------------------------------------- geometry helpers

def _wallcap(x, y):
    return np.minimum.reduce([x - LO, HI - x, y - LO, HI - y])


def _repair(x, y, r):
    """Exact feasibility projection: shrink radii until every wall and pair constraint holds with a
    small margin.  If r_i *= f_i with f_i <= min_j d_ij/(r_i+r_j) then r_i f_i + r_j f_j <= d_ij."""
    r = np.clip(r, 0.0, None)
    r = np.minimum(r, _wallcap(x, y))
    n = r.shape[0]
    if n >= 2:
        d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        np.fill_diagonal(d, np.inf)
        s = r[:, None] + r[None, :]
        f = np.where(s > 1e-300, d / np.maximum(s, 1e-300), np.inf)
        np.fill_diagonal(f, np.inf)
        r = r * np.minimum(f.min(axis=1), 1.0)
    r = np.maximum(r - 1e-12, 1e-9)
    return np.minimum(r, np.maximum(_wallcap(x, y), 1e-9))


def _pack(x, y, r):
    return np.stack([x, y, r], axis=1)


# ---------------------------------------------------------------- the LP step (HiGHS, warm-started)

try:                                    # direct HiGHS: lets us keep the simplex BASIS across SLP steps
    import highspy                      # (scipy.linprog re-solves every LP from scratch: ~12ms vs ~1ms)
    _HAVE_HS = True
except Exception:                       # pragma: no cover -- fall back to scipy if highspy is absent
    _HAVE_HS = False

_HS = None


def _hs():
    global _HS
    if _HS is None:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)          # single-threaded: the CPU cap must stay host-independent
        h.setOptionValue("presolve", "off")     # presolve throws the warm basis away
        h.setOptionValue("solver", "simplex")
        h.setOptionValue("simplex_strategy", 1)  # dual simplex: re-optimises fastest from a stale basis
        _HS = h
    return _HS


class _Run:
    """State of ONE sequential-LP run.  The HiGHS model is built ONCE and thereafter updated IN
    PLACE -- `changeColsBounds` for the trust region, `changeCoeff` for the linearisation
    directions that actually moved, `addRows` for newly-near pairs.  Nothing is ever re-passed, so
    HiGHS keeps its own basis and factorisation internally and every LP is a warm dual re-solve.

    (Iteration 3 carried the basis by hand across `passModel`, but any growth of the pair set nulled
    it -- and after a perturbation the pair set grows on nearly every step, so a screen's LPs were
    effectively cold: 11.4 ms each at n=99 against 0.7 ms on an unchanged model.)

    Rows stay ordered [4n wall rows | pair rows] with pairs appended at the end, so a row index never
    changes meaning.  A STALE direction u is still a valid restriction -- ||c_i-c_j|| >= u.(c_i-c_j)
    holds for ANY unit u (Cauchy-Schwarz) -- it is merely conservative, so directions that drifted by
    less than _UTOL are left alone."""

    __slots__ = ("n", "mask", "I", "J", "ux", "uy", "nrow", "w")

    def __init__(self, n):
        self.n = n
        self.w = None
        self.mask = np.zeros((n, n), dtype=bool)
        self.I = np.empty(0, dtype=np.int64)
        self.J = np.empty(0, dtype=np.int64)
        self.ux = np.empty(0, dtype=float)
        self.uy = np.empty(0, dtype=float)
        self.nrow = 0


_UTOL = 1e-4          # leave a linearisation direction in place until it drifts this far
_OWNER = None         # the _Run whose model is currently loaded in the shared HiGHS instance


def _pair_rows(n, I, J, ux, uy):
    """Rowwise (indices, values) for the pair rows: r_i + r_j - u.(c_i - c_j) <= 0."""
    cols = np.stack([I, J, n + I, n + J, 2 * n + I, 2 * n + J], axis=1).ravel()
    vals = np.stack([-ux, ux, -uy, uy, np.ones(I.shape[0]), np.ones(I.shape[0])], axis=1).ravel()
    return cols.astype(np.int32), vals


def _build_model(h, n, I, J, ux, uy, lb, ub, w=None):
    m = I.shape[0]
    idx = np.arange(n)
    wr, wc, wv = [], [], []
    for k, (sgn, off) in enumerate(((1.0, 0), (-1.0, 0), (1.0, n), (-1.0, n))):
        base = k * n + idx
        wr.append(np.repeat(base, 2))
        wc.append(np.stack([off + idx, 2 * n + idx], axis=1).ravel())
        wv.append(np.stack([np.full(n, sgn), np.ones(n)], axis=1).ravel())
    cols, vals = _pair_rows(n, I, J, ux, uy)
    rows = 4 * n + np.repeat(np.arange(m), 6)
    nrow = 4 * n + m
    A = coo_matrix((np.concatenate(wv + [vals]),
                    (np.concatenate(wr + [rows]), np.concatenate(wc + [cols.astype(np.int64)]))),
                   shape=(nrow, 3 * n)).tocsc()
    lp = highspy.HighsLp()
    lp.num_col_ = 3 * n
    lp.num_row_ = nrow
    cw = np.ones(n) if w is None else np.asarray(w, dtype=float)
    lp.col_cost_ = np.concatenate([np.zeros(2 * n), -cw])
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = np.full(nrow, -1e30)
    lp.row_upper_ = np.concatenate([np.full(4 * n, 0.5), np.zeros(m)])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    h.passModel(lp)
    return nrow


def _slp_step(x, y, r, tr, margin, run=None, w=None, rcap=None):
    """One convex-restriction LP.  Returns (x, y, r) or None if the LP failed.

    `rcap` is a per-circle UPPER BOUND on the radius column.  Left None it is the plain 0.5 and
    the LP is the usual one; set below the current radii for a chosen few it turns the same LP
    into a different question -- 'best packing in which THESE circles stay small' -- whose optimum
    lives in another contact topology.  It rides the trust-region bound update, so it costs
    nothing extra and works on the warm-basis path."""
    global _OWNER
    n = x.shape[0]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    need = np.triu(d <= r[:, None] + r[None, :] + margin, 1)

    lo_c = np.concatenate([x - tr, y - tr, np.zeros(n)])
    up_c = np.concatenate([x + tr, y + tr,
                           np.full(n, 0.5) if rcap is None else np.asarray(rcap, float)])

    if run is None or not _HAVE_HS:
        I, J = np.nonzero(need)
        m = I.shape[0]
        dij = np.maximum(d[I, J], 1e-12)
        ux = (x[I] - x[J]) / dij
        uy = (y[I] - y[J]) / dij
        if not _HAVE_HS:                       # pragma: no cover -- scipy fallback path
            idx = np.arange(n)
            wr, wc, wv = [], [], []
            for k, (sgn, off) in enumerate(((1.0, 0), (-1.0, 0), (1.0, n), (-1.0, n))):
                base = k * n + idx
                wr.append(np.repeat(base, 2))
                wc.append(np.stack([off + idx, 2 * n + idx], axis=1).ravel())
                wv.append(np.stack([np.full(n, sgn), np.ones(n)], axis=1).ravel())
            cols, vals = _pair_rows(n, I, J, ux, uy)
            rows = 4 * n + np.repeat(np.arange(m), 6)
            A = coo_matrix((np.concatenate(wv + [vals]),
                            (np.concatenate(wr + [rows]),
                             np.concatenate(wc + [cols.astype(np.int64)]))),
                           shape=(4 * n + m, 3 * n)).tocsc()
            try:
                res = linprog(np.concatenate([np.zeros(2 * n),
                                          -(np.ones(n) if w is None else np.asarray(w, float))]), A_ub=A,
                              b_ub=np.concatenate([np.full(4 * n, 0.5), np.zeros(m)]),
                              bounds=np.stack([lo_c, up_c], axis=1), method="highs")
            except Exception:
                return None
            if not res.success or res.x is None:
                return None
            z = res.x
            return z[:n].copy(), z[n:2 * n].copy(), z[2 * n:].copy()
        h = _hs()
        _build_model(h, n, I, J, ux, uy, lo_c, up_c, w)
        _OWNER = None
    else:
        h = _hs()
        fresh = need & ~run.mask
        rebuild = _OWNER is not run
        if rebuild:
            run.mask |= fresh
            I, J = np.nonzero(run.mask)
            run.I, run.J = I, J
            dij = np.maximum(d[I, J], 1e-12)      # coincident centres would give NaN directions
            run.ux = (x[I] - x[J]) / dij
            run.uy = (y[I] - y[J]) / dij
            run.nrow = _build_model(h, n, I, J, run.ux, run.uy, lo_c, up_c, w)
            run.w = None if w is None else np.asarray(w, dtype=float).copy()
            _OWNER = run
        else:
            # (a) refresh only the directions that actually moved -- a stale u is still valid
            m0 = run.I.shape[0]
            if m0:
                dij = np.maximum(d[run.I, run.J], 1e-12)
                nux = (x[run.I] - x[run.J]) / dij
                nuy = (y[run.I] - y[run.J]) / dij
                move = np.nonzero((np.abs(nux - run.ux) > _UTOL) |
                                  (np.abs(nuy - run.uy) > _UTOL))[0]
                for p in move:
                    ip, jp, a, b = int(run.I[p]), int(run.J[p]), float(nux[p]), float(nuy[p])
                    rw = 4 * n + int(p)
                    h.changeCoeff(rw, ip, -a)
                    h.changeCoeff(rw, jp, a)
                    h.changeCoeff(rw, n + ip, -b)
                    h.changeCoeff(rw, n + jp, b)
                    run.ux[p] = a
                    run.uy[p] = b
            # (b) append newly-near pairs; HiGHS keeps the basis (new slacks enter basic)
            if fresh.any():
                fi, fj = np.nonzero(fresh)
                run.mask |= fresh
                fd = np.maximum(d[fi, fj], 1e-12)
                fux = (x[fi] - x[fj]) / fd
                fuy = (y[fi] - y[fj]) / fd
                cols, vals = _pair_rows(n, fi, fj, fux, fuy)
                k = fi.shape[0]
                h.addRows(k, np.full(k, -1e30), np.zeros(k), 6 * k,
                          (6 * np.arange(k)).astype(np.int32), cols, vals)
                run.I = np.concatenate([run.I, fi])
                run.J = np.concatenate([run.J, fj])
                run.ux = np.concatenate([run.ux, fux])
                run.uy = np.concatenate([run.uy, fuy])
                run.nrow += k
            # (c) objective weights -- the TILT kernel re-costs the radii in place, which
            #     keeps the basis and costs one call instead of a rebuild
            cw = None if w is None else np.asarray(w, dtype=float)
            same = (run.w is None and cw is None) or (
                run.w is not None and cw is not None and run.w.shape == cw.shape
                and bool(np.array_equal(run.w, cw)))
            if not same:
                nw = np.ones(n) if cw is None else cw
                h.changeColsCost(n, np.arange(2 * n, 3 * n, dtype=np.int32), -nw)
                run.w = None if cw is None else cw.copy()
            # (d) trust region
            h.changeColsBounds(3 * n, np.arange(3 * n, dtype=np.int32), lo_c, up_c)

    try:
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            _OWNER = None
            return None
        z = np.asarray(h.getSolution().col_value, dtype=float)
    except Exception:
        _OWNER = None
        return None
    if z.shape[0] != 3 * n or not np.all(np.isfinite(z)):
        _OWNER = None
        return None
    return z[:n].copy(), z[n:2 * n].copy(), z[2 * n:].copy()


def _slp(x, y, r, tr0=0.02, max_lp=400, deadline=None, trmin=1e-7, maxfail=64, rtol=0.0,
         rcap=None):
    """Iterate the restriction LP to a KKT point.  Monotone in sum(r) by construction.

    Two control knobs make this serve as BOTH a cheap screen and an exact polish:
      trmin/maxfail -- how far the trust region is driven down before giving up;
      rtol          -- relative-gain floor below which the run is declared converged.
    The trust region GROWS on success and reverts-to-best on failure, so the improving phase
    needs far fewer LPs than a pure halving schedule (which spent ~18 of its ~26 LPs in a tail
    worth ~1e-10)."""
    tr = tr0
    run = _Run(x.shape[0])
    best = float(r.sum())
    bx, by, br = x, y, r
    fails = 0
    for _ in range(max_lp):
        if deadline is not None and time.process_time() > deadline:
            break
        out = _slp_step(x, y, r, tr, 2.0 * tr + 0.012, run, rcap=rcap)
        if out is None:
            break
        x, y, r = out
        r = _repair(x, y, r)
        s = float(r.sum())
        if s > best + max(1e-13, rtol * best):
            best, bx, by, br = s, x, y, r
            tr = min(tr * 1.7, 4.0 * tr0)
            fails = 0
        else:
            x, y, r = bx.copy(), by.copy(), br.copy()   # revert: only descend from the incumbent
            tr *= 0.35
            fails += 1
            if tr < trmin or fails >= maxfail:
                break
    return bx, by, br, best


def _screen(x, y, r, deadline):
    """Cheap local solve used inside basin hopping -- stops as soon as gains go below ~1e-7 rel."""
    return _slp(x, y, r, tr0=0.02, max_lp=60, deadline=deadline,
                trmin=6e-4, maxfail=3, rtol=1e-8)


def _polish(x, y, r, deadline):
    """Exact local solve, run only on candidates that already beat the incumbent."""
    return _slp(x, y, r, tr0=6e-3, max_lp=400, deadline=deadline, trmin=1e-7, maxfail=64)


# ---------------------------------------------------------------- starts & moves

def _read_pack(n):
    """Warm start from the committed census, or None.  Existence-guarded: solve() is called with
    arbitrary targets, including n that have never been packed."""
    p = os.path.join(PACK_DIR, "csqv%d.pck" % n)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        rows = []
        for ln in lines[2:]:
            parts = ln.split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        a = np.asarray(rows, dtype=float)
        if a.shape != (n, 3) or not np.all(np.isfinite(a)):
            return None
        return a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy()
    except Exception:
        return None


_RECS = None


def _records():
    """The read-only record table, cached.  Guarded: solve() must work with no records file at
    all (an offline re-run elsewhere), in which case every target is simply treated as unsolved."""
    global _RECS
    if _RECS is None:
        try:
            with open(os.path.join("bench", "records.json")) as fh:
                _RECS = {int(k): float(v) for k, v in json.load(fh)["records"].items()}
        except Exception:
            _RECS = {}
    return _RECS


def _cap_target(n):
    """The sum-of-radii at which n earns the full 7 digits (relgap <= 1e-7), or None if unknown.
    A small safety factor keeps us clear of the scorer's own re-derivation."""
    R = _records().get(int(n))
    return None if R is None else R * (1.0 - 8e-8)


def _grid_start(n, k, rng):
    cs = (np.arange(k) + 0.5) / k - 0.5
    gx, gy = np.meshgrid(cs, cs)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    if pts.shape[0] > n:
        pts = pts[rng.choice(pts.shape[0], n, replace=False)]
    elif pts.shape[0] < n:
        pts = np.concatenate([pts, rng.uniform(LO + 0.02, HI - 0.02, (n - pts.shape[0], 2))])
    pts = pts + rng.normal(0.0, 0.002, pts.shape)
    return np.clip(pts[:, 0], LO, HI), np.clip(pts[:, 1], LO, HI)


def _hex_start(n, rng):
    """Shifted-row (hexagonal-ish) start -- a topology class the square grids cannot reach.
    Rows of alternating offset; the row count is chosen so the cells are roughly square."""
    rows = max(1, int(round(math.sqrt(n / 1.1547))))
    per = [n // rows + (1 if i < n % rows else 0) for i in range(rows)]
    pts = []
    for i, cnt in enumerate(per):
        yy = LO + (i + 0.5) / rows
        off = 0.5 * (i % 2) / max(cnt, 1)
        for j in range(cnt):
            pts.append((LO + ((j + 0.5) / cnt + off) % 1.0, yy))
    p = np.asarray(pts, dtype=float) + rng.normal(0.0, 0.002, (n, 2))
    return np.clip(p[:, 0], LO, HI), np.clip(p[:, 1], LO, HI)


def _rand_start(n, rng):
    return rng.uniform(LO + 0.05, HI - 0.05, n), rng.uniform(LO + 0.05, HI - 0.05, n)


def _hole_points(x, y, r, rng, want):
    """Sample points and keep those with the largest insertable radius -- the empty holes."""
    m = max(256, 24 * x.shape[0])
    px = rng.uniform(LO, HI, m)
    py = rng.uniform(LO, HI, m)
    d = np.sqrt((px[:, None] - x[None, :]) ** 2 + (py[:, None] - y[None, :]) ** 2) - r[None, :]
    cap = np.minimum(d.min(axis=1), _wallcap(px, py))
    order = np.argsort(-cap)[:want]
    return px[order], py[order]


def _greedy_insert(x, y, r, k, rng, top=1):
    """Place k new circles one at a time, each at the sampled point with the largest insertable
    radius.  Capacities are updated after every placement, so the k circles are packed into
    genuinely distinct holes rather than all onto the single best one.

    `top` > 1 picks uniformly among the `top` best holes instead of the single best -- the same
    machinery then yields DISTINCT insertion topologies from one donor, which is what makes a
    transfer ticket a lottery draw rather than a deterministic (and already-failed) start."""
    n0 = x.shape[0]
    m = max(512, 40 * (n0 + k))
    px = rng.uniform(LO, HI, m)
    py = rng.uniform(LO, HI, m)
    cap = _wallcap(px, py)
    if n0:
        d = np.sqrt((px[:, None] - x[None, :]) ** 2 + (py[:, None] - y[None, :]) ** 2) - r[None, :]
        cap = np.minimum(cap, d.min(axis=1))
    nx = np.empty(k); ny = np.empty(k); nr = np.empty(k)
    for t in range(k):
        if top <= 1:
            i = int(np.argmax(cap))
        else:
            cand = np.argsort(-cap)[:top]
            i = int(cand[rng.randint(cand.shape[0])])
        nx[t], ny[t] = px[i], py[i]
        nr[t] = max(float(cap[i]), 1e-6)
        cap = np.minimum(cap, np.sqrt((px - nx[t]) ** 2 + (py - ny[t]) ** 2) - nr[t])
    return (np.concatenate([x, nx]), np.concatenate([y, ny]), np.concatenate([r, nr]))


def _contacts(x, y, r, tol=1e-6):
    """Active-contact count per circle (touching neighbours + touching walls).  A circle with few
    contacts is a rattler the LP is free to move; dropping it changes the contact graph far less
    than dropping a load-bearing one -- which makes "fewest contacts" a genuinely different
    reduction rule from "smallest radius"."""
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2) - r[:, None] - r[None, :]
    np.fill_diagonal(d, 1e9)
    c = (d < tol).sum(axis=1).astype(float)
    c += (x - LO - r < tol).astype(float) + (HI - x - r < tol).astype(float)
    c += (y - LO - r < tol).astype(float) + (HI - y - r < tol).astype(float)
    return c


def _transfer_starts(n, rng, max_donors=10, nrules=3):
    """CROSS-n TOPOLOGY TRANSFER -- the one start that is not derived from n's own history.

    Every other start in this solver is either a synthetic lattice or a perturbation of my own
    committed pack for n, so the whole portfolio shares one failure mode: it can only reach the
    basins reachable from where n already is.  But the census holds packings for the NEIGHBOURING
    sizes, and many of those sit exactly ON the record -- their contact graph is the real optimal
    topology.  A packing for m and one for n = m +/- k differ by k circles, so:

        n < m : drop the (m-n) SMALLEST circles (the fillers) and let the LP re-inflate the rest
        n > m : greedily insert (n-m) circles into the largest holes

    yields a start carrying m's topology.  It usually loses to the warm start -- and that is fine,
    it costs ~0.05 CPU-s and the portfolio keeps the max, not the mean.

    The window is +/- _DONOR_WIN, not +/- 2 neighbours: once most of the census sits on the record
    a NEAR donor and a FAR donor are equally authentic topologies, and they differ only in how big
    an edit the transfer is.  A big edit from a far donor is a worse start on average and a
    different basin either way -- which is the only property a lottery ticket has to have.  Donors
    are still ranked record-topology-first, then nearest, and tickets are emitted rule-major, so
    the cheapest, most-proven draws are always the ones spent first when the slice is short.  When it wins it wins by
    landing in a basin local search cannot walk to.  It also COMPOUNDS: every n driven to the cap
    becomes a donor of record topology to its neighbours on the next iteration.

    Fully existence-guarded: an n with no committed neighbour (a fresh n, an offline re-run on
    unseen sizes) simply gets an empty list and the cold portfolio is untouched.
    """
    if n < 1:
        return []
    cand = []
    for m in range(n - _DONOR_WIN, n + _DONOR_WIN + 1):
        if m == n or m < 1:
            continue
        p = _read_pack(m)
        if p is None or p[0].shape[0] != m:
            continue
        cap = _cap_target(m)
        at_cap = 1 if (cap is not None and float(p[2].sum()) >= cap) else 0
        cand.append((-at_cap, abs(m - n), m, p))
    cand.sort(key=lambda t: (t[0], t[1], t[2]))          # record-topology donors first, then near
    per_donor = []
    for _, _, m, p in cand[:max_donors]:
        tix = []
        for rule in range(nrules):
            x, y, r = p[0].copy(), p[1].copy(), p[2].copy()
            k = abs(m - n)
            if n < m:
                if rule == 0:                            # drop the k SMALLEST (the fillers)
                    keep = np.argsort(-r)[:n]
                elif rule == 1:                          # drop the k LOOSEST (fewest contacts)
                    c = _contacts(x, y, r)
                    keep = np.argsort(c + 1e-3 * (r / max(float(r.max()), 1e-12)))[k:]
                else:                                    # drop k at random, biased small
                    w = 1.0 / np.maximum(r, 1e-9) ** 2
                    drop = rng.choice(m, k, replace=False, p=w / w.sum())
                    keep = np.setdiff1d(np.arange(m), drop)
                x, y, r = x[keep], y[keep], r[keep]
            elif n > m:
                x, y, r = _greedy_insert(x, y, r, k, rng, top=1 if rule == 0 else 3 * rule)
            elif rule > 0:
                continue                                 # m == n never happens; keep it total
            if x.shape[0] != n:
                continue
            tix.append((x, y, _repair(x, y, r)))
        per_donor.append(tix)
    # rule-major order: every donor's PROVEN rule-0 ticket first, then the exploratory variants.
    out = []
    for i in range(nrules):
        for tix in per_donor:
            if i < len(tix):
                out.append(tix[i])
    return out


# ---------------------------------------------------------------- SYMMETRY PROJECTION SOURCE
# Measured on my OWN census (artifacts/probe_sym.py), not assumed: of the 36 committed packings
# that sit ON the record, five are exactly (or near-exactly) invariant under a square symmetry --
# n=27 anti-diagonal and n=49 under a whole order-4 group at residual 0.0000, n=33/41/43/45 within
# 0.15 mean radii -- while the remaining thirty are amorphous at ~1 mean radius of residual.  So
# symmetry is not a universal prior; it is a BIMODAL one, which is exactly the shape a lottery
# ticket wants.  It is usually wrong, it costs a few CPU-s, and where it is right it hands over a
# configuration that no amount of asymmetric hopping will ever walk to -- the symmetric layouts
# form a measure-zero subspace that every other source in this file has probability zero of
# entering.
_SYM_OPS = [(np.array([[-1.0, 0.0], [0.0, 1.0]]), 2),      # mirror across the vertical axis
            (np.array([[1.0, 0.0], [0.0, -1.0]]), 2),      # mirror across the horizontal axis
            (np.array([[0.0, 1.0], [1.0, 0.0]]), 2),       # diagonal
            (np.array([[0.0, -1.0], [-1.0, 0.0]]), 2),     # anti-diagonal
            (np.array([[-1.0, 0.0], [0.0, -1.0]]), 2),     # rotation by 180 degrees
            (np.array([[0.0, -1.0], [1.0, 0.0]]), 4)]      # rotation by 90 degrees


def _greedy_assign(C):
    """Cheap permutation minimising sum C[i, sig[i]] -- globally-smallest-first, n^2 log n."""
    n = C.shape[0]
    sig = -np.ones(n, dtype=int)
    used = np.zeros(n, dtype=bool)
    for idx in np.argsort(C, axis=None):
        i, j = divmod(int(idx), n)
        if sig[i] >= 0 or used[j]:
            continue
        sig[i] = j
        used[j] = True
    return sig


def _symmetrize(x, y, r, M, K):
    """Project a layout onto the nearest arrangement invariant under the group generated by M.

    Circles are first matched to their own images (greedy, on displacement plus radius mismatch),
    giving a permutation sigma with p_{sigma(i)} ~= M p_i.  Each CYCLE of sigma is then replaced by
    a genuine orbit of M, so the result is invariant by construction rather than merely closer to
    invariant.  A cycle whose length L leaves M^L != I -- a fixed point of any op, or a 2-cycle
    under the 90-degree rotation -- is additionally averaged over <M^L>, which is what pins those
    circles onto the mirror axis / the centre instead of leaving them off-orbit.
    """
    n = x.shape[0]
    P = np.stack([x, y], axis=1)
    Q = P @ M.T
    C = ((Q[:, None, :] - P[None, :, :]) ** 2).sum(-1) + (4.0 * (r[:, None] - r[None, :])) ** 2
    sig = _greedy_assign(C)
    Mi = M.T                                       # every square symmetry is orthogonal
    out = P.copy()
    rr = np.array(r, dtype=float, copy=True)
    seen = np.zeros(n, dtype=bool)
    for i in range(n):
        if seen[i]:
            continue
        cyc = [i]
        seen[i] = True
        j = int(sig[i])
        while j != i and 0 <= j < n and not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = int(sig[j])
        L = len(cyc)
        acc = np.zeros(2)
        A = np.eye(2)
        for idx in cyc:                            # q = mean_m M^-m p_{cyc[m]}
            acc += A.dot(P[idx])
            A = Mi.dot(A)
        q = acc / L
        t = K // math.gcd(K, L)                    # order of M^L inside <M>
        if t > 1:
            ML = np.linalg.matrix_power(M, L)
            qa = np.zeros(2)
            A = np.eye(2)
            for _ in range(t):
                qa += A.dot(q)
                A = ML.dot(A)
            q = qa / t
        rq = float(rr[cyc].mean())
        A = np.eye(2)
        for idx in cyc:                            # lay the orbit back down: p_{cyc[m]} = M^m q
            out[idx] = A.dot(q)
            rr[idx] = rq
            A = M.dot(A)
    ox = np.clip(out[:, 0], LO, HI)
    oy = np.clip(out[:, 1], LO, HI)
    return ox, oy, _repair(ox, oy, rr)


def _sym_push(x, y, r, M, K, deadline, rounds=6):
    """ALTERNATING PROJECTION: symmetrise, let the LP climb a little, symmetrise again.

    Seeding a hop chain from a single symmetrised layout is not enough -- the free LP breaks the
    symmetry on its very first step and the chain slides straight back into an amorphous basin,
    which is the same failure that made plain transfer-and-polish useless in iteration 8.  
    Alternating the two keeps the iterate ON the symmetric subspace while sum(r) climbs, so the
    hop chain starts from a symmetric LOCAL OPTIMUM rather than from a symmetric guess.
    """
    for _ in range(max(1, rounds)):
        if time.process_time() >= deadline:
            break
        x, y, r = _symmetrize(x, y, r, M, K)
        x, y, r, _ = _slp(x, y, r, tr0=0.01, max_lp=12, deadline=deadline,
                          trmin=1e-4, maxfail=3, rtol=1e-9)
    return _symmetrize(x, y, r, M, K)


_TGT_HS = None


def _tgt_hs():
    """A PRIVATE HiGHS instance for the elastic target LP.  Kept apart from `_hs()` so a target
    push never invalidates the shared warm basis that `_slp` depends on."""
    global _TGT_HS
    if _TGT_HS is None:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("presolve", "off")
        h.setOptionValue("solver", "simplex")
        _TGT_HS = h
    return _TGT_HS


def _target_push(x, y, r, T, deadline, rng, steps=40, tr0=0.03):
    """TARGET FOLLOWING -- change what the machinery is asked to optimise.

    Every other source in this solver samples the SAME distribution: perturb, then MAXIMISE
    sum(r) locally.  Measured on n=81, that distribution is a dense cloud of basin optima near
    relgap 1e-3 whose spread is ~0.002 in sum_r, and the record sits ~1.8 spreads above the best
    of hundreds of draws -- so no amount of extra draws from that generator reaches it.

    So this one does not maximise sum(r) at all.  It FIXES sum(r) >= T (a value ABOVE every local
    optimum ever found) as a hard row, makes the pair constraints ELASTIC with slacks s_k >= 0,
    and minimises sum(s) -- phase-1 on an infeasible target.  The walls stay hard, so the iterate
    is always a genuine arrangement of n circles whose radii already sum past the record; the only
    thing wrong with it is overlap, and the LP is spending all its freedom removing that.

    The point is not that sum(s) reaches 0 (it usually will not).  It is that the resulting
    POSITIONS are drawn from a different distribution than any local ascent produces: they are the
    layout that comes closest to admitting a record-sized packing.  `_repair` + `_polish`
    afterwards then descends from there, which is a start no perturbation of an incumbent can
    reach.  Feasibility of the returned packing is never at risk -- the caller repairs and
    re-polishes, and only the metered oracle decides what is kept."""
    n = x.shape[0]
    if n < 3 or not _HAVE_HS:
        return x.copy(), y.copy(), r.copy()
    x, y, r = x.copy(), y.copy(), r.copy()
    h = _tgt_hs()
    tr = float(tr0)
    idx = np.arange(n)
    best = None
    prev, stall = np.inf, 0
    for _ in range(int(steps)):
        if time.process_time() > deadline:
            break
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        margin = 2.0 * tr + 0.02
        need = np.triu(d <= r[:, None] + r[None, :] + margin, 1)
        I, J = np.nonzero(need)
        m = I.shape[0]
        if m == 0:
            break
        dij = np.maximum(d[I, J], 1e-12)
        ux = (x[I] - x[J]) / dij
        uy = (y[I] - y[J]) / dij

        # rows: [4n wall | m elastic pair | 1 target] ; cols: [x | y | r | s]
        wr, wc, wv = [], [], []
        for k, (sgn, off) in enumerate(((1.0, 0), (-1.0, 0), (1.0, n), (-1.0, n))):
            base = k * n + idx
            wr.append(np.repeat(base, 2))
            wc.append(np.stack([off + idx, 2 * n + idx], axis=1).ravel())
            wv.append(np.stack([np.full(n, sgn), np.ones(n)], axis=1).ravel())
        cols, vals = _pair_rows(n, I, J, ux, uy)
        prows = 4 * n + np.repeat(np.arange(m), 6)
        srows = 4 * n + np.arange(m)                 # the -s_k coefficient of each pair row
        scols = 3 * n + np.arange(m)
        trow = np.full(n, 4 * n + m)
        nrow = 4 * n + m + 1
        ncol = 3 * n + m
        A = coo_matrix((np.concatenate(wv + [vals, -np.ones(m), np.ones(n)]),
                        (np.concatenate(wr + [prows, srows, trow]),
                         np.concatenate(wc + [cols.astype(np.int64), scols, 2 * n + idx]))),
                       shape=(nrow, ncol)).tocsc()
        lp = highspy.HighsLp()
        lp.num_col_ = ncol
        lp.num_row_ = nrow
        lp.col_cost_ = np.concatenate([np.zeros(3 * n), np.ones(m)])
        lp.col_lower_ = np.concatenate([x - tr, y - tr, np.zeros(n), np.zeros(m)])
        lp.col_upper_ = np.concatenate([x + tr, y + tr, np.full(n, 0.5), np.full(m, 1e30)])
        lp.row_lower_ = np.concatenate([np.full(4 * n + m, -1e30), [float(T)]])
        lp.row_upper_ = np.concatenate([np.full(4 * n, 0.5), np.zeros(m), [1e30]])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        try:
            h.passModel(lp)
            h.run()
            if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                break
            z = np.asarray(h.getSolution().col_value, dtype=float)
        except Exception:
            break
        if z.shape[0] != ncol or not np.all(np.isfinite(z)):
            break
        viol = float(z[3 * n:].sum())
        x, y, r = z[:n].copy(), z[n:2 * n].copy(), z[2 * n:3 * n].copy()
        x = np.clip(x, LO, HI)
        y = np.clip(y, LO, HI)
        if best is None or viol < best[0]:
            best = (viol, x.copy(), y.copy(), r.copy())
        if viol <= 1e-11:
            break
        # The trust region is deliberately NOT driven to zero.  Phase-1 here converges in ~2 LPs
        # to a stationary point of the violation (measured: move -> 0.00000 while viol sticks at
        # 1.2e-2), and shrinking tr only freezes it there.  Hold the region and jitter out of the
        # stall instead -- the escape, not the tolerance, is what this routine is for.
        tr = max(tr * 0.97, 0.004)
        if abs(viol - prev) <= 1e-9 * max(viol, 1e-12):
            stall += 1
            if stall >= 2:
                sc = 0.004 * (1.0 + stall)
                x = np.clip(x + rng.normal(0.0, sc, n), LO, HI)
                y = np.clip(y + rng.normal(0.0, sc, n), LO, HI)
                stall = 0
        else:
            stall = 0
        prev = viol
    if best is None:
        return x, y, r
    return best[1], best[2], best[3]


def _ladder_push(x, y, r, T_hi, deadline, rng, rungs=6, over=1.004):
    """CONTINUATION LADDER -- target following as a homotopy instead of a leap.

    `_target_push` pins sum(r) >= T at the RECORD in one shot and minimises overlap.  That target
    sits ~1.8 basin-spreads above anything the ascent sources reach, so the phase-1 LP starts deep
    inside an infeasible region and, measured, converges in ~2 LPs to a stationary point of the
    violation: the layout it hands back is the closest thing to a record-sized packing *reachable
    from that seed in one jump*, and its returns on the last stuck n decayed from 0.20 digits to
    0.03 across two iterations.

    The ladder asks the same question in rungs.  T is raised from the seed's own sum(r) toward
    (slightly past) the record in a few steps, and BETWEEN rungs the iterate is repaired and
    screened back to a genuinely feasible local optimum, which is where the next rung starts.
    Two things follow that a single jump cannot get: the violation solved at each rung is small
    enough that the LP is still moving when it stops, and the path is a chain of real packings, so
    the ladder ratchets -- it can only report a layout at least as good as its seed.  Overshooting
    (`over` > 1) keeps the last rung pulling PAST the record rather than relaxing onto it.

    T_hi may be NaN/None for an n with no known record; the schedule then climbs a fixed relative
    step above the seed, so this source works on sizes the census has never seen."""
    n = x.shape[0]
    if n < 3 or not _HAVE_HS:
        return np.clip(x, LO, HI), np.clip(y, LO, HI), _repair(x, y, r)
    x = np.clip(np.asarray(x, float).copy(), LO, HI)
    y = np.clip(np.asarray(y, float).copy(), LO, HI)
    r = _repair(x, y, r)
    s0 = float(r.sum())
    Tend = float(T_hi) * float(over) if T_hi is not None else float("nan")
    if not np.isfinite(Tend) or Tend <= s0:
        Tend = s0 * (1.0 + 3e-3)               # no record for this n: climb a fixed relative step
    best = (s0, x.copy(), y.copy(), r.copy())
    cur = (x, y, r)
    rungs = max(2, int(rungs))
    for k in range(rungs):
        now = time.process_time()
        if now >= deadline:
            break
        share = (deadline - now) / float(rungs - k)
        T = s0 + (Tend - s0) * ((k + 1.0) / rungs)
        px, py, pr = _target_push(cur[0], cur[1], cur[2], T,
                                  min(deadline, now + 0.62 * share), rng, steps=10, tr0=0.022)
        pr = _repair(px, py, pr)
        qx, qy, qr, qs = _screen(px, py, pr,
                                 min(deadline, time.process_time() + 0.45 * share))
        if not np.isfinite(qs):
            break
        if qs > best[0]:
            best = (float(qs), qx.copy(), qy.copy(), qr.copy())
        cur = (qx, qy, qr)      # RATCHET: the next rung climbs from a feasible local optimum
    return best[1], best[2], best[3]


def _ruin(x, y, r, rng, frac, deadline):
    """RUIN & RECREATE -- the move the jitter kernels structurally cannot make.

    Delete k circles OUTRIGHT, re-solve the (n-k)-circle subproblem so the survivors actually
    spread into the vacated space, then greedily re-insert k circles into the largest holes.
    The existing 'relocate smallest into holes' kernel moves a circle but keeps all n present,
    so the neighbours never get the intermediate relaxation -- and it is exactly that relaxation
    that rewires the contact graph globally instead of locally.

    Two victim rules: a SPATIAL ruin (a contiguous disc of circles) opens one large connected
    void and is the strong topology move; a SMALLEST ruin recycles filler circles."""
    n = x.shape[0]
    k = int(min(max(1, int(round(frac * n))), max(1, n // 3), n - 2))
    if k < 1:
        return x.copy(), y.copy(), r.copy()
    if rng.rand() < 0.6:                                  # spatial: a disc of neighbours
        i = int(rng.randint(n))
        d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
        victims = np.argsort(d2)[:k]
    else:                                                 # recycle the filler circles
        victims = np.argsort(r)[:k]
    keep = np.ones(n, dtype=bool)
    keep[victims] = False
    sx, sy, sr = x[keep].copy(), y[keep].copy(), r[keep].copy()
    if sx.shape[0] >= 2:
        # a SHORT relaxation: enough for the survivors to move into the void, not so long that
        # they close it entirely and leave no room for the k circles coming back.
        sx, sy, sr, _ = _slp(sx, sy, sr, tr0=0.02, max_lp=4, deadline=deadline,
                             trmin=4e-3, maxfail=1, rtol=1e-6)
    return _greedy_insert(sx, sy, sr, k, rng)


def _tilt(x, y, r, rng, deadline, steps=None):
    """OBJECTIVE TILT -- a topology move made in the LP's own space, not in geometry.

    Every geometric kernel (jitter, kick, relocate, swap, ruin) proposes a new *position* vector
    and lets the SLP decide the radii.  But the committed packs are exact KKT points -- a
    generous re-polish moves them by 0.0000 digits -- so the residual loss is entirely which
    contact topology the packing settled into, and jitter re-descends into the same one.

    Here the packing is instead pushed with a DIFFERENT objective, max sum_i w_i r_i, for a few
    LP steps.  Circles with large w_i inflate and push their neighbours aside; circles with small
    w_i give way.  The contact graph is rewired *by the LP itself*, which is the only agent in
    this solver that knows which constraints are active.  Restoring w = 1 afterwards then lets
    the true objective descend into a genuinely different basin.  Feasibility is untouched: the
    restriction LP is feasible-by-construction for ANY cost vector.

    Four weight fields, chosen because they are the structural asymmetries a random jitter cannot
    express: a linear gradient across the square, a local blob, an inversion of the size ranking,
    and iid noise."""
    n = x.shape[0]
    if n < 3:
        return x.copy(), y.copy(), r.copy()
    a = float(rng.uniform(0.15, 0.9))
    m = rng.rand()
    if m < 0.3:                                   # linear gradient: one side of the square inflates
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        t = math.cos(th) * x + math.sin(th) * y
        lo, hi = float(t.min()), float(t.max())
        f = (t - lo) / max(hi - lo, 1e-12)
    elif m < 0.6:                                 # blob: one local cluster inflates
        i = int(rng.randint(n))
        sg = float(rng.uniform(0.06, 0.25))
        f = np.exp(-((x - x[i]) ** 2 + (y - y[i]) ** 2) / (2.0 * sg * sg))
    elif m < 0.85:                                # invert the size hierarchy: fillers become bosses
        f = np.empty(n)
        f[np.argsort(r)] = np.linspace(1.0, 0.0, n)
    else:                                         # iid
        f = rng.rand(n)
    w = 1.0 + a * f
    if steps is None:
        steps = int(rng.randint(2, 6))
    tr = float(rng.uniform(0.015, 0.045))
    run = _Run(n)
    for _ in range(steps):
        if deadline is not None and time.process_time() > deadline:
            break
        out = _slp_step(x, y, r, tr, 2.0 * tr + 0.012, run, w)
        if out is None:
            break
        x, y, r = out
        r = _repair(x, y, r)
    return np.clip(x, LO, HI), np.clip(y, LO, HI), r


def _squeeze(x, y, r, rng, deadline):
    """SQUEEZE -- a DIRECTED topology move made in the LP's own constraint set.

    Every other kernel in this file moves COORDINATES and hopes the contact graph changes as a
    side effect; measured on the last stuck n, 306 such hops moved sum(r) by exactly zero, which
    says the incumbent's topology is what is rigid, not its geometry.  This kernel changes the
    FEASIBLE SET instead: k circles get a radius ceiling well below what they currently hold, and
    the SAME solver is re-run under it, so its neighbours flow into the space those circles are
    forbidden to occupy and the contact graph re-forms around the hole.  Releasing the ceiling
    (the caller's screen/polish) then lets the squeezed circles regrow inside the NEW topology.

    `_ruin` deletes circles outright and re-inserts them greedily at hole centres -- a crude,
    random, and discontinuous move; this one is continuous in the constraint, so the layout it
    hands back is a genuine local optimum of a nearby problem rather than a guess."""
    n = x.shape[0]
    if n < 4 or not _HAVE_HS:
        return _perturb(x, y, r, rng, None)
    k = int(rng.randint(1, max(2, n // 12) + 1))
    # bias the victims toward well-connected circles: squeezing a circle with many contacts is
    # what actually opens a topology, while a loose one just shrinks and regrows unchanged.
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    np.fill_diagonal(d, np.inf)
    slack = d - (r[:, None] + r[None, :])
    deg = (slack < 1e-4).sum(axis=1) + (_wallcap(x, y) - r < 1e-4)
    pw = deg.astype(float) + 0.25
    pw /= pw.sum()
    who = rng.choice(n, size=min(k, n), replace=False, p=pw)
    alpha = float(rng.uniform(0.25, 0.80))
    cap = np.full(n, 0.5)
    cap[who] = alpha * r[who]
    half = time.process_time() + 0.5 * max(0.0, deadline - time.process_time())
    sx, sy, sr, _ = _slp(x.copy(), y.copy(), np.minimum(r, cap), tr0=0.02, max_lp=45,
                         deadline=half, trmin=6e-4, maxfail=3, rtol=1e-8, rcap=cap)
    # release the ceiling: regrow the squeezed circles inside the topology that formed without
    # them.  The caller screens/polishes from here, so one cheap uncapped pass is enough.
    sr = _repair(sx, sy, np.maximum(sr, 1e-9))
    return np.clip(sx, LO, HI), np.clip(sy, LO, HI), sr


def _perturb(x, y, r, rng, deadline=None, ruin_frac=0.06):
    """Four kernels.  Metric moves (global jitter, single-circle kick) change geometry only;
    topology moves (relocate-smallest-into-holes, size swap) change the contact graph, which is
    what actually decides which local optimum the LP falls into."""
    n = x.shape[0]
    if deadline is not None:
        u0 = rng.rand()
        if u0 < 0.18:                                     # ruin & recreate
            return _ruin(x, y, r, rng, ruin_frac, deadline)
        if u0 < 0.36:                                     # objective tilt (LP-space topology move)
            return _tilt(x, y, r, rng, deadline)
        if u0 < 0.50:                                     # squeeze (constraint-space topology move)
            return _squeeze(x, y, r, rng, deadline)
    x2, y2 = x.copy(), y.copy()
    u = rng.rand()
    if u < 0.30:                                     # global Gaussian jitter
        sc = float(np.exp(rng.uniform(math.log(0.003), math.log(0.05))))
        x2 += rng.normal(0.0, sc, n)
        y2 += rng.normal(0.0, sc, n)
    elif u < 0.50:                                   # kick a few circles hard, leave the rest
        k = int(rng.randint(1, max(2, n // 6) + 1))
        who = rng.choice(n, k, replace=False)
        sc = float(np.exp(rng.uniform(math.log(0.02), math.log(0.15))))
        x2[who] += rng.normal(0.0, sc, k)
        y2[who] += rng.normal(0.0, sc, k)
    elif u < 0.85:                                   # relocate the smallest circles into holes
        k = int(rng.randint(1, max(2, n // 8) + 1))
        victims = np.argsort(r)[:k]
        hx, hy = _hole_points(x, y, r, rng, k)
        take = min(k, hx.shape[0])
        x2[victims[:take]] = hx[:take]
        y2[victims[:take]] = hy[:take]
        x2 += rng.normal(0.0, 0.003, n)
        y2 += rng.normal(0.0, 0.003, n)
    else:                                            # swap a small circle with a large one
        k = int(rng.randint(1, max(2, n // 10) + 1))
        order = np.argsort(r)
        a, b = order[:k], order[-k:]
        x2[a], x2[b] = x[b], x[a]
        y2[a], y2[b] = y[b], y[a]
        x2 += rng.normal(0.0, 0.002, n)
        y2 += rng.normal(0.0, 0.002, n)
    return np.clip(x2, LO, HI), np.clip(y2, LO, HI), r.copy()


# ---------------------------------------------------------------- the entry point

def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    t_end = time.process_time() + CPU_BUDGET      # AT ENTRY -- never a module-level constant
    todo = sorted({int(t) for t in targets if int(t) >= 1}, reverse=True)
    if not todo:
        return

    # ---- VALUE-DRIVEN CPU ALLOCATION ---------------------------------------------------------
    # SCORE caps each n at 7 digits (relgap <= 1e-7).  An n whose committed pack is already at the
    # cap cannot yield another point of SCORE at ANY budget, yet the old equal split handed it a
    # full slice -- 27% of the census budget in iteration 6's state.  Such an n is re-offered once
    # (one evaluation, ~0 CPU: the driver is append-or-improve, so the committed pack survives)
    # and its entire slice flows to the n that can still move.  A target with no record, or no
    # committed pack, is never treated as solved -- an unseen n always gets a full slice.
    solved, active = [], []
    for n in todo:
        cap = _cap_target(n)
        p0 = _read_pack(n) if cap is not None else None
        (solved if (p0 is not None and float(p0[2].sum()) >= cap) else active).append(n)
    for n in solved:                       # keep the census honest: every target is still offered
        if meter.left() <= 0:
            break
        p0 = _read_pack(n)
        if p0 is None:
            continue
        rr = _repair(p0[0], p0[1], p0[2])
        evaluate(n, _pack(p0[0], p0[1], rr))
    todo = active
    if not todo:
        return
    # Equal CPU per remaining n: every n weighs the same in SCORE, and an offline A/B put an equal
    # split ahead of an n-proportional one (mean digits 2.854 vs 2.782 on n=35,67,99 at 9 CPU-s).
    wsum = float(len(todo))

    for pos, n in enumerate(todo):
        now = time.process_time()
        if meter.left() <= 0 or now >= t_end:
            break
        slice_end = min(t_end, now + (t_end - now) * (1.0 / wsum))
        wsum -= 1.0
        cap_s = _cap_target(n)             # reaching this mid-slice returns the rest to the pool

        best_s, best = -np.inf, None

        def offer(x, y, r):
            """Route a candidate through the metered oracle; keep our own incumbent."""
            nonlocal best_s, best
            if meter.left() <= 0:
                return -np.inf
            r = _repair(x, y, r)
            feas, s = evaluate(n, _pack(x, y, r))
            if feas and s > best_s:
                best_s, best = float(s), (x.copy(), y.copy(), r.copy())
            return float(s) if feas else -np.inf

        # ---- portfolio of starts -----------------------------------------------------------
        ks = sorted({max(1, int(math.floor(math.sqrt(n)))),
                     max(1, int(math.ceil(math.sqrt(n)))),
                     max(1, int(math.ceil(math.sqrt(n))) + 1)})
        starts = []
        warm = _read_pack(n)
        if warm is not None:
            starts.append(warm)

        # Cross-n topology transfer tickets are collected here but SPENT below, as the seeds of
        # their own hop chains -- polishing them in place (iteration 8) abandoned the transferred
        # basin the instant it scored below the incumbent, which is almost always.
        tickets = _transfer_starts(n, rng)
        hx, hy = _hex_start(n, rng)
        starts.append((hx, hy, _repair(hx, hy, np.full(n, 0.5))))
        for k in ks:
            gx, gy = _grid_start(n, k, rng)
            starts.append((gx, gy, _repair(gx, gy, np.full(n, 0.5))))
        rx, ry = _rand_start(n, rng)
        starts.append((rx, ry, _repair(rx, ry, np.full(n, 0.5))))

        # explore phase: cheap screens only -- the warm start almost always wins, so spend
        # little here and leave the slice to basin hopping.
        # A warm start almost always wins the screen, so the cold portfolio is mostly a hedge
        # there; keep the full explore budget only when there is nothing to warm-start from.
        ex_frac = 0.12 if warm is not None else 0.22
        t_explore = time.process_time() + ex_frac * max(0.0, slice_end - time.process_time())
        for (sx, sy, sr) in starts:
            if meter.left() <= 0:
                break
            x, y, r, _ = _screen(sx.copy(), sy.copy(), _repair(sx, sy, sr),
                                 min(t_explore, slice_end))
            offer(x, y, r)
            if time.process_time() >= t_explore:
                break
        if best is None:
            k = max(1, int(math.ceil(math.sqrt(n))))
            gx, gy = _grid_start(n, k, rng)
            offer(gx, gy, np.full(n, 0.5 / k))
            if best is None:
                continue
        bx, by, br, bs = _polish(best[0], best[1], best[2], slice_end)
        offer(bx, by, br)
        if cap_s is not None and best_s >= cap_s:
            continue                       # already at the 7-digit cap: hand the slice onward

        # ---- MULTI-BASIN SLICE: one hop chain per topology source ---------------------------
        # Two-tier basin hopping with a Metropolis walk.  A strictly-monotone hop chain
        # re-perturbs the SAME incumbent forever and never leaves its basin; here the walker
        # `cur` may accept a small loss (temperature adapted to the observed downhill steps),
        # while the reported best is kept separately.  Every hop is screened cheaply (~6 LPs
        # instead of ~26); only a screen that beats the CHAIN'S OWN best earns a full-precision
        # polish, so hop throughput stays high.
        #
        # The gate and the temperature are LOCAL to the chain, never global.  That is the whole
        # point: a transferred topology starts BELOW the incumbent, and a global gate would
        # refuse to polish anything it ever produced -- so its basin would never be searched at
        # all, which is exactly what happened when transfers were merely polished in place.
        def hop(sx, sy, sr, chain_end):
            """Hop from one start until chain_end.  Returns True if n reached the 7-digit cap."""
            cx, cy, cr, cs = _polish(sx, sy, sr, min(chain_end, time.process_time() + 1.0))
            offer(cx, cy, cr)
            if cap_s is not None and best_s >= cap_s:
                return True
            cur, cur_s = (cx, cy, cr), cs
            loc, loc_s = (cx, cy, cr), cs
            temp = 1e-4 * max(abs(cs), 1e-6)
            dsum, dcnt, stale = 0.0, 0, 0
            while time.process_time() < chain_end and meter.left() > 0:
                frac = 0.02 + 0.10 * (min(stale, 24) / 24.0)   # stagnate -> ruin harder
                px, py, pr = _perturb(cur[0], cur[1], cur[2], rng, chain_end, frac)
                pr = _repair(px, py, pr)
                x, y, r, s = _screen(px, py, pr, chain_end)
                if s > loc_s - 4.0 * temp:
                    x, y, r, s = _polish(x, y, r, chain_end)
                if s > loc_s:
                    loc, loc_s = (x, y, r), s
                if s > best_s:
                    offer(x, y, r)
                    if cap_s is not None and best_s >= cap_s:
                        return True               # capped mid-slice: the remaining CPU is worth
                                                  # more to the next n than another digit here
                d = s - cur_s
                if d >= 0.0:
                    cur, cur_s, stale = (x, y, r), s, 0
                else:
                    dsum -= d
                    dcnt += 1
                    temp = max(1e-12, 0.25 * dsum / dcnt)
                    if rng.rand() < math.exp(max(-40.0, d / temp)):
                        cur, cur_s = (x, y, r), s
                    stale += 1
                    if stale >= 30:                 # walker wandered off: reset to chain best
                        cur, cur_s, stale = loc, loc_s, 0
            return False

        # Spend the ticket share FIRST -- a ticket that cracks the cap returns the whole rest of
        # the slice, and any ticket time left over flows into the incumbent chain, never the
        # other way.  The payoff here is bimodal (an n is either ON the record topology or ~3
        # digits out), so this is a portfolio of cheap draws judged on its MAX, not its mean:
        # a chain that loses costs ~_TICK_MIN CPU-s, a chain that wins is worth ~4 digits.
        rest = slice_end - time.process_time()
        if tickets and rest > 3.0 * _TICK_MIN:
            tbud = _TICK_FRAC * rest
            kt = max(1, min(len(tickets), int(tbud / _TICK_MIN)))
            cracked = False
            for ti in range(kt):
                if meter.left() <= 0:
                    break
                c_end = min(slice_end, time.process_time() + tbud / kt)
                if c_end - time.process_time() < 0.25:
                    break
                if hop(tickets[ti][0], tickets[ti][1], tickets[ti][2], c_end):
                    cracked = True
                    break
            if cracked:
                continue

        # ---- SQUEEZE SOURCE: many memoryless draws in the LP's CONSTRAINT SET ---------------
        # The other four sources all move GEOMETRY -- they re-seed the same sum(r)-maximising
        # solver from somewhere else in coordinate space.  On an n whose incumbent is rigid in
        # its TOPOLOGY rather than its coordinates that is the wrong axis, and it is measured:
        # 306 coordinate hops moved the stuck n by exactly zero, and so did transfer, symmetry
        # and the ladder.  `_squeeze` re-runs the SAME solver under a radius CEILING on a few
        # well-connected circles, so the contact graph re-forms around the hole they are
        # forbidden to fill; iteration 14 measured 245 of 616 such draws landing in a genuinely
        # different contact graph, and one of them beating a four-iteration-old incumbent.
        #
        # Deliberately MEMORYLESS, exactly as that probe was funded: every draw restarts from the
        # current incumbent rather than from the previous draw.  A squeeze walk would drift down
        # the way the hop chains do; re-reading `best` each draw instead makes the source a
        # ratchet whose footing only ever improves -- the moment one draw is accepted, every
        # later draw in the slice squeezes the BETTER layout.
        rest = slice_end - time.process_time()
        if best is not None and rest > 3.0 * _SQZ_MIN:
            sq_end = min(slice_end, time.process_time() + _SQZ_FRAC * rest)
            sq_top = -np.inf                      # best SCREEN seen here: a self-calibrating gate
            global _SQZ_DRAWS
            while time.process_time() < sq_end - _SQZ_MIN and meter.left() > 0:
                _SQZ_DRAWS += 1
                b = best
                px, py, pr = _squeeze(b[0].copy(), b[1].copy(), b[2].copy(), rng,
                                      min(sq_end, time.process_time() + 1.0))
                qx, qy, qr, qs = _screen(px, py, _repair(px, py, pr),
                                         min(sq_end, time.process_time() + 1.2))
                offer(qx, qy, qr)                 # the meter is nowhere near binding; never let a
                                                  # polish gate be the thing that loses a winner
                if qs > sq_top or qs > best_s:
                    sq_top = max(sq_top, qs)
                    qx, qy, qr, qs = _polish(qx, qy, qr,
                                             min(sq_end, time.process_time() + 2.0))
                    offer(qx, qy, qr)
                if cap_s is not None and best_s >= cap_s:
                    break
            if cap_s is not None and best_s >= cap_s:
                continue

        # ---- TARGET-FOLLOWING SOURCE: chains seeded from an infeasible record-sized layout ----
        # Sources 1 and 2 (incumbent hops, transfer tickets) both end every move by MAXIMISING
        # sum(r) locally, so both draw from one distribution of basin optima -- measured on n=81,
        # a cloud whose spread is ~0.002 in sum(r) with the record ~1.8 spreads above the best of
        # hundreds of draws.  More draws from that generator cannot reach it.  `_target_push`
        # asks a different question (minimise overlap at sum(r) = record), so its seeds are not
        # samples of that cloud; hopping FROM them is what makes it a third attack rather than a
        # third parameter setting.
        rest = slice_end - time.process_time()
        if cap_s is not None and rest > 3.0 * _TICK_MIN:
            tb = _TGT_FRAC * rest
            nsrc = max(1, int(tb / _TICK_MIN))
            cracked = False
            for si in range(nsrc):
                if meter.left() <= 0:
                    break
                c_end = min(slice_end, time.process_time() + tb / nsrc)
                if c_end - time.process_time() < 0.4:
                    break
                if tickets and (si % 2 == 0):
                    t0 = tickets[si % len(tickets)]
                    sx, sy, sr = t0[0].copy(), t0[1].copy(), t0[2].copy()
                else:
                    cx, cy = _rand_start(n, rng)
                    sx, sy, sr = cx, cy, _repair(cx, cy, np.full(n, 0.5))
                sx, sy, sr, _ = _screen(sx, sy, _repair(sx, sy, sr),
                                        min(c_end, time.process_time() + 0.6))
                px, py, pr = _target_push(sx, sy, sr, cap_s,
                                          min(c_end, time.process_time() + 1.2), rng)
                pr = _repair(px, py, pr)
                if hop(px, py, pr, c_end):
                    cracked = True
                    break
            if cracked:
                continue

        # ---- CONTINUATION-LADDER SOURCE: a homotopy path to the record sum, not a leap -----
        # Same question as the target follower, asked in rungs.  The single-jump version freezes
        # at a stationary point of the violation because it is dropped far inside the infeasible
        # region; walking T up and returning to a FEASIBLE local optimum between rungs keeps every
        # LP in a regime where it is still moving, and makes the source a ratchet -- its report is
        # never worse than its seed.  Seeded from the incumbent, from transfer tickets, and from
        # screened cold starts, so the ladder is climbed from three different footings.
        rest = slice_end - time.process_time()
        if rest > 3.0 * _TICK_MIN:
            lb = _LAD_FRAC * rest
            nsrc = max(1, int(lb / _TICK_MIN))
            cracked = False
            for si in range(nsrc):
                if meter.left() <= 0:
                    break
                c_end = min(slice_end, time.process_time() + lb / nsrc)
                if c_end - time.process_time() < 0.5:
                    break
                if best is not None and si % 3 == 0:
                    sx, sy, sr = best[0].copy(), best[1].copy(), best[2].copy()
                elif tickets and si % 3 == 1:
                    t0 = tickets[si % len(tickets)]
                    sx, sy, sr = t0[0].copy(), t0[1].copy(), t0[2].copy()
                else:
                    if si % 2:
                        cx, cy = _hex_start(n, rng)
                    else:
                        cx, cy = _rand_start(n, rng)
                    sx, sy, sr = cx, cy, _repair(cx, cy, np.full(n, 0.5))
                    sx, sy, sr, _ = _screen(sx, sy, sr,
                                            min(c_end, time.process_time() + 0.6))
                l_end = time.process_time() + 0.55 * max(0.0, c_end - time.process_time())
                lx, ly, lr = _ladder_push(sx, sy, sr, cap_s, min(c_end, l_end), rng,
                                          rungs=int(rng.randint(4, 8)),
                                          over=float(rng.uniform(1.0, 1.010)))
                lr = _repair(lx, ly, lr)
                offer(lx, ly, lr)
                if cap_s is not None and best_s >= cap_s:
                    cracked = True
                    break
                if hop(lx, ly, lr, c_end):
                    cracked = True
                    break
            if cracked:
                continue

        # ---- SYMMETRY-PROJECTION SOURCE: chains seeded from a SYMMETRIC local optimum -------
        # A fourth source, and the first one whose draws are not even in the same space as the
        # others: sources 1-3 all wander a full 3n-dimensional amorphous cloud, and a layout with
        # an exact square symmetry has probability zero of ever appearing in it.  The census says
        # such layouts are real -- some record packings ARE exactly symmetric -- so this buys the
        # other mode of a bimodal prior for a fixed slice, judged on its max and not its mean.
        rest = slice_end - time.process_time()
        if rest > 3.0 * _TICK_MIN:
            sb = _SYM_FRAC * rest
            nsrc = max(1, int(sb / _TICK_MIN))
            bases = []
            if best is not None:
                bases.append((best[0], best[1], best[2]))
            bases.extend(tickets[:4])
            cracked = False
            for si in range(nsrc):
                if meter.left() <= 0:
                    break
                c_end = min(slice_end, time.process_time() + sb / nsrc)
                if c_end - time.process_time() < 0.4:
                    break
                if bases and si % 3 != 2:
                    b0 = bases[si % len(bases)]
                    sx, sy, sr = b0[0].copy(), b0[1].copy(), b0[2].copy()
                elif si % 2:
                    cx, cy = _hex_start(n, rng)
                    sx, sy, sr = cx, cy, _repair(cx, cy, np.full(n, 0.5))
                else:
                    cx, cy = _rand_start(n, rng)
                    sx, sy, sr = cx, cy, _repair(cx, cy, np.full(n, 0.5))
                Msym, Ksym = _SYM_OPS[int(rng.randint(len(_SYM_OPS)))]
                px, py, pr = _sym_push(sx, sy, sr, Msym, Ksym,
                                       min(c_end, time.process_time() + 1.5))
                if hop(px, py, _repair(px, py, pr), c_end):
                    cracked = True
                    break
            if cracked:
                continue

        if best is None or meter.left() <= 0:
            continue
        hop(best[0].copy(), best[1].copy(), best[2].copy(), slice_end)


# ---------------------------------------------------------------- self-test

def _self_test():
    """Mimics the harness oracle locally (no bench imports) and calls solve() TWICE in one process."""
    import sys

    records_ok = os.path.isfile(os.path.join("bench", "records.json"))
    state = {"used": 0, "budget": 20000, "best": {}}

    class M:
        budget = state["budget"]

        @property
        def used(self):
            return state["used"]

        def left(self):
            return state["budget"] - state["used"]

    def evaluate(n, packing):
        a = np.asarray(packing, dtype=float)
        single = a.ndim == 2
        if single:
            a = a[None]
        assert a.shape[1] == n and a.shape[2] == 3
        B = a.shape[0]
        grant = max(0, min(B, state["budget"] - state["used"]))
        state["used"] += grant
        x, y, r = a[..., 0], a[..., 1], a[..., 2]
        sum_r = r.sum(axis=1)
        wall = np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min(axis=1)
        dx = x[:, :, None] - x[:, None, :]
        dy = y[:, :, None] - y[:, None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        slack = dist - (r[:, :, None] + r[:, None, :])
        slack[:, np.eye(n, dtype=bool)] = np.inf
        feas = (r.min(axis=1) > 0) & (wall >= -1e-9) & (slack.reshape(B, -1).min(axis=1) >= -1e-9)
        feas[grant:] = False
        sum_r = np.where(feas, sum_r, -np.inf)
        for i in range(grant):
            if feas[i] and sum_r[i] > state["best"].get(n, (-np.inf,))[0]:
                state["best"][n] = (float(sum_r[i]), a[i].copy())
        return (bool(feas[0]), float(sum_r[0])) if single else (feas, sum_r)

    global CPU_BUDGET
    CPU_BUDGET = 6.0
    rng = np.random.RandomState(0)

    solve(evaluate, M(), rng, [27, 45])
    assert set(state["best"]) == {27, 45}, "first call must pack every target"
    first = {k: v[0] for k, v in state["best"].items()}

    # SECOND call in the SAME process, and with a SINGLE target -- both must still work.
    state["best"] = {}
    solve(evaluate, M(), rng, [31])
    assert 31 in state["best"], "second solve() call in one process returned nothing"

    # a target with no committed pack must cold-start cleanly
    state["best"] = {}
    solve(evaluate, M(), rng, [12])
    assert 12 in state["best"], "unseen n must cold-start"

    # a BELOW-cap n exercises the multi-basin ticket path (its neighbours are committed donors,
    # so _transfer_starts must yield several rule-variant tickets and each must seed a hop chain).
    if os.path.isfile(os.path.join(PACK_DIR, "csqv79.pck")):
        tix = _transfer_starts(81, np.random.RandomState(1))
        assert len(tix) >= 2, "transfer must yield rule-variant tickets when donors exist"
        # the donor window is +/-_DONOR_WIN, not +/-4: with a full census around 81 the source
        # must reach past the immediate neighbours, or the widening silently did nothing.
        ndon = sum(1 for m in range(81 - _DONOR_WIN, 81 + _DONOR_WIN + 1)
                   if m != 81 and _read_pack(m) is not None)
        assert len(tix) == 3 * min(10, ndon), "ticket count must track the widened donor window"
        assert ndon >= 5, "expected several committed donors within the window"
        for (tx, ty, tr) in tix:
            assert tx.shape == (81,) and ty.shape == (81,) and tr.shape == (81,)
            assert float(tr.min()) > 0.0
        state["best"] = {}
        global _SQZ_DRAWS
        _SQZ_DRAWS = 0
        solve(evaluate, M(), rng, [81])
        assert 81 in state["best"], "below-cap n must return a packing from the multi-basin slice"
        # the SQUEEZE slice is a FUNDING change, and under-funding is invisible in the output --
        # an unfunded source and a funded-but-unlucky one both just fail to improve.  So police
        # the funding directly: _SQZ_FRAC of a slice at the probe's measured 0.039 CPU-s/draw
        # must actually buy draws, not break out on the first pass.
        assert _SQZ_DRAWS >= 25, "squeeze slice took only %d draws -- it is not funded" % _SQZ_DRAWS
        print("  squeeze slice: %d draws in the metered solve()" % _SQZ_DRAWS)

        # the TARGET-FOLLOWING source must actually drive sum(r) TO the infeasible target and
        # hand back an n-circle layout -- if it silently returned its input, the third attack
        # would be a no-op that still charges _TGT_FRAC of every slice.
        w81 = _read_pack(81)
        T = _cap_target(81)
        px, py, pr = _target_push(w81[0].copy(), w81[1].copy(), w81[2].copy(), T,
                                  time.process_time() + 3.0, np.random.RandomState(5))
        assert px.shape == (81,) and py.shape == (81,) and pr.shape == (81,)
        assert float(pr.sum()) >= T - 1e-7, "target row must be binding: sum(r) must reach T"
        assert float(np.abs(px).max()) <= 0.5 + 1e-12 and float(np.abs(py).max()) <= 0.5 + 1e-12
        rr = _repair(px, py, pr)
        assert float(rr.min()) > 0.0, "repaired target-push layout must stay a real packing"

        # the SYMMETRY source must return a layout that is GENUINELY invariant under its op, not
        # one merely nudged towards it -- otherwise it is just another restart still charging
        # _SYM_FRAC of every slice.  Checked as a set: every image point must land on a circle.
        for Ms, Ks in _SYM_OPS:
            sx, sy, sr = _symmetrize(w81[0].copy(), w81[1].copy(), w81[2].copy(), Ms, Ks)
            Ps = np.stack([sx, sy], axis=1)
            d2 = ((Ps.dot(Ms.T)[:, None, :] - Ps[None, :, :]) ** 2).sum(-1).min(axis=1)
            res = math.sqrt(float(d2.mean())) / float(sr.mean())
            assert res < 1e-6, "symmetrised layout is not invariant (res=%.2e)" % res
            assert float(sr.min()) > 0.0 and sx.shape == (81,)
            assert float(np.abs(sx).max()) <= 0.5 + 1e-12 and float(np.abs(sy).max()) <= 0.5 + 1e-12
        # and _sym_push must still be symmetric AFTER the LP rounds (the whole point: the free LP
        # would otherwise break it on the first step).
        qx, qy, qr = _sym_push(w81[0].copy(), w81[1].copy(), w81[2].copy(),
                               _SYM_OPS[2][0], _SYM_OPS[2][1], time.process_time() + 4.0)
        Ps = np.stack([qx, qy], axis=1)
        d2 = ((Ps.dot(_SYM_OPS[2][0].T)[:, None, :] - Ps[None, :, :]) ** 2).sum(-1).min(axis=1)
        assert math.sqrt(float(d2.mean())) / float(qr.mean()) < 1e-6, "_sym_push lost its symmetry"
        assert float(_repair(qx, qy, qr).min()) > 0.0

        # the CONTINUATION LADDER must (a) be a RATCHET -- never report worse than its seed, which
        # is the property the whole rung schedule is for -- and (b) actually climb, i.e. from a
        # mediocre seed it must strictly improve.  Without (b) a ladder whose every rung failed
        # would look identical to a working one while charging _LAD_FRAC of every slice.
        lrng = np.random.RandomState(11)
        cx0, cy0 = _rand_start(81, lrng)
        cr0 = _repair(cx0, cy0, np.full(81, 0.5))
        cx0, cy0, cr0, cs0 = _screen(cx0, cy0, cr0, time.process_time() + 0.5)
        lx, ly, lr = _ladder_push(cx0.copy(), cy0.copy(), cr0.copy(), T,
                                  time.process_time() + 6.0, lrng, rungs=4, over=1.004)
        assert lx.shape == (81,) and ly.shape == (81,) and lr.shape == (81,)
        assert float(np.abs(lx).max()) <= 0.5 + 1e-12 and float(np.abs(ly).max()) <= 0.5 + 1e-12
        lrr = _repair(lx, ly, lr)
        assert float(lrr.min()) > 0.0, "repaired ladder layout must stay a real packing"
        assert float(lrr.sum()) >= float(cr0.sum()) - 1e-12, "ladder must ratchet, never regress"
        assert float(lrr.sum()) > float(cr0.sum()) + 1e-9, "ladder never climbed off its seed"
        # and with no record for this n the schedule must still be finite and usable
        mx, my, mr = _ladder_push(cx0.copy(), cy0.copy(), cr0.copy(), None,
                                  time.process_time() + 3.0, lrng, rungs=3)
        assert np.all(np.isfinite(mx)) and float(_repair(mx, my, mr).min()) > 0.0

    if records_ok:
        rec = json.load(open(os.path.join("bench", "records.json")))["records"]
        for n, s in first.items():
            if str(n) in rec:
                R = rec[str(n)]
                g = max((R - s) / R, 1e-16)
                print("  n=%d sum_r=%.9f gap=%.2e digits=%.2f" % (n, s, g, min(7.0, -math.log10(g))))
    # --- the SQUEEZE kernel: two things it could silently fake ------------------------------
    rs = np.random.RandomState(9)
    gx, gy = _hex_start(37, rs)
    sx, sy, sr, _ = _screen(gx, gy, _repair(gx, gy, np.full(37, 0.5)),
                            time.process_time() + 2.0)
    # (1) the radius ceiling must actually bind inside the LP, not just be passed around
    cap = np.full(37, 0.5)
    cap[:4] = 0.4 * sr[:4]
    cx, cy, cr, _ = _slp(sx.copy(), sy.copy(), np.minimum(sr, cap), max_lp=25,
                         deadline=time.process_time() + 2.0, rcap=cap)
    assert np.all(cr <= cap + 1e-9), "rcap not honoured by the LP: %.3e" % (cr - cap).max()
    # (2) it must reach a DIFFERENT contact topology -- a squeeze that always regrows into the
    #     same graph is an expensive no-op wearing a kernel's costume
    def _g(x, y, r):
        d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        np.fill_diagonal(d, np.inf)
        return set(map(tuple, np.argwhere(np.triu(d - (r[:, None] + r[None, :]) < 1e-6, 1))))
    g0, moved = _g(sx, sy, sr), 0
    for _ in range(12):
        qx, qy, qr = _squeeze(sx.copy(), sy.copy(), sr.copy(), rs, time.process_time() + 0.8)
        qx, qy, qr, qs = _screen(qx, qy, _repair(qx, qy, qr), time.process_time() + 0.8)
        assert np.isfinite(qs) and qr.min() > 0, "squeeze produced a degenerate layout"
        if len(_g(qx, qy, qr) ^ g0) > 2:
            moved += 1
    assert moved >= 2, "squeeze never changed the contact graph in 12 draws"
    print("  squeeze: rcap honoured, topology changed in %d/12 draws" % moved)
    print("self-test OK (used %d evals)" % state["used"])
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print("usage: python tools/solver.py --self-test")
