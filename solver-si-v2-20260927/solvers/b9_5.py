"""Circle-packing solver: maximize sum of radii of n circles in the unit square (packomania csqv).

STRATEGY (v2 -- replaces the seed baseline's equal-radius repulsion descent)
---------------------------------------------------------------------------
The seed solver gave every circle the *guaranteed* radius r_i = min(wall_i, min_j d_ij/2). That rule
forces near-equal radii, which is badly suboptimal for max-SUM-of-radii: the optimum mixes large and
small circles. Two structural facts drive this rewrite:

  1. FOR FIXED CENTRES THE RADII ARE AN LP.  maximize sum r  s.t.  r_i + r_j <= d_ij,  0 <= r_i <= w_i.
     That is exactly solvable (scipy/HiGHS), so the real search space is only the 2n centre coordinates.
     A valid pair filter keeps the LP small: r_i <= u_i := min(w_i, min_j d_ij), so any pair with
     d_ij >= u_i + u_j can never bind and is dropped (exact, not a heuristic relaxation).

  2. THE CENTRES ARE FOUND BY SMOOTH PENALTY ASCENT.  Minimise
         f(x,y,r) = -sum r + (rho/2) * ( sum_{i<j} relu(r_i+r_j-d_ij)^2 + sum_i relu(r_i-w_i)^2 )
     with analytic gradients, L-BFGS-B, over a ramp rho = 1e2 ... 1e7.  Every operation is a vectorized
     O(n^2) numpy pass, so one gradient is ~0.1-0.3 ms even at n=99 and a full multi-stage run is ~0.1 s.

Each restart is: init centres -> penalty ramp -> exact LP radii -> short high-rho re-polish -> LP again
-> shrink-to-strictly-feasible -> ONE metered evaluate().  Restarts alternate diverse cold inits
(random / jittered square grid / hex rows / ring-shell / two-scale) with basin-hopping perturbations of
the incumbent, plus a warm start from the committed census when one exists.

METERING.  The binding meter here is the driver's 120 s process-CPU backstop, not the 500k evaluation
count (one deep restart = one evaluation).  The CPU deadline is computed RELATIVE to
time.process_time() at entry of every call, so a second solve() in the same process gets its own full
slice (MISSION's CPU_STOP trap).  Per-n time slices are recomputed from the ACTUAL remaining time and
the remaining targets, so they are correct for len(targets) == 1 as well.  Every warm-start read is
guarded by os.path.exists with a cold-start fallback.  All randomness comes from `rng`.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog, minimize
from scipy.sparse import csr_matrix

LO, HI = -0.5, 0.5
CPU_BUDGET = 104.0        # process-CPU seconds this solve() call may spend (driver backstop is 120 s)
EDGE = 1e-7               # keep centres strictly inside the square
RMIN = 1e-11              # strictly-positive radius floor (far below the 1e-9 feasibility tolerance)
RHO_RAMP = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)


# --------------------------------------------------------------------------- geometry helpers
def _wall(xy):
    """Distance from each centre to the nearest wall -> (n,)."""
    x, y = xy[:, 0], xy[:, 1]
    return np.minimum.reduce([x - LO, HI - x, y - LO, HI - y])


def _dists(xy):
    """Pairwise centre distances with +inf on the diagonal -> (n,n)."""
    d = xy[:, None, :] - xy[None, :, :]
    out = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(out, np.inf)
    return out


def _safe_radii(xy):
    """The guaranteed-feasible equal-ish rule, used as an LP fallback / penalty warm start."""
    n = xy.shape[0]
    w = np.maximum(_wall(xy), 0.0)
    if n >= 2:
        r = np.minimum(w, _dists(xy).min(axis=1) / 2.0)
    else:
        r = w
    return np.maximum(r - 2e-9, RMIN)


def _shrink_feasible(xy, r):
    """Scale radii down by the largest factor that makes the packing strictly feasible, then floor them
    strictly positive. Scaling is safe: every constraint (r_i+r_j <= d_ij, r_i <= w_i) is homogeneous."""
    n = xy.shape[0]
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    w = np.maximum(_wall(xy), 0.0)
    s = 1.0
    pos = r > 0
    if pos.any():
        s = min(s, float(np.min(w[pos] / r[pos])))
    if n >= 2:
        d = _dists(xy)
        ss = r[:, None] + r[None, :]
        np.fill_diagonal(ss, 0.0)
        m = ss > 0
        if m.any():
            s = min(s, float(np.min(d[m] / ss[m])))
    if not np.isfinite(s) or s < 0.0:
        s = 0.0
    r = r * (s * (1.0 - 1e-12))
    return np.maximum(r, RMIN)


def _lp_radii(xy):
    """EXACT optimal radii for fixed centres: max sum r s.t. r_i+r_j <= d_ij, 0 <= r_i <= w_i.

    Pair filter (exact): r_i <= u_i := min(w_i, min_j d_ij) because r_j >= 0, so a pair with
    d_ij >= u_i + u_j can never be tight and is omitted from the LP."""
    n = xy.shape[0]
    w = np.maximum(_wall(xy), 0.0)
    if n < 2:
        return np.maximum(w - 1e-12, RMIN)
    d = _dists(xy)
    u = np.minimum(w, d.min(axis=1))
    iu, ju = np.triu_indices(n, 1)
    dv = d[iu, ju]
    keep = dv < (u[iu] + u[ju])
    ik, jk, bk = iu[keep], ju[keep], dv[keep]
    m = ik.size
    try:
        if m:
            rows = np.repeat(np.arange(m), 2)
            cols = np.empty(2 * m, dtype=np.int64)
            cols[0::2], cols[1::2] = ik, jk
            A = csr_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n))
            res = linprog(-np.ones(n), A_ub=A, b_ub=bk,
                          bounds=np.stack([np.zeros(n), u], axis=1), method="highs")
        else:
            return _shrink_feasible(xy, u)
        if res.x is None:
            return _safe_radii(xy)
        r = np.asarray(res.x, dtype=float)
    except Exception:
        return _safe_radii(xy)
    if not np.all(np.isfinite(r)):
        return _safe_radii(xy)
    return _shrink_feasible(xy, r)


# --------------------------------------------------------------------------- penalty objective
def _fg(z, n, rho):
    """f, grad of  -sum r + (rho/2)*( sum_{i<j} relu(r_i+r_j-d_ij)^2 + sum_i relu(r_i-w_i)^2 )."""
    x, y, r = z[:n], z[n:2 * n], z[2 * n:]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    d = np.maximum(d, 1e-12)
    G = r[:, None] + r[None, :] - d
    np.maximum(G, 0.0, out=G)
    np.fill_diagonal(G, 0.0)

    w4 = np.stack([x - LO, HI - x, y - LO, HI - y])          # (4,n)
    k = np.argmin(w4, axis=0)
    w = w4[k, np.arange(n)]
    h = np.maximum(r - w, 0.0)

    f = -r.sum() + 0.5 * rho * (0.5 * float(np.sum(G * G)) + float(np.dot(h, h)))

    coef = G / d
    gx = -rho * (coef * dx).sum(axis=1)
    gy = -rho * (coef * dy).sum(axis=1)
    gx -= rho * h * ((k == 0) * 1.0 + (k == 1) * -1.0)
    gy -= rho * h * ((k == 2) * 1.0 + (k == 3) * -1.0)
    gr = -1.0 + rho * (G.sum(axis=1) + h)
    return f, np.concatenate([gx, gy, gr])


def _bounds(n):
    b = [(LO + EDGE, HI - EDGE)] * (2 * n) + [(0.0, 0.5)] * n
    return b


def _penalty_refine(xy, r, n, ramp, maxiter):
    """Run the L-BFGS-B penalty ramp from (xy, r); returns refined centres."""
    z = np.concatenate([xy[:, 0], xy[:, 1], r])
    bnds = _bounds(n)
    for rho in ramp:
        res = minimize(_fg, z, args=(n, rho), jac=True, method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": maxiter, "maxcor": 12})
        z = res.x
        if not np.all(np.isfinite(z)):
            break
    z = np.where(np.isfinite(z), z, 0.0)
    return np.clip(np.stack([z[:n], z[n:2 * n]], axis=1), LO + EDGE, HI - EDGE)


# --------------------------------------------------------------------------- sequential LP (SLP)
def _slp_refine(xy, r, slice_end, delta0=0.03, max_it=400, dmin=1e-9):
    """JOINT ascent on (centres, radii) by SEQUENTIAL LINEAR PROGRAMMING with a trust region.

    Key structural fact that makes this exact rather than heuristic: d_ij(p) = ||p_i - p_j|| is a CONVEX
    function of the centres, so its first-order model is a global UNDERESTIMATE,

        d_ij(p + D) >= d_ij(p) + u_ij . (D_i - D_j),      u_ij = (p_i - p_j)/d_ij.

    Imposing  r_i + r_j <= d_ij + u_ij.(D_i - D_j)  is therefore a RESTRICTION of the true constraint:
    ANY solution of the LP is genuinely non-overlapping, no line search or feasibility repair needed.
    The wall constraints  r_i +- D_i <= dist-to-wall  are linear, hence exact.  D = 0 with the current r
    is always LP-feasible, so the LP optimum can never be worse: every iterate is feasible AND monotone.

    Variables z = [Dx (n) | Dy (n) | r (n)]; objective max sum r.  Trust region |D| <= delta both bounds
    the LP and makes the pair filter EXACT: with U_i := min(w_i + delta, 0.5) an upper bound on r_i, a
    pair with d_ij >= U_i + U_j + 2*delta still has d_ij(p+D) >= U_i + U_j >= r_i + r_j after the move,
    so it can be dropped from the LP without ever being violated.

    delta shrinks on stall -- the model's conservatism is O(delta^2/d), so the fixed point of the LP at
    trust radius delta sits O(delta^2) below the true local optimum and the shrinking ramp drives that to
    machine precision.  That is where the DIGITS come from.
    """
    n = xy.shape[0]
    xy = np.array(xy, dtype=float)
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    if n < 1:
        return xy, r
    if n == 1:
        w = max(float(_wall(xy)[0]), 0.0)
        return xy, np.array([max(w - 1e-15, RMIN)])
    cur = float(r.sum())
    delta = float(delta0)
    for _ in range(max_it):
        if time.process_time() > slice_end or delta < dmin:
            break
        out = _lp_step(xy, r, delta)
        if out is None:
            delta *= 0.3
            continue
        nxy, nr, step = out
        s = float(nr.sum())
        if s > cur + 1e-15:
            gain = s - cur
            xy, r, cur = nxy, nr, s
            if gain < 1e-12 * max(cur, 1.0):
                delta *= 0.3
            elif float(np.max(np.abs(step))) > 0.9 * delta:
                delta = min(delta * 1.5, 0.08)
        else:
            delta *= 0.3
    return xy, r


def _lp_step(xy, r, delta, cD=None, cR=None, red=None):
    """ONE trust-region LP: max cR.r + cD.D  s.t. the conservatively linearised constraints.

    Returns (new_xy, new_r, step) or None.  `cD` (2n,) adds an arbitrary linear pull on the centre
    displacement: with cD = 0 this is the pure ascent step, and with a random cD it is a FEASIBLE
    RANDOM WALK -- the iterate moves sideways along the feasible manifold (the linearisation is a
    restriction, so the result is still genuinely non-overlapping) instead of hopping out of it.
    That is what lets basin hopping change the contact graph without ever producing an infeasible
    layout that has to be repaired.

    `red` = (M, isD) SUBSTITUTES a linear reduction z_full = M @ z_red of the 3n variables, so the LP
    is solved over a LOWER-DIMENSIONAL affine subspace: A M z_red <= b, objective (M^T c) . z_red.  The
    feasible set is the exact intersection of the original one with range(M), so a reduced step is still
    genuinely non-overlapping -- the reduction only forbids directions, it never relaxes a constraint.
    That is what makes a SYMMETRY-CONSTRAINED SLP possible (iteration 11): tie every circle to its
    mirror partner and the optimiser can no longer drift off the symmetric subspace.  `isD` marks which
    reduced columns are displacements (bounded by the trust radius) rather than radii.

    `cR` (n,) reweights the RADIUS objective (default: all ones = plain max sum r).  The feasible set
    is untouched, so a reweighted step is still exactly non-overlapping -- only the direction the
    optimiser pushes changes.  That is the whole of objective-space perturbation."""
    n = xy.shape[0]
    x, y = xy[:, 0], xy[:, 1]
    iu, ju = np.triu_indices(n, 1)
    wx0, wx1 = x - LO, HI - x
    wy0, wy1 = y - LO, HI - y
    w = np.minimum.reduce([wx0, wx1, wy0, wy1])
    U = np.minimum(np.maximum(w, 0.0) + delta, 0.5)
    dxm = x[:, None] - x[None, :]
    dym = y[:, None] - y[None, :]
    dm = np.sqrt(dxm * dxm + dym * dym)
    dv = dm[iu, ju]
    keep = dv < (U[iu] + U[ju] + 2.0 * delta)
    ik, jk, dk = iu[keep], ju[keep], np.maximum(dv[keep], 1e-14)
    m = ik.size
    ux = (x[ik] - x[jk]) / dk
    uy = (y[ik] - y[jk]) / dk
    rows = np.repeat(np.arange(m), 6)
    cols = np.empty(6 * m, dtype=np.int64)
    vals = np.empty(6 * m, dtype=float)
    cols[0::6], vals[0::6] = 2 * n + ik, 1.0
    cols[1::6], vals[1::6] = 2 * n + jk, 1.0
    cols[2::6], vals[2::6] = ik, -ux
    cols[3::6], vals[3::6] = jk, ux
    cols[4::6], vals[4::6] = n + ik, -uy
    cols[5::6], vals[5::6] = n + jk, uy
    ar = np.arange(n)
    wrow = np.concatenate([m + ar, m + ar, m + n + ar, m + n + ar,
                           m + 2 * n + ar, m + 2 * n + ar, m + 3 * n + ar, m + 3 * n + ar])
    wcol = np.concatenate([2 * n + ar, ar, 2 * n + ar, ar,
                           2 * n + ar, n + ar, 2 * n + ar, n + ar])
    wval = np.concatenate([np.ones(n), -np.ones(n), np.ones(n), np.ones(n),
                           np.ones(n), -np.ones(n), np.ones(n), np.ones(n)])
    b = np.concatenate([dk, wx0, wx1, wy0, wy1])
    A = csr_matrix((np.concatenate([vals, wval]),
                    (np.concatenate([rows, wrow]), np.concatenate([cols, wcol]))),
                   shape=(m + 4 * n, 3 * n))
    lo = np.concatenate([np.full(2 * n, -delta), np.zeros(n)])
    hi = np.concatenate([np.full(2 * n, delta), np.full(n, 0.5)])
    c = np.concatenate([np.zeros(2 * n) if cD is None else -np.asarray(cD, dtype=float),
                        -np.ones(n) if cR is None else -np.asarray(cR, dtype=float)])
    if red is not None:
        M, isD = red
        A = A.dot(M)
        c = np.asarray(M.T.dot(c), dtype=float).ravel()
        lo = np.where(isD, -delta, 0.0)
        hi = np.where(isD, delta, 0.5)
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=np.stack([lo, hi], axis=1), method="highs")
    except Exception:
        return None
    if res.x is None or not np.all(np.isfinite(res.x)):
        return None
    z = np.asarray(res.x, dtype=float)
    if red is not None:
        z = np.asarray(red[0].dot(z), dtype=float).ravel()
    step = z[:2 * n]
    nxy = np.clip(xy + np.stack([step[:n], step[n:]], axis=1), LO, HI)
    return nxy, _shrink_feasible(nxy, z[2 * n:]), step


def _equalize_move(xy, r, rng, delta=0.02, steps=2, beta=None):
    """OBJECTIVE-SPACE perturbation: re-solve the SAME trust-region LP under a radius objective that
    pays more for SMALL circles, then hand the result back to the plain sum-r ascent.

    Why this move class exists.  Every operator built up to iteration 6 perturbs the *layout*
    (jitter, hole teleport, LP shake, region graft); none perturbs the *objective*.  But the census
    itself says the defect is in objective space: by Cauchy-Schwarz  sum r <= sqrt(n * sum r^2), so
    for a given amount of packed area the sum is maximised by EQUAL radii, and measured over the
    committed census the packs already at the record carry visibly less radius spread than their
    neighbours that are stuck (CV(r) 0.120-0.126 at n=43/85/87 against 0.139-0.164 at the n stalled
    near 2.5 digits).  A plain sum-r local optimum is happy to let one circle grow large while its
    neighbours are crushed -- that is a local optimum no positional move can leave, because escaping
    it means every radius changing at once.

    So: weight w_i = (rbar / r_i)^beta.  The LP still enforces the exact linearised non-overlap
    constraints, so every iterate is genuinely feasible; only the direction of travel changes.  Small
    circles buy displacement from their large neighbours, contacts open and close, and the layout is
    dropped into a DIFFERENT basin whose radius distribution is flatter.  sum r is expected to fall
    during the move -- that is the point; the caller re-maximises it afterwards."""
    n = xy.shape[0]
    if n < 2:
        return xy.copy(), np.asarray(r, dtype=float).copy()
    if beta is None:
        beta = float(rng.uniform(0.5, 3.0))
    xy = np.asarray(xy, dtype=float).copy()
    r = np.maximum(np.asarray(r, dtype=float).copy(), 0.0)
    for _ in range(max(1, int(steps))):
        rb = float(r.mean())
        if not np.isfinite(rb) or rb <= 0.0:
            break
        w = (rb / np.maximum(r, rb * 1e-3)) ** beta
        mx = float(w.max())
        if not np.isfinite(mx) or mx <= 0.0:
            break
        out = _lp_step(xy, r, delta, cR=w / mx)
        if out is None:
            break
        xy, r = out[0], out[1]
    return xy, r


def _lp_shake(xy, r, rng, delta=0.02, steps=3, pull=1.0):
    """FEASIBLE random walk: `steps` trust-region LPs driven by a random centre-pull.

    Every intermediate layout is strictly non-overlapping (the linearised constraints are a
    restriction of the true ones), so this explores the feasible manifold rather than jumping off it.
    Unlike coordinate jitter, the walk is CONSTRAINT-AWARE: circles slide along their contacts, which
    is precisely the motion that opens and closes contacts and lands the iterate in another basin."""
    n = xy.shape[0]
    if n < 2:
        return xy, r
    for _ in range(steps):
        cD = rng.normal(0.0, 1.0, size=2 * n)
        nrm = np.linalg.norm(cD)
        if nrm > 0:
            cD *= (pull * n) / nrm
        out = _lp_step(xy, r, delta, cD)
        if out is None:
            break
        xy, r = out[0], out[1]
    return xy, r


# --------------------------------------------------------------------------- centre initialisers
def _init_random(n, rng):
    return rng.uniform(LO + 0.02, HI - 0.02, size=(n, 2))


def _init_grid(n, rng):
    k = int(np.ceil(np.sqrt(n)))
    g = (np.arange(k) + 0.5) / k - 0.5
    pts = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    idx = rng.permutation(pts.shape[0])[:n]
    return np.clip(pts[idx] + rng.normal(0, 0.25 / k, size=(n, 2)), LO + 0.01, HI - 0.01)


def _init_hex(n, rng):
    rows = max(1, int(round(np.sqrt(n / 0.866))))
    per = int(np.ceil(n / rows))
    pts = []
    for i in range(rows):
        yy = LO + (i + 0.5) / rows
        off = 0.5 / per if i % 2 else 0.0
        for j in range(per):
            pts.append((LO + (j + 0.5) / per + off, yy))
    pts = np.array(pts[:max(n, 1)], dtype=float)
    if pts.shape[0] < n:                                     # pad if the lattice came up short
        pts = np.vstack([pts, rng.uniform(LO + 0.05, HI - 0.05, size=(n - pts.shape[0], 2))])
    return np.clip(pts + rng.normal(0, 0.2 / per, size=(n, 2)), LO + 0.01, HI - 0.01)


def _init_shell(n, rng):
    """Concentric rings -- a paradigm the grid/hex families cannot reach."""
    pts, left, ring = [], n, 0
    while left > 0:
        cnt = min(left, max(1, int(round(6 * (ring + 1) * 0.8))))
        rad = 0.46 * (1.0 - ring / (0.5 + np.sqrt(n) / 2.0))
        rad = max(rad, 0.02)
        ang = rng.uniform(0, 2 * np.pi) + np.arange(cnt) * (2 * np.pi / cnt)
        pts.append(np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1))
        left -= cnt
        ring += 1
    p = np.vstack(pts)[:n]
    return np.clip(p + rng.normal(0, 0.01, size=(n, 2)), LO + 0.01, HI - 0.01)


def _init_two_scale(n, rng):
    """A few deliberately large circles plus a fine background -- max-sum-of-radii optima are mixed."""
    big = max(1, int(rng.randint(1, max(2, n // 6))))
    kb = int(np.ceil(np.sqrt(big)))
    gb = (np.arange(kb) + 0.5) / kb - 0.5
    pb = np.stack(np.meshgrid(gb, gb, indexing="ij"), axis=-1).reshape(-1, 2)[:big]
    rest = n - big
    ks = int(np.ceil(np.sqrt(max(rest, 1))))
    gs = (np.arange(ks) + 0.5) / ks - 0.5
    ps = np.stack(np.meshgrid(gs, gs, indexing="ij"), axis=-1).reshape(-1, 2)
    ps = ps[rng.permutation(ps.shape[0])[:rest]]
    p = np.vstack([pb, ps]) if rest > 0 else pb
    return np.clip(p + rng.normal(0, 0.02, size=(n, 2)), LO + 0.01, HI - 0.01)


_INITS = (_init_grid, _init_hex, _init_random, _init_two_scale, _init_shell)


def _read_pack(n):
    """Centres of the committed census pack for n, or None. GUARDED: any missing/odd file -> cold start."""
    p = "bench/packs/csqv%d.pck" % n
    try:
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            raw = [ln for ln in fh.read().splitlines() if ln.strip()]
        rows = [[float(v) for v in ln.split()] for ln in raw[2:]]
        if len(rows) != n or any(len(t) != 3 for t in rows):
            return None
        a = np.array(rows, dtype=float)
        if not np.all(np.isfinite(a)):
            return None
        return np.clip(a[:, :2], LO + EDGE, HI - EDGE)
    except (OSError, ValueError):
        return None


def _read_pack_full(n):
    """(centres, radii) of the committed pack for n, or None.  GUARDED exactly like _read_pack: an
    absent / short / malformed file returns None and the caller must work without it."""
    p = "bench/packs/csqv%d.pck" % n
    try:
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            raw = [ln for ln in fh.read().splitlines() if ln.strip()]
        rows = [[float(v) for v in ln.split()] for ln in raw[2:]]
        if len(rows) != n or any(len(t) != 3 for t in rows):
            return None
        a = np.array(rows, dtype=float)
        if not np.all(np.isfinite(a)):
            return None
        return np.clip(a[:, :2], LO + EDGE, HI - EDGE), np.maximum(a[:, 2], 0.0)
    except (OSError, ValueError):
        return None


def _clearance(p, xy, r):
    """min(distance to the nearest circle's boundary, distance to the nearest wall) at each probe p."""
    if xy.shape[0]:
        d = np.sqrt(((p[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]
        clear = d.min(axis=1)
    else:
        clear = np.full(p.shape[0], np.inf)
    wallc = np.minimum.reduce([p[:, 0] - LO, HI - p[:, 0], p[:, 1] - LO, HI - p[:, 1]])
    return np.minimum(clear, wallc)


def _insert_at_holes(xy, r, k, rng, probes=384):
    """Append k circles at the k successively emptiest points of the layout (greedy, re-probed each
    time so two inserts never land on top of each other)."""
    xy = np.array(xy, dtype=float)
    r = np.array(r, dtype=float)
    for _ in range(int(k)):
        p = rng.uniform(LO + 1e-4, HI - 1e-4, size=(probes, 2))
        c = _clearance(p, xy, r)
        j = int(np.argmax(c))
        xy = np.vstack([xy, p[j]])
        r = np.concatenate([r, [max(min(float(c[j]) * 0.5, 0.5), RMIN)]])
    return xy, r


def _transfer_start(src_xy, src_r, n, rng, mode=0):
    """CROSS-n TRANSFER: turn an m-circle layout into an n-circle start.

    m > n: drop m-n circles -- what is left is an already-optimised skeleton with room to grow.
    m < n: insert n-m circles at the emptiest points.  Either way the result is still strictly
    non-overlapping, so SLP can take it straight away.  The census is a warm-start GRAPH, not 37
    independent problems: neighbouring n share almost all of their structure.

    `mode` is iteration 5's addition.  Mode 0 is the iteration-4 rule (drop the smallest / insert at
    holes) -- deterministic, so a donor could only ever be *tried once*.  Modes 1-2 make the transfer a
    STOCHASTIC operator, so the same donor pair (m, n) is an inexhaustible supply of distinct starts:
      1  drop a random subset of the 2(m-n) smallest / churn: also evict a few smalls before growing;
      2  drop a spatially CONTIGUOUS cluster -- one large hole to refill instead of scattered slivers,
         which is the only variant that can change the layout's coarse structure rather than its edges.
    Mode >= 1 with m == n is a RESEED: same size, small circles evicted and re-inserted elsewhere."""
    m = int(src_xy.shape[0])
    xy0 = np.array(src_xy, dtype=float)
    r0 = np.array(src_r, dtype=float)
    if m == n:
        if mode >= 1 and m >= 4:
            k = int(rng.randint(1, max(2, m // 10)))
            keep = np.argsort(r0)[k:]
            return _insert_at_holes(xy0[keep], r0[keep], m - int(keep.size), rng)
        return xy0, r0
    if m > n:
        d = m - n
        if mode == 0 or m < 4:
            keep = np.argsort(r0)[d:]
        elif mode == 1:
            cand = np.argsort(r0)[:min(2 * d, m - 1)]
            drop = rng.choice(cand, size=min(d, cand.size), replace=False)
            keep = np.setdiff1d(np.arange(m), drop)
        else:
            c = int(rng.randint(m))
            order = np.argsort(((xy0 - xy0[c]) ** 2).sum(1))
            keep = np.setdiff1d(np.arange(m), order[:d])
        keep = keep[:n] if keep.size >= n else keep
        xy1, r1 = xy0[keep], r0[keep]
        if xy1.shape[0] < n:                               # defensive: never return the wrong count
            xy1, r1 = _insert_at_holes(xy1, r1, n - xy1.shape[0], rng)
        return np.array(xy1, dtype=float), np.array(r1, dtype=float)
    g = n - m
    if mode >= 1 and m >= 4:
        k = min(int(rng.randint(1, max(2, m // 8))), m - 1)
        keep = np.argsort(r0)[k:]
        return _insert_at_holes(xy0[keep], r0[keep], g + k, rng)
    return _insert_at_holes(xy0, r0, g, rng)


def _region_move(host_xy, host_r, rng, donor_xy=None, donor_r=None, frac=None, probes=256):
    """SPATIAL RECOMBINATION -- ruin a contiguous REGION of one layout and recreate it, optionally out
    of the corresponding region of ANOTHER layout.

    Every operator up to iteration 5 either moves the whole layout a little (jitter, LP shake) or moves
    the *dead weight* (hole move relocates the k smallest circles) or swaps the layout wholesale for a
    neighbouring n's (transfer).  None of them can rebuild a PATCH: take the k circles nearest a random
    point -- large ones included -- delete them, and fill the hole they leave with the circles that
    another elite put in that same part of the square.  The surrounding (n-k) circles are untouched, so
    the graft is constrained by real optimised structure on all sides instead of being a free-floating
    seam, and the patch carries a genuinely different contact pattern into the host.

    donor may have a DIFFERENT circle count than the host (cross-n recombination) or be None, in which
    case the region is refilled greedily at the clearance maxima -- large-neighbourhood ruin & recreate,
    which `_hole_move` cannot do because it only ever relocates the smallest circles.

    Donor circles are accepted one at a time with the radius clipped to the clearance actually left at
    that point, and any circle that would be crushed is skipped, so the result is non-overlapping BY
    CONSTRUCTION -- no repair phase, and `_shrink_feasible` afterwards is a formality."""
    xy0 = np.array(host_xy, dtype=float)
    r0 = np.array(host_r, dtype=float)
    n = xy0.shape[0]
    if n < 3:
        return xy0, r0
    if frac is None:
        frac = rng.uniform(0.08, 0.35)
    k = int(min(max(2, round(frac * n)), n - 1))
    c = xy0[int(rng.randint(n))] + rng.normal(0.0, 0.05, size=2)
    c = np.clip(c, LO, HI)
    order = np.argsort(((xy0 - c[None, :]) ** 2).sum(1))
    keep = np.sort(order[k:])
    xy = xy0[keep]
    r = r0[keep]
    if donor_xy is not None and donor_xy.shape[0] >= 1:
        dxy = np.asarray(donor_xy, dtype=float)
        dr = np.asarray(donor_r, dtype=float)
        dorder = np.argsort(((dxy - c[None, :]) ** 2).sum(1))
        for j in dorder:
            if xy.shape[0] >= n:
                break
            p = dxy[j:j + 1]
            cl = float(_clearance(p, xy, r)[0])
            if cl <= 1e-5:
                continue                                    # crushed by the surviving structure -- skip
            xy = np.vstack([xy, dxy[j]])
            r = np.concatenate([r, [max(min(float(dr[j]), cl * (1.0 - 1e-9)), RMIN)]])
    if xy.shape[0] < n:
        xy, r = _insert_at_holes(xy, r, n - xy.shape[0], rng, probes=probes)
    return np.array(xy[:n], dtype=float), np.array(r[:n], dtype=float)


# --------------------------------------------------------------------------- symmetry projection
# The five INVOLUTIONS of the square's dihedral group, as 2x2 matrices acting on centred coordinates.
# (The two 90-degree rotations are order 4, so a single averaging pass cannot make a layout exactly
# invariant under them; the involutions can, and they are the symmetries the published csqv optima
# actually show.)
_SYM_G = (
    np.array([[-1.0, 0.0], [0.0, 1.0]]),      # mirror in the vertical axis   x -> -x
    np.array([[1.0, 0.0], [0.0, -1.0]]),      # mirror in the horizontal axis y -> -y
    np.array([[0.0, 1.0], [1.0, 0.0]]),       # mirror in the diagonal        (x,y) -> (y,x)
    np.array([[0.0, -1.0], [-1.0, 0.0]]),     # mirror in the anti-diagonal
    np.array([[-1.0, 0.0], [0.0, -1.0]]),     # rotation by 180 degrees
)


def _symmetry_move(xy, r, rng, g=None, alpha=None):
    """REPRESENTATION-SPACE perturbation (iteration 10): project a layout onto the nearest EXACTLY
    symmetric layout and hand the result back to SLP.

    Every operator before this one moves within the 2n-dimensional space of free centres: jitter and
    the LP shake take small steps, region/hole/transfer rebuild a patch.  None of them can impose a
    GLOBAL relation between distant circles -- and that is precisely the structure the published
    optima have.  Under a symmetry g of the square, the optimal layout of many n is invariant as a
    SET: g maps circle i onto some circle pi(i) of the same radius.  A near-optimal layout is usually
    *almost* invariant, but the local optimiser has no force pulling it the rest of the way, so it
    settles into a slightly-broken-symmetry basin that is a strict local optimum.

    The move:
      1. match the layout to its own image under g with an exact assignment (Hungarian on the squared
         distances) -- pi = the correspondence "which circle plays the role of g(i)";
      2. average each circle with the g-image of its partner.  For an INVOLUTIVE pi this yields a
         configuration that satisfies g(Y_i) = Y_{pi(i)} EXACTLY, not approximately: the projection
         lands on the symmetric subspace rather than near it.  Non-involutive indices (a mismatch the
         assignment could not pair up) are left where they are, which only makes the move gentler.
      3. `alpha` interpolates: 1.0 is the full projection, smaller values a partial pull, so the arm
         can offer both "snap onto the symmetric subspace" and "lean towards it".

    Radii are averaged over the pair and then shrunk, so the result is strictly feasible by
    construction.  Returns (xy, r); a no-op copy if the layout is too small to symmetrize."""
    X = np.array(xy, dtype=float)
    r0 = np.array(r, dtype=float)
    n = X.shape[0]
    if n < 2:
        return X, r0
    if g is None:
        g = _SYM_G[int(rng.randint(len(_SYM_G)))]
    G = np.asarray(g, dtype=float)
    if alpha is None:
        alpha = 1.0 if rng.rand() < 0.6 else float(rng.uniform(0.35, 0.9))
    GX = X.dot(G.T)                                        # the layout seen through the mirror
    cost = ((GX[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    # radii should match too: a big circle must not be paired with a speck just because they are close
    cost = cost + 4.0 * (r0[:, None] - r0[None, :]) ** 2
    try:
        _, pi = linear_sum_assignment(cost)
    except Exception:
        return X, r0
    pi = np.asarray(pi, dtype=int)
    inv = pi[pi] == np.arange(n)                           # indices where pi is an involution
    Y = X.copy()
    rr = r0.copy()
    if inv.any():
        # partner of i is pi(i); its mirror pre-image is g(X_{pi(i)}) because g is an involution
        mate = X[pi].dot(G.T)
        Y[inv] = X[inv] + alpha * 0.5 * (mate[inv] - X[inv])
        rr[inv] = r0[inv] + alpha * 0.5 * (r0[pi][inv] - r0[inv])
    Y = np.clip(Y, LO + EDGE, HI - EDGE)
    rr = np.maximum(rr, RMIN)
    return Y, _shrink_feasible(Y, rr)


def _sym_defect(xy, r, g):
    """Mean squared distance from `xy` to its own g-image under the best matching -- 0 iff the layout
    is exactly g-symmetric.  Used by the self-test and to pick the CLOSEST symmetry to project onto."""
    X = np.asarray(xy, dtype=float)
    GX = X.dot(np.asarray(g, dtype=float).T)
    cost = ((GX[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    _, pi = linear_sum_assignment(cost)
    return float(cost[np.arange(X.shape[0]), pi].mean())


def _nearest_sym(xy, r):
    """The element of `_SYM_G` this layout is already closest to being invariant under."""
    d = [_sym_defect(xy, r, g) for g in _SYM_G]
    return _SYM_G[int(np.argmin(d))], float(min(d))


# ------------------------------------------------------- symmetry-CONSTRAINED optimisation (iter 11)
def _sym_perm(xy, r, g):
    """The correspondence pi: circle i plays the role of g(circle pi(i)).  Same matching cost as
    `_symmetry_move` (squared centre distance + a radius-mismatch term), exposed on its own so the
    constrained optimiser can BUILD the constraint from it rather than only apply it once."""
    X = np.asarray(xy, dtype=float)
    r0 = np.asarray(r, dtype=float)
    GX = X.dot(np.asarray(g, dtype=float).T)
    cost = ((GX[:, None, :] - X[None, :, :]) ** 2).sum(-1) + 4.0 * (r0[:, None] - r0[None, :]) ** 2
    try:
        _, pi = linear_sum_assignment(cost)
    except Exception:
        return np.arange(X.shape[0])
    return np.asarray(pi, dtype=int)


def _sym_reduction(n, pi, g):
    """Build the linear map  z_full (3n) = M @ z_red  whose range is exactly the g-SYMMETRIC subspace.

    Iteration 10 could only PROJECT onto that subspace and then hand the layout back to a free SLP,
    which is at liberty to walk straight back off it -- symmetry was a perturbation, never a property
    of the search.  Here it becomes the coordinate system:

      * a 2-cycle {i, j=pi(i)} keeps ONE displacement pair and ONE radius: D_j = g D_i, r_j = r_i.
        (Y_j = g Y_i is preserved to machine precision by every step, not restored after the fact.)
      * a FIXED point (pi(i)=i) sits on the mirror axis and may only slide ALONG it: its displacement
        is restricted to the +1 eigenspace of g -- one direction for a mirror, none at all for the
        180-degree rotation, whose only fixed point is the centre of the square.
      * an index the assignment could not pair involutively is left fully free, so a partly-symmetric
        layout still gets a partial constraint instead of nothing.

    Roughly half the variables disappear.  The basis vectors are scaled to max|entry| = 1 and every g
    here is a SIGNED PERMUTATION, so |D|_inf is preserved by the substitution and the box |z_red| <=
    delta is still exactly the trust region |D_i|_inf <= delta on every circle, image ones included.

    Returns (M, isD) or None when no reduction is possible."""
    G = np.asarray(g, dtype=float)
    _, sv, vt = np.linalg.svd(G - np.eye(2))
    basis = []
    for v in np.atleast_2d(vt[sv < 1e-9]):
        mx = float(np.max(np.abs(v)))
        if mx > 1e-9:
            basis.append(v / mx)
    rows, cols, vals, isD = [], [], [], []
    k = 0
    seen = np.zeros(n, dtype=bool)
    for i in range(n):
        if seen[i]:
            continue
        j = int(pi[i])
        if j != i and 0 <= j < n and int(pi[j]) == i:
            seen[i] = seen[j] = True
            for a in range(2):                       # D_i = (t0, t1),  D_j = G D_i
                rows.append(i + a * n); cols.append(k); vals.append(1.0)
                for b in range(2):
                    if abs(G[b, a]) > 1e-12:
                        rows.append(j + b * n); cols.append(k); vals.append(float(G[b, a]))
                isD.append(True); k += 1
            rows += [2 * n + i, 2 * n + j]; cols += [k, k]; vals += [1.0, 1.0]
            isD.append(False); k += 1
        else:
            seen[i] = True
            dirs = basis if j == i else [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
            for v in dirs:
                for a in range(2):
                    if abs(v[a]) > 1e-12:
                        rows.append(i + a * n); cols.append(k); vals.append(float(v[a]))
                isD.append(True); k += 1
            rows.append(2 * n + i); cols.append(k); vals.append(1.0)
            isD.append(False); k += 1
    if k == 0 or k >= 3 * n:
        return None
    M = csr_matrix((vals, (rows, cols)), shape=(3 * n, k))
    return M, np.array(isD, dtype=bool)


def _sym_slp_refine(xy, r, g, slice_end, delta0=0.02, max_it=250, dmin=1e-9):
    """SLP run INSIDE the symmetric subspace: same monotone, always-feasible trust-region ascent as
    `_slp_refine`, but every LP is posed in the reduced coordinates of `_sym_reduction`.  Half the
    variables, half the search directions -- and the ones removed are exactly the ones that leak out
    of the symmetric basin.  The constraint rows are untouched, so every iterate is still strictly
    non-overlapping; this cannot return something worse than it was given."""
    n = int(np.asarray(xy).shape[0])
    xy = np.array(xy, dtype=float)
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    if n < 2:
        return xy, r
    red = _sym_reduction(n, _sym_perm(xy, r, g), g)
    if red is None:
        return xy, r
    cur = float(r.sum())
    delta = float(delta0)
    for _ in range(max_it):
        if time.process_time() > slice_end or delta < dmin:
            break
        out = _lp_step(xy, r, delta, red=red)
        if out is None:
            delta *= 0.3
            continue
        nxy, nr, step = out
        s = float(nr.sum())
        if s > cur + 1e-15:
            gain = s - cur
            xy, r, cur = nxy, nr, s
            if gain < 1e-12 * max(cur, 1.0):
                delta *= 0.3
            elif float(np.max(np.abs(step))) > 0.9 * delta:
                delta = min(delta * 1.5, 0.08)
        else:
            delta *= 0.3
    return xy, r


def _sym_constrained(xy, r, rng, slice_end, g=None):
    """The ninth arm: snap onto a symmetry, then POLISH THE BASIN AS A SYMMETRIC OBJECT.

    Arm 8 (`symmetry`) and this one keep both lines alive on purpose: arm 8 asks "is there a better
    layout near the symmetric one?", arm 9 asks "what is the best SYMMETRIC layout in this basin?".
    They answer different questions and the bandit is free to prefer either per n."""
    if g is None:
        g = _nearest_sym(xy, r)[0] if rng.rand() < 0.667 else _SYM_G[int(rng.randint(len(_SYM_G)))]
    Y = _symmetry_move(xy, r, rng, g=g, alpha=1.0)[0]
    return _sym_slp_refine(Y, _lp_radii(Y), g, slice_end)


ELITES = 4                 # distinct layouts kept per n in the archive
_ELITE_FLOOR = 0.98        # an elite may be at most 2% below the best known sum for its n


def _records():
    """The read-only packomania table, {n: record}.  Cached; every failure degrades to {} (no records
    means nothing is ever treated as capped, which is the safe direction)."""
    global _RECORDS
    if _RECORDS is None:
        rec = {}
        try:
            with open("bench/records.json") as f:
                raw = json.load(f)
            src = raw.get("records", {}) if isinstance(raw, dict) else {}
            for k, v in src.items():
                try:
                    rec[int(k)] = float(v)
                except (TypeError, ValueError):
                    pass
        except Exception:                                  # noqa: BLE001 -- diagnostics only, never fatal
            rec = {}
        _RECORDS = rec
    return _RECORDS


_RECORDS = None


def _sig(r):
    """Structural signature of a layout: its radii, sorted.  Two layouts with the same contact graph
    have near-identical signatures; two genuinely different basins do not.  Cheap and permutation- and
    symmetry-invariant, which is exactly what a diversity filter needs."""
    return np.sort(r)[::-1]


def _pool_init(pool, m):
    """The elite list for m, seeded (once) from the committed pack.  [] means 'no pack, no elites' --
    the guarded warm-start miss, cached so it costs one stat() per m and not one per trial."""
    if m not in pool:
        got = _read_pack_full(m)
        pool[m] = [] if got is None else [(got[0], got[1], float(got[1].sum()))]
    return pool[m]


def _pool_add(pool, m, xy, r, s):
    """Insert a layout into m's ELITE ARCHIVE.

    The archive, not the single incumbent, is what iteration 5 adds.  A monotone pool keeps exactly one
    layout per n, so the moment an n settles into a deep local optimum every operator -- perturbation
    AND transfer -- restarts from that same structure forever, and the runner-up basins that were
    already found are thrown away.  Here the top `ELITES` structurally DISTINCT layouts survive
    (distinct by radius signature), so a donation can carry a basin that is not the current best, and a
    perturbation can walk away from the incumbent without losing it."""
    el = _pool_init(pool, m)
    if el and s < el[0][2] * _ELITE_FLOOR:
        return                                             # junk donor -- would only waste trials
    sg = _sig(r)
    for i, e in enumerate(el):
        if e[1].shape[0] == r.shape[0] and float(np.abs(_sig(e[1]) - sg).sum()) < 1e-4:
            if s > e[2]:                                   # same basin, better point -> replace
                el[i] = (np.array(xy, float), np.array(r, float), float(s))
                el.sort(key=lambda t: -t[2])
            return
    el.append((np.array(xy, float), np.array(r, float), float(s)))
    el.sort(key=lambda t: -t[2])
    del el[ELITES:]


def _pool_get(pool, m):
    """Best known layout for m: this run's improved one if we have it, else the committed pack."""
    el = _pool_init(pool, m)
    return None if not el else (el[0][0], el[0][1])


def _pool_pick(pool, m, rng):
    """A donor layout for m -- usually the best, sometimes a runner-up basin (that is the point).

    Returns (xy, r, s); callers that only want the layout index [0] and [1] as before."""
    el = _pool_init(pool, m)
    if not el:
        return None
    i = 0 if (len(el) == 1 or rng.rand() < 0.55) else int(rng.randint(1, len(el)))
    return (el[i][0], el[i][1], el[i][2])


def _pick_donor(pool, n, rng):
    """A layout to graft a REGION from.  Three sources, deliberately mixed:
      * another elite of n itself  -- crossover between two distinct basins of the SAME problem, the
        operator the archive was built for and the one iteration 5 never actually used;
      * a census neighbour's elite -- regional cross-n transfer, finer-grained than whole-layout
        transfer, which can only drop or insert circles wholesale;
      * nothing                    -- pure ruin & recreate, refilling the hole from scratch."""
    u = rng.rand()
    if u < 0.20:
        return None, None
    el = _pool_init(pool, n)
    if u < 0.65 and len(el) >= 2:
        i = int(rng.randint(1, len(el)))
        return el[i][0], el[i][1]
    cand = [m for m in range(max(1, n - 10), n + 11, 2) if m != n and _pool_init(pool, m)]
    if cand:
        got = _pool_pick(pool, cand[int(rng.randint(len(cand)))], rng)
        if got is not None:
            return got[0], got[1]
    if len(el) >= 2:
        i = int(rng.randint(1, len(el)))
        return el[i][0], el[i][1]
    return None, None


def _capped(pool, n):
    """True iff n's committed pack already scores the 7-digit cap.  Such an n cannot gain another
    digit, so spending a time slice on it is pure waste -- it stays in the archive as a DONOR and its
    pack survives untouched (the driver is append-or-improve)."""
    rec = _records().get(int(n))
    cur = _pool_get(pool, n)
    if rec is None or rec <= 0.0 or cur is None:
        return False                                       # unknown n / no pack -> always work on it
    return (rec - float(cur[1].sum())) / rec <= 1e-7


def _transfer_trial(evaluate, meter, pool, rng, n, src_m, slice_end, best_s, mode=0):
    """One cross-n transfer: pick a donor ELITE for src_m, build the start with `mode`, converge it
    with SLP, score it once.  Every feasible result -- winner or not -- enters n's own archive, so a
    losing transfer still contributes the basin it found."""
    src = _pool_pick(pool, src_m, rng)
    if src is None or meter.left() <= 0:
        return None
    xy, r = _transfer_start(src[0], src[1], n, rng, mode)
    if xy.shape[0] != n:
        return None
    xy = np.clip(xy, LO + EDGE, HI - EDGE)
    r = _shrink_feasible(xy, r)
    xy, r = _slp_refine(xy, r, min(slice_end + 0.35, time.process_time() + 2.5), delta0=0.02)
    feas, s = evaluate(n, np.concatenate([xy, r[:, None]], axis=1))
    if not feas:
        return None
    _pool_add(pool, n, xy, r, float(s))
    return xy, r, float(s)                                 # ALWAYS reported: the bandit needs the
    #                                                        value of a losing trial as much as a win


# =============================================================================== operator bandit
# WHY THIS EXISTS (iteration 8).  Iterations 4-7 each added a move class, and each was wired in at a
# hand-tuned FIXED share of one branch of one `if`.  The census then measured the cost of that: the
# iter-7 head-to-head found `_equalize_move` alone reaching 7.000 digits at n=93 in 8 s, but the
# shipped solver -- where that operator fires on ~25% of ~55% of trials, i.e. under a second of a
# ~2.5 s slice -- delivered 2.924 there.  The operator was not weak; it was DILUTED.  A fixed mix also
# cannot express the thing every head-to-head has shown: the best operator DIFFERS BY n (equalize won
# at 75/93 and lost at 49/51).  So the allocation itself becomes learned, per n, online.
#
# Seven arms; six are the existing move classes plus the cold multistart, so this adds no new
# geometry -- it only decides.  Per-n Beta posteriors over a reward in [0,1], sampled by Thompson.
# Two things make ~15-30 trials per n enough to learn from:
#   * HIERARCHY -- each per-n posterior is pulled toward the GLOBAL rate for that arm, pooled over
#     every n in the call (~600 trials), so an n with 8 trials still inherits census-wide evidence;
#   * a PRIOR equal to the iter-7 fixed mix, so trial 1 behaves like the tuned solver and the data
#     has to earn any departure from it.  The bandit can only be as bad as the mix it starts from.
# Counts DECAY, so an arm that stops paying at this n is forgotten and re-explored later.
# Iteration 10 adds an EIGHTH arm, `symmetry`, at zero tuning cost -- which is the whole point of
# having made the allocation learned: a new move class no longer needs a hand-picked share, it needs
# only a prior, and the census decides per n whether it pays.  Its prior share is set to the mean of
# the other perturbation arms, so it starts on equal footing with them and neither the old mix nor the
# new operator is privileged.
_ARMS = ("jitter", "hole", "shake", "equalize", "region", "transfer", "cold", "symmetry", "symslp")
A_JIT, A_HOLE, A_SHAKE, A_EQ, A_REG, A_XFER, A_COLD, A_SYM, A_SYMC = range(9)
# the iteration-7 shipped mix, as selection probabilities (0.40 transfer, then 0.25 of the rest cold,
# the remainder split 15/15/15/25/30 across jitter/hole/shake/equalize/region), renormalised with the
# new arm at the mean perturbation share (0.0956)
# ...and iteration 11's symmetry-CONSTRAINED arm enters at the SAME prior as the projection arm, so
# neither symmetry variant is privileged over the other and the census decides per n which pays.
_ARM_MIX = np.array([0.0675, 0.0675, 0.0675, 0.1125, 0.1350, 0.4000, 0.1500, 0.0956, 0.0956])
_ARM_MIX = _ARM_MIX / _ARM_MIX.sum()
_ARM_P0 = _ARM_MIX / _ARM_MIX.max()        # prior mean per arm, in (0, 1]
_BD_LAM = 3.0                              # pseudo-observations drawn at the GLOBAL rate
_BD_P0 = 2.0                               # pseudo-observations drawn at the iter-7 fixed mix
_BD_DECAY_N = 0.97                         # per-n recency (effective memory ~33 trials)
_BD_DECAY_G = 0.997                        # global recency (~330 trials)


def _bandit():
    return {"g": np.zeros((2, len(_ARMS)))}


def _arm_pick(bd, n, rng, avail):
    """Thompson sample an arm from the hierarchical posterior, restricted to the available arms."""
    A = len(_ARMS)
    loc = bd.get(n)
    if loc is None:
        loc = bd[n] = np.zeros((2, A))
    g = bd["g"]
    grate = (g[0] + 1.0) / (g.sum(axis=0) + 2.0)
    a = 1.0 + loc[0] + _BD_LAM * grate + _BD_P0 * _ARM_P0
    b = 1.0 + loc[1] + _BD_LAM * (1.0 - grate) + _BD_P0 * (1.0 - _ARM_P0)
    th = np.asarray(rng.beta(a, b), dtype=float)
    av = np.asarray(avail, dtype=bool)
    if not av.any():
        return A_COLD
    th[~av] = -1.0
    return int(np.argmax(th))


def _arm_reward(s, best_s, start_s):
    """Credit in [0, 1] for one scored trial.  Three tiers, deliberately: a bandit fed only 'did it
    set a new record for this n' sees almost all zeros once an n stalls and degenerates to its prior,
    which is exactly the regime that matters most here."""
    if not np.isfinite(s):
        return 0.0
    if not np.isfinite(best_s):
        return 1.0                                         # first feasible layout for this n
    if s > best_s + 1e-12:
        return 1.0                                         # new best for n
    if start_s is not None and np.isfinite(start_s) and s > start_s + 1e-9:
        return 0.35                                        # beat the elite it was launched from
    if best_s > 0.0 and s > best_s * 0.999:
        return 0.15                                        # a competitive distinct basin
    return 0.0


def _arm_update(bd, n, arm, w):
    if arm is None or arm < 0:
        return
    loc = bd.get(n)
    if loc is None:
        loc = bd[n] = np.zeros((2, len(_ARMS)))
    loc *= _BD_DECAY_N
    bd["g"] *= _BD_DECAY_G
    w = float(min(1.0, max(0.0, w)))
    loc[0, arm] += w
    loc[1, arm] += 1.0 - w
    bd["g"][0, arm] += w
    bd["g"][1, arm] += 1.0 - w


# ============================================================ time scheduler (the OUTER allocation)
# WHY THIS EXISTS (iteration 9).  Iteration 8 learned WHICH MOVE to make at each n; the allocation it
# ran inside was still the one written in iteration 4: one pass over the work list, each n handed a
# slice proportional to sqrt(n), never revisited.  That constant is a claim about where a CPU second
# buys the most DIGITS -- and the census refutes it: iter-8's own table shows +0.712 d bought at n=61
# and +0.000 at n=33/45/55/57/59/81 in slices sized by the same rule.  sqrt(n) prices the COST of a
# restart; it says nothing about its YIELD, and the score is a mean over digits, not over seconds.
#
# So the sweep becomes ROUNDS.  Every n is visited in each round, its digit gain per CPU second is
# measured, and the next round's shares are set from those measurements: share ~ sqrt(n) * (rate +
# eps * mean_rate).  The eps term is the exploration floor -- an n that yielded nothing still gets a
# fixed fraction of an average n's slice, because "nothing this round" is very often "nothing YET".
# Three second-order effects come free:
#   * an n that stalls no longer holds a large slice hostage for the whole call;
#   * rounds ALTERNATE direction, so an improvement propagates up and down the transfer chain
#     several times instead of once;
#   * the archive persists across rounds, so a later visit RESUMES the search rather than restarting.
_SCHED_ROUNDS = 2          # visits per n (measured: 3 rounds fragments the ~3 s/n slice too far)
_SCHED_EPS = 0.15          # a zero-yield n still gets 15% of a mean-yield n's share
_SCHED_EWMA = 0.60         # weight on the newest rate observation


def _cur_sum(pool, n):
    """Best sum r known for n right now: this run's archive if it has one, else the committed pack,
    else None.  GUARDED for an n with no pack (fresh n / offline re-run on other sizes)."""
    cur = _pool_get(pool, n)
    if cur is not None:
        return float(cur[1].sum())
    got = _read_pack_full(n)
    if got is None:
        return None
    return float(got[1].sum())


def _digits_at(n, s):
    """The scorer's own digit rule, re-implemented locally (importing harness is forbidden).  An n
    with no record contributes no measurable yield, so it reads as 0 and rides the exploration floor."""
    rec = _records().get(int(n))
    if rec is None or rec <= 0.0 or s is None:
        return 0.0
    rel = (rec - float(s)) / rec
    if rel <= 1e-7:
        return 7.0
    return float(max(0.0, min(7.0, -np.log10(rel))))


def _sched_weights(live, rate, eps=_SCHED_EPS):
    """Time shares for one round: sqrt(n) (the cost prior) scaled by measured digits/second (the
    yield), with an exploration floor.  With no measurements at all this is exactly the iteration-4
    sqrt(n) split, so round 1 behaves like the solver it replaces and data has to earn any departure."""
    base = np.array([np.sqrt(float(m)) for m in live], dtype=float)
    base = base / max(base.sum(), 1e-12)
    known = [rate[m] for m in live if rate.get(m) is not None]
    mr = float(np.mean(known)) if known else 0.0
    if mr <= 0.0:
        return base                                        # nothing measured yet -> the cost prior
    rr = np.array([mr if rate.get(m) is None else float(rate[m]) for m in live], dtype=float)
    w = base * (np.maximum(rr, 0.0) + eps * mr)
    tot = w.sum()
    return base if not np.isfinite(tot) or tot <= 0.0 else w / tot


# --------------------------------------------------------------------------- the solver
def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    deadline = time.process_time() + CPU_BUDGET               # RELATIVE: correct on every call
    tg = sorted(set(int(t) for t in targets if int(t) >= 1))
    if not tg:
        return
    pool = {}                                                 # n -> elite archive (list of (xy, r, s))
    bd = _bandit()                                            # operator credit, shared across all n
    # --- CPU REALLOCATION -------------------------------------------------------------------------
    # An n whose committed pack is already at the 7-digit cap has NOTHING left to win: its digits are
    # clamped, so every second spent there is a second not spent on the 33 n that can still move.  It
    # is dropped from the work list but stays in the archive as a DONOR -- and because the driver is
    # append-or-improve, its pack survives untouched even though solve() never scores it again.
    work = [n for n in tg if not _capped(pool, n)]
    if not work:
        work = list(tg)                                       # everything capped -> keep grinding anyway
    # Hold back a slice of the budget for a DESCENDING transfer sweep at the end: the ascending main
    # sweep can only propagate an improvement from n to n+2, and the coupling runs both ways.
    back = 0.13 * CPU_BUDGET if len(work) > 2 else 0.0
    main_end = deadline - back
    # --- ROUND-BASED ADAPTIVE TIME SCHEDULE (iteration 9) -----------------------------------------
    # Round 1 splits by sqrt(n) exactly as before; every later round splits by the digits/second each
    # n actually delivered.  All clocks are read from the ACTUAL process time at each step, so the
    # split is self-correcting, survives len(targets) == 1, and never overruns main_end.
    rate = {}                                                 # n -> EWMA of measured digits / CPU s
    live = list(work)
    for rd in range(_SCHED_ROUNDS):
        if not live or meter.left() <= 0:
            break
        t_left = main_end - time.process_time()
        if t_left <= 0.05:
            break
        round_end = time.process_time() + t_left / (_SCHED_ROUNDS - rd)
        order = list(live) if rd % 2 == 0 else list(reversed(live))
        w = dict(zip(order, _sched_weights(order, rate)))
        for pos, n in enumerate(order):
            now = time.process_time()
            left_t = round_end - now
            if left_t <= 0.03 or meter.left() <= 0:
                break
            wrem = sum(w[m] for m in order[pos:])
            slice_end = now + left_t * (w[n] / max(wrem, 1e-12))
            s0 = _digits_at(n, _cur_sum(pool, n))
            _solve_one(evaluate, meter, rng, n, slice_end, pool, bd)
            used = max(time.process_time() - now, 1e-3)
            obs = max(0.0, _digits_at(n, _cur_sum(pool, n)) - s0) / used
            rate[n] = obs if rate.get(n) is None else _SCHED_EWMA * obs + (1.0 - _SCHED_EWMA) * rate[n]
        live = [m for m in live if not _capped(pool, m)]       # an n that reached the cap retires
    # --- descending transfer sweep: every n now pulls from the (possibly just improved) n+2, n+4 -----
    # Now with structural modes as well as sizes, so the sweep samples five DIFFERENT starts, not three.
    plan = ((2, 0), (4, 0), (-2, 0), (2, 2), (6, 1), (-4, 1))
    for pos, n in enumerate(reversed(work)):
        left_t = deadline - time.process_time()
        if left_t <= 0.05 or meter.left() <= 0:
            break
        rest = len(work) - pos
        sub_end = time.process_time() + max(left_t / rest, 0.05)
        cur = _pool_get(pool, n)
        best_s = -np.inf if cur is None else float(cur[1].sum()) - 1e-12
        for off, mode in plan:
            if time.process_time() >= sub_end or meter.left() <= 0:
                break
            if n + off < 1:
                continue
            got = _transfer_trial(evaluate, meter, pool, rng, n, n + off, sub_end, best_s, mode)
            if got is not None:
                _arm_update(bd, n, A_XFER, _arm_reward(got[2], best_s, None))
                if got[2] > best_s:
                    best_s = got[2]


def _hole_move(xy, r, rng, k=1, probes=256):
    """Relocate the k SMALLEST circles into the emptiest spots of the current layout.

    The discrete layer the continuous method cannot cross: a circle squeezed to near-zero radius is
    dead weight worth exactly nothing to sum r, and no local move can walk it across its neighbours.
    Teleporting it to the point that maximises the clearance  min(wall, min_j(d_j - r_j))  changes the
    CONTACT GRAPH -- a different basin, not a different point in the same one."""
    xy = xy.copy()
    order = np.argsort(r)
    for t in range(min(k, xy.shape[0])):
        i = order[t]
        p = rng.uniform(LO + 1e-4, HI - 1e-4, size=(probes, 2))
        mask = np.ones(xy.shape[0], dtype=bool)
        mask[order[:t + 1]] = False
        if mask.any():
            d = np.sqrt(((p[:, None, :] - xy[None, mask, :]) ** 2).sum(-1)) - r[mask][None, :]
            clear = d.min(axis=1)
        else:
            clear = np.full(probes, np.inf)
        wallc = np.minimum.reduce([p[:, 0] - LO, HI - p[:, 0], p[:, 1] - LO, HI - p[:, 1]])
        xy[i] = p[int(np.argmax(np.minimum(clear, wallc)))]
    return xy


def _solve_one(evaluate, meter, rng, n, slice_end, pool=None, bd=None):
    """Multi-start SLP with basin hopping.  One restart = init -> (optional penalty pre-conditioning)
    -> LP radii -> SLP to convergence -> ONE metered evaluate().  Measured cost at n=51: ~0.16 s, so a
    2-3 s slice buys 15-25 genuinely converged local optima instead of one under-converged one."""
    if pool is None:
        pool = {}
    if bd is None:
        bd = _bandit()
    got = _pool_get(pool, n)
    warm = None if got is None else got[0]
    best_xy, best_r = None, None
    # Start the bar at the committed pack's own value (the warm SLP restart can only match or beat it,
    # since _lp_radii is optimal for those centres and SLP is monotone).  Without this the first
    # transfer trial would report something WORSE than the census as an improvement.
    best_s = -np.inf if got is None else float(got[1].sum()) - 1e-12
    # Cross-n transfer starts are tried FIRST and once each: a neighbouring n's optimised layout is a
    # far better start than any cold init, and it is the only start that imports another basin.
    srcs = [m for m in (n - 2, n + 2, n - 4, n + 4) if m >= 1]
    # ...and once those are spent, the donor set WIDENS to every census neighbour within +-10.  A basin
    # can therefore travel several hops (n-6 -> n-4 -> n-2 -> n across one ascending sweep, or straight
    # across in one trial), which single-hop transfer could not do.
    wide = [m for m in range(max(1, n - 10), n + 11, 2) if m != n]
    trial = 0
    maxiter = 60 if n <= 60 else 45
    while time.process_time() < slice_end and meter.left() > 0:
        pen = False
        d0 = 0.03
        r = None
        arm = -1
        start_s = None
        # --- pick a start -------------------------------------------------------------------------
        if trial == 0 and warm is not None:
            xy = warm.copy()
        elif srcs:
            # the four deterministic single-hop transfers still go first and once each -- they are the
            # cheapest known-good starts AND they hand the bandit four free transfer-arm samples.
            src_m = srcs.pop(0)
            got = _transfer_trial(evaluate, meter, pool, rng, n, src_m, slice_end, best_s)
            trial += 1
            if got is not None:
                _arm_update(bd, n, A_XFER, _arm_reward(got[2], best_s, None))
                if got[2] > best_s:
                    best_xy, best_r, best_s = got[0], got[1], got[2]
            continue
        else:
            # --- WHICH OPERATOR?  Learned per n, not hand-wired (iteration 8) ----------------------
            have_el = bool(_pool_init(pool, n))
            cand = [m for m in wide if _pool_init(pool, m)]
            avail = np.array([have_el, have_el, have_el, have_el, have_el, bool(cand), True,
                              have_el, have_el])
            arm = _arm_pick(bd, n, rng, avail)
            if arm == A_XFER:
                # STOCHASTIC TRANSFER: a random available donor within +-10, a random structural mode.
                src_m = cand[int(rng.randint(len(cand)))]
                got = _transfer_trial(evaluate, meter, pool, rng, n, src_m, slice_end, best_s,
                                      int(rng.randint(3)))
                trial += 1
                if got is not None:
                    _arm_update(bd, n, A_XFER, _arm_reward(got[2], best_s, None))
                    if got[2] > best_s:
                        best_xy, best_r, best_s = got[0], got[1], got[2]
                else:
                    _arm_update(bd, n, A_XFER, 0.0)            # infeasible/refused counts as a loss
                continue
            el = _pool_pick(pool, n, rng) if arm != A_COLD else None
            if el is not None:
                # perturb an ELITE, not always the incumbent: non-monotone acceptance, so a run can
                # leave a deep local optimum without ever risking the best pack (the driver keeps it).
                xy, rr = el[0].copy(), el[1].copy()
                start_s = float(el[2])
                struct = False
                eq = False
                if arm == A_JIT:                               # jitter the whole layout
                    xy = np.clip(xy + rng.normal(0, 10.0 ** rng.uniform(-2.8, -1.5), size=(n, 2)),
                                 LO + EDGE, HI - EDGE)
                elif arm == A_HOLE:                            # relocate the dead-weight circles
                    xy = _hole_move(xy, rr, rng, k=int(rng.randint(1, max(2, n // 12))))
                elif arm == A_SHAKE:                           # FEASIBLE LP random walk (iteration 3)
                    xy, r = _lp_shake(xy, rr, rng,
                                      delta=10.0 ** rng.uniform(-2.3, -1.4),
                                      steps=int(rng.randint(1, 5)))
                elif arm == A_EQ:                              # OBJECTIVE-SPACE perturbation (iter 7)
                    xy, r = _equalize_move(xy, rr, rng,
                                           delta=10.0 ** rng.uniform(-2.3, -1.5),
                                           steps=int(rng.randint(1, 4)))
                    eq = True
                elif arm == A_SYM:                             # REPRESENTATION-SPACE projection (10)
                    # 2/3 of the draws snap onto the symmetry the layout is ALREADY closest to (a
                    # cheap, high-acceptance polish that finishes a nearly-symmetric basin); 1/3 pick
                    # a random element, which is the genuinely exploratory jump to a different
                    # symmetry class.
                    gsel = None
                    if rng.rand() < 0.667:
                        gsel = _nearest_sym(xy, rr)[0]
                    xy = _symmetry_move(xy, rr, rng, g=gsel)[0]
                    # r stays None on purpose: the projected CENTRES are the whole content of this
                    # move, and `_lp_radii` is EXACT for fixed centres, so re-solving the LP recovers
                    # every radius the averaging gave away instead of handing SLP a shrunken start.
                    struct = True
                elif arm == A_SYMC:                            # SYMMETRY-CONSTRAINED SLP (iter 11)
                    # project, then optimise WITHIN the symmetric subspace: the free SLP below still
                    # gets the last word, so the arm can only propose the symmetric optimum, never
                    # force the layout to stay there.
                    xy, r = _sym_constrained(xy, rr, rng,
                                             min(slice_end, time.process_time() + 1.2))
                    struct = True
                else:                                          # SPATIAL RECOMBINATION (iteration 6)
                    arm = A_REG
                    dxy, dr = _pick_donor(pool, n, rng)
                    xy, r = _region_move(xy, rr, rng, dxy, dr)
                    xy = np.clip(xy, LO + EDGE, HI - EDGE)
                    r = _shrink_feasible(xy, r)
                    struct = True
                # a perturbed incumbent is already near a local optimum: a small trust radius reaches
                # the new optimum in far fewer LPs than the cold-start delta0, so more basins/second.
                # A REGRAFTED patch is not: its seam needs room to settle, so it gets a wider region.
                d0 = 0.02 if struct else (0.015 if eq else 0.008)
            else:
                arm = A_COLD
                xy = _INITS[trial % len(_INITS)](n, rng)
                pen = rng.rand() < 0.5                         # half the cold starts get the penalty ramp
        trial += 1

        if pen:
            xy = _penalty_refine(xy, _safe_radii(xy), n, RHO_RAMP, maxiter)
        if r is None:
            r = _lp_radii(xy)
        xy, r = _slp_refine(xy, r, min(slice_end + 0.35, time.process_time() + 2.5), delta0=d0)

        pack = np.concatenate([xy, r[:, None]], axis=1)
        feas, s = evaluate(n, pack)
        if feas:
            _arm_update(bd, n, arm, _arm_reward(float(s), best_s, start_s))
            _pool_add(pool, n, xy, r, float(s))
            if float(s) > best_s:
                best_s, best_xy, best_r = float(s), xy.copy(), r.copy()
        else:
            _arm_update(bd, n, arm, 0.0)
        if best_xy is None:
            best_xy, best_r = xy.copy(), r.copy()


# =============================================================================== self-test
def _self_test():
    import sys
    rs = np.random.RandomState(0)
    ok = True

    # 1. analytic gradient vs central differences
    n = 9
    z = np.concatenate([rs.uniform(-0.4, 0.4, 2 * n), rs.uniform(0.02, 0.12, n)])
    f0, g = _fg(z, n, 500.0)
    num = np.zeros_like(z)
    for i in range(z.size):
        e = np.zeros_like(z); e[i] = 1e-6
        num[i] = (_fg(z + e, n, 500.0)[0] - _fg(z - e, n, 500.0)[0]) / 2e-6
    err = np.max(np.abs(num - g)) / max(1.0, np.max(np.abs(g)))
    print("grad rel-err %.3e" % err)
    ok &= err < 1e-5

    # 2. LP radii are feasible and beat the equal-radius rule
    for m in (1, 2, 13, 40):
        xy = rs.uniform(-0.45, 0.45, size=(m, 2))
        r = _lp_radii(xy)
        assert r.shape == (m,) and np.all(r > 0), "LP radii must be strictly positive"
        w = _wall(xy)
        assert np.min(w - r) >= -1e-9, "wall violation"
        if m >= 2:
            d = _dists(xy)
            assert np.min(d - (r[:, None] + r[None, :])) >= -1e-9, "pair violation"
        base = _safe_radii(xy).sum()
        print("n=%d  lp=%.6f  equal-rule=%.6f" % (m, r.sum(), base))
        ok &= r.sum() >= base - 1e-9

    # 2b. SLP must keep every iterate STRICTLY FEASIBLE (the conservative-linearisation claim) and must
    #     not lose ground against the LP-radii-only value it starts from.
    for m in (2, 7, 31, 60):
        xy = rs.uniform(-0.45, 0.45, size=(m, 2))
        r0 = _lp_radii(xy)
        xy2, r2 = _slp_refine(xy, r0, time.process_time() + 1.5)
        w = _wall(xy2)
        wslack = float(np.min(w - r2))
        pslack = float(np.min(_dists(xy2) - (r2[:, None] + r2[None, :]))) if m >= 2 else np.inf
        print("slp n=%d  %.6f -> %.6f  wall %.1e pair %.1e" % (m, r0.sum(), r2.sum(), wslack, pslack))
        ok &= wslack >= -1e-9 and pslack >= -1e-9 and np.all(r2 > 0)
        ok &= r2.sum() >= r0.sum() - 1e-12
        ok &= xy2.shape == (m, 2) and r2.shape == (m,)
    # 2c. _hole_move must return a valid layout for any k, including k >= n
    xy = rs.uniform(-0.45, 0.45, size=(5, 2))
    hm = _hole_move(xy, _lp_radii(xy), rs, k=9, probes=32)
    ok &= hm.shape == (5, 2) and np.all(np.abs(hm) <= 0.5)

    # 2d. the LP random walk must land on a STRICTLY FEASIBLE layout every time (its whole point)
    for m in (2, 9, 44):
        xy = rs.uniform(-0.45, 0.45, size=(m, 2))
        r0 = _lp_radii(xy)
        sx, sr = _lp_shake(xy.copy(), r0.copy(), rs, delta=0.02, steps=4)
        w = float(np.min(_wall(sx) - sr))
        pp = float(np.min(_dists(sx) - (sr[:, None] + sr[None, :]))) if m >= 2 else np.inf
        moved = float(np.max(np.abs(sx - xy)))
        print("shake n=%d  wall %.1e pair %.1e moved %.3f" % (m, w, pp, moved))
        ok &= w >= -1e-9 and pp >= -1e-9 and np.all(sr > 0) and sx.shape == (m, 2)

    # 2e. CROSS-n TRANSFER: shrinking and growing a layout must both give a strictly feasible n-pack,
    #     and the guarded pool read must return None (not raise) for an n with no committed pack.
    for m, k in ((30, 24), (18, 25), (9, 9)):
        xy = rs.uniform(-0.45, 0.45, size=(m, 2))
        r0 = _lp_radii(xy)
        tx, tr = _transfer_start(xy, r0, k, rs)
        assert tx.shape == (k, 2) and tr.shape == (k,), "transfer must return exactly k circles"
        tr = _shrink_feasible(np.clip(tx, LO + EDGE, HI - EDGE), tr)
        tx = np.clip(tx, LO + EDGE, HI - EDGE)
        w = float(np.min(_wall(tx) - tr))
        pp = float(np.min(_dists(tx) - (tr[:, None] + tr[None, :]))) if k >= 2 else np.inf
        tx2, tr2 = _slp_refine(tx, tr, time.process_time() + 1.0, delta0=0.02)
        w2 = float(np.min(_wall(tx2) - tr2))
        pp2 = float(np.min(_dists(tx2) - (tr2[:, None] + tr2[None, :]))) if k >= 2 else np.inf
        print("transfer %d->%d  wall %.1e pair %.1e  slp %.6f wall %.1e pair %.1e"
              % (m, k, w, pp, tr2.sum(), w2, pp2))
        ok &= w >= -1e-9 and pp >= -1e-9 and w2 >= -1e-9 and pp2 >= -1e-9
        ok &= np.all(tr2 > 0) and tx2.shape == (k, 2)
        ok &= tr2.sum() >= tr.sum() - 1e-12
    # 2f. every STRUCTURAL MODE of the transfer must return exactly the requested count and a feasible
    #     layout after shrink -- modes 1/2 are randomized, so this is checked over repeats.
    for m, k in ((30, 24), (18, 25), (21, 21), (9, 5)):
        for mode in (0, 1, 2):
            for _ in range(3):
                xy = rs.uniform(-0.45, 0.45, size=(m, 2))
                r0 = _lp_radii(xy)
                tx, tr = _transfer_start(xy, r0, k, rs, mode)
                assert tx.shape == (k, 2) and tr.shape == (k,), \
                    "transfer mode %d %d->%d gave %s" % (mode, m, k, tx.shape)
                tx = np.clip(tx, LO + EDGE, HI - EDGE)
                tr = _shrink_feasible(tx, tr)
                w = float(np.min(_wall(tx) - tr))
                pp = float(np.min(_dists(tx) - (tr[:, None] + tr[None, :]))) if k >= 2 else np.inf
                ok &= w >= -1e-9 and pp >= -1e-9 and np.all(tr > 0)
    print("transfer modes 0/1/2: exact count + feasible after shrink")

    # 2g. the ELITE ARCHIVE: guarded miss is cached; distinct basins accumulate up to ELITES; the same
    #     basin is replaced rather than duplicated; junk (>2% below best) is refused; best stays first.
    pool = {}
    ok &= _pool_get(pool, 100003) is None and 100003 in pool     # guarded miss, cached
    ok &= _pool_get(pool, 100003) is None
    ok &= _pool_pick(pool, 100003, rs) is None
    key = 100005
    _pool_init(pool, key)
    for j in range(6):                                           # 6 distinct signatures, ELITES kept
        xyj = rs.uniform(-0.45, 0.45, size=(5, 2))
        _pool_add(pool, key, xyj, _lp_radii(xyj) * 0.0 + (1.0 + 0.01 * j) / 20.0, 1.0 + 0.001 * j)
    ok &= len(pool[key]) == ELITES
    ok &= all(pool[key][i][2] >= pool[key][i + 1][2] for i in range(len(pool[key]) - 1))
    top = pool[key][0][2]
    dup = (pool[key][0][0], pool[key][0][1])
    _pool_add(pool, key, dup[0], dup[1], top + 0.5)              # same signature -> replace, not append
    ok &= len(pool[key]) == ELITES and pool[key][0][2] == top + 0.5
    junk = np.full(5, 0.001)
    _pool_add(pool, key, rs.uniform(-0.4, 0.4, size=(5, 2)), junk, 0.005)
    ok &= len(pool[key]) == ELITES and all(e[2] > 0.005 for e in pool[key])
    print("archive: %d elites, sorted, dedup-by-signature, junk refused" % len(pool[key]))

    # 2i. SPATIAL RECOMBINATION (iteration 6): grafting a region must return EXACTLY the host's circle
    #     count and a strictly non-overlapping layout BY CONSTRUCTION -- before _shrink_feasible, not
    #     because of it (the graft clips each donor radius to the clearance actually left at that
    #     point).  Checked with a same-size donor (crossover), a bigger and a smaller donor (cross-n
    #     recombination), and no donor at all (pure ruin & recreate), over repeats because the region,
    #     its size and the donor order are all randomized.
    worst_w, worst_p = np.inf, np.inf
    for hn, dn in ((24, 24), (24, 31), (24, 17), (24, None), (5, 7), (3, None)):
        for _ in range(4):
            hx = rs.uniform(-0.45, 0.45, size=(hn, 2))
            hr = _lp_radii(hx)
            if dn is None:
                dx = dr = None
            else:
                dx = rs.uniform(-0.45, 0.45, size=(dn, 2))
                dr = _lp_radii(dx)
            gx, gr = _region_move(hx, hr, rs, dx, dr)
            assert gx.shape == (hn, 2) and gr.shape == (hn,), \
                "region_move host %s donor %s gave %s" % (hn, dn, gx.shape)
            w = float(np.min(_wall(gx) - gr))
            pp = float(np.min(_dists(gx) - (gr[:, None] + gr[None, :]))) if hn >= 2 else np.inf
            worst_w, worst_p = min(worst_w, w), min(worst_p, pp)
            ok &= w >= -1e-9 and pp >= -1e-9 and np.all(gr > 0)
            gx2 = np.clip(gx, LO + EDGE, HI - EDGE)
            gr2 = _shrink_feasible(gx2, gr)
            sx, sr2 = _slp_refine(gx2, gr2, time.process_time() + 0.8, delta0=0.02)
            w2 = float(np.min(_wall(sx) - sr2))
            pp2 = float(np.min(_dists(sx) - (sr2[:, None] + sr2[None, :]))) if hn >= 2 else np.inf
            ok &= w2 >= -1e-9 and pp2 >= -1e-9 and np.all(sr2 > 0) and sx.shape == (hn, 2)
            ok &= sr2.sum() >= gr2.sum() - 1e-12        # SLP is monotone on the grafted layout too
    print("region_move: exact count, pre-shrink wall %.1e pair %.1e, SLP monotone" % (worst_w, worst_p))
    #     ...and the donor picker must never raise on an n the archive has never seen.
    dxy, dr = _pick_donor({}, 100007, rs)
    ok &= (dxy is None) or (dxy.ndim == 2 and dr.shape[0] == dxy.shape[0])

    # 2j. OBJECTIVE-SPACE perturbation (iteration 7): a reweighted LP step is still a RESTRICTION of
    #     the true constraints, so the move must come back strictly feasible for ANY weight vector,
    #     must keep exactly n circles, and must actually FLATTEN the radius distribution (that is the
    #     whole point -- a move that only jiggles sum r is the positional mix again).  Degenerate
    #     sizes (n = 1, 2) must not raise.  cR is also checked to genuinely change the LP: the
    #     small-favouring weight must not reproduce the plain sum-r step.
    worst_w, worst_p, flat = np.inf, np.inf, 0
    for en in (1, 2, 7, 19, 34):
        for t in range(3):
            ex = rs.uniform(-0.45, 0.45, size=(en, 2))
            er = _lp_radii(ex)
            # The GUARANTEE the reweighted LP gives is on its OWN objective: D = 0 with the current
            # r is always LP-feasible, so one step can never lower  sum w_i r_i .  (Flattening of the
            # radius spread is the *intent* and is reported below, but on a random cold layout the
            # smalls often have no clearance to grow into, so it is a diagnostic, not an invariant.)
            wq = (er.mean() / np.maximum(er, er.mean() * 1e-3)) ** 2.0
            one = _lp_step(ex, er, 0.02, cR=wq / wq.max())
            if one is not None:
                ok &= float((wq / wq.max() * one[1]).sum()) >= float((wq / wq.max() * er).sum()) - 1e-12
            qx, qr = _equalize_move(ex, er, rs, delta=0.02, steps=int(1 + t))
            assert qx.shape == (en, 2) and qr.shape == (en,), \
                "equalize_move n=%d gave %s" % (en, qx.shape)
            ok &= np.all(np.isfinite(qx)) and np.all(np.isfinite(qr)) and np.all(qr >= 0)
            w = float(np.min(_wall(qx) - qr))
            pp = float(np.min(_dists(qx) - (qr[:, None] + qr[None, :]))) if en >= 2 else np.inf
            worst_w, worst_p = min(worst_w, w), min(worst_p, pp)
            ok &= w >= -1e-9 and pp >= -1e-9
            if en >= 7 and er.std() > 0 and qr.std() / max(qr.mean(), 1e-12) < er.std() / er.mean():
                flat += 1
            # SLP must still be able to re-maximise from the perturbed layout, monotonically.
            qx2 = np.clip(qx, LO + EDGE, HI - EDGE)
            qr2 = _shrink_feasible(qx2, qr)
            ux, ur = _slp_refine(qx2, qr2, time.process_time() + 0.6, delta0=0.015)
            ok &= ur.sum() >= qr2.sum() - 1e-12 and ux.shape == (en, 2)
    #     ...and a weighted step must differ from the unweighted one on a spread-out layout.
    ex = rs.uniform(-0.45, 0.45, size=(23, 2))
    er = _lp_radii(ex)
    a = _lp_step(ex, er, 0.02)
    b = _lp_step(ex, er, 0.02, cR=(er.mean() / np.maximum(er, 1e-9)) ** 2.0)
    ok &= a is not None and b is not None and float(np.max(np.abs(a[1] - b[1]))) > 1e-9
    print("equalize_move: wall %.1e pair %.1e, weighted-obj monotone, flattened %d/9 cold, "
          "cR changes the LP" % (worst_w, worst_p, flat))

    # 2k. OPERATOR BANDIT (iteration 8).  Four properties, each one a way the controller could be
    # silently broken while still "running": the reward tiers, availability, learning, and pooling.
    ok &= _arm_reward(2.0, 1.0, None) == 1.0                    # new best for n
    ok &= _arm_reward(1.0, 2.0, 0.5) == 0.35                    # beat its own start, not the best
    ok &= _arm_reward(1.9995, 2.0, 2.5) == 0.15                 # competitive distinct basin
    ok &= _arm_reward(1.0, 2.0, 2.5) == 0.0                     # nothing
    ok &= _arm_reward(1.0, -np.inf, None) == 1.0                # first feasible layout for this n
    ok &= _arm_reward(np.nan, 1.0, None) == 0.0
    bd = _bandit()
    none_av = np.zeros(len(_ARMS), dtype=bool)
    ok &= _arm_pick(bd, 7, rs, none_av) == A_COLD               # no arm available -> cold, never raise
    only_cold = none_av.copy(); only_cold[A_COLD] = True
    ok &= all(_arm_pick(bd, 7, rs, only_cold) == A_COLD for _ in range(40))
    all_av = np.ones(len(_ARMS), dtype=bool)
    # (i) with NO data the prior must reproduce the iteration-7 mix's ordering: transfer is modal.
    bd0 = _bandit()
    cnt0 = np.zeros(len(_ARMS))
    for _ in range(3000):
        cnt0[_arm_pick(bd0, 5, rs, all_av)] += 1
    ok &= int(np.argmax(cnt0)) == A_XFER
    # (ii) it must LEARN: reward only equalize at n=41 and the pick share must concentrate there.
    bd1 = _bandit()
    for _ in range(300):
        a = _arm_pick(bd1, 41, rs, all_av)
        _arm_update(bd1, 41, a, 1.0 if a == A_EQ else 0.0)
    share = np.zeros(len(_ARMS))
    for _ in range(400):
        share[_arm_pick(bd1, 41, rs, all_av)] += 1
    ok &= share[A_EQ] / share.sum() > 0.60
    ok &= np.isfinite(bd1[41]).all() and (bd1[41] >= 0).all() and bd1[41].sum() < 200.0  # decay bounds
    # (iii) the HIERARCHY must transport that evidence to an n with no data of its own.
    fresh = np.zeros(len(_ARMS))
    for _ in range(1500):
        fresh[_arm_pick(bd1, 999, rs, all_av)] += 1
    ok &= fresh[A_EQ] / fresh.sum() > cnt0[A_EQ] / cnt0.sum()
    # (iv) an unavailable arm is NEVER chosen, however strong its posterior.
    no_eq = all_av.copy(); no_eq[A_EQ] = False
    ok &= all(_arm_pick(bd1, 41, rs, no_eq) != A_EQ for _ in range(200))
    print("bandit: prior modal=%s, learned share eq=%.2f (fresh n %.2f vs prior %.2f), "
          "availability + reward tiers hold"
          % (_ARMS[int(np.argmax(cnt0))], share[A_EQ] / share.sum(),
             fresh[A_EQ] / fresh.sum(), cnt0[A_EQ] / cnt0.sum()))

    # 2h. records read + cap detection must never raise and must default to "not capped" when unknown
    ok &= _capped({}, 100003) is False
    recs = _records()
    print("records loaded: %d entries" % len(recs))

    # 2l. TIME SCHEDULER (iteration 9).  The controller decides how many CPU seconds each n gets, so
    #     it can be silently broken in four ways while still running: it can ignore the measurements,
    #     starve an n to zero, blow up on a bad number, or misprice a digit.  One assertion each.
    #     (i) with NO measurements it must reproduce the iteration-4 sqrt(n) split exactly
    lv = [27, 51, 99]
    w0 = _sched_weights(lv, {})
    exp = np.array([np.sqrt(m) for m in lv]); exp /= exp.sum()
    ok &= np.allclose(w0, exp, atol=1e-12) and abs(w0.sum() - 1.0) < 1e-12
    #     (ii) it must actually FOLLOW the measurements: same n, higher rate -> strictly larger share
    w1 = _sched_weights(lv, {27: 1.0, 51: 0.0, 99: 0.0})
    ok &= w1[0] > w0[0] and w1[1] < w0[1] and abs(w1.sum() - 1.0) < 1e-12
    #     (iii) the exploration floor: a zero-yield n keeps a fixed, predictable fraction of a
    #           top-yield n's share (per unit of sqrt(n) prior) -- eps*mean/(top + eps*mean), > 0.
    #           (I first asserted eps/(1+eps) here and it FAILED at 0.0909 vs 0.1304: the floor is
    #            scaled by the MEAN rate over the live n, 2/3 here, not by the top rate.  The rule
    #            asserted below is the one the code actually implements.)
    w2 = _sched_weights(lv, {27: 1.0, 51: 1.0, 99: 0.0})
    ratio = (w2[2] / np.sqrt(99.0)) / (w2[0] / np.sqrt(27.0))
    _fl = _SCHED_EPS * (2.0 / 3.0)
    ok &= w2.min() > 0.0 and abs(ratio - _fl / (1.0 + _fl)) < 1e-9
    #     (iv) a non-finite rate must degrade to the prior, never to NaN shares
    w3 = _sched_weights(lv, {27: np.nan, 51: 0.5, 99: 0.5})
    ok &= np.all(np.isfinite(w3)) and np.allclose(w3, exp, atol=1e-12)
    ok &= np.all(np.isfinite(_sched_weights([27], {27: 0.0}))) and len(_sched_weights([27], {})) == 1
    print("sched shares  none=%s  hot=%s  floor-ratio=%.4f" % (np.round(w0, 3), np.round(w1, 3), ratio))
    #     (v) the digit rule the scheduler prices yield with must BE the scorer's rule
    rr = _records()
    if rr:
        m0 = sorted(rr)[0]
        ok &= abs(_digits_at(m0, rr[m0]) - 7.0) < 1e-12                  # at the record -> capped
        ok &= abs(_digits_at(m0, rr[m0] * 0.999) - 3.0) < 0.02           # 1e-3 relgap -> 3 digits
        ok &= _digits_at(m0, None) == 0.0 and _digits_at(10 ** 6, 1.0) == 0.0
        ok &= _digits_at(m0, rr[m0] * 2.0) == 7.0                        # above the record is capped
    #     (vi) _cur_sum is GUARDED (an n with no pack) and agrees with the pack it reads
    ok &= _cur_sum({}, 100003) is None
    if rr:
        m1 = sorted(rr)[0]
        gp = _read_pack_full(m1)
        if gp is not None:
            ok &= abs(_cur_sum({}, m1) - float(gp[1].sum())) < 1e-12
    print("scheduler checks ok=%s" % ok)

    # 2m. SYMMETRY PROJECTION (iteration 10).  A projection operator can be broken in ways that a
    #     "did it run?" check never sees: it can land NEAR the symmetric subspace instead of ON it,
    #     it can return an INFEASIBLE layout (averaging two circles' centres pushes them together),
    #     it can pair a big circle with a speck, or it can silently do nothing.  One assertion each.
    rs2 = np.random.RandomState(7)
    sxy = np.clip(rs2.uniform(LO + 0.06, HI - 0.06, size=(24, 2)), LO + EDGE, HI - EDGE)
    sr = _shrink_feasible(sxy, _safe_radii(sxy))
    for _gi, _g in enumerate(_SYM_G):
        Y, RR = _symmetry_move(sxy, sr, rs2, g=_g, alpha=1.0)
        #  (i) EXACT invariance, not approximate -- the whole point of the assignment step.  The
        #      four mirrors give a perfectly involutive matching; the 180-degree rotation may leave
        #      unpaired indices, so it is only required not to get WORSE.
        d_before, d_after = _sym_defect(sxy, sr, _g), _sym_defect(Y, RR, _g)
        ok &= (d_after < 1e-24) if _gi < 4 else (d_after <= d_before + 1e-12)
        #  (ii) strictly feasible BY CONSTRUCTION -- no repair pass runs after this operator
        dd = _dists(Y)
        iu, ju = np.triu_indices(24, 1)
        ok &= RR.min() > 0.0 and (_wall(Y) - RR).min() >= -1e-12
        ok &= (dd[iu, ju] - RR[iu] - RR[ju]).min() >= -1e-12
        ok &= Y.shape == sxy.shape and np.all(np.isfinite(Y)) and np.all(np.isfinite(RR))
    #  (iii) alpha interpolates: a partial pull must MOVE the layout but stop short of the subspace
    Yp, _RP = _symmetry_move(sxy, sr, rs2, g=_SYM_G[0], alpha=0.5)
    dp = _sym_defect(Yp, _RP, _SYM_G[0])
    ok &= 0.0 < dp < _sym_defect(sxy, sr, _SYM_G[0])
    #  (iv) an ALREADY-symmetric layout is a fixed point (idempotence): projecting twice changes
    #       nothing, so the operator cannot drift a converged symmetric optimum away.
    Y1 = _symmetry_move(sxy, sr, rs2, g=_SYM_G[1], alpha=1.0)[0]
    Y2 = _symmetry_move(Y1, _shrink_feasible(Y1, _safe_radii(Y1)), rs2, g=_SYM_G[1], alpha=1.0)[0]
    ok &= np.abs(Y2 - Y1).max() < 1e-9
    #  (v) the radius term in the cost matrix does its job: with two well-separated size classes the
    #      matching must never pair a large circle with a small one.
    big = np.array([[-0.3, -0.3], [0.3, -0.3], [-0.3, 0.3], [0.3, 0.3]])
    sml = np.array([[-0.05, -0.05], [0.05, -0.05], [-0.05, 0.05], [0.05, 0.05]])
    txy = np.vstack([big, sml])
    tr = np.concatenate([np.full(4, 0.15), np.full(4, 0.02)])
    Yt, RT = _symmetry_move(txy, tr, rs2, g=_SYM_G[0], alpha=1.0)
    ok &= RT[:4].min() > 0.1 and RT[4:].max() < 0.05
    #  (vi) _nearest_sym returns an element of the group and its own defect, and the degenerate
    #       n < 2 case is a no-op rather than a crash.
    gbest, dbest = _nearest_sym(sxy, sr)
    ok &= any(gbest is _g for _g in _SYM_G)
    ok &= abs(dbest - min(_sym_defect(sxy, sr, _g) for _g in _SYM_G)) < 1e-15
    Y0, R0 = _symmetry_move(sxy[:1], sr[:1], rs2)
    ok &= Y0.shape == (1, 2) and R0.shape == (1,)
    #  (vii) the arm is REACHABLE: with an elite available, symmetry must be picked by the bandit,
    #        and its prior share must sit between the small perturbation arms and transfer.
    _bd = _bandit()
    _av = np.ones(len(_ARMS), dtype=bool)
    _prng = np.random.RandomState(3)                # ONE stream: a fresh seed per draw would make
    picks = [_arm_pick(_bd, 41, _prng, _av) for _ in range(400)]   # all 400 draws identical
    ok &= picks.count(A_SYM) > 10
    #        ...and unavailable (no elite yet) it must NEVER be picked.
    _av2 = np.array([False] * 5 + [False, True, False, False])
    ok &= all(_arm_pick(_bd, 41, _prng, _av2) == A_COLD for _ in range(40))
    ok &= len(_ARM_MIX) == len(_ARMS) and abs(_ARM_MIX.sum() - 1.0) < 1e-12
    ok &= picks.count(A_SYMC) > 10                  # ...and so must the constrained arm (iter 11)
    print("symmetry: defect->%.1e  sym-arm share %.3f  symslp share %.3f  ok=%s"
          % (_sym_defect(_symmetry_move(sxy, sr, rs2, g=_SYM_G[2], alpha=1.0)[0], sr, _SYM_G[2]),
             picks.count(A_SYM) / 400.0, picks.count(A_SYMC) / 400.0, ok))

    # 2c. SYMMETRY-CONSTRAINED SLP (iteration 11).  Four ways a reduced-variable optimiser can be
    #     silently wrong while still returning numbers: the reduction can not actually reduce; it can
    #     admit directions that LEAVE the symmetric subspace; the reduced LP can lose the guarantee
    #     that every iterate is strictly feasible; and it can go BACKWARDS (SLP must be monotone).
    cxy = np.array([[-0.30, -0.30], [0.30, -0.30], [-0.30, 0.30], [0.30, 0.30],
                    [0.00, -0.12], [0.00, 0.12], [-0.02, 0.00], [0.34, 0.02],
                    [-0.33, 0.01]], dtype=float)
    cr = _lp_radii(cxy)
    for _gi, _g in enumerate(_SYM_G):
        _pi = _sym_perm(cxy, cr, _g)
        _red = _sym_reduction(cxy.shape[0], _pi, _g)
        #  (i) it really is a reduction, and M is exactly (3n x k)
        ok &= _red is not None and _red[0].shape == (3 * cxy.shape[0], _red[0].shape[1])
        ok &= _red[0].shape[1] < 3 * cxy.shape[0] and _red[1].size == _red[0].shape[1]
        #  (ii) EVERY direction in range(M) preserves the symmetry relation to machine precision:
        #       for a 2-cycle {i,j} the displacement must satisfy D_j = g D_i exactly, and for a
        #       fixed point g D_i = D_i.  Tested on random reduced vectors, not just the optimum.
        _M = _red[0].toarray()
        _rs3 = np.random.RandomState(7 + _gi)
        for _ in range(6):
            _zr = _rs3.normal(size=_red[0].shape[1])
            _zf = _M.dot(_zr)
            _D = np.stack([_zf[:cxy.shape[0]], _zf[cxy.shape[0]:2 * cxy.shape[0]]], axis=1)
            _res = _D[_pi].dot(np.asarray(_g).T) - _D          # D_{pi(i)} mapped back vs D_i
            _inv = _pi[_pi] == np.arange(cxy.shape[0])
            ok &= float(np.max(np.abs(_res[_inv]))) < 1e-12 if _inv.any() else True
            _rf = _zf[2 * cxy.shape[0]:]
            ok &= float(np.max(np.abs((_rf[_pi] - _rf)[_inv]))) < 1e-12 if _inv.any() else True
        #  (iii) the constrained ascent is MONOTONE and lands strictly feasible with no repair pass
        _Y = _symmetry_move(cxy, cr, rs2, g=_g, alpha=1.0)[0]
        _rY = _lp_radii(_Y)
        _Z, _rZ = _sym_slp_refine(_Y, _rY, _g, time.process_time() + 0.6)
        ok &= float(_rZ.sum()) >= float(_rY.sum()) - 1e-12
        ok &= float(_rZ.min()) > 0.0
        _dd = _dists(_Z)
        ok &= float(np.min((_dd - (_rZ[:, None] + _rZ[None, :]))[np.triu_indices(_Z.shape[0], 1)])) > 0.0
        ok &= float(np.min(_wall(_Z) - _rZ)) > 0.0
        #  (iv) it STAYS on the subspace it started on: the defect must not grow (up to fp noise)
        ok &= _sym_defect(_Z, _rZ, _g) <= _sym_defect(_Y, _rY, _g) + 1e-9
    #  (v) the 180-degree rotation pins its ONE fixed point (the centre of the square) completely:
    #      that circle keeps a radius variable and NO displacement variable at all.  n<2 is a no-op.
    _r1 = _sym_reduction(1, np.array([0]), _SYM_G[4])
    ok &= _r1 is not None and _r1[0].shape == (3, 1) and not bool(_r1[1].any())
    #      ...while a mirror leaves it one direction to slide along, so 2 columns, one of them a
    #      displacement.  (A reduction that silently froze everything would still "work".)
    _r2 = _sym_reduction(1, np.array([0]), _SYM_G[0])
    ok &= _r2 is not None and _r2[0].shape == (3, 2) and int(_r2[1].sum()) == 1
    _o1, _o2 = _sym_slp_refine(cxy[:1], cr[:1], _SYM_G[0], time.process_time() + 0.1)
    ok &= _o1.shape == (1, 2) and _o2.shape == (1,)
    #  (vi) the whole arm, end to end, returns a strictly feasible layout
    _cz, _crz = _sym_constrained(cxy, cr, rs2, time.process_time() + 0.6)
    _dd = _dists(_cz)
    ok &= float(np.min((_dd - (_crz[:, None] + _crz[None, :]))[np.triu_indices(_cz.shape[0], 1)])) > -1e-12
    ok &= float(np.min(_wall(_cz) - _crz)) > -1e-12 and float(_crz.min()) > 0.0
    print("sym-constrained: %d -> %d vars, sum %.6f -> %.6f  ok=%s"
          % (3 * cxy.shape[0], _sym_reduction(cxy.shape[0], _sym_perm(cxy, cr, _SYM_G[0]),
                                              _SYM_G[0])[0].shape[1],
             cr.sum(), _crz.sum(), ok))
    assert ok, "symmetry-constrained SLP self-test failed"

    # 3. solve() called TWICE in one process: the second call must still deliver a feasible packing.
    global CPU_BUDGET
    keep = CPU_BUDGET
    CPU_BUDGET = 3.0
    try:
        results = []
        class _M:
            def __init__(s, b): s.budget, s.used = b, 0
            def left(s): return max(0, s.budget - s.used)
        mm = _M(500000)
        best = {}
        def ev(k, pack):
            a = np.asarray(pack, float)
            single = a.ndim == 2
            if single:
                a = a[None]
            mm.used += a.shape[0]
            x, y, r = a[..., 0], a[..., 1], a[..., 2]
            wall = np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min(1)
            dx = x[:, :, None] - x[:, None, :]; dy = y[:, :, None] - y[:, None, :]
            dd = np.sqrt(dx * dx + dy * dy) - (r[:, :, None] + r[:, None, :])
            for b in range(a.shape[0]):
                np.fill_diagonal(dd[b], np.inf)
            pair = dd.reshape(a.shape[0], -1).min(1)
            fe = (r.min(1) > 0) & (wall >= -1e-9) & (pair >= -1e-9)
            sr = r.sum(1)
            for b in range(a.shape[0]):
                if fe[b] and sr[b] > best.get(k, (-np.inf,))[0]:
                    best[k] = (float(sr[b]), a[b].copy())
            return (bool(fe[0]), float(sr[0])) if single else (fe, sr)
        for call in (1, 2):
            best.clear()
            _t0 = time.process_time()
            solve(ev, mm, np.random.RandomState(call), [12, 31])
            _spent = time.process_time() - _t0
            print("call %d CPU %.2f s (budget %.1f)" % (call, _spent, CPU_BUDGET))
            ok &= _spent < CPU_BUDGET + 1.5      # the round schedule must still honour the deadline
            got = sorted(best)
            print("call %d -> feasible packings for n=%s, sums=%s"
                  % (call, got, ["%.4f" % best[k][0] for k in got]))
            results.append(got)
            ok &= got == [12, 31]
        # 4. single-target call must also work (budget split with len(targets)==1)
        best.clear()
        solve(ev, mm, np.random.RandomState(9), [27])
        print("single-target call -> %s" % sorted(best))
        ok &= sorted(best) == [27]
    finally:
        CPU_BUDGET = keep

    print("SELF-TEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
