"""Circle-packing census solver -- maximize the sum of radii of n circles in the centered unit square.

TRANSFORMATION (iteration 3): the SEARCH is rebuilt around its real currency -- *repaired topology
moves per CPU-second* -- and the clock is handed to a bandit that buys DIGITS, not seconds.

  WHY.  Iteration 2 proved the residual is contact topology (its move library took SCORE 2.38 -> 2.74
  and beat the n=89 record) but its gain was heavy-tailed: 5 of 37 n did not move at all.  Two things
  caused that.  (i) Every candidate move paid a FULL n-circle SLP repair (0.25 s at n=99), so a 2.8 s
  per-n slice bought only ~11 moves -- far too few samples of a combinatorial neighbourhood.
  (ii) The clock was split flat (proportional to n) although the marginal digits per second differ by
  orders of magnitude across n.

  (a) HIERARCHICAL REPAIR.  A relocation is a LOCAL event.  `_slp` now takes a `free` mask: frozen
      circles keep their (x,y,r) and act as fixed obstacles, so the LP loses their columns (bounds
      pinned to 0) and every row that touches only frozen circles.  A move is therefore screened by a
      ~14-circle repair (~10-20x cheaper) and only promoted to a 40-circle and then a full-n repair
      when it survives.  Same neighbourhood, an order of magnitude more samples of it.

  (b) AN ACTUAL HOLE ORACLE.  Reinsertion used `40+3n` uniform random probes -- hopeless at finding
      the largest hole of a dense 99-disk packing.  Candidates are now the DELAUNAY structure of the
      centres (triangle centroids, circumcentres, edge midpoints -- the largest empty circle of a
      disk system is centred at an Apollonius vertex, which these bracket) plus wall projections and
      random probes, all scored vectorized, then refined by a shrinking-sigma local search.

  (c) A DIGIT-RATE BANDIT INSTEAD OF A FLAT SPLIT.  The score is mean_n clamp(-log10 relgap, 0, 7),
      so the true marginal value of a second spent on n is the DIGIT gain it produces.  bench/
      records.json is readable (MISSION.md says so explicitly), so the solver measures exactly that:
      each n is given a short slice, its observed digits/second feeds an EWMA, and the next slice goes
      to the argmax (epsilon-greedy, optimistic init so every n is tried).  Clock flows to whichever n
      is still moving instead of being spread evenly over n that have stopped.

  (d) THRESHOLD ACCEPTANCE.  The ILS kept only strict improvements, so a stuck n stayed stuck.  The
      walker now has its own `cur` state that may accept a slightly worse repair (threshold ~ 3e-5
      relative, halved on stagnation, reset to `best` after a barren stretch) while `best` -- the only
      thing ever submitted to evaluate() -- is still monotone.

Contract notes (MISSION.md):
  * solve() may be called MORE THAN ONCE per process -> every deadline is derived from
    time.process_time() AT ENTRY, never from a module constant; the split survives len(targets)==1.
  * warm-start reads (packs AND records.json) are GUARDED; every n has a working cold start.
  * randomness comes only from `rng`; every packing that should count goes through evaluate(), and
    evaluate()'s own feasibility verdict is what gates acceptance.
  * no per-n coordinate tables: the only n-dependent inputs are n itself, its record scalar, and our
    own committed pack.
"""
import json
import math
import os
import time

import numpy as np

LO, HI = -0.5, 0.5
PACKS = "bench/packs"
RECORDS = "bench/records.json"

CPU_FRACTION = 0.86      # of the driver's 120 s process-CPU backstop that solve() will spend
CPU_DEFAULT = 120.0
EPS = 1e-12


# --------------------------------------------------------------------------- geometry helpers
def _walls(xy):
    """(n,2) -> (n,) distance from each centre to the nearest wall (negative if outside)."""
    x, y = xy[:, 0], xy[:, 1]
    return np.minimum.reduce([x - LO, HI - x, y - LO, HI - y])


def _dmat(xy):
    """(n,2) -> (n,n) pairwise centre distances, +inf on the diagonal."""
    d = xy[:, None, :] - xy[None, :, :]
    m = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(m, np.inf)
    return m


def _feasible_radii(xy, r):
    """Shrink r until it is STRICTLY feasible at these centres, losing as little Sum r as possible.
    A uniform shrink by t/2 removes a pair violation of size t exactly, so this terminates fast."""
    w = _walls(xy)
    D = _dmat(xy)
    r = np.clip(np.asarray(r, dtype=float), 0.0, None)
    r = np.minimum(r, w)
    for _ in range(80):
        t = max(0.0, (r[:, None] + r[None, :] - D).max() / 2.0, (r - w).max())
        if t <= 0.0:
            break
        r = np.minimum(r - t, w)
    r = np.maximum(r - EPS, EPS)
    return np.minimum(r, np.maximum(w - EPS, EPS))


def _lp_radii(xy, linprog, csr_matrix):
    """EXACT optimal radii for FIXED centres (a genuine LP), on a provably equivalent pruned row set:
    a pair with d_ij >= ub_i + ub_j can never bind, where ub_i = min(wall_i, min_j d_ij)."""
    n = xy.shape[0]
    w = np.maximum(_walls(xy), 0.0)
    D = _dmat(xy)
    ub = np.maximum(np.minimum(w, D.min(axis=1)), 0.0)
    if n < 2 or not np.all(np.isfinite(ub)):
        return _feasible_radii(xy, w)
    iu, ju = np.triu_indices(n, 1)
    keep = D[iu, ju] < ub[iu] + ub[ju] - 1e-15
    ii, jj = iu[keep], ju[keep]
    if ii.size == 0:
        return _feasible_radii(xy, ub)
    m = ii.size
    rows = np.repeat(np.arange(m), 2)
    cols = np.empty(2 * m, dtype=int)
    cols[0::2] = ii
    cols[1::2] = jj
    try:
        A = csr_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n))
        res = linprog(-np.ones(n), A_ub=A, b_ub=D[ii, jj],
                      bounds=np.stack([np.zeros(n), ub], axis=1), method="highs")
    except Exception:
        return _feasible_radii(xy, ub)
    if res.x is None:
        return _feasible_radii(xy, ub)
    return _feasible_radii(xy, res.x)


# --------------------------------------------------------------------------- symmetry algebra
_SYM_M = {}
_SYM_COMP = None


def _sym_matrix(k):
    """The 2x2 orthogonal matrix of dihedral element k, matching `_sym`'s (swap, negate) order."""
    G = _SYM_M.get(k)
    if G is None:
        G = np.zeros((2, 2))
        for a in range(2):
            e = np.zeros((1, 2))
            e[0, a] = 1.0
            G[:, a] = _sym(e, k)[0]
        _SYM_M[k] = G
    return G


def _sym_comp():
    """The 8x8 composition table of the square's dihedral group: `_sym_comp()[a, b]` is the index of
    `M_a @ M_b`.  Derived from `_sym_matrix` itself, so it can never disagree with `_sym`."""
    global _SYM_COMP
    if _SYM_COMP is None:
        M = [_sym_matrix(k) for k in range(8)]
        T = np.zeros((8, 8), int)
        for a in range(8):
            for b in range(8):
                P = M[a] @ M[b]
                T[a, b] = int(np.argmin([np.abs(P - M[c]).max() for c in range(8)]))
        _SYM_COMP = T
    return _SYM_COMP


def _sym_closure(ks):
    """The subgroup of D4 generated by the element indices `ks` (always contains the identity)."""
    T = _sym_comp()
    H = set([0])
    H.update(int(k) for k in ks)
    while True:
        new = set(int(T[a, b]) for a in H for b in H)
        if new <= H:
            return tuple(sorted(H))
        H |= new


def _sym_rows(triples, x, y, r, n, ncol, csr_matrix):
    """Equality rows pinning the SLP step to the fixed-point subspace of a symmetry structure.

    `triples` is a list of `(i, j, k)` incidences meaning `p_j = g_k p_i` -- the general form that
    covers a single involution's pairing AND a whole subgroup's orbit structure.  Columns are
    (dx_0..dx_{n-1}, dy_.., dr_.., [t]).  For an incidence with `i != j`:
    `dp_j - G dp_i = G p_i - p_j` and `dr_j - dr_i = r_i - r_j` (the right-hand sides are ~0 when
    the incoming packing is already on the orbit, and self-correcting when float error has nudged it
    off).  For `i == j` -- the STABILIZER case, a circle `g_k` fixes: `(I - G) dp_i = (G - I) p_i`,
    of which the rows that are identically zero (the directions `g_k` does not constrain) are
    dropped rather than handed to the LP as 0 = 0.  The identity element constrains nothing and is
    skipped."""
    rr, cc, vv, rhs = [], [], [], []
    m = 0
    for i, j, k in triples:
        if k == 0:
            continue
        G = _sym_matrix(k)
        if i == j:
            for a in range(2):                      # ((I - G) dp_i)_a = ((G - I) p_i)_a
                coef = ((1.0 if a == 0 else 0.0) - G[a, 0],
                        (1.0 if a == 1 else 0.0) - G[a, 1])
                if abs(coef[0]) < 1e-12 and abs(coef[1]) < 1e-12:
                    continue
                rr += [m, m]
                cc += [i, n + i]
                vv += [coef[0], coef[1]]
                rhs.append(G[a, 0] * x[i] + G[a, 1] * y[i] - (x[i] if a == 0 else y[i]))
                m += 1
        else:
            for a in range(2):                      # dp_j,a - (G dp_i)_a = (G p_i)_a - p_j,a
                rr += [m, m, m]
                cc += [(j if a == 0 else n + j), i, n + i]
                vv += [1.0, -G[a, 0], -G[a, 1]]
                rhs.append(G[a, 0] * x[i] + G[a, 1] * y[i] - (x[j] if a == 0 else y[j]))
                m += 1
            rr += [m, m]                            # dr_j - dr_i = r_i - r_j
            cc += [2 * n + j, 2 * n + i]
            vv += [1.0, -1.0]
            rhs.append(r[i] - r[j])
            m += 1
    if m == 0:
        return None, None
    A = csr_matrix((np.asarray(vv, float), (np.asarray(rr), np.asarray(cc))), shape=(m, ncol))
    return A, np.asarray(rhs, float)


# --------------------------------------------------------------------------- the SLP core
def _slp(xy, r, linprog, csr_matrix, iters=140, delta0=0.02, t_end=None, free=None, w_eq=0.0,
         sym=None):
    """Sequential LP by convex restriction.  Monotone in Sum r, every iterate truly feasible.

    `free` (bool (n,)) selects which circles may MOVE and GROW.  Frozen circles keep (x,y,r) exactly,
    so their columns are pinned to zero and every row involving only frozen circles is already
    satisfied and dropped -- that is what makes a local repair cheap.

    `w_eq` > 0 switches on the RADIUS-EQUALIZATION HOMOTOPY: one extra variable t with rows
    `t <= r_i` for every i, and objective `-(Sum_free r_i) - w_eq * n * t`, i.e. maximize
    `Sum r + w_eq * n * min_i r_i`.  w_eq = 0 is exactly the old sum-of-radii problem (the column is
    then not created at all); w_eq -> infinity is the classical EQUAL-circle packing (max-min radius).
    Sweeping w_eq moves the iterate between those two structures, which is a move no relocation can
    make: it changes the radius CLASS structure and hence the contact topology globally.  Feasibility
    is preserved for every w_eq -- the linearization of the concave constraint is still a restriction,
    and `t <= r_i` is linear.

    `sym` = a list of (i, j, k) incidences switches on the SYMMETRY QUOTIENT: the LP step is
    confined to the fixed-point subspace of that symmetry by LINEAR EQUALITY rows on the step
    variables -- `dp_j = G dp_i`
    and `dr_j = dr_i` for a matched pair, `(I - G) dp_i = 0` for a circle on the axis.  The iterate
    then stays EXACTLY g-symmetric at every iteration instead of drifting off the orbit at the first
    step, and the effective dimension is halved -- the LP is solving for the best SYMMETRIC packing
    rather than merely starting from one.  dp = 0 always satisfies the rows (the incoming
    configuration is already on the orbit), so the trust region is never infeasible and the
    monotone-in-Sum-r guarantee is untouched."""
    n = xy.shape[0]
    x = np.ascontiguousarray(xy[:, 0], dtype=float)
    y = np.ascontiguousarray(xy[:, 1], dtype=float)
    r = np.maximum(np.asarray(r, dtype=float), EPS)
    if free is None:
        free = np.ones(n, bool)
    fa = np.nonzero(free)[0]
    nf = fa.size
    if nf == 0:
        return np.stack([x, y], axis=1), r
    fr = np.nonzero(~free)[0]
    pinned = np.concatenate([fr, n + fr, 2 * n + fr]) if fr.size else None
    iu, ju = np.triu_indices(n, 1)
    both = free[iu] | free[ju]
    iu, ju = iu[both], ju[both]                       # frozen-frozen pairs can never be violated
    afn = np.arange(nf)
    delta = float(delta0)
    eq = float(w_eq) > 0.0
    ncol = 3 * n + (1 if eq else 0)
    nrq = n if eq else 0                              # the `t <= r_i` rows
    prev = r.sum() + (float(w_eq) * n * r.min() if eq else 0.0)
    c = np.zeros(ncol)
    c[2 * n + fa] = -1.0
    if eq:
        c[3 * n] = -float(w_eq) * n
    for it in range(iters):
        if t_end is not None and (it & 7) == 0 and time.process_time() >= t_end:
            break
        dxp = x[iu] - x[ju]
        dyp = y[iu] - y[ju]
        dij = np.sqrt(dxp * dxp + dyp * dyp)
        keep = (dij - (r[iu] + r[ju])) < 5.0 * delta
        ii, jj = iu[keep], ju[keep]
        M = ii.size
        dij = dij[keep]
        dij = np.maximum(dij, 1e-15)
        ux = dxp[keep] / dij
        uy = dyp[keep] / dij
        ar = np.arange(M)
        rows = [ar, ar, ar, ar, ar, ar]
        cols = [ii, jj, n + ii, n + jj, 2 * n + ii, 2 * n + jj]
        vals = [-ux, ux, -uy, uy, np.ones(M), np.ones(M)]
        b = [dij - r[ii] - r[jj]]
        for k, (cvar, sgn, rhs) in enumerate(((fa, -1.0, x[fa] - LO - r[fa]),
                                              (fa, 1.0, HI - x[fa] - r[fa]),
                                              (n + fa, -1.0, y[fa] - LO - r[fa]),
                                              (n + fa, 1.0, HI - y[fa] - r[fa]))):
            base = M + k * nf
            rows += [base + afn, base + afn]
            cols += [cvar, 2 * n + fa]
            vals += [np.full(nf, sgn), np.ones(nf)]
            b.append(rhs)
        if eq:                                        # t - dr_i <= r_i   (t is column 3n)
            base = M + 4 * nf
            an = np.arange(n)
            rows += [base + an, base + an]
            cols += [np.full(n, 3 * n), 2 * n + an]
            vals += [np.ones(n), -np.ones(n)]
            b.append(r.copy())
        A_eq = b_eq = None
        if sym is not None:
            A_eq, b_eq = _sym_rows(sym, x, y, r, n, ncol, csr_matrix)
        try:
            A = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(M + 4 * nf + nrq, ncol))
            lb = np.concatenate([np.full(2 * n, -delta), np.maximum(-r + EPS, -delta)])
            ub = np.concatenate([np.full(2 * n, delta), np.full(n, delta)])
            if eq:
                lb = np.concatenate([lb, [0.0]])
                ub = np.concatenate([ub, [0.5]])
            if pinned is not None:
                lb[pinned] = 0.0
                ub[pinned] = 0.0
            res = linprog(c, A_ub=A, b_ub=np.concatenate(b), A_eq=A_eq, b_eq=b_eq,
                          bounds=np.stack([lb, ub], axis=1), method="highs")
        except Exception:
            break
        if res.x is None:
            delta *= 0.5
            if delta < 1e-10:
                break
            continue
        x = np.clip(x + res.x[:n], LO, HI)
        y = np.clip(y + res.x[n:2 * n], LO, HI)
        r = np.maximum(r + res.x[2 * n:3 * n], EPS)
        cur = r.sum() + (float(w_eq) * n * r.min() if eq else 0.0)
        if cur - prev < 1e-13 * max(1.0, cur):
            delta *= 0.4
            if delta < 1e-10:
                break
        prev = cur
    return np.stack([x, y], axis=1), r


# --------------------------------------------------------------------------- the hole oracle
def _gaps(C, xy, r):
    """(m,2) probe points -> (m,) radius of the largest circle centred there that fits."""
    d = np.sqrt(((C[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) - r[None, :]
    w = np.minimum.reduce([C[:, 0] - LO, HI - C[:, 0], C[:, 1] - LO, HI - C[:, 1]])
    return np.minimum(d.min(axis=1), w)


def _hole(xy, r, rng, ncand, delaunay):
    """Largest empty circle among the disks.  Its centre is an Apollonius vertex of the disk system;
    the Delaunay structure of the centres brackets those vertices, so we score centroids /
    circumcentres / edge midpoints / wall projections / random probes and then refine the winner."""
    n = xy.shape[0]
    cands = [rng.uniform(LO + 1e-4, HI - 1e-4, size=(ncand, 2))]
    if n >= 4 and delaunay is not None:
        try:
            tri = delaunay(xy).simplices
            a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
            cands.append((a + b + c) / 3.0)
            cands.append((a + b) / 2.0)
            cands.append((b + c) / 2.0)
            cands.append((a + c) / 2.0)
            # circumcentres (the classic largest-empty-circle candidates for equal disks)
            ax, ay = a[:, 0], a[:, 1]
            bx, by = b[:, 0] - ax, b[:, 1] - ay
            cx, cy = c[:, 0] - ax, c[:, 1] - ay
            dd = 2.0 * (bx * cy - by * cx)
            ok = np.abs(dd) > 1e-12
            if ok.any():
                b2 = bx[ok] ** 2 + by[ok] ** 2
                c2 = cx[ok] ** 2 + cy[ok] ** 2
                ux = (cy[ok] * b2 - by[ok] * c2) / dd[ok]
                uy = (bx[ok] * c2 - cx[ok] * b2) / dd[ok]
                cands.append(np.stack([ax[ok] + ux, ay[ok] + uy], axis=1))
        except Exception:
            pass
    # wall projections: a hole hugging a side is common in these packings
    for k, v in ((0, LO), (0, HI), (1, LO), (1, HI)):
        p = xy.copy()
        p[:, k] = v
        cands.append(p)
    C = np.clip(np.vstack(cands), LO + 1e-9, HI - 1e-9)
    g = _gaps(C, xy, r)
    j = int(np.argmax(g))
    p, gb = C[j].copy(), float(g[j])
    sig = 0.06
    for _ in range(5):                       # shrinking-sigma refinement of the winner
        S = np.clip(p + rng.normal(0.0, sig, size=(40, 2)), LO + 1e-9, HI - 1e-9)
        gs = _gaps(S, xy, r)
        k = int(np.argmax(gs))
        if gs[k] > gb:
            p, gb = S[k].copy(), float(gs[k])
        sig *= 0.4
    return p, max(gb, 1e-6)


# --------------------------------------------------------------------------- topology moves
def _move(xy, r, rng, kind, ncand, delaunay):
    """Apply one topology move; return (start for the SLP repair, indices of the disturbed circles)."""
    n = xy.shape[0]
    xy = xy.copy()
    r = r.copy()
    if kind == "jitter":
        s = 0.004 + 0.02 * rng.rand()
        xy = np.clip(xy + rng.normal(0.0, s, size=xy.shape), LO + 1e-4, HI - 1e-4)
        return xy, r * 0.9, np.arange(n)
    if kind == "shake":                       # loosen the neighbourhood of the smallest circle
        i = int(np.argmin(r))
        dd = np.sqrt(((xy - xy[i]) ** 2).sum(-1))
        near = np.argsort(dd)[:max(3, n // 8)]
        xy[near] = np.clip(xy[near] + rng.normal(0.0, 0.03, size=(near.size, 2)),
                           LO + 1e-4, HI - 1e-4)
        return xy, r * 0.9, near
    k = 1 if kind == "reloc1" else (2 if kind == "reloc2" else 3)
    idx = np.argsort(r)[:min(k, n)]
    if kind == "relocR" and n > k:            # random victims rather than the smallest
        idx = rng.choice(n, size=min(k, n), replace=False)
    mask = np.ones(n, bool)
    mask[idx] = False
    for t in idx:
        p, g = _hole(xy[mask], r[mask], rng, ncand, delaunay)
        xy[t] = p
        r[t] = max(0.9 * g, 1e-6)
        mask[t] = True
    return xy, r, np.asarray(idx)


def _equalize(xy, r, linprog, csr_matrix, rng, t_end, w=None, hard=25, soft=110):
    """RADIUS-CLASS MOVE.  Drag the packing toward the equal-circle structure (max-min radius) with
    the w_eq homotopy, then release it back to pure Sum r.

    Why this is the right move.  Maximizing Sum r under the area/density budget the square imposes
    (Sum r^2 <~ rho/pi) is maximized, by Cauchy-Schwarz, when all r_i are EQUAL -- so a record csqv
    packing is a near-equal-circle packing with a few rattlers grown, not a heap of distinct radii.
    Our own census confirms it: the two n that escaped to 4.4 and 7.0 digits carry ~0.66n distinct
    radii, while every n stuck in the 2.3-3.0 band carries ~0.95n.  Relocation cannot fix that -- it
    moves one circle.  Equalization pushes EVERY radius toward a common value at once, which slides
    the whole contact graph, and the release step then re-grows whatever the new topology allows."""
    n = xy.shape[0]
    if w is None and rng.rand() < 0.34:               # a descending homotopy PATH, not a single jump
        xy1, r1 = xy, r
        for wk in (4.0, 1.6, 0.6, 0.2):
            xy1, r1 = _slp(xy1, r1, linprog, csr_matrix, max(6, hard // 4), 0.012, t_end, None, wk)
        xy1, r1 = _slp(xy1, r1, linprog, csr_matrix, soft, 0.014, t_end)
        return xy1, r1, np.arange(n)
    if w is None:
        w = float(10.0 ** rng.uniform(-1.0, 0.8))
    xy1, r1 = _slp(xy, r, linprog, csr_matrix, hard, 0.012, t_end, None, w)
    xy1, r1 = _slp(xy1, r1, linprog, csr_matrix, soft, 0.014, t_end)
    return xy1, r1, np.arange(n)


def _sym(xy, k):
    """One element of the square's dihedral group (the objective and the domain are invariant under
    all 8, so a donor packing may be re-oriented for free before it is spliced into another n)."""
    x = xy[:, 0].copy()
    y = xy[:, 1].copy()
    if k & 4:
        x, y = y, x
    if k & 1:
        x = -x
    if k & 2:
        y = -y
    return np.column_stack([x, y])


def _transfer(n, donor_xy, donor_r, rng, ncand, delaunay):
    """CROSS-n STRUCTURAL TRANSFER -- the census as a POPULATION, not 37 independent searches.

    Adjacent n are not independent problems: an optimal csqv packing of m circles and one of
    m+-2 share almost all of their contact structure.  Our own census makes that exploitable, because
    it is now bimodal -- 6 of 37 n sit at 6.6-7.0 digits (the true basin) while their immediate
    neighbours are stuck at ~2.6 (relative gap ~2e-3, i.e. the WRONG basin).  n=91 sits between two
    7-digit packings; n=95 next to one; 85 next to one; 73 and 69 next to one; 33 next to one.

    So: take a donor packing of m circles, re-orient it by a random symmetry of the square, and turn
    it into an n-circle configuration -- delete the (m-n) smallest circles if m > n, or grow (n-m)
    new ones in the largest empty gaps if m < n -- then hand the result to the SLP for repair.  No
    relocation, equalization or jitter move can reach this: it imports a contact graph discovered
    somewhere else in the census."""
    m = donor_xy.shape[0]
    xy = _sym(donor_xy, int(rng.randint(8)))
    r = np.asarray(donor_r, dtype=float).copy()
    if m > n:
        k = m - n
        order = np.argsort(r)
        cand = order[:min(m, k + 2)]                 # drop the smallest, with a little randomness
        if cand.size > k:
            idx = rng.choice(cand, size=k, replace=False)
        else:
            idx = cand
        keep = np.ones(m, bool)
        keep[np.asarray(idx, dtype=int)] = False
        xy, r = xy[keep], r[keep]
    elif m < n:
        for _ in range(n - m):
            p, g = _hole(xy, r, rng, ncand, delaunay)
            xy = np.vstack([xy, p[None, :]])
            r = np.append(r, max(0.9 * g, 1e-6))
    return np.clip(xy, LO, HI), np.maximum(r, EPS)


def _splice(n, inc_xy, inc_r, donor_xy, donor_r, rng, ncand, delaunay, frac=None):
    """PARTIAL SPLICE CROSSOVER -- transplant a HALF-PLANE PATCH of a donor into the incumbent.

    `_transfer` (iteration 5) imports a donor WHOLE and then resizes it, so the incumbent's own
    structure is discarded entirely and the donor must have nearly the right count: it is a resize,
    not a crossover.  The census, though, is bimodal by PATCH as much as by n -- a stuck packing is
    typically right over most of the square and wrong in one region, and a 7-digit packing of a quite
    different size still carries a correct corner/edge motif.  So: cut the square with a random line,
    keep the donor's circles on one side and the incumbent's on the other, and let the SLP heal the
    seam.  Two consequences no other move has:
      * the donor no longer has to be close in n -- ANY census packing can donate a patch, which
        turns a 12-member good set into a donor pool for all 37 targets; and
      * the incumbent's own good regions survive, so a failed splice costs one region, not everything.

    The seam is resolved by a greedy largest-first admission (a circle is admitted unless it overlaps
    an admitted one by more than `1 - ovl`), then the count is forced to exactly n by dropping the
    smallest / growing in the largest empty gaps."""
    d_xy = _sym(np.asarray(donor_xy, dtype=float), int(rng.randint(8)))
    d_r = np.asarray(donor_r, dtype=float)
    i_xy = np.asarray(inc_xy, dtype=float)
    i_r = np.asarray(inc_r, dtype=float)
    th = float(rng.uniform(0.0, 2.0 * math.pi))
    u = np.array([math.cos(th), math.sin(th)])
    if frac is None:
        frac = float(rng.uniform(0.25, 0.65))
    pd = d_xy @ u
    cut = float(np.quantile(pd, frac))                 # cut so ~frac of the DONOR crosses over
    take_d = pd <= cut
    take_i = (i_xy @ u) > cut
    if take_d.sum() == 0 or take_i.sum() == 0:         # degenerate cut -> nothing to cross over
        return None
    cx = np.vstack([d_xy[take_d], i_xy[take_i]])
    cr = np.concatenate([d_r[take_d], i_r[take_i]])
    # greedy largest-first admission: keeps the big structure, discards the seam's losers
    ovl = 0.80
    order = np.argsort(-cr)
    ax = np.empty((cx.shape[0], 2)); ar = np.empty(cx.shape[0]); m = 0
    for t in order:
        p, rad = cx[t], cr[t]
        if m:
            dd = np.sqrt(((ax[:m] - p) ** 2).sum(-1))
            if np.any(dd < ovl * (ar[:m] + rad)):
                continue
        ax[m] = p; ar[m] = rad; m += 1
    xy, r = ax[:m], ar[:m]
    if m > n:
        keep = np.argsort(-r)[:n]
        xy, r = xy[keep], r[keep]
    elif m < n:
        for _ in range(n - m):
            p, g = _hole(xy, r, rng, ncand, delaunay)
            xy = np.vstack([xy, p[None, :]])
            r = np.append(r, max(0.9 * g, 1e-6))
    return np.clip(xy, LO, HI), np.maximum(r, EPS)


def _rebuild(xy, r, linprog, csr_matrix, rng, ncand, delaunay, mode=None, kfrac=None):
    """REGIONAL RUIN-AND-RECREATE -- delete a contiguous PATCH and build a brand-new one in the hole.

    Every move the solver had so far either perturbs O(1) circles (reloc / shake / jitter), slides all
    of them along one homotopy (equalize), or IMPORTS a region from another census packing
    (transfer / splice).  None of them can *invent* regional structure: if no member of the census
    carries the motif a stuck n needs, the splice has nothing to donate and the relocations cannot
    assemble it one circle at a time.  Yet the diagnosis of the stuck band is precisely regional --
    relgap pinned at ~1e-3, i.e. right over most of the square and wrong in one place.

    So: freeze everything outside a disc, throw away the k circles inside it, and RECREATE them from
    nothing, by one of two generators that are deliberately unlike the incumbent --
      * ``hex``    stamps a fresh lattice patch (square or hexagonal) at a RANDOM orientation, spacing
                   and offset.  A rotated sublattice is unreachable by any sequence of single-circle
                   relocations -- each intermediate step is far worse -- but is one move from here; and
      * ``greedy`` re-fills the hole largest-empty-circle first, which finds the locally densest
                   admissible arrangement rather than an inherited one.
    The frozen circles are untouched, so a failed rebuild costs one region -- and, exactly as with the
    splice, the incumbent's good areas survive.  Radii are then set by the EXACT LP at these centres,
    so the child is judged on its true value, not on a heuristic guess for the new circles.

    Returns (xy, r, seeds) with exactly n circles, or None if the region is degenerate."""
    n = xy.shape[0]
    if n < 8:
        return None
    xy = np.asarray(xy, dtype=float).copy()
    r = np.asarray(r, dtype=float).copy()
    if kfrac is None:
        kfrac = float(rng.uniform(0.10, 0.34))
    k = int(min(max(4, round(kfrac * n)), n - 4))
    if k < 4:
        return None
    # ---- ruin: the k circles nearest an anchor.  Half the time the anchor is the WORST place in the
    # packing (the smallest circle), half the time it is uniform -- exploit vs explore over regions.
    if rng.rand() < 0.5:
        c = xy[int(np.argmin(r))] + rng.normal(0.0, 0.03, size=2)
    else:
        c = rng.uniform(LO, HI, size=2)
    d = np.sqrt(((xy - c) ** 2).sum(-1))
    idx = np.argsort(d)[:k]
    keep = np.ones(n, bool)
    keep[idx] = False
    fxy, fr = xy[keep], r[keep]
    rad = float(d[idx].max()) + float(r[idx].max())
    rmed = float(np.median(r[idx]))
    if not np.isfinite(rad) or rad <= 0.0 or rmed <= 0.0:
        return None
    lo = np.maximum(c - rad, LO + 1e-9)
    hi = np.minimum(c + rad, HI - 1e-9)
    if np.any(hi - lo <= 1e-6):
        return None

    if mode is None:
        mode = "hex" if rng.rand() < 0.5 else "greedy"
    new = []
    if mode == "hex":
        s = 2.0 * rmed * float(rng.uniform(0.86, 1.16))          # lattice spacing ~ the local scale
        th = float(rng.uniform(0.0, math.pi))
        ca, sa = math.cos(th), math.sin(th)
        m = int(math.ceil(2.2 * rad / max(s, 1e-6))) + 2
        ii, jj = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
        ii = ii.ravel().astype(float); jj = jj.ravel().astype(float)
        if rng.rand() < 0.6:                                     # hexagonal, else square
            px = s * (ii + 0.5 * jj)
            py = s * (math.sqrt(3.0) / 2.0) * jj
        else:
            px, py = s * ii, s * jj
        off = rng.uniform(-0.5, 0.5, size=2) * s
        gx = c[0] + off[0] + ca * px - sa * py
        gy = c[1] + off[1] + sa * px + ca * py
        P = np.column_stack([gx, gy])
        inside = (P[:, 0] > LO) & (P[:, 0] < HI) & (P[:, 1] > LO) & (P[:, 1] < HI) \
                 & (((P - c) ** 2).sum(-1) <= rad * rad)
        P = P[inside]
        if P.shape[0]:
            g = _gaps(P, fxy, fr)
            ok = g > 0.18 * s
            P, g = P[ok], g[ok]
            order = np.argsort(-g)                               # roomiest lattice sites first
            for t in order:
                if len(new) >= k:
                    break
                p = P[t]
                if new:
                    dd = np.sqrt(((np.asarray(new) - p) ** 2).sum(-1))
                    if np.any(dd < 0.75 * s):
                        continue
                new.append(p.copy())
    # ---- greedy largest-empty-circle fill (also the top-up when the lattice under-delivers)
    nfill = k - len(new)
    if nfill > 0:
        base_xy = np.vstack([fxy] + ([np.asarray(new)] if new else []))
        base_r = np.concatenate([fr] + ([np.full(len(new), 0.5 * rmed)] if new else []))
        nc = max(120, int(ncand) // 2)
        for _ in range(nfill):
            C = np.vstack([rng.uniform(lo, hi, size=(nc, 2)),
                           np.clip(c + rng.normal(0.0, 0.5 * rad, size=(nc // 2, 2)),
                                   LO + 1e-9, HI - 1e-9)])
            g = _gaps(C, base_xy, base_r)
            j = int(np.argmax(g))
            p, gb = C[j].copy(), float(g[j])
            sig = 0.5 * rmed
            for _q in range(4):                                  # shrinking-sigma refinement
                S = np.clip(p + rng.normal(0.0, sig, size=(32, 2)), LO + 1e-9, HI - 1e-9)
                gs = _gaps(S, base_xy, base_r)
                q = int(np.argmax(gs))
                if gs[q] > gb:
                    p, gb = S[q].copy(), float(gs[q])
                sig *= 0.45
            new.append(p)
            base_xy = np.vstack([base_xy, p[None, :]])
            base_r = np.append(base_r, max(0.85 * gb, 1e-6))
    N = np.clip(np.asarray(new, dtype=float)[:k], LO, HI)
    if N.shape[0] != k:
        return None
    xy2 = np.vstack([fxy, N])
    # EXACT radii at the new centres -- the rebuilt region is judged on its true LP value.
    r2 = _lp_radii(xy2, linprog, csr_matrix)
    seeds = np.arange(n - k, n)
    return xy2, r2, seeds


# --------------------------------------------------------------------------- the CHEAP repair
def _overlap(p, rt, n):
    """Total squared overlap of the configuration `p` (2n flat) when circle i has radius rt[i],
    plus its exact gradient.  This is the objective of the INFEASIBLE-PATH repair: unlike `_slp`,
    which is monotone in Sum r and never leaves the feasible set, this lets circles pass THROUGH
    one another on the way to a new contact graph.  It needs no LP at all -- one call is a couple
    of O(n^2) numpy passes -- which is the entire point: it is ~50x cheaper than an SLP repair."""
    xy = p.reshape(n, 2)
    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, 1e9)
    ov = (rt[:, None] + rt[None, :]) - d
    np.fill_diagonal(ov, 0.0)
    ov = np.where(ov > 0.0, ov, 0.0)
    E = 0.5 * float(np.sum(ov * ov))
    w = 2.0 * ov / np.maximum(d, 1e-12)     # 0.5*sum_{i!=j} ov^2 = sum_{i<j} ov^2 -> factor 2
    gx = -np.sum(w * dx, axis=1)
    gy = -np.sum(w * dy, axis=1)
    for g, c in ((gx, xy[:, 0]), (gy, xy[:, 1])):
        oL = np.maximum(rt - (c - LO), 0.0)
        oR = np.maximum(rt - (HI - c), 0.0)
        E += float(np.sum(oL * oL) + np.sum(oR * oR))
        g -= 2.0 * oL
        g += 2.0 * oR
    return E, np.column_stack([gx, gy]).ravel()


def _relax(xy, rt, minimize, maxit=250):
    """Push the centres apart until the circles of radius `rt` (deliberately too big) fit -- or
    until the descent stalls.  Returns (centres, final overlap energy)."""
    n = xy.shape[0]
    try:
        res = minimize(_overlap, xy.ravel(), args=(np.asarray(rt, float), n), jac=True,
                       method="L-BFGS-B", bounds=[(LO, HI)] * (2 * n),
                       options=dict(maxiter=int(maxit), ftol=1e-18, gtol=1e-14))
    except Exception:
        return xy.copy(), np.inf
    q = np.clip(np.asarray(res.x, float).reshape(n, 2), LO, HI)
    if not np.all(np.isfinite(q)):
        return xy.copy(), np.inf
    return q, float(res.fun)


def _inflate_targets(r, rng, eps):
    """Three ways to ask for more than the packing currently has: scale everything, pull the radii
    toward EQUAL (the Cauchy-Schwarz shape of an optimum) while scaling, or spend the whole increase
    on the smallest circles (the ones a jammed local optimum is starving)."""
    m = int(rng.randint(3))
    if m == 0:
        return r * (1.0 + eps)
    if m == 1:
        lam = 0.25 + 0.5 * rng.rand()
        return ((1.0 - lam) * r + lam * float(r.mean())) * (1.0 + eps)
    return r + eps * (float(r.max()) - r) + 0.2 * eps * float(r.mean())


def _jam(xy, r, rng, linprog, csr_matrix, minimize, ncand, delaunay, t_end, hops=14):
    """MANY CHEAP BASIN-HOPS FOR THE PRICE OF ONE SLP REPAIR.

    Measured on this census, one move through the SLP polish chain costs ~0.3-1.0 s, so a stuck n
    sees on the order of TEN candidate topologies in a whole iteration.  This move instead runs a
    burst of `hops` perturb -> relax -> exact-LP-radii cycles, each ~6 ms because the repair is the
    penalty relaxation above and the only LP is the one that prices the result.  The perturbation
    alternates a topology kick (relocate the smallest circles into the roomiest holes) with a pure
    inflation, so the burst walks over contact graphs rather than jiggling inside one.

    Returns the best (xy, r) the burst saw -- strictly feasible -- for the caller's polish chain."""
    n = xy.shape[0]
    b_xy, b_r = xy.copy(), np.asarray(r, float).copy()
    b_v = float(b_r.sum())
    c_xy, c_r = b_xy.copy(), b_r.copy()
    for h in range(int(hops)):
        if t_end is not None and time.process_time() >= t_end:
            break
        eps = 10.0 ** rng.uniform(-2.8, -1.2)
        if rng.rand() < 0.5:                       # topology kick, then let the relaxation settle it
            k_xy, k_r, _sd = _move(c_xy, c_r, rng, ("reloc1", "reloc2", "relocR")[int(rng.randint(3))],
                                   ncand, delaunay)
        else:
            k_xy, k_r = c_xy, c_r
        rt = _inflate_targets(k_r, rng, eps)
        q, _E = _relax(k_xy, rt, minimize)
        if not np.all(np.isfinite(q)):
            continue
        nr = _lp_radii(q, linprog, csr_matrix)     # exact radii at the relaxed centres
        v = float(nr.sum())
        if not np.isfinite(v) or nr.min() <= 0.0:
            continue
        if v > b_v:
            b_xy, b_r, b_v = q.copy(), nr.copy(), v
            c_xy, c_r = q, nr
        elif v > b_v - 2e-3 * max(1.0, b_v):       # threshold acceptance INSIDE the burst
            c_xy, c_r = q, nr
        else:
            c_xy, c_r = b_xy.copy(), b_r.copy()
    return b_xy, _feasible_radii(b_xy, b_r)


_SYM_INVOL = (1, 2, 3, 4, 7)   # mirror-x, mirror-y, rot180, mirror-diagonal, mirror-antidiagonal
_SYM_GROUPS = ((0, 1, 2, 3),          # Klein: both axis mirrors + rot180
               (0, 3, 4, 7),          # Klein: both diagonal mirrors + rot180
               (0, 3, 5, 6),          # cyclic C4: the four rotations
               (0, 1, 2, 3, 4, 5, 6, 7))   # the full dihedral group D4


def _symmetrize(xy, r, rng, minimize, linprog, csr_matrix, k=None, quotient=False,
                t_end=None):
    """PROJECT THE PACKING ONTO A SYMMETRY ORBIT -- a representation change, not a perturbation.

    Every move the solver had until now walks configuration space one circle (or one patch) at a
    time, so the symmetric configurations that dominate the published csqv optima are reachable
    only by an improbable coincidence of many independent relocations.  This move instead *imposes*
    the symmetry: pick an involution g of the square (mirror-x, mirror-y, rot180, or either
    diagonal), pair each circle with the circle nearest to its image under g, and replace the pair
    by its g-symmetric average -- a circle exactly on the axis if it pairs with itself.  The result
    lies exactly in the fixed-point set of g, which is a measure-zero subspace no local move ever
    lands in, and it is a GLOBAL rearrangement: every circle moves at once, coherently.

    The projection overlaps circles, so it is repaired on the infeasible path (`_relax`) and priced
    by the exact radius LP -- the caller's polish chain then refines it freely (the symmetry is a
    seed for a new contact topology, not a constraint that has to survive).

    `quotient=True` takes the other road: it keeps the symmetry THROUGH the repair.  The projection
    is made feasible by radius shrinkage alone (which cannot break the symmetry, because the wall
    and pair distances are themselves g-invariant), and the SLP then runs on the QUOTIENT -- the
    step confined to the fixed-point subspace by equality rows -- so what comes out is the locally
    optimal packing *within* that symmetry class rather than whatever the free repair happened to
    drift into.  Both roads are kept: the free repair can leave the orbit and find something the
    symmetry only pointed at, the quotient road searches a half-dimensional space exactly."""
    n = xy.shape[0]
    if n < 2:
        return None
    if k is None:
        k = int(_SYM_INVOL[int(rng.randint(len(_SYM_INVOL)))])
    Q = _sym(xy, k)                                     # images of every centre under g
    dx = xy[None, :, 0] - Q[:, None, 0]
    dy = xy[None, :, 1] - Q[:, None, 1]
    D = dx * dx + dy * dy                               # D[i, j] = |p_j - g(p_i)|^2, symmetric
    iu, ju = np.triu_indices(n, 0)                      # j == i allowed: "pair i with itself"
    order = np.argsort(D[iu, ju], kind="stable")
    taken = np.zeros(n, bool)
    nxy = np.empty_like(xy)
    nr = np.asarray(r, float).copy()
    pairs = []
    for t in order:
        i = int(iu[t]); j = int(ju[t])
        if taken[i] or taken[j]:
            continue
        taken[i] = taken[j] = True
        pairs.append((i, j))
        if i == j:                                      # fixed point: project onto the axis of g
            a = 0.5 * (xy[i] + Q[i])
            nxy[i] = a
        else:
            a = 0.5 * (xy[i] + _sym(xy[j:j + 1], k)[0])
            nxy[i] = a
            nxy[j] = _sym(a.reshape(1, 2), k)[0]
            rm = 0.5 * (nr[i] + nr[j])
            nr[i] = nr[j] = rm
    nxy = np.clip(nxy, LO, HI)
    if quotient:
        # radius-only repair keeps the centres, hence the symmetry, EXACT
        r0 = _feasible_radii(nxy, nr)
        if r0.min() <= 0.0 or not np.all(np.isfinite(r0)):
            return None
        tri = [(i, j, k) for i, j in pairs]
        q, qr = _slp(nxy, r0, linprog, csr_matrix, 90, 0.02, t_end, None, 0.0, tri)
        if not np.all(np.isfinite(q)):
            q, qr = nxy, r0
        out = _lp_radii(q, linprog, csr_matrix)
        if not np.all(np.isfinite(out)) or out.min() <= 0.0:
            return None
        return q, _feasible_radii(q, out)
    q, _E = _relax(nxy, nr, minimize)                   # repair through infeasible space
    if not np.all(np.isfinite(q)):
        q = nxy
    out = _lp_radii(q, linprog, csr_matrix)             # exact radii at the symmetrized centres
    if not np.all(np.isfinite(out)) or out.min() <= 0.0:
        return None
    return q, _feasible_radii(q, out)


def _group_project(xy, r, H):
    """PROJECT ONTO A WHOLE SYMMETRY GROUP -- the orbit structure, not one mirror.

    Iterations 10-11 could impose a single involution: pair each circle with its mirror image and
    average.  That is the order-2 case of a much stronger structure.  A subgroup `H <= D4` of order
    4 or 8 partitions the circles into H-ORBITS, and imposing it quarters (Klein, C4) or eighths
    (D4) the dimension of the search instead of halving it -- and the published csqv optima are
    full of 4-fold and 8-fold configurations that no chain of single mirrors ever reaches, because
    a second mirror imposed on top of the first is destroyed by the repair of the first.

    Greedy nearest-image matching builds the orbits: for each still-unassigned circle (largest
    radius first) and each `g in H`, claim the free circle nearest to `g p_i` -- unless `p_i` is
    itself nearer, which is the "sits on the axis / at the centre" case and makes `g` part of the
    orbit's STABILIZER.  The stabilizer is closed into a genuine subgroup `K` and the assignment is
    then collapsed onto the COSETS of `K`, so `g` and `g s` (`s in K`) necessarily claim the same
    circle -- without that step the greedy matching is not a group action and the equality rows it
    implies are contradictory.

    The orbit is then replaced by an exact H-orbit: the representative is the group average
    `q = (1/|H|) sum_g g^-1 p_{m(g)}`, which is automatically K-fixed once the assignment is
    coset-consistent, and each member is re-placed at `g q`.  Radii are averaged over the orbit.

    Returns `(nxy, nr, triples)` with `triples` the `(i, j, k)` incidences `p_j = g_k p_i` ready for
    `_sym_rows`, or None if `H` is degenerate for this packing."""
    n = xy.shape[0]
    h = len(H)
    if n < h or h < 2:
        return None
    T = _sym_comp()
    Ms = dict((k, _sym_matrix(k)) for k in H)
    xy = np.asarray(xy, float)
    r = np.asarray(r, float)
    free = np.ones(n, bool)
    nxy = xy.copy()
    nr = r.copy()
    triples = []
    claimed = {}                                        # circle -> orbit id, to catch collisions
    oid = 0
    for i0 in np.argsort(-r, kind="stable"):
        i0 = int(i0)
        if not free[i0]:
            continue
        free[i0] = False
        member = {0: i0}
        stab = [0]
        for t in range(1, h):
            tgt = Ms[H[t]] @ xy[i0]
            d_self = float(((xy[i0] - tgt) ** 2).sum())
            cand = np.nonzero(free)[0]
            if cand.size:
                d = ((xy[cand] - tgt[None, :]) ** 2).sum(1)
                b = int(np.argmin(d))
                if d[b] < d_self:
                    j = int(cand[b])
                    free[j] = False
                    member[t] = j
                    continue
            member[t] = i0                              # g fixes this circle: stabilizer element
            stab.append(int(H[t]))
        # ---- collapse onto cosets of the closed stabilizer, or the rows would contradict
        K = _sym_closure(stab)
        cos = {}
        for t in range(h):
            key = min(int(T[H[t], s]) for s in K)
            cos.setdefault(key, []).append(t)
        for ts in cos.values():
            keep = i0 if any(member[t] == i0 for t in ts) else member[ts[0]]
            for t in ts:
                if member[t] != keep:
                    if member[t] != i0:
                        free[member[t]] = True          # release the duplicate for another orbit
                    member[t] = keep
        mem = sorted(set(member.values()))
        if any(claimed.get(j, oid) != oid for j in mem):
            return None                                 # two orbits fighting over a circle
        for j in mem:
            claimed[j] = oid
        oid += 1
        # ---- exact H-orbit of the group-averaged representative
        q = np.zeros(2)
        for t in range(h):
            q += Ms[H[t]].T @ xy[member[t]]             # M^-1 == M^T (the group is orthogonal)
        q /= float(h)
        rm = float(np.mean(r[mem]))
        for t in range(h):
            nxy[member[t]] = Ms[H[t]] @ q
            nr[member[t]] = rm
            if H[t] != 0:
                triples.append((i0, member[t], int(H[t])))
    if not triples:
        return None
    return nxy, nr, triples


def _group_symmetrize(xy, r, linprog, csr_matrix, H, t_end=None, iters=90, delta0=0.02):
    """The QUOTIENT solve for a whole subgroup H: project onto the orbit structure, repair by
    radius shrinkage alone (which moves no centre, and wall/pair distances are H-invariant, so the
    configuration stays exactly on the orbit), then run the SLP confined to the orbit's fixed-point
    subspace and price with the exact radius LP.  `dp = 0` still satisfies every equality row, so
    the trust region is never infeasible and the monotone-in-Sum-r guarantee is untouched."""
    pr = _group_project(xy, r, H)
    if pr is None:
        return None
    nxy, nr, triples = pr
    if not np.all(np.isfinite(nxy)) or nxy.min() < LO - 1e-9 or nxy.max() > HI + 1e-9:
        return None
    r0 = _feasible_radii(nxy, nr)
    if not np.all(np.isfinite(r0)) or r0.min() <= 0.0:
        return None
    q, qr = _slp(nxy, r0, linprog, csr_matrix, iters, delta0, t_end, None, 0.0, triples)
    if not np.all(np.isfinite(q)):
        q, qr = nxy, r0
    out = _lp_radii(q, linprog, csr_matrix)
    if not np.all(np.isfinite(out)) or out.min() <= 0.0:
        return None
    return q, _feasible_radii(q, out)


def _neighbourhood(xy, seeds, k):
    """The k circles closest to the disturbed seeds -- the free set of a local repair."""
    n = xy.shape[0]
    if k >= n:
        return np.ones(n, bool)
    d = np.sqrt(((xy[:, None, :] - xy[None, np.asarray(seeds), :]) ** 2).sum(-1)).min(axis=1)
    free = np.zeros(n, bool)
    free[np.argsort(d)[:k]] = True
    free[np.asarray(seeds)] = True
    return free


# --------------------------------------------------------------------------- initial configurations
def _grid_xy(n, rng, hexagonal=False, jitter=0.0):
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pts = []
    for i in range(rows):
        yy = LO + (i + 0.5) / rows
        off = 0.5 * (i % 2) if hexagonal else 0.0
        for j in range(cols):
            pts.append((LO + (j + 0.5 + off) / (cols + (1 if hexagonal else 0)), yy))
    pts = np.array(pts[:n], dtype=float)
    if pts.shape[0] < n:
        pts = np.vstack([pts, rng.uniform(LO + 0.05, HI - 0.05, size=(n - pts.shape[0], 2))])
    if jitter > 0.0:
        pts = pts + rng.normal(0.0, jitter / cols, size=pts.shape)
    return np.clip(pts, LO + 1e-3, HI - 1e-3)


def _read_pack(n):
    """Warm start from our own committed census -- GUARDED; returns (xy, r) or None."""
    p = os.path.join(PACKS, "csqv%d.pck" % n)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            raw = [ln for ln in fh.read().splitlines() if ln.strip()]
        rows = [tuple(float(t) for t in ln.split()) for ln in raw[2:]]
    except (OSError, ValueError):
        return None
    if len(rows) != n or any(len(t) != 3 for t in rows):
        return None
    a = np.array(rows, dtype=float)
    if not np.all(np.isfinite(a)) or a[:, 2].min() <= 0.0:
        return None
    return np.clip(a[:, :2], LO, HI), a[:, 2].copy()


def _read_records():
    """GUARDED read of the record table (MISSION.md explicitly allows it); {} if absent."""
    try:
        if not os.path.exists(RECORDS):
            return {}
        with open(RECORDS) as fh:
            raw = json.load(fh)
        if isinstance(raw, dict) and isinstance(raw.get("records"), dict):
            raw = raw["records"]              # the file wraps the table in metadata
        out = {}
        for k, v in raw.items():              # skip every non-numeric metadata key
            try:
                out[int(k)] = float(v)
            except (ValueError, TypeError):
                continue
        return out
    except (OSError, ValueError, TypeError):
        return {}


def _digits(s, rec):
    """The scorer's own per-n value: clamp(-log10(relgap), 0, 7)."""
    if rec is None or not (rec > 0.0) or not np.isfinite(s):
        return 0.0
    g = max(0.0, (rec - s) / rec)
    if g <= 1e-7:
        return 7.0
    return max(0.0, min(7.0, -math.log10(g)))


# --------------------------------------------------------------------------- the ELITE ARCHIVE
def _descriptor(r):
    """Permutation-invariant fingerprint of a packing's STRUCTURE: the sorted radius profile,
    normalized to sum 1.

    Two packings that realise the same arrangement of radius ROLES (same multiset of "big corner
    circle / medium interior / small filler") sit a tiny L1 distance apart here even if every circle
    is relabelled, the packing is mirrored, or the whole thing is rotated a quarter turn -- all of
    which are the same topology and must not count as diversity.  Two genuinely different contact
    topologies move a visible fraction of the radius mass between roles and separate.  O(n log n),
    no LP and no geometry, so it can be computed on every candidate the search touches."""
    a = np.sort(np.asarray(r, dtype=float))
    t = a.sum()
    return a / t if t > 0 else a


class _Archive(object):
    """A per-n POPULATION of structurally distinct elites -- the diversity axis the solver never had.

    Until now every n carried exactly one incumbent plus a monotone best, so every other basin the
    search visited was discarded: a `jam` burst walks 14 contact graphs and keeps its argmax, a
    `rebuild` invents a rotated sublattice and throws it away if it is momentarily behind, and each
    threshold-accepted drift overwrites the configuration it came from.  This keeps them, and makes
    them PARENTS.

    Insertion is threshold-free by CROWDING: append, and while over capacity find the closest pair in
    descriptor space and drop the weaker of the two.  So the archive auto-calibrates to whatever
    spread it is shown -- no magic distance constant that would be wrong at n=27 and at n=99 -- and
    it converges to `cap` mutually-distant, high-Sigma-r members.  The champion is never evicted, so
    the archive can never lose the incumbent."""
    __slots__ = ("cap", "xy", "r", "s", "d")

    def __init__(self, cap=6):
        self.cap = max(1, int(cap))
        self.xy = []
        self.r = []
        self.s = []
        self.d = []

    def __len__(self):
        return len(self.s)

    def add(self, xy, r, s):
        s = float(s)
        if not np.isfinite(s):
            return
        self.xy.append(np.asarray(xy, dtype=float).copy())
        self.r.append(np.asarray(r, dtype=float).copy())
        self.s.append(s)
        self.d.append(_descriptor(r))
        while len(self.s) > self.cap:
            self._evict()

    def _evict(self):
        m = len(self.s)
        champ = int(np.argmax(self.s))
        bi, bj, bd = 0, m - 1, np.inf
        for i in range(m):
            for j in range(i + 1, m):
                dd = float(np.abs(self.d[i] - self.d[j]).sum())
                if dd < bd:
                    bi, bj, bd = i, j, dd
        k = bi if self.s[bi] < self.s[bj] else bj      # weaker of the closest pair
        if k == champ:                                 # never evict the champion
            k = bj if k == bi else bi
        if k == champ:
            k = int(np.argmin(self.s))
            if k == champ:
                k = m - 1
        for L in (self.xy, self.r, self.s, self.d):
            L.pop(k)

    def sample(self, rng, skip=-1):
        """Pick a parent index, rank-weighted by Sigma-r (weight ~ rank^2): a good elite is favoured,
        but a DISTINCT weaker one is still drawn regularly -- which is the entire reason to keep it."""
        idx = [i for i in range(len(self.s)) if i != skip]
        if not idx:
            return None
        order = sorted(idx, key=lambda i: self.s[i])
        w = np.array([(k + 1.0) ** 2 for k in range(len(order))], dtype=float)
        return order[int(rng.choice(len(order), p=w / w.sum()))]

    def spread(self):
        """Mean pairwise descriptor distance -- diagnostics only."""
        m = len(self.s)
        if m < 2:
            return 0.0
        tot = 0.0
        cnt = 0
        for i in range(m):
            for j in range(i + 1, m):
                tot += float(np.abs(self.d[i] - self.d[j]).sum())
                cnt += 1
        return tot / cnt


# --------------------------------------------------------------------------- per-n walker state
class _State(object):
    __slots__ = ("n", "best_xy", "best_r", "best_s", "cur_xy", "cur_r", "cur_s",
                 "k", "stall", "thresh", "rate", "spent", "rec", "arch")

    def __init__(self, n, rec):
        self.n = n
        self.arch = _Archive(6)
        self.best_xy = self.best_r = None
        self.best_s = -np.inf
        self.cur_xy = self.cur_r = None
        self.cur_s = -np.inf
        self.k = 0
        self.stall = 0
        self.thresh = 3e-5
        self.rate = 1e9          # optimistic init -> every n is tried at least once
        self.spent = 0.0
        self.rec = rec


# --------------------------------------------------------------------------- the solver
def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    from scipy.optimize import linprog, minimize
    from scipy.sparse import csr_matrix
    try:
        from scipy.spatial import Delaunay
    except Exception:
        Delaunay = None

    # EVERY deadline is derived from the clock AT ENTRY (solve() may be called repeatedly per process).
    t0 = time.process_time()
    total_cpu = float(os.environ.get("CP_SOLVER_CPU", CPU_DEFAULT)) * CPU_FRACTION
    deadline = t0 + total_cpu

    tg = sorted(int(t) for t in targets)
    recs = _read_records()
    st = {n: _State(n, recs.get(n)) for n in tg}

    # ---- the census POPULATION: every committed pack near the targets is a possible donor.
    # All reads are guarded (_read_pack returns None for a missing/odd file), so an n with no pack
    # -- a fresh size, an offline re-run -- simply contributes nothing and costs nothing.
    pool = {}
    for m in range(max(2, min(tg) - 8), max(tg) + 9):
        pk = _read_pack(m)
        if pk is not None:
            pool[m] = (pk[0], pk[1], float(pk[1].sum()))

    def pick_donor(n, span=6):
        """Choose a donor m != n: close in size, and (via the record table) as good as possible.
        A 7-digit neighbour therefore outweighs a 2.6-digit one by ~7x."""
        cand = [m for m in pool if m != n and 0 < abs(m - n) <= span]
        if not cand:
            return None
        w = np.array([(0.35 + _digits(pool[m][2], recs.get(m))) / (1.0 + abs(m - n))
                      for m in cand], dtype=float)
        tot = w.sum()
        if not np.isfinite(tot) or tot <= 0.0:
            return cand[int(rng.randint(len(cand)))]
        return cand[int(rng.choice(len(cand), p=w / tot))]
    def pick_patch_donor(n):
        """A donor for the PARTIAL SPLICE.  A patch is size-agnostic, so the whole census is eligible
        and quality dominates: weight ~ (0.2 + digits)^2 with only a mild size penalty.  That turns
        the 12 capped packings into a donor pool for all 37 targets instead of only their neighbours."""
        cand = [m for m in pool if m != n]
        if not cand:
            return None
        w = np.array([(0.2 + _digits(pool[m][2], recs.get(m))) ** 2 / (1.0 + 0.06 * abs(m - n))
                      for m in cand], dtype=float)
        tot = w.sum()
        if not np.isfinite(tot) or tot <= 0.0:
            return cand[int(rng.randint(len(cand)))]
        return cand[int(rng.choice(len(cand), p=w / tot))]

    kinds = ("reloc1", "cross", "jam", "symm", "equal", "rebuild", "transfer", "reloc2", "equal",
             "jam", "splice", "rebuild", "relocR", "symm", "shake", "equal", "jam", "rebuild",
             "transfer", "splice", "reloc1", "jitter", "equal", "jam", "symm", "rebuild", "splice",
             "reloc1", "cross", "cross")

    def score(n, xy, r):
        """Submit a candidate through the METERED evaluate(); its verdict gates acceptance."""
        if meter.left() <= 0:
            return -np.inf
        feas, s = evaluate(n, np.column_stack([xy, r]))
        return float(s) if feas else -np.inf

    # ---------------- phase A: seed every n (warm submit is ~free; cold start only when needed)
    init_end = t0 + 0.30 * total_cpu
    for n in tg:
        if meter.left() <= 0 or time.process_time() >= deadline:
            break
        s0 = st[n]
        warm = _read_pack(n)
        if warm is not None:                  # never regress: re-submit the committed pack as is
            v = score(n, warm[0], warm[1])
            if v > s0.best_s:
                s0.best_s, s0.best_xy, s0.best_r = v, warm[0], warm[1]
            # ...then a short PURE-SLP tail on it.  Measured on this census, the committed packs are
            # each ~1e-6 short of their own basin's optimum -- irrelevant at a 1e-3 gap, but it is
            # exactly what separates 6.4 digits from the 7-digit cap on the n that are already
            # at the record.  _slp is monotone in Sum r and feasible at every iterate, so this can
            # never regress; it just costs a tenth of a second.
            # Iteration 10 shipped this tail at 0.05 + 0.0022n s with a single delta0 = 0.01 pass
            # and it did NOT reproduce the probe's measured +1.2e-6 on n=29/33.  Two corrections:
            # a real budget (the probe used seconds, not a tenth of one) and a second pass at a
            # tight trust region -- the last 1e-6 lives inside delta = 0.01, so the first pass'
            # steps are far too coarse to resolve it -- finished with the exact radius LP at the
            # polished centres.  Every step here is monotone in Sum r, so it still cannot regress.
            t_pol = min(time.process_time() + 0.18 + 0.008 * n, init_end, deadline)
            if meter.left() > 0 and time.process_time() < t_pol:
                pxy, pr = _slp(warm[0].copy(), warm[1].copy(), linprog, csr_matrix,
                               400, 0.01, t_pol)
                pxy, pr = _slp(pxy, pr, linprog, csr_matrix, 400, 1e-3, t_pol)
                pr = _feasible_radii(pxy, pr)
                pl = _lp_radii(pxy, linprog, csr_matrix)
                if pl.sum() > pr.sum():
                    pr = pl
                if pr.sum() > s0.best_s:
                    pv = score(n, pxy, pr)
                    if pv > s0.best_s:
                        s0.best_s, s0.best_xy, s0.best_r = pv, pxy, pr
        else:
            t_end = min(init_end, deadline)
            dm = pick_donor(n, span=10)        # a nearby census packing is the best cold start there is
            if dm is not None:
                dxy, dr, _dv = pool[dm]
                xy_t, r_t = _transfer(n, dxy, dr, rng, 200 + 6 * n, Delaunay)
                xy_t, r_t = _slp(xy_t, r_t, linprog, csr_matrix, 150, 0.03, t_end)
                r_t = _feasible_radii(xy_t, r_t)
                v = score(n, xy_t, r_t)
                s0.arch.add(xy_t, r_t, v)
                if v > s0.best_s:
                    s0.best_s, s0.best_xy, s0.best_r = v, xy_t, r_t
            seeds0 = [_grid_xy(n, rng, hexagonal=False), _grid_xy(n, rng, hexagonal=True),
                      _grid_xy(n, rng, hexagonal=False, jitter=0.3),
                      rng.uniform(LO + .02, HI - .02, (n, 2))]
            for pi, xy0 in enumerate(seeds0):
                if time.process_time() >= t_end or meter.left() <= 0:
                    break
                if pi % 2 == 0:               # half the pool is seeded through an EQUAL packing
                    xy1, r1 = _slp(xy0, np.full(n, 1e-6), linprog, csr_matrix, 120, 0.05, t_end,
                                   None, 4.0)
                    xy1, r1 = _slp(xy1, r1, linprog, csr_matrix, 160, 0.03, t_end)
                else:
                    xy1, r1 = _slp(xy0, np.full(n, 1e-6), linprog, csr_matrix, 200, 0.05, t_end)
                r1 = _feasible_radii(xy1, r1)
                v = score(n, xy1, r1)
                # a cold start throws away 4 whole distinct basins per n -- keep them as parents
                s0.arch.add(xy1, r1, v)
                if v > s0.best_s:
                    s0.best_s, s0.best_xy, s0.best_r = v, xy1, r1
            if s0.best_xy is None:            # last resort: a guaranteed-feasible shrink of a grid
                xy1 = _grid_xy(n, rng)
                r1 = _feasible_radii(xy1, np.minimum(_walls(xy1), _dmat(xy1).min(axis=1) / 2.0))
                v = score(n, xy1, r1)
                if v > s0.best_s:
                    s0.best_s, s0.best_xy, s0.best_r = v, xy1, r1
        if s0.best_xy is not None:
            s0.cur_xy, s0.cur_r, s0.cur_s = s0.best_xy.copy(), s0.best_r.copy(), s0.best_s
            pool[n] = (s0.best_xy, s0.best_r, s0.best_s)
            s0.arch.add(s0.best_xy, s0.best_r, s0.best_s)

    live = [n for n in tg if st[n].best_xy is not None]
    # An n already at the DIGIT CAP cannot gain another point of SCORE no matter how long we search
    # it, so it must not consume phase-B clock -- it stays a donor (pool[n]) and its committed pack is
    # already re-submitted above.  Only drop it if some n is still improvable, so a fully capped
    # census (or a single capped target) still gets searched rather than returning early.
    improvable = [n for n in live if _digits(st[n].best_s, st[n].rec) < 7.0 - 1e-9]
    if improvable:
        live = improvable
    if not live:
        return

    # ---------------- phase B: digit-rate bandit over short per-n slices
    while time.process_time() < deadline and meter.left() > 0:
        if rng.rand() < 0.20:                 # exploration
            n = live[int(rng.randint(len(live)))]
        else:
            n = max(live, key=lambda m: st[m].rate)
        s0 = st[n]
        slice_len = min(0.25 + 0.010 * n, max(0.02, deadline - time.process_time()))
        t_end = min(time.process_time() + slice_len, deadline)
        d_before = _digits(s0.best_s, s0.rec)
        t_slice0 = time.process_time()

        ncand = 200 + 6 * n
        k_small = min(n, max(8, n // 6))
        k_mid = min(n, max(20, n // 2))
        while time.process_time() < t_end and meter.left() > 0:
            kind = kinds[s0.k % len(kinds)]
            s0.k += 1
            base_xy, base_r = s0.cur_xy, s0.cur_r
            dm = pick_donor(n) if kind == "transfer" else None
            if kind == "splice":
                dm = pick_patch_donor(n)
            if kind in ("transfer", "splice") and dm is None:
                kind = "reloc1"                      # no donor in the census -> ordinary move
            if kind == "cross" and len(s0.arch) < 2:
                kind = "rebuild"                     # nothing to recombine with yet
            if kind == "equal":
                xy2, r2, seeds = _equalize(base_xy, base_r, linprog, csr_matrix, rng, t_end)
            elif kind in ("transfer", "splice", "rebuild", "jam", "cross", "symm"):
                # A transferred / rebuilt structure is FAR from the incumbent: repaired once it is
                # usually still behind, so give it its own short polish chain before judging it --
                # otherwise a better basin is thrown away for being momentarily worse.
                t_tr = min(time.process_time() + 0.9 + 0.022 * n, deadline)
                if kind == "symm":
                    # Hedge the THREE roads out of the projection: the free repair (which may leave
                    # the orbit), the involution QUOTIENT (which stays on it and optimizes there),
                    # and the SUBGROUP quotient (order 4 or 8 -- a quarter or an eighth of the
                    # dimension, and a structure no chain of single mirrors reaches).
                    kk = int(_SYM_INVOL[int(rng.randint(len(_SYM_INVOL)))])
                    if rng.rand() < 0.5:
                        HH = _SYM_GROUPS[int(rng.randint(len(_SYM_GROUPS)))]
                        sy = _group_symmetrize(base_xy, base_r, linprog, csr_matrix, HH,
                                               t_end=t_tr)
                        if sy is None:
                            sy = _symmetrize(base_xy, base_r, rng, minimize, linprog, csr_matrix,
                                             k=kk, quotient=True, t_end=t_tr)
                    else:
                        sy = _symmetrize(base_xy, base_r, rng, minimize, linprog, csr_matrix, k=kk,
                                         quotient=True, t_end=t_tr)
                    sf = None
                    if sy is None or rng.rand() < 0.5:
                        sf = _symmetrize(base_xy, base_r, rng, minimize, linprog, csr_matrix, k=kk)
                    if sy is None and sf is None:
                        continue
                    if sy is None or (sf is not None and sf[1].sum() > sy[1].sum()):
                        sy = sf
                    xy2, r2 = sy
                elif kind == "jam":
                    xy2, r2 = _jam(base_xy, base_r, rng, linprog, csr_matrix, minimize,
                                   ncand, Delaunay, t_tr)
                elif kind == "rebuild":
                    rb = _rebuild(base_xy, base_r, linprog, csr_matrix, rng, ncand, Delaunay)
                    if rb is None:
                        continue
                    xy2, r2 = rb[0], rb[1]
                elif kind == "cross":
                    # INTRA-n RECOMBINATION: splice two elites of the SAME n.  Every donor the
                    # solver had until now came from another size, so a patch always arrived at the
                    # wrong scale and the seam had to absorb it; two same-n elites are exactly
                    # commensurate, and _splice's random mirror/rotation of the donor means even
                    # crossing an elite with a symmetry image of itself is a real move.
                    ia = int(np.argmax(s0.arch.s)) if rng.rand() < 0.5 else s0.arch.sample(rng)
                    ib = s0.arch.sample(rng, skip=ia)
                    if ia is None or ib is None:
                        continue
                    sp = _splice(n, s0.arch.xy[ia], s0.arch.r[ia],
                                 s0.arch.xy[ib], s0.arch.r[ib], rng, ncand, Delaunay)
                    if sp is None:
                        continue
                    xy2, r2 = sp
                else:
                    dxy, dr, _dv = pool[dm]
                    if kind == "splice":
                        sp = _splice(n, base_xy, base_r, dxy, dr, rng, ncand, Delaunay)
                        if sp is None:
                            continue
                        xy2, r2 = sp
                    else:
                        xy2, r2 = _transfer(n, dxy, dr, rng, ncand, Delaunay)
                xy2, r2 = _slp(xy2, r2, linprog, csr_matrix, 130, 0.025, t_tr)
                r2 = _feasible_radii(xy2, r2)
                bv = r2.sum()
                for _p in range(3):
                    if time.process_time() >= t_tr or meter.left() <= 0:
                        break
                    a_xy, a_r, _sd = _equalize(xy2, r2, linprog, csr_matrix, rng, t_tr,
                                               w=float(10.0 ** rng.uniform(-0.7, 0.6)),
                                               hard=18, soft=90)
                    a_r = _feasible_radii(a_xy, a_r)
                    if a_r.sum() > bv:
                        xy2, r2, bv = a_xy, a_r, a_r.sum()
                seeds = np.arange(n)
            else:
                xy2, r2, seeds = _move(base_xy, base_r, rng, kind, ncand, Delaunay)
            if kind in ("equal", "transfer", "splice", "rebuild", "jam", "cross", "symm"):
                pass
            elif kind in ("jitter", "shake"):
                xy2, r2 = _slp(xy2, r2, linprog, csr_matrix, 140, 0.02, t_end)
            else:
                # hierarchical repair: cheap local screen -> medium -> full
                free = _neighbourhood(xy2, seeds, k_small)
                xy2, r2 = _slp(xy2, r2, linprog, csr_matrix, 60, 0.02, t_end, free)
                if r2.sum() > s0.cur_s - 1.5e-3 * max(1.0, s0.cur_s):
                    free = _neighbourhood(xy2, seeds, k_mid)
                    xy2, r2 = _slp(xy2, r2, linprog, csr_matrix, 80, 0.015, t_end, free)
                    if r2.sum() > s0.cur_s - 4e-4 * max(1.0, s0.cur_s):
                        xy2, r2 = _slp(xy2, r2, linprog, csr_matrix, 140, 0.012, t_end)
            r2 = _feasible_radii(xy2, r2)
            v = r2.sum()
            if (kind in ("transfer", "splice", "rebuild", "jam", "cross", "equal", "symm")
                    and v > s0.best_s - 4e-3 * max(1.0, s0.best_s)):
                s0.arch.add(xy2, r2, v)      # near-elite: crowding decides whether it is kept
            if v > s0.best_s:
                rl = _lp_radii(xy2, linprog, csr_matrix)     # exact radii at these centres
                if rl.sum() > v:
                    r2, v = rl, rl.sum()
                sc = score(n, xy2, r2)
                if sc > s0.best_s:
                    s0.best_s, s0.best_xy, s0.best_r = sc, xy2.copy(), r2.copy()
                    s0.stall = 0
                    pool[n] = (s0.best_xy, s0.best_r, sc)   # the population improves as we go
            if v > s0.cur_s - s0.thresh * max(1.0, s0.cur_s):     # threshold acceptance
                s0.cur_xy, s0.cur_r, s0.cur_s = xy2, r2, v
            else:
                s0.stall += 1
                if s0.stall >= 12:
                    # Barren stretch.  Restarting from `best` every time is what pinned 20 n at
                    # ~1e-3 for six iterations: the walker keeps re-entering the basin it just
                    # failed to escape.  Two thirds of the time, restart from a DIFFERENT elite.
                    j = None
                    if len(s0.arch) > 1:
                        floor = s0.best_s - 1.5e-3 * max(1.0, s0.best_s)
                        ok = [i for i in range(len(s0.arch)) if s0.arch.s[i] >= floor]
                        if len(ok) > 1:
                            sub = _Archive(len(ok))
                            for i in ok:
                                sub.add(s0.arch.xy[i], s0.arch.r[i], s0.arch.s[i])
                            j = sub.sample(rng)
                            if j is not None:
                                s0.cur_xy, s0.cur_r = sub.xy[j].copy(), sub.r[j].copy()
                                s0.cur_s = sub.s[j]
                    if j is not None and rng.rand() < 0.66:
                        pass                      # restarted from a distinct, comparably good elite
                    else:
                        s0.cur_xy, s0.cur_r = s0.best_xy.copy(), s0.best_r.copy()
                        s0.cur_s = s0.best_s
                    s0.stall = 0
                    s0.thresh = max(3e-6, s0.thresh * 0.5)

        dt = max(1e-3, time.process_time() - t_slice0)
        s0.spent += dt
        gained = _digits(s0.best_s, s0.rec) - d_before
        obs = gained / dt
        s0.rate = obs if s0.rate > 1e8 else 0.45 * s0.rate + 0.55 * obs
        if gained <= 0.0:
            s0.rate = min(s0.rate, 0.6 * max(s0.rate, 0.0)) - 1e-9 * s0.spent


# --------------------------------------------------------------------------- self test
def _self_test():
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    class _Meter(object):
        def __init__(self, b):
            self.budget = b
            self.used = 0

        def left(self):
            return self.budget - self.used

        def tick(self, k=1):
            g = max(0, min(k, self.left()))
            self.used += g
            return g

    best = {}

    def make_eval(meter):
        def evaluate(n, packing):
            a = np.asarray(packing, dtype=float)
            single = (a.ndim == 2)
            if single:
                a = a[None]
            B = a.shape[0]
            g = meter.tick(B)
            feas = np.zeros(B, bool)
            sr = np.full(B, -np.inf)
            for i in range(min(g, B)):
                x, y, r = a[i, :, 0], a[i, :, 1], a[i, :, 2]
                wall = np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min()
                d = np.sqrt(((a[i, :, None, :2] - a[i, None, :, :2]) ** 2).sum(-1))
                sl = d - (r[:, None] + r[None, :])
                np.fill_diagonal(sl, np.inf)
                ok = bool(a[i].shape[0] == n and r.min() > 0 and wall >= -1e-9 and sl.min() >= -1e-9)
                feas[i] = ok
                sr[i] = r.sum()
                if ok and sr[i] > best.get(n, (-np.inf,))[0]:
                    best[n] = (float(sr[i]), a[i].copy())
            if single:
                return bool(feas[0]), float(sr[0])
            return feas, sr
        return evaluate

    # --- unit: a FROZEN circle must not move, and a local repair must stay strictly feasible
    rng0 = np.random.RandomState(7)
    xy0 = _grid_xy(20, rng0, jitter=0.4)
    r0 = _feasible_radii(xy0, np.minimum(_walls(xy0), _dmat(xy0).min(axis=1) / 2.0))
    fmask = np.zeros(20, bool)
    fmask[:6] = True
    xy1, r1 = _slp(xy0, r0, linprog, csr_matrix, 25, 0.02, None, fmask)
    assert np.allclose(xy1[~fmask], xy0[~fmask]) and np.allclose(r1[~fmask], r0[~fmask]), "frozen moved"
    rr = _feasible_radii(xy1, r1)
    assert rr.sum() >= r0.sum() - 1e-9, "local SLP lost sum_r"
    assert (_dmat(xy1) - (rr[:, None] + rr[None, :])).min() >= -1e-9, "local SLP infeasible"

    # --- unit: the equalization homotopy must RAISE the min radius and stay strictly feasible,
    #     and w_eq = 0 must reproduce the old sum-of-radii path exactly (the column is not created)
    xy2, r2 = _slp(xy0, r0, linprog, csr_matrix, 30, 0.02, None, None, 0.0)
    xy3, r3 = _slp(xy0, r0, linprog, csr_matrix, 30, 0.02, None, None)
    assert np.array_equal(xy2, xy3) and np.array_equal(r2, r3), "w_eq=0 changed the sum-r path"
    xye, re_ = _slp(xy0, r0, linprog, csr_matrix, 30, 0.02, None, None, 5.0)
    re_ = _feasible_radii(xye, re_)
    assert (_dmat(xye) - (re_[:, None] + re_[None, :])).min() >= -1e-9, "equalized packing overlaps"
    assert _walls(xye).min() >= re_.max() - 1e-9 or (_walls(xye) - re_).min() >= -1e-9, "wall escape"
    assert re_.min() > r0.min() + 1e-6, "equalization did not raise the min radius"
    assert re_.max() / re_.min() < r2.max() / max(r2.min(), 1e-15), "equalization not more uniform"

    # --- unit: the hole oracle must find a genuine empty disk and beat pure random probing
    from scipy.spatial import Delaunay
    p, g = _hole(xy1, rr, rng0, 200, Delaunay)
    assert g > 0 and float(_gaps(p[None, :], xy1, rr)[0]) >= g - 1e-12, "hole oracle inconsistent"

    # --- unit: digits matches the scorer's formula
    assert abs(_digits(0.999, 1.0) - 3.0) < 1e-9 and _digits(1.0, 1.0) == 7.0
    _rc = _read_records()
    assert len(_rc) >= 37 and 2.6 < _rc[27] < 2.7, "record table not parsed (metadata-wrapped)"

    # --- THE CHEAP INFEASIBLE-PATH REPAIR -------------------------------------------------------
    # (a) the analytic gradient of the overlap energy must match finite differences -- a wrong
    #     gradient would still "converge", just to nonsense, and nothing downstream would notice;
    # (b) the relaxation must strictly REDUCE the overlap of a deliberately inflated packing;
    # (c) a jam burst must return exactly n strictly-feasible circles and must never LOSE Sum r
    #     (it returns the best of the burst, and the incumbent is in the burst);
    # (d) it must work for an n with no committed pack.
    from scipy.optimize import minimize as _mini
    _rt = r0 * 1.06
    _p0 = xy0.ravel().copy() + rng0.normal(0.0, 1e-3, 2 * xy0.shape[0])
    _E0, _G = _overlap(_p0, _rt, xy0.shape[0])
    assert _E0 > 0.0, "inflated packing had no overlap to relax"
    for _t in (0, 5, 17, 23):
        _h = 1e-7
        _pp = _p0.copy(); _pp[_t] += _h
        _pm = _p0.copy(); _pm[_t] -= _h
        _fd = (_overlap(_pp, _rt, xy0.shape[0])[0] - _overlap(_pm, _rt, xy0.shape[0])[0]) / (2 * _h)
        assert abs(_fd - _G[_t]) <= 1e-4 * max(1.0, abs(_fd)), "overlap gradient wrong"
    _q, _E1 = _relax(xy0, _rt, _mini)
    assert _E1 < _E0 - 1e-12 and _q.shape == xy0.shape, "relaxation did not reduce overlap"
    assert _q.min() >= LO - 1e-12 and _q.max() <= HI + 1e-12, "relaxation left the square"
    for _nn, (_jx, _jr) in ((xy0.shape[0], (xy0, r0)),):
        _bx, _br = _jam(_jx, _jr, rng0, linprog, csr_matrix, _mini, 120, None, None, hops=4)
        assert _bx.shape[0] == _nn and _br.shape[0] == _nn, "jam changed n"
        assert _br.min() > 0.0, "jam produced a non-positive radius"
        assert (_dmat(_bx) - (_br[:, None] + _br[None, :])).min() >= -1e-9, "jam overlaps"
        assert (_walls(_bx) - _br).min() >= -1e-9, "jam escaped the square"
        assert _br.sum() >= _jr.sum() - 1e-9, "jam lost sum_r relative to its own input"
    _cx = _grid_xy(14, rng0, jitter=0.2)          # an n with no committed pack whatsoever
    _cr = _feasible_radii(_cx, np.minimum(_walls(_cx), _dmat(_cx).min(axis=1) / 2.0))
    _bx, _br = _jam(_cx, _cr, rng0, linprog, csr_matrix, _mini, 90, None, None, hops=3)
    assert _bx.shape[0] == 14 and (_dmat(_bx) - (_br[:, None] + _br[None, :])).min() >= -1e-9, \
        "jam broke on an uncommitted n"

    # --- CROSS-n TRANSFER ---------------------------------------------------------------------
    _s8 = set()
    for _k in range(8):                     # the 8 symmetries are distinct and stay in the square
        _q = _sym(xy0, _k)
        assert _q.min() >= LO - 1e-12 and _q.max() <= HI + 1e-12, "symmetry left the square"
        assert abs(np.sort(_dmat(_q)[np.triu_indices(xy0.shape[0], 1)]).sum()
                   - np.sort(_dmat(xy0)[np.triu_indices(xy0.shape[0], 1)]).sum()) < 1e-9, "sym warped"
        _s8.add(tuple(np.round(_q.ravel(), 9)))
    assert len(_s8) == 8, "the dihedral group collapsed"
    _m = xy0.shape[0]
    for _dn in (-3, -1, 2, 4):              # both directions: delete-smallest and grow-into-holes
        _xt, _rt = _transfer(_m + _dn, xy0, r0, rng0, 120, None)
        assert _xt.shape[0] == _m + _dn and _rt.shape[0] == _m + _dn, "transfer produced wrong n"
        assert _rt.min() > 0.0 and np.all(np.isfinite(_xt)), "transfer produced a degenerate circle"
        _xt, _rt = _slp(_xt, _rt, linprog, csr_matrix, 40, 0.02, None)
        _rt = _feasible_radii(_xt, _rt)
        assert (_dmat(_xt) - (_rt[:, None] + _rt[None, :])).min() >= -1e-9, "transfer+repair overlaps"
        assert (_walls(_xt) - _rt).min() >= -1e-9, "transfer+repair escaped the square"

    # --- PARTIAL SPLICE CROSSOVER --------------------------------------------------------------
    # A splice must (a) return exactly n circles for n above AND below both parents' sizes,
    # (b) be a genuine MIX (it must contain circles from each parent, else it is a no-op or a
    # plain transfer), and (c) be strictly feasible once repaired.
    _don_xy, _don_r = _slp(_grid_xy(_m + 5, rng0, hexagonal=True), np.full(_m + 5, 1e-6),
                           linprog, csr_matrix, 60, 0.04, None)
    _don_r = _feasible_radii(_don_xy, _don_r)
    _mixes = 0
    for _tn in (_m - 2, _m, _m + 3, _m + 7):
        _sp = _splice(_tn, xy0, r0, _don_xy, _don_r, rng0, 120, None, frac=0.5)
        assert _sp is not None, "splice degenerated at a mid cut"
        _sx, _sr = _sp
        assert _sx.shape[0] == _tn and _sr.shape[0] == _tn, "splice produced wrong n"
        assert _sr.min() > 0.0 and np.all(np.isfinite(_sx)), "splice produced a degenerate circle"
        assert _sx.min() >= LO - 1e-12 and _sx.max() <= HI + 1e-12, "splice left the square"
        # parentage is read off the RADII: they are carried verbatim and, unlike the centres, are
        # invariant under the random dihedral re-orientation the splice applies to the donor
        _from_i = int(np.isin(_sr, r0).sum())
        _from_d = int(np.isin(_sr, _don_r).sum())
        if _from_i > 0 and _from_d > 0:
            _mixes += 1
        _sx, _sr = _slp(_sx, _sr, linprog, csr_matrix, 40, 0.02, None)
        _sr = _feasible_radii(_sx, _sr)
        assert (_dmat(_sx) - (_sr[:, None] + _sr[None, :])).min() >= -1e-9, "splice+repair overlaps"
        assert (_walls(_sx) - _sr).min() >= -1e-9, "splice+repair escaped the square"
    assert _mixes >= 3, "splice is not crossing over: a child took circles from only one parent"
    # the admission rule must never emit two circles that coincide
    _sx, _sr = _splice(_m, xy0, r0, _don_xy, _don_r, rng0, 120, None, frac=0.5)
    assert _dmat(_sx).min() > 1e-9, "splice admitted coincident centres"

    os.environ["CP_SOLVER_CPU"] = "10"
    m1 = _Meter(500000)
    rng = np.random.RandomState(0)
    solve(make_eval(m1), m1, rng, [27, 31])
    assert 27 in best and 31 in best, "first call produced no feasible packing"
    print("  call 1:", {k: round(v[0], 6) for k, v in best.items()}, "nfev", m1.used)

    # CRITICAL: a SECOND solve() in the SAME process must still work (no module-level absolute
    # deadline), and must survive len(targets) == 1.
    first = dict((k, v[0]) for k, v in best.items())
    best.clear()
    m2 = _Meter(500000)
    solve(make_eval(m2), m2, rng, [29])
    assert 29 in best, "second call in the same process produced NO feasible packing"
    print("  call 2:", {k: round(v[0], 6) for k, v in best.items()}, "nfev", m2.used)

    # an n with NO committed pack (and no record entry) must cold-start cleanly
    best.clear()
    m3 = _Meter(500000)
    solve(make_eval(m3), m3, rng, [12])
    assert 12 in best, "cold start on an uncommitted n failed"
    print("  call 3 (cold n=12):", round(best[12][0], 6))

    # a starved evaluation budget must not crash or emit anything infeasible
    m4 = _Meter(3)
    solve(make_eval(m4), m4, rng, [27, 29])
    assert m4.used <= 3

    for n, (s, pk) in best.items():
        x, y, r = pk[:, 0], pk[:, 1], pk[:, 2]
        assert pk.shape[0] == n and r.min() > 0
        assert np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min() >= -1e-9
    assert first[27] > 0 and first[31] > 0

    # ---- REGIONAL RUIN-AND-RECREATE: exactly n circles out, a genuinely NEW region in, and the
    # frozen complement untouched.  The property that actually matters is the last one: a _rebuild
    # that silently returned its input would pass a count/feasibility check and be a no-op move.
    for _n in (27, 40):
        _p = _read_pack(_n)
        if _p is None:
            xy0 = _grid_xy(_n, rng0, jitter=0.2)
            r0 = _feasible_radii(xy0, np.minimum(_walls(xy0), _dmat(xy0).min(axis=1) / 2.0))
        else:
            xy0, r0 = _p
        for _mode in ("hex", "greedy"):
            _rb = _rebuild(xy0, r0, linprog, csr_matrix, rng0, 160, None, mode=_mode, kfrac=0.25)
            assert _rb is not None, "rebuild returned None on a healthy packing"
            _rx, _rr, _sd = _rb
            assert _rx.shape == (_n, 2) and _rr.shape == (_n,), "rebuild changed the circle count"
            _rr = _feasible_radii(_rx, _rr)
            assert _rr.min() > 0.0
            assert _dmat(_rx).min() > 1e-9, "rebuild produced coincident centres"
            assert np.min(_dmat(_rx) - (_rr[:, None] + _rr[None, :])) >= -1e-9
            assert np.min(_walls(_rx) - _rr) >= -1e-9
            _k = _sd.size
            assert 4 <= _k < _n
            # the (n-k) frozen circles are carried over VERBATIM ...
            _old = set(map(tuple, np.round(xy0, 12).tolist()))
            _kept = sum(1 for _q in np.round(_rx[: _n - _k], 12).tolist() if tuple(_q) in _old)
            assert _kept == _n - _k, "rebuild disturbed the frozen complement"
            # ... and the k rebuilt centres are NEW, not the deleted ones put back
            _fresh = sum(1 for _q in np.round(_rx[_n - _k:], 12).tolist() if tuple(_q) not in _old)
            assert _fresh >= max(1, _k // 2), "rebuild reproduced the region it deleted"
    # a rebuild must also decline gracefully rather than crash on a packing too small to cut
    assert _rebuild(np.zeros((5, 2)) + np.array([[0.0, 0.0], [.1, 0], [0, .1], [-.1, 0], [0, -.1]]),
                    np.full(5, 0.01), linprog, csr_matrix, rng0, 60, None) is None
    print("  rebuild: hex+greedy OK on n=27,40 (frozen complement intact, region genuinely new)")

    # --- unit: the elite ARCHIVE.  Its whole value is that it keeps DISTINCT things, so assert
    #     exactly that: the descriptor ignores relabelling, crowding evicts the near-duplicate and
    #     not the distinct-but-weaker member, and the champion is never lost.
    _d1 = _descriptor(np.array([0.1, 0.2, 0.3]))
    _d2 = _descriptor(np.array([0.3, 0.1, 0.2]))
    assert np.allclose(_d1, _d2), "descriptor is not permutation-invariant"
    assert abs(_d1.sum() - 1.0) < 1e-12, "descriptor is not normalized"
    assert _descriptor(np.zeros(3)).shape == (3,), "descriptor blew up on an all-zero packing"
    _far = float(np.abs(_descriptor(np.array([0.3, 0.3, 0.3])) -
                        _descriptor(np.array([0.05, 0.1, 0.7]))).sum())
    _near = float(np.abs(_descriptor(np.array([0.3, 0.3, 0.3])) -
                         _descriptor(np.array([0.3, 0.3, 0.301]))).sum())
    assert _far > 20.0 * _near > 0.0, "descriptor does not separate distinct radius profiles"

    _ar = _Archive(3)
    _z = np.zeros((3, 2))
    _ar.add(_z, np.array([0.30, 0.30, 0.30]), 0.90)      # champion, profile A
    _ar.add(_z, np.array([0.05, 0.10, 0.70]), 0.85)      # distinct profile B, weaker
    _ar.add(_z, np.array([0.30, 0.30, 0.2999]), 0.899)   # near-duplicate of the champion
    _ar.add(_z, np.array([0.02, 0.40, 0.40]), 0.82)      # distinct profile C, weakest
    assert len(_ar) == 3, "archive exceeded its capacity"
    assert max(_ar.s) == 0.90, "archive evicted its champion"
    assert 0.899 not in _ar.s, "crowding kept the near-duplicate over a distinct member"
    assert 0.85 in _ar.s and 0.82 in _ar.s, "crowding evicted a distinct member"
    assert _ar.spread() > 0.05, "archive collapsed onto one structure"
    _cap1 = _Archive(1)                                   # degenerate capacity must not loop forever
    _cap1.add(_z, np.array([0.3, 0.3, 0.3]), 0.9)
    _cap1.add(_z, np.array([0.3, 0.3, 0.3]), 0.8)
    assert len(_cap1) == 1 and _cap1.s[0] == 0.9
    _seen = set()
    for _t in range(60):
        _j = _ar.sample(rng0)
        assert _j is not None and 0 <= _j < len(_ar)
        _seen.add(_j)
    assert len(_seen) == 3, "archive sampling never draws the weaker elites"
    assert _ar.sample(rng0, skip=0) != 0, "archive sampling ignored `skip`"
    assert _Archive(4).sample(rng0) is None, "empty archive must return None, not crash"

    # --- unit: intra-n CROSSOVER of two archive elites yields a repairable n-circle packing
    _n = 24
    _xa = _grid_xy(_n, rng0, hexagonal=False, jitter=0.15)
    _ra = _feasible_radii(_xa, np.minimum(_walls(_xa), _dmat(_xa).min(axis=1) / 2.0))
    _xb = _grid_xy(_n, rng0, hexagonal=True, jitter=0.15)
    _rb = _feasible_radii(_xb, np.minimum(_walls(_xb), _dmat(_xb).min(axis=1) / 2.0))
    _ar2 = _Archive(4)
    _ar2.add(_xa, _ra, _ra.sum())
    _ar2.add(_xb, _rb, _rb.sum())
    assert len(_ar2) == 2 and _ar2.spread() > 0.0, "two different lattices collapsed to one niche"
    _got = 0
    for _t in range(6):
        _sp = _splice(_n, _ar2.xy[0], _ar2.r[0], _ar2.xy[1], _ar2.r[1], rng0, 200, None)
        if _sp is None:
            continue
        _cx, _cr = _slp(_sp[0], _sp[1], linprog, csr_matrix, 40, 0.02, None)
        _cr = _feasible_radii(_cx, _cr)
        assert _cx.shape == (_n, 2) and _cr.shape == (_n,), "cross changed the circle count"
        assert _cr.min() > 0.0
        assert np.min(_dmat(_cx) - (_cr[:, None] + _cr[None, :])) >= -1e-9, "cross infeasible"
        assert np.min(_walls(_cx) - _cr) >= -1e-9, "cross left the square"
        _got += 1
    assert _got >= 4, "intra-n crossover almost always degenerated"
    print("  archive: crowding keeps %d distinct elites; intra-n crossover feasible %d/6"
          % (len(_ar), _got))

    # ---- SYMMETRY PROJECTION.  Three properties, each of which a broken version would pass the
    # others on: (1) the projection really lands in the fixed-point set of g -- the invariant that
    # the whole move exists for, checked BEFORE any repair, as a set equality of centres under g;
    # (2) the repaired result is exactly n strictly-feasible circles inside the square; (3) it is
    # a real MOVE -- an asymmetric input must actually be changed, or "symmetrize" is a no-op that
    # would silently burn one slot in 10 of the kinds cycle forever.
    from scipy.optimize import minimize as _minz
    _moved = 0
    for _n in (27, 33, 40):
        _p = _read_pack(_n)
        if _p is None:
            xy0 = _grid_xy(_n, rng0, jitter=0.35)
            r0 = _feasible_radii(xy0, np.minimum(_walls(xy0), _dmat(xy0).min(axis=1) / 2.0))
        else:
            xy0, r0 = _p
        for _k in _SYM_INVOL:
            # (1) the raw projection is exactly g-symmetric
            _Q = _sym(xy0, _k)
            _dx = xy0[None, :, 0] - _Q[:, None, 0]
            _dy = xy0[None, :, 1] - _Q[:, None, 1]
            assert np.allclose(_dx * _dx + _dy * _dy,
                               (_dx * _dx + _dy * _dy).T, atol=1e-12), "g is not an involution"
            _sy = _symmetrize(xy0, r0, rng0, _minz, linprog, csr_matrix, k=_k)
            assert _sy is not None, "symmetrize returned None on a feasible pack"
            _sxy, _sr = _sy
            assert _sxy.shape == (_n, 2) and _sr.shape == (_n,), "symmetrize changed circle count"
            assert _sr.min() > 0.0, "symmetrize emitted a zero radius"
            assert np.min(_dmat(_sxy) - (_sr[:, None] + _sr[None, :])) >= -1e-9, "symm overlaps"
            assert np.min(_walls(_sxy) - _sr) >= -1e-9, "symm left the square"
            if np.max(np.abs(_sxy - xy0)) > 1e-7:
                _moved += 1
    assert _moved >= 12, "symmetrize is a no-op on almost every input"
    # (1) again, but on the PRE-repair projection, which is where the invariant actually lives:
    # rebuild it here rather than trusting the post-_relax centres (the repair may break symmetry).
    for _k in _SYM_INVOL:
        _xy = rng0.uniform(LO + 0.05, HI - 0.05, (21, 2))
        _r = np.full(21, 0.01)
        _Q = _sym(_xy, _k)
        _dd = ((_xy[None, :, 0] - _Q[:, None, 0]) ** 2 + (_xy[None, :, 1] - _Q[:, None, 1]) ** 2)
        _iu, _ju = np.triu_indices(21, 0)
        _order = np.argsort(_dd[_iu, _ju], kind="stable")
        _tk = np.zeros(21, bool)
        _nxy = np.empty_like(_xy)
        for _t in _order:
            _i, _j = int(_iu[_t]), int(_ju[_t])
            if _tk[_i] or _tk[_j]:
                continue
            _tk[_i] = _tk[_j] = True
            if _i == _j:
                _nxy[_i] = 0.5 * (_xy[_i] + _Q[_i])
            else:
                _a = 0.5 * (_xy[_i] + _sym(_xy[_j:_j + 1], _k)[0])
                _nxy[_i] = _a
                _nxy[_j] = _sym(_a.reshape(1, 2), _k)[0]
        assert _tk.all(), "symmetrize left a circle unpaired"
        # the projected set must be setwise invariant under g
        _img = _sym(_nxy, _k)
        _dm = ((_nxy[None, :, 0] - _img[:, None, 0]) ** 2
               + (_nxy[None, :, 1] - _img[:, None, 1]) ** 2).min(axis=1)
        assert _dm.max() < 1e-20, "projection is NOT in the fixed-point set of g (k=%d)" % _k
    # ---- SYMMETRY QUOTIENT.  The point of the quotient road is that the symmetry SURVIVES the
    # repair -- a version that silently fell back to the free path would still emit a feasible
    # n-circle packing and pass every check above, so assert the invariant on the OUTPUT.
    _qok = 0
    _qsym = 0
    for _n in (17, 28, 41):
        _rng0 = np.random.RandomState(9 + _n)
        _xy0 = _rng0.uniform(-0.45, 0.45, (_n, 2))
        _r0 = _feasible_radii(_xy0, np.full(_n, 0.05))
        for _k in _SYM_INVOL:
            _q = _symmetrize(_xy0, _r0, _rng0, _minz, linprog, csr_matrix, k=_k,
                             quotient=True, t_end=time.process_time() + 1.5)
            assert _q is not None, "quotient symmetrize returned None"
            _qxy, _qr = _q
            assert _qxy.shape == (_n, 2) and _qr.shape == (_n,), "quotient changed circle count"
            assert _qr.min() > 0.0, "quotient emitted a zero radius"
            assert _walls(_qxy).min() >= _qr.max() - 1e-9 or (_qr <= _walls(_qxy) + 1e-9).all(), \
                "quotient left a circle out of the square"
            _D = _dmat(_qxy)
            assert (_D - (_qr[:, None] + _qr[None, :])).min() >= -1e-9, "quotient overlaps"
            _qok += 1
            # setwise g-invariance of the CENTRES: every centre's image must be a centre
            _img = _sym(_qxy, _k)
            _dd = np.sqrt(((_img[:, None, :] - _qxy[None, :, :]) ** 2).sum(-1)).min(axis=1)
            if _dd.max() < 1e-6:      # HiGHS meets A_eq to ~1e-9 per step; 90 steps accumulate
                _qsym += 1
    assert _qok == 15, "quotient symmetrize failed on some involution"
    assert _qsym == 15, "quotient SLP left the symmetry orbit (%d/15)" % _qsym
    # the equality rows must actually constrain: a pair row set on a real pairing is non-empty
    _pk = [(0, 1), (2, 2)]
    _A, _b = _sym_rows([(_i, _j, 1) for _i, _j in _pk], np.array([0.1, -0.1, 0.0]),
                       np.array([0.2, 0.2, 0.3]),
                       np.array([0.05, 0.05, 0.04]), 3, 9, csr_matrix)
    assert _A is not None and _A.shape == (4, 9), "sym rows wrong shape: %r" % (_A.shape,)
    assert abs(float(_b[0])) < 1e-15 and abs(float(_b[1])) < 1e-15, "sym rows rhs not 0 on-orbit"
    for _k in _SYM_INVOL:                       # every element of _SYM_INVOL must be an involution
        _G = _sym_matrix(_k)
        assert np.abs(_G.dot(_G) - np.eye(2)).max() < 1e-15, "_SYM_INVOL[%d] is not an involution" % _k
    print("  quotient: 15/15 involution x size solves feasible and EXACTLY on the symmetry orbit")

    # ---- SUBGROUP QUOTIENT.  An order-4/8 group is only a real generalisation if (a) the group
    # table is a genuine group, (b) the projection lands EXACTLY on an H-orbit -- setwise invariant
    # under every element of H, not just one -- and (c) that invariance SURVIVES the quotient SLP.
    # A broken subgroup road that silently degenerated to a single mirror would pass feasibility.
    _T = _sym_comp()
    assert _T.shape == (8, 8) and sorted(set(_T[0].tolist())) == list(range(8)), "bad group table"
    for _a in range(8):                          # every row/col a permutation; identity is 0
        assert sorted(_T[_a].tolist()) == list(range(8)), "D4 table row %d not a permutation" % _a
        assert int(_T[_a, 0]) == _a and int(_T[0, _a]) == _a, "element 0 is not the identity"
    for _H in _SYM_GROUPS:                       # each declared subgroup is closed and inverse-closed
        assert _sym_closure(_H) == tuple(sorted(_H)), "_SYM_GROUPS entry %r is not a subgroup" % (_H,)
        for _a in _H:
            assert any(int(_T[_a, _b]) == 0 for _b in _H), "subgroup %r lacks an inverse" % (_H,)
    _gok = _ginv = _gtot = 0
    for _n in (13, 24, 33):
        _rng0 = np.random.RandomState(4242 + _n)
        _xy0 = np.clip(_rng0.uniform(LO + 0.06, HI - 0.06, size=(_n, 2)), LO, HI)
        _r0 = _feasible_radii(_xy0, np.full(_n, 0.5))
        for _H in _SYM_GROUPS:
            _gtot += 1
            _pr = _group_project(_xy0, _r0, _H)
            if _pr is None:
                continue
            _pxy, _pr2, _tri = _pr
            assert _pxy.shape == (_n, 2), "group projection changed circle count"
            # (a) the RAW projection is setwise invariant under EVERY element of H
            for _k in _H:
                _im = _sym(_pxy, _k)
                _dd = ((_im[:, None, :] - _pxy[None, :, :]) ** 2).sum(-1).min(axis=1)
                assert _dd.max() < 1e-20, "projection not invariant under g_%d of %r" % (_k, _H)
            # (b) every declared incidence really holds
            for _i, _j, _k in _tri:
                assert np.abs(_sym_matrix(_k).dot(_pxy[_i]) - _pxy[_j]).max() < 1e-12, \
                    "declared incidence p_%d = g_%d p_%d is false" % (_j, _k, _i)
            _q = _group_symmetrize(_xy0, _r0, linprog, csr_matrix, _H)
            if _q is None:
                continue
            _qxy, _qr = _q
            assert _qxy.shape == (_n, 2) and _qr.shape == (_n,), "group quotient changed count"
            assert _qr.min() > 0.0, "group quotient emitted a zero radius"
            assert np.min(_dmat(_qxy) - (_qr[:, None] + _qr[None, :])) >= -1e-9, "group q overlaps"
            assert np.min(_walls(_qxy) - _qr) >= -1e-9, "group q left the square"
            _gok += 1
            # (c) the SLP kept the whole orbit structure, not just feasibility
            _bad = 0
            for _k in _H:
                _im = _sym(_qxy, _k)
                _dd = ((_im[:, None, :] - _qxy[None, :, :]) ** 2).sum(-1).min(axis=1)
                if _dd.max() > 1e-12:
                    _bad += 1
            if _bad == 0:
                _ginv += 1
    assert _gtot == 12, "subgroup coverage changed"
    assert _gok >= 10, "subgroup quotient failed feasibility too often (%d/12)" % _gok
    assert _ginv == _gok, "subgroup quotient LEFT the orbit (%d/%d stayed)" % (_ginv, _gok)
    # orbit sizes must actually be > 2 somewhere, or this is just the involution road again
    _pr = _group_project(_xy0, _r0, _SYM_GROUPS[3])
    assert _pr is not None and max(len(set(x for x in (t[0], t[1]))) for t in _pr[2]) >= 1
    _cnt = {}
    for _i, _j, _k in _pr[2]:
        _cnt.setdefault(_i, set()).add(_j)
    assert max(len(v) for v in _cnt.values()) >= 4, "D4 produced no orbit larger than an involution"
    print("  subgroup quotient: %d/%d (H = Klein x2, C4, D4) feasible, %d/%d exactly on the "
          "H-orbit after the SLP" % (_gok, _gtot, _ginv, _gok))

    print("  symmetrize: 5 involutions x 3 sizes feasible; %d/15 were real moves; "
          "projection exactly g-invariant" % _moved)

    os.environ.pop("CP_SOLVER_CPU", None)
    print("SELF-TEST: PASS")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
