#!/usr/bin/env python3
"""Broad multistart ranked by SLP-KKT value -- the one stage never rebuilt.

WHY THIS EXISTS
---------------
Iterations 3 and 4 proved, by measurement, that *deforming* the incumbent is
finished: a monotone rule cannot leave it (1884 hops, 0 accepts) and a
threshold-accepting walk that CAN leave it reaches hundreds of distinct KKT
points, all worse (859 hops, 101 walk steps, 0 accepts, cliff below rank 2).
Everything measured so far is local to one funnel.

The untested hypothesis is that a better packing lives in a *different* funnel
and must be reached by construction or restart, not deformation.  And the broad
stage is the only part of the pipeline never rebuilt after the SLP insight:
`tools/solve.py` ranks its starts by the value of a half-converged penalty
descent -- exactly the comparator error iteration 1 already got burned by, one
stage earlier.  This module ranks starts by the value of an actual KKT point.

WHAT IS DIFFERENT FROM tools/solve.py's STAGE 1
-----------------------------------------------
1. Every surviving candidate is driven to a KKT point by `slp.slp_polish`
   before it is ranked, so basins are compared by what they are worth, not by
   how far Adam got.
2. Topology-diverse construction.  The optimum here is provably *unequal*
   (radii 0.0485-0.0943), and uniform-random or axis-aligned-lattice starts
   are a narrow slice of contact topologies.  Added: size-graded greedy
   (Apollonian-ish, large circles first), rotated hex lattices (a random
   lattice angle reaches topologies an axis-aligned one cannot), and
   random row-partition layouts (row counts chosen at random, which is what
   actually fixes the contact graph in a jammed square packing).
3. The screen is audited, not trusted.  A random sample of NON-elite
   candidates is polished too, and the Spearman correlation between the cheap
   screen and the KKT value is reported.  If that correlation is weak, the
   screen is the instrument at fault and the elite slice is meaningless --
   this is the check iteration 1 did not run.

PRICE, STATED BEFORE THE RUN (values: price a move in the problem's currency)
-----------------------------------------------------------------------------
Iteration 1 measured best-of-~500 penalty-descent starts at Sigma-r = 3.8352,
i.e. 7.6e-3 BELOW the incumbent 3.842635794938.  For this stage to produce a
gain, SLP polish must add more than 7.6e-3 to the top of the multistart
distribution -- whereas on an already-good point it added 6.2e-5.  So P(gain)
is low and stated as low up front.  What is bought regardless is the first
measurement of the *independent* KKT-value distribution at n=54: how far a
fresh funnel gets, and how many random funnels beat the incumbent (expected: 0,
which is itself the evidence that deformation-vs-restart is settled and that
the incumbent is not merely a lucky local basin).

    python3 tools/broad.py --seconds 240 --report artifacts/iter5_broad.json
    python3 tools/broad.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packlib as P  # noqa: E402
import slp as S  # noqa: E402

KINDS = ("uniform", "grid", "hexrow", "graded", "rothex", "rowpart")


def budget_seconds(default=240.0):
    v = os.environ.get("CP_SOLVE_SECONDS")
    try:
        return float(v) if v else default
    except ValueError:
        return default


# --------------------------------------------------------------------------
# constructions -- each returns (points (n,2), radii (n,) or None)
# --------------------------------------------------------------------------
def _init_uniform(n, rng):
    return rng.random((n, 2)), None


def _init_grid(n, rng):
    k = int(math.ceil(math.sqrt(n)))
    gx, gy = np.meshgrid((np.arange(k) + 0.5) / k, (np.arange(k) + 0.5) / k)
    p = np.stack([gx.ravel(), gy.ravel()], axis=1)
    p = p[rng.permutation(len(p))[:n]] + rng.normal(0, 0.35 / k, (n, 2))
    return p, None


def _init_hexrow(n, rng):
    rows = int(rng.integers(6, 10))
    per = int(math.ceil(n / rows))
    pts = []
    for i in range(rows):
        off = 0.5 / per if i % 2 else 0.0
        for j in range(per):
            pts.append(((j + 0.5) / per + off, (i + 0.5) / rows))
    p = np.array(pts)[rng.permutation(len(pts))[:n]]
    return p + rng.normal(0, 0.25 / per, (n, 2)), None


def _init_graded(n, rng):
    """Size-graded greedy: sample unequal target radii, place biggest first.

    The n=54 optimum is genuinely unequal (0.0485..0.0943, ratio 1.94).  A
    uniform-random cloud is a start with no size structure at all, so the
    descent has to invent the grading; here it is built in, which lands the
    start in a different contact-topology family.
    """
    spread = float(rng.uniform(1.0, 2.6))            # r_max / r_min target
    w = np.linspace(1.0, spread, n)[::-1].copy()     # descending target sizes
    w *= float(rng.uniform(0.85, 1.15))
    scale = 0.5 / math.sqrt(n) / w.mean()            # rough area normalisation
    tgt = w * scale
    p = np.empty((n, 2))
    cand = 96
    for i in range(n):
        c = rng.random((cand, 2))
        edge = np.minimum(np.minimum(c[:, 0], c[:, 1]),
                          np.minimum(1 - c[:, 0], 1 - c[:, 1]))
        if i == 0:
            room = edge
        else:
            d = np.sqrt((c[:, 0][:, None] - p[:i, 0][None, :]) ** 2 +
                        (c[:, 1][:, None] - p[:i, 1][None, :]) ** 2) - tgt[:i][None, :]
            room = np.minimum(d.min(axis=1), edge)
        p[i] = c[int(np.argmax(room - tgt[i]))]
    return p, tgt


def _init_rothex(n, rng):
    """Hexagonal lattice at a random angle and spacing, cropped to the square."""
    theta = float(rng.uniform(0.0, math.pi / 3.0))
    s = float(rng.uniform(0.9, 1.25)) / math.sqrt(n / 0.9)
    ct, st = math.cos(theta), math.sin(theta)
    m = int(2.5 / s) + 2
    pts = []
    for i in range(-m, m + 1):
        for j in range(-m, m + 1):
            u = (j + 0.5 * (i & 1)) * s
            v = i * s * math.sqrt(3) / 2.0
            pts.append((0.5 + u * ct - v * st, 0.5 + u * st + v * ct))
    q = np.array(pts)
    inside = q[(q[:, 0] > 0.02) & (q[:, 0] < 0.98) &
               (q[:, 1] > 0.02) & (q[:, 1] < 0.98)]
    if len(inside) >= n:
        idx = rng.permutation(len(inside))[:n]
        p = inside[idx]
    else:
        extra = rng.random((n - len(inside), 2))
        p = np.vstack([inside, extra]) if len(inside) else rng.random((n, 2))
    return p + rng.normal(0, 0.2 * s, (n, 2)), None


def _init_rowpart(n, rng):
    """Random row partition: rows get random counts summing to n.

    In a jammed square packing the contact graph is essentially "how many
    circles sit in each row", so sampling that partition directly samples
    topologies, which jittering a fixed lattice never does.
    """
    # n>=11 keeps the original 5..10 band byte-for-byte (same single rng draw),
    # so nothing about the measured n=54 behaviour changes.  Below that, 5 rows
    # can need more cut positions than there are gaps and rng.choice raises --
    # found the first time this construction was ever run at n=8 (iteration 6).
    if n >= 11:
        rows = int(rng.integers(5, 11))
    else:
        rows = int(rng.integers(1, max(2, math.isqrt(n) + 1)))
    cuts = np.sort(rng.choice(np.arange(1, n), size=rows - 1, replace=False))
    counts = np.diff(np.concatenate([[0], cuts, [n]]))
    counts = np.maximum(counts, 1)
    while counts.sum() > n:
        i = int(np.argmax(counts))
        counts[i] -= 1
    while counts.sum() < n:
        counts[int(np.argmin(counts))] += 1
    pts = []
    for i, cnt in enumerate(counts):
        yy = (i + 0.5) / rows
        for j in range(int(cnt)):
            pts.append(((j + 0.5) / cnt, yy))
    p = np.array(pts, float)
    p += rng.normal(0, 0.12 / rows, p.shape)
    if rng.random() < 0.5:                      # half the time, columns not rows
        p = p[:, ::-1].copy()
    return p, None


_INIT = {
    "uniform": _init_uniform,
    "grid": _init_grid,
    "hexrow": _init_hexrow,
    "graded": _init_graded,
    "rothex": _init_rothex,
    "rowpart": _init_rowpart,
}


def init_batch_v2(B, n, rng, kinds=KINDS):
    """(X (3,B,n), kind names per start).  Radii warm-started where known."""
    X = np.empty((3, B, n))
    X[2] = 0.02
    names = []
    for b in range(B):
        k = kinds[b % len(kinds)]
        p, r0 = _INIT[k](n, rng)
        p = np.clip(np.asarray(p, float), 0.01, 0.99)
        X[0, b], X[1, b] = p[:, 0], p[:, 1]
        if r0 is not None:
            X[2, b] = np.clip(r0, 1e-3, 0.4)
        names.append(k)
    return X, names


# --------------------------------------------------------------------------
# statistics helper (numpy only -- the audit must not need scipy)
# --------------------------------------------------------------------------
def spearman(a, b):
    """Spearman rank correlation; nan if fewer than 3 points or no variance."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or len(a) != len(b):
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _rank(v):
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), float)
    ranks[order] = np.arange(len(v), dtype=float)
    # average ties
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    return ranks


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------
def broad(n=P.N_DEFAULT, seconds=None, seed=0, verbose=True, B=96,
          adam_iters=2200, refine_iters=6000, refine=24, audit=4,
          polish_iters=30, polish_time=1.5, incumbent=None):
    """Waves of construct -> Adam -> screen -> Adam refine -> SLP-KKT polish.

    Measured wave economics at n=54 (see the docstring above): broad Adam is
    0.06 s/start, refine Adam 0.16 s/start, an SLP polish to KKT 0.07 s and a
    *deep* polish adds exactly 0.0 beyond it once refined.  So polish is so
    cheap that EVERY refined candidate is polished -- there is no second
    screening stage to be biased.  The one surviving screen (which starts get
    refined) is audited: `audit` random non-top starts are pushed through the
    same refine+polish, and the report gives the Spearman correlation between
    the screen and the final KKT value.  A weak correlation there would mean
    the elite slice is meaningless -- the iteration-1 comparator error, one
    stage earlier.

    Returns (x, y, r, report).  Every returned config is strictly feasible.
    """
    t0 = time.time()
    seconds = budget_seconds() if seconds is None else seconds
    rng = np.random.default_rng(seed)
    c = P.Counters()
    slpc = S.SLPCounters()

    best = (-1.0, None, None, None)
    recs = []          # one dict per POLISHED candidate
    screens = []       # screen score of every candidate ever generated
    waves = 0
    t_adam = t_polish = 0.0

    wave_cost = None               # measured, so the guard never guesses
    while True:
        left = seconds - (time.time() - t0)
        # always run wave 1; after that only start a wave we can afford
        if waves > 0 and left < 1.15 * wave_cost:
            break
        t_wave = time.time()
        ta = time.time()
        X, kinds = init_batch_v2(B, n, rng)
        c.restarts += B
        P.adam_run(X, adam_iters, 60.0, 3e4, 4e-3, 1e-4, c)
        t_adam += time.time() - ta

        # --- the one screen: LP-score every start, rank, keep a slice
        order = []
        for b in range(B):
            xx, yy, rr = P.max_radii(X[0, b], X[1, b], X[2, b])
            s = float(rr.sum())
            screens.append(s)
            order.append((s, b))
        order.sort(key=lambda t: -t[0])
        picks = [b for _, b in order[:min(refine, B)]]
        top_set = set(picks)
        rest = [b for _, b in order[min(refine, B):]]
        if rest and audit > 0:                     # the instrument check
            picks += [int(v) for v in rng.choice(rest, size=min(audit, len(rest)),
                                                 replace=False)]
        screen_of = {b: s for s, b in order}

        # --- refine only the picked starts, then polish ALL of them
        ta = time.time()
        Xr = np.ascontiguousarray(X[:, picks, :])
        P.adam_run(Xr, refine_iters, 3e4, 3e6, 3e-4, 3e-6, c)
        t_adam += time.time() - ta

        tp = time.time()
        stop_at = max(0.3, min(2.0, 0.05 * seconds))
        for j, b in enumerate(picks):
            if seconds - (time.time() - t0) < stop_at:
                break
            xx, yy, rr = P.max_radii(Xr[0, j], Xr[1, j], Xr[2, j])
            s1 = float(rr.sum())
            px, py, pr = S.slp_polish(xx, yy, rr, iters=polish_iters, delta0=0.01,
                                      max_stall=5, time_limit=polish_time,
                                      counters=slpc)
            s2 = float(pr.sum())
            recs.append({"screen": screen_of[b], "refined": s1, "kkt": s2,
                         "kind": kinds[b], "elite": b in top_set, "wave": waves})
            if s2 > best[0]:
                best = (s2, px, py, pr)
        t_polish += time.time() - tp
        waves += 1
        dt = time.time() - t_wave
        wave_cost = dt if wave_cost is None else 0.5 * (wave_cost + dt)
        if verbose:
            print(f"[wave {waves:3d} {time.time()-t0:6.1f}s] starts={c.restarts} "
                  f"polished={len(recs)} best_kkt={best[0]:.12f}", flush=True)

    # deep-polish the top few KKT points found (shallow polish is a screen too)
    top = sorted(recs, key=lambda d: -d["kkt"])[:3]
    if best[1] is not None:
        px, py, pr = S.slp_polish(best[1], best[2], best[3], iters=150,
                                  delta0=0.02, counters=slpc)
        if float(pr.sum()) > best[0]:
            best = (float(pr.sum()), px, py, pr)

    kkt = np.array([d["kkt"] for d in recs]) if recs else np.array([])
    scr = np.array([d["screen"] for d in recs]) if recs else np.array([])
    ref = np.array([d["refined"] for d in recs]) if recs else np.array([])
    el = np.array([d["elite"] for d in recs], bool) if recs else np.array([], bool)
    wall = time.time() - t0
    report = {
        "n": n, "seed": seed, "budget_s": seconds, "wall_s": round(wall, 2),
        "waves": waves, "starts": int(c.restarts), "polished": len(recs),
        "t_adam_s": round(t_adam, 1), "t_polish_s": round(t_polish, 1),
        "best_kkt": best[0],
        "best_screen": float(max(screens)) if screens else None,
        "screen_to_kkt_spearman": spearman(scr, kkt),
        "refined_to_kkt_spearman": spearman(ref, kkt),
        "elite_median_kkt": float(np.median(kkt[el])) if el.any() else None,
        "audit_median_kkt": float(np.median(kkt[~el])) if (~el).any() else None,
        "audit_best_kkt": float(kkt[~el].max()) if (~el).any() else None,
        "n_audit": int((~el).sum()),
        "mean_polish_gain": float((kkt - ref).mean()) if len(kkt) else None,
        "max_polish_gain": float((kkt - ref).max()) if len(kkt) else None,
        "mean_refine_gain": float((ref - scr).mean()) if len(kkt) else None,
        "kkt_quantiles": ({q: float(np.quantile(kkt, q))
                           for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)}
                          if len(kkt) else None),
        "by_kind": {},
        "cost": c.as_dict(wall),
        "slp": {"nlp": slpc.nlp, "nit": slpc.nit, "polishes": slpc.polishes},
        "top3_kkt": [d["kkt"] for d in top],
        "records": recs,
    }
    for k in KINDS:
        sub = [d["kkt"] for d in recs if d["kind"] == k]
        allk = [1 for d in recs if d["kind"] == k]
        if sub:
            report["by_kind"][k] = {"polished": len(allk), "best": max(sub),
                                    "median": float(np.median(sub))}
    if incumbent is not None:
        report["incumbent"] = incumbent
        report["n_beating_incumbent"] = int((kkt > incumbent).sum()) if len(kkt) else 0
        report["gap_to_incumbent"] = incumbent - best[0]
    s, x, y, r = best
    if x is None:
        raise RuntimeError("no candidate produced -- budget too small")
    x, y, r = P.repair(x, y, r)
    return x, y, r, report


def _fmt(report):
    q = report["kkt_quantiles"] or {}
    lines = [
        f"waves={report['waves']} starts={report['starts']} "
        f"polished={report['polished']} wall={report['wall_s']}s "
        f"(adam {report['t_adam_s']}s / polish {report['t_polish_s']}s)",
        f"best KKT      = {report['best_kkt']:.12f}",
        f"best screen   = {report['best_screen']:.12f}"
        if report["best_screen"] else "",
        f"refine gain   mean={report['mean_refine_gain']:.3e}   "
        f"polish gain mean={report['mean_polish_gain']:.3e} "
        f"max={report['max_polish_gain']:.3e}" if report["mean_polish_gain"] else "",
        f"screen->KKT spearman = {report['screen_to_kkt_spearman']:.3f}   "
        f"refined->KKT spearman = {report['refined_to_kkt_spearman']:.3f}",
        f"screen audit: elite median {report['elite_median_kkt']:.6f} vs "
        f"{report['n_audit']} audited non-elite median "
        f"{report['audit_median_kkt']:.6f} (audit best "
        f"{report['audit_best_kkt']:.6f})" if report["audit_median_kkt"] else "",
        "KKT quantiles " + "  ".join(f"p{int(k*100)}={v:.6f}" for k, v in q.items()),
        "by kind: " + "  ".join(
            f"{k}[{v['polished']}] best={v['best']:.6f} med={v['median']:.6f}"
            for k, v in report["by_kind"].items()),
    ]
    if "incumbent" in report:
        lines.append(f"incumbent {report['incumbent']:.12f}: "
                     f"{report['n_beating_incumbent']} of {report['polished']} "
                     f"polished starts beat it; best is "
                     f"{report['gap_to_incumbent']:+.6e} away")
    return "\n".join(x for x in lines if x)


# --------------------------------------------------------------------------
def _self_test():
    ok = True
    rng = np.random.default_rng(7)

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")

    # 1. every construction is well formed and inside the square
    for k in KINDS:
        p, r0 = _INIT[k](54, rng)
        p = np.asarray(p, float)
        good = (p.shape == (54, 2) and np.isfinite(p).all())
        chk(f"init {k}: shape/finite", good, f"{p.shape}")
        p2 = np.clip(p, 0.01, 0.99)
        chk(f"init {k}: clips into square", (p2 >= 0).all() and (p2 <= 1).all())
        if r0 is not None:
            chk(f"init {k}: warm radii positive/finite",
                np.isfinite(r0).all() and (np.asarray(r0) > 0).all())

    # 2. constructions are not degenerate: two seeds give different layouts
    for k in KINDS:
        a, _ = _INIT[k](54, np.random.default_rng(1))
        b, _ = _INIT[k](54, np.random.default_rng(2))
        chk(f"init {k}: seed-dependent", not np.allclose(a, b))

    # 3. graded construction really is graded (that is its whole point)
    _, tg = _init_graded(54, np.random.default_rng(3))
    chk("graded: target radii span a real range", tg.max() / tg.min() > 1.05,
        f"ratio={tg.max()/tg.min():.2f}")

    # 4. batch assembles all kinds
    X, names = init_batch_v2(12, 54, rng)
    chk("init_batch_v2 shape", X.shape == (3, 12, 54), f"{X.shape}")
    chk("init_batch_v2 covers every kind", set(names) == set(KINDS))
    chk("init_batch_v2 finite", np.isfinite(X).all())

    # 5. spearman is right on known data
    chk("spearman monotone = 1", abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12)
    chk("spearman antitone = -1", abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1) < 1e-12)
    chk("spearman short = nan", math.isnan(spearman([1, 2], [1, 2])))
    chk("spearman constant = nan", math.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4])))

    # 6. MECHANISM: the SLP polish never loses ground against the screen
    #    (this is what makes ranking by KKT value legitimate)
    Xs, _ = init_batch_v2(4, 24, np.random.default_rng(5))
    P.adam_run(Xs, 400, 60.0, 3e4, 4e-3, 1e-4, None)
    worst = 0.0
    for b in range(4):
        xx, yy, rr = P.max_radii(Xs[0, b], Xs[1, b], Xs[2, b])
        px, py, pr = S.slp_polish(xx, yy, rr, iters=12, delta0=0.01, max_stall=4)
        worst = min(worst, float(pr.sum()) - float(rr.sum()))
        if not P.is_feasible(px, py, pr):
            worst = -1.0
    chk("polish is monotone and feasible on raw starts", worst >= -1e-12,
        f"min gain {worst:.2e}")

    # 7. end to end on a small instance
    x, y, r, rep = broad(n=20, seconds=9.0, seed=2, verbose=False, B=18,
                         adam_iters=500, refine_iters=800, refine=3, audit=1,
                         polish_time=0.5, incumbent=99.0)
    p, bx = P.violations(x, y, r)
    chk("short run is strictly feasible", p <= 0 and bx <= 0 and len(r) == 20,
        f"n={len(r)} sum_r={r.sum():.6f} pair={p:.1e}")
    chk("report has the audit fields",
        {"screen_to_kkt_spearman", "refined_to_kkt_spearman", "by_kind",
         "kkt_quantiles", "polished", "n_beating_incumbent", "n_audit",
         "audit_median_kkt"} <= set(rep))
    chk("run respects its budget", rep["wall_s"] < 16.0, f"{rep['wall_s']}s")
    chk("audited non-elites were polished too",
        rep["n_audit"] > 0 and any(not d["elite"] for d in rep["records"]))
    chk("every polished record carries screen/refined/kkt",
        all({"screen", "refined", "kkt"} <= set(d) for d in rep["records"]))
    chk("KKT value never below its own refined screen (monotone polish)",
        all(d["kkt"] >= d["refined"] - 1e-12 for d in rep["records"]))
    chk("best_kkt is the max over records",
        rep["best_kkt"] >= max(d["kkt"] for d in rep["records"]) - 1e-12)
    chk("returned config matches best_kkt",
        abs(float(r.sum()) - rep["best_kkt"]) < 1e-9,
        f"{r.sum():.9f} vs {rep['best_kkt']:.9f}")
    chk("impossible incumbent is reported as unbeaten",
        rep["n_beating_incumbent"] == 0)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=P.N_DEFAULT)
    ap.add_argument("--B", type=int, default=96)
    ap.add_argument("--refine", type=int, default=24)
    ap.add_argument("--audit", type=int, default=4)
    ap.add_argument("--out", default=None, help="write config here IF it improves")
    ap.add_argument("--report", default=None, help="write the full JSON report here")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        print("broad self-test")
        sys.exit(0 if _self_test() else 1)

    inc = None
    if a.out and os.path.exists(a.out):
        with open(a.out) as fh:
            inc = json.load(fh).get("sum_r", None)
    x, y, r, rep = broad(n=a.n, seconds=a.seconds, seed=a.seed, B=a.B,
                         refine=a.refine, audit=a.audit, incumbent=inc)
    print()
    print(_fmt(rep))
    print(f"\nsum_r = {r.sum():.12f}   (record {P.RECORD_N54:.12f}, "
          f"gap {P.RECORD_N54 - r.sum():+.6e})")
    pv, bv = P.violations(x, y, r)
    print(f"feasibility: max_pair={pv:.3e} max_box={bv:.3e}")
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(rep, fh, indent=1)
        print(f"wrote {a.report}")
    if a.out:
        if inc is not None and inc >= r.sum():
            print(f"NOT writing: stored {inc:.12f} >= new {r.sum():.12f}")
        else:
            P.save_config(a.out, x, y, r, cost=rep["cost"],
                          meta={"kind": "best", "seed": a.seed,
                                "method": "broad multistart ranked by SLP-KKT value",
                                "budget_s": a.seconds})
            print(f"wrote {a.out}")
