#!/usr/bin/env python3
"""Endgame search: basin hopping where every candidate is driven to a KKT point.

Iteration 1's search plateaued (160 basin hops, zero improvement) because its
acceptance test compared half-converged penalty points: two basins were ranked
by how far Adam happened to get, not by what they are worth.  slp.slp_polish
converges a config to an actual KKT point in ~1 s, so here the hop loop is

    perturb incumbent  ->  LP radii  ->  SLP polish to KKT  ->  accept if better

which compares local optima to local optima.  No Adam at all: SLP is monotone
and feasible from any starting point, so it can absorb the perturbation itself.

MOVE DESIGN (rewritten iteration 3 -- see the arithmetic, it is the whole point)
--------------------------------------------------------------------------
Iteration 2 measured every `teleport`/`shed` hop returning to the *identical*
12-digit Sigma-r.  The reason is not a bug, it is the geometry: the incumbent is
fully jammed (0 rattlers, mean 4.96 contacts, radii 0.0485-0.0943).  Pull one
circle out of its seat and drop it anywhere else and you have forfeited its
whole radius -- ~0.05, which is 50x the entire remaining gap to the record --
while the hole it left can only be reclaimed by its neighbours growing by
second-order amounts.  No local polish repays that.  So *any* move that
displaces a single circle far is dead on arrival, and `_best_gap_spot` merely
made the deadness silent by putting it straight back where it came from.

What can be cheap in a jammed packing is a move where many circles travel
together, so that no single seat is abandoned: the packing deforms instead of
tearing.  Hence the move set is now entirely COHERENT rearrangements --

  * jitter    -- gaussian shake of every center (the only iter-2 move that ever
                 landed an accept: 4/1094; all three accepted scales were large)
  * region    -- large shake confined to a random disc: re-tiles a neighbourhood
  * affine    -- global stretch/rotate about the square's center, then clip
  * swirl     -- rotate a disc of circles about its own axis, tapering to zero
                 at the rim so the deformation is continuous; changes contact
                 topology without opening a hole
  * recluster -- lift k spatially-adjacent circles and re-drop them *inside
                 their own footprint*, so the mass stays local and recoverable

Every hop records whether it was DEGENERATE (polished back onto the incumbent),
a NEAR MISS, or an accept, so the next iteration inherits a measurement instead
of a guess.

ACCEPT RULE (rewritten iteration 4 -- the measurement that forced it)
--------------------------------------------------------------------------
Iteration 3 ran the coherent move set for 1884 hops and accepted NOTHING.  The
ledger and `--probe` explain why, and it is not the moves: from the incumbent,
sigma <= 0.005 snaps back onto the identical point 100% of the time, sigma =
0.01/0.02 escapes half the time but its best escapee is 1.39e-4 BELOW the
incumbent, and sigma >= 0.04 leaves the good basin family entirely (0.037-0.070
worse).  So every local optimum this move set can reach is worse than where we
stand -- and an accept-only-if-better test can never traverse to any of them,
no matter how many hops it is given.  The binding constraint is the ACCEPT
RULE, not the move generator (iteration 2's lesson, one level up).

Hence a threshold-accepting walk (Dueck & Scheuer) over KKT points:

    * the WALKER may step downhill by up to T, so the 1.4e-4-down neighbours
      become traversable;
    * T cycles geometrically from T_HI down to T_LO and resets, so each cycle
      is an explore-then-settle pass and exploration stays alive for the whole
      budget;
    * T_HI = 8e-4 is chosen from the ladder: comfortably above the 1.4e-4 /
      7.0e-4 neighbour band, far below the 3.7e-2 junk band, so the walk can
      reach neighbours but can never buy its way out of the good family;
    * BEST is a separate incumbent that only ever rises, so the reported Sigma-r
      cannot fall no matter how far the walker wanders;
    * if the walker drifts more than MAX_DRIFT below best it is teleported back
      to best (a diffusing walk has no other floor).

Also measured-driven: `recluster` was the only move with a low snap-back rate
(3.7% vs 66-84%) and the best near-miss rate, yet got 8.5% of the hops -- it is
reweighted up.  Jitter's scale range is narrowed to the band the probe showed
actually escapes without leaving the family (5e-3 .. 2.5e-2).

    python3 tools/endgame.py --in bench/bench1/best.json --out bench/bench1/best.json
    python3 tools/endgame.py --accept monotone ...     # iteration-3 behaviour
    python3 tools/endgame.py --self-test
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


def budget_seconds(default=240.0):
    v = os.environ.get("CP_SOLVE_SECONDS")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(math.log(lo), math.log(hi))))


def _mv_jitter(x, y, r, rng):
    """Gaussian shake of every center.  Coherent in the weak sense that no
    circle is singled out: each seat moves a little, none is abandoned."""
    n = len(x)
    # band narrowed iteration 4: --probe measured sigma<=5e-3 snapping back 100%
    # of the time and sigma>=4e-2 landing outside the good basin family.
    s = _loguniform(rng, 5e-3, 2.5e-2)
    return x + rng.normal(0, s, n), y + rng.normal(0, s, n), f"s{s:.4f}"


def _mv_region(x, y, r, rng):
    """Large shake confined to a random disc -- re-tile one neighbourhood while
    the rest of the packing holds still and provides the boundary condition."""
    p = rng.random(2)
    R = float(rng.uniform(0.12, 0.38))
    m = (x - p[0]) ** 2 + (y - p[1]) ** 2 < R * R
    k = int(m.sum())
    if k < 3:
        return _mv_jitter(x, y, r, rng)
    s = _loguniform(rng, 5e-3, 6e-2)
    x2, y2 = x.copy(), y.copy()
    x2[m] += rng.normal(0, s, k)
    y2[m] += rng.normal(0, s, k)
    return x2, y2, f"k{k}s{s:.3f}"


def _mv_affine(x, y, r, rng):
    """Global stretch + rotation about the square's center, then clip.  A
    coherent deformation of the whole packing: relative order is preserved but
    every contact is re-tensioned at once."""
    a = float(rng.normal(0, 0.030))
    b = float(rng.normal(0, 0.030))
    th = float(rng.normal(0, 0.060))
    u = (x - 0.5) * (1.0 + a)
    v = (y - 0.5) * (1.0 + b)
    ct, st = math.cos(th), math.sin(th)
    return 0.5 + ct * u - st * v, 0.5 + st * u + ct * v, f"a{a:+.3f}t{th:+.3f}"


def _mv_swirl(x, y, r, rng):
    """Rotate a disc of circles about its own axis, with the angle tapering to
    zero at the rim so the field is continuous.  Changes the contact topology
    inside the disc without opening a hole anywhere."""
    p = rng.random(2)
    R = float(rng.uniform(0.15, 0.42))
    d2 = (x - p[0]) ** 2 + (y - p[1]) ** 2
    m = d2 < R * R
    k = int(m.sum())
    if k < 4:
        return _mv_jitter(x, y, r, rng)
    th = float(rng.uniform(0.12, 1.0)) * (1.0 if rng.random() < 0.5 else -1.0)
    w = np.zeros_like(x)
    w[m] = 1.0 - np.sqrt(d2[m]) / R          # 1 at the axis, 0 at the rim
    ang = th * w
    u, v = x - p[0], y - p[1]
    ct, st = np.cos(ang), np.sin(ang)
    return p[0] + ct * u - st * v, p[1] + st * u + ct * v, f"k{k}t{th:+.2f}"


def _mv_recluster(x, y, r, rng):
    """Lift k spatially-adjacent circles and re-drop them at random *inside
    their own footprint*, then let the ring around them breathe.  Unlike a
    teleport the abandoned mass stays local, so the polish can reclaim it."""
    n = len(x)
    # k <= n, else argsort(d2)[:k] returns fewer indices than the k angles drawn
    # below and the assignment raises.  Unreachable at n=54 (k <= 5); found the
    # first time the endgame ran at n=4 (iteration 6's cross-n calibration).
    k = int(rng.integers(2, 6)) if n >= 6 else max(1, n - 1)
    seed = int(rng.integers(0, n))
    d2 = (x - x[seed]) ** 2 + (y - y[seed]) ** 2
    grp = np.argsort(d2)[:k]
    cx, cy = float(x[grp].mean()), float(y[grp].mean())
    rad = float(max(np.sqrt(d2[grp]).max(), 0.05))
    x2, y2 = x.copy(), y.copy()
    ang = rng.random(k) * 2.0 * math.pi
    rr = rad * np.sqrt(rng.random(k))
    x2[grp] = cx + rr * np.cos(ang)
    y2[grp] = cy + rr * np.sin(ang)
    nb = (x - cx) ** 2 + (y - cy) ** 2 < (2.2 * rad) ** 2
    j = int(nb.sum())
    x2[nb] += rng.normal(0, 6e-3, j)
    y2[nb] += rng.normal(0, 6e-3, j)
    return x2, y2, f"k{k}nb{j}"


# name, cumulative probability, function
# Reweighted iteration 4 from the iteration-3 ledger: recluster was the only
# move that reliably left the incumbent's basin (3.7% snap-back vs 66-84% for
# the rest) and had the best near-miss rate, on 8.5% of the hops.
MOVES = (
    ("jitter",    0.26, _mv_jitter),
    ("region",    0.48, _mv_region),
    ("swirl",     0.62, _mv_swirl),
    ("affine",    0.72, _mv_affine),
    ("recluster", 1.00, _mv_recluster),
)


def perturb(x, y, r, rng):
    """Return (x, y, name, detail) for one basin-hopping move."""
    u = rng.random()
    for name, cum, fn in MOVES:
        if u < cum:
            x2, y2, detail = fn(x, y, r, rng)
            return (np.clip(x2, 0.0, 1.0), np.clip(y2, 0.0, 1.0), name, detail)
    raise AssertionError("move table does not reach 1.0")


# --- acceptance rule (iteration 4) ----------------------------------------
# Factored out as pure functions so the self-test can check the mechanism
# directly rather than hoping a short run happens to exercise it.
T_HI = 8e-4       # top of the threshold cycle: above the neighbour band (1.4e-4
                  # .. 7.0e-4), far below the junk band (3.7e-2)
T_LO = 2e-5       # bottom of the cycle: effectively monotone, settles the walk
CYCLE = 140       # hops per explore->settle cycle
MAX_DRIFT = 3e-3  # walker is teleported back to best if it falls this far


def threshold(hop, t_hi=T_HI, t_lo=T_LO, cycle=CYCLE):
    """Geometric decay from t_hi to t_lo across a cycle, then reset."""
    u = (hop % cycle) / float(cycle)
    return float(t_hi * (t_lo / t_hi) ** u)


def accepts_walk(s_new, s_cur, t):
    """Threshold-accepting test: a step downhill by up to t is allowed."""
    return s_new > s_cur - t


def endgame(x, y, r, seconds, seed=0, verbose=True, hop_iters=25,
            hop_stall=6, delta0=0.02, accept="threshold", t_hi=T_HI,
            t_lo=T_LO, cycle=CYCLE, max_drift=MAX_DRIFT):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    c = S.SLPCounters()

    # start from the KKT point of the incumbent basin so hops are compared fairly
    x, y, r = S.slp_polish(x, y, r, iters=150, delta0=delta0, counters=c)
    best = (float(r.sum()), x.copy(), y.copy(), r.copy())
    cur = best                       # the WALKER; best only ever rises
    if verbose:
        print(f"[seed  {time.time()-t0:6.1f}s] incumbent KKT = {best[0]:.12f}"
              f"   accept={accept}", flush=True)

    # per-move ledger: hops, accepts (new best), walk-steps taken, degenerate
    # (polished back onto the point we hopped from), near misses
    NEAR = 1e-4
    NEARBEST = 5e-5   # trigger a medium refine: could this become a new best?
    stats = {name: dict(n=0, acc=0, walk=0, deg=0, near=0, best=-1.0)
             for name, _, _ in MOVES}

    hops = 0
    accepts = 0        # strict improvements to best
    walk_steps = 0     # walker moved to a different KKT point
    walk_down = 0      # ... and that point was WORSE than where it stood
    refines = 0
    resets = 0
    worst_drift = 0.0
    # census of the KKT points the walk actually visits: sum_r rounded to 1e-11.
    # Iteration 3's probe suggested the optima graph might be nearly a single
    # point; this measures it directly instead of inferring it.
    census = {}

    while time.time() - t0 < seconds - 2.0:
        s_cur, xc, yc, rc = cur
        t = 0.0 if accept == "monotone" else threshold(hops, t_hi, t_lo, cycle)
        xp, yp, name, detail = perturb(xc, yc, rc, rng)
        xn, yn, rn = P.max_radii(xp, yp)
        xn, yn, rn = S.slp_polish(xn, yn, rn, iters=hop_iters, delta0=delta0,
                                  max_stall=hop_stall, counters=c,
                                  time_limit=max(0.5, seconds - (time.time() - t0)))
        hops += 1
        st = stats[name]
        st["n"] += 1
        s = float(rn.sum())
        st["best"] = max(st["best"], s)
        # did the polish just walk back onto the point we hopped from?
        disp = float(np.max(np.hypot(xn - xc, yn - yc)))
        degenerate = disp < 1e-9
        if degenerate:
            st["deg"] += 1
        else:
            census[round(s, 11)] = census.get(round(s, 11), 0) + 1
            if s > best[0] - NEAR:
                st["near"] += 1

        # a candidate that could plausibly become a new best earns a medium
        # refine first -- a shallow hop polish routinely stops a few e-5 short.
        if not degenerate and s > best[0] - NEARBEST:
            xn, yn, rn = S.slp_polish(xn, yn, rn, iters=40, delta0=delta0,
                                      counters=c)
            s = float(rn.sum())
            refines += 1
            st["best"] = max(st["best"], s)

        if s > best[0] + 1e-13:
            # a hit gets a deep polish before it becomes the incumbent
            xn, yn, rn = S.slp_polish(xn, yn, rn, iters=150, delta0=delta0,
                                      counters=c)
            s = float(rn.sum())
            if s > best[0] + 1e-13:
                if verbose:
                    print(f"[hop {hops:5d} {time.time()-t0:6.1f}s] "
                          f"{name}/{detail:>12s} {best[0]:.12f} -> {s:.12f} "
                          f"(+{s-best[0]:.3e})", flush=True)
                best = (s, xn.copy(), yn.copy(), rn.copy())
                accepts += 1
                st["acc"] += 1

        # --- the walker's own move (this is the iteration-4 change) ---
        if not degenerate and accepts_walk(s, s_cur, t):
            cur = (s, xn.copy(), yn.copy(), rn.copy())
            walk_steps += 1
            st["walk"] += 1
            if s < s_cur:
                walk_down += 1
        # a diffusing walk has no floor of its own; give it one
        worst_drift = max(worst_drift, best[0] - cur[0])
        if cur[0] < best[0] - max_drift:
            cur = best
            resets += 1

        if verbose and hops % 100 == 0:
            print(f"[hop {hops:5d} {time.time()-t0:6.1f}s] best={best[0]:.12f} "
                  f"cur={cur[0]:.12f} T={t:.2e} gap={P.RECORD_N54-best[0]:.3e} "
                  f"LPs={c.nlp} walk={walk_steps}/{walk_down}down "
                  f"optima={len(census)}", flush=True)

    s, x, y, r = best
    x, y, r = P.repair(x, y, r)
    wall = time.time() - t0
    top = sorted(census.items(), key=lambda kv: -kv[0])[:12]
    cost = {"wall_s": round(wall, 3), "nfev": int(c.nlp), "nit": int(c.nit),
            "restarts": int(hops), "lp_solves": int(c.nlp),
            "slp_polishes": int(c.polishes), "hops": hops, "accepts": accepts,
            "accept_rule": accept, "walk_steps": walk_steps,
            "walk_down": walk_down, "refines": refines, "resets": resets,
            "worst_drift": worst_drift, "distinct_optima": len(census),
            "moves": {k: dict(v) for k, v in stats.items()}}
    if verbose:
        p, bx = P.violations(x, y, r)
        print(f"\nsum_r = {r.sum():.12f}   (record {P.RECORD_N54:.12f}, "
              f"gap {P.RECORD_N54 - r.sum():+.6e})")
        print(f"feasibility: max_pair={p:.3e} max_box={bx:.3e}")
        print(f"{'move':>10s} {'hops':>6s} {'acc':>4s} {'walk':>6s} "
              f"{'degen':>7s} {'near':>6s} {'best':>16s}")
        for k, v in stats.items():
            if v["n"]:
                print(f"{k:>10s} {v['n']:6d} {v['acc']:4d} "
                      f"{v['walk']/v['n']:5.1%} {v['deg']/v['n']:6.1%} "
                      f"{v['near']/v['n']:5.1%} {v['best']:16.12f}")
        print(f"\ndistinct KKT points visited: {len(census)}   "
              f"(top 12 by sum_r, with visit counts)")
        for val, cnt in top:
            print(f"   {val:.11f}  x{cnt:<5d} ({val - best[0]:+.3e} vs best)")
        print(f"cost: { {k: v for k, v in cost.items() if k != 'moves'} }")
    return x, y, r, cost


def probe_snapback(x, y, r, scales=(0.005, 0.01, 0.02, 0.04, 0.08, 0.16),
                   trials=16, seed=0, hop_iters=25, hop_stall=6, delta0=0.02):
    """How wide is the incumbent's basin of attraction under slp_polish?

    Iteration 3's per-move ledger showed 66-84% of hops polishing back onto the
    incumbent to within 1e-9 -- i.e. most of the search budget is spent
    re-deriving the point we already have.  This sweeps the gaussian-jitter
    scale and reports, per scale, the snap-back rate and how much value the
    escaping hops keep.  The useful sigma is the smallest one that escapes.
    """
    rng = np.random.default_rng(seed)
    c = S.SLPCounters()
    s_inc = float(r.sum())
    rows = []
    print(f"  incumbent = {s_inc:.12f}   n={len(x)}")
    print(f"  {'sigma':>7s} {'snapback':>9s} {'mean loss':>11s} {'best kept':>16s}")
    for s in scales:
        back, losses, bestk = 0, [], -1.0
        for _ in range(trials):
            xp = np.clip(x + rng.normal(0, s, len(x)), 0.0, 1.0)
            yp = np.clip(y + rng.normal(0, s, len(y)), 0.0, 1.0)
            xn, yn, rn = P.max_radii(xp, yp)
            xn, yn, rn = S.slp_polish(xn, yn, rn, iters=hop_iters, delta0=delta0,
                                      max_stall=hop_stall, counters=c)
            if float(np.max(np.hypot(xn - x, yn - y))) < 1e-9:
                back += 1
            else:
                v = float(rn.sum())
                losses.append(s_inc - v)
                bestk = max(bestk, v)
        ml = float(np.mean(losses)) if losses else float("nan")
        rows.append((s, back / trials, ml, bestk))
        print(f"  {s:7.3f} {back/trials:8.1%} {ml:11.3e} {bestk:16.12f}")
    return rows


def _self_test():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")

    rng = np.random.default_rng(3)
    n = 24
    X = P.init_batch(1, n, rng)
    P.adam_run(X, 1200, 60.0, 3e4, 4e-3, 1e-4)
    x0, y0, r0 = P.max_radii(X[0, 0], X[1, 0], X[2, 0])
    s0 = float(r0.sum())
    x, y, r, cost = endgame(x0, y0, r0, seconds=8.0, seed=1, verbose=False)
    p, b = P.violations(x, y, r)
    chk("output strictly feasible", p <= 0 and b <= 0 and len(r) == n,
        f"pair={p:.2e} box={b:.2e}")
    chk("never loses ground to its input", r.sum() >= s0 - 1e-12,
        f"{s0:.9f} -> {r.sum():.9f}")
    chk("improves on the input", r.sum() > s0 + 1e-9, f"+{r.sum()-s0:.3e}")
    chk("respects the budget", cost["wall_s"] < 14.0, f"{cost['wall_s']:.1f}s")
    chk("telemetry complete",
        {"wall_s", "nfev", "nit", "restarts", "hops", "accepts", "moves",
         "accept_rule", "walk_steps", "walk_down", "resets", "distinct_optima"}
        <= set(cost))
    chk("per-move ledger populated",
        sum(v["n"] for v in cost["moves"].values()) == cost["hops"],
        str({k: v["n"] for k, v in cost["moves"].items()}))
    chk("walker never drops the reported best below its input",
        cost["worst_drift"] >= 0.0 and r.sum() >= s0 - 1e-12,
        f"worst walker drift {cost['worst_drift']:.3e}")

    # --- the acceptance rule itself, tested as a mechanism, not by luck ------
    chk("threshold accepts a small downhill step",
        accepts_walk(1.0 - 0.4e-3, 1.0, 8e-4))
    chk("threshold rejects a large downhill step",
        not accepts_walk(1.0 - 2.0e-3, 1.0, 8e-4))
    chk("threshold rejects the junk band (3.7e-2 down) at every T in the cycle",
        not any(accepts_walk(1.0 - 3.7e-2, 1.0, threshold(h))
                for h in range(CYCLE)))
    chk("monotone rule (T=0) accepts only strict improvements",
        accepts_walk(1.0 + 1e-12, 1.0, 0.0) and not accepts_walk(1.0, 1.0, 0.0))
    ts = [threshold(h) for h in range(CYCLE)]
    chk("threshold decays within a cycle",
        all(ts[i] > ts[i + 1] for i in range(len(ts) - 1)),
        f"{ts[0]:.2e} -> {ts[-1]:.2e}")
    chk("threshold stays inside [T_LO, T_HI] and resets each cycle",
        abs(ts[0] - T_HI) < 1e-15 and min(ts) >= T_LO * 0.999
        and max(ts) <= T_HI * 1.001
        and abs(threshold(CYCLE) - threshold(0)) < 1e-15)

    # monotone mode must still be reachable (the iteration-3 behaviour) and must
    # never take a downhill walk step
    _, _, rm, cm = endgame(x0, y0, r0, seconds=5.0, seed=2, verbose=False,
                           accept="monotone")
    chk("monotone mode takes zero downhill steps", cm["walk_down"] == 0,
        f"walk={cm['walk_steps']} down={cm['walk_down']}")
    chk("monotone mode still feasible and non-losing", rm.sum() >= s0 - 1e-12)

    # the move table must be a proper distribution over the named moves
    chk("move table reaches 1.0", abs(MOVES[-1][1] - 1.0) < 1e-12)
    chk("move table is increasing",
        all(MOVES[i][1] < MOVES[i + 1][1] for i in range(len(MOVES) - 1)))

    # every move must (a) preserve n, (b) stay in the square, and (c) actually
    # move something -- the iteration-2 bug was a move that was silently a no-op
    for name, _, fn in MOVES:
        moved, inside, sized = [], True, True
        for t in range(24):
            xp, yp, _det = fn(x, y, r, np.random.default_rng(1000 + t))
            xp, yp = np.clip(xp, 0.0, 1.0), np.clip(yp, 0.0, 1.0)
            sized = sized and len(xp) == n and len(yp) == n
            inside = inside and xp.min() >= 0 and xp.max() <= 1 \
                and yp.min() >= 0 and yp.max() <= 1
            moved.append(float(np.max(np.hypot(xp - x, yp - y))))
        chk(f"move {name}: preserves n and stays in the square", sized and inside)
        chk(f"move {name}: never a no-op", min(moved) > 1e-6,
            f"min displacement {min(moved):.2e}, max {max(moved):.2e}")

    xp, yp, name, detail = perturb(x, y, r, rng)
    chk("perturb preserves n", len(xp) == n and len(yp) == n, f"{name}/{detail}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", dest="out", default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accept", choices=("threshold", "monotone"),
                    default="threshold",
                    help="threshold = non-monotone walk (iter 4); "
                         "monotone = iteration-3 accept-only-if-better")
    ap.add_argument("--t-hi", type=float, default=T_HI)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="measure the incumbent's basin width, then exit")
    a = ap.parse_args()
    if a.self_test:
        print("endgame self-test")
        sys.exit(0 if _self_test() else 1)
    if a.probe:
        doc = P.load_config(a.inp or "bench/bench1/best.json")
        arr = np.asarray(doc["circles"], float)
        print("snap-back probe")
        probe_snapback(arr[:, 0], arr[:, 1], arr[:, 2], seed=a.seed)
        sys.exit(0)

    secs = budget_seconds() if a.seconds is None else a.seconds
    if a.inp:
        doc = P.load_config(a.inp)
        arr = np.asarray(doc["circles"], float)
        x0, y0, r0 = arr[:, 0], arr[:, 1], arr[:, 2]
    else:
        rng = np.random.default_rng(a.seed)
        X = P.init_batch(1, P.N_DEFAULT, rng)
        P.adam_run(X, 3000, 60.0, 3e4, 4e-3, 1e-4)
        x0, y0, r0 = P.max_radii(X[0, 0], X[1, 0], X[2, 0])

    x, y, r, cost = endgame(x0, y0, r0, seconds=secs, seed=a.seed,
                            accept=a.accept, t_hi=a.t_hi)
    if a.out:
        prev = None
        if os.path.exists(a.out):
            with open(a.out) as fh:
                prev = json.load(fh).get("sum_r", None)
        if prev is not None and prev >= r.sum():
            print(f"NOT writing: stored {prev:.12f} >= new {r.sum():.12f}")
        else:
            P.save_config(a.out, x, y, r, cost=cost,
                          meta={"kind": "best", "seed": a.seed,
                                "method": "SLP-KKT basin hopping (tools/endgame.py)",
                                "budget_s": secs})
            print(f"wrote {a.out}"
                  + (f" (improved from {prev:.12f})" if prev is not None else ""))
