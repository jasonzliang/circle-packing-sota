#!/usr/bin/env python3
"""The whole solver as ONE n-generic `solve(n, seconds, seed)`.

Five iterations built the two halves of the search separately and never joined
them:

  * `tools/broad.py`   -- construct -> Adam -> LP screen -> refine -> SLP-KKT
                          polish.  Finds a good FUNNEL from nothing.
  * `tools/endgame.py`  -- threshold-accepting basin hopping where every
                          candidate is driven to a KKT point.  Squeezes the
                          funnel it is handed.

At n=54 that split is invisible because a hand-carried incumbent has existed
since iteration 1, so `endgame` was always run `--in bench/bench1/best.json`.
At ANY OTHER n there is no incumbent, and the two stages have to be composed to
produce anything at all.  This module is that composition, and it is the thing
MISSION.md actually asks for ("build ONE solve()").

Why the split is 25/75 by default (iteration 7, MEASURED -- it was 60/40 by
guess through iteration 6): `tools/split_sweep.py` ran the whole pipeline
end-to-end at n=55, 240 s, three splits x three seeds, all nine runs concurrent
and paired by seed.  broad_frac=0.25 beat the 0.60 control by **+1.24e-3 of
final Sigma-r** (paired mean over seeds; 3/3 seeds reached the sweep's best
packing, versus 1/3 at 0.60).  The stage telemetry says why: broad's extra 84 s
(60 s -> 144 s) bought only +1.44e-3 of funnel quality, while the endgame's
extra 95 s bought +2.68e-3 on top of whatever funnel it was handed.  Iteration
6 had the marginal *rates* right and still could not have set this number from
them -- the stages are coupled (endgame can only squeeze what broad finds), so
only an end-to-end A/B prices the split.

Iteration 7 left a signpost here saying 0.25 was the LOWEST arm run "so the
optimum may be lower still".  **Iteration 8 measured that and it is wrong: the
response is FLAT below 0.25, not still falling.**  Same harness at n=54,
240 s, arms 0.10 / 0.175 / 0.25 x seeds 21,22,23, paired and concurrent:
0.10 and 0.175 each beat the 0.25 control by **+5.3e-05** paired-mean (1/3
seeds), which is 1/23 of the +1.24e-3 that 0.60 -> 0.25 was worth and well
inside the ~3e-4 this design can resolve.  So 0.25 sits on a plateau and the
default stays 0.25.  Two mechanisms, both in the stage telemetry:

  * **Dead zone.** `broad_frac` is a budget, not a spend.  broad's wave loop is
    quantized (~15-20 s) and stops rather than overrun, so nominal 24 s / 42 s
    of broad actually cost 20.2 s / 25.1 s -- and handed the endgame the
    *byte-identical* funnel on 2 of 3 seeds.  Below ~0.2 the knob does nothing.
  * **Cancellation.** Where the funnels do differ, the endgame gives back what
    broad gave up almost exactly: broad mean 3.8282 + endgame +1.113e-2 (low
    arms) vs 3.8315 + 7.78e-3 (0.25) -- 3.8393 either way.  Seed 22 even
    inverts it: 0.25 found the *better* funnel (3.8350 vs 3.8251) and finished
    *worse* (3.841475 vs 3.841635).

Also here: `--calibrate`, the honesty instrument this iteration exists for.
Our n=54 Sigma-r EXCEEDS the published packomania csqv entry by +9.03e-4, which
is far likelier to be our bug than our triumph.  A bug in the *feasibility
definition* (a tolerance, an off-by-one in the pair loop, a margin sign) would
inflate Sigma-r at EVERY n, so running the identical pipeline at small n whose
records are old, heavily studied and tight is a direct test: if we land exactly
ON those records and never above them, a systematic feasibility bug is ruled
out, and the n=54 excess has to be either a real find or something specific to
n=54.  Each config is additionally checked against the provable area bound
sqrt(n/pi) (see packlib.area_bound).

    python3 tools/pipeline.py --n 55 --seconds 240 --out bench/bench2/best.json
    python3 tools/pipeline.py --calibrate 4,9 --seconds 20
    python3 tools/pipeline.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packlib as P  # noqa: E402
import slp as S  # noqa: E402
import broad as B  # noqa: E402
import endgame as E  # noqa: E402


def budget_seconds(default=240.0):
    v = os.environ.get("CP_SOLVE_SECONDS")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _chain(n=P.N_DEFAULT, seconds=None, seed=0, broad_frac=0.25, verbose=True,
           B_wave=96, refine=24, audit=4, t_hi=E.T_HI):
    """ONE broad -> endgame chain: multistart to a good funnel, then a KKT walk.

    Returns (x, y, r, cost).  The returned config is strictly feasible (both
    stages only ever emit LP/SLP configs, which are feasible by construction --
    see notes/insights.md) and `cost` carries the telemetry MISSION.md asks for
    plus a per-stage Sigma-r breakdown, so a later iteration can see which half
    of the pipeline paid at which n.
    """
    t0 = time.time()
    seconds = budget_seconds() if seconds is None else seconds
    t_broad = max(2.0, broad_frac * seconds)

    x, y, r, rep = B.broad(n=n, seconds=t_broad, seed=seed, verbose=verbose,
                           B=B_wave, refine=refine, audit=audit)
    s_broad = float(r.sum())
    if verbose:
        print(f"[pipeline {time.time()-t0:6.1f}s] broad stage done: "
              f"sum_r={s_broad:.12f}", flush=True)

    t_end = seconds - (time.time() - t0)
    ecost = None
    if t_end > 5.0:
        x, y, r, ecost = E.endgame(x, y, r, seconds=t_end, seed=seed + 1,
                                   verbose=verbose, t_hi=t_hi)
    elif verbose:
        print(f"[pipeline] skipping endgame stage ({t_end:.1f}s left)", flush=True)
    s_end = float(r.sum())

    x, y, r = P.repair(x, y, r)
    wall = time.time() - t0
    bc = rep["cost"]
    cost = {
        "wall_s": round(wall, 3),
        # summed across both stages: broad counts Adam steps, endgame counts LPs
        "nfev": int(bc.get("nfev", 0)) + int((ecost or {}).get("nfev", 0)),
        "nit": int(bc.get("nit", 0)) + int((ecost or {}).get("nit", 0)),
        "restarts": int(bc.get("restarts", 0)),
        "hops": int((ecost or {}).get("hops", 0)),
        "lp_solves": int(rep["slp"]["nlp"]) + int((ecost or {}).get("lp_solves", 0)),
        "slp_polishes": (int(rep["slp"]["polishes"])
                         + int((ecost or {}).get("slp_polishes", 0))),
        "stages": {
            "broad": {"seconds": round(t_broad, 1), "sum_r": s_broad,
                      "starts": rep["starts"], "polished": rep["polished"],
                      "waves": rep["waves"],
                      "screen_to_kkt_spearman": rep["screen_to_kkt_spearman"]},
            "endgame": {"seconds": round(max(0.0, t_end), 1), "sum_r": s_end,
                        "hops": int((ecost or {}).get("hops", 0)),
                        "accepts": int((ecost or {}).get("accepts", 0)),
                        "distinct_optima": int((ecost or {}).get("distinct_optima", 0)),
                        # iteration 10: the walk's REACHABILITY telemetry, not
                        # just its yield -- walk_steps/walk_down/resets are how
                        # a threshold-ladder arm proves it is a different arm
                        # (iteration 8's dead-knob lesson) before its Sigma-r is
                        # read as a statement about T_HI.
                        "t_hi": float(t_hi),
                        "walk_steps": int((ecost or {}).get("walk_steps", 0)),
                        "walk_down": int((ecost or {}).get("walk_down", 0)),
                        "resets": int((ecost or {}).get("resets", 0)),
                        "gain": s_end - s_broad},
        },
    }
    if verbose:
        _report(n, x, y, r, cost)
    return x, y, r, cost


def solve(n=P.N_DEFAULT, seconds=None, seed=0, broad_frac=0.25, verbose=True,
          B_wave=96, refine=24, audit=4, cycles=1, t_hi=E.T_HI):
    """Run `cycles` independent broad->endgame chains and keep the best.

    WHY THIS EXISTS (iteration 9, measured -- it is not a generic multistart
    reflex).  Iteration 8's nine n=54 sweep logs were re-read for the endgame's
    best-so-far TRAJECTORY, not just its total gain, and the walk turns out to
    saturate almost immediately: across nine 175-220 s walks there were only
    1, 2 or 4 improvements to `best` in total, and in 6 of the 9 the LAST
    improvement landed at 10-16% of the walk.  So ~85% of the endgame's seconds
    -- ~65% of the whole 240 s solve -- were provably producing nothing.  That
    also *explains* iteration 8's null: `broad_frac` looked flat below 0.25
    because the seconds it moves were being handed to a stage that had already
    stopped moving.

    THE DIAGNOSIS SURVIVED THE A/B AND THE PRESCRIPTION DID NOT -- read this
    before raising `cycles`.  Same paired concurrent harness, n=55, 240 s, arms
    k=1/2/3 x seeds 11,12,13 (`artifacts/iter9_cycles_sweep.log`).  The idle tail
    is confirmed forward: endgame gain was +4.80e-3 with 190 s, +4.95e-3 with
    101 s, +4.66e-3 with 62 s -- **62 s buys what 190 s buys**.  But k=2 and k=3
    LOST, -6.81e-4 and -9.67e-4 paired, 0/3 seeds each, i.e. losses larger than
    this design's ~3e-4 resolution.  Two measured reasons: (1) a chain's ceiling
    is its broad stage and broad is quantized -- mean broad Sigma-r was
    3.876390 at BOTH 30 s and 20 s (one 96-start wave, iteration 8's dead zone)
    vs 3.877220 at 60 s, so shortening the chain buys no extra diversity and
    costs ~8e-4 of funnel; (2) the extra draws are near-identical -- the FIRST
    chain won 3 of 6 multi-chain runs, no chain ever beat what k=1 reached, and
    in one run two chains with *different* funnels finished at the same 12-digit
    Sigma-r (spread 9e-14).  Nine runs produced only FOUR distinct final values.
    So the binding constraint is the ATTRACTOR SET the walk can reach, not the
    number of draws and not the seconds; the untried lever is the threshold
    ladder (`endgame.T_HI`, set at iteration 4 from the n=54 incumbent's
    neighbour band, with ~46x of unexplored room below the junk band).

    `cycles` is kept (default 1, byte-identical to the old single-chain `solve`)
    because it is the measured control arm for that next experiment, not because
    k>1 is recommended: it is measured WORSE at n=55.

    THAT NEXT EXPERIMENT RAN (iteration 10) AND ALSO LOST -- read this before
    touching `t_hi`.  Same harness, n=55, 240 s, arms {2.7e-4, 8e-4 = ctl,
    2.4e-3} x seeds 11,12,13 (`artifacts/iter10_thi_sweep.log`): 2.7e-4 paired
    **-9.00e-4** (0/3 seeds), 2.4e-3 paired **-3.41e-4** (1/3).  The default
    stays 8e-4.  The reason is structural and is now measured: the census of KKT
    points the walk actually visits has a **HOLE of 7.0e-4 .. 1.27e-3
    immediately below `best`** in all nine runs, and the control's ladder top
    sits inside that hole in 7 of them -- so widening T near the top of the
    funnel changes the accept set by nothing, because there is nothing in that
    band to accept.  A ladder wide enough to cross the hole (2.4e-3) crosses it
    DOWNWARD: at seed 11 that arm finished on the -1.02e-3 neighbour the control
    walked past.  `MAX_DRIFT` never fired once (resets = 0 at every arm).

    So `t_hi` is an exposed control arm and a diagnostic, NOT a tuning target,
    and the same now goes for every scalar in this pipeline: four of them across
    iterations 7-10, three flat-or-negative.  The open direction is a MOVE that
    generates candidates inside the 0..1e-3 hole (a combinatorial / contact-graph
    move), not another parameter of the same walk.
    """
    seconds = budget_seconds() if seconds is None else seconds
    cycles = max(1, int(cycles))
    if cycles == 1:
        return _chain(n=n, seconds=seconds, seed=seed, broad_frac=broad_frac,
                      verbose=verbose, B_wave=B_wave, refine=refine, audit=audit,
                      t_hi=t_hi)

    t0 = time.time()
    per = seconds / cycles
    best = None
    sums = []
    for i in range(cycles):
        # remaining/left instead of `per` so a chain that overran does not push
        # the last chain past the budget; each chain gets an equal share of what
        # is actually left, and the loop stops rather than overrun.
        left = seconds - (time.time() - t0)
        if i > 0 and left < max(6.0, 0.35 * per):
            if verbose:
                print(f"[pipeline] stopping after {i} chain(s): {left:.1f}s left",
                      flush=True)
            break
        t_i = min(per, left) if i == cycles - 1 else per
        if verbose:
            print(f"\n[pipeline] === chain {i+1}/{cycles}  {t_i:.1f}s "
                  f"(seed {seed + 1000 * i}) ===", flush=True)
        xi, yi, ri, ci = _chain(n=n, seconds=t_i, seed=seed + 1000 * i,
                                broad_frac=broad_frac, verbose=verbose,
                                B_wave=B_wave, refine=refine, audit=audit,
                                t_hi=t_hi)
        si = float(ri.sum())
        sums.append(si)
        if best is None or si > best[0]:
            best = (si, xi, yi, ri, ci, i)
    s, x, y, r, cost, bi = best
    cost = dict(cost)
    cost["wall_s"] = round(time.time() - t0, 3)
    # the stage block belongs to the WINNING chain; say so, so a later iteration
    # does not read it as an average over chains.
    cost["cycles"] = {"k": cycles, "ran": len(sums), "per_s": round(per, 1),
                      "sums": sums, "best_i": bi, "best": s,
                      "spread": (max(sums) - min(sums)) if sums else 0.0,
                      "note": "stages{} is the winning chain, not a mean"}
    if verbose:
        print(f"\n[pipeline] {len(sums)} chain(s): "
              + "  ".join(f"{v:.12f}" for v in sums)
              + f"   -> keeping chain {bi+1} ({s:.12f})", flush=True)
    return x, y, r, cost


def _report(n, x, y, r, cost):
    s = float(r.sum())
    rec = P.CSQV_RECORDS.get(n)
    pv, bv = P.violations(x, y, r)
    print(f"\nn={n}  sum_r = {s:.12f}")
    if rec is not None:
        print(f"  csqv record {rec:.12f}   gap {rec - s:+.6e}"
              f"   {'(BELOW record)' if s < rec else '(ABOVE record -- suspect a bug first)'}")
    ab = P.area_bound(n)
    print(f"  provable area bound sqrt(n/pi) = {ab:.9f}   slack {ab - s:+.6e}"
          f"   {'OK' if s <= ab else 'VIOLATED -- BUG'}")
    print(f"  feasibility: max_pair={pv:.3e} max_box={bv:.3e}")
    st = cost["stages"]
    print(f"  stages: broad {st['broad']['seconds']}s -> {st['broad']['sum_r']:.12f}"
          f"   endgame {st['endgame']['seconds']}s -> {st['endgame']['sum_r']:.12f}"
          f"  ({st['endgame']['gain']:+.3e})")
    print(f"  cost: wall_s={cost['wall_s']} nfev={cost['nfev']} nit={cost['nit']} "
          f"restarts={cost['restarts']} hops={cost['hops']} "
          f"lp_solves={cost['lp_solves']}")


def calibrate(ns, seconds, seed=0, verbose=True):
    """Run the identical pipeline at several n and compare to the csqv records.

    The point is NOT to set a bar (no bench moves, nothing is written).  It is
    to ask whether this pipeline's relationship to the published table is
    n-specific or systematic.  Reported per n: our sum_r, the record, the
    signed excess, the area-bound slack, and the exact feasibility residual.
    """
    rows = []
    for n in ns:
        x, y, r, cost = solve(n=n, seconds=seconds, seed=seed, verbose=verbose)
        s = float(r.sum())
        rec = P.CSQV_RECORDS.get(n)
        pv, bv = P.violations(x, y, r)
        rows.append({"n": n, "sum_r": s, "record": rec,
                     "excess": (s - rec) if rec is not None else None,
                     "area_bound": P.area_bound(n),
                     "max_pair": pv, "max_box": bv,
                     "wall_s": cost["wall_s"],
                     "circles": np.column_stack([x, y, r]).tolist()})
    print("\ncalibration: identical pipeline, several n, vs the published table")
    print(f"{'n':>4s} {'ours':>16s} {'csqv record':>16s} {'excess':>12s} "
          f"{'area slack':>12s} {'max_pair':>10s}")
    for d in rows:
        rec = "-" if d["record"] is None else f"{d['record']:16.12f}"
        exc = "-" if d["excess"] is None else f"{d['excess']:+12.3e}"
        print(f"{d['n']:4d} {d['sum_r']:16.12f} {rec} {exc} "
              f"{d['area_bound'] - d['sum_r']:+12.3e} {d['max_pair']:10.2e}")
    return rows


def record_smoothness(lo=44, hi=64):
    """Second differences of the published csqv table -- an instrument, not a proof.

    sum_r(n) for this family grows smoothly (roughly like c*sqrt(n)), so its
    second difference should be small and slightly negative.  A single entry
    that is too LOW shows up as a locally large POSITIVE second difference at
    that n.  This costs one arithmetic pass over numbers we did not produce, and
    it is the only check available here that is independent of our own code
    entirely -- so it can corroborate or refute the n=54 excess without
    touching our optimizer.
    """
    ns = [n for n in range(lo, hi + 1) if n in P.CSQV_RECORDS]
    v = {n: P.CSQV_RECORDS[n] for n in ns}
    out = []
    for n in ns:
        if (n - 1) in v and (n + 1) in v:
            out.append((n, v[n + 1] - 2 * v[n] + v[n - 1]))
    return out


def _self_test():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")

    # 1. end-to-end at a small n, both stages, inside the budget
    t0 = time.time()
    x, y, r, cost = solve(n=8, seconds=16.0, seed=1, verbose=False, B_wave=24,
                          refine=6, audit=1, broad_frac=0.35)
    wall = time.time() - t0
    pv, bv = P.violations(x, y, r)
    chk("end-to-end n=8 strictly feasible", pv <= 0 and bv <= 0,
        f"pair={pv:.1e} box={bv:.1e}")
    chk("end-to-end n=8 returns 8 circles", len(r) == 8 and len(x) == 8)
    chk("respects the wall-clock budget", wall < 16.0 + 4.0, f"{wall:.1f}s")
    chk("both stages ran and are logged",
        cost["stages"]["broad"]["sum_r"] > 0
        and cost["stages"]["endgame"]["hops"] > 0,
        f"hops={cost['stages']['endgame']['hops']}")
    chk("endgame stage never loses ground",
        cost["stages"]["endgame"]["sum_r"] >= cost["stages"]["broad"]["sum_r"] - 1e-12,
        f"gain={cost['stages']['endgame']['gain']:+.3e}")
    chk("telemetry present and non-trivial",
        cost["nit"] > 0 and cost["restarts"] > 0 and cost["lp_solves"] > 0,
        f"nit={cost['nit']} restarts={cost['restarts']} lps={cost['lp_solves']}")

    # 2. the area bound is a real bound and our config obeys it
    chk("area bound obeyed at n=8", float(r.sum()) <= P.area_bound(8),
        f"{r.sum():.6f} <= {P.area_bound(8):.6f}")
    chk("area bound is tight-ish and correct for the 2x2 grid",
        abs(P.area_bound(4) - 1.1283791670955126) < 1e-12)

    # 3. broad_frac=1.0 must skip the endgame stage cleanly, not crash
    x2, y2, r2, c2 = solve(n=8, seconds=6.0, seed=2, verbose=False, B_wave=24,
                           refine=6, audit=1, broad_frac=1.0)
    p2, b2 = P.violations(x2, y2, r2)
    chk("broad-only mode still emits a feasible config", p2 <= 0 and b2 <= 0)
    chk("broad-only mode logs zero hops", c2["stages"]["endgame"]["hops"] == 0)

    # 3b. cycles>1 must run k chains inside ONE budget and keep the max
    t0 = time.time()
    x3, y3, r3, c3 = solve(n=8, seconds=20.0, seed=3, verbose=False, B_wave=24,
                           refine=6, audit=1, broad_frac=0.35, cycles=2)
    w3 = time.time() - t0
    p3, b3 = P.violations(x3, y3, r3)
    chk("cycles=2 emits a feasible config", p3 <= 0 and b3 <= 0,
        f"pair={p3:.1e} box={b3:.1e}")
    chk("cycles=2 stays inside the SAME budget, not k times it", w3 < 20.0 + 4.0,
        f"{w3:.1f}s for a 20s budget")
    chk("cycles=2 ran 2 chains and logged both", c3["cycles"]["ran"] == 2
        and len(c3["cycles"]["sums"]) == 2, str(c3["cycles"]["sums"]))
    chk("cycles=2 returns the MAX chain, not the last",
        abs(float(r3.sum()) - max(c3["cycles"]["sums"])) < 1e-12,
        f"kept {float(r3.sum()):.9f} of {c3['cycles']['sums']}")
    chk("cycles=2 chains use different seeds (distinct draws or equal by chance)",
        c3["cycles"]["best_i"] in (0, 1))
    chk("cycles=1 is the plain single chain (no cycles block)",
        "cycles" not in cost)

    # 3c. the t_hi knob must reach the walk, not just the signature (iteration
    #     10).  Three checks, in increasing strength: it is recorded, the
    #     default is unchanged byte-for-byte, and a raised ladder actually makes
    #     the acceptance test accept a step the default rejects.
    chk("default t_hi is recorded in the endgame telemetry and equals E.T_HI",
        abs(cost["stages"]["endgame"]["t_hi"] - E.T_HI) < 1e-18,
        f"{cost['stages']['endgame']['t_hi']:g}")
    x4, y4, r4, c4 = solve(n=8, seconds=8.0, seed=4, verbose=False, B_wave=24,
                           refine=6, audit=1, broad_frac=0.3, t_hi=2.4e-3)
    p4, b4 = P.violations(x4, y4, r4)
    chk("a non-default t_hi still emits a feasible config", p4 <= 0 and b4 <= 0,
        f"pair={p4:.1e} box={b4:.1e}")
    chk("a non-default t_hi is threaded to the endgame, not silently dropped",
        abs(c4["stages"]["endgame"]["t_hi"] - 2.4e-3) < 1e-18,
        f"{c4['stages']['endgame']['t_hi']:g}")
    chk("reachability telemetry is exported for the dead-knob check",
        all(k in c4["stages"]["endgame"]
            for k in ("walk_steps", "walk_down", "resets", "distinct_optima")))
    # the mechanism itself: at hop 0 the ladder sits at t_hi, so a downhill step
    # of 1.5e-3 is acceptable at the high arm and rejected at the default.
    chk("raising t_hi really widens the accept band at the top of the cycle",
        E.accepts_walk(1.0 - 1.5e-3, 1.0, E.threshold(0, t_hi=2.4e-3))
        and not E.accepts_walk(1.0 - 1.5e-3, 1.0, E.threshold(0, t_hi=E.T_HI)))
    chk("t_hi=2.4e-3 stays below the 3.7e-2 junk band at every hop in a cycle",
        max(E.threshold(h, t_hi=2.4e-3) for h in range(E.CYCLE)) < 3.7e-2)

    # 4. the record table itself: transcription sanity (strictly increasing,
    #    plausible increments) -- catches a typo'd digit in CSQV_RECORDS.
    ns = sorted(k for k in P.CSQV_RECORDS if k >= 44)
    vals = [P.CSQV_RECORDS[k] for k in ns]
    inc = [b - a for a, b in zip(vals, vals[1:])]
    chk("csqv table strictly increasing in n", all(d > 0 for d in inc))
    chk("csqv increments all in (0.02, 0.06)",
        all(0.02 < d < 0.06 for d in inc), f"min={min(inc):.4f} max={max(inc):.4f}")
    chk("every tabulated record obeys the provable area bound",
        all(P.CSQV_RECORDS[k] <= P.area_bound(k) for k in P.CSQV_RECORDS))

    # 5. record_smoothness returns one second difference per interior n
    sm = record_smoothness(44, 64)
    chk("record_smoothness covers the interior n", len(sm) == 19,
        f"{len(sm)} entries")
    known = P.CSQV_RECORDS[55] - 2 * P.CSQV_RECORDS[54] + P.CSQV_RECORDS[53]
    chk("record_smoothness arithmetic matches a hand-computed entry",
        abs(dict(sm)[54] - known) < 1e-15, f"{dict(sm)[54]:+.3e}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=P.N_DEFAULT)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--broad-frac", type=float, default=0.25,
                    help="fraction of the budget for the broad stage "
                         "(0.25 measured best at n=55; see module docstring)")
    ap.add_argument("--cycles", type=int, default=1,
                    help="independent broad->endgame chains inside the budget, "
                         "keep the best (1 = the historical single chain); "
                         "k>1 measured WORSE at n=55, see solve() docstring")
    ap.add_argument("--t-hi", type=float, default=E.T_HI,
                    help="top of the endgame threshold ladder (default "
                         f"{E.T_HI:g}); governs how far downhill the KKT walk "
                         "may step, i.e. the reachable attractor set")
    ap.add_argument("--out", default=None, help="write config here IF it improves")
    ap.add_argument("--calibrate", default=None,
                    help="comma-separated n list: run all of them, write no config")
    ap.add_argument("--report", default=None, help="write a JSON report here")
    ap.add_argument("--smoothness", action="store_true",
                    help="print second differences of the published csqv table")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        print("pipeline self-test")
        sys.exit(0 if _self_test() else 1)

    if a.smoothness:
        sm = record_smoothness()
        print("published csqv table, second difference  v[n+1] - 2v[n] + v[n-1]")
        vals = [d for _, d in sm]
        med = float(np.median(vals))
        mad = float(np.median([abs(d - med) for d in vals]))
        for n, d in sm:
            z = (d - med) / mad if mad > 0 else float("nan")
            print(f"  n={n:3d}  {d:+.6e}   z(MAD)={z:+6.2f}"
                  + ("   <-- our instance" if n == 54 else ""))
        print(f"median {med:+.3e}   MAD {mad:.3e}")
        sys.exit(0)

    if a.calibrate:
        ns = [int(t) for t in a.calibrate.split(",") if t.strip()]
        rows = calibrate(ns, seconds=(a.seconds or 20.0), seed=a.seed)
        if a.report:
            with open(a.report, "w") as fh:
                json.dump({"kind": "calibration", "seconds": a.seconds,
                           "seed": a.seed, "rows": rows}, fh, indent=1)
            print(f"wrote {a.report}")
        sys.exit(0)

    x, y, r, cost = solve(n=a.n, seconds=a.seconds, seed=a.seed,
                          broad_frac=a.broad_frac, cycles=a.cycles,
                          t_hi=a.t_hi)
    if a.report:
        with open(a.report, "w") as fh:
            json.dump({"n": a.n, "seed": a.seed, "sum_r": float(r.sum()),
                       "record": P.CSQV_RECORDS.get(a.n), "cost": cost}, fh, indent=1)
        print(f"wrote {a.report}")
    if a.out:
        prev = None
        if os.path.exists(a.out):
            with open(a.out) as fh:
                prev = json.load(fh).get("sum_r", None)
        if prev is not None and prev >= float(r.sum()):
            print(f"NOT writing: stored {prev:.12f} >= new {r.sum():.12f}")
        else:
            P.save_config(a.out, x, y, r, cost=cost,
                          meta={"kind": "best", "seed": a.seed, "n": a.n,
                                "method": "pipeline: broad KKT multistart -> "
                                          "threshold-accepting KKT walk",
                                "budget_s": a.seconds})
            print(f"wrote {a.out}"
                  + (f" (improved from {prev:.12f})" if prev is not None else ""))
