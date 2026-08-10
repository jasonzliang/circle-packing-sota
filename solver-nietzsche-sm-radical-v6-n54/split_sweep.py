#!/usr/bin/env python3
"""Paired, concurrent A/B harness over `pipeline.solve` knobs.

Iteration 6 measured the two pipeline stages separately at n=55 and found the
endgame stage returned ~8.7x per second what the tail of the broad stage did.
That is a *hint about an allocation*, not a measurement of one: the stages are
not independent (endgame can only squeeze the funnel broad hands it), so the
only honest way to price the split is to run the whole pipeline end-to-end at
several splits and compare final Sigma-r.

Why this file exists rather than a shell loop:

  * **Paired.** One run per (seed, arm).  Every seed is run at EVERY arm, so
    the comparison is a within-seed difference, not a difference of two noisy
    means.  At n=55 the seed-to-seed spread of a cold pipeline run is of the
    same order as the effect being measured, so an unpaired sweep of this size
    could not see the effect at all.
  * **Concurrent under equal contention.** All arms launch together with
    `OMP_NUM_THREADS=1`, so every run is single-threaded and every run sees the
    same machine load.  Absolute Sigma-r from a sweep is therefore NOT
    comparable to a solo run -- only arm-vs-arm within one sweep is.  That
    caveat is printed with the results so a later iteration cannot forget it.
  * **Every run keeps its config.** Each arm writes a full config through
    `pipeline.py --out`, so a winning arm is a stored, re-checkable packing and
    not just a number in a log.

The arms are `--broad-frac` values by default but the harness takes any
`pipeline.py` flag, so a later iteration can A/B a different knob without
rewriting this.

    python3 tools/split_sweep.py --n 55 --seconds 240 --seeds 11,12,13 \
        --arms 0.25,0.40,0.60 --outdir artifacts/iter7_sweep
    python3 tools/split_sweep.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PIPELINE = os.path.join(HERE, "pipeline.py")


def arm_label(flag, value):
    return f"{flag.lstrip('-')}={value}"


def launch(n, seconds, seed, flag, value, outdir, extra=()):
    """Start one pipeline run as a single-threaded subprocess.  Non-blocking."""
    tag = f"n{n}_{arm_label(flag, value).replace('=', '')}_s{seed}"
    cfg = os.path.join(outdir, f"cfg_{tag}.json")
    rep = os.path.join(outdir, f"rep_{tag}.json")
    log = open(os.path.join(outdir, f"log_{tag}.txt"), "w")
    env = dict(os.environ)
    # keep every run single-threaded so N concurrent runs do not oversubscribe
    # and so the arms see identical machine load
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = "1"
    env.pop("CP_SOLVE_SECONDS", None)  # --seconds is explicit; do not let env win
    # arms are parsed as float, but integer-valued knobs (--cycles) are
    # argparse type=int and would reject "2.0"; emit the integral form.
    sval = str(int(value)) if float(value).is_integer() else str(value)
    cmd = [sys.executable, PIPELINE, "--n", str(n), "--seconds", str(seconds),
           "--seed", str(seed), flag, sval, "--out", cfg, "--report", rep,
           *extra]
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    return {"proc": p, "log": log, "seed": seed, "value": value, "cfg": cfg,
            "rep": rep, "tag": tag}


def collect(job):
    """Wait for one job and read back its Sigma-r + stage telemetry."""
    rc = job["proc"].wait()
    job["log"].close()
    out = {"seed": job["seed"], "value": job["value"], "rc": rc,
           "cfg": job["cfg"], "sum_r": None}
    if os.path.exists(job["rep"]):
        with open(job["rep"]) as fh:
            d = json.load(fh)
        st = d.get("cost", {}).get("stages", {})
        out.update({
            "sum_r": d.get("sum_r"),
            "record": d.get("record"),
            "wall_s": d.get("cost", {}).get("wall_s"),
            "broad_sum_r": st.get("broad", {}).get("sum_r"),
            "broad_s": st.get("broad", {}).get("seconds"),
            "broad_starts": st.get("broad", {}).get("starts"),
            "endgame_gain": st.get("endgame", {}).get("gain"),
            "endgame_s": st.get("endgame", {}).get("seconds"),
            "hops": st.get("endgame", {}).get("hops"),
            "accepts": st.get("endgame", {}).get("accepts"),
            # iteration 10: reachability, not yield.  An arm that changes a
            # search knob has to PROVE it changed the search before its Sigma-r
            # is read as a statement about that knob -- iteration 8 A/B'd three
            # broad_frac arms that were, on 2 of 3 seeds, byte-identical runs.
            "distinct_optima": st.get("endgame", {}).get("distinct_optima"),
            "walk_steps": st.get("endgame", {}).get("walk_steps"),
            "walk_down": st.get("endgame", {}).get("walk_down"),
            "resets": st.get("endgame", {}).get("resets"),
        })
    return out


def aggregate(rows, arms, control):
    """Per-arm summary plus the paired difference against the control arm.

    Paired difference = mean over seeds of (arm - control) on the SAME seed,
    reported with the per-seed spread and the win count.  With 3 seeds no
    p-value is meaningful, so none is printed: the honest summary is the
    per-seed differences themselves plus how many of them are positive.
    """
    by = {a: {r["seed"]: r for r in rows if r["value"] == a} for a in arms}
    seeds = sorted({r["seed"] for r in rows})
    summ = []
    for a in arms:
        vals = [by[a][s]["sum_r"] for s in seeds
                if s in by[a] and by[a][s]["sum_r"] is not None]
        paired = [(by[a][s]["sum_r"] - by[control][s]["sum_r"])
                  for s in seeds
                  if s in by[a] and s in by[control]
                  and by[a][s]["sum_r"] is not None
                  and by[control][s]["sum_r"] is not None]
        summ.append({
            "arm": a,
            "n_runs": len(vals),
            "best": max(vals) if vals else None,
            "mean": float(np.mean(vals)) if vals else None,
            "median": float(np.median(vals)) if vals else None,
            "paired_mean_vs_control": float(np.mean(paired)) if paired else None,
            "paired_min": float(np.min(paired)) if paired else None,
            "paired_max": float(np.max(paired)) if paired else None,
            "wins_vs_control": int(sum(1 for d in paired if d > 0)),
            "paired_n": len(paired),
        })
    return seeds, summ


def print_table(n, seconds, rows, arms, control, seeds, summ):
    rec = None
    for r in rows:
        rec = r.get("record") or rec
    print(f"\nsplit sweep: n={n}  {seconds:.0f}s/run  "
          f"{len(arms)} arms x {len(seeds)} seeds = {len(rows)} runs, all concurrent")
    print("  CAVEAT: concurrent + OMP_NUM_THREADS=1, so absolute sum_r here is "
          "NOT comparable to a solo run; only arm-vs-arm within this sweep is.")
    hdr = "  " + f"{'seed':>6s}" + "".join(f"{('f=' + str(a)):>18s}" for a in arms)
    print(hdr)
    for s in seeds:
        cells = []
        for a in arms:
            v = next((r["sum_r"] for r in rows
                      if r["value"] == a and r["seed"] == s), None)
            cells.append("             -    " if v is None else f"{v:18.12f}")
        print(f"  {s:6d}" + "".join(cells))
    print(f"\n  {'arm':>8s} {'best':>16s} {'mean':>16s} {'median':>16s} "
          f"{'paired vs ctl':>14s} {'wins':>6s}")
    for d in summ:
        pm = "-" if d["paired_mean_vs_control"] is None else f"{d['paired_mean_vs_control']:+14.3e}"
        print(f"  {str(d['arm']):>8s} {d['best']:16.12f} {d['mean']:16.12f} "
              f"{d['median']:16.12f} {pm} {d['wins_vs_control']:>3d}/{d['paired_n']:<2d}"
              + ("   <-- control" if d["arm"] == control else ""))
    if rec:
        b = max(d["best"] for d in summ if d["best"] is not None)
        print(f"\n  best of sweep {b:.12f}   csqv record {rec:.12f}   gap {rec - b:+.6e}")
    # stage telemetry: what each arm actually bought with its seconds
    print(f"\n  {'arm':>8s} {'broad_s':>8s} {'broad sum_r':>16s} {'starts':>7s} "
          f"{'end_s':>7s} {'endgame gain':>14s} {'hops':>6s} {'acc':>4s}")
    for a in arms:
        rs = [r for r in rows if r["value"] == a and r.get("broad_sum_r")]
        if not rs:
            continue
        print(f"  {str(a):>8s} {np.mean([r['broad_s'] for r in rs]):8.1f} "
              f"{np.mean([r['broad_sum_r'] for r in rs]):16.12f} "
              f"{np.mean([r['broad_starts'] for r in rs]):7.0f} "
              f"{np.mean([r['endgame_s'] for r in rs]):7.1f} "
              f"{np.mean([r['endgame_gain'] for r in rs]):+14.3e} "
              f"{np.mean([r['hops'] for r in rs]):6.0f} "
              f"{np.mean([r['accepts'] for r in rs]):4.0f}")
    # DEAD-KNOB CHECK: did the arms actually search differently?  If these
    # columns are flat across arms the Sigma-r column is not evidence about the
    # knob -- it is evidence the knob did nothing (iteration 8).
    reach = [r for r in rows if r.get("walk_steps") is not None]
    if reach:
        print(f"\n  reachability (dead-knob check -- these MUST differ across arms)")
        print(f"  {'arm':>8s} {'walk':>7s} {'wlk_dn':>7s} {'optima':>7s} "
              f"{'resets':>7s} {'walk/hop':>9s}")
        for a in arms:
            rs = [r for r in reach if r["value"] == a]
            if not rs:
                continue
            hp = np.mean([r["hops"] for r in rs]) or 1.0
            print(f"  {str(a):>8s} {np.mean([r['walk_steps'] for r in rs]):7.1f} "
                  f"{np.mean([r['walk_down'] for r in rs]):7.1f} "
                  f"{np.mean([r['distinct_optima'] for r in rs]):7.1f} "
                  f"{np.mean([r['resets'] for r in rs]):7.1f} "
                  f"{np.mean([r['walk_steps'] for r in rs]) / hp:8.1%}")


def sweep(n, seconds, seeds, arms, outdir, flag="--broad-frac", control=None,
          extra=()):
    os.makedirs(outdir, exist_ok=True)
    control = arms[-1] if control is None else control
    t0 = time.time()
    jobs = [launch(n, seconds, s, flag, a, outdir, extra)
            for a in arms for s in seeds]
    print(f"launched {len(jobs)} concurrent runs ({n=}, {seconds}s each); "
          f"expect ~{seconds + 20:.0f}s wall", flush=True)
    rows = [collect(j) for j in jobs]
    wall = time.time() - t0
    bad = [r for r in rows if r["rc"] != 0 or r["sum_r"] is None]
    if bad:
        print(f"  WARNING: {len(bad)} run(s) failed: "
              f"{[(r['seed'], r['value'], r['rc']) for r in bad]}")
    seeds_seen, summ = aggregate(rows, arms, control)
    print_table(n, seconds, rows, arms, control, seeds_seen, summ)
    print(f"\n  sweep wall {wall:.1f}s")
    return {"n": n, "seconds": seconds, "flag": flag, "arms": arms,
            "control": control, "seeds": seeds, "wall_s": round(wall, 1),
            "rows": rows, "summary": summ}


def _self_test():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")

    # 1. aggregation arithmetic on hand-computed synthetic data.  Arm 0.3 beats
    #    the control on every seed by exactly +0.01; arm 0.9 loses on 2 of 3.
    syn = [
        {"seed": 1, "value": 0.3, "sum_r": 1.01, "rc": 0},
        {"seed": 2, "value": 0.3, "sum_r": 2.01, "rc": 0},
        {"seed": 3, "value": 0.3, "sum_r": 3.01, "rc": 0},
        {"seed": 1, "value": 0.6, "sum_r": 1.00, "rc": 0},
        {"seed": 2, "value": 0.6, "sum_r": 2.00, "rc": 0},
        {"seed": 3, "value": 0.6, "sum_r": 3.00, "rc": 0},
        {"seed": 1, "value": 0.9, "sum_r": 1.05, "rc": 0},
        {"seed": 2, "value": 0.9, "sum_r": 1.90, "rc": 0},
        {"seed": 3, "value": 0.9, "sum_r": 2.90, "rc": 0},
    ]
    seeds, summ = aggregate(syn, [0.3, 0.6, 0.9], 0.6)
    d = {s["arm"]: s for s in summ}
    chk("paired mean is exact on synthetic data",
        abs(d[0.3]["paired_mean_vs_control"] - 0.01) < 1e-12,
        f"{d[0.3]['paired_mean_vs_control']:+.4f}")
    chk("control arm pairs to exactly zero",
        d[0.6]["paired_mean_vs_control"] == 0.0)
    chk("win count is per-seed, not per-mean", d[0.9]["wins_vs_control"] == 1,
        f"wins={d[0.9]['wins_vs_control']}")
    chk("best is the per-arm max, not the last or the paired winner",
        d[0.9]["best"] == 2.90 and d[0.3]["best"] == 3.01,
        f"{d[0.9]['best']} {d[0.3]['best']}")
    chk("an arm that loses on pairing can still win on one seed",
        d[0.9]["wins_vs_control"] == 1 and d[0.9]["paired_max"] > 0
        and d[0.9]["paired_mean_vs_control"] < 0)
    chk("paired spread reported", abs(d[0.9]["paired_min"] + 0.10) < 1e-12
        and abs(d[0.9]["paired_max"] - 0.05) < 1e-12)
    # mean of arm 0.9 (1.95) is LOWER than control (2.0) AND paired says lose:
    # the point of pairing is that these can disagree, so check both exist
    chk("unpaired mean also computed",
        abs(d[0.9]["mean"] - 1.95) < 1e-12 and abs(d[0.6]["mean"] - 2.0) < 1e-12)

    # 2. missing runs must not crash the aggregation or fake a pair
    seeds2, summ2 = aggregate(syn[:-1], [0.3, 0.6, 0.9], 0.6)
    d2 = {s["arm"]: s for s in summ2}
    chk("a dead run drops its pair rather than pairing wrongly",
        d2[0.9]["paired_n"] == 2 and d2[0.9]["n_runs"] == 2)

    # 3. a real end-to-end sweep at tiny n: 2 arms x 2 seeds, concurrent
    outdir = os.path.join(ROOT, "artifacts", "_selftest_sweep")
    t0 = time.time()
    res = sweep(n=6, seconds=8.0, seeds=[1, 2], arms=[0.4, 0.7], outdir=outdir)
    wall = time.time() - t0
    chk("all 4 subprocess runs exited 0",
        all(r["rc"] == 0 for r in res["rows"]),
        str([r["rc"] for r in res["rows"]]))
    chk("every run returned a sum_r",
        all(r["sum_r"] and r["sum_r"] > 0 for r in res["rows"]))
    chk("concurrency is real (4 x 8s runs in well under 32s)", wall < 26.0,
        f"{wall:.1f}s")
    chk("every arm honoured its requested split",
        all(abs(r["broad_s"] - r["value"] * 8.0) < 0.5 for r in res["rows"]),
        str([(r["value"], r["broad_s"]) for r in res["rows"]]))

    # 4. the winning config is a real stored packing, feasible at zero tolerance
    best = max(res["rows"], key=lambda r: r["sum_r"])
    with open(best["cfg"]) as fh:
        cfg = json.load(fh)
    a = np.asarray(cfg["circles"], float)
    x, y, r = a[:, 0], a[:, 1], a[:, 2]
    dd = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(dd, np.inf)
    pair = float(np.max(r[:, None] + r[None, :] - dd))
    box = float(np.max(r - np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y))))
    chk("stored winner has exactly n circles", a.shape == (6, 3))
    chk("stored winner strictly feasible at zero tolerance", pair <= 0 and box <= 0,
        f"pair={pair:.2e} box={box:.2e}")
    chk("stored sum_r matches the reported sum_r",
        abs(float(r.sum()) - best["sum_r"]) < 1e-12)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=55)
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--seeds", default="11,12,13")
    ap.add_argument("--arms", default="0.25,0.40,0.60")
    ap.add_argument("--flag", default="--broad-frac")
    ap.add_argument("--control", type=float, default=None)
    ap.add_argument("--outdir", default="artifacts/sweep")
    ap.add_argument("--report", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        print("split_sweep self-test")
        sys.exit(0 if _self_test() else 1)

    res = sweep(n=a.n, seconds=a.seconds,
                seeds=[int(t) for t in a.seeds.split(",") if t.strip()],
                arms=[float(t) for t in a.arms.split(",") if t.strip()],
                outdir=a.outdir, flag=a.flag, control=a.control)
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"wrote {a.report}")
