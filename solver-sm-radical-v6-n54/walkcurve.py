#!/usr/bin/env python3
"""When does the endgame walk stop paying?  Reads it off the logs we already have.

WHY THIS EXISTS.  Through iteration 8 the endgame stage was summarised by ONE
number -- its total gain, `stages.endgame.gain`.  That number cannot distinguish
"improved steadily for 200 s" from "improved twice in the first 25 s and then
sat idle for 175 s", and those two worlds call for opposite changes.  Iteration 8
had already written nine 240 s n=54 runs to `artifacts/iter8_sweep/log_*.txt`,
and every improvement to `best` is printed there with its hop index and its
timestamp.  So the saturation curve cost zero new compute: it was sitting in
logs that had been read only for their last line.

What it found (see notes/insights.md, iteration 9): across nine 175-220 s walks
there were 1, 2 or 4 improvements TOTAL, and in 6 of 9 the last one landed at
10-16% of the walk.  ~85% of the endgame's seconds produced nothing -- which is
also why iteration 8's `broad_frac` sweep read as flat below 0.25: the seconds
that knob moves were being handed to a stage that had already stopped moving.

This is deliberately a PARSER of stdout, not a new instrumentation hook: it can
be pointed at logs written by earlier iterations, including ones written before
it existed, so it can audit the past rather than only the future.

    python3 tools/walkcurve.py artifacts/iter8_sweep/log_*.txt
    python3 tools/walkcurve.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import re
import sys

import numpy as np

# "[hop  1234   45.6s] jitter/s0.0123 3.8300 -> 3.8310 (+1.0e-03)"
RE_IMPROVE = re.compile(
    r"^\[hop\s+(\d+)\s+([0-9.]+)s\]\s+\S+\s+([0-9.]+)\s*->\s*([0-9.]+)")
# "[hop   100   12.3s] best=3.8300 cur=... T=... "
RE_PROGRESS = re.compile(r"^\[hop\s+(\d+)\s+([0-9.]+)s\]\s+best=([0-9.]+)")
# "[seed    1.2s] incumbent KKT = 3.8290   accept=threshold"
RE_KKT = re.compile(r"incumbent KKT\s*=\s*([0-9.]+)")


def parse_walk(lines):
    """Return {start, improvements: [(hop, t, sum_r)], t_last_seen, hops_seen}.

    `improvements` are the strict rises of `best`; `t_last_seen` is the last
    timestamp of ANY hop line, i.e. how long the walk actually ran.  Both come
    from the same log, so the ratio t_last_improve / t_last_seen is the number
    that matters: the fraction of the walk that was still buying something.
    """
    start, imp, t_last, hops = None, [], 0.0, 0
    for line in lines:
        m = RE_KKT.search(line)
        if m and start is None:
            start = float(m.group(1))
        m = RE_PROGRESS.match(line)
        if m:
            t_last = max(t_last, float(m.group(2)))
            hops = max(hops, int(m.group(1)))
            continue
        m = RE_IMPROVE.match(line)
        if m:
            hop, t, new = int(m.group(1)), float(m.group(2)), float(m.group(4))
            t_last = max(t_last, t)
            hops = max(hops, hop)
            imp.append((hop, t, new))
    return {"start": start, "improvements": imp, "t_last_seen": t_last,
            "hops_seen": hops}


def curve(w, fracs=(0.25, 0.5, 0.75)):
    """Fraction of the walk's TOTAL gain that had been banked by t = f * T."""
    s0, imp, T = w["start"], w["improvements"], w["t_last_seen"]
    if s0 is None or not imp or T <= 0:
        return None
    total = imp[-1][2] - s0
    out = {"gain": total, "n_improve": len(imp), "T": T,
           "t_last_improve": imp[-1][1], "frac_last": imp[-1][1] / T,
           "hops": w["hops_seen"]}
    for f in fracs:
        got = max([e[2] for e in imp if e[1] <= f * T], default=s0) - s0
        out[f"f{int(f*100)}"] = (got / total) if total > 0 else float("nan")
    return out


def report(paths, idle_frac=0.5):
    rows = []
    for p in paths:
        with open(p) as fh:
            c = curve(parse_walk(fh.read().splitlines()))
        if c:
            c["path"] = p
            rows.append(c)
    if not rows:
        print("no parseable endgame walks found")
        return rows
    print(f"{'log':>42s} {'walk_s':>7s} {'gain':>10s} {'#imp':>5s} "
          f"{'f25':>5s} {'f50':>5s} {'f75':>5s} {'last@':>6s} {'idle_s':>7s}")
    for c in rows:
        print(f"{c['path'][-42:]:>42s} {c['T']:7.1f} {c['gain']:10.3e} "
              f"{c['n_improve']:5d} {c['f25']:5.2f} {c['f50']:5.2f} "
              f"{c['f75']:5.2f} {c['frac_last']:6.2f} "
              f"{c['T'] - c['t_last_improve']:7.1f}")
    fl = np.array([c["frac_last"] for c in rows])
    idle = np.array([c["T"] - c["t_last_improve"] for c in rows])
    print(f"\n{len(rows)} walks: last improvement at median "
          f"{np.median(fl)*100:.0f}% of the walk "
          f"({int((fl <= idle_frac).sum())}/{len(rows)} at or before "
          f"{idle_frac*100:.0f}%)")
    print(f"  idle tail after the last improvement: median {np.median(idle):.0f}s, "
          f"total {idle.sum():.0f}s across all walks")
    print(f"  improvements per walk: min {min(c['n_improve'] for c in rows)}, "
          f"max {max(c['n_improve'] for c in rows)}, "
          f"median {np.median([c['n_improve'] for c in rows]):.0f}")
    return rows


SYNTH = """
[seed    1.0s] incumbent KKT = 3.000000000000   accept=threshold
[hop     5   10.0s] jitter/s0.010 3.000000000000 -> 3.100000000000 (+1.0e-01)
[hop   100   20.0s] best=3.100000000000 cur=3.099 T=1e-4 gap=1e-3
[hop   150   25.0s] recluster/k3 3.100000000000 -> 3.400000000000 (+3.0e-01)
[hop   200   40.0s] best=3.400000000000 cur=3.399 T=1e-4 gap=1e-3
[hop   900  100.0s] best=3.400000000000 cur=3.399 T=1e-4 gap=1e-3
""".strip().splitlines()


def _self_test():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")

    w = parse_walk(SYNTH)
    chk("start read from the incumbent-KKT line", w["start"] == 3.0)
    chk("both improvements parsed, progress lines not counted as improvements",
        len(w["improvements"]) == 2, str(w["improvements"]))
    chk("walk length is the last hop line of ANY kind",
        w["t_last_seen"] == 100.0, f"{w['t_last_seen']}")
    chk("hop index tracked", w["hops_seen"] == 900)

    c = curve(w)
    # total gain 0.4; by t=25s (25% of 100s) both improvements are in -> 1.00
    chk("total gain is last_best - start", abs(c["gain"] - 0.4) < 1e-12,
        f"{c['gain']}")
    chk("f25 counts only improvements at or before 25% of the walk",
        abs(c["f25"] - 1.0) < 1e-12, f"{c['f25']}")
    chk("last improvement fraction is exact", abs(c["frac_last"] - 0.25) < 1e-12,
        f"{c['frac_last']}")

    # a walk with a late improvement must NOT report f25 == 1
    late = list(SYNTH)
    late[3] = ("[hop   150   80.0s] recluster/k3 3.100000000000 -> "
               "3.400000000000 (+3.0e-01)")
    c2 = curve(parse_walk(late))
    chk("a late improvement lowers f25 and f50",
        abs(c2["f25"] - 0.25) < 1e-12 and abs(c2["f50"] - 0.25) < 1e-12,
        f"f25={c2['f25']:.2f} f50={c2['f50']:.2f}")
    chk("a monotone-in-time curve never exceeds 1",
        all(c2[k] <= 1.0 + 1e-12 for k in ("f25", "f50", "f75")))

    # a log with no improvements at all must return None, not crash or divide
    chk("a walk with zero improvements yields None",
        curve(parse_walk(SYNTH[:1])) is None)
    chk("an empty log yields None", curve(parse_walk([])) is None)

    # 3. the real iteration-8 logs: the finding this tool exists to state
    real = sorted(glob.glob("artifacts/iter8_sweep/log_*.txt"))
    if real:
        rows = [curve(parse_walk(open(p).read().splitlines())) for p in real]
        rows = [r for r in rows if r]
        chk("all nine iteration-8 walks parse", len(rows) == 9, f"{len(rows)}")
        chk("every iteration-8 walk had <= 4 improvements TOTAL",
            all(r["n_improve"] <= 4 for r in rows),
            str([r["n_improve"] for r in rows]))
        chk("majority of iteration-8 walks finished improving before halfway",
            sum(1 for r in rows if r["frac_last"] <= 0.5) >= 5,
            f"{sum(1 for r in rows if r['frac_last'] <= 0.5)}/9")
        chk("the idle tail is the dominant cost (>1000s across 9 walks)",
            sum(r["T"] - r["t_last_improve"] for r in rows) > 1000.0,
            f"{sum(r['T'] - r['t_last_improve'] for r in rows):.0f}s")
    else:
        print("  [skip] iteration-8 logs not present")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="endgame/pipeline stdout logs")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        print("walkcurve self-test")
        sys.exit(0 if _self_test() else 1)
    paths = []
    for pat in a.logs:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    if not paths:
        ap.error("give at least one log path")
    report(paths)
