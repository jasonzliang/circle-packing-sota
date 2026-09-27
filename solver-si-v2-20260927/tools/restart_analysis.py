#!/usr/bin/env python3
"""restart_analysis.py -- seeds vs. run length, from the anytime traces recorded in job JSONs.

For each (band, mode) pool of runs with traces, and each total budget B (CPU-s per size), evaluate the
k-restart policy: k runs of length T = B/k, value = E[max over k of q_i(T)] where q_i(T) is the run's
best-so-far at CPU time T (relative to its start incumbent). Estimated by resampling k runs (with
replacement) from the pool of runs at the same (solver, n). Reports the optimal k per budget, i.e. whether
restarts (k>1) or a single long run (k=1) is better, plus the run-time distribution of first improvement.
Usage: restart_analysis.py --rounds roundC roundD ... [--budgets 60,120,240,480]
"""
import argparse, glob, json, os, random, collections, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--rounds", nargs="+", required=True)
ap.add_argument("--budgets", default="60,120,240,480,960"); ap.add_argument("--samples", type=int, default=400)
a = ap.parse_args(); random.seed(1)
budgets = [float(x) for x in a.budgets.split(",")]
pool = collections.defaultdict(list)     # (band, mode, solver, n) -> list of traces [(t, gain over the round's incumbent)]
def band(n): return "51-77" if n <= 77 else "78-100" if n <= 100 else "x"
def pck_sum(p):
    import math
    return math.fsum(float(l.split()[2]) for l in open(p).read().split("\n")[2:] if len(l.split()) >= 3)
inc_cache = {}
for rd in a.rounds:
    warm_dir = os.environ.get("CP_CENSUS", "/tmp/si_tools/cp_census") + "/warm_%s" % rd.replace("round", "round")
    for jp in glob.glob(os.path.join(HERE, rd, "jobs", "*", "*", "n*_s*.json")):
        try: r = json.load(open(jp))
        except ValueError: continue
        tr = r.get("trace"); n = int(r["n"])
        if not tr or not r.get("feasible") or n < 51: continue
        key_inc = (rd, n)
        if key_inc not in inc_cache:
            wp = os.path.join(warm_dir, "csqv%d.pck" % n)
            inc_cache[key_inc] = pck_sum(wp) if os.path.isfile(wp) else tr[0][1]
        base = inc_cache[key_inc]            # the incumbent this round started from: gain = improvement over it
        pts = [(t, max(0.0, v - base)) for t, v in tr]
        solver, mode = jp.split(os.sep)[-3], jp.split(os.sep)[-2]
        pool[(band(n), mode, solver, n)].append((pts, r.get("solve_cpu_s") or pts[-1][0]))
def q_at(pts, T):
    best = 0.0
    for t, g in pts:
        if t <= T: best = max(best, g)
        else: break
    return best
groups = collections.defaultdict(list)   # (band, mode) -> keys
for k in pool: groups[(k[0], k[1])].append(k)
print("%-8s %-5s %6s | %s" % ("band", "mode", "budget", "E[max gain] for k = 1,2,4,8 restarts of length B/k   (best k)"))
for (bd, mode), keys in sorted(groups.items()):
    runs = sum(len(pool[k]) for k in keys); maxT = max(T for k in keys for _, T in pool[k])
    first = [next((t for t, g in pts if g > 1e-9), None) for k in keys for pts, _ in pool[k]]
    hit = [t for t in first if t is not None]
    print("--- %s %s: %d runs from %d (solver,n) cells, run length up to %.0f CPU-s; %d/%d runs improved, median first-improvement at %.0f s, 90%% at %.0f s" % (
        bd, mode, runs, len(keys), maxT, len(hit), len(first), statistics.median(hit) if hit else float('nan'), sorted(hit)[int(0.9 * len(hit)) - 1] if len(hit) > 1 else float('nan')))
    for B in budgets:
        if B > maxT * 1.05: continue
        vals = {}
        for k in (1, 2, 4, 8):
            T = B / k
            if T < 5: continue
            acc = []
            for _ in range(a.samples):
                key = random.choice(keys); rs = pool[key]
                if any(Tr < T * 0.95 for _, Tr in rs): continue   # need runs at least that long
                acc.append(max(q_at(random.choice(rs)[0], T) for _ in range(k)))
            if acc: vals[k] = sum(acc) / len(acc)
        if vals:
            bestk = max(vals, key=vals.get)
            print("%-8s %-5s %6.0f | %s   (k=%d)" % (bd, mode, B, "  ".join("k=%d: %.2e" % (k, v) for k, v in sorted(vals.items())), bestk))
