#!/usr/bin/env python3
"""escalate.py -- successive-halving schedule: from the aggregate of earlier rounds, keep the top-K solvers
per n (ranked by their best sum_r over all prior jobs at that n, tie-break by improvement over the warm
baseline and by digits), and emit new (solver, n, seed, mode, cpu) jobs with a bigger budget.

  python escalate.py --agg AGG_DIR --out roundB.jsonl --top 4 --seeds 1-2 --base 60 --slope 3 \
        [--modes best|both] [--seeds-below 1-4] [--only-open]
  --seeds-below: extra seed range for n whose published best is still below/tie vs the live record
  --only-open : skip n whose best already BEATS the live record by >= 1e-6 (leave them alone)
"""
import argparse, csv, json, os, collections
ap = argparse.ArgumentParser()
ap.add_argument("--agg", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=4); ap.add_argument("--seeds", default="1-2")
ap.add_argument("--seeds-below", default=None); ap.add_argument("--modes", default="best")
ap.add_argument("--base", type=float, default=60); ap.add_argument("--slope", type=float, default=3)
ap.add_argument("--only-open", action="store_true"); ap.add_argument("--n", default="1-100")
ap.add_argument("--min-solvers", type=int, default=2, help="always keep at least this many distinct solvers per n")
a = ap.parse_args()
# optional focus file (read at every invocation, so a running campaign picks it up at its next round):
#   {"skip_n": "1-25", "single_solver_n": "26-50", "single_seeds": 1,
#    "boost_n": "51-77", "boost_extra_top": 1, "boost_solvers_first": ["nalt6", "tv14pf6", "deon6"]}
FOCUS = {}
_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "focus.json")
if os.path.isfile(_fp):
    FOCUS = json.load(open(_fp))
def rng(s):
    out = set()
    for p in s.split(","):
        if "-" in p: lo, hi = p.split("-"); out |= set(range(int(lo), int(hi) + 1))
        else: out.add(int(p))
    return sorted(out)
ns = set(rng(a.n)); seeds = rng(a.seeds); seeds_below = rng(a.seeds_below) if a.seeds_below else seeds
per_n_status = {}
for r in csv.DictReader(open(os.path.join(a.agg, "per_n.csv"))):
    per_n_status[int(r["n"])] = r["verdict"]
best = collections.defaultdict(dict)   # n -> (solver, mode) -> (sum_r, d_warm)
for r in csv.DictReader(open(os.path.join(a.agg, "jobs.csv"))):
    if r["feasible"] != "1" or not r["sum_r"]: continue
    n = int(r["n"]); key = (r["solver"], r["mode"]); s = float(r["sum_r"]); dw = float(r["d_warm"]) if r["d_warm"] else 0.0
    if key not in best[n] or s > best[n][key][0]: best[n][key] = (s, dw)
# ---- yield statistics for Thompson-sampling allocation (FOCUS["alloc"] == "thompson") -------------------
import math, random, glob
random.seed(int(FOCUS.get("alloc_seed", 0)) + sum(seeds))
WARM_OF = {"phase1": os.environ.get("CP_CENSUS", "/tmp/si_tools/cp_census") + "/warm_p1", "roundA": os.environ.get("CP_CENSUS", "/tmp/si_tools/cp_census") + "/warm_p1"}
def warm_dir(phase): return WARM_OF.get(phase, os.environ.get("CP_CENSUS", "/tmp/si_tools/cp_census") + "/warm_%s" % phase)
_inc = {}
def incumbent(phase, n):
    k = (phase, n)
    if k not in _inc:
        p = os.path.join(warm_dir(phase), "csqv%d.pck" % n)
        try: _inc[k] = math.fsum(float(l.split()[2]) for l in open(p).read().split("\n")[2:] if len(l.split()) >= 3)
        except OSError: _inc[k] = None
    return _inc[k]
def band_of(n): return "51-77" if n <= 77 else "78-100"
ystat = collections.defaultdict(lambda: [0, 0])      # (solver, band, mode) -> [jobs, beats]
for r in csv.DictReader(open(os.path.join(a.agg, "jobs.csv"))):
    n = int(r["n"])
    if n < 51: continue
    key = (r["solver"], band_of(n), r["mode"]); ystat[key][0] += 1
    if r["feasible"] == "1" and r["sum_r"]:
        i = incumbent(r["phase"], n)
        if i is not None and float(r["sum_r"]) > i + 1e-9: ystat[key][1] += 1
distinct = collections.defaultdict(set)              # n -> distinct nbr outcomes (rounded 1e-9) within 2e-3 of the best
_vals = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(a.agg, "jobs.csv"))):
    if r["mode"] == "nbr" and r["feasible"] == "1" and r["sum_r"]: _vals[int(r["n"])].append(float(r["sum_r"]))
for n_, vs in _vals.items():
    top = max(vs); distinct[n_] = {round(v, 9) for v in vs if top - v < 2e-3}
def sample_yield(solver, n, mode):
    j, b = ystat.get((solver, band_of(n), mode), [0, 0])
    return random.betavariate(1 + b, 1 + max(0, j - b))     # posterior draw of P(beat incumbent)
skip_n = set(rng(FOCUS["skip_n"])) if FOCUS.get("skip_n") else set()
single_n = set(rng(FOCUS["single_solver_n"])) if FOCUS.get("single_solver_n") else set()
boost_n = set(rng(FOCUS["boost_n"])) if FOCUS.get("boost_n") else set()
boost_first = FOCUS.get("boost_solvers_first", [])
rows = []; kept = collections.Counter()
for n in sorted(ns):
    if n not in best or n in skip_n: continue
    verdict = per_n_status.get(n, "-")
    if a.only_open and verdict == "BEAT": continue
    # rank solvers by their best over modes
    per_solver = {}
    for (s, m), (v, dw) in best[n].items():
        if s not in per_solver or v > per_solver[s][0]: per_solver[s] = (v, dw, m)
    pref = {s: i for i, s in enumerate(boost_first)}          # tie-break: preferred solvers first
    ranked = sorted(per_solver.items(), key=lambda kv: (-kv[1][0], -kv[1][1], pref.get(kv[0], 99)))
    if n in single_n:
        keep = ranked[:1]
        sd = (seeds if verdict != "below" else seeds_below)[:max(1, int(FOCUS.get("single_seeds", 1)))]
    elif n in boost_n and FOCUS.get("alloc") == "thompson":
        k = max(a.top, a.min_solvers) + int(FOCUS.get("boost_extra_top", 0))
        cands = list(per_solver)                                # every solver seen at this n
        nbr_pick = sorted(cands, key=lambda s: -sample_yield(s, n, "nbr"))[:k]
        self_pick = sorted(cands, key=lambda s: -sample_yield(s, n, "self"))[:int(FOCUS.get("self_top", 2))]
        kick_pick = sorted(cands, key=lambda s: -sample_yield(s, n, "kick") - 0.5 * sample_yield(s, n, "self"))[:int(FOCUS.get("kick_top", 0))] if FOCUS.get("kick") else []
        allk = list(dict.fromkeys(nbr_pick + self_pick + kick_pick))
        keep = [(s, per_solver[s]) for s in allk]
        keep_modes = {s: (["nbr"] if s in nbr_pick else []) + (["self"] if s in self_pick else []) + (["kick"] if s in kick_pick else []) for s in allk}
        sd = seeds_below
    elif n in boost_n:
        k = max(a.top, a.min_solvers) + int(FOCUS.get("boost_extra_top", 0))
        keep = ranked[:k]
        have = {s for s, _ in keep}
        for s in boost_first:                                   # make sure the mid-range specialists are in
            if s in per_solver and s not in have: keep.append((s, per_solver[s])); have.add(s)
        sd = seeds_below                                        # the larger seed set for every boosted n
    else:
        keep = ranked[:max(a.top, a.min_solvers)]
        sd = seeds_below if verdict in ("below", "tie", "beat(<1e-6)") else seeds
    cpu = round(a.base + a.slope * n, 1)
    modes_override = FOCUS.get("modes_override")          # "both": always run self AND nbr
    self_seeds = FOCUS.get("self_seeds")                  # e.g. 1: refine-the-incumbent is ~deterministic, one run is enough
    keep_modes = locals().get("keep_modes") if (n in boost_n and FOCUS.get("alloc") == "thompson") else None
    for s, (v, dw, m) in keep:
        modes = ["self", "nbr"] if (modes_override == "both" or a.modes == "both") else [m]
        if keep_modes: modes = keep_modes[s]
        self_cap = FOCUS.get("self_cpu_cap")                # refine-the-incumbent converges in seconds: cap its CPU
        for mode in modes:
            seeds_for_mode = sd[:int(self_seeds)] if (mode == "self" and self_seeds) else sd
            cpu_mode = min(cpu, float(self_cap)) if (mode == "self" and self_cap) else cpu
            if mode == "kick":
                cpu_mode = round(cpu * float(FOCUS.get("kick_cpu_scale", 0.5)), 1)
                if verdict != "BEAT" and len(distinct.get(n, ())) >= int(FOCUS.get("rugged_bonus_min_distinct", 10**9)):
                    seeds_for_mode = list(sd) + [x + 1000 for x in sd]        # rugged, unbeaten size: double the kicks
            fr = FOCUS.get("kick_fracs", [0.15, 0.25, 0.35, 0.45])
            for seed in seeds_for_mode:
                job = {"solver": s, "n": n, "seed": seed, "mode": mode, "cpu": cpu_mode}
                if mode == "kick": job["kick"] = fr[seed % len(fr)]
                rows.append(job); kept[s] += 1
with open(a.out, "w") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
tot = sum(r["cpu"] for r in rows)
print("%d jobs, %.0f CPU-s = %.1f CPU-h, ~%.2f h wall at 62 wide" % (len(rows), tot, tot / 3600, tot * 1.08 / 3600 / 62))
print("jobs per solver:", dict(kept.most_common()))
