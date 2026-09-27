#!/usr/bin/env python3
"""aggregate.py -- collect every sweep job JSON under the given phase dirs, re-validate each packing
independently (pure python, tol 1e-9 AND strict slack >= 0), and write:
   <agg>/jobs.csv               every job, flat
   <agg>/per_n.md / per_n.csv   per n: live record, warm baseline, best strict sweep packing, margin, who found it
   <agg>/solvers.md             per-solver stats + specialization by n-band
   <agg>/best/csqv<n>.pck+json  best STRICT packing per n (from sweeps; falls back to the warm census if strict)
   <agg>/warm_next/             warm dir for the next phase (best at tol 1e-9 of census + sweeps)
Usage: aggregate.py --phase DIR [--phase DIR2 ...] --warm <census>/warm_p1 --out AGG
"""
import sys, argparse, csv, glob, json, math, os, sys, shutil, hashlib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_one   # validate_local, write_pck, parse_pck
LIVE = os.environ.get("CP_LIVE_RECORDS", os.path.join(HERE, "..", "..", "sota", "packomania", "packomania_csqv.json"))
CENSUS = os.environ.get("CP_CENSUS_MANIFEST", "")

def digits(s, rec):
    if rec is None or s is None: return None
    rel = max(0.0, (rec - s) / rec)
    return max(0.0, min(7.0, -math.log10(max(rel, 1e-7))))

def val(circ, n):
    v = sweep_one.validate_local(circ, n)
    strict = bool(v["feasible"]) and v["wall_slack"] is not None and v["wall_slack"] >= 0 and (v["pair_slack"] is None or v["pair_slack"] >= 0)
    return v, strict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", action="append", required=True)
    ap.add_argument("--warm", required=True, help="warm dir the phase used (baseline per n)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nmax", type=int, default=100)
    a = ap.parse_args()
    live = {int(k): float(v) for k, v in json.load(open(LIVE))["records"].items()}
    os.makedirs(a.out, exist_ok=True)
    # baseline: the warm packs (validated here, not trusted)
    warm = {}
    for p in glob.glob(os.path.join(a.warm, "csqv*.pck")):
        n = int(os.path.basename(p)[4:-4])
        try:
            circ = sweep_one.parse_pck(p)
        except Exception:
            continue
        v, strict = val(circ, n)
        if v["feasible"]:
            warm[n] = {"sum_r": v["sum_r"], "strict": strict, "circ": circ, "path": p}
    rows = []
    for ph in a.phase:
        for jp in glob.glob(os.path.join(ph, "jobs", "*", "*", "n*_s*.json")):
            try:
                r = json.load(open(jp))
            except ValueError:
                continue
            solver, mode = jp.split(os.sep)[-3], jp.split(os.sep)[-2]
            n = int(r.get("n") or os.path.basename(jp)[1:4]); seed = int(r.get("seed") or 0)
            row = {"phase": os.path.basename(ph.rstrip("/")), "solver": solver, "mode": mode, "n": n, "seed": seed,
                   "status": r.get("status"), "cpu_limit": r.get("solver_cpu_allowance"), "solve_cpu_s": r.get("solve_cpu_s"),
                   "sum_r": None, "feasible": False, "strict": False, "gap_live": None, "digits_live": None,
                   "d_warm": None, "job": jp, "circ": None}
            if r.get("centers") and r.get("radii"):
                circ = [(c[0], c[1], rr) for c, rr in zip(r["centers"], r["radii"])]
                v, strict = val(circ, n)
                if v["feasible"]:
                    row.update(sum_r=v["sum_r"], feasible=True, strict=strict, circ=circ,
                               gap_live=(v["sum_r"] - live[n]) if n in live else None, digits_live=digits(v["sum_r"], live.get(n)),
                               d_warm=(v["sum_r"] - warm[n]["sum_r"]) if n in warm else None)
            rows.append(row)
    rows.sort(key=lambda r: (r["n"], r["solver"], r["mode"], r["seed"]))
    with open(os.path.join(a.out, "jobs.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["phase", "solver", "mode", "n", "seed", "status", "cpu_limit", "solve_cpu_s", "feasible", "strict", "sum_r", "gap_live", "digits_live", "d_warm"])
        for r in rows:
            w.writerow([r["phase"], r["solver"], r["mode"], r["n"], r["seed"], r["status"], r["cpu_limit"], r["solve_cpu_s"], int(r["feasible"]), int(r["strict"]),
                        "" if r["sum_r"] is None else "%.15f" % r["sum_r"], "" if r["gap_live"] is None else "%+.3e" % r["gap_live"],
                        "" if r["digits_live"] is None else "%.3f" % r["digits_live"], "" if r["d_warm"] is None else "%+.3e" % r["d_warm"]])
    # per n
    ns = sorted(n for n in (set(r["n"] for r in rows) | set(warm)) if n in live and n <= a.nmax)
    rows = [r for r in rows if r["n"] in live]
    per_n = {}
    best_dir = os.path.join(a.out, "best"); os.makedirs(best_dir, exist_ok=True)
    warm_next = os.path.join(a.out, "warm_next"); os.makedirs(warm_next, exist_ok=True)
    wins = collections.Counter(); near = collections.Counter(); improved = collections.Counter()
    for n in ns:
        cand = [r for r in rows if r["n"] == n and r["feasible"]]
        strict_c = [r for r in cand if r["strict"]]
        best_any = max(cand, key=lambda r: r["sum_r"]) if cand else None
        best_strict = max(strict_c, key=lambda r: r["sum_r"]) if strict_c else None
        wb = warm.get(n)
        # publishable best = best strict among sweeps and (strict) warm baseline
        pool = []
        if best_strict: pool.append(("sweep", best_strict["sum_r"], best_strict))
        if wb and wb["strict"]: pool.append(("warm", wb["sum_r"], wb))
        pub = max(pool, key=lambda t: t[1]) if pool else None
        rec = live[n]
        if pub:
            src, s, obj = pub
            label = ("SI-v2 sweep %s seed %d %s" % (obj["solver"], obj["seed"], obj["mode"])) if src == "sweep" else ("SI-v2 census " + os.path.basename(os.path.dirname(os.path.dirname(obj["path"]))))
            circ = obj["circ"]
            sweep_one.write_pck(os.path.join(best_dir, "csqv%d.pck" % n), circ, "Jason Liang")
            json.dump({"n": n, "sum_r": s, "record_live": rec, "margin": s - rec, "source": src,
                       "solver": obj.get("solver"), "mode": obj.get("mode"), "seed": obj.get("seed"), "job": obj.get("job"), "warm_path": obj.get("path"),
                       "label": label}, open(os.path.join(best_dir, "csqv%d.json" % n), "w"), indent=1)
        # warm_next: best at tol 1e-9 among warm + sweeps
        wpool = ([(wb["sum_r"], wb["circ"])] if wb else []) + ([(best_any["sum_r"], best_any["circ"])] if best_any else [])
        if wpool:
            s2, c2 = max(wpool, key=lambda t: t[0])
            sweep_one.write_pck(os.path.join(warm_next, "csqv%d.pck" % n), c2, "warm")
        # specialization bookkeeping (feasible at 1e-9, sweeps only)
        if best_any:
            top = best_any["sum_r"]
            for r in cand:
                if r["sum_r"] >= top - 1e-9: near[r["solver"]] += 1
            uniq = {r["solver"] for r in cand if r["sum_r"] >= top - 1e-9}
            if len(uniq) == 1: wins[next(iter(uniq))] += 1
            for r in cand:
                if r["d_warm"] is not None and r["d_warm"] > 1e-9: improved[r["solver"]] += 1
        per_n[n] = {"rec": rec, "warm": wb["sum_r"] if wb else None, "warm_strict": wb["strict"] if wb else None,
                    "best_any": best_any, "best_strict": best_strict, "pub": pub, "n_jobs": len([r for r in rows if r["n"] == n]),
                    "n_feas": len(cand), "n_improved": sum(1 for r in cand if r["d_warm"] is not None and r["d_warm"] > 1e-9)}
    def verdict(m):
        if m is None: return "-"
        if m >= 1e-6: return "BEAT"
        if m > 1e-9: return "beat(<1e-6)"
        if m >= -1e-9: return "tie"
        return "below"
    with open(os.path.join(a.out, "per_n.md"), "w") as fh, open(os.path.join(a.out, "per_n.csv"), "w", newline="") as fc:
        w = csv.writer(fc); w.writerow(["n", "record_live", "warm_baseline", "warm_strict", "best_strict_sum", "margin_vs_live", "verdict", "source", "solver", "mode", "seed", "jobs", "feasible", "improved_over_warm", "best_any_sum", "best_any_solver"])
        fh.write("| n | live record | warm baseline | best (strict) | margin | verdict | found by | jobs | improved |\n|---:|---|---|---|---:|---|---|---:|---:|\n")
        for n in ns:
            p = per_n[n]; pub = p["pub"]
            s = pub[1] if pub else None; m = (s - p["rec"]) if s is not None else None
            who = "-"
            if pub:
                who = "%s/%s/s%d" % (pub[2]["solver"], pub[2]["mode"], pub[2]["seed"]) if pub[0] == "sweep" else "census"
            fh.write("| %d | %.12f | %s | %s | %s | %s | %s | %d | %d |\n" % (n, p["rec"], ("%.12f" % p["warm"]) if p["warm"] else "-",
                     ("%.12f" % s) if s is not None else "-", ("%+.2e" % m) if m is not None else "-", verdict(m), who, p["n_jobs"], p["n_improved"]))
            ba = p["best_any"]
            w.writerow([n, "%.12f" % p["rec"], "" if p["warm"] is None else "%.15f" % p["warm"], p["warm_strict"], "" if s is None else "%.15f" % s,
                        "" if m is None else "%+.3e" % m, verdict(m), pub[0] if pub else "", pub[2].get("solver") if pub else "", pub[2].get("mode") if pub else "",
                        pub[2].get("seed") if pub else "", p["n_jobs"], p["n_feas"], p["n_improved"], "" if not ba else "%.15f" % ba["sum_r"], ba["solver"] if ba else ""])
    # solver stats
    solvers = sorted(set(r["solver"] for r in rows))
    bands = [(1, 25), (26, 50), (51, 75), (76, 100)]
    with open(os.path.join(a.out, "solvers.md"), "w") as fh:
        fh.write("| solver | jobs | ok | feasible | strict | mean digits vs live | n at best (±1e-9) | unique best | improved over warm | beats live | " + " | ".join("digits %d-%d" % b for b in bands) + " |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(bands) + "\n")
        for s in solvers:
            rs = [r for r in rows if r["solver"] == s]
            feas = [r for r in rs if r["feasible"]]
            dig = [r["digits_live"] for r in feas]
            beats = len({r["n"] for r in feas if r["gap_live"] >= 1e-6})
            bd = []
            for lo, hi in bands:
                # per n best digits for this solver in the band
                pn = collections.defaultdict(float)
                for r in feas:
                    if lo <= r["n"] <= hi: pn[r["n"]] = max(pn[r["n"]], r["digits_live"])
                bd.append(("%.2f" % (sum(pn.values()) / len(pn))) if pn else "-")
            fh.write("| %s | %d | %d | %d | %d | %s | %d | %d | %d | %d | %s |\n" % (s, len(rs), sum(1 for r in rs if r["status"] == "ok"), len(feas),
                     sum(1 for r in feas if r["strict"]), ("%.3f" % (sum(dig) / len(dig))) if dig else "-", near[s], wins[s], improved[s], beats, " | ".join(bd)))
    # summary
    pubs = [per_n[n]["pub"] for n in ns if per_n[n]["pub"]]
    margins = {n: per_n[n]["pub"][1] - per_n[n]["rec"] for n in ns if per_n[n]["pub"]}
    summ = {"jobs": len(rows), "ok": sum(1 for r in rows if r["status"] == "ok"), "feasible": sum(1 for r in rows if r["feasible"]),
            "strict": sum(1 for r in rows if r["strict"]), "sizes": len(ns),
            "BEAT(>=1e-6)": sorted(n for n, m in margins.items() if m >= 1e-6), "beat(<1e-6)": sorted(n for n, m in margins.items() if 1e-9 < m < 1e-6),
            "tie": sorted(n for n, m in margins.items() if -1e-9 <= m <= 1e-9), "below": sorted(n for n, m in margins.items() if m < -1e-9),
            "improved_over_warm_sizes": sorted(n for n in ns if per_n[n]["n_improved"] > 0),
            "new_beats_from_sweep": sorted(n for n in ns if per_n[n]["pub"] and per_n[n]["pub"][0] == "sweep" and margins[n] >= 1e-6 and (per_n[n]["warm"] is None or per_n[n]["warm"] < per_n[n]["rec"] + 1e-6))}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
main()
