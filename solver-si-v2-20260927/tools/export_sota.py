#!/usr/bin/env python3
"""export_sota.py -- turn an aggregate best/ dir into a sota/<name>/ artifact in the circle-packing-sota
repo's conventions: pck/csqv<N>.pck (Packomania text, centred square, %.17g, line 2 = author),
json/out<N>.json (corner-frame sidecar like the other sota dirs), results.csv, manifest.json.
Re-validates every packing at ZERO tolerance from the coordinates; a non-strict packing is NOT exported.
Then runs the repo's verify_and_compare.py compare --tol 0 against the given records snapshot.
Usage: export_sota.py --best AGG/best --records sota/packomania/history/packomania_csqv_2026-09-27.json \
                      --out <repo>/sota/si-v2-20260927 [--author "Jason Liang"] [--rounds-root DIR]
"""
import argparse, csv, glob, hashlib, json, os, subprocess, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep_one
import exact_verify
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True); ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--author", default="Jason Liang")
    ap.add_argument("--nmax", type=int, default=100)
    ap.add_argument("--rounds-root", default=os.environ.get("CP_SWEEP_ROOT", "/tmp/si_tools/cp_sweep"),
                    help="rounds root; provenance job paths are recorded relative to it")
    a = ap.parse_args()
    recs = {int(k): float(v) for k, v in json.load(open(a.records))["records"].items()}
    port = json.load(open(os.path.join(HERE, "portfolio.json")))
    rec_tag = json.load(open(a.records)).get("retrieved", "").replace("-", "")
    os.makedirs(os.path.join(a.out, "pck"), exist_ok=True); os.makedirs(os.path.join(a.out, "json"), exist_ok=True)
    rows, man = [], {}
    for p in sorted(glob.glob(os.path.join(a.best, "csqv*.pck")), key=lambda p: int(os.path.basename(p)[4:-4])):
        n = int(os.path.basename(p)[4:-4])
        if n > a.nmax: continue
        circ = sweep_one.parse_pck(p)
        v = sweep_one.validate_local(circ, n, tol=0.0)
        prov = json.load(open(p[:-4] + ".json")) if os.path.isfile(p[:-4] + ".json") else {}
        sp = port.get(prov.get("solver") or "", {})
        prov["solver_origin"] = sp.get("origin"); prov["solver_sha256"] = sp.get("sha256")
        if prov.get("job"): prov["job"] = os.path.relpath(prov["job"], a.rounds_root)
        if not v["feasible"]:
            print("SKIP n=%d: not strictly feasible (wall %.2e pair %s)" % (n, v["wall_slack"], v["pair_slack"])); continue
        # Packomania-grade gate: feasibility decided in EXACT rational arithmetic (zero tolerance). A packing
        # that fails by an ulp is repaired by shrinking every radius by k*2^-50 (sum drops <~1e-14).
        ex_ok, ex_info = exact_verify.exact_check(circ); ex_factor = 1.0
        if not ex_ok:
            fixed, ex_factor = exact_verify.repair(circ)
            if fixed is None:
                print("SKIP n=%d: not exact-feasible and not repairable (%s)" % (n, ex_info)); continue
            circ = fixed; v = sweep_one.validate_local(circ, n, tol=0.0); ex_ok, ex_info = exact_verify.exact_check(circ)
            print("n=%d: exact repair, radius factor %.17g" % (n, ex_factor))
        dst = os.path.join(a.out, "pck", "csqv%d.pck" % n)
        landed = sweep_one.write_pck(dst, circ, a.author)
        lv = sweep_one.validate_local(landed, n, tol=0.0)
        assert lv["feasible"] and abs(lv["sum_r"] - v["sum_r"]) < 1e-15, n
        corner = [[x + 0.5, y + 0.5, r] for x, y, r in sorted(circ, key=lambda c: c[2])]
        side = {"n": n, "sum_radii": v["sum_r"], "max_violation": max(0.0, -min(v["wall_slack"], v["pair_slack"] if v["pair_slack"] is not None else 0.0)),
                "min_wall_slack": v["wall_slack"], "min_pair_slack": v["pair_slack"], "frame": "corner [0,1]^2 (pck is centred [-0.5,0.5]^2)",
                "seed": prov.get("seed"), "solver": prov.get("solver"), "warm_mode": prov.get("mode"), "source": prov.get("source"),
                "exact_feasible": ex_ok, "exact_min_wall_slack": ex_info[0], "exact_min_pair_slack_sq": ex_info[1], "exact_repair_factor": ex_factor,
                "record_%s" % rec_tag: recs.get(n), "delta": (v["sum_r"] - recs[n]) if n in recs else None, "circles": corner}
        json.dump(side, open(os.path.join(a.out, "json", "out%d.json" % n), "w"), indent=1)
        d = v["sum_r"] - recs[n] if n in recs else None
        rows.append([n, "%.12f" % v["sum_r"], "%.2e" % side["max_violation"], "%.12f" % recs[n] if n in recs else "", ("%+.3e" % d) if d is not None else "", 1])
        man[str(n)] = {"sum_r": v["sum_r"], "record": recs.get(n), "margin": d, "verdict": ("WIN" if d is not None and d > 1e-9 else "tie" if d is not None and abs(d) <= 1e-9 else "below"),
                       "min_wall_slack": v["wall_slack"], "min_pair_slack": v["pair_slack"], "min_r": v["min_r"],
                       "exact_feasible": ex_ok, "exact_repair_factor": ex_factor,
                       "pck_sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(), "provenance": prov}
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["n", "sum_radii", "max_violation", "record_%s" % rec_tag, "delta", "feasible"]); w.writerows(rows)
    wins = sorted(int(n) for n, m in man.items() if m["verdict"] == "WIN"); ties = sorted(int(n) for n, m in man.items() if m["verdict"] == "tie")
    below = sorted(int(n) for n, m in man.items() if m["verdict"] == "below")
    json.dump({"artifact": os.path.basename(a.out.rstrip("/")), "records_file": os.path.relpath(a.records, REPO) if a.records.startswith(REPO) else a.records,
               "records_sha256": hashlib.sha256(open(a.records, "rb").read()).hexdigest(), "records_retrieved": json.load(open(a.records)).get("retrieved"),
               "rule": "feasible in EXACT rational arithmetic with zero tolerance (fractions.Fraction, squared comparisons; ulp-level overlaps repaired by shrinking radii) AND sum_r > record + 1e-9 (verify_and_compare.WIN_EPS)",
               "counts": {"exported": len(man), "WIN": len(wins), "tie": len(ties), "below": len(below)}, "wins": wins, "ties": ties, "below": below,
               "per_n": man}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("exported %d packings: WIN %d %s | tie %d | below %d %s" % (len(man), len(wins), wins, len(ties), len(below), below))
    r = subprocess.run([sys.executable, os.path.join(REPO, "verify_and_compare.py"), "compare", "--pck-dir", a.out, "--records", a.records, "--tol", "0",
                        "--out", os.path.join(a.out, "comparison.md")], capture_output=True, text=True)
    print(r.stdout[-600:], r.stderr[-400:])
if __name__ == "__main__":
    main()
