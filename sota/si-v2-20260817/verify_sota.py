#!/usr/bin/env python3
"""verify_sota.py -- re-verify every record-beat claim in this directory FROM COORDINATES.

Runs no optimizer and trusts nothing in manifest.json: for each pck/csqv<n>.pck it re-parses the
(x, y, r) triples, recomputes sum-of-radii and the two feasibility slacks with the FROZEN mission
scorer (missions/circle-packing/scorer/{harness,verify}.py), and re-tests the beat against the
packomania csqv snapshot retrieved 2026-08-14. Every number printed here is recomputed; the manifest
is only read so its claims can be CONTRADICTED.

A per-n line is PASS iff all of:
  * the file parses and holds exactly n circles, every r > 0
  * strictly feasible at tol = 0.0   (wall slack >= 0 and pairwise slack >= 0, recomputed)
  * harness.counts_for(sum_r, record)  ->  sum_r >= record + harness.WIN_MARGIN (1e-6, ABSOLUTE)
  * the recomputed sum_r / margins / slacks agree with manifest.json to the last bit

Usage:
    python3 verify_sota.py                       # uses the in-repo scorer + the local table copy
    python3 verify_sota.py --scorer /path/to/missions/circle-packing/scorer
    python3 verify_sota.py --records /path/to/packomania_csqv_20260814.json
Exit code 0 iff every n PASSes. Stdlib + the frozen scorer only.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCORER = "/Users/jason/Desktop/science_moonshot/self_improvement_v2/missions/circle-packing/scorer"
DEFAULT_RECORDS = os.path.join(HERE, "packomania_csqv_20260814.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorer", default=os.environ.get("CP_SCORER", DEFAULT_SCORER),
                    help="frozen scorer dir holding harness.py + verify.py")
    ap.add_argument("--records", default=DEFAULT_RECORDS,
                    help="packomania csqv snapshot json (dict under 'records')")
    ap.add_argument("--packs", default=os.path.join(HERE, "pck"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.json"))
    args = ap.parse_args(argv)

    if not os.path.isdir(args.scorer):
        print("FATAL: scorer dir not found: %s\n       pass --scorer <dir> or set CP_SCORER."
              % args.scorer)
        return 2
    sys.path.insert(0, os.path.abspath(args.scorer))
    import harness                                    # noqa: E402  frozen geometry + beat rule
    import verify                                     # noqa: E402  frozen .pck parser

    man = json.load(open(args.manifest))
    table = json.load(open(args.records))
    rec = {int(k): float(v) for k, v in table["records"].items()}

    print("scorer      : %s" % os.path.abspath(args.scorer))
    print("  harness.py  sha256 %s  %s" % (
        sha256(os.path.join(args.scorer, "harness.py"))[:16],
        "== manifest" if sha256(os.path.join(args.scorer, "harness.py"))
        == man["verification"]["harness_sha256"] else "!! DIFFERS FROM MANIFEST"))
    print("  verify.py   sha256 %s  %s" % (
        sha256(os.path.join(args.scorer, "verify.py"))[:16],
        "== manifest" if sha256(os.path.join(args.scorer, "verify.py"))
        == man["verification"]["verify_sha256"] else "!! DIFFERS FROM MANIFEST"))
    print("  FEAS_TOL %g   WIN_MARGIN %g (ABSOLUTE: beat iff sum_r >= record + WIN_MARGIN)"
          % (harness.FEAS_TOL, harness.WIN_MARGIN))
    tsha = sha256(args.records)
    print("record table: %s" % os.path.abspath(args.records))
    print("  retrieved %s, site last update %s, %d values, sha256 %s  %s"
          % (table.get("retrieved"), table.get("site_last_update"), len(rec), tsha[:16],
             "== manifest" if tsha == man["record_table"]["sha256"] else "!! DIFFERS FROM MANIFEST"))
    print("packs       : %s" % os.path.abspath(args.packs))
    print("")

    claimed = sorted(int(n) for n in man["records"])
    on_disk = sorted(int(f[4:-4]) for f in os.listdir(args.packs)
                     if f.startswith("csqv") and f.endswith(".pck"))
    if claimed != on_disk:
        print("FATAL: manifest lists n=%s but pck/ holds n=%s" % (claimed, on_disk))
        return 2

    print("%-5s %-18s %-18s %-12s %-11s %-10s %-10s %s"
          % ("n", "sum_r", "record", "abs margin", "rel margin", "pair slack", "wall slack", "verdict"))
    failures, n_pass = [], 0
    for n in claimed:
        path = os.path.join(args.packs, "csqv%d.pck" % n)
        why = []
        try:
            circles = verify.parse_pck(path)
        except Exception as exc:                       # noqa: BLE001
            failures.append(n)
            print("%-5d %s" % (n, "FAIL  unparseable: %s" % exc))
            continue
        v0 = harness.validate(circles, n, tol=0.0)
        v9 = harness.validate(circles, n, tol=harness.FEAS_TOL)
        r = rec.get(n)
        m = man["records"][str(n)]

        if r is None:
            why.append("n absent from record table")
        if not v0["count_ok"]:
            why.append("expected %d circles, found %d" % (n, v0["n_found"]))
        if not v0["feasible"]:
            why.append("infeasible at tol=0.0")
        if not v9["feasible"]:
            why.append("infeasible at tol=%g" % harness.FEAS_TOL)
        if r is not None and not harness.counts_for(v0["sum_r"], r):
            why.append("does not beat record by >= %g" % harness.WIN_MARGIN)
        # manifest must not have rounded, softened or otherwise improved on the coordinates
        if v0["sum_r"] != float(m["sum_r"]):
            why.append("sum_r != manifest (%r vs %r)" % (v0["sum_r"], m["sum_r"]))
        if r is not None and r != m["record_20260814"]:
            why.append("record != manifest")
        if v0["pair_slack"] != m["min_pair_slack"] or v0["wall_slack"] != m["min_wall_slack"]:
            why.append("slacks != manifest")
        if sha256(path) != m["pack_sha256"]:
            why.append("pack sha256 != manifest")

        ok = not why
        n_pass += ok
        if not ok:
            failures.append(n)
        print("%-5d %-18.12f %-18.12f %+.4e  %+.4e  %+.3e %+.3e %s"
              % (n, v0["sum_r"], r if r is not None else float("nan"),
                 (v0["sum_r"] - r) if r is not None else float("nan"),
                 ((v0["sum_r"] - r) / r) if r is not None else float("nan"),
                 v0["pair_slack"], v0["wall_slack"],
                 "PASS" if ok else "FAIL  " + "; ".join(why)))

    print("")
    print("checked %d packings against the %s packomania csqv snapshot" % (len(claimed),
                                                                          table.get("retrieved")))
    print("PASS %d / %d   FAIL %d%s" % (n_pass, len(claimed), len(failures),
                                        ("  -> n = %s" % failures) if failures else ""))
    if not failures:
        margins = [harness.validate(verify.parse_pck(os.path.join(args.packs, "csqv%d.pck" % n)),
                                    n, tol=0.0)["sum_r"] - rec[n] for n in claimed]
        print("every claim reproduces from coordinates; absolute margins %.3e .. %.3e"
              % (min(margins), max(margins)))
    print("VERDICT: %s" % ("ALL CLAIMS VERIFIED" if not failures else "VERIFICATION FAILED"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
