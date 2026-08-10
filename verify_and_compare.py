#!/usr/bin/env python3
"""Fetch Packomania records, independently verify `.pck` packings, and compare Σr against the records.

One tool, three subcommands (pure standard library — no numpy, no solver code, so `verify` is a genuine
independent check of any packing, ours or anyone else's):

    # 1. fetch the LATEST best-known records straight from packomania.com into a json (with provenance)
    python3 verify_and_compare.py fetch                    # -> sota/packomania/packomania_csqv.json
    python3 verify_and_compare.py fetch --out records.json

    # 2. verify ONE packing's strict feasibility + Σr (optionally classify vs a record)
    python3 verify_and_compare.py verify sota/.../pck/csqv54.pck --records sota/packomania/packomania_csqv.json
    python3 verify_and_compare.py verify some.pck --record 3.841733103296

    # 3. verify EVERY packing in a dir AND compare vs records -> comparison.md (a WIN must be strictly
    #    feasible AND strictly exceed the record). --refresh re-fetches the records first.
    python3 verify_and_compare.py compare --pck-dir sota/nietzsche-sm-radical-v6-n54 --refresh

`fetch` reads https://www.packomania.com/csqv/txt/sumradii.txt (the maintainer's plain-text sum-of-radii
table: `N<TAB>Σr`, `#` comments) — robust to parse and the single source of truth, so the records can
never silently drift from a hardcoded copy. Coordinate convention for `.pck`: unit square centred at the
origin ([-0.5, 0.5]); use --side/--corner for others. Exit code 0 = ok/feasible, 1 = infeasible/error.
"""
import argparse
import csv
import datetime
import json
import math
import os
import re
import sys
import urllib.request

CSQV_TXT = "https://www.packomania.com/csqv/txt/sumradii.txt"
REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RECORDS = os.path.join(REPO, "sota/packomania/packomania_csqv.json")
WIN_EPS = 1e-9      # Σr must strictly exceed the record by this to count as a win
TIE_EPS = 1e-6      # |Σr - record| within this is a tie (reproduces the best-known)


# --------------------------------------------------------------------------- fetch
def fetch_records(url=CSQV_TXT, timeout=30):
    """Download and parse the packomania csqv sum-of-radii table -> ({N: Σr}, source_url)."""
    req = urllib.request.Request(url, headers={"User-Agent": "circle-packing-sota verify_and_compare"})
    txt = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")
    recs = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            recs[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    if not recs:
        raise ValueError(f"parsed 0 records from {url} (format changed?)")
    return recs, url


def write_records_json(recs, source, path):
    ns = sorted(recs)
    doc = {
        "source": source,
        "problem": "packomania csqv: maximize sum of radii of N variable circles in a unit square",
        "retrieved": datetime.date.today().isoformat(),
        "n_min": ns[0], "n_max": ns[-1], "count": len(ns),
        "records": {str(n): recs[n] for n in ns},
    }
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    for p in (path, os.path.join(d, "history", f"{os.path.splitext(os.path.basename(path))[0]}_{doc['retrieved']}.json")):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")   # canonical latest + a dated snapshot in history/ (records change over time)
    return doc


def load_records(path):
    doc = json.load(open(path))
    return {int(k): float(v) for k, v in doc["records"].items()}, doc.get("retrieved", "?"), doc.get("source", "?")


# --------------------------------------------------------------------------- verify (independent, pure geometry)
def parse_pck(path):
    """(author, [(x,y,r), ...]) from a .pck: line1 = largest radius, line2 = author, rest = 'x y r'."""
    raw = [ln for ln in open(path).read().splitlines() if ln.strip()]
    if len(raw) < 3:
        raise ValueError("file has fewer than 3 non-empty lines")
    circ = []
    for ln in raw[2:]:
        parts = ln.split()
        if len(parts) != 3:
            raise ValueError(f"expected 'x y r', got: {ln!r}")
        circ.append(tuple(float(p) for p in parts))
    return raw[1], circ


def verify(circ, side=1.0, corner=False, tol=1e-9):
    """Feasibility of `circ` in a square of the given side, straight from the coordinates."""
    n = len(circ)
    lo, hi = (0.0, side) if corner else (-side / 2.0, side / 2.0)
    S = sum(r for *_, r in circ)
    cont = min(min(x - r - lo, hi - x - r, y - r - lo, hi - y - r) for x, y, r in circ)
    pair = math.inf
    for i in range(n):
        xi, yi, ri = circ[i]
        for j in range(i + 1, n):
            xj, yj, rj = circ[j]
            pair = min(pair, math.hypot(xi - xj, yi - yj) - (ri + rj))
    neg_r = min(r for *_, r in circ)
    feasible = neg_r >= -tol and cont >= -tol and (n < 2 or pair >= -tol)
    degenerate = neg_r <= tol   # a zero radius is not a circle -> not a packing of n circles
    return {"n": n, "sum_radii": S, "min_radius": neg_r,
            "min_containment_slack": cont, "min_pairwise_slack": (None if n < 2 else pair),
            "feasible": feasible, "degenerate": degenerate, "valid": feasible and not degenerate, "tol": tol}


def classify(sr, rec):
    d = sr - rec
    return ("WIN" if d > WIN_EPS else "tie" if abs(d) <= TIE_EPS else "below"), d


# --------------------------------------------------------------------------- subcommands
def cmd_fetch(a):
    recs, src = fetch_records(a.url, a.timeout)
    doc = write_records_json(recs, src, a.out)
    print(f"fetched {doc['count']} records (N {doc['n_min']}..{doc['n_max']}) "
          f"retrieved {doc['retrieved']} -> {a.out}")
    for n in (2, 27, 54, 55, 100):
        if str(n) in doc["records"]:
            print(f"  n={n:>3}  {doc['records'][str(n)]:.12f}")
    return 0


def cmd_verify(a):
    author, circ = parse_pck(a.pck)
    r = verify(circ, side=a.side, corner=a.corner, tol=a.tol)
    rec = a.record
    if rec is None and a.records:
        recs, _, _ = load_records(a.records)
        rec = recs.get(r["n"])
    print(f"file            : {a.pck}")
    print(f"author          : {author}")
    print(f"circles (N)     : {r['n']}")
    print(f"sum of radii Σr : {r['sum_radii']:.12f}")
    print(f"min radius      : {r['min_radius']:.3e}  (must be > 0)")
    print(f"min wall slack  : {r['min_containment_slack']:.3e}  (>= 0 => all inside the square)")
    ps = r["min_pairwise_slack"]
    print(f"min pair slack  : {'n/a (N<2)' if ps is None else f'{ps:.3e}'}  (>= 0 => no overlap)")
    print(f"STRICTLY FEASIBLE (tol {a.tol:g}): {r['feasible']}")
    if r["degenerate"]:
        print(f"DEGENERATE      : a radius is 0, so this is not a packing of {r['n']} circles")
    if rec is not None:
        v, d = classify(r["sum_radii"], rec)
        tag = {"WIN": "BEATS the record", "tie": "ties", "below": "below"}[v]
        print(f"record          : {rec:.12f}")
        print(f"Δ (ours-record) : {d:+.3e}   -> {tag}{'' if r['valid'] else ' (but INFEASIBLE)'}")
    return 0 if r["valid"] else 1


def cmd_compare(a):
    records_path = a.records or DEFAULT_RECORDS
    if a.refresh or not os.path.exists(records_path):
        recs, src = fetch_records(a.url, a.timeout)
        write_records_json(recs, src, records_path)
        print(f"[fetched latest records -> {records_path}]")
    recs, retrieved, source = load_records(records_path)

    pdir = os.path.join(a.pck_dir, "pck")
    if not os.path.isdir(pdir):
        pdir = a.pck_dir
    files = sorted((f for f in os.listdir(pdir) if re.fullmatch(r"csqv\d+\.pck", f)),
                   key=lambda f: int(f[4:-4]))
    rows, wins, ties, below, infeasible = [], [], [], [], []
    for f in files:
        n = int(f[4:-4])
        try:
            _, circ = parse_pck(os.path.join(pdir, f))
            r = verify(circ, side=a.side, corner=a.corner, tol=a.tol)
        except (OSError, ValueError) as e:
            infeasible.append(n); rows.append((n, None, recs.get(n), None, None, "PARSE_ERR", str(e))); continue
        rec = recs.get(n)
        if not r["valid"]:
            infeasible.append(n); rows.append((n, r["sum_radii"], rec, None, None, "INFEASIBLE",
                                               f"pair={r['min_pairwise_slack']:.1e} wall={r['min_containment_slack']:.1e}"))
            continue
        if rec is None:
            rows.append((n, r["sum_radii"], None, None, None, "no-record", "")); continue
        v, d = classify(r["sum_radii"], rec)
        gap = d / rec * 100.0
        rows.append((n, r["sum_radii"], rec, d, gap, v, ""))
        (wins if v == "WIN" else ties if v == "tie" else below).append(n)

    out = a.out or os.path.join(a.pck_dir, "comparison.md")
    lines = [
        "# Our solver vs Packomania `csqv` records (max Σr, N variable circles in a unit square)",
        "",
        f"Records: {source} — retrieved {retrieved}. Every row below is **independently re-verified from "
        f"its `.pck` coordinates** (pure geometry, no solver); a **WIN** is strictly feasible AND Σr strictly "
        f"exceeds the record.",
        "",
        f"**WINS: {len(wins)}** · ties (≤1e-6): {len(ties)} · below: {len(below)} · "
        f"infeasible: {len(infeasible)} · sizes: {len(files)}.",
        (f"Wins at N = {sorted(wins)}." if wins else "No strictly-feasible N beats the record."),
        "",
        "| N | ours Σr | record | Δ (ours−rec) | gap % | verdict |",
        "|---:|---:|---:|---:|---:|:--|",
    ]
    mark = {"WIN": "**WIN** 🏆", "tie": "tie"}
    for n, sr, rec, d, gap, v, note in rows:
        if v in ("INFEASIBLE", "PARSE_ERR"):
            sr_s = f"{sr:.12f}" if sr is not None else "—"
            rec_s = f"{rec:.12f}" if rec is not None else "—"
            lines.append(f"| {n} | {sr_s} | {rec_s} | — | — | ⚠ {v} ({note}) |")
        elif v == "no-record":
            lines.append(f"| {n} | {sr:.12f} | — | — | — | no record |")
        else:
            m = mark.get(v, f"{gap:.4f}%")
            lines.append(f"| {n} | {sr:.12f} | {rec:.12f} | {d:+.2e} | {gap:+.4f} | {m} |")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    print(f"\nwrote {out}")
    if infeasible:
        print(f"WARNING: {len(infeasible)} infeasible/parse-error packings: {sorted(infeasible)}")
    return 1 if infeasible else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="download latest packomania csqv records -> json")
    pf.add_argument("--out", default=DEFAULT_RECORDS)
    pf.add_argument("--url", default=CSQV_TXT)
    pf.add_argument("--timeout", type=int, default=30)
    pf.set_defaults(func=cmd_fetch)

    pv = sub.add_parser("verify", help="independent feasibility + Σr of one .pck")
    pv.add_argument("pck")
    pv.add_argument("--record", type=float, default=None)
    pv.add_argument("--records", default=None, help="records json to auto-look-up this N's record")
    pv.add_argument("--side", type=float, default=1.0)
    pv.add_argument("--corner", action="store_true")
    pv.add_argument("--tol", type=float, default=1e-9)
    pv.set_defaults(func=cmd_verify)

    pc = sub.add_parser("compare", help="verify every .pck in a dir and compare vs records -> comparison.md")
    pc.add_argument("--pck-dir", required=True, help="sweep dir (containing pck/) or a pck/ dir")
    pc.add_argument("--records", default=None, help=f"records json (default {DEFAULT_RECORDS})")
    pc.add_argument("--refresh", action="store_true", help="fetch latest records before comparing")
    pc.add_argument("--out", default=None, help="comparison.md path (default <pck-dir>/comparison.md)")
    pc.add_argument("--url", default=CSQV_TXT)
    pc.add_argument("--timeout", type=int, default=30)
    pc.add_argument("--side", type=float, default=1.0)
    pc.add_argument("--corner", action="store_true")
    pc.add_argument("--tol", type=float, default=1e-9)
    pc.set_defaults(func=cmd_compare)

    a = ap.parse_args()
    try:
        return a.func(a)
    except (OSError, ValueError, urllib.error.URLError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
