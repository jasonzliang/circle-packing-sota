#!/usr/bin/env python3
"""Independently verify a packomania-format `.pck` packing — pure standard library, no solver, no numpy.

Given a `.pck` for N variable circles in a unit-side square, this recomputes the sum of radii and checks
strict feasibility (exactly N circles, every circle inside the square, no two overlapping) straight from
the coordinates. It deliberately shares NO code with the solver, so it is a genuine independent check of
any packing — ours or anyone else's. Optionally compares against a known record.

    python3 verify_pck.py ../sota/ours/pck/csqv27.pck --record 2.685350025228
    python3 verify_pck.py ../sota/ours/pck/csqv27.pck                 # just verify + print Σr
    # exit code 0 = strictly feasible, 1 = infeasible/parse error (usable in scripts)

Coordinate convention (matches what run_sweep.py writes): a square of side 1 centred at the origin, i.e.
x,y in [-0.5, 0.5]. Use --side S / --corner for other conventions.
"""
import argparse
import math
import sys


def parse_pck(path):
    """Return (author, [(x,y,r), ...]) from a .pck: line1 = largest radius, line2 = author, rest = x y r."""
    raw = [ln for ln in open(path).read().splitlines() if ln.strip()]
    if len(raw) < 3:
        raise ValueError("file has fewer than 3 non-empty lines")
    author = raw[1]
    circ = []
    for ln in raw[2:]:
        parts = ln.split()
        if len(parts) != 3:
            raise ValueError(f"expected 'x y r', got: {ln!r}")
        circ.append(tuple(float(p) for p in parts))
    return author, circ


def verify(circ, side=1.0, corner=False, tol=1e-9):
    """Feasibility of `circ` in a square of the given side. corner=False => centred at origin
    ([-side/2, side/2]); corner=True => [0, side]. Returns a dict with Σr, margins, feasibility."""
    n = len(circ)
    lo, hi = (0.0, side) if corner else (-side / 2.0, side / 2.0)
    S = sum(r for *_, r in circ)
    # containment slack: distance from each circle to the nearest wall, minus 0 (must be >= 0)
    cont = min(min(x - r - lo, hi - x - r, y - r - lo, hi - y - r) for x, y, r in circ)
    # pairwise slack: dist(centres) - (r_i + r_j) must be >= 0
    pair = math.inf
    for i in range(n):
        xi, yi, ri = circ[i]
        for j in range(i + 1, n):
            xj, yj, rj = circ[j]
            pair = min(pair, math.hypot(xi - xj, yi - yj) - (ri + rj))
    neg_r = min(r for *_, r in circ)
    feasible = neg_r >= -tol and cont >= -tol and (n < 2 or pair >= -tol)
    # A radius of zero is not a circle, so a config containing one is not a packing of n circles even
    # though it violates nothing. This is the failure mode a pure constraint check cannot see: a collapsed
    # config (every radius scaled to 0 by repair()) is trivially "feasible" and scores 0.
    degenerate = neg_r <= tol
    return {"n": n, "sum_radii": S, "min_radius": neg_r,
            "min_containment_slack": cont, "min_pairwise_slack": (None if n < 2 else pair),
            "feasible": feasible, "degenerate": degenerate, "valid": feasible and not degenerate,
            "tol": tol}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pck")
    ap.add_argument("--record", type=float, default=None, help="compare Σr against this best-known value")
    ap.add_argument("--side", type=float, default=1.0, help="square side length (default 1)")
    ap.add_argument("--corner", action="store_true", help="square is [0,side]^2 instead of centred at origin")
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()
    try:
        author, circ = parse_pck(a.pck)
        r = verify(circ, side=a.side, corner=a.corner, tol=a.tol)
    except (OSError, ValueError) as e:
        print(f"PARSE/READ ERROR: {e}"); return 1
    print(f"file            : {a.pck}")
    print(f"author          : {author}")
    print(f"circles (N)     : {r['n']}")
    print(f"sum of radii Σr : {r['sum_radii']:.12f}")
    print(f"min radius      : {r['min_radius']:.3e}  (must be > 0)")
    print(f"min wall slack  : {r['min_containment_slack']:.3e}  (>= 0 => all circles inside the square)")
    ps = r["min_pairwise_slack"]
    print(f"min pair slack  : {'n/a (N<2)' if ps is None else f'{ps:.3e}'}  (>= 0 => no two circles overlap)")
    print(f"STRICTLY FEASIBLE (tol {a.tol:g}): {r['feasible']}")
    if r["degenerate"]:
        print(f"DEGENERATE      : a radius is 0, so this is not a packing of {r['n']} circles")
    if a.record is not None:
        d = r["sum_radii"] - a.record
        tag = "BEATS the record" if (d > 1e-9 and r["valid"]) else ("ties" if abs(d) <= 1e-6 else "below")
        print(f"record          : {a.record:.12f}")
        print(f"Δ (ours-record) : {d:+.3e}   -> {tag}")
    return 0 if r["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
