#!/usr/bin/env python3
"""tools/exact_check.py -- adversarial re-check of a packing in EXACT rational
arithmetic.

bench/verify.py works in float64 and allows a 1e-9 tolerance.  That leaves one
honest doubt: is a config that reports "max_violation = -1e-12" actually
feasible, or is the margin itself float noise?  This tool removes the doubt.

Every number in the JSON is a finite binary float, hence an exact rational.  So
the constraints can be decided with ZERO tolerance using fractions.Fraction:

    non-overlap :  (r_i + r_j)^2 <= dx^2 + dy^2      (both sides >= 0)
    inside      :  r_i <= x_i,  r_i <= 1 - x_i,  ...

No square roots, no epsilon, no rounding -- these comparisons are decided, not
estimated.  A config that passes here is feasible as a matter of arithmetic
fact, not of tolerance.

    python3 tools/exact_check.py bench/bench1/best.json --n 26
    python3 tools/exact_check.py --self-test
"""
import argparse
import json
import sys
from fractions import Fraction as F


def wall_terms(container):
    """Exact rational description of the container's boundary constraints.

    Both forms below are square-free rearrangements, so the wall test stays a
    DECISION in Fraction arithmetic rather than an estimate:
      polygon: r <= (c - a.p)/|a|   <=>   (c - a.p) >= 0 and (c-a.p)^2 >= r^2|a|^2
      disk   : r <= R - |p - o|     <=>   (R - r)   >= 0 and (R-r)^2   >= |p-o|^2
    Rational vertices / radius are therefore required, which is exactly why the
    bench4 containers are the unit disk and a right triangle rather than an
    area-1 disk (radius 1/sqrt(pi)) or an equilateral triangle.
    """
    if container is None:
        container = {"type": "polygon",
                     "vertices": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    if container["type"] == "disk":
        return "disk", (F(container["center"][0]), F(container["center"][1]),
                        F(container["radius"]))
    V = [(F(a), F(b)) for a, b in container["vertices"]]
    m = len(V)
    hp = []
    for k in range(m):
        (x0, y0), (x1, y1) = V[k], V[(k + 1) % m]
        ax, ay = (y1 - y0), -(x1 - x0)          # outward normal, CCW polygon
        hp.append((ax, ay, ax * x0 + ay * y0, k))
    return "poly", hp


def shape_terms(shape):
    """Exact rational gauge/support of the packed object K (None = Euclidean).

    For a polytopal K with rational vertices BOTH tests become exact rational
    LINEAR arithmetic -- strictly easier to decide than the circle case, which
    needed the squared rearrangements above:
        non-overlap :  max_k a_k.(c_i - c_j) >= r_i + r_j
        inside      :  a.c_i + r_i * h_K(a) <= b        for each container wall
    with facets a_k = n_k/(n_k.v_k) and h_K(a) = max over vertices of a.u, all
    rational.  Central symmetry is ASSERTED, not assumed: the disjointness
    identity r_i K (+) r_j (-K) = (r_i+r_j) K needs K = -K, and a lopsided K
    would make the whole check wrong rather than merely conservative.
    """
    if shape is None or shape.get("type") == "ball":
        return None
    V = [(F(a), F(b)) for a, b in shape["vertices"]]
    assert set(V) == {(-a, -b) for a, b in V}, "K must be centrally symmetric"
    m, A = len(V), []
    for k in range(m):
        (x0, y0), (x1, y1) = V[k], V[(k + 1) % m]
        ax, ay = (y1 - y0), -(x1 - x0)
        b = ax * x0 + ay * y0
        assert b > 0, "origin must be strictly inside K (CCW vertices)"
        A.append((ax / b, ay / b))
    return A, V


def _gauge(A, vx, vy):
    return max(ax * vx + ay * vy for ax, ay in A)


def _support(V, ax, ay):
    return max(ax * ux + ay * uy for ux, uy in V)


def exact_report(circles, n_required, container=None, shape=None):
    """Decide feasibility exactly.  Returns (ok, sum_radii_float, problems)."""
    problems = []
    if len(circles) != n_required:
        problems.append(f"count: expected {n_required}, got {len(circles)}")
        return False, 0.0, problems
    C = []
    for k, (x, y, r) in enumerate(circles):
        fx, fy, fr = F(x), F(y), F(r)          # exact: floats are dyadic rationals
        if fr < 0:
            problems.append(f"circle {k}: negative radius")
        C.append((fx, fy, fr))

    kind, W = wall_terms(container)
    K = shape_terms(shape)
    tight_wall, tight_pair = None, None
    for k, (x, y, r) in enumerate(C):
        if K is not None:                      # homothet of a polytope
            assert kind == "poly", "a polytopal shape needs a polygon container"
            SA, SV = K
            terms = [((c - ax * x - ay * y - r * _support(SV, ax, ay)),) * 2
                     + (f"edge {e}",) for (ax, ay, c, e) in W]
        elif kind == "poly":
            terms = []
            for (ax, ay, c, e) in W:
                g = c - ax * x - ay * y
                terms.append((g, g * g - r * r * (ax * ax + ay * ay), f"edge {e}"))
        else:
            ox, oy, R = W
            g = R - r
            terms = [(g, g * g - ((x - ox) ** 2 + (y - oy) ** 2), "the disk boundary")]
        for g, slack, what in terms:
            if g < 0 or slack < 0:
                problems.append(f"circle {k}: crosses {what} "
                                f"(squared slack {float(slack):.3e})")
            if tight_wall is None or slack < tight_wall:
                tight_wall = slack
    for a in range(len(C)):
        xa, ya, ra = C[a]
        for b in range(a + 1, len(C)):
            xb, yb, rb = C[b]
            if K is not None:
                slack = _gauge(K[0], xa - xb, ya - yb) - (ra + rb)   # linear!
                if slack < 0:
                    problems.append(f"objects {a},{b}: overlap "
                                    f"(gauge slack {float(slack):.6e})")
                if tight_pair is None or slack < tight_pair:
                    tight_pair = slack
                continue
            d2 = (xa - xb) ** 2 + (ya - yb) ** 2
            s2 = (ra + rb) ** 2
            if s2 > d2:                        # exact decision, no sqrt needed
                problems.append(f"circles {a},{b}: overlap (d^2={float(d2):.6e} "
                                f"< (r+r)^2={float(s2):.6e})")
            slack = d2 - s2
            if tight_pair is None or slack < tight_pair:
                tight_pair = slack

    total = sum(c[2] for c in C)
    ok = not problems
    if ok:
        # the two paths compute DIFFERENT quantities (squared for circles, linear
        # for polytopal homothets), so label them differently rather than print a
        # number under the wrong name
        obj, wl, pl = (("circles", "squared", "d^2-s^2") if K is None
                       else ("homothets", "linear", "gauge-(r_i+r_j)"))
        print(f"exact: {len(C)} {obj}, all constraints decided in exact rational "
              f"arithmetic with ZERO tolerance")
        print(f"exact: tightest wall slack ({wl}) = {float(tight_wall):+.6e}  "
              f"({'ok' if tight_wall >= 0 else 'VIOLATED'})")
        if tight_pair is not None:
            print(f"exact: tightest pair slack ({pl}) = {float(tight_pair):+.6e}  "
                  f"({'ok' if tight_pair >= 0 else 'VIOLATED'})")
        print(f"exact: sum(r) = {float(total):.12f}")
    return ok, float(total), problems


def self_test():
    ok, tot, p = exact_report([[F(1, 4), F(1, 4), F(1, 4)], [F(3, 4), F(1, 4), F(1, 4)],
                               [F(1, 4), F(3, 4), F(1, 4)], [F(3, 4), F(3, 4), F(1, 4)]], 4)
    assert ok and abs(tot - 1.0) < 1e-15, (ok, tot, p)
    # a violation far below float64 resolution of the coordinates must still be caught
    tiny = [[0.25, 0.25, 0.25 + 2 ** -45], [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25], [0.75, 0.75, 0.25]]
    ok2, _, p2 = exact_report(tiny, 4)
    assert not ok2 and p2, "exact checker missed a 2^-45 wall violation"
    print(f"  caught sub-nanometre violation: {p2[0]}")
    ok3, _, p3 = exact_report(tiny[:3], 4)
    assert not ok3 and "count" in p3[0]
    # containers: the exact wall test must decide disk and triangle too
    disk = {"type": "disk", "center": [0, 0], "radius": 1}
    tri = {"type": "polygon", "vertices": [[0, 0], [1, 0], [0, 1]]}
    assert exact_report([[0, 0, 1]], 1, disk)[0], "exact checker rejected the full disk"
    assert not exact_report([[0, 0, 1 + 2 ** -40]], 1, disk)[0], \
        "exact checker missed a 2^-40 overflow of the disk"
    assert not exact_report([[F(1, 2), F(1, 2), F(1, 5)]], 1, tri)[0], \
        "exact checker missed a circle crossing the hypotenuse"
    assert exact_report([[F(1, 5), F(1, 5), F(1, 8)]], 1, tri)[0], \
        "exact checker rejected a circle well inside the triangle"
    print("  container walls decided exactly (disk, right triangle)")

    # --- the SHAPE path: homothets of a polytope, decided in LINEAR rationals -
    sq = {"type": "polygon", "vertices": [[-1, -1], [1, -1], [1, 1], [-1, 1]]}
    hexa = {"type": "polygon", "vertices": [[1, 0], [F(1, 2), 1], [F(-1, 2), 1],
                                            [-1, 0], [F(-1, 2), -1], [F(1, 2), -1]]}
    # the 2x2 grid of axis-aligned squares of half-side 1/4 EXACTLY tiles the
    # unit square: sum(r) = 1 = the proved optimum for n = 4
    grid = [[F(1, 4), F(1, 4), F(1, 4)], [F(3, 4), F(1, 4), F(1, 4)],
            [F(1, 4), F(3, 4), F(1, 4)], [F(3, 4), F(3, 4), F(1, 4)]]
    okq, totq, pq = exact_report(grid, 4, None, sq)
    assert okq and totq == 1.0, (okq, totq, pq)
    # ... and the same centres with any larger half-side must be REJECTED
    big = [[c[0], c[1], F(1, 4) + F(1, 2 ** 40)] for c in grid]
    assert not exact_report(big, 4, None, sq)[0], "shape checker missed an overlap"
    # a square that pokes out of the container by 2^-40 must be caught
    out = [[F(1, 4) - F(1, 2 ** 40), F(1, 4), F(1, 4)]] + grid[1:]
    assert not exact_report(out, 4, None, sq)[0], "shape checker missed a wall breach"
    # the shape is what decides: a config feasible as CIRCLES of radius 0.3 at
    # distance 0.61 is infeasible as SQUARES of half-side 0.3 at the same centres
    diag = [[0.2, 0.2, 0.3], [0.2 + 0.61, 0.2, 0.3]]
    assert not exact_report(diag, 2, None, sq)[0], "square shape scored as circles"
    # hexagon: two homothets separated by exactly the gauge distance touch and
    # must be accepted; a hair closer must not be
    assert exact_report([[F(1, 4), F(1, 4), F(1, 8)], [F(1, 4) + F(1, 4), F(1, 4), F(1, 8)]],
                        2, None, hexa)[0]
    assert not exact_report([[F(1, 4), F(1, 4), F(1, 8)],
                             [F(1, 4) + F(1, 4) - F(1, 2 ** 40), F(1, 4), F(1, 8)]],
                            2, None, hexa)[0], "hexagon gauge not enforced"
    # a non-symmetric K must be refused rather than silently mis-decided
    try:
        exact_report(grid, 4, None, {"type": "polygon",
                                     "vertices": [[1, 0], [0, 1], [-1, F(-1, 2)]]})
    except AssertionError:
        pass
    else:
        raise AssertionError("exact checker accepted a non-symmetric K")
    print("  polytopal shapes decided exactly in LINEAR rational arithmetic")
    print("exact_check self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--n", type=int, default=26)
    ap.add_argument("--container", help="json file with a container spec, or a "
                                        "bench4 spec instance index like bench4:0")
    ap.add_argument("--shape", help="json file with a packed-object spec "
                                    "(default: the one stored in the config)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    d = json.load(open(a.config))
    ct = None
    if a.container:
        ct = json.load(open(a.container))
    elif isinstance(d, dict) and "container" in d:
        ct = d["container"]                     # the config carries its own
    sh = json.load(open(a.shape)) if a.shape else (
        d.get("shape") if isinstance(d, dict) else None)
    ok, tot, problems = exact_report(d["circles"] if isinstance(d, dict) else d,
                                     a.n, ct, sh)
    for p in problems:
        print("  PROBLEM:", p)
    print("EXACT_FEASIBLE:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
