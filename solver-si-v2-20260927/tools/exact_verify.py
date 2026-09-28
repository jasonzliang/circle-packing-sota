#!/usr/bin/env python3
"""exact_verify.py -- decide feasibility of a centred-square .pck in EXACT rational arithmetic (zero tolerance).
Every double in the file is an exact rational; containment and pairwise non-overlap are decided with
squared comparisons (no sqrt, no epsilon). Prints one line per file; exit 1 if any fails.
   exact_verify.py csqv63.pck [...]     |  --repair-out DIR: write a minimally shrunk copy that passes"""
import sys, os, math
from fractions import Fraction as F
def parse(path):
    rows = [l.split() for l in open(path).read().split("\n")[2:] if len(l.split()) >= 3]
    return [(float(a), float(b), float(c)) for a, b, c in rows]
def exact_check(circ):
    H = F(1, 2)
    worst_wall, worst_pair = None, None
    for i, (x, y, r) in enumerate(circ):
        X, Y, R = F(x), F(y), F(r)
        if R <= 0: return False, ("r<=0", i)
        for s in (X + H - R, H - X - R, Y + H - R, H - Y - R):
            if s < 0: return False, ("wall", i, float(s))
            worst_wall = s if worst_wall is None or s < worst_wall else worst_wall
    n = len(circ)
    for i in range(n):
        Xi, Yi, Ri = F(circ[i][0]), F(circ[i][1]), F(circ[i][2])
        for j in range(i + 1, n):
            Xj, Yj, Rj = F(circ[j][0]), F(circ[j][1]), F(circ[j][2])
            d2 = (Xi - Xj) ** 2 + (Yi - Yj) ** 2; s2 = (Ri + Rj) ** 2
            if d2 < s2: return False, ("pair", i, j, float(d2 - s2))
            g = d2 - s2
            worst_pair = g if worst_pair is None or g < worst_pair else worst_pair
    return True, (float(worst_wall) if worst_wall is not None else None, float(worst_pair) if worst_pair is not None else None)
def repair(circ, max_steps=8):
    """Shrink every radius by k ulps-scale factors until exact-feasible; returns (circ, factor)."""
    f = 1.0
    for k in range(1, max_steps + 1):
        f = 1.0 - k * 2.0 ** -50
        cand = [(x, y, r * f) for x, y, r in circ]
        ok, _ = exact_check(cand)
        if ok: return cand, f
    return None, None
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rep = None
    if "--repair-out" in sys.argv: rep = sys.argv[sys.argv.index("--repair-out") + 1]; args = [a for a in args if a != rep]
    bad = 0
    for p in args:
        circ = parse(p); ok, info = exact_check(circ)
        print("%s n=%d EXACT_FEASIBLE=%s %s" % (os.path.basename(p), len(circ), ok, info))
        if not ok:
            bad += 1
            if rep:
                fixed, f = repair(circ)
                if fixed:
                    os.makedirs(rep, exist_ok=True)
                    with open(os.path.join(rep, os.path.basename(p)), "w") as fh:
                        fh.write("%.17g\n%s\n" % (max(r for *_, r in fixed), "Jason Liang"))
                        for x, y, r in sorted(fixed, key=lambda t: t[2]): fh.write("%.17g %.17g %.17g\n" % (x, y, r))
                    print("   repaired with radius factor %.17g -> sum %.15f (was %.15f)" % (f, math.fsum(r for *_, r in fixed), math.fsum(r for *_, r in circ)))
    sys.exit(1 if bad else 0)
