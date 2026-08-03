#!/usr/bin/env python3
"""Compare our swept Σr (results.csv) against the Packomania `csqv` records and write comparison.md.

Records fetched from https://www.packomania.com/csqv/csqv.html on 2026-08-03 (E. Specht; the table was
refreshed 01-Aug-2026 with AI-driven contributions). A WIN = our Σr strictly exceeds the record.
"""
import argparse
import csv
import os

# Packomania csqv best-known Σr (variable circles in a unit square), N -> record.
PACKO = {
    1: 0.500000000000, 2: 0.585786437626, 3: 0.796287456147,
    4: 1.006788474668, 5: 1.103553390593, 6: 1.202838910871, 7: 1.306546350758,
    8: 1.423813915320, 9: 1.524365359370, 10: 1.591012852634, 11: 1.680058380778,
    12: 1.765978318170, 13: 1.829542411692, 14: 1.905667205293, 15: 1.980266508444,
    16: 2.053080418439, 17: 2.111185327830, 18: 2.178530295972, 19: 2.236704571294,
    20: 2.301122834497, 21: 2.362117319374, 22: 2.420202649747, 23: 2.478013611963,
    24: 2.530311586971, 25: 2.587275055266, 26: 2.635983084918, 27: 2.685350025228,
    28: 2.737739985536, 29: 2.790344154631, 30: 2.842668747462, 31: 2.889969851933,
    32: 2.939572771205, 33: 2.987285008591, 34: 3.029799271188, 35: 3.074036363728,
    36: 3.121754486102, 37: 3.161498916179, 38: 3.205945627974, 39: 3.248110798173,
    40: 3.292391572609, 41: 3.336245021959, 42: 3.380421747611, 43: 3.422605518046,
    44: 3.466043568234, 45: 3.503095055216, 46: 3.539568936862, 47: 3.578955041750,
    48: 3.614921770409, 49: 3.655413576541, 50: 3.686474098753, 51: 3.725628939334,
    52: 3.760937808394, 53: 3.803308399164, 54: 3.839105155920, 55: 3.880994791082,
    56: 3.918533533830, 57: 3.954683754497, 58: 3.990343027691, 59: 4.023613974606,
    60: 4.057375010904, 61: 4.087621293004, 62: 4.117005967587, 63: 4.151017533533,
    64: 4.190585830354, 65: 4.218091021409, 66: 4.251441553087, 67: 4.285588116689,
    68: 4.320867327799, 69: 4.356202760849, 70: 4.392179819755, 71: 4.423446251009,
    72: 4.458515412631, 73: 4.488004575020, 74: 4.516547167624, 75: 4.546182199834,
    76: 4.572442898484, 77: 4.600844976472, 78: 4.635159192020, 79: 4.660114059297,
    80: 4.690798042719, 81: 4.726033735177, 82: 4.749812789172, 83: 4.780543519724,
    84: 4.812216737073, 85: 4.840788086402, 86: 4.875442875490, 87: 4.903706396279,
    88: 4.934199720258, 89: 4.963123644181, 90: 4.996999276706, 91: 5.019290432665,
    92: 5.042913382821, 93: 5.067858107547, 94: 5.095772569675, 95: 5.120799320222,
    96: 5.151567041717, 97: 5.175713921632, 98: 5.199716369613, 99: 5.229812460219,
    100: 5.261573443983,
}
WIN_EPS = 1e-9      # strictly beats the record
TIE_EPS = 1e-6      # matches to ~6 decimal places


def verdict(ours, rec):
    d = ours - rec
    if d > WIN_EPS:
        return "WIN", d
    if abs(d) <= TIE_EPS:
        return "tie", d
    return "below", d


def main():
    # Defaults are resolved against the repo root (this file's parent dir), not the caller's cwd,
    # so `python3 solver/compare.py` works from anywhere.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(repo, "sota/ours/results.csv"))
    ap.add_argument("--out", default=os.path.join(repo, "sota/ours/comparison.md"))
    a = ap.parse_args()
    rows, wins, ties, below = [], [], [], []
    with open(a.results) as f:
        for r in csv.DictReader(f):
            if not r.get("sum_radii"):
                continue
            n = int(r["n"]); ours = float(r["sum_radii"])
            rec = PACKO.get(n)
            if rec is None:
                continue
            v, d = verdict(ours, rec)
            gap = (rec - ours) / rec * 100.0
            rows.append((n, ours, rec, d, gap, v))
            (wins if v == "WIN" else ties if v == "tie" else below).append(n)
    rows.sort()   # the parallel sweep completes out of order; present by N
    lines = [
        "# Our evolved solver vs Packomania `csqv` records (max Σr, N circles in a unit square)",
        "",
        f"Sweep: {len(rows)} sizes.  **WINS: {len(wins)}**  ·  ties (≤1e-6): {len(ties)}  ·  below: {len(below)}.",
        (f"Wins at N = {wins}." if wins else "No N beats the Packomania record (expected; see README)."),
        "",
        "| N | ours Σr | Packomania record | Δ (ours−rec) | gap % | verdict |",
        "|---:|---:|---:|---:|---:|:--|",
    ]
    for n, ours, rec, d, gap, v in rows:
        mark = {"WIN": "**WIN** 🏆", "tie": "tie", "below": f"−{gap:.4f}%"}[v]
        lines.append(f"| {n} | {ours:.12f} | {rec:.12f} | {d:+.2e} | {gap:+.4f} | {mark} |")
    if below:
        worst = max(rows, key=lambda t: t[4]); best = min((r for r in rows if r[5] == "below"), key=lambda t: t[4], default=None)
        lines += ["", f"Closest below: N={best[0]} ({best[4]:+.4f}%). Largest gap: N={worst[0]} ({worst[4]:+.4f}%)." if best else ""]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
