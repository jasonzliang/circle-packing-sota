#!/usr/bin/env python3
"""Side-by-side N=27: the previous record value vs our new record, both drawn with their
D1 mirror axis (X+Y=1) and mirror-pairs colored. Left = a packing at the previous record's
Sigma r (2.685350) found by our own solver (seed 1); Cantrell's exact 2011/12 coordinates
are not public, so this stands in for the previous optimum at that value. Right = our record.
    python3 writeup/make_compare_fig.py  ->  writeup/fig4_prev_vs_new.png"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
SOTA = os.path.dirname(HERE)

def load(p):
    d = json.load(open(p)); return np.array(d["circles"]), d["sum_radii"]

prev, sp = load(f"{HERE}/prev_value_packing.json")
ours, so = load(f"{SOTA}/sota/sm-radical-v6-n27/json/out27.json")

def pairs_axis(C, tol=1.5e-3):
    """Mirror pairs and on-axis circles for the reflection (x,y)->(1-y,1-x) [axis X+Y=1]."""
    n = len(C); used = [False]*n; pairs = []; axis = []
    for i in range(n):
        if used[i]: continue
        xi, yi, ri = C[i]
        if abs(xi + yi - 1) < tol:
            axis.append(i); used[i] = True; continue
        for j in range(n):
            if used[j] or j == i: continue
            if abs(C[j,0]-(1-yi)) < tol and abs(C[j,1]-(1-xi)) < tol and abs(C[j,2]-ri) < tol:
                pairs.append((i, j)); used[i] = used[j] = True; break
        else:
            axis.append(i); used[i] = True
    return pairs, axis

def draw(ax, C, title, sub):
    pairs, axis = pairs_axis(C)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=2.0, ec="#222222"))
    pal = plt.cm.tab20(np.linspace(0, 1, max(len(pairs), 1)))
    col = {}
    for k, (i, j) in enumerate(pairs):
        col[i] = pal[k]; col[j] = pal[k]
    for i, (x, y, r) in enumerate(C):
        c = col.get(i, (0.72, 0.72, 0.72, 1.0))     # on-axis circles: grey
        ax.add_patch(Circle((x, y), r, facecolor=c, edgecolor="white", lw=0.7, alpha=0.92))
    ax.plot([0, 1], [1, 0], color="#c0392b", ls="--", lw=1.5, zorder=6)     # mirror axis X+Y=1
    ax.text(0.90, 0.14, "mirror axis", color="#c0392b", fontsize=8.5, rotation=-45,
            ha="center", va="center", zorder=7)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
    ax.text(0.5, -0.055, sub, ha="center", va="top", fontsize=10, transform=ax.transAxes)
    return len(pairs), len(axis)

fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.9))
pa = draw(axes[0], prev, "Previous record value",
          f"$\\Sigma r$ = {sp:.9f}")
pb = draw(axes[1], ours, "New record (now listed on Packomania)",
          f"$\\Sigma r$ = {so:.9f}   (+6.29$\\times10^{{-4}}$)")
fig.suptitle("N=27: the previous record value vs the new record (both D1-symmetric; mirror-pairs share a color)",
             fontsize=12.5, fontweight="bold", y=0.98)
fig.text(0.5, 0.03,
         "Left: a packing at the previous record's $\\Sigma r$ (2.685350), found by our own solver; "
         "Cantrell's exact 2011/12 coordinates are not public. Right: our record packing.",
         ha="center", fontsize=8.6, color="#555555")
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(f"{HERE}/fig4_prev_vs_new.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"prev: {pa[0]} pairs + {pa[1]} on-axis;  ours: {pb[0]} pairs + {pb[1]} on-axis")
print("wrote", f"{HERE}/fig4_prev_vs_new.png")
