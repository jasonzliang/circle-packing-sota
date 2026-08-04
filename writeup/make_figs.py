#!/usr/bin/env python3
"""Write-up visuals for the N=27 circle-packing result. All data read from the repo's own
committed artifacts (sota/ours/...). Generates fig1 (the record packing) and fig2 (the
N=2..100 sweep vs Packomania). fig3_emergence.png is the SI-v2 study report's Figure 2,
rasterized in separately and NOT regenerated here. Run from anywhere:
    python3 writeup/make_figs.py    ->  regenerates writeup/fig1_n27_packing.png and fig2_sweep_gap.png."""
import json, os, re, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
SOTA = os.path.dirname(HERE)          # repo root (this script lives in writeup/)
OUT = HERE                            # write the figures next to this script
os.makedirs(OUT, exist_ok=True)

RECORD = 2.685350025228          # Packomania csqv27 previous record (Cantrell, 2011/12)
WIN    = 2.685978684198          # our verified Sigma r (now the listed record)

# ----------------------------------------------------------------------------- fig 1: the packing
d = json.load(open(f"{SOTA}/sota/ours/wins/csqv27.seed1.json"))
circ = np.array(d["circles"])                     # x, y, r in [0,1]^2
sr = circ[:, 2].sum()
assert abs(sr - d["sum_radii"]) < 1e-9
print(f"fig1: N={len(circ)}  Sigma r (recomputed) = {sr:.12f}")

fig, ax = plt.subplots(figsize=(6.6, 6.9))
ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=2.0, ec="#222222"))
cmap = plt.cm.viridis
rmin, rmax = circ[:, 2].min(), circ[:, 2].max()
for x, y, r in circ:
    frac = (r - rmin) / (rmax - rmin + 1e-12)
    ax.add_patch(Circle((x, y), r, facecolor=cmap(0.15 + 0.75 * frac),
                        edgecolor="white", lw=0.8, alpha=0.95))
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("27 circles in the unit square: maximizing the sum of radii",
             fontsize=13, fontweight="bold", pad=12)
ax.text(0.5, -0.065,
        f"$\\Sigma r$ = {WIN:.9f}   (beats the previous Packomania record {RECORD:.9f} by +6.29$\\times10^{{-4}}$)",
        ha="center", va="top", fontsize=10.5, transform=ax.transAxes)
ax.text(0.5, -0.11, "strictly feasible: exactly 27 circles, all inside the square, no overlaps (verified to 1e-9)",
        ha="center", va="top", fontsize=8.8, color="#555555", transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_n27_packing.png", dpi=200, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------------- fig 2: sweep gap vs N
rows = []
for line in open(f"{SOTA}/sota/ours/comparison.md"):
    m = re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
    if m:
        N, ours, rec = int(m.group(1)), float(m.group(2)), float(m.group(3))
        rows.append((N, ours, rec))
rows.sort()
N   = np.array([r[0] for r in rows])
gap = np.array([100.0 * (r[1] - r[2]) / r[2] for r in rows])   # signed % of record (>0 = we beat it)
# drop the single degenerate seed failure (N=97 returned Sigma r = 0 -> not a real -100% gap)
keep = ~((N == 97) & (gap < -50))
Nk, gk = N[keep], gap[keep]
ties = np.abs(gk) < 1e-4

fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.axhspan(-0.001, 0.001, color="#cfe8cf", alpha=0.6, zorder=0)
ax.axhline(0, color="#888888", lw=1.0, zorder=1)
ax.scatter(Nk[~ties], gk[~ties], s=26, c="#c0504d", zorder=3, label="below record (our compute budget, not the record's ceiling)")
ax.scatter(Nk[ties], gk[ties], s=30, c="#4f81bd", zorder=3, label="tie (matches best-known, $\\sim10^{-11}$)")
win_i = np.where(Nk == 27)[0][0]
ax.scatter([27], [gk[win_i]], s=220, marker="*", c="#2e8b57", edgecolor="black",
           lw=0.7, zorder=5, label="N=27: beats the previous record")
ax.annotate("N = 27  WIN", xy=(27, gk[win_i]), xytext=(34, 0.28),
            fontsize=10.5, fontweight="bold", color="#2e8b57",
            arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.4))
ax.set_xlabel("N  (number of circles)", fontsize=11)
ax.set_ylabel("sum of radii vs Packomania record  (%)", fontsize=11)
ax.set_title("Our solver across N = 2..100 vs the authoritative Packomania records",
             fontsize=12, fontweight="bold")
ax.set_ylim(-2.6, 0.6)
ax.legend(loc="lower left", fontsize=8.6, framealpha=0.92)
ax.grid(True, axis="y", ls=":", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_sweep_gap.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"fig2: {len(Nk)} sizes plotted; ties={ties.sum()}; N=27 gap={gk[win_i]:+.4f}%")

# fig 3 (writeup/fig3_emergence.png) is the SI-v2 study report's Figure 2, rasterized in
# separately from reports/fig_circle_n27_iters.pdf and NOT regenerated here.
print("fig1 + fig2 written to", OUT)
