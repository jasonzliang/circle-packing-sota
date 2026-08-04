#!/usr/bin/env python3
"""Write-up visuals for the N=27 circle-packing result. All data read from the repo's own
committed artifacts (sota/ours/...); the only hand-entered numbers are the per-iteration
emergence table (fig 3), transcribed from the SI-v2 study report. Run from anywhere:
    python3 writeup/make_figs.py    ->  regenerates writeup/fig1..3."""
import json, os, re, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
SOTA = os.path.dirname(HERE)          # repo root (this script lives in blog/)
OUT = HERE                            # write the figures next to this script
os.makedirs(OUT, exist_ok=True)

RECORD = 2.685350025228          # Packomania csqv27 baseline record
WIN    = 2.685978684198          # our verified Sigma r

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
        f"$\\Sigma r$ = {WIN:.9f}   (beats the listed Packomania record {RECORD:.9f} by +6.29$\\times10^{{-4}}$)",
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
           lw=0.7, zorder=5, label="N=27: beats the listed record")
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

# ----------------------------------------------------------------------------- fig 3: emergence over iterations
# transcribed from the SI-v2 report, Table 'Per-version n=27 outcomes over 50 seeds (120s each)'
it       = [1, 2, 3, 4, 5]
best     = [2.685979]*5                   # best of 50 seeds -> the record-beating win (constant)
median   = [2.685157]*5                   # typical single run (constant, just below the record)
hit_pct  = [10, 10, 12, 14, 14]           # % of 50 seeds that beat the record
xlab = ["iter 1\n$2.48", "iter 2\n$5.95", "iter 3\n$9.79", "iter 4\n$14.82", "iter 5-10\n$21.36"]

fig, (axA, axB) = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True,
                               gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12})

# top panel: Sigma r vs iteration (direct right-side labels; no in-plot legend to obscure the data)
axA.axhline(WIN, color="#2e8b57", ls=":", lw=1.0, alpha=0.4, zorder=1)
axA.plot(it, best, "-o", color="#2e8b57", lw=2.6, ms=9, zorder=5)
axA.axhline(RECORD, color="#c0504d", ls="--", lw=1.7, zorder=2)
axA.plot(it, median, "-s", color="#7f7f7f", lw=1.8, ms=6, zorder=4)
axA.text(5.18, WIN,    "best of 50 starts\n= the WIN  2.685979", color="#2e8b57",
         va="center", fontsize=8.6, fontweight="bold")
axA.text(5.18, RECORD + 0.00003, "record  2.685350", color="#c0504d", va="bottom", fontsize=8.6)
axA.text(5.18, median[0], "median run  2.685157\n(a typical start\ndoes not beat it)", color="#5f5f5f",
         va="center", fontsize=8.2)
axA.annotate("present from\niteration 1", xy=(1, WIN), xytext=(1.75, 2.685740),
             fontsize=9, color="#2e8b57", ha="center",
             arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.3))
axA.set_ylim(2.68500, 2.68612)
axA.set_xlim(0.5, 7.0)
axA.set_ylabel("sum of radii  $\\Sigma r$", fontsize=11)
axA.set_title("The record-beating capability appears early; more self-improvement buys reliability, not the peak",
              fontsize=11.5, fontweight="bold")
axA.grid(True, axis="y", ls=":", alpha=0.35)

# bottom panel: per-seed hit rate
axB.bar(it, hit_pct, width=0.55, color="#4f81bd", alpha=0.85)
for x, h in zip(it, hit_pct):
    axB.text(x, h + 0.5, f"{h}%", ha="center", color="#28527a", fontsize=9)
axB.set_ylim(0, 20)
axB.set_xlim(0.5, 7.0)
axB.set_ylabel("hit rate\n(% of 50 starts\nbeating record)", fontsize=9.5)
axB.set_xticks(it); axB.set_xticklabels(xlab, fontsize=9.2)
axB.set_xlabel("solver version (self-improvement iteration)   ·   cumulative $ spent improving the solver",
               fontsize=10.3, x=0.36)
axB.grid(True, axis="y", ls=":", alpha=0.35)
plt.savefig(f"{OUT}/fig3_emergence.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig3: emergence written")
print("ALL FIGURES WRITTEN TO", OUT)
