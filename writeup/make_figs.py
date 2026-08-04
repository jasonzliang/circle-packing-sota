#!/usr/bin/env python3
"""Write-up visuals for the N=27 circle-packing result. Data from the repo's own committed
artifacts (sota/ours/comparison.md) plus the SI-v2 study's per-iteration emergence numbers.
Generates:
  fig2_sweep_gap.png  -- our solver vs Packomania across N=2..100
  fig3_emergence.png  -- when a record-capable N=27 solver emerges (also saved as .pdf, the twin
                         used as Figure 2 of the SI-v2 circle-packing report)
(fig4_prev_vs_new.png is built by make_compare_fig.py.)
    python3 writeup/make_figs.py"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SOTA = os.path.dirname(HERE)
OUT = HERE
RECORD = 2.685350025228          # Packomania csqv27 previous record (Cantrell, 2011/12)
WIN    = 2.685978684198          # our verified Sigma r (now the listed record)

# ----------------------------------------------------------------------------- fig 2: sweep gap vs N
rows = []
for line in open(f"{SOTA}/sota/ours/comparison.md"):
    m = re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
rows.sort()
N = np.array([r[0] for r in rows])
gap = np.array([100.0 * (r[1] - r[2]) / r[2] for r in rows])
keep = ~((N == 97) & (gap < -50))          # drop the single degenerate seed failure at N=97
Nk, gk = N[keep], gap[keep]
ties = np.abs(gk) < 1e-4

fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.axhspan(-0.001, 0.001, color="#cfe8cf", alpha=0.6, zorder=0)
ax.axhline(0, color="#888888", lw=1.0, zorder=1)
ax.scatter(Nk[~ties], gk[~ties], s=26, c="#c0504d", zorder=3, label="below record (our compute budget, not the record's ceiling)")
ax.scatter(Nk[ties], gk[ties], s=30, c="#4f81bd", zorder=3, label="tie (matches best-known, $\\sim10^{-11}$)")
wi = np.where(Nk == 27)[0][0]
ax.scatter([27], [gk[wi]], s=220, marker="*", c="#2e8b57", edgecolor="black", lw=0.7, zorder=5,
           label="N=27: beats the previous record")
ax.annotate("N = 27  WIN", xy=(27, gk[wi]), xytext=(34, 0.28), fontsize=10.5, fontweight="bold",
            color="#2e8b57", arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.4))
ax.set_xlabel("N  (number of circles)", fontsize=11)
ax.set_ylabel("sum of radii vs Packomania record  (%)", fontsize=11)
ax.set_title("Our solver across N = 2..100 vs the authoritative Packomania records", fontsize=12, fontweight="bold")
ax.set_ylim(-2.6, 0.6)
ax.legend(loc="lower left", fontsize=8.6, framealpha=0.92)
ax.grid(True, axis="y", ls=":", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_sweep_gap.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"fig2: {len(Nk)} sizes; ties={ties.sum()}; N=27 gap={gk[wi]:+.4f}%")

# ----------------------------------------------------------------------------- fig 3: emergence
# Data: SI-v2 study report, Table 'Per-version n=27 outcomes over 50 seeds (120 s each)'.
# best-of-50 = the win (constant); `worst` is the lower edge of the 50-seed spread; seed 1 lands in
# the dominant near-record basin (2.685157) for iters 1-3, then reaches the win from iter 4.
it    = list(range(1, 11))
best  = [WIN] * 10
worst = [2.681726, 2.681726, 2.681541, 2.681508, 2.681508, 2.681508, 2.681508, 2.681508, 2.681508, 2.681508]
seed1 = [2.685157, 2.685157, 2.685157] + [WIN] * 7
hit   = [10, 10, 12, 14, 14, 14, 14, 14, 14, 14]

fig, (axA, axB) = plt.subplots(2, 1, figsize=(8.8, 6.8), sharex=True,
                               gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.13})
axA.fill_between(it, worst, best, color="#cfcfcf", alpha=0.7, zorder=0, label="seed spread (50 seeds)")
axA.axhline(RECORD, color="#000000", ls=":", lw=1.6, zorder=2)
axA.text(10.15, RECORD, "Packomania\nrecord", fontsize=8.6, va="center")
axA.plot(it, best, "-o", color="#1f6fb2", lw=2.4, ms=7, zorder=5, label="best of 50 seeds (capability)")
axA.plot(it, seed1, "--s", color="#c0392b", lw=2.2, ms=7, zorder=5, label="seed 1 only (sota-repo condition)")
axA.annotate("SOTA-capable at iter 1\n(best of seeds = the win; ~$2.48)", xy=(1, WIN), xytext=(1.35, 2.6835),
             color="#1f6fb2", fontsize=9, arrowprops=dict(arrowstyle="->", color="#1f6fb2", lw=1.2))
axA.annotate("seed-1 win from iter 4\n(~$14.82 cum.)", xy=(4, WIN), xytext=(4.7, 2.6840),
             color="#c0392b", fontsize=9, arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
axA.set_ylim(2.6810, 2.6864)
axA.set_xlim(0.7, 11.0)
axA.set_ylabel("N=27  $\\Sigma r$  (higher = better)", fontsize=10.5)
axA.set_title("When does a SOTA-capable N=27 solver emerge?   (120 s/seed, 50 seeds)", fontsize=12, fontweight="bold")
axA.legend(loc="lower right", fontsize=8.4, framealpha=0.93)
axA.grid(True, axis="y", ls=":", alpha=0.3)

axB.bar(it, hit, width=0.6, color=["#2e8b57" if h >= 14 else "#c0504d" for h in hit], alpha=0.9)
for x, h in zip(it, hit):
    axB.text(x, h + 0.4, f"{h}%", ha="center", fontsize=8.6, color="#333333")
axB.set_ylim(0, 17)
axB.set_xticks(it)
axB.set_ylabel("% of 50 seeds\nbeating the record", fontsize=9.2)
axB.set_xlabel("solver iteration", fontsize=10.3)
axB.grid(True, axis="y", ls=":", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_emergence.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/fig3_emergence.pdf", bbox_inches="tight")   # PDF twin for the report's Figure 2
plt.close()
print("fig3: emergence written (png + pdf)")
print("fig2 + fig3 written to", OUT)
