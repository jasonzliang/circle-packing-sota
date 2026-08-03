# sota/ours — our results vs the Packomania records

Our evolved solver's results for **N variable circles in a unit square, maximize Σr**, measured against
the Packomania `csqv` best-known values in [`../theirs/`](../theirs/). The complete per-N packings from
the sweep are in [`pck/`](pck/) and the raw table in [`results.csv`](results.csv).

## Headline (sweep N=2..100, from scratch, 120s/N, seed 1)

| outcome | count | N |
|---|---|---|
| **beat the record (WIN)** | **1** | **27** |
| tie (≤1e-6) | 26 | 2–24 and a few others |
| below (our under-search at large N) | 72 | mostly 25+ |

Full table: [`comparison.md`](comparison.md). The 72 "below" are **our compute budget, not the records'
ceiling** — large N need more search time than 120s (and, like N=27, some may be beatable baseline entries
with a harder re-run).

## The one win: N = 27

- **ours Σr = 2.685978684198**  vs  **Packomania record 2.685350025228**  →  **+6.29e-4 (+0.023%)**
- Strictly feasible (independently re-checked with `verify_pck.py`, below):
  27 circles, min wall slack +1.0e-12, min pairwise slack +7.2e-13 — no overlaps, all inside the square;
  the win margin is ~9 orders larger than the feasibility slack, so it is real, not numerical noise.
- Why this one is beatable: the N=27 record is a Packomania **baseline** entry (reference [1] = Specht's
  own `csqv` program), **not** one of the recent AlphaEvolve/AI-optimized entries (e.g. N=26 = 2.635983,
  which we do not beat).
- Everything for the win is in [`wins/`](wins/): [`csqv27.pck`](wins/csqv27.pck) (the canonical packing,
  Packomania format), `csqv27.seed1.json` (the full float64 config), and `csqv27.verify.txt` (the
  independent feasibility + record check). The same packing is also part of the complete set in
  [`pck/`](pck/).

Verify it yourself (no solver needed):
```bash
python3 ../../solver/verify_pck.py wins/csqv27.pck --record 2.685350025228
```

## How the seed works, and where the N=27 result's seed comes from

**Where the seed is set.** The sweep was launched as `run_sweep.py ... --seeds 1`, so for every N the
driver calls `pack.search(n, seed=1, budget=120)` and keeps the best strictly-feasible result. (`--seeds`
accepts a comma list; the best over all listed seeds is kept. We used a single seed, `1`.)

**What the seed does.** Inside `solver/pack.py`, `search(n, seed, budget)` creates **one** numpy PCG64
generator, `rng = np.random.default_rng(seed)`. That single `rng` drives *everything* stochastic: the
random initial layouts (`start_random`/`start_grid`), the basin-hopping perturbations
(`perturb`/`perturb_dual`), and the per-restart parameter choices. So the seed fully determines the random
stream. The loop repeatedly draws a fresh start (or perturbs the current best), runs the exact radius LP +
SLSQP refine + feasibility repair, and keeps the best — **until the wall-clock `budget` expires**.

**The N=27 result's seed is therefore `1`.** To regenerate it:
```bash
python3 ../../solver/pack.py -n 27 --seed 1 --time 120 -o out27.json
```

**Reproducibility caveat (important).** The budget is *wall-clock*, so the number of restarts that fit in
120s depends on machine speed and load. `(seed=1, 120s)` is deterministic in its RNG stream, but the
*result* depends on how many restarts complete — on a slower/busier machine seed 1 may land in a different
(possibly worse) local optimum. More time is **not** monotonically better either: a re-run at **seed 7,
300s** found a *worse* config (2.683803) because it explored a different basin. The takeaway: the search
is a stochastic multi-start, so a specific win is tied to `(seed, budget, machine)` — but the saved
packing (`wins/csqv27.pck`) is a fixed artifact that is strictly feasible and beats the record regardless
of how it was found, and anyone can confirm that with `verify_pck.py`.

## Contents

```
comparison.md            our full ours-vs-Packomania table (N=2..100)
results.csv              raw sweep output (N, Σr, feasibility)
wins/                    the record-beating N=27 result: csqv27.pck + seed1.json (full float64) + verify.txt
pck/csqv<N>.pck          all 99 packings, Packomania format (csqv27 also here, for completeness)
chase/                   harder re-runs of the closest near-misses (more time + seeds)
```
