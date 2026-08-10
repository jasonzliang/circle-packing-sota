# solver-sm-radical-v6-n27: the original N=27 record solver

The **original** AI-generated circle-packing solver (lineage `sm-radical-v6`)
that set the **N=27** Packomania `csqv` best-known, Σr = **2.685978684198** (now
the _listed_ best-known, a tie; see the
[repo README's Verify section](../README.md#verify-the-results)).

## Provenance / entrypoint

Entry point `search(n, seed, budget)` in `pack.py` (alongside `container.py`,
`shape.py`, and the zero-tolerance `exact_check.py`). Needs **numpy** +
**scipy** (top-level `requirements.txt`).

## Algorithm

The radii-as-exact-LP inner layer that both solvers share (for _fixed_ centres
the optimal radii are an exact linear program, with notation `c_i`, `r_i`,
`d_ij = |c_i − c_j|`, `w_i` (wall distance)) is described in the
[n54 solver's algorithm](../solver-sm-radical-v6-n54/README.md#algorithm). This
solver optimises the centres around it, and additionally solves that LP for its
**duals** and applies a **provable contact-graph reduction**, both below.

`search(n, seed, budget)` is a multi-start over centres. ~65% of iterations hop
from the incumbent and the rest are fresh starts (random or staggered-row
grids); each candidate is run through `refine()`, checked for feasibility, and
scored.

**Outer optimiser: joint SLSQP + greedy hopping.**

- **Joint SLSQP over all of `(x, y, r)`** with analytic constraint Jacobians
  (`slsqp`), so the local solver trades radius against position in one step;
  `refine()` alternates SLSQP with the exact radius LP for up to 3 rounds,
  keeping the LP's radii whenever they beat SLSQP's.
- **Uniform subset hopping** (`perturb`): perturb a random 12/25/45% of circles
  at one of three jump scales, deflating their radii to 30%. Acceptance is
  **strictly greedy** (`if s > best_s`), no temperature, no Metropolis, so this
  is closer to iterated local search than to stochastic basin hopping.

**A provable, exact contact-graph reduction.** The formulation has `n(n−1)/2`
pair constraints, but a packing's contact graph is essentially planar (≤ `3n−6`
contacts). The reduction is **provable, not heuristic**: since
`r_i + r_j ≤ d_ij` and `r_j ≥ 0`, every `j` forces `r_i ≤ d_ij`, so

```text
u_i := min( w_i, min_{j≠i} d_ij )      is a valid bound on r_i in every feasible config
drop pair (i, j)  when  d_ij ≥ u_i + u_j
```

Impose `r_i ≤ u_i` as a variable bound (valid bounds never cut off the optimum)
and the dropped pairs are **implied by those two bounds**, so deletion loses
provably nothing (`valid_caps`, `live_pairs`). The bound and the drop rule are
one argument and must be used together. This applies to the **radius LP only**;
`--self-test` checks the reduced LP against a naive all-pairs reference and
finds the optimum preserved to float precision while keeping only a small
fraction of the rows. `u_i` is valid only for the centres it was computed from,
so it goes stale the moment SLSQP moves anything, and only the LP _value_ is
preserved, not the duals of dropped rows (which report `λ = 0`).

**Strict feasibility by construction, not by tolerance.** `repair()` projects
centres into the container, then applies **one uniform radius scale**
`min(1, s)`, the largest factor making every pair and wall constraint hold at
once, and shaves a further `1e-12`. Any candidate still violating a constraint
by more than `1e-9` is dropped, and `main()` refuses to emit an infeasible
config. Uniformity is cheap but brittle: `s = 0` zeroes every radius, which is
exactly how the N=97 sweep collapsed (see
[Known issues](../README.md#known-issues)). The float repair is separate from
the tolerance-free guarantee, which comes from `exact_check.py` (run for N=27 in
the repo README's
[zero-tolerance check](../README.md#zero-tolerance-check-in-exact-arithmetic)).

**LP duals as a search signal (opt-in, `--dual`).** The radius LP also returns
its duals: `λ_k` prices each tight contact (with `Σ_j λ_ij + μ_i = 1` by
complementary slackness), so `perturb_dual()` can sample contacts ∝ `λ` and
spend hops on the load-bearing ones. The LP is often degenerate, so `λ` is one
optimal dual vector, not a true gradient, and no measurement here shows `--dual`
beats uniform hopping.

**What the N=27 win used (and did not).** The record came from the **default
path only**: multi-start joint SLSQP, exact-LP radii with the contact-graph
reduction, uniform subset hopping, disks in the unit square. It did **not** use
`--dual`, the trust-region QP (`--sparse`, off: measured no faster), or the
polytope joint-LP path; the container/shape abstractions
(`--container`/`--shape`) and `--self-test` (dual identity, zero duality gap,
reduction-vs-naive-LP, a proved `√n/2` optimum) are exercised but contributed
nothing to the result.
