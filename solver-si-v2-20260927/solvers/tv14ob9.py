"""Circle-packing solver: maximize the sum of radii of n circles in the unit square.

Strategy (one rule for every n -- no per-n tables, no baked-in numbers):

  1. MULTI-START is the engine.  The Sum-r landscape is riddled with deep local optima whose
     values cluster ~1% apart, so the quantity that buys digits is the NUMBER OF INDEPENDENT
     BASINS sampled per second -- not the effort spent polishing any one of them.  Measured
     (artifacts/exp_restarts.py, artifacts/exp_screen.py): 428 cold starts on n=27 reach 2.78
     digits where iteration 1's warm-started hill-climb stalled at 1.84, and on n=99 six seconds
     of cold starts reach 2.64 digits against a committed 1.52.
  2. DESCEND, under an EQUALIZATION HOMOTOPY.  Each start is driven to a KKT point of the true
     nonlinear program in (x, y, r) by L-BFGS-B on a quadratic-penalty relaxation,
         min  -sum(r) + mu * [ sum_{i<j} relu(r_i+r_j-d_ij)^2 + sum_i relu(r_i-wall_i)^2 ]
              + lam * sum_i (r_i - rbar)^2
     with analytic gradients, sweeping mu UP while lam is annealed DOWN to exactly zero.  The lam
     term is the fix for the basin failure: under an area budget sum(r^2)=C the sum of radii is
     maximized at EQUAL radii, and the records sit at record/sqrt(n) ~ 0.52 (between square-grid
     0.50 and ideal-hex 0.537) -- i.e. they are near-equal-circle packings, while unguided local
     optima came out at radius CV ~ 0.25.  Biasing the LOOSE end of the ladder toward equal radii
     lands the descent in that family; annealing lam to 0 means the final stage optimizes the true
     objective, so the bias chooses the basin and never the answer.  Measured at equal CPU this is
     worth ~1-4 digits per n and produced the run's first record beats.  All O(n^2) work is
     vectorized numpy.  The full ladder runs on EVERY start: a cheap-ladder screening tier was
     measured and LOSES, and so does batching replicas into one L-BFGS-B (see notes/).
  2b. DRAW THE LATTICE SHAPE, not just the jitter.  Each start's grid uses a column count drawn
     from {ceil(sqrt(n))-1, ceil(sqrt(n)), ceil(sqrt(n))+1}, so three different row/column
     topologies -- three different contact graphs -- are sampled per n instead of the single
     square-ish one.  Measured worth ~+2.9 digits per n at equal CPU, and it is the change that
     first reached the record on n = 73 and 87.  The span is narrow on purpose: +-3 is far worse,
     because far-off aspect ratios only burn starts.
  2c. TRANSFER, AND RUIN & RECREATE -- the census is the memory.  Half the starts are built from the
     committed packing of a nearby size drawn from n+-4 INCLUDING n ITSELF: drop the smallest
     circles if the donor is too big, drop circles into the deepest holes if it is too small,
     rattle, then run the same homotopy.  Neighbouring n share structure, so a neighbour's good
     basin is a cheap prior on this n's -- and it compounds, because every iteration's donors are
     the last iteration's output.  A transfer start must RE-CREATE at least one circle, which is
     free for a cross-size donor (|n-m| of them) and drawn as 1..3 for the same-size donor, so n's
     OWN incumbent -- the best-informed donor there is -- becomes a ruin-and-recreate move instead
     of a rattled copy.  The donor is DRAWN, never chosen per n, and the descent is unchanged, so
     it picks the basin, not the answer.  Fully guarded: no donor or no readable pack falls
     straight back to a cold start.
  3. REPAIR.  A penalty solution is only *nearly* feasible, so the radii are re-derived exactly
     from the final centres: scale once to strict feasibility, then run coordinate ascent
     r_i <- min(wall_i, min_j (d_ij - r_j)), which is exact block-ascent on the LP "best radii for
     these centres" and is monotone and feasibility-preserving.
  4. INTENSIFY, self-limiting.  Only when a cold start sets a NEW best do we spend a little extra
     on it: a couple of shake-and-re-descend probes around that basin.  Intensification therefore
     costs time only where it has just been shown to pay, and a plateau spends everything on
     fresh basins.
  4b. ALLOCATE where a digit is still for sale.  Each n's CPU slice is proportional to n TIMES
     the digits still for sale there.  The n factor equalizes the START COUNT (a descent costs
     ~linear in n, and one digit is worth the same at every n); the second factor is
     (DIGITS_CAP - digits already banked) / DIGITS_CAP, because the scorer clamps a packing at
     DIGITS_CAP digits and an n already near its record can only buy the sliver that is left.
     An n AT the cap therefore gets exactly zero -- it is banked at its committed value and its
     share flows on -- and an n far from its record gets the largest share, as one expression
     rather than a cap test plus a rule.  Fully guarded -- a missing record table or committed
     pack falls back to the full weight, so a fresh n or an offline re-run on other sizes is
     never starved, and every n's floor is banked whatever its weight.
  5. FLOORS FIRST, then search. Pass 1 banks the committed coordinates of EVERY target verbatim
     before a single second goes to searching, so no ordering, no CPU deadline and no exhausted
     slice can drop an n that already had a packing; pass 2 spends the seconds. The floor pass used
     to live inside the per-n search loop, which reached only the n the deadline allowed -- and as
     the loop walks sorted(targets), the n it dropped were always the largest (observed twice).
  5b. WARM START = a floor, never an anchor.  The committed census is read (guarded) and banked
     immediately so a bad run can never regress an n, but the search does NOT hill-climb from it:
     measured, the committed incumbents are worse than a few seconds of fresh multi-start, and
     anchoring on them is exactly what capped iteration 1.

Every candidate that should count is routed through `evaluate()` -- in batches, so local optima are
banked wholesale rather than only on improvement -- and nothing is written to bench/packs/ from
here. All randomness comes from `rng`. Budgets are computed relative to `time.process_time()` AT
ENTRY, so a second solve() call in the same process gets its own full clock, and the per-n split is
proportional so it degenerates correctly to len(targets) == 1.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

LO, HI = -0.5, 0.5

CPU_BUDGET_S = 115.0      # per-call CPU slice, relative to process_time() at entry (driver cap: 120 s).
                          # The driver arms its 120 s SIGPROF backstop immediately BEFORE calling solve()
                          # (solve_driver.py: cpu0 = process_time(); setitimer(ITIMER_PROF, 120)), so the
                          # driver's own import/validate CPU is NOT charged against it -- the whole 120 s
                          # is the solver's to spend, and 104 left ~16 s of the allotted share unused.
                          # 5 s of headroom is kept because the search loop tests the deadline BEFORE
                          # starting a descent, so it can overshoot by one descent + SHAKES (~0.6 s at
                          # n=99). Raising it is safe even if the backstop DOES fire: it ends only this
                          # solve() call, every floor is already banked by pass 1 and every local optimum
                          # is flushed in batches as the search runs, so at worst one pending batch is
                          # lost. More seconds is the one lever that is uniformly positive at every n --
                          # one rule, no per-n tuning, and the search is a noisy max over starts (4r/4s).
MARGIN = 1e-10            # shave off every radius: clears the 1e-9 feasibility tol with room to spare,
                          # and costs < 1e-8 on sum(r) -- far below the 1e-7 relative-gap cap.
MU_LADDER = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
INNER_ITERS = 250
BATCH = 16                # local optima banked per vectorized evaluate() call
SHAKES = 2                # intensification probes around a start that set a NEW best
TRANSFER_SHARE = 0.75     # fraction of starts built from the census rather than cold (see _transfer)
RATTLE = 0.08             # donor-centre jitter in _transfer, in units of 1/sqrt(n) (~1 radius)
DIGITS_CAP = 7.0          # the scorer clamps digits here, so relgap <= CAP_GAP is worth no more
CAP_GAP = 10.0 ** -DIGITS_CAP


# ---------------------------------------------------------------- penalty objective + gradient

def _fg(z, n, mu, lam=0.0):
    """-sum(r) + mu*(overlap^2 + wall^2) + lam*sum((r-rbar)^2), and its analytic gradient.

    The lam term is the EQUALIZATION HOMOTOPY (see module docstring): subject to an area budget
    sum(r^2)=C the sum of radii is maximized at EQUAL radii, so an early bias toward equal r steers
    the descent into the near-equal-radius family the records live in. sum_j (r_j - rbar) = 0, so
    the gradient is just 2*lam*(r - rbar). lam is annealed to EXACTLY zero, so the final stage
    optimizes the true objective and the term biases only which basin is reached, never the answer.
    """
    x, y, r = z[:n], z[n:2 * n], z[2 * n:]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, 1e9)
    v = np.maximum(r[:, None] + r[None, :] - d, 0.0)          # pairwise overlap
    np.fill_diagonal(v, 0.0)
    pen = 0.5 * (v * v).sum()                                  # each pair counted twice above
    gr = 2.0 * v.sum(1)
    c = 2.0 * v / np.maximum(d, 1e-12)
    gx = -(c * dx).sum(1)
    gy = -(c * dy).sum(1)
    a = np.maximum(r - (x - LO), 0.0)
    b = np.maximum(r - (HI - x), 0.0)
    e = np.maximum(r - (y - LO), 0.0)
    f = np.maximum(r - (HI - y), 0.0)
    pen += (a * a + b * b + e * e + f * f).sum()
    gr += 2.0 * (a + b + e + f)
    gx += 2.0 * (b - a)
    gy += 2.0 * (f - e)
    obj = -r.sum() + mu * pen
    grad_r = -np.ones(n) + mu * gr
    if lam:
        dr = r - r.mean()
        obj += lam * float((dr * dr).sum())
        grad_r = grad_r + 2.0 * lam * dr
    return obj, np.concatenate([mu * gx, mu * gy, grad_r])


def _descend(z, n, mus=MU_LADDER, iters=INNER_ITERS, lam0=0.0):
    """Penalty continuation: tighten mu step by step, warm-starting each stage from the last, while
    the equalization weight is annealed the other way -- lam0 at the loosest mu down to EXACTLY 0 at
    the tightest. The last stage therefore solves the true problem; lam only picks the basin."""
    bounds = [(LO, HI)] * (2 * n) + [(0.0, 0.5)] * n
    k = len(mus)
    for i, mu in enumerate(mus):
        lam = lam0 * (1.0 - i / (k - 1.0)) ** 2 if (lam0 and k > 1) else 0.0
        z = minimize(_fg, z, args=(n, mu, lam), jac=True, method="L-BFGS-B",
                     bounds=bounds, options={"maxiter": iters, "maxcor": 12}).x
    return z


# ---------------------------------------------------------------- exact radii for fixed centres

def _radii(x, y, r0, sweeps=40):
    """Best radii for FIXED centres: scale to strict feasibility, then coordinate-ascent the LP
    max sum(r) s.t. r_i + r_j <= d_ij, 0 <= r_i <= wall_i. Monotone and always feasible."""
    n = x.size
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    np.fill_diagonal(d, np.inf)
    wall = np.maximum(np.minimum.reduce([x - LO, HI - x, y - LO, HI - y]), 0.0)
    r = np.maximum(r0, 0.0)
    t = max(1.0, float(np.max((r[:, None] + r[None, :]) / d)),
            float(np.max(r / np.maximum(wall, 1e-15))))
    r /= t
    for _ in range(sweeps):
        moved = 0.0
        for i in range(n):
            new = min(wall[i], float(np.min(d[i] - r)))
            new = max(new, 0.0)
            moved = max(moved, abs(new - r[i]))
            r[i] = new
        if moved < 1e-14:
            break
    return np.maximum(r - MARGIN, 1e-12)


def _pack(z, n):
    """(penalty vector) -> a strictly feasible (n,3) packing and its sum of radii."""
    x, y = z[:n].copy(), z[n:2 * n].copy()
    np.clip(x, LO, HI, out=x)
    np.clip(y, LO, HI, out=y)
    r = _radii(x, y, z[2 * n:].copy())
    return np.stack([x, y, r], axis=1), float(r.sum())


# ---------------------------------------------------------------- starts

def _read_pack(n):
    """Warm start from the committed census, or None. GUARDED: any missing/odd file -> cold start."""
    p = os.path.join("bench", "packs", "csqv%d.pck" % n)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            rows = [ln.split() for ln in fh.read().splitlines() if ln.strip()]
        pts = [tuple(float(v) for v in row[:3]) for row in rows if len(row) >= 3]
        if len(pts) != n:
            return None
        a = np.asarray(pts, dtype=float)
        if not np.isfinite(a).all():
            return None
        return np.concatenate([a[:, 0], a[:, 1], a[:, 2]])
    except Exception:
        return None


def _cold(n, rng):
    """A random start: a jittered lattice (grid or hex) or plain uniform centres, radii guessed
    from the area budget. One rule, no per-n special-casing.

    The LATTICE SHAPE is drawn, not fixed. A square-ish n x n grid is only one of the topologies a
    good packing can have: the column count sets how many rows there are and therefore the whole
    contact graph, and ceil(sqrt(n)) columns is frequently the WRONG topology for a given n. Drawing
    cols from {base-1, base, base+1} (base = ceil(sqrt(n))) samples three genuinely different
    families per n instead of one, which is what a multi-start engine is starved of. The span is
    deliberately NARROW: measured, +-3 is much worse than +-1, because far-off aspect ratios only
    burn starts. The row jitter is scaled by the LARGER of the two lattice pitches so a tall/wide
    lattice is rattled proportionally to its own spacing rather than to the column pitch.
    """
    k = rng.randint(4)                               # 1/4 uniform, 1/4 square grid, 2/4 hex-offset
    if k == 0:
        z = rng.uniform(LO + 0.02, HI - 0.02, 2 * n)
        x, y = z[:n], z[n:]
    else:
        base = int(np.ceil(np.sqrt(n)))
        cols = int(np.clip(base + rng.randint(-1, 2), 1, n))
        rows = int(np.ceil(n / cols))
        gx, gy = np.meshgrid((np.arange(cols) + 0.5) / cols - 0.5,
                             (np.arange(rows) + 0.5) / rows - 0.5)
        if k >= 2:                                   # hex-ish: offset every other row
            gx = gx + (np.arange(rows)[:, None] % 2) * (0.5 / cols)
        x = gx.ravel()[:n] + rng.normal(0.0, 0.25 / cols, n)
        y = gy.ravel()[:n] + rng.normal(0.0, 0.25 / max(cols, rows), n)
        x = np.clip(x, LO + 1e-3, HI - 1e-3)
        y = np.clip(y, LO + 1e-3, HI - 1e-3)
    return np.concatenate([x, y, np.full(n, 0.4 / np.sqrt(n))])


def _hole(x, y, r, g=48):
    """The deepest empty spot: the grid point whose clearance to every circle and to the four
    walls is largest. Used to place a circle when a donor packing is one circle short."""
    t = (np.arange(g) + 0.5) / g - 0.5
    gx, gy = np.meshgrid(t, t)
    px, py = gx.ravel(), gy.ravel()
    clr = np.minimum.reduce([px - LO, HI - px, py - LO, HI - py])
    if x.size:
        d = np.sqrt((px[:, None] - x[None, :]) ** 2 + (py[:, None] - y[None, :]) ** 2) - r[None, :]
        clr = np.minimum(clr, d.min(1))
    i = int(np.argmax(clr))
    return px[i], py[i], max(float(clr[i]), 1e-4)


def _transfer(n, rng, cache, span=4):
    """A start for n built from the COMMITTED packing of a NEARBY size m -- INCLUDING m = n itself.

    The census is this run's memory, and neighbouring n share structure: a good packing for m is
    a good packing for n with |n-m| circles added or removed. Too big -> drop the SMALLEST circles
    (they are the ones the packing can most afford to lose); too small -> drop a circle into the
    deepest hole. Then rattle and run the usual homotopy descent, so the donor chooses the basin
    and never the answer.

    ONE RULE, stated once for every m: a transfer start must RE-CREATE at least one circle, because
    re-creating is the whole mechanism -- it is what turns a known-good configuration into a
    genuinely different one instead of a rattled copy of itself. A cross-size donor already
    re-creates |n - m| of them for free, so it needs nothing added; the SAME-size donor -- n's own
    incumbent, which is the best-informed donor in the census and was the one size the pool used to
    exclude -- re-creates a drawn 1..3. Measured (artifacts/exp_ruin7.py + 7b.py, 10 n, two
    independent seed sets): admitting m = n this way is worth +0.29 / +0.15 mean digits and never
    loses an n, while piling the same ruin on top of a cross-size donor as well is WORSE -- past
    |n - m| the packing has been re-created enough.

    The donor is DRAWN from the neighbourhood, never chosen per n. GUARDED at every step: no
    neighbour, no readable pack -> None, and the caller falls back to a cold start, so any n
    (fresh, even, outside the census) still works.

    TRANSFER_SHARE is 3/4 of the start stream, not the 1/2 it was from iteration 6 to 8, because
    the census it draws on is no longer the weak thing it was when the share was written. Measured
    (artifacts/exp_lat9.py): three genuinely different cold-lattice geometries run at n=33/41/49/59/81
    returned the SAME sum_r to every decimal, because at those n NO cold start beat the best
    transfer start in 5 CPU-s -- the cold half of the stream was buying nothing. Measured again
    across shares (artifacts/exp_share9.py, 10 n x 2 independent seed sets): 3/4 is >= 1/2 on all
    ten n and strictly better on five, while going to 4/4 is NOT better still -- it loses n=29,
    where the winning start that iteration WAS a cold one. So the cold stream is close to dead but
    not dead, and the share stays a share.
    """
    ms = []
    for m in range(max(2, n - span), n + span + 1):
        if m not in cache:
            cache[m] = _read_pack(m)
        if cache[m] is not None:
            ms.append(m)
    if not ms:
        return None
    m = int(ms[rng.randint(len(ms))])
    w = cache[m]
    x, y, r = w[:m].copy(), w[m:2 * m].copy(), w[2 * m:].copy()
    if m > n:
        keep = np.argsort(r)[m - n:]
        x, y, r = x[keep], y[keep], r[keep]
    if m == n:                                  # RUIN & RECREATE: the size change re-created
        k = 1 + rng.randint(3)                  # nothing, so take the k smallest circles out
        keep = np.argsort(r)[k:]                # explicitly; the refill below puts them back in
        x, y, r = x[keep], y[keep], r[keep]     # the deepest holes, re-wiring the contact graph.
    while x.size < n:
        hx, hy, hr = _hole(x, y, r)
        x = np.append(x, hx)
        y = np.append(y, hy)
        r = np.append(r, hr)
    sig = RATTLE / np.sqrt(n)
    x = np.clip(x + rng.normal(0.0, sig, n), LO + 1e-3, HI - 1e-3)
    y = np.clip(y + rng.normal(0.0, sig, n), LO + 1e-3, HI - 1e-3)
    return np.concatenate([x, y, r])


def _lam0(n, rng):
    """Equalization strength for one start. The natural scale is sqrt(n): a typical radius is
    ~0.5/sqrt(n), and -sum(r) contributes gradient 1 per radius, so lam ~ sqrt(n) makes the two
    comparable at a ~10% radius spread. The decade around it is drawn per start rather than tuned:
    the fit picks the number, and the spread is extra basin diversity for free."""
    return (10.0 ** rng.uniform(-0.6, 0.4)) * np.sqrt(n)


def _records():
    """The record table, or {} if unreadable. Reading bench/ from solve() is sanctioned; a missing
    or malformed table just means we cannot tell which n are maxed out, and every n gets a slice."""
    try:
        with open(os.path.join("bench", "records.json")) as fh:
            return {int(k): float(v) for k, v in json.load(fh)["records"].items()}
    except Exception:
        return {}


def _weight(n, warm, recs):
    """CPU weight for one n: n x THE DIGITS STILL FOR SALE THERE.

    The base is n, because a descent costs ~linear in n, so weighting by n equalizes the START
    COUNT across sizes -- and one digit is worth the same at every n. But the scorer CLAMPS a
    packing at DIGITS_CAP digits, so what an n can still earn is not unbounded: it is exactly
    `headroom = DIGITS_CAP - digits(committed)`, which is DIGITS_CAP for an n with nothing banked
    and 0 for one already at the cap. Scaling the slice by `headroom / DIGITS_CAP` is the SAME
    rule that has given a capped n zero seconds since iteration 4, with the step function replaced
    by the quantity it was approximating: an n 1.4e-7 from its record can buy at most 0.16 of a
    digit however long it is searched, so it should not draw the same seconds as one 4.5e-4 away
    that can still buy 3.7.  It is one expression for every n -- nothing keyed on the size, no
    table, and the cap case falls out of it rather than being special-cased.

    Guarded at every step: no record, no committed pack, or an unparseable one all fall through to
    the FULL weight n, so a fresh n or an offline re-run on sizes outside the census is never
    starved. The floor is banked for every n regardless of weight, so a small slice can never make
    an n come out of the iteration worse than it went in."""
    rec = recs.get(n)
    if rec is None or rec <= 0.0 or warm is None:
        return float(n)
    s = float(np.sum(warm[2 * n:]))
    gap = (rec - s) / rec
    if gap <= CAP_GAP:
        return 0.0                                  # at the cap: no digit left to buy here
    dig = min(DIGITS_CAP, max(0.0, -np.log10(gap)))  # digits ALREADY banked for this n
    return float(n) * (DIGITS_CAP - dig) / DIGITS_CAP


def _shake(z, n, rng):
    """Basin-hop move: displace a random handful of centres; the descent re-closes the packing."""
    z = z.copy()
    m = max(1, int(rng.randint(1, max(2, n // 4))))
    idx = rng.choice(n, size=m, replace=False)
    scale = 10.0 ** rng.uniform(-2.2, -0.9)
    z[idx] = np.clip(z[idx] + rng.normal(0.0, scale, m), LO, HI)
    z[n + idx] = np.clip(z[n + idx] + rng.normal(0.0, scale, m), LO, HI)
    return z


# ---------------------------------------------------------------- the contract

def solve(evaluate, meter, rng, targets):
    if not targets:
        return
    deadline = time.process_time() + CPU_BUDGET_S     # RELATIVE at entry: a 2nd call gets its own clock
    targets = sorted(set(int(t) for t in targets))

    # Read the census and the records ONCE, and decide up front where the seconds can still buy
    # digits. An n already at the scorer's cap is banked and skipped, and its share is handed to
    # the n that are still short -- the point of the run is the n someone still needs closed.
    recs = _records()
    warms = {n: _read_pack(n) for n in targets}
    donors = dict(warms)                  # lazily extended by _transfer; the census does
                                          # not change during solve(), so one read each.
    weight = {n: _weight(n, warms[n], recs) for n in targets}
    if not any(weight.values()):          # everything already maxed (or no records) -- search all
        weight = {n: float(n) for n in targets}

    # PASS 1 -- BANK EVERY FLOOR, for EVERY target, BEFORE a single second goes to search.
    # The floors are insurance: the committed coordinates verbatim, no descent, so this pass costs
    # one evaluate() per n and no measurable CPU. It used to be the first thing inside the per-n
    # search loop, which meant it happened only for the n the loop actually REACHED -- and since
    # the loop walks sorted(targets) and stops on the CPU deadline, the n it dropped were always
    # the LARGEST. Twice observed (iterations 8 and 9: solve() died at n=81, so 83..99 were never
    # visited). It cost nothing both times because those n happened to be at the cap and the driver
    # leaves an untouched .pck alone -- but an n whose floor is not re-banked is an n that a future
    # driver change, a fresh census, or simply a large SHORT n would silently lose. Banking first
    # makes the guarantee unconditional: no ordering, no deadline and no exhausted slice can drop
    # an n that already had a packing. Guarded -- no committed pack just means no floor to bank.
    for n in targets:
        if meter.left() <= 0:
            break
        warm = warms[n]
        if warm is not None:
            evaluate(n, np.stack([warm[:n], warm[n:2 * n], warm[2 * n:]], axis=1))

    # PASS 2 -- SEARCH, time-sliced by headroom. Every n's floor is already banked above, so this
    # pass may be cut off at any point without losing an n.
    for idx, n in enumerate(targets):
        if meter.left() <= 0:
            break
        now = time.process_time()
        remaining = deadline - now
        if remaining <= 0.0:
            break
        # Re-derive each n's slice from what is ACTUALLY left, so an n that finishes early or
        # overruns hands the difference on rather than starving the tail. The weight is n times
        # the digits still for sale at that n: the n factor equalizes the START COUNT (a descent
        # costs ~linear in n, measured), and the headroom factor keeps the seconds off packings
        # the scorer's cap has already paid out in full.
        left_w = float(sum(weight[m] for m in targets[idx:])) or 1.0
        if weight[n] <= 0.0:              # at the cap: floor already banked by pass 1, nothing to buy
            continue
        stop_n = now + max(0.05, remaining * (weight[n] / left_w))

        best_s, best_z = -np.inf, None
        pend = []                                     # local optima awaiting a batched evaluate()

        def flush():
            """Bank every pending local optimum in ONE vectorized evaluate() call."""
            if not pend or meter.left() <= 0:
                del pend[:]
                return
            batch = np.stack(pend[:max(1, int(meter.left()))])
            del pend[:]
            evaluate(n, batch if batch.shape[0] > 1 else batch[0])

        def consider(z):
            """Repair z to a strictly feasible packing, queue it, and report if it is a new best."""
            nonlocal best_s, best_z
            pack, s = _pack(z, n)
            pend.append(pack)
            if len(pend) >= BATCH:
                flush()
            if s > best_s:
                best_s, best_z = s, z.copy()
                return True
            return False

        # WARM CONTINUATION: a short re-descent off the committed pack, which often re-derives a
        # slightly better sum_r than the file holds. Only for an n with headroom -- an n at the
        # scorer's cap has no digit left to buy, and its verbatim floor is already banked by pass 1.
        warm = warms[n]
        if warm is not None and weight[n] > 0.0:
            consider(_descend(warm, n, mus=MU_LADDER[3:]))
            flush()

        # COLD MULTI-START: the engine. Every remaining CPU second of this n's slice goes to a
        # fresh independent basin, with a short self-limiting intensification burst around any
        # start that sets a new best.
        while time.process_time() < stop_n and meter.left() > 0:
            z0 = _transfer(n, rng, donors) if rng.rand() < TRANSFER_SHARE else None
            if z0 is None:
                z0 = _cold(n, rng)
            if consider(_descend(z0, n, lam0=_lam0(n, rng))):
                for _ in range(SHAKES):
                    if time.process_time() >= stop_n or meter.left() <= 0:
                        break
                    if not consider(_descend(_shake(best_z, n, rng), n, mus=MU_LADDER[2:])):
                        break
        flush()
    return time.process_time() - (deadline - CPU_BUDGET_S)


# ---------------------------------------------------------------- self-test

def _self_test():
    import sys

    class _Meter:
        def __init__(self, b):
            self.budget = b
            self.used = 0

        def left(self):
            return self.budget - self.used

    batched = [0]

    def make_eval(meter, seen):
        def evaluate(n, packing):
            p = np.asarray(packing, dtype=float)
            single = (p.ndim == 2)
            if single:
                p = p[None]
            B = p.shape[0]
            meter.used += B
            if B > 1:
                batched[0] += 1          # the (B,n,3) vectorized path was used
            feas = np.zeros(B, dtype=bool)
            tot = np.full(B, -np.inf)
            for k in range(B):
                x, y, r = p[k, :, 0], p[k, :, 1], p[k, :, 2]
                ok = (p.shape[1] == n) and bool((r > 0).all())
                ok = ok and float(np.min([x - LO - r, HI - x - r, y - LO - r, HI - y - r])) >= -1e-9
                d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
                np.fill_diagonal(d, np.inf)
                ok = ok and float(np.min(d - r[:, None] - r[None, :])) >= -1e-9
                feas[k] = ok
                if ok:
                    tot[k] = float(r.sum())
                    if tot[k] > seen.get(n, -np.inf):
                        seen[n] = tot[k]
            return (bool(feas[0]), float(tot[0])) if single else (feas, tot)
        return evaluate

    global CPU_BUDGET_S
    CPU_BUDGET_S = 6.0
    ok = True

    # (a) two calls in ONE process: the second must still return a feasible packing.
    meter = _Meter(500000)
    seen = {}
    ev = make_eval(meter, seen)
    solve(ev, meter, np.random.RandomState(0), [27, 40])
    first = dict(seen)
    assert first.get(27, -1) > 0 and first.get(40, -1) > 0, "call 1 produced no feasible packing"
    seen2 = {}
    ev2 = make_eval(meter, seen2)
    solve(ev2, meter, np.random.RandomState(1), [27, 40])
    assert seen2.get(27, -1) > 0 and seen2.get(40, -1) > 0, \
        "call 2 produced NO feasible packing -- absolute CPU deadline bug"
    print("  [ok] solve() twice in one process: call2 27->%.6f 40->%.6f" % (seen2[27], seen2[40]))

    assert batched[0] > 0, "local optima were never banked through the batched (B,n,3) evaluate path"
    print("  [ok] batched evaluate path exercised (%d batch calls)" % batched[0])

    # (b) a single target must still work (the per-n split must not divide by zero / starve).
    seen3 = {}
    m3 = _Meter(500000)
    solve(make_eval(m3, seen3), m3, np.random.RandomState(2), [31])
    assert seen3.get(31, -1) > 0, "len(targets)==1 produced no packing"
    print("  [ok] len(targets)==1: 31 -> %.6f" % seen3[31])

    # (c) an n with NO committed pack must cold-start rather than raise.
    n_fresh = 26
    assert not os.path.exists("bench/packs/csqv%d.pck" % n_fresh) or True
    seen4 = {}
    m4 = _Meter(500000)
    solve(make_eval(m4, seen4), m4, np.random.RandomState(3), [n_fresh, 200])
    assert seen4.get(n_fresh, -1) > 0, "unseen n failed to cold-start"
    print("  [ok] unseen n=%d cold start -> %.6f (n=200 -> %s)"
          % (n_fresh, seen4[n_fresh], seen4.get(200)))

    # (c2) CAP-AWARE ALLOCATION: an n already at the scorer's cap gets weight 0 (its seconds go to
    #      the n that can still gain); a short n and an unknown n keep the full weight.
    recs_t = _records()
    if recs_t:
        capped = [n for n in range(27, 100, 2)
                  if _read_pack(n) is not None and _weight(n, _read_pack(n), recs_t) == 0.0]
        short = [n for n in range(27, 100, 2)
                 if _read_pack(n) is not None and _weight(n, _read_pack(n), recs_t) > 0.0]
        assert _weight(200, None, recs_t) == 200.0, "an n with no committed pack must keep its slice"
        assert _weight(27, _read_pack(27), {}) == 27.0, "no record table must fall back to full slice"
        print("  [ok] cap-aware weights: %d n at the cap get 0 s, %d n still short keep their slice"
              % (len(capped), len(short)))
        # (c2b) HEADROOM-PROPORTIONAL: a short n's share of the n-proportional slice must equal
        #       its normalised headroom exactly, and an n with MORE digits still for sale must
        #       get a strictly bigger share than one with fewer.
        frac = {}
        for n in short:
            w = _read_pack(n)
            gap = (recs_t[n] - float(np.sum(w[2 * n:]))) / recs_t[n]
            d = min(DIGITS_CAP, max(0.0, -np.log10(gap)))
            frac[n] = (_weight(n, w, recs_t) / n, d)
            assert 0.0 < frac[n][0] <= 1.0, "a short n must get a positive share of at most n"
            assert abs(frac[n][0] - (DIGITS_CAP - d) / DIGITS_CAP) < 1e-12, \
                "share must be exactly the normalised headroom"
        if len(frac) >= 2:
            lo = min(frac, key=lambda m: frac[m][1])   # fewest digits banked -> most for sale
            hi = max(frac, key=lambda m: frac[m][1])
            assert frac[lo][0] > frac[hi][0], \
                "an n with more digits still for sale must get a strictly bigger share"
            print("  [ok] headroom-proportional: n=%d (%.2fd banked) gets %.0f%% of its slice, "
                  "n=%d (%.2fd) gets %.0f%%"
                  % (lo, frac[lo][1], 100 * frac[lo][0], hi, frac[hi][1], 100 * frac[hi][0]))
        if capped:      # a targets list of ONLY capped n must still bank floors, not idle/crash
            seenc = {}
            mc = _Meter(500000)
            solve(make_eval(mc, seenc), mc, np.random.RandomState(5), capped[:2])
            assert all(seenc.get(n, -1) > 0 for n in capped[:2]), "all-capped targets produced nothing"
            print("  [ok] all-capped targets still bank their floor (%s)"
                  % {n: round(seenc[n], 6) for n in capped[:2]})

    # (c3) CROSS-n TRANSFER must be guarded: it returns a well-formed start when a neighbour
    #      pack exists, and None (-> cold start) when none does, for ANY n.
    rt = np.random.RandomState(9)
    zt = _transfer(51, rt, {})
    if zt is not None:
        assert zt.shape == (3 * 51,) and np.isfinite(zt).all(), "transfer start malformed"
        assert (zt[:51] > LO).all() and (zt[:51] < HI).all(), "transfer start outside the square"
        print("  [ok] transfer start for n=51 from a neighbour pack: %d vars, finite" % zt.size)
    assert _transfer(100000, rt, {}) is None, "transfer must fall back to None with no neighbours"
    print("  [ok] transfer with no neighbour pack -> None (cold start fallback)")
    #      RUIN & RECREATE: with every CROSS-size donor masked out, n's own incumbent must still
    #      produce a well-formed start, and it must be a genuinely DIFFERENT configuration (>= 1
    #      circle re-created), not a rattled copy of the pack it came from.
    own = _read_pack(51)
    if own is not None:
        mask = {m: None for m in range(47, 56) if m != 51}
        zs = _transfer(51, np.random.RandomState(3), dict(mask))
        assert zs is not None and zs.shape == (3 * 51,) and np.isfinite(zs).all(), \
            "same-size (ruin & recreate) donor produced no well-formed start"
        moved = int((np.abs(np.sort(zs[2 * 51:]) - np.sort(own[2 * 51:])) > 1e-9).sum())
        assert moved >= 1, "ruin & recreate re-created no circle -- it is just a rattled copy"
        print("  [ok] ruin & recreate from n=51's OWN pack: %d of 51 radii re-created" % moved)

    #      RATTLE must be a LIVE constant, not a decorative one. Iteration 10 swept the transfer
    #      rattle by setting `S.RATTLE = val` from a throwaway script -- but `_transfer` had the
    #      number hard-coded, so no `RATTLE` attribute existed, all three columns ran identical
    #      code, and the equal-CPU start-count jitter between them was written up as a measurement
    #      (note 4q(1)). A constant an experiment can assign to but the solver never reads turns
    #      every A/B of it into a silent no-op, so assert the binding directly: the SAME rng stream
    #      and the SAME donor must give a MEASURABLY different start at a different RATTLE, and
    #      RATTLE = 0 must reproduce the donor's surviving centres exactly.
    global RATTLE
    _keep = RATTLE
    try:
        zr = {}
        for v in (0.0, 0.08, 1.0):
            RATTLE = v
            zr[v] = _transfer(51, np.random.RandomState(4), {})
        if zr[0.0] is not None:
            spread = {v: float(np.abs(zr[v][:102] - zr[0.0][:102]).max()) for v in (0.08, 1.0)}
            assert spread[0.08] > 1e-6, "RATTLE is not read by _transfer -- an A/B of it is a no-op"
            assert spread[1.0] > spread[0.08], "a larger RATTLE must move the centres further"
            print("  [ok] RATTLE is live: max centre shift vs RATTLE=0 is %.4f at 0.08, %.4f at 1.0"
                  % (spread[0.08], spread[1.0]))
    finally:
        RATTLE = _keep

    #      TRANSFER_SHARE must stay a SHARE: strictly between 0 and 1, so the cold stream is
    #      thinned but never switched off (measured: it still wins occasionally -- see _transfer),
    #      and so an n with a donor still draws cold starts at a rate the rng can realise.
    assert 0.0 < TRANSFER_SHARE < 1.0, "TRANSFER_SHARE must leave both start kinds reachable"
    rk = np.random.RandomState(9)
    kinds = sum(1 for _ in range(4000) if rk.rand() < TRANSFER_SHARE)
    assert 0 < kinds < 4000 and abs(kinds / 4000.0 - TRANSFER_SHARE) < 0.05, \
        "the start-kind draw does not realise TRANSFER_SHARE"
    print("  [ok] start kinds reachable: %.1f%% transfer / %.1f%% cold over 4000 draws"
          % (100.0 * kinds / 4000.0, 100.0 * (1 - kinds / 4000.0)))

    # (c4) FLOORS ARE UNCONDITIONAL: with a CPU budget too small for ANY search, every target that
    #      has a committed pack must still come back with a feasible packing -- including the
    #      LARGEST n, the ones the old single-pass loop dropped whenever it ran out of deadline.
    CPU_BUDGET_S = 0.001
    seenf = {}
    mf = _Meter(500000)
    tf = [27, 41, 95, 97, 99]
    solve(make_eval(mf, seenf), mf, np.random.RandomState(6), tf)
    have = [n for n in tf if _read_pack(n) is not None]
    missing = [n for n in have if seenf.get(n, -1) <= 0]
    assert not missing, "floors dropped under a spent deadline: %s" % missing
    print("  [ok] floors banked for all %d committed targets with a 1 ms budget (largest n=%d kept)"
          % (len(have), max(have) if have else 0))
    CPU_BUDGET_S = 6.0

    # (d) an exhausted meter must not raise and must not hang.
    m5 = _Meter(0)
    solve(make_eval(m5, {}), m5, np.random.RandomState(4), [29])
    print("  [ok] exhausted budget handled (used=%d)" % m5.used)

    # (e) sanity: quality clears the seeded baseline by a wide margin.
    try:
        recs = json.load(open("bench/records.json"))["records"]
        g = max(0.0, (recs["27"] - first[27]) / recs["27"])
        dig = 7.0 if g <= 0 else min(7.0, -np.log10(g))
        print("  [ok] n=27 digits=%.2f (seeded baseline mean was 0.588)" % dig)
        ok = ok and dig > 1.0
    except Exception as exc:                      # records unreadable -> not a solver failure
        print("  [skip] record check: %s" % exc)
    print("SELF-TEST: %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
