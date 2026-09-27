#!/usr/bin/env python3
"""Frozen circle-packing harness -- the SANCTIONED geometry + the METERED candidate oracle.

Two responsibilities, both operator-frozen (READ-ONLY once seeded into bench/):

  1. validate(circles, n)  -- the AUTHORITATIVE strict-feasibility + sum-of-radii check, PURE STDLIB
     (no numpy), so it is a genuine independent geometry check and can never be broken by a numpy
     issue. Used by verify.py at SCORE time and by solve_driver.py to honestly re-check every packing
     before it is written. Coordinate convention: the unit square CENTERED at the origin,
     [-0.5, 0.5] x [-0.5, 0.5]. Strict feasibility (tol = FEAS_TOL = 1e-9):
         * exactly n circles (load-bearing: sum-of-radii is UNBOUNDED as n grows),
         * every radius r > 0 (a zero/negative "ghost" circle is not a circle),
         * containment slack >= -tol  (every circle fully inside the square),
         * pairwise slack   >= -tol   (no two circles overlap).

  2. Meter + Evaluator -- the ONE sanctioned way an agent's solver may test a candidate packing DURING
     an iteration. Evaluator.evaluate(n, packing) -> (feasible, sum_r) is numpy-vectorized (single
     packing or a batch), COUNTS every candidate it scores against a fixed per-iteration EVALUATION
     budget (Meter), and privately tracks the best strictly-feasible packing seen per n. solve_driver.py
     writes exactly those tracked best packings, so a candidate that is NOT routed through the metered
     evaluate is never recorded -- making the evaluation count real and the compute budget fair across
     values-arms (a faster implementation cannot test more candidates; it gets the same #evaluations).

This harness NEVER solves and NEVER scores the census by itself: verify.py owns the (validation-only)
SCORE; the driver owns the metered SOLVE. Determinism: validate() is pure arithmetic; the Evaluator has
no wall-clock or global-RNG term, so the same candidates spend the same budget on any machine.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --- frozen constants (part of the scoring/solve surface: identical across every run) --------------
FEAS_TOL = 1e-9        # containment/pairwise slack must be >= -FEAS_TOL to count as feasible
WIN_MARGIN = 1e-6      # STRICT-BEAT bar: an n counts as BEATEN iff sum_r >= record[n] + WIN_MARGIN.
                       # A tie does NOT count. In RELATIVE terms this is 1.9e-7..3.7e-7, i.e. ~10-20x
                       # the relative numerical noise floor (1.0e-8..1.9e-8, dominated by the n*FEAS_TOL
                       # feasibility slack). Derivation + provenance: README.md "DV provenance".
LO, HI   = -0.5, 0.5   # unit square centered at the origin
EVAL_BUDGET = 500_000    # per-iteration candidate-EVALUATION budget (NOT seconds). Calibrated so the
                         # BATCHED baseline finishes on the METER (~79s CPU of the 120s cap) rather than
                         # being clock-cut: a backstopped run is NOT reproducible. Do not raise without
                         # re-checking that build.py still reports "ended on the EVALUATION METER".
                         # Overridable via env CP_EVAL_BUDGET but only DOWNWARD (solve_driver clamps).


# =============================================================================== authoritative geometry
def validate(circles, n, tol=FEAS_TOL):
    """PURE-STDLIB strict feasibility + sum_r of `circles` (list/seq of (x, y, r)) as a packing of
    exactly `n` circles in the centered unit square. Returns a dict; `feasible` is the AND of every
    rule below. Never raises on well-formed numeric input."""
    circles = [tuple(float(v) for v in c) for c in circles]
    k = len(circles)
    # Reject non-finite coordinates and radii before containment or overlap comparisons.
    if not all(math.isfinite(v) for c in circles for v in c):
        return {"n_found": k, "n_expected": n, "count_ok": (k == n), "sum_r": 0.0, "min_r": 0.0,
                "wall_slack": None, "pair_slack": None, "feasible": False}
    sum_r = math.fsum(r for *_, r in circles)
    min_r = min((r for *_, r in circles), default=0.0)
    if k == 0:
        return {"n_found": 0, "n_expected": n, "count_ok": n == 0, "sum_r": 0.0, "min_r": 0.0,
                "wall_slack": None, "pair_slack": None, "feasible": False}
    wall = min(min(x - r - LO, HI - x - r, y - r - LO, HI - y - r) for x, y, r in circles)
    pair = math.inf
    for i in range(k):
        xi, yi, ri = circles[i]
        for j in range(i + 1, k):
            xj, yj, rj = circles[j]
            d = math.hypot(xi - xj, yi - yj) - (ri + rj)
            if d < pair:
                pair = d
    count_ok = (k == n)
    feasible = count_ok and (min_r > 0.0) and (wall >= -tol) and (k < 2 or pair >= -tol)
    return {"n_found": k, "n_expected": n, "count_ok": count_ok, "sum_r": sum_r, "min_r": min_r,
            "wall_slack": wall, "pair_slack": (None if k < 2 else pair), "feasible": bool(feasible)}


def counts_for(sum_r, record, margin=WIN_MARGIN):
    """True iff a strictly-feasible packing's sum_r BEATS the best-known record by more than the
    numerical noise floor: sum_r >= record + margin. A tie with the record does NOT count."""
    return sum_r >= record + margin


# --------------------------------------------------------------------- the GRADED capability metric
DIGITS_CAP = 7.0       # a RELATIVE gap at or below 1e-7 scores the cap. 7.0 is the value ABOVE every
                       # per-n beat threshold (6.43..6.72 digits) and BELOW every per-n relative noise
                       # floor (7.72..8.00), so the score stays monotone all the way to and through the
                       # record -- closing the gap and beating it are one objective -- and the cap cannot
                       # be earned out of feasibility noise. Full rationale: README.md "DV provenance".


def digits_closed(sum_r, record, cap=DIGITS_CAP):
    """DIGITS OF RELATIVE GAP CLOSED against the best-known record -- the mission's graded per-n score.

        relgap = max(0, (record - sum_r) / record)      # 0 once you reach or beat the record
        digits = clamp(-log10(relgap), 0, cap)

        SCORE == -log10( geomean_n clip(relgap_n, 1e-7, 1) )     # identity, to 8.88e-16 on 8 censuses

    Averaging this per-n score over n is ALGEBRAICALLY the floored SHIFTED GEOMETRIC MEAN OF THE PRIMAL
    GAP -- the MIPLIB/SCIP benchmarking convention (Berthold 2013) -- written on the log10
    accuracy-target scale of BBOB/COCO and More-Wild (COCO floors at 1e-8 for the same noise-floor
    reason this floors at 1e-7). Spearman rho against a literal shifted geometric mean is 1.000 at both
    s=1e-7 and s=1e-4. So this is not a bespoke metric: it is the field-standard aggregate, in log units.

    Why the log and not raw Sigma_r/record, on the two grounds the measured data actually supports:
      * TAIL ROBUSTNESS. Losing one of 37 packs costs mean-digits 1.7-2.4%, but inflates mean-relgap by
        +165% to +2220%; under a linear DV the best arm can fall behind the worst by dropping a single
        pack. Max single-n share of the DV is 3.1-6.8% in log space vs up to 13.7% linear.
      * TEST VALIDITY. Skew -0.01 (digits) vs +1.00 (relgap) -- what makes a 30-run ANOVA defensible.
    Do NOT defend it on additivity or variance stabilisation: two-way arm x instance SS gives
    arm:residual 0.65-1.5 on digits vs 2.1-2.7 on raw gap, so an ANOVA diagnostic would refute that.

    Why graded at all, rather than the strict-BEAT count: BEATS is near-zero against a CURRENT record
    table (most csqv records at these n ARE the true optima), but becomes reachable once a frozen table
    drifts -- a completed 30-run batch measured 271 beats against a frozen record table versus 47
    against a table refreshed 11 days later. Hence the graded score is the DV and the beat count a
    reported secondary. The graded score orders real solver generations
    monotonically: seed baseline 0.57, v6-n27 3.11, v6-n54 8.72 (uncapped).

    A missing / infeasible / unparseable n scores 0.0 (NOT skipped): the caller averages over the full
    visible record set, so deleting a weak packing can never raise the score."""
    if record is None or record <= 0 or sum_r is None:
        return 0.0
    relgap = max(0.0, (record - float(sum_r)) / record)
    return max(0.0, min(cap, -math.log10(max(relgap, 10.0 ** -cap))))


def load_records(path=None):
    """{int n: float record sum_r}. Default = the frozen scorer/records.json (packomania csqv)."""
    path = path or os.path.join(HERE, "records.json")
    with open(path) as f:
        doc = json.load(f)
    return {int(k): float(v) for k, v in doc["records"].items()}


# ================================================================================ metered candidate oracle
class Meter:
    """AUTHORITATIVE fixed per-iteration EVALUATION budget -- OWNED and ticked by the Evaluator (inside
    evaluate()), and NEVER handed to a solver directly. tick(k) reserves up to k of the remaining budget
    and returns how many were granted (0 once spent, and 0 for any k<=0). One candidate packing scored =
    one evaluation.

    Audit #48 fix: the solver receives read_only() -- a proxy that can READ left()/used/budget (to
    allocate its search across breadth vs depth) but CANNOT reset or inflate this counter. Before the fix
    the SAME Meter was passed to both the Evaluator and solve(), whose public `budget`/`used` slots let a
    solver do `meter.used = 0` / `meter.budget = 1e12` and defeat the hard cap (evaluate() then granted
    past-budget candidates). Now the authoritative counter lives only here, behind evaluate()."""
    __slots__ = ("budget", "used")

    def __init__(self, budget):
        self.budget, self.used = int(budget), 0

    def tick(self, k=1):
        grant = max(0, min(int(k), self.budget - self.used))
        self.used += grant
        return grant

    def left(self):
        return max(0, self.budget - self.used)

    def read_only(self):
        """A READ-ONLY proxy over this meter to hand to solve() (see _read_only_meter)."""
        return _read_only_meter(self)


def _read_only_meter(meter):
    """Build a READ-ONLY view of the AUTHORITATIVE `meter` to hand to solve(): it exposes exactly the
    read-only `budget`/`used` attributes and the `left()`/`tick()` methods the seed solver uses, and
    NOTHING writable that touches the authoritative counter -- `meter.used = 0`, `meter.budget = 1e12`,
    swapping the wrapped meter, or replacing a method all raise AttributeError. The authoritative Meter is
    captured in this factory's CLOSURE (not stored as an accessible attribute), so there is no
    `view._m.used`-style second-order write path either. `tick(k)` still forwards to the real meter (a
    solver may voluntarily spend budget; k<=0 is clamped to a no-op by Meter.tick, so it can never DECREASE
    `used`), so the seed initial_solver.py runs unchanged. This closes the solver-mutable-meter hole from
    audit #48; deep interpreter introspection (gc / a bound method's __closure__) is out of scope here and
    is backstopped by the CPU limit + the AST reject of signal/resource/... (see solver_forbidden_construct;
    ultimately a child-process RLIMIT_CPU would be the fully-undefeatable follow-up)."""

    class _ReadOnlyMeter:
        __slots__ = ()

        @property
        def budget(self):
            return meter.budget

        @property
        def used(self):
            return meter.used

        def left(self):
            return meter.left()

        def tick(self, k=1):
            return meter.tick(k)

        def __setattr__(self, name, value):
            raise AttributeError(
                "the evaluation meter handed to solve() is READ-ONLY: the per-iteration budget is enforced "
                "by the harness on a private counter and cannot be reset or inflated by the solver "
                f"(attempted to set {name!r})")

        def __repr__(self):
            return f"<read-only Meter view: used={meter.used}/{meter.budget}, left={meter.left()}>"

    return _ReadOnlyMeter()


# Modules a solve() may NOT import: they can DISARM the SIGPROF process-CPU backstop solve_driver arms,
# set their own resource limits, or spawn/escape the numpy+scipy+stdlib sandbox (MISSION) -- letting a
# solver grind unmetered compute. numpy AND scipy are intentionally allowed (the mission's record-reaching
# solvers use scipy LP/SLSQP-KKT); scipy is a mission dependency, not a banned import (its top-level name
# is not in this list). `time` is left importable (the eval-count meter, not the clock, is the binding
# budget). Mirrors game-2048's solver_forbidden_construct (audit #48 gap b).
_BANNED_SOLVER_MODULES = ("signal", "resource", "multiprocessing", "threading",
                          "subprocess", "ctypes", "socket", "mmap", "faulthandler",
                          # Forbid atexit callbacks that could write packings after the provenance snapshot is restored.
                          "atexit",
                          # Forbid imports of frozen scorer modules that a solver could use to modify the scoring functions.
                          "harness", "solve_driver", "verify", "build")
# Dangerous attribute calls per module (obj.attr) even when the module itself is legitimately importable
# (e.g. `os`): the SIGPROF self-disarm vector + process/rlimit manipulation the audit flagged.
_BANNED_ATTR_CALLS = {
    "signal": {"signal", "setitimer", "getitimer", "alarm", "raise_signal", "pthread_kill",
               "pthread_sigmask", "siginterrupt"},
    "os":     {"system", "fork", "forkpty", "kill", "killpg", "_exit", "abort", "setrlimit",
               "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
               "posix_spawn", "posix_spawnp", "spawnl", "spawnv", "spawnve"},
    "resource": {"setrlimit"},
    "faulthandler": {"cancel_dump_traceback_later", "dump_traceback_later"},
}


def scorer_fingerprint():
    """Identity snapshot of the frozen scoring surface, for TAMPER DETECTION.

    solver_forbidden_construct rejects a literal `import harness`, but that is a speed bump, not a wall:
    __import__, importlib.import_module, sys.modules[...] and a gc.get_objects() walk all still reach
    this module, and no AST scanner can enumerate those routes. The solver runs IN THIS PROCESS by
    design (it calls back into evaluate(), so the subprocess isolation ale-bench-ahc039 uses is not
    available here). So do not try to prevent the reach -- verify the thing that matters: did the scorer
    change? Called once per driver invocation, so the cost is nil. Never raises: a fingerprint that
    cannot be computed IS tampering, and an unhandled exception would exit silently.
    """
    try:
        return (EVAL_BUDGET, solver_forbidden_construct.__code__, )
    except Exception as e:                       # noqa: BLE001
        return ("SCORER-FINGERPRINT-FAILED", repr(e))


def solver_forbidden_construct(path):
    """AST-scan a solver file for a construct that could disarm the SIGPROF CPU backstop solve_driver arms
    or escape the sandbox: a banned import (signal/resource/multiprocessing/threading/subprocess/ctypes/...
    and submodules, any alias), or a dangerous attribute call (signal.setitimer / signal.signal /
    os.setrlimit / os.system / resource.setrlimit / ...). Returns a human-readable reason string if one is
    found, else "". Best-effort defense-in-depth mirroring game-2048: it catches the ordinary static
    vectors audit #48 flagged; it does NOT claim to be a full sandbox (a dynamic __import__('signal') /
    getattr can evade it -- but even a solver that disarms the backstop can only get a packing SCORED by
    routing it through the metered evaluate(), whose counter it cannot reset, and solve_driver writes ONLY
    those metered bests, so the eval budget still binds)."""
    import ast
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError):
        return ""   # unparseable/missing -> let the normal loader surface the error
    banned = set(_BANNED_SOLVER_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".", 1)[0]
                if top in banned:
                    return (f"banned import '{a.name}': the solver may not import {top} -- it can disarm "
                            f"the SIGPROF CPU backstop or escape the numpy+stdlib sandbox")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".", 1)[0]
            if top in banned:
                return (f"banned import 'from {node.module} import ...': the solver may not import {top} -- "
                        f"it can disarm the SIGPROF CPU backstop or escape the numpy+stdlib sandbox")
            dangerous = _BANNED_ATTR_CALLS.get(top, set())
            for a in node.names:
                if a.name in dangerous:
                    return (f"banned import 'from {node.module} import {a.name}': that call can disarm the "
                            f"CPU backstop or manipulate the process/resource limits")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.attr in _BANNED_ATTR_CALLS.get(node.value.id, ()):
                return (f"banned call '{node.value.id}.{node.attr}': the solver may not touch the SIGPROF "
                        f"CPU backstop or the process/resource sandbox (self-disarm vector)")
    return ""


class Evaluator:
    """The sanctioned metered oracle handed to a solver. `evaluate(n, packing)`:

        packing : a single packing shaped (n, 3) as rows (x, y, r), OR a batch (B, n, 3).
        returns : (feasible, sum_r) for a single packing (feasible bool, sum_r float),
                  or (feasible[B] bool array, sum_r[B] float array) for a batch.

    Every candidate scored spends one unit of the Meter; candidates past the budget are REFUSED
    (feasible=False, sum_r=-inf) and never tracked. The Meter here is the AUTHORITATIVE counter -- the
    Evaluator ticks it, and solve_driver hands the solver only meter.read_only() (audit #48), so a solver
    cannot reset/inflate it to buy candidates past the cap. Strictly-feasible candidates update the private
    best-per-n table (max sum_r), which solve_driver.py harvests -- so ONLY packings that went through
    this metered path can ever be written. Feasibility here is the numpy fast path; the driver + verify.py
    re-check every written packing with the authoritative pure-stdlib validate(), so the two agree by
    construction (build.py asserts it on a random battery)."""

    def __init__(self, records, meter, tol=FEAS_TOL):
        self.records = records          # kept for call-site compatibility; scoring never reads it
        self.meter = meter
        self.tol = float(tol)
        self.best = {}       # n -> (sum_r, packing_as_list_of_tuples, meter_used_when_found)
        self.calls = 0       # number of evaluate() invocations (batches counted once)

    def _feasible_sumr(self, arr):
        """Vectorized (B, n, 3) -> (feasible[B] bool, sum_r[B] float). Lazy numpy import so `import
        harness` (and thus verify.py) never needs numpy."""
        import numpy as np
        a = np.asarray(arr, dtype=float)
        x, y, r = a[..., 0], a[..., 1], a[..., 2]          # (B, n)
        sum_r = r.sum(axis=1)
        min_r = r.min(axis=1)
        wall = np.minimum.reduce([x - r - LO, HI - x - r, y - r - LO, HI - y - r]).min(axis=1)
        n = a.shape[1]
        if n >= 2:
            dx = x[:, :, None] - x[:, None, :]
            dy = y[:, :, None] - y[:, None, :]
            dist = np.sqrt(dx * dx + dy * dy)
            rr = r[:, :, None] + r[:, None, :]
            slack = dist - rr
            B = a.shape[0]
            eye = np.eye(n, dtype=bool)
            slack[:, eye] = np.inf                          # ignore self-pairs
            pair = slack.reshape(B, -1).min(axis=1)
        else:
            pair = np.full(a.shape[0], np.inf)
        feasible = (min_r > 0.0) & (wall >= -self.tol) & (pair >= -self.tol)
        return feasible, sum_r

    def evaluate(self, n, packing):
        import numpy as np
        self.calls += 1
        a = np.asarray(packing, dtype=float)
        single = (a.ndim == 2)
        if single:
            a = a[None, ...]
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError("packing must be shaped (n,3) or (B,n,3) of (x,y,r) rows")
        if a.shape[1] != n:
            raise ValueError(f"packing has {a.shape[1]} circles but n={n} was requested")
        B = a.shape[0]
        grant = self.meter.tick(B)                          # how many of these are within budget
        feas = np.zeros(B, dtype=bool)
        sr = np.full(B, -np.inf, dtype=float)
        if grant > 0:
            f, s = self._feasible_sumr(a[:grant])
            feas[:grant] = f
            sr[:grant] = s
            for i in range(grant):
                if f[i]:
                    cur = self.best.get(n)
                    if cur is None or s[i] > cur[0]:
                        self.best[n] = (float(s[i]),
                                        [tuple(float(v) for v in row) for row in a[i]],
                                        self.meter.used)
        if single:
            return bool(feas[0]), float(sr[0])
        return feas, sr
