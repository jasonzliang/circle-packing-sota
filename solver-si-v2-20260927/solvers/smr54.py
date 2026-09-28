"""Adapter: the circle-packing-sota `solver-sm-radical-v6-n54` pipeline under the SI-v2 solve() contract.
With an incumbent in bench/packs it runs the pipeline's ENDGAME walk from that packing (warm ILS); without one
it runs the full broad->endgame pipeline.solve. Frame: pipeline works in the corner square [0,1]^2."""
import os, sys, time, math, io, contextlib
import numpy as np
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "solver-sm-radical-v6-n54"))   # this repo's solver-sm-radical-v6-n54
if REPO not in sys.path: sys.path.insert(0, REPO)
import pipeline as PL   # noqa: E402
import endgame as EG    # noqa: E402
CPU_BUDGET = 100.0    # seconds per solve() call (the sweep overrides this)

def _warm(n):
    p = os.path.join("bench", "packs", "csqv%d.pck" % n)
    if not os.path.isfile(p): return None
    rows = [l.split() for l in open(p).read().split("\n")[2:] if len(l.split()) >= 3]
    if len(rows) != n: return None
    c = np.array([[float(a), float(b), float(r)] for a, b, r in rows])
    return c[:, 0] + 0.5, c[:, 1] + 0.5, c[:, 2].copy()

def _submit(evaluate, n, x, y, r):
    pk = np.column_stack([np.asarray(x) - 0.5, np.asarray(y) - 0.5, np.asarray(r)])
    feas, s = evaluate(n, pk)
    if not feas:
        for k in range(1, 12):
            pk2 = pk.copy(); pk2[:, 2] *= (1.0 - k * 1e-12)
            feas, s = evaluate(n, pk2)
            if feas: break
    return feas

def solve(evaluate, meter, rng, targets):
    for n in targets:
        seed = int(rng.randint(0, 2**31 - 1))
        w = _warm(n)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                if w is not None:
                    _submit(evaluate, n, *w)
                    x, y, r, cost = EG.endgame(w[0], w[1], w[2], float(CPU_BUDGET), seed=seed, verbose=False)
                else:
                    x, y, r, cost = PL.solve(n=n, seconds=float(CPU_BUDGET), seed=seed, verbose=False)
            except Exception:          # e.g. "no candidate produced -- budget too small"
                continue
        _submit(evaluate, n, x, y, r)
