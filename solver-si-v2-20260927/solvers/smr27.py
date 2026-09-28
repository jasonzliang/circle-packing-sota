"""Adapter: the circle-packing-sota `solver-sm-radical-v6-n27` search (pack.search) under the SI-v2 solve() contract.
Frame: pack.py works in the corner square [0,1]^2 with z = [x(n), y(n), r(n)]; the mission is centred [-0.5,0.5]^2."""
import os, sys, time, math
import numpy as np
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "solver-sm-radical-v6-n27"))   # this repo's solver-sm-radical-v6-n27
if REPO not in sys.path: sys.path.insert(0, REPO)
import pack as PK   # noqa: E402
CPU_BUDGET = 100.0    # seconds per solve() call (the sweep overrides this)

def _warm(n):
    p = os.path.join("bench", "packs", "csqv%d.pck" % n)
    if not os.path.isfile(p): return None
    rows = [l.split() for l in open(p).read().split("\n")[2:] if len(l.split()) >= 3]
    if len(rows) != n: return None
    c = np.array([[float(a), float(b), float(r)] for a, b, r in rows])
    return np.concatenate([c[:, 0] + 0.5, c[:, 1] + 0.5, c[:, 2]])

def solve(evaluate, meter, rng, targets):
    for n in targets:
        seed = int(rng.randint(0, 2**31 - 1))
        z0 = _warm(n)
        if z0 is not None:
            evaluate(n, np.column_stack([z0[:n] - 0.5, z0[n:2*n] - 0.5, z0[2*n:]]))
        best, best_s = PK.search(n, seed=seed, budget=float(CPU_BUDGET), warm=z0, verbose=False)
        if best is None: continue
        z = np.asarray(best, dtype=float)
        pk = np.column_stack([z[:n] - 0.5, z[n:2*n] - 0.5, z[2*n:]])
        feas, s = evaluate(n, pk)
        if not feas:                                   # shave radii until the metered oracle accepts it
            for k in range(1, 12):
                pk2 = pk.copy(); pk2[:, 2] *= (1.0 - k * 1e-12)
                feas, s = evaluate(n, pk2)
                if feas: break
