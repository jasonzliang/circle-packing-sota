#!/usr/bin/env python3
"""The SHAPE of the packed objects -- homothets of a symmetric convex body.

WHY THIS EXISTS
---------------
iter 4 removed the container assumption by finding the single place it entered
(the wall rows).  This removes the *object* assumption the same way.  Replace
"circle of radius r at c" by "c + r*K" for a fixed centrally-symmetric convex
body K.  Then, and this is the whole content of the file:

  * two homothets are disjoint  <=>  gamma_K(c_i - c_j) >= r_i + r_j,
    because r_i*K (+) r_j*(-K) = (r_i + r_j)*K when K = -K.  gamma_K is the
    gauge (Minkowski functional) of K -- the Euclidean case is |v|.
  * containment in a half-plane a.u <= b is  a.c + r*h_K(a) <= b, where
    h_K is the support function.  The Euclidean case is h = |a| = 1.

Both are STILL LINEAR IN r.  So the exactly-solvable inner layer the whole
search rests on -- radii-given-centres is an LP -- survives a change of object
exactly as it survived a change of container.  The object enters in two
coefficient slots: a distance (hypot -> gauge) and a wall divisor (1 -> h_K).

AND IT GETS BETTER FOR POLYTOPES
--------------------------------
If K is a polytope, gamma_K(v) = max_k a_k.v over its facets {u : a_k.u <= 1}.
Then non-overlap is a DISJUNCTION of linear constraints: it holds as soon as
ONE facet direction separates the pair.  Fix, per pair, which facet does the
separating, and the whole problem -- CENTRES AND RADII TOGETHER -- is a linear
program.  Its solution is automatically feasible for the true problem (a chosen
a_k.v >= r_i+r_j implies gamma = max_k' a_k'.v >= r_i+r_j), so iterating
"pick the separating directions, solve the LP" is a monotone ascent whose every
step is a GLOBAL optimum inside its combinatorial cell, and every iterate is
exactly feasible with no repair and no tolerance.  See pack.lp_joint().

EXACT-CHECKABILITY
------------------
With rational vertices, the facet normals a_k = n_k / (n_k.v_k) are rational and
h_K(a) = max over vertices of a.u is rational, so BOTH tests above are exact
rational LINEAR arithmetic -- strictly easier to decide than the circle case,
which needed squaring.  tools/exact_check.py does exactly that.

    python3 tools/shape.py --self-test
"""
import argparse
import math

import numpy as np


class Shape:
    name = "shape"
    is_ball = False
    area_factor = 1.0            # area of r*K = area_factor * r^2

    def gauge(self, vx, vy):
        """gamma_K(v): the t with v on the boundary of t*K.  Elementwise."""
        raise NotImplementedError

    def gauge_grad(self, vx, vy):
        """(d gamma/d vx, d gamma/d vy) -- a subgradient at kinks."""
        raise NotImplementedError

    def support(self, ax, ay):
        """h_K(a) = max_{u in K} a.u.  Elementwise."""
        raise NotImplementedError

    def spec(self):
        raise NotImplementedError

    def bound(self, n, area):
        """sum(r) <= sqrt(n * A / area_factor).

        Disjoint homothets in a container of area A give sum(area_factor r^2)
        <= A; Cauchy-Schwarz gives (sum r)^2 <= n * sum r^2.  Derived here, not
        cited.  Reduces to sqrt(n*A/pi) for the disk, which is what bench2-4 use.
        """
        return math.sqrt(n * area / self.area_factor)


class Ball(Shape):
    """The Euclidean disk: the original problem, recovered exactly."""
    name = "disk"
    is_ball = True
    area_factor = math.pi

    def gauge(self, vx, vy):
        return np.hypot(vx, vy)

    def gauge_grad(self, vx, vy):
        # the +1e-15 guard is the one pack.py's pre-shape Jacobian used, kept
        # byte-for-byte so the circle path is not perturbed in its last digit
        d = np.hypot(vx, vy) + 1e-15
        return vx / d, vy / d

    def support(self, ax, ay):
        return np.hypot(ax, ay)          # = 1 for the unit normals pack.py uses

    def spec(self):
        return {"type": "ball", "name": self.name}


class PolyShape(Shape):
    """Homothets of a centrally symmetric convex polygon with rational vertices.

    Vertices counter-clockwise, centred at the origin, and K = -K (asserted --
    the disjointness identity r_i K (+) r_j (-K) = (r_i+r_j) K needs it, and a
    silently non-symmetric K would make every feasibility claim in this run
    wrong rather than merely conservative).
    """

    def __init__(self, vertices, name="polygon"):
        self.name = name
        self.V = np.asarray(vertices, dtype=float)
        assert self.V.ndim == 2 and self.V.shape[1] == 2 and len(self.V) >= 3
        v0, v1 = self.V, np.roll(self.V, -1, axis=0)
        nrm = np.stack([v1[:, 1] - v0[:, 1], -(v1[:, 0] - v0[:, 0])], axis=1)
        b = np.einsum("ij,ij->i", nrm, v0)
        assert np.all(b > 0), "origin must be strictly inside K"
        self.A = nrm / b[:, None]                     # K = {u : A u <= 1}
        self.area_factor = 0.5 * float(np.sum(v0[:, 0] * v1[:, 1]
                                              - v1[:, 0] * v0[:, 1]))
        assert self.area_factor > 0, "vertices must be counter-clockwise"
        # central symmetry: -V must be a permutation of V
        s = {(round(a, 12), round(c, 12)) for a, c in self.V}
        assert s == {(-a, -c) for a, c in s}, "K must be centrally symmetric"
        self.rout = float(np.max(np.hypot(self.V[:, 0], self.V[:, 1])))

    def gauge(self, vx, vy):
        vx, vy = np.asarray(vx, float), np.asarray(vy, float)
        p = (self.A[:, 0].reshape((-1,) + (1,) * vx.ndim) * vx
             + self.A[:, 1].reshape((-1,) + (1,) * vx.ndim) * vy)
        return p.max(axis=0)

    def gauge_grad(self, vx, vy):
        vx, vy = np.asarray(vx, float), np.asarray(vy, float)
        p = (self.A[:, 0].reshape((-1,) + (1,) * vx.ndim) * vx
             + self.A[:, 1].reshape((-1,) + (1,) * vx.ndim) * vy)
        k = p.argmax(axis=0)
        return self.A[k, 0], self.A[k, 1]

    def active(self, vx, vy):
        """Index of the facet attaining the gauge -- the separating direction."""
        vx, vy = np.asarray(vx, float), np.asarray(vy, float)
        p = (self.A[:, 0].reshape((-1,) + (1,) * vx.ndim) * vx
             + self.A[:, 1].reshape((-1,) + (1,) * vx.ndim) * vy)
        return p.argmax(axis=0)

    def support(self, ax, ay):
        ax, ay = np.asarray(ax, float), np.asarray(ay, float)
        p = (self.V[:, 0].reshape((-1,) + (1,) * ax.ndim) * ax
             + self.V[:, 1].reshape((-1,) + (1,) * ax.ndim) * ay)
        return p.max(axis=0)

    def spec(self):
        return {"type": "polygon", "name": self.name,
                "vertices": [[float(a), float(b)] for a, b in self.V]}


# ------------------------------------------------------------- the registry --

BALL = Ball()
# L-infinity ball: axis-aligned squares of half-side r.  Chosen because its
# optimum is PROVABLE at n = k^2 (see bench5 spec): the k x k grid attains the
# Cauchy-Schwarz bound sqrt(n*A)/2 exactly, so the search can be tested against
# a theorem at full instance size, not just at n = 1, 2.
SQUARE = PolyShape([(-1, -1), (1, -1), (1, 1), (-1, 1)], name="square")
# L1 ball: the same squares rotated 45 degrees against an axis-aligned box.
DIAMOND = PolyShape([(1, 0), (0, 1), (-1, 0), (0, -1)], name="diamond")
# A symmetric hexagon with rational vertices (it tiles the plane, so the area
# bound is not obviously slack, but the box corners cost something).
HEXAGON = PolyShape([(1, 0), (0.5, 1), (-0.5, 1), (-1, 0), (-0.5, -1), (0.5, -1)],
                    name="hexagon")

REGISTRY = {s.name: s for s in (BALL, SQUARE, DIAMOND, HEXAGON)}


def get(name):
    if name in REGISTRY:
        return REGISTRY[name]
    raise SystemExit(f"unknown shape {name!r}; have {sorted(REGISTRY)}")


def from_spec(spec):
    if spec is None or spec.get("type") == "ball":
        return BALL
    return PolyShape(spec["vertices"], name=spec.get("name", "polygon"))


# --------------------------------------------------------------- self-test ---

def _brute_gauge(K, v, hi=8.0):
    """gamma by bisection on 'is v/t inside K', sharing no code with gauge()."""
    def inside(u):
        for k in range(len(K.V)):
            a, b = K.V[k], K.V[(k + 1) % len(K.V)]
            if (b[0] - a[0]) * (u[1] - a[1]) - (b[1] - a[1]) * (u[0] - a[0]) < -1e-12:
                return False
        return True
    lo = 1e-12
    if not inside(np.asarray(v) / hi):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if inside(np.asarray(v) / mid):
            hi = mid
        else:
            lo = mid
    return hi


def self_test():
    rng = np.random.default_rng(0)
    # 1. closed forms: the L-inf / L1 gauges are max|.| and sum|.|
    v = rng.normal(0, 1, (2, 200))
    assert np.allclose(SQUARE.gauge(v[0], v[1]), np.maximum(np.abs(v[0]), np.abs(v[1])))
    assert np.allclose(DIAMOND.gauge(v[0], v[1]), np.abs(v[0]) + np.abs(v[1]))
    assert np.allclose(BALL.gauge(v[0], v[1]), np.hypot(v[0], v[1]))
    # 2. gauge against an INDEPENDENT point-in-polygon bisection
    for K in (SQUARE, DIAMOND, HEXAGON):
        for _ in range(40):
            u = rng.normal(0, 1, 2)
            g, gb = float(K.gauge(u[0], u[1])), _brute_gauge(K, u)
            assert abs(g - gb) < 1e-8, (K.name, u, g, gb)
    # 3. gauge is a norm: symmetric, positively homogeneous, and gamma(u)=1 on dK
    for K in (SQUARE, DIAMOND, HEXAGON, BALL):
        u = rng.normal(0, 1, (2, 50))
        assert np.allclose(K.gauge(u[0], u[1]), K.gauge(-u[0], -u[1]))
        assert np.allclose(K.gauge(3.5 * u[0], 3.5 * u[1]), 3.5 * K.gauge(u[0], u[1]))
        if K is not BALL:
            assert np.allclose(K.gauge(K.V[:, 0], K.V[:, 1]), 1.0)
    # 4. support function against a brute max over a dense boundary sample
    for K in (SQUARE, DIAMOND, HEXAGON):
        t = np.linspace(0, 1, 4001)
        P = np.concatenate([K.V[k] + t[:, None] * (K.V[(k + 1) % len(K.V)] - K.V[k])
                            for k in range(len(K.V))])
        for _ in range(20):
            a = rng.normal(0, 1, 2)
            h, hb = float(K.support(a[0], a[1])), float(np.max(P @ a))
            assert abs(h - hb) < 1e-9, (K.name, a, h, hb)
    # 5. gauge gradient by finite differences (away from kinks)
    for K in (SQUARE, DIAMOND, HEXAGON, BALL):
        for _ in range(30):
            u = rng.normal(0, 1, 2)
            gx, gy = K.gauge_grad(np.array([u[0]]), np.array([u[1]]))
            e = 1e-7
            fx = (K.gauge(np.array([u[0] + e]), np.array([u[1]]))
                  - K.gauge(np.array([u[0] - e]), np.array([u[1]]))) / (2 * e)
            fy = (K.gauge(np.array([u[0]]), np.array([u[1] + e]))
                  - K.gauge(np.array([u[0]]), np.array([u[1] - e]))) / (2 * e)
            assert abs(gx[0] - fx[0]) < 1e-5 and abs(gy[0] - fy[0]) < 1e-5, (K.name, u)
    # 6. the DISJOINTNESS identity, tested geometrically: two homothets touch
    #    exactly when gamma(c_i-c_j) == r_i+r_j.  Sample points of one body and
    #    check none is interior to the other at the touching separation.
    for K in (SQUARE, DIAMOND, HEXAGON):
        for _ in range(20):
            d = rng.normal(0, 1, 2)
            ri, rj = float(rng.uniform(0.2, 1.0)), float(rng.uniform(0.2, 1.0))
            g = float(K.gauge(d[0], d[1]))
            c = d * (ri + rj) / g                      # exactly touching
            t = np.linspace(0, 1, 401)
            P = np.concatenate([K.V[k] + t[:, None] * (K.V[(k + 1) % len(K.V)] - K.V[k])
                                for k in range(len(K.V))])
            inner = ri * P                             # boundary of body i at 0
            rel = inner - c                            # relative to body j
            gg = K.gauge(rel[:, 0], rel[:, 1])
            assert gg.min() > rj - 1e-9, (K.name, gg.min(), rj)   # no overlap
            assert gg.min() < rj + 1e-6, (K.name, gg.min(), rj)   # but touching
    # 7. area factors
    assert abs(SQUARE.area_factor - 4.0) < 1e-12
    assert abs(DIAMOND.area_factor - 2.0) < 1e-12
    assert abs(HEXAGON.area_factor - 3.0) < 1e-12      # shoelace of the hexagon
    assert abs(BALL.area_factor - math.pi) < 1e-15
    # 8. a non-symmetric body must be REJECTED, not silently mishandled
    try:
        PolyShape([(1, 0), (0, 1), (-1, -0.5)], name="scalene")
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted a non-centrally-symmetric K")
    # 9. spec round-trip
    for K in (SQUARE, DIAMOND, HEXAGON, BALL):
        K2 = from_spec(K.spec())
        u = rng.normal(0, 1, (2, 20))
        assert np.allclose(K.gauge(u[0], u[1]), K2.gauge(u[0], u[1]))
    print("shape self-test OK (gauge vs bisection, support vs boundary sweep, "
          "gradients vs finite differences, touching identity, symmetry guard)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
