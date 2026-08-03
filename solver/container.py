#!/usr/bin/env python3
"""Convex containers for the circle-packing search.

WHY THIS EXISTS
---------------
The container enters the packing problem in exactly ONE place: the boundary
constraint

    r_i  <=  dist(c_i, boundary).

For every convex container that distance is a *min of smooth functions of the
centre* and -- this is the whole point -- it is **linear in r**.  So the
exactly-solvable inner layer the entire search rests on (radii-given-centres is
an LP) survives verbatim when the unit square is replaced by a disk, a triangle
or any convex polygon.  Only the wall rows change.  Generalising the container
therefore costs one small abstraction and buys a whole new axis of instances,
none of which have published sum-of-radii answers.

The derived bound generalises too: disjoint circles inside a container of area A
have total area <= A, so sum(pi r_i^2) <= A and Cauchy-Schwarz gives

    sum(r) <= sqrt(n * A / pi)                       [derived, not cited]

which is what makes efficiency comparable *across* containers.

EXACT-CHECKABILITY (why these particular containers)
----------------------------------------------------
`tools/exact_check.py` decides feasibility in exact rationals, with zero
tolerance.  That needs the wall test to be rational-representable:

  * polygon, rational vertices: with the UNNORMALISED outward normal a_k and
    offset c_k, "r_i <= (c_k - a_k.p_i)/|a_k|" is equivalent to
    (c_k - a_k.p_i) >= 0  AND  (c_k - a_k.p_i)^2 >= r_i^2 * |a_k|^2 -- exact.
  * disk of rational radius R: R - |p - o| >= r  is equivalent to
    (R - r) >= 0  AND  (R - r)^2 >= |p - o|^2 -- exact.

A unit-AREA disk (R = 1/sqrt(pi)) would break that, so the disk instances use
R = 1 and the comparison is made on efficiency, which is area-normalised anyway.

Interface (numpy arrays; x, y are length n):
    walls(x, y) -> (F, Fx, Fy), each (m, n).  Wall k for circle i is the
                   constraint F[k,i] - r_i >= 0 with dF/dx = Fx[k,i].
    cap(x, y)   -> per-circle max radius allowed by the boundary = min_k F[:,i].
    project(x, y, inset) -> centres pushed inside the (inset) container.
    sample(n, rng, inset) -> random centres inside the inset container.
    area, bbox, name, spec()

    python3 tools/container.py --self-test
"""
import argparse
import json
import math

import numpy as np


class Container:
    name = "container"
    is_box = False        # axis-aligned rectangle fast path

    def walls(self, x, y):
        raise NotImplementedError

    def cap(self, x, y):
        F, _, _ = self.walls(x, y)
        return np.maximum(F.min(axis=0), 0.0)

    def project(self, x, y, inset=0.0):
        raise NotImplementedError

    def sample(self, n, rng, inset=0.0):
        raise NotImplementedError

    def spec(self):
        raise NotImplementedError

    # -- derived quantities, all recomputable ---------------------------------
    def bound(self, n):
        """sum(r) <= sqrt(n*A/pi) -- derived above, no citation needed."""
        return math.sqrt(n * self.area / math.pi)

    def inradius(self):
        """Largest single circle = max over the container of dist-to-boundary."""
        raise NotImplementedError


class Polygon(Container):
    """Convex polygon, vertices in COUNTER-CLOCKWISE order.

    Stores both the unit-normal form (for the optimiser) and the exact rational
    unnormalised form (for the checker), so the two never drift apart.
    """

    def __init__(self, vertices, name="polygon"):
        self.name = name
        self.V = np.asarray(vertices, dtype=float)
        assert self.V.ndim == 2 and self.V.shape[1] == 2 and len(self.V) >= 3
        v0 = self.V
        v1 = np.roll(self.V, -1, axis=0)
        e = v1 - v0
        # CCW polygon: outward normal of edge (v0 -> v1) is (dy, -dx).
        nrm = np.stack([e[:, 1], -e[:, 0]], axis=1)
        self.A_raw = nrm                                   # unnormalised
        self.c_raw = np.einsum("ij,ij->i", nrm, v0)
        L = np.hypot(nrm[:, 0], nrm[:, 1])
        assert np.all(L > 0), "degenerate edge"
        self.A = nrm / L[:, None]                          # unit outward normals
        self.c = self.c_raw / L
        # shoelace; positive iff CCW
        self.area = 0.5 * float(np.sum(v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]))
        assert self.area > 0, "vertices must be counter-clockwise"
        self.bbox = (float(self.V[:, 0].min()), float(self.V[:, 0].max()),
                     float(self.V[:, 1].min()), float(self.V[:, 1].max()))
        # axis-aligned rectangle fast path: keeps the unit square bit-identical
        # to the pre-container code (clip per coordinate, uniform per coordinate)
        self._is_box = (len(self.V) == 4 and
                        np.allclose(np.abs(self.A), np.array([[0., 1.], [1., 0.],
                                                              [0., 1.], [1., 0.]]),
                                    atol=0) is False)
        xs, ys = sorted(set(np.round(self.V[:, 0], 15))), sorted(set(np.round(self.V[:, 1], 15)))
        self._is_box = len(self.V) == 4 and len(xs) == 2 and len(ys) == 2
        self.is_box = self._is_box

    def walls(self, x, y):
        F = self.c[:, None] - self.A[:, 0:1] * x[None, :] - self.A[:, 1:2] * y[None, :]
        m = len(self.c)
        Fx = np.repeat(-self.A[:, 0:1], len(x), axis=1)
        Fy = np.repeat(-self.A[:, 1:2], len(x), axis=1)
        assert F.shape == (m, len(x))
        return F, Fx, Fy

    def inset_vertices(self, inset):
        """Vertices of the polygon shrunk by `inset` (adjacent inset edges meet)."""
        m = len(self.c)
        W = np.empty((m, 2))
        for k in range(m):
            k2 = (k + 1) % m
            M = np.array([self.A[k], self.A[k2]])
            b = np.array([self.c[k] - inset, self.c[k2] - inset])
            W[k] = np.linalg.solve(M, b)
        return W

    def project(self, x, y, inset=0.0):
        """Exact Euclidean projection onto the inset polygon.

        One sweep of half-plane clipping (POCS) is NOT enough near a corner --
        measured: a triangle left points 0.0045 inside a 0.01 inset after 8
        sweeps.  So: if the point is already inside, keep it; otherwise take the
        nearest point on the inset boundary, which is exact in one pass.
        """
        x, y = np.array(x, dtype=float), np.array(y, dtype=float)
        if self._is_box:                       # axis-aligned: clipping IS exact
            for k in range(len(self.c)):
                g = self.c[k] - inset - self.A[k, 0] * x - self.A[k, 1] * y
                bad = g < 0
                if np.any(bad):
                    x[bad] += self.A[k, 0] * g[bad]
                    y[bad] += self.A[k, 1] * g[bad]
            return x, y
        slack = self.c[:, None] - inset - self.A[:, 0:1] * x - self.A[:, 1:2] * y
        out = np.flatnonzero(slack.min(axis=0) < 0)
        if out.size == 0:
            return x, y
        W = self.inset_vertices(inset)
        px, py = x[out], y[out]
        bestd = np.full(out.size, np.inf)
        bx, by = px.copy(), py.copy()
        m = len(W)
        for k in range(m):
            a, b = W[k], W[(k + 1) % m]
            ex, ey = b[0] - a[0], b[1] - a[1]
            L2 = ex * ex + ey * ey
            t = np.clip(((px - a[0]) * ex + (py - a[1]) * ey) / max(L2, 1e-300), 0.0, 1.0)
            cx, cy = a[0] + t * ex, a[1] + t * ey
            d = (px - cx) ** 2 + (py - cy) ** 2
            better = d < bestd
            bestd = np.where(better, d, bestd)
            bx = np.where(better, cx, bx)
            by = np.where(better, cy, by)
        x[out], y[out] = bx, by
        return x, y

    def sample(self, n, rng, inset=0.0):
        xlo, xhi, ylo, yhi = self.bbox
        if self._is_box:
            # exactly the pre-container behaviour for the unit square
            x = rng.uniform(xlo + inset, xhi - inset, n)
            y = rng.uniform(ylo + inset, yhi - inset, n)
            return x, y
        x = rng.uniform(xlo, xhi, n)
        y = rng.uniform(ylo, yhi, n)
        return self.project(x, y, inset)

    def inradius(self):
        """Chebyshev radius, solved exactly rather than searched.

        max_p min_k (c_k - a_k.p) is an LP whose optimum has (generically) three
        active constraints, so enumerate triples: c_i - a_i.p = c_j - a_j.p =
        c_k - a_k.p = t is a 3x3 linear system in (px, py, t).  Keep the largest
        t that is actually feasible.  Exact, and no solver dependency.
        """
        m = len(self.c)
        best = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    M = np.array([[self.A[i, 0], self.A[i, 1], 1.0],
                                  [self.A[j, 0], self.A[j, 1], 1.0],
                                  [self.A[k, 0], self.A[k, 1], 1.0]])
                    b = np.array([self.c[i], self.c[j], self.c[k]])
                    try:
                        px, py, t = np.linalg.solve(M, b)
                    except np.linalg.LinAlgError:
                        continue
                    if t <= best:
                        continue
                    f = self.cap(np.array([px]), np.array([py]))[0]
                    if f >= t - 1e-12:
                        best = float(t)
        return best

    def spec(self):
        return {"type": "polygon", "name": self.name,
                "vertices": [[float(a), float(b)] for a, b in self.V]}


class Disk(Container):
    def __init__(self, cx=0.0, cy=0.0, R=1.0, name="disk"):
        self.name = name
        self.o = np.array([float(cx), float(cy)])
        self.R = float(R)
        self.area = math.pi * self.R ** 2
        self.bbox = (cx - R, cx + R, cy - R, cy + R)

    def walls(self, x, y):
        dx, dy = x - self.o[0], y - self.o[1]
        rho = np.hypot(dx, dy)
        safe = np.where(rho > 1e-15, rho, 1.0)
        F = (self.R - rho)[None, :]
        Fx = (-dx / safe)[None, :]
        Fy = (-dy / safe)[None, :]
        return F, Fx, Fy

    def project(self, x, y, inset=0.0):
        x, y = np.array(x, dtype=float), np.array(y, dtype=float)
        dx, dy = x - self.o[0], y - self.o[1]
        rho = np.hypot(dx, dy)
        lim = max(self.R - inset, 0.0)
        bad = rho > lim
        if np.any(bad):
            s = lim / np.where(rho[bad] > 0, rho[bad], 1.0)
            x[bad] = self.o[0] + dx[bad] * s
            y[bad] = self.o[1] + dy[bad] * s
        return x, y

    def sample(self, n, rng, inset=0.0):
        lim = max(self.R - inset, 0.0)
        u = rng.uniform(0.0, 1.0, n)
        t = rng.uniform(0.0, 2 * math.pi, n)
        rad = lim * np.sqrt(u)
        return self.o[0] + rad * np.cos(t), self.o[1] + rad * np.sin(t)

    def inradius(self):
        return self.R

    def spec(self):
        return {"type": "disk", "name": self.name,
                "center": [float(self.o[0]), float(self.o[1])], "radius": self.R}


# ------------------------------------------------------------- the registry --

UNIT_SQUARE = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], name="unit_square")
UNIT_DISK = Disk(0.0, 0.0, 1.0, name="unit_disk")
RIGHT_TRI = Polygon([(0, 0), (1, 0), (0, 1)], name="right_triangle")

REGISTRY = {c.name: c for c in (UNIT_SQUARE, UNIT_DISK, RIGHT_TRI)}


def from_spec(spec):
    if spec["type"] == "disk":
        return Disk(spec["center"][0], spec["center"][1], spec["radius"],
                    name=spec.get("name", "disk"))
    return Polygon(spec["vertices"], name=spec.get("name", "polygon"))


def get(name):
    if name in REGISTRY:
        return REGISTRY[name]
    raise SystemExit(f"unknown container {name!r}; have {sorted(REGISTRY)}")


# ------------------------------------------------------------------ tests ----

def self_test():
    rng = np.random.default_rng(0)

    # 1. unit square: the wall rows must reproduce the hand-written ones exactly
    x = rng.uniform(0.05, 0.95, 7)
    y = rng.uniform(0.05, 0.95, 7)
    F, Fx, Fy = UNIT_SQUARE.walls(x, y)
    want = np.stack([y, 1 - x, 1 - y, x])       # edge order: bottom,right,top,left
    assert np.array_equal(np.sort(F, axis=0), np.sort(want, axis=0)), F
    cap = UNIT_SQUARE.cap(x, y)
    assert np.allclose(cap, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    assert abs(UNIT_SQUARE.area - 1.0) < 1e-15
    assert abs(UNIT_SQUARE.inradius() - 0.5) < 1e-6

    # 2. gradients: finite differences on every container
    for ct in (UNIT_SQUARE, UNIT_DISK, RIGHT_TRI):
        px, py = ct.sample(6, rng, inset=0.15)
        F0, Fx0, Fy0 = ct.walls(px, py)
        h = 1e-7
        Fh = ct.walls(px + h, py)[0]
        assert np.allclose((Fh - F0) / h, Fx0, atol=1e-5), (ct.name, "d/dx")
        Fh = ct.walls(px, py + h)[0]
        assert np.allclose((Fh - F0) / h, Fy0, atol=1e-5), (ct.name, "d/dy")

    # 3. areas and inradii against closed forms
    assert abs(UNIT_DISK.area - math.pi) < 1e-12
    assert abs(RIGHT_TRI.area - 0.5) < 1e-15
    # right triangle legs 1,1, hypotenuse sqrt2 -> inradius (a+b-c)/2
    assert abs(RIGHT_TRI.inradius() - (2 - math.sqrt(2)) / 2) < 1e-6, RIGHT_TRI.inradius()
    assert abs(UNIT_DISK.inradius() - 1.0) < 1e-15

    # 4. project() really lands inside, sample() really starts inside
    for ct in (UNIT_SQUARE, UNIT_DISK, RIGHT_TRI):
        bx = rng.uniform(-1.5, 2.0, 200)
        by = rng.uniform(-1.5, 2.0, 200)
        qx, qy = ct.project(bx, by, inset=0.01)
        assert np.all(ct.cap(qx, qy) >= 0.01 - 1e-9), (ct.name, ct.cap(qx, qy).min())
        sx, sy = ct.sample(200, rng, inset=0.02)
        assert np.all(ct.cap(sx, sy) >= 0.02 - 1e-9), (ct.name, ct.cap(sx, sy).min())

    # 5. the derived bound, and round-tripping through spec()
    for ct in (UNIT_SQUARE, UNIT_DISK, RIGHT_TRI):
        assert abs(ct.bound(26) - math.sqrt(26 * ct.area / math.pi)) < 1e-15
        rt = from_spec(json.loads(json.dumps(ct.spec())))
        tx, ty = ct.sample(5, rng, inset=0.05)
        assert np.allclose(rt.cap(tx, ty), ct.cap(tx, ty)), ct.name
        assert abs(rt.area - ct.area) < 1e-12

    # 6. a NON-convex / clockwise polygon must be rejected, not silently wrong
    try:
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])      # clockwise
    except AssertionError:
        pass
    else:
        raise AssertionError("clockwise polygon was accepted")

    print("container self-test OK "
          f"(square A=1 in={UNIT_SQUARE.inradius():.6f}, "
          f"disk A={UNIT_DISK.area:.6f}, tri A={RIGHT_TRI.area} "
          f"in={RIGHT_TRI.inradius():.6f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
