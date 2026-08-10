"""Canonical entrypoint for the circle-packing SOTA solver.

The record-beating solver is the n-generic pipeline; this module re-exports it so
the conventional `solve(n, seconds=None, seed=0)` front door IS the best solver.
The original iteration-1 multistart solver is preserved as solve_legacy.py.
"""
from pipeline import solve  # noqa: F401
