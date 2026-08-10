#!/usr/bin/env bash
# End-to-end reproduction: sweep the self-improved solver across N, compare to the Packomania records, and
# independently verify every packing. One command, from a clean checkout.
#
#   ./reproduce.sh                 # full sweep N=2..100 @ 120s/N, 8 workers (~25 min on a 10-core mac)
#   ./reproduce.sh 20 30 60 4      # quick: N=20..30 @ 60s/N, 4 workers
#   NMIN=27 NMAX=27 TIME=120 ./reproduce.sh   # a single N
#
# Args (all optional): <nmin> <nmax> <time_s_per_N> <workers>. Env vars NMIN/NMAX/TIME/WORKERS/SEEDS override.
set -euo pipefail
cd "$(dirname "$0")"

NMIN="${1:-${NMIN:-2}}"; NMAX="${2:-${NMAX:-100}}"; TIME="${3:-${TIME:-120}}"
WORKERS="${4:-${WORKERS:-8}}"; SEEDS="${SEEDS:-1}"; OUT="repro"    # fresh dir; does NOT clobber the committed sota/nietzsche-sm-radical-v6-n27 reference

echo "== deps ==" && python3 -c "import numpy,scipy; print(' numpy',numpy.__version__,'scipy',scipy.__version__)" \
  || { echo "Missing deps. Run: pip install -r requirements.txt"; exit 1; }

echo "== sweep N=$NMIN..$NMAX (${TIME}s/N, $WORKERS workers, seeds $SEEDS) -> $OUT/ =="
# BLAS pinned to 1 thread/worker so the pool doesn't oversubscribe cores.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 solver-nietzsche-sm-radical-v6-n27/run_sweep.py --nmin "$NMIN" --nmax "$NMAX" --time "$TIME" \
          --workers "$WORKERS" --seeds "$SEEDS" --out-dir "$OUT"

echo "== compare to LIVE Packomania csqv records -> $OUT/comparison.md =="
python3 verify_and_compare.py compare --pck-dir "$OUT" --refresh --out "$OUT/comparison.md" | sed -n '1,6p'

echo "== independently verify every emitted packing (pure-stdlib, no solver) =="
fail=0
for f in "$OUT"/pck/csqv*.pck; do
  python3 verify_and_compare.py verify "$f" >/dev/null || { echo "  INFEASIBLE: $f"; fail=1; }
done
[ "$fail" = 0 ] && echo "  all packings strictly feasible" || { echo "  *** some packings failed verification ***"; exit 1; }
echo "== done. See $OUT/comparison.md and $OUT/pck/ =="
