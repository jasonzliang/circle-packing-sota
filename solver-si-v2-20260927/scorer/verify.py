#!/usr/bin/env python3
"""bench/verify.py --check : score the workspace's committed circle-packing CENSUS -- VALIDATION ONLY.

The census is the workspace's bench/packs/ directory of Packomania `.pck` files. This scorer NEVER
solves: it only re-derives, from each committed packing's own coordinates, whether that packing is a
strictly-feasible packing of exactly n circles whose sum-of-radii reaches the frozen record. It prints
the two machine-readable lines analysis/capability.py reads (it takes the LAST of each):

    SCORE: <float>
    # BEATS: <count>     # the strict record-BEAT count (sum_r >= record + WIN_MARGIN, ties excluded).
                         #   Reported as the prestige integer; it is NOT the SCORE and NOT part of PASS.
    CHECK: PASS|FAIL     # PASS iff SCORE strictly > anchor.json["anchor"] (the same graded score for a
                         #   simple fixed-seed baseline census over the visible set, frozen at build time).

Because the score is a pure function of the committed coordinates -- no optimizer, no randomness, no
wall-clock term -- it is deterministic and contention-immune: identical on two runs, safe under -j16,
and fast (validating <=100 packings of <=100 circles each is a few milliseconds).

The scoring surface (this file, harness.py, records.json, anchor.json) is OPERATOR-FROZEN and byte-
identical across every run that seeds this dir; only bench/packs/ (the agent's census) varies per
iteration. Editing/regenerating the surface voids the cross-run comparison. Feasibility + sum_r use the
authoritative PURE-STDLIB harness.validate() -- no numpy in the score path.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                 # scorer dir FIRST so `import harness` is never shadowed
import harness  # noqa: E402

_PCK_RE = re.compile(r"csqv(\d+)\.pck\Z")


def _record_eval_check():
    """Ground-truth candidate-evaluation counter for analysis/evalcount.py.

    Append EXACTLY ONE line per `--check` PROCESS to an out-of-workspace sink, so the "samples-to-target"
    count is immune to Bash batching (a for-loop of N `--check`s = N processes = N lines, never 1) and
    SURVIVES the per-iteration `git reset --hard` (the sink is a SIBLING of the workspace, never inside
    it). Sink = $SI_EVAL_OUT when the harness exported it (operator/postrun context), else DERIVED from
    this scorer's own location -- the agent's in-iteration `--check` has NO $SI_EVAL_OUT set. Best-effort:
    any failure is swallowed so a grading NEVER crashes on an unwritable sink (SCORE/CHECK unaffected)."""
    import json as _json, os as _os, time as _time
    try:
        sink = _os.environ.get("SI_EVAL_OUT") or ""
        if not sink.strip():
            root = _os.path.dirname(HERE)               # workspace root (bench/ sits directly under it)
            sink = _os.path.join(_os.path.dirname(root), "_eval", _os.path.basename(root))
        _os.makedirs(sink, exist_ok=True)
        line = _json.dumps({"ts": _time.time(), "pid": _os.getpid(), "event": "check"}) + "\n"
        with open(_os.path.join(sink, "evalcount.jsonl"), "a", encoding="utf-8") as _fh:
            _fh.write(line)                             # O_APPEND single-line write: atomic per process
    except Exception:                                   # never let counting break a grading
        pass


def parse_pck(path):
    """[(x, y, r), ...] from a .pck: line1 = largest radius, line2 = label, remaining non-empty lines
    = 'x y r'. Raises ValueError on malformed content."""
    raw = [ln for ln in open(path).read().splitlines() if ln.strip()]
    if len(raw) < 3:
        raise ValueError("fewer than 3 non-empty lines")
    circ = []
    for ln in raw[2:]:
        parts = ln.split()
        if len(parts) != 3:
            raise ValueError(f"expected 'x y r', got {ln!r}")
        circ.append(tuple(float(p) for p in parts))
    return circ


def score_census(packs_dir, records, margin=harness.WIN_MARGIN, tol=harness.FEAS_TOL):
    """Return (count, rows). count = #n whose committed packing BEATS the record by >= margin.
    rows = per-n diagnostics for present files."""
    count, rows = 0, []
    if not os.path.isdir(packs_dir):
        return 0, rows
    files = {}
    for fn in os.listdir(packs_dir):
        m = _PCK_RE.match(fn)
        if m:
            files[int(m.group(1))] = os.path.join(packs_dir, fn)
    for n in sorted(files):
        rec = records.get(n)
        try:
            circ = parse_pck(files[n])
        except (OSError, ValueError) as e:
            rows.append((n, None, rec, "PARSE_ERR", str(e)))
            continue
        v = harness.validate(circ, n, tol=tol)
        if not v["feasible"]:
            why = ("count %d!=%d" % (v["n_found"], n) if not v["count_ok"]
                   else "min_r=%.2e" % v["min_r"] if v["min_r"] <= 0
                   else "wall=%.1e pair=%s" % (v["wall_slack"], v["pair_slack"]))
            rows.append((n, v["sum_r"], rec, "INFEASIBLE", why))
            continue
        if rec is None:
            rows.append((n, v["sum_r"], None, "no-record", ""))
            continue
        if harness.counts_for(v["sum_r"], rec, margin):
            count += 1
            rows.append((n, v["sum_r"], rec, "BEAT", "%+.2e" % (v["sum_r"] - rec)))
        else:
            rows.append((n, v["sum_r"], rec, "below", "%+.2e" % (v["sum_r"] - rec)))
    return count, rows


def graded_score(rows, records):
    """The mission SCORE: mean harness.digits_closed over the FULL visible record set.

    Averaged over `records`, NOT over the files present -- an n with no pack, an infeasible pack or a
    parse error contributes 0.0. That makes the denominator a frozen constant (37 today) instead of an
    agent-controllable one: under the old "mean over present files" rule an arm could delete its 32
    weakest packings and watch its score rise, and two arms with different census sizes were not
    comparable at all."""
    by_n = {r[0]: r for r in rows}
    total = 0.0
    for n, rec in records.items():
        r = by_n.get(n)
        if r is None or r[3] in ("PARSE_ERR", "INFEASIBLE") or r[1] is None:
            continue                                     # contributes 0.0
        total += harness.digits_closed(r[1], rec)
    return total / len(records) if records else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="run the PASS/FAIL census check")
    ap.add_argument("--packs", default=None, help="census dir (default: <bench>/packs)")
    ap.add_argument("--verbose", action="store_true", help="print every present n's verdict")
    args = ap.parse_args(argv)
    if args.check:
        _record_eval_check()             # ground-truth eval count (one line/process; see the helper)

    import json
    packs_dir = args.packs or os.path.join(HERE, "packs")
    records = harness.load_records()
    anchor_doc = json.load(open(os.path.join(HERE, "anchor.json")))
    anchor = float(anchor_doc["anchor"])

    count, rows = score_census(packs_dir, records)
    score = graded_score(rows, records)
    matched = sorted(r[0] for r in rows if r[3] == "BEAT")
    infeasible = [r[0] for r in rows if r[3] in ("INFEASIBLE", "PARSE_ERR")]
    passed = score > anchor

    print(f"SCORE: {score:.6f}")
    print(f"CHECK: {'PASS' if passed else 'FAIL'}")
    print(f"# SCORER: mean DIGITS of relative gap closed vs the best-known record, over ALL "
          f"{len(records)} visible n (missing/infeasible n score 0; cap {harness.DIGITS_CAP:g} "
          f"digits = at-or-above record). Validation only, no solving.  "
          f"feas_tol={harness.FEAS_TOL:g}  present={len(rows)}  infeasible/parse-err={len(infeasible)}")
    print(f"# BEATS: {count}  (strict record beats: sum_r >= record + {harness.WIN_MARGIN:g}; ties do "
          f"NOT count. This is the prestige integer, NOT the SCORE.)")
    print(f"# beaten n = {matched}")
    if infeasible:
        print(f"# INFEASIBLE/parse-error (NOT counted): {sorted(infeasible)}")
    print(f"# bar (anchor) = {anchor}  -> PASS needs SCORE > {anchor}")
    if args.verbose:
        for n, sr, rec, verdict, note in sorted(rows):
            sr_s = "-" if sr is None else f"{sr:.12f}"
            rec_s = "-" if rec is None else f"{rec:.12f}"
            print(f"#   n={n:>3}  sr={sr_s}  rec={rec_s}  {verdict} {note}")
    return 0 if passed else 1


if __name__ == "__main__":
    # Match the other frozen scorers: emit our own SCORE/CHECK, flush, then os._exit so nothing loaded
    # later can append a forged line (capability.py records the LAST match on merged stdout+stderr).
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc if isinstance(_rc, int) else 1)
