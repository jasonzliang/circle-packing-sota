#!/usr/bin/env python3
"""publish.py -- aggregate every sweep round, export the best strict packing per n into the
circle-packing-sota repo (sota/<artifact>/), regenerate that directory's README from the manifest, and
commit + push when anything changed. Safe to run every few minutes.

  python publish.py [--no-push] [--rounds-root /tmp/si_tools/cp_sweep] [--repo ~/Desktop/circle-packing-sota]
                    [--artifact sota/si-v2-20260927] [--records sota/packomania/history/packomania_csqv_2026-09-27.json]
"""
import argparse, glob, json, os, subprocess, sys, time, csv, collections
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

def sh(cmd, cwd=None, timeout=1800):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)

def write_readme(repo, artifact, records_rel, manifest, solvers_md, per_n_csv, campaign_note):
    m = manifest; per = m["per_n"]
    wins = m["wins"]; ties = m["ties"]; below = m["below"]
    port = json.load(open(os.path.join(HERE, "portfolio.json")))
    used = collections.Counter(v["provenance"].get("solver") or "census" for v in per.values())
    win_rows = []
    for n in wins:
        v = per[str(n)]; p = v["provenance"]
        who = ("`%s` %s seed %s" % (p.get("solver"), p.get("mode"), p.get("seed"))) if p.get("source") == "sweep" else "stored census packing (%s)" % (os.path.basename(os.path.dirname(os.path.dirname(p.get("warm_path", "") or ""))) or "pre-sweep")
        win_rows.append("| %d | %.12f | %.12f | %+.3e | %+.2e | %s |" % (n, v["sum_r"], v["record"], v["margin"], v["margin"] / v["record"], who))
    below_rows = ["| %d | %.12f | %.12f | %+.3e |" % (n, per[str(n)]["sum_r"], per[str(n)]["record"], per[str(n)]["margin"]) for n in below]
    prov_rows = []
    for k, v in port.items():
        rr = v.get("run_record") or {}
        prov_rows.append("| `%s` | %s | %s | %s | %s | `%s` | %d |" % (k, v.get("origin", ""), rr.get("values_file", "-"), (rr.get("values_md5") or "-")[:8],
                         ("%.3f" % rr["heldout_endpoint"]) if isinstance(rr.get("heldout_endpoint"), (int, float)) else ("void" if rr.get("heldout_endpoint_state") == "void" else "-"),
                         v["sha256"][:12], used.get(k, 0)))
    txt = f"""# {artifact.split('/')[-1]}: best packing per N from the SI-v2 solver portfolio sweep

**{len(per)} packings (N = 1..100)**, the best strictly feasible packing per N found by sweeping fifteen
LLM-evolved circle-packing solvers from the self-improvement-v2 (SI-v2) `circle-packing` mission, seeded
from every packing those runs ever stored. Compared against the Packomania `csqv` table
**retrieved 2026-09-27** (`{records_rel}`): **{len(wins)} WIN / {len(ties)} tie / {len(below)} below**.

{campaign_note}

This directory is regenerated automatically by the sweep campaign; every number in it is recomputed
from the `.pck` coordinates at each update. Read *Margins, honestly* before quoting any of it.

## The problem and the rule

Place N circles of freely chosen radii in the unit square (centred, `[-0.5, 0.5]^2`), no overlaps,
all inside, maximise the sum of radii. Packomania's values are best-known, not proven optima.

A packing is exported only if it is **strictly feasible at zero tolerance** (every wall slack and every
pairwise slack `>= 0` in double precision, recomputed from the coordinates by two independent
implementations). A **WIN** means `sum_r > record + 1e-9` (the repository's `WIN_EPS`); `tie` means
`|sum_r - record| <= 1e-9`; `below` is everything else.

## Wins against the 2026-09-27 table

| N | ours Σr | record | Δ (abs) | Δ (rel) | found by |
|---:|---|---|---:|---:|---|
{chr(10).join(win_rows) if win_rows else '| - | - | - | - | - | - |'}

## Sizes still below the table

| N | ours Σr | record | Δ |
|---:|---|---|---:|
{chr(10).join(below_rows) if below_rows else '| - | - | - | - |'}

Every other N ties the tabulated value to within 1e-9.

## Margins, honestly

The wins are refinements in the fourth to sixth significant figure of known configurations, not new
packing families, at relative margins of about 1e-5 to 3e-4. The 2026-09-27 table already absorbed
many values this programme submitted in August; a row marked "stored census packing" was found by an
SI-v2 run before this sweep and merely re-verified here. Packomania updates continuously: re-fetch
before repeating any claim (`python3 verify_and_compare.py fetch`).

## Verify

From the repository root, re-verify every packing from coordinates at zero tolerance and rebuild the
comparison table:

```bash
python3 verify_and_compare.py compare --pck-dir {artifact} \\
  --records {records_rel} --tol 0 --out /tmp/comparison-si-v2.md
```

`manifest.json` records, per N, the recomputed Σr, both minimum slacks, the sha256 of the `.pck`, and
full provenance (solver key, warm mode, seed, job path, solver sha256 and origin run).

## Solver portfolio and per-N specialisation

Fifteen solvers were swept. Each is a single `tools/solver.py` evolved by an LLM agent inside one
SI-v2 run (values arm and brief noted below); the solver files and the sweep tooling are in
`solver-si-v2-20260927/`. "packings" counts how many of the exported bests each solver produced
("census" = a stored pre-sweep packing was never beaten).

| key | origin run | values file | values md5 | held-out endpoint | solver sha256 | packings |
|---|---|---|---|---:|---|---:|
{chr(10).join(prov_rows)}

Per-solver sweep statistics (mean digits vs the live table by N band, unique bests, improvements):

{solvers_md}

## Files

```
pck/csqv<N>.pck     Packomania text format: line 1 = largest radius, line 2 = author, then N lines
                    "x y r" at %.17g (bit-exact doubles), centred square [-0.5, 0.5]^2
json/out<N>.json    sidecar in the corner frame [0,1]^2 (x+0.5, y+0.5, r) with Σr, slacks, provenance
results.csv         n, sum_radii, max_violation, record_20260927, delta, feasible
comparison.md       output of verify_and_compare.py compare --tol 0 against the 2026-09-27 snapshot
manifest.json       per-N recomputed values, sha256 of every pck, provenance, counts
survey_all_runs.md  survey of every packing ever stored by the SI-v2 runs on all three machines:
                    which stored packings beat the current table, and how many independent runs found each
README.md           this file (generated)
```
"""
    with open(os.path.join(repo, artifact, "README.md"), "w") as fh:
        fh.write(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds-root", default=HERE)
    ap.add_argument("--repo", default=os.path.expanduser("~/Desktop/circle-packing-sota"))
    ap.add_argument("--artifact", default="sota/si-v2-20260927")
    ap.add_argument("--records", default="sota/packomania/history/packomania_csqv_2026-09-27.json")
    ap.add_argument("--warm", default=os.environ.get("CP_CENSUS", "/tmp/si_tools/cp_census") + "/warm_p1")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    t0 = time.time()
    phases = sorted(d for d in glob.glob(os.path.join(a.rounds_root, "*")) if os.path.isdir(os.path.join(d, "jobs")))
    agg = os.path.join(a.rounds_root, "agg")
    rc, out = sh([PY, os.path.join(HERE, "aggregate.py")] + sum([["--phase", p] for p in phases], []) + ["--warm", a.warm, "--out", agg])
    if rc != 0:
        print("AGGREGATE FAILED:", out[-800:]); sys.exit(1)
    summ = json.load(open(os.path.join(agg, "summary.json")))
    art = os.path.join(a.repo, a.artifact); rec = os.path.join(a.repo, a.records)
    rc, out = sh([PY, os.path.join(HERE, "export_sota.py"), "--best", os.path.join(agg, "best"), "--records", rec, "--out", art])
    if rc != 0:
        print("EXPORT FAILED:", out[-800:]); sys.exit(1)
    man = json.load(open(os.path.join(art, "manifest.json")))
    solvers_md = open(os.path.join(agg, "solvers.md")).read()
    note = a.note or ("Sweep campaign status at %s UTC: %d jobs aggregated from %s." % (time.strftime("%Y-%m-%d %H:%M"), summ["jobs"], ", ".join(os.path.basename(p) for p in phases)))
    write_readme(a.repo, a.artifact, a.records, man, solvers_md, os.path.join(agg, "per_n.csv"), note)
    # git
    rc, st = sh(["git", "status", "--porcelain", "--", a.artifact], cwd=a.repo)
    changed = [ln[3:] for ln in st.splitlines() if ln.strip()]
    changed_n = sorted({int(os.path.basename(c)[4:-4]) for c in changed if os.path.basename(c).startswith("csqv") and c.endswith(".pck")})
    # a README whose only change is its timestamp note is not worth a commit: revert it unless something substantive moved
    substantive = [c for c in changed if not c.endswith("README.md")]
    if changed and not substantive:
        sh(["git", "checkout", "--", os.path.join(a.artifact, "README.md")], cwd=a.repo)
        changed = []
    line = "%s publish: jobs=%d WIN=%d tie=%d below=%d exported=%d" % (time.strftime("%H:%M:%S"), summ["jobs"], man["counts"]["WIN"], man["counts"]["tie"], man["counts"]["below"], man["counts"]["exported"])
    if changed:
        sh(["git", "add", "--", a.artifact], cwd=a.repo)
        msg = "%s: %d WIN / %d tie / %d below vs packomania 2026-09-27" % (a.artifact.split("/")[-1], man["counts"]["WIN"], man["counts"]["tie"], man["counts"]["below"])
        if changed_n: msg += "; new/improved N = %s" % ",".join(map(str, changed_n))
        msg += "\n\nAutomated update from the SI-v2 solver-portfolio sweep (%d jobs aggregated).\nEvery packing re-verified from coordinates at zero tolerance.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" % summ["jobs"]
        rc, out = sh(["git", "commit", "-q", "-m", msg], cwd=a.repo)
        rc2, sha = sh(["git", "rev-parse", "--short", "HEAD"], cwd=a.repo)
        line += " | COMMIT %s (%d files, N %s)" % (sha.strip(), len(changed), changed_n)
        if not a.no_push:
            for attempt in range(3):
                rc, out = sh(["git", "push", "-q", "origin", "HEAD"], cwd=a.repo, timeout=300)
                if rc == 0: line += " pushed"; break
                time.sleep(20)
            else: line += " PUSH FAILED: " + out.strip()[-200:]
    else:
        line += " | no change"
    line += " (%.0fs)" % (time.time() - t0)
    # WINS against the CURRENT Packomania table: re-fetch at most hourly, fall back to the dated snapshot
    live_path = os.environ.get("CP_LIVE", "/tmp/si_tools/cp_live") + "/latest.json"
    try:
        if not os.path.isfile(live_path) or time.time() - os.path.getmtime(live_path) > 3600:
            r = subprocess.run([PY, os.path.join(a.repo, "verify_and_compare.py"), "fetch", "--out", live_path + ".tmp"], capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and os.path.isfile(live_path + ".tmp"): os.replace(live_path + ".tmp", live_path)
        live_doc = json.load(open(live_path)); live_src = "live %s" % time.strftime("%H:%M", time.gmtime(os.path.getmtime(live_path)))
    except Exception:
        live_doc = json.load(open(rec)); live_src = "snapshot 2026-09-27"
    live = {int(k): float(v) for k, v in live_doc["records"].items()}
    snap = {int(k): float(v) for k, v in json.load(open(rec))["records"].items()}
    drift = sorted(n for n in range(1, 101) if n in live and n in snap and abs(live[n] - snap[n]) > 1e-12)
    wins = [(int(n), v["sum_r"] - live[int(n)]) for n, v in man["per_n"].items() if int(n) in live and v["sum_r"] > live[int(n)] + 1e-9]
    wins.sort()
    line += " || BEATEN vs %s (%d): " % (live_src, len(wins)) + ", ".join("%d(%+.1e)" % w for w in wins)
    if drift: line += " || table moved since 09-27 at N=%s" % ",".join(map(str, drift))
    print(line)
main()
