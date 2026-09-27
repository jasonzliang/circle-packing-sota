#!/usr/bin/env python3
"""sweep_all.py -- launch sweep_one.py over n x seeds through a process pool, collect results, keep the
best independently-validated packing per n, and print a table.

    python sweep_all.py --solver <tools/solver.py> --wall-seconds 300 [--cpu-seconds 280] \
        [--n 1-100] [--seeds 3 | --seed-list 0 1 2] [--jobs 60] [--warm <dir>] --out-dir <dir> [--resume]

Layout under --out-dir:
    jobs/n{N:03d}_s{S}.json     per-job result JSON written by sweep_one.py (never trusted blindly:
                                the packing is re-validated here again before it can become a best)
    results.jsonl               one line per finished job (n, seed, status, feasible, sum_radii, gap, ...)
    best/csqv{N}.pck + .json    best feasible packing per n (Packomania format, %.17g) + provenance
    table.txt                   the final table
    failures.txt                summary of non-ok jobs

Each job is single-threaded (OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 1). The per-job launcher timeout is
wall + grace + 60 s; sweep_one itself kills its child at wall + grace, so the launcher timeout only
fires if sweep_one is wedged (recorded as status=launcher_timeout).

--resume skips every (n, seed) already present in results.jsonl (add --retry-failed to redo jobs whose
status is not ok/infeasible/no_packing, e.g. killed or crashed ones).
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep_one  # noqa: E402  (validate_local, write_pck, load_records, THREAD_ENV)

SWEEP_ONE = os.path.join(HERE, "sweep_one.py")
FINAL_STATUSES = {"ok", "infeasible", "no_packing"}     # definitive: not retried by --resume


def parse_nlist(spec):
    out = []
    for part in spec.replace(",", " ").split():
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


class Collector:
    """Thread-safe results.jsonl appender + best-per-n keeper."""

    def __init__(self, out_dir, records):
        self.out_dir = out_dir
        self.records = records
        self.lock = threading.Lock()
        self.rows = []                                   # every finished job row
        self.best = {}                                   # n -> row (feasible, validated here)
        self.jsonl = os.path.join(out_dir, "results.jsonl")
        self.best_dir = os.path.join(out_dir, "best")
        os.makedirs(self.best_dir, exist_ok=True)

    def load_existing(self):
        if not os.path.isfile(self.jsonl):
            return
        for ln in open(self.jsonl):
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            self.rows.append(row)
        # rebuild best from the per-job JSONs (they carry the coordinates; results.jsonl does not)
        for row in self.rows:
            if row.get("feasible"):
                jp = os.path.join(self.out_dir, "jobs", row["job"])
                if os.path.isfile(jp):
                    try:
                        self._consider(json.load(open(jp)), row["job"], write=False)
                    except (OSError, ValueError):
                        pass
        # make sure best/ on disk reflects the rebuilt table
        for n, row in self.best.items():
            pck = os.path.join(self.best_dir, "csqv%d.pck" % n)
            if not os.path.isfile(pck):
                self._write_best(n, row)

    def done_pairs(self, retry_failed):
        done = set()
        for row in self.rows:
            if retry_failed and row.get("status") not in FINAL_STATUSES:
                continue
            done.add((int(row["n"]), int(row["seed"])))
        return done

    def _consider(self, res, job, write=True):
        """Independently re-validate the job's packing; if feasible and better, it becomes best[n]."""
        n = int(res["n"])
        centers, radii = res.get("centers"), res.get("radii")
        if not centers or not radii or len(centers) != len(radii):
            return False
        circ = [(c[0], c[1], r) for c, r in zip(centers, radii)]
        v = sweep_one.validate_local(circ, n)
        if not v["feasible"]:
            return False
        cur = self.best.get(n)
        if cur is not None and v["sum_r"] <= cur["sum_r"] + 1e-15:
            return False
        rec = self.records.get(n, (None, None))[0]
        row = {"n": n, "seed": int(res["seed"]), "sum_r": v["sum_r"], "record": rec,
               "gap": (None if rec is None else v["sum_r"] - rec),
               "digits": sweep_one.digits_closed(v["sum_r"], rec), "job": job,
               "solver_sha": res.get("solver_sha"), "wall_s": res.get("wall_s"),
               "cpu_s": res.get("cpu_s"), "nfev": res.get("nfev"), "circles": circ}
        self.best[n] = row
        if write:
            self._write_best(n, row)
        return True

    def _write_best(self, n, row):
        pck = os.path.join(self.best_dir, "csqv%d.pck" % n)
        landed = sweep_one.write_pck(pck, row["circles"],
                                     "sweep best n=%d seed=%d sha=%s" % (n, row["seed"], row["solver_sha"]))
        lv = sweep_one.validate_local(landed, n)
        if not lv["feasible"]:                           # what landed on disk must itself pass
            os.remove(pck)
            return
        prov = {k: v for k, v in row.items() if k != "circles"}
        prov["sum_r_on_disk"] = lv["sum_r"]
        with open(os.path.join(self.best_dir, "csqv%d.json" % n), "w") as fh:
            json.dump(prov, fh, indent=1)

    def add(self, res, job, launcher_status=None, n=None, seed=None):
        """Record a finished job. `res` may be None (sweep_one produced no JSON)."""
        with self.lock:
            if res is None:
                row = {"n": n, "seed": seed, "status": launcher_status or "no_json", "feasible": False,
                       "job": job}
            else:
                row = {"n": res["n"], "seed": res["seed"], "status": res.get("status"),
                       "feasible": bool(res.get("feasible")), "sum_radii": res.get("sum_radii"),
                       "record": res.get("record"), "gap": res.get("gap"), "digits": res.get("digits"),
                       "wall_s": res.get("wall_s"), "cpu_s": res.get("cpu_s"), "nfev": res.get("nfev"),
                       "killed": res.get("killed"), "wall_alarm_fired": res.get("wall_alarm_fired"),
                       "cpu_backstop_fired": res.get("cpu_backstop_fired"),
                       "max_overlap": res.get("max_overlap"), "max_outside": res.get("max_outside"),
                       "solver_error": (str(res.get("solver_error") or "").strip().splitlines() or [""])[-1][:200],
                       "job": job}
                if launcher_status:
                    row["status"] = launcher_status
                improved = False
                if row["feasible"]:
                    improved = self._consider(res, job)
                row["new_best"] = improved
            self.rows.append(row)
            with open(self.jsonl, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            return row


def run_job(a, n, seed, out_dir, env, warm):
    job = "n%03d_s%d.json" % (n, seed)
    jp = os.path.join(out_dir, "jobs", job)
    cmd = [a.python, SWEEP_ONE, "--solver", a.solver, "--n", str(n), "--seed", str(seed),
           "--wall-seconds", str(a.wall_seconds), "--out", jp, "--evals", str(a.evals),
           "--mission-dir", a.mission_dir, "--tmp-root", os.path.join(out_dir, "tmp")]
    if a.cpu_seconds:
        cmd += ["--cpu-seconds", str(a.cpu_seconds)]
    if a.solver_cpu is not None:
        cmd += ["--solver-cpu", str(a.solver_cpu)]
    if a.grace is not None:
        cmd += ["--grace", str(a.grace)]
    if warm:
        cmd += ["--warm", warm]
    if a.hide_records:
        cmd += ["--hide-records"]
    if a.solver_kwargs:
        cmd += ["--solver-kwargs", a.solver_kwargs]
    if a.module_attrs:
        cmd += ["--module-attrs", a.module_attrs]
    if a.records_file:
        cmd += ["--records-file", a.records_file]
    if a.warm_exclude_self:
        cmd += ["--warm-exclude-self"]
    if a.allow_forbidden:
        cmd += ["--allow-forbidden"]
    grace = a.grace if a.grace is not None else max(15.0, 0.25 * a.wall_seconds)
    timeout = a.wall_seconds + grace + 60.0
    t0 = time.time()
    launcher_status = None
    try:
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        rc, tail = p.returncode, p.stdout.decode("utf-8", "replace")[-600:]
    except subprocess.TimeoutExpired:
        rc, tail, launcher_status = -1, "", "launcher_timeout"
    res = None
    if os.path.isfile(jp):
        try:
            res = json.load(open(jp))
        except ValueError:
            res = None
    return job, rc, res, tail, launcher_status, time.time() - t0


def fmt_table(collector, ns, seeds_by_n):
    lines = ["%4s  %-16s  %-18s  %-12s  %-6s  %-9s  %-5s  %s" % (
        "n", "record", "best_sum_r", "gap", "digits", "seeds", "ok", "best_seed")]
    lines.append("-" * len(lines[0]))
    tot_d, n_rec = 0.0, 0
    for n in ns:
        rec = collector.records.get(n, (None, None))[0]
        b = collector.best.get(n)
        tried = seeds_by_n.get(n, [])
        ok = sum(1 for r in tried if r.get("feasible"))
        rec_s = ("%.12f" % rec) if rec is not None else "-"
        if b is None:
            lines.append("%4d  %-16s  %-18s  %-12s  %-6s  %-9s  %-5s  %s" % (
                n, rec_s, "-", "-", "-", "%d" % len(tried), ok, "-"))
            continue
        gap_s = ("%+.3e" % b["gap"]) if b["gap"] is not None else "-"
        dg = b["digits"]
        if dg is not None:
            tot_d += dg
            n_rec += 1
        lines.append("%4d  %-16s  %-18.12f  %-12s  %-6s  %-9s  %-5s  %s" % (
            n, rec_s, b["sum_r"], gap_s, ("%.3f" % dg) if dg is not None else "-", len(tried), ok, b["seed"]))
    if n_rec:
        lines.append("mean digits over %d n with records: %.4f" % (n_rec, tot_d / n_rec))
    beats = sorted(n for n, b in collector.best.items() if b["gap"] is not None and b["gap"] >= 1e-6)
    lines.append("strict beats (sum_r >= record + 1e-6): %d %s" % (len(beats), beats))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--n", default="1-100", help="n list/ranges, e.g. '1-100' or '10,27,55-60' (default 1-100)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds 0..K-1 (default 3)")
    ap.add_argument("--seed-list", type=int, nargs="*", default=None, help="explicit seeds (overrides --seeds)")
    ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--wall-seconds", type=float, required=True)
    ap.add_argument("--cpu-seconds", type=float, default=None)
    ap.add_argument("--solver-cpu", type=float, default=None)
    ap.add_argument("--evals", type=int, default=sweep_one.DEFAULT_EVALS)
    ap.add_argument("--grace", type=float, default=None)
    ap.add_argument("--warm", default=None, help="dir of csqv*.pck (or a single file) passed to every job")
    ap.add_argument("--warm-from-best", action="store_true",
                    help="pass <out-dir>/best as --warm (grows during the sweep; NOT reproducible)")
    ap.add_argument("--hide-records", action="store_true")
    ap.add_argument("--solver-kwargs", default=None)
    ap.add_argument("--module-attrs", default=None, help="JSON dict of solver module constants to override")
    ap.add_argument("--records-file", default=None, help="custom bench/records.json for every job")
    ap.add_argument("--warm-exclude-self", action="store_true")
    ap.add_argument("--allow-forbidden", action="store_true")
    ap.add_argument("--mission-dir", default=sweep_one.DEFAULT_MISSION)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args(argv)

    ns = parse_nlist(a.n)
    seeds = a.seed_list if a.seed_list is not None else list(range(a.seeds))
    out_dir = os.path.abspath(a.out_dir)
    os.makedirs(os.path.join(out_dir, "jobs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tmp"), exist_ok=True)
    records = sweep_one.load_records(os.path.abspath(a.mission_dir))
    col = Collector(out_dir, records)
    if a.resume:
        col.load_existing()
    done = col.done_pairs(a.retry_failed) if a.resume else set()
    todo = [(n, s) for s in seeds for n in ns if (n, s) not in done]     # seed-outer: full pass first
    warm = os.path.abspath(a.warm) if a.warm else None
    if a.warm_from_best:
        warm = col.best_dir
    env = dict(os.environ)
    for v in sweep_one.THREAD_ENV:
        env[v] = "1"
    print("sweep: %d n x %d seeds = %d jobs (%d skipped by --resume), jobs=%d, wall=%.0fs cpu=%s warm=%s out=%s"
          % (len(ns), len(seeds), len(todo), len(done), a.jobs, a.wall_seconds, a.cpu_seconds, warm, out_dir),
          flush=True)
    with open(os.path.join(out_dir, "sweep_args.json"), "a") as fh:
        fh.write(json.dumps({"t": time.time(), "argv": sys.argv[1:], "n_jobs": len(todo)}) + "\n")

    t0 = time.time()
    n_done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(run_job, a, n, s, out_dir, env, warm): (n, s) for n, s in todo}
        for fut in cf.as_completed(futs):
            n, s = futs[fut]
            try:
                job, rc, res, tail, lstat, dt = fut.result()
            except Exception as e:           # noqa: BLE001
                job, rc, res, tail, lstat, dt = "n%03d_s%d.json" % (n, s), -2, None, repr(e), "launcher_error", 0.0
            row = col.add(res, job, launcher_status=lstat, n=n, seed=s)
            n_done += 1
            b = col.best.get(n)
            print("[%4d/%4d %6.0fs] n=%3d seed=%d rc=%s status=%-14s feas=%-5s sum=%s gap=%s%s"
                  % (n_done, len(todo), time.time() - t0, n, s, rc, row.get("status"), row.get("feasible"),
                     ("%.12f" % row["sum_radii"]) if row.get("sum_radii") is not None else "-",
                     ("%+.2e" % row["gap"]) if row.get("gap") is not None else "-",
                     "  NEW BEST" if row.get("new_best") else ""), flush=True)
            if rc not in (0, 2, 3) and tail:
                print("    " + tail.strip().splitlines()[-1][:200] if tail.strip() else "", flush=True)

    seeds_by_n = {}
    for r in col.rows:
        if r.get("n") is not None:
            seeds_by_n.setdefault(int(r["n"]), []).append(r)
    table = fmt_table(col, ns, seeds_by_n)
    print("\n" + table)
    with open(os.path.join(out_dir, "table.txt"), "w") as fh:
        fh.write(table + "\n")
    fails = [r for r in col.rows if r.get("status") != "ok" or not r.get("feasible")]
    by_status = {}
    for r in fails:
        by_status.setdefault(str(r.get("status")), []).append((r.get("n"), r.get("seed")))
    lines = ["failures: %d of %d jobs" % (len(fails), len(col.rows))]
    for st, lst in sorted(by_status.items()):
        lines.append("  %-24s %4d  e.g. %s" % (st, len(lst), sorted(lst)[:12]))
    for r in fails[:50]:
        if r.get("solver_error"):
            lines.append("  n=%s seed=%s: %s" % (r.get("n"), r.get("seed"), r["solver_error"]))
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(out_dir, "failures.txt"), "w") as fh:
        fh.write(summary + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
