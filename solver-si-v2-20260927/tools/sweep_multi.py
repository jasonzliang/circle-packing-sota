#!/usr/bin/env python3
"""sweep_multi.py -- run a schedule of (solver, n, seed, mode, cpu) jobs through one pool of workers.

    python sweep_multi.py --schedule phase1.jsonl --out-dir phase1 --jobs 60 \
        --warm $CP_CENSUS/warm_p1 --records-file $CP_CENSUS/records_live_inflated.json

schedule line: {"solver": "<portfolio key>", "n": 27, "seed": 1, "mode": "self"|"nbr", "cpu": 240}
  mode self = warm dir as is (incumbent + neighbours); nbr = incumbent csqv<n>.pck withheld (transfer/cold start)
Each job -> <out-dir>/jobs/<solver>/<mode>/n###_s#.json (sweep_one result; re-validated later by aggregate.py).
Resumable: jobs whose JSON exists with a status are skipped. Longest jobs first (LPT).
"""
import argparse, json, os, random, subprocess, sys, time, threading, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

def attrs_for(spec, cpu):
    v = cpu
    if spec.get("cpu_div"): v = cpu / spec["cpu_div"]
    if spec.get("cpu_add"): v = cpu + spec["cpu_add"]
    return {a: v for a in spec["attrs"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--portfolio", default=os.path.join(HERE, "portfolio.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--warm", required=True)
    ap.add_argument("--records-file", default=None)
    ap.add_argument("--hide-records", action="store_true")
    ap.add_argument("--wall-factor", type=float, default=1.25)
    ap.add_argument("--wall-add", type=float, default=30.0)
    ap.add_argument("--grace", type=float, default=60.0)
    a = ap.parse_args()
    port = json.load(open(a.portfolio))
    sched = [json.loads(l) for l in open(a.schedule) if l.strip()]
    os.makedirs(os.path.join(a.out_dir, "jobs"), exist_ok=True)
    todo = []
    for j in sched:
        jp = os.path.join(a.out_dir, "jobs", j["solver"], j["mode"], "n%03d_s%d.json" % (j["n"], j["seed"]))
        if os.path.isfile(jp):
            try:
                if json.load(open(jp)).get("status"): continue
            except ValueError: pass
        todo.append((j, jp))
    # Order: every size band progresses at the same pace through the round (fractional rank within the
    # band, longest jobs first inside a band). Pure longest-first would run all of n=78..100 before any
    # n=51..77 job started, leaving the mid sizes untouched for hours.
    def band(n): return 0 if n <= 50 else 1 if n <= 77 else 2
    groups = {}
    for t in todo: groups.setdefault((band(t[0]["n"]), t[0]["mode"]), []).append(t)   # per band AND mode
    ordered = []
    for g in groups.values():
        g.sort(key=lambda t: -t[0]["cpu"])
        for i, t in enumerate(g): ordered.append(((i + 0.5) / len(g), -t[0]["cpu"], t))
    ordered.sort(key=lambda x: (x[0], x[1]))
    todo = [t for _, _, t in ordered]
    total, skipped = len(todo), len(sched) - len(todo)
    log = open(os.path.join(a.out_dir, "launcher.log"), "a")
    def say(msg):
        line = "%s %s" % (time.strftime("%H:%M:%S"), msg); print(line, flush=True); log.write(line + "\n"); log.flush()
    say("schedule %d jobs, %d already done, %d to run, %d workers, cpu-s total %.0f (%.1f CPU-h, ~%.1f h wall at %d wide)" % (
        len(sched), skipped, total, a.jobs, sum(j["cpu"] for j, _ in todo), sum(j["cpu"] for j, _ in todo) / 3600,
        sum(j["cpu"] for j, _ in todo) * 1.08 / 3600 / a.jobs, a.jobs))
    lock = threading.Lock(); done = [0]; t_start = time.time(); stats = {"ok": 0, "feasible": 0, "other": 0}
    def run(t):
        j, jp = t
        spec = port[j["solver"]]; cpu = float(j["cpu"]); wall = cpu * a.wall_factor + a.wall_add
        os.makedirs(os.path.dirname(jp), exist_ok=True)
        cmd = [PY, os.path.join(HERE, "sweep_one.py"), "--solver", spec["solver"], "--n", str(j["n"]), "--seed", str(j["seed"]),
               "--wall-seconds", str(wall), "--cpu-seconds", str(cpu + 5.0), "--solver-cpu", str(cpu), "--grace", str(a.grace),
               "--warm", a.warm, "--module-attrs", json.dumps(attrs_for(spec, cpu)),
               "--tmp-root", os.path.join(a.out_dir, "tmp"), "--out", jp]
        if a.records_file: cmd += ["--records-file", a.records_file]
        if a.hide_records: cmd += ["--hide-records"]
        if j["mode"] == "nbr": cmd += ["--warm-exclude-self"]
        if j["mode"].startswith("kick"): cmd += ["--warm-kick", str(j.get("kick", 0.1))]
        env = dict(os.environ); env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        st = "?"
        try:
            subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=wall + a.grace + 90)
            r = json.load(open(jp)); st = r.get("status"); feas = r.get("feasible")
        except Exception as e:                       # noqa: BLE001
            st, feas = "launcher_error", False
            try: json.dump({"status": "launcher_error", "error": repr(e), **j}, open(jp, "w"))
            except OSError: pass
        with lock:
            done[0] += 1
            stats["ok" if st == "ok" else "other"] += 1; stats["feasible"] += bool(feas)
            if done[0] % 25 == 0 or done[0] == total:
                el = time.time() - t_start
                say("done %d/%d (%.0f%%) ok=%d feasible=%d other=%d elapsed %.1f h, ETA %.1f h" % (
                    done[0], total, 100.0 * done[0] / total, stats["ok"], stats["feasible"], stats["other"], el / 3600,
                    el / done[0] * (total - done[0]) / 3600))
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(run, todo))
    say("SWEEP DONE %s" % a.out_dir)
    open(os.path.join(a.out_dir, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
if __name__ == "__main__":
    main()
