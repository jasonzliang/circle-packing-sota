#!/usr/bin/env python3
"""sweep_one.py -- run ONE (n, seed) of a circle-packing tools/solver.py under the SI-v2 mission
contract with a generous budget, then INDEPENDENTLY re-validate the best packing it produced.

    python sweep_one.py --solver <path/to/tools/solver.py> --n 27 --seed 0 --wall-seconds 300 \
        [--cpu-seconds 280] [--warm <csqvN.pck | dir of csqv*.pck>] --out result.json [--pck best.pck]

Contract this mirrors (missions/circle-packing/scorer):
  * solve_driver.py:87-110 _load_solver  -> importlib spec_from_file_location("ws_cp_solver", path);
    workspace NOT on sys.path; AST-rejected via harness.solver_forbidden_construct (harness.py:252).
  * solve_driver.py:244-248, 277        -> Meter(budget); Evaluator(records, meter); the solver gets
    meter.read_only(); rng = np.random.RandomState(int(seed)); solve(ev.evaluate, ro_meter, rng, [n]).
  * harness.py:341-369 Evaluator.evaluate(n, packing) -> (feasible, sum_r) for (n,3) or (B,n,3), ticks
    the meter by B, tracks best strictly-feasible per n in ev.best[n] = (sum_r, [(x,y,r)...], nfev).
  * harness.py:49-76 validate(): exactly n circles, all finite, min r > 0, wall slack >= -1e-9,
    pairwise slack >= -1e-9, centered square [-0.5, 0.5]^2 (harness.py:35,40).
  * solve_driver.py:271-272 CPU backstop = SIGPROF/ITIMER_PROF raising into solve(); the driver then
    harvests ev.best regardless (solve_driver.py:274-300). Reproduced here, plus a wall-clock alarm.
  * MISSION.md: cwd is the workspace root; solvers read bench/packs/csqv<n>.pck and bench/records.json.
    A throwaway workspace with exactly that layout is built per call.

The child process (this same file, --_child) does the metered run and writes a raw result; the parent
enforces the wall limit (SIGKILL at wall + grace), salvages the checkpoint if the child was killed, and
re-validates the packing with its OWN pure-Python implementation of the rule AND (when importable) the
scorer's harness.validate. Nothing the solver or the child says about feasibility is trusted.

Exit codes: 0 feasible; 2 infeasible; 3 no packing produced; 4 solver load/AST error; 5 child killed
with nothing to salvage; 6 validator disagreement (treated as infeasible).
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _v in THREAD_ENV:                      # BEFORE any numpy import (solve_driver.py:68-75)
    os.environ[_v] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MISSION = os.path.join(HERE, "..")   # <solver dir>/scorer/{harness,verify}.py
FEAS_TOL = 1e-9          # harness.FEAS_TOL (harness.py:35)
LO, HI = -0.5, 0.5       # harness.LO/HI (harness.py:40)
DIGITS_CAP = 7.0         # harness.DIGITS_CAP (harness.py:86)
DEFAULT_EVALS = 10 ** 9  # "effectively unlimited" evaluation budget (mission: 500_000)
MODULE_NAME = "ws_cp_solver"   # solve_driver.py:102


# ------------------------------------------------------------------ independent geometry (pure python)
def validate_local(circles, n, tol=FEAS_TOL):
    """Exact re-implementation of harness.validate (harness.py:49-76) plus max_overlap/max_outside.
    circles: iterable of (x, y, r). Never trusts anything but the coordinates."""
    try:
        circ = [tuple(float(v) for v in c) for c in circles]
    except (TypeError, ValueError):
        return {"feasible": False, "n_found": -1, "count_ok": False, "sum_r": None, "min_r": None,
                "wall_slack": None, "pair_slack": None, "max_overlap": None, "max_outside": None,
                "why": "unparseable"}
    k = len(circ)
    if any(len(c) != 3 for c in circ):
        return {"feasible": False, "n_found": k, "count_ok": k == n, "sum_r": None, "min_r": None,
                "wall_slack": None, "pair_slack": None, "max_overlap": None, "max_outside": None,
                "why": "row arity != 3"}
    if not all(math.isfinite(v) for c in circ for v in c):
        return {"feasible": False, "n_found": k, "count_ok": k == n, "sum_r": 0.0, "min_r": 0.0,
                "wall_slack": None, "pair_slack": None, "max_overlap": None, "max_outside": None,
                "why": "non-finite"}
    if k == 0:
        return {"feasible": False, "n_found": 0, "count_ok": n == 0, "sum_r": 0.0, "min_r": 0.0,
                "wall_slack": None, "pair_slack": None, "max_overlap": None, "max_outside": None,
                "why": "empty"}
    sum_r = math.fsum(r for _, _, r in circ)
    min_r = min(r for _, _, r in circ)
    wall = min(min(x - r - LO, HI - x - r, y - r - LO, HI - y - r) for x, y, r in circ)
    pair = math.inf
    for i in range(k):
        xi, yi, ri = circ[i]
        for j in range(i + 1, k):
            xj, yj, rj = circ[j]
            d = math.hypot(xi - xj, yi - yj) - (ri + rj)
            if d < pair:
                pair = d
    count_ok = (k == n)
    feasible = count_ok and (min_r > 0.0) and (wall >= -tol) and (k < 2 or pair >= -tol)
    why = ""
    if not count_ok:
        why = "count %d != %d" % (k, n)
    elif min_r <= 0.0:
        why = "min_r=%.3e <= 0" % min_r
    elif wall < -tol:
        why = "wall slack %.3e < -tol" % wall
    elif k >= 2 and pair < -tol:
        why = "pair slack %.3e < -tol" % pair
    return {"feasible": bool(feasible), "n_found": k, "count_ok": count_ok, "sum_r": sum_r,
            "min_r": min_r, "wall_slack": wall, "pair_slack": (None if k < 2 else pair),
            "max_overlap": (0.0 if k < 2 else max(0.0, -pair)), "max_outside": max(0.0, -wall),
            "why": why}


def digits_closed(sum_r, record, cap=DIGITS_CAP):
    """harness.digits_closed (harness.py:93-127)."""
    if record is None or record <= 0 or sum_r is None:
        return None
    relgap = max(0.0, (record - float(sum_r)) / record)
    return max(0.0, min(cap, -math.log10(max(relgap, 10.0 ** -cap))))


def load_records(mission_dir, include_heldout=True):
    """{n: (record, source)} from scorer/records.json (visible odd n) plus, optionally, the even-n
    held-out table heldout/records_20260803.json['splits']['even_interpolation']."""
    out = {}
    p = os.path.join(mission_dir, "scorer", "records.json")
    if os.path.isfile(p):
        for k, v in json.load(open(p))["records"].items():
            out[int(k)] = (float(v), "scorer/records.json")
    if include_heldout:
        q = os.path.join(mission_dir, "heldout", "records_20260803.json")
        if os.path.isfile(q):
            doc = json.load(open(q)).get("splits", {})
            for split in ("even_interpolation", "gt100_extrapolation"):
                for k, v in doc.get(split, {}).get("records", {}).items():
                    out.setdefault(int(k), (float(v), "heldout/records_20260803.json:" + split))
    return out


def write_pck(path, circles, label):
    """Packomania .pck exactly as solve_driver._write_pck (solve_driver.py:210-219): line 1 = max r,
    line 2 = label, then n lines 'x y r' at %.17g, radii ascending. Returns re-parsed circles."""
    circ = sorted((tuple(float(v) for v in c) for c in circles), key=lambda t: t[2])
    with open(path, "w") as f:
        f.write("%.17g\n%s\n" % (max(r for *_, r in circ), label))
        for x, y, r in circ:
            f.write("%.17g %.17g %.17g\n" % (x, y, r))
    raw = [ln for ln in open(path).read().splitlines() if ln.strip()]
    return [tuple(float(p) for p in ln.split()) for ln in raw[2:]]


def parse_pck(path):
    """verify.parse_pck (verify.py:82-94)."""
    raw = [ln for ln in open(path).read().splitlines() if ln.strip()]
    if len(raw) < 3:
        raise ValueError("fewer than 3 non-empty lines")
    circ = []
    for ln in raw[2:]:
        parts = ln.split()
        if len(parts) != 3:
            raise ValueError("expected 'x y r', got %r" % ln)
        circ.append(tuple(float(p) for p in parts))
    return circ


# ------------------------------------------------------------------ throwaway workspace
def build_workspace(root, n, warm, mission_dir, hide_records, records_file=None, warm_exclude_self=False):
    """<root>/bench/packs/ (+ warm packs) and <root>/bench/records.json (the mission's visible table,
    unless hide_records). Mirrors what seed_bench.py gives an agent (records.json + packs/)."""
    packs = os.path.join(root, "bench", "packs")
    os.makedirs(packs, exist_ok=True)
    os.makedirs(os.path.join(root, "tools"), exist_ok=True)
    copied = []
    if warm:
        if os.path.isdir(warm):
            for fn in sorted(os.listdir(warm)):
                if warm_exclude_self and fn == "csqv%d.pck" % n:
                    continue
                if fn.startswith("csqv") and fn.endswith(".pck"):
                    shutil.copy2(os.path.join(warm, fn), os.path.join(packs, fn))
                    copied.append(fn)
        elif os.path.isfile(warm):
            dst = "csqv%d.pck" % n
            shutil.copy2(warm, os.path.join(packs, dst))
            copied.append(dst)
        else:
            raise FileNotFoundError("--warm %r is neither a file nor a directory" % warm)
    if not hide_records:
        src = records_file or os.path.join(mission_dir, "scorer", "records.json")
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(root, "bench", "records.json"))
        elif records_file:
            raise FileNotFoundError("--records-file %r missing" % records_file)
    return copied


# ------------------------------------------------------------------ CHILD: the metered run
def _child_main(req_path):
    t_child0 = time.time()
    with open(req_path) as fh:
        req = json.load(fh)
    n, seed = int(req["n"]), int(req["seed"])
    result = {"status": "init", "n": n, "seed": seed, "solver_error": None, "import_wall_s": None,
              "solve_wall_s": None, "solve_cpu_s": None, "child_cpu_s": None, "nfev": 0, "calls": 0,
              "wall_alarm_fired": False, "cpu_backstop_fired": False, "packing": None,
              "sum_r_claimed": None, "nfev_at_find": None, "solver_kwargs": {}, "forbidden": ""}

    def _emit(code):
        result["child_cpu_s"] = time.process_time()
        tmp = req["result_out"] + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(result, fh)
        os.replace(tmp, req["result_out"])
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(code)

    sys.path.insert(0, req["scorer_dir"])
    try:
        import harness                       # noqa: F401  (scorer's Meter/Evaluator/AST scan)
        import numpy as np
    except Exception as e:                   # noqa: BLE001
        result.update(status="load_error", solver_error="import harness/numpy: %r" % (e,))
        _emit(4)

    # --- load the solver exactly as solve_driver._load_solver does (solve_driver.py:87-110)
    import importlib.util
    import inspect
    import random
    import signal
    import traceback
    path = req["solver"]
    bad = harness.solver_forbidden_construct(path)
    result["forbidden"] = bad
    if bad and not req.get("allow_forbidden"):
        result.update(status="load_error", solver_error="AST-rejected: " + bad)
        _emit(4)
    t_imp = time.time()
    try:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        solve_fn = mod.solve
        applied, missing = {}, []
        for k, v in (req.get("module_attrs") or {}).items():
            if hasattr(mod, k):
                setattr(mod, k, v)
                applied[k] = v
            else:
                missing.append(k)
        result["module_attrs_applied"] = applied
        result["module_attrs_missing"] = missing
    except BaseException as e:               # noqa: BLE001
        result.update(status="load_error",
                      solver_error="import solver: %s: %s\n%s" % (type(e).__name__, e,
                                                                 traceback.format_exc()[-2000:]))
        _emit(4)
    result["import_wall_s"] = round(time.time() - t_imp, 3)

    # --- metered oracle (solve_driver.py:244-248)
    budget = int(req["evals"])
    meter = harness.Meter(budget)
    ev = harness.Evaluator({}, meter)
    ro_meter = meter.read_only()
    rng = np.random.RandomState(seed)
    random.seed(seed)                        # belt and braces for solvers that touch global RNGs
    np.random.seed(seed)

    ckpt = {"last_sum": None, "last_t": 0.0}
    trace = []                                   # anytime curve: [cpu_s_since_entry, sum_r] at every new best
    t_entry = {"cpu": None}

    def _dump_ckpt(force=False):
        cur = ev.best.get(n)
        if cur is None:
            return
        if not trace or cur[0] > trace[-1][1]:
            trace.append([round(time.process_time() - (t_entry["cpu"] or 0.0), 3), cur[0]])
        now = time.time()
        if not force and (cur[0] == ckpt["last_sum"] or now - ckpt["last_t"] < 1.0):
            return
        ckpt["last_sum"], ckpt["last_t"] = cur[0], now
        try:
            tmp = req["ckpt_out"] + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"n": n, "seed": seed, "sum_r_claimed": cur[0], "packing": cur[1],
                           "nfev_at_find": cur[2], "nfev": meter.used, "calls": ev.calls,
                           "t": now}, fh)
            os.replace(tmp, req["ckpt_out"])
        except Exception:                    # noqa: BLE001 -- checkpointing must never break the run
            pass

    def evaluate(nn, packing):
        out = ev.evaluate(nn, packing)
        _dump_ckpt()
        return out

    # --- extra kwargs: hand the CPU allowance to any `*cpu*` parameter the solver declares
    kwargs = {}
    try:
        sig = inspect.signature(solve_fn)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        cpu_params = [nm for nm in params if "cpu" in nm.lower() and nm not in
                      ("evaluate", "meter", "rng", "targets")]
        if cpu_params and req.get("solver_cpu") is not None:
            kwargs[cpu_params[0]] = float(req["solver_cpu"])
        for k, v in (req.get("solver_kwargs") or {}).items():
            if k in params or has_varkw:
                kwargs[k] = v
    except (TypeError, ValueError):
        pass
    result["solver_kwargs"] = kwargs

    class _WallStop(BaseException):
        pass

    class _CpuStop(BaseException):
        pass

    def _on_alrm(sig, frm):
        raise _WallStop()

    def _on_prof(sig, frm):
        raise _CpuStop()

    remaining = float(req["wall_seconds"]) - (time.time() - t_child0)
    signal.signal(signal.SIGALRM, _on_alrm)
    signal.setitimer(signal.ITIMER_REAL, max(0.5, remaining))
    if req.get("cpu_seconds"):
        signal.signal(signal.SIGPROF, _on_prof)
        signal.setitimer(signal.ITIMER_PROF, float(req["cpu_seconds"]))
    t0, c0 = time.time(), time.process_time()
    t_entry["cpu"] = c0
    status = "ok"
    try:
        solve_fn(evaluate, ro_meter, rng, [n], **kwargs)
    except _WallStop:
        result["wall_alarm_fired"] = True
    except _CpuStop:
        result["cpu_backstop_fired"] = True
    except BaseException as e:               # noqa: BLE001 -- harvest whatever was tracked (driver does too)
        status = "solver_error"
        result["solver_error"] = "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-3000:])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)
    result["solve_wall_s"] = round(time.time() - t0, 3)
    result["solve_cpu_s"] = round(time.process_time() - c0, 3)
    result["trace"] = trace[-400:]
    result["nfev"], result["calls"] = meter.used, ev.calls
    cur = ev.best.get(n)
    if cur is None:
        result["status"] = "no_packing" if status == "ok" else status
        _emit(3)
    result["sum_r_claimed"], result["packing"], result["nfev_at_find"] = cur[0], cur[1], cur[2]
    result["status"] = status
    _emit(0)


# ------------------------------------------------------------------ PARENT
def run_one(a):
    mission = os.path.abspath(a.mission_dir)
    scorer_dir = os.path.join(mission, "scorer")
    solver = os.path.abspath(a.solver)
    if not os.path.isfile(solver):
        sys.exit("no such solver file: %s" % solver)
    records = load_records(mission, include_heldout=not a.records_only_visible)
    rec, rec_src = records.get(a.n, (None, None))
    tmp_root = a.tmp_root or os.path.join(HERE, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    ws = tempfile.mkdtemp(prefix="ws_n%d_s%d_" % (a.n, a.seed), dir=tmp_root)
    solver_cpu = a.solver_cpu
    if solver_cpu is None:
        solver_cpu = a.cpu_seconds if a.cpu_seconds else 0.9 * a.wall_seconds
    grace = a.grace if a.grace is not None else max(15.0, 0.25 * a.wall_seconds)
    out = {"n": a.n, "seed": a.seed, "wall_s": None, "cpu_s": None, "sum_radii": None,
           "feasible": False, "max_overlap": None, "max_outside": None, "record": rec,
           "record_source": rec_src, "gap": None, "relgap": None, "digits": None,
           "centers": None, "radii": None,
           "status": None, "solver": solver, "solver_sha": None, "wall_limit_s": a.wall_seconds,
           "cpu_limit_s": a.cpu_seconds, "solver_cpu_allowance": solver_cpu, "evals_budget": a.evals,
           "warm": (os.path.abspath(a.warm) if a.warm else None), "warm_files": None,
           "hide_records": bool(a.hide_records), "tol": FEAS_TOL, "killed": False,
           "checkpoint_used": False, "validators": {}, "python": sys.executable}
    try:
        import hashlib
        out["solver_sha"] = hashlib.sha256(open(solver, "rb").read()).hexdigest()[:12]
        out["warm_files"] = build_workspace(ws, a.n, a.warm, mission, a.hide_records, a.records_file,
                                            a.warm_exclude_self)
        out["warm_exclude_self"] = bool(a.warm_exclude_self)
        out["records_file"] = (os.path.abspath(a.records_file) if a.records_file else None)
        out["module_attrs"] = (json.loads(a.module_attrs) if a.module_attrs else {})
        req = {"n": a.n, "seed": a.seed, "solver": solver, "scorer_dir": scorer_dir,
               "wall_seconds": float(a.wall_seconds), "cpu_seconds": a.cpu_seconds,
               "solver_cpu": solver_cpu, "evals": int(a.evals),
               "solver_kwargs": (json.loads(a.solver_kwargs) if a.solver_kwargs else {}),
               "module_attrs": (json.loads(a.module_attrs) if a.module_attrs else {}),
               "allow_forbidden": bool(a.allow_forbidden),
               "result_out": os.path.join(ws, "child_result.json"),
               "ckpt_out": os.path.join(ws, "child_ckpt.json")}
        req_path = os.path.join(ws, "request.json")
        with open(req_path, "w") as fh:
            json.dump(req, fh)
        env = dict(os.environ)
        for v in THREAD_ENV:
            env[v] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        log = open(os.path.join(ws, "child.log"), "wb")
        t0 = time.time()
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--_child", req_path],
                                cwd=ws, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
        try:
            rc = proc.wait(timeout=a.wall_seconds + grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = -9
            out["killed"] = True
        out["wall_s"] = round(time.time() - t0, 3)
        log.close()
        out["child_rc"] = rc
        child = None
        if os.path.isfile(req["result_out"]):
            try:
                child = json.load(open(req["result_out"]))
            except ValueError:
                child = None
        packing = None
        if child is not None:
            out.update({k: child.get(k) for k in ("nfev", "calls", "nfev_at_find", "import_wall_s",
                                                   "solve_wall_s", "solve_cpu_s", "solver_kwargs",
                                                   "wall_alarm_fired", "cpu_backstop_fired",
                                                   "solver_error", "forbidden",
                                                   "module_attrs_applied", "module_attrs_missing", "trace")})
            out["cpu_s"] = child.get("child_cpu_s")
            out["status"] = child.get("status")
            out["sum_radii_claimed"] = child.get("sum_r_claimed")
            packing = child.get("packing")
        elif os.path.isfile(req["ckpt_out"]):
            try:
                ck = json.load(open(req["ckpt_out"]))
                packing = ck.get("packing")
                out["checkpoint_used"] = True
                out["sum_radii_claimed"] = ck.get("sum_r_claimed")
                out["nfev"], out["calls"], out["nfev_at_find"] = ck.get("nfev"), ck.get("calls"), ck.get("nfev_at_find")
                out["status"] = "killed_salvaged"
            except ValueError:
                pass
        if packing is None and out["status"] is None:
            out["status"] = "killed_no_result" if out["killed"] else "child_failed_rc%d" % rc
        try:
            tail = open(os.path.join(ws, "child.log"), "rb").read()[-4000:].decode("utf-8", "replace")
        except OSError:
            tail = ""
        out["child_log_tail"] = tail

        # --- INDEPENDENT re-validation: own implementation + the scorer's harness.validate
        if packing is not None:
            v = validate_local(packing, a.n)
            out["validators"]["local"] = {k: v[k] for k in ("feasible", "why", "n_found")}
            agree = True
            try:
                sys.path.insert(0, scorer_dir)
                import harness
                hv = harness.validate(packing, a.n)
                out["validators"]["harness"] = {"feasible": hv["feasible"], "sum_r": hv["sum_r"],
                                                "wall_slack": hv["wall_slack"], "pair_slack": hv["pair_slack"]}
                agree = (bool(hv["feasible"]) == bool(v["feasible"]))
            except Exception as e:           # noqa: BLE001
                out["validators"]["harness"] = {"error": repr(e)}
            out["validators"]["agree"] = agree
            out["feasible"] = bool(v["feasible"]) and agree
            out["sum_radii"] = v["sum_r"]
            out["min_r"], out["wall_slack"], out["pair_slack"] = v["min_r"], v["wall_slack"], v["pair_slack"]
            out["max_overlap"], out["max_outside"] = v["max_overlap"], v["max_outside"]
            out["centers"] = [[float(c[0]), float(c[1])] for c in packing]
            out["radii"] = [float(c[2]) for c in packing]
            if rec is not None and v["sum_r"] is not None:
                out["gap"] = v["sum_r"] - rec
                out["relgap"] = max(0.0, (rec - v["sum_r"]) / rec)
                out["digits"] = digits_closed(v["sum_r"], rec)
            if not agree:
                out["status"] = "validator_disagreement"
            elif not v["feasible"]:
                out["status"] = (out["status"] or "") + "+infeasible" if out["status"] not in (None, "ok") else "infeasible"
            if out["feasible"] and a.pck:
                landed = write_pck(a.pck, packing, "sweep n=%d seed=%d %s" % (a.n, a.seed, out["solver_sha"]))
                lv = validate_local(landed, a.n)
                out["pck"] = a.pck
                out["pck_feasible_on_disk"] = lv["feasible"]
                out["pck_sum_r_on_disk"] = lv["sum_r"]
    finally:
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
            tmp = a.out + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(out, fh)
            os.replace(tmp, a.out)
        if a.keep_tmp:
            out["workspace"] = ws
        else:
            shutil.rmtree(ws, ignore_errors=True)

    rec_s = "%.12f" % rec if rec is not None else "-"
    sr_s = "%.12f" % out["sum_radii"] if out["sum_radii"] is not None else "-"
    gap_s = "%+.3e" % out["gap"] if out["gap"] is not None else "-"
    dg_s = "%.3f" % out["digits"] if out["digits"] is not None else "-"
    print("n=%d seed=%d status=%s feasible=%s sum_r=%s record=%s gap=%s digits=%s wall=%.1fs cpu=%s "
          "nfev=%s overlap=%s outside=%s"
          % (a.n, a.seed, out["status"], out["feasible"], sr_s, rec_s, gap_s, dg_s, out["wall_s"] or 0.0,
             out["cpu_s"], out.get("nfev"), out["max_overlap"], out["max_outside"]))
    if out.get("solver_error"):
        print("# solver_error: " + str(out["solver_error"]).strip().splitlines()[-1][:300])
    if out["status"] == "load_error":
        return 4
    if out["status"] == "validator_disagreement":
        return 6
    if out["feasible"]:
        return 0
    if out["centers"] is None:
        return 5 if out["killed"] else 3
    return 2


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 2 and argv[0] == "--_child":
        _child_main(argv[1])
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", required=True, help="path to a tools/solver.py")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--wall-seconds", type=float, required=True,
                    help="wall limit: the child's SIGALRM interrupts solve() here; SIGKILL at wall+grace")
    ap.add_argument("--cpu-seconds", type=float, default=None,
                    help="optional process-CPU cap (SIGPROF/ITIMER_PROF, as the driver does)")
    ap.add_argument("--solver-cpu", type=float, default=None,
                    help="CPU allowance handed to a solver that declares a *cpu* kwarg "
                         "(default: --cpu-seconds, else 0.9*wall)")
    ap.add_argument("--evals", type=int, default=DEFAULT_EVALS, help="evaluation budget (default 1e9)")
    ap.add_argument("--warm", default=None, help="csqv<N>.pck file, or a dir of csqv*.pck (neighbours too)")
    ap.add_argument("--hide-records", action="store_true",
                    help="do NOT give the solver bench/records.json (some solvers stop at the record)")
    ap.add_argument("--records-only-visible", action="store_true",
                    help="report only scorer/records.json records (no even-n heldout fallback)")
    ap.add_argument("--solver-kwargs", default=None, help="JSON dict of extra kwargs for solve()")
    ap.add_argument("--module-attrs", default=None,
                    help="JSON dict of module-level attributes to set on the solver after import "
                         "(e.g. CPU budget constants); names the module lacks are reported, not created")
    ap.add_argument("--warm-exclude-self", action="store_true",
                    help="with a --warm dir: copy every pack EXCEPT csqv<n>.pck (neighbour transfer only, no incumbent)")
    ap.add_argument("--records-file", default=None,
                    help="records.json to place at bench/records.json instead of the mission's frozen table")
    ap.add_argument("--allow-forbidden", action="store_true", help="skip the mission's AST rejection")
    ap.add_argument("--grace", type=float, default=None, help="kill margin after wall (default max(15, 0.25*wall))")
    ap.add_argument("--mission-dir", default=DEFAULT_MISSION)
    ap.add_argument("--tmp-root", default=None, help="where throwaway workspaces go (default <here>/tmp)")
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--out", required=True, help="result JSON path")
    ap.add_argument("--pck", default=None, help="also write the (feasible) best packing as .pck here")
    a = ap.parse_args(argv)
    return run_one(a)


if __name__ == "__main__":
    sys.exit(main())
