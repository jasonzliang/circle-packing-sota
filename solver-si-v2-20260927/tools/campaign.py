#!/usr/bin/env python3
"""campaign.py -- run a 24 h series of successive-halving sweep rounds, publishing the best packing per n
into the circle-packing-sota repo every 10 minutes (publish.py: aggregate -> export -> README -> commit -> push).

  nohup python campaign.py --hours 24 --wait-for roundA > campaign.stdout 2>&1 &
Status lines go to campaign.log (one per publish, i.e. every 10 min) -- tail it for monitoring.
"""
import sys, argparse, json, os, shutil, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
CP_CENSUS = os.environ.get("CP_CENSUS", os.path.join(HERE, "census"))   # warm dirs + records_live_inflated.json
LOG = os.path.join(HERE, "campaign.log")

def say(msg):
    line = "%s %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as fh: fh.write(line + "\n")

def publish(note=None):
    cmd = [PY, os.path.join(HERE, "publish.py")] + (["--note", note] if note else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
    out = (r.stdout.strip().splitlines() or [""])[-1]
    say(("PUBLISH " + out) if r.returncode == 0 else ("PUBLISH FAILED rc=%d %s" % (r.returncode, (r.stdout + r.stderr)[-300:].replace("\n", " | "))))

def launcher_alive(pid_file):
    try: pid = int(open(pid_file).read().split()[0])
    except Exception: return False
    try: os.kill(pid, 0); return True
    except OSError: return False

def wait_round(name, pid_file, poll=600):
    d = os.path.join(HERE, name); last = 0
    while not os.path.isfile(os.path.join(d, "DONE")):
        if not launcher_alive(pid_file):
            say("%s: launcher gone without DONE -- treating as finished" % name); break
        time.sleep(30)
        if time.time() - last >= poll:
            last = time.time()
            prog = subprocess.run(["bash", "-c", "command grep 'done ' %s/launcher.log | tail -1" % d], capture_output=True, text=True).stdout.strip()
            say("%s progress: %s" % (name, prog[9:] if prog else "starting"))
            publish()
    say("%s finished" % name)

def sched_cost(path):
    tot = 0.0; n = 0
    for ln in open(path):
        if ln.strip(): tot += json.loads(ln)["cpu"]; n += 1
    return n, tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--wait-for", default="roundA")
    ap.add_argument("--jobs", type=int, default=62)
    a = ap.parse_args()
    t_end = time.time() + a.hours * 3600
    say("CAMPAIGN START: %.1f h budget, deadline %s UTC, %d workers" % (a.hours, time.strftime("%m-%d %H:%M", time.gmtime(t_end)), a.jobs))
    if a.wait_for:
        wait_round(a.wait_for, os.path.join(HERE, a.wait_for, "launcher.pid"))
    publish()
    # round specs: successive halving with growing budgets and fresh seeds
    specs = []
    seed_lo = 1
    plan = [  # (top, n_seeds, n_seeds_open, base, slope, modes)
        (4, 2, 4, 60, 3, "best"), (3, 3, 6, 120, 6, "best"), (2, 4, 8, 200, 10, "both"),
        (2, 4, 8, 240, 12, "best"), (2, 4, 8, 300, 14, "both"), (2, 4, 8, 360, 16, "best"),
        (2, 4, 8, 400, 18, "both"), (2, 4, 8, 450, 20, "best"), (2, 4, 8, 500, 22, "both"), (2, 4, 8, 550, 24, "best")]
    for i, (top, ns, nso, base, slope, modes) in enumerate(plan):
        specs.append(dict(name="round%s" % chr(ord("B") + i), top=top, seeds="%d-%d" % (seed_lo, seed_lo + ns - 1),
                          seeds_below="%d-%d" % (seed_lo, seed_lo + nso - 1), base=base, slope=slope, modes=modes))
        seed_lo += nso
    for spec in specs:
        remaining = t_end - time.time()
        if remaining < 1200:
            say("less than 20 min left -- no further rounds"); break
        name = spec["name"]; sched = os.path.join(HERE, name + ".jsonl")
        base, slope = spec["base"], spec["slope"]
        for attempt in range(6):   # fit the round into the remaining time (cap one round at 5 h)
            cmd = [PY, os.path.join(HERE, "escalate.py"), "--agg", os.path.join(HERE, "agg"), "--out", sched, "--top", str(spec["top"]),
                   "--seeds", spec["seeds"], "--seeds-below", spec["seeds_below"], "--base", str(base), "--slope", str(slope), "--modes", spec["modes"]]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                say("%s: escalate failed: %s" % (name, (r.stdout + r.stderr)[-300:])); break
            njobs, cpu_tot = sched_cost(sched)
            est = cpu_tot * 1.10 / a.jobs
            budget = min(remaining * 0.92, 5 * 3600)
            if est <= budget or base <= 30: break
            f = max(0.3, budget / est); base, slope = max(30.0, base * f), max(0.5, slope * f)
            say("%s: est %.1f h > %.1f h -> scaling budget by %.2f (base %.0f slope %.1f)" % (name, est / 3600, budget / 3600, f, base, slope))
        else:
            njobs, cpu_tot = sched_cost(sched)
        if r.returncode != 0: continue
        warm = os.path.join(CP_CENSUS, "warm_%s" % name)
        shutil.rmtree(warm, ignore_errors=True); shutil.copytree(os.path.join(HERE, "agg", "warm_next"), warm)
        out_dir = os.path.join(HERE, name); os.makedirs(out_dir, exist_ok=True)
        say("%s LAUNCH: %d jobs, %.1f CPU-h, est %.1f h wall; top=%d seeds=%s/%s cpu=%.0f+%.1fn modes=%s; warm=%s (%d packs)" % (
            name, njobs, cpu_tot / 3600, cpu_tot * 1.10 / a.jobs / 3600, spec["top"], spec["seeds"], spec["seeds_below"], base, slope, spec["modes"], warm, len(os.listdir(warm))))
        p = subprocess.Popen([PY, os.path.join(HERE, "sweep_multi.py"), "--schedule", sched, "--out-dir", out_dir, "--jobs", str(a.jobs),
                              "--warm", warm, "--records-file", os.path.join(CP_CENSUS, "records_live_inflated.json")],
                             stdout=open(os.path.join(out_dir, "stdout.log"), "a"), stderr=subprocess.STDOUT)
        open(os.path.join(out_dir, "launcher.pid"), "w").write(str(p.pid))
        last = time.time()
        while p.poll() is None:
            time.sleep(30)
            if time.time() - last >= 600:
                last = time.time()
                prog = subprocess.run(["bash", "-c", "command grep 'done ' %s/launcher.log | tail -1" % out_dir], capture_output=True, text=True).stdout.strip()
                say("%s progress: %s" % (name, prog[9:] if prog else "starting"))
                publish()
            if time.time() > t_end + 1800:     # hard stop 30 min past the deadline
                say("%s: deadline passed, stopping launcher" % name); p.terminate(); time.sleep(5)
                subprocess.run(["pkill", "-f", "sweep_one[.]py --solver"]); subprocess.run(["pkill", "-f", "sweep_one[.]py --_child"]); break
        say("%s finished (rc=%s)" % (name, p.poll()))
        publish()
    publish(note="Sweep campaign finished at %s UTC." % time.strftime("%Y-%m-%d %H:%M"))
    say("CAMPAIGN DONE")
    open(os.path.join(HERE, "CAMPAIGN_DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
main()
