#!/usr/bin/env python3
"""final_campaign3.py -- deadline extended by the operator (09-28) to 2026-09-29 12:09 UTC. Attaches to the running round K,
then keeps planning ~3 h rounds (L, M, N, ...) from refreshed size signals and a fresh warm census, one full-length nbr run per
seed on the 10 audited solvers; budgets alternate x1.0 / x1.5 across rounds as a cutoff hedge (Luby: unknown run-time
distribution -> mix cutoffs). Publishes every 10 min; ends at the deadline."""
import json, os, subprocess, sys, time, shutil, string
HERE=os.path.dirname(os.path.abspath(__file__)); PY=sys.executable; LOG=os.path.join(HERE,"campaign.log")
T_END_RECORDED = 1790683740   # 2026-09-29 12:09:00 UTC (original 24 h deadline + 24 h, operator request 09-28 ~07:10 UTC) -- the value this driver ran with
T_END = float(os.environ.get("CP_T_END", T_END_RECORDED))   # overridable: env CP_T_END or --t-end (epoch seconds)
CENSUS=os.environ.get("CP_CENSUS", "/tmp/si_tools/cp_census"); WORKERS=62; TAIL_H=0.17; RESERVE_H=0.20; ROUND_WALL_H=3.0
def say(m):
    line="%s %s"%(time.strftime("%m-%d %H:%M:%S"),m); print(line,flush=True); open(LOG,"a").write(line+"\n")
def publish(note=None):
    r=subprocess.run([PY,os.path.join(HERE,"publish.py")]+(["--note",note] if note else []),capture_output=True,text=True,timeout=3000)
    say(("PUBLISH "+(r.stdout.strip().splitlines() or [""])[-1]) if r.returncode==0 else "PUBLISH FAILED %s"%(r.stdout+r.stderr)[-300:].replace("\n"," | "))
class _Attached:
    def __init__(self, pid): self.pid=pid
    def poll(self):
        try: os.kill(self.pid,0); return None
        except OSError: return 0
    def terminate(self):
        try: os.kill(self.pid,15)
        except OSError: pass
def run_round(name, sched, warm, jobs=WORKERS):
    out=os.path.join(HERE,name); os.makedirs(out,exist_ok=True)
    pidf=os.path.join(out,"launcher.pid"); p=None
    if os.path.isfile(pidf) and not os.path.isfile(os.path.join(out,"DONE")):
        try:
            pid=int(open(pidf).read().split()[0]); os.kill(pid,0); say("%s: attaching to running launcher pid %d"%(name,pid)); p=_Attached(pid)
        except (OSError, ValueError): p=None
    if p is None:
        p=subprocess.Popen([PY,os.path.join(HERE,"sweep_multi.py"),"--schedule",sched,"--out-dir",out,"--jobs",str(jobs),"--warm",warm,
                            "--records-file",os.path.join(CENSUS,"records_live_inflated.json")],stdout=open(os.path.join(out,"stdout.log"),"a"),stderr=subprocess.STDOUT)
        open(pidf,"w").write(str(p.pid))
    last=time.time()
    while p.poll() is None:
        time.sleep(30)
        if time.time()-last>=600:
            last=time.time(); prog=subprocess.run(["bash","-c","command grep 'done ' %s/launcher.log | tail -1"%out],capture_output=True,text=True).stdout.strip()
            say("%s progress: %s"%(name,prog[9:] if prog else "starting")); publish()
        if time.time()>T_END+300:
            say("%s: deadline passed, stopping"%name); p.terminate(); time.sleep(5)
            subprocess.run(["pkill","-f","sweep_one[.]py --solver"]); subprocess.run(["pkill","-f","sweep_one[.]py --_child"]); break
    say("%s finished (rc=%s)"%(name,p.poll()))
def plan(name, prev_warm, cpu_hours, seed0, cpu_scale):
    warm=os.path.join(CENSUS,"warm_%s"%name); shutil.rmtree(warm,ignore_errors=True); shutil.copytree(os.path.join(HERE,"agg","warm_next"),warm)
    sigf=os.path.join(HERE,"tmp","size_signals_%s.json"%name)
    subprocess.run([PY,os.path.join(HERE,"size_signals.py"),sigf,os.path.join(HERE,"agg")],capture_output=True,text=True,check=True)
    sched=os.path.join(HERE,"%s.jsonl"%name)
    r=subprocess.run([PY,os.path.join(HERE,"alloc_round.py"),"--signals",sigf,"--warm-prev",prev_warm,"--warm",warm,"--cpu-hours","%.1f"%cpu_hours,
                      "--seed0",str(seed0),"--cpu-scale",str(cpu_scale),"--out",sched],capture_output=True,text=True,check=True)
    for l in r.stdout.strip().splitlines(): say("%s plan: %s"%(name,l.strip()))
    return sched, warm
def main():
    global T_END
    import argparse
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--t-end",type=float,default=T_END,help="campaign deadline, epoch seconds (default: env CP_T_END or the recorded value)")
    T_END=ap.parse_args().t_end
    say("FINAL DRIVER 3: deadline extended to %s UTC; attach to round K, then ~%.0f h rounds L, M, N... (nbr only, 10 audited solvers, softness-weighted sizes, budgets x1.0/x1.5 alternating)"%(time.strftime("%Y-%m-%d %H:%M",time.gmtime(T_END)),ROUND_WALL_H))
    run_round("roundK",os.path.join(HERE,"roundK.jsonl"),os.path.join(CENSUS,"warm_roundK"))
    publish()
    prev="roundK"; idx=1
    while True:
        remaining_h=(T_END-time.time())/3600-RESERVE_H
        if remaining_h<0.4: break
        name="round"+string.ascii_uppercase[10+idx]          # L, M, N, ...
        wall=min(ROUND_WALL_H,remaining_h); cpu=max(20.0,(wall-TAIL_H)*WORKERS*0.98)
        scale=1.5 if idx%2==1 else 1.0
        sched,warm=plan(name,os.path.join(CENSUS,"warm_%s"%prev),cpu,61+100*idx,scale)
        say("%s LAUNCH: %.1f CPU-h for %.2f h wall; budgets x%.1f; seeds %d+; warm=%s"%(name,cpu,wall,scale,61+100*idx,warm))
        run_round(name,sched,warm)
        publish(); prev=name; idx+=1
    publish(note="Sweep campaign finished at %s UTC."%time.strftime("%Y-%m-%d %H:%M"))
    say("CAMPAIGN DONE"); open(os.path.join(HERE,"CAMPAIGN_DONE"),"w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
if __name__ == "__main__":
    main()
