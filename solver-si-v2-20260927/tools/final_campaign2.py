#!/usr/bin/env python3
"""final_campaign2.py -- audited allocation (09-28 07:10 UTC): two rungs K then L of nbr-only runs on the 10 audited
solvers, sizes weighted by softness, one full-length run per seed; publish every 10 min; ends at the 24 h deadline."""
import json, os, subprocess, sys, time, shutil
HERE=os.path.dirname(os.path.abspath(__file__)); PY=sys.executable; LOG=os.path.join(HERE,"campaign.log")
T_END_RECORDED = 1790597340   # 2026-09-28 12:09:00 UTC -- the value this driver ran with
T_END = float(os.environ.get("CP_T_END", T_END_RECORDED))   # overridable: env CP_T_END or --t-end (epoch seconds)
CENSUS=os.environ.get("CP_CENSUS", "/tmp/si_tools/cp_census"); WORKERS=62; TAIL_H=0.17; RESERVE_H=0.20
def say(m):
    line="%s %s"%(time.strftime("%m-%d %H:%M:%S"),m); print(line,flush=True); open(LOG,"a").write(line+"\n")
def publish(note=None):
    r=subprocess.run([PY,os.path.join(HERE,"publish.py")]+(["--note",note] if note else []),capture_output=True,text=True,timeout=3000)
    say(("PUBLISH "+(r.stdout.strip().splitlines() or [""])[-1]) if r.returncode==0 else "PUBLISH FAILED %s"%(r.stdout+r.stderr)[-300:].replace("\n"," | "))
def run_round(name, sched, warm, jobs=WORKERS):
    out=os.path.join(HERE,name); os.makedirs(out,exist_ok=True)
    p=subprocess.Popen([PY,os.path.join(HERE,"sweep_multi.py"),"--schedule",sched,"--out-dir",out,"--jobs",str(jobs),"--warm",warm,
                        "--records-file",os.path.join(CENSUS,"records_live_inflated.json")],stdout=open(os.path.join(out,"stdout.log"),"a"),stderr=subprocess.STDOUT)
    open(os.path.join(out,"launcher.pid"),"w").write(str(p.pid))
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
def plan(name, prev_warm, cpu_hours, seed0):
    warm=os.path.join(CENSUS,"warm_%s"%name); shutil.rmtree(warm,ignore_errors=True); shutil.copytree(os.path.join(HERE,"agg","warm_next"),warm)
    sigf=os.path.join(HERE,"tmp","size_signals_%s.json"%name)
    subprocess.run([PY,os.path.join(HERE,"size_signals.py"),sigf,os.path.join(HERE,"agg")],capture_output=True,text=True,check=True)
    sched=os.path.join(HERE,"%s.jsonl"%name)
    r=subprocess.run([PY,os.path.join(HERE,"alloc_round.py"),"--signals",sigf,"--warm-prev",prev_warm,"--warm",warm,"--cpu-hours","%.1f"%cpu_hours,"--seed0",str(seed0),"--out",sched],capture_output=True,text=True,check=True)
    for l in r.stdout.strip().splitlines(): say("%s plan: %s"%(name,l.strip()))
    return sched, warm
def main():
    global T_END
    import argparse
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--t-end",type=float,default=T_END,help="campaign deadline, epoch seconds (default: env CP_T_END or the recorded value)")
    T_END=ap.parse_args().t_end
    say("FINAL DRIVER 2: audited allocation, rungs K then L (nbr only, 10 solvers, softness-weighted sizes), until %s UTC"%time.strftime("%H:%M",time.gmtime(T_END)))
    publish()
    remaining_h=(T_END-time.time())/3600-RESERVE_H
    wall_k=remaining_h/2; cpu_k=max(20.0,(wall_k-TAIL_H)*WORKERS*0.98)
    sched,warm=plan("roundK",os.path.join(CENSUS,"warm_roundJ"),cpu_k,61)
    say("roundK LAUNCH: %.1f CPU-h for %.2f h wall; warm=%s"%(cpu_k,wall_k,warm))
    run_round("roundK",sched,warm)
    publish()
    remaining_h=(T_END-time.time())/3600-RESERVE_H
    if remaining_h>0.4:
        cpu_l=max(20.0,(remaining_h-TAIL_H)*WORKERS*0.98)
        sched,warm=plan("roundL",os.path.join(CENSUS,"warm_roundK"),cpu_l,81)
        say("roundL LAUNCH: %.1f CPU-h for %.2f h wall; warm=%s"%(cpu_l,remaining_h,warm))
        run_round("roundL",sched,warm)
    publish(note="Sweep campaign finished at %s UTC."%time.strftime("%Y-%m-%d %H:%M"))
    say("CAMPAIGN DONE"); open(os.path.join(HERE,"CAMPAIGN_DONE"),"w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
if __name__ == "__main__":
    main()
