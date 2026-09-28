#!/usr/bin/env python3
"""finish_campaign.py -- run the screening round I (new solvers), then one final Thompson-allocated round J over
the full 27-solver portfolio sized to the remaining time, publishing every 10 minutes; ends at the deadline."""
import json, os, subprocess, sys, time, shutil
HERE=os.path.dirname(os.path.abspath(__file__)); PY=sys.executable; LOG=os.path.join(HERE,"campaign.log")
T_END_RECORDED = 1790597340   # 2026-09-28 12:09:00 UTC (campaign start 12:09 UTC 09-27 + 24 h) -- the value this driver ran with
T_END = float(os.environ.get("CP_T_END", T_END_RECORDED))   # overridable: env CP_T_END or --t-end (epoch seconds)
CENSUS = os.environ.get("CP_CENSUS", "/tmp/si_tools/cp_census")
def say(m):
    line="%s %s"%(time.strftime("%m-%d %H:%M:%S"),m); print(line,flush=True); open(LOG,"a").write(line+"\n")
def publish(note=None):
    r=subprocess.run([PY,os.path.join(HERE,"publish.py")]+(["--note",note] if note else []),capture_output=True,text=True,timeout=3000)
    say(("PUBLISH "+(r.stdout.strip().splitlines() or [""])[-1]) if r.returncode==0 else "PUBLISH FAILED %s"%(r.stdout+r.stderr)[-300:].replace("\n"," | "))
class _Attached:
    """poll()-compatible handle on an already-running launcher pid"""
    def __init__(self, pid): self.pid=pid
    def poll(self):
        try: os.kill(self.pid,0); return None
        except OSError: return 0
    def terminate(self):
        try: os.kill(self.pid,15)
        except OSError: pass
def run_round(name, sched, warm, jobs=62):
    out=os.path.join(HERE,name); os.makedirs(out,exist_ok=True)
    pidf=os.path.join(out,"launcher.pid")
    if os.path.isfile(pidf) and not os.path.isfile(os.path.join(out,"DONE")):
        try:
            pid=int(open(pidf).read().split()[0]); os.kill(pid,0); say("%s: attaching to running launcher pid %d"%(name,pid)); p=_Attached(pid)
        except (OSError, ValueError): p=None
    else: p=None
    if p is None: p=subprocess.Popen([PY,os.path.join(HERE,"sweep_multi.py"),"--schedule",sched,"--out-dir",out,"--jobs",str(jobs),"--warm",warm,
                        "--records-file",os.path.join(CENSUS,"records_live_inflated.json")],stdout=open(os.path.join(out,"stdout.log"),"a"),stderr=subprocess.STDOUT)
    if not isinstance(p,_Attached): open(os.path.join(out,"launcher.pid"),"w").write(str(p.pid))
    last=time.time()
    while p.poll() is None:
        time.sleep(30)
        if time.time()-last>=600:
            last=time.time(); prog=subprocess.run(["bash","-c","command grep 'done ' %s/launcher.log | tail -1"%out],capture_output=True,text=True).stdout.strip()
            say("%s progress: %s"%(name,prog[9:] if prog else "starting")); publish()
        if time.time()>T_END+600:
            say("%s: deadline passed, stopping"%name); p.terminate(); time.sleep(5)
            subprocess.run(["pkill","-f","sweep_one[.]py --solver"]); subprocess.run(["pkill","-f","sweep_one[.]py --_child"]); break
    say("%s finished (rc=%s)"%(name,p.poll()))
def main():
    global T_END
    import argparse
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--t-end",type=float,default=T_END,help="campaign deadline, epoch seconds (default: env CP_T_END or the recorded value)")
    T_END=ap.parse_args().t_end
    say("FINISH DRIVER: round I (12 new solvers, N 51-100, nbr+self, seed 0) then round J (all 27, Thompson) until %s UTC"%time.strftime("%H:%M",time.gmtime(T_END)))
    run_round("roundI", os.path.join(HERE,"roundI.jsonl"), os.path.join(CENSUS,"warm_roundI"))
    publish()
    remaining=T_END-time.time()
    if remaining>1500:
        base,slope=300.0,14.0
        for _ in range(8):
            r=subprocess.run([PY,os.path.join(HERE,"escalate.py"),"--agg",os.path.join(HERE,"agg"),"--out",os.path.join(HERE,"roundJ.jsonl"),"--top","2","--seeds","51-54","--seeds-below","51-58","--base",str(base),"--slope",str(slope),"--modes","both"],capture_output=True,text=True)
            rows=[json.loads(l) for l in open(os.path.join(HERE,"roundJ.jsonl")) if l.strip()]; cpu=sum(x["cpu"] for x in rows); est=cpu*1.10/62
            budget=(T_END-time.time())*0.92
            if est<=budget or base<=30: break
            f=max(0.3,budget/est); base,slope=max(30.0,base*f),max(0.5,slope*f)
        warm=os.path.join(CENSUS,"warm_roundJ"); shutil.rmtree(warm,ignore_errors=True); shutil.copytree(os.path.join(HERE,"agg","warm_next"),warm)
        say("roundJ LAUNCH: %d jobs, %.1f CPU-h, est %.2f h; cpu=%.0f+%.1fn; warm=%s"%(len(rows),cpu/3600,est/3600,base,slope,warm))
        run_round("roundJ", os.path.join(HERE,"roundJ.jsonl"), warm)
    publish(note="Sweep campaign finished at %s UTC."%time.strftime("%Y-%m-%d %H:%M"))
    say("CAMPAIGN DONE"); open(os.path.join(HERE,"CAMPAIGN_DONE"),"w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
if __name__ == "__main__":
    main()
