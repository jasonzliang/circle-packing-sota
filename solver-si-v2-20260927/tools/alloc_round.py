#!/usr/bin/env python3
"""alloc_round.py -- allocation synthesised from the 09-28 audits (agents A/B/C, quantitative, literature):

  * nbr mode only (kick 0/1,318 beats -> dropped; self is a deterministic polish -> one short job only for sizes whose
    incumbent changed since the previous round);
  * one full-length run per seed, never split; nbr budget = clip(9.4n-154, 350, 800) s (tie-hazard t90 fit), except the
    fast-saturating SLP solvers sm7/v2enh5 (100+2n), sm2 (100+3n), deon10 (120+4n);
  * solver share of nbr CPU per band from beats-per-CPU-h (rounds C-J) plus the two audit picks anchor6/deon10 (lo);
  * sizes: unbeaten sizes get 80% of the CPU, split by a softness score = exp(0.35 z(-log rebuild) + 0.25 z(log distinct))
    x (1 + 0.4 clip(-anomaly/2.5e-3, 0, 1)), tempered half-uniform (backtest rho ~0.3); already-beaten sizes share 20% in
    proportion to their recent beat counts (margin durability); per-size cap 6% of CPU, >= 4 jobs per unbeaten size;
    26-50 keep one tv14pf6 run per size; --cpu-scale multiplies nbr budgets (cap 1200 s).

    python alloc_round.py --signals tmp/size_signals.json --warm-prev warm_roundJ --warm warm_roundK \
        --cpu-hours 140 --seed0 61 --out roundK.jsonl
"""
import argparse, json, math, os, collections, hashlib
ap=argparse.ArgumentParser()
ap.add_argument("--signals",required=True); ap.add_argument("--warm-prev",default=None); ap.add_argument("--warm",required=True)
ap.add_argument("--cpu-hours",type=float,required=True); ap.add_argument("--seed0",type=int,required=True); ap.add_argument("--out",required=True)
ap.add_argument("--cpu-scale",type=float,default=1.0,help="multiply nbr budgets (alternate 1.0 / 1.5 across rounds as a cutoff hedge; capped at 1200 s)")
ap.add_argument("--margin-share",type=float,default=0.20); ap.add_argument("--cap",type=float,default=0.06); ap.add_argument("--min-jobs",type=int,default=4)
a=ap.parse_args()
sig={int(k):v for k,v in json.load(open(a.signals)).items()}
MIX={"lo":{"tv14pf6":.33,"tv16ob6":.22,"v2enh5":.13,"sm2":.13,"nalt6":.09,"anchor6":.05,"deon10":.05},
     "hi":{"tv14pf6":.33,"tv16ob6":.25,"none7":.10,"sm7":.09,"sm2":.09,"anchor1":.05,"v2enh5":.05,"nalt6":.04}}
def band(n): return "lo" if n<=77 else "hi"
def cpu_nbr(solver,n):
    if solver in("sm7","v2enh5"): b=100+2*n
    elif solver=="sm2": b=100+3*n
    elif solver=="deon10": b=120+4*n
    else: b=float(min(800,max(350,9.4*n-154)))
    return float(min(1200.0,round(b*a.cpu_scale,1)))
total=a.cpu_hours*3600
mid_cpu=sum(cpu_nbr("tv14pf6",n) for n in range(26,51)); rows=[]
for n in range(26,51): rows.append({"solver":"tv14pf6","n":n,"seed":a.seed0,"mode":"nbr","cpu":cpu_nbr("tv14pf6",n),"tier":"mid"})
# self polish only where the incumbent moved since the previous round
def h(p):
    try: return hashlib.md5(open(p,"rb").read()).hexdigest()
    except OSError: return None
moved=[]
if a.warm_prev:
    for n in range(51,101):
        if h(os.path.join(a.warm_prev,"csqv%d.pck"%n))!=h(os.path.join(a.warm,"csqv%d.pck"%n)): moved.append(n)
for n in moved:
    rows.append({"solver":"tv14pf6","n":n,"seed":a.seed0,"mode":"self","cpu":30.0,"tier":"moved"})
    rows.append({"solver":"nalt6","n":n,"seed":a.seed0,"mode":"self","cpu":60.0,"tier":"moved"})
pool=total-sum(r["cpu"] for r in rows)
unb=[n for n in range(51,101) if sig[n]["verdict"]!="BEAT"]; won=[n for n in range(51,101) if sig[n]["verdict"]=="BEAT"]
def z(xs):
    m=sum(xs)/len(xs); s=math.sqrt(sum((x-m)**2 for x in xs)/max(1,len(xs)-1)) or 1.0
    return [(x-m)/s for x in xs]
x1=z([-math.log(max(sig[n]["rebuild_rate"],0.01)) for n in unb]); x2=z([math.log(max(sig[n]["distinct"],1)) for n in unb])
score={}
for i,n in enumerate(unb):
    an=sig[n]["anom"]; score[n]=math.exp(0.35*x1[i]+0.25*x2[i])*(1+0.4*min(1.0,max(0.0,-an/2.5e-3)))
S0=sum(score.values()); score={n:0.5*score[n]/S0+0.5/len(unb) for n in unb}   # temper: half uniform, half softness (backtest rho ~0.3)
S=sum(score.values()); cpu_n={n:pool*(1-a.margin_share)*score[n]/S for n in unb}
wb={n:sig[n].get("beats_recent",0)+0.5 for n in won}; W=sum(wb.values())
for n in won: cpu_n[n]=pool*a.margin_share*wb[n]/W
capc=a.cap*pool
for _ in range(20):   # cap per size, redistribute the excess over the uncapped unbeaten sizes
    over=sum(max(0,c-capc) for c in cpu_n.values())
    if over<1: break
    free=[n for n in unb if cpu_n[n]<capc-1]; fs=sum(score[n] for n in free) or 1
    for n in cpu_n:
        if cpu_n[n]>capc: cpu_n[n]=capc
    for n in free: cpu_n[n]+=over*score[n]/fs
for n in unb:   # floor: >= min-jobs at the two top solvers' budgets
    need=a.min_jobs/2*(cpu_nbr("tv14pf6",n)+cpu_nbr("tv16ob6",n))
    if cpu_n[n]<need: cpu_n[n]=need
per_size_jobs=collections.Counter()
for n in sorted(cpu_n):
    mix=MIX[band(n)]; sd=a.seed0
    for s,w in sorted(mix.items(),key=lambda x:-x[1]):
        c=cpu_nbr(s,n); k=int(round(w*cpu_n[n]/c))
        if k==0 and w>=0.2: k=1
        for j in range(k):
            rows.append({"solver":s,"n":n,"seed":sd,"mode":"nbr","cpu":c,"tier":"unbeaten" if n in unb else "margin"}); sd+=1
        per_size_jobs[n]+=k
with open(a.out,"w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
tot=sum(r["cpu"] for r in rows)
by=collections.defaultdict(lambda:[0,0.0])
for r in rows: by[(r["solver"],r["mode"],band(r["n"]) if r["n"]>50 else "mid")][0]+=1; by[(r["solver"],r["mode"],band(r["n"]) if r["n"]>50 else "mid")][1]+=r["cpu"]
print("jobs=%d cpu=%.1f h (target %.1f h); moved incumbents: %s"%(len(rows),tot/3600,a.cpu_hours,moved))
for k,v in sorted(by.items(),key=lambda x:-x[1][1]): print("  %-8s %-4s %-3s jobs=%4d cpu=%6.1f h"%(k[0],k[1],k[2],v[0],v[1]/3600))
print("  jobs per size:", " ".join("%d:%d"%(n,per_size_jobs[n]) for n in sorted(per_size_jobs)))
print("  unbeaten share: %.0f%% of nbr jobs"%(100*sum(per_size_jobs[n] for n in unb)/max(1,sum(per_size_jobs.values()))))
