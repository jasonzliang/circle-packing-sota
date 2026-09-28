"""per-size softness signals for N=51..100: verdict, increment anomaly, nbr ruggedness, nbr best gap, rebuild rate
   size_signals.py [OUT.json] [AGG_DIR]   (defaults: $CP_SWEEP_ROOT/tmp/size_signals.json, $CP_SWEEP_ROOT/agg;
   reads $CP_LIVE/latest.json and the warm dirs under $CP_CENSUS)"""
import json, csv, math, collections, os, glob, sys
ROOT=os.environ.get('CP_SWEEP_ROOT','/tmp/si_tools/cp_sweep'); CENSUS=os.environ.get('CP_CENSUS','/tmp/si_tools/cp_census'); LIVE_DIR=os.environ.get('CP_LIVE','/tmp/si_tools/cp_live')
OUT=sys.argv[1] if len(sys.argv)>1 else os.path.join(ROOT,'tmp','size_signals.json')
AGG=sys.argv[2] if len(sys.argv)>2 else os.path.join(ROOT,'agg')
live=json.load(open(os.path.join(LIVE_DIR,'latest.json')))
rec={}
for k,v in live.items():
    if k.isdigit(): rec[int(k)]=float(v) if not isinstance(v,dict) else float(v.get('sum_r', v.get('value')))
    elif isinstance(v,dict) and k in('records','values','data'):
        for kk,vv in v.items(): rec[int(kk)]=float(vv) if not isinstance(vv,dict) else float(vv.get('sum_r', vv.get('value')))
    elif isinstance(v,list) and k in('records','values','data'):
        for it in v: rec[int(it['n'])]=float(it.get('sum_r', it.get('value')))
if not rec: print(list(live.keys())[:10]); raise SystemExit
# increment anomaly: d(N)=rec(N)-rec(N-1); trend via 7-point median around N (excluding N); anomaly = d - trend (negative = soft)
d={n:rec[n]-rec[n-1] for n in range(46,106) if n in rec and n-1 in rec}
anom={}
for n in range(51,101):
    win=[d[m] for m in range(n-4,n+5) if m!=n and m in d]
    win.sort(); med=win[len(win)//2]
    anom[n]=d[n]-med
per={int(r['n']):r for r in csv.DictReader(open(os.path.join(AGG,'per_n.csv')))}
WARM={'phase1':'warm_p1','roundA':'warm_p1'}
_inc={}
def inc(phase,n):
    k=(phase,n)
    if k not in _inc:
        p=os.path.join(CENSUS,WARM.get(phase,'warm_%s'%phase),'csqv%d.pck'%n)
        try:_inc[k]=math.fsum(float(l.split()[2]) for l in open(p).read().split('\n')[2:] if len(l.split())>=3)
        except OSError:_inc[k]=None
    return _inc[k]
vals=collections.defaultdict(list); rebuilt=collections.Counter(); njobs=collections.Counter(); gaps=collections.defaultdict(list); beats=collections.Counter()
EARLY={'phase1','roundA','roundB'}
for r in csv.DictReader(open(os.path.join(AGG,'jobs.csv'))):
    n=int(r['n'])
    if n<51 or r['mode']!='nbr' or r['feasible']!='1' or not r['sum_r']: continue
    s=float(r['sum_r']); i=inc(r['phase'],n)
    if i is None: continue
    njobs[n]+=1; vals[n].append(s)
    if s>i+1e-9 and r['phase'] not in EARLY: beats[n]+=1
    if s>i-1e-7: rebuilt[n]+=1
    else: gaps[n].append(i-s)
print('n,verdict,margin,record,anom,distinct,nbr_jobs,rebuild_rate,best_gap,med_gap,beats_recent')
rows=[]
for n in range(51,101):
    vs=vals[n]; top=max(vs) if vs else 0; dist=len({round(v,9) for v in vs if top-v<2e-3})
    g=sorted(gaps[n]); bg=g[0] if g else float('nan'); mg=g[len(g)//2] if g else float('nan')
    rows.append((n,per[n]['verdict'],per[n]['margin_vs_live'],rec[n],anom[n],dist,njobs[n],rebuilt[n]/max(1,njobs[n]),bg,mg))
    print('%d,%s,%s,%.9f,%+.2e,%d,%d,%.2f,%.1e,%.1e,%d'%(rows[-1]+(beats[n],)))
json.dump({str(r[0]):{'verdict':r[1],'anom':r[4],'distinct':r[5],'nbr_jobs':r[6],'rebuild_rate':r[7],'best_gap':r[8] if r[8]==r[8] else None,'med_gap':r[9] if r[9]==r[9] else None,'beats_recent':beats[int(r[0])]} for r in rows},open(OUT,'w'),indent=1)
