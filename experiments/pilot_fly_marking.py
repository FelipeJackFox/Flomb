"""Dev pilot for point 4: does a fly that starts never marking learn to mark from delayed reward alone? Curves of theta and wins."""
import json,sys,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from experiments import evaluate_fly_density as E
from experiments.fly_agents import play,LearnedMarker
from experiments.fly_senses import sniffs_raw

OUT=Path('benchmarks/fly-marking-pilot');DEV=[(20_000_000+i,9,12) for i in range(150)]+[(21_000_000+i,7,7) for i in range(100)]
VARIANTS={'lr02-burn3':dict(lr=.02,burn=3.),'lr005-burn3':dict(lr=.005,burn=3.),'lr02-burn1':dict(lr=.02,burn=1.)}

def run(job):
    variant,seed=job;agent,mb,_=E.build('fly-near-var',seed);marker=LearnedMarker(**VARIANTS[variant]);rng=np.random.default_rng(seed+1);eta0=mb.eta;log=[];t=time.time()
    for g in range(15000):
        mb.eta=eta0/(1+g/3000);marker.lr=marker.lr0/(1+g/3000);size=[7,9][g%2];play(agent,10_000_000+g,size,E.training_mines(g,size,True),rng,True,sense=sniffs_raw,lethal=False,marker=marker)
        if (g+1)%2500==0:
            rows=[(s,play(agent,sd,s,m,rng,False,sense=sniffs_raw,marker=marker)) for sd,s,m in DEV];nine=[r for s,r in rows if s==9]
            rec=dict(variant=variant,seed=seed,games=g+1,theta=round(marker.theta,2),threshold_pct=round(100/(1+np.exp(-marker.theta)),2),win9=round(100*np.mean([r['won'] for r in nine]),1),win7=round(100*np.mean([r['won'] for s,r in rows if s==7]),1),
                marks=round(float(np.mean([r['marks'] for r in nine])),2),false_marks=sum(r['false_marks'] for _,r in rows),seconds=round(time.time()-t));log.append(rec);print(rec,flush=True);(OUT/f'{variant}-{seed}.json').write_text(json.dumps(log))
    return log

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with Pool(9) as pool:pool.map(run,[(v,s) for v in VARIANTS for s in (100,101,102)])
