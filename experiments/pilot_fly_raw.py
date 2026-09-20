"""Dev-seed pilot: can anything learn from raw (N,k,m) sniffs with outcome-only feedback? Learning curves, no reserved layouts."""
import json,sys,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from experiments.fly_senses import sniffs_raw
from experiments.fly_agents import play,RawFlyAgent,TableAgent,AdditiveAgent
from experiments.mushroom_body import MushroomBody

OUT=Path('benchmarks/fly-raw-pilot');CONFIG=dict(rule='rpe',sparsity=.2,tuning=.35,beta=100.);GAMES=40000;EVERY=5000
JOBS=[('fly',100,{}),('fly',101,{}),('fly',102,{}),('fly-shuffled',100,dict(shuffled=True)),('fly-shuffled',101,dict(shuffled=True)),('table',100,None),('table',101,None),('additive',100,None),('additive',101,None)]
DEV=[(20_000_000+i,9,12) for i in range(150)]+[(21_000_000+i,7,7) for i in range(100)]

def make(kind,seed,options,lethal=True):
    if kind.startswith('fly'):mb=MushroomBody(**CONFIG,seed=seed,**options);return RawFlyAgent(mb),mb
    return (TableAgent() if kind=='table' else AdditiveAgent()),None

def run(job,games=GAMES,lethal=True,density=None,tag=''):
    kind,seed,options=job;agent,mb=make(kind,seed,options);rng=np.random.default_rng(seed+1);eta0=mb.eta if mb is not None else agent.lr0;t=time.time();log=[]
    for g in range(games):
        eta=eta0/(1+g/3000)
        if mb is not None:mb.eta=eta
        else:agent.eta=eta
        size=[7,9][g%2];mines=[7,12][g%2] if density is None else density(g,size,rng)
        play(agent,10_000_000+g,size,mines,rng,True,sense=sniffs_raw,lethal=lethal)
        if (g+1)%EVERY==0:
            rows=[(s,play(agent,seed_,s,m,rng,False,sense=sniffs_raw)) for seed_,s,m in DEV];nine=[r for s,r in rows if s==9];seven=[r for s,r in rows if s==7]
            rec=dict(kind=kind+tag,seed=seed,games=g+1,win9=round(100*np.mean([r['won'] for r in nine]),1),win7=round(100*np.mean([r['won'] for r in seven]),1),marks=round(float(np.mean([r['marks'] for r in nine])),2),false_marks=sum(r['false_marks'] for _,r in rows),
                safe_rate=round(sum(r['safe_steps'] for _,r in rows)/sum(r['steps'] for _,r in rows),3),stimuli=len(agent.index) if hasattr(agent,'index') else 27,seconds=round(time.time()-t))
            log.append(rec);print(rec,flush=True);(OUT/f'{kind}{tag}-{seed}.json').write_text(json.dumps(log))
    return log

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with Pool(9) as pool:pool.map(run,JOBS)
