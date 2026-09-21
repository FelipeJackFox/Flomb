"""Dev pilot for front 4: can a sharper Kenyon code cut the per-step error (false marks, confusions between neighbouring triples)?"""
import json,itertools
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from experiments import evaluate_fly_density as D

OUT=Path('benchmarks/fly-precision-pilot');GRID=[dict(sparsity=s,tuning=t) for s,t in itertools.product((.05,.1,.2,.4),(.2,.35,.5))]

def run(job):
    from experiments.fly_agents import play,RawFlyAgent
    from experiments.fly_senses import sniffs_raw
    from experiments.mushroom_body import MushroomBody
    cfg,seed=job;mb=MushroomBody(**{**D.CONFIG,**cfg},seed=seed);agent=RawFlyAgent(mb);rng=np.random.default_rng(seed+1);eta0=mb.eta
    for g in range(8000):mb.eta=eta0/(1+g/3000);size=[7,9][g%2];play(agent,10_000_000+g,size,D.training_mines(g,size,True),rng,True,sense=sniffs_raw,lethal=False)
    nine=[play(agent,24_000_000+i,9,12,rng,False,sense=sniffs_raw,zero_start=True) for i in range(300)];big=[play(agent,24_500_000+i,16,40,rng,False,sense=sniffs_raw,zero_start=True) for i in range(80)]
    rec=dict(**cfg,seed=seed,win9=100*np.mean([r['won'] for r in nine]),win16=100*np.mean([r['won'] for r in big]),err16=100*(1-sum(r['safe_steps'] for r in big)/sum(r['steps'] for r in big)),false=sum(r['false_marks'] for r in nine+big))
    (OUT/f"s{cfg['sparsity']}-t{cfg['tuning']}-{seed}.json").write_text(json.dumps(rec));return rec

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with Pool(9) as pool:
        for r in pool.imap_unordered(run,[(c,s) for c in GRID for s in (100,101,102,103)]):print(r,flush=True)
