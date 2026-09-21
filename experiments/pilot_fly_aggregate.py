"""Dev pilot: should the fly SUM or AVERAGE what its sniffs tell it? Same information, same brain, learned marking gate."""
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from experiments import evaluate_fly_density as D

OUT=Path('benchmarks/fly-aggregate-pilot')
def run(job):
    from experiments.mushroom_body import MushroomBody
    from experiments.fly_agents import RawFlyAgent,LearnedMarker,play
    from experiments.fly_senses import sniffs_raw
    agg,seed=job;reflex=agg.endswith('-reflex');label=agg;agg=agg.split('-')[0];mb=MushroomBody(**D.CONFIG,seed=seed);agent=RawFlyAgent(mb,aggregate=agg);marker=None if reflex else LearnedMarker(theta=-8.,tau=.25,lr=.02,burn=3.,ceiling=0. if 'cap' in job[0] else 4.);rng=np.random.default_rng(seed+1);eta0=mb.eta
    for g in range(12000):
        f=1/(1+g/3000);mb.eta=eta0*f
        if marker is not None:marker.lr=marker.lr0*f
        size=[7,9][g%2];play(agent,10_000_000+g,size,D.training_mines(g,size,True),rng,True,sense=sniffs_raw,lethal=False,marker=marker)
    nine=[play(agent,24_000_000+i,9,12,rng,False,sense=sniffs_raw,zero_start=True,marker=marker) for i in range(400)];big=[play(agent,24_500_000+i,16,40,rng,False,sense=sniffs_raw,zero_start=True,marker=marker) for i in range(80)]
    rec=dict(aggregate=label,seed=seed,win9=100*np.mean([r['won'] for r in nine]),win16=100*np.mean([r['won'] for r in big]),err9=100*(1-sum(r['safe_steps'] for r in nine)/sum(r['steps'] for r in nine)),blind=sum(r['unknown_steps'] for r in nine)/sum(r['steps'] for r in nine),theta=marker.theta if marker is not None else -3.48,false=sum(r['false_marks'] for r in nine+big))
    (OUT/f'{label}-{seed}.json').write_text(json.dumps(rec));return rec
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with Pool(8) as pool:
        for r in pool.imap_unordered(run,[(a,s) for s in (100,101,102,103,104,105,106,107) for a in ('mean-cap',)]):print(r,flush=True)
