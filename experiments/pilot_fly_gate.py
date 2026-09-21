"""Dev pilot for front 5: can the marking gate converge to a useful threshold from both sides if it explores more?"""
import json,itertools
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from experiments import evaluate_fly_density as D

OUT=Path('benchmarks/fly-gate-pilot2');GRID=[dict(theta=t0,tau=tau,lr=lr) for t0,tau,lr in itertools.product((-8.,-1.),(.25,.5),(.02,))]

def run(job):
    from experiments.fly_agents import play,RawFlyAgent,LearnedMarker
    from experiments.fly_senses import sniffs_raw
    from experiments.mushroom_body import MushroomBody
    cfg,seed=job;mb=MushroomBody(**D.CONFIG,seed=seed);agent=RawFlyAgent(mb);marker=LearnedMarker(burn=3.,**cfg);rng=np.random.default_rng(seed+1);eta0=mb.eta;curve=[]
    for g in range(12000):
        mb.eta=eta0/(1+g/3000);size=[7,9][g%2];play(agent,10_000_000+g,size,D.training_mines(g,size,True),rng,True,sense=sniffs_raw,lethal=False,marker=marker)
        if (g+1)%3000==0:curve.append(round(marker.theta,2))
    rows=[play(agent,24_000_000+i,9,12,rng,False,sense=sniffs_raw,zero_start=True,marker=marker) for i in range(300)];ref=[play(agent,24_000_000+i,9,12,rng,False,sense=sniffs_raw,zero_start=True) for i in range(300)]
    rec=dict(**cfg,seed=seed,theta_curve=curve,win=100*np.mean([r['won'] for r in rows]),win_with_3pct_reflex_same_brain=100*np.mean([r['won'] for r in ref]),marks=float(np.mean([r['marks'] for r in rows])),false=sum(r['false_marks'] for r in rows))
    (OUT/f"t{cfg['theta']}-tau{cfg['tau']}-lr{cfg['lr']}-{seed}.json").write_text(json.dumps(rec));return rec

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with Pool(9) as pool:
        for r in pool.imap_unordered(run,[(c,s) for c in GRID for s in (100,101,102)]):print(r,flush=True)
