"""Dev-seed pilot: does a more lenient stress threshold remove the never-marks collapse without creating false marks?"""
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from experiments import evaluate_fly_validation as base
from experiments.pilot_fly_reference import DEV

OUT=Path('benchmarks/fly-punish2-pilot');FLIES=[dict(seed=102),dict(seed=103),dict(seed=3),dict(seed=100),dict(seed=101),dict(seed=1,shuffled=True),dict(seed=3,shuffled=True),dict(seed=5,shuffled=True),dict(seed=6,logistic=True)]

def run(job):
    dread,options=job;base.DREAD=.03;base.EXTRA=dict(punish_gain=dread);agent,mb=base.build(options);rng=base.train(agent,mb,options['seed'],base.GAMES);rows=base.evaluate(agent,DEV,rng)
    rec=dict(dread=dread,fly=json.dumps(options),win9=100*np.mean([r['won'] for r in rows if r['size']==9]),win7=100*np.mean([r['won'] for r in rows if r['size']==7]),marks=float(np.mean([r['marks'] for r in rows])),false_marks=sum(r['false_marks'] for r in rows))
    (OUT/f"{dread}-{'-'.join(f'{k}{v}' for k,v in options.items())}.json").write_text(json.dumps(rec));return rec

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with Pool(9) as pool:
        for rec in pool.imap_unordered(run,[(d,f) for f in FLIES for d in (2.,3.)]):print(rec,flush=True)
