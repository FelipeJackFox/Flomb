"""Dev-seed pilot for step 2: where does 'weak clue' cross 'odourless tile'? Never touches reserved layouts."""
import json,sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from experiments import evaluate_fly_validation as base

OUT=Path('benchmarks/fly-reference-pilot');VARIANTS=dict(current={},context_reference=dict(context_reference=True),far_field=dict(far_field=True),far_field_no_tonic=dict(far_field=True,context_reference=True))
DEV=[dict(seed=20_000_000+i,size=9,mines=12) for i in range(300)]+[dict(seed=21_000_000+i,size=7,mines=7) for i in range(200)]

def run(job):
    variant,seed=job;agent,mb=base.build(dict(seed=seed,**VARIANTS[variant]));rng=base.train(agent,mb,seed,base.GAMES);rows=base.evaluate(agent,DEV,rng)
    rec=dict(variant=variant,seed=seed,win9=100*np.mean([r['won'] for r in rows if r['size']==9]),win7=100*np.mean([r['won'] for r in rows if r['size']==7]),unknown_share=sum(r['unknown_steps'] for r in rows)/sum(r['steps'] for r in rows),
        false_marks=sum(r['false_marks'] for r in rows),weak=float(mb.valence(mb.kenyon(np.array([[.125,1.]],np.float32)))[0])*100,context=float(mb.baseline))
    (OUT/f'{variant}-{seed}.json').write_text(json.dumps(rec));return rec

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with Pool(9) as pool:
        for rec in pool.imap_unordered(run,[(v,s) for s in range(100,105) for v in VARIANTS]):print(rec,flush=True)
