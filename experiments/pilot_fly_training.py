"""Dev-seed pilot for point 2: non-lethal training and a density curriculum, on raw senses. Evaluation is always lethal at 15% mines."""
from multiprocessing import Pool
from pathlib import Path
from experiments import pilot_fly_raw as base
base.OUT=Path('benchmarks/fly-training-pilot')

def ramp(g,size,rng):
    """Generic curriculum (no solver): mine count rises linearly from ~40% of the target to the target over the first 4000 games."""
    target=[7,12][size==9];return max(2,int(round(target*(.4+.6*min(1.,g/4000)))))

VARIANTS=dict(lethal=dict(),soft=dict(lethal=False),ramp=dict(density=ramp),soft_ramp=dict(lethal=False,density=ramp))
def run(job):variant,seed=job;return base.run(('fly',seed,{}),games=20000,tag='-'+variant,**VARIANTS[variant])

if __name__=='__main__':
    base.OUT=Path('benchmarks/fly-training-pilot');base.OUT.mkdir(parents=True,exist_ok=True)
    with Pool(9) as pool:pool.map(run,[(v,s) for s in (100,101) for v in VARIANTS]+[('soft',102)])
