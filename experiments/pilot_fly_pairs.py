"""Dev pilot: the mushroom-body fly with and without the working memory of clue pairs. Outcome-only learning, everything learned at once."""
import json,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from experiments import evaluate_fly_density as D

OUT=Path('benchmarks/fly-pairs-pilot');CHECKPOINTS=(5000,10000,20000,40000)

def run(job):
    from experiments.mushroom_body import MushroomBody
    from experiments.fly_agents import RawFlyAgent,PairFlyAgent,play
    from experiments.fly_senses import sniffs_raw,sniffs_pairs
    from experiments.fly_agents import PairTableAgent,TableAgent
    kind,seed=job;mb=MushroomBody(**D.CONFIG,seed=seed) if 'table' not in kind else None
    agent,sense={'pairs':lambda:(PairFlyAgent(mb),sniffs_pairs),'single':lambda:(RawFlyAgent(mb),sniffs_raw),'table-pairs':lambda:(PairTableAgent(),sniffs_pairs),'table-single':lambda:(TableAgent(),sniffs_raw)}[kind]()
    rng=np.random.default_rng(seed+1);eta0=mb.eta if mb is not None else agent.lr0;log=[];t=time.time()
    for g in range(CHECKPOINTS[-1]):
        eta=eta0/(1+g/10000)
        if mb is not None:mb.eta=eta
        else:agent.eta=eta
        size=[7,9][g%2];play(agent,10_000_000+g,size,D.training_mines(g,size,True),rng,True,sense=sense,lethal=False)
        if g+1 in CHECKPOINTS:
            nine=[play(agent,24_000_000+i,9,12,rng,False,sense=sense,zero_start=True) for i in range(300)];big=[play(agent,24_500_000+i,16,40,rng,False,sense=sense,zero_start=True) for i in range(50)]
            rec=dict(kind=kind,seed=seed,games=g+1,win9=round(100*np.mean([r['won'] for r in nine]),1),win16=round(100*np.mean([r['won'] for r in big]),1),err9=round(100*(1-sum(r['safe_steps'] for r in nine)/sum(r['steps'] for r in nine)),2),
                false=sum(r['false_marks'] for r in nine+big),pair_keys=len(getattr(agent,'pair_rows',{})),seconds=round(time.time()-t));log.append(rec);print(rec,flush=True);(OUT/f'{kind}-{seed}.json').write_text(json.dumps(log))
    return log

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    import sys
    kinds=sys.argv[1:] or ['pairs','single']
    with Pool(8) as pool:pool.map(run,[(k,s) for s in (100,101,102,103) for k in kinds])
