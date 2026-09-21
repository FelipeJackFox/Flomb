"""Does the real wiring matter early in learning? See research/PROTOCOLO_CABLEADO.md."""
import json,pickle,shutil
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats
from minesweeper import Minesweeper
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from experiments import evaluate_fly_density as D

OUT=Path('runs/fly-wiring-001');SEEDS=tuple(range(10));CHECKPOINTS=(250,500,1000,2000,4000,8000);ARMS={'real':False,'shuffled':True,'input':'input','output':'output','random':'random'};EXTRA={}

def configure(out,extra):
    global OUT,EXTRA;OUT=Path(out);EXTRA=dict(extra)

def run(job):
    from experiments.fly_agents import play,RawFlyAgent
    from experiments.fly_senses import sniffs_raw
    from experiments.mushroom_body import MushroomBody
    arm,seed=job;name=f'{arm}-{seed}'
    if (OUT/f'{name}.json').exists():return name
    games=json.loads((OUT/'games.json').read_text());mb=MushroomBody(**D.CONFIG,**EXTRA,seed=seed,shuffled=ARMS[arm]);agent=RawFlyAgent(mb);rng=np.random.default_rng(seed+1);eta0=mb.eta;curve=[]
    for g in range(CHECKPOINTS[-1]):
        mb.eta=eta0/(1+g/3000);size=[7,9][g%2];play(agent,10_000_000+g,size,D.training_mines(g,size,True),rng,True,sense=sniffs_raw,lethal=False)
        if g+1 in CHECKPOINTS:
            rows=[play(agent,x['seed'],9,12,rng,False,sense=sniffs_raw,zero_start=True) for x in games]
            curve.append(dict(games=g+1,win=100*float(np.mean([r['won'] for r in rows])),safe=100*sum(r['safe_steps'] for r in rows)/sum(r['steps'] for r in rows),false_marks=int(sum(r['false_marks'] for r in rows))))
    write_json(OUT/f'{name}.json',dict(arm=arm,seed=seed,curve=curve));return name

def summarise():
    R={(r['arm'],r['seed']):r for r in (json.loads(p.read_text()) for p in OUT.glob('*-[0-9]*.json'))}
    def interval(v):v=np.asarray(v,float);half=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v));return dict(mean=round(float(v.mean()),1),ci95=[round(float(v.mean()-half),1),round(float(v.mean()+half),1)])
    wins=lambda a,s:np.array([c['win'] for c in R[a,s]['curve']]);summary=dict(curve={},early={},plateau={})
    for a in ARMS:
        summary['curve'][a]={str(c):interval([wins(a,s)[i] for s in SEEDS]) for i,c in enumerate(CHECKPOINTS)};summary['false_marks_8k']=summary.get('false_marks_8k',{})|{a:int(sum(R[a,s]['curve'][-1]['false_marks'] for s in SEEDS))}
    for a in list(ARMS)[1:]:
        e=[wins('real',s)[:4].mean()-wins(a,s)[:4].mean() for s in SEEDS];p=[wins('real',s)[-1]-wins(a,s)[-1] for s in SEEDS]
        summary['early'][f'real minus {a}']=dict(**interval(e),p=round(float(stats.ttest_1samp(e,0).pvalue),4));summary['plateau'][f'real minus {a}']=dict(**interval(p),p=round(float(stats.ttest_1samp(p,0).pvalue),4))
    write_json(OUT/'summary.json',summary);return summary

def main(out=None,extra=None):
    global OUT,EXTRA
    from experiments.evaluate_difficulty_transfer import prior_layouts
    if out:OUT=Path(out)
    if extra:EXTRA=dict(extra)
    if not (OUT/'games.json').exists():
        OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
        for p in ['experiments/evaluate_fly_wiring.py','experiments/fly_agents.py','experiments/mushroom_body.py','experiments/fly_senses.py','research/PROTOCOLO_CABLEADO.md']:shutil.copy2(p,OUT/'source'/Path(p).name)
        used=prior_layouts(exclude=OUT);games=[];seed=8_900_000_000+sum(map(ord,OUT.name))*100_000
        while len(games)<300:
            env=Minesweeper(seed,9,12,zero_start=True);key=identity(env);seed+=1
            if key in used or env.won:continue
            used.add(key);games.append(dict(seed=seed-1,size=9,mines=12,layout_hash=key))
        write_json(OUT/'games.json',games)
        with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
        write_json(OUT/'manifest.json',dict(arms={k:str(v) for k,v in ARMS.items()},extra=EXTRA,seeds=list(SEEDS),checkpoints=CHECKPOINTS,unit='seed'))
    with Pool(9,initializer=configure,initargs=(str(OUT),EXTRA)) as pool:
        for name in pool.imap_unordered(run,[(a,s) for s in SEEDS for a in ARMS]):print('done',name,flush=True)
    print(json.dumps(summarise(),indent=1));write_json(OUT/'completed.json',dict(completed=True))

if __name__=='__main__':main()
