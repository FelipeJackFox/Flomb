"""Where does a brain beat a table? Sample-efficiency curves and sensory noise. See research/PROTOCOLO_MOSCA_VS_TABLA.md."""
import json,pickle,shutil
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats
from minesweeper import Minesweeper
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from experiments import evaluate_fly_density as D

OUT=Path('runs/fly-vs-table-001');SEEDS=tuple(range(8));CHECKPOINTS=(250,500,1000,2000,4000,8000,16000);SIGMAS=(0.,.15,.30);NOISE_GAMES=10000
JOBS=[('curve',k,l,s) for k in ('fly','table') for l in ('lethal','soft') for s in SEEDS]+[('noise',k,sg,s) for k in ('fly','table-int','table-fine') for sg in SIGMAS for s in SEEDS]

def configure(out):
    global OUT;OUT=Path(out)

def noisy(sigma,seed):
    from experiments.fly_senses import sniffs_raw
    if not sigma:return sniffs_raw
    rng=np.random.default_rng([seed,99])
    def sense(visible,size,marked=None):
        cells,raw=sniffs_raw(visible,size,marked);return cells,raw*np.exp(sigma*rng.standard_normal(raw.shape)).astype(np.float32)
    return sense

def make(kind,seed,sigma=0.):
    from experiments.mushroom_body import MushroomBody
    from experiments.fly_agents import RawFlyAgent,TableAgent
    if kind=='fly':mb=MushroomBody(**D.CONFIG,seed=seed);return RawFlyAgent(mb,resolution=1. if not sigma else .05,cache=not sigma),mb
    return TableAgent(resolution=.5 if kind=='table-fine' else 1.),None

def run(job):
    from experiments.fly_agents import play
    part,kind,setting,seed=job;name=f'{part}-{kind}-{setting}-{seed}'
    if (OUT/f'{name}.json').exists():return name
    games=json.loads((OUT/'games.json').read_text());sigma=setting if part=='noise' else 0.;agent,mb=make(kind,seed,sigma);sense=noisy(sigma,seed);rng=np.random.default_rng(seed+1);eta0=mb.eta if mb is not None else agent.lr0
    total=CHECKPOINTS[-1] if part=='curve' else NOISE_GAMES;marks=set(CHECKPOINTS) if part=='curve' else {total};steps=0;curve=[]
    for g in range(total):
        eta=eta0/(1+g/3000);size=[7,9][g%2]
        if mb is not None:mb.eta=eta
        else:agent.eta=eta
        steps+=play(agent,10_000_000+g,size,D.training_mines(g,size,True),rng,True,sense=sense,lethal=setting=='lethal')['steps']
        if g+1 in marks:
            rows=[play(agent,x['seed'],9,12,rng,False,sense=sense,zero_start=True) for x in games]
            curve.append(dict(games=g+1,training_steps=steps,win=100*float(np.mean([r['won'] for r in rows])),safe=100*sum(r['safe_steps'] for r in rows)/sum(r['steps'] for r in rows),marks=float(np.mean([r['marks'] for r in rows])),false_marks=int(sum(r['false_marks'] for r in rows))))
    write_json(OUT/f'{name}.json',dict(part=part,kind=kind,setting=setting,seed=seed,curve=curve,stimuli=len(getattr(agent,'codes',{}))));return name

def summarise():
    R=[json.loads(p.read_text()) for p in OUT.glob('curve-*.json')]+[json.loads(p.read_text()) for p in OUT.glob('noise-*.json')]
    def interval(v):v=np.asarray(v,float);half=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v));return dict(mean=round(float(v.mean()),1),ci95=[round(float(v.mean()-half),1),round(float(v.mean()+half),1)])
    def get(part,kind,setting,i,what='win'):return [next(r for r in R if r['part']==part and r['kind']==kind and r['setting']==setting and r['seed']==s)['curve'][i][what] for s in SEEDS]
    summary=dict(curve={},noise={})
    for l in ('lethal','soft'):
        for i,c in enumerate(CHECKPOINTS):
            f,t=get('curve','fly',l,i),get('curve','table',l,i);d=np.array(f)-np.array(t)
            summary['curve'][f'{l}-{c}']=dict(fly=interval(f),table=interval(t),fly_minus_table=dict(**interval(d),p=round(float(stats.ttest_rel(f,t).pvalue),4)),steps_fly=int(np.mean(get('curve','fly',l,i,'training_steps'))),steps_table=int(np.mean(get('curve','table',l,i,'training_steps'))))
    for sg in SIGMAS:
        per={k:get('noise',k,sg,0) for k in ('fly','table-int','table-fine')};cell={k:interval(v) for k,v in per.items()}
        for k in ('table-int','table-fine'):d=np.array(per['fly'])-np.array(per[k]);cell[f'fly minus {k}']=dict(**interval(d),p=round(float(stats.ttest_rel(per['fly'],per[k]).pvalue),4))
        cell['false_marks']={k:int(np.sum(get('noise',k,sg,0,'false_marks'))) for k in per};summary['noise'][str(sg)]=cell
    write_json(OUT/'summary.json',summary);return summary

def main():
    from experiments.evaluate_difficulty_transfer import prior_layouts
    if not (OUT/'games.json').exists():
        OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
        for p in ['experiments/evaluate_fly_vs_table.py','experiments/fly_agents.py','experiments/mushroom_body.py','experiments/fly_senses.py','minesweeper.py','research/PROTOCOLO_MOSCA_VS_TABLA.md']:shutil.copy2(p,OUT/'source'/Path(p).name)
        used=prior_layouts(exclude=OUT);games=[];seed=8_800_000_000
        while len(games)<300:
            env=Minesweeper(seed,9,12,zero_start=True);key=identity(env);seed+=1
            if key in used or env.won:continue
            used.add(key);games.append(dict(seed=seed-1,size=9,mines=12,zero=True,layout_hash=key))
        write_json(OUT/'games.json',games)
        with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
        write_json(OUT/'manifest.json',dict(seeds=list(SEEDS),checkpoints=CHECKPOINTS,sigmas=SIGMAS,noise_games=NOISE_GAMES,unit='seed'))
    with Pool(9,initializer=configure,initargs=(str(OUT),)) as pool:
        for name in pool.imap_unordered(run,sorted(JOBS,key=lambda j:(j[0]=='curve',j[1]!='fly'))):print('done',name,flush=True)
    print(json.dumps(summarise(),indent=1));write_json(OUT/'completed.json',dict(completed=True))

if __name__=='__main__':main()
