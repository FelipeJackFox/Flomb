"""Sum or average the sniffs? Formal. See research/PROTOCOLO_PROMEDIAR_OLFATEOS.md."""
import json,pickle,shutil
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats
from minesweeper import Minesweeper
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from experiments import evaluate_fly_density as D

OUT=Path('runs/fly-aggregate-001');GAMES=20000;SEEDS=tuple(range(10));ARMS={'sum':('sum',4.),'sum-cap':('sum',0.),'mean-cap':('mean',0.)};SETS=((9,12,500),(7,7,250),(16,40,150))
def configure(out):
    global OUT;OUT=Path(out)
def run(job):
    from experiments.mushroom_body import MushroomBody
    from experiments.fly_agents import RawFlyAgent,LearnedMarker,play
    from experiments.fly_senses import sniffs_raw
    arm,seed=job;name=f'{arm}-{seed}'
    if (OUT/f'{name}-games.json').exists():return name
    agg,cap=ARMS[arm];games=json.loads((OUT/'games.json').read_text());mb=MushroomBody(**D.CONFIG,seed=seed);agent=RawFlyAgent(mb,aggregate=agg);marker=LearnedMarker(theta=-8.,tau=.25,lr=.02,burn=3.,ceiling=cap);rng=np.random.default_rng(seed+1);eta0=mb.eta
    for g in range(GAMES):
        f=1/(1+g/3000);mb.eta=eta0*f;marker.lr=marker.lr0*f;size=[7,9][g%2];play(agent,10_000_000+g,size,D.training_mines(g,size,True),rng,True,sense=sniffs_raw,lethal=False,marker=marker)
    rows=[dict(**g,**play(agent,g['seed'],g['size'],g['mines'],rng,False,sense=sniffs_raw,marker=marker,zero_start=True)) for g in games]
    write_json(OUT/f'{name}-games.json',rows);write_json(OUT/f'{name}-extra.json',dict(theta=marker.theta));return name
def hand(_):
    from experiments import evaluate_fly_start as S
    S.OUT=OUT;games=json.loads((OUT/'games.json').read_text());write_json(OUT/'games.json',[dict(**g,zero=True,set=str(g['size'])) for g in games]) if 'zero' not in games[0] else None;return S.hand(0)
def summarise():
    rows={p.name[:-11]:json.loads(p.read_text()) for p in OUT.glob('*-games.json')};summary={}
    def interval(v):v=np.asarray(v,float);half=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v));return dict(mean=round(float(v.mean()),1),sd=round(float(v.std(ddof=1)),1),ci95=[round(float(v.mean()-half),1),round(float(v.mean()+half),1)],per_seed=[round(float(x),1) for x in v])
    for size,_,_ in SETS:
        per={a:[100*np.mean([r['won'] for r in rows[f'{a}-{s}'] if r['size']==size]) for s in SEEDS] for a in ARMS};block={a:interval(v) for a,v in per.items()}
        for a,b in (('mean-cap','sum'),('sum-cap','sum'),('mean-cap','sum-cap')):d=np.array(per[a])-np.array(per[b]);block[f'{a} minus {b}']=dict(**interval(d),p=round(float(stats.ttest_rel(per[a],per[b]).pvalue),4))
        block['hand']=round(float(100*np.mean([r['won'] for r in rows['hand'] if r['size']==size])),1)
        block['safe_step_pct']={a:round(float(np.mean([100*sum(r['safe_steps'] for r in rows[f'{a}-{s}'] if r['size']==size)/sum(r['steps'] for r in rows[f'{a}-{s}'] if r['size']==size) for s in SEEDS])),2) for a in ARMS}
        block['false_marks']={a:int(sum(r['false_marks'] for s in SEEDS for r in rows[f'{a}-{s}'] if r['size']==size)) for a in ARMS};summary[str(size)]=block
    summary['theta']={a:[round(json.loads((OUT/f'{a}-{s}-extra.json').read_text())['theta'],2) for s in SEEDS] for a in ARMS};write_json(OUT/'summary.json',summary);return summary
def main():
    from experiments.evaluate_difficulty_transfer import prior_layouts
    if not (OUT/'games.json').exists():
        OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
        for p in ['experiments/evaluate_fly_aggregate.py','experiments/fly_agents.py','experiments/mushroom_body.py','experiments/fly_senses.py','research/PROTOCOLO_PROMEDIAR_OLFATEOS.md']:shutil.copy2(p,OUT/'source'/Path(p).name)
        used=prior_layouts(exclude=OUT);games=[]
        for j,(size,mines,n) in enumerate(SETS):
            seed=9_200_000_000+j*1_000_000;found=0
            while found<n:
                env=Minesweeper(seed,size,mines,zero_start=True);key=identity(env);seed+=1
                if key in used or env.won:continue
                used.add(key);games.append(dict(seed=seed-1,size=size,mines=mines,zero=True,set=str(size),layout_hash=key));found+=1
        write_json(OUT/'games.json',games)
        with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
        write_json(OUT/'manifest.json',dict(arms={k:list(v) for k,v in ARMS.items()},seeds=list(SEEDS),training_games=GAMES,unit='seed'))
    with Pool(9,initializer=configure,initargs=(str(OUT),)) as pool:
        for name in pool.imap_unordered(run,[(a,s) for s in SEEDS for a in ARMS]):print('done',name,flush=True)
        print('done',pool.apply(hand,(0,)),flush=True)
    print(json.dumps(summarise(),indent=1));write_json(OUT/'completed.json',dict(completed=True))
if __name__=='__main__':main()
