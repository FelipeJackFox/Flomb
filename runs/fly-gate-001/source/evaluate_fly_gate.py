"""Front 5: learned marking gate, formal. See research/PROTOCOLO_COMPUERTA_MARCA.md."""
import json,pickle,shutil
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats
from minesweeper import Minesweeper
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from experiments import evaluate_fly_density as D

OUT=Path('runs/fly-gate-001');GAMES=20000;SEEDS=tuple(range(10));GROUPS=('gate','reflex','sham','never')
def configure(out):
    global OUT;OUT=Path(out)
def marker_for(group):
    from experiments.fly_agents import LearnedMarker
    if group=='reflex':return None
    if group=='never':return LearnedMarker(theta=-30.,lr=0.)
    return LearnedMarker(theta=-8.,tau=.25,lr=.02,burn=3.,sham=group=='sham')
def run(arm):
    from experiments.fly_agents import play
    from experiments.fly_senses import sniffs_raw
    group,seed=arm;name=f'{group}-{seed}'
    if (OUT/f'{name}-games.json').exists():return name
    games=json.loads((OUT/'games.json').read_text());agent,mb,_=D.build('fly-near-var',seed);marker=marker_for(group);rng=np.random.default_rng(seed+1);eta0=mb.eta;curve=[]
    for g in range(GAMES):
        f=1/(1+g/3000);size=[7,9][g%2];mb.eta=eta0*f
        if marker is not None:marker.lr=marker.lr0*f
        play(agent,10_000_000+g,size,D.training_mines(g,size,True),rng,True,sense=sniffs_raw,lethal=False,marker=marker)
        if marker is not None and (g+1)%2000==0:curve.append(round(marker.theta,3))
    rows=[dict(**g,**play(agent,g['seed'],g['size'],g['mines'],rng,False,sense=sniffs_raw,marker=marker,zero_start=True)) for g in games]
    write_json(OUT/f'{name}-games.json',rows);write_json(OUT/f'{name}-extra.json',dict(theta=None if marker is None else marker.theta,theta_curve=curve));return name
def summarise():
    rows={p.name[:-11]:json.loads(p.read_text()) for p in OUT.glob('*-games.json')};summary={}
    def interval(v):v=np.asarray(v,float);half=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v));return dict(mean=round(float(v.mean()),1),sd=round(float(v.std(ddof=1)),1),ci95=[round(float(v.mean()-half),1),round(float(v.mean()+half),1)],per_seed=[round(float(x),1) for x in v])
    for size in (9,7):
        per={g:[100*np.mean([r['won'] for r in rows[f'{g}-{s}'] if r['size']==size]) for s in SEEDS] for g in GROUPS};block={g:interval(v) for g,v in per.items()}
        for b in GROUPS[1:]:d=np.array(per['gate'])-np.array(per[b]);block[f'gate minus {b}']=dict(**interval(d),p=round(float(stats.ttest_rel(per['gate'],per[b]).pvalue),4))
        block['marks_per_game']={g:round(float(np.mean([r['marks'] for s in SEEDS for r in rows[f'{g}-{s}'] if r['size']==size])),2) for g in GROUPS}
        block['false_marks']={g:int(sum(r['false_marks'] for s in SEEDS for r in rows[f'{g}-{s}'] if r['size']==size)) for g in GROUPS};summary[str(size)]=block
    summary['theta']={g:dict(final=[round(json.loads((OUT/f'{g}-{s}-extra.json').read_text())['theta'],2) for s in SEEDS],mean_curve=np.mean([json.loads((OUT/f'{g}-{s}-extra.json').read_text())['theta_curve'] for s in SEEDS],0).round(2).tolist()) for g in ('gate','sham')};summary['reflex_theta_equivalent']=-3.48
    write_json(OUT/'summary.json',summary);return summary
def main():
    from experiments.evaluate_difficulty_transfer import prior_layouts
    if not (OUT/'games.json').exists():
        OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
        for p in ['experiments/evaluate_fly_gate.py','experiments/fly_agents.py','experiments/mushroom_body.py','experiments/fly_senses.py','research/PROTOCOLO_COMPUERTA_MARCA.md']:shutil.copy2(p,OUT/'source'/Path(p).name)
        used=prior_layouts(exclude=OUT);games=[]
        for size,mines,n,seed in ((9,12,500,9_100_000_000),(7,7,250,9_101_000_000)):
            found=0
            while found<n:
                env=Minesweeper(seed,size,mines,zero_start=True);key=identity(env);seed+=1
                if key in used or env.won:continue
                used.add(key);games.append(dict(seed=seed-1,size=size,mines=mines,layout_hash=key));found+=1
        write_json(OUT/'games.json',games)
        with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
        write_json(OUT/'manifest.json',dict(groups=GROUPS,seeds=list(SEEDS),training_games=GAMES,gate=dict(theta0=-8,tau=.25,lr=.02,decay=.9,burn=3),unit='seed'))
    with Pool(9,initializer=configure,initargs=(str(OUT),)) as pool:
        for name in pool.imap_unordered(run,[(g,s) for s in SEEDS for g in GROUPS]):print('done',name,flush=True)
    print(json.dumps(summarise(),indent=1));write_json(OUT/'completed.json',dict(completed=True))
if __name__=='__main__':main()
