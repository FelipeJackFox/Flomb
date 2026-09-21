"""Fair start (guaranteed opening) and large boards. See research/PROTOCOLO_INICIO_Y_TAMANO.md."""
import json,pickle,shutil
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats
from minesweeper import Minesweeper
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from experiments import evaluate_fly_density as D

OUT=Path('runs/fly-start-001');SEEDS=tuple(range(8))
ARMS={'old':dict(zero=False,sizes=(7,9),games=20000),'zero':dict(zero=True,sizes=(7,9),games=20000),'zero-big':dict(zero=True,sizes=(9,16),games=10000)}
SETS={'9-old':dict(size=9,mines=12,zero=False,n=400),'9-zero':dict(size=9,mines=12,zero=True,n=400),'16-zero':dict(size=16,mines=40,zero=True,n=150),'30-zero':dict(size=30,mines=140,zero=True,n=30)}

def configure(out):
    global OUT;OUT=Path(out)

def run(arm):
    from experiments.fly_agents import play
    from experiments.fly_senses import sniffs_raw
    group,seed=arm;name=f'{group}-{seed}';assert not (OUT/'completed.json').exists()
    if (OUT/f'{name}-games.json').exists():return name
    cfg=ARMS[group];games=json.loads((OUT/'games.json').read_text());agent,mb,_=D.build('fly-near-var',seed);rng=np.random.default_rng(seed+1);eta0=mb.eta;steps=0
    for g in range(cfg['games']):
        mb.eta=eta0/(1+g/(3000*cfg['games']/20000));size=cfg['sizes'][g%2];steps+=play(agent,10_000_000+g,size,D.training_mines(g,size,True),rng,True,sense=sniffs_raw,lethal=False,zero_start=cfg['zero'])['steps']
    rows=[dict(**g,**play(agent,g['seed'],g['size'],g['mines'],rng,False,sense=sniffs_raw,zero_start=g['zero'])) for g in games]
    write_json(OUT/f'{name}-games.json',rows);write_json(OUT/f'{name}-extra.json',dict(training_steps=steps));return name

def hand(_):
    from experiments.fly_senses import sniffs
    rng=np.random.default_rng(0);rows=[]
    for g in json.loads((OUT/'games.json').read_text()):
        e=Minesweeper(g['seed'],g['size'],g['mines'],zero_start=g['zero']);marked=np.zeros(g['size']**2,bool);steps=safe=0
        while not e.done:
            while True:
                cells,sh=sniffs(e.visible,g['size'],marked);new=cells[(np.nan_to_num(sh,nan=0.)>=1).any(1)&~marked[cells]]
                if not len(new):break
                marked[new]=True
            ok=~marked[cells];cells,sh=cells[ok],sh[ok];smelled=~np.isnan(sh).all(1)
            score=np.where(smelled,np.nanmean(np.where(smelled[:,None],sh,0.),1),(g['mines']-marked.sum())/max(1,len(cells)));score=np.where((np.nan_to_num(sh,nan=1.)==0).any(1),-1.,score)
            dead=e.step(cells[rng.choice(np.nonzero(score==score.min())[0])])<0;steps+=1;safe+=not dead
        rows.append(dict(**g,won=bool(e.won),steps=steps,safe_steps=safe,cleared=round(float((e.visible>=0).sum())/(g['size']**2-g['mines']),4)))
    write_json(OUT/'hand-games.json',rows);return 'hand'

def summarise():
    rows={p.name[:-11]:json.loads(p.read_text()) for p in OUT.glob('*-games.json')};summary={}
    def interval(v):v=np.asarray(v,float);half=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v));return dict(mean=round(float(v.mean()),1),ci95=[round(float(v.mean()-half),1),round(float(v.mean()+half),1)],per_seed=[round(float(x),1) for x in v])
    def metric(arm,name,what):
        rr=[r for r in rows[arm] if r['set']==name]
        return 100*(np.mean([r['won'] for r in rr]) if what=='win' else np.mean([r['cleared'] for r in rr]) if what=='cleared' else sum(r['safe_steps'] for r in rr)/sum(r['steps'] for r in rr))
    for name in SETS:
        block={}
        for what in ('win','cleared','safe'):
            per={a:[metric(f'{a}-{s}',name,what) for s in SEEDS] for a in ARMS};cell={a:interval(v) for a,v in per.items()};cell['hand']=round(float(metric('hand',name,what)),1)
            for a,b in (('zero','old'),('zero-big','zero')):d=np.array(per[a])-np.array(per[b]);cell[f'{a} minus {b}']=dict(**interval(d),p=round(float(stats.ttest_rel(per[a],per[b]).pvalue),4))
            block[what]=cell
        block['blind_step_share']={a:round(float(np.mean([sum(r['unknown_steps'] for r in rows[f'{a}-{s}'] if r['set']==name)/sum(r['steps'] for r in rows[f'{a}-{s}'] if r['set']==name) for s in SEEDS])),3) for a in ARMS};summary[name]=block
    summary['training_steps']={a:int(np.mean([json.loads((OUT/f'{a}-{s}-extra.json').read_text())['training_steps'] for s in SEEDS])) for a in ARMS}
    per={a:[metric(f'{a}-{s}','9-zero','win')-metric(f'{a}-{s}','9-old','win') for s in SEEDS] for a in ARMS};summary['same_fly_zero_minus_old_start']={a:interval(v) for a,v in per.items()}
    write_json(OUT/'summary.json',summary);return summary

def main():
    from experiments.evaluate_difficulty_transfer import prior_layouts
    resume=(OUT/'games.json').exists() and not (OUT/'completed.json').exists()
    if not resume:
        OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
        for p in ['experiments/evaluate_fly_start.py','experiments/fly_agents.py','experiments/mushroom_body.py','experiments/fly_senses.py','experiments/test_fly.py','minesweeper.py','research/PROTOCOLO_INICIO_Y_TAMANO.md']:shutil.copy2(p,OUT/'source'/Path(p).name)
        used=prior_layouts(exclude=OUT);games=[]
        for j,(name,c) in enumerate(SETS.items()):
            seed=8_700_000_000+j*1_000_000;found=0
            while found<c['n']:
                env=Minesweeper(seed,c['size'],c['mines'],zero_start=c['zero']);key=identity(env);seed+=1
                if key in used or env.won:continue
                used.add(key);games.append(dict(seed=seed-1,size=c['size'],mines=c['mines'],zero=c['zero'],set=name,layout_hash=key));found+=1
        reserved={g['layout_hash'] for g in games}
        for a in ARMS.values():assert not any(identity(Minesweeper(10_000_000+g,a['sizes'][g%2],D.training_mines(g,a['sizes'][g%2],True),zero_start=a['zero'])) in reserved for g in range(a['games']))
        write_json(OUT/'games.json',games)
        with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
        write_json(OUT/'manifest.json',dict(arms=ARMS,sets=SETS,seeds=list(SEEDS),recipe='raw sniffs, non-lethal, variable density, 3% reflex',unit='seed',training_layouts_in_test=0))
    with Pool(9,initializer=configure,initargs=(str(OUT),)) as pool:
        for name in pool.imap_unordered(run,[(a,s) for s in SEEDS for a in ARMS]):print('done',name,flush=True)
        print('done',pool.apply(hand,(0,)),flush=True)
    print(json.dumps(summarise(),indent=1));write_json(OUT/'completed.json',dict(completed=True))

if __name__=='__main__':main()
