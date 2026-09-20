"""Point 4: marking as a learned action. See research/PROTOCOLO_MARCAR_APRENDIDO.md."""
import json,pickle,shutil
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats
from minesweeper import Minesweeper
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from experiments import evaluate_fly_density as D

OUT=Path('runs/fly-marking-001');GAMES=20000;SEEDS=tuple(range(10));GROUPS=('fly-learned','fly-reflex','fly-nomarks','fly-sham','table-learned','table-reflex')

def marker_for(group):
    from experiments.fly_agents import LearnedMarker
    kind=group.split('-')[1]
    if kind=='reflex':return None
    if kind=='nomarks':return LearnedMarker(theta=-12.,lr=0.)
    return LearnedMarker(theta=-8.,lr=.005,burn=3.,sham=kind=='sham')

def run(arm):
    from experiments.fly_agents import play
    from experiments.fly_senses import sniffs_raw
    group,seed=arm;name=f'{group}-{seed}';assert not (OUT/'completed.json').exists()
    if (OUT/f'{name}-games.json').exists():return name
    games=json.loads((OUT/'games.json').read_text());agent,mb,_=D.build(('fly' if group.startswith('fly') else 'table')+'-near-var',seed);marker=marker_for(group);rng=np.random.default_rng(seed+1);eta0=mb.eta if mb is not None else agent.lr0;curve=[]
    for g in range(GAMES):
        f=1/(1+g/3000);size=[7,9][g%2]
        if mb is not None:mb.eta=eta0*f
        else:agent.eta=eta0*f
        if marker is not None:marker.lr=marker.lr0*f
        play(agent,10_000_000+g,size,D.training_mines(g,size,True),rng,True,sense=sniffs_raw,lethal=False,marker=marker)
        if marker is not None and (g+1)%2000==0:curve.append(round(marker.theta,3))
    rows=[dict(**g,automatic=bool(Minesweeper(g['seed'],g['size'],g['mines']).won),**play(agent,g['seed'],g['size'],g['mines'],rng,False,sense=sniffs_raw,marker=marker)) for g in games]
    write_json(OUT/f'{name}-games.json',rows);write_json(OUT/f'{name}-extra.json',dict(theta=None if marker is None else marker.theta,theta_curve=curve));return name

def configure(out):
    global OUT;OUT=Path(out)

def summarise():
    rows={p.name[:-11]:json.loads(p.read_text()) for p in OUT.glob('*-games.json')};summary={}
    def interval(v):v=np.asarray(v,float);half=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v));return dict(mean=round(float(v.mean()),1),sd=round(float(v.std(ddof=1)),1),ci95=[round(float(v.mean()-half),1),round(float(v.mean()+half),1)],per_seed=[round(float(x),1) for x in v])
    for size in (9,7):
        per={g:[100*np.mean([r['won'] for r in rows[f'{g}-{s}'] if r['size']==size and not r['automatic']]) for s in SEEDS] for g in GROUPS};block={g:interval(v) for g,v in per.items()}
        for a,b in (('fly-learned','fly-nomarks'),('fly-learned','fly-reflex'),('fly-learned','fly-sham'),('table-learned','table-reflex'),('fly-learned','table-learned')):
            d=np.array(per[a])-np.array(per[b]);block[f'{a} minus {b}']=dict(**interval(d),p=round(float(stats.ttest_rel(per[a],per[b]).pvalue),4))
        block['marks_per_game']={g:round(float(np.mean([r['marks'] for s in SEEDS for r in rows[f'{g}-{s}'] if r['size']==size])),2) for g in GROUPS}
        block['never_marks']={g:int(sum(np.mean([r['marks'] for r in rows[f'{g}-{s}'] if r['size']==size])<.5 for s in SEEDS)) for g in GROUPS}
        block['false_marks']={g:int(sum(r['false_marks'] for s in SEEDS for r in rows[f'{g}-{s}'] if r['size']==size)) for g in GROUPS}
        block['hand-policy']=round(float(100*np.mean([r['won'] for r in rows['hand-policy'] if r['size']==size])),1);summary[str(size)]=block
    thetas={g:[json.loads((OUT/f'{g}-{s}-extra.json').read_text()) for s in SEEDS] for g in GROUPS if 'reflex' not in g and 'nomarks' not in g}
    summary['theta']={g:dict(final=[round(x['theta'],2) for x in v],mean_curve=np.mean([x['theta_curve'] for x in v],0).round(2).tolist()) for g,v in thetas.items()};summary['reflex_theta_equivalent']=round(float(np.log(.03/.97)),2)
    write_json(OUT/'summary.json',summary);return summary

def hand(_):
    from experiments.evaluate_fly_marker import hand as policy
    write_json(OUT/'hand-policy-games.json',policy(json.loads((OUT/'games.json').read_text())));return 'hand-policy'

def main():
    from experiments.train_nine_dagger import reserve
    resume=(OUT/'games.json').exists() and not (OUT/'completed.json').exists()
    if not resume:
        OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
        for p in ['experiments/evaluate_fly_marking.py','experiments/fly_agents.py','experiments/mushroom_body.py','experiments/fly_senses.py','experiments/test_fly.py','research/PROTOCOLO_MARCAR_APRENDIDO.md']:shutil.copy2(p,OUT/'source'/Path(p).name)
        games=reserve(OUT)['games'];write_json(OUT/'games.json',games)
        with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
        reserved={g['layout_hash'] for g in games};assert not any(identity(Minesweeper(10_000_000+g,[7,9][g%2],D.training_mines(g,[7,9][g%2],True))) in reserved for g in range(GAMES))
        write_json(OUT/'manifest.json',dict(config=D.CONFIG,training='non-lethal, variable density',training_games=GAMES,seeds=list(SEEDS),groups=GROUPS,marker=dict(theta0=-8,tau=1,lr=.005,decay=.9,burn=3),unit='seed'))
    with Pool(9,initializer=configure,initargs=(str(OUT),)) as pool:
        for name in pool.imap_unordered(run,[(g,s) for s in SEEDS for g in GROUPS]):print('done',name,flush=True)
        print('done',pool.apply(hand,(0,)),flush=True)
    print(json.dumps(summarise(),indent=1));write_json(OUT/'completed.json',dict(completed=True))

if __name__=='__main__':main()
