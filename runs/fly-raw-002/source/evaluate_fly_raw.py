"""Formal evaluation of raw (N,k,m) senses. See research/PROTOCOLO_OLFATEOS_CRUDOS.md."""
import json,pickle,shutil
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats
from minesweeper import Minesweeper
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity

OUT=Path('runs/fly-raw-001');GAMES=10000;CONFIG=dict(rule='rpe',sparsity=.2,tuning=.35,beta=100.);SEEDS=range(10);GROUPS=('raw-real','raw-shuffled','table','additive','share-real')
TRAINING=dict(lethal=True)

def configure(out,games,groups):
    """Pool initializer: spawned workers re-import this module, so overrides made by a launcher must be passed explicitly."""
    global OUT,GAMES,GROUPS;OUT,GAMES,GROUPS=Path(out),games,tuple(groups)

def build(kind,seed):
    kind=kind.replace('-soft','')
    from experiments.mushroom_body import MushroomBody
    from experiments.fly_agents import FlyAgent,RawFlyAgent,TableAgent,AdditiveAgent
    from experiments.fly_senses import sniffs,sniffs_raw
    if kind=='table':return TableAgent(),None,sniffs_raw
    if kind=='additive':return AdditiveAgent(),None,sniffs_raw
    if kind=='naive-raw':kind='raw-real'
    mb=MushroomBody(**CONFIG,seed=seed,shuffled=kind.endswith('shuffled'));return (FlyAgent(mb),mb,sniffs) if kind.startswith('share') else (RawFlyAgent(mb),mb,sniffs_raw)

def run(arm):
    from experiments.fly_agents import play
    kind,seed,games_n=arm;name=f'{kind}-{seed}';assert not (OUT/'completed.json').exists(),f'refusing to write into finished run {OUT}';games=json.loads((OUT/'games.json').read_text());agent,mb,sense=build(kind,seed);rng=np.random.default_rng(seed+1);eta0=mb.eta if mb is not None else agent.lr0
    for g in range(games_n):
        eta=eta0/(1+g/3000)
        if mb is not None:mb.eta=eta
        else:agent.eta=eta
        play(agent,10_000_000+g,[7,9][g%2],[7,12][g%2],rng,True,sense=sense,lethal=not kind.endswith('-soft'))
    rows=[dict(**g,automatic=bool(Minesweeper(g['seed'],g['size'],g['mines']).won),**play(agent,g['seed'],g['size'],g['mines'],rng,False,sense=sense)) for g in games]
    write_json(OUT/f'{name}-games.json',rows)
    if mb is not None:np.save(OUT/f'weights-{name}.npy',mb.w);write_json(OUT/f'{name}-extra.json',dict(context=float(mb.baseline),stimuli=len(agent.index)))
    else:write_json(OUT/f'{name}-extra.json',dict(context=float(agent.b),weights={str(k):float(agent.w[i]) for k,i in getattr(agent,'index',{}).items()} or [float(x) for x in agent.w]))
    return name

def ensemble(_):
    from experiments.fly_agents import Ensemble,play
    from experiments.fly_senses import sniffs_raw
    games=json.loads((OUT/'games.json').read_text());members=[]
    for s in SEEDS:
        agent,mb,_=build('raw-real',s);mb.w=np.load(OUT/f'weights-{GROUPS[0]}-{s}.npy');mb.baseline=json.loads((OUT/f'{GROUPS[0]}-{s}-extra.json').read_text())['context'];members.append(agent)
    rng=np.random.default_rng(0);write_json(OUT/'ens-raw-10-games.json',[dict(**g,automatic=False,**play(Ensemble(members),g['seed'],g['size'],g['mines'],rng,False,sense=sniffs_raw)) for g in games]);return 'ens-raw-10'

def hand(_):
    from experiments.evaluate_fly_marker import hand as policy
    write_json(OUT/'hand-policy-games.json',policy(json.loads((OUT/'games.json').read_text())));return 'hand-policy'

def summarise():
    rows={p.name[:-11]:json.loads(p.read_text()) for p in OUT.glob('*-games.json')};summary={}
    def interval(v):v=np.asarray(v,float);half=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v));return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),ci95=[float(v.mean()-half),float(v.mean()+half)],per_seed=[round(float(x),1) for x in v])
    for size in (9,7):
        pct={a:100*np.mean([r['won'] for r in rr if r['size']==size and not r['automatic']]) for a,rr in rows.items()};groups={k:[pct[f'{k}-{s}'] for s in SEEDS] for k in GROUPS};block={k:interval(v) for k,v in groups.items()}
        for other in GROUPS[1:]:
            d=np.array(groups[GROUPS[0]])-np.array(groups[other]);t=stats.ttest_rel(groups[GROUPS[0]],groups[other]);block[f'{GROUPS[0]}_minus_{other}_paired_by_seed']=dict(**interval(d),t=float(t.statistic),p=float(t.pvalue))
        block['single']={a:round(float(v),1) for a,v in pct.items() if a in ('naive-raw-0','hand-policy','ens-raw-10')}
        block['never_marks']={k:int(sum(np.mean([r['marks'] for r in rows[f'{k}-{s}'] if r['size']==size])<.5 for s in SEEDS)) for k in GROUPS}
        block['marks_per_game']={k:round(float(np.mean([r['marks'] for s in SEEDS for r in rows[f'{k}-{s}'] if r['size']==size])),2) for k in GROUPS}
        block['false_marks']={k:int(sum(r['false_marks'] for s in SEEDS for r in rows[f'{k}-{s}'] if r['size']==size)) for k in GROUPS};summary[str(size)]=block
    write_json(OUT/'summary.json',summary);return summary

def main():
    from experiments.train_nine_dagger import reserve
    OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
    for p in ['experiments/evaluate_fly_raw.py','experiments/fly_agents.py','experiments/mushroom_body.py','experiments/fly_senses.py','experiments/test_fly.py','research/PROTOCOLO_OLFATEOS_CRUDOS.md']:shutil.copy2(p,OUT/'source'/Path(p).name)
    games=reserve(OUT)['games'];write_json(OUT/'games.json',games)
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
    reserved={g['layout_hash'] for g in games};assert not any(identity(Minesweeper(10_000_000+g,[7,9][g%2],[7,12][g%2])) in reserved for g in range(GAMES))
    write_json(OUT/'manifest.json',dict(config=CONFIG,training=TRAINING,training_games=GAMES,seeds=list(SEEDS),groups=GROUPS,unit='seed'))
    with Pool(9,initializer=configure,initargs=(str(OUT),GAMES,GROUPS)) as pool:
        for name in pool.imap_unordered(run,[(k,s,GAMES) for s in SEEDS for k in GROUPS]+[('naive-raw',0,0)]):print('done',name,flush=True)
        print('done',pool.apply(ensemble,(0,)),flush=True);print('done',pool.apply(hand,(0,)),flush=True)
    print(json.dumps(summarise(),indent=1));write_json(OUT/'completed.json',dict(completed=True))

if __name__=='__main__':main()
