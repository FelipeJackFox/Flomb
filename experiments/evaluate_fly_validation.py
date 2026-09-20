"""Seed-as-unit validation of the sniff-and-mark fly. See research/PROTOCOLO_MOSCA_VALIDACION.md."""
import json,pickle,shutil,sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats
from minesweeper import Minesweeper
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity

OUT=Path('runs/fly-marker-002');GAMES=10000;CONFIG=dict(rule='rpe',sparsity=.2,tuning=.35,beta=100.);SEEDS=range(10);EXTRA={};DREAD=.03
def arms():return [(f'real-{s}',dict(seed=s)) for s in SEEDS]+[(f'shuffled-{s}',dict(seed=s,shuffled=True)) for s in SEEDS]+[(f'logistic-{s}',dict(seed=s,logistic=True)) for s in SEEDS]+[('naive',dict(seed=0,games=0))]

def build(options):
    from experiments.mushroom_body import MushroomBody
    from experiments.fly_agents import FlyAgent,LogisticAgent
    options=dict(options);options.pop('games',None);far=options.pop('far_field',False)
    if options.pop('logistic',False):return LogisticAgent(punish_gain=EXTRA.get('punish_gain',1.)),None
    mb=MushroomBody(**CONFIG,**EXTRA,**options);return FlyAgent(mb,far),mb

def train(agent,mb,seed,n):
    from experiments.fly_agents import play
    rng=np.random.default_rng(seed+1);eta0=mb.eta if mb is not None else agent.lr0
    for g in range(n):
        eta=eta0/(1+g/3000)
        if mb is not None:mb.eta=eta
        else:agent.eta=eta
        play(agent,10_000_000+g,[7,9][g%2],[7,12][g%2],rng,True,dread=DREAD)
    return rng

def evaluate(agent,games,rng):
    from experiments.fly_agents import play
    return [dict(**g,automatic=bool(Minesweeper(g['seed'],g['size'],g['mines']).won),**play(agent,g['seed'],g['size'],g['mines'],rng,False,dread=DREAD)) for g in games]

def run(arm):
    name,options=arm;games=json.loads((OUT/'games.json').read_text());agent,mb=build(options);rng=train(agent,mb,options['seed'],options.get('games',GAMES))
    write_json(OUT/f'{name}-games.json',evaluate(agent,games,rng))
    if mb is not None:
        np.save(OUT/f'weights-{name}.npy',mb.w);write_json(OUT/f'{name}-extra.json',dict(context=float(mb.baseline),valence_by_share={str(s):float(mb.valence(mb.kenyon(np.array([[s,1.]],np.float32)))[0]) for s in (0.,.125,.25,.333,.5,.667,1.)}))
    else:write_json(OUT/f'{name}-extra.json',dict(context=float(agent.b),valence_by_share={str(k):float(agent.w[i]) for k,i in sorted(agent.index.items())}))
    return name

def ensemble(item):
    from experiments.fly_agents import Ensemble
    name,members=item;games=json.loads((OUT/'games.json').read_text());agents=[]
    for m in members:
        agent,mb=build(dict(seed=int(m.split('-')[1]),shuffled=m.startswith('shuffled')));mb.w=np.load(OUT/f'weights-{m}.npy');mb.baseline=json.loads((OUT/f'{m}-extra.json').read_text())['context'];agents.append(agent)
    write_json(OUT/f'{name}-games.json',evaluate(Ensemble(agents),games,np.random.default_rng(0)));return name

def hand(_):
    from experiments.evaluate_fly_marker import hand as policy
    write_json(OUT/'hand-policy-games.json',policy(json.loads((OUT/'games.json').read_text())));return 'hand-policy'

def summarise():
    rows={p.name[:-11]:json.loads(p.read_text()) for p in OUT.glob('*-games.json')};summary={}
    def interval(v):v=np.asarray(v,float);half=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v));return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),ci95=[float(v.mean()-half),float(v.mean()+half)],per_seed=[round(float(x),1) for x in v])
    for size in (9,7):
        pct={a:100*np.mean([r['won'] for r in rr if r['size']==size and not r['automatic']]) for a,rr in rows.items()};groups={k:[pct[f'{k}-{s}'] for s in SEEDS] for k in ('real','shuffled','logistic') if f'{k}-0' in pct};block={k:interval(v) for k,v in groups.items()}
        if 'shuffled' in groups:d=np.array(groups['real'])-np.array(groups['shuffled']);t=stats.ttest_rel(groups['real'],groups['shuffled']);block['real_minus_shuffled_paired']=dict(**interval(d),t=float(t.statistic),p=float(t.pvalue))
        if 'logistic' in groups:t=stats.ttest_ind(groups['real'],groups['logistic'],equal_var=False);block['real_minus_logistic']=dict(diff=float(np.mean(groups['real'])-np.mean(groups['logistic'])),t=float(t.statistic),p=float(t.pvalue))
        block['single']={a:round(float(v),1) for a,v in pct.items() if '-' not in a or a.startswith(('ens','hand'))}
        won={a:np.array([r['won'] for r in rr if r['size']==size and not r['automatic']]) for a,rr in rows.items()}
        if 'real-2' in won:R=np.stack([won[f'real-{s}'] for s in range(3)]);block['oracle_union_of_real_0_1_2']=float(R.any(0).mean()*100);block['all_of_real_0_1_2']=float(R.all(0).mean()*100)
        block['unknown_step_share']={k:round(float(np.mean([sum(r['unknown_steps'] for r in rows[f'{k}-{s}'] if r['size']==size)/sum(r['steps'] for r in rows[f'{k}-{s}'] if r['size']==size) for s in SEEDS])),3) for k in groups}
        block['false_marks']={k:int(sum(r['false_marks'] for s in SEEDS for r in rows[f'{k}-{s}'] if r['size']==size)) for k in groups};summary[str(size)]=block
    flies=[a for a in rows if a.startswith(('real-','shuffled-'))];weak=[json.loads((OUT/f'{a}-extra.json').read_text())['valence_by_share']['0.125']*100 for a in flies];wins9=[100*np.mean([r['won'] for r in rows[a] if r['size']==9]) for a in flies]
    summary['weak_clue_valence_vs_wins']=dict(n=len(flies),pearson=float(np.corrcoef(weak,wins9)[0,1]),valence=dict(zip(flies,[round(x,2) for x in weak])));write_json(OUT/'summary.json',summary);return summary

def main(members=None):
    from experiments.train_nine_dagger import reserve
    OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
    for p in ['experiments/evaluate_fly_validation.py','experiments/fly_agents.py','experiments/mushroom_body.py','experiments/fly_senses.py','experiments/test_fly.py']:shutil.copy2(p,OUT/'source'/Path(p).name)
    games=reserve(OUT)['games'];write_json(OUT/'games.json',games)
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
    reserved={g['layout_hash'] for g in games};assert not any(identity(Minesweeper(10_000_000+g,[7,9][g%2],[7,12][g%2])) in reserved for g in range(GAMES))
    write_json(OUT/'manifest.json',dict(config=CONFIG,extra=EXTRA,training_games=GAMES,seeds=list(SEEDS),unit='seed',training_layouts_in_test=0))
    with Pool(9) as pool:
        for name in pool.imap_unordered(run,arms()):print('done',name,flush=True)
        jobs=[('ens-real-3',[f'real-{s}' for s in range(3)]),('ens-real-10',[f'real-{s}' for s in SEEDS]),('ens-shuffled-10',[f'shuffled-{s}' for s in SEEDS])]
        for name in pool.imap_unordered(ensemble,jobs):print('done',name,flush=True)
        print('done',pool.apply(hand,(0,)),flush=True)
    print(json.dumps(summarise(),indent=1));write_json(OUT/'completed.json',dict(completed=True))

if __name__=='__main__':main()
