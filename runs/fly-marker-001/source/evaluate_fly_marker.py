"""Formal evaluation of the sniff-and-mark fly on reserved layouts. See research/PROTOCOLO_MOSCA_MARCADORA.md."""
import json,pickle,shutil,sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from minesweeper import Minesweeper
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity

OUT=Path('runs/fly-marker-001');GAMES=10000;CONFIG=dict(rule='rpe',sparsity=.2,tuning=.35,beta=100.)
ARMS=[(f'real-{s}',dict(seed=s)) for s in range(3)]+[(f'shuffled-{s}',dict(seed=s,shuffled=True)) for s in range(3)]+[('naive',dict(seed=0,games=0)),('no-marks',dict(seed=0,dread=0.)),('hand-policy',None)]

def hand(games):
    from experiments.fly_senses import sniffs
    rng=np.random.default_rng(0);rows=[]
    for g in games:
        e=Minesweeper(g['seed'],g['size'],g['mines']);auto=bool(e.won);marked=np.zeros(g['size']**2,bool)
        while not e.done:
            while True:
                cells,sh=sniffs(e.visible,g['size'],marked);new=cells[(np.nan_to_num(sh,nan=0.)>=1).any(1)&~marked[cells]]
                if not len(new):break
                marked[new]=True
            ok=~marked[cells];cells,sh=cells[ok],sh[ok];smelled=~np.isnan(sh).all(1)
            score=np.where(smelled,np.nanmean(np.where(smelled[:,None],sh,0.),1),(g['mines']-marked.sum())/max(1,len(cells)));score=np.where((np.nan_to_num(sh,nan=1.)==0).any(1),-1.,score)
            e.step(cells[rng.choice(np.nonzero(score==score.min())[0])])
        rows.append(dict(**g,automatic=auto,won=bool(e.won)))
    return rows

def run(arm):
    name,options=arm;games=json.loads((OUT/'games.json').read_text())
    if options is None:rows=hand(games);extra={}
    else:
        from experiments.mushroom_body import MushroomBody
        from experiments.train_fly_marker import Nose,play
        options=dict(options);n=options.pop('games',GAMES);dread=options.pop('dread',.03);mb=MushroomBody(**CONFIG,**options);nose=Nose(mb);rng=np.random.default_rng(options['seed']+1);eta0=mb.eta
        for g in range(n):mb.eta=eta0/(1+g/3000);play(mb,nose,10_000_000+g,[7,9][g%2],[7,12][g%2],rng,True,1.5,dread)
        rows=[]
        for g in games:
            auto=bool(Minesweeper(g['seed'],g['size'],g['mines']).won);won,steps,safe,false_marks,marks=play(mb,nose,g['seed'],g['size'],g['mines'],rng,False,dread=dread)
            rows.append(dict(**g,automatic=auto,won=bool(won),steps=steps,safe_steps=safe,marks=marks,false_marks=false_marks))
        extra=dict(valence_by_share={str(s):float(mb.valence(mb.kenyon(np.array([[s,1.]],np.float32)))[0]) for s in (0.,.125,.25,.333,.5,.667,1.)},context=float(mb.baseline),
            synapses_changed=int(((mb.w!=1)&mb.exists).sum()),synapses_total=int(mb.exists.sum()));np.save(OUT/f'weights-{name}.npy',mb.w)
    write_json(OUT/f'{name}-games.json',rows);write_json(OUT/f'{name}-extra.json',extra);return name

def main():
    from experiments.train_nine_dagger import reserve
    OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
    for p in ['experiments/evaluate_fly_marker.py','experiments/train_fly_marker.py','experiments/mushroom_body.py','experiments/fly_senses.py','experiments/prepare_mushroom_body.py','research/PROTOCOLO_MOSCA_MARCADORA.md']:shutil.copy2(p,OUT/'source'/Path(p).name)
    games=reserve(OUT)['games'];write_json(OUT/'games.json',games)
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
    reserved={g['layout_hash'] for g in games};overlap=sum(identity(Minesweeper(10_000_000+g,[7,9][g%2],[7,12][g%2])) in reserved for g in range(GAMES));assert overlap==0
    write_json(OUT/'manifest.json',dict(config=CONFIG,training_games=GAMES,arms=[a for a,_ in ARMS],training_layouts_in_test=overlap,protocol='research/PROTOCOLO_MOSCA_MARCADORA.md'))
    with Pool(9) as pool:
        for name in pool.imap_unordered(run,ARMS):print('done',name,flush=True)
    rows={a:json.loads((OUT/f'{a}-games.json').read_text()) for a,_ in ARMS};summary={}
    for size in (9,7):
        played={a:[r for r in rr if r['size']==size and not r['automatic']] for a,rr in rows.items()};wins={a:np.array([int(r['won']) for r in rr]) for a,rr in played.items()};rng=np.random.default_rng(20260919+size);n=len(wins['naive'])
        def ci(v):boot=[rng.choice(v,len(v),replace=True).mean()*100 for _ in range(10000)];return dict(percent=float(v.mean()*100),ci95=np.quantile(boot,[.025,.975]).tolist())
        real=np.mean([wins[f'real-{s}'] for s in range(3)],0);shuffled=np.mean([wins[f'shuffled-{s}'] for s in range(3)],0)
        summary[str(size)]=dict(n=n,per_arm={a:f"{int(w.sum())}/{len(w)}" for a,w in wins.items()},real_mean=ci(real),shuffled_mean=ci(shuffled),real_minus_shuffled=ci(real-shuffled),naive=ci(wins['naive']),no_marks=ci(wins['no-marks']),hand_policy=ci(wins['hand-policy']),
            false_marks={a:int(sum(r.get('false_marks',0) for r in rr)) for a,rr in played.items() if a!='hand-policy'},safe_step_rate={a:round(sum(r['safe_steps'] for r in rr)/max(1,sum(r['steps'] for r in rr)),4) for a,rr in played.items() if a!='hand-policy'})
    summary['extra']={a:json.loads((OUT/f'{a}-extra.json').read_text()) for a,_ in ARMS};write_json(OUT/'summary.json',summary);write_json(OUT/'completed.json',dict(completed=True));print(json.dumps(summary,indent=1))

if __name__=='__main__':main()
