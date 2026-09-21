"""Train one legitimate fly (raw sniffs, non-lethal, variable density) and export decision-by-decision replays for scene/dist/arena.html."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from minesweeper import Minesweeper
from experiments import evaluate_fly_density as E
from experiments.fly_agents import play,sigmoid
from experiments.fly_senses import sniffs_raw

SHIFTS=[(dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0)];GAMES=20000;SEED=0;DREAD=.03
W=ROOT/'runs/fly-viz/weights.npy'

def trained():
    agent,mb,_=E.build('fly-near-var',SEED)
    if W.exists():mb.w=np.load(W);mb.baseline=float(json.loads((W.parent/'context.json').read_text())['context']);return agent,mb
    rng=np.random.default_rng(SEED+1);eta0=mb.eta
    for g in range(GAMES):
        mb.eta=eta0/(1+g/3000);size=[7,9][g%2];play(agent,10_000_000+g,size,E.training_mines(g,size,True),rng,True,sense=sniffs_raw,lethal=False)
        if (g+1)%2000==0:print('trained',g+1,flush=True)
    W.parent.mkdir(parents=True,exist_ok=True);np.save(W,mb.w);(W.parent/'context.json').write_text(json.dumps(dict(context=float(mb.baseline))));return agent,mb

def sniff_list(agent,mb,raw_row,cell,size):
    r,c=divmod(int(cell),size);out=[]
    for (dr,dc),t in zip(SHIFTS,raw_row):
        if np.isnan(t[0]):continue
        key=tuple(int(x) for x in t);v=float(mb.valence(agent.rows[agent.index[key]][None])[0])
        out.append(dict(src=(r-dr)*size+(c-dc),N=key[0],k=key[1],m=key[2],v=round(v*100,2),safe=round(float(sigmoid(mb.beta*v+mb.baseline)),4)))
    return out

def record(agent,mb,seed,size,mines):
    env=Minesweeper(seed,size,mines,zero_start=True);marked=np.zeros(size*size,bool);steps=[];layout=env._mines.astype(int).tolist()   # fair start: opens on a cascade
    first=dict(visible=env.visible.astype(int).tolist())
    while not env.done:
        before=env.visible.astype(int).tolist();marks=[]
        while True:
            cells,raw=sniffs_raw(env.visible,size,marked);h,v,worst=agent.assess(raw,0.);new=np.nonzero((worst<DREAD)&~marked[cells])[0]
            if not len(new):break
            for i in new:
                sn=sniff_list(agent,mb,raw[i],cells[i],size);marks.append(dict(cell=int(cells[i]),trigger=min(sn,key=lambda s:s['v']),correct=bool(env._mines[cells[i]])))
            marked[cells[new]]=True
        free=np.nonzero(~marked[cells])[0]
        if not len(free):free=np.arange(len(cells))
        k=free[int(v[free].argmax())];cell=int(cells[k]);hk=h[k];mbon=((hk@(mb.kc_mbon*(mb.w-1)).T)*mb.sign*100)
        active=np.argsort(hk)[::-1][:160];active=active[hk[active]>0]
        step=dict(before=before,marks=marks,cell=cell,sniffs=sniff_list(agent,mb,raw[k],cell,size),valence=round(float(v[k])*100,2),blind=bool(np.isnan(raw[k][:,0]).all()),
            field={int(c):round(float(x)*100,2) for c,x in zip(cells,v)},kc=[int(i) for i in active],kc_total=int((hk>0).sum()),mbon=[round(float(x),3) for x in mbon])
        mine=env.step(cell)<0;step.update(mine=bool(mine),after=env.visible.astype(int).tolist(),flags=np.nonzero(marked)[0].astype(int).tolist());steps.append(step)
    return dict(seed=int(seed),size=size,mines=mines,won=bool(env.won),layout=layout,first=first,steps=steps)

if __name__=='__main__':
    agent,mb=trained();games=[];wins=losses=0
    for seed in range(23_000_000,23_000_400):
        g=record(agent,mb,seed,9,12)
        if len(g['steps'])<6:continue
        if g['won'] and wins<7:games.append(g);wins+=1
        elif not g['won'] and losses<5 and len(g['steps'])>=8:games.append(g);losses+=1
        if wins==7 and losses==5:break
    rate=np.mean([record(agent,mb,s,9,12)['won'] for s in range(23_100_000,23_100_200)])
    meta=dict(title='Mosca con olfateos crudos (N, k, m)',training='20,000 partidas, solo calor y azúcar, entrenamiento no letal, densidad variable; partidas con inicio en cascada',win_rate_dev=round(float(rate)*100,1),
        mbon_sign=[int(x) for x in mb.sign],mbon_types=[str(t) for t in mb.mbon_types],kc_count=int(mb.kc_mbon.shape[1]),dread=DREAD,context=round(float(mb.baseline),2))
    out=ROOT/'scene/dist/data/arena.json';out.write_text(json.dumps(dict(meta=meta,games=games),separators=(',',':')));print('games',len(games),'wins',wins,'losses',losses,'dev win rate',meta['win_rate_dev'],'bytes',out.stat().st_size)
