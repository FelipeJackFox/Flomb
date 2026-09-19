"""The fly plays: walk to a covered tile, step on it, feel heat (mine) or taste sugar (safe). Outcome-only learning."""
import json,sys,time
from pathlib import Path
import numpy as np
from minesweeper import Minesweeper
from experiments.fly_senses import windows,plume
from experiments.mushroom_body import MushroomBody

def play(mb,seed,size,mines,rng,learn,silent,temperature=.5):
    env=Minesweeper(seed,size,mines);steps=0;safe=0
    while not env.done:
        cells,odor=plume(env.visible,size) if silent=='plume' else windows(env.visible,size,silent);h=mb.kenyon(odor);v=mb.valence(h)
        if learn:p=np.exp((v-v.max())/(temperature*v.std()+1e-9));k=int(rng.choice(len(cells),p=p/p.sum()))
        else:k=int(v.argmax())
        mine=env.step(cells[k])<0;steps+=1;safe+=not mine
        if learn:mb.learn(h[k],mine,float(v[k]))
    return env.won,steps,safe

def main(name,games=30000,**config):
    silent=config.pop('silent',False);temperature=config.pop('temperature',.5);decay=config.pop('decay',0);rng=np.random.default_rng(config.get('seed',0)+1);mb=MushroomBody(**config);eta0=mb.eta;t=time.time();log=[]
    for g in range(games):
        if decay:mb.eta=eta0/(1+g/decay)
        play(mb,10_000_000+g,[7,9][g%2],[7,12][g%2],rng,True,silent,temperature)
        if (g+1)%5000==0:
            r9=[play(mb,20_000_000+i,9,12,rng,False,silent) for i in range(200)];r7=[play(mb,21_000_000+i,7,7,rng,False,silent) for i in range(200)];dev9=sum(r[0] for r in r9);dev7=sum(r[0] for r in r7);safe_rate=sum(r[2] for r in r9+r7)/sum(r[1] for r in r9+r7)
            rec=dict(name=name,games=g+1,dev9=f'{dev9}/200',dev7=f'{dev7}/200',safe_rate=round(safe_rate,3),w_mean=float(mb.w[mb.exists].mean()),w_min=float(mb.w[mb.exists].min()),seconds=round(time.time()-t));log.append(rec);print(rec,flush=True)
    return mb,log

if __name__=='__main__':
    name=sys.argv[1];config=json.loads(sys.argv[2]) if len(sys.argv)>2 else {};main(name,**config)
