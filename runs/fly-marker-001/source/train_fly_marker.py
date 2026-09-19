"""The fly sniffs each open neighbour in turn, stress-marks tiles it has learned to dread, and steps where valence is best.

Nothing is told to the fly: heat (mine -> PPL1) and sugar (safe -> PAM) are the only teaching signals, and only KC->MBON
synapses change. Marking is an innate reaction to learned extreme aversion (predicted safety below `dread`).
"""
import json,sys,time
import numpy as np
from minesweeper import Minesweeper
from experiments.fly_senses import sniffs
from experiments.mushroom_body import MushroomBody

class Nose:
    """Caches the Kenyon pattern of each distinct sniff (open-tile odour at 1, danger odour at the clue's share)."""
    def __init__(self,mb):self.mb=mb;self.index={};self.table=np.zeros((0,mb.kc_mbon.shape[1]),np.float32)
    def patterns(self,shares):
        values=np.round(np.nan_to_num(shares,nan=-1.),3);counts=np.zeros((len(shares),len(self.index)+64),np.float32)
        for v in np.unique(values):
            if v<0:continue
            if v not in self.index:
                self.index[float(v)]=len(self.index);self.table=np.vstack([self.table,self.mb.kenyon(np.array([[v,1.]],np.float32))])
            counts[:,self.index[float(v)]]=(values==v).sum(1)
        counts=counts[:,:len(self.table)];single=self.mb.valence(self.table)
        # worst single sniff: the stress reflex is triggered by one overwhelming smell, not by many mild ones adding up
        worst=np.where(counts>0,single[None],np.inf).min(1);return counts@self.table,worst

def safety(mb,value):return 1/(1+np.exp(-(mb.beta*value+mb.baseline)))

def play(mb,nose,seed,size,mines,rng,learn,temperature=1.5,dread=.03):
    env=Minesweeper(seed,size,mines);marked=np.zeros(size*size,bool);steps=safe=0;false_marks=0
    while not env.done:
        while True:
            cells,shares=sniffs(env.visible,size,marked);h,worst=nose.patterns(shares);v=mb.valence(h)
            new=cells[(safety(mb,worst)<dread)&~marked[cells]]
            if not len(new):break
            marked[new]=True;false_marks+=int((~env._mines[new]).sum())
        free=~marked[cells]
        if not free.any():free[:]=True
        cells,h,v=cells[free],h[free],v[free]
        if learn:p=np.exp((v-v.max())/(temperature*v.std()+1e-9));k=int(rng.choice(len(cells),p=p/p.sum()))
        else:k=int(v.argmax())
        mine=env.step(cells[k])<0;steps+=1;safe+=not mine
        if learn:mb.learn(h[k],mine,float(v[k])) if h[k].any() else mb.learn_context(mine)
    return env.won,steps,safe,false_marks,int(marked.sum())

def main(name,games=30000,**config):
    temperature=config.pop('temperature',1.5);decay=config.pop('decay',3000);dread=config.pop('dread',.03);rng=np.random.default_rng(config.get('seed',0)+1)
    mb=MushroomBody(**config);nose=Nose(mb);eta0=mb.eta;t=time.time()
    for g in range(games):
        mb.eta=eta0/(1+g/decay);play(mb,nose,10_000_000+g,[7,9][g%2],[7,12][g%2],rng,True,temperature,dread)
        if (g+1)%2500==0:
            r9=[play(mb,nose,20_000_000+i,9,12,rng,False,dread=dread) for i in range(200)];r7=[play(mb,nose,21_000_000+i,7,7,rng,False,dread=dread) for i in range(200)]
            rec=dict(name=name,games=g+1,dev9=f'{sum(r[0] for r in r9)}/200',dev7=f'{sum(r[0] for r in r7)}/200',safe_rate=round(sum(r[2] for r in r9+r7)/sum(r[1] for r in r9+r7),3),marks=sum(r[4] for r in r9+r7),false_marks=sum(r[3] for r in r9+r7),
                v={s:round(float(mb.valence(mb.kenyon(np.array([[s,1.]],np.float32)))[0])*100,1) for s in (0.,.125,.25,.5,1.)},bias=round(float(mb.baseline),2),seconds=round(time.time()-t));print(rec,flush=True)
    return mb

if __name__=='__main__':main(sys.argv[1],**(json.loads(sys.argv[2]) if len(sys.argv)>2 else {}))
