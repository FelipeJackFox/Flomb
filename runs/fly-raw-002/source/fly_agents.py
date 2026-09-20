"""Interchangeable players for the sniff-and-mark arena: mushroom-body fly, a plain logistic learner, and ensembles."""
import numpy as np
from minesweeper import Minesweeper
from experiments.fly_senses import sniffs

def sigmoid(x):return 1/(1+np.exp(-x))

class FlyAgent:
    def __init__(self,mb,far_field=False):
        self.mb,self.far_field=mb,far_field;self.index={};self.table=np.zeros((0,mb.kc_mbon.shape[1]),np.float32);self.far={}
    def assess(self,shares,density):
        values=np.round(np.nan_to_num(shares,nan=-1.),3);counts=np.zeros((len(shares),len(self.index)+64),np.float32)
        for v in np.unique(values):
            if v<0:continue
            if float(v) not in self.index:
                self.index[float(v)]=len(self.index);self.table=np.vstack([self.table,self.mb.kenyon(np.array([[v,1.]],np.float32))])
            counts[:,self.index[float(v)]]=(values==v).sum(1)
        counts=counts[:,:len(self.table)];single=self.mb.valence(self.table);worst=np.where(counts>0,single[None],np.inf).min(1);h=counts@self.table
        if self.far_field:
            # Far from any open tile the fly still smells the arena-wide danger odour (mines left / tiles left), without 'open' odour.
            key=round(float(density),2)
            if key not in self.far:self.far[key]=self.mb.kenyon(np.array([[max(key,.01),0.]],np.float32))[0]
            h[~(counts>0).any(1)]=self.far[key]
        return h,self.mb.valence(h),sigmoid(self.mb.beta*worst+self.mb.baseline)
    def learn(self,handle,mine,value):
        if handle.any():self.mb.learn(handle,mine,value)
        else:self.mb.learn_context(mine)

class LogisticAgent:
    """Same senses, same outcomes, same reflex; no brain: one free weight per distinct sniff value plus a bias, delta rule."""
    def __init__(self,lr=.3,punish_gain=1.):self.lr0=self.eta=lr;self.punish_gain=punish_gain;self.index={};self.w=np.zeros(0);self.b=0.
    def assess(self,shares,density):
        values=np.round(np.nan_to_num(shares,nan=-1.),3)
        for v in np.unique(values):
            if v>=0 and float(v) not in self.index:self.index[float(v)]=len(self.index);self.w=np.append(self.w,0.)
        counts=np.zeros((len(shares),len(self.w)))
        for v,i in self.index.items():counts[:,i]=(values==v).sum(1)
        worst=np.where(counts>0,self.w[None],np.inf).min(1);return counts,counts@self.w,sigmoid(worst+self.b)
    def learn(self,handle,mine,value):
        error=(0. if mine else 1.)-sigmoid(value+self.b);self.w[:len(handle)]+=self.eta*error*(self.punish_gain if mine else 1.)*handle;self.b+=self.eta*error

class Ensemble:
    def __init__(self,members):self.members=members
    def assess(self,shares,density):
        parts=[m.assess(shares,density) for m in self.members];return None,np.mean([p[1] for p in parts],0),np.mean([p[2] for p in parts],0)

def play(agent,seed,size,mines,rng,learn,temperature=1.5,dread=.03,sense=sniffs,lethal=True):
    env=Minesweeper(seed,size,mines);marked=np.zeros(size*size,bool);steps=safe=false_marks=unknown=burns=0
    while not env.done:
        while True:
            cells,shares=sense(env.visible,size,marked);density=(mines-marked.sum())/max(1,len(cells)-marked[cells].sum());handle,v,worst=agent.assess(shares,density)
            new=cells[(worst<dread)&~marked[cells]]
            if not len(new):break
            marked[new]=True;false_marks+=int((~env._mines[new]).sum())   # bookkeeping only; never shown to the agent
        free=~marked[cells]
        if not free.any():free[:]=True
        ids=np.nonzero(free)[0];vv=v[ids]
        if learn:p=np.exp((vv-vv.max())/(temperature*vv.std()+1e-9));k=ids[int(rng.choice(len(ids),p=p/p.sum()))]
        else:k=ids[int(vv.argmax())]
        unknown+=bool(np.isnan(shares[k]).all());mine=env.step(cells[k])<0;steps+=1;safe+=not mine
        if learn:agent.learn(handle[k],mine,float(v[k]))
        if mine and not lethal:
            # Training wheels: the burn hurts and teaches, but the fly survives, stress-marks the tile that burned it and plays on.
            env.done=False;marked[cells[k]]=True;burns+=1
    return dict(won=bool(env.won),steps=steps,safe_steps=safe,marks=int(marked.sum()),false_marks=false_marks,unknown_steps=unknown,burns=burns)


def keys_of(stimulus):
    """One hashable key per sniff: the share (old senses) or the raw triple (N,k,m). None where there is nothing to smell."""
    if stimulus.ndim==2:return [[None if np.isnan(v) else (round(float(v),3),) for v in row] for row in stimulus]
    return [[None if np.isnan(t[0]) else tuple(int(x) for x in t) for t in row] for row in stimulus]

class RawFlyAgent(FlyAgent):
    """Mushroom-body fly on raw (N,k,m) sniffs: one Kenyon pattern per distinct triple, valence summed over sniffs."""
    def assess(self,stimulus,density):
        keys=keys_of(stimulus)
        for row in keys:
            for key in row:
                if key is not None and key not in self.index:
                    self.index[key]=len(self.index);self.table=np.vstack([self.table,self.mb.kenyon(np.array([key],np.float32))])
        counts=np.zeros((len(keys),len(self.table)),np.float32)
        for i,row in enumerate(keys):
            for key in row:
                if key is not None:counts[i,self.index[key]]+=1
        single=self.mb.valence(self.table) if len(self.table) else np.zeros(0);worst=np.where(counts>0,single[None],np.inf).min(1) if len(self.table) else np.full(len(keys),np.inf)
        h=counts@self.table;return h,self.mb.valence(h),sigmoid(self.mb.beta*worst+self.mb.baseline)

class TableAgent(LogisticAgent):
    """No brain: one free weight per distinct sniff key (exact conjunction of N,k,m) plus a bias."""
    def assess(self,stimulus,density):
        keys=keys_of(stimulus)
        for row in keys:
            for key in row:
                if key is not None and key not in self.index:self.index[key]=len(self.index);self.w=np.append(self.w,0.)
        counts=np.zeros((len(keys),len(self.w)))
        for i,row in enumerate(keys):
            for key in row:
                if key is not None:counts[i,self.index[key]]+=1
        worst=np.where(counts>0,self.w[None],np.inf).min(1) if len(self.w) else np.full(len(keys),np.inf);return counts,counts@self.w,sigmoid(worst+self.b)

class AdditiveAgent(LogisticAgent):
    """No brain and no conjunctions: separate one-hot weights for N, for k and for m; a sniff is worth a[N]+b[k]+c[m]."""
    def __init__(self,lr=.3,punish_gain=1.):super().__init__(lr,punish_gain);self.w=np.zeros(27)
    def assess(self,stimulus,density):
        keys=keys_of(stimulus);counts=np.zeros((len(keys),27));worst=np.full(len(keys),np.inf)
        for i,row in enumerate(keys):
            for key in row:
                if key is not None:
                    ix=[min(key[0],8),9+min(key[1],8),18+min(key[2],8)];counts[i,ix]+=1;worst[i]=min(worst[i],self.w[ix].sum())
        return counts,counts@self.w,sigmoid(worst+self.b)
