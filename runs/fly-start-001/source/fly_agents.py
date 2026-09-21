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

class LearnedMarker:
    """Marking as a learned action instead of an innate reflex with a hand-set threshold.

    The fly marks a tile with probability sigmoid((theta - z)/tau), z being the log-odds of safety its brain predicts from the
    most dreaded single sniff. Marking gives no heat and no sugar, so credit arrives late: every mark/no-mark decision leaves an
    eligibility trace that decays over the following steps, and the dopamine that those steps bring (sugar +1, heat -burn,
    relative to a running average) nudges the one 'gate' parameter theta. Nothing tells the fly whether a mark was right."""
    def __init__(self,theta=-8.,tau=1.,lr=.02,decay=.9,burn=3.,sham=False):self.sham,self.rate=sham,.1;self.theta,self.tau,self.lr0,self.lr,self.decay,self.burn=theta,tau,lr,lr,decay,burn;self.trace=0.;self.average=0.
    def begin(self):self.trace=0.
    def decide(self,z,rng,learn):
        if not learn:return z<self.theta
        p=sigmoid((self.theta-z)/self.tau);act=rng.random(len(z))<p;self.trace+=float((act-p).sum())/self.tau;return act
    def reward(self,mine,rng=None):
        if self.sham:   # control: same amount of heat and sugar, but unrelated to what the fly just did
            self.rate+=.002*(float(mine)-self.rate);mine=rng.random()<self.rate
        r=-self.burn if mine else 1.;self.theta+=self.lr*(r-self.average)*self.trace;self.theta=float(np.clip(self.theta,-12.,4.));self.average+=.01*(r-self.average);self.trace*=self.decay

def play(agent,seed,size,mines,rng,learn,temperature=1.5,dread=.03,sense=sniffs,lethal=True,marker=None,zero_start=False):
    env=Minesweeper(seed,size,mines,zero_start=zero_start);marked=np.zeros(size*size,bool);steps=safe=false_marks=unknown=burns=0;total_safe=size*size-mines
    if marker is not None:marker.begin()
    while not env.done:
        asked={}
        while True:
            cells,shares=sense(env.visible,size,marked,**(dict(mines=mines) if getattr(sense,'wants_counter',False) else {}));density=(mines-marked.sum())/max(1,len(cells)-marked[cells].sum());handle,v,worst=agent.assess(shares,density)
            if marker is None:new=cells[(worst<dread)&~marked[cells]]
            else:
                z=np.clip(np.log(np.maximum(worst,1e-13))-np.log(np.maximum(1-worst,1e-13)),-30,30)
                # one decision per tile per smell: a tile is reconsidered only if its most dreaded sniff changed (e.g. a new mark nearby)
                fresh=np.array([not marked[c] and asked.get(int(c))!=round(float(v_),3) for c,v_ in zip(cells,z)],bool)
                for c,v_ in zip(cells[fresh],z[fresh]):asked[int(c)]=round(float(v_),3)
                new=cells[fresh][marker.decide(z[fresh],rng,learn)] if fresh.any() else cells[:0]
            if not len(new):break
            marked[new]=True;false_marks+=int((~env._mines[new]).sum())   # bookkeeping only; never shown to the agent
        free=~marked[cells]
        if not free.any():free[:]=True
        ids=np.nonzero(free)[0];vv=v[ids]
        if learn:p=np.exp((vv-vv.max())/(temperature*vv.std()+1e-9));k=ids[int(rng.choice(len(ids),p=p/p.sum()))]
        else:k=ids[int(vv.argmax())]
        unknown+=bool(np.isnan(shares[k][:8]).all());mine=env.step(cells[k])<0;steps+=1;safe+=not mine
        if learn:agent.learn(handle[k],mine,float(v[k]))
        if learn and marker is not None:marker.reward(mine,rng)
        if mine and not lethal:
            # Training wheels: the burn hurts and teaches, but the fly survives, stress-marks the tile that burned it and plays on.
            env.done=False;marked[cells[k]]=True;burns+=1
    return dict(won=bool(env.won),steps=steps,safe_steps=safe,marks=int(marked.sum()),false_marks=false_marks,unknown_steps=unknown,burns=burns,cleared=round(float((env.visible>=0).sum())/total_safe,4))


def keys_of(stimulus):
    """One hashable key per sniff: the share (old senses) or the raw triple (N,k,m). None where there is nothing to smell."""
    if stimulus.ndim==3:   # raw sniffs: skip odourless tiles outright (most of a large board)
        some=~np.isnan(stimulus[:,:,0]);return [[tuple(int(x) for x in t) for t,ok in zip(row,m) if ok] if m.any() else [] for row,m in zip(stimulus,some)]
    if stimulus.ndim==2:return [[None if np.isnan(v) else (round(float(v),3),) for v in row] for row in stimulus]
    return [[None if np.isnan(t[0]) else tuple(int(x) for x in t) for t in row] for row in stimulus]

class RawFlyAgent(FlyAgent):
    """Mushroom-body fly on raw (N,k,m) sniffs: one Kenyon pattern per distinct triple, valence summed over sniffs."""
    def __init__(self,mb,far_field=False):super().__init__(mb,far_field);self.rows=[]
    def assess(self,stimulus,density):
        keys=keys_of(stimulus)
        for row in keys:
            for key in row:
                if key is not None and key not in self.index:
                    self.index[key]=len(self.index);self.rows.append(self.mb.kenyon(np.array([key],np.float32))[0])
        # Only the smells present on this board matter: cost must not grow with everything the fly has ever smelled.
        present=sorted({key for row in keys for key in row if key is not None},key=self.index.get);local={key:j for j,key in enumerate(present)}
        table=np.stack([self.rows[self.index[key]] for key in present]) if present else np.zeros((0,self.mb.kc_mbon.shape[1]),np.float32);counts=np.zeros((len(keys),len(present)),np.float32)
        for i,row in enumerate(keys):
            for key in row:
                if key is not None:counts[i,local[key]]+=1
        single=self.mb.valence(table) if present else np.zeros(0);near=np.array([len(k)<4 or k[3]==0 for k in present],bool)   # the far field never triggers the stress reflex
        worst=np.where((counts>0)&near[None],single[None],np.inf).min(1) if present else np.full(len(keys),np.inf)
        h=counts@table;return h,self.mb.valence(h),sigmoid(self.mb.beta*worst+self.mb.baseline)

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
        near=np.array([len(k)<4 or k[3]==0 for k in self.index],bool)
        worst=np.where((counts>0)&near[None],self.w[None],np.inf).min(1) if len(self.w) else np.full(len(keys),np.inf);return counts,counts@self.w,sigmoid(worst+self.b)

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
