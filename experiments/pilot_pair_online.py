"""Front 6 pilot, step 2: does the pair advantage survive OUTCOME-ONLY learning (heat/sugar on the tile actually stepped on)?
Same single-point player as pilot_pair_memory; the guesser is a hashed table trained online with a delta rule, non-lethal training."""
import json,sys,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from minesweeper import Minesweeper
from experiments.pilot_pair_memory import closure,triples,features,BITS

OUT=Path('benchmarks/pair-online-pilot');CHECKPOINTS=(5000,20000,80000)

def guess_values(w,b,env,mines,cands,tri,use_pairs):
    feats=[features(env,mines,t,tri) for t in cands];keys=[np.array(s+(p if use_pairs else []),np.int64) for s,p in feats]
    # mean over pair keys keeps a tile with many pairs from shouting louder than one with few
    z=np.array([w[k[:len(f[0])]].sum()+(w[k[len(f[0]):]].mean() if use_pairs and len(f[1]) else 0.) for k,f in zip(keys,feats)])+b[0];return keys,feats,z

def episode(w,b,seed,size,mine_count,rng,use_pairs,learn,lr):
    env=Minesweeper(seed,size,mine_count,zero_start=learn is False or True);burned=set();guesses=safe_g=0
    while not env.done:
        mines,safe=closure(env);mines|=burned
        if safe-burned:env.step(min(safe-burned));continue
        cands=[int(t) for t in np.nonzero(env.visible==-1)[0] if t not in mines]
        if not cands:break
        tri=triples(env,mines);keys,feats,z=guess_values(w,b,env,mines,cands,tri,use_pairs)   # z = logit of being SAFE
        if learn:p=np.exp((z-z.max())/max(1.5*z.std(),1e-6));i=int(rng.choice(len(cands),p=p/p.sum()))
        else:i=int(rng.choice(np.nonzero(z==z.max())[0]))
        t=cands[i];mine=bool(env._mines[t]);guesses+=1;safe_g+=not mine
        if learn:
            err=(0. if mine else 1.)-1/(1+np.exp(-z[i]));ns=len(feats[i][0]);k=keys[i];w[k[:ns]]+=lr*err;b[0]+=lr*err*.1
            if use_pairs and len(k)>ns:w[k[ns:]]+=lr*err/ (len(k)-ns)**.5
            if mine:burned.add(t);continue        # non-lethal training: burned, remembered as a mine, play on
        env.step(t)
    return env.won,guesses,safe_g

def run(job):
    name,use_pairs,seed=job;rng=np.random.default_rng(seed);w=np.zeros(1<<BITS);b=[1.];log=[];t=time.time()
    for g in range(CHECKPOINTS[-1]):
        size,mc=((9,12),(16,40))[g%4==3];episode(w,b,30_000_000+seed*1_000_000+g,size,mc,rng,use_pairs,True,.3/(1+g/20000))
        if g+1 in CHECKPOINTS:
            r9=[episode(w,b,31_000_000+i,9,12,rng,use_pairs,False,0.) for i in range(1500)];r16=[episode(w,b,31_500_000+i,16,40,rng,use_pairs,False,0.) for i in range(200)]
            rec=dict(name=name,seed=seed,games=g+1,win9=round(100*np.mean([r[0] for r in r9]),1),win16=round(100*np.mean([r[0] for r in r16]),1),safe_guess9=round(100*sum(r[2] for r in r9)/max(1,sum(r[1] for r in r9)),1),keys=int((w!=0).sum()),seconds=round(time.time()-t))
            log.append(rec);print(rec,flush=True);(OUT/f'{name}-{seed}.json').write_text(json.dumps(log))
    return log

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with Pool(8) as pool:pool.map(run,[(n,u,s) for s in (0,1,2,3) for n,u in (('single',False),('pairs',True))])
