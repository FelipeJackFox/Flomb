"""Front 6 pilot, no fly involved: does remembering PAIRS of clues predict mines better than single sniffs, where it matters?

Setting: a player with perfect single-point logic (marks sure mines, opens sure-safe tiles). Whenever it is forced to guess we ask a
learned guesser. Guessers see only raw, legitimate quantities:
  single : the bag of raw triples (N,k,m) of the tile's open neighbours  -> what today's fly can represent
  pairs  : single + for every open neighbour B, every other clue A within two tiles of B: (triple_B, triple_A, where A sits relative
           to B, where B sits relative to the tile), up to rotation/reflection -> a short working memory of the previous sniff plus the step taken
Yardsticks: the hand heuristic (mean danger share / density) and the exact constraint solver.
"""
import json,sys,time,zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
from minesweeper import Minesweeper

OUT=Path('benchmarks/pair-memory-pilot');BITS=20;DIRS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
SYM=[lambda r,c:(r,c),lambda r,c:(c,-r),lambda r,c:(-r,-c),lambda r,c:(-c,r),lambda r,c:(r,-c),lambda r,c:(-r,c),lambda r,c:(c,r),lambda r,c:(-c,-r)]

def closure(env):
    vis=env.visible;mines=set();safe=set();changed=True
    while changed:
        changed=False
        for j in np.nonzero(vis>0)[0]:
            nb=[q for q in env.neighbors(j) if vis[q]==-1];known=sum(q in mines for q in nb);rest=[q for q in nb if q not in mines and q not in safe]
            if rest and vis[j]-known==len(rest):mines|=set(rest);changed=True
            elif rest and vis[j]-known==0:safe|=set(rest);changed=True
    return mines,safe

def triples(env,mines):
    n=env.size;vis=env.visible.reshape(n,n);out={}
    for j in np.nonzero(env.visible>0)[0]:
        nb=env.neighbors(j);out[int(j)]=(int(env.visible[j]),sum(env.visible[q]==-1 for q in nb),sum(q in mines for q in nb))
    return out

def h(*key):return zlib.crc32(repr(key).encode())%(1<<BITS)

def features(env,mines,tile,tri):
    n=env.size;r,c=divmod(tile,n);single=[];pairs=[]
    for dr,dc in DIRS:
        rr,cc=r+dr,c+dc
        if not (0<=rr<n and 0<=cc<n):continue
        b=rr*n+cc
        if b not in tri:continue
        single.append(h('s',tri[b]))
        for ar in range(-2,3):
            for ac in range(-2,3):
                a_r,a_c=rr+ar,cc+ac
                if (ar,ac)==(0,0) or not (0<=a_r<n and 0<=a_c<n):continue
                a=a_r*n+a_c
                if a in tri:pairs.append(h('p',tri[b],tri[a],min((s(-dr,-dc),s(ar,ac)) for s in SYM)))   # tile->B and B->A, canonical under the 8 symmetries
    if not single:
        left=env.mine_count-len(mines);cov=int((env.visible==-1).sum())-len(mines);single=[h('blind',round(left/max(cov,1)*20))]
    return single,pairs

def heuristic(env,mines,cands,tri):
    score=[]
    for t in cands:
        sh=[(tri[b][0]-tri[b][2])/max(tri[b][1]-tri[b][2],1) for b in env.neighbors(t) if b in tri]
        score.append(np.mean(sh) if sh else (env.mine_count-len(mines))/max(int((env.visible==-1).sum())-len(mines),1))
    return np.array(score)

def collect(args):
    """Play with single-point logic + hand heuristic for guesses; log every forced-guess decision."""
    lo,hi,size,mine_count=args;rng=np.random.default_rng(lo);rows=[]
    for seed in range(lo,hi):
        env=Minesweeper(seed,size,mine_count,zero_start=True)
        while not env.done:
            mines,safe=closure(env)
            if safe:env.step(min(safe));continue
            cands=[int(t) for t in np.nonzero(env.visible==-1)[0] if t not in mines];tri=triples(env,mines)
            for t in cands:s,p=features(env,mines,t,tri);rows.append((s,p,int(env._mines[t])))
            score=heuristic(env,mines,cands,tri);env.step(cands[int(rng.choice(np.nonzero(score==score.min())[0]))])
    return rows

class Bag(torch.nn.Module):
    def __init__(self):super().__init__();self.e=torch.nn.EmbeddingBag(1<<BITS,1,mode='sum');torch.nn.init.zeros_(self.e.weight);self.b=torch.nn.Parameter(torch.zeros(1))
    def forward(self,idx,off):return self.e(idx,off).squeeze(1)+self.b

def pack(rows,use_pairs):
    idx=[];off=[];y=[]
    for s,p,label in rows:off.append(len(idx));idx.extend(s+(p if use_pairs else []));y.append(label)
    return torch.tensor(idx),torch.tensor(off),torch.tensor(y,dtype=torch.float32)

def fit(rows,use_pairs,epochs=6):
    torch.manual_seed(0);model=Bag();opt=torch.optim.Adam(model.parameters(),lr=.03);order=np.arange(len(rows))
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(order)
        for i in range(0,len(order),4096):
            idx,off,y=pack([rows[j] for j in order[i:i+4096]],use_pairs);opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(idx,off),y)+1e-6*model.e.weight.pow(2).sum();loss.backward();opt.step()
    return model

def play(args):
    guesser,path,lo,hi,size,mine_count=args;rng=np.random.default_rng(lo);wins=guesses=safe_guesses=0
    model=torch.load(path,weights_only=False) if path else None
    if guesser=='exact':from solver import choose
    for seed in range(lo,hi):
        env=Minesweeper(seed,size,mine_count,zero_start=True)
        while not env.done:
            if guesser=='exact':env.step(choose(env.observation(),env.legal_mask(),size,mine_count,rng)[0]);continue
            mines,safe=closure(env)
            if safe:env.step(min(safe));continue
            cands=[int(t) for t in np.nonzero(env.visible==-1)[0] if t not in mines];tri=triples(env,mines)
            if model is None:score=heuristic(env,mines,cands,tri)
            else:
                with torch.no_grad():idx,off,_=pack([(*features(env,mines,t,tri),0) for t in cands],guesser=='pairs');score=model(idx,off).numpy()
            t=cands[int(rng.choice(np.nonzero(score==score.min())[0]))];guesses+=1;safe_guesses+=not env._mines[t];env.step(t)
        wins+=env.won
    return guesser,size,wins,hi-lo,guesses,safe_guesses

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);t=time.time()
    with Pool(9) as pool:
        train=sum(pool.map(collect,[(27_000_000+i*2000,27_000_000+(i+1)*2000,9,12) for i in range(18)]+[(27_500_000+i*300,27_500_000+(i+1)*300,16,40) for i in range(9)]),[])
        print('training rows',len(train),'mine rate',round(np.mean([r[2] for r in train]),3),'mean pairs per tile',round(np.mean([len(r[1]) for r in train]),1),'seconds',round(time.time()-t),flush=True)
        test=sum(pool.map(collect,[(28_000_000+i*500,28_000_000+(i+1)*500,9,12) for i in range(9)]),[])
        results={}
        for name,use in (('single',False),('pairs',True)):
            model=fit(train,use);torch.save(model,OUT/f'{name}.pt')
            with torch.no_grad():idx,off,y=pack(test,use);z=model(idx,off)
            results[name]=dict(test_logloss=float(torch.nn.functional.binary_cross_entropy_with_logits(z,y)),distinct_keys_used=int((model.e.weight.abs()>1e-4).sum()))
            print(name,results[name],'seconds',round(time.time()-t),flush=True)
        jobs=[(g,str(OUT/f'{g}.pt') if g in ('single','pairs') else None,29_000_000+i*250,29_000_000+(i+1)*250,9,12) for g in ('heuristic','single','pairs','exact') for i in range(8)]
        jobs+=[(g,str(OUT/f'{g}.pt') if g in ('single','pairs') else None,29_500_000+i*40,29_500_000+(i+1)*40,16,40) for g in ('heuristic','single','pairs','exact') for i in range(8)]
        agg={}
        for g,size,w,n,gu,sg in pool.imap_unordered(play,jobs):
            a=agg.setdefault((g,size),[0,0,0,0]);a[0]+=w;a[1]+=n;a[2]+=gu;a[3]+=sg
    for (g,size),(w,n,gu,sg) in sorted(agg.items(),key=lambda x:(x[0][1],x[0][0])):
        results[f'{g}-{size}']=dict(win_pct=round(100*w/n,1),games=n,guesses_per_game=round(gu/n,2) if gu else None,safe_guess_pct=round(100*sg/gu,1) if gu else None);print(f'{size}x{size}',g.ljust(10),results[f'{g}-{size}'],flush=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=1));print('total seconds',round(time.time()-t))
