"""Paired raw vs fixed local 3x3 input, with identical full-brain training."""
import hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from scipy import sparse
from minesweeper import Minesweeper
from solver import analyze
from experiments.backbone import Backbone,encode_visible
from experiments.train import visible,teacher,supervised,public_context

def spatial_input(boards):
    flat,legal=encode_visible(boards);x=flat.reshape(-1,16,16,10)
    valid=(np.asarray(boards).reshape(-1,16,16)>=-1).astype(np.float32)
    padded=np.pad(x,((0,0),(1,1),(1,1),(0,0)));pv=np.pad(valid,((0,0),(1,1),(1,1)))
    neighbors=np.zeros_like(x);count=np.zeros_like(valid)
    for dr in range(3):
        for dc in range(3):
            if (dr,dc)!=(1,1):neighbors+=padded[:,dr:dr+16,dc:dc+16];count+=pv[:,dr:dr+16,dc:dc+16]
    y=(x+.5*neighbors/np.maximum(count[...,None],1))*valid[...,None]
    # Match each board's total input energy, so amplification is not the intervention.
    energy=np.sum(y*y,axis=(1,2,3));original=np.sum(x*x,axis=(1,2,3));y*=np.sqrt(original/np.maximum(energy,1e-12))[:,None,None,None]
    return y.reshape(len(x),-1),legal

def make_data(start,n,used):
    rows=[];rng=np.random.default_rng(start)
    for size,mines in [(5,3),(7,7)]:
        seed=start+size*100000;count=0
        while count<n:
            env=Minesweeper(seed,size,mines);seed+=1
            # Hidden layout is used ONLY to enforce disjoint board splits, never features/labels.
            identity=f'{size}:'+hashlib.sha256(env._mines.tobytes()).hexdigest()
            if identity in used:continue
            used.add(identity);candidates=[];move=0
            while not env.done:
                info=analyze(env.observation(),env.legal_mask(),size,mines);labels,_=teacher(env,info)
                if info.safe:candidates.append({'board':visible(env),'labels':labels,'size':size,'mines':mines,'seed':seed-1,'layout_hash':identity,'move':move})
                actions=sorted(info.safe) if info.safe else sorted(info.mine_probability,key=info.mine_probability.get)[:1]
                env.step(int(rng.choice(actions)));move+=1
            if candidates:rows.append(candidates[int(rng.integers(len(candidates)))]);count+=1
    return rows

def main():
    from experiments.qr_core import DuelingQuantileHead
    out=Path('runs/spatial-diagnostic-001');out.mkdir(exist_ok=False)
    used=set();train=make_data(3_200_000_000,128,used);holdout=make_data(3_300_000_000,64,used)
    games=[]
    for size,mines in [(5,3),(7,7)]:
        seed=3_400_000_000+size*100000
        while sum(g['size']==size for g in games)<64:
            env=Minesweeper(seed,size,mines);seed+=1;identity=f'{size}:'+hashlib.sha256(env._mines.tobytes()).hexdigest()
            if identity in used:continue
            used.add(identity);games.append({'seed':seed-1,'size':size,'mines':mines,'layout_hash':identity})
    with (out/'dataset.pkl').open('wb') as f:pickle.dump({'train':train,'holdout':holdout,'games':games},f)
    init='runs/curriculum-001/initial-expanded.npz'
    manifest={'updates':200,'batch':16,'seeds':[20260920,20260921],'train_positions':len(train),'heldout_positions':len(holdout),'game_boards':len(games),'initializer':'common fixed input/output maps; log gains ZERO and heads freshly seeded; no pretrained weights','local_encoder':'fixed residual 3x3 neighbor category averages, weight0.5, per-board energy matched; no new trainable parameters','paused_hybrid_sha256':hashlib.sha256(Path('runs/hybrid-001/checkpoint.pkl').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    brain=Backbone(sparse.load_npz('data/processed/graph.npz'),init,workers=4);results=[]
    def arrays(rows,variant):
        x,mask=(encode_visible if variant=='raw' else spatial_input)([r['board'] for r in rows]);context=np.array([public_context(r['size'],r['mines']) for r in rows]);labels=np.array([r['labels'] for r in rows]);return x,mask,context,labels
    def measure(rows,variant,head):
        hits=[];loss=0.
        for i in range(0,len(rows),16):
            chunk=rows[i:i+16];x,mask,context,labels=arrays(chunk,variant);f,_=brain.forward(x,False);z,_=head.forward(np.concatenate([f,context],axis=1));a=np.argmax(np.where(mask,z.mean(axis=2),-np.inf),axis=1);hits.extend(labels[np.arange(len(a)),a].astype(int).tolist());loss+=supervised(z,mask,labels,np.ones(len(a)),np.ones(len(a)))[0]*len(a)
        return {'correct':sum(hits),'n':len(rows),'loss':loss/len(rows),'hits':hits,'by_size':{str(size):{'correct':sum(hit for row,hit in zip(rows,hits) if row['size']==size),'n':sum(row['size']==size for row in rows)} for size in (5,7)}}
    for seed in manifest['seeds']:
        for variant in (['raw','local'] if seed==manifest['seeds'][0] else ['local','raw']):
            brain.log_gain.fill(0);brain.m.fill(0);brain.v.fill(0);brain.updates=0
            head=DuelingQuantileHead(features=1026,actions=256,quantiles=16,seed=seed);rng=np.random.default_rng(seed+1)
            result={'seed':seed,'variant':variant,'before_holdout':measure(holdout,variant,head),'history':[]};t=time.monotonic()
            for step in range(1,201):
                rows=[train[i] for i in rng.choice(len(train),16,replace=False)];x,mask,context,labels=arrays(rows,variant);f,c=brain.forward(x);z,h=head.forward(np.concatenate([f,context],axis=1));loss,dz=supervised(z,mask,labels,np.ones(16),np.ones(16));df,g=head.backward(h,dz);dg=brain.backward(c,df[:,:1024]);head.apply(g);brain.apply(dg)
                if step%50==0:
                    progress={'seed':seed,'variant':variant,'updates':step,'loss':loss,'seconds':time.monotonic()-t};result['history'].append(progress);(out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
            result.update(train=measure(train,variant,head),holdout=measure(holdout,variant,head),changed_gains=int(np.count_nonzero(brain.log_gain)),training_seconds=time.monotonic()-t)
            result['games']=[]
            for game in games:
                size,mines=game['size'],game['mines'];env=Minesweeper(game['seed'],size,mines);automatic=env.won;clicks=0
                while not env.done:
                    b=visible(env);x,mask=(encode_visible if variant=='raw' else spatial_input)([b]);f,_=brain.forward(x,False);z,_=head.forward(np.concatenate([f,public_context(size,mines)[None,:]],axis=1));a=int(np.argmax(np.where(mask[0],z[0].mean(axis=1),-np.inf)));env.step((a//16)*size+a%16);clicks+=1
                result['games'].append({**game,'won':bool(env.won),'automatic':bool(automatic),'clicks':clicks})
            with (out/f'{variant}-{seed}.pkl').open('wb') as f:pickle.dump({'brain':brain.state_dict(),'head':head.state_dict()},f)
            results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2));print('FINISHED',variant,seed,result['holdout']['correct'],flush=True)
    brain.close();(out/'completed.json').write_text(json.dumps({'variants':4,'time':time.time()}))
if __name__=='__main__':main()
