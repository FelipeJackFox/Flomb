"""Bounded full-brain memorization and held-out safe-action diagnostic."""
import json,pickle,time,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from minesweeper import Minesweeper
from solver import analyze
from experiments.backbone import Backbone,encode_visible
from experiments.qr_core import DuelingQuantileHead
from experiments.train import visible,teacher,supervised,public_context

def dataset(start):
    rows=[]
    for size,mines in [(5,3),(7,7)]:
        seed=start+size*10000
        while sum(r['size']==size for r in rows)<50:
            env=Minesweeper(seed,size,mines);seed+=1
            while not env.done:
                info=analyze(env.observation(),env.legal_mask(),size,mines)
                labels,_=teacher(env,info)
                if info.safe:
                    rows.append({'board':visible(env),'labels':labels,'size':size,'mines':mines,'seed':seed-1});break
                env.step(min(info.mine_probability,key=info.mine_probability.get))
    return rows

def main():
    out=Path('runs/learning-diagnostic-001');out.mkdir(exist_ok=False)
    source=Path('runs/hybrid-001/checkpoint.pkl');blob=source.read_bytes();(out/'parent-checkpoint.pkl').write_bytes(blob);state=pickle.loads(blob)
    train=dataset(3_000_000_000);test=dataset(3_100_000_000)
    np.savez_compressed(out/'positions.npz',train_boards=np.array([r['board'] for r in train]),train_labels=np.array([r['labels'] for r in train]),train_seeds=[r['seed'] for r in train],test_boards=np.array([r['board'] for r in test]),test_labels=np.array([r['labels'] for r in test]),test_seeds=[r['seed'] for r in test])
    brain=Backbone(sparse.load_npz('data/processed/graph.npz'),'runs/curriculum-001/initial-expanded.npz',workers=4);brain.load_state_dict(state['brain'])
    head=DuelingQuantileHead(features=1026,actions=256,quantiles=16);head.load_state_dict(state['head'])
    # Preserve learned weights, start fresh supervised optimizers to isolate fitting.
    brain.m.fill(0);brain.v.fill(0);brain.updates=0
    for values in (head.m,head.v):
        for a in values:a.fill(0)
    head.step=0
    initial_gain=brain.log_gain.copy();rng=np.random.default_rng(20260917)
    def batch(rows):
        x,mask=encode_visible([r['board'] for r in rows]);context=np.array([public_context(r['size'],r['mines']) for r in rows]);labels=np.array([r['labels'] for r in rows]);return x,mask,context,labels
    def measure(rows):
        correct=0;loss=0.;random=0.;counts={}
        for start in range(0,len(rows),16):
            chunk=rows[start:start+16];x,mask,context,labels=batch(chunk);f,_=brain.forward(x,False);z,_=head.forward(np.concatenate([f,context],axis=1));a=np.argmax(np.where(mask,z.mean(axis=2),-np.inf),axis=1)
            hits=labels[np.arange(len(a)),a];correct+=int(hits.sum());l,_=supervised(z,mask,labels,np.ones(len(a)),np.ones(len(a)));loss+=l*len(a);random+=float(np.sum(labels.sum(axis=1)/mask.sum(axis=1)))
            for row,hit in zip(chunk,hits):
                c=counts.setdefault(str(row['size']),{'correct':0,'n':0});c['correct']+=int(hit);c['n']+=1
        return {'correct':correct,'n':len(rows),'safe_action_rate':correct/len(rows),'cross_entropy':loss/len(rows),'random_expected_rate':random/len(rows),'by_size':counts}
    report={'parent_episode':state['episode'],'parent_sha256':hashlib.sha256(blob).hexdigest(),'protocol':'100 distinct-board safe positions, 50 each 5x5/3 and7x7/7; 100 unseen distinct-board positions. Learned hybrid weights, fresh supervised Adam; full connectome gains trained, no hidden locations in labels. Memorization is not game win rate.','initial_train':measure(train),'initial_holdout':measure(test),'history':[]}
    t=time.monotonic()
    for step in range(1,401):
        rows=[train[i] for i in rng.choice(len(train),16,replace=False)];x,mask,context,labels=batch(rows);f,c=brain.forward(x);z,h=head.forward(np.concatenate([f,context],axis=1));loss,dz=supervised(z,mask,labels,np.ones(len(rows)),np.ones(len(rows)));df,g=head.backward(h,dz);dg=brain.backward(c,df[:,:1024]);head.apply(g);brain.apply(dg)
        if step%50==0:
            result={'updates':step,'train':measure(train),'seconds':time.monotonic()-t};report['history'].append(result);print(json.dumps(result),flush=True)
            (out/'progress.json').write_text(json.dumps(report,indent=2))
            if result['train']['correct']>=99:break
    report.update(final_train=measure(train),final_holdout=measure(test),changed_gains=int(np.count_nonzero(brain.log_gain!=initial_gain)),seconds=time.monotonic()-t)
    with (out/'supervised-checkpoint.pkl').open('wb') as f:pickle.dump({'brain':brain.state_dict(),'head':head.state_dict()},f)
    (out/'result.json').write_text(json.dumps(report,indent=2));brain.close();print(json.dumps(report),flush=True)
if __name__=='__main__':main()
