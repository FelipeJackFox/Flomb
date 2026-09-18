"""Does the connectome itself learn? Frozen interface, trainable synapses, 3 propagation cycles."""
import copy,hashlib,pickle,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.train_joint_interface import BASE,digest,tensor_hash,save,validation
from experiments.train_nine_dagger import reserve,MAPPING
from experiments.capacity_probe import arrays,write_json
from experiments.expand_dagger_experience import features
from experiments.synaptic_learning import SynapticBrain
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead,activity_map
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.extend_spatial_decoder import validate
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games

OUT=Path('runs/synaptic-learning-001');PARENT=Path('runs/early-interface-001/latest-joint.pt');SEED=20261111
CYCLES=3;WARMUP=3000;UPDATES=3000;ARMS=('baseline','reader','brain')

def draw(rng,step,groups,sets):
    indices=[np.array([int(rng.choice(groups[(step*64+j)%len(groups)])) for j in range(16)],np.int64)]
    return indices+[rng.integers(len(r),size=16,dtype=np.int64) for r in sets[1:]]

def train_head(head,opt,rng,caches,groups,sets,first,last,label,blocks):
    block=hashlib.sha256()
    for step in range(first,last+1):
        indices=draw(rng,step,groups,sets)
        for ix in indices:block.update(ix.tobytes())
        a,l,c,y=[torch.cat([p[ix] for p,ix in zip(parts,indices)]) for parts in zip(*caches)]
        opt.zero_grad();loss=equivalent_loss(head(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
        if step%250==0:
            blocks.append(block.hexdigest());block=hashlib.sha256()
            rec=dict(phase='training',arm=label,step=step,loss=float(loss));write_json(OUT/'progress.json',rec);print(rec,flush=True)

def main(out=OUT,seed=SEED,workers=8):
    global OUT
    OUT=Path(out);torch.set_num_threads(1);resume=(OUT/'warmup.pt').exists()
    if not resume:OUT.mkdir(exist_ok=False)
    paths=[PARENT,MAPPING,BASE/'training/retina_plastic-20260926.pt',BASE/'dataset.pkl',Path('runs/hybrid-001/checkpoint.pkl'),Path('runs/adapted-dagger-001/dataset.pkl'),Path('runs/nine-dagger-001/dataset.pkl'),Path('runs/deep-coverage-001/states-autonomous.pkl')]
    hashes={str(p):digest(p) for p in paths}
    if resume:
        import json;assert json.loads((OUT/'manifest.json').read_text())['hashes']==hashes;games=json.loads((OUT/'games.json').read_text())
    else:
        write_json(OUT/'manifest.json',dict(seed=seed,cycles=CYCLES,warmup_updates=WARMUP,updates=UPDATES,batch=64,microbatch=16,hashes=hashes,
            trainable=dict(baseline='nothing after warm-up',reader='ActivityHead 12769 parameters',brain='25582938 edge log-gains and 166700 node log-gains; encoder and reader frozen'),
            lr=.001,clip=5,primary='9 brain minus baseline, single inference',secondary=['9 brain minus reader','7'],checkpoint='fixed last update',
            scope='encoder from early-interface-001/latest-joint frozen throughout; graph topology and signs fixed'))
        (OUT/'source').mkdir()
        for p in [Path(__file__),Path('experiments/synaptic_learning.py'),Path('experiments/report_synaptic_learning.py'),Path('experiments/plastic_sparse.py'),Path('experiments/spatial_decoder.py')]:shutil.copy2(p,OUT/'source'/p.name)
        games=reserve(OUT)['games'];write_json(OUT/'games.json',games)
        with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
    data=pickle.loads(paths[3].read_bytes());sets=[data['train'],pickle.loads(paths[5].read_bytes())['train'],pickle.loads(paths[6].read_bytes())['train'],pickle.loads(paths[7].read_bytes())]
    groups={}
    for i,r in enumerate(sets[0]):groups.setdefault((r['size'],r['category']),[]).append(i)
    groups=[np.array(groups[k]) for k in sorted(groups)]
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=workers)
    try:
        brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
        brain.load_state_dict(torch.load(paths[2],weights_only=False)['model']);brain.encoder.load_state_dict(torch.load(PARENT,weights_only=False)['encoder']);brain.requires_grad_(False)
        for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
            if not hasattr(brain,f'out_{size}'):
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False);brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        brain.cycles=CYCLES;initial=copy.deepcopy(brain.state_dict());initial_hash=tensor_hash(initial)
        if not resume:
            # Shared warm-up: a fresh reader learns the 3-cycle activity of the untouched brain.
            caches=[]
            for index,rows in enumerate(sets):
                write_json(OUT/'progress.json',dict(phase='activity_cache',dataset=index,total=4));caches.append(features(ActivityMemo(brain),rows))
            valid=features(ActivityMemo(brain),data['holdout'])
            torch.manual_seed(seed);head=ActivityHead(True);opt=torch.optim.Adam(head.parameters(),lr=.001);rng=np.random.default_rng(seed)
            train_head(head,opt,rng,caches,groups,sets,1,WARMUP,'warmup',[])
            save(OUT/'warmup.pt',dict(head=copy.deepcopy(head.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=copy.deepcopy(rng.bit_generator.state),step=WARMUP,validation_loss=validate(head,valid)))
            # Control: the reader keeps learning on the same future draws while the brain stays fixed.
            blocks=[];train_head(head,opt,rng,caches,groups,sets,WARMUP+1,WARMUP+UPDATES,'reader',blocks)
            save(OUT/'latest-reader.pt',dict(head=copy.deepcopy(head.state_dict()),step=UPDATES,arm='reader',draw_blocks=blocks,validation_loss=validate(head,valid)))
            del caches,valid
        warm=torch.load(OUT/'warmup.pt',weights_only=False);head=ActivityHead(True);head.load_state_dict(warm['head']);head_hash=tensor_hash(head.state_dict())
        model=SynapticBrain(brain,head);encoder_hash=tensor_hash(brain.encoder.state_dict())
        opt=torch.optim.Adam(model.synapses(),lr=.001);rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(warm['rng']);first=1;history=[]
        if (OUT/'latest-brain.pt').exists():
            state=torch.load(OUT/'latest-brain.pt',weights_only=False)
            brain.edge_log_gain.data.copy_(state['edge_log_gain']);brain.node_log_gain.data.copy_(state['node_log_gain']);opt.load_state_dict(state['optimizer'])
            rng.bit_generator.state=state['rng'];first=state['step']+1;history=state['history'];prior=state['draw_blocks']
        else:
            prior=[]
            write_json(OUT/'progress.json',dict(phase='gradient_preflight'))
            x,l,c,y=arrays([sets[0][0],sets[1][0],sets[2][0],sets[3][0]])
            torch.testing.assert_close(model.activity(x,c).detach(),activity_map(brain,x,c),rtol=1e-5,atol=2e-6)
            model.zero_grad();equivalent_loss(model(x,c),l,y).backward()
            edge=brain.edge_log_gain.grad;assert torch.isfinite(edge).all() and int((edge!=0).sum())>0 and all(p.grad is None for p in list(head.parameters())+list(brain.encoder.parameters()))
            write_json(OUT/'preflight.json',dict(forward_parity=True,edges_with_gradient_in_4_examples=int((edge!=0).sum()),nodes_with_gradient=int((brain.node_log_gain.grad!=0).sum()),interface_without_gradient=True))
            model.zero_grad(set_to_none=True)
        block=hashlib.sha256();started=time.time()
        for step in range(first,UPDATES+1):
            indices=draw(rng,WARMUP+step,groups,sets)
            for ix in indices:block.update(ix.tobytes())
            rows=[r for s,ix in zip(sets,indices) for r in (s[i] for i in ix)]
            opt.zero_grad(set_to_none=True);total=0.
            for i in range(0,64,16):
                x,l,c,y=arrays(rows[i:i+16]);loss=equivalent_loss(model(x,c),l,y)/4;loss.backward();total+=float(loss)
            torch.nn.utils.clip_grad_norm_(model.synapses(),5.,error_if_nonfinite=True);opt.step()
            if step%25==0:
                rec=dict(phase='training',arm='brain',step=step,loss=total,seconds_per_update=(time.time()-started)/(step-first+1));write_json(OUT/'progress.json',rec);print(rec,flush=True)
            if step%250==0 or step==UPDATES:
                value=validation(model,data['holdout']) if step%500==0 or step==UPDATES else None
                moved=int((brain.edge_log_gain.detach()!=initial['edge_log_gain']).sum());history.append(dict(step=step,validation_loss=value,edges_modified=moved))
                prior=prior+[block.hexdigest()];block=hashlib.sha256()
                save(OUT/'latest-brain.pt',dict(edge_log_gain=brain.edge_log_gain.detach().clone(),node_log_gain=brain.node_log_gain.detach().clone(),optimizer=opt.state_dict(),rng=copy.deepcopy(rng.bit_generator.state),step=step,arm='brain',history=history,draw_blocks=prior))
                write_json(OUT/'history-brain.json',history)
        assert prior==torch.load(OUT/'latest-reader.pt',weights_only=False)['draw_blocks'],'reader and brain must see identical positions'
        assert tensor_hash(head.state_dict())==head_hash and tensor_hash(brain.encoder.state_dict())==encoder_hash
        learned=dict(edge_log_gain=brain.edge_log_gain.detach().clone(),node_log_gain=brain.node_log_gain.detach().clone())
        chosen={a:dict(path=str(OUT/p),sha256=digest(OUT/p)) for a,p in dict(warmup='warmup.pt',reader='latest-reader.pt',brain='latest-brain.pt').items()}
        write_json(OUT/'evaluation-checkpoints.json',chosen);seal=digest(OUT/'evaluation-checkpoints.json');write_json(OUT/'evaluation-seal.json',dict(sha256=seal))
        brain.requires_grad_(False)
        for arm in ARMS:
            brain.load_state_dict(initial)
            if arm=='brain':brain.edge_log_gain.data.copy_(learned['edge_log_gain']);brain.node_log_gain.data.copy_(learned['node_log_gain'])
            else:assert tensor_hash(brain.state_dict())==initial_hash
            reader=ActivityHead(True);reader.load_state_dict(torch.load(OUT/('latest-reader.pt' if arm=='reader' else 'warmup.pt'),weights_only=False)['head']);reader.requires_grad_(False);rows=[]
            for start in range(0,750,50):
                rows.extend(evaluate_games(MemoPolicy(ActivityMemo(brain),reader),games[start:start+50]));write_json(OUT/f'{arm}-games.json',sorted(rows,key=lambda r:(r['size'],r['seed'])))
                rec=dict(phase='evaluation',arm=arm,completed=len(rows),total=750);write_json(OUT/'progress.json',rec);print(rec,flush=True)
        assert all(digest(p)==sha for p,sha in hashes.items()) and digest(OUT/'evaluation-checkpoints.json')==seal and all(digest(v['path'])==v['sha256'] for v in chosen.values())
        write_json(OUT/'verification.json',dict(originals_intact=True,identical_positions_reader_and_brain=True,interface_frozen_during_brain_arm=True,topology_fixed=True,endpoint_fixed=True))
        write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'))
    finally:op.close()

if __name__=='__main__':main(workers=int(sys.argv[1]) if len(sys.argv)>1 else 8)
