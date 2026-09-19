"""Stage 1: make the clues legible after the first synaptic hop. Arms: encoder | synapses (encoder + synapses)."""
import copy,hashlib,pickle,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.train_joint_interface import BASE,digest,tensor_hash,save
from experiments.train_nine_dagger import MAPPING
from experiments.train_synaptic_learning import draw
from experiments.capacity_probe import arrays,write_json
from experiments.legible_entry import OpenBrain,ClueDecoder
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import activity_map

OUT=Path('runs/legible-entry-001');PARENT=Path('runs/early-interface-001/latest-joint.pt');SEED=20261113;UPDATES=3000;PROBE_SEED=20261112

def held_out_games(rows):
    seeds=sorted({r['seed'] for r in rows});np.random.default_rng(PROBE_SEED).shuffle(seeds);return set(seeds[:150])

def sources():
    paths=[PARENT,MAPPING,BASE/'training/retina_plastic-20260926.pt',BASE/'dataset.pkl',Path('runs/adapted-dagger-001/dataset.pkl'),Path('runs/nine-dagger-001/dataset.pkl'),Path('runs/deep-coverage-001/states-autonomous.pkl')]
    nine=pickle.loads(paths[5].read_bytes())['train'];held=held_out_games(nine);banned={r['layout_hash'] for r in nine if r['seed'] in held}
    sets=[pickle.loads(paths[3].read_bytes())['train'],pickle.loads(paths[4].read_bytes())['train'],nine,pickle.loads(paths[6].read_bytes())]
    # The information probe tests on 150 held-out games; none of their layouts may shape the entrance.
    return paths,[[r for r in rows if r['layout_hash'] not in banned] for rows in sets],len(banned)

def build_brain(op):
    brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
    brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']);brain.encoder.load_state_dict(torch.load(PARENT,weights_only=False)['encoder']);brain.requires_grad_(False)
    for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
        if not hasattr(brain,f'out_{size}'):
            brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False);brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
    brain.cycles=1;return brain

def main(arm,workers=4):
    assert arm in ('encoder','synapses');torch.set_num_threads(1);OUT.mkdir(exist_ok=True);(OUT/'source').mkdir(exist_ok=True)
    paths,sets,banned=sources();hashes={str(p):digest(p) for p in paths}
    write_json(OUT/f'manifest-{arm}.json',dict(arm=arm,seed=SEED,updates=UPDATES,cycles=1,batch=64,microbatch=16,hashes=hashes,excluded_probe_layouts=banned,training_positions=[len(s) for s in sets],
        objective='cross-entropy reconstruction of every visible cell (hidden,0..8) from pooled output activity; no play loss',lr=dict(decoder=.001,encoder=.0003,synapses=.001),clip=5,checkpoint='fixed last update'))
    for p in [Path(__file__),Path('experiments/legible_entry.py'),Path('experiments/evaluate_legible_entry.py'),Path('experiments/probe_information.py')]:shutil.copy2(p,OUT/'source'/p.name)
    groups={}
    for i,r in enumerate(sets[0]):groups.setdefault((r['size'],r['category']),[]).append(i)
    groups=[np.array(groups[k]) for k in sorted(groups)]
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=workers)
    try:
        brain=build_brain(op);initial=copy.deepcopy(brain.state_dict());model=OpenBrain(brain,arm=='synapses');torch.manual_seed(SEED);decoder=ClueDecoder()
        x,_,c,_=arrays([s[0] for s in sets]);torch.testing.assert_close(model.activity(x,c).detach(),activity_map(brain,x,c),rtol=1e-5,atol=2e-6)
        loss,_=decoder.loss(model.activity(x,c),x);loss.backward();assert sum(float(p.grad.abs().sum()) for p in brain.encoder.parameters())>0
        edges=int((brain.edge_log_gain.grad!=0).sum()) if arm=='synapses' else 0;write_json(OUT/f'preflight-{arm}.json',dict(forward_parity=True,encoder_gradient=True,edges_with_gradient_in_4_examples=edges))
        groups_=[dict(params=decoder.parameters(),lr=.001),dict(params=brain.encoder.parameters(),lr=.0003)]+([dict(params=model.synapses(),lr=.001)] if arm=='synapses' else [])
        opt=torch.optim.Adam(groups_);rng=np.random.default_rng(SEED);draws=hashlib.sha256();started=time.time();recent=[]
        for step in range(1,UPDATES+1):
            indices=draw(rng,step,groups,sets)
            for ix in indices:draws.update(ix.tobytes())
            rows=[r for s,ix in zip(sets,indices) for r in (s[i] for i in ix)];opt.zero_grad(set_to_none=True);total=0.;hits=0.
            for i in range(0,64,16):
                x,_,c,_=arrays(rows[i:i+16]);loss,acc=decoder.loss(model.activity(x,c),x);(loss/4).backward();total+=float(loss)/4;hits+=acc/4
            for g in opt.param_groups:torch.nn.utils.clip_grad_norm_(g['params'],5.,error_if_nonfinite=True)
            opt.step();recent=(recent+[hits])[-100:]
            if step%50==0:
                rec=dict(phase='training',arm=arm,step=step,loss=total,train_clue_accuracy_last100=float(np.mean(recent)),seconds_per_update=(time.time()-started)/step);write_json(OUT/f'progress-{arm}.json',rec);print(rec,flush=True)
            if step%500==0 or step==UPDATES:
                save(OUT/f'entry-{arm}.pt',dict(arm=arm,step=step,encoder=copy.deepcopy(brain.encoder.state_dict()),edge_log_gain=brain.edge_log_gain.detach().clone() if arm=='synapses' else None,node_log_gain=brain.node_log_gain.detach().clone() if arm=='synapses' else None,
                    decoder=copy.deepcopy(decoder.state_dict()),draws=draws.hexdigest() if step==UPDATES else None,edges_modified=int((brain.edge_log_gain.detach()!=initial['edge_log_gain']).sum())))
        if arm=='encoder':assert torch.equal(brain.edge_log_gain,initial['edge_log_gain']) and torch.equal(brain.node_log_gain,initial['node_log_gain'])
        assert all(digest(p)==sha for p,sha in hashes.items());write_json(OUT/f'completed-{arm}.json',dict(completed=True))
    finally:op.close()

if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 4)
