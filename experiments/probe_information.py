"""How much of the visible board survives in brain activity after each propagation cycle?

Diagnostic, not a game benchmark: decoders reconstruct each cell's clue (hidden, 0..8) from the
pooled output activity that the reader sees. Encoder output is the reference for what enters the brain.
"""
import pickle,shutil,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import sparse
from experiments.train_joint_interface import BASE,digest
from experiments.train_nine_dagger import MAPPING
from experiments.capacity_probe import arrays,write_json
from experiments.retina_policy import RetinaPolicy
from experiments.plastic_sparse import PlasticOperator

OUT=Path('runs/information-probe-001');PARENT=Path('runs/early-interface-001/latest-joint.pt');LEARNED=Path('runs/synaptic-learning-001/latest-brain.pt')
CYCLES=4;SEED=20261112

@torch.no_grad()
def trace(brain,x,context,cycles=None):
    """Pooled output activity after every cycle, plus the encoder map that feeds the brain."""
    b=brain;count=len(x);sizes=(context[:,0]*16).round().long();assert len(set(sizes.tolist()))==1;size=int(sizes[0])
    grid=x.reshape(count,16,16,10).permute(0,3,1,2)
    encoded=b.encoder(torch.cat([grid,context[:,:,None,None].expand(-1,-1,16,16)],1))*(grid.sum(1,keepdim=True)!=0)
    sg=2*b.input_xy[None]*(sizes[:,None,None]-1)/15-1
    samples=F.grid_sample(encoded,sg[:,:,None],align_corners=True).squeeze(-1)
    sensory=samples[:,b.input_channel,torch.arange(len(b.input_channel))]
    drive=x.new_zeros(b.operator.shape[0],count).index_copy(0,b.input_index,sensory.T)
    state=torch.tanh(drive);values=(b.base*b.edge_log_gain.clamp(-2,2).exp()).numpy();gain=b.node_log_gain.clamp(-1,1).exp()[:,None]
    b.operator.bind(values,force=True);ix,wt=getattr(b,f'out_{size}'),getattr(b,f'weight_{size}');maps=[]
    for _ in range(cycles or CYCLES):
        state=torch.tanh(torch.from_numpy(b.operator.multiply(values,(gain*state).numpy()))+.15*drive)
        pooled=(state[ix][:,:,:,torch.arange(count)]*wt[:,:,:,None]).sum(2).permute(2,1,0)
        result=x.new_zeros(count,10,16,16);result[:,:,:size,:size]=pooled.reshape(count,10,size,size)*10;maps.append(result)
    return encoded,maps

def decode(train,test,kernel,hidden,seed):
    """Per-cell classifier over a kernel x kernel window; returns accuracies on held-out games."""
    (a,y,m),(ta,ty,tm)=train,test;mean=a.mean((0,2,3),keepdim=True);std=a.std((0,2,3),keepdim=True)+1e-6
    a,ta=(a-mean)/std,(ta-mean)/std;torch.manual_seed(seed);pad=kernel//2
    net=nn.Conv2d(a.shape[1],10,kernel,padding=pad) if not hidden else nn.Sequential(nn.Conv2d(a.shape[1],hidden,kernel,padding=pad),nn.Tanh(),nn.Conv2d(hidden,10,1))
    opt=torch.optim.Adam(net.parameters(),lr=.01);rng=np.random.default_rng(seed)
    for step in range(1500):
        ix=torch.from_numpy(rng.integers(len(a),size=128));logits=net(a[ix]).permute(0,2,3,1)[m[ix]]
        opt.zero_grad();F.cross_entropy(logits,y[ix][m[ix]]).backward();opt.step()
    with torch.no_grad():pred=net(ta).argmax(1)[tm];true=ty[tm]
    clue=true>0;recall=[float((pred[true==k]==k).float().mean()) for k in range(10) if (true==k).any()]
    return dict(accuracy=float((pred==true).float().mean()),clue_accuracy=float((pred[clue]==true[clue]).float().mean()),balanced_accuracy=float(np.mean(recall)),
        number_error_mean=float((pred[clue]-true[clue]).abs().float().mean()))

def main(workers=8):
    torch.set_num_threads(4);OUT.mkdir(exist_ok=False)
    rows=pickle.loads(Path('runs/nine-dagger-001/dataset.pkl').read_bytes())['train'];seeds=sorted({r['seed'] for r in rows})
    rng=np.random.default_rng(SEED);rng.shuffle(seeds);held=set(seeds[:150]);train=[r for r in rows if r['seed'] not in held][:3000];test=[r for r in rows if r['seed'] in held][:1000]
    paths=[PARENT,MAPPING,BASE/'training/retina_plastic-20260926.pt',LEARNED,Path('runs/nine-dagger-001/dataset.pkl')]
    write_json(OUT/'manifest.json',dict(seed=SEED,cycles=CYCLES,train_positions=len(train),test_positions=len(test),split='by game seed, 150 held-out games',hashes={str(p):digest(p) for p in paths},
        decoders=dict(linear1='1x1 linear',linear3='3x3 linear',mlp3='3x3, 64 tanh units'),target='clue class per cell: hidden or 0..8'))
    (OUT/'source').mkdir();shutil.copy2(Path(__file__),OUT/'source'/Path(__file__).name)
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=workers)
    try:
        brain=RetinaPolicy(op,pickle.loads((BASE/'mapping.pkl').read_bytes()),True)
        brain.load_state_dict(torch.load(paths[2],weights_only=False)['model']);brain.encoder.load_state_dict(torch.load(PARENT,weights_only=False)['encoder']);brain.requires_grad_(False)
        for size,(ix,wt) in pickle.loads(MAPPING.read_bytes())['outputs'].items():
            if not hasattr(brain,f'out_{size}'):
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False);brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        learned=torch.load(LEARNED,weights_only=False);original=(brain.edge_log_gain.clone(),brain.node_log_gain.clone());features={}
        for variant in ('original','learned'):
            for p,v in zip((brain.edge_log_gain,brain.node_log_gain),original if variant=='original' else (learned['edge_log_gain'],learned['node_log_gain'])):p.data.copy_(v)
            for name,part in (('train',train),('test',test)):
                chunks=[]
                for i in range(0,len(part),32):
                    x,_,c,_=arrays(part[i:i+32]);encoded,maps=trace(brain,x,c);chunks.append((x,encoded,maps))
                    if i%640==0:rec=dict(phase='activity',variant=variant,split=name,done=i,total=len(part));write_json(OUT/'progress.json',rec);print(rec,flush=True)
                x=torch.cat([q[0] for q in chunks]).reshape(-1,16,16,10);mask=x.sum(-1)>0
                features[variant,name]=dict(y=x.argmax(-1),m=mask,raw=x.permute(0,3,1,2).contiguous(),encoder=torch.cat([q[1] for q in chunks]),**{f'cycle{k+1}':torch.cat([q[2][k] for q in chunks]) for k in range(CYCLES)})
        results={}
        sources=[('original','raw'),('original','encoder')]+[(v,f'cycle{k}') for v in ('original','learned') for k in range(1,CYCLES+1)]
        for variant,source in sources:
            tr,te=features[variant,'train'],features[variant,'test'];key=source if source in ('raw','encoder') else f'{variant}_{source}';results[key]={}
            for label,kernel,hidden in (('linear1',1,0),('linear3',3,0),('mlp3',3,64)):
                results[key][label]=decode((tr[source],tr['y'],tr['m']),(te[source],te['y'],te['m']),kernel,hidden,SEED)
            rec=dict(phase='decoding',source=key,**{k:round(v['clue_accuracy'],4) for k,v in results[key].items()});write_json(OUT/'progress.json',rec);print(rec,flush=True)
            write_json(OUT/'results.json',results)
        y=features['original','test'];true=y['y'][y['m']];results['majority_class']=dict(accuracy=float((true==true.mode().values).float().mean()),clue_share=float((true>0).float().mean()))
        write_json(OUT/'results.json',results);write_json(OUT/'completed.json',dict(completed=True))
    finally:op.close()

if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 8)
