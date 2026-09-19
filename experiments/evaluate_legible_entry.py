"""Stage 2: independent information probe, fresh play readers and new games for each entrance."""
import copy,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from experiments.train_joint_interface import digest,tensor_hash,save
from experiments.train_nine_dagger import reserve
from experiments.train_synaptic_learning import draw
from experiments.train_legible_entry import OUT,SEED,UPDATES,sources,build_brain,held_out_games
from experiments.probe_information import trace,decode
from experiments.capacity_probe import arrays,write_json
from experiments.expand_dagger_experience import features
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import ActivityHead
from experiments.validate_spatial_decoder import ActivityMemo,MemoPolicy
from experiments.expressive_models import equivalent_loss
from experiments.scaled_train import evaluate_games

ARMS=('baseline','encoder','synapses');LABELS=dict(baseline='Entrada original',encoder='Encoder legible',synapses='Encoder + sinapsis')

def probe(brain,nine):
    held=held_out_games(nine);parts=dict(train=[r for r in nine if r['seed'] not in held][:3000],test=[r for r in nine if r['seed'] in held][:1000]);packed={}
    for name,rows in parts.items():
        xs,maps=[],[]
        for i in range(0,len(rows),32):
            x,_,c,_=arrays(rows[i:i+32]);xs.append(x);maps.append(trace(brain,x,c,1)[1][0])
        x=torch.cat(xs).reshape(-1,16,16,10);packed[name]=(torch.cat(maps),x.argmax(-1),x.sum(-1)>0)
    return {label:decode(packed['train'],packed['test'],kernel,hidden,20261112) for label,kernel,hidden in (('linear1',1,0),('linear3',3,0),('mlp3',3,64))}

def main(workers=8):
    torch.set_num_threads(1);assert all((OUT/f'completed-{a}.json').exists() for a in ARMS[1:]) and not (OUT/'completed.json').exists()
    paths,sets,_=sources();hashes={str(p):digest(p) for p in paths};nine=pickle.loads(paths[5].read_bytes())['train']
    entries={a:torch.load(OUT/f'entry-{a}.pt',weights_only=False) for a in ARMS[1:]};assert all(e['step']==UPDATES for e in entries.values()) and entries['encoder']['draws']==entries['synapses']['draws']
    sealed={a:digest(OUT/f'entry-{a}.pt') for a in ARMS[1:]};write_json(OUT/'evaluation-seal.json',sealed)
    games=reserve(OUT)['games'];write_json(OUT/'games.json',games)
    with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
    groups={}
    for i,r in enumerate(sets[0]):groups.setdefault((r['size'],r['category']),[]).append(i)
    groups=[np.array(groups[k]) for k in sorted(groups)]
    op=PlasticOperator(sparse.load_npz('data/processed/graph.npz'),workers=workers);results={};pairs={}
    try:
        brain=build_brain(op);initial=copy.deepcopy(brain.state_dict())
        for arm in ARMS:
            brain.load_state_dict(initial)
            if arm!='baseline':
                brain.encoder.load_state_dict(entries[arm]['encoder'])
                if arm=='synapses':brain.edge_log_gain.data.copy_(entries[arm]['edge_log_gain']);brain.node_log_gain.data.copy_(entries[arm]['node_log_gain'])
            frozen=tensor_hash(brain.state_dict());write_json(OUT/'progress.json',dict(phase='probe',arm=arm));results[arm]=dict(probe=probe(brain,nine))
            print(dict(arm=arm,probe={k:round(v['clue_accuracy'],4) for k,v in results[arm]['probe'].items()}),flush=True)
            caches=[]
            for index,rows in enumerate(sets):
                write_json(OUT/'progress.json',dict(phase='activity_cache',arm=arm,dataset=index,total=4));caches.append(features(ActivityMemo(brain),rows))
            torch.manual_seed(SEED);head=ActivityHead(True);init=tensor_hash(head.state_dict());opt=torch.optim.Adam(head.parameters(),lr=.001);rng=np.random.default_rng(SEED+1);sample=hashlib.sha256()
            for step in range(1,3001):
                indices=draw(rng,step,groups,sets)
                for ix in indices:sample.update(ix.tobytes())
                a,l,c,y=[torch.cat([p[ix] for p,ix in zip(parts,indices)]) for parts in zip(*caches)]
                opt.zero_grad();loss=equivalent_loss(head(a,c),l,y);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
                if step%500==0:rec=dict(phase='reader',arm=arm,step=step,loss=float(loss));write_json(OUT/'progress.json',rec);print(rec,flush=True)
            pairs[arm]=dict(initial=init,draws=sample.hexdigest());save(OUT/f'reader-{arm}.pt',dict(head=copy.deepcopy(head.state_dict()),step=3000,arm=arm));del caches
            head.requires_grad_(False);rows=[]
            for start in range(0,750,50):
                rows.extend(evaluate_games(MemoPolicy(ActivityMemo(brain),head),games[start:start+50]));write_json(OUT/f'{arm}-games.json',sorted(rows,key=lambda r:(r['size'],r['seed'])))
                rec=dict(phase='evaluation',arm=arm,completed=len(rows),total=750);write_json(OUT/'progress.json',rec);print(rec,flush=True)
            assert frozen==tensor_hash(brain.state_dict())
        assert len({json.dumps(p) for p in pairs.values()})==1 and all(digest(p)==sha for p,sha in hashes.items()) and all(digest(OUT/f'entry-{a}.pt')==s for a,s in sealed.items())
    finally:op.close()
    summary=dict(probe={a:results[a]['probe'] for a in ARMS},edges_modified=entries['synapses']['edges_modified'])
    for size in (7,9):
        counts={};wins={}
        for arm in ARMS:
            played=[r for r in json.loads((OUT/f'{arm}-games.json').read_text()) if r['size']==size and not r['automatic']];wins[arm]=np.array([int(r['won']) for r in played])
            counts[arm]=dict(wins=int(wins[arm].sum()),n=len(played),**{k:int(sum(r[k] for r in played)) for k in ('safe_choices','safe_opportunities','known_mine_choices','death_with_safe_available')})
        summary[str(size)]=dict(results=counts)
        for a,b in (('synapses','baseline'),('encoder','baseline'),('synapses','encoder')):
            d=wins[a]-wins[b];rng=np.random.default_rng(SEED+size);boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
            summary[str(size)][f'{a}_minus_{b}']=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())
    write_json(OUT/'summary.json',summary)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    v=[100*summary['probe'][a]['mlp3']['clue_accuracy'] for a in ARMS];axes[0].bar(range(3),v,color=['#8a8f98','#099caa','#aa7945']);axes[0].set_title('Números legibles tras el primer salto (sonda MLP 3×3)');axes[0].set_ylabel('Acierto (%)');axes[0].set_ylim(0,105)
    for i,t in enumerate(v):axes[0].text(i,t+1,f'{t:.1f}%',ha='center')
    axes[0].set_xticks(range(3),[LABELS[a] for a in ARMS],fontsize=8)
    for ax,size in zip(axes[1:],(7,9)):
        cc=[summary[str(size)]['results'][a] for a in ARMS];v=[100*c['wins']/c['n'] for c in cc];ax.bar(range(3),v,color=['#8a8f98','#099caa','#aa7945'])
        for i,c in enumerate(cc):ax.text(i,v[i]+.5,f"{c['wins']}/{c['n']}",ha='center')
        ax.set_xticks(range(3),[LABELS[a] for a in ARMS],fontsize=8);ax.set_title(f'Victorias {size}×{size}');ax.set_ylabel('Victorias (%)');ax.set_ylim(0,max(v)+7)
    fig.savefig('research/legible-entry-results.png',dpi=150);plt.close(fig)
    write_json(OUT/'completed.json',dict(completed=True));write_json(OUT/'progress.json',dict(phase='completed'));print(json.dumps(summary,indent=2))

if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 8)
