"""Prepare preserved biological graph, reconfigured control, fresh games and cost evidence."""
import hashlib,json,pickle,time,resource
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from minesweeper import Minesweeper
from experiments.scaled_data import identity
from experiments.capacity_probe import arrays,write_json
from experiments.expressive_models import equivalent_loss
from experiments.plastic_sparse import PlasticOperator,reconfigured
from experiments.retina_policy import build_mapping,RetinaPolicy


def main():
    torch.set_num_threads(1)
    root=Path('runs/structural-learning-001');root.mkdir(exist_ok=True)
    if (root/'prepared.json').exists():raise FileExistsError('Already prepared')
    data=pickle.loads(Path('runs/scaled-learning-001/dataset.pkl').read_bytes())
    used={r['layout_hash'] for rows in data.values() for r in rows}
    for name in ('capacity-diagnostic-001','spatial-diagnostic-001'):
        old=pickle.loads((Path('runs')/name/'dataset.pkl').read_bytes())
        used.update(r['layout_hash'] for rows in old.values() for r in rows)
    games=[]
    for size,mines in ((5,3),(7,7)):
        seed=4_400_000_000+size*100000;count=0;attempts=0
        while count<250:
            attempts+=1
            if attempts>100000:raise RuntimeError('Fresh layout pool exhausted')
            env=Minesweeper(seed,size,mines);key=identity(env)
            if key not in used:
                games.append(dict(seed=seed,size=size,mines=mines,layout_hash=key));used.add(key);count+=1
            seed+=1
    data['games']=games
    (root/'dataset.pkl').write_bytes(pickle.dumps(data))
    mapping=build_mapping();(root/'mapping.pkl').write_bytes(pickle.dumps(mapping))
    assert len(np.unique(mapping['input_index']))==len(mapping['input_index'])
    for indices,_ in mapping['outputs'].values():
        assert not np.intersect1d(mapping['input_index'],indices).size
    graph=sparse.load_npz('data/processed/graph.npz')
    signs=np.load('data/processed/neurotransmitter_signs.npy')
    start=time.monotonic();rewired,report=reconfigured(graph,signs)
    report['seconds']=time.monotonic()-start
    sparse.save_npz(root/'rewired.npz',rewired)
    write_json(root/'rewiring.json',report)
    del rewired
    op=PlasticOperator(graph,workers=4)
    measurements=[]
    for batch in (8,16):
        torch.manual_seed(26)
        model=RetinaPolicy(op,mapping,plastic=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        x,legal,context,labels=arrays(data['train'][:batch])
        times=[]
        for step in range(3):
            start=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss=equivalent_loss(model(x,context),legal,labels);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            optimizer.step();times.append(time.monotonic()-start)
        measurements.append(dict(batch=batch,step_seconds=times,seconds_per_example=float(np.mean(times[1:]))/batch,
                                 changed_edges=int(model.edge_log_gain.detach().count_nonzero()),
                                 changed_nodes=int(model.node_log_gain.detach().count_nonzero())))
        print(json.dumps(measurements[-1]),flush=True)
        del model,optimizer
    op.close()
    write_json(root/'benchmark.json',dict(measurements=measurements,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    inputs=['data/processed/graph.npz','data/raw/annotations.feather','data/processed/neuron_ids.npy','data/processed/neurotransmitter_signs.npy']
    inputs += [str(root/name) for name in ('dataset.pkl','mapping.pkl','rewired.npz')]
    write_json(root/'prepared.json',dict(input_neurons=len(mapping['input_index']),nodes=graph.shape[0],edges=graph.nnz,
               protected_sha256=hashlib.sha256(Path('runs/hybrid-001/checkpoint.pkl').read_bytes()).hexdigest(),
               hashes={p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest() for p in inputs},
               fresh_final_games=500,training_positions=len(data['train']),validation_positions=len(data['holdout'])))


if __name__=='__main__':main()
