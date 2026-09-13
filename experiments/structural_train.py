"""Paired structural trial with full-edge plasticity and functional controls."""
import argparse,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.capacity_probe import arrays,measure,write_json
from experiments.expressive_models import ConvPolicy,equivalent_loss
from experiments.plastic_sparse import PlasticOperator
from experiments.retina_policy import RetinaPolicy
from experiments.scaled_train import evaluate_games,save_checkpoint


class Intervention(torch.nn.Module):
    def __init__(self,model,mode):
        super().__init__();self.model=model;self.mode=mode
    def forward(self,x,context):
        return self.model(x,context,ablation=self.mode)


def make_model(name,seed,operator,mapping):
    torch.manual_seed(seed)
    if name=='cnn':return ConvPolicy()
    return RetinaPolicy(operator,mapping,plastic=name!='retina_fixed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('runs/structural-learning-001'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--updates',type=int,default=3000)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--seed',type=int,default=20260926)
    parser.add_argument('--variants',nargs='+',choices=['cnn','retina_fixed','retina_plastic','rewired_plastic'],
                        default=['cnn','retina_fixed','retina_plastic','rewired_plastic'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args();torch.set_num_threads(1)
    out=args.out or args.root/'training';out.mkdir(parents=True,exist_ok=True)
    prepared=json.loads((args.root/'prepared.json').read_text())
    for name,digest in prepared['hashes'].items():
        if hashlib.file_digest(Path(name).open('rb'),'sha256').hexdigest()!=digest:
            raise ValueError(f'Prepared input changed: {name}')
    protected=Path('runs/hybrid-001/checkpoint.pkl')
    if hashlib.sha256(protected.read_bytes()).hexdigest()!=prepared['protected_sha256']:
        raise ValueError('Protected hybrid changed since preparation')
    sources=[Path(__file__),*[Path('experiments')/n for n in ('retina_policy.py','plastic_sparse.py','plastic_kernel.cpp',
             'scaled_train.py','capacity_probe.py','expressive_models.py','train.py')],Path('minesweeper.py'),Path('solver.py')]
    manifest=dict(seed=args.seed,updates=args.updates,batch=args.batch,variants=args.variants,lr=.001,
                  protected_sha256=prepared['protected_sha256'],input_hashes=prepared['hashes'],
                  torch=torch.__version__,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text())!=manifest:
            raise ValueError('Resume requires identical config/source/inputs')
    else:
        if args.resume:raise FileNotFoundError('Manifest missing')
        write_json(manifest_path,manifest);(out/'source').mkdir()
        for p in sources:(out/'source'/p.name).write_bytes(p.read_bytes())
    data=pickle.loads((args.root/'dataset.pkl').read_bytes())
    mapping=pickle.loads((args.root/'mapping.pkl').read_bytes())
    groups={}
    for i,row in enumerate(data['train']):groups.setdefault((row['size'],row['category']),[]).append(i)
    groups=[np.array(groups[k]) for k in sorted(groups)]
    result_path=out/'results.json';results=json.loads(result_path.read_text()) if result_path.exists() else []
    for name in args.variants:
        if any(r['variant']==name for r in results):continue
        operator=None
        if name!='cnn':
            graph_path=args.root/'rewired.npz' if name=='rewired_plastic' else Path('data/processed/graph.npz')
            operator=PlasticOperator(sparse.load_npz(graph_path),workers=4)
        model=make_model(name,args.seed,operator,mapping)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        rng=np.random.default_rng(args.seed+1)
        history,step,train_seconds=[],0,0.
        state_path=out/f'{name}-{args.seed}.pt'
        if args.resume and state_path.exists():
            state=torch.load(state_path,weights_only=False)
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng'])
            history,step,train_seconds,before=state['history'],state['step'],state['train_seconds'],state['before']
        else:
            before=measure(model,data['holdout'])
        start_step=step;started=time.monotonic()
        for step in range(step+1,args.updates+1):
            indices=[int(rng.choice(groups[(step*args.batch+j)%len(groups)])) for j in range(args.batch)]
            x,legal,context,labels=arrays([data['train'][i] for i in indices])
            optimizer.zero_grad(set_to_none=True)
            loss=equivalent_loss(model(x,context),legal,labels);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            if isinstance(model,RetinaPolicy):
                with torch.no_grad():
                    model.node_log_gain.clamp_(-1,1)
                    if model.plastic:model.edge_log_gain.clamp_(-2,2)
            stop=args.stop_after is not None and step-start_step>=args.stop_after
            if step%100==0 or step==args.updates or stop:
                elapsed=train_seconds+time.monotonic()-started
                progress=dict(phase='training',variant=name,seed=args.seed,updates=step,target=args.updates,
                              loss=float(loss.detach()),gradient_norm=float(norm),training_seconds=elapsed)
                history.append(progress);write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                # Large edge/Adam snapshots are saved less often than lightweight progress.
                if step%250==0 or step==args.updates or stop:
                    save_checkpoint(state_path,model,optimizer,rng,step,history,before,elapsed)
            elif step%250==0:
                elapsed=train_seconds+time.monotonic()-started
                save_checkpoint(state_path,model,optimizer,rng,step,history,before,elapsed)
            if stop and step<args.updates:
                if operator:operator.close()
                return
        train_seconds+=time.monotonic()-started
        write_json(out/'progress.json',dict(phase='evaluation',variant=name,seed=args.seed,updates=step))
        result=dict(variant=name,seed=args.seed,updates=step,batch=args.batch,before=before,
                    holdout=measure(model,data['holdout']),training_seconds=train_seconds,
                    parameters=sum(p.numel() for p in model.parameters()),games=evaluate_games(model,data['games']),
                    interventions={})
        if isinstance(model,RetinaPolicy):
            result['changed_nodes']=int(model.node_log_gain.detach().count_nonzero())
            result['changed_edges']=int(model.edge_log_gain.detach().count_nonzero()) if model.plastic else 0
            for mode in ('zero','shuffle'):
                write_json(out/'progress.json',dict(phase='intervention',variant=name,mode=mode,seed=args.seed,updates=step))
                intervention=Intervention(model,mode)
                result['interventions'][mode]=dict(holdout=measure(intervention,data['holdout']),
                                                   games=evaluate_games(intervention,data['games']))
        results.append(result);write_json(result_path,results)
        print('FINISHED',name,result['holdout']['correct'],flush=True)
        del model,optimizer
        if operator:operator.close()
    intact=hashlib.sha256(protected.read_bytes()).hexdigest()==manifest['protected_sha256']
    if not intact:raise RuntimeError('Protected checkpoint changed')
    write_json(out/'completed.json',dict(models=len(results),protected_checkpoint_intact=True,time=time.time()))


if __name__=='__main__':main()
