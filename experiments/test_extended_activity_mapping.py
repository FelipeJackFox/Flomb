"""Real-graph regression and sensitivity checks for newly supported board sizes."""
import importlib.util
import pickle
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from experiments.retina_policy import RetinaPolicy, build_mapping
from experiments.plastic_sparse import PlasticOperator
from experiments.spatial_decoder import activity_map
from experiments.train_risk_auxiliary import BASE, PARENT, digest
from experiments.backbone import encode_visible
from experiments.train import visible, public_context
from experiments.capacity_probe import write_json
from minesweeper import Minesweeper


def main():
    torch.set_num_threads(1)
    original = digest(PARENT)
    spec = importlib.util.spec_from_file_location('legacy_activity',
        'runs/difficulty-transfer-001/source/spatial_decoder-legacy.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    op = PlasticOperator(sparse.load_npz('data/processed/graph.npz'), workers=4)
    try:
        brain = RetinaPolicy(op, pickle.loads((BASE/'mapping.pkl').read_bytes()), True)
        brain.load_state_dict(torch.load(BASE/'training/retina_plastic-20260926.pt', weights_only=False)['model'])
        brain.encoder.load_state_dict(torch.load(PARENT, weights_only=False)['encoder'])
        brain.requires_grad_(False)
        try:
            activity_map(brain, torch.zeros(1,2560), torch.tensor([[9/16,12/256]]))
        except ValueError as exc:
            assert 'Missing output mapping' in str(exc)
        else:
            raise AssertionError('Missing mapping must fail loudly')
        mapping = build_mapping(sizes=(5,7,9,12,16))
        for size,(ix,wt) in mapping['outputs'].items():
            if hasattr(brain,f'out_{size}'):
                assert torch.equal(getattr(brain,f'out_{size}'),torch.from_numpy(ix))
                assert torch.equal(getattr(brain,f'weight_{size}'),torch.from_numpy(wt))
            else:
                brain.register_buffer(f'out_{size}',torch.from_numpy(ix),persistent=False)
                brain.register_buffer(f'weight_{size}',torch.from_numpy(wt),persistent=False)
        checks = {}
        for size,mines in ((5,3),(7,7),(9,12),(12,22),(16,38)):
            first=Minesweeper(20261028,size,mines)
            envs=[first]
            for offset in range(1,100):
                candidate=Minesweeper(20261028+offset,size,mines)
                if not np.array_equal(visible(first),visible(candidate)):
                    envs.append(candidate)
                    break
            assert len(envs)==2, 'Fixture requires distinct visible observations'
            x,_=encode_visible([visible(e) for e in envs])
            x=torch.from_numpy(x)
            c=torch.tensor(np.array([public_context(size,mines)]*2))
            actual=activity_map(brain,x,c)
            assert torch.isfinite(actual).all() and int(torch.count_nonzero(actual)) > 0
            assert not torch.equal(actual[0],actual[1]), 'Activity must respond to different visible boards'
            if size in (5,7):
                torch.testing.assert_close(actual,legacy.activity_map(brain,x,c),rtol=0,atol=0)
            else:
                assert torch.count_nonzero(legacy.activity_map(brain,x,c)) == 0
            assert torch.count_nonzero(actual[:,:,size:,:]) == 0
            assert torch.count_nonzero(actual[:,:,:,size:]) == 0
            checks[str(size)]=dict(nonzero=True,finite=True,observation_sensitive=True,
                legacy_equal=size in (5,7),padding_zero=True)
        assert digest(PARENT)==original
        write_json(Path('benchmarks/extended-mapping-verification.json'),
            dict(checks=checks,missing_mapping_rejected=True,parent_intact=True))
        print('PASS: real graph parity 5/7; nonzero and observation-sensitive 9/12/16; missing mapping rejected')
    finally:
        op.close()


if __name__=='__main__':
    main()
