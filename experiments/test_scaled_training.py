import contextlib
import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scipy import sparse
from brain import BrainPolicy
from experiments.backbone import Backbone
from experiments.capacity_probe import build_dataset, play
from experiments.scaled_train import FLAGS, make_model, evaluate_games, main


class Tests(unittest.TestCase):
    def test_factorial_flags_and_common_head(self):
        torch.set_num_threads(1)
        graph = sparse.eye(12,format='csr',dtype=np.float32)*.3
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'init.npz'
            BrainPolicy(graph,inputs=2560,actions=256,readouts=8).save(path)
            operator=Backbone(graph,path)
        models={name:make_model(name,17,operator) for name in FLAGS}
        for name,model in models.items():
            self.assertEqual(model.channels,1)
            self.assertEqual(hasattr(model,'encoder'),FLAGS[name][0])
            self.assertEqual(hasattr(model,'bias'),FLAGS[name][1])
            torch.testing.assert_close(model.head.weight,models['current'].head.weight,rtol=0,atol=0)
            model(torch.randn(2,2560),torch.zeros(2,2)).square().sum().backward()
            self.assertGreater(float(model.log_gain.grad.abs().sum()),0)
            if hasattr(model,'encoder'):
                self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()),0)
            if hasattr(model,'bias'):
                self.assertGreater(float(model.bias.grad.abs().sum()),0)

    def test_batched_games_match_serial(self):
        torch.set_num_threads(1)
        games=[dict(seed=500+i,size=size,mines=mines) for size,mines in ((5,3),(7,7)) for i in range(5)]
        model=make_model('cnn',18)
        batch=evaluate_games(model,games,batch=4)
        single=evaluate_games(model,games,batch=1)
        self.assertEqual(batch,single)
        old=sorted(play(model,games),key=lambda r:(r['size'],r['seed']))
        self.assertEqual([(r['won'],r['clicks']) for r in batch],[(r['won'],r['clicks']) for r in old])
        for row in batch:
            self.assertLessEqual(row['safe_choices'],row['safe_opportunities'])
            self.assertEqual(row['safe_opportunities']+row['guesses'],row['clicks'])

    def test_cli_resume_weights_and_adam_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            data=root/'dataset.pkl'
            data.write_bytes(pickle.dumps(build_dataset(2,2,2)))
            def invoke(name,extra):
                argv=['scaled_train','--data',str(data),'--out',str(root/name),'--variants','cnn',
                      '--seeds','18','--updates','3','--batch','4',*extra]
                with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                    main()
            invoke('continuous',[])
            invoke('resumed',['--stop-after','1'])
            invoke('resumed',['--resume'])
            a=torch.load(root/'continuous/cnn-18.pt',weights_only=False)
            b=torch.load(root/'resumed/cnn-18.pt',weights_only=False)
            self.assertEqual(a['step'],b['step'])
            self.assertEqual(a['rng'],b['rng'])
            for key in a['model']:
                torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            for index,state in a['optimizer']['state'].items():
                for key,value in state.items():
                    torch.testing.assert_close(value,b['optimizer']['state'][index][key],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
