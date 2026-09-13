import copy
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from brain import BrainPolicy
from experiments.backbone import Backbone
from experiments.expressive_models import ConnectomePolicy, SparseMessage, equivalent_loss
from experiments.capacity_probe import build_dataset


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        rng = np.random.default_rng(12)
        graph = sparse.csr_matrix(rng.normal(0, .08, (12, 12)).astype(np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'init.npz'
            BrainPolicy(graph, inputs=2560, actions=256, readouts=8).save(path)
            self.operator = Backbone(graph, path)
        self.x = torch.randn(2, 2560)*.1
        self.context = torch.zeros(2, 2)

    def test_current_forward_and_gain_gradient_parity(self):
        model = ConnectomePolicy(self.operator)
        actual = model.features(self.x)
        expected, cache = self.operator.forward(self.x.numpy())
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-6, rtol=2e-5)
        upstream = torch.randn_like(actual)
        (actual*upstream).sum().backward()
        np.testing.assert_allclose(model.log_gain.grad[:, 0].numpy(),
                                   self.operator.backward(cache, upstream.numpy()), atol=2e-6, rtol=2e-5)

    def test_sparse_backward_numerical(self):
        x = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda v: SparseMessage.apply(v, self.operator), (x,)))

    def test_expressive_encoder_and_internal_gradients(self):
        model = ConnectomePolicy(self.operator, True).double()
        x, context = self.x.double(), self.context.double()
        def objective():
            return model(x, context).square().sum()
        objective().backward()
        for name in ('log_gain', 'bias', 'mix', 'retention', 'encoder.0.weight', 'encoder.2.weight'):
            parameter = dict(model.named_parameters())[name]
            index = int(parameter.grad.abs().flatten().argmax())
            gradient = float(parameter.grad.flatten()[index])
            self.assertGreater(abs(gradient), 1e-12, name)
            with torch.no_grad():
                original = float(parameter.flatten()[index])
                parameter.flatten()[index] = original+1e-5
                plus = float(objective())
                parameter.flatten()[index] = original-1e-5
                minus = float(objective())
                parameter.flatten()[index] = original
            self.assertAlmostEqual(gradient, (plus-minus)/2e-5, delta=1e-6, msg=name)

    def test_optimizer_resume_exact(self):
        first = ConnectomePolicy(self.operator, True)
        optimizer = torch.optim.Adam(first.parameters(), lr=.001)
        legal = torch.ones(2, 256, dtype=torch.bool)
        labels = torch.zeros_like(legal)
        labels[:, 3:6] = True
        def step(model, opt):
            opt.zero_grad()
            equivalent_loss(model(self.x, self.context), legal, labels).backward()
            opt.step()
        step(first, optimizer)
        buffer = io.BytesIO()
        torch.save({'model': first.state_dict(), 'optimizer': optimizer.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        second = ConnectomePolicy(self.operator, True)
        second.load_state_dict(saved['model'])
        other = torch.optim.Adam(second.parameters(), lr=.001)
        other.load_state_dict(saved['optimizer'])
        step(first, optimizer)
        step(second, other)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_equivalent_actions_and_illegal_mask(self):
        logits = torch.tensor([[.2, 100., -.3]], requires_grad=True)
        legal = torch.tensor([[True, False, True]])
        labels = legal.clone()
        loss = equivalent_loss(logits, legal, labels)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 1]), 0.)
        swapped = logits.detach()[:, [2, 1, 0]]
        torch.testing.assert_close(loss.detach(), equivalent_loss(swapped, legal, labels))

    def test_dataset_board_and_visible_disjoint(self):
        data = build_dataset(3, 3, 2)
        train = {r['layout_hash'] for r in data['train']}
        heldout = {r['layout_hash'] for r in data['holdout']}
        games = {r['layout_hash'] for r in data['games']}
        self.assertFalse(train & heldout or train & games or heldout & games)
        self.assertFalse({r['board'].tobytes() for r in data['train']} &
                         {r['board'].tobytes() for r in data['holdout']})
        for split in ('train', 'holdout'):
            for row in data[split]:
                self.assertTrue(row['labels'].any())
                self.assertFalse(np.any(row['labels'] & (row['board'] != -1)))


if __name__ == '__main__':
    unittest.main()
