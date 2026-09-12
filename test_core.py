import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from minesweeper import Minesweeper
from brain import BrainPolicy


class EnvironmentTests(unittest.TestCase):
    def test_first_click_and_counts(self):
        for seed in range(100):
            env = Minesweeper(seed)
            self.assertFalse(env._mines[12])
            self.assertEqual(int(env._mines.sum()), 3)
            self.assertTrue(np.all(~env._mines[env.visible >= 0]))
            for i in np.flatnonzero(env.visible >= 0):
                self.assertEqual(env.visible[i], sum(env._mines[j] for j in env.neighbors(i)))

    def test_hidden_state_not_observed(self):
        env = Minesweeper(1)
        before = env.observation()
        mask = env.legal_mask()
        env._mines[:] = ~env._mines
        env._counts[:] = 8
        np.testing.assert_array_equal(before, env.observation())
        np.testing.assert_array_equal(mask, env.legal_mask())

    def test_loss_and_illegal(self):
        env = Minesweeper(10)
        with self.assertRaises(ValueError):
            env.step(12)
        self.assertEqual(env.step(np.flatnonzero(env._mines)[0]), -1.)
        self.assertTrue(env.done)
        self.assertFalse(env.won)

    def test_win(self):
        env = Minesweeper(10)
        for i in np.flatnonzero(~env._mines):
            if not env.done and env.visible[i] == -1:
                env.step(i)
        self.assertTrue(env.won)


class GradientTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.policy = BrainPolicy(sparse.csr_matrix(rng.normal(0, .2, (12, 12)).astype(np.float32)),
                                  inputs=6, actions=3, readouts=8, cycles=3)
        self.obs = rng.normal(size=6).astype(np.float32)
        self.legal = np.array([True, False, True])

    def test_analytical_gradient_against_finite_difference(self):
        policy = self.policy
        probs, cache = policy.forward(self.obs, self.legal, True)
        self.assertEqual(probs[1], 0.)
        score = -probs.copy()
        score[0] += 1.
        gradients = policy.backward(cache, score)
        for param, grad in zip(policy.parameters(), gradients):
            indices = list(np.ndindex(param.shape))
            for idx in indices[::max(1, len(indices)//6)]:
                original, eps = param[idx].copy(), .002
                param[idx] = original + eps
                a = np.log(policy.forward(self.obs, self.legal)[0][0])
                param[idx] = original - eps
                b = np.log(policy.forward(self.obs, self.legal)[0][0])
                param[idx] = original
                self.assertAlmostEqual(float(grad[idx]), float((a-b)/(2*eps)), delta=.001)

    def test_checkpoint_and_internal_update(self):
        p = self.policy
        probs, cache = p.forward(self.obs, self.legal, True)
        score = -probs.copy(); score[0] += 1
        p.apply(p.backward(cache, score))
        self.assertGreater(float(np.linalg.norm(p.log_gain)), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.npz'
            p.save(path)
            q = BrainPolicy(p.graph, inputs=6, actions=3, readouts=8)
            q.load(path)
            np.testing.assert_array_equal(p.forward(self.obs, self.legal)[0], q.forward(self.obs, self.legal)[0])
            for a, b in zip(p.m, q.m):
                np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
