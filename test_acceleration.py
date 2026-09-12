"""Execution optimizations must preserve sampling, gradients, optimizer and resume."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from brain import BrainPolicy
from minesweeper import Minesweeper
from train_curriculum import train_episode


class AccelerationTests(unittest.TestCase):
    def policy(self, workers=1):
        graph = sparse.random(80, 80, density=.2, random_state=6, format='csr', dtype=np.float32)
        return BrainPolicy(graph*.1, inputs=2560, actions=256, readouts=40, sparse_workers=workers)

    def compare(self, p, q):
        self.assertEqual(p.updates, q.updates)
        for a, b in zip(p.parameters()+p.m+p.v, q.parameters()+q.m+q.v):
            np.testing.assert_array_equal(a, b)

    def test_cache_fallback_thread_and_resume_parity(self):
        p, q = self.policy(), self.policy(4)
        self.addCleanup(q.close)
        r1, r2 = np.random.default_rng(73), np.random.default_rng(73)
        for seed in (5, 123, 78, 82):
            e1, e2 = Minesweeper(seed, 16, 40), Minesweeper(seed, 16, 40)
            a = train_episode(p, e1, r1, .12)
            b = train_episode(q, e2, r2, .12, q.activation_cache_bytes()/1024**2)
            self.assertEqual(a, b)
            self.assertEqual(r1.bit_generator.state, r2.bit_generator.state)
            np.testing.assert_array_equal(e1.visible, e2.visible)
            self.compare(p, q)
            self.assertLessEqual(q.last_episode_compute['cached_steps'], 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.npz'
            q.save(path)
            resumed = self.policy(2)
            self.addCleanup(resumed.close)
            resumed.load(path)
            self.compare(q, resumed)
            a = train_episode(q, Minesweeper(143, 9, 10), r2, 0, 1)
            b = train_episode(resumed, Minesweeper(143, 9, 10), r1, 0, 0)
            self.assertEqual(a, b)
            self.compare(q, resumed)

    def test_parallel_gradients_and_cache_budget(self):
        p, q = self.policy(), self.policy(4)
        self.addCleanup(q.close)
        obs = np.random.default_rng(12).normal(size=2560).astype(np.float32)
        mask = np.ones(256, bool)
        a, ca = p.forward(obs, mask, True)
        b, cb = q.forward(obs, mask, True)
        np.testing.assert_array_equal(a, b)
        actual = a.nbytes + sum(s.nbytes for s in ca[0])+ca[1].nbytes+ca[2].nbytes
        self.assertEqual(actual, p.activation_cache_bytes())
        d = -a.copy(); d[3] += 1
        for x, y in zip(p.backward(ca, d), q.backward(cb, d)):
            np.testing.assert_array_equal(x, y)
        # Existing finite-difference tests cover the mathematical derivative.
        with self.assertRaises(ValueError):
            q.configure_compute(0)
        for cap in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                train_episode(p, Minesweeper(123), np.random.default_rng(3), 0, cap)


if __name__ == '__main__':
    unittest.main()
