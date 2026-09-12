import unittest
import numpy as np
from experiments.qr_core import DuelingQuantileHead, quantile_huber, double_dqn_targets, PrioritizedReplay


class QRCoreTests(unittest.TestCase):
    def test_head_numeric_gradient_and_resume(self):
        head = DuelingQuantileHead(3, 2, 3, seed=13)
        rng = np.random.default_rng(12)
        x = rng.normal(size=(2, 3)).astype(np.float32)
        dz = rng.normal(size=(2, 2, 3)).astype(np.float32)
        z, cache = head.forward(x)
        dx, grads = head.backward(cache, dz)
        epsilon = 1e-3
        for p, g in zip(head.params, grads):
            for index in np.ndindex(p.shape):
                original = p[index].copy()
                p[index] = original + epsilon
                plus = float((head.forward(x)[0] * dz).sum())
                p[index] = original - epsilon
                minus = float((head.forward(x)[0] * dz).sum())
                p[index] = original
                self.assertAlmostEqual(float(g[index]), (plus-minus)/(2*epsilon), delta=1e-4)
        for index in np.ndindex(x.shape):
            original = x[index].copy()
            x[index] = original + epsilon
            plus = float((head.forward(x)[0] * dz).sum())
            x[index] = original - epsilon
            minus = float((head.forward(x)[0] * dz).sum())
            x[index] = original
            self.assertAlmostEqual(float(dx[index]), (plus-minus)/(2*epsilon), delta=1e-4)
        head.apply(grads)
        restored = head.clone()
        head.apply(grads)
        restored.apply(grads)
        for a, b in zip(head.params, restored.params):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(head.step, 2)

    def test_quantile_numeric_gradient(self):
        pred = np.array([[-1.8, .2, 1.7], [-.3, .4, 2.1]])
        target = np.array([[-.7, 1.1, 3.2], [-2.2, -.1, .9]])
        weights = np.array([.3, 1.])
        loss, grad, errors = quantile_huber(pred, target, weights)
        for index in np.ndindex(pred.shape):
            original = pred[index]
            pred[index] = original + 1e-6
            plus = quantile_huber(pred, target, weights)[0]
            pred[index] = original - 1e-6
            minus = quantile_huber(pred, target, weights)[0]
            pred[index] = original
            self.assertAlmostEqual(grad[index], (plus-minus)/2e-6, places=8)
        self.assertGreater(loss, 0)
        np.testing.assert_allclose(errors, np.abs(target[:, None, :] - pred[:, :, None]).mean((1, 2)))

    def test_double_target_mask_terminal(self):
        online = np.array([[[99., 99.], [3, 4], [1, 2]], [[0, 0], [0, 0], [0, 0]]])
        target = np.array([[[1., 1.], [5, 7], [100, 200]], [[np.nan, np.nan]] * 3])
        legal = np.array([[False, True, True], [False, False, False]])
        result = double_dqn_targets([1, -1], [False, True], online, target, legal, .5)
        np.testing.assert_array_equal(result, [[3.5, 4.5], [-1, -1]])
        with self.assertRaises(ValueError):
            double_dqn_targets([1, -1], [False, False], online, target, legal)

    def test_replay_sampling_and_exact_resume(self):
        replay = PrioritizedReplay(3, seed=18, alpha=1)
        for i in range(4):
            replay.add({'visible': np.array([i], np.int8)}, priority=i+1)
        self.assertEqual(len(replay), 3)
        self.assertEqual(replay.items[0]['visible'][0], 3)
        restored = PrioritizedReplay(3)
        restored.load_state_dict(replay.state_dict())
        a, idx, weights = replay.sample(10000)
        _, idx2, weights2 = restored.sample(10000)
        np.testing.assert_array_equal(idx, idx2)
        np.testing.assert_array_equal(weights, weights2)
        np.testing.assert_allclose(np.bincount(idx)/len(idx), [4/9, 2/9, 3/9], atol=.001)
        replay.update_priorities([1], [100])
        _, idx, _ = replay.sample(1000)
        self.assertGreater(np.mean(idx == 1), .9)
        with self.assertRaises(ValueError):
            PrioritizedReplay(2).sample(1)


if __name__ == '__main__':
    unittest.main()
