import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from brain import BrainPolicy
from minesweeper import Minesweeper
from curriculum import mixture, draw_board, encode, decode, reward_step
from train_curriculum import train_episode


class CurriculumTests(unittest.TestCase):
    def test_schedule_decays_and_finishes_large_only(self):
        np.testing.assert_allclose(mixture(0, 50000), [.6, .3, .1])
        previous = mixture(0, 50000)
        for i in range(1, 50000, 113):
            current = mixture(i, 50000)
            self.assertAlmostEqual(current.sum(), 1.)
            self.assertTrue(np.all(current >= 0))
            self.assertLessEqual(current[0], previous[0])
            self.assertLessEqual(current[1], previous[1])
            previous = current
        np.testing.assert_array_equal(mixture(35000, 50000), [0, 0, 1])
        rng = np.random.default_rng(19)
        for i in range(49000, 50000):
            group, size, mines, _ = draw_board(rng, i, 50000)
            self.assertEqual((group, size), ('large', 16))
            self.assertTrue(31 <= mines <= 56)

    def test_padding_mask_and_action_mapping(self):
        for size in (5, 7, 9, 12, 16):
            env = Minesweeper(78, size, round(size*size*.15))
            obs, mask = encode(env)
            self.assertEqual(obs.shape, (2560,))
            self.assertEqual(int(mask.sum()), int(env.legal_mask().sum()))
            self.assertTrue(np.all(obs.reshape(16, 16, 10)[size:] == 0))
            self.assertTrue(np.all(obs.reshape(16, 16, 10)[:, size:] == 0))
            for action in np.flatnonzero(mask):
                self.assertTrue(env.legal_mask()[decode(action, size)])
            if size < 16:
                with self.assertRaises(ValueError):
                    decode(size, size)

    def test_size_independent_reward_bound(self):
        for size in (5, 9, 16):
            env = Minesweeper(66, size, round(size*size*.2))
            total = 0.
            for cell in np.flatnonzero(~env._mines):
                if not env.done and env.visible[cell] == -1:
                    total += reward_step(env, cell)
            self.assertTrue(env.won)
            self.assertLessEqual(total, 1.2500001)
            self.assertGreaterEqual(total, 1.)


class MigrationTests(unittest.TestCase):
    def policy(self):
        rng = np.random.default_rng(9)
        return BrainPolicy(sparse.csr_matrix(rng.normal(0, .08, (60, 60)).astype(np.float32)), readouts=32)

    def test_old_policy_preserved_and_extra_regions_connected(self):
        p = self.policy()
        env = Minesweeper(3)
        before, _ = p.forward(env.observation(), env.legal_mask())
        original_gain = p.log_gain.copy()
        p.expand_canvas()
        obs, mask = encode(env)
        after, _ = p.forward(obs, mask)
        locations = np.arange(25)//5*16 + np.arange(25)%5
        np.testing.assert_allclose(after[locations], before, atol=2e-7)
        np.testing.assert_array_equal(p.log_gain, original_gain)
        large = Minesweeper(3, 16, 40)
        large_obs, large_mask = encode(large)
        probs, cache = p.forward(large_obs, large_mask, True)
        self.assertAlmostEqual(float(probs.sum()), 1., places=6)
        self.assertTrue(np.all(probs[~large_mask] == 0))
        altered = large_obs.copy().reshape(16, 16, 10)
        altered[5:, :, :] = 0
        altered[:, 5:, :] = 0
        _, changed = p.forward(altered.reshape(-1), large_mask, True)
        self.assertGreater(float(np.linalg.norm(cache[0][0]-changed[0][0])), 0)

    def test_expanded_checkpoint_and_episode_update(self):
        p = self.policy()
        p.expand_canvas()
        env = Minesweeper(123, 9, 14)
        obs, mask = encode(env)
        _, _, steps = train_episode(p, env, np.random.default_rng(91), 0.)
        self.assertGreater(steps, 0)
        self.assertTrue(all(np.isfinite(x).all() for x in p.parameters()))
        self.assertGreater(float(np.linalg.norm(p.log_gain)), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.npz'
            p.save(path)
            q = self.policy()
            q.load(path)
            np.testing.assert_array_equal(p.forward(obs, mask)[0], q.forward(obs, mask)[0])


if __name__ == '__main__':
    unittest.main()
