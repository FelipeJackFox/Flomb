import inspect
import unittest
import numpy as np
import solver
from minesweeper import Minesweeper


class SolverTests(unittest.TestCase):
    def test_subset_and_full_zero(self):
        safe, mines, _ = solver.deduce([({0, 1}, 1), ({0, 1, 2}, 1), ({3}, 1)])
        self.assertIn(2, safe)
        self.assertIn(3, mines)

    def test_global_deduction(self):
        safe, mines, _ = solver.deduce([({0, 1, 2, 3}, 1), ({0, 1}, 1)])
        self.assertTrue({2, 3} <= safe)

    def test_exact_probability(self):
        visible = np.full(9, -1)
        visible[4] = 2
        obs = np.eye(10)[visible+1].reshape(-1)
        info = solver.analyze(obs, visible == -1, 3, 2)
        self.assertEqual(info.probability_method, 'exact_uniform_consistent_boards')
        self.assertFalse(info.safe)
        for prob in info.mine_probability.values():
            self.assertAlmostEqual(prob, .25)
        heuristic = solver.analyze(obs, visible == -1, 3, 2, exact_limit=0)
        self.assertEqual(heuristic.probability_method, 'global_density_heuristic')

    def test_exact_global_outside_weighting(self):
        visible = np.full(16, -1)
        visible[0] = 1
        info = solver.analyze(np.eye(10)[visible+1].reshape(-1), visible == -1, 4, 3)
        self.assertAlmostEqual(info.mine_probability[1], 1/3)
        self.assertAlmostEqual(info.mine_probability[15], 2/12)
        self.assertAlmostEqual(sum(info.mine_probability.values()), 3)

    def test_no_hidden_state_api_and_legal_actions(self):
        source = inspect.getsource(solver)
        self.assertNotIn('env.', source)
        self.assertNotIn('._mines', source)
        self.assertNotIn('._counts', source)
        for seed in range(40):
            size, mines = (5, 3) if seed < 20 else (16, 40)
            env = Minesweeper(seed, size, mines)
            rng = np.random.default_rng(seed)
            while not env.done:
                obs, legal = env.observation(), env.legal_mask()
                action, info = solver.choose(obs, legal, size, mines, rng)
                self.assertTrue(legal[action])
                # Hidden state is used exclusively as a test oracle, never solver input.
                self.assertTrue(all(not env._mines[i] for i in info.safe))
                self.assertTrue(all(env._mines[i] for i in info.mines))
                result = env.step(action)
                if info.classify(action) == 'proven_safe':
                    self.assertNotEqual(result, -1)

    def test_reject_mismatched_mask(self):
        obs = np.eye(10)[np.zeros(9, dtype=int)].reshape(-1)
        with self.assertRaises(ValueError):
            solver.analyze(obs, np.zeros(9, bool), 3, 2)


if __name__ == '__main__':
    unittest.main()
