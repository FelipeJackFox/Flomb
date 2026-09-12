import unittest
import numpy as np
from experiments.game_metrics import analyze_move, summarize_moves
from minesweeper import Minesweeper


class GameMetricsTests(unittest.TestCase):
    def test_final_fair_coin_both_outcomes(self):
        move = analyze_move([1, 1, -1, -1], 2, 1, 2)
        self.assertTrue(move['final_5050_candidate'])
        for won in (False, True):
            result = summarize_moves([dict(move, survived=won, revealed_delta=int(won))], won)
            self.assertEqual(result['final_5050_neutral_score_sum'], .5)
            self.assertEqual(result['wins'], int(won))
            self.assertEqual(result['exact_guess_expected_survivals'], .5)

    def test_final_fair_coin_with_proven_mines_and_reused_analysis(self):
        from unittest.mock import patch
        from solver import analyze
        visible = np.array([-1, -1, -1, 2, 1, 3, -1, 2, 0, 1, 1, 1, 0, 0, 0, 0])
        analysis = analyze(np.eye(10)[visible + 1], visible == -1, 4, 3)
        self.assertEqual(analysis.mines, {2, 6})
        with patch('experiments.game_metrics.analyze', side_effect=AssertionError('Duplicate solver')):
            move = analyze_move(visible, 4, 3, 0, analysis=analysis)
        self.assertTrue(move['final_5050_candidate'])
        for won in (False, True):
            summary = summarize_moves([dict(move, survived=won)], won)
            self.assertEqual(summary['final_5050_certified'], 1)
            self.assertEqual(summary['final_5050_neutral_score_sum'], .5)
        mine_move = analyze_move(visible, 4, 3, 2, analysis=analysis)
        self.assertFalse(mine_move['final_5050_candidate'])
        self.assertTrue(mine_move['proven_mine_chosen'])
        with self.assertRaises(ValueError):
            analyze_move([1, 1, -1, -1], 2, 1, 2, analysis=analysis)

    def test_no_certification_with_missing_or_bad_history(self):
        move = dict(analyze_move([1, 1, -1, -1], 2, 1, 2), survived=False)
        for history in ([{'metrics_sampled': True}], [dict(move, logical_error=True)]):
            r = summarize_moves(history + [move], False)
            self.assertEqual(r['final_5050_certified'], 0)
            self.assertEqual(r['final_5050_neutral_score_sum'], 0)

    def test_exact_visible_probabilities_match_oracle_support(self):
        opportunities = 0
        for seed in range(12):
            env = Minesweeper(seed, size=5, mines=3)
            if env.done:
                continue
            action = int(np.flatnonzero(env.legal_mask())[0])
            record = analyze_move(env.visible.copy(), env.size, env.mine_count, action)
            if record['safe_taken']:
                self.assertFalse(env._mines[action])
            if record['proven_mine_chosen']:
                self.assertTrue(env._mines[action])
            opportunities += record['safe_opportunity']
            self.assertEqual(record['interior'], not record['frontier'])
        self.assertGreater(opportunities, 0)

    def test_heuristic_is_not_probability_evidence(self):
        # Patch only the solver interface to exercise the bounded-solver path.
        from unittest.mock import patch
        from solver import Analysis
        info = Analysis(set(), set(), {2: .5, 3: .5}, 'global_density_heuristic')
        with patch('experiments.game_metrics.analyze', return_value=info):
            move = analyze_move([1, 1, -1, -1], 2, 1, 2)
        result = summarize_moves([dict(move, survived=False)], False)
        self.assertIsNone(move['chosen_mine_risk'])
        self.assertEqual(result['exact_moves'], 0)
        self.assertEqual(result['final_5050_certified'], 0)
        self.assertIsNone(result['mean_excess_risk'])

    def test_denominators_and_auto_wins(self):
        result = summarize_moves([], True, automatic_win=True)
        self.assertEqual(result['wins'], 1)
        self.assertEqual(result['policy_episodes'], 0)
        self.assertEqual(result['final_5050_neutral_score_count'], 0)
        self.assertIsNone(result['safe_take_rate'])

    def test_reject_hidden_map_or_invalid_action(self):
        with self.assertRaises(ValueError):
            analyze_move([-1, -1, 9, 1], 2, 1, 0)
        with self.assertRaises(ValueError):
            analyze_move([1, 1, -1, -1], 2, 1, 0)


if __name__ == '__main__':
    unittest.main()
