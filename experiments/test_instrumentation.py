import unittest
from unittest.mock import patch
import numpy as np
from minesweeper import Minesweeper
from solver import analyze
from experiments.train import teacher, metric_summary, aggregate_metrics, evaluate
from experiments.game_metrics import analyze_move


class InstrumentationTests(unittest.TestCase):
    def test_teacher_reuses_analysis(self):
        env=Minesweeper(23,5,3)
        info=analyze(env.observation(),env.legal_mask(),5,3)
        expected=teacher(env)
        with patch('experiments.train.analyze',side_effect=AssertionError('duplicate solver call')):
            actual=teacher(env,info)
        np.testing.assert_array_equal(actual[0],expected[0]);self.assertEqual(actual[1],expected[1])

    def test_partial_controller_never_certifies_5050(self):
        move={'action':0,'final_5050_candidate':True,'survived':True,'logical_error':False}
        self.assertEqual(metric_summary([move],True,False,True)['final_5050_certified'],1)
        self.assertEqual(metric_summary([move],True,False,False)['final_5050_certified'],0)

    def test_aggregate_recomputes_denominators(self):
        a=metric_summary([{'action':0,'safe_opportunity':True,'safe_taken':True}],False,False)
        b=metric_summary([{'action':i,'safe_opportunity':True,'safe_taken':False} for i in range(3)],False,False)
        result=aggregate_metrics([a,b]);self.assertEqual(result['safe_take_rate'],.25)
        self.assertEqual(result['episodes'],2)

    def test_evaluation_observational_and_teacher_free(self):
        class Brain:
            def forward(self,x,cache):return np.zeros((len(x),2),np.float32),None
        class Head:
            def forward(self,x):return np.zeros((len(x),256,2),np.float32),None
        with patch('experiments.train.teacher',side_effect=AssertionError('evaluation teacher leak')):
            result=evaluate(Brain(),Head(),1)
        self.assertEqual(len(result),6)
        for row in result.values():
            self.assertFalse(row['teacher']);self.assertFalse(row['exploration'])
            self.assertEqual(row['game_metrics']['analyzed_moves'],row['clicks'])
            self.assertEqual(row['game_metrics']['episodes'],1)

if __name__=='__main__':unittest.main()
