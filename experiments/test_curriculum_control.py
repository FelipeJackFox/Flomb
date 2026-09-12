import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from curriculum import draw_board
from experiments.curriculum_control import (STRATA, applied_control, request_control,
                                           sample_board, validate_control)


class CurriculumControlTests(unittest.TestCase):
    def test_scheduled_rng_equivalence(self):
        with tempfile.TemporaryDirectory() as run:
            for explicit in (False, True):
                if explicit:
                    request_control(run, {'mode': 'scheduled', 'weights': [1]*6})
                left = np.random.default_rng(51)
                right = np.random.default_rng(51)
                for episode in range(100):
                    old = draw_board(left, episode, 100)
                    new = sample_board(right, episode, 100, run)
                    self.assertEqual(old[:3], new[:3])
                    np.testing.assert_array_equal(old[3], new[3])
                    self.assertEqual(left.bit_generator.state, right.bit_generator.state)
                    self.assertEqual(new[4]['mode'], 'scheduled')

    def test_validation(self):
        bad = [[0]*6, [float('nan')]*6, [float('inf')]*6, [-1,1,1,1,1,1],
               [True]*6, ['1']*6, [1]*5]
        for weights in bad:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                validate_control({'mode': 'manual', 'weights': weights})
        for payload in ({'mode':'bad','weights':[1]*6},
                        {'mode':'manual','weights':[1]*6,'path':'/tmp/other'},
                        {'mode':'manual'}):
            with self.assertRaises(ValueError): validate_control(payload)
        result = validate_control({'mode':'manual','weights':[1e308]*6})
        self.assertAlmostEqual(sum(result['weights']), 1)

    def test_manual_sampling_each_stratum(self):
        with tempfile.TemporaryDirectory() as run:
            for index, expected in enumerate(STRATA):
                weights = [0]*6
                weights[index] = 5
                requested = request_control(run, {'mode':'manual','weights':weights})
                group, size, mines, mix, meta = sample_board(np.random.default_rng(1), 99, 100, run)
                self.assertEqual((group,size,mines), expected)
                self.assertEqual(meta['revision'], requested['revision'])
                self.assertEqual(meta['weights'][index], 1)
                self.assertEqual(mix[index//2], 1)

    def test_revisions_and_explicit_adoption(self):
        with tempfile.TemporaryDirectory() as run:
            first = request_control(run, {'mode':'manual','weights':[1]*6})
            meta = sample_board(np.random.default_rng(1), 0, 100, run)[4]
            second = request_control(run, {'mode':'manual','weights':[0,0,0,0,1,1]})
            self.assertFalse((Path(run)/'control-applied.json').exists())
            ack = applied_control(run, meta, 10)
            self.assertEqual(ack['revision'], first['revision'])
            self.assertEqual(json.loads((Path(run)/'control.json').read_text())['revision'], second['revision'])
            newer = sample_board(np.random.default_rng(1), 1, 100, run)[4]
            self.assertEqual(applied_control(run, newer, 11)['revision'], 2)
            self.assertEqual(applied_control(run, meta, 12)['revision'], 2)
            self.assertEqual(len((Path(run)/'control-audit.jsonl').read_text().splitlines()), 2)

    def test_concurrent_requests_are_monotonic(self):
        with tempfile.TemporaryDirectory() as run:
            with ThreadPoolExecutor(4) as pool:
                states = list(pool.map(lambda _: request_control(run, {'mode':'manual','weights':[1]*6}), range(12)))
            self.assertEqual(sorted(s['revision'] for s in states), list(range(1,13)))
            logs = [json.loads(line) for line in (Path(run)/'control-audit.jsonl').read_text().splitlines()]
            self.assertEqual([s['revision'] for s in logs], list(range(1,13)))


if __name__ == '__main__':
    unittest.main()
