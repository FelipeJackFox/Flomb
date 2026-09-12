import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import dashboard_data as d


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def lines(path, data):
    path.write_text(''.join(json.dumps(r)+'\n' for r in data))


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'runs'
        self.run = self.root / 'suite' / 'dagger'
        write(self.run/'config.json', {'mode':'dagger'})
        d._CACHE.clear()

    def tearDown(self):
        self.temp.cleanup()

    def test_registry_security_and_capabilities(self):
        write(self.root/'curriculum-001/config.json', {'total':50000})
        write(self.root/'direct/config.json', {'mode':'hybrid'})
        write(self.root/'ignored/config.json', {'mode':'other'})
        outside = Path(self.temp.name)/'outside'
        write(outside/'config.json', {'mode':'qrdqn'})
        (self.root/'escaped').symlink_to(outside, target_is_directory=True)
        self.assertEqual(set(d.registry(self.root)), {'curriculum-001','direct','suite/dagger'})
        for value in ('../outside', '/tmp/foo', 'suite/../direct','suite//dagger','escaped','ignored'):
            with self.subTest(value=value), self.assertRaises(ValueError): d.resolve_run(self.root, value)
        self.assertEqual(d.resolve_run(self.root,'suite/dagger'),self.run.resolve())
        self.assertFalse(next(r for r in d.build_dashboard(self.root)['runs'] if r['id']=='suite/dagger')['control_supported'])
        write(self.run/'capabilities.json', {'runtime_curriculum':True})
        self.assertTrue(next(r for r in d.build_dashboard(self.root)['runs'] if r['id']=='suite/dagger')['control_supported'])

    def test_dedup_old_new_normalization_and_automatic_exclusion(self):
        first = self.run/'metrics-from-old.jsonl'
        last = self.run/'metrics-new.jsonl'
        lines(first,[{'episode':1,'size':5,'mines':3,'won':False,'steps':1,'total_clicks':1},
                     {'episode':2,'size':5,'mines':3,'won':False}])
        lines(last,[{'episode':1,'size':5,'mines':3,'won':True,'automatic_win':True,'steps':0,'total_clicks':0},
                    {'episode':2,'size':5,'mines':3,'won':True,'episode_steps':10,'steps':10,
                     'teacher_actions':4,'loss':{'qr':.2,'teacher':.3},'training_seconds':4,'peak_rss_bytes_macos':42},
                    {'episode':3,'size':5,'mines':3,'won':False,'steps':3,'total_clicks':13,
                     'training_wall_seconds':6,'safe_fraction':.7,'return':-.3}])
        os.utime(first, ns=(1000,1000))
        data=d.build_dashboard(self.root)['runs'][0]
        self.assertEqual(data['metric_episodes'],3)
        a,b,c=data['series']['5x5-3']
        self.assertIsNone(a['win_rate'])
        self.assertEqual(a['automatic_wins'],1)
        self.assertEqual(b['win_rate'],1)
        self.assertEqual(b['teacher_fraction'],.4)
        self.assertEqual(b['loss_qr'],.2)
        self.assertEqual(b['rss'],42)
        self.assertEqual(c['episode_steps'],3)
        self.assertEqual(c['steps'],13)
        self.assertEqual(c['training_seconds'],6)
        self.assertIsNone(c['teacher_fraction'])

    def test_bucket_game_ratios_and_controllers(self):
        rows=[]
        for episode in range(1,301):
            gm={'safe_opportunities':1 if episode%2 else 9,'safe_taken':1,
                'safe_take_rate':1 if episode%2 else 1/9,
                'analyzed_moves':10,'exact_moves':5,'excess_risk_sum':1}
            rows.append({'episode':episode,'size':16,'mines':40,'won':episode%2==0,
                         'episode_steps':episode,'steps':episode*100,'game_metrics':gm,
                         'controller_metrics':{'policy':gm}})
        lines(self.run/'metrics.jsonl',rows)
        data=d.build_dashboard(self.root)['runs'][0]
        self.assertEqual(len(data['overall_series']),150)
        point=data['series']['16x16-40'][0]
        self.assertEqual(point['episodes'],2)
        self.assertEqual(point['win_rate'],.5)
        ratio=point['game_metrics']['ratios']['safe_take_rate']
        self.assertEqual(ratio,{'numerator':2,'denominator':10,'value':.2})
        self.assertEqual(point['controller_metrics']['policy']['ratios']['safe_take_rate'],ratio)
        self.assertIsNone(point['controller_metrics']['random'])
        self.assertNotIn('safe_take_rate',point['game_metrics']['counters'])

    def test_evaluations_partial_lines_and_final(self):
        result={'5x5-3':{'episodes':5,'wins':3,'automatic_wins':1,'mean_safe_fraction':.8}}
        lines(self.run/'evaluations.jsonl',[{'episode':10,'results':result},{'episode':10,'results':result}])
        with (self.run/'evaluations.jsonl').open('a') as f:f.write('{unfinished')
        write(self.run/'completed.json',{'episodes':20})
        write(self.run/'evaluation-final.json',result)
        data=d.build_dashboard(self.root)['runs'][0]
        self.assertEqual(len(data['evaluations']),2)
        final=data['evaluations'][-1]
        self.assertEqual(final['episode'],20)
        self.assertEqual(final['kind'],'final')
        self.assertEqual(final['results']['5x5-3']['win_rate'],.5)

    def test_common_bucket_end_and_unmeasured_diagnostics(self):
        gm={'policy_episodes':1,'final_5050_neutral_score_sum':1,
            'final_5050_neutral_score_count':1,'analyzed_moves':1}
        rows=[{'episode':1,'size':5,'mines':3,'won':True,'metrics_measured':False,
               'game_metrics':gm,'controller_metrics':{'policy':gm},
               'timing':{'metrics_seconds':.3},'epsilon':.8,'teacher_beta':.9},
              {'episode':2,'size':7,'mines':7,'won':False,'timing_metrics_seconds':.1},
              {'episode':300,'size':5,'mines':3,'won':False}]
        lines(self.run/'metrics.jsonl',rows)
        data=d.build_dashboard(self.root)['runs'][0]
        first=data['series']['5x5-3'][0]
        other=data['series']['7x7-7'][0]
        overall=data['overall_series'][0]
        self.assertEqual(first['bucket_end'],other['bucket_end'])
        self.assertEqual(first['bucket_end'],overall['bucket_end'])
        self.assertIsNone(first['game_metrics'])
        self.assertIsNone(first['controller_metrics']['policy'])
        self.assertAlmostEqual(overall['timing_metrics_seconds'],.2)
        self.assertEqual(first['epsilon'],.8)
        self.assertEqual(first['teacher_beta'],.9)

    def test_size_series_aggregates_counts_and_before_evaluators_stay_separate(self):
        lines(self.run/'metrics.jsonl',[
            {'episode':1,'size':16,'mines':40,'won':True},
            {'episode':2,'size':16,'mines':56,'won':False},
            {'episode':300,'size':5,'mines':3,'won':True}])
        result={'16x16-40':{'episodes':8,'wins':3,'automatic_wins':0}}
        write(self.run/'evaluation-before.json',{'random':result,'brain':result})
        lines(self.run/'evaluations.jsonl',[{'episode':300,'results':result}])
        write(self.run/'completed.json',{'episodes':300})
        write(self.run/'evaluation-final.json',result)
        data=d.build_dashboard(self.root)['runs'][0]
        point=data['size_series']['16x16'][0]
        self.assertEqual(point['episodes'],2)
        self.assertEqual(point['win_rate'],.5)
        self.assertIsNone(point['mines'])
        self.assertEqual(set(data['size_series']),{'5x5','16x16'})
        self.assertEqual(set(e['kind'] for e in data['evaluations']),
                         {'before/random','before/brain','periodic','final'})
        self.assertEqual(data['evaluations'][-1]['kind'],'final')
        self.assertTrue(all('16x16-40' in e['results'] for e in data['evaluations']))

    def test_cache_metric_reads_independent_of_progress(self):
        lines(self.run/'metrics.jsonl',[{'episode':1,'size':5,'mines':3,'won':False}])
        d.build_dashboard(self.root)
        with patch.object(d,'_aggregate',wraps=d._aggregate) as aggregate:
            data=d.build_dashboard(self.root)
            data['runs'][0]['series'].clear()
            write(self.run/'progress.json',{'episode':2})
            fresh=d.build_dashboard(self.root)
            self.assertTrue(fresh['runs'][0]['series'])
            self.assertEqual(fresh['runs'][0]['progress']['episode'],2)
            aggregate.assert_not_called()
            with (self.run/'metrics.jsonl').open('a') as f:f.write(json.dumps({'episode':2,'size':5,'mines':3,'won':True})+'\n')
            d.build_dashboard(self.root)
            aggregate.assert_called_once()


if __name__=='__main__':unittest.main()
