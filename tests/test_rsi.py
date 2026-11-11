"""Offline end-to-end and failure-recovery contracts for recursive experiments."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from bottle_scan import rsi
from bottle_scan.rsi_methods import baseline_method, validate_method
from tests.test_agentic_campaign import config


def settings(root):
    s = rsi.load_settings(rsi.ROOT / 'config.rsi.yaml')
    s.update(simulate=True, seed=4)
    s['memory']['path'] = str(root / 'memory')
    s['provider']['backend'] = 'mock'
    s['budget'].update(max_cycles=1, max_trials=240)
    s['contact_policy'].update(epochs=25, ensemble_size=3)
    s['comparison'].update(collection_trials_per_branch=12, batch_scan_trials=6, checkpoints=[6, 12])
    return s


def propose(provider_settings, context):
    m = copy.deepcopy(baseline_method())
    for k in ('schema', 'method_id'):
        m.pop(k)
    m.update(name='Diversity and conflict hypothesis', parent_id=context['baseline_method_id'],
             hypothesis='Conflict and coverage may yield more informative data',
             evidence_refs=['acquisition_features'],
             score={'op': 'add', 'args': [{'op': 'feature', 'name': 'action_conflict'},
                                        {'op': 'feature', 'name': 'coverage_gap'}]})
    m['sampling'].update(step_mm=2.5, diversity_mm=2., max_reps_per_cell=4)
    return {'action': 'propose_method', 'method': validate_method(m), 'method_id': None,
            'reason': '离线测试方法，不能据此宣称实机改善', 'audit': {'live_inference': False}}


class RSITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.resolve = patch('bottle_scan.agentic.resolve_base', return_value=config())
        self.resolve.start()
        self.addCleanup(self.resolve.stop)

    def campaign(self, **kw):
        return rsi.RSICampaign(self.root / 'run', settings=settings(self.root),
                               say=lambda _: None, proposer=propose, **kw)

    def test_end_to_end_two_branches_train_evaluate_remember_and_final_holdout(self):
        c = self.campaign()
        result = c.run()
        self.assertEqual(result, 0, c.state.get('last_error'))
        self.assertEqual(c.state['status'], 'complete', c.state.get('last_error'))
        cycle = c.state['cycles'][0]
        self.assertEqual(cycle['phase'], 'complete')
        self.assertEqual(set(cycle['branches']), {'old', 'new'})
        self.assertEqual(cycle['comparison']['old']['spent_trials'], cycle['comparison']['new']['spent_trials'])
        self.assertNotEqual(cycle['result']['outcome'], 'invalid', cycle['result'])
        self.assertTrue(cycle['experience_id'])
        self.assertEqual(len(cycle['checkpoints']), 2)
        self.assertTrue((c.root / 'cycles/001/learning_curve.json').exists())
        self.assertEqual(cycle['learning_curve']['old']['budget_range'], [6, 13])
        for key in ('old', 'new'):
            policies = [c.model(point['branches'][key]['model_id']) for point in cycle['checkpoints']]
            self.assertGreater(policies[1]['training_count'], policies[0]['training_count'])
            self.assertEqual(policies[0]['split']['validation_groups'], policies[1]['split']['validation_groups'])
        final_roots = c.roots([c.state['final_batch']])
        for branch in cycle['branches'].values():
            model = c.model(branch['model_id'])
            self.assertEqual(model['skill_conditions']['version'], 'contact_policy_conditions_v1')
            self.assertEqual(model['skill_conditions']['thresholds']['tol_mm'], c.cfg['servo']['tol_mm'])
            self.assertFalse(set(model['source_sessions']) & set(final_roots + c.roots(branch['evaluation'])))
        self.assertTrue((c.root / 'REPORT.md').exists())
        frozen_model = c.model(c.state['frozen']['model_id'])
        self.assertEqual(c.state['frozen']['skill_conditions'], frozen_model['skill_conditions'])
        self.assertFalse(set(frozen_model['source_sessions']) & set(final_roots))
        report = (c.root / 'REPORT.md').read_text()
        for phrase in ('模型：', '何时调用：', '何时成功退出：', 'contact_policy_conditions_v1'):
            self.assertIn(phrase, report)
        self.assertTrue(list((c.root / 'prompt_buffer').glob('*_decision.json')))
        self.assertLessEqual(c.spent(), c.settings['budget']['max_trials'])
        no_hardware = Mock(side_effect=AssertionError('completed run must not connect'))
        resumed = rsi.RSICampaign(c.root, executor=no_hardware, say=lambda _: None)
        self.assertEqual(resumed.run(), 0)
        no_hardware.assert_not_called()

    def test_pause_after_completed_batch_resume_does_not_repeat(self):
        from bottle_scan.agentic_executor import execute_batch
        calls = []
        def paused(*args, **kw):
            result = execute_batch(*args, **kw)
            calls.append(str(args[0]))
            result['quit'] = True
            return result
        c = self.campaign(executor=paused)
        self.assertEqual(c.run(), 0)
        self.assertEqual(c.state['status'], 'paused')
        first = calls[0]
        def resumed_executor(*args, **kw):
            self.assertNotEqual(str(args[0]), first)
            self.assertTrue(kw['authorize'] if not calls[1:] else True)
            calls.append(str(args[0]))
            return execute_batch(*args, **kw)
        again = rsi.RSICampaign(c.root, say=lambda _: None, executor=resumed_executor, proposer=propose)
        self.assertEqual(again.run(), 0, again.state.get('last_error'))
        self.assertEqual(again.state['status'], 'complete')

    def test_provider_failure_pauses_before_method_collection(self):
        c = self.campaign()
        c.proposer = Mock(side_effect=RuntimeError('network failure'))
        self.assertEqual(c.run(), 2)
        self.assertEqual(c.state['status'], 'paused')
        self.assertEqual(c.state['cycles'], [])
        self.assertTrue(all(b['phase'] == 'bootstrap' for b in c.state['batches']))

    def test_simulated_state_cannot_resume_real(self):
        c = self.campaign()
        messages = []
        self.assertEqual(rsi.main(['resume', '--real', '--out', str(c.root)], output=messages.append), 2)
        self.assertIn('模拟/实机', messages[-1])

    def test_status_is_read_only_and_does_not_resolve_hardware(self):
        c = self.campaign()
        before = c.path.read_bytes()
        executor = Mock(side_effect=AssertionError())
        self.assertEqual(rsi.main(['status', '--out', str(c.root)], output=lambda _: None, executor=executor), 0)
        executor.assert_not_called()
        self.assertEqual(c.path.read_bytes(), before)

    def test_nominal_target_bounds_are_checked(self):
        from bottle_scan.agentic_sim import default_T0
        cfg = config()
        cfg['axes_tcp']['a'] = [0., 0., -1.]
        # Identity TCP rotation makes positive a move down.
        import numpy as np
        T0 = np.eye(4)
        p = rsi.plan_for_cells([{'offset_mm': 20, 'reps': 2}], cfg, cond='A', seed=1)
        with self.assertRaisesRegex(ValueError, '限幅'):
            rsi.check_targets(p, cfg, T0)

    def test_invalid_or_missing_eval_labels_do_not_vanish_from_acceptance(self):
        root = self.root / 'eval'
        root.mkdir()
        c = [{'offset_mm': 0., 'bottle': 'light', 'trials': 2}]
        from bottle_scan.session import atomic_json
        for i, label in enumerate(('normal', 'invalid')):
            f = root / 'trials' / str(i)
            atomic_json(f / 'meta.json', dict(id=str(i), kind='scan', bottle='light', delta_mm=0., complete=True,
                        recording_complete=True, label=label, hold_complete=True, lifted=True, dropped=False))
            atomic_json(f / 'recording_status.json', {'complete': True})
        self.assertFalse(rsi.summarize([root], c)['records_complete'])

    def test_contact_policy_abstention_never_causes_a_zero_length_regrasp(self):
        from bottle_scan.e1e2 import Runner
        import numpy as np
        runner = object.__new__(Runner)
        runner.cfg = config()
        runner.hw = Mock()
        runner.hw.now.return_value = 1
        runner.current = {}
        runner.save = lambda: None
        runner.bilateral = lambda _: True
        runner.release = Mock(side_effect=AssertionError('must leave release to trial cleanup'))
        runner.correction = Mock()
        runner.correction.decide.return_value = {'accepted': False, 'rejected': True, 'delta_mm': 0., 'reason': 'policy_uncertain'}
        target, status, probe = runner.apply_B(np.eye(4), 'contact', {'descriptors': []})
        self.assertEqual(status, 'policy_rejected')
        self.assertIsNone(probe)
        self.assertTrue(runner.current['servo_exhausted'])
        runner.release.assert_not_called()

    def test_resumed_evaluation_attempt_is_not_silently_replaced(self):
        from bottle_scan.session import atomic_json
        root = self.root / 'eval'
        row = dict(id='1', kind='scan', bottle='light', delta_mm=0., complete=True,
                   recording_complete=True, label='normal', hold_complete=True, lifted=True, dropped=False)
        atomic_json(root/'trials/1/meta.json', row)
        atomic_json(root/'trials/1/recording_status.json', {'complete': True})
        cells = [{'offset_mm': 0., 'bottle': 'light', 'trials': 1}]
        self.assertTrue(rsi.summarize([root], cells)['records_complete'])
        atomic_json(root/'interrupted/1_old/meta.json', {'complete': False})
        self.assertFalse(rsi.summarize([root], cells)['records_complete'])


if __name__ == '__main__':
    unittest.main()
