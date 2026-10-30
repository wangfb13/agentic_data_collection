"""Offline campaign orchestration contracts; no ROS or physical hardware."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bottle_scan import agentic
from bottle_scan.agentic_sim import default_T0
from bottle_scan.e1e2_core import defaults
from bottle_scan.session import atomic_json


def settings():
    value = agentic.load_settings(agentic.ROOT / 'config.agentic.yaml')
    value['simulate'] = True
    value['provider']['backend'] = 'deterministic'
    return value


def config():
    value = defaults()
    value['gripper']['min_normalized'] = .336
    return value


def candidate(cond='A', *, reps=4, model_id=None):
    value = {'id': 'candidate-1', 'cond': cond, 'cells': [
        {'offset_mm': d, 'reps': reps} for d in (-2., 0., 10.)],
        'reason': 'offline contract test', 'policy': {'gain': 1.}}
    if model_id is not None:
        value['model_id'] = model_id
    return value


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'campaign'
        self.resolve = patch('bottle_scan.agentic.resolve_base', return_value=config())
        self.resolve.start()
        self.addCleanup(self.resolve.stop)

    def campaign(self, **kwargs):
        return agentic.Campaign(self.root, settings=settings(), say=lambda _: None, **kwargs)

    def model(self, campaign):
        fit = {'model_id': 'm1', 'usable_for_B': True}
        atomic_json(campaign.root / 'models/m1.json', fit)
        campaign.state['models']['m1'] = 'models/m1.json'
        return fit

    def test_plan_is_reproducible_balanced_and_counts_monitors(self):
        plan = agentic.make_plan(candidate(), config(), seed=5)
        self.assertEqual(plan, agentic.make_plan(candidate(), config(), seed=5))
        self.assertEqual(len(plan), 13)
        self.assertEqual(sum(p['kind'] == 'zero_monitor' for p in plan), 1)
        for bottle in config()['design']['bottles']:
            for offset in (-2., 0., 10.):
                self.assertEqual(sum(p['bottle'] == bottle and p['delta_mm'] == offset and
                                     p['kind'] == 'scan' for p in plan), 2)
        self.assertEqual(len({p['id'] for p in plan}), len(plan))

    def test_validation_contains_paired_A_B_and_excludes_monitors_from_pairs(self):
        s = settings()
        plan = agentic.make_plan(agentic.validation_candidate(s), config(), seed=5, validation=True)
        self.assertEqual(len(plan), 44)
        scans = [p for p in plan if p['kind'] == 'scan']
        for pair in zip(scans[::2], scans[1::2]):
            self.assertEqual({p['cond'] for p in pair}, {'A', 'B'})
            for key in ('bottle', 'delta_mm', 'repeat'):
                self.assertEqual(pair[0][key], pair[1][key])

    def test_candidate_budget_counts_zero_monitors(self):
        with self.assertRaisesRegex(ValueError, '监测'):
            agentic.validate_candidate(candidate(), settings(), config(), 12, {})
        self.assertEqual(len(agentic.validate_candidate(candidate(), settings(), config(), 13, {})), 13)

    def test_candidates_cannot_change_scope_models_or_policy(self):
        for field, value in [('offset_mm', 25.), ('reps', 3), ('reps', True)]:
            c = candidate(); c['cells'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                agentic.validate_candidate(c, settings(), config(), 120, {})
        with self.assertRaisesRegex(ValueError, '模型'):
            agentic.validate_candidate(candidate('B', model_id='unapproved'), settings(), config(), 120, {})
        for policy in ({'gain': .1}, {'gain': 1., 'force_target': 100}):
            c = candidate('B', model_id='m'); c['policy'] = policy
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, '策略'):
                agentic.validate_candidate(c, settings(), config(), 120, {'m': 'model.json'})

    def test_agent_decisions_are_finite_candidate_selection_only(self):
        c = candidate('B', model_id='m')
        valid = dict(action='select_batch', candidate_id=c['id'], model_id='m', reason='test')
        self.assertIs(agentic.validate_decision(valid, [c], {'eligible': False}), c)
        for change in ({'candidate_id': 'invented'}, {'model_id': 'other'}, {'force_target': 100},
                       {'action': 'move'}, {'reason': ''}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                agentic.validate_decision({**valid, **change}, [c], {'eligible': False})
        with self.assertRaisesRegex(ValueError, '停止条件'):
            agentic.validate_decision({'action': 'freeze', 'reason': 'done'}, [c], {'eligible': False})
        self.assertIsNone(agentic.validate_decision({'action': 'pause', 'reason': 'uncertain'}, [c], {'eligible': False}))

    def test_invalid_numeric_settings_rejected(self):
        for path, value in [(('scope', 'min_mm'), -26), (('scope', 'step_mm'), float('nan')),
                            (('budget', 'max_trials'), True), (('validation', 'reps'), 3),
                            (('policy', 'gains'), [.1]), (('learning', 'plateau_batches'), 0)]:
            s = settings(); s[path[0]][path[1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                agentic.validate_settings(s)

    def test_validation_reserved_before_exploration(self):
        c = self.campaign()
        self.assertEqual(c.state['validation_reserved_trials'], 44)
        with patch('bottle_scan.agentic_learning.propose_candidates', return_value=[]) as propose:
            c.candidates({})
        self.assertEqual(propose.call_args.args[2], 76)
        s = settings(); s['budget']['max_trials'] = 44
        with self.assertRaisesRegex(ValueError, '预留'):
            agentic.Campaign(Path(self.temp.name) / 'insufficient', settings=s, say=lambda _: None)

    def test_resume_settings_are_immutable(self):
        c = self.campaign()
        altered = settings(); altered['scope']['max_mm'] = 12.
        with self.assertRaisesRegex(ValueError, '续跑不能更改'):
            agentic.Campaign(self.root, settings=altered)
        resumed = agentic.Campaign(self.root)
        self.assertEqual(c.state['base_config'], resumed.state['base_config'])
        self.assertEqual(c.state['settings'], resumed.state['settings'])

    def test_seed_metadata_edits_rejected_on_resume(self):
        seed = Path(self.temp.name) / 'seed'
        atomic_json(seed / 'session_meta.json', {'simulated': True, 'T0': default_T0().tolist()})
        atomic_json(seed / 'trials/00001/meta.json', {'label': 'normal'})
        s = settings(); s['seed_sessions'] = [str(seed)]
        agentic.Campaign(self.root, settings=s, say=lambda _: None)
        atomic_json(seed / 'trials/00001/meta.json', {'label': 'invalid'})
        with self.assertRaisesRegex(ValueError, '种子数据已被更改'):
            agentic.Campaign(self.root)

    def test_consumed_seed_TCP_data_is_pinned_but_unconsumed_images_are_not(self):
        try:
            import h5py
        except ImportError:
            self.skipTest('h5py is installed in the robot/offline integration environment')
        seed = Path(self.temp.name) / 'seed'
        atomic_json(seed / 'session_meta.json', {'simulated': True})
        atomic_json(seed / 'trials/00001/meta.json', {'label': 'normal'})
        path = seed / 'trials/00001/streams.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('streams/tcp_measured/data', data=np.eye(4)[None])
            handle.create_dataset('streams/tcp_measured/monotonic_ns', data=[1])
            handle.create_dataset('streams/wrist/data', data=[0])
        before = agentic.source_manifest([seed])
        with h5py.File(path, 'a') as handle:
            handle['streams/wrist/data'][0] = 1
        self.assertEqual(before, agentic.source_manifest([seed]))
        with h5py.File(path, 'a') as handle:
            handle['streams/tcp_measured/data'][0, 0, 3] = 1.
        self.assertNotEqual(before, agentic.source_manifest([seed]))

    def test_hardware_config_change_rejected_before_executor(self):
        hardware = Path(self.temp.name) / 'hardware.yaml'; hardware.write_text('tcp: original\n')
        cfg = config(); cfg['hardware_config'] = str(hardware)
        s = settings(); s['simulate'] = False
        execute = Mock(side_effect=AssertionError('configuration changed'))
        with patch('bottle_scan.agentic.resolve_base', return_value=cfg):
            c = agentic.Campaign(self.root, settings=s, executor=execute, say=lambda _: None)
        batch = c.prepare_batch(candidate())
        hardware.write_text('tcp: modified\n')
        with self.assertRaisesRegex(ValueError, '硬件配置'):
            c.execute(batch)
        execute.assert_not_called()

    def test_simulated_seed_cannot_seed_real_campaign(self):
        seed = Path(self.temp.name) / 'seed'
        atomic_json(seed / 'session_meta.json', {'simulated': True, 'T0': default_T0().tolist()})
        s = settings(); s['simulate'] = False; s['seed_sessions'] = [str(seed)]
        with self.assertRaisesRegex(ValueError, '种子不能混用'):
            agentic.Campaign(self.root, settings=s, say=lambda _: None)

    def test_completed_campaign_never_invokes_executor_or_analyzer(self):
        execute = Mock(side_effect=AssertionError('must not connect'))
        c = self.campaign(executor=execute)
        c.state['status'] = 'complete'; c.save()
        with patch.object(c, 'analyze', side_effect=AssertionError('no remaining work')):
            self.assertEqual(c.run(), 0)
        execute.assert_not_called()

    def test_cli_resume_cannot_change_mode_or_budget(self):
        self.campaign()
        for extra in (['--real'], ['--simulate', '--max-trials', '140']):
            execute = Mock(side_effect=AssertionError('must not connect'))
            result = agentic.main(['resume', '--out', str(self.root), *extra], output=lambda _: None, executor=execute)
            self.assertEqual(result, 2)
            execute.assert_not_called()

    def test_interrupted_attempts_count_against_pending_exploration_budget(self):
        execute = Mock(side_effect=AssertionError('budget must reject before execution'))
        c = self.campaign(executor=execute); batch = c.prepare_batch(candidate())
        for i in range(64):
            atomic_json(c.root / batch['path'] / f'interrupted/{i:04d}/meta.json', {'complete': False})
        self.assertEqual(c._spent(), 64)
        with self.assertRaisesRegex(ValueError, '中断尝试'):
            c.execute(batch)
        execute.assert_not_called()

    def test_resume_counts_completed_trials_only_once(self):
        execute = Mock(return_value={'complete': True, 'quit': False, 'context': {}})
        c = self.campaign(executor=execute); batch = c.prepare_batch(candidate())
        # 63 failed attempts + 13 planned = the exploration ceiling of 76.
        for i in range(63):
            atomic_json(c.root / batch['path'] / f'interrupted/{i:04d}/meta.json', {'complete': False})
        atomic_json(c.root / batch['path'] / 'trials/00001/meta.json', {'id': '00001', 'complete': True})
        c.execute(batch)
        execute.assert_called_once()

    def test_executor_gets_role_authorization_deadline_and_fixed_config(self):
        captured = []
        def execute(root, plan, cfg, fit, context, **kwargs):
            captured.append((copy.deepcopy(cfg), copy.deepcopy(context), kwargs))
            return {'complete': True, 'quit': False, 'context': context}
        c = self.campaign(executor=execute); before = copy.deepcopy(c.cfg)
        c.state['elapsed_s'] = 10.; c.state['active_run_unix_ns'] = time.time_ns()
        batch = c.prepare_batch(candidate()); c.execute(batch)
        second = candidate(); second['id'] = 'second'; c.execute(c.prepare_batch(second))
        self.assertEqual(before, c.cfg)
        self.assertEqual(captured[0][1]['batch_role'], 'exploration')
        self.assertFalse(captured[0][1]['continuous_campaign'])
        self.assertTrue(captured[1][1]['continuous_campaign'])
        self.assertTrue(captured[0][2]['authorize']); self.assertFalse(captured[1][2]['authorize'])
        remaining = (captured[0][1]['deadline_unix_ns'] - time.time_ns()) / 1e9
        self.assertGreater(remaining, 14380.); self.assertLessEqual(remaining, 14390.)

    def test_execute_accepts_cleared_active_run_timestamp(self):
        execute = Mock(return_value={'complete': True, 'quit': False, 'context': {}})
        c = self.campaign(executor=execute)
        c.state['active_run_unix_ns'] = None
        c.execute(c.prepare_batch(candidate()))
        execute.assert_called_once()

    def test_validation_executor_role_and_frozen_gain(self):
        execute = Mock(return_value={'complete': True, 'quit': False, 'context': {}})
        c = self.campaign(executor=execute); self.model(c)
        c.state['frozen'] = {'model_id': 'm1', 'policy': {'gain': .75}}
        batch = c.prepare_batch(agentic.validation_candidate(c.settings), validation=True)
        c.execute(batch)
        args = execute.call_args.args
        self.assertEqual(args[4]['batch_role'], 'validation')
        self.assertEqual(args[2]['servo']['gain'], .75)
        self.assertEqual(len(args[1]), 44)

    def test_q_on_last_validation_trial_finishes_on_resume_without_duplicate_batch(self):
        execute = Mock(return_value={'complete': True, 'quit': True, 'context': {}})
        c = self.campaign(executor=execute); self.model(c)
        c.state.update(phase='validation', frozen={'model_id': 'm1', 'policy': {'gain': 1.}, 'reason': 'test'})
        c.prepare_batch(agentic.validation_candidate(c.settings), validation=True)
        self.assertEqual(c.run(), 0)
        self.assertEqual(c.state['status'], 'paused'); self.assertIsNone(c.state['active_batch'])
        self.assertEqual(len(c.state['batches']), 1)
        resumed = agentic.Campaign(self.root, executor=Mock(side_effect=AssertionError('no next batch')), say=lambda _: None)
        def finish(): resumed.state['status'] = 'complete'
        with patch.object(resumed, 'finish_validation', side_effect=finish) as finalize:
            self.assertEqual(resumed.run(), 0)
        finalize.assert_called_once(); self.assertEqual(len(resumed.state['batches']), 1)
        resumed.executor.assert_not_called()

    def test_recovery_requires_explicit_operator_assertion_and_logs_no_motion(self):
        c = self.campaign(ask=lambda _: 'RECOVERED', executor=Mock(side_effect=AssertionError('no motion')))
        batch = c.prepare_batch(candidate()); path = c.root / batch['path']
        context = {'T0': default_T0().tolist(), 'airborne': True, 'unsafe_recovery': True}
        c.state['context'] = copy.deepcopy(context)
        atomic_json(path / 'session_meta.json', {'T0': context['T0'], 'agentic_context': context})
        atomic_json(path / 'progress.json', {'airborne': True, 'unsafe_recovery': True, 'completed': []})
        c.acknowledge_recovery()
        self.assertFalse(c.state['context']['airborne']); self.assertFalse(c.state['context']['unsafe_recovery'])
        self.assertFalse(json.loads((path / 'progress.json').read_text())['airborne'])
        records = list((c.root / 'recovery_acknowledgements').glob('*.json'))
        self.assertEqual(len(records), 1)
        event = json.loads(records[0].read_text())
        self.assertFalse(event['motion_sent']); self.assertTrue(event['campaign_context_before']['airborne'])
        c.executor.assert_not_called()

    def test_recovery_cancel_keeps_airborne_latch(self):
        c = self.campaign(ask=lambda _: '')
        batch = c.prepare_batch(candidate()); path = c.root / batch['path']
        context = {'T0': default_T0().tolist(), 'airborne': True, 'unsafe_recovery': True}
        c.state['context'] = context
        atomic_json(path / 'session_meta.json', {'T0': context['T0'], 'agentic_context': context})
        c.acknowledge_recovery()
        self.assertTrue(c.state['context']['airborne'])
        self.assertFalse((c.root / 'recovery_acknowledgements').exists())

    def test_recovery_rejects_mismatched_fixed_T0(self):
        c = self.campaign(); batch = c.prepare_batch(candidate())
        c.state['context']['T0'] = default_T0().tolist()
        changed = default_T0(); changed[0, 3] += 1.
        atomic_json(c.root / batch['path'] / 'session_meta.json', {'T0': changed.tolist()})
        with self.assertRaisesRegex(ValueError, 'T0'):
            c.recover_context()

    def test_seed_and_exploration_only_are_training_sources(self):
        c = self.campaign(); self.model(c)
        explore = c.prepare_batch(candidate())
        c.state['frozen'] = {'model_id': 'm1', 'policy': {'gain': 1.}}
        validation = c.prepare_batch(agentic.validation_candidate(c.settings), validation=True)
        for batch in (explore, validation):
            atomic_json(c.root / batch['path'] / 'session_meta.json', {})
        self.assertEqual(c.training_sessions(), [str(c.root / explore['path'])])

    def test_mapping_stop_requires_successful_B_on_current_model(self):
        c = self.campaign()
        report = {'best_model': {'model_id': 'm2'}, 'stopping_eligible': True}
        batch = c.prepare_batch(candidate('B', model_id='m1')); batch['status'] = 'complete'
        with patch('bottle_scan.e1e2.rebuild_summary', return_value=[
                dict(kind='scan', complete=True, recording_complete=True, label='normal', success=True)]):
            self.assertFalse(c.stop_evidence(report)['eligible'])
            batch['model_id'] = 'm2'
            self.assertTrue(c.stop_evidence(report)['eligible'])
        with patch('bottle_scan.e1e2.rebuild_summary', return_value=[
                dict(kind='scan', complete=True, recording_complete=True, label='dropped', success=False)]):
            self.assertFalse(c.stop_evidence(report)['eligible'])

    def test_untrusted_decision_error_pauses_without_fallback_execution(self):
        provider = Mock()
        provider.propose.return_value = {'action': 'select_batch', 'candidate_id': 'candidate-1',
                                         'reason': 'unsafe', 'force_target': 100}
        c = self.campaign(executor=Mock(side_effect=AssertionError('must not execute')), provider=provider)
        c.settings['provider']['backend'] = 'chatgpt'
        report = {'quality': {'blocking_reasons': []}, 'metrics': {}}
        with patch.object(c, 'analyze', return_value=report), patch.object(c, 'candidates', return_value=[candidate()]), \
             patch.object(c, 'stop_evidence', return_value={'eligible': False}):
            self.assertEqual(c.run(), 2)
        self.assertEqual(c.state['status'], 'paused'); c.executor.assert_not_called(); provider.close.assert_called_once()
        self.assertEqual(len(list((c.root / 'errors').glob('*.json'))), 1)

    def test_data_quality_blocker_prevents_next_batch(self):
        c = self.campaign(executor=Mock(side_effect=AssertionError('must not execute')))
        with patch.object(c, 'analyze', return_value={'quality': {'blocking_reasons': ['force protocol changed']}}):
            self.assertEqual(c.run(), 0)
        self.assertEqual(c.state['status'], 'paused'); self.assertEqual(c.state['batches'], [])
        c.executor.assert_not_called()

    def test_elapsed_budget_exhaustion_never_starts_validation(self):
        c = self.campaign(executor=Mock(side_effect=AssertionError('time limit')))
        c.state['elapsed_s'] = c.settings['budget']['max_seconds'] + 1
        report = {'quality': {'blocking_reasons': []}, 'metrics': {}}
        with patch.object(c, 'analyze', return_value=report), patch.object(c, 'candidates', return_value=[candidate()]), \
             patch.object(c, 'stop_evidence', return_value={'eligible': False}):
            self.assertEqual(c.run(), 0)
        self.assertEqual(c.state['status'], 'budget_exhausted'); self.assertIsNone(c.state['frozen'])
        c.executor.assert_not_called()

    def test_executor_deadline_return_stops_loop_and_terminal_resume_is_noop(self):
        execute = Mock(return_value={'complete': False, 'quit': False, 'context': {},
                                     'deadline_reached': True})
        c = self.campaign(executor=execute)
        batch = c.prepare_batch(candidate())
        with patch.object(c, 'analyze', side_effect=AssertionError('deadline must stop before analysis')):
            self.assertEqual(c.run(), 0)
        self.assertEqual(c.state['status'], 'budget_exhausted')
        self.assertEqual(c.state['active_batch'], batch['id'])
        self.assertEqual(batch['status'], 'paused')
        self.assertIn('时间预算', c.state['stop_reason'])
        self.assertEqual(len(c.state['batches']), 1)
        execute.assert_called_once()
        resumed = agentic.Campaign(self.root, executor=Mock(side_effect=AssertionError('budget is terminal')), say=lambda _: None)
        self.assertEqual(resumed.run(), 0)
        resumed.executor.assert_not_called()

    def test_validation_deadline_never_reports_incomplete_fixed_plan_as_complete(self):
        execute = Mock(return_value={'complete': False, 'quit': False, 'context': {},
                                     'deadline_reached': True})
        c = self.campaign(executor=execute); self.model(c)
        c.state.update(phase='validation', frozen={'model_id': 'm1', 'policy': {'gain': 1.}, 'reason': 'test'})
        c.prepare_batch(agentic.validation_candidate(c.settings), validation=True)
        with patch.object(c, 'finish_validation', side_effect=AssertionError('incomplete fixed validation')):
            self.assertEqual(c.run(), 0)
        self.assertEqual(c.state['status'], 'budget_exhausted')
        self.assertNotIn('validation', c.state)
        execute.assert_called_once()

    def test_B_only_rounds_do_not_manufacture_mapping_plateau(self):
        c = self.campaign(executor=Mock(side_effect=AssertionError('agent pauses')))
        c.state['history'] = [{'cond': 'B', 'metrics': {'mae_mm': .2}} for _ in range(5)]
        report = {'quality': {'blocking_reasons': []}, 'metrics': {}}
        with patch.object(c, 'analyze', return_value=report), patch.object(c, 'candidates', return_value=[candidate()]), \
             patch.object(c, 'stop_evidence', return_value={'eligible': False}), \
             patch.object(c, 'decide', return_value=({'action': 'pause', 'reason': 'test'}, None)) as decide, \
             patch.object(c, 'freeze', side_effect=AssertionError('B is not new A evidence')):
            self.assertEqual(c.run(), 0)
        decide.assert_called_once()


if __name__ == '__main__':
    unittest.main()
