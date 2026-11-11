"""Learned head integration, physical guard regressions, and fake collection only."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bottle_scan.agentic_executor import execute_batch
from bottle_scan.recovery_skill import RecoverySkill, load_skill_bundle
from bottle_scan.skill_collection import SkillBatchRunner, load_skill_config, main
from tests.test_agentic_executor import FakeHardware, ask_default, plan
from tests.test_agentic_learning import desc, servo
from tests.test_skill_collection import skill_cfg, skill_model


TARGET = {'id': 'contact_target_test_v1', 'description': 'Independent tactile contact target',
          'normal': 'Contact lies inside the independently specified target region',
          'needs_recovery': 'Contact lies outside target but remains within the allowed gel region',
          'completion': 'Stable bilateral interior target contact'}


class CollectionControlTests(unittest.TestCase):
    def test_limit_preserves_frozen_plan_and_resume_does_not_repeat_completed_trials(self):
        config = skill_cfg()
        items = plan(cond='A', offsets=(0., 4.))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first_hw = FakeHardware(config)
            first = execute_batch(root, items, config, None, {}, limit=1, step=True,
                hardware_factory=lambda _: first_hw, ask=ask_default, say=lambda _: None)
            self.assertFalse(first['complete'])
            self.assertEqual(first['completed_count'], 1)
            initial = (root/'trials/00001/meta.json').read_bytes()
            self.assertFalse((root/'trials/00002').exists())
            saved = json.loads((root/'session_meta.json').read_text())
            self.assertEqual(saved['plan'], items)
            second_hw = FakeHardware(config, pose=first_hw.pose())
            second = execute_batch(root, items, config, None, {}, limit=1,
                hardware_factory=lambda _: second_hw, ask=ask_default, say=lambda _: None)
            self.assertTrue(second['complete'])
            self.assertEqual(second['completed_count'], 1)
            self.assertEqual((root/'trials/00001/meta.json').read_bytes(), initial)
            forbidden = Mock(side_effect=AssertionError('completed batch must not open hardware'))
            third = execute_batch(root, items, config, None, {}, limit=1,
                hardware_factory=forbidden, ask=ask_default, say=lambda _: None)
            self.assertEqual(third['completed_count'], 0)
            forbidden.assert_not_called()

    def test_new_window_labels_are_unique_unknown_preserved_and_no_label_moves(self):
        config = skill_cfg()
        config['recovery_skill']['target_definition'] = TARGET
        hw = FakeHardware(config)
        command_count = []
        def ask(message):
            if '回看首次轻触' in message:
                command_count.append(len(hw.events))
                return '1'
            if '交回控制权前' in message:
                return '2'
            if '只判断这个时间窗口' in message:
                return ''
            return ask_default(message)
        with tempfile.TemporaryDirectory() as td:
            execute_batch(td, plan(cond='B', offsets=(0.,)), config, skill_model(), {},
                hardware_factory=lambda _: hw, ask=ask, say=lambda _: None)
            row = json.loads((Path(td)/'trials/00001/meta.json').read_text())
        gt = row['skill_ground_truth']
        self.assertEqual(gt['target_definition_id'], TARGET['id'])
        self.assertEqual(gt['initial_class'], 'needs_recovery')
        labels = row['skill_window_labels']
        timestamps = {e['observation']['timestamp_ns'] for e in row['skill_observations']}
        self.assertEqual({r['timestamp_ns'] for r in labels}, timestamps)
        self.assertEqual(len(labels), len(timestamps))
        self.assertTrue(all(r['target_reached'] is None for r in labels))
        self.assertTrue(all(r['target_definition_id'] == TARGET['id'] for r in labels))
        # Only close of the connection happens after the labels, no robot command.
        self.assertEqual(hw.events[command_count[0]:], ['close'])

    def test_rejection_does_not_misreport_motion_budget_exhaustion(self):
        config = skill_cfg(mode='deployment')
        with tempfile.TemporaryDirectory() as td:
            hw = FakeHardware(config)
            execute_batch(td, plan(cond='B', offsets=(0.,)), config, skill_model(), {},
                hardware_factory=lambda _: hw, ask=ask_default, say=lambda _: None)
            row = json.loads((Path(td)/'trials/00001/meta.json').read_text())
        self.assertFalse(row['servo_exhausted'])
        self.assertTrue(row['skill_incomplete'])
        self.assertEqual(row['skill_failure_reason'], 'missing_or_mismatched_validation_certificate')

    def test_incomplete_annotation_protocol_rejected_before_collection(self):
        import yaml
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'config.yaml'
            path.write_text(yaml.safe_dump({'schema':'bottle_skill_config_v1', 'learned_heads':True,
                                            'target_definition':{'id':'too-little'}}))
            with self.assertRaisesRegex(ValueError, 'target_definition requires'):
                load_skill_config(path)

    def test_invalid_limits_rejected_before_factory(self):
        factory = Mock(side_effect=AssertionError('must not initialize'))
        for limit in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'positive'):
                execute_batch('unused', plan(), skill_cfg(), None, {}, limit=limit, hardware_factory=factory)
        factory.assert_not_called()

    def test_interrupted_post_retreat_label_resumes_without_hardware_or_replay(self):
        config = skill_cfg()
        config['recovery_skill']['target_definition'] = TARGET
        hw = FakeHardware(config)
        def interrupted_review(message):
            if '回看首次轻触' in message:
                raise KeyboardInterrupt('review interrupted')
            return ask_default(message)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(KeyboardInterrupt):
                execute_batch(root, plan(cond='A'), config, None, {}, limit=1,
                    hardware_factory=lambda _: hw, ask=interrupted_review, say=lambda _: None)
            path = root/'trials/00001/meta.json'
            before = json.loads(path.read_text())
            self.assertTrue(before['skill_label_pending'])
            self.assertTrue(before['skill_physical_finished'])
            self.assertFalse(before['complete'])
            blocked = Mock(side_effect=AssertionError('review must not connect hardware'))
            resumed = execute_batch(root, plan(cond='A'), config, None, {}, limit=1,
                hardware_factory=blocked, ask=ask_default, say=lambda _: None)
            blocked.assert_not_called()
            self.assertTrue(resumed['labels_only'])
            self.assertEqual(resumed['completed_count'], 0)
            self.assertEqual(resumed['labelled_existing_count'], 1)
            after = json.loads(path.read_text())
            self.assertTrue(after['complete'])
            self.assertFalse(after['skill_label_pending'])
            self.assertEqual(before['stages'], after['stages'])
            self.assertFalse((root/'trials/00002').exists())
            self.assertFalse((root/'interrupted').exists())


    def test_joint_train_cli_selects_joint_learner_with_explicit_target(self):
        import yaml
        from bottle_scan.learned_skill import train_learned_skill
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root/'joint.yaml'
            config.write_text(yaml.safe_dump({'schema': 'bottle_skill_config_v1',
                'target_definition': TARGET, 'learning': {'simulate': True}}))
            with patch('bottle_scan.learned_skill.train_learned_skill',
                       return_value={'usable_for_B': False, 'blocking_reasons': ['needs labels']}) as train:
                status = main(['train', '--learning-mode', 'joint', '--config', str(config),
                               '--session', str(root/'evidence'), '--out', str(root/'model.json')])
            self.assertEqual(status, 2)
            self.assertEqual(train.call_args.args[1]['target_definition_id'], TARGET['id'])
            self.assertEqual(train.call_args.kwargs['out'], root/'model.json')


from tests.test_learned_skill import LearnedSkillFixture


class LearnedRuntimeTests(LearnedSkillFixture):
    def setUp(self):
        super().setUp()
        _, self.model = self.train_learned()

    def skill(self, **kwargs):
        return RecoverySkill(self.model, {**servo(), 'max_regrasps': 3},
                             mode=kwargs.pop('mode', 'development'), **kwargs)

    def observation(self, x=0., timestamp_ns=1_000_000_000, **kwargs):
        return {'status': 'contact', 'bilateral': True,
                'descriptors': [desc(60-3*x), desc(55-3*x)],
                'timestamp_ns': timestamp_ns, 'now_ns': timestamp_ns,
                'first_touch_complete': True, 'safety_ok': True,
                'force_stable': True, 'stable_duration_s': .2,
                'context': {'target_definition_id': TARGET['id']}, **kwargs}

    def test_actual_heads_normal_requires_two_fresh_windows_and_no_regressor(self):
        skill = self.skill()
        skill.policy.predict = Mock(side_effect=AssertionError('no-action needs no regression'))
        first = skill.start(self.observation())
        self.assertFalse(first['learned_conditions']['invocation']['positive'])
        self.assertTrue(first['learned_conditions']['exit']['positive'])
        self.assertFalse(first['exit']['completed'])
        last = skill.advance(self.observation(timestamp_ns=2_000_000_000))
        self.assertTrue(last['exit']['completed'])
        skill.policy.predict.assert_not_called()

    def test_positive_invocation_uses_small_action_below_legacy_two_mm(self):
        skill = self.skill()
        skill.policy.predict = Mock(return_value={'valid': True, 'action_mm': -.5, 'uncertainty_mm': .1})
        first = skill.start(self.observation(6))
        self.assertTrue(first['invocation']['invoke'])
        self.assertAlmostEqual(first['recovery']['action']['delta_mm'], -.5)
        self.assertFalse(first['exit']['completed'])
        fresh = skill.advance(self.observation(timestamp_ns=3_000_000_000),
            applied_action={'applied': True, 'action_mm': -.5, 'completed_ns': 2_000_000_000})
        self.assertEqual(fresh['exit']['status'], 'observing')
        last = skill.advance(self.observation(timestamp_ns=4_000_000_000))
        self.assertTrue(last['exit']['completed'])

    def test_uncertain_heads_or_recovery_never_issue_action_or_success(self):
        for head in ('invocation', 'exit'):
            skill = self.skill()
            setattr(skill.learned_state, head, lambda _: {'valid': False, 'positive': None,
                'probability': .5, 'reason': 'test_uncertain'})
            result = skill.start(self.observation(6))
            self.assertEqual(result['exit']['reason'], 'test_uncertain')
            self.assertFalse(result['exit']['completed'])
            self.assertIsNone(result['recovery']['action'])
        skill = self.skill()
        skill.policy.predict = lambda _: {'valid': False, 'reason': 'policy_uncertain'}
        result = skill.start(self.observation(6))
        self.assertEqual(result['exit']['reason'], 'policy_uncertain')
        self.assertIsNone(result['recovery']['action'])

    def test_head_disagreement_never_moves_or_completes(self):
        for state in (True, False):
            skill = self.skill()
            predict = lambda _: {'valid': True, 'positive': state, 'probability': float(state)}
            skill.learned_state.invocation = predict
            skill.learned_state.exit = predict
            result = skill.start(self.observation())
            self.assertEqual(result['exit']['reason'], 'learned_state_disagreement')
            self.assertIsNone(result['recovery']['action'])
            self.assertFalse(result['exit']['completed'])

    def test_guard_rejections_remain_in_front_of_learned_states(self):
        for modified in ({'safety_ok': False}, {'first_touch_complete': False},
                         {'now_ns': 9_000_000_000}, {'context': {}}, {'bilateral': False}):
            result = self.skill().start(self.observation(**modified))
            self.assertFalse(result['exit']['completed'])
            self.assertIsNone(result['recovery']['action'])
            self.assertEqual(result['exit']['status'], 'incomplete')
        observation = self.observation()
        observation['descriptors'][0]['touches_edge'] = True
        self.assertEqual(self.skill().start(observation)['exit']['status'], 'incomplete')
        skill = self.skill()
        skill.start(self.observation())
        self.assertEqual(skill.advance(self.observation())['exit']['reason'], 'new_observation_required')
        result = self.skill(mode='deployment').start(self.observation())
        self.assertEqual(result['exit']['reason'], 'missing_or_mismatched_validation_certificate')

    def test_receipt_stability_and_budget_still_required(self):
        skill = self.skill()
        first = skill.start(self.observation(6))
        self.assertIsNotNone(first['recovery']['action'])
        result = skill.advance(self.observation(timestamp_ns=3_000_000_000))
        self.assertEqual(result['exit']['reason'], 'unverified_execution_or_reobservation')
        skill = self.skill(config={'max_regrasps': 0})
        self.assertEqual(skill.start(self.observation(6))['exit']['reason'], 'recovery_budget_exhausted')
        for state in ({'force_stable': False}, {'stable_duration_s': .1}):
            skill = self.skill()
            self.assertEqual(skill.start(self.observation(**state))['exit']['status'], 'observing')
            self.assertFalse(skill.advance(self.observation(timestamp_ns=3_000_000_000, **state))['exit']['completed'])

    def test_new_bundle_round_trip_checks_model_and_contains_all_heads(self):
        root = self.root/'bundle'
        skill = self.skill()
        skill.save_bundle(root)
        loaded = load_skill_bundle(root, mode='development')
        self.assertEqual(loaded.model_id, skill.model_id)
        self.assertEqual(loaded.evaluate(self.observation()), skill.evaluate(self.observation()))

    def test_learned_artifact_runs_through_fake_collection_and_records_heads(self):
        from bottle_scan.agentic_learning import _hash, _signature
        from bottle_scan.contact_policy import _seal as seal_recovery
        from bottle_scan.learned_skill import _seal, LearnedSkillModel
        config = skill_cfg()
        config['force']['normal_axis'] = self.model['normal_axis']
        config['force']['normal_signs'] = self.model['normal_signs']
        config['recovery_skill']['target_definition'] = TARGET
        # This is a fake transport fixture, explicitly resealed for its protocol.
        model = copy.deepcopy(self.model)
        model['axes_tcp'] = copy.deepcopy(config['axes_tcp'])
        model['recovery_model']['axes_tcp'] = copy.deepcopy(config['axes_tcp'])
        model['recovery_model']['fallback_fit']['axes_tcp'] = copy.deepcopy(config['axes_tcp'])
        model['signature'] = _hash(_signature({'config': config, 'simulated': False}))
        model['recovery_model']['signature'] = model['signature']
        seal_recovery(model['recovery_model'])
        _seal(model)
        hw = FakeHardware(config)
        normal = {'valid': True, 'positive': False, 'probability': .01, 'reason': 'synthetic_normal'}
        reached = {'valid': True, 'positive': True, 'probability': .99, 'reason': 'synthetic_target'}
        with patch.object(LearnedSkillModel, 'invocation', return_value=normal), \
             patch.object(LearnedSkillModel, 'exit', return_value=reached):
            execute_batch(self.root/'fake', plan(cond='B', offsets=(0.,)), config, model, {},
                hardware_factory=lambda _: hw, ask=ask_default, say=lambda _: None)
        row = json.loads((self.root/'fake/trials/00001/meta.json').read_text())
        self.assertTrue(row['skill_handoff']['completed'])
        self.assertIn('LIFT', hw.events)
        self.assertNotIn('REGRASP_CORRECT', hw.events)
        self.assertEqual(row['skill_outputs']['model_id'], model['model_id'])
        self.assertEqual(row['skill_outputs']['learned_conditions']['exit'], reached)
        self.assertTrue(all(e['observation']['context']['target_definition_id'] == TARGET['id']
                            for e in row['skill_observations']))

    def test_actual_collection_receipts_and_terminal_labels_admit_joint_B_action(self):
        """Guard against trainer/collector schema drift; all I/O is fake.

        The real guarded collection stages produce the receipt, measurements,
        decision-window association and operator labels. Only predictions and
        the hardware transport are controlled here; no metadata is patched
        after collection to make eligibility pass.
        """
        from bottle_scan.agentic_learning import _hash, _signature, _load
        from bottle_scan.contact_policy import ContactPolicy, _B_observations, DEFAULTS, _seal as seal_recovery
        from bottle_scan.learned_skill import LearnedSkillModel, _seal
        config = skill_cfg()
        config['force']['normal_axis'] = self.model['normal_axis']
        config['force']['normal_signs'] = self.model['normal_signs']
        config['recovery_skill']['target_definition'] = TARGET
        model = copy.deepcopy(self.model)
        model['axes_tcp'] = copy.deepcopy(config['axes_tcp'])
        model['recovery_model']['axes_tcp'] = copy.deepcopy(config['axes_tcp'])
        model['recovery_model']['fallback_fit']['axes_tcp'] = copy.deepcopy(config['axes_tcp'])
        model['signature'] = _hash(_signature({'config': config, 'simulated': False}))
        model['recovery_model']['signature'] = model['signature']
        seal_recovery(model['recovery_model'])
        _seal(model)
        state = {'observations': 0, 'needs_recovery': None, 'labelled_windows': 0}
        def invocation(_model, observation):
            state['observations'] += 1
            state['needs_recovery'] = state['observations'] == 1
            return {'valid': True, 'positive': state['needs_recovery'],
                    'probability': float(state['needs_recovery']), 'reason': 'synthetic_state'}
        def completion(_model, observation):
            return {'valid': True, 'positive': not state['needs_recovery'],
                    'probability': float(not state['needs_recovery']), 'reason': 'synthetic_state'}
        def review(message):
            if '回看首次轻触' in message or '交回控制权前' in message:
                return '1'
            if '只判断这个时间窗口' in message:
                state['labelled_windows'] += 1
                return '2' if state['labelled_windows'] == 1 else '1'
            return ask_default(message)
        root = self.root/'actual_B_shape'
        hw = FakeHardware(config)
        with patch.object(LearnedSkillModel, 'invocation', invocation), \
             patch.object(LearnedSkillModel, 'exit', completion), \
             patch.object(ContactPolicy, 'predict', return_value={
                 'valid': True, 'action_mm': -4., 'uncertainty_mm': .1}):
            result = execute_batch(root, plan(cond='B', offsets=(4.,)), config, model,
                {'batch_role': 'development'}, hardware_factory=lambda _: hw,
                ask=review, say=lambda _: None)
        self.assertTrue(result['complete'])
        self.assertIn('REGRASP_CORRECT', hw.events)
        self.assertIn('LIFT', hw.events)
        sessions, rows, issues = _load([root], {'simulate': False})
        self.assertFalse(issues)
        row = rows[0]
        self.assertEqual(row['status'], 'hold_complete')
        self.assertTrue(row['skill_handoff']['completed'])
        self.assertEqual(len(row['correction_history']), 2)  # action + terminal
        self.assertTrue(row['correction_history'][-1]['terminal'])
        self.assertEqual(len(row['probe_regrasps']), 1)
        self.assertEqual(len(row['skill_window_labels']), 3)
        self.assertFalse(row['skill_window_labels'][0]['target_reached'])
        self.assertTrue(row['skill_window_labels'][-1]['target_reached'])
        settings = {'target_definition_id': TARGET['id']}
        admitted, excluded = _B_observations(rows, sessions[0]['meta'], settings, DEFAULTS)
        self.assertEqual(excluded, {})
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0]['label_source'], 'B_verified_successful_executed_action_not_optimal')
        self.assertAlmostEqual(admitted[0]['target_action_mm'], -4.)
        self.assertTrue(admitted[0]['measurement']['source'].startswith('measured_tcp'))
        # The identical physical record with unknown/negative terminal annotation
        # must no longer become action supervision, even when lifting succeeded.
        for label in (None, False):
            contradictory = copy.deepcopy(rows)
            contradictory[0]['skill_window_labels'][-1]['target_reached'] = label
            rejected, excluded = _B_observations(contradictory, sessions[0]['meta'], settings, DEFAULTS)
            self.assertEqual(rejected, [])
            self.assertEqual(excluded, {'B_joint_target_or_handoff_unverified': 1})
