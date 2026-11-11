"""Offline supervisor lifecycle tests. Every collector, trainer and LLM is fake."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from bottle_scan import skill_loop as loop, skill_director as director
from tests.test_skill_loop import CampaignFixture


class ScriptedDirector:
    def __init__(self, actions):
        self.actions = list(actions)
        self.contexts = []
        self.decisions = []

    def direct(self, context):
        self.contexts.append(deepcopy(context))
        action = self.actions.pop(0)
        if isinstance(action, dict):
            answer = deepcopy(action)
        else:
            answer = {'action': action, 'reason': '根据本轮证据选择下一步，并保留预先定义的任务标准。',
                      'evidence_refs': list(context['valid_evidence_refs']), 'plan': None}
            if action in ('collect_A', 'validate_B'):
                answer['plan'] = {'cells': [{'offset_mm': 0., 'reps': 4},
                    {'offset_mm': -5., 'reps': 4}, {'offset_mm': 0., 'reps': 2},
                    {'offset_mm': 5., 'reps': 4}],
                    'hypothesis': '前后边界的恢复行为不同，需要在两只瓶子上核对。',
                    'expected_outcome': '得到已达标、未达标及恢复后状态证据。'}
        self.decisions.append(answer)
        return answer


class DirectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def fixture(self, **kw):
        f = CampaignFixture(self.temp.name, planner='llm', **kw)
        f.config['planner']['authority'] = 'director'
        f.config['budget']['max_agent_decisions'] = 32
        f.config_path.write_text(yaml.safe_dump(f.config))
        return f

    def step(self, f, provider, **kw):
        return loop.advance(f.root, allow_agent=True, provider=provider,
                            trainer=f.trainer, evaluator=kw.pop('evaluator', f.evaluator), **kw)

    def test_init_and_every_gate_await_agent_instead_of_auto_progressing(self):
        f = self.fixture(); s = f.init(seed=True)
        self.assertEqual(s['phase'], 'director')
        self.assertFalse(f.train_calls)
        with patch('bottle_scan.agentic_provider.AgentProvider', side_effect=AssertionError('offline')):
            s = loop.advance(f.root, trainer=f.trainer)
        self.assertEqual(s['phase'], 'awaiting_agent')
        self.assertFalse(f.train_calls)
        p = ScriptedDirector(['train', 'validate_B', 'freeze', 'complete'])
        s = self.step(f, p)
        self.assertEqual(s['phase'], 'director')
        self.assertIsNone(s['batch'])  # Ready model does not schedule B itself.
        s = self.step(f, p); self.assertEqual(s['phase'], 'collect_B')
        f.write_batch()
        s = loop.advance(f.root, evaluator=f.evaluator)
        self.assertTrue(s['current_development']['result']['passed'])
        self.assertEqual(s['phase'], 'awaiting_agent')
        self.assertFalse(s['frozen_independent'])
        s = self.step(f, p); self.assertEqual(s['phase'], 'collect_independent')
        f.write_batch()
        s = loop.advance(f.root, evaluator=f.evaluator)
        self.assertTrue(s['independent_evaluation']['result']['passed'])
        self.assertEqual(s['phase'], 'awaiting_agent')
        self.assertNotIn('completion_decision', s)
        s = self.step(f, p); self.assertEqual(s['phase'], 'complete')
        audit = loop.read(s['completion_decision']['path'])
        self.assertEqual(audit['answer']['action'], 'complete')
        self.assertEqual(audit['model_id'], s['model']['model_id'])
        self.assertEqual(loop.digest(audit), s['completion_decision']['hash'])

    def test_full_failed_B_new_A_retrain_independent_agent_completion(self):
        f = self.fixture(confirmed=True); f.init(seed=True)
        p = ScriptedDirector(['train', 'validate_B', 'collect_A', 'train', 'validate_B', 'freeze', 'complete'])
        n = [0]
        def evaluator(session, **kw):
            n[0] += 1
            return f.evaluator(session, fail=n[0] == 1, **kw)
        s = loop.run(f.root, real=True, allow_agent=True, provider=p,
                     trainer=f.trainer, evaluator=evaluator, collector=f.collector)
        self.assertEqual(s['phase'], 'complete')
        self.assertEqual(len(f.train_calls), 2)
        self.assertEqual(len(f.collect_calls), 4)
        self.assertEqual([x['action'] for x in p.decisions],
                         ['train', 'validate_B', 'collect_A', 'train', 'validate_B', 'freeze', 'complete'])
        self.assertEqual(len(s['director_decisions']), 7)
        self.assertEqual(len({args[args.index('--out')+1] for args in f.collect_calls}), 4)
        self.assertTrue(all('independent' not in path for sources in f.train_calls for path in sources))
        self.assertEqual(s['model']['model_id'], 'frozen-2')
        self.assertEqual(s['current_development']['model_id'], 'frozen-2')
        self.assertEqual(s['independent_evaluation']['model_id'], 'frozen-2')
        self.assertFalse('collect_A' in p.contexts[-1]['allowed_actions'])
        self.assertTrue('complete' in p.contexts[-1]['allowed_actions'])
        self.assertEqual(len(s['sealed_evaluation_sources']), 1)
        trace = {'schema': 'offline_director_lifecycle_trace_v1', 'simulated': True,
                 'hardware_started': False, 'live_llm_called': False,
                 'actions': [x['action'] for x in p.decisions],
                 'trained_versions': [r['model_id'] for r in s['rounds']],
                 'development_passes': [r['development']['passed'] for r in s['rounds']],
                 'independent_passed': s['independent_evaluation']['result']['passed'],
                 'final_phase': s['phase']}
        # Useful for explicitly requested offline execution traces, without test-path dependencies.
        self.assertEqual(trace['development_passes'], [False, True])

    def test_agent_can_choose_more_A_after_ready_model(self):
        f = self.fixture(); f.init(seed=True)
        p = ScriptedDirector(['train', 'collect_A'])
        self.step(f, p); s = self.step(f, p)
        self.assertEqual(s['phase'], 'collect_A')
        self.assertIn('validate_B', p.contexts[-1]['allowed_actions'])

    def test_agent_selects_initial_A_before_any_training(self):
        f = self.fixture(); s = f.init()
        self.assertIsNone(s['batch'])
        p = ScriptedDirector(['collect_A'])
        s = self.step(f, p)
        self.assertEqual(s['phase'], 'collect_A')
        self.assertFalse(f.train_calls)
        self.assertNotIn('train', p.contexts[0]['allowed_actions'])

    def test_no_data_retraining_or_premature_B_or_completion_rejected(self):
        f = self.fixture(); f.init(seed=True)
        p = ScriptedDirector(['validate_B', 'complete', 'train', 'train'])
        for _ in range(2):
            s = self.step(f, p)
            self.assertEqual(s['phase'], 'awaiting_agent')
            self.assertFalse(loop.read(s['director_decisions'][-1]['path'])['validation']['accepted'])
        self.step(f, p); s = self.step(f, p)
        self.assertEqual(len(f.train_calls), 1)
        self.assertIn('rejected', s['stop_reason'])
        self.assertNotIn('train', p.contexts[-1]['allowed_actions'])

    def test_missing_class_model_cannot_be_sent_to_B(self):
        f = self.fixture(); f.init(seed=True)
        original = f.trainer
        def blocked(sources, settings):
            model = original(sources, settings)
            model.update(usable_for_B=False, blocking_reasons=['missing target negative examples'])
            return model
        f.trainer = blocked
        p = ScriptedDirector(['train', 'validate_B'])
        self.step(f, p); s = self.step(f, p)
        self.assertEqual(s['phase'], 'awaiting_agent')
        self.assertIsNone(s['batch'])
        self.assertNotIn('validate_B', p.contexts[-1]['allowed_actions'])

    def test_unknown_reference_and_unsafe_plan_are_recorded_but_not_executed(self):
        f = self.fixture(); f.init(seed=True)
        p = ScriptedDirector([{'action': 'train', 'reason': 'test', 'evidence_refs': ['made_up'], 'plan': None}])
        s = self.step(f, p)
        self.assertEqual(s['phase'], 'awaiting_agent'); self.assertFalse(f.train_calls)
        good = ScriptedDirector(['collect_A']).direct(director.context(f.root, s, loop.read(f.root/'config.json')))
        good['plan']['cells'][1]['offset_mm'] = 500
        s = self.step(f, ScriptedDirector([good]))
        self.assertIsNone(s['batch'])
        self.assertEqual(len(s['director_decisions']), 2)
        self.assertFalse(loop.read(s['director_decisions'][-1]['path'])['validation']['accepted'])

    def test_pause_review_and_decision_budget(self):
        f = self.fixture(); f.config['budget']['max_agent_decisions'] = 2
        f.config_path.write_text(yaml.safe_dump(f.config)); f.init(seed=True)
        p = ScriptedDirector(['pause', 'pause'])
        self.assertEqual(self.step(f, p)['phase'], 'awaiting_agent')
        self.assertEqual(self.step(f, p)['phase'], 'awaiting_agent')
        self.assertEqual(self.step(f, p)['phase'], 'budget_exhausted')
        self.assertEqual(len(p.contexts), 2)

    def test_review_keeps_truth_immutable_and_does_not_recall_provider(self):
        f = self.fixture(); f.init(seed=True)
        original = loop.source_snapshot(f.reference)
        p = ScriptedDirector(['review_labels'])
        s = self.step(f, p)
        self.assertEqual(s['phase'], 'awaiting_label_review')
        self.assertEqual(loop.source_snapshot(f.reference), original)
        self.assertEqual(self.step(f, p)['phase'], 'awaiting_label_review')
        self.assertEqual(len(p.contexts), 1)

    def test_independent_failure_never_reopens_training(self):
        f = self.fixture(confirmed=True); f.init(seed=True)
        p = ScriptedDirector(['train', 'validate_B', 'freeze', 'pause'])
        def evaluator(session, **kw):
            return f.evaluator(session, fail=kw['role'] == 'independent', **kw)
        s = loop.run(f.root, real=True, allow_agent=True, provider=p, trainer=f.trainer,
                     evaluator=evaluator, collector=f.collector)
        self.assertEqual(s['phase'], 'independent_failed')
        self.assertEqual(p.contexts[-1]['allowed_actions'], ['pause'])
        self.assertEqual(len(f.train_calls), 1)

    def test_stale_development_and_independent_cannot_freeze_or_complete(self):
        f = self.fixture(); f.init(seed=True); p = ScriptedDirector(['train'])
        s = self.step(f, p); c = loop.read(f.root/'config.json')
        fake = {'result': {'passed': True}, 'model_id': 'old', 'model_hash': 'wrong',
                'report_path': str(f.root/'nonexistent'), 'session': 'old'}
        s['current_development'] = fake
        self.assertNotIn('freeze', director.context(f.root, s, c)['allowed_actions'])
        s.update(current_development=None, independent_evaluation=fake, frozen_independent=True,
                 frozen_model=deepcopy(s['model']))
        self.assertNotIn('complete', director.context(f.root, s, c)['allowed_actions'])

    def test_pending_train_resumes_without_new_LLM_or_duplicate_training(self):
        f = self.fixture(); f.init(seed=True)
        p = ScriptedDirector(['train'])
        real_execute = director._execute
        with patch.object(director, '_execute', side_effect=RuntimeError('crash before effect')):
            with self.assertRaisesRegex(RuntimeError, 'crash'):
                self.step(f, p)
        s = loop.read(f.root/'state.json')
        self.assertIsNotNone(s['pending_director_decision'])
        s = self.step(f, p)
        self.assertEqual(s['phase'], 'director')
        self.assertEqual(len(p.contexts), 1)
        self.assertEqual(len(f.train_calls), 1)
        self.assertEqual(len(s['rounds']), 1)
        self.assertIsNone(s['pending_director_decision'])

    def test_decision_tampering_is_detected_before_any_new_action(self):
        f = self.fixture(); f.init(seed=True); p = ScriptedDirector(['train'])
        s = self.step(f, p)
        path = Path(s['director_decisions'][0]['path']); data = loop.read(path)
        data['answer']['reason'] = 'changed'; path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'sealed evaluation report changed'):
            self.step(f, p)

    def test_consumed_A_crash_boundary_persists_director_and_sources_once(self):
        f = self.fixture(); f.init()
        p = ScriptedDirector(['collect_A', 'train'])
        s = self.step(f, p); f.write_batch()
        root, s, c = loop._load(f.root)
        self.assertTrue(loop._consume(root, s, c))
        persisted = loop.read(root/'state.json')
        self.assertEqual(persisted['phase'], 'director')
        self.assertEqual(len(persisted['training_sources']), 1)
        result = self.step(f, p)
        self.assertEqual(result['phase'], 'director')
        self.assertEqual(len(result['training_sources']), 1)
        self.assertEqual(len(f.train_calls), 1)

    def test_pretrain_context_has_current_class_coverage_without_fake_B(self):
        f = self.fixture(); f.init(seed=True)
        root, s, c = loop._load(f.root)
        expected = {'class_counts': {'exit': {'positive': 21, 'negative': 0}},
                    'blocking_reasons': ['contradictory_contact_labels'],
                    'data_gaps': [{'component': 'exit', 'class': 'negative', 'missing': 8}]}
        with patch('bottle_scan.learned_skill.diagnose_training_data', return_value=expected) as diagnose:
            ctx = director.context(root, s, c)
        self.assertIsNone(ctx['development_assessment'])
        self.assertEqual(ctx['current_training_diagnosis']['class_counts'], expected['class_counts'])
        self.assertEqual(ctx['current_training_diagnosis']['blocking_reasons'], expected['blocking_reasons'])
        diagnose.assert_called_once()

    def test_frozen_operator_feedback_reaches_provider_without_relabelling_or_training_input(self):
        f = self.fixture()
        feedback = {'source': 'operator_requested_outcome_rule',
                    'rule': {'force_not_reached': 'needs_recovery', 'hold_complete': 'already_normal'},
                    'interpretation': 'weak hypothesis; position cause is not independently confirmed'}
        f.config['operator_feedback'] = feedback
        f.config_path.write_text(yaml.safe_dump(f.config))
        state = f.init(seed=True)
        original_sources = deepcopy(state['training_sources'])
        original_evidence = loop.source_snapshot(f.reference)
        frozen = loop.read(f.root/'config.json')
        self.assertEqual(frozen['operator_feedback'], feedback)
        provider = ScriptedDirector(['train'])
        with patch.object(f, 'trainer', wraps=f.trainer) as train:
            result = self.step(f, provider)
        ctx = provider.contexts[0]
        self.assertEqual(ctx['operator_feedback'], feedback)
        self.assertIn(str(f.root/'config.json'), ctx['valid_evidence_refs'])
        self.assertEqual(result['training_sources'], original_sources)
        self.assertEqual(loop.source_snapshot(f.reference), original_evidence)
        self.assertEqual(train.call_args.args[0], [s['path'] for s in original_sources])
        self.assertEqual(train.call_args.args[1], frozen['learning'])
        self.assertNotIn('operator_feedback', train.call_args.args[1])
        # A provider receives a detached copy, never a writable label/config view.
        ctx['operator_feedback']['rule']['force_not_reached'] = 'invented_label'
        self.assertEqual(loop.read(f.root/'config.json')['operator_feedback'], feedback)
        self.assertEqual(loop.source_snapshot(f.reference), original_evidence)

    def test_invalid_operator_feedback_is_rejected_before_provider_call(self):
        f = self.fixture(); f.init(seed=True)
        root, state, config = loop._load(f.root)
        provider = Mock()
        for bad in (['not an object'], {'score': float('nan')}, {'not_json': object()}):
            with self.subTest(value=type(bad).__name__):
                modified = deepcopy(config); modified['operator_feedback'] = bad
                with self.assertRaisesRegex(ValueError, 'operator_feedback'):
                    director.advance(root, deepcopy(state), modified, allow_agent=True, provider=provider)
        provider.direct.assert_not_called()

    def test_B_decision_must_cite_current_model_evidence(self):
        f = self.fixture(); f.init(seed=True)
        p = ScriptedDirector(['train']); s = self.step(f, p)
        context = director.context(f.root, s, loop.read(f.root/'config.json'))
        answer = ScriptedDirector(['validate_B']).direct(context)
        answer['evidence_refs'] = [str(f.root/'config.json')]
        s = self.step(f, ScriptedDirector([answer]))
        self.assertEqual(s['phase'], 'awaiting_agent')
        self.assertIsNone(s['batch'])

    def test_shape_validation_rejects_code_and_malformed_values(self):
        answer = {'action': 'pause', 'reason': 'test', 'evidence_refs': ['config'], 'plan': None}
        for mutate in (lambda x: x.update(command='move robot'),
                       lambda x: x.update(action='execute'),
                       lambda x: x.update(plan={}),
                       lambda x: x.update(evidence_refs=[])):
            with self.subTest(mutate=mutate):
                bad = deepcopy(answer); mutate(bad)
                with self.assertRaises(ValueError): director.validate_response(bad)

    def test_set_terminal_handoff_is_bound_to_saved_fresh_contact_features(self):
        terminal={'completed': True, 'hand_back': True, 'achieved_state': {
            'timestamp_ns': 123, 'target_definition_id': 'set', 'contact_features': [1.,2.,3.,4.]}}
        observation={'timestamp_ns':123,'status':'contact','bilateral':True,'force_stable':True,
            'stable_duration_s':.3,'context':{'target_definition_id':'set'},
            'descriptors':[{'valid':True,'centroid_px':[1.,2.]},{'valid':True,'centroid_px':[3.,4.]}]}
        row={'skill_observations':[{'observation':observation,
            'skill_outputs':{'model_id':'m','exit':deepcopy(terminal)}}]}
        detail={'terminal_exit':terminal}
        self.assertTrue(loop._verified_set_terminal(row,detail,'set','m'))
        for key,value in [('timestamp_ns',124),('bilateral',False),('force_stable',False),('stable_duration_s',float('inf')),
                          ('stable_duration_s',.1),('context',{'target_definition_id':'other'})]:
            with self.subTest(key=key):
                changed=deepcopy(row);changed['skill_observations'][0]['observation'][key]=value
                self.assertFalse(loop._verified_set_terminal(changed,detail,'set','m'))
        changed=deepcopy(detail);changed['terminal_exit']['achieved_state']['contact_features'][0]=99
        self.assertFalse(loop._verified_set_terminal(row,changed,'set','m'))
        self.assertFalse(loop._verified_set_terminal(row,detail,'set','wrong_model'))
        changed=deepcopy(row)
        changed['skill_observations'][0]['observation']['descriptors'][0]['touches_edge']=True
        self.assertFalse(loop._verified_set_terminal(changed,detail,'set','m'))


if __name__ == '__main__':
    unittest.main()
