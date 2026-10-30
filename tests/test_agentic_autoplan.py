"""Campaign integration for proposed plans; all execution/providers are offline."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from bottle_scan import agentic
from bottle_scan.agentic_provider import ProviderError, ProposalError
from bottle_scan.agentic_sim import default_T0
from bottle_scan.session import atomic_json
from tests.test_agentic_campaign import adaptive_settings, config


def proposal(offsets=(-1.5, 1.25, 4.75)):
    return {'action': 'propose_batch', 'reason': '补测非均匀位置以区分局部偏差和噪声',
            'plan': {'cond': 'A', 'cells': [{'offset_mm': x, 'reps': 2} for x in offsets],
                     'model_id': None, 'gain': 1., 'hypothesis': '局部映射存在系统偏差',
                     'evidence_refs': ['metrics'], 'expected_outcome': '比较新位置的实测偏移与反推误差'}}


def report():
    return {'metrics': {}, 'models': [], 'diagnostics': [],
            'quality': {'blocking_reasons': []}, 'evidence': {'level_stats': [], 'observations': []}}


class AutoplanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'campaign'
        s = adaptive_settings()
        s['planning'].update(mode='agent', max_attempts=3)
        s['provider']['backend'] = 'chatgpt'
        self.provider = Mock()
        self.executor = Mock(side_effect=AssertionError('no hardware expected'))
        with patch('bottle_scan.agentic.resolve_base', return_value=config()):
            self.c = agentic.Campaign(self.root, settings=s, provider=self.provider,
                                     executor=self.executor, say=lambda _: None)
        self.c.state['design'] = {'min_mm': -2., 'max_mm': 10., 'step_mm': 2.}
        self.c.state['context']['T0'] = default_T0().tolist()
        self.c.save()

    def decide(self, responses):
        self.provider.propose.side_effect = responses
        return self.c.decide(report(), [], {'eligible': False})

    def test_non_grid_plan_without_any_candidates_is_accepted_and_persisted(self):
        value = proposal()
        decision, chosen = self.decide([value])
        self.assertEqual(decision, value)
        self.assertEqual(chosen['origin'], 'agent_proposal')
        self.assertEqual(chosen['cells'], value['plan']['cells'])
        batch = self.c.prepare_batch(chosen)
        self.assertEqual(len(batch['plan']), 6)
        self.assertEqual({r['delta_mm'] for r in batch['plan']}, {-1.5, 1.25, 4.75})
        self.assertEqual(self.c.state['design']['min_mm'], -2.)
        self.assertEqual(self.c.state['design']['max_mm'], 10.)
        self.assertTrue((self.root / 'decisions/0001_attempt_01_proposal.json').exists())
        self.assertTrue((self.root / 'decisions/0001_validated_plan.json').exists())
        self.executor.assert_not_called()

    def test_invalid_plan_is_returned_for_bounded_repair(self):
        bad = proposal((26.,))
        decision, chosen = self.decide([bad, proposal()])
        self.assertEqual(self.provider.propose.call_count, 2)
        repaired_input = json.loads((self.root / 'decisions/0001_attempt_02_input.json').read_text())
        self.assertIn('previous_rejection', repaired_input['revision'])
        self.assertEqual(repaired_input['revision']['previous_proposal'], bad)
        self.assertEqual(self.c.state['decisions'][0]['attempts'], 2)
        self.assertEqual(chosen['origin'], 'agent_proposal')
        self.executor.assert_not_called()

    def test_json_failure_is_repairable_without_raw_response(self):
        self.decide([ProposalError('JSON 格式错误', code='invalid_json'), proposal()])
        repaired = json.loads((self.root / 'decisions/0001_attempt_02_input.json').read_text())
        self.assertEqual(repaired['revision']['previous_rejection']['code'], 'invalid_json')
        self.assertNotIn('previous_proposal', repaired['revision'])

    def test_three_rejections_do_not_fall_back_or_create_batch(self):
        with self.assertRaisesRegex(ValueError, '连续 3 次'):
            self.decide([proposal((26.,))] * 3)
        self.assertEqual(self.provider.propose.call_count, 3)
        self.assertEqual(self.c.state['batches'], [])
        self.assertEqual(self.c.state['decisions'][0]['action'], 'rejected')
        self.executor.assert_not_called()

    def test_transport_failure_pauses_without_repair(self):
        with self.assertRaises(ProviderError):
            self.decide([ProviderError('连接失败')])
        self.assertEqual(self.provider.propose.call_count, 1)
        self.assertEqual(self.c.state['decisions'][0]['action'], 'provider_error')
        self.executor.assert_not_called()

    def test_premature_freeze_is_rejected_and_can_be_repaired_to_pause(self):
        pause = {'action': 'pause', 'reason': '证据不足，检查数据', 'plan': None}
        decision, chosen = self.decide([{'action': 'freeze', 'reason': 'done', 'plan': None}, pause])
        self.assertEqual(decision, pause)
        self.assertIsNone(chosen)
        self.assertEqual(self.c.state['phase'], 'explore')

    def test_budget_includes_monitors_without_silent_truncation(self):
        self.c.settings['budget']['max_trials'] = self.c.state['validation_reserved_trials'] + 12
        bad = proposal((0., 1., 2., 3., 4., 5.))  # 12 scans + one monitor
        _, chosen = self.decide([bad, proposal((0., 2.))])
        self.assertEqual(self.provider.propose.call_count, 2)
        self.assertEqual(sum(c['reps'] for c in chosen['cells']), 4)

    def test_forbidden_control_fields_and_fake_references_rejected(self):
        control = proposal(); control['plan']['force_target'] = 100
        evidence = proposal(); evidence['plan']['evidence_refs'] = ['fabricated:123']
        self.decide([control, evidence, proposal()])
        self.assertEqual(self.provider.propose.call_count, 3)

    def test_plan_cannot_change_after_local_validation(self):
        _, chosen = self.decide([proposal()])
        changed = copy.deepcopy(chosen); changed['cells'][0]['offset_mm'] = -1.
        with self.assertRaisesRegex(ValueError, '修改'):
            self.c.prepare_batch(changed)
        self.assertEqual(self.c.state['batches'], [])

    def test_crashed_decision_files_are_never_overwritten(self):
        atomic_json(self.root / 'decisions/0001_input.json', {'interrupted': True})
        _, chosen = self.decide([proposal()])
        self.assertEqual(chosen['decision_number'], 2)
        self.assertEqual(json.loads((self.root / 'decisions/0001_input.json').read_text()), {'interrupted': True})

    def test_context_has_actual_previous_plan_and_hypothesis(self):
        _, chosen = self.decide([proposal()])
        batch = self.c.prepare_batch(chosen)
        context = self.c.planning_context(report(), [], {'eligible': False})
        history = context['history'][-1]
        self.assertEqual(history['id'], batch['id'])
        self.assertEqual(history['cells'], proposal()['plan']['cells'])
        self.assertEqual(history['hypothesis'], proposal()['plan']['hypothesis'])
        self.assertEqual(history['policy']['gain'], 1.)

    def test_A_gain_one_is_independent_of_permitted_B_gains(self):
        self.c.settings['policy']['gains'] = [.5, .75]
        _, chosen = self.decide([proposal()])
        self.c.prepare_batch(chosen)
        self.assertEqual(self.c.state['batches'][0]['candidate']['policy']['gain'], 1.)

    def test_coarse_focus_does_not_erase_finest_design_spacing(self):
        self.c.state['design']['step_mm'] = .5
        _, chosen = self.decide([proposal((0., 10.))])
        self.c.prepare_batch(chosen)
        self.assertEqual(self.c.state['design']['step_mm'], .5)

    def test_new_mode_cannot_execute_unsaved_legacy_candidate(self):
        from tests.test_agentic_campaign import candidate
        with self.assertRaisesRegex(ValueError, '完整提案'):
            self.c.prepare_batch(candidate())

    def test_single_side_expansion_persists_before_resume_without_shrinking(self):
        _, chosen = self.decide([proposal((10., 13.5))])
        before = copy.deepcopy(self.c.state['design'])
        self.c.prepare_batch(chosen)
        self.assertEqual(self.c.state['design']['min_mm'], before['min_mm'])
        self.assertEqual(self.c.state['design']['max_mm'], 13.5)
        resumed = agentic.Campaign(self.root, say=lambda _: None)
        self.assertTrue(resumed.agent_planning())
        self.assertEqual(resumed.state['design'], self.c.state['design'])
        self.assertEqual(resumed.state['batches'][0]['candidate']['proposal'], proposal((10., 13.5)))

    def test_old_campaign_missing_mode_remains_legacy(self):
        self.c.state['settings'].pop('planning')
        self.c.save()
        resumed = agentic.Campaign(self.root, say=lambda _: None)
        self.assertFalse(resumed.agent_planning())

    def test_plan_preview_never_invokes_provider_or_executor(self):
        with patch.object(agentic.Campaign, 'analyze', return_value=report()), \
             patch.object(agentic.Campaign, 'write_report'), \
             patch.object(agentic.Campaign, 'candidates', return_value=[]):
            outputs = []
            result = agentic.main(['plan', '--out', str(self.root)], output=outputs.append,
                                  executor=self.executor, provider=self.provider)
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(outputs[-1])['planning_mode'], 'agent')
        self.provider.propose.assert_not_called()
        self.executor.assert_not_called()

    def test_no_suggestions_or_plateau_does_not_bypass_agent(self):
        self.c.state['history'] = [{'cond': 'A', 'design': self.c.state['design'],
                                   'metrics': {'mae_mm': .2}} for _ in range(4)]
        self.provider.propose.return_value = {'action': 'pause', 'reason': '需检查漂移', 'plan': None}
        with patch.object(self.c, 'analyze', return_value=report()), \
             patch.object(self.c, 'candidates', return_value=[]), \
             patch.object(self.c, 'freeze', side_effect=AssertionError('must not auto-freeze')):
            self.assertEqual(self.c.run(), 0)
        self.provider.propose.assert_called_once()
        self.executor.assert_not_called()
        self.assertEqual(self.c.state['status'], 'paused')

    def test_transport_error_run_is_paused_and_audited(self):
        self.provider.propose.side_effect = ProviderError('服务不可用')
        with patch.object(self.c, 'analyze', return_value=report()), \
             patch.object(self.c, 'candidates', return_value=[]):
            self.assertEqual(self.c.run(), 2)
        self.assertEqual(self.c.state['status'], 'paused')
        self.assertTrue((self.root / 'decisions/0001_attempt_01_check.json').exists())
        self.executor.assert_not_called()

    def test_advisory_unresolved_does_not_override_scientific_stop(self):
        evidence = {'eligible': True, 'chosen_gain': .75, 'required_cells': [],
                    'B_batches_on_current_model': 1, 'success_rate': 1.}
        with patch('bottle_scan.agentic_evaluation.prospective_B_evidence', return_value=evidence):
            stop = self.c.stop_evidence({'stopping_eligible': True}, [{'id': 'advice', 'unresolved': True}])
        self.assertTrue(stop['eligible'])
        self.assertEqual(stop['advisory_design_candidates'], ['advice'])

    def test_freeze_uses_same_gain_and_records_budget_not_convergence(self):
        fit = {'model_id': 'm1', 'usable_for_B': True}
        atomic_json(self.root / 'models/m1.json', fit)
        self.c.state['models']['m1'] = 'models/m1.json'
        evidence = {'eligible': False, 'chosen_gain': .75, 'required_cells': [],
                    'B_batches_on_current_model': 1, 'success_rate': .7}
        with patch('bottle_scan.agentic_evaluation.prospective_B_evidence', return_value=evidence):
            with self.assertRaisesRegex(ValueError, '停止条件'):
                self.c.freeze({'best_model': fit, 'stopping_eligible': True}, 'agent request')
            self.c.freeze({'best_model': fit, 'stopping_eligible': False}, '预算耗尽', trigger='budget')
        frozen = self.c.state['frozen']
        self.assertEqual(frozen['policy']['gain'], .75)
        self.assertEqual(frozen['freeze_trigger'], 'budget')
        self.assertFalse(frozen['exploration_eligible'])
        self.assertEqual(self.c.state['phase'], 'validation')


if __name__ == '__main__':
    unittest.main()
