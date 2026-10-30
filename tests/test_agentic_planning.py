"""Offline tests for free agent plans and explicit local execution constraints."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from bottle_scan.agentic_planning import (
    PlanningError, build_context, enabled, validate_plan_response, validate_response,
)
from bottle_scan.e1e2_core import defaults


def settings():
    return {'planning': {'mode': 'agent'}, 'simulate': True,
            'scope': {'min_mm': -25., 'max_mm': 25., 'step_mm': 5., 'min_step_mm': .5,
                      'allow_expand': True, 'expansion_step_mm': 5.},
            'budget': {'batch_trials': 12}, 'policy': {'gains': [.5, .75, 1.]}}


def config():
    cfg = defaults(); cfg['gripper']['min_normalized'] = .336
    cfg['axes_tcp'] = {'a': [1., 0., 0.], 'l': [0., 0., 1.], 'c': [0., -1., 0.]}
    return cfg


def model():
    return {'model_id': 'approved1', 'usable_for_B': True, 'schema': 'agentic_fit_v1',
            'common_range_mm': [-5., 5.], 'validation': {'mae_mm': .5, 'p90_mm': 1.}, 'fingers': []}


def response(cond='A', cells=None):
    return {'action': 'propose_batch', 'reason': 'Test local departure from a linear mapping.',
            'plan': {'cond': cond, 'cells': cells or [{'offset_mm': 0., 'reps': 2}, {'offset_mm': 3.7, 'reps': 2}],
                     'model_id': None if cond == 'A' else 'approved1', 'gain': 1.,
                     'hypothesis': 'The offset-to-centroid relation may have local curvature.',
                     'evidence_refs': ['metrics'], 'expected_outcome': 'Independent repeated PROBE samples resolve local model error.'}}


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings(); self.cfg = config()
        self.design = {'min_mm': -10., 'max_mm': 10., 'step_mm': 5.}
        self.models = {'approved1': model()}
        self.context = {'references': {'metrics': {}, 'no_data': {}}, 'valid_evidence_refs': ['metrics', 'no_data']}
        self.T0 = np.eye(4); self.T0[2, 3] = 800.

    def validate(self, proposal=None, **changes):
        args = {'settings': self.settings, 'cfg': self.cfg, 'design': self.design,
                'approved_models': self.models, 'remaining_trials': 100, 'context': self.context,
                'round_index': 2, 'T0': self.T0}
        args.update(changes)
        return validate_plan_response(proposal or response(), **args)

    def test_legacy_missing_mode_stays_candidate_mode(self):
        self.assertTrue(enabled(self.settings))
        self.assertFalse(enabled({}))
        cfg = copy.deepcopy(self.settings); cfg.pop('planning')
        with self.assertRaises(PlanningError) as got: self.validate(settings=cfg)
        self.assertEqual(got.exception.code, 'planning_mode')

    def test_response_closed_shape_disallows_commands_and_control_changes(self):
        for mutate in [lambda v: v.update(command='PRIVATE_COMMAND'),
                       lambda v: v['plan'].update(speed_mm_s=100),
                       lambda v: v['plan']['cells'][0].update(T0=[1, 2, 3])]:
            value = response(); mutate(value)
            with self.assertRaises(PlanningError) as got: validate_response(value)
            self.assertNotIn('PRIVATE_COMMAND', str(got.exception))
            self.assertTrue(got.exception.code)

    def test_response_requires_finite_values_and_bounded_strings_lists(self):
        variations = [lambda v: v['plan']['cells'][0].update(offset_mm=float('nan')),
                      lambda v: v['plan']['cells'][0].update(offset_mm=True),
                      lambda v: v['plan']['cells'][0].update(reps=True),
                      lambda v: v['plan'].update(gain=float('inf')),
                      lambda v: v['plan'].update(hypothesis=''),
                      lambda v: v['plan'].update(expected_outcome='x' * 2001),
                      lambda v: v['plan'].update(evidence_refs=[]),
                      lambda v: v['plan'].update(cells=[]),
                      lambda v: v['plan'].update(model_id='x' * 129)]
        for mutate in variations:
            value = response(); mutate(value)
            with self.subTest(value=value), self.assertRaises(PlanningError): validate_response(value)

    def test_freeze_and_pause_require_null_plan_and_return_none(self):
        for action in ('freeze', 'pause'):
            value = {'action': action, 'reason': 'Stop for evidence review.', 'plan': None}
            self.assertIsNone(self.validate(value))
            value['plan'] = response()['plan']
            with self.assertRaises(PlanningError): validate_response(value)

    def test_arbitrary_nonuniform_A_not_limited_to_old_candidate_grid(self):
        value = response(cells=[{'offset_mm': -.7, 'reps': 2}, {'offset_mm': 1.4, 'reps': 4}, {'offset_mm': 7.3, 'reps': 2}])
        candidate = self.validate(value)
        self.assertEqual(candidate['origin'], 'agent_proposal')
        self.assertEqual(candidate['design_action'], 'agent_plan')
        self.assertAlmostEqual(candidate['design_next']['step_mm'], 2.1)
        self.assertEqual(candidate['design_next']['min_mm'], -10.)
        self.assertEqual(candidate['design_next']['max_mm'], 10.)
        self.assertTrue(candidate['motion_bounds_checked'])
        self.assertEqual(candidate['proposal'], value)
        self.assertEqual(candidate['id'], self.validate(value)['id'])

    def test_no_heuristic_score_or_failure_is_a_hidden_gate(self):
        self.context['references']['diagnostic:0'] = {'code': 'previous_no_contact', 'failures': 8}
        value = response(cells=[{'offset_mm': 12., 'reps': 2}])
        value['plan']['evidence_refs'] = ['diagnostic:0']
        candidate = self.validate(value)
        self.assertEqual(candidate['design_next']['max_mm'], 12.)
        self.assertEqual(candidate['score'], 0.)

    def test_every_reference_must_be_present_in_context(self):
        value = response(); value['plan']['evidence_refs'] = ['PRIVATE_UNKNOWN_REFERENCE']
        with self.assertRaises(PlanningError) as got: self.validate(value)
        self.assertEqual(got.exception.code, 'unknown_evidence')
        self.assertNotIn('PRIVATE_UNKNOWN_REFERENCE', str(got.exception))

    def test_A_cannot_use_model_or_gain_change(self):
        value = response(); value['plan']['model_id'] = 'approved1'
        with self.assertRaises(PlanningError): self.validate(value)
        value = response(); value['plan']['gain'] = .75
        with self.assertRaises(PlanningError): self.validate(value)

    def test_B_requires_approved_model_gain_and_allows_outside_model_support(self):
        value = response('B', [{'offset_mm': 8., 'reps': 2}])
        candidate = self.validate(value)
        self.assertTrue(any('模型支持范围外' in s for s in candidate['warnings']))
        value['plan']['gain'] = .75
        self.assertEqual(self.validate(value)['policy']['gain'], .75)
        value['plan']['gain'] = .6
        with self.assertRaises(PlanningError): self.validate(value)
        value['plan']['gain'] = 1.; value['plan']['model_id'] = 'unknown'
        with self.assertRaises(PlanningError): self.validate(value)

    def test_B_cannot_expand_current_scope(self):
        with self.assertRaises(PlanningError) as got:
            self.validate(response('B', [{'offset_mm': 12., 'reps': 2}]))
        self.assertEqual(got.exception.code, 'B_expansion')

    def test_A_expansion_one_side_at_most_five_and_never_shrink(self):
        value = response(cells=[{'offset_mm': 15., 'reps': 2}])
        new = self.validate(value)['design_next']
        self.assertEqual(new, {'min_mm': -10., 'max_mm': 15., 'step_mm': 5.})
        for cells in [[{'offset_mm': 15.1, 'reps': 2}], [{'offset_mm': -11., 'reps': 2}, {'offset_mm': 11., 'reps': 2}]]:
            with self.assertRaises(PlanningError): self.validate(response(cells=cells))
        self.settings['scope']['allow_expand'] = False
        with self.assertRaises(PlanningError): self.validate(value)

    def test_duplicate_or_too_close_points_and_odd_reps_rejected(self):
        variations = [[{'offset_mm': 1., 'reps': 2}, {'offset_mm': 1., 'reps': 2}],
                      [{'offset_mm': 1., 'reps': 2}, {'offset_mm': 1.49, 'reps': 2}],
                      [{'offset_mm': 1., 'reps': 3}]]
        for cells in variations:
            with self.assertRaises(PlanningError): self.validate(response(cells=cells))
        self.validate(response(cells=[{'offset_mm': 1., 'reps': 2}, {'offset_mm': 1.5, 'reps': 2}]))

    def test_exact_monitor_insertion_is_in_total_budget(self):
        value = response(cells=[{'offset_mm': 0., 'reps': 6}, {'offset_mm': 4., 'reps': 6}])
        candidate = self.validate(value, remaining_trials=13)
        self.assertEqual(candidate['scan_count'], 12)
        self.assertEqual(candidate['planned_total'], 13)
        self.assertEqual(candidate['monitor_count'], 1)
        with self.assertRaises(PlanningError) as got: self.validate(value, remaining_trials=12)
        self.assertEqual(got.exception.code, 'total_budget')
        value['plan']['cells'][0]['reps'] = 8
        with self.assertRaises(PlanningError) as got: self.validate(value)
        self.assertEqual(got.exception.code, 'scan_budget')

    def test_static_grasp_height_violation_rejects_whole_proposal(self):
        self.cfg['axes_tcp']['a'] = [0., 0., -1.]
        value = response(cells=[{'offset_mm': 0., 'reps': 2}, {'offset_mm': 12., 'reps': 2}])
        with self.assertRaises(PlanningError) as got: self.validate(value)
        self.assertEqual(got.exception.code, 'motion_bounds')
        self.assertEqual(len(value['plan']['cells']), 2)

    def test_retreat_and_lift_targets_are_checked_too(self):
        self.cfg['axes_tcp']['a'] = [0., 0., 1.]
        with self.assertRaises(PlanningError): self.validate(response(cells=[{'offset_mm': 0., 'reps': 2}]))
        self.cfg['axes_tcp']['a'] = [1., 0., 0.]; self.cfg['motion']['lift_mm'] = 61.
        with self.assertRaises(PlanningError): self.validate(response(cells=[{'offset_mm': 0., 'reps': 2}]))

    def test_inserted_zero_monitor_also_gets_motion_check(self):
        self.design = {'min_mm': -10., 'max_mm': 10., 'step_mm': 5.}
        self.cfg['axes_tcp']['a'] = [0., 0., 1.]
        self.cfg['motion']['pre_mm'] = 12.
        self.cfg['design']['zero_every'] = 2
        # Scan at +5 permits retreat to -7; inserted zero retreats to -12 below
        # the fixed -10 limit, so the entire proposal must be rejected.
        with self.assertRaises(PlanningError) as got:
            self.validate(response(cells=[{'offset_mm': 5., 'reps': 2}]))
        self.assertEqual(got.exception.code, 'motion_bounds')

    def test_missing_T0_warns_and_invalid_T0_is_rejected(self):
        candidate = self.validate(T0=None)
        self.assertFalse(candidate['motion_bounds_checked'])
        self.assertTrue(candidate['warnings'])
        with self.assertRaises(PlanningError): self.validate(T0=np.zeros((4, 4)))

    def test_candidate_is_json_serializable_and_does_not_mutate_inputs(self):
        value = response(); before = copy.deepcopy(value)
        candidate = self.validate(value)
        json.dumps(candidate, allow_nan=False)
        candidate['proposal']['plan']['cells'][0]['reps'] = 10
        self.assertEqual(value, before)


class ContextTests(unittest.TestCase):
    def test_cold_start_and_stop_evidence_references(self):
        r = {'metrics': {}, 'quality': {}, 'evidence': {}, 'diagnostics': []}
        stop = {'allowed': False, 'missing_cells': [{'offset_mm': 5., 'bottle': 'heavy'}]}
        ctx = build_context(r, settings(), config(), None, [], {}, 100, stop, [], 0)
        self.assertEqual(ctx['planning_mode'], 'agent')
        self.assertIn('metrics', ctx['valid_evidence_refs'])
        self.assertIn('no_data', ctx['valid_evidence_refs'])
        self.assertEqual(ctx['references']['stop']['missing_cells'][0]['bottle'], 'heavy')
        json.dumps(ctx, allow_nan=False)

    def test_context_contains_all_key_diagnostics_and_no_raw_DD(self):
        r = {'metrics': {'mae_mm': 1.2}, 'quality': {'raw_dd': [99] * 1000},
             'diagnostics': [{'code': 'B_offset_feedback', 'offset_mm': 2., 'initial_mae_mm': 3.}, {'code': 'execution_bound'}],
             'evidence': {'level_stats': [{'offset_mm': 2., 'actual_mean_mm': 2.1, 'actual_sd_mm': .3, 'n': 4}],
                          'observations': [{'raw_dd': [99] * 10000}]}}
        history = [{'id': 'h1', 'status': 'completed', 'config': {'secret': 'not_for_context'}}]
        ctx = build_context(r, settings(), config(), {'min_mm': -10., 'max_mm': 10., 'step_mm': 5.},
                            history, {'approved1': model()}, 50, {}, [{'score': 999, 'cond': 'A', 'cells': []}], 2)
        encoded = json.dumps(ctx)
        self.assertNotIn('raw_dd', encoded)
        self.assertNotIn('not_for_context', encoded)
        self.assertNotIn('score', ctx['optional_examples'][0])
        self.assertIn('level:2', ctx['references'])
        self.assertIn('diagnostic:1', ctx['references'])
        self.assertIn('model:approved1', ctx['references'])
        self.assertEqual(ctx['references']['level:2']['actual_sd_mm'], .3)

    def test_failures_are_evidence_not_hidden_prohibitions(self):
        r = {'evidence': {}, 'metrics': {}, 'diagnostics': []}
        outcome = {'offset_mm': 5., 'passed': False, 'contact_force_failure': True, 'blocked': False,
                   'complete': True, 'recording': True, 'reason': 'no_contact'}
        with patch('bottle_scan.agentic_design._load_outcomes', return_value=[outcome]):
            ctx = build_context(r, settings(), config(), None, [], {}, 50, {}, [], 1)
        self.assertEqual(ctx['references']['outcome:5']['contact_force_failures'], 1)


if __name__ == '__main__':
    unittest.main()
