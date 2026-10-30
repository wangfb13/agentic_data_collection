"""Adaptive range/resolution planning tests; only temporary offline JSON."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from bottle_scan.agentic_design import enabled, initialize, effective_settings, propose_adaptive


def settings():
    return {'simulate': True,
            'scope': {'adaptive': True, 'min_mm': -25., 'max_mm': 25., 'step_mm': 5.,
                      'initial_min_mm': -10., 'initial_max_mm': 10., 'min_step_mm': .5,
                      'expansion_step_mm': 5., 'allow_expand': True, 'boundary_min_trials': 4},
            'budget': {'batch_trials': 8},
            'learning': {'min_per_level': 2, 'target_mae_mm': 1.}}


def report(levels=(-10., -5., 0., 5., 10.), *, noise=.1, actual_sd=.1, cv=0., model=True):
    return {'best_model': {'model_id': 'frozen1', 'fingers': [{'coordinate': 'u'}]} if model else None,
            'quality': {'blocking_reasons': []}, 'diagnostics': [],
            'evidence': {'sessions': [], 'observations': [],
                         'level_stats': [{'offset_mm': float(x), 'n': 4, 'measured_n': 4,
                                          'actual_mean_mm': float(x), 'actual_sd_mm': actual_sd,
                                          'cv_bias_mm': cv, 'cv_mae_mm': abs(cv),
                                          'fingers': [{'sd_equivalent_mm': noise}, {'sd_equivalent_mm': noise}]}
                                         for x in levels]}}


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def add_outcomes(self, rep, offset, passed=True, blocked=False, n=4, *, name=None):
        root = self.root / (name or str(offset)); root.mkdir(exist_ok=True)
        (root / 'session_meta.json').write_text(json.dumps({'schema': 'e1e2_session_v1', 'simulated': True,
                                                         'agentic': {'role': 'exploration'}}))
        for i in range(n):
            folder = root / 'trials' / str(i); folder.mkdir(parents=True, exist_ok=True)
            row = {'id': str(i), 'cond': 'A', 'delta_mm': float(offset), 'bottle': 'light' if i % 2 == 0 else 'heavy',
                   'complete': True, 'recording_complete': not blocked, 'label': 'normal' if passed else 'invalid',
                   'status': 'hold_complete' if passed else 'skipped_bound' if blocked else 'no_contact',
                   'probe_initial': {'valid': passed, 'descriptors': [{'valid': passed, 'touches_edge': False}] * 2},
                   'bilateral': passed, 'target_reached': passed, 'lifted': passed,
                   'hold_complete': passed, 'dropped': False}
            (folder / 'meta.json').write_text(json.dumps(row))
            (folder / 'recording_status.json').write_text(json.dumps({'complete': not blocked}))
        rep['evidence']['sessions'].append({'path': str(root)})

    def test_adaptive_is_opt_in_and_empty_proposals_when_disabled(self):
        cfg = settings(); cfg['scope'].pop('adaptive')
        self.assertFalse(enabled(cfg))
        self.assertEqual(propose_adaptive(report(), cfg, initialize(cfg, report()), 8, 0), [])

    def test_initialize_seed_span_median_spacing_clipped_hard_contains_zero(self):
        cfg = settings(); r = report((-40., -15., -10., -5., 0., 5., 10., 40.))
        design = initialize(cfg, r)
        self.assertEqual(design['min_mm'], -15.)
        self.assertEqual(design['max_mm'], 10.)
        self.assertEqual(design['step_mm'], 5.)
        self.assertEqual(design['origin'], 'measured_A_seed_levels')
        self.assertEqual(initialize(cfg, report((2., 4., 9.)))['min_mm'], 0.)

    def test_initialize_fallback_and_no_settings_mutation(self):
        cfg = settings(); old = copy.deepcopy(cfg)
        design = initialize(cfg, report(()))
        self.assertEqual((design['min_mm'], design['max_mm'], design['step_mm']), (-10., 10., 5.))
        design['max_mm'] = 15.
        effective = effective_settings(cfg, design)
        self.assertEqual(effective['scope']['initial_max_mm'], 15.)
        self.assertEqual(effective['scope']['max_mm'], 15.)
        self.assertEqual(cfg, old)
        self.assertEqual(effective['budget'], cfg['budget'])

    def test_seed_initialization_excludes_edge_observations(self):
        cfg = settings(); r = report((-10., -5., 0., 5., 10.))
        r['evidence']['observations'] = [
            {'planned_delta_mm': x, 'measurement': {'source': 'measured_tcp_probe_end'},
             'descriptors': [{'valid': True, 'touches_edge': abs(x) == 10.}] * 2}
            for x in (-10., -5., 0., 5., 10.)]
        d = initialize(cfg, r)
        self.assertEqual((d['min_mm'], d['max_mm']), (-5., 5.))

    def test_refine_halves_step_and_proposes_midpoints_even_counts(self):
        cfg = settings(); r = report(cv=2.); d = initialize(cfg, r)
        candidates = propose_adaptive(r, cfg, d, 8, 2)
        refine = next(c for c in candidates if c['design_action'] == 'refine')
        self.assertEqual(refine['design_next'], {'min_mm': -10., 'max_mm': 10., 'step_mm': 2.5})
        self.assertTrue(all(c['offset_mm'] in (-7.5, -2.5, 2.5, 7.5) for c in refine['cells']))
        self.assertEqual(refine['trial_count'], 8)
        self.assertTrue(all(c['reps'] % 2 == 0 for c in refine['cells']))
        self.assertTrue(refine['unresolved'])
        self.assertEqual(candidates, propose_adaptive(r, cfg, d, 8, 2))

    def test_numerically_linear_good_model_needs_no_finer_data(self):
        cfg = settings(); cfg['scope']['allow_expand'] = False
        r = report(); d = initialize(cfg, r)
        self.assertEqual(propose_adaptive(r, cfg, d, 8, 1), [])

    def test_large_random_noise_repeats_without_smaller_nominal_step(self):
        cfg = settings(); r = report(noise=5., cv=2.); d = initialize(cfg, r)
        result = propose_adaptive(r, cfg, d, 8, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['design_action'], 'repeat')
        self.assertEqual(result[0]['evidence']['mode'], 'repeat_before_refine')
        self.assertEqual(result[0]['design_next']['step_mm'], 5.)
        self.assertTrue(all(c['offset_mm'] in (-10., -5., 0., 5., 10.) for c in result[0]['cells']))

    def test_noise_repeat_cap_counts_physical_trials(self):
        cfg = settings(); cfg['scope']['noise_repeat_cap'] = 12
        r = report((-5., 0., 5.), noise=10., cv=2.)
        for s in r['evidence']['level_stats']: s['n'] = 10
        result = propose_adaptive(r, cfg, initialize(cfg, r), 8, 1)
        self.assertTrue(all(cell['reps'] == 2 for cell in result[0]['cells']))
        self.assertEqual(result[0]['trial_count'], 6)
        for s in r['evidence']['level_stats']: s['n'] = 12
        self.assertEqual(propose_adaptive(r, cfg, initialize(cfg, r), 8, 1), [])

    def test_poor_actual_tcp_separation_prevents_claimed_refinement(self):
        cfg = settings(); r = report(cv=2., actual_sd=2.)
        d = initialize(cfg, r)
        result = propose_adaptive(r, cfg, d, 8, 1)
        self.assertTrue(result)
        self.assertTrue(all(c['evidence']['mode'] == 'repeat_before_refine' for c in result))
        self.assertTrue(all(c['design_next']['step_mm'] == 5. for c in result))
        cfg2 = settings(); r2 = report(cv=2.)
        for s in r2['evidence']['level_stats']: s['actual_mean_mm'] = 0.
        result2 = propose_adaptive(r2, cfg2, initialize(cfg2, r2), 8, 1)
        self.assertEqual(result2[0]['evidence']['mode'], 'repeat_before_refine')

    def test_expand_each_side_only_after_measured_successful_boundary(self):
        cfg = settings(); r = report(); d = initialize(cfg, r)
        self.assertEqual(propose_adaptive(r, cfg, d, 8, 1), [])
        self.add_outcomes(r, -10.); self.add_outcomes(r, 10.)
        candidates = propose_adaptive(r, cfg, d, 8, 1)
        self.assertEqual({c['design_action'] for c in candidates}, {'expand_left', 'expand_right'})
        left = next(c for c in candidates if c['design_action'] == 'expand_left')
        right = next(c for c in candidates if c['design_action'] == 'expand_right')
        self.assertEqual(left['design_next']['min_mm'], -15.)
        self.assertEqual(left['design_next']['max_mm'], 10.)
        self.assertEqual(right['design_next']['max_mm'], 15.)
        self.assertEqual(right['design_next']['min_mm'], -10.)
        self.assertEqual({v['offset_mm'] for v in right['cells']}, {10., 15.})
        self.assertEqual(right['trial_count'], 4)
        self.assertTrue(all(c['reps'] == 2 for c in right['cells']))

    def test_expansion_clipped_at_25mm_and_never_crosses_known_failure(self):
        cfg = settings(); r = report((-20., -15., -10., -5., 0., 5., 10., 15., 20., 23.)); d = initialize(cfg, r)
        self.add_outcomes(r, 23.)
        candidates = propose_adaptive(r, cfg, d, 8, 1)
        self.assertEqual(candidates[0]['design_next']['max_mm'], 25.)
        self.add_outcomes(r, 24., passed=False)
        self.assertFalse(any(c['design_action'] == 'expand_right' for c in propose_adaptive(r, cfg, d, 8, 1)))

    def test_failed_boundary_bisects_inward_preserves_scope(self):
        cfg = settings(); r = report((-10., -5., 0., 5.)); d = {'min_mm': -10., 'max_mm': 10., 'step_mm': 5.}
        self.add_outcomes(r, 5.); self.add_outcomes(r, 10., passed=False)
        result = propose_adaptive(r, cfg, d, 8, 1)
        refine = next(c for c in result if c['design_action'] == 'refine')
        self.assertIn(7.5, [c['offset_mm'] for c in refine['cells']])
        self.assertEqual(refine['design_next']['max_mm'], 10.)
        self.assertTrue(refine['evidence']['intervals'][0]['contact_transition'])
        self.assertFalse(any(c['design_action'] == 'expand_right' for c in result))

    def test_execution_height_skip_is_not_tactile_boundary(self):
        cfg = settings(); r = report((-10., -5., 0., 5.)); d = {'min_mm': -10., 'max_mm': 10., 'step_mm': 5.}
        self.add_outcomes(r, 5.); self.add_outcomes(r, 10., passed=False, blocked=True)
        result = propose_adaptive(r, cfg, d, 8, 1)
        self.assertFalse(any(7.5 in [cell['offset_mm'] for cell in c['cells']] for c in result))
        self.assertFalse(any(c['design_action'] == 'expand_right' for c in result))

    def test_nonzero_B_feedback_drives_local_refinement(self):
        cfg = settings(); r = report(); r['diagnostics'] = [
            {'code': 'B_offset_feedback', 'offset_mm': 5., 'n': 4, 'failures': 4, 'execution_blocked': 0, 'initial_mae_mm': 4.}]
        result = propose_adaptive(r, cfg, initialize(cfg, r), 4, 2)
        self.assertEqual({cell['offset_mm'] for cell in result[0]['cells']}, {2.5, 7.5})

    def test_small_B_residual_does_not_force_endless_refinement(self):
        cfg = settings(); cfg['scope']['allow_expand'] = False
        for residual in (.01, .5, 1.):
            r = report(); r['diagnostics'] = [
                {'code': 'B_offset_feedback', 'offset_mm': 5., 'n': 4, 'failures': 0,
                 'servo_exhausted': 0, 'execution_blocked': 0, 'initial_mae_mm': residual}]
            self.assertEqual(propose_adaptive(r, cfg, initialize(cfg, r), 4, 2), [])

    def test_nonbound_servo_exhaustion_can_request_refinement(self):
        cfg = settings(); cfg['scope']['allow_expand'] = False
        r = report(); r['diagnostics'] = [
            {'code': 'B_offset_feedback', 'offset_mm': 5., 'n': 4, 'failures': 0,
             'servo_exhausted': 2, 'execution_blocked': 0, 'initial_mae_mm': .1}]
        self.assertTrue(propose_adaptive(r, cfg, initialize(cfg, r), 4, 2))
        r['diagnostics'][0]['execution_blocked'] = 2
        self.assertEqual(propose_adaptive(r, cfg, initialize(cfg, r), 4, 2), [])

    def test_unknown_invalid_or_incomplete_is_not_contact_boundary(self):
        cfg = settings(); r = report((-10., -5., 0., 5.))
        d = {'min_mm': -10., 'max_mm': 10., 'step_mm': 5.}
        self.add_outcomes(r, 5.); self.add_outcomes(r, 10., passed=False)
        failed_root = Path(r['evidence']['sessions'][-1]['path'])
        for path in failed_root.glob('trials/*/meta.json'):
            row = json.loads(path.read_text()); row['status'] = 'hold_complete'
            path.write_text(json.dumps(row))
        self.assertEqual(propose_adaptive(r, cfg, d, 8, 1), [])
        for path in failed_root.glob('trials/*/meta.json'):
            row = json.loads(path.read_text()); row['status'] = 'no_contact'; row['recording_complete'] = False
            path.write_text(json.dumps(row))
        self.assertEqual(propose_adaptive(r, cfg, d, 8, 1), [])
        for path in failed_root.glob('trials/*/meta.json'):
            row = json.loads(path.read_text()); row['recording_complete'] = True
            path.write_text(json.dumps(row))
        self.assertTrue(any(c['design_action'] == 'refine' for c in propose_adaptive(r, cfg, d, 8, 1)))

    def test_below_minimum_step_never_requested_and_fills_remaining_gaps(self):
        cfg = settings(); cfg['scope']['min_step_mm'] = 1.
        r = report((-2., 0., 2.), cv=2.)
        d = {'min_mm': -2., 'max_mm': 2., 'step_mm': 1.}
        result = propose_adaptive(r, cfg, d, 4, 1)
        self.assertEqual(result[0]['design_next']['step_mm'], 1.)
        self.assertEqual(result[0]['design_action'], 'sample')
        self.assertEqual({c['offset_mm'] for c in result[0]['cells']}, {-1., 1.})
        fine = report((-1., 0., 1.), cv=2.)
        self.assertEqual(propose_adaptive(fine, cfg, {'min_mm': -1., 'max_mm': 1., 'step_mm': 1.}, 4, 1), [])

    def test_budget_blocking_and_json_validity(self):
        cfg = settings(); r = report(cv=2.); d = initialize(cfg, r)
        self.assertEqual(propose_adaptive(r, cfg, d, 1, 1), [])
        result = propose_adaptive(r, cfg, d, 5, 1)
        self.assertTrue(all(c['trial_count'] <= 4 for c in result))
        json.dumps(result, allow_nan=False)
        r['quality']['blocking_reasons'] = ['protocol changed']
        self.assertEqual(propose_adaptive(r, cfg, d, 8, 1), [])

    def test_expansion_larger_than_five_mm_rejected(self):
        cfg = settings(); cfg['scope']['expansion_step_mm'] = 6.
        with self.assertRaises(ValueError): initialize(cfg, report())


if __name__ == '__main__':
    unittest.main()
