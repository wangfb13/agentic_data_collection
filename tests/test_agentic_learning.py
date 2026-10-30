"""Offline independent-trial learning tests. No hardware, ROS, or network."""
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bottle_scan.agentic_learning import (
    AgenticCorrection, analyze_sessions, finger_estimate, frozen_evaluation,
    model_error, propose_candidates, write_plots,
)


def settings():
    return {'simulate': True, 'scope': {'min_mm': -2., 'max_mm': 10., 'step_mm': 2., 'allow_expand': False},
            'budget': {'batch_trials': 12},
            'learning': {'min_levels': 3, 'min_per_level': 2, 'target_mae_mm': 1., 'target_p90_mm': 2., 'min_gain_mm': .1},
            'policy': {'gains': [.5, .75, 1.]},
            'validation': {'offsets_mm': [-2., 0., 2., 6., 10.], 'reps': 4, 'success_target': .8}}


def desc(x, *, edges=None):
    sides = {key: False for key in ('left', 'right', 'top', 'bottom')}
    sides.update(edges or {})
    return {'valid': True, 'centroid_px': [float(x), 40.], 'touches_edge': any(sides.values()),
            'edge_sides': sides, 'area_px': 50, 'bbox_xyxy_px': [10, 10, 70, 70]}


def servo():
    return {'tol_mm': 2., 'max_delta_mm': 15., 'no_contact_delta_mm': 8., 'stall_delta_mm': -8.,
            'root_edge_delta_mm': -6., 'tip_edge_delta_mm': 6., 'gain': 1.}


class EvidenceFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.counter = 0

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, name='train', cond='A', *, offsets=(-2., 0., 2., 6., 10.), reps=4,
             fun=None, actual_shift=0., role='exploration', model=None, measured=True,
             label='normal', simulated=True, T0=None, patch_cfg=None):
        root = self.root / name
        root.mkdir()
        T = np.eye(4) if T0 is None else np.asarray(T0, float)
        cfg = {'arm': 'right', 'axes_tcp': {'a': [1., 0., 0.], 'l': [0., 0., 1.], 'c': [0., 1., 0.]},
               'force': {'normal_axis': 2, 'normal_signs': [-1., -1.], 'target': 5., 'probe_fraction': .3},
               'tactile': {'dd_threshold': .022}, 'hold_s': 2., 'servo': servo()}
        cfg.update(patch_cfg or {})
        meta = {'schema': 'e1e2_session_v1', 'simulated': simulated, 'T0': T.tolist(), 'config': cfg,
                'cond': cond, 'agentic': {'role': role}, 'fit': model}
        (root / 'session_meta.json').write_text(json.dumps(meta))
        for offset in offsets:
            for repeat in range(reps):
                self.counter += 1
                actual = float(offset + actual_shift)
                y = fun(actual, repeat) if fun else 60. - 3. * actual
                pose = T.copy(); pose[0, 3] += actual
                probe = {'valid': True, 'descriptors': [desc(y), desc(y - 5)], 'timestamp_ns': 1000}
                if measured:
                    probe['measured_tcp_mm'] = pose.tolist()
                    probe['actual_delta_mm'] = actual
                row = {'id': f'{self.counter:05d}', 'cond': cond, 'kind': 'scan', 'complete': True,
                       'recording_complete': True, 'delta_mm': offset, 'repeat': repeat // 2,
                       'bottle': 'light' if repeat % 2 == 0 else 'heavy', 'label': label,
                       'status': 'hold_complete', 'probe_initial': probe, 'probe_final': probe,
                       'bilateral': True, 'lifted': True, 'hold_complete': True, 'dropped': False,
                       'regrasps': 0, 'correction_history': []}
                folder = root / 'trials' / row['id']; folder.mkdir(parents=True)
                (folder / 'meta.json').write_text(json.dumps(row))
                (folder / 'recording_status.json').write_text(json.dumps({'complete': True}))
        return root

    @staticmethod
    def mutate(root, fn):
        for path in sorted((root / 'trials').glob('*/meta.json')):
            row = json.loads(path.read_text()); fn(row); path.write_text(json.dumps(row))


class LearningTests(EvidenceFixture):
    def test_measured_position_not_command_used_for_fit(self):
        root = self.make(actual_shift=.4)
        report = analyze_sessions([root], settings())
        self.assertFalse(report['quality']['blocking_reasons'])
        model = report['best_model']; self.assertIsNotNone(model)
        self.assertAlmostEqual(model['fingers'][0]['intercept_px'], 60., places=6)
        self.assertAlmostEqual(model['fingers'][0]['slope_px_per_mm'], -3., places=6)
        self.assertAlmostEqual(report['evidence']['observations'][0]['x_mm'], -1.6)
        self.assertLess(report['metrics']['mae_mm'], 1e-8)

    def test_actual_tcp_is_unified_to_common_T0(self):
        first = self.make('first')
        T = np.eye(4); T[0, 3] = 1.
        second = self.make('second', T0=T, fun=lambda x, rep: 60. - 3. * (x + 1.))
        report = analyze_sessions([first, second], settings())
        self.assertFalse(report['quality']['blocking_reasons'])
        observations = [o for o in report['evidence']['observations'] if o['session'] == str(second)]
        self.assertAlmostEqual(observations[0]['x_mm'], -1.)
        self.assertAlmostEqual(report['best_model']['fingers'][0]['intercept_px'], 60.)
        self.assertEqual(report['metrics']['method'], 'held_out_session')

    def test_B_and_validation_do_not_train(self):
        A = self.make('A')
        B = self.make('B', cond='B', fun=lambda x, r: 1000 + x)
        validation = self.make('v', role='validation', fun=lambda x, r: -1000 + x)
        result = analyze_sessions([A, B, validation], settings())
        self.assertAlmostEqual(result['best_model']['fingers'][0]['slope_px_per_mm'], -3.)
        self.assertEqual(result['quality']['mapping_trials'], 20)
        self.assertEqual(result['quality']['excluded_counts']['B_diagnostic_only'], 20)
        self.assertEqual(result['quality']['excluded_counts']['frozen_validation_never_trains'], 20)
        self.assertEqual(result['best_model']['source_sessions'], [str(A)])

    def test_moved_invalid_and_incomplete_excluded_without_erasing_outcomes(self):
        A = self.make()
        labels = iter(['pushed', 'tipped', 'invalid'] + ['normal'] * 17)
        self.mutate(A, lambda row: row.update(label=next(labels)))
        result = analyze_sessions([A], settings())
        self.assertEqual(result['quality']['mapping_trials'], 17)
        self.assertEqual(result['quality']['trials_total'], 20)
        self.assertEqual(result['quality']['excluded_counts']['invalid_or_object_moved'], 3)

    def test_force_or_T0_incompatibility_blocks_models(self):
        A = self.make('A')
        other = self.make('other', patch_cfg={'force': {'target': 7.}})
        result = analyze_sessions([A, other], settings())
        self.assertTrue(result['quality']['blocking_reasons'])
        self.assertIsNone(result['best_model'])
        self.assertEqual(propose_candidates(result, settings(), 12, 1), [])
        T = np.eye(4); T[0, 3] = 10.
        far = self.make('far', T0=T)
        self.assertTrue(analyze_sessions([A, far], settings())['quality']['blocking_reasons'])

    def test_simulated_evidence_cannot_be_used_in_real_mode(self):
        A = self.make()
        cfg = settings(); cfg['simulate'] = False
        result = analyze_sessions([A], cfg)
        self.assertTrue(any('simulation/real' in x for x in result['quality']['blocking_reasons']))
        self.assertIsNone(result['best_model'])

    def test_planned_only_cannot_claim_precision_stopping(self):
        A = self.make(measured=False)
        result = analyze_sessions([A], settings())
        self.assertIsNotNone(result['best_model'])
        self.assertEqual(result['metrics']['measured_fraction'], 0.)
        self.assertFalse(result['stopping_eligible'])
        self.assertTrue(any(x['code'] == 'planned_position_only' for x in result['diagnostics']))

    def test_verified_zero_anchor_and_no_unmeasured_anchor(self):
        A = self.make()
        result = analyze_sessions([A], settings())
        anchored = [m for m in result['models'] if m['anchored']]
        self.assertTrue(anchored)
        self.assertTrue(anchored[0]['fingers'][0]['zero_anchor']['verified'])
        legacy = self.make('legacy', measured=False)
        self.assertFalse(any(m['anchored'] for m in analyze_sessions([legacy], settings())['models']))

    def test_nonlinear_candidate_selected_by_heldout_mm_errors(self):
        A = self.make(offsets=(-2., 0., 2., 4., 6., 8., 10.), reps=8,
                      fun=lambda x, r: 100. - (1.5 * x + .20 * x * x))
        report = analyze_sessions([A], settings())
        self.assertEqual(report['best_model']['family'], 'monotone_piecewise')
        self.assertLess(report['metrics']['mae_mm'], 1e-8)
        self.assertGreater(next(m['validation']['mae_mm'] for m in report['models'] if m['family'] == 'linear'), .2)

    def test_cv_keeps_trial_groups_independent(self):
        A = self.make()
        report = analyze_sessions([A], settings())
        seen = {}
        for p in report['best_model']['validation_predictions']:
            seen.setdefault(p['group'], set()).add(p['fold'])
        self.assertTrue(all(len(folds) == 1 for folds in seen.values()))
        self.assertGreater(len(seen), 1)

    def test_duplicate_roots_not_counted_twice(self):
        A = self.make()
        self.assertEqual(analyze_sessions([A, A], settings())['quality']['mapping_trials'], 20)

    def test_initial_empty_campaign_proposes_bounded_balanced_A(self):
        report = analyze_sessions([], settings())
        proposals = propose_candidates(report, settings(), 12, 0)
        self.assertEqual(len(proposals), 1)
        p = proposals[0]; self.assertEqual(p['cond'], 'A')
        self.assertEqual(p['trial_count'], 12)
        self.assertTrue(all(-2 <= cell['offset_mm'] <= 10 and cell['reps'] % 2 == 0 for cell in p['cells']))
        self.assertEqual(propose_candidates(report, settings(), 1, 0), [])

    def test_policy_candidates_require_actual_prospective_test(self):
        A = self.make()
        report = analyze_sessions([A], settings())
        candidates = propose_candidates(report, settings(), 10, 2)
        B = [c for c in candidates if c['cond'] == 'B']
        self.assertEqual({c['policy']['gain'] for c in B}, {.5, .75, 1.})
        self.assertTrue(all(c['requires_prospective_evaluation'] and c['trial_count'] <= 10 for c in B))
        self.assertTrue(all(c['model_id'] == report['best_model']['model_id'] for c in B))

    def test_nonzero_B_failure_and_residual_prioritize_A_neighbors(self):
        A = self.make('A', offsets=(-2., 0., 2., 4., 6., 8., 10.))
        B = self.make('B', cond='B', offsets=(6.,))
        self.mutate(B, lambda r: r.update(lifted=False, hold_complete=False, servo_exhausted=True,
                                        correction_history=[{'e_mm': 10., 'delta_mm': -10., 'gain': 1.}]))
        cfg = settings(); cfg['budget']['batch_trials'] = 4
        result = analyze_sessions([A, B], cfg)
        feedback = next(d for d in result['diagnostics'] if d['code'] == 'B_offset_feedback')
        self.assertEqual(feedback['offset_mm'], 6.)
        self.assertEqual(feedback['failures'], 4)
        self.assertEqual(feedback['servo_exhausted'], 4)
        self.assertAlmostEqual(feedback['initial_mae_mm'], 4.)
        self.assertEqual(feedback['position_sources'], {'measured_tcp_probe_end': 4})
        candidate = next(c for c in propose_candidates(result, cfg, 4, 2) if c['cond'] == 'A')
        self.assertEqual(candidate['cells'][0]['offset_mm'], 6.)
        self.assertIn(candidate['cells'][1]['offset_mm'], (4., 8.))
        self.assertIn('B_feedback_near_6', candidate['reason'])
        self.assertEqual(result['quality']['mapping_trials'], 28)

    def test_B_planned_residual_fallback_is_explicit(self):
        A = self.make('A')
        B = self.make('B', cond='B', offsets=(6.,), measured=False)
        self.mutate(B, lambda r: r.update(correction_history=[{'e_mm': 8.}]))
        result = analyze_sessions([A, B], settings())
        feedback = next(d for d in result['diagnostics'] if d['code'] == 'B_offset_feedback')
        self.assertEqual(feedback['position_sources'], {'planned_delta_fallback': 4})
        self.assertEqual(feedback['examples'][0]['position_source'], 'planned_delta_fallback')

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib optional plot test')
    def test_exploration_figures_are_written(self):
        report = analyze_sessions([self.make()], settings())
        artifacts = write_plots(report, self.root / 'figures')
        self.assertEqual(len(artifacts['files']), 3)
        self.assertFalse(artifacts['warnings'])
        self.assertTrue(all(Path(p).read_bytes().startswith(b'\x89PNG') for p in artifacts['files']))

    def test_reports_are_json_finite_and_serializable(self):
        A = self.make()
        result = analyze_sessions([A], settings())
        json.dumps(result, allow_nan=False)
        json.dumps(propose_candidates(result, settings(), 12, 1), allow_nan=False)

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'h5py optional for legacy H5 test')
    def test_legacy_H5_measured_tcp_preferred_over_command_and_inline(self):
        import h5py
        A = self.make(actual_shift=.4)
        for path in sorted((A / 'trials').glob('*/meta.json')):
            row = json.loads(path.read_text())
            actual = row['delta_mm'] + .4
            row['probe_initial']['window'] = [{'frame_ns': [990, 1000, 1010]}]
            row['probe_initial']['measured_tcp_mm'][0][3] = 999.
            path.write_text(json.dumps(row))
            with h5py.File(path.parent / 'streams.h5', 'w') as f:
                g = f.create_group('streams/tcp_measured')
                g.create_dataset('monotonic_ns', data=[990, 1000, 1010])
                pose = np.repeat(np.eye(4)[None], 3, axis=0); pose[:, 0, 3] = actual
                g.create_dataset('data', data=pose)
        report = analyze_sessions([A], settings())
        self.assertEqual(report['quality']['coordinate_sources']['measured_tcp_probe_window'], 20)
        self.assertAlmostEqual(report['best_model']['fingers'][0]['intercept_px'], 60.)


class CorrectionTests(EvidenceFixture):
    def get_model(self):
        A = self.make()
        return analyze_sessions([A], settings())['best_model']

    def test_tactile_only_interface_and_estimation(self):
        model = self.get_model()
        correction = AgenticCorrection(model, servo())
        self.assertEqual(list(inspect.signature(correction.decide).parameters), ['status', 'descriptors', 'bilateral'])
        self.assertAlmostEqual(correction.estimate([desc(48), desc(43)]), 4.)
        decision = correction.decide('contact', [desc(48), desc(43)], True)
        self.assertFalse(decision['accepted']); self.assertAlmostEqual(decision['delta_mm'], -4.)

    def test_no_extrapolation_no_silent_clamping(self):
        model = self.get_model()
        self.assertIsNone(model_error(model, [desc(-100), desc(-105)]))
        correction = AgenticCorrection(model, servo())
        d = correction.decide('contact', [desc(-100), desc(-105)], True)
        self.assertFalse(d['accepted']); self.assertEqual(d['delta_mm'], 0.)
        self.assertIn('outside_model_domain', d['reason'])

    def test_gain_changes_centroid_delta_only(self):
        model = self.get_model(); cfg = servo(); cfg['gain'] = .5
        correction = AgenticCorrection(model, cfg)
        self.assertAlmostEqual(correction.decide('contact', [desc(48), desc(43)], True)['delta_mm'], -2.)
        self.assertEqual(correction.decide('no_contact', [], False)['delta_mm'], 8.)
        self.assertEqual(correction.decide('stall_no_contact', [], False)['delta_mm'], -8.)
        self.assertEqual(correction.decide('contact', [desc(48, edges={'left': True}), desc(43)], True)['delta_mm'], -6.)

    def test_acceptance_uses_both_fingers_and_edge(self):
        model = self.get_model(); correction = AgenticCorrection(model, servo())
        self.assertTrue(correction.decide('contact', [desc(60), desc(55)], True)['accepted'])
        self.assertFalse(correction.decide('contact', [desc(60), desc(55)], False)['accepted'])
        self.assertFalse(correction.decide('contact', [desc(60, edges={'top': True}), desc(55)], True)['accepted'])
        self.assertIsNone(correction.estimate([desc(60)]))

    def test_malformed_and_unvalidated_models_fail_closed(self):
        model = self.get_model()
        model['fingers'][0]['knots_centroid_px'] = [2., 2.]
        with self.assertRaises(ValueError):
            AgenticCorrection(model, servo())
        model = copy.deepcopy(model)
        model['usable_for_B'] = False
        with self.assertRaises(ValueError):
            AgenticCorrection(model, servo())


class FrozenTests(EvidenceFixture):
    def validation(self, reps=4):
        train = self.make('train')
        model = analyze_sessions([train], settings())['best_model']
        A = self.make('A', role='validation', reps=reps)
        B = self.make('B', cond='B', role='validation', reps=reps, model=model)
        cfg = settings(); cfg['validation']['reps'] = reps
        return A, B, model, cfg

    def test_all_success_small_sample_is_inconclusive(self):
        A, B, model, cfg = self.validation()
        report = frozen_evaluation([A], [B], model, cfg)
        self.assertTrue(report['fixed_plan_complete'])
        self.assertTrue(report['accuracy_met'])
        self.assertEqual(report['outcome'], 'inconclusive')
        self.assertFalse(report['success_met'])
        self.assertEqual(report['totals']['B']['success_rate'], 1.)

    def test_predeclared_sufficient_all_success_can_pass(self):
        A, B, model, cfg = self.validation(20)
        report = frozen_evaluation([A], [B], model, cfg)
        self.assertEqual(report['outcome'], 'passed')
        self.assertTrue(all(b['wilson_95'][0] > .8 for b in report['bins']))

    def test_validation_failure_not_erased_as_invalid(self):
        A, B, model, cfg = self.validation(20)
        self.mutate(B, lambda row: row.update(lifted=False, hold_complete=False, status='no_contact'))
        report = frozen_evaluation([A], [B], model, cfg)
        self.assertEqual(report['outcome'], 'not_met')
        self.assertEqual(report['totals']['B']['success_rate'], 0.)

    def test_training_overlap_prevents_validation_claim(self):
        A, B, model, cfg = self.validation(20)
        model['source_sessions'].append(str(A))
        report = frozen_evaluation([A], [B], model, cfg)
        self.assertEqual(report['outcome'], 'inconclusive')
        self.assertTrue(any('overlap' in x for x in report['issues']))

    def test_wrong_model_or_incomplete_plan_is_inconclusive(self):
        A, B, model, cfg = self.validation(4)
        meta_path = B / 'session_meta.json'; meta = json.loads(meta_path.read_text())
        meta['fit']['model_id'] = 'changed'; meta_path.write_text(json.dumps(meta))
        report = frozen_evaluation([A], [B], model, cfg)
        self.assertEqual(report['outcome'], 'inconclusive')
        self.assertTrue(any('different model' in x for x in report['issues']))

    def test_mixed_session_model_and_frozen_gain_are_verified(self):
        A, B, model, cfg = self.validation(20)
        for folder in list((A / 'trials').iterdir()):
            folder.rename(B / 'trials' / folder.name)
        path = B / 'session_meta.json'; meta = json.loads(path.read_text())
        meta['cond'] = 'mixed'; path.write_text(json.dumps(meta))
        cfg['validation']['frozen_gain'] = 1.
        good = frozen_evaluation([B], [B], model, cfg)
        self.assertEqual(good['outcome'], 'passed')
        self.assertEqual(good['totals']['B']['n'], 100)
        meta['fit']['model_id'] = 'wrong'; path.write_text(json.dumps(meta))
        bad = frozen_evaluation([B], [B], model, cfg)
        self.assertEqual(bad['outcome'], 'inconclusive')
        self.assertTrue(any('different model' in x for x in bad['issues']))
        meta['fit']['model_id'] = model['model_id']; path.write_text(json.dumps(meta))
        cfg['validation']['frozen_gain'] = .75
        bad = frozen_evaluation([B], [B], model, cfg)
        self.assertEqual(bad['outcome'], 'inconclusive')
        self.assertTrue(any('frozen gain' in x for x in bad['issues']))

    def test_B_gain_cannot_change_inside_trial(self):
        A, B, model, cfg = self.validation(20)
        cfg['validation']['frozen_gain'] = 1.
        self.mutate(B, lambda row: row.update(correction_history=[{'gain': .5}]))
        report = frozen_evaluation([A], [B], model, cfg)
        self.assertEqual(report['outcome'], 'inconclusive')
        self.assertTrue(any('within a trial' in x for x in report['issues']))

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib optional plot test')
    def test_final_validation_figures_are_written(self):
        A, B, model, cfg = self.validation()
        report = frozen_evaluation([A], [B], model, cfg)
        artifacts = write_plots(report, self.root / 'figures')
        self.assertEqual(len(artifacts['files']), 2)
        self.assertTrue(all(Path(p).read_bytes().startswith(b'\x89PNG') for p in artifacts['files']))


if __name__ == '__main__':
    unittest.main()
