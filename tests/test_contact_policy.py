"""Offline ContactPolicy training/runtime boundaries; never connects hardware."""
import copy
import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from bottle_scan.contact_policy import ContactPolicy, feature_vector, train_contact_policy
from tests.test_agentic_learning import EvidenceFixture, desc, servo, settings


def policy_settings(**overrides):
    cfg = settings()
    cfg['contact_policy'] = {'epochs': 120, 'ensemble_size': 3, **overrides}
    return cfg


class ContactPolicyTests(EvidenceFixture):
    def train(self, **overrides):
        root = self.make()
        model = train_contact_policy([root], policy_settings(**overrides))
        self.assertTrue(model['usable_for_B'], model['blocking_reasons'])
        return root, model

    def test_train_and_dispatch_without_truth_inputs(self):
        from bottle_scan.agentic_executor import _correction
        root, model = self.train()
        policy = _correction(model, servo())
        descriptors = [desc(42), desc(37)]  # measured +6 -> correction -6
        result = policy.decide('contact', descriptors, True)
        self.assertEqual(result['reason'], 'learned_contact_recovery')
        self.assertAlmostEqual(result['delta_mm'], -6, delta=.5)
        poisoned = copy.deepcopy(descriptors)
        for d in poisoned:
            d.update(delta_mm=-200, tcp=[999, 999, 999], actual_error=999)
        self.assertEqual(result, policy.decide('contact', poisoned, True))
        self.assertEqual(list(inspect.signature(policy.decide).parameters), ['status', 'descriptors', 'bilateral'])
        self.assertTrue(policy.decide('contact', [desc(60), desc(55)], True)['accepted'])
        self.assertIn('not optimal', model['label_definition'])

    def test_train_validation_groups_disjoint_and_reproducible(self):
        root, first = self.train()
        second = train_contact_policy([root, root], policy_settings())
        self.assertEqual(first['model_id'], second['model_id'])
        split = first['split']
        self.assertFalse(set(split['training_groups']) & set(split['validation_groups']))
        self.assertEqual(split['method'], 'held_out_trial_group')
        model_file = self.root / 'saved.json'
        train_contact_policy([root], policy_settings(), out=model_file)
        self.assertEqual(json.loads(model_file.read_text()), first)

    def test_holdout_cannot_change_scaling_weights_or_direction(self):
        A = self.make('A')
        B = self.make('other')
        first = train_contact_policy([A, B], policy_settings())
        self.assertTrue(first['usable_for_B'])
        held = first['split']['validation_groups'][0]
        self.mutate(Path(held), lambda row: row['probe_initial']['descriptors'][0].update(centroid_px=[10000., 40.]))
        second = train_contact_policy([A, B], policy_settings())
        self.assertEqual(first['scaling'], second['scaling'])
        self.assertEqual(first['networks'], second['networks'])
        self.assertEqual(first['fallback_fit'], second['fallback_fit'])
        self.assertFalse(second['usable_for_B'])
        self.assertEqual(first['split']['method'], 'held_out_session')

    def test_frozen_holdout_stays_out_of_all_incremental_training(self):
        seed, initial = self.train()
        frozen = sorted({initial['source_trials'][i]['group'] for i in initial['split']['validation_indices']})
        cfg = policy_settings(frozen_validation_groups=frozen)
        first_batch = self.make('first_new')
        first = train_contact_policy([seed, first_batch], cfg)
        self.assertTrue(first['usable_for_B'], first['blocking_reasons'])
        second_batch = self.make('second_new')
        second = train_contact_policy([seed, first_batch, second_batch], cfg)
        self.assertTrue(second['usable_for_B'], second['blocking_reasons'])
        self.assertEqual(first['training_count'], initial['training_count'] + 20)
        self.assertEqual(second['training_count'], first['training_count'] + 20)
        expected = {initial['source_trials'][i]['trial_path'] for i in initial['split']['validation_indices']}
        for model in (first, second):
            split = model['split']
            self.assertEqual(split['method'], 'frozen_trial_groups')
            self.assertEqual(split['validation_groups'], frozen)
            actual = {model['source_trials'][i]['trial_path'] for i in split['validation_indices']}
            self.assertEqual(actual, expected)
            self.assertFalse(set(split['training_groups']) & set(frozen))
        # Changing held-out features affects its reported errors only, not any
        # normalization, learned weights or explicit direction fallback.
        def change_held(row):
            group = f"{seed}|{row['bottle']}|{row['repeat']}"
            if group in frozen:
                row['probe_initial']['descriptors'][0]['centroid_px'] = [10000., 40.]
        self.mutate(seed, change_held)
        changed = train_contact_policy([seed, first_batch, second_batch], cfg)
        self.assertEqual(second['scaling'], changed['scaling'])
        self.assertEqual(second['networks'], changed['networks'])
        self.assertEqual(second['fallback_fit'], changed['fallback_fit'])
        self.assertFalse(changed['usable_for_B'])

    def test_missing_frozen_group_blocks_training_instead_of_resplitting(self):
        seed, model = self.train()
        frozen = sorted({model['source_trials'][i]['group'] for i in model['split']['validation_indices']})
        other = self.make('other')
        missing = train_contact_policy([other], policy_settings(frozen_validation_groups=frozen))
        self.assertFalse(missing['usable_for_B'])
        self.assertTrue(any('frozen validation groups missing' in reason for reason in missing['blocking_reasons']))

    def test_frozen_group_settings_reject_empty_duplicate_or_nonstring_ids(self):
        for groups in ([], ['one', 'one'], [''], [1], 'one', None):
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, 'frozen_validation_groups'):
                train_contact_policy([], policy_settings(frozen_validation_groups=groups))

    def test_simulation_hardware_and_T0_mismatch_block(self):
        A = self.make()
        cfg = policy_settings(); cfg['simulate'] = False
        model = train_contact_policy([A], cfg)
        self.assertFalse(model['usable_for_B'])
        self.assertTrue(any('simulation/real' in r for r in model['blocking_reasons']))
        other = self.make('different', patch_cfg={'force': {'target': 7}})
        self.assertFalse(train_contact_policy([A, other], policy_settings())['usable_for_B'])

    def test_planned_only_and_insufficient_samples_cannot_train(self):
        root = self.make(measured=False)
        model = train_contact_policy([root], policy_settings())
        self.assertFalse(model['usable_for_B'])
        self.assertEqual(model['excluded_counts']['unmeasured_position_not_policy_supervision'], 20)
        tiny = self.make('tiny', offsets=(0.,), reps=2)
        self.assertFalse(train_contact_policy([tiny], policy_settings())['usable_for_B'])

    def test_actual_tilted_operator_label_never_supervises_geometry(self):
        root = self.make(label='tilted')
        model = train_contact_policy([root], policy_settings())
        self.assertFalse(model['usable_for_B'])
        self.assertEqual(model['source_trials'], [])

    def test_final_validation_and_default_B_excluded(self):
        A = self.make('A')
        final = self.make('final', role='final_validation', fun=lambda x, rep: 9999)
        B = self.make('B', cond='B', fun=lambda x, rep: -1000)
        first = train_contact_policy([A], policy_settings())
        combined = train_contact_policy([A, final, B], policy_settings())
        self.assertEqual(first['networks'], combined['networks'])
        self.assertEqual(combined['source_sessions'], [str(A)])

    def test_missing_OOD_and_nonbilateral_rejected_without_action(self):
        _, model = self.train()
        policy = ContactPolicy(model, servo())
        for descriptors, bilateral, reason in (
            ([], True, 'policy_missing_features'),
            ([desc(10000), desc(10000)], True, 'policy_out_of_distribution'),
            ([desc(60), desc(55)], False, 'policy_requires_bilateral_contact'),
        ):
            result = policy.decide('contact', descriptors, bilateral)
            self.assertTrue(result['rejected']); self.assertEqual(result['delta_mm'], 0.)
            self.assertEqual(result['reason'], reason)

    def test_tampering_and_unusable_model_rejected(self):
        _, model = self.train()
        changed = copy.deepcopy(model)
        changed['networks'][0]['bias'][0] += .1
        with self.assertRaisesRegex(ValueError, 'integrity'):
            ContactPolicy(changed, servo())
        changed = copy.deepcopy(model); changed['usable_for_B'] = False
        with self.assertRaises(ValueError):
            ContactPolicy(changed, servo())

    def test_even_resealed_malformed_network_and_gate_rejected(self):
        from bottle_scan.contact_policy import _seal
        _, model = self.train()
        bad = copy.deepcopy(model)
        bad['networks'][0]['W'].pop()
        with self.assertRaisesRegex(ValueError, 'shape'):
            ContactPolicy(_seal(bad), servo())
        bad = copy.deepcopy(model)
        bad['uncertainty']['max_mm'] = 1000.
        with self.assertRaisesRegex(ValueError, 'gate'):
            ContactPolicy(_seal(bad), servo())

    def test_fallback_rules_kept_and_transverse_edge_rejected(self):
        _, model = self.train()
        policy = ContactPolicy(model, servo())
        self.assertEqual(policy.decide('no_contact', [], False)['delta_mm'], 8.)
        self.assertEqual(policy.decide('stall_no_contact', [], False)['delta_mm'], -8.)
        self.assertEqual(policy.decide('contact', [desc(42, edges={'left': True}), desc(37)], True)['delta_mm'], -6.)
        result = policy.decide('contact', [desc(42, edges={'top': True}), desc(37)], True)
        self.assertTrue(result['rejected'])

    def test_dd_pool_feature_recording_and_required_presence(self):
        from bottle_scan.e1e2_core import defaults, describe
        cfg = defaults()
        dd = np.zeros((12, 16, 3)); dd[3:9, 4:12, 0] = .2
        d = describe(dd, np.zeros_like(dd), cfg)
        self.assertEqual(np.asarray(d['dd_pool_4x4']).shape, (4, 4))
        self.assertEqual(feature_vector([d, d], dd_pool=True).shape, (52,))
        with self.assertRaises(ValueError):
            feature_vector([desc(10), desc(10)], dd_pool=True)

    def test_optional_B_requires_measured_successful_applied_transition(self):
        A = self.make('A', reps=6)
        B = self.make('B', cond='B', offsets=(6.,), reps=2)
        def transition(row):
            initial = copy.deepcopy(row['probe_initial'])
            final = copy.deepcopy(initial)
            final['descriptors'] = [desc(60), desc(55)]
            final['measured_tcp_mm'][0][3] = 0.
            row.update(probe_final=final, regrasps=1,
                       probe_regrasps=[{'probe': final}],
                       correction_history=[{'applied_delta_mm': -6.}, {'accepted': True}])
        self.mutate(B, transition)
        model = train_contact_policy([A, B], policy_settings(include_successful_B=True))
        added = [r for r in model['source_trials'] if r['label_source'].startswith('B_')]
        self.assertEqual(len(added), 2)
        self.assertTrue(all(r['target_action_mm'] == -6. for r in added))
        self.mutate(B, lambda row: row.update(dropped=True))
        bad = train_contact_policy([A, B], policy_settings(include_successful_B=True))
        self.assertFalse(any(r['label_source'].startswith('B_') for r in bad['source_trials']))


class IncrementalSplitTests(unittest.TestCase):
    def test_large_seed_is_not_entire_holdout_when_only_small_new_batch_trains(self):
        from bottle_scan.contact_policy import _split, DEFAULTS
        observations = [{'session': 'seed', 'group': 's' + str(i)} for i in range(33)]
        observations += [{'session': 'new', 'group': 'n' + str(i)} for i in range(6)]
        train, valid, info = _split(observations, DEFAULTS)
        self.assertGreaterEqual(len(train), DEFAULTS['min_train_samples'])
        self.assertGreaterEqual(len(valid), DEFAULTS['min_validation_samples'])
        self.assertFalse(set(info['training_groups']) & set(info['validation_groups']))


class JointBRecoveryTests(EvidenceFixture):
    """Successful B actions train joint recovery only with actual target evidence."""
    def make_joint_B(self):
        root = self.make('B', cond='B', offsets=(6.,), reps=2, role='development',
            patch_cfg={'recovery_skill': {'target_definition': {'id': 'target_v1'}}})
        def transition(row):
            final = copy.deepcopy(row['probe_initial'])
            final.update(timestamp_ns=2000, descriptors=[desc(60), desc(55)])
            final['measured_tcp_mm'][0][3] = 0.
            decision = {'recovery': {'action': {'axis': 'a', 'delta_mm': -6.}}}
            row.update(probe_final=final, regrasps=1,
                probe_regrasps=[{'probe': final, 'receipt': {'applied': True, 'action_mm': -6., 'completed_ns': 1500}}],
                correction_history=[{'delta_mm': -6., 'applied_delta_mm': -6., 'timestamp_ns': 1200, 'skill_decision': decision},
                                    {'accepted': True, 'terminal': True}],
                skill_ground_truth={'source': 'operator_review_of_observed_contact',
                    'target_definition_id': 'target_v1', 'independent_of_experimental_offset': True,
                    'initial_class': 'needs_recovery', 'final_target_reached': True},
                skill_handoff={'hand_back': True, 'completed': True},
                skill_observations=[{'phase': 'skill_decision', 'skill_outputs': decision,
                    'observation': {'timestamp_ns': 1000, 'descriptors': copy.deepcopy(row['probe_initial']['descriptors']),
                                    'context': {'target_definition_id': 'target_v1'}}},
                    {'phase': 'skill_decision', 'skill_outputs': {'exit': {'hand_back': True, 'completed': True}},
                     'observation': {'timestamp_ns': 2000, 'descriptors': copy.deepcopy(final['descriptors']),
                                     'context': {'target_definition_id': 'target_v1'}}}],
                skill_window_labels=[{'timestamp_ns': 2000, 'target_reached': True,
                    'source': 'operator_review_of_observed_contact', 'independent_of_experimental_offset': True,
                    'target_definition_id': 'target_v1'}])
        self.mutate(root, transition)
        return root

    def extract(self, root):
        from bottle_scan.agentic_learning import _load
        from bottle_scan.contact_policy import _B_observations, DEFAULTS
        cfg = policy_settings(); cfg['target_definition_id'] = 'target_v1'
        sessions, rows, issues = _load([root], cfg)
        self.assertFalse(issues)
        return _B_observations(rows, sessions[0]['meta'], cfg, DEFAULTS)

    def test_joint_success_adds_actual_decision_observation(self):
        root = self.make_joint_B()
        def confirmation(row):
            observation = row['skill_observations'][0]['observation']
            observation.update(timestamp_ns=1100, descriptors=[desc(43), desc(38)])
        self.mutate(root, confirmation)
        rows, excluded = self.extract(root)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['descriptors'][0]['centroid_px'][0], 43.)
        self.assertEqual(rows[0]['target_action_mm'], -6.)
        self.assertFalse(excluded)

    def test_joint_contact_success_survives_downstream_force_failure_but_legacy_does_not(self):
        root=self.make_joint_B()
        self.mutate(root,lambda row:row.update(status='force_not_reached',lifted=False,hold_complete=False))
        rows,excluded=self.extract(root)
        self.assertEqual(len(rows),2)
        self.assertFalse(excluded)
        from bottle_scan.agentic_learning import _load
        from bottle_scan.contact_policy import _B_observations,DEFAULTS
        cfg=policy_settings()
        sessions,raw,issues=_load([root],cfg)
        legacy,rejected=_B_observations(raw,sessions[0]['meta'],cfg,DEFAULTS)
        self.assertEqual(legacy,[])
        self.assertEqual(rejected['B_not_reliable_success'],2)
        for label in ('tipped','tilted','pushed','invalid'):
            self.mutate(root,lambda row:row.update(label=label))
            rows,rejected=self.extract(root)
            self.assertEqual(rows,[])
            self.assertEqual(rejected['B_not_reliable_success'],2)

    def test_lift_success_does_not_replace_independent_contact_target_labels(self):
        root = self.make_joint_B()
        original = {p:p.read_text() for p in (root/'trials').glob('*/meta.json')}
        changes = [
            lambda r: r['skill_ground_truth'].update(initial_class='already_normal'),
            lambda r: r['skill_ground_truth'].update(final_target_reached=False),
            lambda r: r['skill_ground_truth'].update(independent_of_experimental_offset=False),
            lambda r: r['skill_ground_truth'].update(target_definition_id='other'),
            lambda r: r['skill_handoff'].update(hand_back=False),
            lambda r: r['skill_handoff'].update(completed=False),
        ]
        for change in changes:
            for p, content in original.items(): p.write_text(content)
            self.mutate(root, change)
            rows, excluded = self.extract(root)
            self.assertEqual(rows, [])
            self.assertEqual(excluded['B_joint_target_or_handoff_unverified'], 2)

    def test_bad_receipt_stale_probe_wrong_axis_or_missing_decision_never_train(self):
        root = self.make_joint_B()
        original = {p:p.read_text() for p in (root/'trials').glob('*/meta.json')}
        changes = [
            lambda r: r['probe_regrasps'][0]['receipt'].update(applied=False),
            lambda r: r['probe_regrasps'][0]['receipt'].update(action_mm=-5.),
            lambda r: r['probe_regrasps'][0]['receipt'].update(completed_ns=2000),
            lambda r: r['probe_regrasps'][0]['receipt'].update(completed_ns=1100),
            lambda r: r['correction_history'][0]['skill_decision']['recovery']['action'].update(axis='l'),
            lambda r: r.update(skill_observations=r['skill_observations'][1:]),
            lambda r: r['skill_observations'][0]['observation']['context'].update(target_definition_id='other'),
        ]
        for change in changes:
            for p, content in original.items(): p.write_text(content)
            self.mutate(root, change)
            rows, excluded = self.extract(root)
            self.assertEqual(rows, [])
            self.assertEqual(excluded['B_joint_receipt_or_fresh_observation_unverified'], 2)

    def test_joint_unknown_or_independent_role_never_trains(self):
        root = self.make_joint_B()
        path = root/'session_meta.json'; meta = json.loads(path.read_text())
        for role in ('exploration', 'independent', 'final_validation'):
            meta['agentic']['role'] = role; path.write_text(json.dumps(meta))
            rows, excluded = self.extract(root)
            self.assertEqual(rows, [])
            self.assertTrue(excluded)

    def test_global_target_true_cannot_override_unknown_conflicting_or_unmatched_terminal_window(self):
        root = self.make_joint_B()
        original = {p:p.read_text() for p in (root/'trials').glob('*/meta.json')}
        changes = [
            lambda r: r['skill_window_labels'][0].update(target_reached=None),
            lambda r: r['skill_window_labels'][0].update(target_reached=False),
            lambda r: r['skill_window_labels'][0].update(timestamp_ns=1999),
            lambda r: r['skill_window_labels'][0].update(target_definition_id='other'),
            lambda r: r['skill_window_labels'][0].update(independent_of_experimental_offset=False),
            lambda r: r['skill_window_labels'].append({**r['skill_window_labels'][0], 'target_reached': False}),
            lambda r: r['skill_observations'][-1]['skill_outputs']['exit'].update(completed=False),
        ]
        for change in changes:
            for p, content in original.items(): p.write_text(content)
            self.mutate(root, change)
            rows, excluded = self.extract(root)
            self.assertEqual(rows, [])
            self.assertEqual(excluded['B_joint_target_or_handoff_unverified'], 2)


if __name__ == '__main__':
    unittest.main()

