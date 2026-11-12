"""Offline labelled-state learning tests. No ROS, hardware or network."""
import copy
import json
from pathlib import Path

import numpy as np

from bottle_scan.learned_skill import (
    LearnedSkillModel, diagnose_training_data, train_learned_skill, _seal,
)
from tests.test_agentic_learning import EvidenceFixture, desc
from tests.test_contact_policy import policy_settings

TARGET_ID = 'contact_target_test_v1'
SOURCE = 'operator_review_of_observed_contact'


def learned_settings():
    cfg = policy_settings()
    cfg['target_definition_id'] = TARGET_ID
    return cfg


class LearnedSkillFixture(EvidenceFixture):
    def make_learned(self, name='train', **kwargs):
        kwargs.setdefault('reps', 12)
        root = self.make(name, **kwargs)
        def annotate(row):
            first = copy.deepcopy(row['probe_initial'])
            first['context'] = {'target_definition_id': TARGET_ID}
            # Synthetic observation classes have known independently specified
            # semantics. The production learner never derives labels from offset.
            needs = first['descriptors'][0]['centroid_px'][0] < 45.
            final = {'timestamp_ns': 2000, 'descriptors': [desc(60), desc(55)],
                     'context': {'target_definition_id': TARGET_ID}}
            row['skill_observations'] = [
                {'phase': 'first_touch', 'observation': first},
                {'phase': 'completion', 'observation': final}]
            shared = {'source': SOURCE, 'target_definition_id': TARGET_ID,
                      'independent_of_experimental_offset': True}
            row['skill_ground_truth'] = {**shared,
                'initial_class': 'needs_recovery' if needs else 'already_normal'}
            row['skill_window_labels'] = [
                {**shared, 'timestamp_ns': 1000, 'target_reached': not needs},
                {**shared, 'timestamp_ns': 2000, 'target_reached': True}]
        self.mutate(root, annotate)
        return root

    def train_learned(self):
        root = self.make_learned()
        model = train_learned_skill([root], learned_settings())
        self.assertTrue(model['usable_for_B'], model['blocking_reasons'])
        return root, model


class LearnedSkillTests(LearnedSkillFixture):
    def test_all_three_components_train_and_decide_from_contact(self):
        _, model = self.train_learned()
        skill = LearnedSkillModel(model)
        normal = {'descriptors': [desc(60), desc(55)], 'context': {'target_definition_id': TARGET_ID}}
        needs = {'descriptors': [desc(42), desc(37)], 'context': {'target_definition_id': TARGET_ID}}
        self.assertEqual(model['schema'], 'learned_a_skill_v1')
        self.assertFalse(skill.invocation(normal)['positive'])
        self.assertTrue(skill.exit(normal)['positive'])
        self.assertTrue(skill.invocation(needs)['positive'])
        self.assertFalse(skill.exit(needs)['positive'])
        self.assertTrue(model['three_components_updated_together'])
        self.assertFalse(model['certified'])
        for key in ('reference_T0', 'signature', 'fingers'):
            self.assertEqual(model[key], model['recovery_model'][key])

    def test_all_normal_is_insufficient_despite_geometry_and_many_frames(self):
        root = self.make_learned()
        def all_normal(row):
            row['skill_ground_truth']['initial_class'] = 'already_normal'
            for label in row['skill_window_labels']:
                label['target_reached'] = True
            row['skill_window_labels'] *= 100
        self.mutate(root, all_normal)
        model = train_learned_skill([root], learned_settings())
        self.assertTrue(model['recovery_model']['usable_for_B'])
        self.assertFalse(model['usable_for_B'])
        self.assertEqual(model['invocation_head']['class_counts']['positive'], 0)
        self.assertEqual(model['exit_head']['class_counts']['negative'], 0)
        self.assertEqual(model['exit_head']['class_counts']['positive'], 120)
        self.assertTrue(model['data_gaps'])
        with self.assertRaisesRegex(ValueError, 'not ready'):
            LearnedSkillModel(model)

    def test_unknown_and_legacy_labels_never_become_negative(self):
        root = self.make_learned()
        def unknown(row):
            row['skill_ground_truth'] = {'initial_class': 'already_normal', 'final_target_reached': True}
            for label in row['skill_window_labels']:
                label['target_reached'] = None
        self.mutate(root, unknown)
        report = diagnose_training_data([root], learned_settings())
        self.assertEqual(report['class_counts']['invocation']['physical_trials'], 0)
        self.assertEqual(report['class_counts']['exit']['physical_trials'], 0)
        self.assertEqual(report['diagnostics']['legacy_initial_classes']['already_normal'], 60)

    def test_whole_trial_group_separation_and_train_only_scaling(self):
        root, model = self.train_learned()
        head = model['invocation_head']
        groups = head['split']['groups']
        self.assertFalse(set(groups['train']) & set(groups['calibration']))
        self.assertFalse(set(groups['train']) & set(groups['validation']))
        self.assertFalse(set(groups['calibration']) & set(groups['validation']))
        held = set(groups['validation'])
        def alter_validation(row):
            group = f"{root}|{row['bottle']}|{row['repeat']}"
            if group in held:
                for event in row['skill_observations']:
                    event['observation']['descriptors'][0]['centroid_px'][1] = 10000.
        self.mutate(root, alter_validation)
        changed = train_learned_skill([root], learned_settings())
        for name in ('invocation_head', 'exit_head'):
            self.assertEqual(model[name]['scaling'], changed[name]['scaling'])
            self.assertEqual(model[name]['weights'], changed[name]['weights'])
            self.assertEqual(model[name]['thresholds'], changed[name]['thresholds'])
        self.assertFalse(changed['usable_for_B'])

    def test_final_independent_evidence_excluded_and_B_development_allowed(self):
        A = self.make_learned('A')
        first = train_learned_skill([A], learned_settings())
        final = self.make_learned('final', role='final_validation', fun=lambda x,r: 999.)
        independent = self.make_learned('independent', role='independent', fun=lambda x,r: -999.)
        combined = train_learned_skill([A, final, independent], learned_settings())
        for name in ('invocation_head', 'exit_head'):
            self.assertEqual(first[name]['weights'], combined[name]['weights'])
            self.assertEqual(first[name]['scaling'], combined[name]['scaling'])
            self.assertEqual(first[name]['class_counts'], combined[name]['class_counts'])
        self.assertEqual(first['recovery_model'], combined['recovery_model'])
        B = self.make_learned('B', cond='B', role='development')
        developed = train_learned_skill([A, B], learned_settings())
        self.assertEqual(developed['invocation_head']['class_counts']['physical_trials'], 120)
        self.assertEqual(developed['recovery_model']['source_sessions'], [str(A)])
        self.assertTrue(developed['usable_for_B'], developed['blocking_reasons'])

    def test_exact_timestamp_labels_and_conflicts(self):
        root = self.make_learned()
        def mismatch(row):
            for label in row['skill_window_labels']:
                label['timestamp_ns'] += 1
        self.mutate(root, mismatch)
        report = diagnose_training_data([root], learned_settings())
        self.assertEqual(report['class_counts']['exit']['physical_trials'], 0)
        self.assertEqual(report['class_counts']['invocation']['physical_trials'], 60)

    def test_mixed_target_or_gripper_protocol_blocks_readiness(self):
        A = self.make_learned('A', patch_cfg={'gripper': {'min_normalized': .336}})
        B = self.make_learned('B', cond='B', role='development', patch_cfg={'gripper': {'min_normalized': .334}})
        model = train_learned_skill([A, B], learned_settings())
        self.assertFalse(model['usable_for_B'])
        self.assertTrue(any('gripper contact protocol' in s for s in model['blocking_reasons']))
        self.mutate(A, lambda row: row['skill_ground_truth'].update(target_definition_id='different'))
        report = diagnose_training_data([A], learned_settings())
        self.assertTrue(any('mixed target definitions' in s for s in report['blocking_reasons']))

    def test_initial_and_first_touch_target_contradictions_block_without_relabelling(self):
        root = self.make_learned()
        def contradict(row):
            initial = row['skill_ground_truth']['initial_class']
            row['skill_window_labels'][0]['target_reached'] = initial == 'needs_recovery'
        self.mutate(root, contradict)
        report = diagnose_training_data([root], learned_settings())
        self.assertTrue(any('contradictory_contact_labels' in reason for reason in report['blocking_reasons']))
        self.assertEqual(report['class_counts']['invocation']['positive'], 24)
        self.assertEqual(report['class_counts']['invocation']['negative'], 36)
        model = train_learned_skill([root], learned_settings())
        self.assertFalse(model['usable_for_B'])
        self.assertTrue(any('contradictory_contact_labels' in reason for reason in model['blocking_reasons']))

    def test_artifact_tampering_and_no_overwrite(self):
        root, model = self.train_learned()
        changed = copy.deepcopy(model)
        changed['invocation_head']['weights'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'integrity'):
            LearnedSkillModel(changed)
        changed = copy.deepcopy(model)
        changed['exit_head']['scaling']['scale'][0] = 0.
        with self.assertRaisesRegex(ValueError, 'numerical'):
            LearnedSkillModel(_seal(changed))
        changed = copy.deepcopy(model)
        changed['recovery_model']['networks'][0]['bias'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'recovery'):
            LearnedSkillModel(_seal(changed))
        path = self.root / 'version.json'
        train_learned_skill([root], learned_settings(), out=path)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            train_learned_skill([root], learned_settings(), out=path)
        self.assertEqual(before, path.read_bytes())
        self.assertEqual(json.loads(before)['model_id'], model['model_id'])

    def test_inference_forbidden_inputs_invariance_and_domain_refusal(self):
        _, model = self.train_learned()
        skill = LearnedSkillModel(model)
        observation = {'descriptors': [desc(42), desc(37)], 'context': {'target_definition_id': TARGET_ID}}
        poisoned = copy.deepcopy(observation)
        poisoned.update(tcp_pose=[999]*6, actual_offset_mm=999., label='already_normal', target_reached=True)
        for d in poisoned['descriptors']:
            d.update(delta_mm=-100., tcp_pose=[999]*6, ground_truth=True)
        for method in (skill.invocation, skill.exit):
            self.assertEqual(method(observation), method(poisoned))
            ood = copy.deepcopy(observation)
            ood['descriptors'][0]['centroid_px'] = [10000.,10000.]
            self.assertFalse(method(ood)['valid'])
            self.assertIn('out_of_distribution', method(ood)['reason'])
            self.assertEqual(method({'descriptors': []})['reason'], 'target_definition_mismatch')


if __name__ == '__main__':
    import unittest
    unittest.main()
