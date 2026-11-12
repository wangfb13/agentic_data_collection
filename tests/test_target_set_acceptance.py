"""Set success requires observed exits and coverage of each disconnected region."""
from copy import deepcopy

from bottle_scan.acceptable_contact_set import match_features
from bottle_scan.learned_skill import train_learned_skill
from bottle_scan.skill_loop_feedback import assess_acceptance
from tests.test_learned_skill import LearnedSkillFixture, learned_settings
from tests.test_skill_loop_feedback import physical_report


class TargetSetAcceptanceTests(LearnedSkillFixture):
    def evidence(self):
        root = self.make_learned()
        cfg = learned_settings(); cfg['target_mode'] = 'acceptable_set'
        model = train_learned_skill([root], cfg)
        self.assertTrue(model['usable_for_B'], model['blocking_reasons'])
        report = physical_report()
        report.update(model_ids=[model['model_id']], target_definition_id=model['target_definition_id'])
        # Distinct observed feature neighborhoods, not duplicate physical anchors.
        features = list(dict.fromkeys(tuple(a['features']) for a in model['acceptable_set']['anchors']))
        for i, detail in enumerate(report['details']):
            x = list(features[i % len(features)])
            detail.update(model_id=model['model_id'], set_terminal_observation_verified=True)
            detail['terminal_exit']['achieved_state'] = {
                'contact_features': x, 'timestamp_ns': 1000+i,
                'target_definition_id': model['target_definition_id'],
                **match_features(model['acceptable_set'], x)}
        return model, report

    def check(self, model, report):
        return assess_acceptance(report, model=model, expected_model_id=model['model_id'])

    def test_all_supported_regions_have_observed_B_exits(self):
        model, report = self.evidence()
        result = self.check(model, report)
        self.assertTrue(result['passed'], result['reasons'])
        self.assertGreater(len(result['metrics']['acceptable_set']['components']), 1)

    def test_only_one_successful_region_cannot_validate_the_whole_set(self):
        model, report = self.evidence()
        state = report['details'][0]['terminal_exit']['achieved_state']
        for detail in report['details']:
            detail['terminal_exit']['achieved_state'] = deepcopy(state)
        result = self.check(model, report)
        self.assertFalse(result['passed'])
        self.assertTrue(result['metrics']['acceptable_set']['unobserved_components'])

    def test_missing_raw_match_or_wrong_state_identity_never_pass(self):
        model, original = self.evidence()
        for field, value in [('set_id', 'wrong'), ('supported', False),
                             ('anchor_ids', ['invented']), ('contact_features', [999.]*4)]:
            report = deepcopy(original)
            report['details'][0]['terminal_exit']['achieved_state'][field] = value
            with self.subTest(field=field):
                self.assertFalse(self.check(model, report)['passed'])
        report = deepcopy(original)
        report['details'][0].pop('set_terminal_observation_verified')
        self.assertFalse(self.check(model, report)['passed'])

    def test_completed_exit_without_membership_never_inherits_legacy_pass(self):
        model, report = self.evidence()
        for detail in report['details']:
            detail['terminal_exit'].pop('achieved_state')
        self.assertFalse(self.check(model, report)['passed'])
