"""Pure skill scheduling predicates and sealed artifact compatibility, offline."""
import copy
import json
from unittest.mock import patch

from bottle_scan.contact_policy import ContactPolicy, _seal, train_contact_policy, with_skill_conditions
from tests.test_contact_policy import policy_settings
from tests.test_agentic_learning import EvidenceFixture, desc, servo


def checked_context(active=False):
    return dict(object_supported=True, airborne=False, unsafe_recovery=False,
                frame_compatible=True, motion_bounds_ok=True, observation_current=True,
                timed_out=False, servo_exhausted=False, active=active)


def completed_probe(offset=0.):
    # Same fields returned by Runner.regulate; synthetic test evidence only.
    return {'valid': True, 'descriptors': [desc(60. - 3 * offset), desc(55. - 3 * offset)],
            'stable_start_ns': 1_000_000_000, 'timestamp_ns': 1_300_000_000,
            'window': [{'frame_ns': [1_100_000_000 + i * 20_000_000 for i in range(5)],
                        'frame_fids': list(range(20, 25))} for _ in range(2)]}


class SkillConditionsTests(EvidenceFixture):
    def policy(self, **kwargs):
        model = train_contact_policy(self.make(), policy_settings(), **kwargs)
        self.assertTrue(model['usable_for_B'], model['blocking_reasons'])
        return ContactPolicy(model, servo()), model

    def call(self, policy, kind, probe=None, **kwargs):
        args = {'status': 'contact', 'bilateral': True,
                'context': checked_context(active=kind == 'exit')}
        args.update(kwargs)
        return getattr(policy, 'should_' + kind)(completed_probe() if probe is None else probe, **args)

    def test_needs_correction_starts_and_centered_does_not(self):
        p, _ = self.policy()
        result = self.call(p, 'start', completed_probe(6.))
        self.assertTrue(result['result'], result)
        self.assertEqual(result['reason'], 'correction_required')
        self.assertAlmostEqual(result['decision']['e_mm'], 6., delta=.5)
        centered = self.call(p, 'start')
        self.assertFalse(centered['result'])
        self.assertEqual(centered['reason'], 'already_within_tolerance')

    def test_only_active_stable_centered_probe_exits(self):
        p, _ = self.policy()
        result = self.call(p, 'exit')
        self.assertTrue(result['result'], result)
        self.assertEqual(result['reason'], 'stable_within_tolerance')
        self.assertFalse(self.call(p, 'exit', completed_probe(6.))['result'])
        self.assertFalse(self.call(p, 'exit', context=checked_context(active=False))['result'])
        self.assertFalse(self.call(p, 'start', completed_probe(6.), context=checked_context(active=True))['result'])

    def test_rejected_invalid_and_contact_lost_never_succeed(self):
        p, _ = self.policy()
        invalid = completed_probe(); invalid['valid'] = False
        malformed = completed_probe(); malformed['descriptors'] = [None, None]
        ood = completed_probe(); ood['descriptors'] = [desc(10000), desc(10000)]
        nonfinite = completed_probe(); nonfinite['descriptors'][0]['centroid_px'][0] = float('nan')
        for probe in (invalid, malformed, ood, nonfinite):
            for kind in ('start', 'exit'):
                with self.subTest(kind=kind, probe=probe):
                    self.assertFalse(self.call(p, kind, probe)['result'])
        for status in ('no_contact', 'stall_no_contact', 'probe_contact_failed', 'policy_rejected', 'timeout'):
            self.assertFalse(self.call(p, 'exit', status=status)['result'])
        self.assertFalse(self.call(p, 'exit', bilateral=False)['result'])
        for reason in ('policy_uncertain', 'policy_nonfinite_prediction'):
            with patch.object(p, 'predict', return_value={'valid': False, 'reason': reason}):
                self.assertEqual(self.call(p, 'exit')['reason'], reason)

    def test_context_missing_out_of_scope_and_terminal_failures_fail_closed(self):
        p, _ = self.policy()
        for kind in ('start', 'exit'):
            good = checked_context(active=kind == 'exit')
            probe = completed_probe(6. if kind == 'start' else 0.)
            for key in good:
                with self.subTest(kind=kind, key=key):
                    bad = copy.deepcopy(good); bad[key] = not bad[key]
                    self.assertFalse(self.call(p, kind, probe, context=bad)['result'])
                    bad = copy.deepcopy(good); bad.pop(key)
                    self.assertFalse(self.call(p, kind, probe, context=bad)['result'])
            self.assertFalse(self.call(p, kind, probe, context=None)['result'])
        # A cached centered PROBE cannot override a live loss of contact.
        self.assertFalse(self.call(p, 'exit', bilateral=False)['result'])

    def test_existing_probe_evidence_required_without_invented_sensor_fields(self):
        p, _ = self.policy()
        for field in ('stable_start_ns', 'timestamp_ns', 'window'):
            bad = completed_probe(); bad.pop(field)
            self.assertFalse(self.call(p, 'exit', bad)['result'])
        mutations = [lambda x: x.update(timestamp_ns=1_100_000_000),
                     lambda x: x['window'][0]['frame_fids'].pop(),
                     lambda x: x['window'][0].update(frame_fids=[1] * 5),
                     lambda x: x['window'][1].update(frame_ns=[999] * 5),
                     lambda x: x.update(stable_start_ns=True)]
        for mutate in mutations:
            bad = completed_probe(); mutate(bad)
            self.assertFalse(self.call(p, 'exit', bad)['result'])
        # DD may arrive after the force snapshot before probe_window() returns.
        latest = completed_probe(); latest['window'][0]['frame_ns'][-1] = 1_310_000_000
        self.assertTrue(self.call(p, 'exit', latest)['result'])

    def test_fallback_still_works_but_is_not_a_learned_skill_start_or_success(self):
        p, _ = self.policy()
        edge = completed_probe(6.); edge['descriptors'][0] = desc(42., edges={'left': True})
        before = p.decide('contact', edge['descriptors'], True)
        self.assertEqual(before['controller'], 'explicit_direction_fallback')
        self.assertFalse(self.call(p, 'start', edge)['result'])
        self.assertFalse(self.call(p, 'exit', edge)['result'])
        self.assertEqual(before, p.decide('contact', edge['descriptors'], True))
        self.assertEqual(p.decide('no_contact', [], False)['delta_mm'], 8.)
        self.assertEqual(p.decide('stall_no_contact', [], False)['delta_mm'], -8.)

    def test_saved_contract_reload_and_pure_calls(self):
        path = self.root / 'new.json'
        p, model = self.policy(out=path)
        loaded = json.loads(path.read_text())
        self.assertEqual(loaded, model)
        again = ContactPolicy(loaded, servo())
        probe, context = completed_probe(), checked_context(True)
        before = copy.deepcopy((probe, context, again.fit, again.cfg, again.skill_conditions))
        self.assertTrue(self.call(again, 'exit', probe, context=context)['result'])
        self.assertEqual(before, (probe, context, again.fit, again.cfg, again.skill_conditions))
        self.assertEqual(self.call(p, 'exit'), self.call(again, 'exit'))
        self.assertEqual(again.conditions_source, 'embedded')

    def test_legacy_load_unchanged_and_export_new_copy(self):
        _, model = self.policy()
        legacy = copy.deepcopy(model); legacy.pop('skill_conditions'); _seal(legacy)
        path = self.root / 'legacy.json'; path.write_text(json.dumps(legacy))
        before = path.read_bytes()
        old = ContactPolicy(legacy, servo())
        self.assertTrue(old.decide('contact', completed_probe()['descriptors'], True)['accepted'])
        self.assertTrue(self.call(old, 'exit')['result'])
        self.assertEqual(self.call(old, 'exit')['conditions_source'], 'legacy_runtime_defaults')
        exported = with_skill_conditions(legacy, {'servo': servo()})
        self.assertNotEqual(exported['model_id'], legacy['model_id'])
        self.assertEqual(exported['conditions_parent_model_id'], legacy['model_id'])
        self.assertEqual(exported['networks'], legacy['networks'])
        self.assertNotIn('skill_conditions', legacy)
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue(self.call(ContactPolicy(exported, servo()), 'exit')['result'])

    def test_training_never_overwrites_a_historical_model(self):
        root = self.make()
        path = self.root / 'existing.json'
        first = train_contact_policy(root, policy_settings(), out=path)
        before, modified = path.read_bytes(), path.stat().st_mtime_ns
        self.assertEqual(train_contact_policy(root, policy_settings(), out=path), first)
        self.assertEqual(path.stat().st_mtime_ns, modified)
        with self.assertRaises(FileExistsError):
            train_contact_policy(root, policy_settings(), out=path,
                                 execution_config={'servo': {'tol_mm': 1.}})
        self.assertEqual(path.read_bytes(), before)

    def test_tampered_unknown_and_resealed_definitions_are_rejected(self):
        _, model = self.policy()
        changed = copy.deepcopy(model); changed['skill_conditions']['thresholds']['tol_mm'] = 3.
        with self.assertRaisesRegex(ValueError, 'integrity'):
            ContactPolicy(changed, servo())
        with self.assertRaisesRegex(ValueError, 'tol_mm'):
            ContactPolicy(_seal(changed), servo())
        for path, value in ((('version',), 'agent_python'),
                            (('exit', 'accepted'), False),
                            (('exit', 'evaluator'), 'eval(user_python)'),
                            (('thresholds', 'probe_frames'), 0)):
            changed = copy.deepcopy(model)
            target = changed['skill_conditions']
            for key in path[:-1]: target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                ContactPolicy(_seal(changed), servo())
        changed_servo = {**servo(), 'tol_mm': 3.}
        with self.assertRaisesRegex(ValueError, 'tol_mm'):
            ContactPolicy(model, changed_servo)

    def test_custom_actual_thresholds_and_unusable_artifacts_have_contract(self):
        cfg = {'servo': {'tol_mm': 1.25}, 'force': {'probe_stable_s': .35}, 'tactile': {'probe_frames': 7}}
        model = train_contact_policy(self.make(), policy_settings(), execution_config=cfg)
        self.assertEqual(model['skill_conditions']['thresholds'], {'tol_mm': 1.25, 'probe_stable_s': .35, 'probe_frames': 7})
        p = ContactPolicy(model, {**servo(), 'tol_mm': 1.25})
        self.assertFalse(self.call(p, 'exit')['result'])  # old .3 s/5-frame evidence is insufficient
        model = train_contact_policy([], policy_settings(), execution_config=cfg)
        self.assertFalse(model['usable_for_B'])
        self.assertEqual(model['skill_conditions']['thresholds']['tol_mm'], 1.25)

    def test_boundary_uses_same_inclusive_tolerance_as_decide(self):
        p, _ = self.policy()
        for e, start, exit in ((2., False, True), (-2., False, True), (2.001, True, False)):
            with patch.object(p, 'predict', return_value={'valid': True, 'action_mm': -e}):
                self.assertEqual(self.call(p, 'start')['result'], start)
                self.assertEqual(self.call(p, 'exit')['result'], exit)
