"""Tests use an offline learned model and a fake guarded execution adapter."""
import copy
import json
import unittest

from bottle_scan.contact_policy import train_contact_policy
from bottle_scan.recovery_skill import RecoverySkill, load_skill_bundle
from tests.test_agentic_learning import EvidenceFixture, desc, servo
from tests.test_contact_policy import policy_settings


class RecoverySkillTests(EvidenceFixture):
    def setUp(self):
        super().setUp()
        self.model = train_contact_policy([self.make()], policy_settings(epochs=30))
        self.assertTrue(self.model["usable_for_B"], self.model.get("blocking_reasons"))

    def skill(self, *, certified=False, mode="deployment", **kwargs):
        skill = RecoverySkill(self.model, {**servo(), "max_regrasps": 3}, mode=mode, **kwargs)
        if certified:
            skill.certificate = {
                "schema": "skill_validation_certificate_v1",
                "model_id": skill.model_id, "model_sha256": skill.model_sha256,
                "config_sha256": skill.config_sha256,
                "observed_cells": [{
                    "status": "contact", "centroid_bounds_px": [
                        [[30, 80], [0, 100]], [[25, 75], [0, 100]]],
                    "context": {}, "invocation_validated": True,
                    "recovery_validated": True, "exit_validated": True,
                    "evidence_refs": ["synthetic-test-fixture-not-real-evidence"],
                }],
            }
        return skill

    def observation(self, x=0, timestamp_ns=1000000000, **kwargs):
        return {"status": "contact", "bilateral": True,
                "descriptors": [desc(60 - 3*x), desc(55 - 3*x)],
                "timestamp_ns": timestamp_ns, "first_touch_complete": True,
                "safety_ok": True, "force_stable": True, "stable_duration_s": .2,
                "context": {}, **kwargs}

    def test_three_outputs_need_and_validated_scope_are_separate(self):
        skill = self.skill()
        result = skill.evaluate(self.observation(6))
        self.assertEqual(set(("invocation", "recovery", "exit")) - set(result), set())
        self.assertTrue(result["invocation"]["needs_recovery"])
        self.assertTrue(result["invocation"]["candidate_supported"])
        self.assertFalse(result["invocation"]["applicable"])
        self.assertFalse(result["invocation"]["invoke"])
        self.assertIsNone(result["recovery"]["action"])
        self.assertEqual(skill.start(self.observation(6))["exit"]["status"], "incomplete")

    def test_rejected_policy_reason_precedes_missing_certificate(self):
        for mode in ("development", "deployment"):
            skill = self.skill(mode=mode)
            skill.policy.decide = lambda *args: {"accepted": False, "rejected": True,
                "delta_mm": 0., "reason": "policy_uncertain"}
            result = skill.start(self.observation())
            self.assertEqual(result["exit"]["reason"], "policy_uncertain")
            self.assertEqual(result["invocation"]["validation_reason"], "missing_or_mismatched_validation_certificate")
            self.assertFalse(result["invocation"]["permitted"])
            self.assertFalse(result["exit"]["completed"])
        skill = self.skill(mode="deployment")
        result = skill.start(self.observation())
        self.assertEqual(result["exit"]["reason"], "missing_or_mismatched_validation_certificate")
        self.assertFalse(result["invocation"]["permitted"])

    def test_safety_reason_precedes_rejected_policy_reason(self):
        skill = self.skill(mode="development")
        skill.policy.decide = lambda *args: {"accepted": False, "rejected": True,
            "delta_mm": 0., "reason": "policy_uncertain"}
        result = skill.start(self.observation(safety_ok=False))
        self.assertEqual(result["exit"]["reason"], "safety_state_unavailable_or_unsafe")

    def test_old_model_or_config_cannot_inherit_certificate(self):
        skill = self.skill(certified=True)
        self.assertTrue(skill.evaluate(self.observation(6))["invocation"]["applicable"])
        for key in ("model_id", "model_sha256", "config_sha256"):
            cert = copy.deepcopy(skill.certificate)
            skill.certificate[key] = "different"
            self.assertFalse(skill.evaluate(self.observation(6))["invocation"]["applicable"])
            skill.certificate = cert

    def test_no_truth_leakage(self):
        skill = self.skill(certified=True)
        original = self.observation(6)
        poisoned = copy.deepcopy(original)
        poisoned.update(delta_mm=999, measured_tcp_mm=999, actual_error=-999)
        poisoned["context"]["planned_offset_mm"] = -999
        for d in poisoned["descriptors"]:
            d.update(delta_mm=999, measured_tcp_mm=999)
        self.assertEqual(skill.evaluate(original), skill.evaluate(poisoned))

    def test_already_normal_confirmed_without_recovery_motion(self):
        skill = self.skill(certified=True)
        first = skill.start(self.observation())
        self.assertFalse(first["invocation"]["needs_recovery"])
        self.assertFalse(first["invocation"]["invoke"])
        self.assertEqual(first["exit"]["status"], "observing")
        last = skill.advance(self.observation(timestamp_ns=2000000000))
        self.assertTrue(last["exit"]["completed"])
        self.assertTrue(last["exit"]["hand_back"])
        self.assertEqual(last["exit"]["continuation"], "squeeze_lift_allowed")

    def test_first_touch_and_freshness_required(self):
        for patch in ({"first_touch_complete": False}, {"safety_ok": False},
                      {"timestamp_ns": None}, {"now_ns": 9000000000}):
            skill = self.skill(certified=True)
            self.assertEqual(skill.start(self.observation(**patch))["exit"]["status"], "incomplete")
        skill = self.skill(certified=True)
        skill.start(self.observation())
        stale = skill.advance(self.observation())
        self.assertEqual(stale["exit"]["reason"], "new_observation_required")
        self.assertFalse(stale["exit"]["completed"])

    def test_recovery_requires_receipt_then_new_window_then_confirmation(self):
        skill = self.skill(certified=True)
        first = skill.start(self.observation(6))
        self.assertAlmostEqual(first["recovery"]["action"]["delta_mm"], -6, delta=.6)
        action = first["recovery"]["action"]["delta_mm"]
        second = skill.advance(self.observation(timestamp_ns=3000000000),
                               applied_action={"applied": True, "action_mm": action,
                                               "completed_ns": 2000000000})
        self.assertEqual(second["exit"]["status"], "observing")
        last = skill.advance(self.observation(timestamp_ns=4000000000))
        self.assertEqual(last["exit"]["status"], "success")
        self.assertEqual(skill.regrasps, 1)
        with self.assertRaises(RuntimeError):
            skill.advance(self.observation(timestamp_ns=5000000000))

    def test_unapplied_or_unmeasured_actions_never_count_as_recovery(self):
        for receipt in (None, {"applied": False}, {"applied": True, "action_mm": 999, "completed_ns": 2},
                        {"applied": True, "action_mm": -6, "completed_ns": 4000000000}):
            skill = self.skill(certified=True)
            skill.start(self.observation(6))
            last = skill.advance(self.observation(timestamp_ns=3000000000), applied_action=receipt)
            self.assertEqual(last["exit"]["status"], "incomplete")
            self.assertEqual(last["exit"]["reason"], "unverified_execution_or_reobservation")

    def test_bounds_context_and_uncertainty_block_deployment(self):
        skill = self.skill(certified=True)
        skill.certificate["observed_cells"][0]["centroid_bounds_px"][0][0] = [50, 80]
        self.assertFalse(skill.evaluate(self.observation(6))["invocation"]["applicable"])
        self.assertFalse(skill.evaluate(self.observation(1000))["invocation"]["permitted"])
        skill = self.skill(config={"required_context": {"force_protocol": "fixed"}})
        result = skill.evaluate(self.observation())
        self.assertIn("context_mismatch:force_protocol", result["invocation"]["reasons"])

    def test_target_is_contact_not_geometric_estimate_only(self):
        skill = self.skill(certified=True)
        for patch in ({"force_stable": False}, {"stable_duration_s": .1}, {"bilateral": False}):
            self.assertFalse(skill.evaluate(self.observation(**patch))["exit"]["target_observed"])
        o = self.observation()
        o["descriptors"][0]["touches_edge"] = True
        self.assertFalse(skill.evaluate(o)["exit"]["target_observed"])

    def test_budget_exhaustion_is_not_success(self):
        skill = self.skill(certified=True, config={"max_regrasps": 0})
        result = skill.start(self.observation(6))
        self.assertEqual(result["exit"]["reason"], "recovery_budget_exhausted")
        self.assertFalse(result["exit"]["completed"])
        self.assertEqual(result["exit"]["continuation"], "replan_or_operator")

    def test_development_does_not_claim_certified_applicability(self):
        skill = self.skill(mode="development")
        result = skill.start(self.observation(6))
        self.assertTrue(result["invocation"]["invoke"])
        self.assertFalse(result["invocation"]["applicable"])
        self.assertFalse(result["invocation"]["verified"])

    def test_adapter_reobserves_after_each_action_and_hands_control_back(self):
        test = self
        calls = []
        class Adapter:
            def initial_probe(self):
                calls.append("first_touch")
                return test.observation(6)
            def execute(self, action):
                calls.append("execute")
                return {"applied": True, "action_mm": action["delta_mm"], "completed_ns": 2000000000}
            def observe(self):
                calls.append("observe")
                return test.observation(timestamp_ns=2000000000 + calls.count("observe")*1000000000)
            def handoff(self, result):
                calls.append("handoff:" + result["exit"]["status"])
        result = self.skill(certified=True).run(Adapter())
        self.assertTrue(result["exit"]["completed"])
        self.assertEqual(calls, ["first_touch", "execute", "observe", "observe", "handoff:success"])

    def test_unstable_contact_observation_budget_ends_incomplete(self):
        test = self
        class Adapter:
            stamp = 1000000000
            def initial_probe(self):
                return test.observation(force_stable=False)
            def observe(self):
                self.stamp += 1000000000
                return test.observation(timestamp_ns=self.stamp, force_stable=False)
            def execute(self, action):
                raise AssertionError("normal geometry must not cause motion")
            def handoff(self, result):
                self.result = result
        adapter = Adapter()
        result = self.skill(certified=True).run(adapter, max_observations=3)
        self.assertEqual(result["exit"]["reason"], "observation_budget_exhausted")
        self.assertFalse(result["exit"]["completed"])


    def test_overlapping_windows_do_not_confirm_and_external_abort_returns_failure(self):
        skill = self.skill(certified=True)
        skill.start(self.observation())
        result = skill.advance(self.observation(timestamp_ns=1000000001))
        self.assertEqual(result["exit"]["status"], "observing")
        result = skill.abort("operator_stop")
        self.assertFalse(result["exit"]["completed"])
        self.assertTrue(result["exit"]["hand_back"])
        self.assertEqual(result["exit"]["reason"], "operator_stop")

    def test_experimental_truth_cannot_become_required_runtime_context(self):
        with self.assertRaisesRegex(ValueError, "experimental truth"):
            self.skill(config={"required_context": {"planned_offset_mm": 6}})
        skill = self.skill(certified=True)
        skill.certificate["observed_cells"][0]["context"] = {"planned_offset_mm": 6}
        observation = self.observation(6, context={"planned_offset_mm": 6})
        self.assertFalse(skill.evaluate(observation)["invocation"]["applicable"])

    def test_initial_probe_failure_returns_control_as_incomplete(self):
        class Adapter:
            result = None
            def initial_probe(self):
                raise RuntimeError("sensor failed")
            def handoff(self, result):
                self.result = result
        adapter = Adapter()
        with self.assertRaisesRegex(RuntimeError, "sensor failed"):
            self.skill(certified=True).run(adapter)
        self.assertEqual(adapter.result["exit"]["status"], "incomplete")
        self.assertEqual(adapter.result["exit"]["reason"], "adapter_failure:RuntimeError")



    def test_bundle_roundtrip_default_deployment_and_tamper_rejection(self):
        skill = self.skill(mode="development")
        root = self.root / "bundle"
        manifest_path = skill.save_bundle(root)
        loaded = load_skill_bundle(manifest_path)
        self.assertEqual(loaded.mode, "deployment")
        self.assertEqual(loaded.model_sha256, skill.model_sha256)
        self.assertFalse(loaded.start(self.observation(6))["exit"]["completed"])
        with self.assertRaises(FileExistsError):
            skill.save_bundle(root)
        manifest = json.loads((root / "skill.json").read_text())
        manifest["config"]["target"]["confirmation_windows"] = 3
        (root / "skill.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            load_skill_bundle(root)

    def test_bundle_model_tampering_is_rejected(self):
        skill = self.skill()
        root = self.root / "bundle"
        skill.save_bundle(root)
        model = json.loads((root / "model.json").read_text())
        model["model_id"] = "another-model"
        (root / "model.json").write_text(json.dumps(model))
        with self.assertRaisesRegex(ValueError, "model hash mismatch"):
            load_skill_bundle(root)

    def test_l_axis_requires_separately_measured_model_and_blocks_a_fallback(self):
        from bottle_scan.contact_policy import _seal
        with self.assertRaisesRegex(ValueError, "physical action axis"):
            self.skill(config={"action_axis": "l"})
        model = copy.deepcopy(self.model)
        model.update(recovery_action_axis="l",
                     recovery_label_source="measured_tcp_projection_l",
                     recovery_controlled_axes=["a"])
        # Synthetic metadata ONLY for this unit test; no real l evidence claimed.
        model = _seal(model)
        skill = RecoverySkill(model, {**servo(), "max_regrasps": 3},
                              {"action_axis": "l"}, mode="development")
        result = skill.start(self.observation(6))
        self.assertEqual(result["recovery"]["action"]["axis"], "l")
        self.assertFalse(result["invocation"]["verified"])
        result = skill.evaluate(self.observation(status="no_contact", descriptors=[], bilateral=False))
        self.assertIsNone(result["recovery"]["action"])
        self.assertEqual(result["recovery"]["decision"]["reason"], "l_directional_fallback_not_validated")
        with self.assertRaisesRegex(ValueError, "physical action axis"):
            RecoverySkill(model, {**servo(), "max_regrasps": 3})


if __name__ == "__main__":
    unittest.main()
