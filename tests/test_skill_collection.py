"""Complete skill integration with original stages, using fake I/O only."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bottle_scan.agentic_executor import BatchRunner, execute_batch
from bottle_scan.recovery_skill import RecoverySkill
from bottle_scan.skill_collection import SkillBatchRunner, selection_plan, main
from tests.test_agentic_executor import FakeHardware, cfg, plan, ask_default


def skill_model(center=13.5):
    return {"schema": "agentic_fit_v1", "usable_for_B": True, "model_id": "fake-model-for-tests",
            "fingers": [{"coordinate": "u", "direction_sign": -1,
                         "knots_delta_mm": [-10., 0., 10.],
                         "knots_centroid_px": [center+10, center, center-10]} for _ in range(2)]}


def skill_cfg(mode="development"):
    config = cfg()
    config["force"]["probe_stable_s"] = .2
    config["recovery_skill"] = {"mode": mode, "context": {}, "runtime": {}, "certificate": None}
    return config


class CollectionTests(unittest.TestCase):
    def run_batch(self, config, model=None, cond="B", **kwargs):
        root = Path(self.temp.name)
        hw = FakeHardware(config)
        result = execute_batch(root, plan(cond=cond, offsets=(0.,)), config,
                               model, {}, hardware_factory=lambda _: hw,
                               ask=ask_default, say=lambda _: None, **kwargs)
        row = json.loads((root/"trials/00001/meta.json").read_text())
        return result, row, hw

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_development_normal_contact_confirms_then_original_lift(self):
        result, row, hw = self.run_batch(skill_cfg(), skill_model())
        self.assertTrue(result["complete"])
        self.assertTrue(row["skill_handoff"]["completed"])
        self.assertEqual(row["skill_handoff"]["status"], "success")
        self.assertIn("LIFT", hw.events)
        self.assertTrue(row["probe_confirmations"])
        self.assertEqual(row["regrasps"], 0)
        self.assertTrue(row["correction_history"][-1]["accepted"])
        self.assertTrue(row["skill_observations"])
        self.assertEqual(row["skill_cost"]["B_development_attempts"], 1)

    def test_uncertified_deployment_does_not_lift_or_correct(self):
        result, row, hw = self.run_batch(skill_cfg(mode="deployment"), skill_model())
        self.assertTrue(result["complete"])
        self.assertFalse(row["skill_handoff"]["completed"])
        self.assertEqual(row["status"], "skill_incomplete")
        self.assertNotIn("LIFT", hw.events)
        self.assertNotIn("SQUEEZE", hw.events)
        self.assertNotIn("REGRASP_CORRECT", hw.events)

    def test_A_collects_observations_and_independent_labels_without_recovery(self):
        result, row, hw = self.run_batch(skill_cfg(), cond="A")
        self.assertTrue(result["complete"])
        self.assertEqual(row["skill_observations"][0]["phase"], "first_touch")
        self.assertIn("skill_ground_truth", row)
        self.assertIsNone(row["skill_ground_truth"]["initial_class"])
        self.assertEqual(row["skill_cost"]["A_exploration_attempts"], 1)
        self.assertNotIn("REGRASP_CORRECT", hw.events)

    def test_recovery_actions_are_reobserved_and_exhaustion_cannot_lift(self):
        # Fake DD does not change with movement, deliberately prevents recovery.
        result, row, hw = self.run_batch(skill_cfg(), skill_model(center=19.5))
        self.assertTrue(result["complete"])
        self.assertEqual(row["regrasps"], 3)
        self.assertEqual(len(row["probe_regrasps"]), 3)
        self.assertEqual(row["skill_handoff"]["status"], "incomplete")
        self.assertEqual(row["skill_handoff"]["reason"], "recovery_budget_exhausted")
        self.assertNotIn("LIFT", hw.events)
        for entry in row["probe_regrasps"]:
            self.assertLess(entry["receipt"]["completed_ns"], entry["probe"]["timestamp_ns"])
        self.assertEqual(len([e for e in row["skill_runtime_events"] if e["event"] == "execution"]), 3)

    def test_unstable_target_observation_loop_is_bounded(self):
        config = skill_cfg()
        config["force"]["probe_stable_s"] = .001
        result, row, hw = self.run_batch(config, skill_model())
        self.assertEqual(row["skill_handoff"]["reason"], "observation_budget_exhausted")
        self.assertFalse(row["skill_handoff"]["completed"])
        self.assertLessEqual(len(row["probe_confirmations"]), 11)
        self.assertNotIn("LIFT", hw.events)

    def test_failed_first_touch_explicitly_hands_back_incomplete(self):
        original = SkillBatchRunner.touch
        from bottle_scan.agentic_executor import BatchRunner
        inherited = BatchRunner.touch
        def failed_trial_only(runner, prefix=""):
            if runner.current:
                return "probe_contact_failed", None
            return inherited(runner, prefix)
        with patch.object(BatchRunner, "touch", failed_trial_only):
            result, row, hw = self.run_batch(skill_cfg(), skill_model())
        self.assertFalse(row["skill_handoff"]["completed"])
        self.assertEqual(row["skill_handoff"]["status"], "incomplete")
        self.assertNotIn("LIFT", hw.events)
        self.assertTrue(any(e["phase"] == "failed_first_touch" for e in row["skill_observations"]))




def selected_experiment():
    return {
        "schema": "skill_experiment_selection_v1", "role": "development",
        "stage": "a_closed_loop", "status": "planned_unexecuted",
        "planned_trial_attempts": 4,
        "selected": [
            {"condition": {"a_mm": a, "contact_height_l_mm": 0, "probe_force": 1.5},
             "reps": 2, "bottle_blocks": ["light", "heavy"],
             "purpose": "reference" if a == 0 else "missing_invocation_evidence",
             "reason": "keep a budgeted reference and test observed feedback gap",
             "problem_refs": ["development/decision-0001"]}
            for a in (0., 4.)
        ],
    }


class SelectionAndProtocolTests(unittest.TestCase):
    def test_frozen_selection_includes_reference_in_exact_balanced_budget(self):
        selected = selected_experiment()
        rows = selection_plan(selected, skill_cfg(), "B")
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["delta_mm"] for r in rows], [0., 0., 4., 4.])
        self.assertEqual([r["bottle"] for r in rows], ["light", "heavy"]*2)
        self.assertTrue(all(r["cond"] == "B" for r in rows))
        self.assertEqual(rows[-1]["evidence_refs"], ["development/decision-0001"])
        for bad_count in (3, 5):
            bad = copy.deepcopy(selected)
            bad["planned_trial_attempts"] = bad_count
            with self.assertRaisesRegex(ValueError, "budget"):
                selection_plan(bad, skill_cfg(), "A")

    def test_force_height_and_missing_budgeted_reference_rejected(self):
        for key, value in (("probe_force", 2.), ("contact_height_l_mm", 1.)):
            selected = selected_experiment()
            selected["selected"][1]["condition"][key] = value
            with self.assertRaisesRegex(ValueError, "skill-extension"):
                selection_plan(selected, skill_cfg(), "B")
        selected = selected_experiment()
        selected["selected"][0]["condition"]["a_mm"] = -4.
        with self.assertRaisesRegex(ValueError, "reference"):
            selection_plan(selected, skill_cfg(), "B")

    def test_selection_cannot_reassign_feedback_to_independent_evaluation(self):
        selected = selected_experiment()
        selected["role"] = "final_validation"
        with self.assertRaisesRegex(ValueError, "development selections"):
            selection_plan(selected, skill_cfg(), "B")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"selection.json"
            path.write_text(json.dumps(selected_experiment()))
            with patch("bottle_scan.skill_collection.execute_batch") as execute:
                with self.assertRaisesRegex(ValueError, "independent evaluation"):
                    main(["plan", "--config", str(Path(__file__).resolve().parents[1]/"config.skill.bottle.yaml"),
                          "--selection", str(path), "--role", "final_validation", "--out", str(Path(td)/"out.json")])
                execute.assert_not_called()
            self.assertFalse((Path(td)/"out.json").exists())

    def test_repetition_balance_and_reference_gap_preserve_original_monitoring(self):
        for patch_value in (3, True, 0):
            selected = selected_experiment()
            selected["selected"][0]["reps"] = patch_value
            with self.assertRaisesRegex(ValueError, "balance bottle"):
                selection_plan(selected, skill_cfg(), "A")
        selected = selected_experiment()
        selected["selected"][1]["reps"] = 12
        selected["planned_trial_attempts"] = 14
        with self.assertRaisesRegex(ValueError, "monitoring interval"):
            selection_plan(selected, skill_cfg(), "B")

    def test_model_protocol_mismatch_blocks_before_hardware_factory(self):
        from unittest.mock import Mock
        config = skill_cfg()
        model = skill_model()
        model["signature"] = "different-force-tactile-protocol"
        factory = Mock(side_effect=AssertionError("hardware must not initialize"))
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                execute_batch(Path(td), plan(cond="B", offsets=(0.,)), config, model, {},
                              hardware_factory=factory, ask=ask_default, say=lambda _: None)
        factory.assert_not_called()

    def test_reference_session_mismatch_blocks_before_execution(self):
        from tests.test_agentic_executor import default_T0
        config = skill_cfg()
        model = skill_model()
        model["reference_T0"] = default_T0().tolist()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_path = root/"model.json"
            model_path.write_text(json.dumps(model))
            reference = root/"reference"
            reference.mkdir()
            different = copy.deepcopy(config)
            different["force"]["target"] = 6.
            (reference/"session_meta.json").write_text(json.dumps({
                "T0": default_T0().tolist(), "config": different}))
            with patch("bottle_scan.skill_collection.resolve_config", return_value=(config, None)):
                with patch("bottle_scan.skill_collection.execute_batch") as execute:
                    with self.assertRaisesRegex(ValueError, "incompatible sensing protocol"):
                        main(["collect", "--real", "--cond", "B", "--mode", "development",
                              "--config", str(Path(__file__).resolve().parents[1]/"config.skill.bottle.yaml"),
                              "--fit", str(model_path), "--reference-session", str(reference),
                              "--out", str(root/"new")])
                    execute.assert_not_called()

    def test_contact_policy_drift_uses_inverse_fallback_without_legacy_slopes(self):
        runner = SimpleNamespace(
            fit={"schema": "contact_policy_v1", "usable_for_B": True, "fallback_fit": skill_model()},
            cfg=skill_cfg(), current={"delta_mm": 0.},
            context={"zero_samples": [{"centroids_px": [[13.5, 8.5], [13.5, 8.5]]} for _ in range(3)]},
            save=lambda: None, read=lambda _: "", quit=False)
        probe = {"valid": True, "descriptors": [{"centroid_px": [12.5, 8.5]} for _ in range(2)]}
        BatchRunner.zero_drift(runner, probe)
        self.assertTrue(runner.current["zero_drift"]["checked"])
        self.assertEqual(runner.current["zero_drift"]["errors_equivalent_mm"], [1., 1.])
        self.assertFalse(runner.current["zero_drift"]["alarm"])


if __name__ == "__main__":
    unittest.main()



def test_extension_initial_status_survives_incomplete_recovery(tmp_path):
    from tests.test_ambiguity_collect import make_runner
    runner, hw, design = make_runner(tmp_path, tilt_deg=0.)
    runner.after_initial_probe = lambda target,status,probe: (target,'skill_incomplete',None)
    runner.trial(design['plan'][0])
    assert runner.current['initial_status'] == 'contact'
    assert runner.current['status'] == 'skill_incomplete'
    assert runner.current['probe_initial']['valid']
    assert runner.current['probe_final'] == {}
