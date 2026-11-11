"""No robot: evidence isolation, cost accounting and bounded extension plans."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from bottle_scan.skill_exploration import (
    default_exploration_config, validate_exploration_config, diagnose_feedback,
    choose_experiments, extension_design, CostLedger, record_round,
    memory_context, persist_method_experience, comparison_protocol)
from bottle_scan.rsi_methods import MethodStore, baseline_method


def config():
    cfg = default_exploration_config()
    cfg["variables"]["probe_force"].update(reference=1.5, levels=[1., 1.5, 2.])
    return cfg


def feedback():
    return {"role": "offline_development", "evidence_refs": ["source/meta.json"],
            "invocation": {"n": 10, "FP": 2, "FN": 1, "unknown": 2},
            "recovery": {"n": 10, "p90_mm": 3.},
            "exit": {"n": 10, "false_success": 1, "proxy_only": True},
            "gaps": ["Missing physical height recovery evidence"]}


class ExplorationTests(unittest.TestCase):
    def test_model_unknown_does_not_invent_missing_human_labels(self):
        report = feedback()
        report["exit"] = {"n": 10, "unknown": 2}
        report["gaps"] = []
        report["details"] = [{"initial_class": "already_normal", "final_target_reached": True,
                              "label_source_verified": True}]
        report["rejection_diagnostics"] = {"terminal_reasons": {"policy_uncertain": 2}}
        issues = diagnose_feedback(report)
        problems = [row["problem"] for row in issues]
        self.assertNotIn("Missing independent labels for need/applicability", problems)
        self.assertIn("Invocation prediction or evidence is unknown", problems)
        self.assertIn("Recovery prediction uncertainty exceeds its frozen gate", problems)
        self.assertNotIn("No independently observed target-state completion", problems)
        self.assertIn("Completion prediction or terminal handoff is unknown", problems)
        report["details"][0]["initial_class"] = None
        problems = [row["problem"] for row in diagnose_feedback(report)]
        self.assertIn("Missing independent labels for need/applicability", problems)

    def test_three_components_and_control_choice(self):
        issues = diagnose_feedback(feedback())
        self.assertTrue({"invocation", "recovery", "exit"} <= {x["component"] for x in issues})
        self.assertTrue({"repeat_reference", "fill_gap", "control_variable"} <= {x["experiment"] for x in issues})

    def test_independent_evidence_never_guides_choice(self):
        report = feedback(); report["role"] = "independent"
        with self.assertRaises(ValueError): diagnose_feedback(report)
        with self.assertRaises(ValueError): choose_experiments(report, config())
        combined = {"development": feedback(), "independent": report}
        self.assertEqual(diagnose_feedback(combined), diagnose_feedback(feedback()))

    def test_budget_and_fixed_controls(self):
        for mode in ("fixed", "simple_feedback", "agent_evidence"):
            result = choose_experiments(feedback(), config(), mode=mode, remaining_trials=5)
            self.assertLessEqual(result["planned_trial_attempts"], 5)
            self.assertEqual(result["planned_trial_attempts"], 4)
            for row in result["selected"]:
                self.assertEqual(row["condition"]["contact_height_l_mm"], 0)
                self.assertEqual(row["condition"]["probe_force"], 1.5)
        self.assertEqual(choose_experiments(feedback(), config(), remaining_trials=0)["selected"], [])

    def test_height_then_force_then_small_combinations(self):
        cfg = config()
        height = extension_design(cfg, "height_controlled")
        self.assertTrue(all(x["condition"]["a_mm"] == 0 and x["condition"]["probe_force"] == 1.5 for x in height["rows"]))
        force = extension_design(cfg, "force_controlled")
        self.assertTrue(all(x["condition"]["a_mm"] == x["condition"]["contact_height_l_mm"] == 0 for x in force["rows"]))
        combos = extension_design(cfg, "small_combinations")
        self.assertLessEqual(sum(x["purpose"] == "small_combinations" for x in combos["rows"]), cfg["max_combinations"])

    def test_force_requires_calibration(self):
        with self.assertRaises(ValueError): extension_design(default_exploration_config(), "height_controlled")
        cfg = config(); cfg["variables"]["probe_force"]["levels"] = []
        with self.assertRaises(ValueError): extension_design(cfg, "force_controlled")

    def test_nonfinite_and_bad_units_rejected(self):
        cfg = config(); cfg["variables"]["a_mm"]["levels"] = [float("nan")]
        with self.assertRaises(ValueError): validate_exploration_config(cfg)
        cfg = config(); cfg["variables"]["a_mm"]["unit"] = ""
        with self.assertRaises(ValueError): validate_exploration_config(cfg)

    def test_cost_unknown_source_and_idempotence(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = CostLedger(root)
            event = ledger.record("old", "source_experience", trial_attempts=1, failed_attempts=1,
                                  evidence_refs=["old/meta.json"], attempt_ids=["old:1"])
            ledger.record("old", "source_experience", trial_attempts=1, failed_attempts=1,
                          evidence_refs=["old/meta.json"], attempt_ids=["old:1"])
            ledger.record("train", "offline_compute", wall_seconds=2., robot_seconds=0.,
                          reset_seconds=0., evidence_refs=["round.json"])
            summary = ledger.summary()
            self.assertEqual(summary["new_physical_attempts"], 0)
            self.assertEqual(summary["source_experience_attempts"], 1)
            self.assertEqual(summary["categories"]["source_experience"]["reset_seconds"]["unknown_events"], 1)
            self.assertEqual(summary["categories"]["offline_compute"]["robot_seconds"]["known_total"], 0.)
            with self.assertRaises(ValueError):
                ledger.record("duplicate", "A_exploration", trial_attempts=1, attempt_ids=["old:1"],
                              evidence_refs=["old/meta.json"])
            with self.assertRaises(ValueError):
                ledger.record("old", "source_experience", trial_attempts=2, evidence_refs=["old/meta.json"])
            with self.assertRaises(ValueError):
                ledger.record("replay", "A_exploration", trial_attempts=1, actual=False, evidence_refs=["old/meta.json"])

    def test_round_immutable_and_independent_result_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            kwargs = dict(feedback=feedback(), selection={}, result={"role":"development"},
                          skill_update={"status":"pending"})
            record_round(root, "r1", **kwargs)
            record_round(root, "r1", **kwargs)
            kwargs["skill_update"] = {"status":"changed"}
            with self.assertRaises(ValueError): record_round(root, "r1", **kwargs)
            kwargs["result"] = {"role":"independent"}
            with self.assertRaises(ValueError): record_round(root, "r2", **kwargs)

    def test_history_off_does_not_read_or_promote(self):
        class NoRead:
            def retrieve(self, *args, **kwargs): raise AssertionError("history disabled")
        self.assertEqual(memory_context(NoRead(), {}, history_enabled=False)["methods"], [])

    def test_real_replay_keeps_existing_method_pending(self):
        with tempfile.TemporaryDirectory() as root:
            store = MethodStore(root)
            record = {"result":{"offline_replay":True}}
            result = persist_method_experience(store, baseline_method(), task_id="bottle",
                context={"hardware_signature":"verified-config-signature", "simulated":False},
                round_record=record, simulated=False)
            self.assertEqual(store.list_methods()[0]["lifecycle"]["by_scope"][0]["status"], "pending")
            with self.assertRaises(ValueError):
                persist_method_experience(store, baseline_method(), task_id="bottle",
                    context={"hardware_signature":"verified-config-signature", "simulated":False},
                    round_record=record, simulated=False, comparison={})

    def test_fair_protocol_same_action_learner_budget(self):
        p = comparison_protocol(config(), interaction_budget=72, source_evidence_refs=["manifest"])
        self.assertEqual(len(p["modes"]), 3)
        self.assertEqual(p["interaction_budget_per_arm"], 72)
        self.assertIn("same source data", p["history_ablation"])
        self.assertEqual(p, comparison_protocol(config(), interaction_budget=72, source_evidence_refs=["manifest"]))
