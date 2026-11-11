"""Pure offline checks: a pilot stop requires measured recovery, not pooled scores."""
from copy import deepcopy
import unittest

from bottle_scan.e1e2_core import defaults
from bottle_scan.skill_collection import selection_plan
from bottle_scan.skill_exploration import default_exploration_config
from bottle_scan.skill_loop_feedback import (
    acceptance_criteria, assess_acceptance, propose_collection, summarize_round)


def learned_model():
    return {"schema": "learned_a_skill_v1", "model_id": "frozen-1",
            "target_definition_id": "white-target-v1", "usable_for_B": True,
            "invocation_head": {"usable": True}, "exit_head": {"usable": True},
            "recovery_model": {"usable_for_B": True}, "data_gaps": []}


def physical_report():
    """Synthetic evidence fixture; never pretends to be a collected dataset."""
    details, actions = [], []
    for i in range(12):
        need = i >= 6
        path = "/synthetic/B/trials/" + str(i)
        details.append({"trial_path": path, "model_id": "frozen-1",
            "label_source_verified": True,
            "initial_class": "needs_recovery" if need else "already_normal",
            "final_target_reached": True,
            "invocation": {"needs_recovery": need, "invoke": need, "permitted": True},
            "initial_model_decision": {"uncertainty_mm": .3, "ood_distance": .1},
            "terminal_exit": {"hand_back": True, "completed": True}})
        if need:
            actions.append({"trial_path": path, "index": 0, "axis": "a",
                "command_mm": -2., "measured_action_mm": -1.95,
                "blocked": False, "receipt_verified": True,
                "fresh_post_action_observation": True})
    return {"role": "development", "model_ids": ["frozen-1"], "session_complete": True,
        "target_definition_id": "white-target-v1", "simulated": False,
        "coverage": {"trials": 12, "recording_verified": 12}, "details": details,
        "invocation": {"n": 12, "TP": 6, "TN": 6, "unknown": 0},
        "recovery": {"n": 6, "successes": 6},
        "exit": {"n": 12, "TP": 12, "false_success": 0},
        "actions": actions, "completion_validation": {
            "TP": 12, "TN": 6, "FP": 0, "FN": 0, "unknown": 0,
            "target_count": 12, "not_target_count": 6, "label_source_verified": True},
        "cells": [{"conditions": {"a_mm": 0., "contact_height_l_mm": 0., "probe_force": 1.5},
                   "n": 6, "failures": 0, "evidence_refs": [d["trial_path"] + "/meta.json" for d in details[:6]]},
                  {"conditions": {"a_mm": 5., "contact_height_l_mm": 0., "probe_force": 1.5},
                   "n": 6, "failures": 0, "evidence_refs": [d["trial_path"] + "/meta.json" for d in details[6:]]}],
        "evidence_refs": [d["trial_path"] + "/meta.json" for d in details]}


def exploration_config():
    cfg = default_exploration_config()
    cfg["variables"]["probe_force"].update(reference=1.5, levels=[1.5])
    return cfg


class StopGateTests(unittest.TestCase):
    def assess(self, report=None, model=None, criteria=None):
        return assess_acceptance(report or physical_report(), criteria,
                                 expected_model_id="frozen-1", model=model or learned_model())

    def test_verified_recovery_evidence_can_pass_pilot_only(self):
        result = self.assess()
        self.assertTrue(result["passed"], result["reasons"])
        self.assertEqual(result["metrics"]["verified_recoveries"], 6)
        self.assertEqual(result["next_stage"], "frozen_independent_B")
        self.assertFalse(result["certificate_issued"])
        report = physical_report(); report["role"] = "independent"
        result = self.assess(report, criteria={"expected_role": "independent"})
        self.assertTrue(result["passed"])
        self.assertEqual(result["next_stage"], "complete")

    def test_all_normal_never_passes(self):
        report = physical_report()
        for detail in report["details"]:
            detail["initial_class"] = "already_normal"
            detail["invocation"].update(needs_recovery=False, invoke=False)
        report["actions"] = []
        result = self.assess(report)
        self.assertFalse(result["passed"])
        self.assertIn("insufficient needs-recovery trials", result["reasons"])

    def test_zero_executed_actions_never_passes(self):
        report = physical_report(); report["actions"] = []
        result = self.assess(report)
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["verified_recoveries"], 0)

    def test_action_receipt_freshness_axis_and_nonzero_movement_are_required(self):
        for field, bad in (("receipt_verified", False), ("fresh_post_action_observation", False),
                           ("axis", "l"), ("measured_action_mm", 0.),
                           ("measured_action_mm", float("nan"))):
            with self.subTest(field=field, bad=bad):
                report = physical_report(); report["actions"][0][field] = bad
                result = self.assess(report)
                self.assertFalse(result["passed"])
                self.assertTrue(any("action receipt" in reason for reason in result["reasons"]))

    def test_model_version_and_all_three_heads_are_required(self):
        for model_ids in (["different"], ["frozen-1", "different"], []):
            report = physical_report(); report["model_ids"] = model_ids
            self.assertFalse(self.assess(report)["passed"])
        for field in ("invocation_head", "exit_head", "recovery_model"):
            model = learned_model(); model[field] = {}
            self.assertFalse(self.assess(model=model)["passed"])
        report = physical_report(); report["details"][0]["model_id"] = "other"
        self.assertFalse(self.assess(report)["passed"])

    def test_unknown_refusals_stay_in_recall_denominator(self):
        report = physical_report()
        for detail in report["details"][6:]:
            detail["invocation"].update(needs_recovery=None, invoke=False, permitted=False)
        report["invocation"]["unknown"] = 6
        result = self.assess(report)
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["invocation_recall"], 0.)
        self.assertEqual(result["metrics"]["rejection_rate"], .5)

    def test_premature_terminal_exit_cannot_hide_behind_good_window_counts(self):
        report = physical_report(); report["details"][0]["final_target_reached"] = False
        result = self.assess(report)
        self.assertFalse(result["passed"])
        self.assertIn("premature terminal completion observed", result["reasons"])

    def test_missed_terminal_completion_cannot_hide_behind_good_window_counts(self):
        report = physical_report()
        for detail in report["details"][:6]:
            detail["terminal_exit"]["completed"] = False
        result = self.assess(report)
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["terminal_exit_recall"], .5)

    def test_window_unknowns_negative_coverage_and_provenance_required(self):
        for patch in ({"TN": 5, "unknown": 1}, {"TN": 0, "not_target_count": 0},
                      {"label_source_verified": False}, {"TP": "invalid"}):
            report = physical_report(); report["completion_validation"].update(patch)
            self.assertFalse(self.assess(report)["passed"])
        report = physical_report(); del report["completion_validation"]
        self.assertFalse(self.assess(report)["passed"])

    def test_missing_truth_incomplete_session_simulation_duplicates_and_role_fail(self):
        for key, value in (("session_complete", False), ("simulated", True),
                           ("target_definition_id", "different"), ("role", "independent")):
            report = physical_report(); report[key] = value
            self.assertFalse(self.assess(report)["passed"])
        report = physical_report(); report["details"][0]["initial_class"] = None
        self.assertFalse(self.assess(report)["passed"])
        report = physical_report(); report["details"][0] = deepcopy(report["details"][1])
        self.assertFalse(self.assess(report)["passed"])

    def test_downstream_optional_but_complete_when_required(self):
        report = physical_report()
        self.assertFalse(self.assess(report, criteria={"require_task_success": True})["passed"])
        report["task_outcomes"] = [{"trial_path": d["trial_path"], "success": True} for d in report["details"]]
        self.assertTrue(self.assess(report, criteria={"require_task_success": True})["passed"])
        for row in report["task_outcomes"]:
            row.update(success=False, status="force_not_reached")
        self.assertFalse(self.assess(report, criteria={"require_task_success": True})["passed"])
        self.assertTrue(self.assess(report)["passed"])

    def test_criteria_are_explicit_and_cannot_relax_false_success(self):
        for bad in ({"unknown": 1}, {"min_trials": 0}, {"max_false_success": 1},
                    {"min_invocation_recall": float("nan")}, {"require_task_success": "false"}):
            with self.assertRaises(ValueError): acceptance_criteria(bad)


class DiagnosisAndProposalTests(unittest.TestCase):
    def test_training_feedback_reports_actual_A_counts_and_unrun_B(self):
        model = {"blocking_reasons": ["contradictory_contact_labels: trial-7"],
                 "data_gaps": [{"component": "exit", "class": "negative", "missing": 4}],
                 "invocation_head": {"class_counts": {"positive": 5, "negative": 17}},
                 "exit_head": {"class_counts": {"positive": 21, "negative": 0}}}
        report = {"role": "development", "feedback_source": "training_data", "evidence_refs": ["model"]}
        diagnosis = summarize_round(report, model)
        self.assertEqual(diagnosis["B_evaluation"], "not_evaluated_for_current_model")
        self.assertEqual(diagnosis["learned_class_counts"]["invocation"]["positive"], 5)
        self.assertEqual(diagnosis["learned_class_counts"]["invocation"]["negative"], 17)
        self.assertEqual(diagnosis["learned_class_counts"]["exit"]["negative"], 0)
        codes = {x["code"] for x in diagnosis["findings"]}
        self.assertIn("contradictory_contact_labels", codes)
        self.assertNotIn("no_normal_examples", codes)
        self.assertNotIn("no_needs_recovery_examples", codes)
        self.assertNotIn("no_verified_recovery_actions", codes)

    def test_unknown_model_is_not_missing_operator_truth(self):
        report = physical_report()
        report["invocation"]["unknown"] = 6
        report["rejection_diagnostics"] = {"terminal_reasons": {"policy_uncertain": 6}}
        report["task_outcomes"] = [{"status": "force_not_reached"}]
        diagnosis = summarize_round(report)
        codes = {x["code"] for x in diagnosis["findings"]}
        self.assertNotIn("missing_contact_labels", codes)
        self.assertIn("model_uncertainty", codes)
        self.assertIn("downstream_force_not_reached", codes)
        self.assertEqual(diagnosis["missing_initial_labels"], 0)
        self.assertEqual(diagnosis["model_unknown_predictions"], 6)

    def test_missing_truth_and_allnormal_training_gap_are_distinct(self):
        report = physical_report()
        for detail in report["details"]:
            detail["initial_class"] = "already_normal"
        codes = {x["code"] for x in summarize_round(report)["findings"]}
        self.assertIn("no_needs_recovery_examples", codes)
        self.assertNotIn("missing_contact_labels", codes)
        report["details"][0]["initial_class"] = None
        codes = {x["code"] for x in summarize_round(report)["findings"]}
        self.assertIn("missing_contact_labels", codes)

    def test_blocked_learner_proposes_collection_with_exit_class_gap(self):
        report = physical_report(); report.pop("completion_validation")
        model = learned_model()
        model.update(usable_for_B=False, blocking_reasons=["insufficient independently labelled classes"],
                     data_gaps=[{"component": "exit", "class": "negative", "missing": 6}])
        proposal = propose_collection(report, exploration_config(), model=model)
        codes = {x["code"] for x in proposal["diagnosis"]["findings"]}
        self.assertIn("no_not_target_examples", codes)
        self.assertIn("training_blocked", codes)
        self.assertTrue(proposal["selected"])
        self.assertEqual(proposal["diagnosis"]["data_gaps"], model["data_gaps"])

    def test_a_only_budgeted_plan_compiles_with_references_and_preserves_sources(self):
        report = physical_report(); original = deepcopy(report)
        cfg = exploration_config(); original_cfg = deepcopy(cfg)
        proposal = propose_collection(report, cfg, model=learned_model())
        self.assertEqual(report, original)
        self.assertEqual(cfg, original_cfg)
        self.assertEqual(proposal["schema"], "skill_experiment_selection_v1")
        self.assertEqual(proposal["selected"][0]["condition"]["a_mm"], 0)
        self.assertLessEqual(proposal["planned_trial_attempts"], cfg["max_round_trials"])
        execution = defaults(); execution["design"]["zero_every"] = 4
        plan = selection_plan(proposal, execution, "A")
        self.assertEqual(len(plan), proposal["planned_trial_attempts"])
        for row in proposal["selected"]:
            self.assertEqual(row["condition_kind"], "A")
            self.assertEqual(row["condition"]["contact_height_l_mm"], 0)
            self.assertEqual(row["condition"]["probe_force"], 1.5)
            self.assertIn("never", row["label_warning"])
        self.assertFalse(proposal["execution_authorized"])
        self.assertFalse(proposal["stop_claim"])

    def test_refusals_affect_selection_even_when_final_operator_labels_are_target(self):
        report = physical_report()
        for detail in report["details"][6:]:
            detail["invocation"].update(needs_recovery=None, invoke=False, permitted=False)
        report["invocation"]["unknown"] = 6
        proposal = propose_collection(report, exploration_config(), remaining_trials=4)
        self.assertEqual({x["condition"]["a_mm"] for x in proposal["selected"]}, {0., 5.})
        self.assertEqual(report["cells"][1]["failures"], 0)

    def test_independent_evidence_cannot_choose_collection(self):
        for role in ("independent", "final_validation", "frozen_validation"):
            report = physical_report(); report["role"] = role
            with self.assertRaises(ValueError): propose_collection(report, exploration_config())

    def test_bounded_zero_reference_and_exhausted_budget(self):
        report = physical_report(); cfg = exploration_config()
        self.assertEqual(propose_collection(report, cfg, remaining_trials=0)["selected"], [])
        cfg["variables"]["a_mm"]["levels"].remove(0.)
        with self.assertRaises(ValueError): propose_collection(report, cfg)
        cfg = exploration_config(); cfg["variables"]["a_mm"]["range"] = [-30., 30.]
        with self.assertRaises(ValueError): propose_collection(report, cfg)
        cfg = exploration_config(); cfg["variables"]["contact_height_l_mm"]["reference"] = 1.
        with self.assertRaises(ValueError): propose_collection(report, cfg)


if __name__ == "__main__":
    unittest.main()
