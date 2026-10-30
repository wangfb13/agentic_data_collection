import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bottle_scan.config import demo_config
from bottle_scan.force_control import (force_components, force_unit, target_definition,
    target_as_mean, in_target_band, summarize_force_cycle, evaluate_force_report,
    force_report_binding, require_force_report)
from bottle_scan.force_control import close_quality_report, feedback_target, validate_control_target
from bottle_scan.session import atomic_json


def force_config():
    cfg = demo_config()
    cfg["force"].update(controller="bridge_loop", target_definition="mean", unit="N",
        validation={"repetitions": 2, "hold_s": .3, "report_path": "unused.json",
                    "max_contact_peak": 3., "max_settle_s": 2., "max_steady_std": .3})
    cfg["force"]["stable_s"] = .2
    return cfg


def samples():
    return [{"t_s": i*.1, "phase": "CLOSE" if i < 2 else "SQUEEZE" if i < 8 else "HOLD",
             "normal": [0., 0.] if i < 2 else [2., 2.]} for i in range(12)]


def passing_report(cfg):
    trial = {"trial_id": "one", "status": "completed", **summarize_force_cycle(samples(), cfg)}
    return {"schema": "bottle_force_validation_v2", "simulated": False, "controller": cfg["force"]["controller"],
            "unit": force_unit(cfg), "target_definition": target_definition(cfg),
            "binding": force_report_binding(cfg), "operator_confirmation": "PASS", "passed": True,
            "trials": [copy.deepcopy(trial) for _ in range(2)]}


class ForceControlTests(unittest.TestCase):
    def test_close_estimate_uses_slower_measured_source_and_five_resolutions(self):
        cfg=force_config()
        cfg["close"].update(observed_force_fps=20.,observed_dd_fps=10.)
        report=close_quality_report(cfg)
        self.assertAlmostEqual(report["detection_delay_s"],.2)
        self.assertAlmostEqual(report["estimated_overtravel_mm"],.4)
        self.assertEqual(report["errors"],[])
        cfg["close"]["max_overtravel_mm"]=.3
        self.assertEqual(len(close_quality_report(cfg)["errors"]),1)
        cfg["close"]["width_resolution_mm"]=.2
        self.assertEqual(len(close_quality_report(cfg)["errors"]),2)
    def test_existing_feedback_and_control_limits_never_mix_sources(self):
        from tests.test_existing_force import existing_config
        cfg = existing_config()
        cfg["force"]["validation"]["evaluation_target"] = 1.2
        cfg["force"]["existing"]["evaluation_probe_target"] = .35
        cfg["force"]["existing"]["control_limits"].update(max=9., target_tolerance=.1)
        self.assertEqual(feedback_target(cfg, "SQUEEZE"), 1.2)
        self.assertEqual(feedback_target(cfg, "PROBE"), .35)
        validate_control_target(cfg, 6.)  # Above 39020 guard max, below separate 39010 command max.
        with self.assertRaises(ValueError): validate_control_target(cfg, 10.)
        validate_control_target(cfg, 0.)
        cfg["force"]["controller"] = "bridge_loop"
        with self.assertRaises(ValueError): validate_control_target(cfg, 6.)

    def test_components_keep_both_fingers_mean_and_sum(self):
        self.assertEqual(force_components([1., 3.]), {"per_finger": [1., 3.], "mean": 2., "sum": 4.})
        self.assertTrue(in_target_band([1., 3.], 2., .1, "mean"))
        self.assertFalse(in_target_band([1., 3.], 2., .1, "per_finger"))
        self.assertTrue(in_target_band([1., 3.], 4., .1, "sum"))
        self.assertEqual(target_as_mean(4., "sum"), 2.)

    def test_invalid_units_definitions_and_values_rejected(self):
        cfg = force_config()
        cfg["force"]["unit"] = "gf"
        with self.assertRaises(ValueError): force_unit(cfg)
        cfg["force"]["target_definition"] = "max"
        with self.assertRaises(ValueError): target_definition(cfg)
        for values in ([1.], [1., 2., 3.], [float("nan"), 2.]):
            with self.assertRaises(ValueError): force_components(values)

    def test_metrics_require_continuous_stability_and_actual_hold(self):
        cfg = force_config()
        result = summarize_force_cycle(samples(), cfg)
        self.assertTrue(result["settled"])
        self.assertAlmostEqual(result["time_to_target_band_s"], .2)
        self.assertEqual(result["statistics"]["left"]["contact_peak"], 2.)
        self.assertEqual(result["statistics"]["sum"]["steady_mean"], 4.)
        self.assertEqual(result["statistics"]["right"]["steady_std"], 0.)

    def test_interrupted_band_restarts_window(self):
        data = samples()
        data[3]["normal"] = [0., 0.]
        result = summarize_force_cycle(data, force_config())
        self.assertAlmostEqual(result["time_to_target_band_s"], .4)

    def test_per_finger_reports_unsettled_asymmetric_contact(self):
        cfg = force_config()
        cfg["force"]["target_definition"] = "per_finger"
        data = samples()
        for item in data[2:]: item["normal"] = [1., 3.]
        result = summarize_force_cycle(data, cfg)
        self.assertFalse(result["settled"])
        self.assertIsNone(result["time_to_target_band_s"])

    def test_no_contact_or_missing_hold_cannot_pass(self):
        cfg = force_config()
        report = passing_report(cfg)
        report["trials"][0]["contact_detected"] = False
        report["trials"][1]["hold_observations"] = 0
        self.assertEqual(len(evaluate_force_report(report, cfg)), 2)

    def test_validation_rejects_steady_wrong_target_or_contact_loss(self):
        cfg=force_config()
        report=passing_report(cfg)
        report["trials"][0]["statistics"]["mean"]["steady_mean"]=1.
        report["trials"][1]["statistics"]["left"]["steady_min"]=0.
        failures=evaluate_force_report(report,cfg)
        self.assertTrue(any("HOLD mean" in x for x in failures))
        self.assertTrue(any("HOLD contact" in x for x in failures))

    def test_unknown_acceptance_limits_are_explicit_failures(self):
        cfg = force_config()
        cfg["force"]["validation"]["max_contact_peak"] = "TODO"
        self.assertIn("Unmeasured", evaluate_force_report(passing_report(cfg), cfg)[0])

    def test_sdk_metrics_are_not_mislabeled_N(self):
        cfg = force_config()
        cfg["force"]["unit"] = "sdk"
        report = summarize_force_cycle(samples(), cfg)
        self.assertEqual(report["unit"], "sdk")
        self.assertNotIn("_N", json.dumps(report))

    def test_report_hash_excludes_T0_and_report_destination(self):
        cfg = force_config()
        original = force_report_binding(cfg)
        cfg["T0"][0][3] += 40.
        cfg["force"]["validation"]["report_path"] = "other.json"
        self.assertEqual(original, force_report_binding(cfg))
        cfg["gripper"]["close_speed_mm_s"] *= 2
        self.assertNotEqual(original, force_report_binding(cfg))

    def test_report_gate_rejects_simulated_unconfirmed_changed_or_failed_reports(self):
        cfg = force_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            report = passing_report(cfg)
            atomic_json(path, report)
            self.assertEqual(require_force_report(cfg, report_path=path)["controller"], "bridge_loop")
            for key, value in (("simulated", True), ("operator_confirmation", None), ("passed", False),
                               ("binding", {}), ("controller", "existing")):
                damaged = copy.deepcopy(report)
                damaged[key] = value
                atomic_json(path, damaged)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    require_force_report(cfg, report_path=path)


if __name__ == "__main__": unittest.main()
