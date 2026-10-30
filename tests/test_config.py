import copy
import unittest
from bottle_scan.config import (CALIBRATE_AT_SESSION_START, demo_config, validate_config,
    require_real_ready, load_config)


class ConfigValidationTests(unittest.TestCase):
    def test_demo_valid_and_template_unresolved(self):
        self.assertEqual(validate_config(demo_config()), [])
        self.assertTrue(validate_config(load_config()))

    def test_invalid_limits_are_rejected_without_clipping(self):
        changes = [
            ("motion", "free_speed_mm_s", 0), ("motion", "near_speed_mm_s", -1),
            ("gripper", "min_mm", 0), ("force", "target_N", 500),
            ("force", "probe_fraction", 1.1), ("units", "distance", "m"),
            ("safety", "path_step_mm", 0), ("tactile", "pixel_pitch_mm", [0, .2]),
            ("check", "min_score", 1.1), ("design", "repeats", 1.5),
        ]
        for section, key, value in changes:
            with self.subTest(field=section + "." + key):
                cfg = demo_config()
                cfg[section][key] = value
                original = copy.deepcopy(cfg)
                self.assertTrue(validate_config(cfg))
                self.assertEqual(cfg, original)

    def test_timeout_must_cover_hold(self):
        cfg = demo_config()
        cfg["motion"]["stage_timeouts_s"]["HOLD"] = 1
        self.assertTrue(any("HOLD" in e for e in validate_config(cfg)))

    def test_real_gate_checks_numeric_and_interface_flags(self):
        cfg = demo_config()
        cfg["demo"] = False
        for key in ("stop_verified", "linear_path_verified", "cartesian_speed_verified", "camera_mounts_declared", "axis_mapping_declared", "sensor_mapping_declared"):
            cfg["hardware"][key] = True
        cfg["force"]["target_N"] = 500
        with self.assertRaises(ValueError):
            require_real_ready(cfg)

    def test_threshold_calibration_sentinel_only_allowed_during_preparation(self):
        cfg = demo_config()
        cfg["demo"] = False
        for key in ("stop_verified", "linear_path_verified", "cartesian_speed_verified", "camera_mounts_declared", "axis_mapping_declared", "sensor_mapping_declared"):
            cfg["hardware"][key] = True
        cfg["check"]["threshold_px"] = CALIBRATE_AT_SESSION_START
        self.assertEqual(validate_config(cfg), [])
        require_real_ready(cfg, preparing=True)
        with self.assertRaisesRegex(ValueError, "calibration"):
            require_real_ready(cfg)
        cfg["check"]["threshold_px"] = {"left_wrist": 3.2, "right_wrist": 4.1}
        require_real_ready(cfg)
        cfg["check"]["threshold_px"]["left_wrist"] = 0
        with self.assertRaises(ValueError):
            require_real_ready(cfg, preparing=True)

    def test_preparation_does_not_bypass_other_unmeasured_values(self):
        cfg = demo_config()
        cfg["demo"] = False
        cfg["check"]["threshold_px"] = CALIBRATE_AT_SESSION_START
        cfg["tactile"]["dd_dtype"] = "TODO_verified_dtype"
        with self.assertRaisesRegex(ValueError, "dd_dtype"):
            require_real_ready(cfg, preparing=True)

    def test_runtime_numeric_settings_cannot_fall_back_silently(self):
        paths = ["motion.angular_speed_deg_s", "motion.pose_tolerance_mm", "motion.pose_tolerance_deg",
            "gripper.open_tolerance_mm", "recording.control_hz", "recording.stale_timeout_s", "recording.source_max_age_s"]
        for path in paths:
            section, key = path.split(".")
            with self.subTest(field=path):
                cfg = demo_config()
                del cfg[section][key]
                self.assertTrue(any(path in e for e in validate_config(cfg)))
                cfg[section][key] = 0
                self.assertTrue(any(path in e for e in validate_config(cfg)))

    def test_free_space_pose_dtype_serials_are_required(self):
        cfg = demo_config()
        cfg["preparation"]["axis_test_pose"][0][0] = 2
        cfg["tactile"]["dd_dtype"] = "object"
        cfg["hardware"]["sensor_serials"]["left"] = ""
        errors = validate_config(cfg)
        for field in ("preparation.axis_test_pose", "tactile.dd_dtype", "hardware.sensor_serials.left"):
            self.assertTrue(any(field in e for e in errors))


if __name__ == "__main__":
    unittest.main()
