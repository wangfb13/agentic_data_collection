"""Offline profile separation: software ROS claims cannot satisfy strict mode."""
import copy
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

from bottle_scan.backend import PUBLIC_ROS_CAPABILITIES, load_hardware
from bottle_scan.config import demo_config, load_config, require_real_ready, unresolved


class PublicProfileTests(unittest.TestCase):
    def setUp(self):
        guard = patch("bottle_scan.command_ownership.guard_all_commands", return_value={})
        guard.start()
        self.addCleanup(guard.stop)
        output = patch("bottle_scan.command_ownership.format_ownership_report", return_value="fake ROS ownership check")
        output.start()
        self.addCleanup(output.stop)

    def measured_fake_config(self):
        cfg = demo_config()
        cfg["demo"] = False
        cfg["hardware"].update(profile="public_ros", backend="public_ros", bridge_module="fake_public_bridge",
                               camera_mounts_declared=True, axis_mapping_declared=True, sensor_mapping_declared=True,
                               public_ros={**cfg["hardware"].get("public_ros", {}),
                                           "software_control_acknowledged": True, "processed_dd_accepted": True})
        return cfg

    def provider(self):
        return types.SimpleNamespace(PROFILE="public_ros",
                                     CAPABILITIES={key: True for key in PUBLIC_ROS_CAPABILITIES},
                                     create=Mock(return_value=object()), validate_settings=Mock())

    def test_public_provider_cannot_be_selected_under_strict_contract(self):
        cfg = self.measured_fake_config()
        cfg["hardware"]["profile"] = "strict"
        provider = self.provider()
        with patch("bottle_scan.backend.importlib.import_module", return_value=provider):
            with self.assertRaisesRegex(ValueError, "strict hardware contract"):
                load_hardware(cfg)
        provider.create.assert_not_called()

    def test_public_claims_and_provider_profile_are_explicit(self):
        cfg = self.measured_fake_config()
        provider = self.provider()
        with patch("bottle_scan.backend.importlib.import_module", return_value=provider):
            result = load_hardware(cfg)
        self.assertIs(result, provider.create.return_value)
        provider.validate_settings.assert_called_once_with(cfg)
        provider.create.assert_called_once_with(cfg)
        provider = self.provider()
        provider.PROFILE = "strict"
        with patch("bottle_scan.backend.importlib.import_module", return_value=provider):
            with self.assertRaisesRegex(ValueError, "matching provider"):
                load_hardware(cfg)
        provider.create.assert_not_called()

    def test_missing_public_capability_rejected_before_creation(self):
        cfg = self.measured_fake_config()
        provider = self.provider()
        provider.CAPABILITIES["software_force_width_limits"] = False
        with patch("bottle_scan.backend.importlib.import_module", return_value=provider):
            with self.assertRaisesRegex(ValueError, "software_force_width_limits"):
                load_hardware(cfg)
        provider.create.assert_not_called()

    def test_unknown_calibration_blocks_before_importing_any_provider(self):
        cfg = self.measured_fake_config()
        cfg["hardware"]["gripper_width_calibration"] = "TODO_actual_mm"
        with patch("bottle_scan.backend.importlib.import_module") as importer:
            with self.assertRaisesRegex(ValueError, "gripper_width_calibration"):
                load_hardware(cfg)
        importer.assert_not_called()
        with self.assertRaisesRegex(ValueError, "gripper_width_calibration"):
            require_real_ready(cfg)

    def test_public_acknowledgements_do_not_bypass_measurement_confirmations(self):
        cfg = self.measured_fake_config()
        # Software profile does not demand falsely asserting native guarantees.
        for key in ("stop_verified", "linear_path_verified", "cartesian_speed_verified"):
            cfg["hardware"].pop(key, None)
        require_real_ready(cfg)
        for key in ("software_control_acknowledged", "processed_dd_accepted"):
            bad = copy.deepcopy(cfg)
            bad["hardware"]["public_ros"][key] = False
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                require_real_ready(bad)
        for key in ("camera_mounts_declared", "axis_mapping_declared", "sensor_mapping_declared"):
            bad = copy.deepcopy(cfg)
            bad["hardware"][key] = False
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                require_real_ready(bad)

    def test_shipped_public_config_preserves_unknown_units_and_widths(self):
        cfg = load_config(Path(__file__).resolve().parents[1]/"config.ros.yaml")
        self.assertFalse(cfg["demo"])
        self.assertEqual(cfg["hardware"]["profile"], "public_ros")
        self.assertEqual(cfg["hardware"]["bridge_module"], "bridge_ros_public")
        self.assertEqual(set(cfg["cameras"]), {"left_wrist", "right_wrist"})
        self.assertEqual(cfg["tactile"]["dd_shape"], [96, 128, 3])
        self.assertEqual(cfg["tactile"]["dd_dtype"], "float32")
        missing = unresolved(cfg)
        # T0 was recorded on hardware; other unmeasured fields must still block.
        self.assertNotIn("T0", missing)
        self.assertEqual(len(cfg["T0"]), 4)
        self.assertIn("hardware.gripper_width_calibration", missing)
        prefix = "hardware.public_ros.sensors."
        self.assertEqual(cfg["force"]["unit"], "sdk")
        self.assertNotIn("force_scale_to_N", cfg["hardware"]["public_ros"]["sensors"]["force_conversion"])
        self.assertIn(prefix+"force_conversion.normal_axis", missing)
        self.assertIn(prefix+"force_conversion.normal_signs.left", missing)
        with patch("bottle_scan.backend.importlib.import_module") as importer:
            with self.assertRaises(ValueError):
                load_hardware(cfg)
        importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
