"""A1 regression tests with nonzero, off-axis measured tool translations."""
import copy
import unittest

import numpy as np

from bottle_scan.config import demo_config, load_config, validate_config
from bottle_scan.design import commissioning_trials, visual_points
from bottle_scan.geometry import (axis_basis, check_pose, grasp_center_metadata,
    grasp_pose, grasp_pose_offset, grasp_transform, nominal_contact_region,
    pose_offset, rotation)


class GraspCenterTests(unittest.TestCase):
    def setUp(self):
        self.cfg = demo_config()
        self.G = np.eye(4)
        self.G[:3, 3] = [113.4, -7.8, 19.2]
        self.cfg["tool"]["grasp_center_in_tcp"] = self.G.tolist()
        self.T0 = np.asarray(self.cfg["T0"])

    def test_all_translations_are_numerically_identical_to_old_tcp_formula(self):
        rng = np.random.default_rng(42)
        for _ in range(40):
            T0 = self.T0.copy()
            T0[:3, :3] = rotation(rng.normal(size=3), rng.uniform(-170, 170))
            delta = rng.uniform(-90, 90, size=3)
            for axis in (None, "a", "c", "l"):
                np.testing.assert_array_equal(
                    grasp_pose_offset(T0, self.cfg, delta, axis, 0),
                    pose_offset(T0, delta, axis, 0, self.cfg["axes"]))

    def test_commissioning_three_tcp_targets_and_paths_are_unchanged(self):
        old_cfg = demo_config()
        old = commissioning_trials(old_cfg, self.T0)
        new = commissioning_trials(self.cfg, self.T0)
        self.assertEqual(len(new), 3)
        for before, after in zip(old, new):
            self.assertEqual(before["kind"], after["kind"])
            for field in ("T_target", "T_pre", "T_check", "T_lift"):
                np.testing.assert_array_equal(before[field], after[field])
            self.assertEqual(before["pose_checks"], after["pose_checks"])
            self.assertEqual(before["path_checks"], after["path_checks"])

    def test_arbitrary_tilt_keeps_grasp_center_fixed(self):
        rng = np.random.default_rng(19)
        for _ in range(40):
            T0 = self.T0.copy()
            T0[:3, :3] = rotation(rng.normal(size=3), rng.uniform(-170, 170))
            axis, angle = rng.normal(size=3), rng.uniform(-170, 170)
            result = grasp_pose_offset(T0, self.cfg, rotation_axis=axis, angle_deg=angle)
            np.testing.assert_allclose(grasp_pose(result, self.cfg)[:3, 3],
                grasp_pose(T0, self.cfg)[:3, 3], atol=1e-10)
            expected = pose_offset(T0 @ self.G, rotation_axis=axis,
                angle_deg=angle, axes=self.cfg["axes"]) @ np.linalg.inv(self.G)
            np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_translation_and_tilt_move_center_by_original_t0_axes(self):
        self.T0[:3, :3] = rotation([1, 2, -3], 47)
        delta = [7, -11, 13]
        result = grasp_pose_offset(self.T0, self.cfg, delta, "c", 33)
        info = grasp_center_metadata(result, self.T0, self.cfg)
        np.testing.assert_allclose(info["grasp_center_offset_base_mm"],
            self.T0[:3, :3] @ axis_basis(self.cfg["axes"]) @ delta, atol=1e-10)
        np.testing.assert_allclose(info["grasp_center_offset_t0_acl_mm"], delta, atol=1e-10)

    def test_visual_orbit_is_about_grasp_origin_and_declared_bottle_axis(self):
        for mode, bottle_axis in (("side", "l"), ("top", "a")):
            with self.subTest(mode=mode):
                cfg = copy.deepcopy(self.cfg)
                cfg["grasp_mode"] = mode
                cfg["geometry"]["bottle"]["axis"] = bottle_axis
                cfg["visual"].update(count=1, orbit_deg=[70, 70],
                    a_range_mm=[-30, -30], c_range_mm=[8, 8], l_range_mm=[12, 12],
                    tilt_deg=[17, 17])
                point = visual_points(cfg, self.T0)[0]
                target = np.asarray(point["T_target"])
                B = self.T0 @ self.G
                axis = axis_basis(cfg["axes"])[:, ("a", "c", "l").index(bottle_axis)]
                expected = B[:3, 3] + B[:3, :3] @ rotation(axis, 70) @ [-30, 8, 12]
                np.testing.assert_allclose(grasp_pose(target, cfg)[:3, 3], expected, atol=1e-10)
                np.testing.assert_allclose(grasp_pose(target, cfg) @ point["label_B_in_grasp"], B, atol=1e-10)
                np.testing.assert_allclose(target @ point["label_B_in_tcp"], self.T0, atol=1e-10)

    def test_nominal_region_uses_center_height_and_original_geometry_frame(self):
        self.G[:3, 3] = [0, 0, 42]
        self.cfg["tool"]["grasp_center_in_tcp"] = self.G.tolist()
        self.assertEqual(nominal_contact_region(self.T0, self.T0, self.cfg), "cap")
        tilted = grasp_pose_offset(self.T0, self.cfg, rotation_axis="c", angle_deg=90)
        self.assertEqual(nominal_contact_region(tilted, self.T0, self.cfg), "cap")
        self.assertEqual(nominal_contact_region(
            grasp_pose_offset(self.T0, self.cfg, [0, 0, -15]), self.T0, self.cfg), "neck")
        self.assertEqual(nominal_contact_region(
            grasp_pose_offset(self.T0, self.cfg, [0, 0, 20]), self.T0, self.cfg), "empty")
        # Geometry dimensions remain relative to TCP; adding G alone cannot move
        # the modeled palm, fingers, table, or bottle or loosen collision checks.
        self.assertEqual(check_pose(self.T0, self.cfg, self.T0),
            check_pose(self.T0, demo_config(), self.T0))

    def test_missing_unmeasured_or_rotated_G_has_no_identity_fallback(self):
        for value in (None, "TODO", [[1, 0], [0, 1]]):
            cfg = copy.deepcopy(self.cfg)
            cfg["tool"]["grasp_center_in_tcp"] = value
            with self.assertRaises(ValueError):
                grasp_transform(cfg)
            self.assertTrue(any("grasp_center_in_tcp" in e for e in validate_config(cfg)))
        cfg = copy.deepcopy(self.cfg)
        del cfg["tool"]
        with self.assertRaises(ValueError):
            grasp_pose(self.T0, cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["tool"]["grasp_center_in_tcp"][0][0] = 0
        cfg["tool"]["grasp_center_in_tcp"][0][1] = -1
        cfg["tool"]["grasp_center_in_tcp"][1][0] = 1
        cfg["tool"]["grasp_center_in_tcp"][1][1] = 0
        with self.assertRaisesRegex(ValueError, "identity"):
            grasp_transform(cfg)

    def test_schema_evidence_and_zero_settings_are_explicit(self):
        self.assertEqual(validate_config(self.cfg), [])
        generic = load_config()
        self.assertEqual(generic["schema_version"], 2)
        self.assertIn("TODO", generic["tool"]["grasp_center_in_tcp"])
        self.assertIn("TODO", generic["tool"]["grasp_center_evidence"])
        public = load_config("config.ros.yaml")
        self.assertEqual(public["schema_version"], 2)
        np.testing.assert_allclose(grasp_transform(public)[:3, 3], [0, 0, 117])
        self.assertTrue(public["tool"]["grasp_center_approximate"])
        self.assertFalse(public["tool"]["grasp_center_measured"])
        self.assertIn("Operator approved", public["tool"]["grasp_center_evidence"])
        for path in (("tool", "grasp_center_evidence"), ("zero_guard", "force_std_max_N")):
            cfg = copy.deepcopy(self.cfg)
            del cfg[path[0]][path[1]]
            self.assertTrue(any(".".join(path) in error for error in validate_config(cfg)))
        self.cfg["schema_version"] = 1
        self.assertTrue(any("schema_version" in error for error in validate_config(self.cfg)))


if __name__ == "__main__":
    unittest.main()
