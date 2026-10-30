import copy
import unittest
import numpy as np

from bottle_scan.config import demo_config, load_config, require_real_ready, unresolved
from bottle_scan.geometry import (axis_basis, check_pose, check_segment, interface_units,
    lift_base, pose_offset, rotation, translate_local, box_cylinder_collision,
    _box, _cylinder_prism)
from bottle_scan.design import commissioning_trials, contact_trials, visual_points


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = demo_config()
        self.T0 = np.asarray(self.cfg["T0"])

    def test_right_multiply_translation_then_rotation(self):
        T0 = np.eye(4)
        T0[:3, :3] = rotation([0, 0, 1], 90)
        T0[:3, 3] = [100, 200, 300]
        result = pose_offset(T0, [10, 20, 30], "c", 45)
        np.testing.assert_allclose(result[:3, 3], [80, 210, 330], atol=1e-9)
        np.testing.assert_allclose(result[:3, :3], T0[:3, :3] @ rotation([0, 1, 0], 45), atol=1e-9)

    def test_axes_and_base_lift(self):
        axes = {"a": [0, 0, -1], "c": [0, 1, 0], "l": [1, 0, 0]}
        result = pose_offset(self.T0, [10, 2, 3], axes=axes)
        np.testing.assert_allclose(result[:3, 3] - self.T0[:3, 3], [3, 2, -10])
        lifted = lift_base(result, 50)
        np.testing.assert_allclose(lifted[:3, 3] - result[:3, 3], [0, 0, 50])
        with self.assertRaises(ValueError):
            axis_basis({"a": [1, 0, 0], "c": [1, 0, 0], "l": [0, 0, 1]})

    def test_single_interface_unit_boundary(self):
        T = pose_offset(self.T0, [1, 2, 3], "l", 25)
        converted = interface_units(T, "mm_to_m")
        np.testing.assert_allclose(converted[:3, 3], T[:3, 3] / 1000)
        np.testing.assert_array_equal(converted[:3, :3], T[:3, :3])
        np.testing.assert_allclose(interface_units(converted, "m_to_mm"), T)
        pose = np.array([100, 200, 300, .1, .2, .3, .9])
        np.testing.assert_allclose(interface_units(interface_units(pose, "mm_to_m"), "m_to_mm"), pose)
        with self.assertRaises(ValueError):
            interface_units(T, "guess")

    def test_open_collision_and_contact_only_pad(self):
        self.assertEqual(check_pose(self.T0, self.cfg, self.T0), [])
        # At width 35, a 36 mm cylinder penetrates 0.5 mm into each soft pad.
        self.assertTrue(check_pose(self.T0, self.cfg, self.T0, width_mm=35))
        self.assertEqual(check_pose(self.T0, self.cfg, self.T0, width_mm=35, allow_contact=True), [])
        self.assertTrue(check_pose(self.T0, self.cfg, self.T0, width_mm=20, allow_contact=True))
        palm_hit = pose_offset(self.T0, [45, 0, 0])
        self.assertTrue(any("palm:bottle" in r for r in check_pose(palm_hit, self.cfg, self.T0, allow_contact=True)))

    def test_workspace_table_step_and_no_clipping(self):
        outside = self.T0.copy()
        outside[0, 3] = 1000
        original = outside.copy()
        self.assertTrue(any("workspace" in r for r in check_pose(outside, self.cfg, self.T0)))
        np.testing.assert_array_equal(outside, original)
        low = self.T0.copy()
        low[2, 3] = 5
        self.assertTrue(any("table" in r for r in check_pose(low, self.cfg, self.T0)))
        self.assertIn("step_translation_exceeded", check_segment(self.T0, outside, self.cfg, self.T0))

    def test_swept_path_catches_between_endpoint_collision(self):
        # Sweep sideways through bottle with both endpoints safely outside it.
        start = pose_offset(self.T0, [0, -100, 0])
        end = pose_offset(self.T0, [0, 100, 0])
        self.cfg["safety"]["max_step_mm"] = 250
        self.cfg["safety"]["path_step_mm"] = 200  # single interval, bound remains conservative
        self.assertEqual(check_pose(start, self.cfg, self.T0), [])
        self.assertEqual(check_pose(end, self.cfg, self.T0), [])
        self.assertTrue(any("bottle" in r for r in check_segment(start, end, self.cfg, self.T0)))

    def test_circumscribed_cylinder_no_gap_between_vertices(self):
        segment = self.cfg["geometry"]["bottle"]["segments"][0]
        prism = _cylinder_prism(self.T0, self.cfg, segment)
        rng = np.random.default_rng(18)
        # Boxes centred at actual cylinder surface must always be rejected,
        # including arbitrary box rotations and arbitrary radial angles.
        for _ in range(40):
            theta = rng.uniform(-np.pi, np.pi)
            center = self.T0[:3, 3] + [18 * np.cos(theta), 18 * np.sin(theta), -10]
            basis = rotation(rng.normal(size=3), rng.uniform(-180, 180))
            self.assertTrue(box_cylinder_collision(_box(center, [1, 1, 1], basis, "probe"), prism))

    def test_config_is_explicitly_synthetic_and_real_blocked(self):
        self.assertTrue(unresolved(load_config()))
        self.assertTrue(load_config(demo=True)["demo"])
        with self.assertRaises(ValueError):
            require_real_ready(self.cfg)

    def test_commissioning_and_repeatable_bottle_blocks(self):
        first = commissioning_trials(self.cfg, self.T0)
        self.assertEqual(len(first), 3)
        self.assertTrue(first[0]["is_zero"])
        self.assertGreater(first[2]["delta_acl_mm"][2], 50)
        self.cfg["design"].update(bottle_axis_range_mm=[0, 0], other_axis_range_mm=[0, 0], closure_offsets_mm=[0], tilt_degrees=[0], zero_every=3)
        one = contact_trials(self.cfg, self.T0)
        two = contact_trials(self.cfg, self.T0)
        self.assertEqual(one, two)
        bottles = [t["bottle"] for t in one]
        self.assertEqual(bottles, sorted(bottles, key=lambda b: b == "heavy"))
        self.assertTrue(any(t["kind"] == "drift_zero" for t in one))

    def test_dry_run_checks_lift_lower_retreat_and_check(self):
        trial = commissioning_trials(self.cfg, self.T0)[0]
        for stage in ("MOVE_PRE", "APPROACH", "LIFT", "LOWER", "RETREAT", "CHECK"):
            self.assertIn(stage, trial["pose_checks"])
            self.assertIn(stage, trial["path_checks"])
        self.assertTrue(trial["requires_runtime_width_check"])
        self.cfg["motion"]["lift_height_mm"] = 400
        rejected = commissioning_trials(self.cfg, self.T0)[0]
        self.assertFalse(rejected["valid"])
        self.assertTrue(any("LIFT:" in reason and "workspace" in reason for reason in rejected["safety_reasons"]))
        self.assertTrue(any("LOWER:path:step_translation_exceeded" == reason for reason in rejected["safety_reasons"]))

    def test_visual_labels_and_count(self):
        self.cfg["visual"]["count"] = 5
        points = visual_points(self.cfg, self.T0)
        self.assertEqual(len(points), 5)
        for p in points:
            np.testing.assert_allclose(np.asarray(p["T_target"]) @ p["label_B_in_tcp"], self.T0, atol=1e-7)

    def test_attached_lift_moves_bottle_and_rejects_rotation(self):
        lifted = lift_base(self.T0, 50)
        self.assertEqual(check_segment(self.T0, lifted, self.cfg, self.T0, 35.8, True, self.T0), [])
        tilted = pose_offset(lifted, rotation_axis="c", angle_deg=1)
        self.assertIn("attached_motion_must_be_base_z_only_without_rotation",
            check_segment(self.T0, tilted, self.cfg, self.T0, 35.8, True, self.T0))
        below = lift_base(self.T0, -30)
        self.assertIn("attached_bottle:table", check_segment(self.T0, below, self.cfg, self.T0, 35.8, True, self.T0))


if __name__ == "__main__":
    unittest.main()
