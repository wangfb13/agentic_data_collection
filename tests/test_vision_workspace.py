"""Offline taught-workspace regression: no ROS imports or device access."""
import copy
import hashlib
import json
import unittest

import numpy as np

from bottle_scan.config import demo_config
from bottle_scan.geometry import pose_offset, rotation, gripper_boxes, _box_vertices
from bottle_scan.vision_workspace import (make_workspace_profile,
    validate_workspace_profile, TaughtWorkspace, teaching_from_captures)


def fixture():
    cfg = demo_config()
    ref = np.asarray(cfg["T0"], dtype=float)
    observation = ref.copy()
    observation[0, 3] -= 20
    teaching = {
        "workspace_min_mm": [100., -200., 5.],
        "workspace_max_mm": [550., 200., 350.],
        "reference_tcp_mm": ref.tolist(),
        "reference_bottle_pose_mm": ref.tolist(),
        "observation_tcp_mm": observation.tolist(),
        "max_translation_step_mm": 25.,
        "orientation_tolerance_deg": 5.,
        "max_total_translation_mm": 100.,
        "open_width_mm": cfg["gripper"]["open_mm"],
        "table_z_mm": cfg["safety"]["table_z_mm"],
        "clear_space_confirmed": True,
        "evidence": "synthetic test fixture only",
    }
    return cfg, teaching


def reseal(profile):
    result = copy.deepcopy(profile)
    result.pop("content_sha256", None)
    result["content_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


class TaughtWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.t = fixture()
        self.ref = np.asarray(self.t["reference_tcp_mm"])
        self.profile = make_workspace_profile(self.cfg, self.t, simulated=True)
        self.ws = TaughtWorkspace(self.profile)

    def test_round_trip_does_not_modify_originals(self):
        cfg, t = copy.deepcopy(self.cfg), copy.deepcopy(self.t)
        self.assertEqual(validate_workspace_profile(json.loads(json.dumps(self.profile))), self.profile)
        self.assertEqual(self.cfg, cfg)
        self.assertEqual(self.t, t)
        self.assertEqual(self.profile["geometry_config"]["safety"]["workspace_min_mm"], [50., -250., 0.])
        self.assertEqual(self.ws.config["safety"]["workspace_min_mm"], self.t["workspace_min_mm"])
        self.assertEqual(self.ws.config["safety"]["max_step_mm"], 25.)
        self.assertEqual(len(self.profile["base_config_sha256"]), 64)

    def test_allowed_open_hand_translation(self):
        target = self.ref.copy()
        target[0, 3] -= 10
        self.assertEqual(self.ws.check_segment(self.ref, target, self.ref), [])

    def test_goal_and_full_tool_envelope_must_be_inside(self):
        target = self.ref.copy()
        # TCP itself stays inside the box; tool exceeds upper y boundary.
        target[1, 3] = 170.
        reasons = self.ws.check_segment(target, target, self.ref)
        self.assertTrue(any("workspace" in r for r in reasons), reasons)

    def test_cannot_enlarge_original_limits(self):
        for patch in ({"workspace_min_mm": [49., -200., 5.]},
                      {"workspace_max_mm": [601., 200., 350.]},
                      {"max_translation_step_mm": 151.},
                      {"table_z_mm": -1.},
                      {"orientation_tolerance_deg": 46.}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                make_workspace_profile(self.cfg, {**self.t, **patch}, simulated=True)

    def test_does_not_clip_oversized_target(self):
        target = self.ref.copy()
        target[0, 3] -= 30.
        before = target.copy()
        self.assertIn("step_translation_exceeded", self.ws.check_segment(self.ref, target, self.ref))
        np.testing.assert_array_equal(target, before)

    def test_rejects_orientation_outside_reference_range(self):
        target = self.ref.copy()
        target[:3, :3] = rotation([0, 0, 1], 6.)
        self.assertIn("end:taught_orientation_exceeded", self.ws.check_segment(self.ref, target, self.ref))

    def test_current_pose_must_also_satisfy_orientation(self):
        start = self.ref.copy()
        start[:3, :3] = rotation([0, 0, 1], -6.)
        self.assertIn("start:taught_orientation_exceeded", self.ws.check_segment(start, self.ref, self.ref))

    def test_closed_hand_is_never_accepted(self):
        self.assertIn("open_hand_required", self.ws.check_segment(self.ref, self.ref, self.ref, 80.))
        with self.assertRaises(ValueError):
            make_workspace_profile(self.cfg, {**self.t, "open_width_mm": 80.}, simulated=True)

    def test_measured_opening_tolerance_is_accepted_without_changing_geometry(self):
        self.assertEqual(self.ws.check_segment(self.ref, self.ref, self.ref, 89.8), [])
        self.assertIn("open_hand_required", self.ws.check_segment(self.ref, self.ref, self.ref, 89.4))
        # A narrow clearance is safe at nominal 90mm, but actual 89.5mm
        # intersects the bottle. Accepting opening tolerance must not replace
        # measured width with the nominal value for the collision calculation.
        cfg, t = fixture()
        cfg["geometry"]["bottle"]["segments"][0]["radius_mm"] = 43.9
        cfg["geometry"]["gripper"]["palm_center_acl_mm"] = [-70., 0., 0.]
        ws = TaughtWorkspace(make_workspace_profile(cfg, t, simulated=True))
        self.assertEqual(ws.check_segment(self.ref, self.ref, self.ref, 90.), [])
        reasons = ws.check_segment(self.ref, self.ref, self.ref, 89.5)
        self.assertNotIn("open_hand_required", reasons)
        self.assertTrue(any("bottle" in reason for reason in reasons), reasons)

    def test_invalid_original_opening_tolerance_is_not_guessed(self):
        for value in (-1., True, float("nan"), None):
            cfg = copy.deepcopy(self.cfg)
            cfg["gripper"]["open_tolerance_mm"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_workspace_profile(cfg, self.t, simulated=True)

    def test_swept_collision_between_clear_endpoints(self):
        cfg, t = fixture()
        cfg["safety"]["max_step_mm"] = 250.
        t["max_translation_step_mm"] = 220.
        t["max_total_translation_mm"] = 300.
        ws = TaughtWorkspace(make_workspace_profile(cfg, t, simulated=True))
        start, end = self.ref.copy(), self.ref.copy()
        start[1, 3] -= 100.
        end[1, 3] += 100.
        # Neither endpoint hits the bottle, but the fingers sweep through it.
        reasons = ws.check_segment(start, end, self.ref)
        self.assertTrue(any("bottle" in r for r in reasons), reasons)

    def test_bottle_motion_updates_geometry_but_not_base_workspace(self):
        bottle = self.ref.copy()
        bottle[1, 3] += 40.
        np.testing.assert_allclose(self.ws.geometry_anchor(bottle), bottle)
        self.assertEqual(self.ws.config["safety"]["workspace_min_mm"], self.t["workspace_min_mm"])
        # The shifted bottle now hits a finger at the formerly valid pose.
        self.assertTrue(any("bottle" in r for r in self.ws.check_segment(self.ref, self.ref, bottle)))

    def test_object_frame_and_geometry_anchor_can_differ_with_rotation(self):
        bottle = np.eye(4)
        bottle[:3, :3] = rotation([0, 0, 1], 90.)
        bottle[:3, 3] = [310., 10., 70.]
        t = {**self.t, "reference_bottle_pose_mm": bottle.tolist(),
             "geometry_reference_tcp_mm": self.ref.tolist()}
        ws = TaughtWorkspace(make_workspace_profile(self.cfg, t, simulated=True))
        np.testing.assert_allclose(ws.geometry_anchor(bottle), self.ref, atol=1e-9)
        delta = np.eye(4)
        delta[:3, :3] = rotation([0, 0, 1], 10.)
        delta[:3, 3] = [5., -3., 2.]
        moved = delta @ bottle
        np.testing.assert_allclose(ws.geometry_anchor(moved), delta @ self.ref, atol=1e-9)

    def test_mutating_public_copies_does_not_change_checks(self):
        for obj in (self.ws.config, self.ws.profile["geometry_config"]):
            obj["safety"]["workspace_max_mm"] = [10000., 10000., 10000.]
        t = self.ws.teaching
        t["max_translation_step_mm"] = 999.
        target = self.ref.copy()
        target[0, 3] -= 30.
        self.assertIn("step_translation_exceeded", self.ws.check_segment(self.ref, target, self.ref))

    def test_hash_and_canonical_profile_checks(self):
        changed = copy.deepcopy(self.profile)
        changed["teaching"]["max_translation_step_mm"] = 26.
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_workspace_profile(changed)
        changed = copy.deepcopy(self.profile)
        changed["motion_mode"] = "contact"
        with self.assertRaises(ValueError):
            validate_workspace_profile(reseal(changed))
        changed = copy.deepcopy(self.profile)
        changed["simulated"] = False
        with self.assertRaisesRegex(ValueError, "synthetic"):
            validate_workspace_profile(reseal(changed))

    def test_missing_confirmation_and_ambiguous_flags_fail(self):
        for patch in ({"clear_space_confirmed": False}, {"clear_space_confirmed": "yes"},
                      {"clear_space_confirmed": 1}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                make_workspace_profile(self.cfg, {**self.t, **patch}, simulated=True)
        with self.assertRaises(ValueError):
            make_workspace_profile(self.cfg, self.t, simulated=False)
        with self.assertRaises(ValueError):
            make_workspace_profile(self.cfg, self.t, simulated="true")

    def test_nonfinite_and_boolean_numbers_fail(self):
        for key, value in (("max_translation_step_mm", True),
                           ("max_total_translation_mm", float("inf")),
                           ("orientation_tolerance_deg", -1.),
                           ("open_width_mm", float("nan")),
                           ("workspace_min_mm", [0., 0., float("nan")])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_workspace_profile(self.cfg, {**self.t, key: value}, simulated=True)
        invalid = self.ref.copy()
        invalid[2, 3] = float("nan")
        self.assertTrue(self.ws.check_segment(self.ref, invalid, self.ref))

    def test_invalid_reference_or_observation_not_certified(self):
        below = self.ref.copy()
        below[2, 3] = -10.
        for key in ("reference_tcp_mm", "observation_tcp_mm"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_workspace_profile(self.cfg, {**self.t, key: below.tolist()}, simulated=True)
        malformed = self.ref.copy()
        malformed[0, 0] = 2.
        with self.assertRaises(ValueError):
            make_workspace_profile(self.cfg, {**self.t, "reference_bottle_pose_mm": malformed.tolist()}, simulated=True)

    def test_teaching_captures_use_complete_margin_expanded_envelope(self):
        low = self.ref.copy()
        low[:3, 3] = [170., -120., 40.]
        high = self.ref.copy()
        high[:3, 3] = [480., 120., 280.]
        constraints = {k: v for k, v in self.t.items() if k not in (
            "workspace_min_mm", "workspace_max_mm", "reference_tcp_mm",
            "observation_tcp_mm", "reference_bottle_pose_mm")}
        captures = [
            {"role": "lower_corner", "tcp_mm": low.tolist()},
            {"role": "upper_corner", "tcp_mm": high.tolist()},
            {"role": "reference", "tcp_mm": self.ref.tolist(), "constraints": constraints},
            {"role": "observation", "tcp_mm": self.t["observation_tcp_mm"]},
            {"role": "bottle_reference", "pose_mm": self.ref.tolist()},
        ]
        t = teaching_from_captures(captures, self.cfg)
        vertices = np.concatenate([_box_vertices(b, 1.) for b in gripper_boxes(low, self.cfg, 90.)])
        np.testing.assert_allclose(t["workspace_min_mm"], vertices.min(0))
        self.assertNotEqual(t["workspace_min_mm"], low[:3, 3].tolist())
        profile = make_workspace_profile(self.cfg, t, simulated=True)
        self.assertEqual(profile["teaching"]["capture_evidence"], captures)
        with self.assertRaises(ValueError):
            teaching_from_captures(captures[:-1], self.cfg)
        with self.assertRaises(ValueError):
            teaching_from_captures(captures + [captures[0]], self.cfg)


if __name__ == "__main__":
    unittest.main()
