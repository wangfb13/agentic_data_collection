"""Synthetic RGB localization through existing planning and guarded SimBackend.

The visual model estimates tabletop XY only. A vertical TCP correction restores
the taught height; it is never presented as visually observed bottle height.
"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bottle_scan.backend import SimBackend
from bottle_scan.bottle_localization import validate_bottle_calibration
from bottle_scan.runtime import Runner, SafetyFault
from bottle_scan.vision_localization import localize_image
from bottle_scan.vision_motion import GuardedVisionAdapter, plan_alignment
from bottle_scan.vision_workspace import make_workspace_profile, TaughtWorkspace
from tests.test_bottle_localization import markerless_fixture, rgb_scene
from tests.test_vision_workspace import fixture as workspace_fixture


def markerless_motion_fixture(teaching_changes=None, *, cap=True):
    """Return profile, canonical calibration, TCP reference, fixed camera.

    The legacy collision anchor is at TCP Z=70; the visual bottle frame is at
    tabletop Z=0. Their explicit relationship is retained in teaching. Collision
    cylinders conservatively cover every measured green/body and cap height.
    """
    cfg, teaching = workspace_fixture()
    ref = np.asarray(teaching["reference_tcp_mm"], float)
    c, _, camera = markerless_fixture(fixed=True, cap=cap)
    camera = camera.copy()
    camera[:2, 3] += ref[:2, 3]
    c.pop("calibration_sha256")
    c["T_base_camera_mm"] = camera.tolist()
    c["fit_limits"]["xy_bounds_mm"] = [[ref[0, 3]-220., ref[0, 3]+220.],
                                       [ref[1, 3]-220., ref[1, 3]+220.]]
    c = validate_bottle_calibration(c)
    # Reserve at least the configured image-localization uncertainty in every
    # existing tool/bottle/table check. This still excludes calibration bias.
    cfg["safety"]["margin_mm"] = c["fit_limits"]["max_position_uncertainty_mm"]
    bottle_ref = np.eye(4)
    bottle_ref[:3, 3] = [ref[0, 3], ref[1, 3], c["model"]["table_z_mm"]]
    cfg["geometry"]["bottle"]["segments"] = [
        {"name": "body", "radius_mm": 17., "axial_min_mm": -70., "axial_max_mm": -10.},
        {"name": "upper", "radius_mm": 23., "axial_min_mm": -10., "axial_max_mm": 10.}]
    teaching.update(reference_bottle_pose_mm=bottle_ref.tolist(),
                    geometry_reference_tcp_mm=ref.tolist(),
                    reference_calibration_sha256=c["calibration_sha256"])
    if teaching_changes:
        teaching.update(teaching_changes)
    profile = make_workspace_profile(cfg, teaching, simulated=True)
    return profile, c, ref, camera


def observation(calibration, camera, xy, stamp):
    """Run actual RGB segmentation and metric geometry, not a fabricated pose."""
    rgb, _ = rgb_scene(calibration, camera, xy)
    result = localize_image(rgb, calibration, T_base_tcp_mm=np.eye(4),
        image_timestamp_ns=int(stamp), pose_timestamp_ns=int(stamp), now_ns=int(stamp))
    if not result["valid"]:
        raise AssertionError("synthetic RGB localization failed: " + str(result))
    return result


class MarkerlessPlanningTests(unittest.TestCase):
    def setUp(self):
        self.profile, self.c, self.ref, self.camera = markerless_motion_fixture()
        self.now = 2_000_000_000

    def obs(self, delta=(0., 0.)):
        return observation(self.c, self.camera, self.ref[:2, 3]+delta, self.now)

    def plan(self, obs, *, profile=None, current=None, **kwargs):
        return plan_alignment(self.profile if profile is None else profile,
            self.c, obs, self.ref if current is None else current,
            now_ns=kwargs.pop("now_ns", self.now), **kwargs)

    def test_rgb_positions_drive_both_signs_of_both_tabletop_axes(self):
        for delta in ([3., 0.], [-3., 0.], [0., 3.], [0., -3.], [-2., 2.]):
            with self.subTest(delta=delta):
                obs = self.obs(delta)
                plan = self.plan(obs)
                self.assertTrue(plan["valid"], plan)
                target = np.asarray(plan["target_tcp_mm"])
                np.testing.assert_allclose(target[:2, 3]-self.ref[:2, 3], delta, atol=.9)
                self.assertEqual(target[2, 3], self.ref[2, 3])
                self.assertEqual(obs["observed_dofs"], ["x", "y"])
                self.assertFalse(obs["orientation_observed"])
                self.assertFalse(plan["contact_allowed"])

    def test_three_dimensional_tcp_correction_uses_taught_z(self):
        current = self.ref.copy()
        current[:3, 3] += [-3., 2., 5.]
        obs = self.obs((-5., 4.))
        plan = self.plan(obs, current=current)
        self.assertTrue(plan["valid"], plan)
        target = np.asarray(plan["target_tcp_mm"])
        np.testing.assert_allclose(target[:3, 3]-current[:3, 3], [-2., 2., -5.], atol=.9)
        self.assertEqual(target[2, 3], self.profile["teaching"]["reference_tcp_mm"][2][3])
        self.assertEqual(obs["T_base_bottle_mm"][2][3], self.c["model"]["table_z_mm"])
        np.testing.assert_array_equal(target[:3, :3], current[:3, :3])

    def test_large_valid_goal_requires_reobservation_after_first_step(self):
        plan = self.plan(self.obs((60., 0.)))
        self.assertTrue(plan["valid"], plan)
        self.assertEqual(len(plan["preview_waypoints_mm"]), 3)
        first = np.asarray(plan["target_tcp_mm"])
        self.assertLessEqual(np.linalg.norm(first[:3, 3]-self.ref[:3, 3]), 25.)
        np.testing.assert_allclose(np.asarray(plan["goal_tcp_mm"])[:2, 3],
                                   self.ref[:2, 3]+[60., 0.], atol=.9)
        self.assertFalse(np.allclose(plan["target_tcp_mm"], plan["goal_tcp_mm"]))

    def test_complete_path_rejects_collision_between_clear_endpoints(self):
        profile = make_workspace_profile(self.profile["geometry_config"],
            {**self.profile["teaching"], "max_total_translation_mm": 200.}, simulated=True)
        start = self.ref.copy()
        start[1, 3] -= 100.
        obs = self.obs()
        ws = TaughtWorkspace(profile)
        bottle = np.asarray(obs["T_base_bottle_mm"])
        goal = bottle @ np.linalg.inv(np.asarray(profile["teaching"]["reference_bottle_pose_mm"])) @ self.ref
        self.assertEqual(ws.check_segment(start, start, bottle), [])
        self.assertEqual(ws.check_segment(goal, goal, bottle), [])
        plan = self.plan(obs, profile=profile, current=start)
        self.assertFalse(plan["valid"], plan)
        self.assertTrue(any("bottle" in reason for reason in plan["reasons"]), plan)
        self.assertIsNone(plan["target_tcp_mm"])

    def test_full_tool_workspace_violation_is_rejected_without_clipping(self):
        profile = make_workspace_profile(self.profile["geometry_config"],
            {**self.profile["teaching"], "max_total_translation_mm": 300.}, simulated=True)
        plan = self.plan(self.obs((0., 170.)), profile=profile)
        self.assertFalse(plan["valid"], plan)
        self.assertTrue(any("workspace" in reason for reason in plan["reasons"]), plan)
        self.assertIsNone(plan["target_tcp_mm"])
        self.assertAlmostEqual(plan["goal_tcp_mm"][1][3], 170., delta=1.2)

    def test_image_pose_age_and_synchronization_are_rechecked(self):
        obs = self.obs()
        for updates, now, reason in (
            ({}, self.now+1_000_000_000, "stale_visual_observation"),
            ({}, self.now-1, "stale_visual_observation"),
            ({"pose_timestamp_ns": self.now-1_000_000_000}, self.now, "stale_visual_pose"),
            ({"pose_timestamp_ns": self.now-100_000_000}, self.now, "image_pose_timestamp_skew")):
            with self.subTest(updates=updates, now=now):
                plan = self.plan({**obs, **updates}, now_ns=now)
                self.assertFalse(plan["valid"], plan)
                self.assertIn(reason, plan["reasons"])

    def test_target_appearance_calibration_and_pose_semantics_are_bound(self):
        obs = self.obs()
        patches = ({"target_id": "different_bottle"}, {"appearance_sha256": "0"*64},
                   {"calibration_sha256": "0"*64}, {"camera_name": "left_wrist"},
                   {"source": "aruco_detection"}, {"observed_dofs": ["x", "y", "z"]},
                   {"orientation_observed": True}, {"simulated": False}, {"valid": False})
        for changes in patches:
            with self.subTest(changes=changes):
                plan = self.plan({**obs, **changes})
                self.assertFalse(plan["valid"], plan)
                self.assertIsNone(plan["target_tcp_mm"])

    def test_changed_calibration_requires_new_taught_reference(self):
        c = deepcopy(self.c)
        c.pop("calibration_sha256")
        c["fit_limits"]["pixel_uncertainty_px"] = .6
        c = validate_bottle_calibration(c)
        obs = observation(c, self.camera, self.ref[:2, 3], self.now)
        plan = plan_alignment(self.profile, c, obs, self.ref, now_ns=self.now)
        self.assertFalse(plan["valid"], plan)
        self.assertIn("taught_reference_calibration_mismatch", plan["reasons"])

    def test_invalid_or_excessive_declared_uncertainty_cannot_authorize_motion(self):
        obs = self.obs()
        for value in (None, -1., float("nan"), float("inf"), 10.01, True):
            with self.subTest(value=value):
                plan = self.plan({**obs, "position_uncertainty_mm": value})
                self.assertFalse(plan["valid"], plan)
                self.assertIsNone(plan["target_tcp_mm"])
        missing = {k: v for k, v in obs.items() if k != "position_uncertainty_mm"}
        self.assertFalse(self.plan(missing)["valid"])

    def test_old_marker_frame_reference_cannot_be_reused_for_axis_bottom(self):
        old = make_workspace_profile(self.profile["geometry_config"],
            {**self.profile["teaching"], "reference_bottle_pose_mm": self.ref.tolist()}, simulated=True)
        plan = self.plan(self.obs(), profile=old)
        self.assertFalse(plan["valid"], plan)
        self.assertIn("markerless_table_or_orientation_convention_mismatch", plan["reasons"])

    def test_collision_geometry_must_cover_calibrated_bottle_and_cap(self):
        obs = self.obs()
        expected = {"anchor": "collision_bottle_axis_position_mismatch",
                    "radius": "collision_envelope_does_not_cover_body_profile_mm",
                    "cap": "collision_envelope_does_not_cover_cap_profile_mm",
                    "table": "collision_table_below_visual_table",
                    "margin": "collision_margin_below_visual_uncertainty"}
        for mismatch in expected:
            cfg, teaching = deepcopy(self.profile["geometry_config"]), deepcopy(self.profile["teaching"])
            if mismatch == "anchor":
                teaching["geometry_reference_tcp_mm"][0][3] += 8.
            elif mismatch == "radius":
                cfg["geometry"]["bottle"]["segments"][0]["radius_mm"] = 16.
            elif mismatch == "cap":
                cfg["geometry"]["bottle"]["segments"] = cfg["geometry"]["bottle"]["segments"][:1]
            elif mismatch == "table":
                cfg["safety"]["table_z_mm"] = -1.
                teaching["table_z_mm"] = -1.
            else:
                cfg["safety"]["margin_mm"] = 9.
            with self.subTest(mismatch=mismatch):
                changed = make_workspace_profile(cfg, teaching, simulated=True)
                plan = self.plan(obs, profile=changed)
                self.assertFalse(plan["valid"], plan)
                self.assertIn(expected[mismatch], plan["reasons"])
                self.assertIsNone(plan["target_tcp_mm"])


class MarkerlessRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.profile, self.c, self.ref, self.camera = markerless_motion_fixture()
        self.cfg = TaughtWorkspace(self.profile).config
        self.backend = SimBackend(self.cfg, self.ref)
        self.session = SimpleNamespace(cfg=self.cfg, T0=self.ref.copy(), meta={}, path=Path(self.temp.name))
        self.runner = Runner(self.session, self.backend, single_step=False, unattended=True)
        self.adapter = GuardedVisionAdapter(self.runner, self.profile, self.c)

    def obs(self, delta=(0., 0.), *, stamp=None):
        return observation(self.c, self.camera, self.ref[:2, 3]+delta,
                           int(self.runner.now()) if stamp is None else stamp)

    def assert_no_motion(self):
        self.assertFalse(any(name == "move_linear" for name, _ in self.backend.calls), self.backend.calls)

    def test_rgb_to_guarded_three_axis_tcp_recovery_restores_anchor(self):
        saved = self.session.T0.copy()
        self.backend.pose[:3, 3] += [-3., 2., 5.]
        self.backend.command_pose = self.backend.pose.copy()
        self.backend.step(0.)
        with patch.object(self.runner, "move", wraps=self.runner.move) as moved:
            receipt = self.adapter.move_step(self.obs((-5., 4.)), tolerance_mm=.2)
        self.assertTrue(receipt["applied"])
        moved.assert_called_once()
        self.assertTrue(moved.call_args.kwargs["noncontact"])
        self.assertFalse(moved.call_args.kwargs["allow_contact"])
        np.testing.assert_allclose(receipt["applied_translation_mm"], [-2., 2., -5.], atol=.9)
        np.testing.assert_allclose(self.backend.pose[:3, 3], self.ref[:3, 3]+[-5., 4., 0.], atol=.9)
        np.testing.assert_array_equal(self.session.T0, saved)
        self.assertFalse(any(name in ("close", "force", "open") for name, _ in self.backend.calls))
        self.assertEqual(self.runner.stage, "VISUAL_MOVE")
        self.assertFalse(self.runner.airborne)

    def test_aligned_rgb_does_not_issue_motion(self):
        receipt = self.adapter.move_step(self.obs(), tolerance_mm=1.5)
        self.assertFalse(receipt["applied"])
        self.assertTrue(receipt["visual_alignment_only"])
        self.assert_no_motion()

    def test_closed_contact_or_carried_object_rejects_before_motion(self):
        obs = self.obs((-2., 0.))
        for condition in ("closed", "contact", "airborne", "grasp_pose"):
            with self.subTest(condition=condition):
                adapter = GuardedVisionAdapter(self.runner, self.profile, self.c)
                self.backend.step(0.)
                self.runner.airborne = False
                self.runner.grasp_pose = None
                if condition == "closed":
                    self.backend.snapshot_cache["width_mm"] = 80.
                elif condition == "contact":
                    self.backend.snapshot_cache["normal_N"] = np.array([.5, 0.])
                elif condition == "airborne":
                    self.runner.airborne = True
                else:
                    self.runner.grasp_pose = self.ref.copy()
                with self.assertRaises((ValueError, SafetyFault)):
                    adapter.move_step(obs)
                self.assert_no_motion()

    def test_invalid_image_time_and_target_hash_never_reach_backend(self):
        obs = self.obs((-2., 0.))
        for changes in ({"timestamp_ns": obs["timestamp_ns"]-1_000_000_000},
                        {"timestamp_ns": obs["timestamp_ns"]+1},
                        {"calibration_sha256": "0"*64}, {"appearance_sha256": "0"*64},
                        {"target_id": "another_bottle"}):
            with self.subTest(changes=changes):
                adapter = GuardedVisionAdapter(self.runner, self.profile, self.c)
                with self.assertRaises(ValueError):
                    adapter.move_step({**obs, **changes})
                self.assert_no_motion()

    def test_replay_is_blocked_and_fresh_rgb_permits_next_step(self):
        obs = self.obs((-2., 0.))
        first = self.adapter.move_step(obs)
        count = sum(name == "move_linear" for name, _ in self.backend.calls)
        with self.assertRaises(ValueError):
            self.adapter.move_step(obs)
        self.assertEqual(sum(name == "move_linear" for name, _ in self.backend.calls), count)
        during = int((obs["timestamp_ns"]+first["completed_ns"])//2)
        with self.assertRaisesRegex(ValueError, "after the previous"):
            self.adapter.move_step({**obs, "timestamp_ns": during, "pose_timestamp_ns": during})
        self.backend.step(.01)
        second = self.adapter.move_step(self.obs((-2., 3.)))
        self.assertTrue(second["applied"])
        self.assertEqual(len(self.adapter.receipts), 2)
        self.assertGreater(second["plan"]["observation"]["timestamp_ns"], first["completed_ns"])
        expected = sum(np.linalg.norm(row["applied_translation_mm"]) for row in self.adapter.receipts)
        self.assertAlmostEqual(self.adapter.travelled_mm, expected, places=7)

    def test_runtime_fault_stops_motion_and_restores_geometry_reference(self):
        before = self.session.T0.copy()
        with patch.object(self.runner, "move", side_effect=SafetyFault("synthetic stop")):
            with self.assertRaisesRegex(SafetyFault, "synthetic stop"):
                self.adapter.move_step(self.obs((-2., 2.)))
        np.testing.assert_array_equal(self.session.T0, before)
        self.assertIn("stop_arm", [name for name, _ in self.backend.calls])
        self.assertTrue(self.adapter.faulted)


if __name__ == "__main__":
    unittest.main()
