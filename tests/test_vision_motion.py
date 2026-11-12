"""Offline visual planning and real Runner integration with the synthetic backend."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bottle_scan.backend import SimBackend
from bottle_scan.geometry import rotation
from bottle_scan.runtime import Runner, SafetyFault
from bottle_scan.vision_localization import localize_marker_corners
from bottle_scan.vision_motion import plan_alignment, GuardedVisionAdapter
from bottle_scan.vision_workspace import make_workspace_profile, TaughtWorkspace
from tests.test_vision_workspace import fixture
from tests.test_vision_localization import calibration, marker_pose, projected_corners


def motion_fixture(teaching_changes=None, cfg_changes=None):
    cfg, teaching = fixture()
    if cfg_changes:
        cfg_changes(cfg)
    if teaching_changes:
        teaching.update(teaching_changes)
    profile = make_workspace_profile(cfg, teaching, simulated=True)
    ref = np.asarray(teaching["reference_tcp_mm"], float)
    c = calibration()
    c["camera_mount"] = "fixed"
    c["image_size_wh"] = [1920, 1080]
    c["camera_matrix"][0][2] = 960.
    c["camera_matrix"][1][2] = 540.
    c.pop("mount_tcp_side")
    c.pop("T_tcp_camera_mm")
    # Measured optical frame is independent of robot axes and nonidentity.
    c["T_base_camera_mm"] = (ref @ np.linalg.inv(
        marker_pose() @ np.asarray(c["T_marker_bottle_mm"]))).tolist()
    return profile, c, ref


def observation(c, bottle, stamp):
    marker_camera = (np.linalg.inv(np.asarray(c["T_base_camera_mm"])) @
                     bottle @ np.linalg.inv(np.asarray(c["T_marker_bottle_mm"])))
    result = localize_marker_corners(projected_corners(c, marker_camera), c,
        T_base_tcp_mm=np.eye(4), image_timestamp_ns=int(stamp),
        pose_timestamp_ns=int(stamp), now_ns=int(stamp))
    if not result["valid"]:
        raise AssertionError("synthetic localization failed: " + str(result))
    return result


class VisionPlanningTests(unittest.TestCase):
    def setUp(self):
        self.profile, self.c, self.ref = motion_fixture()
        self.now = 1_000_000_000

    def plan(self, bottle, **kwargs):
        obs = observation(self.c, bottle, self.now)
        return plan_alignment(self.profile, self.c, obs,
                              kwargs.pop("current", self.ref), now_ns=self.now, **kwargs)

    def test_three_axis_directions_come_from_calibrated_visual_geometry(self):
        for delta in ([4., 0, 0], [0, -4., 0], [0, 0, 4.], [-3., 2., -2.]):
            with self.subTest(delta=delta):
                bottle = self.ref.copy()
                bottle[:3, 3] += delta
                plan = self.plan(bottle)
                self.assertTrue(plan["valid"], plan)
                np.testing.assert_allclose(np.asarray(plan["target_tcp_mm"])[:3, 3]-self.ref[:3, 3], delta, atol=1e-6)
                self.assertFalse(plan["contact_allowed"])

    def test_oversized_but_in_bounds_motion_is_segmented_for_reobservation(self):
        bottle = self.ref.copy()
        bottle[0, 3] += 60.
        plan = self.plan(bottle)
        self.assertTrue(plan["valid"], plan)
        self.assertEqual(len(plan["preview_waypoints_mm"]), 3)
        np.testing.assert_allclose(np.asarray(plan["target_tcp_mm"])[:3, 3]-self.ref[:3, 3], [20, 0, 0], atol=1e-6)
        np.testing.assert_allclose(np.asarray(plan["goal_tcp_mm"])[:3, 3], bottle[:3, 3], atol=1e-6)

    def test_out_of_workspace_goal_is_rejected_not_clipped(self):
        bottle = self.ref.copy()
        bottle[1, 3] += 170.
        # Isolate boundary from total-budget rejection.
        profile = make_workspace_profile(self.profile["geometry_config"],
            {**self.profile["teaching"], "max_total_translation_mm": 300.}, simulated=True)
        obs = observation(self.c, bottle, self.now)
        plan = plan_alignment(profile, self.c, obs, self.ref, now_ns=self.now)
        self.assertFalse(plan["valid"])
        self.assertIsNone(plan["target_tcp_mm"])
        self.assertTrue(any("workspace" in reason for reason in plan["reasons"]), plan)
        np.testing.assert_allclose(np.asarray(plan["goal_tcp_mm"])[:3, 3], bottle[:3, 3], atol=1e-6)

    def test_full_path_is_checked_not_just_first_step(self):
        # Both start and goal are clear; fingers cross the bottle on the way.
        start = self.ref.copy()
        start[1, 3] -= 100.
        profile = make_workspace_profile(self.profile["geometry_config"],
            {**self.profile["teaching"], "max_total_translation_mm": 200.}, simulated=True)
        plan = plan_alignment(profile, self.c, observation(self.c, self.ref, self.now),
                              start, now_ns=self.now)
        self.assertFalse(plan["valid"], plan)
        self.assertTrue(any("bottle" in reason for reason in plan["reasons"]), plan)
        self.assertIsNone(plan["target_tcp_mm"])

    def test_stale_and_future_images_are_rejected(self):
        obs = observation(self.c, self.ref, self.now)
        for now in (self.now-1, self.now+1_000_000_000):
            with self.subTest(now=now):
                plan = plan_alignment(self.profile, self.c, obs, self.ref, now_ns=now)
                self.assertFalse(plan["valid"])
                self.assertIn("stale_visual_observation", plan["reasons"])

    def test_stale_or_unsynchronized_pose_is_rechecked_at_planning(self):
        obs = observation(self.c, self.ref, self.now)
        for stamp, reason in ((self.now-1_000_000_000, "stale_visual_pose"),
                              (self.now-100_000_000, "image_pose_timestamp_skew"),
                              (self.now+1, "stale_visual_pose")):
            with self.subTest(stamp=stamp):
                plan = plan_alignment(self.profile, self.c, {**obs, "pose_timestamp_ns": stamp},
                                      self.ref, now_ns=self.now)
                self.assertFalse(plan["valid"], plan)
                self.assertIn(reason, plan["reasons"])

    def test_missing_invalid_and_domain_mismatched_observations_rejected(self):
        obs = observation(self.c, self.ref, self.now)
        for patch_value in ({"valid": False}, {"simulated": False}, {"simulated": "true"},
                            {"calibration_sha256": "wrong"}, {"camera_name": "left_wrist"}):
            with self.subTest(patch_value=patch_value):
                plan = plan_alignment(self.profile, self.c, {**obs, **patch_value}, self.ref, now_ns=self.now)
                self.assertFalse(plan["valid"], plan)

    def test_total_budget_includes_already_travelled_distance(self):
        bottle = self.ref.copy()
        bottle[0, 3] -= 10.
        plan = self.plan(bottle, travelled_mm=95.)
        self.assertFalse(plan["valid"])
        self.assertIn("total_translation_budget_exceeded", plan["reasons"])

    def test_changed_object_orientation_is_not_silently_commanded(self):
        bottle = self.ref.copy()
        bottle[:3, :3] = rotation([0, 0, 1], 8.)
        plan = self.plan(bottle)
        self.assertFalse(plan["valid"])
        self.assertIn("goal_requires_uncommissioned_rotation", plan["reasons"])

    def test_already_aligned_has_no_translation(self):
        plan = self.plan(self.ref)
        self.assertTrue(plan["valid"], plan)
        self.assertEqual(plan["status"], "aligned")
        np.testing.assert_allclose(plan["target_tcp_mm"], self.ref, atol=1e-7)


class VisionRunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.profile, self.c, self.ref = motion_fixture()
        self.cfg = TaughtWorkspace(self.profile).config
        self.backend = SimBackend(self.cfg, self.ref)
        self.session = SimpleNamespace(cfg=self.cfg, T0=self.ref.copy(), meta={}, path=Path(self.temp.name))
        self.runner = Runner(self.session, self.backend, single_step=False, unattended=True)
        self.adapter = GuardedVisionAdapter(self.runner, self.profile, self.c)

    def obs(self, delta=(0, 0, 0), *, stamp=None):
        bottle = self.ref.copy()
        bottle[:3, 3] += delta
        return observation(self.c, bottle, int(self.runner.now()) if stamp is None else stamp)

    def assert_no_motion(self):
        self.assertFalse(any(name == "move_linear" for name, _ in self.backend.calls), self.backend.calls)

    def test_runner_executes_visual_translation_and_restores_session_reference(self):
        saved = self.session.T0.copy()
        with patch.object(self.runner, "move", wraps=self.runner.move) as called:
            receipt = self.adapter.move_step(self.obs((-4, 3, 2)), tolerance_mm=.2)
        self.assertTrue(receipt["applied"])
        called.assert_called_once()
        self.assertEqual(called.call_args.kwargs["noncontact"], True)
        self.assertEqual(called.call_args.kwargs["allow_contact"], False)
        np.testing.assert_allclose(receipt["applied_translation_mm"], [-4, 3, 2], atol=1e-6)
        np.testing.assert_allclose(self.backend.pose[:3, 3], self.ref[:3, 3]+[-4, 3, 2], atol=1e-6)
        np.testing.assert_array_equal(self.session.T0, saved)
        self.assertAlmostEqual(self.adapter.travelled_mm, np.sqrt(29.), places=6)
        self.assertFalse(any(name in ("close", "force", "open") for name, _ in self.backend.calls))
        self.assertEqual(self.runner.stage, "VISUAL_MOVE")
        self.assertFalse(self.runner.airborne)

    def test_current_pose_not_reference_uses_dynamic_geometry_anchor(self):
        self.backend.pose[:3, 3] += [-10, 0, 0]
        self.backend.command_pose = self.backend.pose.copy()
        self.backend.step(0.)
        receipt = self.adapter.move_step(self.obs((0, 3, 0)))
        self.assertTrue(receipt["applied"])
        np.testing.assert_allclose(receipt["applied_translation_mm"], [10, 3, 0], atol=1e-6)
        np.testing.assert_array_equal(self.session.T0, self.ref)

    def test_alignment_does_not_issue_an_actuator_command(self):
        receipt = self.adapter.move_step(self.obs())
        self.assertFalse(receipt["applied"])
        self.assertTrue(receipt["visual_alignment_only"])
        self.assert_no_motion()

    def test_closed_hand_and_contact_are_rejected_before_move(self):
        for condition in ("closed", "contact", "airborne", "grasp_pose"):
            with self.subTest(condition=condition):
                # Rebuild adapter so a deliberate refusal cannot poison another case.
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
                    adapter.move_step(self.obs((-4, 0, 0)))
                self.assert_no_motion()

    def test_gripper_state_is_rechecked_immediately_before_command(self):
        healthy = self.runner._healthy_snapshot
        reads = 0

        def changed_after_plan():
            nonlocal reads
            reads += 1
            if reads >= 2:
                self.backend.snapshot_cache["width_mm"] = 80.
            return healthy()

        with patch.object(self.runner, "_healthy_snapshot", side_effect=changed_after_plan):
            with self.assertRaises((ValueError, SafetyFault)):
                self.adapter.move_step(self.obs((-3, 0, 0)))
        self.assert_no_motion()

    def test_repeated_image_cannot_trigger_second_action(self):
        obs = self.obs((-4, 0, 0))
        self.adapter.move_step(obs)
        count = sum(name == "move_linear" for name, _ in self.backend.calls)
        with self.assertRaises(ValueError):
            self.adapter.move_step(obs)
        self.assertEqual(sum(name == "move_linear" for name, _ in self.backend.calls), count)

    def test_image_from_during_previous_motion_is_not_post_action_observation(self):
        # A 0.1mm movement is intentionally shorter than max image age, so age
        # alone cannot reject an image captured before action completion.
        before = self.runner.now()
        receipt = self.adapter.move_step(self.obs((-.1, 0, 0)), tolerance_mm=.01)
        completed = receipt["completed_ns"]
        stamp = int((before+completed)//2)
        self.assertGreater(stamp, before)
        self.assertLess(stamp, completed)
        count = sum(name == "move_linear" for name, _ in self.backend.calls)
        with self.assertRaises(ValueError):
            self.adapter.move_step(self.obs((-.2, 0, 0), stamp=stamp), tolerance_mm=.01)
        self.assertEqual(sum(name == "move_linear" for name, _ in self.backend.calls), count)

    def test_fresh_post_action_image_allows_next_step_and_charges_total(self):
        first = self.adapter.move_step(self.obs((-3, 0, 0)))
        self.backend.step(.01)
        second = self.adapter.move_step(self.obs((-3, 3, 0)))
        self.assertGreater(second["plan"]["observation"]["timestamp_ns"], first["completed_ns"])
        self.assertAlmostEqual(self.adapter.travelled_mm, 6., places=6)
        self.assertEqual(len(self.adapter.receipts), 2)

    def test_cumulative_budget_rejects_further_motion(self):
        self.adapter.move_step(self.obs((-3, 0, 0)))
        self.adapter.travelled_mm = 99.
        self.backend.step(.01)
        count = sum(name == "move_linear" for name, _ in self.backend.calls)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.adapter.move_step(self.obs((-6, 0, 0)))
        self.assertEqual(sum(name == "move_linear" for name, _ in self.backend.calls), count)

    def test_config_and_simulation_mismatches_block_adapter_creation(self):
        self.runner.cfg = deepcopy(self.cfg)
        self.runner.cfg["safety"]["max_step_mm"] += 1.
        with self.assertRaises(ValueError):
            GuardedVisionAdapter(self.runner, self.profile, self.c)
        self.runner.cfg = self.cfg
        self.backend.simulated = False
        with self.assertRaises(ValueError):
            GuardedVisionAdapter(self.runner, self.profile, self.c)
        self.backend.simulated = "true"
        with self.assertRaises(ValueError):
            GuardedVisionAdapter(self.runner, self.profile, self.c)

    def test_move_failure_stops_and_restores_reference(self):
        before = self.session.T0.copy()
        with patch.object(self.runner, "move", side_effect=SafetyFault("synthetic motion fault")):
            with self.assertRaisesRegex(SafetyFault, "synthetic motion fault"):
                self.adapter.move_step(self.obs((-3, 2, 0)))
        np.testing.assert_array_equal(self.session.T0, before)
        self.assertIn("stop_arm", [name for name, _ in self.backend.calls])
        self.backend.step(.01)
        # A new image cannot clear a failed-motion latch.
        with self.assertRaises((ValueError, RuntimeError)):
            self.adapter.move_step(self.obs((-3, 2, 0)))
        self.assert_no_motion()

    def test_parked_arm_movement_before_command_rejects_motion(self):
        from bottle_scan.park_guard import ParkGuard, ParkGuardError
        self.session.save_meta = lambda: None
        ParkGuard.capture(self.session, self.backend.park_observation(), now_ns=self.runner.now())
        original_park = self.backend.park_observation
        checks = 0

        def relocated_before_command():
            nonlocal checks
            checks += 1
            result = original_park()
            if checks >= 2:
                result["arms"]["left"]["T_base_tcp_mm"][0][3] += 2.
            return result

        with patch.object(self.backend, "park_observation", side_effect=relocated_before_command):
            with self.assertRaises(ParkGuardError):
                self.adapter.move_step(self.obs((-3, 0, 0)))
        self.assert_no_motion()
        self.assertTrue(self.adapter.faulted)
        np.testing.assert_array_equal(self.session.T0, self.ref)

    def test_parked_arm_movement_during_motion_stops_and_latches(self):
        from bottle_scan.park_guard import ParkGuard, ParkGuardError
        self.session.save_meta = lambda: None
        ParkGuard.capture(self.session, self.backend.park_observation(), now_ns=self.runner.now())
        original_park = self.backend.park_observation
        checks = 0

        def relocated_during_motion():
            nonlocal checks
            checks += 1
            result = original_park()
            if checks >= 3:
                result["arms"]["left"]["T_base_tcp_mm"][0][3] += 2.
            return result

        with patch.object(self.backend, "park_observation", side_effect=relocated_during_motion):
            with self.assertRaises(ParkGuardError):
                self.adapter.move_step(self.obs((-3, 0, 0)))
        names = [name for name, _ in self.backend.calls]
        self.assertIn("move_linear", names)
        self.assertIn("stop_arm", names)
        self.assertTrue(self.adapter.faulted)
        np.testing.assert_array_equal(self.session.T0, self.ref)

    def test_stale_healthy_snapshot_is_still_checked_before_motion(self):
        self.backend.snapshot_cache["timestamp_ns"] -= 1_000_000_000
        with self.assertRaisesRegex(SafetyFault, "Stale"):
            self.adapter.move_step(self.obs((-3, 0, 0)))
        self.assert_no_motion()


if __name__ == "__main__":
    unittest.main()
