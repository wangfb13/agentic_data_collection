"""State-machine safety regressions with synthetic inputs only."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import h5py
import numpy as np

from bottle_scan.backend import SimBackend
from bottle_scan.config import demo_config
from bottle_scan.geometry import translate_local
from bottle_scan.runtime import Runner, SafetyFault, SkippedPoint
from bottle_scan.session import Session
from bottle_scan.sensing import TemplateGuard


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        cfg = demo_config()
        cfg["gripper"]["close_speed_mm_s"] = 100.
        cfg["tactile"]["baseline_s"] = .2
        self.session = Session.create(Path(self.temp.name)/"session", cfg, np.asarray(cfg["T0"]), simulated=True)
        self.backend = SimBackend(cfg, self.session.T0)
        self.runner = Runner(self.session, self.backend, single_step=False, unattended=True)
        self.item = {"trial_id": "unit", "bottle": "light", "kind": "zero", "delta_acl_mm": [0, 0, 0],
                     "angle_deg": 0, "T_target": self.session.T0.tolist(),
                     "T_pre": translate_local(self.session.T0, -50).tolist()}

    def unit_meta(self):
        self.session.start_trial("unit")
        self.runner.meta = {"trial_id": "unit", "stages": [], "auto": {"exceptions": [],
                            "bilateral_contact": False, "contact_times_ns": [None, None],
                            "contact_widths_mm": [None, None], "dropped": False,
                            "slip_suspected": False, "normal_pre_lift_N": [2., 2.]}}

    def test_airborne_fault_stops_then_holds_never_opens(self):
        seen = []
        def move(*args, **kwargs):
            if self.runner.stage == "LIFT":
                persisted = json.loads(self.session.progress_path.read_text())
                seen.append(persisted["requires_operator_recovery"])
                self.assertTrue(self.runner.airborne)
                raise SafetyFault("synthetic failure before lift command")
        with patch.object(self.runner, "move", side_effect=move):
            with self.assertRaises(SafetyFault):
                self.runner.contact_trial(self.item)
        self.assertEqual(seen, [True])
        names = [name for name, _ in self.backend.calls]
        at_stop = names.index("stop_arm")
        self.assertEqual(names[at_stop:at_stop+2], ["stop_arm", "hold_gripper"])
        self.assertNotIn("open", names[at_stop:])
        self.assertNotIn("force", names[at_stop:])
        self.assertNotIn("move_linear", names[at_stop:])
        self.assertTrue(self.session.progress["requires_operator_recovery"])

    def test_airborne_open_is_refused(self):
        self.runner.airborne = True
        with self.assertRaises(SafetyFault):
            self.runner.ensure_open()
        with self.assertRaises(SafetyFault):
            self.runner.emergency_release_pre_lift()
        self.assertFalse(self.backend.calls)

    def test_pre_lift_emergency_open_ignores_failed_tactile_health(self):
        self.backend.width = self.backend.width_command = 36.
        original = self.backend.snapshot
        def tactile_fault():
            snap = dict(original())
            snap["normal_N"] = np.array([100., 100.])
            snap["timestamp_ns"] = self.backend.now_ns-1_000_000_000
            snap["gripper_timestamp_ns"] = self.backend.now_ns
            snap["dd"] = [np.full((32, 48, 3), np.nan)]*2
            return snap
        with patch.object(self.backend, "snapshot", side_effect=tactile_fault):
            self.runner.emergency_release_pre_lift()
        names = [name for name, _ in self.backend.calls]
        self.assertEqual(names[:3], ["stop_arm", "exit_force", "open"])
        self.assertAlmostEqual(self.backend.width, 90., delta=.5)
        self.assertNotIn("move_linear", names)

    def test_failed_pre_lift_retreat_cannot_continue_sequence(self):
        self.unit_meta()
        with patch.object(self.runner, "move", side_effect=SafetyFault("cannot certify safe retreat")):
            with self.assertRaises(SafetyFault):
                self.runner.recover(RuntimeError("original failure"), self.session.T0)
        self.assertFalse(self.runner.meta["auto"]["recovery_succeeded"])
        self.assertEqual(self.runner.meta["status"], "aborted")
        self.assertTrue(self.session.progress["requires_operator_recovery"])
        self.assertEqual(self.backend.calls[-1][0], "stop_arm")

    def test_emergency_open_missing_gripper_feedback_forbids_retreat(self):
        self.unit_meta()
        original = self.backend.snapshot
        def gripper_stale():
            snap = dict(original())
            snap["gripper_timestamp_ns"] = self.backend.now_ns-1_000_000_000
            return snap
        with patch.object(self.backend, "snapshot", side_effect=gripper_stale):
            with self.assertRaises(SafetyFault):
                self.runner.recover(RuntimeError("sensor failure"), self.session.T0)
        names = [name for name, _ in self.backend.calls]
        self.assertIn("open", names)
        self.assertNotIn("move_linear", names)
        self.assertTrue(self.session.progress["requires_operator_recovery"])

    def test_failed_stop_rpc_still_requests_airborne_grip_hold(self):
        self.unit_meta()
        self.runner.airborne = True
        with patch.object(self.backend, "stop_arm", side_effect=OSError("lost arm stop RPC")):
            with self.assertRaises(SafetyFault):
                self.runner.recover(RuntimeError("fault"), self.session.T0)
        self.assertEqual(self.backend.calls[-1][0], "hold_gripper")
        self.assertTrue(json.loads(self.session.progress_path.read_text())["requires_operator_recovery"])
        self.assertNotIn("open", [name for name, _ in self.backend.calls])

    def test_disk_full_cannot_suppress_airborne_stop_and_hold(self):
        self.unit_meta()
        self.runner.airborne = True
        with patch.object(self.session, "save_progress", side_effect=OSError("disk full")):
            with self.assertRaises(SafetyFault):
                self.runner.recover(RuntimeError("sensor fault"), self.session.T0)
        names = [name for name, _ in self.backend.calls]
        self.assertEqual(names[:2], ["stop_arm", "hold_gripper"])
        self.assertNotIn("open", names)
        self.assertTrue(self.runner.logging_failed)

    def test_disk_full_cannot_block_pre_lift_open_or_certified_retreat(self):
        self.unit_meta()
        self.runner.single_step = True
        self.backend.width = self.backend.width_command = 36.
        def failed_pump():
            self.backend.drain_samples()
            raise OSError("raw disk full")
        self.runner.recorder = SimpleNamespace(
            h5=SimpleNamespace(append=Mock(side_effect=OSError("HDF5 disk full"))),
            pump=failed_pump, error=None)
        target = translate_local(self.session.T0, -50)
        with patch.object(self.runner, "ask", side_effect=AssertionError("Emergency must not wait for input")), \
             patch("bottle_scan.runtime.atomic_json", side_effect=OSError("JSON disk full")), \
             patch.object(self.session, "save_progress", side_effect=OSError("progress disk full")):
            with self.assertRaises(SafetyFault):
                self.runner.recover(OSError("original logging error"), target)
        names = [name for name, _ in self.backend.calls]
        self.assertEqual(names[:3], ["stop_arm", "exit_force", "open"])
        self.assertIn("move_linear", names)
        np.testing.assert_allclose(self.backend.pose, target)
        self.assertAlmostEqual(self.backend.width, 90., delta=.5)
        self.assertTrue(self.session.progress["requires_operator_recovery"])
        self.assertTrue(self.runner.logging_failed)

    def test_normal_stage_recording_remains_strict(self):
        self.unit_meta()
        self.runner.recorder = SimpleNamespace(h5=SimpleNamespace(append=Mock(side_effect=OSError("disk full"))))
        entered = False
        with self.assertRaises(OSError):
            with self.runner.phase("RELEASE"):
                entered = True
        self.assertFalse(entered)

    def test_airborne_geometric_rejection_terminates_sequence(self):
        def move(*args, **kwargs):
            if self.runner.stage == "LIFT":
                raise SkippedPoint("synthetic carried-bottle workspace rejection")
        with patch.object(self.runner, "move", side_effect=move):
            with self.assertRaises(SafetyFault):
                self.runner.contact_trial(self.item)
        self.assertTrue(self.session.progress["requires_operator_recovery"])
        self.assertNotIn("unit", self.session.progress["completed"])
        next_item = dict(self.item, trial_id="next")
        previous_calls = len(self.backend.calls)
        with self.assertRaises(SafetyFault):
            self.runner.contact_trial(next_item)
        self.assertEqual(len(self.backend.calls), previous_calls)
        self.assertFalse((self.session.path/"trials"/"next").exists())

    def test_close_bilateral_single_no_contact_and_stall(self):
        for scenario in ("normal", "single_contact", "no_contact", "stall"):
            with self.subTest(scenario=scenario):
                # Keep each subcase independent without reusing progress/trial files.
                self.backend.scenario = scenario
                self.backend.width = self.backend.width_command = self.session.cfg["gripper"]["open_mm"]
                self.backend.force_mode = False
                self.unit_meta()
                bilateral = self.runner.close_contact()
                self.assertEqual(bilateral, scenario == "normal")
                self.assertEqual(self.backend.calls[-1][0], "hold_gripper")
                if scenario == "single_contact":
                    self.assertIsNotNone(self.runner.meta["auto"]["contact_times_ns"][0])
                    self.assertIsNone(self.runner.meta["auto"]["contact_times_ns"][1])
                elif scenario == "no_contact":
                    self.assertTrue(self.runner.meta["auto"]["no_bilateral_contact"])
                    self.assertTrue(self.runner.meta["auto"]["safety_closure_limit"])
                    self.assertEqual(self.runner.meta["auto"]["closure_stop_reason"],
                                     "geometric_width_floor_without_bilateral_contact")
                elif scenario == "stall":
                    self.assertTrue(self.runner.meta["auto"]["stall_without_contact"])

    def test_geometry_width_floor_is_enforced_by_force_controller(self):
        self.unit_meta()
        self.assertTrue(self.runner.close_contact())
        floor = self.runner.closure_min_width
        self.assertGreater(floor, self.runner.cfg["gripper"]["min_mm"])
        self.assertLess(floor, 36.)
        self.assertEqual(self.backend.gripper_min_width, floor)
        self.backend.set_gripper_min_width(36.)
        self.backend.set_force_target(4.)
        self.backend.step(.1)
        self.assertGreaterEqual(self.backend.width, 36.)
        with self.assertRaises(ValueError):
            self.backend.command_close(20., 2.)

    def test_nominal_empty_mock_does_not_fabricate_contact(self):
        target = self.session.T0.copy()
        target[2, 3] += 80.
        item = dict(self.item, T_target=target.tolist(), delta_acl_mm=[0, 0, 80])
        self.backend.set_trial(item, scenario="normal")
        self.backend.set_stage("CLOSE")
        self.backend.command_close(20., 100.)
        self.backend.step(1.)
        np.testing.assert_array_equal(self.backend.snapshot()["normal_N"], [0., 0.])
        self.assertEqual(self.backend.width, 20.)

    def test_hold_flags_any_finger_drop_and_dd_drift(self):
        for scenario, flag in (("dropped", "dropped"), ("slip", "slip_suspected")):
            with self.subTest(scenario=scenario):
                self.unit_meta()
                self.backend.scenario = scenario
                self.backend.set_force_target(2.)
                self.runner.hold()
                self.assertTrue(self.runner.meta["auto"][flag])

    def test_hold_requires_explicit_rotation_calibration(self):
        self.unit_meta()
        del self.runner.cfg["tactile"]["sensor_to_tcp"]
        with self.assertRaises(SafetyFault):
            self.runner.hold()

    def test_visual_never_closes_or_sets_force_target(self):
        point = {"point_id": "view", "bottle": "heavy", "T_target": self.session.T0.tolist()}
        with patch.object(self.runner, "move"):
            meta = self.runner.visual_point(point)
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(meta["bottle"], "heavy")
        saved = json.loads((self.session.path/"trials"/"view"/"meta.json").read_text())
        self.assertEqual(saved["bottle"], "heavy")
        self.assertIn("heavy", (self.session.path/"trial_index.csv").read_text())
        self.assertNotIn("close", [name for name, _ in self.backend.calls])
        self.assertNotIn("force", [name for name, _ in self.backend.calls])

    def test_visual_rejects_missing_or_conflicting_bottle_before_any_action(self):
        point = {"point_id": "view", "T_target": self.session.T0.tolist()}
        before = list(self.backend.calls)
        with self.assertRaisesRegex(ValueError, "explicit planned bottle"):
            self.runner.visual_point(point)
        with self.assertRaisesRegex(ValueError, "override disagrees"):
            self.runner.visual_point(dict(point, bottle="heavy"), bottle="light")
        self.assertEqual(self.backend.calls, before)
        self.assertIsNone(self.session.progress["active"])
        self.assertFalse((self.session.path/"trials"/"view").exists())

    def test_move_rejects_stale_or_nonfinite_before_command(self):
        self.backend.snapshot_cache["timestamp_ns"] -= 1_000_000_000
        with self.assertRaises(SafetyFault):
            self.runner.move(self.session.T0)
        self.assertNotIn("move_linear", [name for name, _ in self.backend.calls])
        self.backend.snapshot_cache["timestamp_ns"] = self.backend.now_ns
        self.backend.snapshot_cache["width_mm"] = float("nan")
        with self.assertRaises(SafetyFault):
            self.runner.move(self.session.T0)
        self.assertNotIn("move_linear", [name for name, _ in self.backend.calls])

    def test_online_actual_pose_must_stay_on_certified_path(self):
        target = translate_local(self.session.T0, -50)
        self.runner.active_motion = (self.session.T0.copy(), target)
        self.backend.snapshot_cache["tcp"] = self.session.T0.copy()
        self.backend.snapshot_cache["tcp"][1, 3] += 3.
        with self.assertRaises(SafetyFault):
            self.runner._healthy_snapshot()

    def test_check_reset_keeps_reference_and_requires_both_cameras(self):
        images = self.backend.images()
        rois = {"left_wrist": (88, 42, 61, 101), "right_wrist": (92, 42, 61, 101)}
        guard = TemplateGuard(images, rois, {name: 3. for name in images}, min_peak_margin=0)
        self.runner.guard = guard
        before = guard.reference_sha256
        self.backend.bottle_shift_px = [8, 0]
        with patch.object(self.runner, "move"):
            with self.assertRaises(SafetyFault):
                self.runner.check_bottle("moved")
            self.backend.bottle_shift_px = [0, 0]
            self.runner.check_bottle("reset")
        self.assertEqual(before, guard.reference_sha256)
        self.assertFalse(guard.compare({"left_wrist": images["left_wrist"]})["passed"])

    def test_baseline_preserves_native_sequences_and_averages(self):
        self.runner.baseline()
        reference = self.session.meta["references"][-1]
        with h5py.File(self.session.path/reference["raw_path"], "r") as h5:
            group = h5["streams/dd_left"]
            n = int(group.attrs["committed_count"])
            self.assertGreaterEqual(n, 2)
            metadata = [json.loads(v) for v in group["metadata_json"][:n]]
            self.assertEqual(len(set(m["source_sequence"] for m in metadata)), n)
            raw_mean = group["data"][:n].mean(axis=0, dtype=np.float64)
        with np.load(self.session.path/reference["path"]) as data:
            np.testing.assert_array_equal(data["left_dd"], raw_mean)

    def test_learned_force_policy_is_not_silently_fixed(self):
        self.unit_meta()
        self.runner.cfg["force"]["squeeze_mode"] = "forcepolicy"
        with self.assertRaises(SafetyFault):
            self.runner.squeeze()
        self.assertNotIn("force", [name for name, _ in self.backend.calls])


if __name__ == "__main__":
    unittest.main()
