"""Fake ROS only: no ROS imports, sockets, device initialization or actuation."""
import json
import subprocess
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from bottle_scan.hardware_gripper import RosNormalizedGripperObserver, WidthCalibration


class FakeRos:
    def __init__(self, initialized=False):
        self.initialized = initialized
        self.core = types.SimpleNamespace(is_initialized=lambda: self.initialized)
        self.calls = []

    def init_node(self, name, **kwargs):
        self.calls.append(("init_node", name, kwargs))
        self.initialized = True

    def Subscriber(self, topic, kind, callback, **kwargs):
        self.calls.append(("subscribe", topic, kwargs))
        self.callback = callback
        return types.SimpleNamespace(unregister=lambda: self.calls.append(("unregister", topic)))

    def Publisher(self, *args, **kwargs):
        raise AssertionError("Read-only observer must never create a publisher")

    def ServiceProxy(self, *args, **kwargs):
        raise AssertionError("Read-only observer must never call a service")

    def signal_shutdown(self, *args):
        raise AssertionError("Observer must not shut down the shared ROS node")


class WidthCalibrationTests(unittest.TestCase):
    def test_piecewise_forward_inverse_and_endpoints(self):
        calibration = WidthCalibration([(.2, 10), (.5, 40), (.9, 60)])
        self.assertEqual(calibration.to_mm(.2), 10.)
        self.assertEqual(calibration.to_mm(.9), 60.)
        self.assertAlmostEqual(calibration.to_mm(.7), 50.)
        self.assertAlmostEqual(calibration.to_normalized(50), .7)
        self.assertAlmostEqual(calibration.to_normalized(25), .35)
        points = calibration.points
        points[0]["width_mm"] = 999
        self.assertEqual(calibration.to_mm(.2), 10.)

    def test_two_points_dicts_and_strict_domains(self):
        calibration = WidthCalibration([{"normalized": .3, "width_mm": 20},
                                        {"normalized": 1, "width_mm": 90}])
        self.assertAlmostEqual(calibration.to_mm(.6), 50.)
        for value in (.29999, 1.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                calibration.to_mm(value)
        for value in (19.99, 90.01, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                calibration.to_normalized(value)

    def test_ambiguous_or_nonphysical_calibration_rejected(self):
        for points in ([], [(0, 0)], [(0, 10), (1, 5)], [(0, 10), (1, 10)],
                       [(0, 0), (0, 1)], [(1, 1), (0, 0)], [(0, -1), (1, 2)],
                       [(-.1, 0), (1, 2)], [(0, 0), (1.1, 2)], [(0, 0), (1, float("nan"))]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                WidthCalibration(points)


class ObserverTests(unittest.TestCase):
    def make_observer(self, **kwargs):
        self.now = 10_000_000_000
        self.ros = FakeRos(kwargs.pop("initialized", False))
        self.observer = RosNormalizedGripperObserver(
            rospy_module=self.ros, message_type=object, clock_ns=lambda: self.now, **kwargs)
        return self.observer

    def receive(self, value):
        self.ros.callback(types.SimpleNamespace(data=value))

    def test_import_constructor_and_lifecycle_do_not_actuate(self):
        subprocess.run([sys.executable, "-c",
                        "import sys; sys.modules['rospy']=None; sys.modules['std_msgs']=None; "
                        "import bottle_scan.hardware_gripper"], check=True, capture_output=True)
        with patch.dict(sys.modules, {"rospy": None, "std_msgs": None}):
            observer = self.make_observer()
            self.assertEqual(self.ros.calls, [])
            self.assertFalse(observer.snapshot()["received"])
            observer.connect().connect()
            self.assertEqual([c[0] for c in self.ros.calls], ["init_node", "subscribe"])
            self.assertEqual(self.ros.calls[0][2], {"anonymous": True, "disable_signals": True})
            observer.close()
            observer.close()
            self.assertEqual([c[0] for c in self.ros.calls], ["init_node", "subscribe", "unregister"])

    def test_receipt_timestamp_freshness_and_unknown_physical_fields(self):
        observer = self.make_observer(initialized=True).connect()
        self.receive(.6)
        snapshot = observer.snapshot()
        self.assertTrue(snapshot["fresh"])
        self.assertEqual(snapshot["monotonic_ns"], self.now)
        self.assertEqual(snapshot["counts"], 600.)
        self.assertIn("not_hardware_encoder", snapshot["counts_kind"])
        for key in ("width_mm", "velocity_mm_s", "current_A", "device_timestamp"):
            self.assertIsNone(snapshot[key])
        self.assertFalse(snapshot["width_calibrated"])
        self.assertEqual(len(self.ros.calls), 1)
        self.now += 251_000_000
        self.assertFalse(observer.snapshot()["fresh"])
        # Calling snapshot does not refresh the source timestamp.
        self.assertEqual(observer.snapshot()["monotonic_ns"], 10_000_000_000)
        self.now -= 252_000_000
        self.assertFalse(observer.snapshot()["fresh"])

    def test_invalid_message_does_not_look_like_last_good_or_fresh_sample(self):
        observer = self.make_observer().connect()
        self.receive(.5)
        for value in (float("nan"), float("inf"), -.01, 1.01, "broken"):
            self.receive(value)
            snapshot = observer.snapshot()
            self.assertFalse(snapshot["valid"])
            self.assertFalse(snapshot["fresh"])
            self.assertIsNotNone(snapshot["error"])
            json.dumps(snapshot, allow_nan=False)

    def test_calibrated_width_outside_domain_is_unavailable_not_clipped(self):
        from bottle_scan.hardware_gripper import WidthCalibration as Calibration
        observer = self.make_observer(calibration=Calibration([(.3, 20), (1, 90)])).connect()
        self.receive(.5)
        self.assertAlmostEqual(observer.snapshot()["width_mm"], 40.)
        self.receive(.2)
        snapshot = observer.snapshot()
        self.assertTrue(snapshot["valid"])
        self.assertIsNone(snapshot["width_mm"])
        self.assertIsNotNone(snapshot["width_calibration_error"])
        observer.close()
        self.assertFalse(observer.snapshot()["fresh"])
        observer.connect()
        self.assertFalse(observer.snapshot()["received"])

    def test_invalid_observer_config_rejected_before_connection(self):
        for kwargs in ({"side": "both"}, {"max_age_s": 0}, {"max_age_s": float("nan")},
                       {"calibration": [(0, 0), (1, 90)]}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                RosNormalizedGripperObserver(**kwargs)


class CommandRos(FakeRos):
    def Publisher(self, topic, kind, **kwargs):
        self.calls.append(("advertise", topic, kwargs))
        self.published = []
        return types.SimpleNamespace(publish=lambda msg: self.published.append(msg.data),
                                     unregister=lambda: self.calls.append(("publisher_unregister", topic)))


class FakeLegacy:
    source_sha256 = "test-fixture-not-a-real-controller"

    def __init__(self):
        self.calls = []
        self.proposal = 499.

    def combine_force(self, config, wrenches):
        normals = [row[2] for row in wrenches]
        self.calls.append(("combine", config, normals))
        return sum(normals)/2., normals

    def tag_step(self, config, measured, actual):
        self.calls.append(("step", config.copy(), measured, actual))
        return "close", self.proposal


class ControllerTests(unittest.TestCase):
    def setUp(self):
        from bottle_scan.hardware_gripper import RosGripperController, WidthCalibration as Calibration
        self.now = 10_000_000_000
        self.ros, self.legacy = CommandRos(), FakeLegacy()
        self.force = {"normal_N": [1., 1.], "monotonic_ns": self.now, "source_sequence": [1, 1]}
        self.controller = RosGripperController(
            "right", Calibration([(0, 0), (1, 100)]), lambda: self.force,
            min_width_mm=20., max_width_mm=90., max_force_N=4., force_deadband_N=.1,
            force_step_counts=1, force_settle_s=.1, max_age_s=.25,
            rospy_module=self.ros, message_type=types.SimpleNamespace,
            clock_ns=lambda: self.now, start_worker=False, legacy_functions=self.legacy).connect()
        self.receive(.5)

    def tearDown(self):
        self.controller.close()

    def receive(self, value):
        self.ros.callback(types.SimpleNamespace(data=value))

    def advance(self, seconds=.05, position=None, sequences=None):
        self.now += int(seconds*1e9)
        self.force["monotonic_ns"] = self.now
        if sequences is not None:
            self.force["source_sequence"] = sequences
        if position is not None:
            self.receive(position)

    def test_connect_snapshot_and_close_do_not_publish(self):
        self.controller._tick()
        self.assertEqual(self.ros.published, [])
        first = self.controller.snapshot()
        self.assertFalse(first["velocity_available"])
        self.assertFalse(first["current_available"])
        self.advance(position=.51)
        second = self.controller.snapshot()
        self.assertAlmostEqual(second["velocity_mm_s"], 20.)
        self.assertIn("finite_difference", second["velocity_kind"])
        samples = self.controller.drain_samples()
        self.assertEqual(len(samples), 2)
        self.assertEqual([sample.stream for sample in samples], ["gripper", "gripper"])
        self.assertNotEqual(samples[0].monotonic_ns, samples[1].monotonic_ns)
        self.assertEqual(self.controller.drain_samples(), [])
        self.controller.close()
        self.assertEqual(self.ros.published, [])

    def test_position_steps_have_requested_software_speed_and_need_new_feedback(self):
        self.controller.command_close(30, 2)
        self.assertEqual(self.ros.published, [])
        self.advance()
        self.controller._tick()
        self.assertAlmostEqual(self.ros.published[-1], .499)  # 0.1 mm / 50 ms.
        self.advance()
        self.controller._tick()
        self.assertEqual(len(self.ros.published), 1)
        self.receive(.499)
        self.controller._tick()
        self.assertAlmostEqual(self.ros.published[-1], .498)

    def test_force_uses_existing_mean_definition_target_and_new_pair(self):
        self.force["normal_N"] = [1., 3.]
        self.controller.set_force_target(2.)
        self.controller._tick()
        step = self.legacy.calls[-1]
        self.assertEqual(step[0], "step")
        self.assertEqual(step[1]["target_force"], 2.)
        self.assertEqual(step[2:], (2., 500.))
        self.assertEqual(self.ros.published, [.499])
        self.advance(.11, position=.499, sequences=[2, 1])
        self.controller._tick()
        self.assertEqual(len(self.ros.published), 1)
        self.force["source_sequence"] = [2, 2]
        self.legacy.proposal = 498
        self.controller._tick()
        self.assertEqual(self.ros.published[-1], .498)

    def test_sum_regulation_uses_sum_but_still_limits_each_finger(self):
        self.controller.target_definition="sum"
        self.force["normal_N"]=[1.,2.]
        self.controller.set_force_target(4.)
        self.controller._tick()
        self.assertEqual(self.legacy.calls[-1][2],3.)
        self.assertEqual(self.legacy.calls[-1][1]["target_force"],4.)
        self.assertEqual(self.controller.snapshot()["force_definition"],"sum")

    def test_sdk_unit_is_explicit_and_mismatched_feedback_refused(self):
        self.controller.force_unit="sdk"
        self.force["unit"]="N"
        with self.assertRaisesRegex(RuntimeError,"unit differs"):
            self.controller.set_force_target(2.)
        self.force["unit"]="sdk"
        self.controller.set_force_target(2.)
        self.assertEqual(self.controller.snapshot()["force_unit"],"sdk")

    def test_loop_metric_reset_clears_idle_gap_without_actuation(self):
        self.controller._loop_intervals.extend([.02, .03])
        self.controller._loop_previous_ns = 123
        before = list(self.ros.published)
        self.controller.reset_loop_metrics()
        self.assertEqual(self.controller.loop_metrics()["count"], 0)
        self.assertIsNone(self.controller._loop_previous_ns)
        self.assertEqual(self.ros.published, before)

    def test_handoff_unregisters_without_further_position_steps(self):
        self.controller.command_close(30,2)
        self.controller.suspend_publication()
        self.assertEqual(self.ros.published,[.5])
        self.assertIsNone(self.controller._publisher)
        self.advance(position=.5)
        self.controller._tick()
        self.assertEqual(self.ros.published,[.5])
        self.controller.resume_publication()
        self.assertEqual(self.ros.published,[])  # New fake publisher starts its own empty history.

    def test_recording_queue_is_bounded_and_overflow_latches_controller(self):
        self.controller.drain_samples()
        self.controller._queue_limit=2
        for value in (.5,.51,.52,.53):self.receive(value)
        self.assertIn("queue overflow",self.controller.snapshot()["fault"])
        self.assertEqual(len(self.controller.drain_samples()),3)  # Final overflowing event retained.
        with self.assertRaisesRegex(RuntimeError,"overflow"):
            self.controller.drain_samples()
        with self.assertRaisesRegex(RuntimeError,"fault is latched"):
            self.controller.command_close(30,2)

    def test_loop_statistics_come_from_actual_active_tick_times(self):
        self.controller.command_close(30,2)
        self.controller._tick()
        self.advance(.05,position=.5)
        self.controller._tick()
        self.advance(.1,position=.499)
        self.controller._tick()
        stats=self.controller.loop_metrics()
        self.assertEqual(stats["count"],2)
        self.assertAlmostEqual(stats["mean_s"],.075)
        self.assertAlmostEqual(stats["max_s"],.1)

    def test_each_finger_max_overload_cannot_be_hidden_by_mean(self):
        self.controller.command_close(30, 2)
        self.force["normal_N"] = [4.1, .1]
        self.advance()
        self.controller._tick()
        self.assertEqual(self.ros.published, [.5])  # Hold measured, never close.
        self.assertIn("F_max", self.controller.snapshot()["fault"])
        with self.assertRaises(RuntimeError):
            self.controller.command_close(30, 2)

    def test_force_target_and_dynamic_floor_reject_without_clipping(self):
        with self.assertRaises(ValueError):
            self.controller.set_force_target(4.)
        self.controller.set_gripper_min_width(40)
        with self.assertRaises(ValueError):
            self.controller.command_close(39, 2)
        self.assertEqual(self.ros.published, [])
        self.controller.set_force_target(2)
        self.legacy.proposal = 399
        self.controller._tick()
        self.assertEqual(self.ros.published, [.5])
        self.assertIn("width bound", self.controller.snapshot()["fault"])

    def test_exit_force_and_hold_use_measured_current_not_last_target(self):
        self.controller.set_force_target(2)
        self.controller._tick()
        self.receive(.497)
        self.controller.exit_force_mode()
        self.assertEqual(self.ros.published[-1], .497)
        self.advance(.11, position=.497, sequences=[3, 3])
        self.controller._tick()
        self.assertEqual(len(self.ros.published), 2)
        self.assertEqual(self.controller.snapshot()["controller_state"], "hold")

    def test_stale_feedback_fault_sends_no_unjustified_hold(self):
        self.controller.command_close(30, 2)
        self.advance(.3)
        self.controller._tick()
        self.assertEqual(self.ros.published, [])
        self.assertIn("hold unavailable", self.controller.snapshot()["fault"])

    def test_scalar_or_stale_force_cannot_masquerade_as_two_new_fingers(self):
        self.force["normal_N"] = 1.
        with self.assertRaises(RuntimeError):
            self.controller.set_force_target(2)
        self.force["normal_N"] = [1., 1.]
        self.force["monotonic_ns"] = [self.now, self.now-300_000_000]
        with self.assertRaises(RuntimeError):
            self.controller.command_close(30, 2)
        self.assertEqual(self.ros.published, [])

    def test_lost_contact_stops_regulation_without_automatic_reclose(self):
        self.controller.set_force_target(2)
        self.force["normal_N"] = [0., 1.]
        self.controller._tick()
        self.assertEqual(self.ros.published, [.5])
        self.assertIsNone(self.controller.snapshot()["fault"])
        self.assertEqual(self.controller.snapshot()["controller_state"], "hold")
        self.advance(.11, position=.5, sequences=[3, 3])
        self.force["normal_N"] = [1., 1.]
        self.controller._tick()
        self.assertEqual(self.ros.published, [.5])

    def test_emergency_open_after_overforce_keeps_fault_and_only_opens(self):
        self.controller.command_close(30, 2)
        self.force["normal_N"] = [4.5, .5]
        self.advance()
        self.controller._tick()
        original_fault = self.controller.snapshot()["fault"]
        self.assertIn("F_max",original_fault)
        self.controller.emergency_open_pre_lift(60, 2)
        self.assertEqual(self.ros.published[-1],.5)  # Cancel old target using measured position.
        self.advance(position=.5)
        self.controller._tick()
        self.assertAlmostEqual(self.ros.published[-1],.501)
        self.advance(position=.501)
        self.controller._tick()
        self.assertAlmostEqual(self.ros.published[-1],.502)
        self.assertEqual(self.controller.snapshot()["fault"],original_fault)
        self.assertEqual(self.controller.snapshot()["controller_state"],"emergency_open")
        self.advance(position=.6)
        self.controller._tick()
        self.assertEqual(self.controller.snapshot()["controller_state"],"hold")
        for operation in (lambda:self.controller.command_close(30,2),
                          lambda:self.controller.command_open(90,2),
                          lambda:self.controller.set_force_target(2)):
            with self.assertRaisesRegex(RuntimeError,"fault is latched"):
                operation()

    def test_emergency_open_force_reader_failure_does_not_prevent_release(self):
        self.controller.set_force_target(2)
        broken = unittest.mock.Mock(side_effect=RuntimeError("force/DD stream disconnected"))
        self.controller._force_reader = broken
        self.controller._tick()
        self.assertEqual(broken.call_count,1)
        self.controller.emergency_open_pre_lift(60,2)
        self.advance(position=.5)
        self.controller._tick()
        self.assertAlmostEqual(self.ros.published[-1],.501)
        self.assertEqual(broken.call_count,1)  # Neither recovery start nor worker uses broken force.
        self.assertIn("stream disconnected",self.controller.snapshot()["fault"])

    def test_emergency_open_stale_position_refuses_and_cancels_old_force_loop(self):
        self.controller.set_force_target(2)
        self.advance(.3)
        with self.assertRaisesRegex(RuntimeError,"stale"):
            self.controller.emergency_open_pre_lift(60,2)
        self.assertEqual(self.ros.published,[])
        self.assertEqual(self.controller.snapshot()["force_target_N"],0.)
        self.receive(.5)
        self.controller._tick()
        self.assertEqual(self.ros.published,[])
        with self.assertRaisesRegex(RuntimeError,"fault is latched"):
            self.controller.set_force_target(2)

    def test_emergency_open_cannot_close_or_exceed_measured_travel(self):
        for target,speed in ((49,2),(91,2),(60,0),(60,float("inf"))):
            with self.subTest(target=target,speed=speed), self.assertRaises(ValueError):
                self.controller.emergency_open_pre_lift(target,speed)
        self.assertEqual(self.ros.published,[])
        self.assertIsNotNone(self.controller.snapshot()["fault"])

    def test_emergency_open_does_not_correct_overshoot_by_closing(self):
        self.controller.emergency_open_pre_lift(50.2,2)
        original_fault = self.controller.snapshot()["fault"]
        self.advance(position=.503)
        self.controller._tick()
        self.assertEqual(self.ros.published[-1],.503)  # Hold overshoot, not a .502 correction.
        self.assertIn("refusing corrective closure",self.controller.snapshot()["fault"])
        self.assertIn(original_fault,self.controller.snapshot()["fault"])

    def test_emergency_open_later_position_loss_preserves_original_fault(self):
        self.controller.emergency_open_pre_lift(60,2)
        original_fault = self.controller.snapshot()["fault"]
        self.advance(.3)
        self.controller._tick()
        self.assertEqual(self.ros.published,[.5])  # No stale-position guess after initial cancellation.
        self.assertIn(original_fault,self.controller.snapshot()["fault"])
        self.assertIn("hold unavailable",self.controller.snapshot()["fault"])

    def test_legacy_loader_does_not_execute_module_or_other_definitions(self):
        from bottle_scan.hardware_gripper import load_legacy_force_functions
        source = '''raise AssertionError("never execute module body")
class Fault(RuntimeError):
    pass
def normal_component(c, f, w):
    return w[2]
def combine_force(c, w):
    return 123, [1, 2]
def tag_step(c, measured_raw, actual):
    return "hold", None
class Dangerous:
    raise AssertionError("never execute unrelated class")
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"existing_controller.py"
            path.write_text(source)
            loaded = load_legacy_force_functions(path)
            self.assertEqual(loaded.combine_force({}, []), (123, [1, 2]))
            self.assertEqual(len(loaded.source_sha256), 64)
            self.assertEqual(path.read_text(), source)


if __name__ == "__main__":
    unittest.main()
