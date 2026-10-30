"""Read-only adapter regression tests: all ROS objects are fakes."""
import json
import copy
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from bottle_scan.hardware_arm import RosArmObserver, RosArmController, ArmObservationError, UnverifiedMotionError
from bottle_scan.geometry import pose_error, rotation

REAL_THREAD = threading.Thread


class Stamp:
    def __init__(self, seconds=100.):
        self.seconds = seconds

    def to_sec(self):
        return self.seconds

    @staticmethod
    def now():
        return Stamp(100.)


def xyz(x=0., y=0., z=0.):
    return NS(x=x, y=y, z=z)


def quat():
    return NS(x=0., y=0., z=0., w=1.)


def state_pose(x):
    return NS(header=NS(frame_id="base_link", stamp=Stamp(), seq=0),
              pose=NS(position=xyz(x, .2, .8), orientation=quat()))


class FakeROS:
    def __init__(self, initialized=False):
        self.initialized = initialized
        self.core = NS(is_initialized=lambda: self.initialized)
        self.Time = Stamp
        self.init_calls = []
        self.subscriptions = []
        self.message = NS(left_arm_state_pose=state_pose(-.1), right_arm_state_pose=state_pose(.1))

    def init_node(self, *args, **kwargs):
        self.init_calls.append((args, kwargs))
        self.initialized = True

    def is_shutdown(self):
        return False

    def Subscriber(self, topic, message_type, callback, queue_size):
        sub = NS(topic=topic, closed=False)
        sub.unregister = lambda: setattr(sub, "closed", True)
        self.subscriptions.append(sub)
        payload = self.message if topic == "/wbc_state_pose" else NS(data="RUN" if topic == "/fsm_state" else True)
        callback(payload)
        return sub

    def Publisher(self, *args, **kwargs):
        raise AssertionError("A read-only observer created a publisher")

    def ServiceProxy(self, *args, **kwargs):
        raise AssertionError("A read-only observer created a service proxy")


class FakeTF:
    def __init__(self):
        self.lookups = []
        self.closed = False
        self.failure = False
        self.stamp = 0.
        self.bad_frame = False

    def Buffer(self):
        return self

    def TransformListener(self, buffer):
        return self

    def unregister(self):
        self.closed = True

    def lookup_transform(self, target, source, stamp):
        self.lookups.append((target, source, stamp.to_sec()))
        if self.failure:
            raise RuntimeError("No TF data")
        return NS(header=NS(frame_id="wrong" if self.bad_frame else target, stamp=Stamp(self.stamp)),
                  child_frame_id=source,
                  transform=NS(translation=xyz(0., 0., .125), rotation=quat()))


class ArmObserverTests(unittest.TestCase):
    def setUp(self):
        self.ros, self.tf = FakeROS(), FakeTF()
        self.dependencies = patch("bottle_scan.hardware_arm._ros_dependencies",
                                  return_value=(self.ros, self.tf, object, object, object))
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)
        self.observer = RosArmObserver(connect_timeout_s=.03)
        self.addCleanup(self.observer.close)

    def test_import_does_not_import_ros_or_connect(self):
        code = """import sys
class RejectROS:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rospy', 'tf2_ros', 'wbc_msgs', 'std_msgs'):
            raise AssertionError('Unexpected ROS import')
sys.meta_path.insert(0, RejectROS())
from bottle_scan.hardware_arm import RosArmObserver
RosArmObserver()
"""
        subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], check=True)

    def test_live_tool_composition_units_and_both_arms(self):
        self.observer.connect()
        snapshot = self.observer.snapshot()
        self.assertEqual(set(snapshot["arms"]), {"left", "right"})
        self.assertTrue(snapshot["read_only"])
        self.assertFalse(snapshot["motion_capabilities_verified"])
        np.testing.assert_allclose(self.observer.read_pose()[:3, 3], [100., 200., 925.])
        np.testing.assert_allclose(self.observer.read_pose("left")[:3, 3], [-100., 200., 925.])
        np.testing.assert_allclose(np.array(snapshot["arms"]["right"]["T_flange_tcp_mm"])[:3, 3], [0., 0., 125.])
        self.assertEqual(snapshot["arms"]["right"]["raw_flange_xyz_m_xyzw"], [.1, .2, .8, 0., 0., 0., 1.])
        self.assertTrue(all(request[2] == 100. for request in self.tf.lookups))
        self.assertEqual(len(self.ros.subscriptions), 3)
        json.dumps(snapshot, allow_nan=False)

    def test_rotated_flange_composes_local_tool_direction(self):
        self.ros.message.right_arm_state_pose.pose.orientation = NS(x=0., y=2**-.5, z=0., w=2**-.5)
        self.observer.connect()
        np.testing.assert_allclose(self.observer.read_pose()[:3, 3], [225., 200., 800.], atol=1e-10)

    def test_shared_ros_node_and_cleanup(self):
        self.ros.initialized = True
        self.observer.connect()
        self.assertEqual(self.ros.init_calls, [])
        self.observer.close()
        self.assertTrue(all(sub.closed for sub in self.ros.subscriptions))
        self.assertTrue(self.tf.closed)
        with self.assertRaises(ArmObservationError):
            self.observer.read_pose()

    def test_all_motion_methods_refuse_even_before_connect(self):
        with self.assertRaises(UnverifiedMotionError):
            self.observer.move_linear(np.eye(4), 5., 5.)
        with self.assertRaises(UnverifiedMotionError):
            self.observer.stop_arm()
        self.assertEqual(self.ros.subscriptions, [])
        self.assertEqual(self.ros.init_calls, [])

    def test_bad_state_frame_refuses_and_cleans_up(self):
        self.ros.message.right_arm_state_pose.header.frame_id = "calibration_target_center"
        with self.assertRaisesRegex(ArmObservationError, "frame differs"):
            self.observer.connect()
        self.assertTrue(all(sub.closed for sub in self.ros.subscriptions))
        self.assertTrue(self.tf.closed)

    def test_stale_device_time_refuses_even_with_fresh_receipt(self):
        self.ros.message.right_arm_state_pose.header.stamp = Stamp(98.)
        with self.assertRaisesRegex(ArmObservationError, "stale or clocks disagree"):
            self.observer.connect()

    def test_stale_receipt_refuses_even_with_valid_device_time(self):
        self.observer.connect()
        message, received = self.observer._state
        self.observer._state = (message, received - 1_000_000_000)
        with self.assertRaisesRegex(ArmObservationError, "receipt is stale"):
            self.observer.read_pose()

    def test_future_time_and_zero_state_time_refuse(self):
        self.observer.connect()
        for stamp in (100.5, 0.):
            self.ros.message.right_arm_state_pose.header.stamp = Stamp(stamp)
            with self.assertRaises(ArmObservationError):
                self.observer.read_pose()

    def test_tf_failure_never_returns_flange_as_tcp(self):
        self.observer.connect()
        self.tf.failure = True
        with self.assertRaisesRegex(ArmObservationError, "TF unavailable"):
            self.observer.read_pose()

    def test_wrong_tf_frames_and_stale_dynamic_tf_refuse(self):
        self.observer.connect()
        self.tf.bad_frame = True
        with self.assertRaisesRegex(ArmObservationError, "unexpected frames"):
            self.observer.read_pose()
        self.tf.bad_frame = False
        self.tf.stamp = 95.
        with self.assertRaisesRegex(ArmObservationError, "stale"):
            self.observer.read_pose()

    def test_invalid_quaternion_refuses(self):
        self.ros.message.right_arm_state_pose.pose.orientation.w = 0.
        with self.assertRaisesRegex(ArmObservationError, "quaternion"):
            self.observer.connect()

    def test_status_is_diagnostic_not_a_safety_capability(self):
        self.observer.connect()
        status = self.observer.status()
        self.assertEqual(status["fsm_state"]["value"], "RUN")
        self.assertTrue(status["motion_active"]["value"])
        self.observer._status["motion_active"] = (True, time.monotonic_ns() - 1_000_000_000)
        self.assertFalse(self.observer.status()["motion_active"]["fresh"])


class FakePublisher:
    def __init__(self):
        self.messages = []
        self.connections = 1
        self.closed = False

    def get_num_connections(self):
        return self.connections

    def publish(self, message):
        self.messages.append(copy.deepcopy(message))

    def unregister(self):
        self.closed = True


class FakeCommand:
    def __init__(self):
        self.left_arm_goal_pose = state_pose(0.)
        self.right_arm_goal_pose = state_pose(0.)
        self.head_goal_pose = state_pose(0.)
        self.hip_goal_pose = state_pose(0.)


class NoThread:
    def __init__(self, **kwargs):
        self.target = kwargs["target"]

    def start(self):
        pass

    def join(self, timeout):
        pass

    def is_alive(self):
        return False


class ArmControllerTests(unittest.TestCase):
    def setUp(self):
        self.ros, self.tf, self.publisher = FakeROS(), FakeTF(), FakePublisher()
        self.ros.Publisher = lambda *args, **kwargs: self.publisher
        for name, replacement in (("_ros_dependencies", lambda: (self.ros, self.tf, object, object, object)),
                                  ("_command_message_type", lambda: FakeCommand),
                                  ("threading.Thread", NoThread)):
            patcher = patch("bottle_scan.hardware_arm." + name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.arm = RosArmController(publish_hz=50., max_step_mm=.15, max_step_deg=.25)
        self.addCleanup(self.arm.close)
        self.arm.connect()
        self.worker_clock = time.monotonic()

    def advance(self, count=1, elapsed=.02):
        # A real worker clock keeps ticking after a completed motion becomes
        # None. Do not derive the fake clock's lifetime from the motion object:
        # float rounding can make a short interpolation finish one tick earlier.
        for _ in range(count):
            if self.arm._motion is not None:
                self.worker_clock = max(self.worker_clock, self.arm._motion["last_monotonic"])
            self.worker_clock += elapsed
            self.arm._advance_locked(self.worker_clock)

    def test_connect_close_do_not_publish(self):
        self.assertEqual(self.publisher.messages, [])
        self.arm.close()
        self.assertEqual(self.publisher.messages, [])
        self.assertTrue(self.publisher.closed)

    def test_per_acquisition_loop_metrics_reset_is_read_only_and_unit_labelled(self):
        self.arm._loop_periods.extend([.02,.03,.04])
        self.arm._loop_period_count=3
        stats=self.arm.loop_metrics()
        self.assertEqual(stats["count"],3)
        self.assertAlmostEqual(stats["mean_s"],.03)
        self.assertAlmostEqual(stats["p95_s"],.039)
        self.assertAlmostEqual(stats["max_s"],.04)
        self.arm.reset_loop_metrics()
        stats=self.arm.loop_metrics()
        self.assertEqual(stats["count"],0)
        self.assertIsNone(stats["mean_s"])
        self.assertEqual(self.publisher.messages,[])

    def test_single_side_mask_and_other_arm_measured_not_default(self):
        target = self.arm.read_pose()
        target[0, 3] += 10.
        self.arm.move_linear(target, 20., 10.)
        self.assertEqual(self.publisher.messages, [])
        self.advance()
        message = self.publisher.messages[-1]
        self.assertEqual(message.pose_selection, 2)
        self.assertEqual(message.left_arm_goal_pose.pose.position.x, -.1)
        self.assertEqual(message.left_arm_goal_pose.pose.position.y, .2)
        self.assertEqual(message.left_arm_goal_pose.pose.orientation.w, 1.)
        self.assertEqual(message.linear_velocity, 0.)
        self.assertEqual(message.angular_velocity, 0.)
        self.assertEqual(message.right_arm_goal_pose.header.frame_id, "base_link")
        self.assertAlmostEqual(message.right_arm_goal_pose.pose.position.z, .8)

    def test_selected_left_is_the_only_mask(self):
        self.arm.arm_side = "left"
        target = self.arm.read_pose()
        target[1, 3] += 3.
        self.arm.move_linear(target, 5., 5.)
        self.advance()
        self.assertEqual(self.publisher.messages[-1].pose_selection, 1)
        self.assertEqual(self.publisher.messages[-1].right_arm_goal_pose.pose.position.x, .1)

    def test_command_path_step_and_speed_limits_survive_long_scheduler_gap(self):
        start = self.arm.read_pose()
        target = start.copy()
        target[:3, 3] += [3., 4., 0.]
        target[:3, :3] = rotation([0, 0, 1], 90.)
        self.arm.move_linear(target, 5., 20.)
        previous = start.copy()
        for elapsed in [.02, .02, 4., .02, .5]:
            self.advance(elapsed=elapsed)
            current = self.arm._command_tcp.copy()
            distance, angle = pose_error(previous, current)
            self.assertLessEqual(distance, .15 + 1e-9)
            self.assertLessEqual(distance, 5. * min(elapsed, .02) + 1e-9)
            self.assertLessEqual(angle, .25 + 1e-9)
            self.assertLessEqual(angle, 20. * min(elapsed, .02) + 1e-9)
            self.assertAlmostEqual((current[0, 3]-start[0, 3]) / 3., (current[1, 3]-start[1, 3]) / 4.)
            previous = current

    def test_completion_requires_measured_feedback(self):
        target = self.arm.read_pose()
        target[0, 3] += .1
        self.arm.move_linear(target, 5., 5.)
        self.advance(3)
        self.assertTrue(self.arm.motion_done())  # within configured 1 mm tolerance
        publications = len(self.publisher.messages)
        self.advance(5)  # Idle worker ticks must not restart a completed command.
        self.assertEqual(len(self.publisher.messages), publications)
        self.arm.pose_tolerance_mm = .01
        self.arm.move_linear(target, 5., 5.)
        self.advance(3)
        np.testing.assert_allclose(self.arm._command_tcp, target)
        self.assertEqual(self.arm._motion["fraction"], 1.)
        self.assertFalse(self.arm.motion_done())
        self.ros.message.right_arm_state_pose.pose.position.x = target[0, 3] / 1000.
        self.advance()
        self.assertTrue(self.arm.motion_done())

    def test_published_final_orientation_still_requires_measured_rotation(self):
        target = self.arm.read_pose()
        target[:3, :3] = rotation([0, 0, 1], 2.)
        self.arm.move_linear(target, 5., 5.)
        self.advance(25)
        np.testing.assert_allclose(self.arm._command_tcp, target, atol=1e-12)
        self.assertEqual(self.arm._motion["fraction"], 1.)
        self.assertFalse(self.arm.motion_done())  # measured error 2 deg > tolerance 1 deg
        radians = np.deg2rad(2.)
        self.ros.message.right_arm_state_pose.pose.orientation = NS(
            x=0., y=0., z=float(np.sin(radians/2)), w=float(np.cos(radians/2)))
        self.advance()
        self.assertTrue(self.arm.motion_done())

    def test_stop_cancels_interpolation_before_publishing_measured_hold(self):
        start = self.arm.read_pose()
        target = start.copy()
        target[0, 3] += 30.
        self.arm.move_linear(target, 20., 20.)
        self.advance(2)
        self.arm.stop_arm()
        count = len(self.publisher.messages)
        np.testing.assert_allclose(self.arm._command_tcp, start)
        self.assertEqual(self.arm.snapshot()["control_mode"], "software_hold")
        for _ in range(10):
            self.arm._advance_locked(time.monotonic() + 100.)
        self.assertEqual(len(self.publisher.messages), count)
        self.assertTrue(self.arm.motion_done())

    def test_stale_feedback_cancels_and_does_not_publish_old_pose_as_hold(self):
        target = self.arm.read_pose()
        target[0, 3] += 30.
        self.arm.move_linear(target, 20., 20.)
        self.advance()
        self.ros.message.right_arm_state_pose.header.stamp = Stamp(95.)
        count = len(self.publisher.messages)
        with self.assertRaises(ArmObservationError) as error:
            self.advance()
        self.arm._abort_locked(error.exception)
        self.assertIsNone(self.arm._motion)
        self.assertEqual(len(self.publisher.messages), count)
        self.assertIn("hold unavailable", self.arm.fault)
        with self.assertRaises(ArmObservationError):
            self.arm.motion_done()

    def test_close_cancels_without_hold_publication(self):
        target = self.arm.read_pose()
        target[0, 3] += 30.
        self.arm.move_linear(target, 20., 20.)
        self.advance()
        count = len(self.publisher.messages)
        self.arm.close()
        self.assertIsNone(self.arm._motion)
        self.assertEqual(len(self.publisher.messages), count)

    def test_stop_failure_still_discards_old_motion(self):
        target = self.arm.read_pose()
        target[0, 3] += 30.
        self.arm.move_linear(target, 20., 20.)
        self.tf.failure = True
        with self.assertRaises(ArmObservationError):
            self.arm.stop_arm()
        self.assertIsNone(self.arm._motion)
        self.assertEqual(self.publisher.messages, [])

    def test_invalid_speed_and_overlapping_command_refused(self):
        target = self.arm.read_pose()
        for speed in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                self.arm.move_linear(target, speed, 5.)
        self.arm.move_linear(target, 5., 5.)
        with self.assertRaises(ArmObservationError):
            self.arm.move_linear(target, 5., 5.)

    def test_running_worker_cannot_publish_old_targets_after_stop_returns(self):
        self.arm.close()
        with patch("bottle_scan.hardware_arm.threading.Thread", REAL_THREAD):
            self.arm = RosArmController(publish_hz=100., max_age_s=1.)
            self.addCleanup(self.arm.close)
            self.arm.connect()
            target = self.arm.read_pose()
            target[0, 3] += 10.
            self.arm.move_linear(target, 5., 5.)
            deadline = time.monotonic() + .3
            while len(self.publisher.messages) < 2 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertGreaterEqual(len(self.publisher.messages), 2)
            self.arm.stop_arm()
            count = len(self.publisher.messages)
            time.sleep(.04)
            self.assertEqual(len(self.publisher.messages), count)
            self.arm.close()

    def test_native_source_events_once_and_command_events_only_on_publish(self):
        self.assertEqual(self.arm.drain_samples(), [])
        for seq in (12, 13, 14):
            self.ros.message.right_arm_state_pose.header.seq = seq
            self.arm._on_state(self.ros.message)
        events = self.arm.drain_samples()
        self.assertEqual([e.stream for e in events], ["tcp_measured"] * 3)
        self.assertEqual([e.metadata["source_sequence"] for e in events], [12, 13, 14])
        self.assertEqual([e.device_timestamp for e in events], [100.] * 3)
        self.assertEqual(self.arm.drain_samples(), [])
        target = self.arm.read_pose()
        target[0, 3] += 10.
        self.arm.move_linear(target, 5., 5.)
        self.assertEqual(self.arm.drain_samples(), [])
        self.advance(2)
        commands = self.arm.drain_samples()
        self.assertEqual([e.stream for e in commands], ["tcp_command"] * 2)
        self.assertEqual(len(commands), len(self.publisher.messages))

    def test_live_fixed_zero_sequence_is_preserved_not_fabricated(self):
        self.arm._on_state(self.ros.message)
        self.arm._on_state(self.ros.message)
        events = self.arm.drain_samples()
        self.assertEqual([e.metadata["source_sequence"] for e in events], [0, 0])
        self.assertTrue(all(not e.metadata["source_sequence_tracking_available"] for e in events))
        self.assertIsNone(self.arm.fault)

    def test_source_gap_fault_does_not_initiate_motion_from_idle_connection(self):
        for seq in (12, 14):
            self.ros.message.right_arm_state_pose.header.seq = seq
            self.arm._on_state(self.ros.message)
        events = self.arm.drain_samples()
        self.assertTrue(events[-1].metadata["source_sequence_gap"])
        self.assertIn("discontinuity", self.arm.fault)
        self.assertEqual(self.publisher.messages, [])
        with self.assertRaisesRegex(ArmObservationError, "latched"):
            self.arm.move_linear(self.arm.read_pose(), 5., 5.)

    def test_source_queue_overflow_fault_preserves_first_event_and_never_moves_idle_arm(self):
        self.arm.sample_queue_size = 1
        self.arm._on_state(self.ros.message)
        self.arm._on_state(self.ros.message)
        self.assertEqual(len(self.arm.drain_samples()), 1)
        self.assertIn("overflow", self.arm.fault)
        self.assertEqual(self.publisher.messages, [])

    def test_missing_source_tool_tf_is_fault_not_a_fake_measured_tcp(self):
        self.tf.failure = True
        self.arm._on_state(self.ros.message)
        self.assertEqual(self.arm.drain_samples(), [])
        self.assertIn("source acquisition fault", self.arm.fault)
        self.assertEqual(self.publisher.messages, [])


if __name__ == "__main__":
    unittest.main()
