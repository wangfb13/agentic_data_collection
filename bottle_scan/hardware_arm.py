"""Read-only ROS1 arm observation, with no control objects or motion fallback.

Imports do not import ROS, connect, publish, call services, or initialize a robot.
``connect`` subscribes only; ROS's own /rosout diagnostic publisher is harmless.
TCP means the live URDF/TF ``{side}_gripper_base_link`` origin, NOT an assumed
fingertip/grasp centre. The experiment geometry must use this same TCP definition.

The running inference adapter documents /wbc_state_pose as base_link -> Link7;
we compose it with the live Link7 -> gripper_base TF, retaining both source
transforms. No old hard-coded tool calibration or model-reference frame is used.
``RosArmObserver`` has no command publisher. ``RosArmController`` adds the
user-requested public ROS pose stream, software interpolation and measured-pose
hold. These do not establish a hardware stop, watchdog, or executed-path guarantee.
"""
from __future__ import annotations

import math
from collections import deque
import threading
import time

import numpy as np

from .geometry import interface_units, validate_transform, pose_error, _rotation_vector, rotation


class ArmObservationError(RuntimeError):
    """Missing, malformed, or stale arm feedback; no fallback pose is returned."""


class UnverifiedMotionError(RuntimeError):
    """A control operation has no verified implementation."""


def _ros_dependencies():
    # Deferred to explicit connect(), allowing offline CLI/config imports.
    import rospy
    import tf2_ros
    from std_msgs.msg import Bool, String
    from wbc_msgs.msg import wbc_state
    return rospy, tf2_ros, wbc_state, Bool, String


def _pose_matrix_mm(position, quaternion):
    """Convert ROS xyz metres and quaternion xyzw through the single unit seam."""
    xyz = np.asarray(position, dtype=float)
    q = np.asarray(quaternion, dtype=float)
    if xyz.shape != (3,) or q.shape != (4,) or not np.all(np.isfinite(np.r_[xyz, q])):
        raise ArmObservationError("Nonfinite or malformed ROS pose")
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.) > .01:
        raise ArmObservationError("ROS quaternion is not unit length")
    x, y, z, w = q / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = xyz
    return validate_transform(interface_units(matrix, "m_to_mm"))


def _xyz(value):
    return [float(value.x), float(value.y), float(value.z)]


def _xyzw(value):
    return [float(value.x), float(value.y), float(value.z), float(value.w)]


class RosArmObserver:
    """Observe both arms while selecting only the requested arm for read_pose.

    This is not a backend satisfying the experiment's motion capabilities.
    Run inside an environment with ROS Noetic and the wbc_msgs workspace sourced.
    ROS_MASTER_URI/ROS_IP must be set by the launcher, not inferred or mutated here.
    ``snapshot`` returns JSON-serializable data; ``read_pose`` returns a numpy
    4x4 transform in millimetres. Both reject stale device and receipt timestamps.
    """

    def __init__(self, arm_side="right", base_frame="base_link",
                 state_topic="/wbc_state_pose", max_age_s=.25,
                 connect_timeout_s=5., flange_frames=None, tcp_frames=None,
                 state_queue_size=1):
        if arm_side not in ("left", "right"):
            raise ValueError("arm_side must be left or right")
        for name, value in (("max_age_s", max_age_s), ("connect_timeout_s", connect_timeout_s)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(name + " must be positive and finite")
        self.arm_side = arm_side
        self.base_frame = str(base_frame).lstrip("/")
        self.state_topic = str(state_topic)
        if not self.base_frame or not self.state_topic:
            raise ValueError("base frame and state topic must be nonempty")
        self.max_age_s = float(max_age_s)
        self.connect_timeout_s = float(connect_timeout_s)
        if not isinstance(state_queue_size, int) or isinstance(state_queue_size, bool) or state_queue_size < 1:
            raise ValueError("state_queue_size must be a positive integer")
        self.state_queue_size = state_queue_size
        self.flange_frames = flange_frames or {s: "arm_" + s + "_7_link" for s in ("left", "right")}
        self.tcp_frames = tcp_frames or {s: s + "_gripper_base_link" for s in ("left", "right")}
        for frames in (self.flange_frames, self.tcp_frames):
            if set(frames) != {"left", "right"} or not all(isinstance(f, str) and f for f in frames.values()):
                raise ValueError("Each frame map must contain nonempty left/right frames")
        self._lock = threading.Lock()
        self._state = None
        self._status = {}
        self._subscriptions = []
        self._rospy = self._buffer = self._listener = None
        self._connected = False

    def _on_state(self, message):
        with self._lock:
            self._state = (message, time.monotonic_ns())

    def _on_status(self, name, message):
        with self._lock:
            self._status[name] = (message.data, time.monotonic_ns())

    def connect(self):
        """Register subscriptions, wait for fresh measured poses; issue no actions."""
        if self._connected:
            return self
        if self._subscriptions:
            raise ArmObservationError("Observer already connecting")
        rospy, tf2_ros, state_type, bool_type, string_type = _ros_dependencies()
        self._rospy = rospy
        try:
            if not rospy.core.is_initialized():
                rospy.init_node("bottle_arm_observer", anonymous=True, disable_signals=True)
            self._buffer = tf2_ros.Buffer()
            self._listener = tf2_ros.TransformListener(self._buffer)
            self._subscriptions.append(rospy.Subscriber(self.state_topic, state_type, self._on_state,
                                                       queue_size=self.state_queue_size))
            self._subscriptions.append(rospy.Subscriber("/fsm_state", string_type,
                lambda message: self._on_status("fsm_state", message), queue_size=1))
            self._subscriptions.append(rospy.Subscriber("/motion_active", bool_type,
                lambda message: self._on_status("motion_active", message), queue_size=1))
            deadline = time.monotonic() + self.connect_timeout_s
            last_error = "No state received"
            while time.monotonic() < deadline and not rospy.is_shutdown():
                try:
                    self.snapshot()
                    self._connected = True
                    return self
                except ArmObservationError as exc:
                    last_error = str(exc)
                    time.sleep(.01)
            raise ArmObservationError("Read-only arm connection timed out: " + last_error)
        except BaseException:
            self.close()
            raise

    def _fresh_age(self, stamp, now_ros, description, *, static_allowed=False):
        stamp = float(stamp)
        if not math.isfinite(stamp) or (stamp <= 0 and not (static_allowed and stamp == 0)):
            raise ArmObservationError(description + " has invalid/zero timestamp")
        if static_allowed and stamp == 0:
            return None
        age = now_ros - stamp
        if not -self.max_age_s <= age <= self.max_age_s:
            raise ArmObservationError(description + " timestamp is stale or clocks disagree: %.6f s" % age)
        return age

    def snapshot(self):
        """Read both TCPs with original flange and live TF provenance; no motion."""
        if self._rospy is None or self._buffer is None:
            raise ArmObservationError("Call connect() before reading arm feedback")
        with self._lock:
            state = self._state
            status = dict(self._status)
        if state is None:
            raise ArmObservationError("No WBC state received")
        message, received_ns = state
        now_ns = time.monotonic_ns()
        receive_age = (now_ns - received_ns) / 1e9
        if not 0 <= receive_age <= self.max_age_s:
            raise ArmObservationError("WBC state receipt is stale")
        now_ros = float(self._rospy.Time.now().to_sec())
        if not math.isfinite(now_ros) or now_ros <= 0:
            raise ArmObservationError("ROS clock unavailable")
        arms = {}
        for side in ("left", "right"):
            stamped = getattr(message, side + "_arm_state_pose")
            if stamped.header.frame_id.lstrip("/") != self.base_frame:
                raise ArmObservationError(side + " WBC frame differs from configured base frame")
            source_stamp = float(stamped.header.stamp.to_sec())
            source_age = self._fresh_age(source_stamp, now_ros, side + " WBC")
            pose = stamped.pose
            xyz, quat = _xyz(pose.position), _xyzw(pose.orientation)
            flange = _pose_matrix_mm(xyz, quat)
            try:
                # Use the measurement stamp, never a newer dynamic transform.
                tool = self._buffer.lookup_transform(self.flange_frames[side], self.tcp_frames[side],
                                                     stamped.header.stamp)
            except Exception as exc:
                raise ArmObservationError(side + " flange->TCP TF unavailable: " + str(exc)) from exc
            if (tool.header.frame_id.lstrip("/") != self.flange_frames[side].lstrip("/")
                    or tool.child_frame_id.lstrip("/") != self.tcp_frames[side].lstrip("/")):
                raise ArmObservationError(side + " TF returned unexpected frames")
            tf_stamp = float(tool.header.stamp.to_sec())
            self._fresh_age(tf_stamp, now_ros, side + " tool TF", static_allowed=True)
            tool_mm = _pose_matrix_mm(_xyz(tool.transform.translation), _xyzw(tool.transform.rotation))
            tcp_mm = validate_transform(flange @ tool_mm)
            arms[side] = {
                "base_frame": self.base_frame, "flange_frame": self.flange_frames[side],
                "tcp_frame": self.tcp_frames[side],
                "T_base_flange_mm": flange.tolist(), "T_flange_tcp_mm": tool_mm.tolist(),
                "T_base_tcp_mm": tcp_mm.tolist(), "raw_flange_xyz_m_xyzw": xyz + quat,
                "source_topic": self.state_topic, "source_ros_timestamp_s": source_stamp,
                "source_age_s": source_age, "received_monotonic_ns": received_ns,
                "tool_transform_source": "live_tf_at_wbc_measurement_stamp",
                "tool_tf_ros_timestamp_s": tf_stamp,
            }
        values = {}
        for name in ("fsm_state", "motion_active"):
            value, received = status.get(name, (None, None))
            age = None if received is None else (now_ns - received) / 1e9
            values[name] = {"value": value, "received_monotonic_ns": received,
                            "receipt_age_s": age, "fresh": age is not None and 0 <= age <= self.max_age_s}
        return {"selected_arm": self.arm_side, "base_frame": self.base_frame,
                "translation_unit": "mm", "quaternion_order": "xyzw",
                "observed_monotonic_ns": now_ns, "state_receipt_age_s": receive_age,
                "arms": arms, "status": values, "read_only": True,
                "motion_capabilities_verified": False,
                "tcp_definition": "live gripper_base_link origin; not a calibrated fingertip centre"}

    def read_pose(self, side=None):
        side = self.arm_side if side is None else side
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        return np.asarray(self.snapshot()["arms"][side]["T_base_tcp_mm"], dtype=float)

    def status(self):
        return self.snapshot()["status"]

    def move_linear(self, *args, **kwargs):
        raise UnverifiedMotionError("Cartesian straight-line speed control is not verified; no command sent")

    def stop_arm(self, *args, **kwargs):
        raise UnverifiedMotionError("Controller stop semantics are not verified; no service called")

    def close(self):
        """Release observation subscriptions only. This is NOT a motor stop."""
        for subscription in self._subscriptions:
            subscription.unregister()
        self._subscriptions.clear()
        if self._listener is not None:
            self._listener.unregister()
        self._listener = self._buffer = self._rospy = None
        self._connected = False
        with self._lock:
            self._state = None
            self._status.clear()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exception):
        self.close()


def _command_message_type():
    from wbc_msgs.msg import wbc_pose_expand
    return wbc_pose_expand


def _matrix_quaternion_xyzw(matrix):
    """Convert a validated rotation without importing control/ROS libraries."""
    r = matrix[:3, :3]
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.) * 2
        q = np.array([(r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s,
                      (r[1, 0]-r[0, 1])/s, .25*s])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1. + r[i, i] - r[j, j] - r[k, k]) * 2
        q = np.zeros(4)
        q[i] = .25*s
        q[j] = (r[j, i] + r[i, j])/s
        q[k] = (r[k, i] + r[i, k])/s
        q[3] = (r[k, j] - r[j, k])/s
    return q / np.linalg.norm(q)


class RosArmController(RosArmObserver):
    """Single-arm public WBC pose streaming with explicit SOFTWARE semantics.

    Targets interpolate linearly in base-frame xyz and by shortest rotation.
    Speed/step limits apply to the published targets, not a guarantee about the
    low-level robot's actual trajectory. stop_arm cancels queued interpolation
    and publishes the latest measured pose once; it is not an emergency stop.
    No stop service, controller settings, initial-pose action, or gripper command
    is issued. Connection and closing never publish a pose.
    """

    def __init__(self, *args, command_topic="/wbc_goal_pose", publish_hz=50.,
                 max_step_mm=1., max_step_deg=1., pose_tolerance_mm=1.,
                 pose_tolerance_deg=1., state_queue_size=100, sample_queue_size=10000, **kwargs):
        super().__init__(*args, state_queue_size=state_queue_size, **kwargs)
        for name, value in (("publish_hz", publish_hz), ("max_step_mm", max_step_mm),
                            ("max_step_deg", max_step_deg), ("pose_tolerance_mm", pose_tolerance_mm),
                            ("pose_tolerance_deg", pose_tolerance_deg)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(name + " must be positive and finite")
            setattr(self, name, float(value))
        if not isinstance(command_topic, str) or not command_topic:
            raise ValueError("command_topic must be nonempty")
        self.command_topic = command_topic
        self._control_lock = threading.RLock()
        self._worker_stop = threading.Event()
        self._worker = None
        self._publisher = None
        self._message_type = None
        self._motion = None
        self._command_tcp = None
        self._command_ns = None
        self._control_mode = "idle_no_command"
        self._fault = None
        if not isinstance(sample_queue_size, int) or isinstance(sample_queue_size, bool) or sample_queue_size < 1:
            raise ValueError("sample_queue_size must be a positive integer")
        self.sample_queue_size = sample_queue_size
        self._sample_lock = threading.Lock()
        self._pending_samples = deque()
        self._Sample = None
        self._recording_active = False
        self._previous_source_sequence = None
        self._data_fault_latched = False
        self._loop_periods = deque(maxlen=4096)
        self._loop_period_count = 0
        self._loop_metrics_generation = 0
        self._loop_metrics_started_ns = time.monotonic_ns()

    def connect(self):
        if self._publisher is not None:
            return self
        try:
            super().connect()
            from .backend import Sample
            self._Sample = Sample
            self._message_type = _command_message_type()
            self._publisher = self._rospy.Publisher(self.command_topic, self._message_type,
                                                   queue_size=1, latch=False)
            self._worker_stop.clear()
            self._recording_active = True
            self._worker = threading.Thread(target=self._worker_loop, name="bottle_arm_pose_stream", daemon=True)
            self._worker.start()
            return self
        except BaseException:
            self.close()
            raise

    def _queue_sample(self, sample):
        with self._sample_lock:
            if len(self._pending_samples) >= self.sample_queue_size:
                raise ArmObservationError("Arm source event queue overflow; acquisition is incomplete")
            self._pending_samples.append(sample)

    def _on_state(self, message):
        received_ns = time.monotonic_ns()
        with self._lock:
            self._state = (message, received_ns)
        if not self._recording_active:
            return  # Bootstrap ends only after TF and both measured poses are ready.
        try:
            side = self.arm_side
            stamped = getattr(message, side + "_arm_state_pose")
            if stamped.header.frame_id.lstrip("/") != self.base_frame:
                raise ArmObservationError("Measured pose source frame changed")
            source_stamp = float(stamped.header.stamp.to_sec())
            self._fresh_age(source_stamp, float(self._rospy.Time.now().to_sec()), "source-event WBC")
            tool = self._buffer.lookup_transform(self.flange_frames[side], self.tcp_frames[side], stamped.header.stamp)
            if (tool.header.frame_id.lstrip("/") != self.flange_frames[side].lstrip("/")
                    or tool.child_frame_id.lstrip("/") != self.tcp_frames[side].lstrip("/")):
                raise ArmObservationError("Source-event tool TF has unexpected frames")
            self._fresh_age(float(tool.header.stamp.to_sec()), float(self._rospy.Time.now().to_sec()),
                            "source-event tool TF", static_allowed=True)
            flange = _pose_matrix_mm(_xyz(stamped.pose.position), _xyzw(stamped.pose.orientation))
            tcp = validate_transform(flange @ _pose_matrix_mm(_xyz(tool.transform.translation), _xyzw(tool.transform.rotation)))
            sequence = int(stamped.header.seq)
            previous = self._previous_source_sequence
            # Live WBC headers have been observed with seq==0 on every message.
            # Preserve zero verbatim; do not invent a monotonic source counter.
            sequence_available = sequence != 0 or (previous is not None and previous != 0)
            delta = None if previous is None else (sequence - previous) % (2**32)
            gap = sequence_available and delta is not None and delta != 1
            self._previous_source_sequence = sequence
            metadata = {"arm": side, "source_topic": self.state_topic, "source_sequence": sequence,
                        "source_sequence_tracking_available": sequence_available,
                        "source_sequence_gap": bool(gap), "source_sequence_delta": delta,
                        "tcp_frame": self.tcp_frames[side], "base_frame": self.base_frame,
                        "tool_transform_source": "live_tf_at_wbc_measurement_stamp"}
            self._queue_sample(self._Sample("tcp_measured", tcp.copy(), received_ns,
                                            device_timestamp=source_stamp, metadata=metadata))
            if gap:
                raise ArmObservationError("WBC source sequence discontinuity: %s -> %s" % (previous, sequence))
        except Exception as exc:
            with self._control_lock:
                if not self._data_fault_latched:
                    self._data_fault_latched = True
                    self._abort_locked(ArmObservationError("Arm source acquisition fault: " + str(exc)))

    def drain_samples(self):
        """Return each native measurement/publication event once, never resample latest."""
        with self._sample_lock:
            samples = list(self._pending_samples)
            self._pending_samples.clear()
        return samples

    def snapshot(self):
        # Observer.connect calls this before the publisher/thread exists.
        result = super().snapshot()
        with self._control_lock:
            result.update({"read_only": False, "control_semantics": "software_pose_stream_and_measured_pose_hold",
                           "command_tcp_mm": None if self._command_tcp is None else self._command_tcp.tolist(),
                           "command_timestamp_ns": self._command_ns, "control_mode": self._control_mode,
                           "fault": self._fault, "motion_capabilities_verified": False,
                           "loop_period_s": self.loop_metrics()})
        return result

    def loop_metrics(self):
        with self._control_lock:
            values=np.asarray(self._loop_periods,dtype=float)
            return {"count":self._loop_period_count,"window_count":len(values),"window":"latest_at_most_4096",
                    "measurement_started_monotonic_ns":self._loop_metrics_started_ns,
                    "mean_s":float(values.mean()) if len(values) else None,
                    "p95_s":float(np.percentile(values,95)) if len(values) else None,
                    "max_s":float(values.max()) if len(values) else None,
                    "source":"wall_monotonic_active_arm_interpolation_loop"}

    def reset_loop_metrics(self):
        """Start a per-acquisition timing interval without publishing commands."""
        with self._control_lock:
            self._loop_periods.clear()
            self._loop_period_count = 0
            self._loop_metrics_generation += 1
            self._loop_metrics_started_ns = time.monotonic_ns()

    @property
    def fault(self):
        with self._control_lock:
            return self._fault

    def _fill_pose(self, stamped, pose_mm, stamp):
        pose_m = interface_units(validate_transform(pose_mm), "mm_to_m")
        stamped.header.frame_id = self.base_frame
        stamped.header.stamp = stamp
        stamped.pose.position.x, stamped.pose.position.y, stamped.pose.position.z = map(float, pose_m[:3, 3])
        q = _matrix_quaternion_xyzw(pose_m)
        (stamped.pose.orientation.x, stamped.pose.orientation.y,
         stamped.pose.orientation.z, stamped.pose.orientation.w) = map(float, q)

    def _publish_locked(self, tcp_mm, observation, mode):
        if self._publisher is None:
            raise ArmObservationError("Controller is not connected")
        if self._publisher.get_num_connections() < 1:
            raise ArmObservationError("No subscriber on public WBC command topic")
        message = self._message_type()
        stamp = self._rospy.Time.now()
        for side in ("left", "right"):
            arm = observation["arms"][side]
            flange = np.asarray(arm["T_base_flange_mm"], dtype=float)
            if side == self.arm_side:
                flange = validate_transform(tcp_mm @ np.linalg.inv(np.asarray(arm["T_flange_tcp_mm"], dtype=float)))
            # Unselected arm field contains measured feedback, never a default
            # pose; pose_selection below explicitly excludes that field.
            self._fill_pose(getattr(message, side + "_arm_goal_pose"), flange, stamp)
        message.pose_selection = 0x01 if self.arm_side == "left" else 0x02
        # These are chassis velocities, not arm Cartesian speed parameters.
        message.linear_velocity = message.angular_velocity = 0.
        self._publisher.publish(message)
        self._command_tcp = np.array(tcp_mm, dtype=float).copy()
        self._command_ns = time.monotonic_ns()
        self._control_mode = mode
        try:
            self._queue_sample(self._Sample("tcp_command", self._command_tcp.copy(), self._command_ns,
                device_timestamp=float(stamp.to_sec()), metadata={"arm": self.arm_side,
                "command_topic": self.command_topic, "pose_selection": message.pose_selection,
                "control_semantics": mode, "timestamp_semantics": "host publication; not robot receipt"}))
        except Exception:
            self._data_fault_latched = True
            raise

    def move_linear(self, pose_mm, speed_mm_s, angular_speed_deg_s):
        target = validate_transform(pose_mm).copy()
        speed, angular_speed = float(speed_mm_s), float(angular_speed_deg_s)
        if not all(math.isfinite(v) and v > 0 for v in (speed, angular_speed)):
            raise ValueError("Commanded linear and angular speeds must be positive and finite")
        with self._control_lock:
            if self._publisher is None:
                raise ArmObservationError("Controller is not connected")
            if self._motion is not None:
                raise ArmObservationError("Previous pose interpolation is still active")
            if self._data_fault_latched:
                raise ArmObservationError("Arm acquisition fault is latched; reconnect in a new recording attempt")
            observation = super().snapshot()
            start = np.asarray(observation["arms"][self.arm_side]["T_base_tcp_mm"], dtype=float)
            axis, angle_rad = _rotation_vector(start[:3, :3].T @ target[:3, :3])
            distance = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
            angle_deg = math.degrees(angle_rad)
            duration = max(distance / speed, angle_deg / angular_speed, 1. / self.publish_hz)
            step_fraction = min(1., self.max_step_mm / distance if distance > 0 else 1.,
                                self.max_step_deg / angle_deg if angle_deg > 0 else 1.)
            self._motion = {"start": start.copy(), "target": target, "axis": axis,
                            "angle_deg": angle_deg, "duration": duration, "fraction": 0.,
                            "step_fraction": step_fraction, "last_monotonic": time.monotonic()}
            self._fault = None
            self._control_mode = "software_interpolating"

    def _advance_locked(self, now):
        motion = self._motion
        if motion is None:
            return
        observation = super().snapshot()  # Mandatory freshness check before every publication.
        measured = np.asarray(observation["arms"][self.arm_side]["T_base_tcp_mm"], dtype=float)
        if motion["fraction"] >= 1.:
            translation_error, angular_error = pose_error(motion["target"], measured)
            if translation_error <= self.pose_tolerance_mm and angular_error <= self.pose_tolerance_deg:
                self._motion = None
                self._control_mode = "target_reached_by_feedback"
                return
        elapsed = max(0., min(now - motion["last_monotonic"], 1. / self.publish_hz))
        fraction = min(1., motion["fraction"] + min(elapsed / motion["duration"], motion["step_fraction"]))
        pose = motion["start"].copy()
        pose[:3, 3] += fraction * (motion["target"][:3, 3] - motion["start"][:3, 3])
        pose[:3, :3] = motion["start"][:3, :3] @ rotation(motion["axis"], motion["angle_deg"] * fraction)
        self._publish_locked(pose, observation, "software_interpolating")
        motion["fraction"] = fraction
        motion["last_monotonic"] = now

    def _abort_locked(self, error):
        should_hold = self._motion is not None or self._command_tcp is not None
        self._motion = None  # Cancel before attempting any observation or hold.
        self._fault = str(error)
        self._control_mode = "cancelled_hold_unavailable"
        if not should_hold:
            self._control_mode = "idle_acquisition_fault"
            return  # A connection/observation fault must never initiate an action.
        try:
            observation = super().snapshot()
            measured = np.asarray(observation["arms"][self.arm_side]["T_base_tcp_mm"], dtype=float)
            self._publish_locked(measured, observation, "software_hold_after_fault")
        except Exception as hold_error:
            self._fault += "; fresh-pose software hold unavailable: " + str(hold_error)

    def _worker_loop(self):
        previous=None
        generation=self._loop_metrics_generation
        while not self._worker_stop.wait(1. / self.publish_hz):
            with self._control_lock:
                try:
                    now=time.monotonic()
                    if generation != self._loop_metrics_generation:
                        previous=None
                        generation=self._loop_metrics_generation
                    if self._motion is not None:
                        if previous is not None:
                            self._loop_periods.append(now-previous)
                            self._loop_period_count+=1
                        previous=now
                    else:previous=None
                    self._advance_locked(now)
                except Exception as exc:
                    self._abort_locked(exc)

    def motion_done(self):
        with self._control_lock:
            if self._fault:
                raise ArmObservationError(self._fault)
            return self._motion is None

    def stop_arm(self):
        """Cancel old targets and publish fresh measured pose once (software hold)."""
        with self._control_lock:
            self._motion = None
            self._control_mode = "cancelled_hold_unavailable"
            try:
                observation = super().snapshot()
                measured = np.asarray(observation["arms"][self.arm_side]["T_base_tcp_mm"], dtype=float)
                self._publish_locked(measured, observation, "software_hold")
            except Exception as exc:
                self._fault = "Pose interpolation cancelled; software hold unavailable: " + str(exc)
                raise ArmObservationError(self._fault) from exc

    def close(self):
        # Stop the software worker without issuing any implicit pose command.
        self._recording_active = False
        with self._control_lock:
            self._motion = None
            self._worker_stop.set()
            worker, self._worker = self._worker, None
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=max(1., 2. / self.publish_hz))
            if worker.is_alive():
                raise ArmObservationError("Pose-stream thread did not terminate")
        with self._control_lock:
            if self._publisher is not None:
                self._publisher.unregister()
            self._publisher = self._message_type = None
        super().close()
