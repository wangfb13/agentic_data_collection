"""ROS gripper observation, measured calibration, and explicit software control.

Importing this module has no ROS/device side effects. The read-only observer
never publishes. The separately selected controller only publishes after an
explicit command and does not imply native force mode or hardware protection.
"""
from __future__ import annotations

from bisect import bisect_right
import math
import numbers
import threading
import time
from dataclasses import dataclass
from collections import deque


class WidthCalibration:
    """Measured increasing normalized opening -> physical width in millimetres.

    ``points`` contains at least two ``(normalized, width_mm)`` pairs, or
    dictionaries with those keys, in increasing order. Outside the declared
    domain conversion always fails. An optional, explicitly tagged first knot
    declares a bounded equivalent gel-compression model, not a measured gap.
    This numerical mapping does not certify motor limits or encoder resolution.
    """

    def __init__(self, points):
        points=tuple(points)
        try:
            pairs = tuple((float(p["normalized"]), float(p["width_mm"]))
                          if isinstance(p, dict) else (float(p[0]), float(p[1]))
                          for p in points)
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise ValueError("Calibration requires normalized/width_mm pairs") from exc
        if len(pairs) < 2:
            raise ValueError("At least two measured width calibration points are required")
        if any(not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or y < 0
               for x, y in pairs):
            raise ValueError("Calibration must have finite normalized [0,1] and nonnegative mm")
        if any(x1 <= x0 or y1 <= y0 for (x0, y0), (x1, y1) in zip(pairs, pairs[1:])):
            raise ValueError("Both normalized opening and width_mm must strictly increase")
        modelled=[i for i,p in enumerate(points) if isinstance(p,dict) and p.get('kind')=='compression_equivalent_estimate']
        self.compression_model=None
        if modelled:
            if modelled!=[0] or len(points)<3:raise ValueError('Compression estimate must precede at least two measured anchors')
            model=dict(points[0]);n,w=pairs[0];contact_n,contact_w=pairs[1]
            required=('contact_normalized','contact_width_mm','max_compression_mm','normalized_hard_min')
            if any(k not in model or not math.isfinite(float(model[k])) for k in required):raise ValueError('Incomplete bounded compression model')
            if (not math.isclose(float(model['contact_normalized']),contact_n,abs_tol=1e-9)
                    or not math.isclose(float(model['contact_width_mm']),contact_w,abs_tol=1e-9)
                    or not 0<float(model['max_compression_mm'])<contact_w
                    or contact_w-w>float(model['max_compression_mm'])+1e-8
                    or not 0<=float(model['normalized_hard_min'])<=n):
                raise ValueError('Compression model exceeds declared contact/travel window')
            slope=(pairs[-1][1]-contact_w)/(pairs[-1][0]-contact_n)
            if not math.isclose(w,contact_w+(n-contact_n)*slope,abs_tol=1e-7):raise ValueError('Compression knot differs from measured-anchor slope')
            self.compression_model=model
        self._points = pairs
        self._normalized = tuple(p[0] for p in pairs)
        self._widths = tuple(p[1] for p in pairs)

    @property
    def points(self):
        result=[{"normalized": x, "width_mm": y} for x,y in self._points]
        if self.compression_model:result[0].update(self.compression_model)
        return tuple(result)

    def width_kind(self, normalized):
        self.to_mm(normalized)  # preserve strict domain checks
        if self.compression_model and float(normalized)<self.compression_model['contact_normalized']-1e-10:
            return 'estimated_equivalent_compression'
        return 'measured_anchor_interpolation'

    @staticmethod
    def _interpolate(value, xs, ys):
        value = float(value)
        if not math.isfinite(value) or not xs[0] <= value <= xs[-1]:
            raise ValueError(f"Value outside declared calibration/model domain [{xs[0]}, {xs[-1]}]")
        index = min(bisect_right(xs, value) - 1, len(xs) - 2)
        ratio = (value - xs[index]) / (xs[index + 1] - xs[index])
        return ys[index] + ratio * (ys[index + 1] - ys[index])

    def to_mm(self, normalized):
        return self._interpolate(normalized, self._normalized, self._widths)

    def to_normalized(self, width_mm):
        return self._interpolate(width_mm, self._widths, self._normalized)


class RosNormalizedGripperObserver:
    """Subscribe only to ``/{side}_gripper_state`` (std_msgs/Float64).

    ``connect`` lazily imports ROS, reuses an existing node or creates an
    anonymous observer node, and subscribes without any command publisher or
    service call. ``close`` unregisters only this subscription. No shutdown,
    motor target, initialization pose, or supposedly safe hold is sent.

    ``counts`` is explicitly the old experiment script's normalized * 1000
    convention, not a measured hardware encoder count. The topic has no source
    timestamp, so freshness can only describe local message receipt.
    Dependency injection is for offline tests; no ROS import occurs when both
    ``rospy_module`` and ``message_type`` are supplied.
    """

    def __init__(self, side="right", max_age_s=0.25, calibration=None, *,
                 rospy_module=None, message_type=None, clock_ns=time.monotonic_ns,
                 state_queue_size=1):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if not math.isfinite(float(max_age_s)) or float(max_age_s) <= 0:
            raise ValueError("max_age_s must be finite and positive")
        if calibration is not None and not isinstance(calibration, WidthCalibration):
            raise TypeError("calibration must be a WidthCalibration or None")
        if not isinstance(state_queue_size, int) or isinstance(state_queue_size, bool) or state_queue_size <= 0:
            raise ValueError("state_queue_size must be a positive integer")
        self.side = side
        self.topic = f"/{side}_gripper_state"
        self.max_age_s = float(max_age_s)
        self.calibration = calibration
        self._ros = rospy_module
        self._message_type = message_type
        self._clock_ns = clock_ns
        self._state_queue_size = state_queue_size
        self._lock = threading.Lock()
        self._subscriber = None
        self._latest = None
        self._sequence = 0

    def connect(self):
        if self._subscriber is not None:
            return self
        if self._ros is None:
            import rospy
            self._ros = rospy
        if self._message_type is None:
            from std_msgs.msg import Float64
            self._message_type = Float64
        if not self._ros.core.is_initialized():
            self._ros.init_node("bottle_gripper_observer", anonymous=True, disable_signals=True)
        with self._lock:
            self._latest = None
            self._sequence = 0
        self._subscriber = self._ros.Subscriber(
            self.topic, self._message_type, self._on_state, queue_size=self._state_queue_size)
        return self

    def _on_state(self, message):
        timestamp_ns = int(self._clock_ns())
        reason = None
        try:
            value = float(message.data)
            if not math.isfinite(value):
                reason = "nonfinite_normalized_state"
            elif not 0 <= value <= 1:
                reason = "normalized_state_outside_0_1"
        except (AttributeError, TypeError, ValueError):
            value = None
            reason = "invalid_Float64_state"
        # Nonfinite numbers must not become nonstandard NaN/Infinity JSON.
        finite_value = value if value is not None and math.isfinite(value) else None
        with self._lock:
            self._sequence += 1
            self._latest = (finite_value, timestamp_ns, self._sequence, reason)

    def snapshot(self):
        now_ns = int(self._clock_ns())
        with self._lock:
            latest = self._latest
        value, stamp, sequence, error = latest if latest else (None, None, None, "no_message")
        age_s = None if stamp is None else (now_ns - stamp) / 1e9
        connected = self._subscriber is not None
        valid = latest is not None and error is None
        width_mm, calibration_error = None, None
        if valid and self.calibration is not None:
            try:
                width_mm = self.calibration.to_mm(value)
            except ValueError as exc:
                calibration_error = str(exc)
        fresh = bool(connected and valid and age_s is not None and 0 <= age_s <= self.max_age_s)
        width_kind=self.calibration.width_kind(value) if width_mm is not None else None
        estimated=width_kind=='estimated_equivalent_compression'
        return {
            "side": self.side, "topic": self.topic, "connected": connected,
            "received": latest is not None, "valid": valid, "fresh": fresh,
            "value": value, "normalized": value,
            "counts": None if value is None else value * 1000.,
            "counts_kind": "legacy_normalized_x1000_not_hardware_encoder_counts",
            "width_mm": width_mm, "width_calibrated": width_mm is not None and not estimated,
            "width_kind":width_kind,"width_is_estimate":estimated,
            "width_calibration_error": calibration_error,
            "velocity_mm_s": None, "current_A": None,
            "velocity_available": False, "current_available": False,
            "monotonic_ns": stamp, "source_sequence": sequence,
            "device_timestamp": None, "timestamp_kind": "host_monotonic_receive",
            "age_s": age_s, "max_age_s": self.max_age_s, "error": error,
        }

    def close(self):
        subscriber, self._subscriber = self._subscriber, None
        if subscriber is not None:
            subscriber.unregister()


def load_legacy_force_functions(source_path):
    """Compile only the existing controller's pure definitions, unchanged.

    No imports, constants, worker processes, CLI, SDK constructors or other
    module-level statements from that file are executed. The file is read only.
    The hash records the actual source version used by this experiment.
    """
    import ast
    import hashlib
    from pathlib import Path
    import types

    path = Path(source_path)
    raw = path.read_bytes()
    tree = ast.parse(raw, filename=str(path))
    required = {"Fault", "normal_component", "combine_force", "tag_step"}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in required]
    if len(nodes) != len(required) or {node.name for node in nodes} != required:
        raise ValueError("Legacy source must define Fault/normal_component/combine_force/tag_step exactly once")
    # Avoid executing expressions in decorators, defaults, annotations, or class
    # bodies if the upstream definitions change in a future version.
    for node in nodes:
        if node.decorator_list:
            raise ValueError("Legacy pure definitions unexpectedly have decorators")
        if isinstance(node, ast.FunctionDef):
            if (node.args.defaults or any(x is not None for x in node.args.kw_defaults)
                    or node.returns is not None
                    or any(arg.annotation is not None for arg in
                           node.args.posonlyargs + node.args.args + node.args.kwonlyargs)):
                raise ValueError("Legacy pure definitions changed: unexpected default/annotation")
        elif (node.name != "Fault" or len(node.bases) != 1
              or not isinstance(node.bases[0], ast.Name) or node.bases[0].id != "RuntimeError"
              or node.keywords or not all(isinstance(x, (ast.Pass, ast.Expr)) for x in node.body)
              or any(isinstance(x, ast.Expr) and not isinstance(x.value, ast.Constant) for x in node.body)):
            raise ValueError("Legacy Fault class changed unexpectedly")
    namespace = {"math": math, "__builtins__": __builtins__}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in required},
                                 source_path=str(path), source_sha256=hashlib.sha256(raw).hexdigest())


@dataclass
class GripperSample:
    """Duck-compatible with backend.Sample without importing robot libraries."""
    stream: str
    data: object
    monotonic_ns: int
    device_timestamp: float | None = None
    metadata: dict | None = None


class RosGripperController(RosNormalizedGripperObserver):
    """Operator-invoked software controller over the existing ROS scalar API.

    Connecting creates a publisher but publishes NOTHING. Explicit open/close
    requests enable small position steps; force requests use the existing
    ``tag_step`` and signed two-finger mean, with calibrated N passed through
    ``combine_force``. A separate worker checks limits and freshness. These are
    SOFTWARE limits: neither ROS publish nor publish-current-position proves a
    hardware stop, hardware speed limit, or protection after process loss.

    force_reader() must return {normal_N: [left, right], monotonic_ns: int or
    pair, source_sequence: int or pair}. Its normals must already be calibrated
    in N and positive for compression. Missing/old/invalid readings stop active
    software regulation. A single actuator regulates their mean; it cannot
    independently guarantee that both fingers reach the target.

    The first velocity is unavailable; later values are finite differences of
    calibrated position at host receive times, not measured motor velocity.
    Current remains NaN with explicit availability=false. close() unregisters
    without an actuation command; call hold_gripper() explicitly when needed.
    """

    def __init__(self, side, calibration, force_reader, *, min_width_mm, max_width_mm,
                 max_force_N, force_deadband_N, force_step_counts, force_settle_s,
                 max_age_s, source_path=None, control_hz=20., rospy_module=None,
                 message_type=None, clock_ns=time.monotonic_ns, start_worker=True,
                 legacy_functions=None, state_queue_size=1024, force_unit="N", target_definition="mean",
                 sample_queue_limit=10000,compression_contact_threshold=None):
        super().__init__(side, max_age_s, calibration, rospy_module=rospy_module,
                         message_type=message_type, clock_ns=clock_ns, state_queue_size=state_queue_size)
        if calibration is None:
            raise ValueError("Measured width calibration is mandatory for command access")
        values = (min_width_mm, max_width_mm, max_force_N, force_deadband_N,
                  force_step_counts, force_settle_s, control_hz)
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("Controller limits must be finite")
        if not (0 <= min_width_mm < max_width_mm and max_force_N > 0
                and force_deadband_N >= 0 and force_step_counts > 0
                and float(force_step_counts).is_integer() and force_settle_s > 0 and control_hz > 0):
            raise ValueError("Invalid controller limits")
        calibration.to_normalized(min_width_mm)
        calibration.to_normalized(max_width_mm)
        if not callable(force_reader):
            raise TypeError("force_reader must be callable")
        self._legacy = legacy_functions or load_legacy_force_functions(source_path)
        if force_unit not in ("sdk", "N") or target_definition not in ("mean", "sum", "per_finger"):
            raise ValueError("Explicit supported force unit and target definition required")
        if type(sample_queue_limit) is not int or sample_queue_limit < 1:
            raise ValueError("sample_queue_limit must be positive integer")
        self.force_unit, self.target_definition = force_unit, target_definition
        self._queue_limit, self._recording_fault = sample_queue_limit, None
        self._loop_intervals = deque(maxlen=sample_queue_limit)
        self._loop_previous_ns = None
        self._force_reader = force_reader
        self._compression_threshold=compression_contact_threshold
        if calibration.compression_model and (compression_contact_threshold is None or not math.isfinite(compression_contact_threshold) or compression_contact_threshold<=0):
            raise ValueError('Bounded compression requires an explicit bilateral contact threshold')
        self._minimum = float(min_width_mm)
        self._floor = float(min_width_mm)
        self._maximum = float(max_width_mm)
        self._max_force = float(max_force_N)
        self._deadband = float(force_deadband_N)
        self._step_counts = int(force_step_counts)
        self._settle_ns = int(force_settle_s * 1e9)
        self._period_s = 1. / float(control_hz)
        self._start_worker = bool(start_worker)
        self._control_lock = threading.RLock()
        self._publisher = None
        self._worker = None
        self._stop_event = threading.Event()
        self._mode = "idle"
        self._goal = None
        self._speed = 0.
        self._force_target = 0.
        self._fault = None
        self._last_tick_ns = None
        self._last_command_ns = None
        self._last_command_normalized = None
        self._last_force_sequences = (-1, -1)
        self._last_position_sequence = None
        self._position_previous = None
        self._velocity = float("nan")
        self._pending_samples = deque()

    def connect(self):
        if self._publisher is not None:
            return self
        super().connect()
        self._publisher = self._ros.Publisher(
            f"/{self.side}_gripper_command", self._message_type, queue_size=1, latch=False)
        self._stop_event.clear()
        if self._start_worker:
            self._worker = threading.Thread(target=self._worker_loop,
                                            name="bottle_gripper_software_loop", daemon=True)
            self._worker.start()
        return self

    def _on_state(self, message):
        super()._on_state(message)
        sample = super().snapshot()
        width, stamp = sample["width_mm"], sample["monotonic_ns"]
        with self._control_lock:
            velocity = float("nan")
            if sample["valid"] and width is not None:
                if self._position_previous is not None:
                    previous_width, previous_stamp = self._position_previous
                    if stamp > previous_stamp:
                        velocity = (width - previous_width) / ((stamp - previous_stamp) / 1e9)
                self._position_previous = (width, stamp)
            else:
                self._position_previous = None
            self._velocity = velocity
            self._append_sample(GripperSample(
                "gripper", [float("nan") if width is None else width, velocity, float("nan")], stamp,
                metadata={"source_sequence": sample["source_sequence"], "normalized": sample["normalized"],
                          "timestamp_kind": "host_monotonic_receive", "width_unit": "mm",
                          "velocity_kind": "position_finite_difference_at_receive_time",
                          "velocity_available": math.isfinite(velocity), "current_available": False,
                          "current_unit": "A", "valid": sample["valid"],
                          "width_calibration_error": sample["width_calibration_error"],
                          "width_kind":sample['width_kind'],"width_is_estimate":sample['width_is_estimate']}))

    def snapshot(self):
        result = super().snapshot()
        with self._control_lock:
            result.update(velocity_mm_s=self._velocity, current_A=float("nan"),
                          velocity_available=math.isfinite(self._velocity), current_available=False,
                          velocity_kind="position_finite_difference_at_receive_time",
                          force_target_N=self._force_target, controller_state=self._mode,
                          force_target=self._force_target, force_unit=self.force_unit,
                          fault=self._fault, min_width_mm=self._floor,
                          command_normalized=self._last_command_normalized,
                          command_timestamp_ns=self._last_command_ns,
                          commanded_speed_mm_s=self._speed,
                          limits_kind="software_only_no_hardware_guarantee",
                          hold_kind="publish_latest_measured_position_not_verified_native_stop",
                          force_definition=self.target_definition,
                          legacy_source_sha256=getattr(self._legacy, "source_sha256", None))
            stats = self.loop_metrics()
            result["loop_period_s"] = {"count":stats["count"],"mean":stats["mean_s"],
                                       "p95":stats["p95_s"],"max":stats["max_s"],
                                       "window":"latest_at_most_"+str(self._queue_limit)}
        return result

    def drain_samples(self):
        with self._control_lock:
            if self._recording_fault and not self._pending_samples:
                raise RuntimeError(self._recording_fault)
            result = list(self._pending_samples)
            self._pending_samples.clear()
        return result

    def _append_sample(self, sample):
        if self._recording_fault:
            return  # Fault has already stopped acquisition; no silent continuation.
        self._pending_samples.append(sample)
        if len(self._pending_samples) > self._queue_limit:
            self._recording_fault = "Gripper recording queue overflow; final batch retained, controller fault latched"
            self._fault = self._recording_fault
            self._mode, self._goal, self._speed, self._force_target = "fault", None, 0., 0.

    def reset_loop_metrics(self):
        """Start a new diagnostic window without commands or timing changes."""
        with self._control_lock:
            self._loop_intervals.clear()
            self._loop_previous_ns = None

    def loop_metrics(self):
        import numpy as np
        with self._control_lock:
            values = np.asarray(self._loop_intervals, dtype=float)
        return {"count": len(values), "mean_s": float(values.mean()) if len(values) else None,
                "p95_s": float(np.percentile(values, 95)) if len(values) else None,
                "max_s": float(values.max()) if len(values) else None,
                "source": "actual active software control tick start intervals"}

    def suspend_publication(self):
        """Explicit handoff after contact: hold, stop local proposals, unregister."""
        with self._control_lock:
            self.hold_gripper()
            self._mode = "external_force"
            publisher, self._publisher = self._publisher, None
            if publisher is not None:
                publisher.unregister()

    def resume_publication(self, *, timeout_s=None, poll_s=None):
        """Only the supervisor may call after old-worker exit and ownership proof."""
        with self._control_lock:
            if self._fault:
                raise RuntimeError("Cannot reacquire a faulted gripper controller")
            self._fresh_position()
            if self._publisher is not None:
                raise RuntimeError("Gripper publisher is already held")
            self._publisher = self._ros.Publisher(
                f"/{self.side}_gripper_command", self._message_type, queue_size=1, latch=False)
            self._mode, self._goal, self._speed = "idle", None, 0.
        if timeout_s is not None:
            if not math.isfinite(float(timeout_s)) or timeout_s <= 0 or not math.isfinite(float(poll_s)) or poll_s <= 0:
                raise ValueError("Explicit positive handoff publisher wait limits required")
            deadline=time.monotonic()+timeout_s
            while self._publisher.get_num_connections()<1:
                self._fresh_position()
                if time.monotonic()>=deadline:
                    raise RuntimeError("Reacquired gripper publisher has no actuator subscriber; hold not confirmed")
                time.sleep(poll_s)

    def _fresh_position(self):
        sample = super().snapshot()
        if not sample["fresh"] or sample["width_mm"] is None:
            raise RuntimeError("Missing/stale/invalid calibrated gripper position")
        return sample

    @staticmethod
    def _pair(value, name):
        if isinstance(value, (str, bytes, bool)):
            raise RuntimeError(f"Invalid force {name}")
        if isinstance(value, numbers.Real):
            value = [value, value]
        try:
            values = tuple(float(x) for x in value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid force {name}") from exc
        if len(values) != 2 or not all(math.isfinite(x) for x in values):
            raise RuntimeError(f"Invalid force {name}")
        return values

    def _force_sample(self, now):
        sample = self._force_reader()
        if not isinstance(sample, dict):
            raise RuntimeError("force_reader must supply normal_N and independent source time/sequence")
        value = sample.get("normal", sample.get("normal_N"))
        if sample.get("unit", self.force_unit) != self.force_unit:
            raise RuntimeError("Force feedback unit differs from selected controller unit")
        if isinstance(value, numbers.Real):
            raise RuntimeError("normal_N must contain two independently observed finger forces")
        normals = self._pair(value, "normal")
        stamps = self._pair(sample.get("monotonic_ns"), "monotonic_ns")
        sequences = self._pair(sample.get("source_sequence"), "source_sequence")
        if any(x < 0 or not x.is_integer() for x in stamps + sequences):
            raise RuntimeError("Force source timestamps/sequences must be nonnegative integers")
        if any(not 0 <= now - stamp <= self.max_age_s * 1e9 for stamp in stamps):
            raise RuntimeError("Stale/future calibrated normal force")
        if max(normals) > self._max_force:
            raise RuntimeError("Per-finger force exceeds configured software F_max")
        return normals, sequences, stamps

    def _publish(self, normalized, *, kind, speed_mm_s=0., allow_hold_current=False,
                 expected_sequence=None, emergency_open=False):
        if self._stop_event.is_set():
            raise RuntimeError("Gripper software loop is closing; no further publication")
        if self._publisher is None:
            raise RuntimeError("Gripper command transport is not connected")
        sample = self._fresh_position()
        if expected_sequence is not None and sample["source_sequence"] != expected_sequence:
            raise RuntimeError("Position feedback advanced while a command was being proposed")
        width = self.calibration.to_mm(normalized)
        if (self.calibration.width_kind(normalized)=='estimated_equivalent_compression'
                and normalized<sample['normalized']-1e-10 and self._mode!='force'):
            raise RuntimeError('Further compression requires bilateral contact and active force regulation')
        if emergency_open:
            # A recovery may start below the dynamic grasp floor after a fault.
            # Only increasing calibrated opening, never further closure, is
            # permitted. Both endpoints remain inside the measured domain.
            if not sample["width_mm"] <= width <= self._maximum:
                raise RuntimeError("Emergency release target must only increase measured opening within travel")
        elif not allow_hold_current and not self._floor <= width <= self._maximum:
            raise RuntimeError("Proposed gripper target violates software width bound")
        if allow_hold_current and normalized != sample["normalized"]:
            raise RuntimeError("Hold must use the latest measured normalized position")
        message = self._message_type()
        message.data = float(normalized)
        self._publisher.publish(message)
        stamp = int(self._clock_ns())
        self._last_command_ns = stamp
        self._last_command_normalized = float(normalized)
        self._append_sample(GripperSample(
            "gripper_command", [width, float(speed_mm_s)], stamp,
            metadata={"kind": kind, "normalized": float(normalized),
                      "width_kind":self.calibration.width_kind(normalized),
                      "width_is_estimate":self.calibration.width_kind(normalized)=='estimated_equivalent_compression',
                      "timestamp_kind": "host_publish_return_not_hardware_ack",
                      "source_sequence": sample["source_sequence"], "limits_kind": "software_only"}))

    def _begin_position(self, width_mm, speed_mm_s, direction):
        with self._control_lock:
            if self._fault is not None:
                raise RuntimeError("Controller fault is latched: " + self._fault)
            width_mm, speed_mm_s = float(width_mm), float(speed_mm_s)
            if (not math.isfinite(width_mm) or not self._floor <= width_mm <= self._maximum
                    or not math.isfinite(speed_mm_s) or speed_mm_s <= 0):
                raise ValueError("Position command must respect width bounds and positive finite speed")
            sample = self._fresh_position()
            if self._publisher is None:
                raise RuntimeError("Gripper is not connected")
            self._force_sample(int(self._clock_ns()))
            if direction == "open" and width_mm < sample["width_mm"]:
                raise ValueError("An open request cannot close the gripper")
            if direction == "close" and width_mm > sample["width_mm"]:
                raise ValueError("A close request cannot open the gripper")
            if direction=='close' and self.calibration.compression_model and width_mm<self.calibration.compression_model['contact_width_mm']-1e-9:
                raise ValueError('Position closure cannot enter the gel-compression interval; bilateral contact and force mode required')
            self._mode, self._goal, self._speed = direction, width_mm, speed_mm_s
            self._force_target = 0.
            self._last_tick_ns = int(self._clock_ns())
            self._last_position_sequence = None

    def command_open(self, width_mm, speed_mm_s):
        self._begin_position(width_mm, speed_mm_s, "open")

    def command_close(self, min_mm, speed_mm_s):
        self._begin_position(min_mm, speed_mm_s, "close")

    def emergency_open_pre_lift(self, width_mm, speed_mm_s):
        """One-way release after the caller confirms pre-lift and arm stopped.

        This explicit recovery entry point does not infer whether a bottle is
        airborne; Runner must enforce that condition before calling. Force/DD
        failure must not prevent release, so only fresh calibrated jaw position
        is required here. Existing faults remain latched and normal commands
        stay disabled, even if the release reaches its target successfully.
        No native hardware stop or speed guarantee is implied.
        """
        with self._control_lock:
            # First cancel all software feedback/position proposals. A bad
            # request must never leave the old force loop running.
            self._mode, self._goal, self._speed, self._force_target = "fault", None, 0., 0.
            if self._fault is None:
                self._fault = "Emergency pre-lift recovery latched; normal commands disabled"
            width_mm, speed_mm_s = float(width_mm), float(speed_mm_s)
            if (not math.isfinite(width_mm) or not self._floor <= width_mm <= self._maximum
                    or not math.isfinite(speed_mm_s) or speed_mm_s <= 0):
                raise ValueError("Emergency opening requires a bounded calibrated target and positive finite speed")
            self.calibration.to_normalized(width_mm)
            sample = self._fresh_position()
            if width_mm < sample["width_mm"]:
                raise ValueError("Emergency pre-lift release cannot close the gripper")
            # Cancel the previous submitted actuator target using the measured
            # position convention before beginning the slower opening steps.
            self._publish(sample["normalized"], kind="emergency_pre_lift_cancel_to_measured_position",
                          allow_hold_current=True, expected_sequence=sample["source_sequence"])
            self._mode, self._goal, self._speed = "emergency_open", width_mm, speed_mm_s
            self._last_tick_ns = int(self._clock_ns())
            self._last_position_sequence = None

    def set_gripper_min_width(self, width_mm):
        with self._control_lock:
            width_mm = float(width_mm)
            if not math.isfinite(width_mm) or not self._minimum <= width_mm < self._maximum:
                raise ValueError("Dynamic floor must remain inside the measured configured range")
            self.calibration.to_normalized(width_mm)
            self._floor = width_mm
            if self._mode in ("close", "force"):
                if (self._goal is not None and self._goal < width_mm) or self._fresh_position()["width_mm"] < width_mm:
                    self.hold_gripper()

    def set_force_target(self, per_finger_N):
        with self._control_lock:
            from .force_control import target_as_mean
            value = float(per_finger_N)
            mean_target = target_as_mean(value, self.target_definition)
            mean_band = self._deadband / (2. if self.target_definition == "sum" else 1.)
            if not math.isfinite(value) or value < 0 or mean_target + mean_band >= self._max_force:
                raise ValueError("Force target/deadband must lie below per-finger software F_max")
            if value == 0:
                self.exit_force_mode()
                return
            if self._fault is not None:
                raise RuntimeError("Controller fault is latched: " + self._fault)
            sample = self._fresh_position()
            if not self._floor <= sample["width_mm"] <= self._maximum:
                raise RuntimeError("Actual width is outside software bounds")
            normals,_,_=self._force_sample(int(self._clock_ns()))
            if self.calibration.compression_model and min(normals)<self._compression_threshold:
                raise RuntimeError('Bilateral contact is required before enabling gel-compression force control')
            self._mode, self._force_target, self._goal, self._speed = "force", value, None, 0.
            self._last_force_sequences = (-1, -1)
            self._last_tick_ns = int(self._clock_ns())

    def hold_gripper(self):
        with self._control_lock:
            self._mode, self._force_target, self._goal, self._speed = "hold", 0., None, 0.
            sample = self._fresh_position()
            self._publish(sample["normalized"], kind="hold_measured_position", allow_hold_current=True)

    def exit_force_mode(self):
        # Stop software regulation first, then cancel its last target using the
        # measured position convention explicitly requested by the operator.
        self.hold_gripper()

    def _tick(self):
        """One software update. Public only for deterministic fake-I/O tests."""
        with self._control_lock:
            if self._stop_event.is_set() or self._mode not in ("open", "close", "force", "emergency_open"):
                self._loop_previous_ns = None
                return
            try:
                now = int(self._clock_ns())
                if self._loop_previous_ns is not None and now > self._loop_previous_ns:
                    self._loop_intervals.append((now-self._loop_previous_ns)/1e9)
                self._loop_previous_ns = now
                sample = self._fresh_position()
                emergency_open = self._mode == "emergency_open"
                if not emergency_open:
                    normals, sequences, stamps = self._force_sample(now)
                actual = sample["width_mm"]
                if actual > self._maximum or (not emergency_open and actual < self._floor):
                    raise RuntimeError("Actual gripper width violates software bounds")
                if self._mode == "force":
                    if any(new <= old for new, old in zip(sequences, self._last_force_sequences)):
                        return
                    if self._last_command_ns is not None and (
                            now - self._last_command_ns < self._settle_ns
                            or min(stamps) < self._last_command_ns):
                        return
                    c = {"position_min": self.calibration.to_normalized(self._floor)*1000.,
                         "position_max": self.calibration.to_normalized(self._maximum)*1000.,
                         "step_size": self._step_counts, "target_force": self._force_target,
                         "deadband": self._deadband, "control_force_mode": "sdk_raw",
                         "fingers": [{"normal_axis": "Fz", "compression_sign": 1, "zero_raw": 0} for _ in range(2)]}
                    # Already-calibrated N is passed through the unchanged mean
                    # function. No abs, second zero subtraction or sum factor.
                    measured, _ = self._legacy.combine_force(c, [[0, 0, n, 0, 0, 0] for n in normals])
                    if self.target_definition == "sum":
                        measured = sum(normals)
                    elif self.target_definition == "per_finger":
                        # One actuator cannot independently balance unequal
                        # fingers: never chase a low finger past a high one.
                        measured = max(normals)
                        if min(normals) < self._force_target-self._deadband and max(normals) <= self._force_target+self._deadband:
                            measured = self._force_target if max(normals) >= self._force_target-self._deadband else max(normals)
                    if min(normals) <= 0 or (self.calibration.compression_model and min(normals)<self._compression_threshold):
                        # Do not chase a lost bottle by continuing to close.
                        # Keep the observation available for runtime HOLD's
                        # dropped label without auto-restarting regulation.
                        self.hold_gripper()
                        return
                    direction, counts = self._legacy.tag_step(c, measured, sample["normalized"]*1000.)
                    self._last_force_sequences = sequences
                    if counts is not None:
                        self._publish(counts/1000., kind="legacy_force_" + direction,
                                      expected_sequence=sample["source_sequence"])
                else:
                    if sample["source_sequence"] == self._last_position_sequence:
                        return
                    elapsed = min(max(0., (now - self._last_tick_ns)/1e9), 2*self._period_s)
                    self._last_tick_ns = now
                    delta = self._goal - actual
                    if emergency_open and delta < 0:
                        raise RuntimeError("Emergency opening target overshot; refusing corrective closure")
                    if abs(delta) <= 1e-9:
                        self._mode = "hold"
                        self._speed = 0.
                        return
                    if elapsed <= 0:
                        return
                    step = min(abs(delta), self._speed*elapsed)
                    target = actual + math.copysign(step, delta)
                    self._publish(self.calibration.to_normalized(target), kind=self._mode,
                                  speed_mm_s=math.copysign(self._speed, delta),
                                  expected_sequence=sample["source_sequence"], emergency_open=emergency_open)
                    self._last_position_sequence = sample["source_sequence"]
            except Exception as exc:
                detail = str(exc)
                if not self._fault:
                    self._fault = detail
                elif detail not in self._fault:
                    self._fault += "; subsequent controller fault: " + detail
                self._mode = "fault"
                self._force_target = 0.
                self._goal = None
                self._speed = 0.
                try:
                    sample = self._fresh_position()
                    self._publish(sample["normalized"], kind="fault_hold_measured_position", allow_hold_current=True)
                except Exception as hold_exc:
                    self._fault += "; measured-position hold unavailable: " + str(hold_exc)

    def _worker_loop(self):
        while not self._stop_event.wait(self._period_s):
            self._tick()

    def close(self):
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=max(1., 4*self._period_s))
            if self._worker.is_alive():
                raise RuntimeError("Software gripper worker did not stop; transport retained")
            self._worker = None
        with self._control_lock:
            self._mode = "disconnected"
            publisher, self._publisher = self._publisher, None
            if publisher is not None:
                publisher.unregister()
            super().close()
