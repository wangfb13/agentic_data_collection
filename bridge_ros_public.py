"""User-selected ROS/X5 adapter. Importing this file has no hardware effects.

Actuation uses public pose/gripper topics with software interpolation/feedback
hold, as requested. It does NOT claim native stop, hard real-time speed limits,
or protection after this process/host disappears. Existing controller settings
and SDK owners are never changed here. DD is explicitly the existing processed
stream; unavailable motor current is preserved as NaN, never invented.
"""
from __future__ import annotations

from collections import deque
import hashlib
import math
from pathlib import Path
import threading
import time

import numpy as np

from bottle_scan.backend import PUBLIC_ROS_CAPABILITIES, Sample
from bottle_scan.hardware_arm import RosArmController
from bottle_scan.hardware_gripper import RosGripperController, WidthCalibration
from bottle_scan.hardware_sensors import ActualX5SensorSource
from bottle_scan.command_ownership import guard_all_commands, current_ros_node_names, format_ownership_report
from bottle_scan.force_control import force_unit, target_definition, force_value, target_as_mean, force_components
from bottle_scan.existing_force import ExistingForceController, validate_existing_settings
from bottle_scan.hardware_observer import RosJointObserver
from bottle_scan.park_guard import image_pose_metadata

PROFILE = "public_ros"
CAPABILITIES = dict.fromkeys(PUBLIC_ROS_CAPABILITIES, True)
CONTROLLER_STATE_CODES = {"idle": 0, "open": 1, "close": 2, "force": 3,
                          "hold": 4, "fault": 5, "disconnected": 6, "emergency_open": 7, "external_force": 8}


def validate_settings(cfg):
    hardware = cfg["hardware"]
    options = hardware["public_ros"]
    if hardware.get("profile") != PROFILE:
        raise ValueError("This provider requires explicit hardware.profile=public_ros")
    if any(options.get(key) is not True for key in ("software_control_acknowledged", "processed_dd_accepted")):
        raise ValueError("Public ROS software control and existing DD must be explicitly selected")
    if hardware["arm_side"] not in ("left", "right"):
        raise ValueError("Select exactly one arm per session")
    calibration = WidthCalibration(hardware["gripper_width_calibration"])
    if calibration.compression_model:
        model=calibration.compression_model
        if options['raw_normalized_min']!=model['normalized_hard_min']:
            raise ValueError('Compression model changes the inherited normalized operating floor')
        if model['max_compression_mm']>2*cfg['geometry']['gripper']['pad_depth_mm']+1e-8:
            raise ValueError('Compression window exceeds the two declared soft gel thicknesses')
        if cfg['force'].get('controller','bridge_loop')!='bridge_loop':
            raise ValueError('Bounded equivalent compression requires the guarded bridge_loop controller')
        if cfg['gripper'].get('precontact_min_mm')!=model['contact_width_mm']:
            raise ValueError('Position closure must stop at the measured contact anchor')
    minimum = calibration.to_normalized(cfg["gripper"]["min_mm"])
    calibration.to_normalized(cfg["gripper"]["open_mm"])
    # Preserve the old controller's explicit operating lower bound. Reject a
    # conflicting experiment configuration; do not silently clip a command.
    floor = options["raw_normalized_min"]
    if not isinstance(floor, (int, float)) or not 0 <= floor <= 1 or minimum < floor:
        raise ValueError("Measured w_min maps below the configured existing normalized operating bound")
    deadband = options["force_deadband_N"]
    if (not isinstance(deadband, (float, int)) or not math.isfinite(deadband) or deadband < 0
            or target_as_mean(force_value(cfg,"target"),target_definition(cfg))+deadband/(2 if target_definition(cfg)=="sum" else 1) >= force_value(cfg,"max")):
        raise ValueError("F_target plus software deadband must be below F_max")
    if options["sensors"]["serials"] != hardware["sensor_serials"]:
        raise ValueError("X5 left/right serial mapping must match session sensor identities")
    if options["sensors"].get("gripper_side", hardware["arm_side"]) != hardware["arm_side"]:
        raise ValueError("Selected arm differs from the explicitly configured tactile gripper side")
    if tuple(cfg["tactile"]["dd_shape"]) != (96, 128, 3) or cfg["tactile"]["dd_dtype"] != "float32":
        raise ValueError("Existing X5 decoded DD uses float32 [96,128,3]; encoded uint16 is saved separately")
    if cfg["force"].get("controller", "bridge_loop") == "existing":
        validate_existing_settings(cfg)


class PublicRosBackend:
    simulated = False

    def __init__(self, cfg, *, arm=None, gripper=None, sensors=None, connect=True,
                 ownership_guard=None, existing_transport_factory=None, passive_joints=None):
        validate_settings(cfg)
        self.cfg, self.side = cfg, cfg["hardware"]["arm_side"]
        options = cfg["hardware"]["public_ros"]
        self._ownership_guard = ownership_guard or guard_all_commands
        self.passive_joints = passive_joints if passive_joints is not None else RosJointObserver(cfg)
        self._session = None
        age = cfg["recording"]["source_max_age_s"]
        self.sensors = sensors if sensors is not None else ActualX5SensorSource(cfg)
        self.arm = arm if arm is not None else RosArmController(
            arm_side=self.side, base_frame=cfg["hardware"]["base_frame"], max_age_s=age,
            publish_hz=options["arm_publish_hz"], max_step_mm=options["arm_command_step_mm"],
            max_step_deg=options["arm_command_step_deg"],
            pose_tolerance_mm=cfg["motion"]["pose_tolerance_mm"],
            pose_tolerance_deg=cfg["motion"]["pose_tolerance_deg"])
        self.gripper = gripper if gripper is not None else RosGripperController(
            self.side, WidthCalibration(cfg["hardware"]["gripper_width_calibration"]), self._force_reader,
            min_width_mm=cfg["gripper"]["min_mm"], max_width_mm=cfg["gripper"]["open_mm"],
            max_force_N=cfg["force"]["max_N"], force_deadband_N=options["force_deadband_N"],
            force_step_counts=options["force_step_counts"], force_settle_s=options["force_settle_s"],
            max_age_s=age, source_path=options["force_source_path"],
            control_hz=cfg["recording"]["control_hz"], force_unit=force_unit(cfg),
            target_definition=target_definition(cfg), sample_queue_limit=options["queue_limit"],
            compression_contact_threshold=force_value(cfg,'contact_min'))
        self.existing_force = None
        if cfg["force"].get("controller", "bridge_loop") == "existing":
            extra = {} if existing_transport_factory is None else {"transport_factory": existing_transport_factory}
            self.existing_force = ExistingForceController(cfg, self.gripper, self._force_reader,
                ownership_check=self._check_ownership, **extra)
        self._dispatch_lock = threading.RLock()
        self._pending = deque()
        self._recording = False
        self._fault = None
        self._stage = "IDLE"
        self._trial_id = None
        self._stage_history = [(0, "IDLE")]
        self._queue_limit = int(options["queue_limit"])
        if self._queue_limit <= 0:
            raise ValueError("Positive bridge queue_limit required")
        self._stop = threading.Event()
        self._worker = None
        self._idle_discarded = 0
        self._closed = False
        self._zero_ready = False
        self._ever_zero_ready = False
        if connect:
            self.connect()

    def connect(self):
        """Connect data and idle publishers; never publish an actuator command."""
        if self._worker is not None:
            return self
        self._worker = threading.Thread(target=self._dispatch_loop, name="bottle_source_dispatch", daemon=True)
        self._worker.start()
        try:
            self._check_ownership(())
            # Dispatcher drains raw startup/zero frames while source caches retain
            # the actual latest state. Human waits outside a recording don't fill
            # an unbounded queue, and are not claimed to be recorded trial data.
            self.sensors.start()
            self.arm.connect()
            self.passive_joints.connect()
            self.gripper.connect()
            deadline = time.monotonic()+5
            last = "waiting for read-only raw state"
            while time.monotonic()<deadline:
                snap = self.observation_snapshot()
                if not snap.get("fault"):
                    return self
                last = snap["fault"]
                time.sleep(.02)
            raise RuntimeError("Public ROS state did not become ready: "+last)
        except BaseException:
            self.close()
            raise

    def monotonic_ns(self):
        return time.monotonic_ns()

    def _force_reader(self):
        self._require_zero_ready()
        snap = self.sensors.snapshot()
        if snap.get("fault"):
            raise RuntimeError(str(snap["fault"]))
        keys = ("wrench_left", "wrench_right")
        return {"normal_N":snap["normal_N"], "normal":snap["normal_N"], "unit":force_unit(self.cfg),
            "monotonic_ns":[snap["source_timestamp_ns"][k] for k in keys],
            "source_sequence":[snap["source_sequences"][k] for k in keys]}

    def _require_zero_ready(self):
        if not self._zero_ready:
            raise RuntimeError("zero_not_ready: run guarded prepare_zero for this connection before any motion")

    def prepare_zero(self, session, *, confirm=None):
        """Apply a guarded software zero and persist this connection's evidence.

        Safe for a just-created session: reference-free first connection requires
        a fresh EMPTY response; an old YAML declaration is never confirmation.
        Repeated explicit calls use all the same guards and append new evidence.
        """
        if self._recording:
            raise RuntimeError("Software zero is forbidden during active acquisition/trial")
        self._zero_ready = False
        if self.gripper.snapshot().get("controller_state") not in ("idle", "hold"):
            raise RuntimeError("Software zero requires an idle/holding gripper controller")
        references = session.meta.get("references", [])
        reference, reference_info = None, None
        if references:
            reference_path = Path(session.path) / references[-1]["path"]
            with np.load(reference_path, allow_pickle=False) as saved:
                reference = {side+"_dd":np.array(saved[side+"_dd"], copy=True) for side in ("left", "right")}
            reference_info = {"path":references[-1]["path"],
                              "sha256":hashlib.sha256(reference_path.read_bytes()).hexdigest()}
        records = session.meta.setdefault("zero_runs", [])
        previous = next((row for row in reversed(records) if row.get("status") == "ready"), None)
        requested = time.time_ns()
        report = None
        try:
            report = self.sensors.zero_unloaded(width_reader=self.gripper.snapshot, reference=reference,
                                               confirm=confirm, previous=previous)
            report = {**report, "run_requested_unix_ns":requested, "unloaded_reference":reference_info}
            records.append(report)
            session.save_meta()
            # Persistence is part of readiness: a failed meta write cannot
            # silently permit unrecorded physical-force offsets to be used.
            self._zero_ready = self._ever_zero_ready = True
            for warning in report.get("warnings", []):
                print("WARNING: " + warning)
            return report
        except BaseException as exc:
            failed = {**(report or getattr(self.sensors,"last_zero_attempt",None) or {}),
                      "status":"failed", "run_requested_unix_ns":requested,
                      "unloaded_reference":reference_info, "error":type(exc).__name__+": "+str(exc)}
            if records and records[-1].get("run_requested_unix_ns") == requested:
                records[-1] = failed
            else:
                records.append(failed)
            session.save_meta()
            raise

    def observation_snapshot(self):
        """Read initial TCP/width/raw DD and force before any software zero."""
        return self._snapshot(observe=True)

    def snapshot(self):
        return self._snapshot(observe=False)

    def _snapshot(self, *, observe):
        try:
            if self._fault:
                raise RuntimeError(self._fault)
            if not observe:
                self._require_zero_ready()
            sensors = self.sensors.observation_snapshot() if observe else self.sensors.snapshot()
            if self.existing_force is not None:
                self.existing_force.check()
            arm, grip = self.arm.snapshot(), self.gripper.snapshot()
            if arm.get("fault") or grip.get("fault"):
                raise RuntimeError(str(arm.get("fault") or grip.get("fault")))
            if not grip.get("fresh") or not grip.get("valid") or not grip.get("velocity_available"):
                raise RuntimeError("Fresh calibrated gripper feedback (at least two source states) required")
            pose = arm["arms"][self.side]
            now = self.monotonic_ns()
            stamps = dict(sensors["source_timestamp_ns"])
            stamps.update(sensor_summary=sensors["timestamp_ns"], tcp=pose["received_monotonic_ns"],
                          gripper=grip["monotonic_ns"])
            for source, stamp in stamps.items():
                if not isinstance(stamp, (int, np.integer)) or not 0 <= now-int(stamp) <= self.cfg["recording"]["source_max_age_s"]*1e9:
                    raise RuntimeError("Stale/future/missing source timestamp: " + source)
            if sensors.get("fault"):
                raise RuntimeError(str(sensors["fault"]))
            if not np.isfinite([grip["width_mm"], grip["velocity_mm_s"]]).all():
                raise RuntimeError("Calibrated gripper width and estimated velocity must be finite")
            result = {"timestamp_ns":min(stamps.values()),
                "gripper_timestamp_ns":grip["monotonic_ns"], "tcp":np.asarray(pose["T_base_tcp_mm"],float),
                "width_mm":grip["width_mm"], "velocity_mm_s":grip["velocity_mm_s"],
                "width_kind":grip.get('width_kind','measured_anchor_interpolation'),
                "width_is_estimate":grip.get('width_is_estimate',False),
                "current_A":float("nan"), "current_A_available":False,
                "velocity_source":"finite_difference_of_received_equivalent_aperture" if grip.get('width_is_estimate') else "finite_difference_of_received_calibrated_position",
                "dd":sensors["dd"], "zero_ready":self._zero_ready,
                "force_target_N":grip["force_target_N"], "force_target":grip["force_target_N"], "controller_state":grip["controller_state"],
                "fault":None, "hardware_profile":PROFILE, "arms":arm["arms"],
                "force_unit":force_unit(self.cfg), "force_definition":target_definition(self.cfg)}
            if observe:
                result.update(wrench_raw=sensors["wrench_raw"], wrench_unit=sensors["wrench_unit"])
            else:
                result.update(wrench=sensors["wrench"], wrench_force=np.asarray(sensors["wrench"])[:,:3],
                              normal_N=sensors["normal_N"], normal=sensors["normal_N"],
                              normal_components=force_components(sensors["normal_N"]),
                              source_sequences=sensors.get("source_sequences"), source_timestamp_ns=stamps)
            if self.existing_force is not None and self.existing_force.active:
                result.update(force_target_N=self.existing_force.target, force_target=self.existing_force.target,
                              controller_state="external_force")
            return result
        except Exception as exc:
            return {"timestamp_ns":self.monotonic_ns(), "fault":str(exc)}

    def _stage_at(self, timestamp_ns):
        for timestamp, name in reversed(self._stage_history):
            if timestamp<=timestamp_ns:
                return name
        return "IDLE"

    def _dispatch_once(self):
        with self._dispatch_lock:
            if self._fault:
                return  # Preserve remaining component queues; acquisition has failed.
            samples = []
            try:
                for component in (self.sensors, self.arm, self.gripper):
                    samples.extend(component.drain_samples())
            except BaseException as exc:
                # Earlier components in this batch have already transferred
                # ownership. Retain their events before surfacing this fault.
                self._fault = "Source recording failure: " + str(exc)
            samples.sort(key=lambda sample: sample.monotonic_ns)
            if not self._recording:
                self._idle_discarded += len(samples)
                return
            for sample in samples:
                metadata = dict(sample.metadata or {})
                if sample.stream.startswith("camera/") and self._session is not None:
                    metadata.update(image_pose_metadata({"arms":self.arm.snapshot()["arms"],
                        "timestamp_ns":self.monotonic_ns()}, self._session, camera_stamp_ns=sample.monotonic_ns))
                metadata.update(stage=self._stage_at(sample.monotonic_ns), hardware_profile=PROFILE,
                                simulated=False, trial_id=self._trial_id)
                self._pending.append(Sample(sample.stream, sample.data, sample.monotonic_ns,
                                            sample.device_timestamp, metadata))
            if len(self._pending)>self._queue_limit:
                # The just-drained batch is retained in full, even if it crosses
                # the normal capacity. No further batches are consumed. The
                # writer may flush this final batch before observing the fault.
                self._fault = "Recording queue overflow: acquisition stopped; final drained batch retained"

    def _dispatch_loop(self):
        while not self._stop.wait(.005):
            try:
                self._dispatch_once()
                if self._fault:
                    return
            except BaseException as exc:
                self._fault = "Source recording failure: "+str(exc)
                return

    def begin_acquisition(self):
        with self._dispatch_lock:
            if self._fault:
                raise RuntimeError(self._fault)
            if self._recording:
                raise RuntimeError("Overlapping acquisition writers are not supported")
            if self._pending:
                raise RuntimeError("Previous acquisition has undrained source events")
            self._dispatch_once()  # discard explicitly unrecorded idle backlog
            if self._fault:
                raise RuntimeError(self._fault)
            for controller in (self.arm, self.gripper):
                if hasattr(controller, "reset_loop_metrics"):
                    controller.reset_loop_metrics()
            self._recording = True

    def end_acquisition(self):
        with self._dispatch_lock:
            self._dispatch_once()
            self._recording = False

    def drain_samples(self):
        with self._dispatch_lock:
            if self._fault and not self._pending:
                raise RuntimeError(self._fault)
            result = list(self._pending)
            self._pending.clear()
            return result

    def _control_event(self):
        snap = self.gripper.snapshot()
        if self.existing_force is not None and self.existing_force.active:
            snap = dict(snap, controller_state="external_force", force_target_N=self.existing_force.target)
        stamp = self.monotonic_ns()
        with self._dispatch_lock:
            if self._recording:
                state_name = str(snap["controller_state"])
                for name, data in (("force_target", np.asarray(snap["force_target_N"],dtype=np.float64)),
                                   ("controller_state", np.asarray(CONTROLLER_STATE_CODES.get(state_name,-1),dtype=np.int16))):
                    self._pending.append(Sample(name,data,stamp,None,
                        {"stage":self._stage,"hardware_profile":PROFILE,"event":"explicit_control_transition",
                         "trial_id":self._trial_id,"simulated":False,"controller_state_name":state_name,
                         "force_unit":force_unit(self.cfg),"force_definition":target_definition(self.cfg),
                         "controller_state_codes":CONTROLLER_STATE_CODES}))
                if len(self._pending)>self._queue_limit:
                    self._fault = "Control event recording queue overflow; final events retained"
                    raise RuntimeError(self._fault)

    def set_stage(self, name):
        with self._dispatch_lock:
            self._stage = str(name)
            self._stage_history.append((self.monotonic_ns(),self._stage))
        self.sensors.set_context(stage=self._stage)

    def set_trial(self, item, scenario="normal"):
        if scenario != "normal":
            raise ValueError("Synthetic faults cannot be injected into hardware")
        self._trial_id = item.get("trial_id", item.get("point_id"))

    def images(self):
        return self.sensors.images()

    def image_samples(self):
        """Capture fresh keyframe samples with source stamps and both TCPs."""
        samples=self.sensors.image_samples()
        if self._session is None:
            return samples
        observation={"arms":self.arm.snapshot()["arms"],"timestamp_ns":self.monotonic_ns()}
        return {name:Sample(sample.stream,sample.data,sample.monotonic_ns,sample.device_timestamp,
                {**(sample.metadata or {}),**image_pose_metadata(observation,self._session,
                    camera_stamp_ns=sample.monotonic_ns)}) for name,sample in samples.items()}

    def bind_session(self, session):
        self._session = session

    def park_observation(self):
        return {"arms":self.arm.snapshot()["arms"], "joints":self.passive_joints.snapshot()}

    def _check_ownership(self, external_nodes=None):
        if external_nodes is None:
            external_nodes = self.existing_force.check() if self.existing_force is not None else ()
        report = self._ownership_guard(self.cfg, own_nodes=tuple(current_ros_node_names())+tuple(external_nodes))
        print(format_ownership_report(report))
        return report

    def move_linear(self, pose_mm, speed_mm_s, angular_speed_deg_s):
        self._require_zero_ready()
        self._check_ownership()
        self.arm.move_linear(pose_mm,speed_mm_s,angular_speed_deg_s)

    def motion_done(self):
        return self.arm.motion_done()

    def stop_arm(self):
        return self.arm.stop_arm()

    def command_open(self, width_mm, speed_mm_s):
        self._require_zero_ready()
        if self.existing_force is not None and self.existing_force.active:
            self.existing_force.exit_force_mode()
        self._check_ownership()
        self.gripper.command_open(width_mm,speed_mm_s)
        self._control_event()

    def command_close(self, lower_width_mm, speed_mm_s):
        self._require_zero_ready()
        if self.existing_force is not None and self.existing_force.active:
            self.existing_force.exit_force_mode()
        self._check_ownership()
        self.gripper.command_close(lower_width_mm,speed_mm_s)
        self._control_event()

    def emergency_open_pre_lift(self, width_mm, speed_mm_s):
        # Runner enforces NOT airborne and a completed arm stop before this
        # narrow recovery operation. Bad force/DD must not prevent opening;
        # real, fresh gripper feedback is still mandatory throughout.
        if not self._ever_zero_ready:
            raise RuntimeError("No guarded zero has succeeded on this connection; no emergency actuation during initialization")
        if self.existing_force is not None and self.existing_force.transport is not None:
            self.existing_force.exit_force_mode()  # Refuse release if external owner cannot exit.
        self.gripper.emergency_open_pre_lift(width_mm,speed_mm_s)
        try:
            self._control_event()
        except Exception as exc:
            # Retain the logging fault, but don't prevent a separately guarded
            # pre-lift release merely because its diagnostic log cannot flush.
            self._fault = self._fault or "Emergency control event recording: " + str(exc)

    def emergency_gripper_snapshot(self):
        grip = self.gripper.snapshot()
        return {"timestamp_ns":grip["monotonic_ns"], "gripper_timestamp_ns":grip["monotonic_ns"],
                "width_mm":grip["width_mm"],
                "gripper_fault":not grip.get("fresh") or not grip.get("valid")}

    def set_force_target(self, per_finger_N):
        self._require_zero_ready()
        self._check_ownership()
        if self.existing_force is None:
            self.gripper.set_force_target(per_finger_N)
        elif float(per_finger_N) == 0:
            self.existing_force.exit_force_mode()
        else:
            self.existing_force.set_target(per_finger_N)
        self._control_event()

    def set_gripper_min_width(self, width_mm):
        self.gripper.set_gripper_min_width(width_mm)

    def exit_force_mode(self):
        self._require_zero_ready()
        if self.existing_force is not None:
            self.existing_force.exit_force_mode()
        else:
            self.gripper.exit_force_mode()
        self._control_event()

    def hold_gripper(self):
        self._require_zero_ready()
        if self.existing_force is not None:
            self.existing_force.exit_force_mode()
        else:
            self.gripper.hold_gripper()
        self._control_event()

    def control_metrics(self):
        return {"arm_interpolation": self.arm.loop_metrics() if hasattr(self.arm,"loop_metrics") else {"count":0},
                "force_control": (self.gripper.loop_metrics() if self.existing_force is None and hasattr(self.gripper,"loop_metrics") else
                    {"count":0,"mean_s":None,"p95_s":None,"max_s":None,
                     "unavailable":"Unchanged existing CLI does not expose loop timing; CSV event intervals are not control periods"})}

    def provenance(self):
        return {"profile":PROFILE, "arm_side":self.side,
            "motion":"public ROS pose stream, software interpolation and measured-pose hold",
            "hardware_stop_verified":False, "hardware_watchdog_verified":False,
            "current_A_available":False, "velocity_source":"received_position_finite_difference",
            "sensor_provenance":dict(self.sensors.provenance),
            "startup_software_zero":self.sensors.startup_zero_report,
            "zero_ready":self._zero_ready,
            "force_unit":force_unit(self.cfg),"force_definition":target_definition(self.cfg),
            "force_controller":self.cfg["force"].get("controller","bridge_loop"),
            "existing_controller_evidence":None if self.existing_force is None else self.existing_force.events,
            "force_controller_source_sha256":self.gripper.snapshot().get("legacy_source_sha256"),
            "unrecorded_idle_events_discarded":self._idle_discarded}

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2)
        # No open/hold/pose command is implicit in transport cleanup.
        errors = []
        for component in (self.existing_force,self.gripper,self.passive_joints,self.arm,self.sensors):
            if component is None:
                continue
            try:
                component.close()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Transport cleanup: "+"; ".join(errors))


def create(cfg):
    return PublicRosBackend(cfg)
