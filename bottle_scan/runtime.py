"""Guarded contact/visual state machines; a hardware bridge is injected explicitly."""
from __future__ import annotations
from collections import deque
from contextlib import contextmanager
import json
import threading
import time
from pathlib import Path
import numpy as np
import h5py
from scipy.spatial.transform import Rotation
from .geometry import check_pose, check_segment, translate_local, lift_base, validate_transform, interpolate, pose_error
from .recording import H5Writer, JpegWriter, AsyncRecorder
from .sensing import TemplateGuard, dd_descriptors
from .session import atomic_json, plain
from .force_control import (force_value, force_unit, target_definition, in_target_band,
    target_as_mean, normal_stream_name, feedback_target, validate_control_target)


STAGES = ["MOVE_PRE", "APPROACH", "CLOSE", "PROBE", "SQUEEZE", "LIFT", "HOLD", "LOWER", "RELEASE", "RETREAT", "CHECK", "LABEL", "VISUAL_MOVE", "VISUAL_CAPTURE", "BASELINE"]


class SafetyFault(RuntimeError):
    pass


class SkippedPoint(RuntimeError):
    pass


class Acquisition:
    """Drain independently timestamped native events, never synthesize missing frames.

    Real bridges acquire in their own threads. Their drain_samples() must be
    nonblocking and return every acquired frame (or report a data-loss fault).
    Source draining continues while the operator is answering a prompt; disk
    compression and persistence belong to a separate writer process.
    """
    def __init__(self, backend, path):
        self.backend = backend
        self.writer = AsyncRecorder(path, backend.cfg)
        self.h5, self.jpg = self.writer.h5, self.writer.jpg
        self.error = None
        self.stop = threading.Event()
        self.thread = None
        try:
            if hasattr(backend, "begin_acquisition"):
                backend.begin_acquisition()
            if not backend.simulated:
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
        except BaseException as exc:
            self.error = exc
            try:self.writer.close(error=exc)
            except Exception:pass
            raise

    def _run(self):
        try:
            while not self.stop.wait(.005):
                self.pump()
        except BaseException as exc:
            self.error = exc
            self.writer.fail(exc)

    def pump(self):
        try:
            self.writer.check()
            for sample in self.backend.drain_samples():
                if sample.stream.startswith("camera/"):
                    metadata = dict(sample.metadata or {})
                    session = getattr(self.backend, "recording_session", None)
                    associated = all(key in metadata for key in ("T_base_leftTCP", "T_base_rightTCP", "B_base_mm", "pose_sources"))
                    if session is not None and not associated:
                        from .park_guard import image_pose_metadata
                        metadata.update(image_pose_metadata(self.backend.snapshot(), session, camera_stamp_ns=sample.monotonic_ns))
                    self.jpg.append(sample.stream.split("/", 1)[1], sample.data, sample.monotonic_ns,
                                    sample.device_timestamp, metadata)
                else:
                    self.h5.append(sample.stream, sample.data, sample.monotonic_ns,
                                   sample.device_timestamp, sample.metadata)
        except BaseException as exc:
            self.error = self.error or exc
            self.writer.fail(exc)
            raise

    def close(self):
        self.stop.set()
        try:
            if self.thread:
                self.thread.join(timeout=3)
                if self.thread.is_alive():
                    raise SafetyFault("Acquisition thread failed to stop")
            if hasattr(self.backend, "end_acquisition"):
                self.backend.end_acquisition()
            if self.error is None:
                self.pump()
        except BaseException as exc:
            self.error = self.error or exc
        finally:
            self.writer.close(error=self.error)


class Runner:
    def __init__(self, session, backend, *, single_step=True, unattended=False):
        self.session, self.backend, self.cfg = session, backend, session.cfg
        self.backend.recording_session = session
        if hasattr(self.backend, "bind_session"):
            self.backend.bind_session(session)
        if unattended and not backend.simulated:
            raise ValueError("Unattended mode is simulation-only")
        self.single_step, self.unattended = single_step, unattended
        self.airborne = False
        self.grasp_pose = None
        self.closure_min_width = None
        self.active_motion = None
        self.actuation_started = False
        self.emergency_logging = False
        self.logging_failed = False
        self.recorder = None
        self.stage = "IDLE"
        self.stage_started_ns = self.now()
        self.meta = None
        self.dt = 1 / float(self.cfg.get("recording", {}).get("control_hz", 20))
        self.threshold = np.array(session.meta.get("contact_threshold", session.meta.get("contact_threshold_N", [force_value(self.cfg, "contact_min")]*2)))
        self.guard = TemplateGuard.load(session.path / "references" / "bottle_position") if (session.path / "references" / "bottle_position").exists() else None
        self.reference = None
        if session.meta.get("references"):
            self.reference = np.load(session.path / session.meta["references"][-1]["path"])

    def now(self):
        return self.backend.monotonic_ns()

    def ask(self, message):
        if self.unattended:
            return ""
        return input(message)

    @contextmanager
    def phase(self, name, *, emergency=False):
        previous_logging = self.emergency_logging
        self.emergency_logging = previous_logging or emergency
        try:
            if self.single_step and not self.emergency_logging:
                self.ask(f"[{name}] 确认现场安全后按回车；Ctrl-C 中止：")
            self.stage = name
            self.stage_started_ns = self.now()
            self.backend.set_stage(name)
            record = {"name": name, "start_ns": self.now(), "end_ns": None, "status": "running"}
            if self.meta is not None:
                self.meta["stages"].append(record)
            if self.recorder:
                self._log(lambda: self.recorder.h5.append("stage", np.array(STAGES.index(name), np.int16), self.now(),
                          attrs={"stage": name, "event": "start"}), "stage_start")
            try:
                yield
                record["status"] = "ok"
            except BaseException as exc:
                record["status"] = "exception"
                record["error"] = str(exc)
                raise
            finally:
                record["end_ns"] = self.now()
                if self.recorder:
                    self._log(lambda: self.recorder.h5.append("stage", np.array(STAGES.index(name), np.int16), self.now(),
                              attrs={"stage": name, "event": "end"}), "stage_end")
                if self.meta and self.recorder:
                    self._log(lambda: self.recorder.writer.json("meta.json", self.meta),
                              "trial_metadata")
        finally:
            self.emergency_logging = previous_logging

    def _note_logging_error(self, error, operation):
        self.logging_failed = True
        if self.meta is not None:
            self.meta["logging_error_count"] = self.meta.get("logging_error_count", 0)+1
            errors = self.meta.setdefault("logging_errors", [])
            if len(errors) < 50:
                errors.append({"operation": operation, "error": str(error), "stage": self.stage})

    def _log(self, action, operation, *, emergency=False):
        try:
            action()
            return True
        except Exception as exc:
            if not (emergency or self.emergency_logging):
                raise
            self._note_logging_error(exc, operation)
            return False

    def tick(self, *, noncontact=False):
        if self.backend.simulated:
            # Synthetic virtual time can otherwise generate minutes of frames
            # in milliseconds. A fixed simulation-only pace lets the real disk
            # worker keep up; it is not queue backpressure or a timing test.
            if self.recorder:
                speedup = float(self.cfg.get("simulation", {}).get("max_speedup", 2.))
                if not np.isfinite(speedup) or speedup <= 0:
                    raise ValueError("simulation.max_speedup must be positive")
                time.sleep(self.dt / speedup)
            self.backend.step(self.dt)
            if self.recorder:
                self._log(self.recorder.pump, "native_stream_write")
            else:
                self.backend.drain_samples()
        else:
            time.sleep(self.dt)
        if self.recorder and self.recorder.error and not self.emergency_logging:
            raise SafetyFault(f"Recording failed: {self.recorder.error}")
        snap = self._healthy_snapshot()
        if noncontact and np.any(np.asarray(snap["normal_N"]) > self.threshold):
            raise SafetyFault("Unexpected contact during open-gripper motion")
        if self.now() - self.stage_started_ns > self.timeout()*1e9:
            raise SafetyFault(f"{self.stage} timeout")
        return snap

    def _healthy_snapshot(self):
        """Validate health before actuator commands as well as after each tick."""
        if self.recorder:
            try:
                self.recorder.writer.check()
            except Exception as error:
                self.recorder.error = error
        if self.recorder and self.recorder.error:
            if not self.emergency_logging:
                raise SafetyFault(f"Recording failed: {self.recorder.error}")
            self._note_logging_error(self.recorder.error, "acquisition_thread")
        snap = self.backend.snapshot()
        if snap is None:
            raise SafetyFault("No synchronized health snapshot")
        age_ns = self.now() - int(snap["timestamp_ns"])
        if age_ns < 0 or age_ns > int(self.cfg.get("recording", {}).get("stale_timeout_s", .25)*1e9):
            raise SafetyFault("Stale robot/tactile state")
        if snap.get("fault"):
            raise SafetyFault(str(snap["fault"]))
        normal = np.asarray(snap["normal_N"], float)
        if normal.shape != (2,) or not np.isfinite(normal).all():
            raise SafetyFault("Invalid per-finger normal force")
        if self.meta is not None:
            auto = self.meta.setdefault("auto", {})
            auto["force_unit"] = force_unit(self.cfg)
            auto["target_definition"] = target_definition(self.cfg)
            auto["max_force"] = np.maximum(auto.get("max_force", normal), normal).tolist()
            if any(x is not None for x in auto.get("contact_times_ns", [])):
                auto["contact_peak"] = np.maximum(auto.get("contact_peak", normal), normal).tolist()
        if np.any(normal > force_value(self.cfg, "max")):
            raise SafetyFault("F_max exceeded")
        for field in ("width_mm", "velocity_mm_s", "current_A"):
            # The user-selected public ROS state exposes no measured current.
            # Preserve an explicit missing datum, rather than fabricate amperes.
            # Width and velocity are still mandatory; a provided invalid current
            # cannot use this exception unless its availability is false.
            if (field == "current_A" and self.cfg.get("hardware", {}).get("profile") == "public_ros"
                    and snap.get("current_A_available") is False
                    and (snap.get(field) is None or np.isnan(snap.get(field)))):
                continue
            if not np.isfinite(snap.get(field, float("nan"))):
                raise SafetyFault(f"Invalid gripper state: {field}")
        if self.meta is not None and snap.get('width_is_estimate'):
            self.meta.setdefault('auto',{})['compressed_aperture_estimated']=True
            self.meta['auto']['width_semantics']='equivalent_aperture_model_mm_not_loaded_gap_measurement'
        if snap["width_mm"] < self.cfg["gripper"]["min_mm"] - .05:
            raise SafetyFault("w_min exceeded")
        if snap["width_mm"] > self.cfg["gripper"]["open_mm"] + .5:
            raise SafetyFault("Open width limit exceeded")
        try:
            validate_transform(snap["tcp"])
        except (KeyError, ValueError) as exc:
            raise SafetyFault("Invalid measured TCP pose") from exc
        if self.active_motion is not None:
            start, target = self.active_motion
            measured = np.asarray(snap["tcp"])
            displacement = target[:3, 3]-start[:3, 3]
            denominator = float(displacement @ displacement)
            if denominator > 1e-10:
                fraction = float(np.clip(np.dot(measured[:3, 3]-start[:3, 3], displacement)/denominator, 0, 1))
            else:
                angle = Rotation.from_matrix(start[:3, :3].T @ target[:3, :3]).magnitude()
                traveled = Rotation.from_matrix(start[:3, :3].T @ measured[:3, :3]).magnitude()
                fraction = float(np.clip(traveled/max(angle, 1e-12), 0, 1)) if angle > 1e-10 else 1.
            expected_pose = interpolate(start, target, fraction)
            position_error = np.linalg.norm(expected_pose[:3, 3]-measured[:3, 3])
            angle_error = np.degrees(Rotation.from_matrix(expected_pose[:3, :3].T @ measured[:3, :3]).magnitude())
            if position_error > self.cfg["motion"].get("pose_tolerance_mm", 1.) or angle_error > self.cfg["motion"].get("pose_tolerance_deg", 1.):
                raise SafetyFault("Measured TCP departed from certified straight/Slerp path")
        wrench = np.asarray(snap.get("wrench"))
        if wrench.shape != (2, 6) or not np.isfinite(wrench).all():
            raise SafetyFault("Invalid raw wrench pair")
        expected = tuple(self.cfg["tactile"]["dd_shape"])
        dd = snap.get("dd", [])
        if len(dd) != 2 or any(np.asarray(frame).shape != expected or not np.isfinite(frame).all() for frame in dd):
            raise SafetyFault("Invalid raw DD shape or nonfinite values")
        if self.stage in ("CLOSE", "PROBE", "SQUEEZE", "LIFT", "HOLD", "LOWER"):
            bottle_pose = self.session.T0
            if self.airborne and self.grasp_pose is not None:
                bottle_pose = np.asarray(snap["tcp"]) @ np.linalg.inv(self.grasp_pose) @ self.session.T0
            errors = check_pose(np.asarray(snap["tcp"]), self.cfg, bottle_pose,
                                width_mm=float(snap["width_mm"]), allow_contact=True)
            if errors:
                raise SafetyFault("Actual contact geometry invalid: " + "; ".join(errors))
            if self.closure_min_width is not None and snap["width_mm"] < self.closure_min_width - .05:
                raise SafetyFault("Dynamic geometric gripper width floor exceeded")
        return snap

    def _command_force(self, target):
        self._healthy_snapshot()
        try:
            validate_control_target(self.cfg, target)
        except ValueError as exc:
            raise SafetyFault("Force target outside source-specific limits: " + str(exc)) from exc
        self.actuation_started = True
        self.backend.set_force_target(float(target))

    def _closure_floor(self, snap):
        """Certify continuous jaw-width sweep using midpoint boxes plus a bound.

        Across a width interval dw, either finger travels dw/2. Inflating its
        midpoint box by dw/4 contains the full continuous swept finger volume.
        This changes a safety width limit, never the requested TCP target pose.
        """
        width = float(snap["width_mm"])
        minimum = float(self.cfg["gripper"]["min_mm"])
        step = float(self.cfg["gripper"].get("geometry_width_step_mm", 1.))
        if not np.isfinite(step) or not 0 < step <= 2.:
            raise SafetyFault("geometry_width_step_mm must be in (0,2]")
        while width > minimum + 1e-9:
            candidate = max(minimum, width-step)
            midpoint = (width+candidate)/2
            errors = check_pose(np.asarray(snap["tcp"]), self.cfg, self.session.T0,
                                width_mm=midpoint, allow_contact=True,
                                extra_margin_mm=(width-candidate)/4)
            if errors:
                return width, errors
            width = candidate
        return minimum, []

    def timeout(self):
        return float(self.cfg["motion"].get("stage_timeouts_s", {}).get(self.stage, 30.))

    def wait(self, seconds, *, noncontact=False):
        until = self.now() + int(seconds*1e9)
        snapshots = []
        while self.now() < until:
            snapshots.append(self.tick(noncontact=noncontact))
        return snapshots

    def ensure_open(self):
        if self.airborne:
            raise SafetyFault("Refusing to open while airborne/recovery is latched")
        self._healthy_snapshot()
        self.actuation_started = True
        self.backend.exit_force_mode()
        self.backend.command_open(self.cfg["gripper"]["open_mm"], self.cfg["gripper"]["close_speed_mm_s"])
        while True:
            snap = self.tick(noncontact=False)
            if abs(snap["width_mm"] - self.cfg["gripper"]["open_mm"]) <= self.cfg["gripper"].get("open_tolerance_mm", .5):
                break
        if np.any(np.asarray(snap["normal_N"]) > self.threshold):
            raise SafetyFault("Open width reached but tactile contact remains")

    def move(self, target, *, speed=None, allow_contact=False, noncontact=False, stop_when=None):
        target = np.asarray(target, float)
        snap = self._healthy_snapshot()
        attached = self.grasp_pose if self.airborne and self.stage in ("LIFT", "LOWER") else None
        errors = check_segment(np.asarray(snap["tcp"]), target, self.cfg, self.session.T0,
                               width_mm=snap["width_mm"], allow_contact=allow_contact,
                               attached_grasp_pose=attached)
        if errors:
            raise SkippedPoint("; ".join(errors))
        motion = self.cfg["motion"]
        # Conservative policy: near speed for the entire segment if either
        # endpoint or swept straight segment comes within 50 mm of T_target.
        reference = np.asarray(self.meta.get("T_target", self.session.T0)) if self.meta else self.session.T0
        p, q, r = snap["tcp"][:3, 3], target[:3, 3], reference[:3, 3]
        v = q-p
        closest = p + np.clip(np.dot(r-p, v)/max(np.dot(v, v), 1e-12), 0, 1)*v
        limit = motion["near_speed_mm_s"] if np.linalg.norm(closest-r) <= motion.get("near_distance_mm", 50) else motion["free_speed_mm_s"]
        commanded = min(float(speed or limit), limit)
        if not np.isfinite(commanded) or commanded <= 0:
            raise SafetyFault("Motion speed must be finite and positive")
        latest = self._healthy_snapshot()
        start_shift = np.linalg.norm(np.asarray(latest["tcp"])[:3, 3]-np.asarray(snap["tcp"])[:3, 3])
        start_rotation = np.degrees(Rotation.from_matrix(np.asarray(snap["tcp"])[:3, :3].T @ np.asarray(latest["tcp"])[:3, :3]).magnitude())
        if start_shift > motion.get("pose_tolerance_mm", 1.) or start_rotation > motion.get("pose_tolerance_deg", 1.):
            raise SafetyFault("TCP changed while the path was being validated; command cancelled")
        self.active_motion = (np.array(snap["tcp"], copy=True), target.copy())
        try:
            self.actuation_started = True
            self.backend.move_linear(target, commanded, motion.get("angular_speed_deg_s", 5.))
            if self.recorder:
                self._log(lambda: self.recorder.h5.append("tcp_command_event", target, self.now(),
                          attrs={"stage": self.stage, "speed_mm_s": commanded}), "tcp_command_event")
            while not self.backend.motion_done():
                sample = self.tick(noncontact=noncontact)
                if stop_when is not None and stop_when(sample):
                    self.backend.stop_arm()
                    return False
            final_sample = self._healthy_snapshot()
            if stop_when is not None and stop_when(final_sample):
                self.backend.stop_arm()
                return False
            measured = np.asarray(final_sample["tcp"])
        finally:
            self.active_motion = None
        poserr = np.linalg.norm(measured[:3, 3]-target[:3, 3])
        roterr = np.degrees(Rotation.from_matrix(measured[:3, :3].T @ target[:3, :3]).magnitude())
        if poserr > motion.get("pose_tolerance_mm", 1.) or roterr > motion.get("pose_tolerance_deg", 1.):
            raise SafetyFault(f"Motion completion outside tolerance: {poserr:.3f} mm, {roterr:.3f} deg")
        return True

    def check_bottle(self, tag="after_retreat"):
        if self.guard is None:
            raise SafetyFault("Session has no immutable wrist reference")
        if "nonexperiment_arm_park" in self.session.meta or not self.backend.simulated:
            from .park_guard import ParkGuard
            park = ParkGuard(self.session).check(self.backend.park_observation(), context=tag, now_ns=self.now())
            if self.meta is not None:
                self.meta.setdefault("park_checks", []).append(park)
        target = translate_local(self.session.T0, -self.cfg["check"].get("retreat_mm", 80), self.cfg["axes"]["a"])
        if tag == "run_start":
            self.confirm_startup_move(target)
        with self.phase("CHECK"):
            # Reject an unsafe return before even changing gripper targets.
            snap = self._healthy_snapshot()
            reasons = check_segment(np.asarray(snap["tcp"]), target, self.cfg, self.session.T0,
                                    width_mm=snap["width_mm"])
            if reasons:
                raise SkippedPoint("T_check path rejected; remain at current position: " + "; ".join(reasons))
            self.ensure_open()
            self.move(target, noncontact=True)
            self.wait(self.cfg["motion"].get("settle_s", .3), noncontact=True)
            attempt = 0
            while True:
                if "nonexperiment_arm_park" in self.session.meta or not self.backend.simulated:
                    from .park_guard import ParkGuard
                    park = ParkGuard(self.session).check(self.backend.park_observation(), context=tag+"_capture", now_ns=self.now())
                    if self.meta is not None:
                        self.meta.setdefault("park_checks", []).append(park)
                samples = self.backend.image_samples()
                images = {name: sample.data for name, sample in samples.items()}
                result = self.guard.compare(images)
                folder = self.session.path / "checks" / f"{self.now()}_{tag}_{attempt}"
                folder.mkdir()
                with AsyncRecorder(folder, self.cfg) as writer:
                    from .park_guard import image_pose_metadata
                    for camera, sample in samples.items():
                        attrs = dict(sample.metadata or {})
                        if "pose_sources" not in attrs:
                            attrs.update(image_pose_metadata(self.backend.snapshot(), self.session, camera_stamp_ns=sample.monotonic_ns))
                        attrs.update(image_encoding="keyframe_jpeg_from_rgb", image_timestamp_kind="source_received_monotonic_ns")
                        writer.jpg.append(camera, sample.data, sample.monotonic_ns,
                            sample.device_timestamp, attrs=attrs)
                    writer.json("result.json", result)
                if self.meta is not None:
                    self.meta.setdefault("position_checks", []).append({"path": str(folder.relative_to(self.session.path)), **result})
                    self.meta.setdefault("auto", {})["check_displacements"] = result
                if result["passed"]:
                    return
                print("瓶位检查未通过：", json.dumps(result, ensure_ascii=False))
                if self.unattended:
                    raise SafetyFault("Synthetic bottle check failed; immutable reference retained")
                self.ask("请把瓶子复位到标记位置，再按回车重拍。参考图不会更新：")
                # Human reset has no stage timeout; restart only the observation window.
                self.stage_started_ns = self.now()
                self.wait(self.cfg["motion"].get("settle_s", .3), noncontact=True)
                attempt += 1

    def confirm_startup_move(self, target):
        """One concrete MOVE confirmation before any startup actuator command."""
        initial = self._healthy_snapshot()
        start = np.array(initial["tcp"], dtype=float, copy=True)
        reasons = check_segment(start, target, self.cfg, self.session.T0,
                                width_mm=initial["width_mm"])
        report = {"start_tcp_mm":start.tolist(), "target_T_check_mm":np.asarray(target).tolist(),
                  "path_check_passed":not reasons, "path_check_reasons":reasons,
                  "simulated":bool(self.backend.simulated)}
        print("首次移动预览：" + json.dumps(report, ensure_ascii=False))
        if reasons:
            raise SkippedPoint("Startup T_check path rejected: " + "; ".join(reasons))
        if not self.backend.simulated and self.ask("首次移动到 T_check；输入 MOVE 执行，其他输入取消：").strip() != "MOVE":
            raise SafetyFault("Initial MOVE not confirmed; no actuator command sent")
        latest = self._healthy_snapshot()
        delta, angle = pose_error(start, latest["tcp"])
        if delta > self.cfg["motion"]["pose_tolerance_mm"] or angle > self.cfg["motion"]["pose_tolerance_deg"]:
            raise SafetyFault("TCP changed during initial MOVE confirmation; restart and inspect new path")
        # Human confirmation is outside the bounded CHECK stage clock.
        self.session.meta.setdefault("startup_moves", []).append(report)
        self.session.save_meta()

    def baseline(self):
        if self.recorder is not None:
            raise SafetyFault("Baseline requires its own independent acquisition record")
        old_meta, self.meta = self.meta, None
        try:
            with self.phase("BASELINE"):
                self.ensure_open()
                stamp = self.now()
                folder = self.session.path / "references" / f"unloaded_{stamp}"
                folder.mkdir()
                path, raw_path = folder / "mean.npz", folder / "streams.h5"
                self.backend.drain_samples()
                self.recorder = Acquisition(self.backend, folder)
                self.recorder.h5.append("stage", np.array(STAGES.index("BASELINE"), np.int16), self.now(),
                                        attrs={"stage": "BASELINE", "event": "start"})
                self.wait(float(self.cfg.get("tactile", {}).get("baseline_s", 10)), noncontact=True)
        finally:
            try:
                if self.recorder:
                    self.recorder.close()
            finally:
                self.recorder, self.meta = None, old_meta
        means, counts = {}, {}
        with h5py.File(raw_path, "r") as raw:
            for side in ("left", "right"):
                for kind in ("dd", "wrench"):
                    name = kind + "_" + side
                    if name not in raw["streams"]:
                        raise SafetyFault(f"Missing native baseline stream {name}")
                    group = raw["streams"][name]
                    count = int(group.attrs["committed_count"])
                    if count < 2:
                        raise SafetyFault(f"Insufficient independent baseline frames for {name}")
                    total = np.zeros(group["data"].shape[1:], dtype=np.float64)
                    for i in range(count):
                        total += group["data"][i]
                    means[side + "_" + kind] = total/count
                    if kind == "dd":
                        means[side + "_dd_std"] = np.std(group["data"][:count], axis=0)
                    counts[name] = count
            normal_name = normal_stream_name(self.cfg)
            if normal_name not in raw["streams"]:
                raise SafetyFault("Baseline needs native " + normal_name + " stream for noise threshold")
            normals = raw["streams/" + normal_name]
            count = int(normals.attrs["committed_count"])
            values = normals["data"][:count]
            if values.shape != (count, 2) or count < 2 or not np.isfinite(values).all():
                raise SafetyFault("Invalid native calibrated normal force baseline")
            sigma = np.std(values, axis=0)
            np.savez_compressed(path, **means)
            self.threshold = np.maximum(5*sigma, force_value(self.cfg, "contact_min"))
            self.session.meta["contact_threshold"] = self.threshold.tolist()
            self.session.meta["force_unit"] = force_unit(self.cfg)
            if force_unit(self.cfg) == "N":
                self.session.meta["contact_threshold_N"] = self.threshold.tolist()
            self.session.meta["references"].append({"path": str(path.relative_to(self.session.path)), "monotonic_ns": stamp,
                "raw_path": str(raw_path.relative_to(self.session.path)), "native_frame_counts": counts})
            self.session.save_meta()
            self.reference = np.load(path)

    def recover(self, error, pre):
        """Airborne is latched BEFORE the first lift command, never after return."""
        self.meta["auto"]["exceptions"].append({"stage": self.stage, "error": str(error), "airborne": self.airborne})
        stop_error = None
        try:
            self.backend.stop_arm()
        except BaseException as exc:
            stop_error = exc
            self.meta["auto"]["exceptions"].append({"stage": "EMERGENCY_STOP", "error": str(exc), "airborne": self.airborne})
        if self.airborne:
            self.session.progress["requires_operator_recovery"] = True
            # A failed arm-stop RPC must not suppress the independent grip hold.
            # Disk persistence comes only AFTER both emergency hardware actions.
            hold_error = None
            try:
                self.backend.hold_gripper()
            except Exception as exc:
                hold_error = exc
                self.meta["auto"]["exceptions"].append({"stage": "EMERGENCY_GRIP_HOLD", "error": str(exc), "airborne": True})
            persisted = self._log(self.session.save_progress, "airborne_recovery_latch", emergency=True)
            if hold_error is not None:
                raise SafetyFault("Airborne grip hold could not be confirmed; operator intervention required") from hold_error
            if stop_error is not None:
                print("\a空中异常：机械臂停止未能确认，已请求夹爪保持；请立即人工接管，不会自动松开或后撤。")
                raise SafetyFault("Arm stop could not be confirmed; gripper hold requested, operator recovery required") from stop_error
            print("\a空中异常：机械臂停止，夹爪保持当前开度；请人工接管。不会自动松开或后撤。")
            if not persisted:
                raise SafetyFault("Arm stopped/grip held, but recovery latch could not be persisted; do not resume")
            return
        if stop_error is not None:
            self.meta["status"] = "aborted"
            self.meta["auto"]["recovery_succeeded"] = False
            self.session.progress["requires_operator_recovery"] = True
            self._log(self.session.save_progress, "failed_stop_recovery_latch", emergency=True)
            raise SafetyFault("Arm stop could not be confirmed; release/retreat refused") from stop_error
        try:
            with self.phase("RELEASE", emergency=True):
                self.emergency_release_pre_lift(already_stopped=True)
            with self.phase("RETREAT", emergency=True):
                self.move(pre, noncontact=True)
            if self.logging_failed:
                raise SafetyFault("Safe release/retreat completed but data logging failed; sequence cannot continue")
            self.check_bottle("recovery_retreat")
            self.meta["auto"]["recovery_succeeded"] = True
        except BaseException as recovery_error:
            try:
                self.backend.stop_arm()
            except BaseException as stop_failure:
                self.meta["auto"]["exceptions"].append({"stage": "RECOVERY_STOP", "error": str(stop_failure), "airborne": False})
            self.meta["status"] = "aborted"
            self.meta["auto"]["recovery_succeeded"] = False
            self.meta["auto"]["exceptions"].append({"stage": "RECOVERY", "error": str(recovery_error), "airborne": False})
            self.session.progress["requires_operator_recovery"] = True
            self._log(self.session.save_progress, "failed_recovery_latch", emergency=True)
            print("\a安全后撤未通过或执行失败；已停止，等待人工处理。")
            raise SafetyFault("Pre-lift recovery failed; sequence terminated for operator recovery") from recovery_error

    def emergency_release_pre_lift(self, *, already_stopped=False):
        """After verified arm stop, open despite failed tactile/DD health.

        Only the explicit open width/speed and fresh gripper feedback are used.
        No arm retreat is issued here. Invalid gripper feedback aborts recovery;
        an airborne or persisted recovery latch always refuses this operation.
        """
        if self.airborne or self.session.progress.get("requires_operator_recovery"):
            raise SafetyFault("Emergency opening forbidden while airborne/recovery-latched")
        if not already_stopped:
            self.backend.stop_arm()
        width = float(self.cfg["gripper"]["open_mm"])
        speed = float(self.cfg["gripper"]["close_speed_mm_s"])
        if not np.isfinite(width) or not np.isfinite(speed) or width <= self.cfg["gripper"]["min_mm"] or speed <= 0:
            raise SafetyFault("Emergency release requires verified open width and positive speed")
        if hasattr(self.backend, "emergency_open_pre_lift"):
            self.backend.emergency_open_pre_lift(width, speed)
        else:
            self.backend.exit_force_mode()
            self.backend.command_open(width, speed)
        timeout = float(self.cfg["motion"].get("stage_timeouts_s", {}).get("RELEASE", 30.))
        deadline = self.now()+int(timeout*1e9)
        while self.now() < deadline:
            if self.backend.simulated:
                self.backend.step(self.dt)
                if self.recorder:
                    try:
                        self.recorder.pump()
                    except Exception as exc:
                        self.recorder.error = exc
                        self._note_logging_error(exc, "emergency_native_stream_write")
                else:
                    self.backend.drain_samples()
            else:
                time.sleep(self.dt)
            snap = (self.backend.emergency_gripper_snapshot()
                    if hasattr(self.backend, "emergency_gripper_snapshot") else self.backend.snapshot())
            if snap is None:
                raise SafetyFault("Emergency opening has no gripper feedback; retreat forbidden")
            timestamp = snap.get("gripper_timestamp_ns", snap.get("timestamp_ns"))
            if timestamp is None:
                raise SafetyFault("Emergency opening has no gripper clock; retreat forbidden")
            age = self.now()-int(timestamp)
            if age < 0 or age > self.cfg.get("recording", {}).get("stale_timeout_s", .25)*1e9:
                raise SafetyFault("Emergency opening feedback is stale; retreat forbidden")
            measured = float(snap.get("width_mm", float("nan")))
            if not np.isfinite(measured) or measured < self.cfg["gripper"]["min_mm"]-.05 or measured > width+.5 or snap.get("gripper_fault"):
                raise SafetyFault("Emergency opening feedback invalid; retreat forbidden")
            if abs(measured-width) <= self.cfg["gripper"].get("open_tolerance_mm", .5):
                return snap
        raise SafetyFault("Emergency opening timed out; retreat forbidden")

    def contact_trial(self, item, *, scenario="normal"):
        if self.airborne or self.session.progress.get("requires_operator_recovery"):
            raise SafetyFault("Operator recovery latch prevents starting another trial")
        trial_id = str(item["trial_id"])
        folder = self.session.start_trial(trial_id)
        delta = item["delta_acl_mm"]
        self.meta = {"trial_id": trial_id, "task": "contact", "bottle": item["bottle"], "kind": item["kind"],
            "offsets": {"a_mm": delta[0], "c_mm": delta[1], "l_mm": delta[2],
                        "rot_c_deg": item["angle_deg"] if item.get("rotation_axis") == "c" else 0,
                        "rot_other_deg": item["angle_deg"] if item.get("rotation_axis") not in (None,"c") else 0},
            "T_target": item["T_target"], "T_pre": item["T_pre"], "stages": [], "status": "running",
            "simulated": self.backend.simulated, "auto": {"bilateral_contact": False, "force_reached": False,
                "contact_times_ns": [None, None], "contact_widths_mm": [None, None],
                "dropped": False, "slip_suspected": False, "exceptions": [],
                "nominal_region": item.get("nominal_contact_region", "unknown")}, "operator": {"label": "", "note": ""}}
        self.airborne = False
        self.backend.set_trial(item, scenario=scenario)
        self.backend.drain_samples()
        self.recorder = Acquisition(self.backend, folder)
        target, pre = np.asarray(item["T_target"]), np.asarray(item["T_pre"])
        try:
            with self.phase("MOVE_PRE"):
                self.ensure_open()
                self.move(pre, noncontact=True)
            with self.phase("APPROACH"):
                self.move(target, speed=self.cfg["motion"]["near_speed_mm_s"], noncontact=True)
                self.wait(self.cfg["motion"].get("settle_s", .3), noncontact=True)
                translation, rotation = pose_error(target, self._healthy_snapshot()["tcp"])
                self.meta["auto"].update(approach_translation_error_mm=translation, approach_rotation_error_deg=rotation)
            bilateral = self.close_contact()
            reached = False
            if bilateral:
                self.probe()
                reached = self.squeeze()
            if reached:
                self.meta["auto"]["normal_pre_lift"] = self.backend.snapshot()["normal_N"].tolist()
                if force_unit(self.cfg) == "N":
                    self.meta["auto"]["normal_pre_lift_N"] = self.meta["auto"]["normal_pre_lift"]
                with self.phase("LIFT"):
                    self.grasp_pose = np.array(self._healthy_snapshot()["tcp"], dtype=float, copy=True)
                    self.airborne = True
                    self.session.progress["requires_operator_recovery"] = True
                    self.session.save_progress()
                    self.move(lift_base(self.grasp_pose, self.cfg["motion"]["lift_height_mm"]),
                              speed=self.cfg["motion"]["lift_speed_mm_s"], allow_contact=True)
                self.hold()
                self.lower()
            with self.phase("RELEASE"):
                self._command_force(0.)
                self.ensure_open()
            with self.phase("RETREAT"):
                self.move(pre, noncontact=True)
            self.check_bottle()
            with self.phase("LABEL"):
                self.label()
            self.meta["status"] = "completed"
        except SkippedPoint as exc:
            self.meta["status"] = "skipped_safety"
            self.meta["skip_reason"] = str(exc)
            self.recover(exc, pre)
            if self.airborne or self.session.progress.get("requires_operator_recovery"):
                self.meta["status"] = "aborted"
                raise SafetyFault("Airborne safety rejection: operator recovery required; sequence terminated") from exc
        except BaseException as exc:
            self.meta["status"] = "aborted"
            self.recover(exc, pre)
            atomic_json(folder / "meta.json", self.meta)
            raise
        finally:
            original_error = __import__("sys").exc_info()[1]
            persistence_error = None
            if self.recorder:
                try:
                    self.recorder.close()
                except BaseException as exc:
                    self._note_logging_error(exc, "acquisition_close")
                    self.meta["status"] = "aborted"
                    persistence_error = exc
                finally:
                    self.recorder = None
            if (folder / "streams.h5").exists():
                from .window_descriptors import enrich_trial_windows
                try:
                    enrich_trial_windows(self.meta, folder / "streams.h5", self.cfg, self.reference)
                    from .recording import finalize_gel_windows
                    finalize_gel_windows(folder, self.meta, self.cfg)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.meta["window_enrichment_error"] = str(exc)
                    self.meta["status"] = "aborted"
                    persistence_error = persistence_error or exc
            if hasattr(self.backend, "control_metrics"):
                self.meta["auto"]["control_loop_periods"] = self.backend.control_metrics()
            atomic_json(folder / "meta.json", self.meta)
            if persistence_error is not None and original_error is None:
                raise SafetyFault("Trial recording incomplete: " + str(persistence_error)) from persistence_error
        self.session.complete(self.meta)
        return self.meta

    def close_contact(self):
        with self.phase("CLOSE"):
            snap = self._healthy_snapshot()
            self.closure_min_width, limit_reasons = self._closure_floor(snap)
            compression_floor=self.closure_min_width
            precontact=self.cfg['gripper'].get('precontact_min_mm')
            if precontact is not None:
                self.closure_min_width=max(self.closure_min_width,float(precontact))
                self.meta['auto']['compression_geometry_floor_mm']=compression_floor
                self.meta['auto']['precontact_floor_mm']=self.closure_min_width
                self.meta['auto']['width_semantics']='equivalent_aperture_model_mm; below measured body contact is estimated'
            self.meta["auto"]["closure_min_width_mm"] = self.closure_min_width
            self.meta["auto"]["closure_geometry_limit_reasons"] = limit_reasons
            self._healthy_snapshot()
            self.backend.set_gripper_min_width(self.closure_min_width)
            self._healthy_snapshot()
            self.actuation_started = True
            self.backend.command_close(self.closure_min_width, self.cfg["gripper"]["close_speed_mm_s"])
            history = deque()
            while True:
                snap = self.tick()
                for i in range(2):
                    if snap["normal_N"][i] > self.threshold[i] and self.meta["auto"]["contact_times_ns"][i] is None:
                        self.meta["auto"]["contact_times_ns"][i] = int(snap["timestamp_ns"])
                        self.meta["auto"]["contact_widths_mm"][i] = float(snap["width_mm"])
                if np.all(np.asarray(snap["normal_N"]) > self.threshold):
                    self.backend.hold_gripper()
                    self.meta["auto"]["bilateral_contact"] = True
                    t = self.meta["auto"]["contact_times_ns"]
                    self.meta["auto"]["contact_time_difference_s"] = (t[1]-t[0])/1e9
                    if precontact is not None:
                        # Unlock only the already certified geometric interval,
                        # after a measured bilateral contact and measured hold.
                        self._healthy_snapshot()
                        self.closure_min_width=compression_floor
                        self.backend.set_gripper_min_width(compression_floor)
                        self.meta['auto']['compression_authorized_after_bilateral_contact']=True
                        self.meta['auto']['closure_min_width_mm']=compression_floor
                    return True
                if snap["width_mm"] <= self.closure_min_width + .05:
                    self.meta["auto"]["no_bilateral_contact"] = True
                    if precontact is not None and abs(self.closure_min_width-float(precontact))<=.05:
                        self.meta['auto']['closure_stop_reason']='measured_contact_floor_without_bilateral_contact'
                    elif self.closure_min_width > self.cfg["gripper"]["min_mm"] + .05:
                        self.meta["auto"]["safety_closure_limit"] = True
                        self.meta["auto"]["closure_stop_reason"] = "geometric_width_floor_without_bilateral_contact"
                    else:
                        self.meta["auto"]["closure_stop_reason"] = "configured_w_min_without_bilateral_contact"
                    self.backend.hold_gripper()
                    return False
                history.append((self.now(), float(snap["width_mm"])))
                window = self.cfg["gripper"].get("stall_window_s", .3)*1e9
                while len(history) > 1 and self.now() - history[1][0] >= window:
                    history.popleft()
                if self.now()-history[0][0] >= window and abs(history[-1][1]-history[0][1]) <= self.cfg["gripper"].get("stall_epsilon_mm", .05):
                    self.meta["auto"]["stall_without_contact"] = True
                    self.backend.hold_gripper()
                    return False

    def probe(self):
        """Require a continuous force band; timeout is a quality flag, not success."""
        with self.phase("PROBE"):
            settings = self.cfg.get("probe", {})
            force = self.cfg["force"]
            target = float(force.get("probe", force.get("probe_N", force["probe_fraction"] * force_value(self.cfg, "target"))))
            self._command_force(target)
            stable_start = None
            deadline = self.now() + int(float(settings.get("timeout_s", 3.)) * 1e9)
            tolerance = float(settings.get("tolerance", force_value(self.cfg, "target_tolerance")))
            while self.now() < deadline:
                snap = self.tick()
                if in_target_band(snap["normal_N"], feedback_target(self.cfg, "PROBE"), tolerance, target_definition(self.cfg)):
                    stable_start = self.now() if stable_start is None else stable_start
                    if self.now() - stable_start >= float(settings.get("settle_s", .2))*1e9:
                        self.meta["auto"].update(probe_not_settled=False,
                            probe_stable_window={"start_ns": stable_start, "end_ns": self.now()})
                        return True
                else:
                    stable_start = None
            self.meta["auto"].update(probe_not_settled=True, probe_stable_window=None)
            return False

    def squeeze(self):
        with self.phase("SQUEEZE"):
            if self.cfg["force"].get("squeeze_mode", "fixed") != "fixed":
                raise SafetyFault("First version supports fixed force only; learned ForcePolicy remains disabled")
            self._command_force(force_value(self.cfg, "target"))
            stable_start = None
            deadline = self.now() + int((self.timeout()-self.dt*2)*1e9)
            while self.now() < deadline:
                snap = self.tick()
                if in_target_band(snap["normal_N"], feedback_target(self.cfg, "SQUEEZE"), force_value(self.cfg, "target_tolerance"), target_definition(self.cfg)):
                    stable_start = self.now() if stable_start is None else stable_start
                    if self.now()-stable_start >= self.cfg["force"].get("stable_s", .5)*1e9:
                        self.meta["auto"]["force_reached"] = True
                        return True
                else:
                    stable_start = None
            self.meta["auto"]["force_not_reached"] = True
            return False

    def tangential_load(self, snap):
        """Magnitude used for unloading, in the selected force unit."""
        tactile = self.cfg["tactile"]
        axis = tactile["normal_axis"]
        forces = np.asarray(snap.get("wrench_force", np.asarray(snap["wrench"])[:, :3]), float)
        if forces.shape != (2, 3) or not np.isfinite(forces).all():
            raise SafetyFault("Invalid calibrated tangential force pair")
        tangents = forces.copy()
        tangents[:, axis] = 0.
        if self.cfg.get("lower", {}).get("vertical_mapping_confirmed", False):
            rotations = tactile["sensor_to_tcp"]
            for side in ("left", "right"):
                rotation = np.asarray(rotations.get(side), float)
                if rotation.shape != (3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1):
                    raise SafetyFault("Confirmed vertical mapping requires valid sensor_to_tcp rotations")
            vertical = sum(float((snap["tcp"][:3,:3] @ np.asarray(rotations[side]) @ tangents[i])[2])
                           for i, side in enumerate(("left", "right")))
            return abs(vertical), "absolute_vertical_sum"
        return float(np.linalg.norm(tangents, axis=1).sum()), "sum_of_tangential_magnitudes"

    def hold(self):
        tactile = self.cfg["tactile"]
        rotations = tactile.get("sensor_to_tcp")
        if not isinstance(rotations, dict) or set(rotations) != {"left", "right"}:
            raise SafetyFault("Explicit left/right sensor_to_tcp rotations are required")
        for rotation in rotations.values():
            if isinstance(rotation,str) and rotation == "unavailable":
                if self.cfg.get("lower",{}).get("vertical_mapping_confirmed"):
                    raise SafetyFault("Vertical mapping cannot use an unavailable sensor rotation")
                continue
            rotation = np.asarray(rotation, float)
            if rotation.shape != (3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation),1):
                raise SafetyFault("Invalid sensor_to_tcp rotation")
        pitch = tactile.get("pixel_pitch_mm")
        metric = "mm" if isinstance(pitch, (list, tuple)) else "px"
        with self.phase("HOLD"):
            initial, maximum = [None, None], [0., 0.]
            normals, loads = [], []
            end = self.now()+int(self.cfg["motion"].get("hold_s", 2)*1e9)
            while self.now() < end:
                snap = self.tick()
                normal = np.asarray(snap["normal_N"])
                normals.append(normal)
                if np.any(normal < self.threshold):
                    self.meta["auto"]["dropped"] = True
                for i, side in enumerate(("left", "right")):
                    baseline = None if self.reference is None else self.reference[side+"_dd"]
                    desc = dd_descriptors(snap["dd"][i], baseline,
                        depth_channel=tactile.get("depth_channel", 0), contact_threshold=tactile.get("dd_threshold", .01),
                        pixel_pitch_mm=pitch)
                    centroid = desc.get("centroid_" + metric)
                    if centroid is not None and np.isfinite(centroid).all():
                        if initial[i] is None:
                            initial[i] = np.array(centroid)
                        drift = float(np.linalg.norm(np.array(centroid)-initial[i]))
                        maximum[i] = max(maximum[i], drift)
                        if drift > float(tactile["centroid_drift_" + metric]):
                            self.meta["auto"]["slip_suspected"] = True
                load, method = self.tangential_load(snap)
                loads.append(load)
            mean = np.mean(normals, axis=0)
            self.meta["auto"].update(normal_hold=mean.tolist(), force_unit=force_unit(self.cfg),
                target_definition=target_definition(self.cfg),
                normal_change=(mean-self.meta["auto"].get("normal_pre_lift", mean)).tolist(),
                hold_vertical_tangent_sum=float(np.mean(loads)), lower_force_method=method,
                **{"max_centroid_drift_" + metric: maximum})
            if force_unit(self.cfg) == "N":
                for key in ("normal_hold", "normal_change", "hold_vertical_tangent_sum"):
                    self.meta["auto"][key + "_N"] = self.meta["auto"][key]

    def lower(self):
        """Never command below the original height; stop early when unloaded."""
        if self.grasp_pose is None:
            raise SafetyFault("LOWER requires measured pre-lift pose")
        original_z = float(self.grasp_pose[2,3])
        if float(self._healthy_snapshot()["tcp"][2,3]) < original_z-1e-6:
            self.backend.stop_arm()
            raise SafetyFault("LOWER starts below original height; operator recovery required")
        threshold = float(self.meta["auto"]["hold_vertical_tangent_sum"]) * float(self.cfg.get("lower", {}).get("unload_ratio", .2))
        detected = None
        with self.phase("LOWER"):
            def stop_on_unload(snap):
                nonlocal detected
                z = float(snap["tcp"][2,3])
                if z < original_z - 1e-6:
                    self.backend.stop_arm()
                    raise SafetyFault("LOWER measured below original height; operator recovery required")
                value, method = self.tangential_load(snap)
                if threshold > 0. and value <= threshold:
                    detected = {"height_mm": z, "force": value, "method": method,
                                "timestamp_ns": self.now()}
                    return True
                return False
            self.move(self.grasp_pose, speed=self.cfg["motion"]["lift_speed_mm_s"],
                      allow_contact=True, stop_when=stop_on_unload)
            for snap in self.wait(self.cfg["motion"].get("settle_s", .3)):
                if float(snap["tcp"][2,3]) < original_z-1e-6:
                    self.backend.stop_arm()
                    raise SafetyFault("LOWER overshot original height; keep recovery latch")
            final_z = float(self._healthy_snapshot()["tcp"][2,3])
            if final_z < original_z-1e-6:
                self.backend.stop_arm()
                raise SafetyFault("LOWER final feedback below original height; keep recovery latch")
            self.meta["auto"].update(lower_stop_reason="unloaded" if detected else "original_height",
                lower_unload=detected, lower_stop_height_mm=final_z,
                lower_unload_height_difference_mm=None if detected is None else detected["height_mm"]-original_z,
                lower_original_height_mm=original_z, lower_force_threshold=threshold)
            self.airborne = False
            self.session.progress["requires_operator_recovery"] = False
            self.session.save_progress()

    def label(self):
        while True:
            answer = self.ask("标注：回车正常，t倾倒 / m移位 / d掉落 / s滑动旋转 / x作废：").strip().lower()
            if answer in ("", "t", "m", "d", "s", "x"):
                break
        note = self.ask("备注（可留空）：")
        self.meta["operator"] = {"label": answer, "note": note}
        if answer in ("t", "m", "d"):
            self.ask("请先将瓶子复位到标记位置，确认后按回车，随后重新检查两腕图像：")
            self.check_bottle("operator_reset")

    def visual_point(self, item, bottle=None):
        if self.airborne or self.session.progress.get("requires_operator_recovery"):
            raise SafetyFault("Operator recovery latch prevents starting another point")
        planned_bottle = item.get("bottle")
        if not isinstance(planned_bottle, str) or not planned_bottle.strip():
            raise ValueError("Visual point requires an explicit planned bottle")
        if bottle is not None and bottle != planned_bottle:
            raise ValueError("Visual bottle override disagrees with the saved point plan")
        bottle = planned_bottle
        trial_id = str(item["point_id"])
        folder = self.session.start_trial(trial_id)
        self.meta = {"trial_id": trial_id, "task": "visual", "bottle": bottle, "kind": "visual", "stages": [],
            "T_target": item["T_target"], "auto": {"exceptions": []}, "status": "running", "simulated": self.backend.simulated}
        self.backend.set_trial(item)
        self.recorder = Acquisition(self.backend, folder)
        try:
            with self.phase("VISUAL_MOVE"):
                # Open only: visual scans have no close or force-control branch.
                self.ensure_open()
                self.move(item["T_target"], speed=self.cfg["motion"]["near_speed_mm_s"], noncontact=True)
            with self.phase("VISUAL_CAPTURE"):
                self.wait(self.cfg["motion"].get("settle_s", .3), noncontact=True)
                measured = self.backend.snapshot()["tcp"]
                self.meta["T_measured"] = measured.tolist()
                from .geometry import grasp_pose
                self.meta["B"] = grasp_pose(self.session.T0, self.cfg).tolist()
                self.meta["label_B_in_grasp"] = (np.linalg.inv(grasp_pose(measured, self.cfg)) @ np.asarray(self.meta["B"])).tolist()
                self.meta["label_B_in_tcp"] = (np.linalg.inv(measured) @ self.session.T0).tolist()
                self.meta["label_B_in_tcp_legacy"] = self.meta["label_B_in_tcp"]
                self.meta["grasp_center_base_mm"] = grasp_pose(measured, self.cfg)[:3, 3].tolist()
                self.meta["grasp_center_offset_base_mm"] = (grasp_pose(measured, self.cfg)[:3, 3]-grasp_pose(self.session.T0, self.cfg)[:3, 3]).tolist()
                self.meta["capture_monotonic_ns"] = self.now()
                paths = {}
                for name, sample in self.backend.image_samples().items():
                    from .park_guard import image_pose_metadata
                    attrs = dict(sample.metadata or {})
                    if "pose_sources" not in attrs:
                        attrs.update(image_pose_metadata(self.backend.snapshot(), self.session, camera_stamp_ns=sample.monotonic_ns))
                    attrs.update(stage=self.stage, keyframe=True, image_encoding="keyframe_jpeg_from_rgb",
                        image_timestamp_kind="source_received_monotonic_ns", selected_monotonic_ns=self.now())
                    paths[name] = "images/" + self.recorder.jpg.append(name, sample.data, sample.monotonic_ns,
                        sample.device_timestamp, attrs=attrs)
                self.meta["images"] = paths
            self.meta["status"] = "completed"
        except SkippedPoint as exc:
            self.backend.stop_arm()
            self.meta.update(status="skipped_safety", skip_reason=str(exc))
        except BaseException as exc:
            self.backend.stop_arm()
            self.meta["auto"]["exceptions"].append({"stage": self.stage, "error": str(exc)})
            self.meta["status"] = "aborted"
            atomic_json(folder / "meta.json", self.meta)
            raise
        finally:
            original_error = __import__("sys").exc_info()[1]
            try:
                self.recorder.close()
            except BaseException as exc:
                self._note_logging_error(exc, "visual_acquisition_close")
                self.meta["status"] = "aborted"
                atomic_json(folder / "meta.json", self.meta)
                if original_error is None:
                    raise SafetyFault("Visual recording incomplete: " + str(exc)) from exc
            finally:
                self.recorder = None
        self.session.complete(self.meta)
        return self.meta
