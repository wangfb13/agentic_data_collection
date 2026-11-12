"""Opt-in visual open-hand positioning through the existing guarded Runner.

Pure planning is separate from execution. Imports never connect hardware.
This stage does not close, touch, lift, train a skill or certify deployment.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from .geometry import validate_transform, pose_error, grasp_transform, axis_basis
from .session import fingerprint, plain
from .vision_workspace import TaughtWorkspace
from .vision_localization import validate_calibration


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def markerless_geometry_reasons(workspace, calibration):
    """The visual shape and old collision envelope must describe one object.

    A generous collision envelope is allowed. An offset, tilted, shorter or
    narrower collision bottle must never authorize an otherwise correct image.
    Check continuous height intervals, including gaps between old cylinders.
    """
    cfg, teaching = workspace.config, workspace.teaching
    model = calibration["model"]
    reasons = []
    if cfg["safety"]["table_z_mm"] < model["table_z_mm"]-1e-8:
        reasons.append("collision_table_below_visual_table")
    if cfg["safety"]["margin_mm"] < calibration["fit_limits"]["max_position_uncertainty_mm"]:
        reasons.append("collision_margin_below_visual_uncertainty")
    anchor = np.asarray(teaching["geometry_reference_tcp_mm"])
    reference = np.asarray(teaching["reference_bottle_pose_mm"])
    bottle = cfg["geometry"]["bottle"]
    center = anchor[:3, 3]+anchor[:3, :3] @ np.asarray(bottle["center_offset_mm"])
    axis = anchor[:3, :3] @ axis_basis(cfg["axes"])[:, ("a", "c", "l").index(bottle["axis"])]
    if not np.allclose(center[:2], reference[:2, 3], atol=1e-6, rtol=0):
        reasons.append("collision_bottle_axis_position_mismatch")
    if not np.allclose(axis[:2], [0., 0.], atol=1e-6, rtol=0) or abs(abs(axis[2])-1.) > 1e-6:
        reasons.append("collision_bottle_axis_not_vertical")
        return reasons
    intervals = []
    # Use nominal cylinders: modelling uncertainty must not hide missing shape.
    for seg in bottle["segments"]:
        ends = sorted(center[2]+axis[2]*float(seg[key]) for key in ("axial_min_mm", "axial_max_mm"))
        intervals.append((*ends, float(seg["radius_mm"])))
    for name in ("body_profile_mm", "cap_profile_mm"):
        if name not in model:
            continue
        profile = np.asarray(model[name], float).copy()
        profile[:, 0] += model["table_z_mm"]
        for (z0, r0), (z1, r1) in zip(profile[:-1], profile[1:]):
            cuts = sorted({z0, z1, *(z for lo, hi, _ in intervals for z in (lo, hi) if z0 < z < z1)})
            for low, high in zip(cuts[:-1], cuts[1:]):
                mid = (low+high)/2
                covered = max((r for lo, hi, r in intervals if lo <= mid <= hi), default=0.)
                required = max(r0+(r1-r0)*(z-z0)/(z1-z0) for z in (low, high))
                if covered+1e-8 < required:
                    reasons.append("collision_envelope_does_not_cover_"+name)
                    break
    return sorted(set(reasons))


def plan_alignment(profile, calibration, observation, current_tcp_mm, *, now_ns,
                   travelled_mm=0., tolerance_mm=1., width_mm=None):
    """Reobserve before EVERY step; never turn an out-of-bounds goal into a valid one."""
    workspace = TaughtWorkspace(profile)
    calibration = validate_calibration(calibration)
    current = validate_transform(current_tcp_mm)
    tolerance = _positive(tolerance_mm, "tolerance_mm")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("now_ns must be a monotonic timestamp")
    if isinstance(travelled_mm, bool) or not math.isfinite(travelled_mm) or travelled_mm < 0:
        raise ValueError("travelled_mm must be finite and nonnegative")
    result = {"schema": "vision_alignment_plan_v1", "valid": False, "reasons": [],
              "status": "rejected", "source_tcp_mm": current.tolist(), "target_tcp_mm": None,
              "goal_tcp_mm": None, "workspace_sha256": profile["content_sha256"],
              "observation": deepcopy(observation), "created_ns": now_ns,
              "travelled_mm": float(travelled_mm), "tolerance_mm": tolerance,
              "simulated": profile["simulated"], "contact_allowed": False}
    reasons = result["reasons"]
    if observation.get("schema") != "vision_observation_v1":
        reasons.append("invalid_observation_schema")
    if observation.get("valid") is not True:
        reasons.append("invalid_visual_observation")
    if type(observation.get("simulated")) is not bool or observation["simulated"] != profile["simulated"] or calibration.get("simulated") != profile["simulated"]:
        reasons.append("simulation_domain_mismatch")
    # Localization supplies its own exact canonical calibration fingerprint.
    from .vision_localization import calibration_digest
    calibration_id = calibration_digest(calibration)
    if observation.get("calibration_sha256") != calibration_id:
        reasons.append("calibration_mismatch")
    reference_id = workspace.teaching.get("reference_calibration_sha256")
    if (reference_id is not None or not profile["simulated"]) and reference_id != calibration_id:
        reasons.append("taught_reference_calibration_mismatch")
    if observation.get("camera_name") != calibration["camera_name"]:
        reasons.append("camera_mismatch")
    markerless = calibration["schema"] == "markerless_bottle_calibration_v1"
    if markerless:
        if observation.get("target_id") != calibration["target_id"] or observation.get("appearance_sha256") != calibration["appearance_profile"]["content_sha256"]:
            reasons.append("target_appearance_mismatch")
        if (observation.get("source") != "known_bottle_silhouette" or
                observation.get("pose_model") != "upright_axisymmetric_on_table" or
                observation.get("orientation_observed") is not False or
                observation.get("observed_dofs") != ["x", "y"]):
            reasons.append("unsupported_markerless_observation")
        uncertainty = observation.get("position_uncertainty_mm")
        if (type(uncertainty) not in (int, float) or not math.isfinite(uncertainty) or
                not 0 <= uncertainty <= calibration["fit_limits"]["max_position_uncertainty_mm"]):
            reasons.append("invalid_markerless_position_uncertainty")
    elif observation.get("marker_id") != calibration["marker_id"]:
        reasons.append("target_marker_mismatch")
    stamp = observation.get("timestamp_ns")
    if type(stamp) is not int or not 0 <= now_ns-stamp <= calibration["max_observation_age_s"]*1e9:
        reasons.append("stale_visual_observation")
    pose_stamp = observation.get("pose_timestamp_ns")
    if type(pose_stamp) is not int or not 0 <= now_ns-pose_stamp <= calibration["max_observation_age_s"]*1e9:
        reasons.append("stale_visual_pose")
    elif type(stamp) is int and abs(pose_stamp-stamp) > calibration["max_pose_skew_s"]*1e9:
        reasons.append("image_pose_timestamp_skew")
    expected_source = "known_bottle_silhouette" if markerless else "aruco_detection"
    if not profile["simulated"] and observation.get("source") != expected_source:
        reasons.append("real_motion_requires_detected_image")
    if reasons:
        return result
    try:
        bottle = validate_transform(observation["T_base_bottle_mm"])
    except (KeyError, TypeError, ValueError):
        reasons.append("invalid_bottle_pose")
        return result
    teaching = workspace.teaching
    if markerless:
        reasons.extend(markerless_geometry_reasons(workspace, calibration))
        reference = np.asarray(teaching["reference_bottle_pose_mm"])
        for pose in (bottle, reference):
            if (not np.allclose(pose[:3, :3], np.eye(3), atol=1e-8, rtol=0) or
                    abs(pose[2, 3]-calibration["model"]["table_z_mm"]) > 1e-8):
                reasons.append("markerless_table_or_orientation_convention_mismatch")
                return result
        if reasons:
            return result
    goal = bottle @ np.linalg.inv(np.asarray(teaching["reference_bottle_pose_mm"])) @ np.asarray(teaching["reference_tcp_mm"])
    result["goal_tcp_mm"] = goal.tolist()
    # This first stage translates only. A changed object orientation is not
    # silently turned into a new wrist rotation or an arbitrary projected goal.
    if pose_error(current, goal)[1] > teaching["orientation_tolerance_deg"]:
        reasons.append("goal_requires_uncommissioned_rotation")
    if pose_error(current, np.asarray(teaching["reference_tcp_mm"]))[1] > teaching["orientation_tolerance_deg"]:
        reasons.append("current_orientation_outside_teaching")
    # Preserve the desired physical grasp-center location when keeping wrist
    # orientation fixed, even with a nonzero measured TCP->grasp offset.
    grasp = grasp_transform(workspace.config)
    center = (goal @ grasp)[:3, 3].copy()
    goal[:3, :3] = current[:3, :3]
    goal[:3, 3] = center-current[:3, :3] @ grasp[:3, 3]
    delta = goal[:3, 3]-current[:3, 3]
    distance = float(np.linalg.norm(delta))
    result["remaining_translation_mm"] = distance
    if travelled_mm+distance > teaching["max_total_translation_mm"]+1e-9:
        reasons.append("total_translation_budget_exceeded")
    # Hold the taught/current wrist orientation exactly; acceptable tiny PnP
    # rotational variation is not commanded as robot rotation.
    result["goal_tcp_mm"] = goal.tolist()
    width = teaching["open_width_mm"] if width_mm is None else width_mm
    reasons.extend(workspace.check_segment(goal, goal, bottle, width_mm=width))
    if reasons:
        return result
    count = max(1, int(math.ceil(distance/teaching["max_translation_step_mm"])))
    previous = current
    waypoints = []
    for index in range(1, count+1):
        target = current.copy()
        target[:3, 3] += delta*(index/count)
        reasons.extend(workspace.check_segment(previous, target, bottle, width_mm=width))
        waypoints.append(target.tolist())
        previous = target
    if reasons:
        result["reasons"] = sorted(set(reasons))
        return result
    result.update(valid=True, status="aligned" if distance <= tolerance else "step_ready",
                  target_tcp_mm=current.tolist() if distance <= tolerance else waypoints[0],
                  preview_waypoints_mm=waypoints,
                  note="Only the first step may execute. Fresh image/pose and replanning required thereafter.")
    return result


class GuardedVisionAdapter:
    """Use an already commissioned runtime.Runner; never create a publisher here.

    The immutable session reference is restored after each dynamic scene check.
    Existing force, fresh-feedback, command-owner and measured-path guards remain.
    """
    def __init__(self, runner, profile, calibration):
        self.runner = runner
        self.workspace = TaughtWorkspace(profile)
        self.profile, self.calibration = deepcopy(profile), deepcopy(calibration)
        validate_calibration(calibration)
        if type(runner.backend.simulated) is not bool or runner.backend.simulated != profile["simulated"]:
            raise ValueError("runner simulation domain differs")
        if calibration["simulated"] != profile["simulated"]:
            raise ValueError("calibration simulation domain differs")
        if not profile["simulated"]:
            if calibration["camera_mount"] != "eye_in_hand":
                raise ValueError("physical stage one requires a calibrated wrist camera with live TCP")
            if calibration["camera_name"] != calibration["mount_tcp_side"]+"_wrist":
                raise ValueError("camera name does not match its physical wrist")
            if "nonexperiment_arm_park" not in runner.session.meta:
                raise ValueError("physical positioning requires an existing parked-arm reference")
        if fingerprint(runner.cfg) != fingerprint(self.workspace.config):
            raise ValueError("runner must use the exact taught effective configuration")
        self.travelled_mm = 0.
        self.last_image_ns = None
        self.last_completion_ns = None
        self.faulted = False
        self.receipts = []

    def _check_park(self):
        if "nonexperiment_arm_park" in self.runner.session.meta or not self.profile["simulated"]:
            from .park_guard import ParkGuard
            ParkGuard(self.runner.session).check(self.runner.backend.park_observation(),
                now_ns=int(self.runner.now()), context="vision_stage1")

    def move_step(self, observation, *, tolerance_mm=1.):
        runner = self.runner
        if self.faulted:
            raise ValueError("adapter faulted; operator recovery and a new session required")
        stamp = observation.get("timestamp_ns")
        if type(stamp) is not int or self.last_image_ns is not None and stamp <= self.last_image_ns:
            raise ValueError("a new image is required after each attempt")
        if self.last_completion_ns is not None and stamp <= self.last_completion_ns:
            raise ValueError("image must be acquired after the previous completed action")
        if runner.airborne or runner.grasp_pose is not None:
            raise ValueError("visual positioning requires no carried object")
        self._check_park()
        snapshot = runner._healthy_snapshot()
        if np.any(np.asarray(snapshot["normal_N"]) > runner.threshold):
            raise ValueError("visual positioning requires unloaded open gripper")
        start = np.asarray(snapshot["tcp"], float)
        plan = plan_alignment(self.profile, self.calibration, observation, start,
                              now_ns=int(runner.now()), travelled_mm=self.travelled_mm,
                              tolerance_mm=tolerance_mm, width_mm=float(snapshot["width_mm"]))
        if not plan["valid"]:
            raise ValueError("visual plan rejected: " + ";".join(plan["reasons"]))
        self.last_image_ns = stamp
        if plan["status"] == "aligned":
            return {"plan": plan, "applied": False, "visual_alignment_only": True}
        target = np.asarray(plan["target_tcp_mm"])
        anchor = self.workspace.geometry_anchor(observation["T_base_bottle_mm"])
        previous_reference = runner.session.T0.copy()
        try:
            runner.session.T0 = anchor
            # phase(single_step=False) is used by the CLI; explicit MOVE happens
            # before reacquiring the image. No prompt can age this observation.
            with runner.phase("VISUAL_MOVE"):
                latest = runner._healthy_snapshot()
                if pose_error(start, latest["tcp"])[0] > runner.cfg["motion"]["pose_tolerance_mm"] or pose_error(start, latest["tcp"])[1] > runner.cfg["motion"]["pose_tolerance_deg"]:
                    raise ValueError("TCP changed after visual plan")
                if runner.now()-stamp > self.calibration["max_observation_age_s"]*1e9:
                    raise ValueError("image became stale before command")
                def open_and_unloaded(snap):
                    return (abs(float(snap["width_mm"])-self.workspace.teaching["open_width_mm"]) <=
                            runner.cfg["gripper"].get("open_tolerance_mm", .5) and
                            not np.any(np.asarray(snap["normal_N"]) > runner.threshold))
                if not open_and_unloaded(latest):
                    raise ValueError("gripper must remain fully open and unloaded")
                self._check_park()
                def interrupted(snap):
                    self._check_park()
                    return not open_and_unloaded(snap)
                completed = runner.move(target, noncontact=True, allow_contact=False,
                            speed=runner.cfg["motion"]["near_speed_mm_s"],
                            stop_when=interrupted)
                if not completed:
                    raise ValueError("open-hand motion interrupted by gripper/contact change")
                end = np.asarray(runner._healthy_snapshot()["tcp"], float)
                travelled = float(np.linalg.norm(end[:3, 3]-start[:3, 3]))
                self.travelled_mm += travelled
                receipt = {"applied": True, "plan": plan, "measured_tcp_mm": end.tolist(),
                           "applied_translation_mm": (end[:3, 3]-start[:3, 3]).tolist(),
                           "completed_ns": int(runner.now()), "travelled_mm": self.travelled_mm}
                self.receipts.append(receipt)
                self.last_completion_ns = receipt["completed_ns"]
                return receipt
        except BaseException:
            self.faulted = True
            runner.backend.stop_arm()
            raise
        finally:
            runner.session.T0 = previous_reference


def localize_sample(sample, arms, calibration, now_ns):
    """Match wrist-camera extrinsics to the correct arm's timestamped pose."""
    from .vision_localization import localize_image
    calibration = validate_calibration(calibration)
    if calibration["camera_mount"] == "eye_in_hand":
        side = calibration["mount_tcp_side"]
        pose = arms[side]
        tcp, stamp = pose["T_base_tcp_mm"], pose["received_monotonic_ns"]
    else:
        side, tcp, stamp = None, np.eye(4), sample.monotonic_ns
    if sample.stream != "camera/"+calibration["camera_name"]:
        raise ValueError("image stream differs from calibration camera")
    return localize_image(sample.data, calibration, T_base_tcp_mm=tcp,
                           image_timestamp_ns=int(sample.monotonic_ns), pose_timestamp_ns=int(stamp),
                           now_ns=int(now_ns), pose_tcp_side=side)
