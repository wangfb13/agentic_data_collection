"""Configuration loading. Unmeasured values remain TODO outside explicit demo."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np
import yaml


CALIBRATE_AT_SESSION_START = "calibrate_at_session_start"


def unresolved(cfg, prefix=""):
    result = []
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            result.extend(unresolved(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(cfg, list):
        for i, value in enumerate(cfg):
            result.extend(unresolved(value, f"{prefix}[{i}]"))
    elif cfg is None or isinstance(cfg, str) and "TODO" in cfg.upper():
        result.append(prefix)
    return result


def _merge(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def demo_config():
    """A wholly synthetic tabletop scene. Never use these values on hardware."""
    result = {
        "schema_version": 2, "demo": True,
        "units": {"distance": "mm", "angle": "deg", "force": "N", "time": "s"},
        "grasp_mode": "side", "axes": {"a": [1., 0, 0], "c": [0, 1., 0], "l": [0, 0, 1.]},
        "T0": [[1., 0, 0, 300.], [0, 1., 0, 0], [0, 0, 1., 70.], [0, 0, 0, 1.]],
        "tool": {"grasp_center_in_tcp": np.eye(4).tolist(),
            "grasp_center_evidence": "synthetic fixture only; center intentionally equals TCP"},
        "zero_guard": {"sample_count": 100, "timeout_s": 8., "force_std_max_N": .05,
            "dd_contact_threshold": .05, "dd_contact_max_fraction": .001, "offset_delta_warn_N": .2},
        "preparation": {"axis_test_pose": [[1., 0, 0, 200.], [0, 1., 0, 0], [0, 0, 1., 150.], [0, 0, 0, 1.]]},
        "geometry": {
            "bottle": {"axis": "l", "center_offset_mm": [0., 0, 0], "segments": [
                {"name": "body", "radius_mm": 18., "axial_min_mm": -45., "axial_max_mm": 20.},
                {"name": "neck", "radius_mm": 11., "axial_min_mm": 20., "axial_max_mm": 35.},
                {"name": "cap", "radius_mm": 14., "axial_min_mm": 35., "axial_max_mm": 50.}]},
            "gripper": {"finger_size_acl_mm": [50., 8., 20.], "finger_center_a_mm": -10.,
                "finger_center_l_mm": 0., "palm_center_acl_mm": [-45., 0, 0],
                "palm_size_acl_mm": [16., 110., 24.], "pad_depth_mm": 3., "pad_size_al_mm": [50., 20.]}},
        "safety": {"workspace_min_mm": [50., -250., 0.], "workspace_max_mm": [600., 250., 400.],
            "table_z_mm": 0., "margin_mm": 1., "max_step_mm": 150., "max_rotation_step_deg": 45.,
            "path_step_mm": 2., "path_step_deg": 2., "cylinder_sides": 32},
        "motion": {"free_speed_mm_s": 20., "near_speed_mm_s": 5., "lift_speed_mm_s": 5.,
            "angular_speed_deg_s": 5., "pose_tolerance_mm": 1., "pose_tolerance_deg": 1.,
            "near_distance_mm": 50., "lift_height_mm": 50., "pre_distance_mm": 50.,
            "settle_s": .3, "hold_s": 2., "stage_timeouts_s": {k: 30. for k in
                ("MOVE_PRE", "APPROACH", "CLOSE", "PROBE", "SQUEEZE", "LIFT", "HOLD", "LOWER", "RELEASE", "RETREAT", "CHECK", "LABEL", "VISUAL_MOVE", "VISUAL_CAPTURE", "BASELINE")}},
        "force": {"contact_min_N": .2, "target_N": 2., "probe_fraction": .3, "max_N": 4.,
            "target_tolerance_N": .2, "stable_s": .5, "probe_s": .3, "squeeze_mode": "fixed"},
        "gripper": {"open_mm": 90., "min_mm": 20., "open_tolerance_mm": .5, "close_speed_mm_s": 2., "stall_window_s": .3, "stall_epsilon_mm": .05, "geometry_width_step_mm": 1.},
        "design": {"bottle_axis_range_mm": [-20., 60.], "bottle_axis_step_mm": 5.,
            "other_axis_range_mm": [-20., 20.], "other_axis_step_mm": 5.,
            "closure_offsets_mm": [-8., -4., 0., 4., 8.], "tilt_degrees": [-10., -5., 0., 5., 10.],
            "repeats": 2, "seed": 20261026, "zero_every": 10, "baseline_every": 30,
            "bottles": ["light", "heavy"]},
        "visual": {"count": 400, "seed": 20261026, "orbit_deg": [-20., 20.],
            "a_range_mm": [-80., 0.], "c_range_mm": [-20., 20.], "l_range_mm": [-30., 30.],
            "tilt_deg": [-10., 10.], "settle_s": .3},
        "check": {"retreat_mm": 80., "threshold_px": {"left_wrist": 5., "right_wrist": 5.},
            "min_score": .7, "calibration_mm": 2., "calibration_fraction": .8, "search_radius_px": 80},
        "tactile": {"pixel_pitch_mm": [.2, .2], "depth_channel": 0, "dd_threshold": .05,
            "centroid_drift_mm": 2., "sensor_to_tcp": {"left": np.eye(3).tolist(), "right": np.eye(3).tolist()},
            "normal_axis": 2, "normal_signs": [1., 1.], "dd_shape": [32, 48, 3], "dd_dtype": "float32", "dd_unit": "synthetic_mm",
            "fps": 20., "baseline_s": 10., "edge_pixels": 3,
            "force_unit": "N", "moment_unit": "Nmm", "cop_plane_z_mm": 0.},
        "analysis": {"dd_threshold": .05, "edge_width_px": 3, "max_stream_skew_s": .1,
            "max_pose_command_age_s": .2, "centroid_to_al": {"left": np.eye(2).tolist(), "right": np.eye(2).tolist()}},
        "cameras": {"left_wrist": {"fps": 20., "width": 640, "height": 480},
            "right_wrist": {"fps": 20., "width": 640, "height": 480}},
        "hardware": {"backend": "demo_only", "bridge_module": "hardware_bridge_template",
            "sensor_serials": {"left": "SYNTHETIC_LEFT", "right": "SYNTHETIC_RIGHT"},
            "pose_unit": "m", "quaternion_order": "xyzw",
            "base_frame": "base_link", "tcp_definition": "synthetic", "arm_side": "right",
            "public_ros": {"command_ownership": {"infrastructure_output_policy": "strict",
                "topics": ["/wbc_goal_pose", "/joint_control", "/joint_commands",
                "/left_gripper_command", "/right_gripper_command"], "timeout_s": 3.}},
            "stop_verified": False, "linear_path_verified": False, "cartesian_speed_verified": False,
            "gripper_width_calibration": "synthetic", "force_calibration": "synthetic",
            "camera_mounts_declared": False, "axis_mapping_declared": False, "sensor_mapping_declared": False},
        "output": {"root": "outputs", "compression": "gzip", "jpeg_quality": 95},
        "recording": {"control_hz": 20., "stale_timeout_s": .25, "source_max_age_s": .25},
        "estimated_bytes_per_image": 80000, "sample_hz": 20.,
    }
    from .force_control import force_defaults, close_defaults
    from .stop_validation import stop_test_defaults
    from .park_guard import park_guard_defaults
    from .recording import async_recording_defaults, record_stress_defaults
    result["force"].update(force_defaults(demo=True))
    result["close"] = close_defaults(demo=True)
    result["stop_test"] = stop_test_defaults(demo=True)
    result["park_guard"] = park_guard_defaults()
    result["recording"]["writer"] = async_recording_defaults()
    result["recording"]["stress"] = record_stress_defaults()
    result["record"] = {"gel_raw": "windows"}
    result["probe"] = {"settle_s": .2, "timeout_s": 3., "window_frames": 5}
    result["hold"] = {"window_s": .5}
    result["opening_repeat"] = {"repetitions": 10, "report_path": "outputs/validation/opening_report.json"}
    result["simulation"] = {"max_speedup": 2.}
    result["lower"] = {"unload_ratio": .2, "vertical_mapping_confirmed": False}
    result["tactile"]["centroid_drift_px"] = 5.
    result["preparation"]["height_crosscheck_tolerance_mm"] = 5.
    return result


def load_config(path=None, demo=False):
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "config.yaml"
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    if demo:
        # Explicit demo always uses a coherent synthetic scene. Real values are
        # not mixed with guessed geometry; paths and seeds can be changed later.
        result = demo_config()
        result["demo_source_config"] = str(path)
        result["demo_notice"] = "SYNTHETIC DEMO: no measured geometry, no hardware execution"
        result["source_unresolved_fields"] = unresolved(config)
        return result
    config["demo"] = False
    return config


def require_real_ready(cfg, preparing=False):
    missing = unresolved(cfg) + validate_config(cfg)
    if cfg.get("demo"):
        missing.append("demo configuration cannot execute hardware")
    threshold = cfg.get("check", {}).get("threshold_px")
    if threshold == CALIBRATE_AT_SESSION_START and not preparing:
        missing.append("check.threshold_px: complete and save session-start calibration before trials")
    hardware = cfg.get("hardware", {})
    public_ros = hardware.get("profile", "strict") == "public_ros"
    keys = ["camera_mounts_declared", "axis_mapping_declared", "sensor_mapping_declared"]
    if public_ros:
        options = hardware.get("public_ros", {})
        for key in ("software_control_acknowledged", "processed_dd_accepted"):
            if options.get(key) is not True:
                missing.append("hardware.public_ros." + key)
    else:
        keys += ["stop_verified", "linear_path_verified", "cartesian_speed_verified"]
    for key in keys:
        if cfg.get("hardware", {}).get(key) is not True:
            missing.append("hardware." + key)
    if missing:
        raise ValueError("Real execution blocked: " + ", ".join(sorted(set(missing))))


def validate_config(cfg):
    """Return actionable validation errors; no coercion/clipping or mutation."""
    from .geometry import axis_basis, validate_transform, grasp_transform
    from .zero_guard import validate_zero_guard_settings
    from .command_ownership import validate_ownership_config
    from .force_control import validate_force_settings, force_unit
    from .stop_validation import validate_stop_settings
    from .park_guard import validate_park_settings
    errors = []
    if cfg.get("schema_version") != 2:
        errors.append("schema_version: version 2 with explicitly measured grasp-center transform is required")
    try:
        grasp_transform(cfg)
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    evidence = cfg.get("tool", {}).get("grasp_center_evidence")
    if not isinstance(evidence, str) or not evidence.strip() or "TODO" in evidence.upper():
        errors.append("tool.grasp_center_evidence: required URDF/CAD/measurement source and method")
    errors.extend(validate_zero_guard_settings(cfg))
    errors.extend(validate_force_settings(cfg))
    errors.extend(validate_stop_settings(cfg))
    errors.extend(validate_park_settings(cfg))
    if cfg.get("record", {}).get("gel_raw") not in ("off", "windows", "full"):
        errors.append("record.gel_raw must be off, windows or full")
    if cfg.get("hardware", {}).get("profile", "strict") == "public_ros":
        errors.extend(validate_ownership_config(cfg))

    def get(path):
        value = cfg
        for key in path.split("."):
            value = value[key]
        return value

    def number(path, minimum=0., strict=True):
        try:
            value = get(path)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError("must be a finite number")
            if value < minimum or (strict and value == minimum):
                raise ValueError(f"must be {'>' if strict else '>='} {minimum}")
            return float(value)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(path + ": " + str(exc))
            return None

    def vector(path, size, positive=False):
        try:
            value = np.asarray(get(path), dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)) or positive and np.any(value <= 0):
                raise ValueError(f"must be {size} finite {'positive ' if positive else ''}numbers")
            return value
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(path + ": " + str(exc))
            return None

    for key, expected in {"distance": "mm", "angle": "deg", "force": cfg.get("force", {}).get("unit"), "time": "s"}.items():
        if cfg.get("units", {}).get(key) != expected:
            errors.append(f"units.{key}: must be {expected}")
    if cfg.get("grasp_mode") not in ("side", "top"):
        errors.append("grasp_mode: must be side or top")
    if cfg.get("hardware", {}).get("profile", "strict") not in ("strict", "public_ros"):
        errors.append("hardware.profile: must be strict or public_ros")
    for key, allowed in {"pose_unit": ("m", "mm"), "quaternion_order": ("xyzw", "wxyz")}.items():
        if cfg.get("hardware", {}).get(key) not in allowed:
            errors.append("hardware." + key + ": invalid unit/convention")
    try:
        axis_basis(cfg["axes"])
        validate_transform(cfg["T0"])
    except (ValueError, TypeError, KeyError) as exc:
        errors.append("axes/T0: " + str(exc))
    try:
        validate_transform(cfg["preparation"]["axis_test_pose"])
    except (ValueError, TypeError, KeyError) as exc:
        errors.append("preparation.axis_test_pose: " + str(exc))
    positive = ["motion.free_speed_mm_s", "motion.near_speed_mm_s", "motion.lift_speed_mm_s",
        "motion.angular_speed_deg_s", "motion.pose_tolerance_mm", "motion.pose_tolerance_deg",
        "motion.near_distance_mm", "motion.lift_height_mm", "motion.pre_distance_mm", "motion.hold_s",
        "force.contact_min_N", "force.target_N", "force.probe_fraction", "force.max_N", "force.target_tolerance_N",
        "force.stable_s", "force.probe_s", "gripper.open_mm", "gripper.min_mm", "gripper.close_speed_mm_s",
        "gripper.stall_window_s", "gripper.stall_epsilon_mm", "gripper.open_tolerance_mm", "safety.max_step_mm", "safety.max_rotation_step_deg",
        "safety.path_step_mm", "safety.path_step_deg", "check.retreat_mm", "check.calibration_mm",
        "check.calibration_fraction", "check.search_radius_px", "tactile.fps",
        "tactile.baseline_s", "design.bottle_axis_step_mm", "design.other_axis_step_mm", "sample_hz",
        "recording.control_hz", "recording.stale_timeout_s", "recording.source_max_age_s"]
    values = {path: number(path) for path in positive}
    for path in ("safety.margin_mm", "motion.settle_s", "visual.settle_s", "tactile.dd_threshold"):
        number(path, strict=False)
    def compare(a, b, operator, message):
        x, y = values.get(a), values.get(b)
        if x is not None and y is not None and not operator(x, y):
            errors.append(message)
    compare("gripper.min_mm", "gripper.open_mm", lambda a, b: a < b, "gripper.min_mm must be below open_mm")
    compare("gripper.open_tolerance_mm", "gripper.open_mm", lambda a, b: a < b, "gripper.open_tolerance_mm must be below open_mm")
    # Unit/definition-aware per-finger limits are checked by validate_force_settings.
    if cfg.get("force", {}).get("controller", "bridge_loop") != "existing":
        compare("force.contact_min_N", "force.target_N", lambda a, b: a < b, "force.contact_min_N must be below target_N")
        compare("force.target_tolerance_N", "force.target_N", lambda a, b: a < b, "force.target_tolerance_N must be below target_N")
    # Existing control SDK and 39020 guard values have independent source bands;
    # validate_force_settings checks the measured feedback target, while
    # validate_existing_settings checks the original source's control target.
    compare("motion.near_speed_mm_s", "motion.free_speed_mm_s", lambda a, b: a <= b, "near speed exceeds free speed")
    for path in ("force.probe_fraction", "check.calibration_fraction"):
        if values[path] is not None and values[path] > 1:
            errors.append(path + " must be <= 1")
    if cfg.get("force", {}).get("squeeze_mode") not in ("fixed", "forcepolicy"):
        errors.append("force.squeeze_mode must be fixed or forcepolicy")
    for path in ("design.repeats", "design.zero_every", "design.baseline_every", "visual.count", "safety.cylinder_sides"):
        value = number(path)
        if value is not None and int(value) != value:
            errors.append(path + " must be an integer")
    if isinstance(cfg.get("safety", {}).get("cylinder_sides"), (int, float)) and cfg["safety"]["cylinder_sides"] < 8:
        errors.append("safety.cylinder_sides must be >= 8")
    for path in ("design.bottle_axis_range_mm", "design.other_axis_range_mm", "visual.orbit_deg", "visual.a_range_mm", "visual.c_range_mm", "visual.l_range_mm", "visual.tilt_deg"):
        value = vector(path, 2)
        if value is not None and value[0] > value[1]:
            errors.append(path + " must have lower <= upper")
    low, high = vector("safety.workspace_min_mm", 3), vector("safety.workspace_max_mm", 3)
    if low is not None and high is not None and np.any(low >= high):
        errors.append("workspace_min_mm must be below workspace_max_mm on every axis")
    number("safety.table_z_mm", minimum=-float("inf"), strict=False)
    for path in ("geometry.gripper.finger_center_a_mm", "geometry.gripper.finger_center_l_mm"):
        number(path, minimum=-float("inf"), strict=False)
    for path, size in (("geometry.gripper.finger_size_acl_mm", 3), ("geometry.gripper.palm_size_acl_mm", 3), ("geometry.gripper.pad_size_al_mm", 2)):
        vector(path, size, True)
    for path in ("geometry.gripper.palm_center_acl_mm", "geometry.bottle.center_offset_mm"):
        vector(path, 3)
    pad = number("geometry.gripper.pad_depth_mm")
    try:
        dimensions = np.asarray(get("geometry.gripper.finger_size_acl_mm"), float)
        pad_size = np.asarray(get("geometry.gripper.pad_size_al_mm"), float)
        if pad is not None and pad >= dimensions[1] or np.any(pad_size > dimensions[[0, 2]]):
            errors.append("soft pad volume must be contained in finger")
        for i,axis in ((0,"a"),(1,"l")):
            center = float(cfg["geometry"]["gripper"]["finger_center_"+axis+"_mm"])
            pad_center = float(cfg["geometry"]["gripper"].get("pad_center_"+axis+"_mm",center))
            extent = dimensions[0 if axis == "a" else 2]
            if not np.isfinite(pad_center) or abs(pad_center-center)+pad_size[i]/2>extent/2+1e-8:
                errors.append("active pad "+axis+" interval must be inside finger envelope")
    except (ValueError, TypeError, KeyError, IndexError):
        pass
    try:
        bottle = cfg["geometry"]["bottle"]
        if bottle["axis"] != ("l" if cfg["grasp_mode"] == "side" else "a"):
            errors.append("bottle axis disagrees with grasp mode")
        names=[s['name'] for s in bottle['segments']]
        if not ((len(names)==3 and set(names)=={'body','neck','cap'}) or names==['body','upper']):
            errors.append("bottle needs ordered body/upper or body/neck/cap cylinder segments")
        for segment in bottle["segments"]:
            dims = np.asarray([segment[k] for k in ("radius_mm", "axial_min_mm", "axial_max_mm")], float)
            if not np.all(np.isfinite(dims)) or dims[0] <= 0 or dims[1] >= dims[2]:
                errors.append("invalid bottle cylinder " + str(segment["name"]))
    except (TypeError, ValueError, KeyError) as exc:
        errors.append("bottle geometry: " + str(exc))
    minima = {"PROBE": "force.probe_s", "SQUEEZE": "force.stable_s", "HOLD": "motion.hold_s", "LOWER": "motion.settle_s", "BASELINE": "tactile.baseline_s", "VISUAL_CAPTURE": "visual.settle_s"}
    for stage in ("MOVE_PRE", "APPROACH", "CLOSE", "PROBE", "SQUEEZE", "LIFT", "HOLD", "LOWER", "RELEASE", "RETREAT", "CHECK", "LABEL", "VISUAL_MOVE", "VISUAL_CAPTURE", "BASELINE"):
        timeout = number("motion.stage_timeouts_s." + stage)
        if stage in minima and timeout is not None:
            try:
                hold = float(get(minima[stage]))
                if timeout <= hold:
                    errors.append("motion.stage_timeouts_s." + stage + " must exceed hold/settle duration")
            except (ValueError, TypeError, KeyError):
                pass
    points=cfg.get('hardware',{}).get('gripper_width_calibration')
    if isinstance(points,list) and any(isinstance(p,dict) and p.get('kind')=='compression_equivalent_estimate' for p in points):
        try:
            from .hardware_gripper import WidthCalibration
            cal=WidthCalibration(points);model=cal.compression_model
            if model['max_compression_mm']>2*float(cfg['geometry']['gripper']['pad_depth_mm'])+1e-8:
                errors.append('compression window exceeds both soft pad thicknesses')
            if cfg['gripper'].get('precontact_min_mm')!=model['contact_width_mm']:
                errors.append('precontact_min_mm must equal the measured body contact width')
            if cfg['hardware']['public_ros']['raw_normalized_min']!=model['normalized_hard_min']:
                errors.append('compression model must preserve the original normalized lower bound')
            if cfg['force'].get('controller','bridge_loop')!='bridge_loop':
                errors.append('compression model requires guarded bridge_loop force control')
        except (KeyError,TypeError,ValueError) as exc:errors.append('compression model: '+str(exc))
    score = number("check.min_score", minimum=-1, strict=False)
    if score is not None and score > 1:
        errors.append("check.min_score must be <= 1")
    for camera in ("left_wrist", "right_wrist"):
        if cfg.get("check", {}).get("threshold_px") != CALIBRATE_AT_SESSION_START:
            number("check.threshold_px." + camera)
        for key in ("fps", "width", "height"):
            number("cameras." + camera + "." + key)
    number("simulation.max_speedup")
    probe_settle = number("probe.settle_s")
    probe_timeout = number("probe.timeout_s")
    if probe_settle is not None and probe_timeout is not None and probe_timeout <= probe_settle:
        errors.append("probe.timeout_s must exceed probe.settle_s")
    window_frames = cfg.get("probe", {}).get("window_frames")
    if type(window_frames) is not int or window_frames < 1:
        errors.append("probe.window_frames must be a positive integer")
    hold_window = number("hold.window_s")
    if hold_window is not None and values.get("motion.hold_s") is not None and hold_window > values["motion.hold_s"]:
        errors.append("hold.window_s must not exceed motion.hold_s")
    unload_ratio = number("lower.unload_ratio")
    if unload_ratio is not None and unload_ratio >= 1:
        errors.append("lower.unload_ratio must be less than 1")
    if type(cfg.get("lower", {}).get("vertical_mapping_confirmed")) is not bool:
        errors.append("lower.vertical_mapping_confirmed must be an explicit bool")
    number("preparation.height_crosscheck_tolerance_mm")
    for key in ("queue_bytes", "queue_items"):
        value = cfg.get("recording", {}).get("writer", {}).get(key)
        if type(value) is not int or value <= 0:
            errors.append("recording.writer." + key + " must be a positive integer")
    for key in ("startup_timeout_s", "close_timeout_s"):
        number("recording.writer." + key)
    from .recording import validate_record_stress_settings
    errors.extend(validate_record_stress_settings(cfg))
    pitch = cfg.get("tactile", {}).get("pixel_pitch_mm")
    if isinstance(pitch, str) and pitch == "unavailable":
        number("tactile.centroid_drift_px")
        inactive_drift = cfg.get("tactile", {}).get("centroid_drift_mm")
        if inactive_drift != "unavailable":
            number("tactile.centroid_drift_mm")
    else:
        vector("tactile.pixel_pitch_mm", 2, positive=True)
        number("tactile.centroid_drift_mm")
    cop_plane = cfg.get("tactile", {}).get("cop_plane_z_mm")
    if cop_plane != "unavailable":
        try:
            if isinstance(cop_plane, bool) or not np.isfinite(float(cop_plane)):
                raise ValueError("must be a finite measurement or explicit unavailable")
        except (ValueError, TypeError):
            errors.append("tactile.cop_plane_z_mm: must be a finite measurement or explicit unavailable")
    opening = cfg.get("opening_repeat", {})
    if type(opening.get("repetitions")) is not int or opening["repetitions"] < 1:
        errors.append("opening_repeat.repetitions: must be a positive integer")
    if not isinstance(opening.get("report_path"), str) or not opening["report_path"].strip():
        errors.append("opening_repeat.report_path: required nonempty path")
    signs = vector("tactile.normal_signs", 2)
    if signs is not None and not np.all(np.isin(signs, [-1, 1])):
        errors.append("tactile.normal_signs must be +1 or -1")
    if cfg.get("tactile", {}).get("normal_axis") not in (0, 1, 2):
        errors.append("tactile.normal_axis must be 0, 1, or 2")
    for side in ("left", "right"):
        try:
            value = cfg["tactile"]["sensor_to_tcp"][side]
            if isinstance(value, str) and value == "unavailable":
                if cfg.get("lower", {}).get("vertical_mapping_confirmed"):
                    raise ValueError("vertical mapping requires measured sensor rotations")
                continue
            R = np.asarray(value, dtype=float)
            if R.shape != (3, 3) or not np.all(np.isfinite(R)) or not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(R), 1):
                raise ValueError("must be proper 3x3 rotation")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append("tactile.sensor_to_tcp." + side + ": " + str(exc))
    try:
        dtype = np.dtype(cfg["tactile"]["dd_dtype"])
        if dtype.kind not in "fiu":
            raise ValueError("must identify the native numeric DD dtype")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("tactile.dd_dtype: " + str(exc))
    for side in ("left", "right"):
        serial = cfg.get("hardware", {}).get("sensor_serials", {}).get(side)
        if not isinstance(serial, str) or not serial.strip():
            errors.append("hardware.sensor_serials." + side + ": required nonempty string")
    bridge = cfg.get("hardware", {}).get("bridge_module")
    if not isinstance(bridge, str) or not bridge.strip():
        errors.append("hardware.bridge_module: required nonempty module name")
    return sorted(set(errors))

