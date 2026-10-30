"""Explicit force units/definitions, validation metrics and evidence gates.

This module is offline-only. It does not connect to ROS or operate actuators.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .session import fingerprint


def force_value(cfg, name):
    """Canonical unit-neutral key; retain old numeric keys for saved V1 sessions."""
    force = cfg["force"]
    key = name if name in force else name + "_N"
    value = float(force[key])
    if not math.isfinite(value):
        raise ValueError("force." + key + " must be finite")
    return value


def force_unit(cfg):
    unit = cfg.get("force", {}).get("unit", "N")
    if unit not in ("sdk", "N"):
        raise ValueError("force.unit must be sdk or N")
    return unit


def target_definition(cfg):
    definition = cfg.get("force", {}).get("target_definition", "per_finger")
    if definition not in ("per_finger", "mean", "sum"):
        raise ValueError("force.target_definition must be per_finger, mean or sum")
    return definition


def force_components(normals):
    values = np.asarray(normals, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("Two finite independently observed normal forces are required")
    return {"per_finger": values.tolist(), "mean": float(values.mean()), "sum": float(values.sum())}


def in_target_band(normals, target, tolerance, definition):
    values = force_components(normals)
    if definition not in values:
        raise ValueError("Unknown force target definition")
    return bool(np.all(np.abs(np.asarray(values[definition]) - float(target)) <= float(tolerance)))


def target_as_mean(target, definition):
    if definition not in ("per_finger", "mean", "sum"):
        raise ValueError("Unknown force target definition")
    return float(target) / 2. if definition == "sum" else float(target)


def feedback_target(cfg, stage):
    """39020 settling target; never infer an original-controller SDK mapping."""
    if stage not in ("PROBE", "SQUEEZE"):
        raise ValueError("feedback_target stage must be PROBE or SQUEEZE")
    force = cfg["force"]
    if force.get("controller", "bridge_loop") == "existing":
        value = force["existing"]["evaluation_probe_target"] if stage == "PROBE" else force["validation"]["evaluation_target"]
    else:
        value = (force.get("probe", force.get("probe_N", force["probe_fraction"]*force_value(cfg,"target")))
                 if stage == "PROBE" else force_value(cfg,"target"))
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("Explicit finite positive 39020 feedback target required")
    return float(value)


def validate_control_target(cfg, target):
    """Check commands against the selected controller's own source limits."""
    if isinstance(target, bool) or not math.isfinite(float(target)) or float(target) < 0:
        raise ValueError("Force target must be nonnegative and finite")
    if float(target) == 0:
        return  # Explicit exit, never a low-force regulation request.
    definition = target_definition(cfg)
    actual = target_as_mean(target, definition)
    if cfg["force"].get("controller", "bridge_loop") == "existing":
        limits = cfg["force"]["existing"]["control_limits"]
        band = limits["target_tolerance"] / (2. if definition == "sum" else 1.)
        if not limits["contact_min"] < actual-band < actual+band < limits["max"]:
            raise ValueError("Requested original-controller target band violates its own 39010 contact/max limits")
    elif actual >= force_value(cfg, "max"):
        raise ValueError("Force target violates 39020 bridge-loop per-finger maximum")


def force_defaults(*, demo=False):
    return {"controller": "bridge_loop", "target_definition": "mean", "unit": "N" if demo else "sdk",
        "validation": {"repetitions": 10, "hold_s": 2., "report_path": "outputs/validation/force_report.json",
                       "evaluation_target": 2. if demo else "TODO_measured_39020_evaluation_target",
                       "max_contact_peak": 3.5 if demo else "TODO_accepted_contact_peak",
                       "max_settle_s": 2. if demo else "TODO_accepted_force_settle_s",
                       "max_steady_std": .3 if demo else "TODO_accepted_steady_force_std"}}


def close_defaults(*, demo=False):
    return {"detection_frames": 2, "observed_force_fps": 20. if demo else "TODO_measured_force_fps",
            "observed_dd_fps": 20. if demo else "TODO_measured_DD_fps",
            "max_overtravel_mm": .5 if demo else "TODO_accepted_overtravel_mm",
            "width_resolution_mm": .05 if demo else "TODO_measured_width_resolution_mm"}


def close_quality_report(cfg):
    """B2 conservative two-source latency estimate and measurable stall travel."""
    c = cfg["close"]
    values = [float(c[k]) for k in ("detection_frames", "observed_force_fps", "observed_dd_fps",
                                      "max_overtravel_mm", "width_resolution_mm")]
    if not all(math.isfinite(v) and v > 0 for v in values) or int(values[0]) != values[0]:
        raise ValueError("close parameters require measured positive values and integer detection_frames")
    frames, force_fps, dd_fps, maximum, resolution = values
    speed = float(cfg["gripper"]["close_speed_mm_s"])
    window = float(cfg["gripper"]["stall_window_s"])
    if not math.isfinite(speed) or speed <= 0 or not math.isfinite(window) or window <= 0:
        raise ValueError("Closure speed and stall window must be positive and finite")
    delay = frames / min(force_fps, dd_fps)
    estimate, travel = speed * delay, speed * window
    errors = []
    if estimate > maximum:
        errors.append("接触过冲行程估计超过 close.max_overtravel_mm")
    if travel < 5. * resolution:
        errors.append("闭合速度 × 停滞窗口小于开度分辨率的 5 倍")
    return {"detection_delay_s": delay, "estimated_overtravel_mm": estimate,
            "max_overtravel_mm": maximum, "stall_window_travel_mm": travel,
            "width_resolution_mm": resolution, "required_stall_travel_mm": 5.*resolution,
            "source": "configured measured DD/force rates; slower source determines delay", "errors": errors}


def validate_force_settings(cfg):
    errors = []
    try:
        unit, definition = force_unit(cfg), target_definition(cfg)
        controller = cfg["force"]["controller"]
        if controller not in ("existing", "bridge_loop"):
            raise ValueError("force.controller must be existing or bridge_loop")
        target = target_as_mean(feedback_target(cfg, "SQUEEZE"), definition)
        band = force_value(cfg, "target_tolerance") / (2. if definition == "sum" else 1.)
        if target + band >= force_value(cfg, "max"):
            raise ValueError("Selected force target/band must be below per-finger F_max")
        if target-band <= force_value(cfg,"contact_min"):
            raise ValueError("Selected target band must lie above each finger contact threshold")
        v = cfg["force"]["validation"]
        if type(v["repetitions"]) is not int or v["repetitions"] < 1:
            raise ValueError("force.validation.repetitions must be a positive integer")
        for key in ("hold_s", "evaluation_target", "max_contact_peak", "max_settle_s", "max_steady_std"):
            if isinstance(v[key], bool) or not math.isfinite(float(v[key])) or float(v[key]) <= 0:
                raise ValueError("force.validation." + key + " must be finite and positive")
        if controller == "existing":
            from .existing_force import validate_existing_settings
            validate_existing_settings(cfg)
        errors.extend(close_quality_report(cfg)["errors"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("force/close configuration: " + str(exc))
    return errors


def normal_stream_name(cfg):
    return "normal_force_" + force_unit(cfg)


def force_report_binding(cfg):
    """Bind evidence to all actuation/sensor settings and executed source versions."""
    selected = {key: copy.deepcopy(cfg.get(key)) for key in (
        "force", "hardware", "tool", "gripper", "recording", "tactile", "motion", "zero",
        "zero_guard", "close", "probe", "hold")}
    force = selected["force"]
    force.get("validation", {}).pop("report_path", None)
    if isinstance(selected.get("recording"),dict):
        selected["recording"].pop("stress",None)  # independent offline test load, not a controller parameter
    # Choosing a report does not change the tested controller. Its thresholds do.
    if force.get("controller", "bridge_loop") != "existing":
        force.pop("existing", None)
    root = Path(__file__).resolve().parents[1]
    files = ["bottle_scan/force_control.py", "bottle_scan/existing_force.py",
             "bottle_scan/hardware_gripper.py", "bottle_scan/hardware_sensors.py",
             "bridge_ros_public.py", "bottle_scan/runtime.py", "bottle_scan/zero_guard.py",
             "bottle_scan/command_ownership.py", "validate_force.py"]
    sources = {}
    for name in files:
        path = root / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
    original = cfg.get("hardware", {}).get("public_ros", {}).get("force_source_path")
    if original:
        path = Path(original)
        if path.is_file():
            sources["original_controller"] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif not cfg.get("demo"):
            sources["original_controller"] = "unavailable: " + str(original)
    return {"config_sha256": fingerprint(selected), "source_sha256": fingerprint(sources), "sources": sources}


def summarize_force_cycle(samples, cfg):
    """Evaluate timestamped independent feedback, never command target snapshots."""
    if not samples:
        raise ValueError("No force observations recorded")
    unit, definition = force_unit(cfg), target_definition(cfg)
    target = float(cfg["force"]["validation"].get("evaluation_target", force_value(cfg, "target")))
    tolerance = force_value(cfg, "target_tolerance")
    contact = force_value(cfg, "contact_min")
    stamps = np.array([float(x["t_s"]) for x in samples])
    normals = np.array([force_components(x["normal"])["per_finger"] for x in samples])
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps) < 0):
        raise ValueError("Force observations need finite monotonically increasing timestamps")
    touch = np.flatnonzero(np.any(normals > contact, axis=1))
    phases = [x["phase"] for x in samples]
    hold = np.array([phase == "HOLD" for phase in phases])
    squeeze = np.flatnonzero(np.array(phases) == "SQUEEZE")
    settled_at, stable_start = None, None
    stable_s = float(cfg["force"]["stable_s"])
    for i in squeeze:
        if in_target_band(normals[i], target, tolerance, definition):
            stable_start = stamps[i] if stable_start is None else stable_start
            if stamps[i] - stable_start >= stable_s - 1e-9:
                settled_at = stamps[i]
                break
        else:
            stable_start = None
    force_stats = {}
    values = {"left": normals[:, 0], "right": normals[:, 1],
              "mean": normals.mean(axis=1), "sum": normals.sum(axis=1)}
    for name, data in values.items():
        force_stats[name] = {
            "maximum": float(data.max()),
            "contact_peak": None if not touch.size else float(data[touch[0]:].max()),
            "steady_mean": float(data[hold].mean()) if hold.any() else None,
            "steady_min": float(data[hold].min()) if hold.any() else None,
            "steady_std": float(data[hold].std()) if hold.any() else None,
        }
    return {"unit": unit, "target_definition": definition, "target": target,
            "controller_target": force_value(cfg, "target"), "evaluation_target": target,
            "target_tolerance": tolerance, "contact_threshold_per_finger": contact,
            "contact_detected": bool(touch.size),
            "time_to_target_band_s": None if settled_at is None or not squeeze.size else float(settled_at-stamps[squeeze[0]]),
            "settled": settled_at is not None, "statistics": force_stats,
            "observations": len(samples), "hold_observations": int(hold.sum())}


def evaluate_force_report(report, cfg):
    """Configured acceptance tests; missing observations/thresholds cannot pass."""
    limits = cfg["force"]["validation"]
    failures = []
    trials = report.get("trials", [])
    if len(trials) != int(limits["repetitions"]):
        failures.append("Incomplete requested repetitions")
    numeric = {}
    for key in ("max_contact_peak", "max_settle_s", "max_steady_std"):
        try:
            numeric[key] = float(limits[key])
            if not math.isfinite(numeric[key]) or numeric[key] <= 0:
                raise ValueError(key)
        except (TypeError, ValueError):
            failures.append("Unmeasured acceptance limit: force.validation." + key)
    for trial in trials:
        label = str(trial.get("trial_id", "unknown"))
        if trial.get("status") != "completed" or not trial.get("settled") or not trial.get("contact_detected"):
            failures.append(label + ": incomplete contact/settling")
        if not trial.get("hold_observations"):
            failures.append(label + ": no HOLD observations")
        delay = trial.get("time_to_target_band_s")
        if delay is None or ("max_settle_s" in numeric and delay > numeric["max_settle_s"]):
            failures.append(label + ": target band timing failed")
        definitions = target_definition(cfg)
        selected = ("left", "right") if definitions == "per_finger" else (definitions,)
        expected = float(limits.get("evaluation_target", feedback_target(cfg, "SQUEEZE")))
        band = force_value(cfg, "target_tolerance")
        for side in selected:
            steady = trial.get("statistics", {}).get(side, {}).get("steady_mean")
            if steady is None or not math.isfinite(float(steady)) or abs(float(steady)-expected) > band:
                failures.append(label + ": " + side + " HOLD mean outside measured feedback target band")
        for side in ("left", "right"):
            stats = trial.get("statistics", {}).get(side, {})
            minimum = stats.get("steady_min")
            if minimum is None or not math.isfinite(float(minimum)) or minimum <= force_value(cfg, "contact_min"):
                failures.append(label + ": " + side + " HOLD contact lost/missing")
            for key, limit in (("contact_peak", "max_contact_peak"), ("steady_std", "max_steady_std")):
                value = stats.get(key)
                if value is None or not math.isfinite(float(value)) or (limit in numeric and value > numeric[limit]):
                    failures.append(label + ": " + side + " " + key + " failed")
            if stats.get("maximum") is None or stats["maximum"] > force_value(cfg, "max"):
                failures.append(label + ": " + side + " F_max exceeded/missing")
    return failures


def require_force_validation(cfg, *, report_path=None):
    path = Path(report_path or cfg["force"]["validation"]["report_path"])
    if not path.is_file():
        raise ValueError("Formal scanning requires an operator-confirmed force validation report: " + str(path))
    report = json.loads(path.read_text())
    if report.get("schema") != "bottle_force_validation_v2" or report.get("simulated") is not False:
        raise ValueError("Synthetic/unknown force report cannot authorize physical scanning")
    if report.get("operator_confirmation") != "PASS" or report.get("passed") is not True:
        raise ValueError("Force validation requires recorded operator PASS and successful measurements")
    if report.get("binding") != force_report_binding(cfg):
        raise ValueError("Force controller/source/config changed since validation")
    if report.get("controller") != cfg["force"]["controller"]:
        raise ValueError("Force report belongs to another controller")
    errors = evaluate_force_report(report, cfg)
    if errors:
        raise ValueError("Force report no longer meets measured acceptance: " + "; ".join(errors))
    return report


# Public gate used by the formal scan CLI.
require_force_report = require_force_validation
