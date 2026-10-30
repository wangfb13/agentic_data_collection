"""Read-only checks required before every host software force offset update.

No robot command or SDK reset belongs here. A constant load cannot be detected
from force variance alone, so width and DD evidence (or the first-session human
confirmation) are independently required and retained in the report.
"""
from __future__ import annotations

import math
import copy
import time

import numpy as np
from .force_control import force_unit


class ZeroGuardError(RuntimeError):
    pass


def validate_zero_guard_settings(cfg):
    errors = []
    options = cfg.get("zero_guard", {})
    for canonical in ("force_std_max", "dd_contact_threshold", "offset_delta_warn", "timeout_s"):
        key = canonical if canonical in options else canonical+"_N" if canonical in ("force_std_max", "offset_delta_warn") else canonical
        value = options.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            errors.append("zero_guard." + key + " requires a measured positive value")
    fraction = options.get("dd_contact_max_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        errors.append("zero_guard.dd_contact_max_fraction must be measured in [0,1]")
    count = options.get("sample_count")
    if type(count) is not int or not 100 <= count <= 4096:
        errors.append("zero_guard.sample_count must be 100..4096 independent frames per finger")
    if isinstance(options.get("timeout_s"), (int, float)) and options["timeout_s"] > 60:
        errors.append("zero_guard.timeout_s must be at most 60 seconds")
    return errors


def guarded_zero_report(cfg, source, width_reader, *, reference=None, confirm=None, previous=None):
    """Validate evidence, then return proposed offsets; this never applies them.

    ``source`` supplies read-only raw snapshots and new independent wrench frames.
    Width and DD checks run before sampling, during waits, and immediately before
    accepting the proposed offset. Existing ``unloaded_confirmed`` is ignored.
    """
    errors = validate_zero_guard_settings(cfg)
    if errors:
        raise ZeroGuardError("; ".join(errors))
    guard = cfg["zero_guard"]
    force = cfg["hardware"]["public_ros"]["sensors"]["force_conversion"]
    unit = force_unit(cfg)
    scale = float(force["force_scale_to_N"]) if unit == "N" else 1.
    std_limit = guard.get("force_std_max", guard.get("force_std_max_N"))
    drift_limit = guard.get("offset_delta_warn", guard.get("offset_delta_warn_N"))
    options = cfg["gripper"]
    inspected = []

    def inspect():
        width = width_reader()
        value = width.get("width_mm")
        stamp = width.get("monotonic_ns")
        if (not width.get("fresh") or not width.get("valid") or value is None
                or not np.isfinite(value) or not isinstance(stamp, (int, np.integer))
                or not 0 <= source.clock()-int(stamp) <= source.max_age_ns):
            raise ZeroGuardError("Software zero refused: fresh measured gripper width is required")
        if abs(float(value)-options["open_mm"]) > options["open_tolerance_mm"]:
            raise ZeroGuardError("Software zero refused: gripper width is outside w_open +/- tolerance")
        observed = source.observation_snapshot()
        checks = {"width_mm": float(value), "width_timestamp_ns": int(stamp),
                  "open_mm": options["open_mm"], "open_tolerance_mm": options["open_tolerance_mm"],
                  "dd_source_fids": {side: observed["source_sequences"]["dd_"+side]
                                     for side in ("left", "right")}, "dd": {}}
        if reference is not None:
            channel = cfg["tactile"]["depth_channel"]
            for i, side in enumerate(("left", "right")):
                current = np.asarray(observed["dd"][i], dtype=float)
                unloaded = np.asarray(reference[side+"_dd"], dtype=float)
                if current.shape != unloaded.shape or not np.isfinite(unloaded).all():
                    raise ZeroGuardError("Software zero refused: unloaded DD reference geometry/values invalid")
                if current.ndim != 3 or type(channel) is not int or not 0 <= channel < current.shape[-1]:
                    raise ZeroGuardError("Software zero refused: explicit DD depth channel required")
                residual = np.maximum(current[..., channel]-unloaded[..., channel], 0.)
                fraction = float(np.mean(residual > guard["dd_contact_threshold"]))
                checks["dd"][side] = {"contact_fraction": fraction, "max_positive_residual": float(residual.max()),
                                       "threshold": guard["dd_contact_threshold"],
                                       "max_contact_fraction": guard["dd_contact_max_fraction"]}
                if fraction > guard["dd_contact_max_fraction"]:
                    raise ZeroGuardError("Software zero refused: " + side + " DD has contact relative to unloaded reference")
        # Retain endpoints and number of intermediate read-only checks, not a
        # repeated copy of every DD image.
        inspected.append(checks)
        return checks

    inspect()
    first_confirmation = reference is None
    if first_confirmation:
        answer = confirm("本 session 尚无空载 DD 参考。确认两指完全空载、没有接触物体，输入 EMPTY 才允许扣零：") if callable(confirm) else None
        if not isinstance(answer, str) or answer.strip() != "EMPTY":
            raise ZeroGuardError("Software zero refused: first-session EMPTY confirmation was not given")
        # Operator deliberation must not exempt the now-current state from checks.
        inspect()
    samples = source.collect_raw_zero_samples(guard["sample_count"], guard["timeout_s"], monitor=inspect)
    fingers = {}
    means = {}
    for side in ("left", "right"):
        rows = samples[side]
        values = np.asarray([row[1] for row in rows], dtype=float)
        fids = [int(row[2]) for row in rows]
        stamps = [int(row[0]) for row in rows]
        if (len(rows) != guard["sample_count"] or values.shape != (guard["sample_count"], 6)
                or not np.isfinite(values).all() or any(b <= a for a, b in zip(fids, fids[1:]))
                or any(b <= a for a, b in zip(stamps, stamps[1:]))):
            raise ZeroGuardError("Software zero refused: need independent increasing source frames for " + side)
        normal = values[:, force["normal_axis"]] * scale * force["normal_signs"][side]
        deviation = float(np.std(normal))
        if deviation >= std_limit:
            raise ZeroGuardError("Software zero refused: " + side + " normal-force fluctuation exceeds threshold")
        means[side] = values.mean(axis=0)
        fingers[side] = {"offset_raw": means[side].tolist(), "samples_raw": values.tolist(),
                         "source_fids": fids, "host_monotonic_ns": stamps,
                         "normal_std_"+unit: deviation, "normal_peak_to_peak_"+unit: float(np.ptp(normal)),
                         "source_gap_count": sum(max(0, b-a-1) for a, b in zip(fids, fids[1:]))}
    final_check = inspect()
    warnings, differences = [], {}
    if previous is not None:
        if previous.get("serials") != source.serials or previous.get("force_conversion") != force:
            warnings.append("Previous software zero uses different source/calibration; numeric comparison unavailable")
        else:
            for side in ("left", "right"):
                old = np.asarray(previous["fingers"][side]["offset_raw"], dtype=float)
                delta = means[side]-old
                normal_delta = float(delta[force["normal_axis"]]*scale*force["normal_signs"][side])
                force_delta = delta[:3]*scale
                differences[side] = {"offset_raw_delta": delta.tolist(), "normal_delta_"+unit: normal_delta,
                                     "force_vector_delta_"+unit: force_delta.tolist(),
                                     "force_max_component_delta_"+unit: float(np.max(np.abs(force_delta)))}
                if np.max(np.abs(force_delta)) > drift_limit:
                    warnings.append(side + " software zero drift exceeds zero_guard.offset_delta_warn ("+unit+")")
    return {"status": "guard_passed", "method": "guarded_host_software_mean_no_SDK_reset",
            "host_unix_ns": time.time_ns(), "host_monotonic_ns": int(source.clock()),
            "samples_per_finger": guard["sample_count"], "source_unit": "sdk_raw", "force_unit": unit,
            "force_conversion": copy.deepcopy(force), "serials": dict(source.serials), "fingers": fingers,
            "guard_settings": dict(guard), "first_session_empty_confirmed": first_confirmation,
            "unloaded_reference_present": reference is not None,
            "initial_check": inspected[0], "final_check": final_check, "state_check_count": len(inspected),
            "previous_zero_difference": differences, "warnings": warnings}
