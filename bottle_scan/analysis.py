"""Offline analysis of raw bottle-scan data. Never imports a robot driver.

DD is retained unchanged on disk. Descriptor computation alone subtracts the
nearest timed unloaded reference and clips negative depth residuals to zero.
Pixel coordinates have u=column, v=row; millimetre coordinates are centred on
the sensor image. Unknown calibration produces NaN, never invented SI units.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default


def _pitch(value: Any) -> tuple[float, float] | None:
    try:
        values = np.asarray(value, dtype=float).ravel()
        if len(values) == 1:
            values = np.repeat(values, 2)
        if len(values) == 2 and np.all(np.isfinite(values)) and np.all(values > 0):
            return float(values[0]), float(values[1])
    except (TypeError, ValueError):
        pass
    return None


def dd_descriptors(dd: np.ndarray, reference: np.ndarray, *, depth_channel: int | None,
                   threshold: float, pixel_pitch_mm: Any, edge_width_px: int = 2) -> dict[str, float]:
    """Describe positive baseline-subtracted depth, without changing raw arrays.

For a scalar H×W field depth_channel must be None. For H×W×C it must
explicitly select the depth/deformation component. DD units remain the SDK's
units; integrated_deformation additionally carries mm² when pitch is known.
    """
    raw = np.asarray(dd, dtype=np.float64)
    baseline = np.asarray(reference, dtype=np.float64)
    if raw.shape != baseline.shape:
        raise ValueError(f"DD/reference shape mismatch: {raw.shape} versus {baseline.shape}")
    if raw.ndim == 3:
        if isinstance(depth_channel, bool) or not isinstance(depth_channel, int):
            raise ValueError("H×W×C DD requires an explicitly configured integer depth_channel")
        if not 0 <= depth_channel < raw.shape[2]:
            raise ValueError("depth_channel is outside raw DD channels")
        raw, baseline = raw[..., depth_channel], baseline[..., depth_channel]
    elif raw.ndim != 2 or depth_channel is not None:
        raise ValueError("Scalar DD requires H×W and depth_channel=None")
    if not np.isfinite(raw).all() or not np.isfinite(baseline).all():
        raise ValueError("DD/reference contains nonfinite values")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("DD threshold must be finite and nonnegative")
    if isinstance(edge_width_px, bool) or not isinstance(edge_width_px, int) or edge_width_px < 0:
        raise ValueError("edge_width_px must be a nonnegative integer")
    weights = np.maximum(raw - baseline, 0)
    height, width = weights.shape
    v, u = np.indices(weights.shape)
    total = float(weights.sum())
    contact_pixels = int(np.count_nonzero(weights > threshold))
    edge = (u < edge_width_px) | (u >= width - edge_width_px) | (v < edge_width_px) | (v >= height - edge_width_px)
    pitch = _pitch(pixel_pitch_mm)
    result = {
        "depth_weight_sum": total,
        "area_px": contact_pixels,
        "area_mm2": contact_pixels * pitch[0] * pitch[1] if pitch else math.nan,
        "integrated_deformation_sdk_mm2": total * pitch[0] * pitch[1] if pitch else math.nan,
        "centroid_u_px": math.nan, "centroid_v_px": math.nan,
        "centroid_u_centered_px": math.nan, "centroid_v_centered_px": math.nan,
        "centroid_u_mm": math.nan, "centroid_v_mm": math.nan,
        "edge_fraction": math.nan, "principal_angle_deg": math.nan,
        "aspect_ratio": math.nan,
        "moment_uu_px2": math.nan, "moment_uv_px2": math.nan, "moment_vv_px2": math.nan,
        "moment_uu_mm2": math.nan, "moment_uv_mm2": math.nan, "moment_vv_mm2": math.nan,
    }
    if total <= 0:
        return result
    cu, cv = float((weights * u).sum() / total), float((weights * v).sum() / total)
    result.update(centroid_u_px=cu, centroid_v_px=cv,
                  centroid_u_centered_px=cu-(width-1)/2, centroid_v_centered_px=cv-(height-1)/2,
                  edge_fraction=float(weights[edge].sum() / total))
    # Moment geometry is in mm when calibration exists, otherwise in pixels.
    du, dv = u - cu, v - cv
    cov_px = np.array([[(weights * du * du).sum(), (weights * du * dv).sum()],
                       [(weights * du * dv).sum(), (weights * dv * dv).sum()]]) / total
    result.update(moment_uu_px2=float(cov_px[0, 0]), moment_uv_px2=float(cov_px[0, 1]),
                  moment_vv_px2=float(cov_px[1, 1]))
    if pitch:
        result.update(centroid_u_mm=(cu - (width - 1) / 2) * pitch[0],
                      centroid_v_mm=(cv - (height - 1) / 2) * pitch[1])
        du, dv = du * pitch[0], dv * pitch[1]
    cov = np.array([[(weights * du * du).sum(), (weights * du * dv).sum()],
                    [(weights * du * dv).sum(), (weights * dv * dv).sum()]]) / total
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major = eigenvectors[:, -1]
    angle = math.degrees(math.atan2(major[1], major[0]))
    if eigenvalues[-1] > 1e-15:
        # A circular/isotropic patch has no identifiable principal direction.
        if not math.isclose(float(eigenvalues[0]), float(eigenvalues[1]), rel_tol=1e-8, abs_tol=1e-15):
            result["principal_angle_deg"] = (angle + 90) % 180 - 90
        result["aspect_ratio"] = math.sqrt(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 1e-15 else math.inf
    if pitch:
        result.update(moment_uu_mm2=float(cov[0, 0]), moment_uv_mm2=float(cov[0, 1]), moment_vv_mm2=float(cov[1, 1]))
    return result


def center_of_pressure(wrench: np.ndarray, *, moment_unit: str | None,
                       force_unit: str | None, plane_z_mm: float | None,
                       min_normal: float = 1e-8, normal_axis: int | None = 2) -> dict[str, float]:
    """CoP on sensor-local z=plane_z_mm, under a no-free-bending-moment model.

Wrench order is Fx,Fy,Fz,Mx,My,Mz at the documented sensor origin. Unknown
moment units, force units or origin-to-plane distance deliberately yield NaN.
    """
    result = {"cop_u_mm": math.nan, "cop_v_mm": math.nan}
    if normal_axis != 2:
        return result  # This formula explicitly assumes the contact normal is sensor z.
    w = np.asarray(wrench, dtype=float)
    if w.shape != (6,) or not np.isfinite(w).all():
        return result
    if force_unit != "N" or moment_unit not in ("Nmm", "N_mm", "Nm", "N_m"):
        return result
    plane_z_mm = _finite_float(plane_z_mm)
    if not math.isfinite(plane_z_mm) or abs(w[2]) <= min_normal:
        return result
    factor = 1000 if moment_unit in ("Nm", "N_m") else 1
    fx, fy, fz, mx, my, _ = w
    result["cop_u_mm"] = float((plane_z_mm * fx - my * factor) / fz)
    result["cop_v_mm"] = float((mx * factor + plane_z_mm * fy) / fz)
    return result


def linear_fit(x: Any, y: Any) -> dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    result = {"n": int(len(x)), "slope": math.nan, "intercept": math.nan, "r2": math.nan}
    if len(x) < 2 or np.ptp(x) < 1e-12:
        return result
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    residual = float(np.square(y - (slope * x + intercept)).sum())
    variance = float(np.square(y - y.mean()).sum())
    result.update(slope=float(slope), intercept=float(intercept),
                  r2=1 - residual / variance if variance > 1e-15 else math.nan)
    return result


def nearest_reference(references: list[dict[str, Any]], monotonic_ns: int) -> dict[str, Any]:
    """Closest reference in session-host monotonic time; earlier wins a tie."""
    if not references:
        raise ValueError("Session contains no unloaded references")
    def timestamp(entry: dict[str, Any]) -> int:
        return int(entry.get("monotonic_ns", entry.get("timestamp_ns")))
    return min(references, key=lambda entry: (abs(timestamp(entry) - monotonic_ns), timestamp(entry)))


def _read_reference(path: Path, side: str) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive[f"{side}_dd"])
    with h5py.File(path, "r") as handle:
        for key in (f"{side}_dd", f"dd_{side}", f"streams/dd_{side}/data"):
            if key in handle:
                data = np.asarray(handle[key])
                return data.mean(axis=0) if key.endswith("/data") else data
    raise ValueError(f"No raw DD reference for {side} in {path}")


def _stream(handle: h5py.File, *names: str) -> dict[str, Any] | None:
    for name in names:
        key = f"streams/{name}"
        if key not in handle:
            continue
        group = handle[key]
        if "data" not in group or "monotonic_ns" not in group:
            continue
        count = int(group.attrs.get("committed_count", len(group["data"])))
        lengths = [len(group[key]) for key in ("data", "monotonic_ns")]
        if "metadata_json" in group:
            lengths.append(len(group["metadata_json"]))
        if count < 0 or count > min(lengths):
            raise ValueError(f"Invalid committed_count for stream: {name}")
        metadata = []
        if "metadata_json" in group:
            for value in group["metadata_json"][:count]:
                metadata.append(json.loads(value.decode() if isinstance(value, bytes) else value))
        timestamps = np.asarray(group["monotonic_ns"][:count], dtype=np.int64)
        if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
            raise ValueError(f"Stream clock is not monotonic: {name}")
        return {"data": group["data"], "times": timestamps, "metadata": metadata}
    return None


def _stage_times(meta: dict[str, Any], stage: str) -> tuple[int, int] | None:
    stages = meta.get("stages", [])
    if isinstance(stages, dict):
        stages = [{"name": key, **value} for key, value in stages.items()]
    matches = [entry for entry in stages if entry.get("name", entry.get("stage")) == stage]
    if not matches:
        return None
    entry = matches[-1]
    start, end = entry.get("start_ns"), entry.get("end_ns")
    if start is None or end is None:
        return None
    return int(start), int(end)


def _stage_frame(stream: dict[str, Any], meta: dict[str, Any], stage: str) -> int | None:
    timing = _stage_times(meta, stage)
    times = stream["times"]
    if not len(times):
        return None
    if timing:
        eligible = np.flatnonzero((times >= timing[0]) & (times <= timing[1]))
        midpoint = sum(timing) // 2
    else:
        eligible = np.asarray([i for i, entry in enumerate(stream["metadata"]) if entry.get("stage") == stage], dtype=int)
        midpoint = (int(times[eligible[0]]) + int(times[eligible[-1]])) // 2 if len(eligible) else 0
    if not len(eligible):
        return None
    return int(eligible[-1] if stage == "PROBE" else eligible[np.argmin(np.abs(times[eligible] - midpoint))])


def _load_config(session: Path) -> dict[str, Any]:
    path = session / "config_snapshot.yaml"
    if path.exists():
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    return {}


def _settings(config: dict[str, Any], side: str) -> dict[str, Any]:
    tactile = config.get("tactile", {})
    analysis = config.get("analysis", {})
    finger = tactile.get(side, tactile.get("fingers", {}).get(side, {}))
    if not isinstance(finger, dict):
        finger = {}
    depth = finger.get("depth_channel", analysis.get("depth_channel", tactile.get("depth_channel")))
    return {
        "depth_channel": depth,
        "threshold": _finite_float(analysis.get("dd_threshold", tactile.get("dd_threshold", 0.1)), 0.1),
        "pixel_pitch_mm": finger.get("pixel_pitch_mm", tactile.get("pixel_pitch_mm", tactile.get("dd_pixel_pitch_mm"))),
        "edge_width_px": int(analysis.get("edge_width_px", 2)),
        "force_unit": config.get("force", {}).get("unit", finger.get("force_unit", tactile.get("force_unit"))),
        "moment_unit": finger.get("moment_unit", tactile.get("moment_unit")),
        "plane_z_mm": finger.get("cop_plane_z_mm", tactile.get("cop_plane_z_mm")),
        "normal_axis": finger.get("normal_axis", tactile.get("normal_axis")),
        "projection": finger.get("centroid_to_al", analysis.get("centroid_to_al", {}).get(side)),
    }


def _projection(desc: dict[str, float], matrix: Any) -> dict[str, float]:
    result = {"centroid_a_mm": math.nan, "centroid_l_mm": math.nan,
              "centroid_a_px": math.nan, "centroid_l_px": math.nan}
    try:
        transform = np.asarray(matrix, float)
        if transform.shape != (2, 2) or not np.isfinite(transform).all():
            return result
        value = transform @ np.array([desc["centroid_u_mm"], desc["centroid_v_mm"]])
        result.update(centroid_a_mm=float(value[0]), centroid_l_mm=float(value[1]))
        pixel = transform @ np.array([desc["centroid_u_centered_px"], desc["centroid_v_centered_px"]])
        result.update(centroid_a_px=float(pixel[0]), centroid_l_px=float(pixel[1]))
    except (TypeError, ValueError):
        pass
    return result


def pose_errors(commands: dict[str, Any], measured: dict[str, Any], *, max_age_s: float = 0.2) -> list[dict[str, Any]]:
    """Pair each measured pose with the latest command at or before it.

No future command is used. Commands older than max_age_s are excluded and
counted separately by the caller, because a stale association is misleading.
    """
    result = []
    if not len(commands["times"]):
        return result
    for index, timestamp in enumerate(measured["times"]):
        command_index = int(np.searchsorted(commands["times"], timestamp, side="right") - 1)
        if command_index < 0:
            continue
        age_s = (int(timestamp) - int(commands["times"][command_index])) / 1e9
        if age_s > max_age_s:
            continue
        actual = np.asarray(measured["data"][index], float)
        target = np.asarray(commands["data"][command_index], float)
        if actual.shape != (4, 4) or target.shape != (4, 4) or not np.isfinite(actual).all() or not np.isfinite(target).all():
            raise ValueError("TCP streams must contain finite 4×4 matrices in mm")
        delta = actual[:3, 3] - target[:3, 3]
        rotation = target[:3, :3].T @ actual[:3, :3]
        angle = math.degrees(math.acos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        metadata = measured["metadata"][index] if index < len(measured["metadata"]) else {}
        result.append({"monotonic_ns": int(timestamp), "stage": metadata.get("stage", "UNKNOWN"),
                       "command_age_s": age_s, "dx_mm": float(delta[0]), "dy_mm": float(delta[1]),
                       "dz_mm": float(delta[2]), "translation_mm": float(np.linalg.norm(delta)),
                       "rotation_deg": angle})
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], default_fields: tuple[str, ...] = ("trial_id",)) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(default_fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _outcomes(meta: dict[str, Any], unit: str = "N") -> dict[str, Any]:
    auto = meta.get("auto", meta.get("auto_results", {}))
    operator = meta.get("operator", {})
    label = operator.get("label", meta.get("operator_label", ""))
    flags = auto.get("flags", meta.get("flags", []))
    if isinstance(flags, str):
        flags = [flags]
    def flag(name: str) -> bool:
        return bool(auto.get(name, False) or name in flags or meta.get("outcome") == name)
    invalid = label == "x"
    errors = auto.get("exceptions", [])
    reached_hold = _stage_times(meta, "HOLD") is not None
    normal = (not invalid and meta.get("status") not in ("aborted", "running", "skipped_safety")
              and label in ("", "normal", "enter") and reached_hold
              and bool(auto.get("bilateral_contact")) and bool(auto.get("force_reached"))
              and not errors and not any(flag(name) for name in ("dropped", "slip_suspected", "no_bilateral_contact", "stall_without_contact", "force_not_reached")))
    times = list(auto.get("contact_times_ns") or [])
    widths = list(auto.get("contact_widths_mm") or [])
    left_time, right_time = (_finite_float(value) for value in (times + [None, None])[:2])
    left_width, right_width = (_finite_float(value) for value in (widths + [None, None])[:2])
    force_fields = {"force_unit": unit, "force_target_definition": auto.get("force_target_definition", meta.get("force_target_definition", "unknown"))}
    for name in ("normal_pre_lift", "normal_hold"):
        value = auto.get(name, auto.get(name + "_N") if unit == "N" else None)
        force_fields[name+"_"+unit] = value
        pair = list(value) if isinstance(value, (list, tuple)) and len(value) == 2 else [None, None]
        for side, measurement in zip(("left", "right"), pair):
            force_fields[f"{name}_{side}_{unit}"] = _finite_float(measurement)
    for side in ("left", "right"):
        force_fields[f"normal_change_{side}_{unit}"] = force_fields[f"normal_hold_{side}_{unit}"] - force_fields[f"normal_pre_lift_{side}_{unit}"]
    if isinstance(force_fields["normal_pre_lift_"+unit], (list, tuple)) or isinstance(force_fields["normal_hold_"+unit], (list, tuple)):
        force_fields["normal_change_"+unit] = [force_fields[f"normal_change_{side}_{unit}"] for side in ("left", "right")]
    else:
        # Old scalar measurements were explicitly aggregated; do not fabricate
        # two per-finger values from them.
        force_fields["normal_change_"+unit] = _finite_float(force_fields["normal_hold_"+unit]) - _finite_float(force_fields["normal_pre_lift_"+unit])
    force_fields["hold_vertical_tangent_sum_"+unit] = auto.get("hold_vertical_tangent_sum", auto.get("hold_vertical_tangent_sum_N") if unit == "N" else None)
    return {"operator_label": label, "operator_note": operator.get("note", ""), "invalid": invalid,
            "normal_completion": normal, "toppled": label == "t", "moved": label == "m",
            "dropped": flag("dropped") or label == "d", "slip_suspected": flag("slip_suspected"),
            "slip_observed": label == "s", "no_bilateral_contact": flag("no_bilateral_contact"),
            "stall_without_contact": flag("stall_without_contact"), "force_not_reached": flag("force_not_reached"),
            "bilateral_contact": auto.get("bilateral_contact"), "force_reached": auto.get("force_reached"),
            "contact_delta_s": (right_time - left_time) / 1e9,
            "contact_width_left_mm": left_width, "contact_width_right_mm": right_width,
            **force_fields,
            "exceptions": json.dumps(errors, ensure_ascii=False), "nominal_region": auto.get("nominal_region", meta.get("nominal_region", "unknown"))}


def _trial_offsets(meta, session_meta, config):
    """Use this session's T0 grasp center; legacy logs retain declared offsets."""
    old = meta.get("offsets", meta.get("offset", {}))
    result = {key: old.get(key, 0.) for key in ("a_mm", "c_mm", "l_mm", "rot_c_deg", "rot_other_deg")}
    try:
        from .geometry import grasp_pose, axis_basis
        T0 = np.asarray(session_meta.get("T0", config.get("T0")), float)
        target = np.asarray(meta["T_target"], float)
        center = grasp_pose(target, config)[:3, 3]
        difference = center-grasp_pose(T0, config)[:3, 3]
        delta = axis_basis(config["axes"]).T @ T0[:3, :3].T @ difference
        result.update({key+"_mm": float(value) for key, value in zip(("a", "c", "l"), delta)})
        result.update({"offset_basis": "target_grasp_center_relative_to_session_T0_grasp_center",
            "grasp_center_base_mm": center.tolist(), "grasp_center_offset_base_mm": difference.tolist()})
    except (KeyError, ValueError, TypeError):
        result["offset_basis"] = "legacy_declared_offset_unverified_grasp_center"
    return result


def _quality_fields(meta, unit):
    auto = meta.get("auto", {})
    result = {key: auto.get(key) for key in ("approach_translation_error_mm", "approach_rotation_error_deg",
        "probe_not_settled", "lower_unload_height_difference_mm")}
    for side in ("left", "right"):
        result["probe_snr_"+side] = auto.get("probe_snr", {}).get(side)
    for field in ("contact_peak", "contact_peak_force", "contact_overshoot"):
        if field in auto:
            value = auto[field]
            if isinstance(value, (list, tuple)):
                for side, item in zip(("left", "right"), value):
                    result[field+"_"+side+"_"+unit] = item
            elif isinstance(value, dict):
                for key, item in value.items():
                    result[field+"_"+str(key)+"_"+unit] = item
            else:
                result[field+"_"+unit] = value
    for controller, values in auto.get("control_loop_periods", {}).items():
        if isinstance(values, dict):
            for key, value in values.items():
                result[str(controller)+"_loop_"+str(key)] = value
    checks = meta.get("position_checks", [])
    result["check_attempt_count"] = len(checks)
    # Preserve every check attempt, including the failed pre-reset comparison.
    result["check_results_json"] = json.dumps(checks, ensure_ascii=False)
    for camera in ("left_wrist", "right_wrist"):
        observed = [entry.get("cameras", {}).get(camera, {}) for entry in checks]
        values = [entry.get("displacement_px", entry.get("offset_px")) for entry in observed]
        distances = [entry.get("distance_px", entry.get("offset_px", entry.get("shift_px"))) for entry in observed]
        result["check_"+camera+"_displacements_px_json"] = json.dumps(values)
        result["check_"+camera+"_distances_px_json"] = json.dumps(distances)
        valid = [_finite_float(value) for value in distances]
        result["check_"+camera+"_max_distance_px"] = max((value for value in valid if math.isfinite(value)), default=math.nan)
    return result


def _pose_statistics(errors):
    result = []
    for stage in sorted({row["stage"] for row in errors}):
        rows = [row for row in errors if row["stage"] == stage]
        row = {"stage": stage, "n": len(rows)}
        for key in ("translation_mm", "rotation_deg"):
            values = np.asarray([entry[key] for entry in rows])
            row.update({f"{key}_mean": float(values.mean()), f"{key}_p95": float(np.percentile(values, 95)), f"{key}_max": float(values.max())})
        result.append(row)
    return result


def analyze_session(session: str | Path, output: str | Path | None = None, *, _return_data=False) -> dict[str, Any]:
    session = Path(session).resolve()
    output = Path(output).resolve() if output else session / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    session_meta = json.loads((session / "session_meta.json").read_text())
    config = _load_config(session)
    references = session_meta.get("references", session_meta.get("reference_files", []))
    descriptors, summary, errors, messages = [], [], [], []
    for side in ("left", "right"):
        settings = _settings(config, side)
        if _pitch(settings["pixel_pitch_mm"]) is None:
            messages.append(f"{side}: DD pixel pitch unavailable; physical area/centroid/moment scale unavailable")
        try:
            projection = np.asarray(settings["projection"], float)
            projection_ok = projection.shape == (2, 2) and np.isfinite(projection).all()
        except (ValueError, TypeError):
            projection_ok = False
        if not projection_ok:
            messages.append(f"{side}: sensor-centroid to a/l mapping unavailable; projected centroid fits unavailable")
        if (settings["force_unit"] != "N" or settings["moment_unit"] not in ("Nmm", "N_mm", "Nm", "N_m")
                or not math.isfinite(_finite_float(settings["plane_z_mm"])) or settings["normal_axis"] != 2):
            messages.append(f"{side}: CoP needs calibrated units, sensor-plane origin and confirmed sensor-z normal; CoP is NaN")
    from .window_descriptors import enrich_trial_windows, source_frame_gaps
    reference_cache = {}
    def reference_at(timestamp):
        selected = nearest_reference(references, timestamp)
        ref_path = Path(selected["path"])
        if not ref_path.is_absolute():
            ref_path = session / ref_path
        if str(ref_path) not in reference_cache:
            if ref_path.suffix == ".npz":
                with np.load(ref_path, allow_pickle=False) as archive:
                    value = {key: np.asarray(archive[key]) for key in archive.files}
            else:
                value = {side+"_dd": _read_reference(ref_path, side) for side in ("left", "right")}
            reference_cache[str(ref_path)] = {**value, "path": selected["path"], "monotonic_ns": selected.get("monotonic_ns", selected.get("timestamp_ns"))}
        return reference_cache[str(ref_path)]
    for meta_path in sorted((session / "trials").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        trial_id = meta.get("trial_id", meta_path.parent.name)
        task = meta.get("task", "preparation" if str(trial_id).startswith("preparation_") else "contact")
        status = meta.get("status", "unknown")
        common = {"session": str(session), "session_name": session.name,
                  "trial_id": trial_id, "task": task, "status": status, "bottle": meta.get("bottle", "unknown"),
                  **_trial_offsets(meta, session_meta, config)}
        common["zero_offset"] = all(abs(_finite_float(common[key], 1)) < 1e-9 for key in ("a_mm", "c_mm", "l_mm", "rot_c_deg", "rot_other_deg"))
        unit = config.get("force", {}).get("unit", config.get("tactile", {}).get("force_unit", "N"))
        row = {**common, **_outcomes(meta, unit), **_quality_fields(meta, unit)}
        row["force_target_definition"] = config.get("force", {}).get("target_definition", row.get("force_target_definition"))
        row["included_in_contact_statistics"] = task == "contact" and status != "skipped_safety" and not row["invalid"]
        path = meta_path.parent / "streams.h5"
        if not path.exists():
            if status != "skipped_safety" and task != "preparation":
                messages.append(f"{trial_id}: streams.h5 missing; summary retained")
            summary.append(row)
            continue
        enrich_trial_windows(meta, path, config, reference_at)
        row.update(_quality_fields(meta, unit))
        windows = meta.get("descriptor_windows", {})
        with h5py.File(path, "r") as handle:
            for side in ("left", "right"):
                dd = _stream(handle, f"dd_{side}", f"tactile/{side}/dd")
                wrench = _stream(handle, f"wrench_{side}", f"tactile/{side}/wrench")
                if dd is None:
                    messages.append(f"{trial_id}/{side}: raw DD stream missing")
                    continue
                for name, stream in (("dd", dd), ("force", wrench)):
                    if stream is not None:
                        gaps = source_frame_gaps(stream)
                        row[f"{name}_{side}_frame_gaps"] = gaps["missing_frames"]
                        row[f"{name}_{side}_missing_source_ids"] = gaps["source_id_missing_count"]
                for stage in ("PROBE", "HOLD"):
                    window = windows.get(stage, {}).get(side)
                    if not window or not window["selected_frames"]:
                        continue
                    desc = {key: math.nan if value is None else value for key, value in window["median"].items()}
                    timestamps = [entry["monotonic_ns"] for entry in window["frames"]]
                    descriptors.append({**common, "finger": side, "stage": stage,
                        "monotonic_ns": int(np.median(timestamps)), "force_unit": unit,
                        "dd_index": window["dd_indices"][-1], "dd_indices_json": json.dumps(window["dd_indices"]),
                        "source_fids_json": json.dumps(window["selected_source_fids"]),
                        "window_frame_count": window["selected_frames"], "window_start_ns": window["start_ns"],
                        "window_end_ns": window["end_ns"], "window_fallback_reason": window["fallback_reason"],
                        "probe_not_settled": window["probe_not_settled"],
                        "dd_window_frame_gaps": window["dd_frame_gaps"]["missing_frames"],
                        "frames_and_actual_force_json": json.dumps(window["frames"], ensure_ascii=False),
                        "probe_snr": window.get("snr", {}).get("value"), **desc})
                    for name, value in desc.items():
                        row[f"{stage.lower()}_{side}_{name}"] = value
            command = _stream(handle, "tcp_command", "tcp/command")
            actual = _stream(handle, "tcp_measured", "tcp/actual")
            if command is not None and actual is not None:
                per_frame = pose_errors(command, actual, max_age_s=config.get("analysis", {}).get("max_pose_command_age_s", 0.2))
                for error in per_frame:
                    if error["stage"] == "UNKNOWN":
                        for stage in meta.get("stages", []):
                            if stage["start_ns"] <= error["monotonic_ns"] <= stage["end_ns"]:
                                error["stage"] = stage["name"]
                                break
                    errors.append({"session": str(session), "trial_id": trial_id, **error})
                row["pose_samples_total"] = len(actual["times"])
                row["pose_samples_paired"] = len(per_frame)
        summary.append(row)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "descriptors.csv", descriptors)
    _write_csv(output / "pose_errors.csv", errors)
    stage_statistics = _pose_statistics(errors)
    _write_csv(output / "pose_error_by_stage.csv", stage_statistics, ("stage", "n"))
    fits = _plots(output, descriptors, summary, stage_statistics, config)
    _write_csv(output / "linear_fits.csv", fits, ("bottle", "stage", "finger", "axis", "slope", "r2", "n"))
    report = {"session": str(session), "output": str(output), "trials": len(summary),
              "descriptor_rows": len(descriptors), "pose_error_rows": len(errors), "warnings": messages,
              "reference_selection": "closest reference to each selected DD frame host monotonic timestamp; earlier wins ties",
              "descriptor_selection": "PROBE median of last valid stable frames (explicit fallback flag); HOLD median across central time window",
              "invalid_trials": "operator x, safety-skipped, visual and preparation records retained in CSV; excluded from contact plots and fit denominators",
              "units": "Raw DD depth-weight sum in SDK units; pixel centroid/area/moments always retained; mm equivalents require measured pitch. Force columns carry configured unit. CoP requires calibrated N, moments and sensor-plane origin.",
              "heatmap": "translation-only c=0 trials; both tilt angles=0; per-cell sample counts exported",
              "normal_completion": "HOLD reached, bilateral contact and force reached, operator normal, no exception/dropped/slip/failure flags"}
    if "zero_drift" in session_meta:
        from .wizard_analysis import write_zero_drift_plot
        report["zero_offset_centroid_drift"] = write_zero_drift_plot(session, output)
    (output / "analysis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if _return_data:
        report["_data"] = {"summary": summary, "descriptors": descriptors, "errors": errors, "config": config}
    return report


def _compatibility_signature(config, metadata):
    tactile = config.get("tactile", {})
    hardware = config.get("hardware", {})
    sensors = hardware.get("public_ros", {}).get("sensors", {})
    return {"G": config.get("tool", {}).get("grasp_center_in_tcp"),
        "DD_processing": {"shape": tactile.get("dd_shape"), "dtype": tactile.get("dd_dtype"),
            "unit": tactile.get("dd_unit"), "source": metadata.get("dd_processing", metadata.get("dd_provenance")),
            "configured_source": sensors.get("dd_processing", sensors.get("dd_source")),
            "processed_dd_accepted": hardware.get("public_ros", {}).get("processed_dd_accepted")},
        "force_unit": config.get("force", {}).get("unit", tactile.get("force_unit")),
        "force_target_definition": config.get("force", {}).get("target_definition"),
        "tool_definition": {key: hardware.get(key) for key in ("tcp_definition", "base_frame", "arm_side")},
        "sensor_pixel_pitch_mm": tactile.get("pixel_pitch_mm"),
        "centroid_projection": config.get("analysis", {}).get("centroid_to_al")}


def analyze_sessions(sessions, output):
    """Write per-session results and an explicitly grouped combined analysis."""
    sessions = [Path(path).resolve() for path in sessions]
    if not sessions or len(set(sessions)) != len(sessions):
        raise ValueError("Provide one or more distinct session directories")
    if output is None:
        raise ValueError("Multiple-session analysis requires an explicit output directory")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries, descriptors, errors, reports, messages, signatures, configs = [], [], [], [], [], [], []
    for index, session in enumerate(sessions, 1):
        destination = output / "per_session" / (str(index).zfill(2)+"_"+session.name)
        report = analyze_session(session, destination, _return_data=True)
        data = report.pop("_data")
        summaries.extend(data["summary"])
        descriptors.extend(data["descriptors"])
        errors.extend(data["errors"])
        configs.append(data["config"])
        metadata = json.loads((session/"session_meta.json").read_text())
        signatures.append(_compatibility_signature(data["config"], metadata))
        reports.append(report)
        messages.extend(str(session)+": "+message for message in report["warnings"])
    for field in signatures[0]:
        representations = {json.dumps(signature[field], sort_keys=True, default=str) for signature in signatures}
        if len(representations) > 1:
            messages.append("SESSION CONFIGURATION MISMATCH: "+field+"; combined results need separate interpretation; per-session outputs retained")
    _write_csv(output/"summary.csv", summaries)
    _write_csv(output/"descriptors.csv", descriptors)
    _write_csv(output/"pose_errors.csv", errors)
    statistics = _pose_statistics(errors)
    _write_csv(output/"pose_error_by_stage.csv", statistics, ("stage", "n"))
    plotting_config = dict(configs[0])
    plotting_config["demo"] = any(config.get("demo") for config in configs)
    fits = _plots(output, descriptors, summaries, statistics, plotting_config)
    _write_csv(output/"linear_fits.csv", fits, ("session", "bottle", "stage", "finger", "axis", "slope", "r2", "n"))
    report = {"sessions": [str(path) for path in sessions], "output": str(output),
        "trials": len(summaries), "descriptor_rows": len(descriptors), "pose_error_rows": len(errors),
        "warnings": messages, "per_session": reports,
        "compatibility": {str(session): signature for session, signature in zip(sessions, signatures)},
        "offset_basis": "each session's T0 grasp center in that session's declared a/c/l axes; legacy offsets explicitly marked",
        "grouping": "session column disambiguates repeated trial IDs; both combined and per_session fits/heatmaps are exported",
        "clock_policy": "separate session monotonic time origins; drift curves never join different session clocks"}
    (output/"analysis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _plots(output: Path, descriptors: list[dict[str, Any]], summary: list[dict[str, Any]],
           pose_statistics: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    identity = lambda row: (row.get("session", ""), row["trial_id"])
    invalid = {identity(row) for row in summary if not row.get("included_in_contact_statistics", not row["invalid"])}
    desc = [row for row in descriptors if identity(row) not in invalid]
    valid = [row for row in summary if identity(row) not in invalid]
    bottles = sorted({row["bottle"] for row in valid})
    sessions = {row.get("session", "") for row in summary}
    group = next(iter(sessions)) if len(sessions) == 1 else "__combined__"
    fits = []
    def save(fig: Any, name: str) -> None:
        if config.get("demo"):
            fig.text(.5, .005, "SYNTHETIC DATA — SOFTWARE VALIDATION ONLY", ha="center", fontsize=9, color="darkred")
            fig.tight_layout(rect=(0,.035,1,1))
        else:
            fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=150)
        plt.close(fig)
    for axis in ("a", "l"):
        for finger in ("left", "right"):
            for unit in ("px", "mm"):
                preferred = f"centroid_{axis}_{unit}"
                fallback = f"centroid_{'u' if axis == 'a' else 'v'}_{unit}"
                coordinate = preferred if any(math.isfinite(_finite_float(row.get(preferred))) for row in desc if row["finger"] == finger) else fallback
                if unit == "mm" and not any(math.isfinite(_finite_float(row.get(coordinate))) for row in desc if row["finger"] == finger):
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
                for ax, stage in zip(axes[0], ("PROBE", "HOLD")):
                    for bottle in bottles:
                        rows = [row for row in desc if row["finger"] == finger and row["stage"] == stage and row["bottle"] == bottle
                                and abs(row["c_mm"]) < 1e-9 and abs(row["rot_c_deg"]) < 1e-9 and abs(row["rot_other_deg"]) < 1e-9]
                        x = np.asarray([row[f"{axis}_mm"] for row in rows], float)
                        y = np.asarray([_finite_float(row.get(coordinate)) for row in rows], float)
                        fit = linear_fit(x, y)
                        fits.append({"session": group, "bottle": bottle, "stage": stage, "finger": finger, "axis": axis,
                            "descriptor": coordinate, "slope_unit": unit+"/mm", **fit})
                        label = f"{bottle}: slope={fit['slope']:.3g} {unit}/mm, R²={fit['r2']:.3g}, n={fit['n']}"
                        ax.scatter(x, y, s=15, label=label)
                        if math.isfinite(fit["slope"]):
                            xx = np.array([np.nanmin(x), np.nanmax(x)])
                            ax.plot(xx, fit["slope"] * xx + fit["intercept"])
                    ax.set(title=f"{finger} / {stage}", xlabel=f"grasp-center offset {axis} (mm)", ylabel=coordinate)
                    ax.legend(fontsize=7)
                    ax.grid(alpha=.2)
                save(fig, f"centroid_vs_{axis}_{finger}_{unit}")
    heat_rows = []
    for bottle in bottles:
        rows = [row for row in valid if row["bottle"] == bottle and abs(row["c_mm"]) < 1e-9
                and abs(row["rot_c_deg"]) < 1e-9 and abs(row["rot_other_deg"]) < 1e-9]
        if not rows:
            continue
        a_values, l_values = sorted({row["a_mm"] for row in rows}), sorted({row["l_mm"] for row in rows})
        metrics = ["normal_completion", "toppled", "no_bilateral_contact", "stall_without_contact",
                   "probe_left_edge_fraction", "probe_right_edge_fraction", "hold_left_edge_fraction", "hold_right_edge_fraction"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
        for metric, ax in zip(metrics, axes.ravel()):
            grid = np.full((len(l_values), len(a_values)), np.nan)
            for j, l in enumerate(l_values):
                for i, a in enumerate(a_values):
                    cell = [row for row in rows if row["a_mm"] == a and row["l_mm"] == l]
                    values = [_finite_float(row.get(metric)) for row in cell]
                    values = [value for value in values if math.isfinite(value)]
                    mean = float(np.mean(values)) if values else math.nan
                    grid[j, i] = mean
                    if cell:
                        heat_rows.append({"session": group, "bottle": bottle, "a_mm": a, "l_mm": l, "metric": metric,
                                          "n_trials": len(cell), "n_valid_metric": len(values), "mean": mean})
            masked = np.ma.masked_invalid(grid)
            im = ax.imshow(masked, origin="lower", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            ax.set(xticks=range(len(a_values)), xticklabels=a_values, yticks=range(len(l_values)), yticklabels=l_values,
                   xlabel="a offset (mm)", ylabel="l offset (mm)", title=metric.replace("_", " "))
            ax.tick_params(axis="x", labelrotation=90, labelsize=7)
            fig.colorbar(im, ax=ax, fraction=.046)
        fig.suptitle(f"{bottle}: translation grid (invalid x excluded)")
        save(fig, f"heatmap_{bottle}")
    _write_csv(output / "heatmap_cells.csv", heat_rows, ("bottle", "a_mm", "l_mm", "metric", "mean", "n_trials"))
    grasp_mode = config.get("grasp_mode", config.get("geometry", {}).get("grasp_mode", "side"))
    bottle_axis = "a" if grasp_mode in ("top", "top_down") else "l"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for bottle in bottles:
        rows = [row for row in valid if row["bottle"] == bottle]
        for finger in ("left", "right"):
            axes[0].scatter([row[f"{bottle_axis}_mm"] for row in rows], [row[f"contact_width_{finger}_mm"] for row in rows], label=f"{bottle}/{finger}", s=16)
        pure_c = [row for row in rows if all(abs(row[key]) < 1e-9 for key in ("a_mm", "l_mm", "rot_c_deg", "rot_other_deg"))]
        axes[1].scatter([row["c_mm"] for row in pure_c], [row["contact_delta_s"] for row in pure_c], label=bottle, s=16)
    axes[0].set(xlabel=f"bottle-axis offset {bottle_axis} (mm)", ylabel="opening at contact (mm)")
    axes[1].set(xlabel="closure-axis c offset (mm)", ylabel="right − left contact onset (s)")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    save(fig, "contact_opening_and_timing")
    drift = [row for row in desc if row["zero_offset"]]
    physical = any(math.isfinite(_finite_float(row.get("centroid_u_mm"))) for row in drift)
    keys = ("depth_weight_sum", "area_mm2" if physical else "area_px", "centroid_u_mm" if physical else "centroid_u_px",
            "centroid_v_mm" if physical else "centroid_v_px", "edge_fraction", "principal_angle_deg", "aspect_ratio", "cop_u_mm", "cop_v_mm")
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    origins = {session: min(row["monotonic_ns"] for row in drift if row.get("session", "") == session)
               for session in {row.get("session", "") for row in drift}}
    for ax, key in zip(axes.ravel(), keys):
        for session, origin in origins.items():
            for bottle in bottles:
                for finger in ("left", "right"):
                    for stage in ("PROBE", "HOLD"):
                        rows = sorted([row for row in drift if row.get("session", "") == session and row["bottle"] == bottle and row["finger"] == finger and row["stage"] == stage], key=lambda row: row["monotonic_ns"])
                        if rows:
                            ax.plot([(row["monotonic_ns"] - origin) / 1e9 for row in rows], [_finite_float(row.get(key)) for row in rows], ".-", label=f"{Path(session).name}/{bottle}/{finger}/{stage}", markersize=3)
        ax.set(title=key, xlabel="elapsed in each session (s)")
        ax.grid(alpha=.2)
    if drift:
        axes.ravel()[0].legend(fontsize=5)
    save(fig, "zero_offset_drift")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, unit in zip(axes, ("translation_mm", "rotation_deg"), ("mm", "deg")):
        positions = np.arange(len(pose_statistics))
        ax.bar(positions, [row[f"{key}_mean"] for row in pose_statistics], label="mean")
        ax.scatter(positions, [row[f"{key}_p95"] for row in pose_statistics], marker="_", color="red", label="p95")
        ax.set(xticks=positions, xticklabels=[row["stage"] for row in pose_statistics], ylabel=f"pose error ({unit})")
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        ax.legend()
    save(fig, "pose_error_by_stage")
    return fits
