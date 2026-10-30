"""Post-recording robust DD windows. No ROS, control, or recording mutation.

Call only after the trial HDF5 writer has closed. Each median is the median of
individually computed descriptors, not a descriptor of a median DD image.
"""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np


def source_frame_gaps(stream, indices=None):
    """Count observable gaps in source IDs; missing source IDs stay unknown."""
    indices = list(range(len(stream["times"]))) if indices is None else list(indices)
    ids = []
    for index in indices:
        metadata = stream["metadata"][index] if index < len(stream["metadata"]) else {}
        value = metadata.get("fid", metadata.get("frame_id", metadata.get("source_sequence")))
        ids.append(int(value) if isinstance(value, (int, np.integer)) and not isinstance(value, bool) else None)
    pairs = [(a, b) for a, b in zip(ids, ids[1:]) if a is not None and b is not None]
    return {"source_fids": ids, "source_id_available_count": sum(value is not None for value in ids),
        "source_id_missing_count": sum(value is None for value in ids),
        "missing_frames": sum(max(0, b-a-1) for a, b in pairs) if pairs else None,
        "nonincreasing_pairs": sum(b <= a for a, b in pairs),
        "observed_adjacent_pairs": len(pairs)}


def _reference(value, timestamp):
    if callable(value):
        value = value(timestamp)
    if isinstance(value, (str, Path)):
        path = Path(value)
        with np.load(path, allow_pickle=False) as archive:
            return {**{key: np.asarray(archive[key]) for key in archive.files}, "path": str(path)}
    if hasattr(value, "files"):
        return {key: np.asarray(value[key]) for key in value.files}
    if not isinstance(value, dict):
        raise ValueError("Unloaded reference must supply measured per-finger DD arrays")
    return value


def _median(items, key):
    values = [float(item[key]) for item in items if key in item and item[key] is not None]
    values = [value for value in values if not math.isnan(value)]
    return float(np.median(values)) if values else None


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def enrich_trial_windows(meta, raw_h5_path, cfg, unloaded_reference):
    """Update/return meta with PROBE/HOLD descriptor medians and frame evidence.

    ``unloaded_reference`` accepts an NPZ path, arrays dict, or callable selecting
    a reference by frame host monotonic ns. ``auto.probe_stable_window`` contains
    start_ns/end_ns. Missing/failed stability is explicitly a fallback, never
    relabeled as settled. A short stable window stays short; no outside frames
    are imported to reach window_frames.
    """
    from .analysis import (_settings, _stage_times, _stream, dd_descriptors,
        _projection, center_of_pressure)
    auto = meta.setdefault("auto", {})
    count = cfg.get("probe", {}).get("window_frames", 5)
    duration = cfg.get("hold", {}).get("window_s", .5)
    if type(count) is not int or count < 1 or not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError("probe.window_frames and hold.window_s must be positive")
    unit = cfg.get("force", {}).get("unit", cfg.get("tactile", {}).get("force_unit", "unknown"))
    stable = auto.get("probe_stable_window")
    windows = {"version": 2, "aggregation": "median_of_per_frame_descriptors",
        "force_unit": unit, "force_target_definition": cfg.get("force", {}).get("target_definition", "unknown"),
        "probe_not_settled": bool(auto.get("probe_not_settled", False)), "PROBE": {}, "HOLD": {}}
    snr = {}
    if not Path(raw_h5_path).exists():
        windows["unavailable_reason"] = "raw_stream_file_missing"
        meta["descriptor_windows"] = windows
        auto["probe_snr"] = {"left": None, "right": None}
        return meta
    with h5py.File(raw_h5_path, "r") as handle:
        for side_index, side in enumerate(("left", "right")):
            dd = _stream(handle, "dd_"+side, "tactile/"+side+"/dd")
            wrench = _stream(handle, "wrench_"+side, "tactile/"+side+"/wrench")
            if dd is None:
                continue
            settings = _settings(cfg, side)
            for stage in ("PROBE", "HOLD"):
                timing = _stage_times(meta, stage)
                if timing is None:
                    continue
                lo, hi = timing
                fallback = None
                if stage == "PROBE":
                    if (not windows["probe_not_settled"] and isinstance(stable, dict)
                            and isinstance(stable.get("start_ns"), int) and isinstance(stable.get("end_ns"), int)):
                        lo, hi = max(lo, stable["start_ns"]), min(hi, stable["end_ns"])
                        if hi < lo:
                            raise ValueError("probe_stable_window does not overlap PROBE")
                    else:
                        fallback = "probe_not_settled" if windows["probe_not_settled"] else "stable_window_unrecorded"
                else:
                    midpoint = (lo+hi)//2
                    half = int(float(duration)*1e9/2)
                    lo, hi = max(lo, midpoint-half), min(hi, midpoint+half)
                eligible = np.flatnonzero((dd["times"] >= lo) & (dd["times"] <= hi)).tolist()
                selected, invalid = [], []
                # Validate before taking the last N so corrupt frames cannot
                # displace an earlier valid frame in the same stable interval.
                for index in eligible:
                    timestamp = int(dd["times"][index])
                    dd_meta = dd["metadata"][index] if index < len(dd["metadata"]) else {}
                    try:
                        if dd_meta.get("valid") is False:
                            raise ValueError("source marked DD invalid")
                        reference = _reference(unloaded_reference, timestamp)
                        desc = dd_descriptors(np.asarray(dd["data"][index]), reference[side+"_dd"],
                            **{key: settings[key] for key in ("depth_channel", "threshold", "pixel_pitch_mm", "edge_width_px")})
                        desc.update(_projection(desc, settings["projection"]))
                        frame = {"index": index, "monotonic_ns": timestamp,
                            "source_fid": dd_meta.get("fid", dd_meta.get("frame_id", dd_meta.get("source_sequence"))),
                            "reference_path": reference.get("path"), "normal_force": None,
                            "force_unit": unit, "force_association": "unavailable", "descriptor": desc}
                        if wrench is not None and len(wrench["times"]):
                            wi = int(np.argmin(np.abs(wrench["times"]-timestamp)))
                            skew = abs(int(wrench["times"][wi])-timestamp)/1e9
                            frame.update(wrench_index=wi, force_skew_s=skew)
                            wm = wrench["metadata"][wi] if wi < len(wrench["metadata"]) else {}
                            declared_unit = wm.get("force_unit", wm.get("unit", unit))
                            if isinstance(declared_unit, (list, tuple)):
                                force_unit_matches = len(declared_unit) == 6 and all(value == unit for value in declared_unit[:3])
                            else:
                                force_unit_matches = declared_unit == unit
                            w = np.asarray(wrench["data"][wi], float)
                            if skew <= cfg.get("analysis", {}).get("max_stream_skew_s", .1) and w.shape == (6,) and np.isfinite(w).all() and force_unit_matches:
                                axis = settings["normal_axis"]
                                signs = cfg.get("tactile", {}).get("normal_signs", [1, 1])
                                if type(axis) is int and 0 <= axis <= 2:
                                    frame.update(normal_force=float(w[axis]*float(signs[side_index])), force_association="nearest_host_timestamp")
                                desc.update(center_of_pressure(w, **{key: settings[key] for key in ("moment_unit", "force_unit", "plane_z_mm", "normal_axis")}))
                            else:
                                frame["force_association"] = "stale_or_invalid_or_unit_mismatch"
                        desc.setdefault("cop_u_mm", math.nan)
                        desc.setdefault("cop_v_mm", math.nan)
                        noise = reference.get(side+"_dd_std")
                        noise_sum = None
                        if noise is not None:
                            noise = np.asarray(noise, float)
                            if noise.shape == np.asarray(reference[side+"_dd"]).shape:
                                if noise.ndim == 3:
                                    noise = noise[..., settings["depth_channel"]]
                                if np.isfinite(noise).all() and np.all(noise >= 0):
                                    noise_sum = float(noise.sum())
                        frame["unloaded_noise_sum"] = noise_sum
                        selected.append(frame)
                    except (ValueError, TypeError, KeyError, OSError, IndexError) as exc:
                        invalid.append({"index": index, "reason": str(exc)})
                if stage == "PROBE":
                    selected = selected[-count:]
                descriptors = [entry["descriptor"] for entry in selected]
                medians = {key: _median(descriptors, key) for key in dict.fromkeys(key for entry in descriptors for key in entry)}
                forces = [entry["normal_force"] for entry in selected if entry["normal_force"] is not None]
                medians["normal_force_"+str(unit)] = float(np.median(forces)) if forces else None
                indices = [entry["index"] for entry in selected]
                record = {"start_ns": lo, "end_ns": hi, "selection": "last_valid_stable_frames" if stage == "PROBE" and fallback is None else "last_valid_stage_frames" if stage == "PROBE" else "central_time_window",
                    "fallback_reason": fallback, "probe_not_settled": windows["probe_not_settled"],
                    "requested_frames": count if stage == "PROBE" else None,
                    "selected_frames": len(selected), "short_window": stage == "PROBE" and len(selected) < count,
                    "eligible_frames": len(eligible), "invalid_frames": invalid,
                    "dd_indices": indices, "dd_frame_gaps": source_frame_gaps(dd, eligible),
                    "selected_source_fids": [entry["source_fid"] for entry in selected],
                    "frames": selected, "median": medians}
                if stage == "PROBE":
                    denominators = [entry["unloaded_noise_sum"] for entry in selected if entry["unloaded_noise_sum"] is not None]
                    denominator = float(np.median(denominators)) if denominators else None
                    signal = medians.get("depth_weight_sum")
                    value = signal/denominator if signal is not None and denominator is not None and denominator > 0 else None
                    record["snr"] = {"value": value, "signal": signal, "noise": denominator,
                        "definition": "median_positive_DD_depth_sum / median_sum_of_unloaded_per_pixel_temporal_std",
                        "unavailable_reason": None if value is not None else "unloaded_temporal_noise_missing_or_zero_or_empty_window"}
                    snr[side] = value
                windows[stage][side] = record
    meta["descriptor_windows"] = _clean(windows)
    auto["probe_snr"] = _clean(snr)
    return meta
