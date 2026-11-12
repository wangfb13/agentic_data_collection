"""Taught, immutable workspace limits for the opt-in vision positioning stage.

All transforms use millimetres in base coordinates. No device, ROS, socket or
motion transport is imported here. A profile records operator declarations;
it is not proof of obstacle clearance, arm collision freedom or physical path
tracking. Every proposed Cartesian segment reuses the existing tool-envelope,
table and bottle collision checks. Only open-hand motion is admitted.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from .geometry import (validate_transform, gripper_boxes, _box_vertices,
                       check_pose, check_segment as geometry_check_segment,
                       pose_error)

SCHEMA = "vision_workspace_v1"


def _hash(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("profile must contain finite JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _number(value, name, *, minimum=None, strictly_positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    value = float(value)
    if strictly_positive and value <= 0 or minimum is not None and value < minimum:
        raise ValueError(name + " is outside its allowed range")
    return value


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(name + " must contain three finite numbers")
    return np.asarray([_number(x, name) for x in value], dtype=float)


def _canonical_inputs(geometry_config, teaching, simulated):
    if type(simulated) is not bool:
        raise ValueError("simulated must be an explicit boolean")
    if not isinstance(geometry_config, dict) or not isinstance(teaching, dict):
        raise ValueError("geometry_config and teaching must be mappings")
    cfg, t = deepcopy(geometry_config), deepcopy(teaching)
    _hash(cfg); _hash(t)
    if type(cfg.get("demo", False)) is not bool:
        raise ValueError("geometry demo flag must be boolean")
    if cfg.get("demo", False) and not simulated:
        raise ValueError("synthetic geometry requires simulated=true")
    if t.get("clear_space_confirmed") is not True:
        raise ValueError("clear_space_confirmed must explicitly confirm the entire workspace interior")
    if cfg.get("units", {}).get("distance") != "mm":
        raise ValueError("geometry distance units must be mm")
    low = _vector(t.get("workspace_min_mm"), "workspace_min_mm")
    high = _vector(t.get("workspace_max_mm"), "workspace_max_mm")
    if np.any(low >= high):
        raise ValueError("workspace bounds must be increasing on all axes")
    safety = cfg.get("safety", {})
    outer_low = _vector(safety.get("workspace_min_mm"), "geometry workspace_min_mm")
    outer_high = _vector(safety.get("workspace_max_mm"), "geometry workspace_max_mm")
    if np.any(outer_low >= outer_high):
        raise ValueError("geometry workspace bounds are invalid")
    if np.any(low < outer_low) or np.any(high > outer_high):
        raise ValueError("teaching must not enlarge the existing workspace")
    for key in ("reference_tcp_mm", "reference_bottle_pose_mm", "observation_tcp_mm"):
        if key not in t:
            raise ValueError("missing teaching field: " + key)
        t[key] = validate_transform(t[key]).tolist()
    t["geometry_reference_tcp_mm"] = validate_transform(
        t.get("geometry_reference_tcp_mm", t["reference_tcp_mm"])).tolist()
    original_step = _number(safety.get("max_step_mm"), "geometry max_step_mm", strictly_positive=True)
    original_angle = _number(safety.get("max_rotation_step_deg"), "geometry max_rotation_step_deg", minimum=0)
    for key in ("path_step_mm", "path_step_deg"):
        _number(safety.get(key), "geometry " + key, strictly_positive=True)
    _number(safety.get("margin_mm"), "geometry margin_mm", minimum=0)
    step = _number(t.get("max_translation_step_mm"), "max_translation_step_mm", strictly_positive=True)
    total = _number(t.get("max_total_translation_mm"), "max_total_translation_mm", strictly_positive=True)
    angle = _number(t.get("orientation_tolerance_deg"), "orientation_tolerance_deg", minimum=0)
    if step > original_step:
        raise ValueError("teaching must not enlarge the existing step limit")
    if total < step:
        raise ValueError("max_total_translation_mm must be at least one step")
    if angle > original_angle or angle >= 90:
        raise ValueError("orientation tolerance must stay within existing rotation limit and below 90 degrees")
    table = _number(t.get("table_z_mm"), "table_z_mm")
    original_table = _number(safety.get("table_z_mm"), "geometry table_z_mm")
    if table < original_table:
        raise ValueError("teaching must not lower the conservative table boundary")
    width = _number(t.get("open_width_mm"), "open_width_mm", strictly_positive=True)
    gripper = cfg.get("gripper", {})
    original_width = _number(gripper.get("open_mm"), "geometry open_mm", strictly_positive=True)
    minimum_width = _number(gripper.get("min_mm"), "geometry min_mm", minimum=0)
    _number(gripper.get("open_tolerance_mm"), "geometry open_tolerance_mm", minimum=0)
    if not minimum_width <= width <= original_width:
        raise ValueError("open_width_mm is outside the existing gripper range")
    # This stage uses the existing fully-open setting rather than inventing a
    # partly closed opening from the taught profile.
    if not math.isclose(width, original_width, rel_tol=0, abs_tol=1e-9):
        raise ValueError("open_width_mm must equal the existing fully-open width")
    t.update(workspace_min_mm=low.tolist(), workspace_max_mm=high.tolist(),
             table_z_mm=table, open_width_mm=width,
             max_translation_step_mm=step, max_total_translation_mm=total,
             orientation_tolerance_deg=angle)
    return cfg, t


def _effective_config(cfg, teaching):
    result = deepcopy(cfg)
    result["safety"].update(
        workspace_min_mm=teaching["workspace_min_mm"],
        workspace_max_mm=teaching["workspace_max_mm"],
        table_z_mm=teaching["table_z_mm"],
        max_step_mm=teaching["max_translation_step_mm"])
    return result


def _validate_taught_poses(cfg, t):
    effective = _effective_config(cfg, t)
    anchor = np.asarray(t["geometry_reference_tcp_mm"], dtype=float)
    reference = np.asarray(t["reference_tcp_mm"], dtype=float)
    for key in ("reference_tcp_mm", "observation_tcp_mm"):
        pose = np.asarray(t[key], dtype=float)
        if pose_error(reference, pose)[1] > t["orientation_tolerance_deg"] + 1e-7:
            raise ValueError(key + " exceeds the taught orientation range")
        reasons = check_pose(pose, effective, anchor, t["open_width_mm"], allow_contact=False)
        if reasons:
            raise ValueError(key + ": " + ", ".join(reasons))


def make_workspace_profile(geometry_config, teaching, *, simulated=False):
    """Freeze measured teaching and the original geometry it may only narrow.

    ``reference_tcp_mm`` is a collision-free, open-hand target, not a contact
    pose. ``reference_bottle_pose_mm`` is the visual object frame measured at
    teaching time. ``geometry_reference_tcp_mm`` is the old cylinder-geometry
    anchor for that same bottle pose; it defaults to the reference TCP.
    """
    cfg, t = _canonical_inputs(geometry_config, teaching, simulated)
    _validate_taught_poses(cfg, t)
    result = {"schema": SCHEMA, "simulated": simulated,
              "coordinate_frame": "base", "distance_unit": "mm",
              "geometry_config": cfg, "base_config_sha256": _hash(cfg), "teaching": t,
              "motion_mode": "open_hand_only",
              "limitations": ["operator-declared obstacle-free interior",
                              "tool-envelope checks are not whole-arm collision checks",
                              "Cartesian path checks require backend path-tracking validation",
                              "cumulative translation must be enforced by the session executor"]}
    result["content_sha256"] = _hash(result)
    return result


def validate_workspace_profile(profile):
    """Validate content and return a detached canonical profile; never mutate."""
    if not isinstance(profile, dict):
        raise ValueError("workspace profile must be a mapping")
    value = deepcopy(profile)
    digest = value.pop("content_sha256", None)
    if not isinstance(digest, str) or digest != _hash(value):
        raise ValueError("workspace profile content/hash mismatch")
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported workspace profile schema")
    try:
        expected = make_workspace_profile(value["geometry_config"], value["teaching"],
                                          simulated=value["simulated"])
    except KeyError as exc:
        raise ValueError("incomplete workspace profile") from exc
    if profile != expected:
        raise ValueError("workspace profile differs from canonical validated content")
    return expected


class TaughtWorkspace:
    """Immutable profile with dynamic bottle geometry and fixed base limits."""

    def __init__(self, profile):
        self._profile = validate_workspace_profile(profile)

    @property
    def profile(self):
        return deepcopy(self._profile)

    @property
    def teaching(self):
        return deepcopy(self._profile["teaching"])

    @property
    def config(self):
        return self.effective_geometry_config()

    def effective_geometry_config(self, bottle_pose=None):
        if bottle_pose is not None:
            validate_transform(bottle_pose)
        return _effective_config(self._profile["geometry_config"], self._profile["teaching"])

    def geometry_anchor(self, bottle_pose):
        t = self._profile["teaching"]
        return (validate_transform(bottle_pose) @
                np.linalg.inv(np.asarray(t["reference_bottle_pose_mm"], dtype=float)) @
                np.asarray(t["geometry_reference_tcp_mm"], dtype=float))

    def check_segment(self, start, end, bottle_pose, width_mm=None):
        """Return blocking reasons; never clip a target or contact the bottle."""
        t = self._profile["teaching"]
        try:
            start, end = validate_transform(start), validate_transform(end)
            anchor = self.geometry_anchor(bottle_pose)
            width = t["open_width_mm"] if width_mm is None else _number(width_mm, "width_mm", strictly_positive=True)
            reasons = []
            open_tolerance = self._profile["geometry_config"]["gripper"]["open_tolerance_mm"]
            if abs(width-t["open_width_mm"]) > open_tolerance:
                reasons.append("open_hand_required")
            reference = np.asarray(t["reference_tcp_mm"], dtype=float)
            for name, pose in (("start", start), ("end", end)):
                if pose_error(reference, pose)[1] > t["orientation_tolerance_deg"] + 1e-7:
                    reasons.append(name + ":taught_orientation_exceeded")
            reasons.extend(geometry_check_segment(start, end, self.effective_geometry_config(),
                                                  anchor, width, allow_contact=False))
            return sorted(set(reasons))
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            return ["invalid_workspace_motion:" + str(exc)]


def teaching_from_captures(captures, cfg):
    """Convert passive teleoperation pose captures into a teaching mapping.

    Exactly one record per role is required. Robot records use ``tcp_mm``;
    ``bottle_reference`` uses ``pose_mm`` (or ``tcp_mm``), explicitly supplied by
    the visual/object measurement. The ``reference`` record must contain all
    non-pose teaching values in ``constraints``. Captures are retained as
    evidence. Lower/upper corners refer to the complete open-tool envelope,
    including the existing geometry margin, never just TCP coordinates.
    """
    if not isinstance(captures, list):
        raise ValueError("captures must be a list")
    required = {"lower_corner", "upper_corner", "reference", "observation", "bottle_reference"}
    records = {}
    for row in captures:
        if not isinstance(row, dict) or row.get("role") not in required or row["role"] in records:
            raise ValueError("capture roles must be known and unique")
        records[row["role"]] = deepcopy(row)
    if set(records) != required:
        raise ValueError("captures require lower_corner, upper_corner, reference, observation and bottle_reference")
    t = deepcopy(records["reference"].get("constraints", {}))
    if not isinstance(t, dict):
        raise ValueError("reference constraints must be a mapping")
    width = _number(cfg["gripper"]["open_mm"], "geometry open_mm", strictly_positive=True)
    margin = _number(cfg["safety"]["margin_mm"], "geometry margin_mm", minimum=0)
    for role, field, reduce in (("lower_corner", "workspace_min_mm", np.min),
                                ("upper_corner", "workspace_max_mm", np.max)):
        pose = validate_transform(records[role]["tcp_mm"])
        vertices = np.concatenate([_box_vertices(box, margin)
                                   for box in gripper_boxes(pose, cfg, width, allow_contact=False)])
        t[field] = reduce(vertices, axis=0).tolist()
    t["reference_tcp_mm"] = validate_transform(records["reference"]["tcp_mm"]).tolist()
    t["observation_tcp_mm"] = validate_transform(records["observation"]["tcp_mm"]).tolist()
    bottle = records["bottle_reference"]
    t["reference_bottle_pose_mm"] = validate_transform(bottle.get("pose_mm", bottle.get("tcp_mm"))).tolist()
    t["open_width_mm"] = width
    t["capture_evidence"] = deepcopy(captures)
    _hash(t)
    return t
