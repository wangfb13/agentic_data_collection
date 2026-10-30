"""Deterministic experiment plans; pure calculation, never hardware access."""
from __future__ import annotations

import copy
import random
import numpy as np

from .geometry import (check_pose, check_segment, lift_base, nominal_contact_region,
    pose_offset, rotation, translate_local, axis_basis, grasp_transform,
    grasp_pose, grasp_pose_offset, grasp_center_metadata)


def _grid(bounds, step):
    lo, hi = map(float, bounds)
    step = float(step)
    if not all(np.isfinite([lo, hi, step])) or hi < lo or step <= 0:
        raise ValueError("invalid grid range/step")
    count = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return [lo + step * i for i in range(count)]


def scan_axes(cfg):
    bottle = "l" if cfg["grasp_mode"] == "side" else "a" if cfg["grasp_mode"] == "top" else None
    if bottle is None or bottle != cfg["geometry"]["bottle"]["axis"]:
        raise ValueError("grasp_mode side/top must agree with bottle axis l/a")
    return bottle, "a" if bottle == "l" else "l"


def _trial(cfg, T0, bottle, kind, delta=(0, 0, 0), axis=None, angle=0, repeat=0):
    target = grasp_pose_offset(T0, cfg, delta, axis, angle)
    pre = translate_local(target, -cfg["motion"]["pre_distance_mm"], "a", cfg["axes"])
    check = translate_local(T0, -cfg["check"]["retreat_mm"], "a", cfg["axes"])
    lift = lift_base(target, cfg["motion"]["lift_height_mm"])
    pose_checks = {"MOVE_PRE": check_pose(pre, cfg, T0), "APPROACH": check_pose(target, cfg, T0),
        "CHECK": check_pose(check, cfg, T0)}
    # Open width bounds the gripper workspace/table extent for every narrower
    # width. It also checks the palm and carried bottle, while actual closed
    # finger/bottle clearance remains a mandatory runtime check.
    attached_bottle_at_lift = lift @ np.linalg.inv(target) @ np.asarray(T0)
    pose_checks["LIFT"] = check_pose(lift, cfg, attached_bottle_at_lift)
    pose_checks["LOWER"] = list(pose_checks["APPROACH"])
    pose_checks["RETREAT"] = list(pose_checks["MOVE_PRE"])
    leg_to_pre = check_segment(check, pre, cfg, T0)
    approach = check_segment(pre, target, cfg, T0)
    lifted = check_segment(target, lift, cfg, T0, width_mm=cfg["gripper"]["open_mm"],
        allow_contact=True, attached_grasp_pose=target)
    # The same sampled path and swept bound certify reverse traversal exactly.
    path_checks = {"MOVE_PRE": leg_to_pre, "APPROACH": approach, "LIFT": lifted,
        "LOWER": list(lifted), "RETREAT": list(approach), "CHECK": list(leg_to_pre)}
    reasons = [phase + ":pose:" + reason for phase, found in pose_checks.items() for reason in found]
    reasons += [phase + ":path:" + reason for phase, found in path_checks.items() for reason in found]
    return {**grasp_center_metadata(target, T0, cfg),
        "bottle": bottle, "kind": kind, "delta_acl_mm": list(map(float, delta)),
        "rotation_axis": axis, "angle_deg": float(angle), "repeat": repeat,
        "T_target": target.tolist(), "T_pre": pre.tolist(), "T_lift": lift.tolist(),
        "T_check": check.tolist(), "nominal_contact_region": nominal_contact_region(target, T0, cfg),
        "safety_reasons": sorted(set(reasons)), "valid": not reasons,
        "pose_checks": pose_checks, "path_checks": path_checks,
        "requires_runtime_width_check": True,
        "geometry_coverage": "All planned target/leg poses checked; lifted bottle follows the grasp. Open-width envelope certifies workspace/table and palm. Closed finger/bottle clearance, actual grasp pose, and actual current-to-target path must be checked at runtime.",
        "is_zero": bool(np.allclose(delta, 0) and float(angle) == 0)}


def commissioning_trials(cfg, T0):
    """First physical run: exactly zero, deepest +a, and beyond bottle cap."""
    bottle_axis, other = scan_axes(cfg)
    d = cfg["design"]
    deep_a = max(d["other_axis_range_mm"] if other == "a" else d["bottle_axis_range_mm"])
    finger = cfg["geometry"]["gripper"]
    axis_index = ("a", "c", "l").index(bottle_axis)
    lower_tip_offset = float(finger["finger_center_a_mm"] if bottle_axis == "a" else finger["finger_center_l_mm"]) - float(finger["finger_size_acl_mm"][axis_index])/2
    beyond_cap = max(float(d["bottle_axis_range_mm"][1]),
        max(float(s["axial_max_mm"]) for s in cfg["geometry"]["bottle"]["segments"])
        - lower_tip_offset + float(cfg["safety"]["margin_mm"]) + float(d["bottle_axis_step_mm"]))
    delta_cap = [0., 0., 0.]
    delta_cap[("a", "c", "l").index(bottle_axis)] = beyond_cap
    specs = [("commission_zero", [0, 0, 0]), ("commission_deep_a", [deep_a, 0, 0]), ("commission_above_cap", delta_cap)]
    trials = [_trial(cfg, T0, d["bottles"][0], kind, delta) for kind, delta in specs]
    for i, trial in enumerate(trials):
        trial.update(trial_id=f"trial_{i + 1:04d}", sequence=i + 1, commissioning=True)
    return trials


def contact_trials(cfg, T0):
    bottle_axis, other_axis = scan_axes(cfg)
    bi, oi = ("a", "c", "l").index(bottle_axis), ("a", "c", "l").index(other_axis)
    d = cfg["design"]
    repeats, zero_every = int(d["repeats"]), int(d["zero_every"])
    if repeats <= 0 or zero_every <= 0:
        raise ValueError("repeats and zero_every must be positive")
    prototypes = []
    for b in _grid(d["bottle_axis_range_mm"], d["bottle_axis_step_mm"]):
        for other in _grid(d["other_axis_range_mm"], d["other_axis_step_mm"]):
            delta = [0., 0., 0.]
            delta[bi], delta[oi] = b, other
            prototypes.append(("grid", delta, None, 0.))
    prototypes.extend(("closure_scan", [0., float(c), 0.], None, 0.) for c in d["closure_offsets_mm"])
    for axis in ("c", other_axis):
        prototypes.extend(("tilt_" + axis, [0., 0., 0.], axis, float(angle)) for angle in d["tilt_degrees"])
    # Geometry is bottle independent for this pair, so certify each offset once.
    cached = [_trial(cfg, T0, d["bottles"][0], kind, delta, axis, angle) for kind, delta, axis, angle in prototypes]
    zero = _trial(cfg, T0, d["bottles"][0], "drift_zero")
    result = []
    for block, bottle in enumerate(d["bottles"]):
        block_trials = [dict(copy.deepcopy(trial), bottle=bottle, repeat=rep) for rep in range(1, repeats + 1) for trial in cached]
        random.Random(int(d["seed"]) + block).shuffle(block_trials)
        for count, trial in enumerate(block_trials, 1):
            trial["bottle_block"] = block
            result.append(trial)
            if count % zero_every == 0:
                result.append(dict(copy.deepcopy(zero), bottle=bottle, bottle_block=block, repeat=0))
    for i, trial in enumerate(result):
        trial.update(trial_id=f"trial_{i + 1:04d}", sequence=i + 1, commissioning=False)
    return result


def visual_points(cfg, T0):
    """Exactly N candidates; safe ones nearest-neighbour ordered, rejected kept.

    Each leg is certified from the preceding accepted point. Rejected points do
    not move the current pose. Labels stored here are nominal; acquisition must
    recompute inv(T_measured @ G) @ B using actual FK after settling. The old
    inv(T_measured) @ T0 label is retained separately for traceability.
    """
    bottle_axis, other = scan_axes(cfg)
    v = cfg["visual"]
    n = int(v["count"])
    if n <= 0:
        raise ValueError("visual.count must be positive")
    rng = random.Random(int(v.get("seed", cfg["design"]["seed"])))
    basis = axis_basis(cfg["axes"])
    G = grasp_transform(cfg)
    B = grasp_pose(T0, cfg)
    generated = []
    for i in range(n):
        theta = rng.uniform(*v["orbit_deg"])
        delta = [rng.uniform(*v[key + "_range_mm"]) for key in ("a", "c", "l")]
        tilts = [rng.uniform(*v["tilt_deg"]), rng.uniform(*v["tilt_deg"])]
        orbit = np.eye(4)
        orbit[:3, :3] = rotation(basis[:, ("a", "c", "l").index(bottle_axis)], theta)
        center_target = pose_offset(B @ orbit, delta, "c", tilts[0], cfg["axes"])
        final_tilt = np.eye(4)
        final_tilt[:3, :3] = rotation(basis[:, ("a", "c", "l").index(other)], tilts[1])
        target = center_target @ final_tilt @ np.linalg.inv(G)
        reasons = check_pose(target, cfg, T0)
        generated.append({**grasp_center_metadata(target, T0, cfg),
            "point_id": f"point_{i + 1:04d}", "candidate_index": i + 1,
            "kind": "visual", "bottle": cfg["design"]["bottles"][0], "T_target": target.tolist(),
            "delta_acl_mm": delta, "orbit_deg": theta, "tilt_deg": tilts,
            "label_B_in_tcp": (np.linalg.inv(target) @ np.asarray(T0)).tolist(),
            "label_B_in_grasp": (np.linalg.inv(grasp_pose(target, cfg)) @ B).tolist(),
            "B": B.tolist(),
            "label_is_nominal": True, "safety_reasons": reasons, "valid": not reasons})
    remaining = [p for p in generated if p["valid"]]
    rejected = [p for p in generated if not p["valid"]]
    ordered = []
    current = translate_local(T0, -cfg["check"]["retreat_mm"], "a", cfg["axes"])
    while remaining:
        nearest = min(range(len(remaining)), key=lambda i: np.linalg.norm(np.asarray(remaining[i]["T_target"])[:3, 3] - current[:3, 3]))
        point = remaining.pop(nearest)
        reasons = check_segment(current, point["T_target"], cfg, T0)
        if reasons:
            point.update(valid=False, safety_reasons=reasons)
            rejected.append(point)
        else:
            ordered.append(point)
            current = np.asarray(point["T_target"])
    for i, point in enumerate(ordered + rejected):
        point["sequence"] = i + 1
    return ordered + rejected
