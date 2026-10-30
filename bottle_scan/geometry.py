"""Pure geometry in millimetres. No robot/ROS imports or device access.

Collision tests use an OUTER polygonal cylinder (32 sides by default) and
separating-axis tests against oriented boxes. Thus approximation may reject a
safe point, but cannot accept an intersection. Path certificates apply only to
linear Cartesian translation plus shortest-path rotation interpolation; a real
backend must guarantee that path (an unverified point-command backend cannot).
"""
from __future__ import annotations

import math
from itertools import product
import numpy as np

_CORNER_SIGNS = np.asarray(list(product((-1., 1.), repeat=3)))


def interface_units(value, direction):
    """The sole mm/m boundary: 4x4 matrix or xyz+xyzw vector; rotations unchanged."""
    factors = {"mm_to_m": 0.001, "m_to_mm": 1000.0}
    if direction not in factors:
        raise ValueError("direction must be mm_to_m or m_to_mm")
    out = np.asarray(value, dtype=float).copy()
    if not np.all(np.isfinite(out)):
        raise ValueError("nonfinite pose")
    if out.shape == (4, 4):
        out[:3, 3] *= factors[direction]
    elif out.shape == (7,):
        out[:3] *= factors[direction]
    else:
        raise ValueError("expected 4x4 transform or xyz+xyzw pose")
    return out


def validate_transform(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("pose must be finite 4x4")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("invalid homogeneous bottom row")
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-7):
        raise ValueError("rotation must be proper orthonormal")
    return T


def axis_basis(axes=None):
    axes = axes or {"a": [1, 0, 0], "c": [0, 1, 0], "l": [0, 0, 1]}
    basis = np.column_stack([axes[k] for k in ("a", "c", "l")]).astype(float)
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)) or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-7):
        raise ValueError("a/c/l must be finite orthogonal unit vectors")
    return basis


def rotation(axis, angle_deg):
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-12:
        raise ValueError("invalid rotation axis")
    x, y, z = axis / np.linalg.norm(axis)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = math.radians(float(angle_deg))
    if not np.isfinite(angle):
        raise ValueError("nonfinite angle")
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def pose_offset(T0, delta_acl_mm=(0, 0, 0), rotation_axis=None, angle_deg=0, axes=None):
    """T_target = T0 @ Trans(delta in original T0 axes) @ Rot(axis)."""
    T0 = validate_transform(T0)
    basis = axis_basis(axes)
    delta = np.asarray(delta_acl_mm, dtype=float)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("delta_acl_mm must be three finite numbers in a,c,l order")
    local = np.eye(4)
    local[:3, 3] = basis @ delta
    if rotation_axis is not None:
        axis = basis[:, ("a", "c", "l").index(rotation_axis)] if isinstance(rotation_axis, str) else rotation_axis
        local[:3, :3] = rotation(axis, angle_deg)
    elif float(angle_deg) != 0:
        raise ValueError("nonzero angle needs rotation_axis")
    return T0 @ local


def grasp_transform(cfg):
    """Measured TCP -> grasp-center transform G, with TCP-aligned axes.

    G translates the origin to the midpoint of the sensor faces. Its rotation
    must be identity: the declared a/c/l axes and all physical solid dimensions
    remain expressed in the existing TCP frame. There is deliberately no
    identity fallback for missing or unmeasured real configurations.
    """
    try:
        G = validate_transform(cfg["tool"]["grasp_center_in_tcp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("tool.grasp_center_in_tcp: " + str(exc)) from exc
    if not np.allclose(G[:3, :3], np.eye(3), atol=1e-9, rtol=0):
        raise ValueError("tool.grasp_center_in_tcp rotation must be identity; grasp axes stay TCP-aligned")
    return G.copy()


def grasp_pose(T, cfg):
    """Grasp-center pose in base coordinates; T is the TCP pose in mm."""
    return validate_transform(T) @ grasp_transform(cfg)


def grasp_pose_offset(T0, cfg, delta_acl_mm=(0, 0, 0), rotation_axis=None, angle_deg=0):
    """T0 @ G @ Trans(delta) @ Rot(theta) @ inv(G).

    Pure translations use the algebraically equivalent old expression so their
    numerical result, including all three commissioning TCP targets, is exactly
    unchanged. Tilt compensation keeps the displaced grasp center fixed.
    """
    G = grasp_transform(cfg)
    if float(angle_deg) == 0:
        return pose_offset(T0, delta_acl_mm, rotation_axis, angle_deg, cfg["axes"])
    return pose_offset(validate_transform(T0) @ G, delta_acl_mm,
        rotation_axis, angle_deg, cfg["axes"]) @ np.linalg.inv(G)


def grasp_center_metadata(T, T0, cfg):
    """Explicit nominal/actual center coordinates for plan and report consumers."""
    center, origin = grasp_pose(T, cfg), grasp_pose(T0, cfg)
    offset = center[:3, 3] - origin[:3, 3]
    return {"grasp_center_base_mm": center[:3, 3].tolist(),
        "grasp_center_offset_base_mm": offset.tolist(),
        "grasp_center_offset_t0_acl_mm": (axis_basis(cfg["axes"]).T @
            validate_transform(T0)[:3, :3].T @ offset).tolist()}


def translate_local(T, distance_mm, axis="a", axes=None):
    basis = axis_basis(axes)
    out = validate_transform(T).copy()
    direction = basis[:, ("a", "c", "l").index(axis)] if isinstance(axis, str) else np.asarray(axis, dtype=float)
    out[:3, 3] += out[:3, :3] @ direction * float(distance_mm)
    return out


def lift_base(T, height_mm):
    out = validate_transform(T).copy()
    out[2, 3] += float(height_mm)
    return out


def _rotation_vector(R):
    angle = math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if angle < 1e-10:
        return np.array([1., 0, 0]), 0.
    if math.pi - angle < 1e-6:
        vals, vecs = np.linalg.eig(R)
        axis = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
        axis /= np.linalg.norm(axis)
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
    return axis, angle


def pose_error(command, measured):
    command, measured = validate_transform(command), validate_transform(measured)
    return float(np.linalg.norm(command[:3, 3] - measured[:3, 3])), math.degrees(_rotation_vector(command[:3, :3].T @ measured[:3, :3])[1])


def interpolate(start, end, fraction):
    start, end = validate_transform(start), validate_transform(end)
    axis, angle = _rotation_vector(start[:3, :3].T @ end[:3, :3])
    out = np.eye(4)
    out[:3, 3] = (1 - fraction) * start[:3, 3] + fraction * end[:3, 3]
    out[:3, :3] = start[:3, :3] @ rotation(axis, math.degrees(angle) * fraction)
    return out


def _box(center, sizes, basis, name):
    sizes = np.asarray(sizes, dtype=float)
    if np.any(sizes <= 0) or not np.all(np.isfinite(sizes)):
        raise ValueError("box sizes must be positive")
    return {"center": np.asarray(center, dtype=float), "half": sizes / 2, "basis": basis, "name": name}


def _box_vertices(box, margin=0.):
    half = box["half"] + margin
    return (_CORNER_SIGNS * half) @ box["basis"].T + box["center"]


def gripper_boxes(T, cfg, width_mm, allow_contact=False):
    """Return collision solids. During controlled closure only pad volumes exempt.

    Full fingers remain checked against table/workspace separately. Padding
    around every remaining hard part protects palm, finger cores, and pad rims.
    """
    basis_local = axis_basis(cfg["axes"])
    basis = T[:3, :3] @ basis_local
    g = cfg["geometry"]["gripper"]
    boxes = [_box(T[:3, 3] + basis @ np.asarray(g["palm_center_acl_mm"]), g["palm_size_acl_mm"], basis, "palm")]
    size = np.asarray(g["finger_size_acl_mm"], dtype=float)
    for side, sign in (("left", -1), ("right", 1)):
        center = np.array([g["finger_center_a_mm"], sign * (width_mm / 2 + size[1] / 2), g["finger_center_l_mm"]], dtype=float)
        if not allow_contact:
            boxes.append(_box(T[:3, 3] + basis @ center, size, basis, side))
            continue
        depth = float(g["pad_depth_mm"])
        pad_a, pad_l = map(float, g["pad_size_al_mm"])
        if not (0 < depth < size[1] and 0 < pad_a <= size[0] and 0 < pad_l <= size[2]):
            raise ValueError("pad must be contained in finger and shallower than thickness")
        core_center = center.copy()
        core_center[1] += sign * depth / 2
        core_size = size.copy()
        core_size[1] -= depth
        boxes.append(_box(T[:3, 3] + basis @ core_center, core_size, basis, side + "_core"))
        # Subtract only the declared active soft rectangle. Its center need
        # not equal the whole-finger center (the actual sensor is at the tip).
        alo, ahi = center[0]-size[0]/2, center[0]+size[0]/2
        llo, lhi = center[2]-size[2]/2, center[2]+size[2]/2
        pa = float(g.get("pad_center_a_mm", center[0]))
        pl = float(g.get("pad_center_l_mm", center[2]))
        pal, pah = pa-pad_a/2, pa+pad_a/2
        pll, plh = pl-pad_l/2, pl+pad_l/2
        if pal < alo-1e-8 or pah > ahi+1e-8 or pll < llo-1e-8 or plh > lhi+1e-8:
            raise ValueError("active pad rectangle must lie within full finger envelope")
        pad_c = sign * (width_mm / 2 + depth / 2)
        rims = [(alo, pal, llo, lhi), (pah, ahi, llo, lhi),
                (pal, pah, llo, pll), (pal, pah, plh, lhi)]
        for index,(a0,a1,l0,l1) in enumerate(rims):
            if a1-a0 <= 1e-10 or l1-l0 <= 1e-10:
                continue
            boxes.append(_box(T[:3,3]+basis@np.array([(a0+a1)/2,pad_c,(l0+l1)/2]),
                              [a1-a0,depth,l1-l0],basis,side+"_rim_"+str(index)))
    return boxes


def _cylinder_prism(T0, cfg, segment):
    b = cfg["geometry"]["bottle"]
    basis = axis_basis(cfg["axes"])
    axis_local = basis[:, ("a", "c", "l").index(b["axis"])]
    axis = T0[:3, :3] @ axis_local
    u = T0[:3, :3] @ basis[:, next(i for i, key in enumerate(("a", "c", "l")) if key != b["axis"])]
    v = np.cross(axis, u)
    n = int(cfg["safety"].get("cylinder_sides", 32))
    if n < 8:
        raise ValueError("cylinder_sides must be >= 8")
    uncertainty = float(b.get("axial_uncertainty_mm", 0.))
    if not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError("invalid axial geometry uncertainty")
    nominal_lo, nominal_hi = float(segment["axial_min_mm"]), float(segment["axial_max_mm"])
    lo, hi = nominal_lo-uncertainty, nominal_hi+uncertainty
    r = float(segment["radius_mm"])
    if nominal_hi <= nominal_lo or r <= 0:
        raise ValueError("invalid cylinder dimensions")
    center = T0[:3, 3] + T0[:3, :3] @ np.asarray(b["center_offset_mm"], dtype=float)
    angles = (np.arange(n) + .5) * 2 * np.pi / n
    radial = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    # Circumscribed, never inscribed. Every true cylinder point is inside.
    ring = radial * (r / math.cos(math.pi / n))
    verts = np.concatenate((center + ring + lo * axis, center + ring + hi * axis))
    edge_dirs = np.roll(ring, -1, axis=0) - ring
    normals = np.array([math.cos(2 * np.pi * i / n) * u + math.sin(2 * np.pi * i / n) * v for i in range(n)])
    return verts, np.concatenate(([axis], normals)), np.concatenate(([axis], edge_dirs))


def box_cylinder_collision(box, prism, margin_mm=0):
    """Conservative distance predicate: False certifies distance > margin."""
    vertices, normals, edges = prism
    bverts = _box_vertices(box, margin_mm)
    if np.any(bverts.max(axis=0) < vertices.min(axis=0) - 1e-8) or np.any(vertices.max(axis=0) < bverts.min(axis=0) - 1e-8):
        return False
    crossed = np.cross(box["basis"].T[:, None, :], edges[None, :, :]).reshape(-1, 3)
    axes = np.concatenate((box["basis"].T, normals, crossed))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms >= 1e-10] / norms[norms >= 1e-10, None]
    a, b = bverts @ axes.T, vertices @ axes.T
    # Touching and numerical uncertainty count as collision.
    separated = (a.max(axis=0) < b.min(axis=0) - 1e-8) | (b.max(axis=0) < a.min(axis=0) - 1e-8)
    return not bool(np.any(separated))


def check_pose(T, cfg, T0, width_mm=None, allow_contact=False, extra_margin_mm=0.):
    try:
        T, T0 = validate_transform(T), validate_transform(T0)
        width = float(cfg["gripper"]["open_mm"] if width_mm is None else width_mm)
        if not np.isfinite(width) or not cfg["gripper"]["min_mm"] <= width <= cfg["gripper"]["open_mm"]:
            return ["gripper_width_out_of_bounds"]
        safety = cfg["safety"]
        margin = float(safety["margin_mm"]) + float(extra_margin_mm)
        if margin < 0:
            raise ValueError("negative margin")
        reasons = []
        lower, upper = np.asarray(safety["workspace_min_mm"]), np.asarray(safety["workspace_max_mm"])
        if np.any(lower >= upper):
            raise ValueError("invalid workspace")
        for box in gripper_boxes(T, cfg, width, False):
            vertices = _box_vertices(box, margin)
            if np.any(vertices < lower) or np.any(vertices > upper):
                reasons.append(box["name"] + ":workspace")
            if np.min(vertices[:, 2]) <= float(safety["table_z_mm"]):
                reasons.append(box["name"] + ":table")
        prisms = [(segment, _cylinder_prism(T0, cfg, segment)) for segment in cfg["geometry"]["bottle"]["segments"]]
        for box in gripper_boxes(T, cfg, width, allow_contact):
            for segment, prism in prisms:
                if box_cylinder_collision(box, prism, margin):
                    reasons.append(box["name"] + ":bottle:" + segment["name"])
        return sorted(set(reasons))
    except (TypeError, ValueError, KeyError) as exc:
        return ["invalid_geometry:" + str(exc)]


def check_segment(start, end, cfg, T0, width_mm=None, allow_contact=False, attached_grasp_pose=None):
    """Check endpoints AND conservative swept volume; never clip the target."""
    try:
        start, end = validate_transform(start), validate_transform(end)
        distance, angle = pose_error(start, end)
        safety = cfg["safety"]
        reasons = []
        if attached_grasp_pose is not None:
            attached_grasp_pose = validate_transform(attached_grasp_pose)
            if not np.allclose(start[:3, :3], end[:3, :3], atol=1e-8, rtol=0) or not np.allclose(start[:2, 3], end[:2, 3], atol=1e-7, rtol=0):
                return ["attached_motion_must_be_base_z_only_without_rotation"]
        if distance > float(safety["max_step_mm"]) + 1e-9:
            reasons.append("step_translation_exceeded")
        if angle > float(safety["max_rotation_step_deg"]) + 1e-9:
            reasons.append("step_rotation_exceeded")
        spatial_step, angular_step = float(safety["path_step_mm"]), float(safety["path_step_deg"])
        if spatial_step <= 0 or angular_step <= 0:
            raise ValueError("path sampling steps must be positive")
        n = max(1, math.ceil(distance / spatial_step), math.ceil(angle / angular_step))
        width = cfg["gripper"]["open_mm"] if width_mm is None else width_mm
        boxes = gripper_boxes(np.eye(4), cfg, width, False)
        radius = max(np.linalg.norm(v) for box in boxes for v in _box_vertices(box))
        # Every path time is <= half an interval from a sampled pose.
        swept_bound = distance / (2 * n) + 2 * radius * math.sin(math.radians(angle) / (4 * n))
        for i in range(n + 1):
            current = interpolate(start, end, i / n)
            bottle_pose = T0
            if attached_grasp_pose is not None:
                bottle_pose = current @ np.linalg.inv(attached_grasp_pose) @ np.asarray(T0)
                # The carried cylinder itself must remain inside the workspace
                # and above the table. Bottle/table equality is allowed for the
                # initial/final supported position. Pure-z monotonic paths make
                # endpoint heights sufficient, so no artificial below-table
                # swept inflation is needed for the bottle.
                low, high = np.asarray(safety["workspace_min_mm"]), np.asarray(safety["workspace_max_mm"])
                for segment in cfg["geometry"]["bottle"]["segments"]:
                    vertices = _cylinder_prism(bottle_pose, cfg, segment)[0]
                    if np.min(vertices[:, 2]) < float(safety["table_z_mm"]) - 1e-8:
                        reasons.append("attached_bottle:table")
                    if np.any(vertices < low) or np.any(vertices > high):
                        reasons.append("attached_bottle:workspace")
            found = check_pose(current, cfg, bottle_pose, width, allow_contact, swept_bound)
            reasons.extend("path:" + reason for reason in found)
        return sorted(set(reasons))
    except (ValueError, TypeError, KeyError) as exc:
        return ["invalid_geometry:" + str(exc)]


def nominal_contact_region(T, T0, cfg):
    # Bottle segment offsets remain measured in the original T0 TCP frame;
    # classify the grasp-center position, not the gripper-base origin.
    local = np.linalg.inv(validate_transform(T0)) @ grasp_pose(T, cfg)
    bottle = cfg["geometry"]["bottle"]
    axis = axis_basis(cfg["axes"])[:, ("a", "c", "l").index(bottle["axis"])]
    coord = float(axis @ (local[:3, 3] - np.asarray(bottle["center_offset_mm"])))
    for seg in bottle["segments"]:
        if seg["axial_min_mm"] <= coord <= seg["axial_max_mm"]:
            return seg["name"]
    return "empty"
