"""Metric localization of a known, upright convex bottle body from its silhouette.

Only the axis centre's X/Y coordinates are estimated. Z comes from a measured
table and the identity rotation is an upright, yaw-unobserved convention, NOT
a measured six-degree-of-freedom pose. Imports have no hardware side effects.

The observed contour is undistorted before fitting continuous geometric support
functions. Final IoU is checked in the original distorted image. The complete
visible green silhouette is required after any explicitly modelled cap
self-occlusion; other missing edges are not silently reconstructed.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import validate_transform


def _finite(value, name, *, positive=False, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(name + " outside range")
    return float(value)


def _profile(value):
    profile = np.asarray(value, dtype=float)
    if profile.ndim != 2 or profile.shape[1] != 2 or not 2 <= len(profile) <= 5 or not np.isfinite(profile).all():
        raise ValueError("body_profile_mm must contain 2..5 [height,radius] pairs")
    if np.any(profile[:, 0] < 0) or np.any(profile[:, 1] <= 0) or np.any(np.diff(profile[:, 0]) <= 0):
        raise ValueError("body heights must be nonnegative/increasing and radii positive")
    slopes = np.diff(profile[:, 1])/np.diff(profile[:, 0])
    if np.any(np.diff(slopes) > 1e-9):
        raise ValueError("body profile must describe a convex solid; concave shoulders are unsupported")
    return profile.tolist()


def validate_geometry_model(model):
    """Validate measured convex body and optional overlying convex cap, in mm."""
    if not isinstance(model, dict):
        raise ValueError("bottle model must be an object")
    if {"body_profile_mm", "table_z_mm"}-model.keys() or model.keys()-{"body_profile_mm", "table_z_mm", "cap_profile_mm"}:
        raise ValueError("bottle model requires body_profile_mm/table_z_mm and optional cap_profile_mm")
    result = {"table_z_mm": _finite(model["table_z_mm"], "table_z_mm"),
              "body_profile_mm": _profile(model["body_profile_mm"])}
    if "cap_profile_mm" in model:
        result["cap_profile_mm"] = _profile(model["cap_profile_mm"])
        if result["cap_profile_mm"][0][0] < result["body_profile_mm"][-1][0]:
            raise ValueError("cap must lie above the modelled green body")
    return result


validate_body_model = validate_geometry_model


def _camera(K, D, T, camera_model):
    K, D = np.asarray(K, float), np.asarray(D, float)
    T = validate_transform(T)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("invalid camera matrix")
    if not np.allclose(K[2], [0, 0, 1], atol=1e-12, rtol=0) or abs(K[0, 1])+abs(K[1, 0]) > 1e-12:
        raise ValueError("camera matrix must use zero-skew calibrated pixels")
    if camera_model not in ("pinhole", "fisheye"):
        raise ValueError("explicit pinhole or fisheye camera_model required")
    lengths = (4,) if camera_model == "fisheye" else (4, 5, 8, 12, 14)
    if D.ndim != 1 or len(D) not in lengths or not np.isfinite(D).all():
        raise ValueError("distortion vector does not match camera_model")
    return K, D, T


def _undistort(points, K, D, camera_model):
    points = np.asarray(points, np.float64).reshape(-1, 1, 2)
    if camera_model == "fisheye":
        out = cv2.fisheye.undistortPoints(points, K, D, P=K)
    else:
        out = cv2.undistortPoints(points, K, D, P=K)
    out = out.reshape(-1, 2)
    if not np.isfinite(out).all() or np.max(np.abs(out)) > 1e7:
        raise ValueError("contour cannot be reliably undistorted")
    return out


def _distort(points, K, D, camera_model):
    normalized = np.column_stack(((np.asarray(points)-K[:2, 2])/[K[0, 0], K[1, 1]], np.ones(len(points))))
    zeros = np.zeros(3)
    if camera_model == "fisheye":
        out, _ = cv2.fisheye.projectPoints(normalized.reshape(-1, 1, 3), zeros, zeros, K, D)
    else:
        out, _ = cv2.projectPoints(normalized, zeros, zeros, K, D)
    return out.reshape(-1, 2)


def _body_points(model, xy, count=128):
    angle = np.arange(count)*2*np.pi/count
    points = []
    for height, radius in model["body_profile_mm"]:
        points.append(np.column_stack((xy[0]+radius*np.cos(angle), xy[1]+radius*np.sin(angle),
                                      np.full(count, model["table_z_mm"]+height))))
    return np.concatenate(points)


def _project_undistorted(model, xy, K, T):
    world = _body_points(model, xy)
    camera = (world-T[:3, 3]) @ T[:3, :3]
    if np.any(camera[:, 2] <= 1.) or not np.isfinite(camera).all():
        raise ValueError("bottle intersects or lies behind camera plane")
    norm = camera[:, :2]/camera[:, 2:]
    return norm*[K[0, 0], K[1, 1]]+K[:2, 2]


def _dense_hull(points):
    hull = _hull(points)
    return _dense_polygon(hull)


def _dense_polygon(hull):
    if len(hull) < 3:
        raise ValueError("degenerate projected silhouette")
    pieces = []
    for first, second in zip(hull, np.roll(hull, -1, axis=0)):
        count = max(1, int(np.ceil(np.linalg.norm(second-first)/2.)))
        if count > 100000:
            raise ValueError("projected silhouette is outside usable range")
        pieces.append(first+(second-first)*np.arange(count)[:, None]/count)
    return np.concatenate(pieces)


def _hull(points):
    points = np.asarray(points, float)
    indices = cv2.convexHull(points.astype(np.float32), returnPoints=False).reshape(-1)
    return points[indices]


def _half_plane(poly, first, second, inside=True):
    if len(poly) < 3:
        return np.empty((0, 2))
    edge = second-first
    signed = (poly[:, 0]-first[0])*edge[1]-(poly[:, 1]-first[1])*edge[0]
    # For a CCW clip polygon the interior is to the left of each edge.
    signed = -signed if inside else signed
    output = []
    for index in range(len(poly)):
        previous = (index-1) % len(poly)
        a, b = poly[previous], poly[index]
        sa, sb = signed[previous], signed[index]
        ina, inb = sa >= -1e-9, sb >= -1e-9
        if ina != inb:
            output.append(a+(b-a)*(sa/(sa-sb)))
        if inb:
            output.append(b)
    return np.asarray(output, float).reshape(-1, 2)


def _visible_polygons(model, xy, K, T):
    body = _hull(_project_undistorted(model, xy, K, T))
    if "cap_profile_mm" not in model:
        return [body]
    if T[2, 3] <= model["table_z_mm"]+model["cap_profile_mm"][-1][0]+1.:
        raise ValueError("cap self-occlusion model requires camera above the complete cap")
    cap_model = {"table_z_mm": model["table_z_mm"], "body_profile_mm": model["cap_profile_mm"]}
    cap = _hull(_project_undistorted(cap_model, xy, K, T))
    signed_area = np.sum(cap[:, 0]*np.roll(cap[:, 1], -1)-cap[:, 1]*np.roll(cap[:, 0], -1))
    if signed_area < 0:
        cap = cap[::-1]
    remaining, pieces = body, []
    for first, second in zip(cap, np.roll(cap, -1, axis=0)):
        exterior = _half_plane(remaining, first, second, inside=False)
        if len(exterior) >= 3 and abs(cv2.contourArea(exterior.astype(np.float32))) > 1e-6:
            pieces.append(exterior)
        remaining = _half_plane(remaining, first, second, inside=True)
        if len(remaining) < 3:
            break
    if not pieces:
        raise ValueError("green body is hidden by the modelled cap")
    return pieces


def render_bottle_mask(image_size_wh, camera_matrix, distortion_coefficients,
                       T_base_camera_mm, model, xy_mm, *, camera_model="pinhole"):
    """Pure synthetic projection; the output is not physical image evidence."""
    if len(image_size_wh) != 2 or any(type(v) is not int or v <= 0 for v in image_size_wh):
        raise ValueError("image_size_wh must be positive integers")
    K, D, T = _camera(camera_matrix, distortion_coefficients, T_base_camera_mm, camera_model)
    model = validate_body_model(model)
    xy = np.asarray(xy_mm, float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("xy_mm must be two finite numbers")
    mask = np.zeros((image_size_wh[1], image_size_wh[0]), np.uint8)
    for polygon in _visible_polygons(model, xy, K, T):
        outline = _distort(_dense_polygon(polygon), K, D, camera_model)
        if not np.isfinite(outline).all() or np.max(np.abs(outline)) > 1e7:
            raise ValueError("invalid distorted projection")
        cv2.fillPoly(mask, [np.rint(outline).astype(np.int32)], 255)
    return mask


def _limits(limits):
    if not isinstance(limits, dict):
        raise ValueError("fit limits must be an object")
    required = {"xy_bounds_mm", "max_contour_error_px", "min_iou", "max_position_uncertainty_mm", "min_mask_area_px"}
    optional = {"min_border_px", "max_condition_number", "pixel_uncertainty_px", "min_ray_plane_cos"}
    if required-limits.keys() or limits.keys()-required-optional:
        raise ValueError("invalid bottle fitting limit fields")
    result = dict(limits)
    bounds = np.asarray(result["xy_bounds_mm"], float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("xy_bounds_mm must be [[xmin,xmax],[ymin,ymax]]")
    result["xy_bounds_mm"] = bounds
    result.setdefault("min_border_px", 2.)
    result.setdefault("max_condition_number", 1000.)
    result.setdefault("pixel_uncertainty_px", 1.)
    result.setdefault("min_ray_plane_cos", .05)
    for key in (required | optional)-{"xy_bounds_mm"}:
        result[key] = _finite(result[key], key, positive=True)
    if result["min_iou"] > 1 or result["min_ray_plane_cos"] >= 1 or result["max_condition_number"] < 1:
        raise ValueError("invalid ratio/condition limit")
    return result


def validate_fit_limits(limits):
    """Return a JSON-compatible normalized copy before opening any hardware."""
    result = _limits(limits)
    result["xy_bounds_mm"] = result["xy_bounds_mm"].tolist()
    return result


def fit_bottle_mask(mask, camera_matrix, distortion_coefficients, T_base_camera_mm,
                    model, limits, *, camera_model="pinhole"):
    """Fit an upright known silhouette; reject insufficient or ambiguous evidence.

    ``contour_error_px`` is RMS support error on the undistorted pixel plane.
    ``position_uncertainty_mm`` is a conservative local image-error sensitivity
    estimate, NOT a calibrated confidence interval. It excludes unknown errors
    in hand-eye, table height, bottle dimensions and segmentation bias.
    Invalid configuration raises ValueError; an unusable mask returns valid=False.
    """
    K, D, T = _camera(camera_matrix, distortion_coefficients, T_base_camera_mm, camera_model)
    model, limits = validate_body_model(model), _limits(limits)
    result = {"valid": False, "reasons": [], "T_base_bottle_mm": None,
              "fit_iou": None, "contour_error_px": None, "position_uncertainty_mm": None,
              "orientation_observed": False, "pose_convention": "base_aligned_upright_axis_origin_on_table",
              "estimated_dofs": ["base_x_mm", "base_y_mm"], "assumed_dofs": ["table_z_mm", "upright_axis"],
              "yaw": "unobserved_axial_symmetry", "camera_model": camera_model,
              "uncertainty_scope": "local image-error sensitivity only; no calibration/model error certificate"}
    def reject(reason):
        result["reasons"].append(reason)
        return result
    if "cap_profile_mm" in model and T[2, 3] <= model["table_z_mm"]+model["cap_profile_mm"][-1][0]+1.:
        return reject("cap_visibility_requires_camera_above_cap")
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0 or mask.dtype not in (np.uint8, np.bool_) or not np.isin(mask, [0, 1, 255]).all():
        return reject("mask_must_be_binary_uint8_or_bool")
    mask = np.where(mask, 255, 0).astype(np.uint8)
    height, width = mask.shape
    area = int(np.count_nonzero(mask))
    if area < limits["min_mask_area_px"]:
        return reject("insufficient_body_pixels")
    ys, xs = np.nonzero(mask)
    border = limits["min_border_px"]
    if xs.min() < border or ys.min() < border or width-1-xs.max() < border or height-1-ys.max() < border:
        return reject("body_silhouette_touches_image_border")
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    significant = np.flatnonzero(component_areas >= max(4, area*.005))
    if len(significant) != 1:
        return reject("body_silhouette_is_disconnected_or_multiple")
    # Small isolated speckles cannot become artificial support-function extrema.
    mask = np.where(labels == int(significant[0])+1, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1 or len(contours[0]) < 12:
        return reject("insufficient_body_contour")
    try:
        observed = _undistort(contours[0].reshape(-1, 2), K, D, camera_model)
        center = _undistort([[float(xs.mean()), float(ys.mean())]], K, D, camera_model)[0]
    except (ValueError, cv2.error):
        return reject("invalid_undistorted_contour")
    ray = T[:3, :3] @ np.array([(center[0]-K[0, 2])/K[0, 0], (center[1]-K[1, 2])/K[1, 1], 1.])
    ray /= np.linalg.norm(ray)
    if abs(ray[2]) < limits["min_ray_plane_cos"]:
        return reject("view_too_parallel_to_support_plane")
    mid_z = model["table_z_mm"] + (model["body_profile_mm"][0][0]+model["body_profile_mm"][-1][0])/2
    distance = (mid_z-T[2, 3])/ray[2]
    if distance <= 0:
        return reject("support_plane_is_behind_camera")
    initial = (T[:3, 3]+distance*ray)[:2]
    bounds = limits["xy_bounds_mm"]
    low, high = bounds[:, 0], bounds[:, 1]
    angles = np.arange(96)*2*np.pi/96
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    observed_support = np.max(observed @ directions.T, axis=0)
    def residual(xy):
        try:
            projected = np.concatenate(_visible_polygons(model, xy, K, T))
            return np.max(projected @ directions.T, axis=0)-observed_support
        except ValueError:
            return np.full(len(directions), 1e6)
    initial = np.clip(initial, low+1e-6, high-1e-6)
    seeds = [initial, (low+high)/2]
    seeds.extend(low+(high-low)*np.asarray(pair) for pair in ((.15, .15), (.15, .85), (.85, .15), (.85, .85)))
    solutions = []
    for seed in seeds:
        fitted = least_squares(residual, seed, bounds=(low, high), max_nfev=100,
                               xtol=1e-8, ftol=1e-8, gtol=1e-8, diff_step=1e-5)
        error = float(np.sqrt(np.mean(residual(fitted.x)**2)))
        if np.isfinite(error):
            solutions.append((error, fitted))
    if not solutions:
        return reject("silhouette_optimization_failed")
    solutions.sort(key=lambda pair: pair[0])
    error, fitted = solutions[0]
    result["contour_error_px"] = error
    result["candidate_errors_px"] = [float(x[0]) for x in solutions]
    if not fitted.success:
        return reject("silhouette_optimization_did_not_converge")
    xy = fitted.x
    if np.any(xy-low <= 1e-4) or np.any(high-xy <= 1e-4):
        return reject("position_reached_search_boundary")
    if error > limits["max_contour_error_px"]:
        return reject("body_contour_does_not_match_measured_model")
    for other_error, other in solutions[1:]:
        if other_error <= error+limits["pixel_uncertainty_px"] and np.linalg.norm(other.x-xy) > limits["max_position_uncertainty_mm"]:
            return reject("ambiguous_position_solutions")
    try:
        rendered = render_bottle_mask([width, height], K, D, T, model, xy, camera_model=camera_model)
    except (ValueError, cv2.error):
        return reject("invalid_fitted_projection")
    intersection = np.count_nonzero((rendered != 0) & (mask != 0))
    union = np.count_nonzero((rendered != 0) | (mask != 0))
    result["fit_iou"] = float(intersection/max(union, 1))
    if result["fit_iou"] < limits["min_iou"]:
        return reject("incomplete_occluded_or_wrong_body_silhouette")
    # Normalize by sqrt(N): contour directions are correlated samples, not N
    # independent measurements that would justify artificially tiny uncertainty.
    eps = .05
    jacobian = np.column_stack([(residual(xy+np.eye(2)[axis]*eps)-residual(xy-np.eye(2)[axis]*eps))/(2*eps) for axis in range(2)])
    singular = np.linalg.svd(jacobian/np.sqrt(len(directions)), compute_uv=False)
    if singular[-1] <= 1e-9 or singular[0]/singular[-1] > limits["max_condition_number"]:
        return reject("position_is_poorly_observable")
    uncertainty = 3*max(limits["pixel_uncertainty_px"], error)/float(singular[-1])
    result["position_uncertainty_mm"] = uncertainty
    result["fit_condition_number"] = float(singular[0]/singular[-1])
    if uncertainty > limits["max_position_uncertainty_mm"]:
        return reject("position_uncertainty_exceeds_limit")
    pose = np.eye(4)
    pose[:3, 3] = [float(xy[0]), float(xy[1]), model["table_z_mm"]]
    result.update(valid=True, T_base_bottle_mm=pose.tolist(),
                  measured_mask_area_px=int(np.count_nonzero(mask)))
    return result
