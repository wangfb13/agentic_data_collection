"""Calibrated bottle localization dispatch, in millimetres.

The markerless schema uses the known upright bottle's appearance and measured
shape. Legacy ArUco configurations retain their explicit marker localizer.
Calibration evidence is recorded, never treated as a deployment certificate.
Imports do not open devices, connect ROS, or publish robot commands.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import cv2
import numpy as np

from .geometry import validate_transform

CALIBRATION_SCHEMA = "vision_calibration_v1"
OBSERVATION_SCHEMA = "vision_observation_v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _positive(value, name, *, allow_zero=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(name + " must be positive")
    return float(value)


def _matrix(value, shape, name):
    out = np.asarray(value, dtype=float)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(name + " has invalid shape or nonfinite entries")
    return out


def _pose(value, name):
    out = validate_transform(value)
    # Existing geometry validator uses allclose's default rtol on its bottom
    # row. Tighten it here for interchange records before composing frames.
    if not np.allclose(out[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError(name + " has invalid homogeneous bottom row")
    return out


def _dictionary(name):
    if not isinstance(name, str) or not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
        raise ValueError("unknown ArUco marker_dictionary")
    identifier = getattr(cv2.aruco, name)
    if type(identifier) is not int:
        raise ValueError("marker_dictionary must identify a predefined dictionary")
    try:
        return cv2.aruco.getPredefinedDictionary(identifier)
    except cv2.error as exc:
        raise ValueError("invalid predefined marker dictionary") from exc


def validate_calibration(calibration):
    """Validate and copy explicit camera/marker geometry; never approve hardware.

    T_parent_child maps child-coordinate points into parent coordinates. For
    eye-in-hand, base->TCP is supplied at image time. Fixed cameras instead use
    a measured T_base_camera_mm. Lengths including marker size are always mm.
    A provided calibration_sha256 must match the normalized configuration.
    """
    if isinstance(calibration, dict) and calibration.get("schema") == "markerless_bottle_calibration_v1":
        from .bottle_localization import validate_bottle_calibration
        return validate_bottle_calibration(calibration)
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be an object")
    c = deepcopy(calibration)
    claimed_hash = c.pop("calibration_sha256", None)
    required = {
        "schema", "camera_name", "image_size_wh", "camera_matrix",
        "distortion_coefficients", "camera_mount", "T_marker_bottle_mm",
        "marker_dictionary", "marker_id", "marker_length_mm",
        "max_reprojection_error_px", "max_observation_age_s", "max_pose_skew_s",
        "min_marker_area_px2", "simulated", "evidence",
    }
    optional = {"T_tcp_camera_mm", "T_base_camera_mm", "unit",
                "pose_ambiguity_ratio", "pose_ambiguity_gap_px",
                "min_depth_mm", "max_depth_mm", "mount_tcp_side"}
    if required - c.keys() or c.keys() - required - optional:
        raise ValueError("invalid calibration fields: missing=" + str(sorted(required - c.keys())) +
                         "; unknown=" + str(sorted(c.keys() - required - optional)))
    if c["schema"] != CALIBRATION_SCHEMA or c.get("unit", "mm") != "mm":
        raise ValueError("expected vision_calibration_v1 in mm")
    c["unit"] = "mm"
    if not isinstance(c["camera_name"], str) or not c["camera_name"].strip():
        raise ValueError("camera_name must be explicit")
    size = c["image_size_wh"]
    if not isinstance(size, (tuple, list)) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("image_size_wh must be two positive integer pixels")
    c["image_size_wh"] = list(size)
    K = _matrix(c["camera_matrix"], (3, 3), "camera_matrix")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError("invalid pinhole camera_matrix")
    if abs(K[0, 1]) > 1e-12 or abs(K[1, 0]) > 1e-12:
        raise ValueError("only zero-skew pinhole camera matrices are supported")
    if not 0 <= K[0, 2] < size[0] or not 0 <= K[1, 2] < size[1]:
        raise ValueError("principal point is outside calibrated image")
    c["camera_matrix"] = K.tolist()
    D = np.asarray(c["distortion_coefficients"], dtype=float)
    if D.ndim != 1 or len(D) not in (4, 5, 8, 12, 14) or not np.isfinite(D).all():
        raise ValueError("invalid pinhole distortion coefficients")
    c["distortion_coefficients"] = D.tolist()
    mount = c["camera_mount"]
    if mount not in ("eye_in_hand", "fixed"):
        raise ValueError("camera_mount must be eye_in_hand or fixed")
    if mount == "eye_in_hand" and c.get("mount_tcp_side") not in ("left", "right"):
        raise ValueError("eye_in_hand requires mount_tcp_side left/right")
    if mount == "fixed" and "mount_tcp_side" in c:
        raise ValueError("fixed camera must not declare mount_tcp_side")
    needed, forbidden = (("T_tcp_camera_mm", "T_base_camera_mm") if mount == "eye_in_hand"
                         else ("T_base_camera_mm", "T_tcp_camera_mm"))
    if needed not in c or forbidden in c:
        raise ValueError("camera mounting requires exactly " + needed)
    c[needed] = _pose(c[needed], needed).tolist()
    c["T_marker_bottle_mm"] = _pose(c["T_marker_bottle_mm"], "T_marker_bottle_mm").tolist()
    dictionary = _dictionary(c["marker_dictionary"])
    if type(c["marker_id"]) is not int or not 0 <= c["marker_id"] < len(dictionary.bytesList):
        raise ValueError("marker_id is outside dictionary")
    for key in ("marker_length_mm", "max_reprojection_error_px", "max_observation_age_s",
                "max_pose_skew_s", "min_marker_area_px2"):
        c[key] = _positive(c[key], key)
    c["pose_ambiguity_ratio"] = _positive(c.get("pose_ambiguity_ratio", 1.5), "pose_ambiguity_ratio")
    if c["pose_ambiguity_ratio"] <= 1:
        raise ValueError("pose_ambiguity_ratio must exceed one")
    c["pose_ambiguity_gap_px"] = _positive(c.get("pose_ambiguity_gap_px", .1), "pose_ambiguity_gap_px")
    c["min_depth_mm"] = _positive(c.get("min_depth_mm", 10.), "min_depth_mm")
    c["max_depth_mm"] = _positive(c.get("max_depth_mm", 2000.), "max_depth_mm")
    if c["max_depth_mm"] <= c["min_depth_mm"]:
        raise ValueError("invalid marker depth interval")
    if type(c["simulated"]) is not bool:
        raise ValueError("simulated must be explicit boolean")
    if not isinstance(c["evidence"], dict) or not c["evidence"]:
        raise ValueError("calibration evidence must be a nonempty object")
    # Evidence may contain filenames/hashes/measurement notes, but cannot
    # silently contain nonfinite numbers or non-JSON objects.
    sha = _digest(c)
    if claimed_hash is not None and claimed_hash != sha:
        raise ValueError("calibration_sha256 mismatch")
    c["calibration_sha256"] = sha
    return c


def calibration_digest(calibration):
    """Content identity used by observations and motion-plan bindings."""
    return validate_calibration(calibration)["calibration_sha256"]


def localize_image(image, calibration, **context):
    """Explicit schema dispatch; never fall back from failed markerless detection."""
    if isinstance(calibration, dict) and calibration.get("schema") == "markerless_bottle_calibration_v1":
        from .bottle_localization import localize_bottle
        return localize_bottle(image, calibration, **context)
    return localize_marker(image, calibration, **context)


def marker_object_points(marker_length_mm):
    """Canonical ArUco corners: top-left, top-right, bottom-right, bottom-left."""
    half = _positive(marker_length_mm, "marker_length_mm") / 2
    return np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], dtype=np.float64)


def _result(c, image_timestamp_ns, pose_timestamp_ns, source):
    return {"schema": OBSERVATION_SCHEMA, "valid": False, "reasons": [],
            "source": source, "camera_name": c["camera_name"], "marker_id": c["marker_id"],
            "calibration_sha256": c["calibration_sha256"], "simulated": c["simulated"],
            "timestamp_ns": image_timestamp_ns, "pose_timestamp_ns": pose_timestamp_ns,
            "T_base_bottle_mm": None, "T_camera_marker_mm": None,
            "T_base_camera_mm": None, "T_base_mount_tcp_mm": None,
            "marker_corners_px": None, "reprojection_error_px": None,
            "candidate_reprojection_errors_px": [], "unit": "mm",
            "deployment_verified": False, "mount_tcp_side": c.get("mount_tcp_side")}


def _context(c, result, T_base_tcp_mm, image_timestamp_ns, pose_timestamp_ns, now_ns, pose_tcp_side):
    if c["camera_mount"] == "eye_in_hand" and pose_tcp_side != c["mount_tcp_side"]:
        result["reasons"].append("camera_mount_tcp_side_mismatch")
        return None
    if any(type(t) is not int or t < 0 for t in (image_timestamp_ns, pose_timestamp_ns, now_ns)):
        result["reasons"].append("invalid_monotonic_timestamps")
        return None
    if not 0 <= now_ns - image_timestamp_ns <= c["max_observation_age_s"] * 1e9:
        result["reasons"].append("stale_or_future_image")
    if not 0 <= now_ns - pose_timestamp_ns <= c["max_observation_age_s"] * 1e9:
        result["reasons"].append("stale_or_future_pose")
    if abs(image_timestamp_ns - pose_timestamp_ns) > c["max_pose_skew_s"] * 1e9:
        result["reasons"].append("image_pose_timestamp_skew")
    try:
        tcp = _pose(T_base_tcp_mm, "T_base_tcp_mm")
    except (TypeError, ValueError, KeyError):
        result["reasons"].append("invalid_tcp_pose")
        return None
    if result["reasons"]:
        return None
    T_base_camera = (tcp @ np.asarray(c["T_tcp_camera_mm"]) if c["camera_mount"] == "eye_in_hand"
                     else np.asarray(c["T_base_camera_mm"]))
    result["T_base_camera_mm"] = T_base_camera.tolist()
    result["T_base_mount_tcp_mm"] = tcp.tolist() if c["camera_mount"] == "eye_in_hand" else None
    return T_base_camera


def _solve(corners, c, result, T_base_camera):
    try:
        points = _matrix(corners, (4, 2), "marker_corners_px")
    except (TypeError, ValueError):
        result["reasons"].append("invalid_marker_corners")
        return result
    result["marker_corners_px"] = points.tolist()
    width, height = c["image_size_wh"]
    if np.any(points[:, 0] < 1) or np.any(points[:, 0] >= width - 1) or np.any(points[:, 1] < 1) or np.any(points[:, 1] >= height - 1):
        result["reasons"].append("marker_outside_or_touching_image_border")
        return result
    contour = points.astype(np.float32)
    if (not cv2.isContourConvex(contour) or
            abs(cv2.contourArea(contour)) < c["min_marker_area_px2"] or
            min(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)) < 2):
        result["reasons"].append("degenerate_or_too_small_marker")
        return result
    object_points = marker_object_points(c["marker_length_mm"])
    K, D = np.asarray(c["camera_matrix"]), np.asarray(c["distortion_coefficients"])
    try:
        solved = cv2.solvePnPGeneric(object_points, points, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        result["reasons"].append("pnp_failed")
        return result
    candidates = []
    if solved[0]:
        for rvec, tvec in zip(solved[1], solved[2]):
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            R, _ = cv2.Rodrigues(rvec)
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = R, np.asarray(tvec).reshape(3)
            camera_points = object_points @ R.T + pose[:3, 3]
            if np.any(camera_points[:, 2] < c["min_depth_mm"]) or np.any(camera_points[:, 2] > c["max_depth_mm"]):
                continue
            # A printed marker is observed on its front (+z) side.
            if float(R[:, 2] @ pose[:3, 3]) >= 0:
                continue
            reprojected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
            error = float(np.sqrt(np.mean(np.sum((reprojected.reshape(4, 2) - points) ** 2, axis=1))))
            if math.isfinite(error):
                candidates.append((error, pose))
    if not candidates:
        result["reasons"].append("no_valid_front_facing_pose_in_depth_range")
        return result
    candidates.sort(key=lambda x: x[0])
    errors = [x[0] for x in candidates]
    result["candidate_reprojection_errors_px"] = errors
    result["reprojection_error_px"] = errors[0]
    if errors[0] > c["max_reprojection_error_px"]:
        result["reasons"].append("reprojection_error_exceeded")
        return result
    if len(candidates) > 1:
        if errors[1] - errors[0] < c["pose_ambiguity_gap_px"] or errors[1] / max(errors[0], 1e-9) < c["pose_ambiguity_ratio"]:
            result["reasons"].append("ambiguous_planar_pose")
            return result
    camera_marker = candidates[0][1]
    result["T_camera_marker_mm"] = camera_marker.tolist()
    result["T_base_bottle_mm"] = (T_base_camera @ camera_marker @ np.asarray(c["T_marker_bottle_mm"])).tolist()
    result["valid"] = True
    return result


def localize_marker_corners(corners, calibration, *, T_base_tcp_mm,
                           image_timestamp_ns, pose_timestamp_ns, now_ns, pose_tcp_side=None):
    """Geometry test/external correspondence path, explicitly not image detection.

    Corners must be supplied in canonical marker order. No marker identity claim
    can be verified by this function. Motion integrations should require
    source='aruco_detection' for actual camera observations.
    """
    c = validate_calibration(calibration)
    result = _result(c, image_timestamp_ns, pose_timestamp_ns, "provided_correspondences")
    T_base_camera = _context(c, result, T_base_tcp_mm, image_timestamp_ns, pose_timestamp_ns, now_ns, pose_tcp_side)
    return result if T_base_camera is None else _solve(corners, c, result, T_base_camera)


def localize_marker(image, calibration, *, T_base_tcp_mm,
                    image_timestamp_ns, pose_timestamp_ns, now_ns, pose_tcp_side=None):
    """Detect the configured ArUco ID in a uint8 grayscale/RGB image and localize.

    The caller supplies image-time TCP feedback; receipt time is not a substitute
    for image/pose association. The result is an observation, never an actuator
    command. Calibration validation errors raise; observation failures return an
    invalid record, preserving their explicit reason.
    """
    c = validate_calibration(calibration)
    result = _result(c, image_timestamp_ns, pose_timestamp_ns, "aruco_detection")
    T_base_camera = _context(c, result, T_base_tcp_mm, image_timestamp_ns, pose_timestamp_ns, now_ns, pose_tcp_side)
    if T_base_camera is None:
        return result
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3) or
            list(image.shape[:2][::-1]) != c["image_size_wh"]):
        result["reasons"].append("invalid_image_or_calibration_size_mismatch")
        return result
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    dictionary = _dictionary(c["marker_dictionary"])
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    indices = [] if ids is None else np.flatnonzero(ids.reshape(-1) == c["marker_id"]).tolist()
    if len(indices) != 1:
        result["reasons"].append("target_marker_not_found" if not indices else "duplicate_target_marker_id")
        return result
    return _solve(np.asarray(corners[indices[0]]).reshape(4, 2), c, result, T_base_camera)
