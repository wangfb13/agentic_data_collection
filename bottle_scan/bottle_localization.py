"""Known unmarked bottle: appearance mask + calibrated upright shape geometry.

Only tabletop XY is visually estimated. Height and vertical axis are assumptions
bound to a measured model; yaw is unobserved. No device or ROS imports.
"""
from copy import deepcopy
import hashlib

import numpy as np

from .vision_localization import _digest, _matrix, _pose, _positive
from .bottle_appearance import validate_appearance_profile, segment_bottle
from .bottle_geometry import validate_geometry_model, validate_fit_limits, fit_bottle_mask

SCHEMA = "markerless_bottle_calibration_v1"


def validate_bottle_calibration(calibration):
    if not isinstance(calibration, dict):
        raise ValueError("markerless calibration must be a mapping")
    c = deepcopy(calibration)
    claimed_hash = c.pop("calibration_sha256", None)
    required = {"schema", "target_id", "camera_name", "camera_model", "image_size_wh",
                "camera_matrix", "distortion_coefficients", "camera_mount", "appearance_profile",
                "model", "fit_limits", "max_observation_age_s", "max_pose_skew_s", "simulated", "evidence"}
    optional = {"unit", "mount_tcp_side", "T_tcp_camera_mm", "T_base_camera_mm"}
    if required-c.keys() or c.keys()-required-optional:
        raise ValueError("invalid markerless calibration fields: missing="+str(sorted(required-c.keys()))+
                         "; unknown="+str(sorted(c.keys()-required-optional)))
    if c["schema"] != SCHEMA or c.get("unit", "mm") != "mm":
        raise ValueError("expected markerless_bottle_calibration_v1 with mm units")
    c["unit"] = "mm"
    if type(c["simulated"]) is not bool:
        raise ValueError("simulated must be explicit boolean")
    for key in ("target_id", "camera_name"):
        if not isinstance(c[key], str) or not c[key].strip():
            raise ValueError(key+" must be nonempty")
    size = c["image_size_wh"]
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("image_size_wh must contain two positive integer pixels")
    K = _matrix(c["camera_matrix"], (3, 3), "camera_matrix")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("invalid camera matrix")
    if abs(K[0, 1]) > 1e-12 or abs(K[1, 0]) > 1e-12 or not 0 <= K[0, 2] < size[0] or not 0 <= K[1, 2] < size[1]:
        raise ValueError("camera matrix must be zero-skew and match image geometry")
    c["camera_matrix"] = K.tolist()
    D = np.asarray(c["distortion_coefficients"], dtype=float)
    if c["camera_model"] not in ("pinhole", "fisheye"):
        raise ValueError("camera_model must explicitly be pinhole or fisheye")
    sizes = (4,) if c["camera_model"] == "fisheye" else (4, 5, 8, 12, 14)
    if D.ndim != 1 or len(D) not in sizes or not np.isfinite(D).all():
        raise ValueError("distortion coefficients do not match camera model")
    c["distortion_coefficients"] = D.tolist()
    if c["camera_mount"] == "eye_in_hand":
        if c.get("mount_tcp_side") not in ("left", "right") or "T_base_camera_mm" in c:
            raise ValueError("eye_in_hand requires the explicit mounting arm and only TCP-camera transform")
        c["T_tcp_camera_mm"] = _pose(c.get("T_tcp_camera_mm"), "T_tcp_camera_mm").tolist()
    elif c["camera_mount"] == "fixed":
        if "mount_tcp_side" in c or "T_tcp_camera_mm" in c:
            raise ValueError("fixed camera must not declare a wrist transform")
        c["T_base_camera_mm"] = _pose(c.get("T_base_camera_mm"), "T_base_camera_mm").tolist()
    else:
        raise ValueError("unknown camera mounting")
    appearance = validate_appearance_profile(c["appearance_profile"])
    if appearance["simulated"] != c["simulated"] or appearance["target_id"] != c["target_id"]:
        raise ValueError("appearance target/simulation domain differs from calibration")
    if appearance["image_size_wh"] != size:
        raise ValueError("appearance image geometry differs from camera calibration")
    c["appearance_profile"] = appearance
    c["model"] = validate_geometry_model(c["model"])
    c["fit_limits"] = validate_fit_limits(c["fit_limits"])
    for key in ("max_observation_age_s", "max_pose_skew_s"):
        c[key] = _positive(c[key], key)
    if not isinstance(c["evidence"], dict) or not c["evidence"]:
        raise ValueError("measured calibration evidence is required")
    digest = _digest(c)
    if claimed_hash is not None and claimed_hash != digest:
        raise ValueError("calibration_sha256 mismatch")
    c["calibration_sha256"] = digest
    return c


def localize_bottle(image, calibration, *, T_base_tcp_mm, image_timestamp_ns,
                    pose_timestamp_ns, now_ns, pose_tcp_side=None):
    c = validate_bottle_calibration(calibration)
    result = {"schema": "vision_observation_v1", "valid": False, "reasons": [],
              "source": "known_bottle_silhouette", "target_id": c["target_id"],
              "camera_name": c["camera_name"], "camera_model": c["camera_model"],
              "calibration_sha256": c["calibration_sha256"],
              "appearance_sha256": c["appearance_profile"]["content_sha256"],
              "simulated": c["simulated"], "timestamp_ns": image_timestamp_ns,
              "pose_timestamp_ns": pose_timestamp_ns, "T_base_bottle_mm": None,
              "unit": "mm", "pose_model": "upright_axisymmetric_on_table",
              "orientation_observed": False, "observed_dofs": ["x", "y"],
              "assumed_dofs": ["z_from_table", "upright_axis"], "unobserved_dofs": ["yaw"],
              "deployment_verified": False}
    reasons = result["reasons"]
    if any(type(t) is not int or t < 0 for t in (image_timestamp_ns, pose_timestamp_ns, now_ns)):
        reasons.append("invalid_timestamps")
        return result
    if not 0 <= now_ns-image_timestamp_ns <= c["max_observation_age_s"]*1e9:
        reasons.append("stale_image")
    if not 0 <= now_ns-pose_timestamp_ns <= c["max_observation_age_s"]*1e9:
        reasons.append("stale_pose")
    if abs(image_timestamp_ns-pose_timestamp_ns) > c["max_pose_skew_s"]*1e9:
        reasons.append("image_pose_timestamp_skew")
    if c["camera_mount"] == "eye_in_hand":
        if pose_tcp_side != c["mount_tcp_side"]:
            reasons.append("camera_tcp_side_mismatch")
            return result
        try:
            tcp = _pose(T_base_tcp_mm, "T_base_tcp_mm")
        except (ValueError, TypeError):
            reasons.append("invalid_tcp_pose")
            return result
        camera = tcp @ np.asarray(c["T_tcp_camera_mm"])
        result["T_base_mount_tcp_mm"] = tcp.tolist()
    else:
        camera = np.asarray(c["T_base_camera_mm"])
        result["T_base_mount_tcp_mm"] = None
    result["T_base_camera_mm"] = camera.tolist()
    if reasons:
        return result
    segmentation = segment_bottle(image, c["appearance_profile"])
    result["segmentation"] = {k: v for k, v in segmentation.items() if k != "mask"}
    if not segmentation["valid"]:
        reasons.extend("segmentation:"+x for x in segmentation["reasons"])
        return result
    mask = segmentation["mask"]
    result["segmentation_mask_sha256"] = hashlib.sha256(mask.tobytes()).hexdigest()
    fit = fit_bottle_mask(mask, c["camera_matrix"], c["distortion_coefficients"], camera,
                          c["model"], c["fit_limits"], camera_model=c["camera_model"])
    result["geometry_fit"] = fit
    if not fit["valid"]:
        reasons.extend("geometry:"+x for x in fit["reasons"])
        return result
    result.update(valid=True, T_base_bottle_mm=fit["T_base_bottle_mm"],
                  fit_iou=fit["fit_iou"], contour_error_px=fit["contour_error_px"],
                  position_uncertainty_mm=fit["position_uncertainty_mm"])
    return result
