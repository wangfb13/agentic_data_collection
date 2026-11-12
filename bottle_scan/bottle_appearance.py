"""Conservative, markerless body masks for a known green or white bottle.

Training requires an explicit operator mask of the bottle body; the
cap is excluded. White requires explicitly labelled background too.
This is a bounded-workcell appearance model, not semantic
object recognition. It never infers pose, fills occlusion, opens a camera or
commands a robot. Images are uint8 RGB, masks use image pixel coordinates.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import cv2
import numpy as np


SCHEMA = "known_bottle_appearance_v1"
WHITE_SCHEMA = "known_white_bottle_appearance_v1"
DEFAULT_SETTINGS = {
    "min_training_pixels": 50,
    "hue_margin": 5.,
    "max_hue_radius": 20.,
    "saturation_margin": 35.,
    "value_margin": 45.,
    "minimum_saturation": 30.,
    "minimum_value": 15.,
    "quantile_low": .01,
    "quantile_high": .99,
    "max_training_outlier_fraction": .05,
    "max_background_match_fraction": .01,
    "min_component_area_px": 100,
    "max_component_area_fraction": .4,
    "min_bbox_fill": .45,
    "min_solidity": .75,
    "max_hole_fraction": .08,
    "min_aspect_ratio": .15,
    "max_aspect_ratio": 3.,
    "border_margin_px": 2,
    "search_roi_xywh": None,
}


def _digest(value):
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("appearance profile must be finite JSON data") from exc
    return hashlib.sha256(data).hexdigest()


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(name + " is outside its finite allowed range")
    return float(value)


def _image(image):
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 4:
        raise ValueError("expected uint8 RGB HxWx3 image")
    return np.ascontiguousarray(array)


def _mask(mask, shape, name):
    array = np.asarray(mask)
    if array.shape != shape or array.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
        raise ValueError(name + " must be same-size bool or uint8 mask")
    return array != 0


def _settings(settings, size):
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("settings must be a mapping")
    if settings and set(settings)-set(DEFAULT_SETTINGS):
        raise ValueError("unknown appearance settings")
    value = {**deepcopy(DEFAULT_SETTINGS), **deepcopy(settings or {})}
    for key in ("min_training_pixels", "min_component_area_px", "border_margin_px"):
        if type(value[key]) is not int or value[key] < (0 if key == "border_margin_px" else 1):
            raise ValueError(key + " must be an integer in its allowed range")
    if value["min_training_pixels"] < 20:
        raise ValueError("at least 20 explicitly labelled pixels are required")
    for key in ("hue_margin", "max_hue_radius"):
        value[key] = _number(value[key], key, 0., 45.)
    if value["max_hue_radius"] <= 0 or value["hue_margin"] > value["max_hue_radius"]:
        raise ValueError("invalid hue interval limits")
    for key in ("saturation_margin", "value_margin", "minimum_saturation", "minimum_value"):
        value[key] = _number(value[key], key, 0., 255.)
    for key in ("quantile_low", "quantile_high", "max_training_outlier_fraction", "max_background_match_fraction",
                "max_component_area_fraction", "min_bbox_fill", "min_solidity", "max_hole_fraction"):
        value[key] = _number(value[key], key, 0., 1.)
    if not 0 <= value["quantile_low"] < value["quantile_high"] <= 1 or value["max_component_area_fraction"] <= 0:
        raise ValueError("invalid quantiles or maximum component fraction")
    if value["max_training_outlier_fraction"] > .25:
        raise ValueError("training outlier allowance cannot exceed 25 percent")
    for key in ("min_aspect_ratio", "max_aspect_ratio"):
        value[key] = _number(value[key], key, .01, 100.)
    if value["min_aspect_ratio"] >= value["max_aspect_ratio"]:
        raise ValueError("invalid component aspect-ratio interval")
    roi = value["search_roi_xywh"]
    if roi is not None:
        if not isinstance(roi, (list, tuple)) or len(roi) != 4 or any(type(x) is not int for x in roi):
            raise ValueError("search_roi_xywh must contain four integers")
        x, y, w, h = roi
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > size[0] or y+h > size[1]:
            raise ValueError("search ROI outside the calibrated image")
        value["search_roi_xywh"] = list(roi)
    return value


def _hue_distance(hue, center):
    return np.abs((np.asarray(hue, dtype=float)-float(center)+90.) % 180.-90.)


def _color_mask(hsv, model):
    if set(model) == {"saturation_range", "value_range"}:
        return ((hsv[..., 1] >= model["saturation_range"][0]) & (hsv[..., 1] <= model["saturation_range"][1]) &
                (hsv[..., 2] >= model["value_range"][0]) & (hsv[..., 2] <= model["value_range"][1]))
    return ((_hue_distance(hsv[..., 0], model["hue_center"]) <= model["hue_radius"]) &
            (hsv[..., 1] >= model["saturation_range"][0]) & (hsv[..., 1] <= model["saturation_range"][1]) &
            (hsv[..., 2] >= model["value_range"][0]) & (hsv[..., 2] <= model["value_range"][1]))


def validate_appearance_profile(profile):
    """Validate hash and all algorithm inputs; return an independent copy."""
    if not isinstance(profile, dict):
        raise ValueError("appearance profile must be a mapping")
    value = deepcopy(profile)
    digest = value.pop("content_sha256", None)
    if digest != _digest(value):
        raise ValueError("appearance content_sha256 mismatch")
    required = {"schema", "target_id", "image_size_wh", "color_space", "color_model", "settings",
                "simulated", "training", "evidence", "limitations"}
    white = value.get("schema") == WHITE_SCHEMA
    if (set(value) != required or value["schema"] not in (SCHEMA, WHITE_SCHEMA) or
            value["color_space"] != ("OpenCV_HSV_SV_from_RGB" if white else "OpenCV_HSV_from_RGB")):
        raise ValueError("invalid appearance schema/fields")
    if type(value["simulated"]) is not bool:
        raise ValueError("simulated must be a boolean")
    if not isinstance(value["target_id"], str) or not value["target_id"].strip() or len(value["target_id"]) > 128:
        raise ValueError("target_id must identify the known bottle")
    size = value["image_size_wh"]
    if not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or x < 4 for x in size):
        raise ValueError("invalid image_size_wh")
    if not isinstance(value["evidence"], dict) or not value["evidence"]:
        raise ValueError("training evidence is required")
    settings = _settings(value["settings"], size)
    if settings != value["settings"]:
        raise ValueError("settings must be fully explicit and canonical")
    color = value["color_model"]
    keys = {"saturation_range", "value_range"} if white else {"hue_center", "hue_radius", "saturation_range", "value_range"}
    if not isinstance(color, dict) or set(color) != keys:
        raise ValueError("invalid colour model")
    if not white:
        _number(color["hue_center"], "hue_center", 0., 179.999999)
        _number(color["hue_radius"], "hue_radius", 0., settings["max_hue_radius"])
    for key, minimum in (("saturation_range", settings["minimum_saturation"]),
                         ("value_range", settings["minimum_value"])):
        interval = color[key]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(key + " must be a pair")
        lo, hi = [_number(x, key, 0., 255.) for x in interval]
        if lo > hi or lo < minimum:
            raise ValueError(key + " violates configured limits")
    training = value["training"]
    if not isinstance(training, dict) or training.get("label_semantics") != ("operator_selected_white_body_pixels" if white else "operator_selected_coloured_body_pixels"):
        raise ValueError("explicit training-mask semantics required")
    if type(training.get("foreground_pixels")) is not int or training["foreground_pixels"] < settings["min_training_pixels"]:
        raise ValueError("not enough labelled foreground pixels")
    if type(training.get("background_pixels")) is not int or training["background_pixels"] < 0:
        raise ValueError("invalid background pixel count")
    if white and training["background_pixels"] < settings["min_training_pixels"]:
        raise ValueError("white bottle requires explicit background negative labels")
    for key in ("image_sha256", "foreground_mask_sha256"):
        if not isinstance(training.get(key), str) or len(training[key]) != 64:
            raise ValueError("training pixel evidence hash missing")
    _number(training.get("foreground_retained_fraction"), "foreground_retained_fraction",
            1.-settings["max_training_outlier_fraction"], 1.)
    fraction = training.get("background_match_fraction")
    if training["background_pixels"]:
        _number(fraction, "background_match_fraction", 0., settings["max_background_match_fraction"])
    elif fraction is not None:
        raise ValueError("no negative labels means unknown background match fraction")
    value["content_sha256"] = digest
    return value


def appearance_digest(profile):
    return validate_appearance_profile(profile)["content_sha256"]


def build_appearance_profile(image, foreground_mask, *, target_id, simulated, evidence,
                             background_mask=None, settings=None, appearance_mode="green"):
    """Learn colour only from explicitly selected body pixels, never the full frame."""
    rgb = _image(image)
    foreground = _mask(foreground_mask, rgb.shape[:2], "foreground_mask")
    background = np.zeros(rgb.shape[:2], bool) if background_mask is None else _mask(background_mask, rgb.shape[:2], "background_mask")
    if np.any(foreground & background):
        raise ValueError("foreground and background labels overlap")
    size = [rgb.shape[1], rgb.shape[0]]
    if appearance_mode not in ("green", "white"):
        raise ValueError("appearance_mode must be green or white")
    white = appearance_mode == "white"
    if white:
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("settings must be a mapping")
        settings = {"minimum_saturation": 0., "saturation_margin": 8., "value_margin": 8., **(settings or {})}
    settings = _settings(settings, size)
    if int(foreground.sum()) < settings["min_training_pixels"]:
        raise ValueError("not enough explicitly selected body pixels")
    if white and int(background.sum()) < settings["min_training_pixels"]:
        raise ValueError("white bottle requires explicit background negative labels")
    if settings["search_roi_xywh"] is not None:
        x, y, w, h = settings["search_roi_xywh"]
        permitted = np.zeros(foreground.shape, bool); permitted[y:y+h, x:x+w] = True
        if np.any(foreground & ~permitted):
            raise ValueError("labelled foreground lies outside declared search ROI")
    if type(simulated) is not bool or not isinstance(evidence, dict) or not evidence:
        raise ValueError("explicit simulated flag and nonempty evidence required")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    pixels = hsv[foreground].astype(float)
    # Robust circular centre handles hue wrapping (e.g. 179 and 1). Saturation
    # is used only to downweight near-grey pixels whose hue is undefined.
    low, high = settings["quantile_low"], settings["quantile_high"]
    color = {"saturation_range": [max(settings["minimum_saturation"], float(np.quantile(pixels[:, 1], low))-settings["saturation_margin"]),
                                  min(255., float(np.quantile(pixels[:, 1], high))+settings["saturation_margin"])],
             "value_range": [max(settings["minimum_value"], float(np.quantile(pixels[:, 2], low))-settings["value_margin"]),
                             min(255., float(np.quantile(pixels[:, 2], high))+settings["value_margin"])]}
    if not white:
        weights = pixels[:, 1]/255.
        angles = pixels[:, 0]*(2*np.pi/180.)
        vector = np.sum(weights*np.exp(1j*angles)) / max(float(weights.sum()), 1e-12)
        if abs(vector) < .7:
            raise ValueError("foreground colours are not a compact hue population")
        center = float((np.angle(vector)*180./(2*np.pi)) % 180.)
        if center >= 179.999999: center = 0.
        radius = float(np.quantile(_hue_distance(pixels[:, 0], center), high))+settings["hue_margin"]
        if radius > settings["max_hue_radius"]:
            raise ValueError("foreground hue spread is too broad; refine body labels")
        color.update(hue_center=center, hue_radius=radius)
    matched = _color_mask(hsv, color)
    retained = float(matched[foreground].mean())
    if retained < 1.-settings["max_training_outlier_fraction"]:
        raise ValueError("too many selected body pixels fail the colour model")
    background_fraction = float(matched[background].mean()) if background.any() else None
    if background_fraction is not None and background_fraction > settings["max_background_match_fraction"]:
        raise ValueError("labelled background confuses the target colour model")
    profile = {"schema": WHITE_SCHEMA if white else SCHEMA, "target_id": target_id, "image_size_wh": size,
               "color_space": "OpenCV_HSV_SV_from_RGB" if white else "OpenCV_HSV_from_RGB", "color_model": color,
               "settings": settings, "simulated": simulated, "evidence": deepcopy(evidence),
               "training": {"label_semantics": "operator_selected_white_body_pixels" if white else "operator_selected_coloured_body_pixels",
                   "foreground_pixels": int(foreground.sum()), "background_pixels": int(background.sum()),
                   "foreground_retained_fraction": retained, "background_match_fraction": background_fraction,
                   "image_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                   "foreground_mask_sha256": hashlib.sha256(foreground.astype(np.uint8).tobytes()).hexdigest(),
                   "background_mask_sha256": hashlib.sha256(background.astype(np.uint8).tobytes()).hexdigest() if background.any() else None},
               "limitations": ["colour model is not semantic identity proof",
                   "single known bottle in an inspected workcell",
                   "same image geometry and validated illumination range",
                   "no occlusion filling; metric geometry must validate the complete silhouette"]}
    profile["content_sha256"] = _digest(profile)
    return validate_appearance_profile(profile)


def segment_bottle(image, profile):
    """Return a mask only for one unambiguous complete colour candidate.

    Noise is filtered by area; multiple meaningful colour components are
    rejected even when only one has a good shape. No largest-object fallback,
    contour repair, hole filling, dilation or arbitrary ROI expansion occurs.
    """
    p = validate_appearance_profile(profile)
    result = {"schema": "known_bottle_segmentation_v1", "valid": False, "reasons": [],
              "mask": None, "bbox_xywh": None, "candidates": [], "target_id": p["target_id"],
              "appearance_sha256": p["content_sha256"], "simulated": p["simulated"],
              "source": "known_bottle_colour_segmentation", "semantic_identity_verified": False}
    try:
        rgb = _image(image)
    except (TypeError, ValueError) as exc:
        result["reasons"].append("invalid_image:"+str(exc)); return result
    if [rgb.shape[1], rgb.shape[0]] != p["image_size_wh"]:
        result["reasons"].append("image_geometry_changed"); return result
    settings = p["settings"]
    mask = _color_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), p["color_model"])
    roi = settings["search_roi_xywh"] or [0, 0, rgb.shape[1], rgb.shape[0]]
    rx, ry, rw, rh = roi
    permitted = np.zeros(mask.shape, bool); permitted[ry:ry+rh, rx:rx+rw] = True
    mask &= permitted
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < settings["min_component_area_px"]:
            continue
        component = np.asarray(labels == label, np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = contour_area/max(hull_area, 1.)
        interior = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(interior, [contour], -1, 1, thickness=cv2.FILLED)
        hole_fraction = float(np.sum((interior != 0) & (component == 0)))/max(int(interior.sum()), 1)
        reasons = []
        if area > settings["max_component_area_fraction"]*rgb.shape[0]*rgb.shape[1]:
            reasons.append("component_too_large")
        if area/(w*h) < settings["min_bbox_fill"]:
            reasons.append("component_bbox_fill_low")
        if solidity < settings["min_solidity"]:
            reasons.append("component_nonconvex_or_occluded")
        if hole_fraction > settings["max_hole_fraction"]:
            reasons.append("component_holes_or_occlusion")
        if not settings["min_aspect_ratio"] <= w/h <= settings["max_aspect_ratio"]:
            reasons.append("component_aspect_outside_range")
        margin = settings["border_margin_px"]
        if x <= rx+margin or y <= ry+margin or x+w >= rx+rw-margin or y+h >= ry+rh-margin:
            reasons.append("component_touches_image_or_roi_border")
        info = {"component": label, "bbox_xywh": [x, y, w, h], "area_px": area,
                "centroid_px": [float(v) for v in centroids[label]],
                "solidity": solidity, "hole_fraction": hole_fraction, "reasons": reasons}
        candidates.append((info, component))
    result["candidates"] = [entry[0] for entry in candidates]
    if not candidates:
        result["reasons"].append("no_body_colour_candidate")
    elif len(candidates) != 1:
        result["reasons"].append("multiple_body_colour_candidates")
    elif candidates[0][0]["reasons"]:
        result["reasons"].extend(candidates[0][0]["reasons"])
    else:
        info, component = candidates[0]
        result.update(valid=True, mask=component*255, bbox_xywh=info["bbox_xywh"])
    return result
