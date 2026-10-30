"""Offline tactile descriptors and fixed-session two-wrist image verification.

This module never imports an SDK, opens a camera, or controls a robot. Image
coordinates are x=column, y=row. Physical distances require measured pixel pitch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


def dd_descriptors(dd, reference=None, *, depth_channel=0, contact_threshold=0.1,
                   pixel_pitch_mm=None, edge_width_px=2):
    """Subtract an unloaded reference, then describe positive depth contact.

    Accept depth HxW or channel-last DD HxWxC. Threshold has the same unit as
    SDK depth. ``pixel_pitch_mm`` is scalar or (x,y); None leaves mm fields null.
    CoP is a positive-depth weighted center, a proxy, not a force-calibrated CoP.
    """
    values = np.asarray(dd, dtype=np.float64)
    if values.ndim not in (2, 3) or not np.isfinite(values).all():
        raise ValueError("DD must be a finite HxW or HxWxC array")
    if values.ndim == 3:
        values = values[..., depth_channel]
    if not values.size or contact_threshold < 0 or not np.isfinite(contact_threshold):
        raise ValueError("DD and nonnegative contact threshold are required")
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim == 3:
            ref = ref[..., depth_channel]
        if ref.shape != values.shape or not np.isfinite(ref).all():
            raise ValueError("Reference must match DD depth geometry and be finite")
        values = values - ref
    positive = np.maximum(values, 0.)
    contact = positive > contact_threshold
    weights = np.where(contact, positive, 0.)
    yy, xx = np.indices(values.shape, dtype=np.float64)
    total = float(weights.sum())
    area = int(contact.sum())
    centroid = None
    geometric_centroid = None
    moments = None
    if total > 0:
        cx, cy = float((xx * weights).sum() / total), float((yy * weights).sum() / total)
        centroid = [cx, cy]
        geometric_centroid = [float(xx[contact].mean()), float(yy[contact].mean())]
        moments = [[float(((xx-cx)**2*weights).sum()/total),
                    float(((xx-cx)*(yy-cy)*weights).sum()/total)],
                   [float(((xx-cx)*(yy-cy)*weights).sum()/total),
                    float(((yy-cy)**2*weights).sum()/total)]]
    border = np.zeros(values.shape, dtype=bool)
    width = int(edge_width_px)
    if width < 0:
        raise ValueError("edge_width_px must be nonnegative")
    if width:
        border[:width] = border[-width:] = True
        border[:, :width] = border[:, -width:] = True
    pitch = None
    unknown_pitch = (pixel_pitch_mm is None or isinstance(pixel_pitch_mm, str) and
                     ("TODO" in pixel_pitch_mm.upper() or pixel_pitch_mm.lower() in ("unknown", "unavailable")))
    if not unknown_pitch:
        pitch = np.broadcast_to(np.asarray(pixel_pitch_mm, dtype=float), (2,))
        if not np.isfinite(pitch).all() or np.any(pitch <= 0):
            raise ValueError("Measured pixel pitch must be positive (x,y) mm/pixel")
    return {"centroid_px": centroid, "cop_px": centroid,
            "geometric_centroid_px": geometric_centroid,
            "centroid_mm": None if centroid is None or pitch is None else (np.array(centroid)*pitch).tolist(),
            "area_px": area, "area_mm2": None if pitch is None else float(area*np.prod(pitch)),
            "depth_weight_sum": total, "max_depth": float(positive.max()),
            "edge_fraction": None if total <= 0 else float(weights[border].sum()/total),
            "edge_area_fraction": None if not area else float(np.count_nonzero(contact & border)/area),
            "second_moment_px2": moments, "contact": area > 0,
            "pixel_pitch_available": pitch is not None,
            "cop_definition": "positive_reference_subtracted_depth_weighted_proxy"}


def _gray(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Wrist image must be uint8 gray or RGB")
    if image.ndim == 3:
        if image.shape[2] != 3:
            raise ValueError("Wrist RGB image requires three channels")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(image)


class TemplateGuard:
    """Immutable image references, with explicit measured thresholds per camera.

    Repositioning/resetting a bottle never changes the stored reference. Only
    constructing a NEW guard for an explicitly re-recorded T0 may replace it.
    A low score, ambiguous peak, missing camera, or unknown threshold fails closed.
    """
    def __init__(self, references: Mapping, rois: Mapping, thresholds_px=None,
                 search_radius_px=80, min_score=.8, min_template_std=2., min_peak_margin=.02,
                 checked_cameras=None):
        if len(references) != 2:
            raise ValueError("Exactly two wrist references are required for recording")
        self.checked_cameras = tuple(sorted(references if checked_cameras is None else checked_cameras))
        if (not self.checked_cameras or len(set(self.checked_cameras)) != len(self.checked_cameras)
                or not set(self.checked_cameras) <= set(references)
                or set(rois) != set(self.checked_cameras)):
            raise ValueError("Checked cameras require exactly their matching ROIs")
        if search_radius_px < 1 or not -1 <= min_score <= 1 or min_peak_margin < 0:
            raise ValueError("Invalid template matching configuration")
        self._references = {name: np.array(img, copy=True) for name, img in references.items()}
        self.rois = {}
        self.search_radius_px = int(search_radius_px)
        self.min_score = float(min_score)
        self.min_template_std = float(min_template_std)
        self.min_peak_margin = float(min_peak_margin)
        self.thresholds_px = dict(thresholds_px or {})
        self.calibration = None
        for name, image in self._references.items():
            image.setflags(write=False)
            if name not in self.checked_cameras:
                continue
            gray = _gray(image)
            x, y, w, h = map(int, rois[name])
            if x < 0 or y < 0 or w < 4 or h < 4 or x+w > gray.shape[1] or y+h > gray.shape[0]:
                raise ValueError(f"{name}: ROI must be an in-frame bottle region at least 4x4")
            if float(gray[y:y+h, x:x+w].std()) < self.min_template_std:
                raise ValueError(f"{name}: flat/textureless template; select a textured bottle region")
            image.setflags(write=False)
            self.rois[name] = (x, y, w, h)
        self._validate_thresholds()

    @property
    def reference_sha256(self):
        return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in self._references.items()}

    def _validate_thresholds(self):
        if set(self.thresholds_px) - set(self.checked_cameras):
            raise ValueError("Threshold names must identify the two wrist cameras")
        if any(not np.isfinite(v) or v <= 0 for v in self.thresholds_px.values()):
            raise ValueError("Thresholds must be positive pixels")

    def _match(self, name, image):
        reference = _gray(self._references[name])
        current = _gray(image)
        if current.shape != reference.shape:
            return {"valid": False, "reason": "image_geometry_changed"}
        x, y, w, h = self.rois[name]
        r = self.search_radius_px
        x0, y0 = max(0, x-r), max(0, y-r)
        x1, y1 = min(current.shape[1], x+w+r), min(current.shape[0], y+h+r)
        result = cv2.matchTemplate(current[y0:y1, x0:x1], reference[y:y+h, x:x+w], cv2.TM_CCOEFF_NORMED)
        if not np.isfinite(result).all():
            return {"valid": False, "reason": "nonfinite_match"}
        _, score, _, (px, py) = cv2.minMaxLoc(result)
        other = result.copy()
        # Adjacent pixels belong to the same peak. Repeated distant peaks are unsafe.
        other[max(0, py-3):py+4, max(0, px-3):px+4] = -np.inf
        second = float(other.max())
        margin = float(score-second) if np.isfinite(second) else 2.
        dx, dy = int(px+x0-x), int(py+y0-y)
        reason = "ok"
        if score < self.min_score:
            reason = "low_match_confidence"
        elif margin < self.min_peak_margin:
            reason = "ambiguous_match"
        elif abs(dx) >= r or abs(dy) >= r:
            reason = "search_boundary"
        return {"valid": reason == "ok", "reason": reason, "dx_px": dx, "dy_px": dy,
                "offset_px": float(np.hypot(dx, dy)), "score": float(score), "peak_margin": margin}

    def compare(self, images):
        results = {}
        for name in self.checked_cameras:
            try:
                result = self._match(name, images[name])
            except (KeyError, ValueError, cv2.error) as error:
                result = {"valid": False, "reason": "missing_or_invalid_image", "detail": str(error)}
            threshold = self.thresholds_px.get(name)
            result["threshold_px"] = threshold
            result["passed"] = bool(result["valid"] and threshold is not None and result["offset_px"] <= threshold)
            if result["valid"] and threshold is None:
                result["reason"] = "threshold_not_calibrated"
            elif result["valid"] and not result["passed"]:
                result["reason"] = "bottle_offset_exceeded"
            results[name] = result
        return {"passed": all(x["passed"] for x in results.values()), "cameras": results,
                "reference_sha256": self.reference_sha256}

    def calibrate(self, moved_images, noise_images=None, movement_mm=2., threshold_fraction=.5):
        """Use an operator-confirmed ~2 mm motion and stationary repeat images.

        Stationary repeats are mandatory: calibration rejects weak displacement
        relative to baseline jitter. References remain byte-for-byte unchanged.
        This estimates a pixel threshold for that motion/view, not a full camera
        calibration or a guarantee of equal sensitivity in every direction.
        """
        if not np.isfinite(movement_mm) or movement_mm <= 0 or not 0 < threshold_fraction < 1:
            raise ValueError("Require positive movement_mm and 0 < threshold_fraction < 1")
        if not noise_images or len(noise_images) < 2:
            raise ValueError("At least two stationary image pairs are required for noise validation")
        records, thresholds = {}, {}
        for name in self.checked_cameras:
            moved = self._match(name, moved_images[name])
            noise = [self._match(name, images[name]) for images in noise_images]
            if not moved["valid"] or any(not n["valid"] for n in noise):
                raise ValueError(f"{name}: calibration image match is unreliable")
            noise_max = max(n["offset_px"] for n in noise)
            threshold = moved["offset_px"] * threshold_fraction
            if moved["offset_px"] < 2 or threshold < max(.5, 3*noise_max):
                raise ValueError(f"{name}: intentional displacement is too small compared with image noise")
            thresholds[name] = threshold
            records[name] = {"moved_offset_px": moved["offset_px"], "dx_px": moved["dx_px"],
                             "dy_px": moved["dy_px"], "noise_max_px": noise_max,
                             "threshold_px": threshold,
                             "approx_px_per_mm_in_demonstrated_direction": moved["offset_px"]/movement_mm}
        self.thresholds_px = thresholds
        self.calibration = {"movement_mm_operator_estimate": movement_mm,
                            "threshold_fraction": threshold_fraction, "cameras": records}
        return self.calibration

    def save(self, directory):
        """Write once; refusing overwrite protects fixed-session references."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / "template_guard.json"
        if metadata_path.exists():
            raise FileExistsError("Fixed session references already exist; create a new T0/session explicitly")
        files = {}
        for index, (name, image) in enumerate(self._references.items()):
            filename = f"wrist_reference_{index}.png"
            path = directory / filename
            if path.exists():
                raise FileExistsError(path)
            png = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), png):
                raise OSError(f"Could not write {path}")
            files[name] = filename
        content = {"schema": "bottle_template_guard_v1", "reference_files": files,
                   "reference_sha256": self.reference_sha256, "rois": self.rois,
                   "thresholds_px": self.thresholds_px, "search_radius_px": self.search_radius_px,
                   "min_score": self.min_score, "min_template_std": self.min_template_std,
                   "min_peak_margin": self.min_peak_margin, "calibration": self.calibration,
                   "checked_cameras": list(self.checked_cameras)}
        with metadata_path.open("x", encoding="utf8") as stream:
            json.dump(content, stream, indent=2, allow_nan=False)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        content = json.loads((directory / "template_guard.json").read_text(encoding="utf8"))
        if content.get("schema") != "bottle_template_guard_v1":
            raise ValueError("Unknown template reference schema")
        images = {}
        for name, filename in content["reference_files"].items():
            path = directory / filename
            if path.resolve().parent != directory.resolve():
                raise ValueError("Reference path escapes session directory")
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Missing reference {path}")
            images[name] = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        obj = cls(images, content["rois"], content["thresholds_px"],
                  checked_cameras=content.get("checked_cameras"),
                  **{k: content[k] for k in ("search_radius_px", "min_score", "min_template_std", "min_peak_margin")})
        if obj.reference_sha256 != content["reference_sha256"]:
            raise ValueError("Fixed reference image checksum mismatch")
        obj.calibration = content["calibration"]
        return obj
