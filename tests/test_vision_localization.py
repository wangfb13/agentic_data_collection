"""Synthetic camera geometry and rendered marker tests; no device access."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from bottle_scan.geometry import rotation
from bottle_scan.vision_localization import (validate_calibration, marker_object_points,
    localize_marker, localize_marker_corners)


def calibration():
    hand_eye = np.eye(4)
    hand_eye[:3, :3] = rotation([0, 1, 0], 17)
    hand_eye[:3, 3] = [12, -8, 24]
    marker_bottle = np.eye(4)
    marker_bottle[:3, :3] = rotation([0, 0, 1], -11)
    marker_bottle[:3, 3] = [8, 18, -35]
    return {
        "schema": "vision_calibration_v1", "camera_name": "right_wrist",
        "camera_mount": "eye_in_hand", "mount_tcp_side": "right",
        "image_size_wh": [640, 480],
        "camera_matrix": [[800., 0, 320], [0, 805., 240], [0, 0, 1]],
        "distortion_coefficients": [0., 0., 0., 0., 0.],
        "T_tcp_camera_mm": hand_eye.tolist(),
        "T_marker_bottle_mm": marker_bottle.tolist(),
        "marker_dictionary": "DICT_4X4_50", "marker_id": 7,
        "marker_length_mm": 60., "max_reprojection_error_px": 1.,
        "max_observation_age_s": .25, "max_pose_skew_s": .03,
        "min_marker_area_px2": 100., "pose_ambiguity_ratio": 1.5,
        "pose_ambiguity_gap_px": .1, "simulated": True,
        "evidence": {"source": "synthetic camera geometry, not a real calibration"},
    }


def marker_pose():
    pose = np.eye(4)
    pose[:3, :3] = rotation([0, 1, 0], 28) @ rotation([1, 0, 0], -12) @ np.diag([1., -1., -1.])
    pose[:3, 3] = [8, -7, 380]
    return pose


def tcp_pose():
    pose = np.eye(4)
    pose[:3, :3] = rotation([0, 0, 1], 57) @ rotation([1, 0, 0], -8)
    pose[:3, 3] = [220, -95, 420]
    return pose


def projected_corners(c, pose=None):
    pose = marker_pose() if pose is None else pose
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    corners, _ = cv2.projectPoints(marker_object_points(c["marker_length_mm"]), rvec,
        pose[:3, 3], np.asarray(c["camera_matrix"]), np.asarray(c["distortion_coefficients"]))
    return corners.reshape(4, 2)


def rendered_marker(c, pose=None, marker_id=None):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, c["marker_dictionary"]))
    marker = cv2.aruco.generateImageMarker(dictionary, c["marker_id"] if marker_id is None else marker_id, 400)
    source = np.float32([[0, 0], [399, 0], [399, 399], [0, 399]])
    transform = cv2.getPerspectiveTransform(source, projected_corners(c, pose).astype(np.float32))
    return cv2.warpPerspective(marker, transform, tuple(c["image_size_wh"]),
                               flags=cv2.INTER_NEAREST, borderValue=255)


def timing(**overrides):
    values = {"T_base_tcp_mm": tcp_pose(), "image_timestamp_ns": 1_000_000_000,
              "pose_timestamp_ns": 1_010_000_000, "now_ns": 1_050_000_000,
              "pose_tcp_side": "right"}
    values.update(overrides)
    return values


class VisionLocalizationTests(unittest.TestCase):
    def test_composes_nonidentity_hand_eye_tcp_and_marker_offset(self):
        c = calibration()
        result = localize_marker_corners(projected_corners(c), c, **timing())
        self.assertTrue(result["valid"], result)
        expected = tcp_pose() @ np.asarray(c["T_tcp_camera_mm"]) @ marker_pose() @ np.asarray(c["T_marker_bottle_mm"])
        np.testing.assert_allclose(result["T_base_bottle_mm"], expected, atol=1e-6)
        self.assertEqual(result["source"], "provided_correspondences")
        self.assertFalse(result["deployment_verified"])
        self.assertTrue(result["simulated"])
        self.assertLess(result["reprojection_error_px"], 1e-6)

    def test_direction_and_mm_are_preserved_in_base_coordinates(self):
        c = calibration()
        poses = [marker_pose(), marker_pose()]
        poses[1][:3, 3] += [4, -3, 2]
        values = [localize_marker_corners(projected_corners(c, p), c, **timing()) for p in poses]
        self.assertTrue(all(v["valid"] for v in values), values)
        delta = np.asarray(values[1]["T_base_bottle_mm"])[:3, 3] - np.asarray(values[0]["T_base_bottle_mm"])[:3, 3]
        expected = (tcp_pose() @ np.asarray(c["T_tcp_camera_mm"]))[:3, :3] @ [4, -3, 2]
        np.testing.assert_allclose(delta, expected, atol=1e-6)

    def test_rendered_tilted_marker_is_detected_and_localized(self):
        c = calibration()
        result = localize_marker(rendered_marker(c), c, **timing())
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["source"], "aruco_detection")
        self.assertEqual(result["marker_id"], 7)
        expected = tcp_pose() @ np.asarray(c["T_tcp_camera_mm"]) @ marker_pose() @ np.asarray(c["T_marker_bottle_mm"])
        # Rasterization/subpixel localization uncertainty is explicit here.
        np.testing.assert_allclose(np.asarray(result["T_base_bottle_mm"])[:3, 3], expected[:3, 3], atol=3.)

    def test_fixed_camera_uses_its_measured_base_transform(self):
        c = calibration()
        c["camera_mount"] = "fixed"
        c.pop("mount_tcp_side")
        c["T_base_camera_mm"] = c.pop("T_tcp_camera_mm")
        result = localize_marker_corners(projected_corners(c), c, **timing())
        self.assertTrue(result["valid"], result)
        expected = np.asarray(c["T_base_camera_mm"]) @ marker_pose() @ np.asarray(c["T_marker_bottle_mm"])
        np.testing.assert_allclose(result["T_base_bottle_mm"], expected, atol=1e-6)

    def test_distortion_coefficients_are_used(self):
        c = calibration()
        c["distortion_coefficients"] = [.12, -.04, .001, -.002, .01]
        result = localize_marker_corners(projected_corners(c), c, **timing())
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(result["T_camera_marker_mm"], marker_pose(), atol=1e-5)

    def test_wrong_or_missing_marker_does_not_fabricate_pose(self):
        c = calibration()
        for image in [np.full((480, 640), 255, np.uint8), rendered_marker(c, marker_id=9)]:
            with self.subTest():
                result = localize_marker(image, c, **timing())
                self.assertEqual(result["reasons"], ["target_marker_not_found"])
                self.assertIsNone(result["T_base_bottle_mm"])

    def test_duplicate_id_is_rejected(self):
        c = calibration()
        image = np.full((480, 640), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        tile = cv2.aruco.generateImageMarker(dictionary, 7, 100)
        image[50:150, 70:170] = tile
        image[250:350, 400:500] = tile
        result = localize_marker(image, c, **timing())
        self.assertEqual(result["reasons"], ["duplicate_target_marker_id"])

    def test_invalid_image_size_dtype_and_shape_rejected(self):
        c = calibration()
        for image in (None, np.zeros((480, 640), float), np.zeros((240, 320), np.uint8),
                      np.zeros((480, 640, 4), np.uint8)):
            with self.subTest():
                result = localize_marker(image, c, **timing())
                self.assertFalse(result["valid"])
                self.assertIn("invalid_image_or_calibration_size_mismatch", result["reasons"])

    def test_timestamp_age_skew_and_future_rejected_before_detection(self):
        c = calibration()
        for change, reason in [({"now_ns": 2_000_000_000}, "stale_or_future_image"),
                               ({"image_timestamp_ns": 1_100_000_000}, "stale_or_future_image"),
                               ({"pose_timestamp_ns": 950_000_000}, "image_pose_timestamp_skew"),
                               ({"image_timestamp_ns": 1.}, "invalid_monotonic_timestamps")]:
            with self.subTest(change=change):
                result = localize_marker(rendered_marker(c), c, **timing(**change))
                self.assertFalse(result["valid"])
                self.assertIn(reason, result["reasons"])

    def test_left_camera_cannot_use_right_tcp_feedback(self):
        c = calibration()
        c["mount_tcp_side"] = "left"
        c["camera_name"] = "left_wrist"
        result = localize_marker_corners(projected_corners(c), c, **timing())
        self.assertEqual(result["reasons"], ["camera_mount_tcp_side_mismatch"])
        self.assertIsNone(result["T_base_bottle_mm"])
        result = localize_marker_corners(projected_corners(c), c, **timing(pose_tcp_side="left"))
        self.assertTrue(result["valid"], result)

    def test_degenerate_nonfinite_and_border_corners_rejected(self):
        c = calibration()
        for corners in (np.zeros((4, 2)), np.full((4, 2), np.nan), [[20, 20], [30, 20], [40, 20], [50, 20]],
                        [[0, 20], [30, 20], [30, 60], [0, 60]]):
            with self.subTest():
                result = localize_marker_corners(corners, c, **timing())
                self.assertFalse(result["valid"])
                self.assertIsNone(result["T_base_bottle_mm"])

    def test_depth_interval_is_enforced(self):
        c = calibration()
        c["max_depth_mm"] = 100.
        result = localize_marker_corners(projected_corners(c), c, **timing())
        self.assertIn("no_valid_front_facing_pose_in_depth_range", result["reasons"])

    def test_back_of_camera_candidates_are_rejected(self):
        c = calibration()
        rvec, _ = cv2.Rodrigues(marker_pose()[:3, :3])
        with patch("cv2.solvePnPGeneric", return_value=(1, (rvec,), (np.array([[0.], [0.], [-400.]]),))):
            result = localize_marker_corners(projected_corners(c), c, **timing())
        self.assertFalse(result["valid"])
        self.assertIn("no_valid_front_facing_pose_in_depth_range", result["reasons"])

    def test_planar_ambiguity_is_not_silently_chosen(self):
        c = calibration()
        p = marker_pose()
        rvec, _ = cv2.Rodrigues(p[:3, :3])
        with patch("cv2.solvePnPGeneric", return_value=(2, (rvec, rvec), (p[:3, 3], p[:3, 3]))):
            result = localize_marker_corners(projected_corners(c), c, **timing())
        self.assertEqual(result["reasons"], ["ambiguous_planar_pose"])
        self.assertIsNone(result["T_base_bottle_mm"])

    def test_reprojection_error_not_ignored(self):
        c = calibration()
        points = projected_corners(c)
        points[2] += [9, 7]
        c["max_reprojection_error_px"] = .1
        result = localize_marker_corners(points, c, **timing())
        self.assertFalse(result["valid"])
        self.assertIn("reprojection_error_exceeded", result["reasons"])

    def test_calibration_validation_and_hash_are_strict(self):
        c = validate_calibration(calibration())
        self.assertEqual(validate_calibration(c), c)
        changed = deepcopy(c)
        changed["marker_length_mm"] += 1
        with self.assertRaisesRegex(ValueError, "sha256"):
            validate_calibration(changed)
        mutations = [lambda x: x.update(simulated="false"),
                     lambda x: x.update(marker_length_mm=float("nan")),
                     lambda x: x.update(marker_id=1000),
                     lambda x: x.update(evidence={}),
                     lambda x: x.update(unit="m"),
                     lambda x: x.update(camera_matrix=[[0, 0, 320], [0, 800, 240], [0, 0, 1]]),
                     lambda x: x.update(T_tcp_camera_mm=np.zeros((4, 4)).tolist()),
                     lambda x: x.update(T_base_camera_mm=np.eye(4).tolist()),
                     lambda x: x.pop("mount_tcp_side"),
                     lambda x: x.update(distortion_coefficients=[1, 2, 3])]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                invalid = calibration()
                mutate(invalid)
                with self.assertRaises(ValueError):
                    validate_calibration(invalid)

    def test_invalid_tcp_pose_rejected(self):
        c = calibration()
        result = localize_marker_corners(projected_corners(c), c, **timing(T_base_tcp_mm=np.zeros((4, 4))))
        self.assertFalse(result["valid"])
        self.assertEqual(result["reasons"], ["invalid_tcp_pose"])


if __name__ == "__main__":
    unittest.main()
