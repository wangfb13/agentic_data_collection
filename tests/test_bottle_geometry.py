"""Known geometric truths and negative silhouettes; no device or ROS access."""
from copy import deepcopy
import unittest

import cv2
import numpy as np

from bottle_scan.bottle_geometry import (fit_bottle_mask, render_bottle_mask,
    validate_geometry_model, validate_fit_limits)


def look_at(position=(120., -420., 300.), target=(0., 0., 35.)):
    position, target = np.asarray(position, float), np.asarray(target, float)
    z = target-position
    z /= np.linalg.norm(z)
    x = np.cross(z, [0., 0., 1.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x, y, z))
    result[:3, 3] = position
    return result


def geometry_fixture(*, cap=False, camera_model="pinhole"):
    K = np.array([[700., 0, 320.], [0, 705., 240.], [0, 0, 1.]])
    D = np.zeros(4 if camera_model == "fisheye" else 5)
    model = {"table_z_mm": 0., "body_profile_mm": [[0., 15.], [60., 17.]]}
    if cap:
        model["cap_profile_mm"] = [[60., 23.], [80., 23.]]
    limits = {"xy_bounds_mm": [[-80., 80.], [-80., 80.]],
              "max_contour_error_px": 1.5, "min_iou": .9,
              "max_position_uncertainty_mm": 10., "min_mask_area_px": 200.,
              "pixel_uncertainty_px": .5, "min_border_px": 2.}
    return K, D, look_at(), model, limits


class BottleGeometryTests(unittest.TestCase):
    def setUp(self):
        self.K, self.D, self.T, self.model, self.limits = geometry_fixture()

    def image(self, xy=(8., -12.), *, model=None, T=None, D=None, camera_model="pinhole"):
        return render_bottle_mask([640, 480], self.K, self.D if D is None else D,
                                 self.T if T is None else T, self.model if model is None else model,
                                 xy, camera_model=camera_model)

    def fit(self, mask, *, model=None, T=None, D=None, limits=None, camera_model="pinhole"):
        return fit_bottle_mask(mask, self.K, self.D if D is None else D,
                              self.T if T is None else T, self.model if model is None else model,
                              self.limits if limits is None else limits, camera_model=camera_model)

    def test_metric_positions_under_nonidentity_camera_rotation(self):
        for xy in ((0., 0.), (15., -20.), (-18., 12.)):
            with self.subTest(xy=xy):
                result = self.fit(self.image(xy))
                self.assertTrue(result["valid"], result)
                np.testing.assert_allclose(np.asarray(result["T_base_bottle_mm"])[:3, 3], [*xy, 0.], atol=.8)
                self.assertFalse(result["orientation_observed"])
                self.assertEqual(result["yaw"], "unobserved_axial_symmetry")
                self.assertEqual(result["estimated_dofs"], ["base_x_mm", "base_y_mm"])

    def test_independent_analytic_cylinder_width_projection(self):
        # Camera exactly horizontal, no distortion, cylinder axis vertical.
        # Tangent-ray geometry gives width = 2*f*r/sqrt(d^2-r^2).
        T = look_at((0., -400., 30.), (0., 0., 30.))
        model = {"table_z_mm": 0., "body_profile_mm": [[0., 15.], [60., 15.]]}
        mask = self.image((0., 0.), T=T, model=model)
        _, xs = np.nonzero(mask)
        analytical_width = 2*self.K[0, 0]*15/np.sqrt(400**2-15**2)
        self.assertAlmostEqual(float(xs.max()-xs.min()), analytical_width, delta=1.1)
        self.assertAlmostEqual(float(xs.mean()), self.K[0, 2], delta=.1)

    def test_pinhole_radial_distortion_is_removed_before_fitting(self):
        D = np.array([-.25, .08, .002, -.001, 0.])
        result = self.fit(self.image((25., 20.), D=D), D=D)
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(np.asarray(result["T_base_bottle_mm"])[:2, 3], [25, 20], atol=.8)

    def test_piecewise_convex_measured_body_profile(self):
        model = {"table_z_mm": 0., "body_profile_mm": [[0., 15.], [35., 18.], [65., 16.]]}
        result = self.fit(self.image((-12., -5.), model=model), model=model)
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(np.asarray(result["T_base_bottle_mm"])[:2, 3], [-12, -5], atol=.8)

    def test_fisheye_projection_and_inverse_are_explicit(self):
        D = np.array([-.12, .02, .001, -.001])
        result = self.fit(self.image((-20., 15.), D=D, camera_model="fisheye"), D=D, camera_model="fisheye")
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(np.asarray(result["T_base_bottle_mm"])[:2, 3], [-20, 15], atol=.8)
        with self.assertRaises(ValueError):
            self.fit(self.image(), camera_model="guess")
        with self.assertRaises(ValueError):
            self.fit(self.image(), camera_model="fisheye")  # Five pinhole coefficients.

    def test_known_white_cap_self_occlusion_is_modelled(self):
        model = {**self.model, "cap_profile_mm": [[60., 23.], [80., 23.]]}
        bare = self.image((10., 6.))
        visible = self.image((10., 6.), model=model)
        self.assertLess(np.count_nonzero(visible), np.count_nonzero(bare))
        result = self.fit(visible, model=model)
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(np.asarray(result["T_base_bottle_mm"])[:2, 3], [10, 6], atol=.9)
        wrong = self.fit(visible)
        self.assertFalse(wrong["valid"], wrong)

    def test_cap_visibility_outside_supported_view_is_rejected(self):
        model = {**self.model, "cap_profile_mm": [[60., 23.], [80., 23.]]}
        result = self.fit(self.image(), model=model, T=look_at((0., -400., 60.), (0., 0., 30.)))
        self.assertFalse(result["valid"])
        self.assertIn("cap_visibility_requires_camera_above_cap", result["reasons"])

    def test_empty_and_insufficient_masks_are_rejected(self):
        for mask in (np.zeros((480, 640), np.uint8), np.eye(20, dtype=np.uint8)):
            with self.subTest(shape=mask.shape):
                result = self.fit(mask)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["T_base_bottle_mm"])

    def test_mask_clipping_at_border_is_rejected(self):
        mask = self.image()
        mask[:20, 300:330] = 255
        result = self.fit(mask)
        self.assertFalse(result["valid"])
        self.assertIn("body_silhouette_touches_image_border", result["reasons"])

    def test_disconnected_second_bottle_is_not_selected_implicitly(self):
        mask = self.image((-45., 0.)) | self.image((45., 0.))
        result = self.fit(mask)
        self.assertFalse(result["valid"])
        self.assertIn("body_silhouette_is_disconnected_or_multiple", result["reasons"])

    def test_external_occlusion_and_wrong_shape_are_rejected(self):
        mask = self.image()
        ys, xs = np.nonzero(mask)
        occluded = mask.copy()
        occluded[:, int(xs.mean()):] = 0
        disk = np.zeros_like(mask)
        cv2.circle(disk, (320, 240), 38, 255, -1)
        for image in (occluded, disk):
            with self.subTest(area=int(np.count_nonzero(image))):
                result = self.fit(image)
                self.assertFalse(result["valid"], result)
                self.assertIsNone(result["T_base_bottle_mm"])

    def test_holes_cannot_be_hidden_by_external_contour_fit(self):
        mask = self.image()
        ys, xs = np.nonzero(mask)
        cv2.circle(mask, (int(xs.mean()), int(ys.mean())), 16, 0, -1)
        result = self.fit(mask)
        self.assertFalse(result["valid"], result)
        self.assertIn("incomplete_occluded_or_wrong_body_silhouette", result["reasons"])

    def test_nearly_parallel_view_is_rejected(self):
        T = look_at((0., -400., 30.), (0., 0., 30.))
        result = self.fit(self.image((0., 0.), T=T), T=T)
        self.assertFalse(result["valid"])
        self.assertIn("view_too_parallel_to_support_plane", result["reasons"])

    def test_uncertainty_limit_is_enforced(self):
        limits = {**self.limits, "max_position_uncertainty_mm": .01}
        result = self.fit(self.image(), limits=limits)
        self.assertFalse(result["valid"])
        self.assertIn("position_uncertainty_exceeds_limit", result["reasons"])

    def test_observability_condition_limit_is_enforced(self):
        limits = {**self.limits, "max_condition_number": 1.001}
        result = self.fit(self.image(), limits=limits)
        self.assertFalse(result["valid"])
        self.assertIn("position_is_poorly_observable", result["reasons"])

    def test_out_of_search_bounds_does_not_clip_into_success(self):
        limits = {**self.limits, "xy_bounds_mm": [[-10., 10.], [-10., 10.]]}
        result = self.fit(self.image((35., 25.)), limits=limits)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["T_base_bottle_mm"])

    def test_table_height_and_visible_body_offset_are_metric(self):
        model = {"table_z_mm": 25., "body_profile_mm": [[5., 14.], [65., 18.]]}
        result = self.fit(self.image((12., 4.), model=model), model=model)
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(np.asarray(result["T_base_bottle_mm"])[:3, 3], [12, 4, 25], atol=.8)

    def test_models_and_limits_are_validated_without_images(self):
        model = deepcopy(self.model)
        limits = deepcopy(self.limits)
        validated = validate_geometry_model(model)
        checked = validate_fit_limits(limits)
        self.assertEqual(validated, model)
        self.assertEqual(checked["xy_bounds_mm"], limits["xy_bounds_mm"])
        for invalid in ({**model, "body_profile_mm": [[0, 15], [30, 10], [60, 15]]},
                        {**model, "body_profile_mm": [[-1, 15], [60, 15]]},
                        {**model, "cap_profile_mm": [[50, 20], [80, 20]]},
                        {**model, "table_z_mm": float("nan")}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_geometry_model(invalid)
        with self.assertRaises(ValueError):
            validate_fit_limits({**limits, "max_position_uncertainty_mm": float("inf")})


if __name__ == "__main__":
    unittest.main()
