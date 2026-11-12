"""RGB-to-metric markerless localization; all data are synthetic and offline.

Besides the production renderer, one scene uses independent per-pixel analytic
ray/cylinder intersections. These tests do not certify real camera calibration.
"""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from bottle_scan.bottle_appearance import build_appearance_profile
from bottle_scan.bottle_geometry import render_bottle_mask
from bottle_scan.bottle_localization import validate_bottle_calibration, localize_bottle
from bottle_scan.vision_localization import (validate_calibration, calibration_digest,
                                           localize_image)
from bottle_scan.vision_motion import (GuardedVisionAdapter, localize_sample,
                                      plan_alignment)
from bottle_scan.vision_workspace import make_workspace_profile
from tests.test_bottle_geometry import geometry_fixture
from tests.test_vision_workspace import fixture as workspace_fixture


NOW_NS = 2_000_000_000


def colorize(mask, *, hue=78):
    """Plain gray background with manually labelled synthetic green pixels."""
    hsv = np.full((*mask.shape, 3), [0, 0, 190], np.uint8)
    hsv[mask != 0] = [hue, 150, 110]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def rgb_scene(calibration, camera, xy=(8., -12.)):
    c = calibration
    mask = render_bottle_mask(c["image_size_wh"], c["camera_matrix"],
        c["distortion_coefficients"], camera, c["model"], xy,
        camera_model=c["camera_model"])
    rgb = colorize(mask)
    if "cap_profile_mm" in c["model"]:
        cap_model = {"table_z_mm": c["model"]["table_z_mm"],
                     "body_profile_mm": c["model"]["cap_profile_mm"]}
        cap_mask = render_bottle_mask(c["image_size_wh"], c["camera_matrix"],
            c["distortion_coefficients"], camera, cap_model, xy,
            camera_model=c["camera_model"])
        # The renderer's green mask has already excluded the cap silhouette.
        # Keep boundary pixels belonging to the body so raster rounding does
        # not introduce an artificial one-pixel gap between the two materials.
        rgb[(cap_mask != 0) & (mask == 0)] = [240, 240, 240]
    return rgb, mask


def markerless_fixture(*, camera_model="pinhole", cap=False, fixed=False,
                       simulated=True):
    K, D, camera, model, limits = geometry_fixture(cap=cap, camera_model=camera_model)
    D = (np.array([-.12, .02, .001, -.001]) if camera_model == "fisheye"
         else np.array([-.25, .08, .002, -.001, 0.]))
    angle = np.deg2rad(27.)
    tcp = np.array([[np.cos(angle), -np.sin(angle), 0., 130.],
                    [np.sin(angle), np.cos(angle), 0., -45.],
                    [0., 0., 1., 175.], [0., 0., 0., 1.]])
    c = {"schema": "markerless_bottle_calibration_v1", "unit": "mm",
         "target_id": "current_green_bottle", "camera_name": "right_wrist",
         "camera_model": camera_model, "image_size_wh": [640, 480],
         "camera_matrix": K.tolist(), "distortion_coefficients": D.tolist(),
         "camera_mount": "fixed" if fixed else "eye_in_hand",
         "model": model, "fit_limits": limits, "max_observation_age_s": .5,
         "max_pose_skew_s": .05, "simulated": simulated,
         "evidence": {"source": "synthetic test geometry; not deployment evidence"}}
    if fixed:
        c["T_base_camera_mm"] = camera.tolist()
    else:
        c["mount_tcp_side"] = "right"
        c["T_tcp_camera_mm"] = (np.linalg.inv(tcp) @ camera).tolist()
    rgb, mask = rgb_scene(c, camera, (0., 0.))
    # Training selects only known body pixels, never an automatically selected
    # whole frame. A real training artifact requires a manually reviewed mask.
    foreground = cv2.erode(mask, np.ones((3, 3), np.uint8))
    c["appearance_profile"] = build_appearance_profile(rgb, foreground,
        target_id=c["target_id"], simulated=simulated,
        evidence={"source": "explicit synthetic body labels"})
    return validate_bottle_calibration(c), tcp, camera


def analytic_cylinder_mask(size, K, camera, xy, *, radius=15., bottom=0., top=60.):
    """Independent closed-cylinder ray intersection, no production projection.

    Pixel centres define calibrated pinhole rays. A ray hits the cylinder iff
    either quadratic side intersection has a height in [bottom, top], or a
    top/bottom-plane intersection lies within the circular end disk.
    """
    width, height = size
    v, u = np.indices((height, width), dtype=float)
    rays = np.stack(((u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1],
                     np.ones_like(u)), axis=-1) @ camera[:3, :3].T
    origin = camera[:3, 3]
    ox, oy = origin[:2]-np.asarray(xy)
    a = rays[..., 0]**2+rays[..., 1]**2
    b = 2*(ox*rays[..., 0]+oy*rays[..., 1])
    discriminant = b*b-4*a*(ox*ox+oy*oy-radius*radius)
    visible = np.zeros((height, width), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for sign in (-1., 1.):
            t = (-b+sign*np.sqrt(np.maximum(discriminant, 0)))/(2*a)
            z = origin[2]+t*rays[..., 2]
            visible |= (discriminant >= 0) & (t > 0) & (z >= bottom) & (z <= top)
        for z in (bottom, top):
            t = (z-origin[2])/rays[..., 2]
            x = ox+t*rays[..., 0]
            y = oy+t*rays[..., 1]
            visible |= (t > 0) & (x*x+y*y <= radius*radius)
    return visible.astype(np.uint8)*255


class BottleLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.c, self.tcp, self.camera = markerless_fixture()
        self.rgb, self.mask = rgb_scene(self.c, self.camera)

    def localize(self, image=None, calibration=None, **kwargs):
        arguments = {"T_base_tcp_mm": self.tcp, "image_timestamp_ns": NOW_NS,
                     "pose_timestamp_ns": NOW_NS, "now_ns": NOW_NS,
                     "pose_tcp_side": "right"}
        arguments.update(kwargs)
        return localize_image(self.rgb if image is None else image,
                              self.c if calibration is None else calibration,
                              **arguments)

    def assert_no_pose(self, result):
        self.assertFalse(result["valid"], result)
        self.assertIsNone(result["T_base_bottle_mm"])
        self.assertTrue(result["reasons"])

    def test_calibration_round_trip_and_generic_dispatch_digest(self):
        c = json.loads(json.dumps(self.c, allow_nan=False))
        self.assertEqual(validate_bottle_calibration(c), self.c)
        self.assertEqual(validate_calibration(c), self.c)
        self.assertEqual(calibration_digest(c), self.c["calibration_sha256"])
        returned = validate_bottle_calibration(c)
        returned["model"]["table_z_mm"] = 123.
        self.assertEqual(self.c["model"]["table_z_mm"], 0.)

    def test_calibration_tampering_and_nested_appearance_hash_are_rejected(self):
        for field, value in (("target_id", "another_bottle"),
                             ("max_observation_age_s", 20.),
                             ("calibration_sha256", "0"*64)):
            changed = deepcopy(self.c)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_bottle_calibration(changed)
        changed = deepcopy(self.c)
        changed.pop("calibration_sha256")
        changed["appearance_profile"]["settings"]["hue_margin"] = 6.
        with self.assertRaises(ValueError):
            validate_bottle_calibration(changed)

    def test_explicit_camera_geometry_and_numeric_domains_are_required(self):
        mutations = (
            lambda c: c.update(camera_model="auto"),
            lambda c: c.update(camera_model="fisheye"),  # Wrong coefficient count.
            lambda c: c["camera_matrix"][0].__setitem__(0, float("nan")),
            lambda c: c["camera_matrix"][0].__setitem__(1, 2.),
            lambda c: c.update(image_size_wh=[1280, 960]),
            lambda c: c.update(simulated="false"),
            lambda c: c.update(max_pose_skew_s=True),
            lambda c: c.update(evidence={}),
            lambda c: c.update(unknown_field="ignored?"),
            lambda c: c.update(T_base_camera_mm=np.eye(4).tolist()),
            lambda c: c["fit_limits"].update(min_iou=1.1),
        )
        for index, mutate in enumerate(mutations):
            c = deepcopy(self.c)
            c.pop("calibration_sha256")
            mutate(c)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_bottle_calibration(c)

    def test_independent_analytic_pinhole_rgb_recovers_metric_xy(self):
        c = deepcopy(self.c)
        c.pop("calibration_sha256")
        c["distortion_coefficients"] = [0.]*5
        c["model"] = {"table_z_mm": 0., "body_profile_mm": [[0., 15.], [60., 15.]]}
        xy = (13., -17.)
        mask = analytic_cylinder_mask(c["image_size_wh"], np.asarray(c["camera_matrix"]),
                                       self.camera, xy)
        self.assertGreater(np.count_nonzero(mask), 200)
        result = self.localize(colorize(mask), c)
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(np.asarray(result["T_base_bottle_mm"])[:3, 3],
                                   [*xy, 0.], atol=1.1)

    def test_rgb_distortion_and_nonidentity_hand_eye_composition(self):
        for camera_model, xy in (("pinhole", (18., -20.)), ("fisheye", (-20., 15.))):
            with self.subTest(camera_model=camera_model):
                c, tcp, camera = markerless_fixture(camera_model=camera_model)
                rgb, _ = rgb_scene(c, camera, xy)
                result = self.localize(rgb, c, T_base_tcp_mm=tcp)
                self.assertTrue(result["valid"], result)
                np.testing.assert_allclose(result["T_base_camera_mm"], camera, atol=1e-10)
                np.testing.assert_allclose(np.asarray(result["T_base_bottle_mm"])[:3, 3],
                                           [*xy, 0.], atol=.9)
                self.assertEqual(result["source"], "known_bottle_silhouette")
                self.assertEqual(result["observed_dofs"], ["x", "y"])
                self.assertEqual(result["unobserved_dofs"], ["yaw"])
                self.assertFalse(result["orientation_observed"])
                self.assertFalse(result["deployment_verified"])
                self.assertTrue(result["simulated"])
                json.dumps(result, allow_nan=False)  # Masks must not escape into metadata.

    def test_white_cap_is_not_green_and_its_self_occlusion_is_modelled(self):
        c, tcp, camera = markerless_fixture(cap=True)
        rgb, visible = rgb_scene(c, camera)
        bare = deepcopy(c)
        bare.pop("calibration_sha256")
        bare["model"].pop("cap_profile_mm")
        _, unoccluded = rgb_scene(bare, camera)
        self.assertLess(np.count_nonzero(visible), np.count_nonzero(unoccluded))
        self.assertGreater(np.count_nonzero(np.all(rgb == [240, 240, 240], axis=2)), 50)
        result = self.localize(rgb, c, T_base_tcp_mm=tcp)
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(np.asarray(result["T_base_bottle_mm"])[:2, 3],
                                   [8., -12.], atol=.9)
        wrong_model = self.localize(rgb, bare, T_base_tcp_mm=tcp)
        self.assert_no_pose(wrong_model)
        self.assertTrue(any(r.startswith("geometry:") for r in wrong_model["reasons"]))

    def test_missing_wrong_colour_and_multiple_targets_never_fabricate_a_pose(self):
        _, mask_a = rgb_scene(self.c, self.camera, (-40., 0.))
        _, mask_b = rgb_scene(self.c, self.camera, (40., 0.))
        images = (np.full_like(self.rgb, 190), colorize(self.mask, hue=10),
                  colorize(np.maximum(mask_a, mask_b)))
        for image in images:
            with self.subTest(non_gray=int(np.count_nonzero(image[..., 0] != image[..., 1]))):
                result = self.localize(image)
                self.assert_no_pose(result)
                self.assertTrue(any(r.startswith("segmentation:") for r in result["reasons"]))

    def test_external_gripper_occlusion_is_not_filled_into_a_bottle(self):
        rgb = self.rgb.copy()
        ys, xs = np.nonzero(self.mask)
        middle = (ys.min()+ys.max())//2
        rgb[middle-5:middle+6, xs.min():xs.max()+1] = [220, 100, 20]
        self.assert_no_pose(self.localize(rgb))

    def test_image_shape_dtype_and_resolution_are_checked(self):
        for image in (self.rgb[:200], self.rgb.astype(float), self.rgb[..., 0],
                      self.rgb[..., :2], np.zeros((960, 1280, 3), np.uint8)):
            with self.subTest(shape=image.shape, dtype=str(image.dtype)):
                self.assert_no_pose(self.localize(image))

    def test_timestamp_age_skew_future_and_type_fail_before_detection(self):
        cases = (
            ({"image_timestamp_ns": NOW_NS-600_000_000}, "stale_image"),
            ({"pose_timestamp_ns": NOW_NS-600_000_000}, "stale_pose"),
            ({"pose_timestamp_ns": NOW_NS-100_000_000}, "image_pose_timestamp_skew"),
            ({"image_timestamp_ns": NOW_NS+1}, "stale_image"),
            ({"pose_timestamp_ns": NOW_NS+1}, "stale_pose"),
            ({"now_ns": True}, "invalid_timestamps"),
            ({"image_timestamp_ns": float(NOW_NS)}, "invalid_timestamps"),
            ({"pose_timestamp_ns": -1}, "invalid_timestamps"),
        )
        with patch("bottle_scan.bottle_localization.segment_bottle",
                   side_effect=AssertionError("invalid timing reached detection")):
            for kwargs, reason in cases:
                with self.subTest(kwargs=kwargs):
                    result = self.localize(**kwargs)
                    self.assert_no_pose(result)
                    self.assertIn(reason, result["reasons"])

    def test_wrong_arm_and_invalid_tcp_are_rejected_before_detection(self):
        with patch("bottle_scan.bottle_localization.segment_bottle",
                   side_effect=AssertionError("invalid pose reached detection")):
            for side in ("left", None):
                result = self.localize(pose_tcp_side=side)
                self.assert_no_pose(result)
                self.assertIn("camera_tcp_side_mismatch", result["reasons"])
            malformed = np.eye(4)
            malformed[0, 0] = 2.
            result = self.localize(T_base_tcp_mm=malformed)
            self.assert_no_pose(result)
            self.assertIn("invalid_tcp_pose", result["reasons"])

    def test_fixed_camera_uses_base_extrinsic_without_a_wrist_pose(self):
        c, _, camera = markerless_fixture(fixed=True)
        rgb, _ = rgb_scene(c, camera)
        result = self.localize(rgb, c, T_base_tcp_mm=None, pose_tcp_side=None)
        self.assertTrue(result["valid"], result)
        self.assertIsNone(result["T_base_mount_tcp_mm"])
        np.testing.assert_allclose(result["T_base_camera_mm"], camera)

    def test_sample_uses_the_mounting_arm_and_rejects_wrong_camera_stream(self):
        sample = SimpleNamespace(data=self.rgb, monotonic_ns=NOW_NS,
                                 stream="camera/right_wrist")
        arms = {"right": {"T_base_tcp_mm": self.tcp,
                           "received_monotonic_ns": NOW_NS},
                "left": {"T_base_tcp_mm": np.zeros((4, 4)),
                          "received_monotonic_ns": 0}}
        result = localize_sample(sample, arms, self.c, NOW_NS)
        self.assertTrue(result["valid"], result)
        sample.stream = "camera/left_wrist"
        with self.assertRaisesRegex(ValueError, "stream"):
            localize_sample(sample, arms, self.c, NOW_NS)

    def test_simulation_domain_cannot_be_changed_without_matching_appearance(self):
        changed = deepcopy(self.c)
        changed.pop("calibration_sha256")
        changed["simulated"] = False
        with self.assertRaisesRegex(ValueError, "simulation domain"):
            validate_bottle_calibration(changed)
        cfg, teaching = workspace_fixture()
        profile = make_workspace_profile(cfg, teaching, simulated=True)
        # This deliberately incomplete runner proves the adapter rejects domain
        # mismatch before reading robot/session state or constructing hardware.
        runner = SimpleNamespace(backend=SimpleNamespace(simulated=False))
        with self.assertRaisesRegex(ValueError, "simulation domain"):
            GuardedVisionAdapter(runner, profile, self.c)

    def test_planning_rechecks_domains_calibration_and_appearance_fingerprints(self):
        cfg, teaching = workspace_fixture()
        profile = make_workspace_profile(cfg, teaching, simulated=True)
        observation = self.localize()
        self.assertTrue(observation["valid"], observation)
        mutations = (("simulated", False, "simulation_domain_mismatch"),
                     ("calibration_sha256", "0"*64, "calibration_mismatch"),
                     ("appearance_sha256", "0"*64, "target_appearance_mismatch"),
                     ("target_id", "other_bottle", "target_appearance_mismatch"),
                     ("orientation_observed", True, "unsupported_markerless_observation"))
        for field, value, reason in mutations:
            changed = {**observation, field: value}
            with self.subTest(field=field):
                plan = plan_alignment(profile, self.c, changed,
                    np.asarray(teaching["reference_tcp_mm"]), now_ns=NOW_NS)
                self.assertFalse(plan["valid"], plan)
                self.assertIn(reason, plan["reasons"])
                self.assertIsNone(plan["target_tcp_mm"])


if __name__ == "__main__":
    unittest.main()
