"""Offline markerless CLI artifacts and refusals; no robot/camera connection."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from bottle_scan.bottle_appearance import validate_appearance_profile
from bottle_scan.vision_stage1 import main
from bottle_scan.vision_workspace import make_workspace_profile
from tests.test_bottle_appearance import scene, model
from tests.test_bottle_localization import markerless_fixture, rgb_scene, NOW_NS
from tests.test_markerless_motion import markerless_motion_fixture
from tests.test_vision_workspace import fixture as workspace_fixture


class MarkerlessCLITests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rgb, self.mask = scene()
        self.image_path = self.write_image("image.png", self.rgb)
        self.mask_path = self.write_mask("foreground.png", self.mask)
        self.evidence_path = self.write_json("evidence.json", {
            "source": "synthetic CLI fixture; not real deployment evidence"})

    def write_json(self, name, value):
        path = self.root/name
        path.write_text(json.dumps(value, allow_nan=False))
        return str(path)

    def write_image(self, name, image):
        path = self.root/name
        self.assertTrue(cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))
        return str(path)

    def write_mask(self, name, mask):
        path = self.root/name
        self.assertTrue(cv2.imwrite(str(path), mask))
        return str(path)

    def call(self, args):
        self.stdout = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(self.stdout))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            for target in ("bottle_scan.backend.load_hardware",
                           "bottle_scan.vision_stage1.teach",
                           "bottle_scan.vision_stage1.commission"):
                stack.enter_context(patch(target, side_effect=AssertionError("hardware forbidden")))
            return main(args)

    def build_args(self, out="appearance.json"):
        return ["build-appearance", "--image", self.image_path, "--mask", self.mask_path,
                "--target-id", "current_green_bottle", "--evidence", self.evidence_path,
                "--simulated", "--out", str(self.root/out)]

    def detect_args(self, out="detection.json", *, save_mask=True):
        appearance_path = self.write_json("input_appearance.json", model(self.rgb, self.mask))
        args = ["detect-bottle", "--image", self.image_path,
                "--appearance", appearance_path, "--out", str(self.root/out)]
        if save_mask:
            args += ["--mask-out", str(self.root/"segmented.png")]
        return args

    def report(self, name):
        return json.loads((self.root/name).read_text())

    def preview_args(self, delta=(6., -3.)):
        profile, c, ref, camera = markerless_motion_fixture()
        rgb, _ = rgb_scene(c, camera, ref[:2, 3]+delta)
        pose = {"T_base_tcp_mm": ref.tolist(), "current_tcp_mm": ref.tolist(),
                "image_timestamp_ns": NOW_NS, "pose_timestamp_ns": NOW_NS, "now_ns": NOW_NS}
        args = ["preview", "--workspace", self.write_json("workspace.json", profile),
                "--calibration", self.write_json("calibration.json", c),
                "--image", self.write_image("preview_image.png", rgb),
                "--pose", self.write_json("pose.json", pose), "--out", str(self.root/"plan.json")]
        return args, ref

    def test_build_saves_frozen_profile_with_explicit_masks_and_settings(self):
        negative = np.zeros_like(self.mask)
        negative[20:40, 20:40] = 255
        args = self.build_args()+["--background-mask", self.write_mask("negative.png", negative),
            "--settings", self.write_json("settings.json", {"hue_margin": 6.})]
        self.assertEqual(self.call(args), 0, self.stdout.getvalue())
        profile = self.report("appearance.json")
        self.assertEqual(validate_appearance_profile(profile), profile)
        self.assertEqual(profile["target_id"], "current_green_bottle")
        self.assertTrue(profile["simulated"])
        self.assertEqual(profile["settings"]["hue_margin"], 6.)
        self.assertEqual(profile["training"]["foreground_pixels"], int(np.count_nonzero(self.mask)))
        self.assertEqual(profile["training"]["background_pixels"], 400)
        self.assertEqual(profile["training"]["background_match_fraction"], 0.)
        self.assertIn("No hardware connected", self.stdout.getvalue())

    def test_build_requires_an_explicit_foreground_mask(self):
        args = self.build_args()
        index = args.index("--mask")
        del args[index:index+2]
        with self.assertRaises(SystemExit) as error:
            self.call(args)
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root/"appearance.json").exists())

    def test_build_invalid_masks_images_evidence_and_settings_leave_no_output(self):
        cases = (
            ("--image", str(self.root/"missing.png")),
            ("--mask", str(self.root/"missing_mask.png")),
            ("--mask", self.write_mask("small_mask.png", np.ones((10, 10), np.uint8)*255)),
            ("--mask", self.write_mask("empty_mask.png", np.zeros_like(self.mask))),
            ("--evidence", self.write_json("empty_evidence.json", {})),
        )
        for option, value in cases:
            args = self.build_args()
            args[args.index(option)+1] = value
            with self.subTest(option=option, value=value):
                self.assertEqual(self.call(args), 2, self.stdout.getvalue())
                self.assertFalse((self.root/"appearance.json").exists())
        for settings in ({"unknown": 1}, {"hue_margin": -1}, {"min_component_area_px": True}):
            with self.subTest(settings=settings):
                args = self.build_args()+["--settings", self.write_json("bad_settings.json", settings)]
                self.assertEqual(self.call(args), 2, self.stdout.getvalue())
                self.assertFalse((self.root/"appearance.json").exists())

    def test_build_refuses_confusing_or_overlapping_negative_labels(self):
        negative = np.zeros_like(self.mask)
        negative[70:80, 100:110] = 255
        args = self.build_args()+["--background-mask", self.write_mask("negative.png", negative)]
        self.assertEqual(self.call(args), 2)
        self.assertFalse((self.root/"appearance.json").exists())
        self.assertIn("overlap", self.stdout.getvalue())

    def test_build_does_not_overwrite_existing_artifact(self):
        path = self.root/"appearance.json"
        path.write_text("existing measured evidence\n")
        self.assertEqual(self.call(self.build_args()), 2)
        self.assertEqual(path.read_text(), "existing measured evidence\n")

    def test_detect_round_trip_saves_binary_mask_and_nonmetric_report(self):
        self.assertEqual(self.call(self.detect_args()), 0, self.stdout.getvalue())
        report = self.report("detection.json")
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["source"], "known_bottle_colour_segmentation")
        self.assertFalse(report["metric_pose_available"])
        self.assertFalse(report["hardware_connected"])
        self.assertFalse(report["semantic_identity_verified"])
        self.assertNotIn("mask", report)
        np.testing.assert_array_equal(cv2.imread(str(self.root/"segmented.png"), cv2.IMREAD_GRAYSCALE),
                                      self.mask)
        self.assertEqual(self.report("input_appearance.json")["content_sha256"],
                         report["appearance_sha256"])

    def test_detect_can_write_only_metadata(self):
        self.assertEqual(self.call(self.detect_args(save_mask=False)), 0)
        self.assertTrue(self.report("detection.json")["valid"])
        self.assertFalse((self.root/"segmented.png").exists())

    def test_detect_missing_or_multiple_bottles_reports_rejection_without_mask(self):
        args = self.detect_args()
        blank = np.full_like(self.rgb, 190)
        _, second = scene(x=200)
        hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)
        hsv[second != 0] = [78, 150, 110]
        multiple = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        for index, image in enumerate((blank, multiple)):
            with self.subTest(index=index):
                args[args.index("--image")+1] = self.write_image("test_image.png", image)
                args[args.index("--out")+1] = str(self.root/f"rejected_{index}.json")
                self.assertEqual(self.call(args), 2, self.stdout.getvalue())
                report = self.report(f"rejected_{index}.json")
                self.assertFalse(report["valid"])
                self.assertTrue(report["reasons"])
                self.assertFalse(report["metric_pose_available"])
                self.assertFalse((self.root/"segmented.png").exists())

    def test_detect_tampered_appearance_and_bad_image_are_rejected(self):
        args = self.detect_args()
        appearance = self.report("input_appearance.json")
        appearance["settings"]["hue_margin"] = 7.
        self.write_json("input_appearance.json", appearance)
        self.assertEqual(self.call(args), 2)
        self.assertFalse((self.root/"detection.json").exists())
        self.assertFalse((self.root/"segmented.png").exists())
        args = self.detect_args()
        args[args.index("--image")+1] = str(self.root/"missing.png")
        self.assertEqual(self.call(args), 2)
        self.assertFalse((self.root/"detection.json").exists())

    def test_detect_refuses_existing_report_or_mask_and_aliasing_output_paths(self):
        for existing in ("detection.json", "segmented.png"):
            with self.subTest(existing=existing):
                path = self.root/existing
                path.write_bytes(b"do not replace")
                self.assertEqual(self.call(self.detect_args()), 2)
                self.assertEqual(path.read_bytes(), b"do not replace")
                other = "segmented.png" if existing == "detection.json" else "detection.json"
                self.assertFalse((self.root/other).exists())
                path.unlink()
        args = self.detect_args()
        args[args.index("--mask-out")+1] = str(self.root/"detection.json")
        self.assertEqual(self.call(args), 2)
        self.assertFalse((self.root/"detection.json").exists())

    def test_blank_markerless_preview_records_no_robot_target(self):
        c, tcp, _ = markerless_fixture()
        cfg, teaching = workspace_fixture()
        profile = make_workspace_profile(cfg, teaching, simulated=True)
        pose = {"T_base_tcp_mm": tcp.tolist(), "pose_tcp_side": "right",
                "current_tcp_mm": teaching["reference_tcp_mm"],
                "image_timestamp_ns": NOW_NS, "pose_timestamp_ns": NOW_NS, "now_ns": NOW_NS}
        args = ["preview", "--workspace", self.write_json("workspace.json", profile),
                "--calibration", self.write_json("calibration.json", c),
                "--image", self.write_image("blank.png", np.full((480, 640, 3), 190, np.uint8)),
                "--pose", self.write_json("pose.json", pose), "--out", str(self.root/"plan.json")]
        self.assertEqual(self.call(args), 2)
        plan = self.report("plan.json")
        self.assertFalse(plan["valid"])
        self.assertFalse(plan["observation"]["valid"])
        self.assertEqual(plan["observation"]["source"], "known_bottle_silhouette")
        self.assertIn("invalid_visual_observation", plan["reasons"])
        self.assertIsNone(plan["target_tcp_mm"])
        self.assertFalse(plan["contact_allowed"])

    def test_markerless_rgb_preview_produces_checked_translation_without_hardware(self):
        args, ref = self.preview_args()
        self.assertEqual(self.call(args), 0, self.stdout.getvalue())
        plan = self.report("plan.json")
        self.assertTrue(plan["valid"], plan)
        self.assertEqual(plan["status"], "step_ready")
        self.assertFalse(plan["contact_allowed"])
        self.assertEqual(plan["observation"]["source"], "known_bottle_silhouette")
        self.assertEqual(plan["observation"]["observed_dofs"], ["x", "y"])
        self.assertFalse(plan["observation"]["orientation_observed"])
        self.assertFalse(plan["observation"]["deployment_verified"])
        np.testing.assert_allclose(np.asarray(plan["target_tcp_mm"])[:3, 3]-ref[:3, 3],
                                   [6., -3., 0.], atol=.9)
        self.assertFalse(json.loads(self.stdout.getvalue())["hardware_connected"])

    def test_markerless_preview_preserves_existing_output_and_rejects_modified_calibration(self):
        args, _ = self.preview_args()
        self.assertEqual(self.call(args), 0, self.stdout.getvalue())
        original = (self.root/"plan.json").read_bytes()
        self.assertEqual(self.call(args), 2)
        self.assertEqual((self.root/"plan.json").read_bytes(), original)
        (self.root/"plan.json").unlink()
        changed = deepcopy(self.report("calibration.json"))
        changed["model"]["table_z_mm"] += 1.
        self.write_json("calibration.json", changed)
        self.assertEqual(self.call(args), 2)
        self.assertFalse((self.root/"plan.json").exists())

    def test_help_lists_markerless_commands_without_entering_hardware(self):
        with self.assertRaises(SystemExit) as error:
            self.call(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("build-appearance", self.stdout.getvalue())
        self.assertIn("detect-bottle", self.stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
