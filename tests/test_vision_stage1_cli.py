"""Saved RGB image through CLI planning; never contacts hardware."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from bottle_scan.vision_stage1 import main
from bottle_scan.vision_motion import plan_alignment, localize_sample
from bottle_scan.vision_localization import calibration_digest
from bottle_scan.vision_workspace import make_workspace_profile
from bottle_scan.geometry import rotation, grasp_transform
from bottle_scan.backend import Sample
from tests.test_vision_motion import motion_fixture, observation
from tests.test_vision_workspace import fixture
from tests.test_vision_localization import rendered_marker


class VisionStageOneCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile, self.c, self.ref = motion_fixture()

    def write(self, name, value):
        path = self.root/name
        path.write_text(json.dumps(value))
        return str(path)

    def call(self, args):
        with contextlib.redirect_stdout(io.StringIO()), patch('bottle_scan.backend.load_hardware', side_effect=AssertionError('hardware forbidden')):
            return main(args)

    def preview_args(self, shift=(6, -3, 2)):
        bottle = self.ref.copy()
        bottle[:3, 3] += shift
        camera_marker = np.linalg.inv(np.asarray(self.c['T_base_camera_mm'])) @ bottle @ np.linalg.inv(np.asarray(self.c['T_marker_bottle_mm']))
        image = rendered_marker(self.c, camera_marker)
        image_path = self.root/'image.png'
        self.assertTrue(cv2.imwrite(str(image_path), image))
        pose = {'T_base_tcp_mm': self.ref.tolist(), 'current_tcp_mm': self.ref.tolist(),
                'image_timestamp_ns': 1_000_000_000, 'pose_timestamp_ns': 1_000_000_000,
                'now_ns': 1_010_000_000}
        return ['preview', '--workspace', self.write('workspace.json', self.profile),
                '--calibration', self.write('calibration.json', self.c), '--image', str(image_path),
                '--pose', self.write('pose.json', pose), '--out', str(self.root/'plan.json')]

    def test_rgb_image_to_three_dimensional_checked_target(self):
        self.assertEqual(self.call(self.preview_args()), 0)
        plan = json.loads((self.root/'plan.json').read_text())
        self.assertTrue(plan['valid'])
        self.assertEqual(plan['observation']['source'], 'aruco_detection')
        self.assertEqual(plan['status'], 'step_ready')
        np.testing.assert_allclose(np.asarray(plan['target_tcp_mm'])[:3, 3]-self.ref[:3, 3], [6, -3, 2], atol=2.)
        self.assertFalse(plan['contact_allowed'])

    def test_target_loss_produces_rejected_report_and_no_hardware(self):
        args = self.preview_args()
        self.assertTrue(cv2.imwrite(str(self.root/'image.png'), np.full((1080, 1920), 255, np.uint8)))
        self.assertEqual(self.call(args), 2)
        plan = json.loads((self.root/'plan.json').read_text())
        self.assertFalse(plan['valid'])
        self.assertIsNone(plan['target_tcp_mm'])

    def test_existing_output_is_not_overwritten(self):
        args = self.preview_args()
        self.assertEqual(self.call(args), 0)
        original = (self.root/'plan.json').read_bytes()
        self.assertEqual(self.call(args), 2)
        self.assertEqual((self.root/'plan.json').read_bytes(), original)

    def test_build_profile_offline_and_tampered_profile_rejected(self):
        cfg, teaching = fixture()
        args = ['build-workspace', '--config', self.write('config.json', cfg),
                '--teaching', self.write('teaching.json', teaching), '--simulated', '--out', str(self.root/'built.json')]
        self.assertEqual(self.call(args), 0)
        self.assertEqual(json.loads((self.root/'built.json').read_text()), make_workspace_profile(cfg, teaching, simulated=True))
        args = self.preview_args()
        self.profile['teaching']['workspace_max_mm'][0] += 20
        self.write('workspace.json', self.profile)
        self.assertEqual(self.call(args), 2)
        self.assertFalse((self.root/'plan.json').exists())

    def test_teach_and_commission_without_real_do_not_connect(self):
        self.assertEqual(self.call(['teach', '--station', 'missing', '--calibration', 'missing', '--limits', 'missing', '--out', str(self.root/'out')]), 2)
        self.assertEqual(self.call(['commission', '--station', 'missing', '--calibration', 'missing', '--workspace', 'missing', '--out', str(self.root/'out')]), 2)

    def test_simulated_artifacts_cannot_enter_real_commissioning(self):
        args = ['commission', '--real', '--station', 'missing', '--calibration', self.write('cal.json', self.c),
                '--workspace', self.write('ws.json', self.profile), '--out', str(self.root/'live')]
        self.assertEqual(self.call(args), 2)
        self.assertFalse((self.root/'live').exists())

    def test_reference_is_bound_to_calibration(self):
        profile = make_workspace_profile(self.profile['geometry_config'],
            {**self.profile['teaching'], 'reference_calibration_sha256': calibration_digest(self.c)}, simulated=True)
        changed = deepcopy(self.c)
        changed['T_marker_bottle_mm'][0][3] += 2
        obs = observation(changed, self.ref, 1_000_000_000)
        plan = plan_alignment(profile, changed, obs, self.ref, now_ns=1_000_000_000)
        self.assertFalse(plan['valid'])
        self.assertIn('taught_reference_calibration_mismatch', plan['reasons'])

    def test_grasp_center_offset_compensates_fixed_wrist_orientation(self):
        cfg, teaching = fixture()
        cfg['tool']['grasp_center_in_tcp'][0][3] = 30.
        profile = make_workspace_profile(cfg, teaching, simulated=True)
        bottle = self.ref.copy()
        bottle[:3, :3] = rotation([0, 0, 1], 2.)
        plan = plan_alignment(profile, self.c, observation(self.c, bottle, 1_000_000_000),
                              self.ref, now_ns=1_000_000_000)
        self.assertTrue(plan['valid'], plan)
        desired = bottle @ np.linalg.inv(self.ref) @ self.ref @ grasp_transform(cfg)
        actual = np.asarray(plan['goal_tcp_mm']) @ grasp_transform(cfg)
        np.testing.assert_allclose(actual[:3, 3], desired[:3, 3], atol=1e-6)
        np.testing.assert_allclose(np.asarray(plan['goal_tcp_mm'])[:3, :3], self.ref[:3, :3])

    def test_import_and_help_do_not_load_ros_or_connect(self):
        root = Path(__file__).resolve().parents[1]
        code = 'import sys; import bottle_scan.vision_stage1; assert "rospy" not in sys.modules; assert "bridge_ros_public" not in sys.modules'
        result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, 'vision_stage1.py', '--help'], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('build-workspace', result.stdout)


if __name__ == '__main__':
    unittest.main()
