"""Unmarked white bottle through existing RGB, metric and guarded planning paths.
Synthetic inputs only: physical accuracy requires separate measured validation.
"""
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

from bottle_scan.bottle_appearance import (build_appearance_profile, segment_bottle,
                                           validate_appearance_profile)
from bottle_scan.bottle_geometry import render_bottle_mask
from bottle_scan.bottle_localization import localize_bottle, validate_bottle_calibration
from bottle_scan.vision_motion import plan_alignment
from tests.test_markerless_motion import markerless_motion_fixture


class WhiteBottleTests(unittest.TestCase):
    def setUp(self):
        self.profile, self.c, self.ref, self.camera = markerless_motion_fixture()
        self.mask = self.mask_at(self.ref[:2, 3])
        self.rgb = self.colour(self.mask)
        self.background = np.where(self.mask == 0, 255, 0).astype(np.uint8)
        self.appearance = build_appearance_profile(self.rgb, self.mask, target_id='white_bottle',
            simulated=True, evidence={'source': 'synthetic labels'}, background_mask=self.background,
            appearance_mode='white')
        self.c.pop('calibration_sha256')
        self.c['target_id'] = 'white_bottle'
        self.c['appearance_profile'] = self.appearance
        self.c = validate_bottle_calibration(self.c)
        from bottle_scan.vision_workspace import make_workspace_profile
        teaching = deepcopy(self.profile['teaching'])
        teaching['reference_calibration_sha256'] = self.c['calibration_sha256']
        self.profile = make_workspace_profile(self.profile['geometry_config'], teaching, simulated=True)

    def mask_at(self, xy):
        c = self.c
        return render_bottle_mask(c['image_size_wh'], c['camera_matrix'], c['distortion_coefficients'],
                                 self.camera, c['model'], xy, camera_model=c['camera_model'])

    def colour(self, mask):
        rgb = np.full((*mask.shape, 3), 220, np.uint8)
        rgb[mask != 0] = [150, 150, 150]
        return rgb

    def test_low_saturation_white_body_reaches_checked_xy_plan(self):
        xy = self.ref[:2, 3] + [3, -2]
        image = self.colour(self.mask_at(xy))
        observation = localize_bottle(image, self.c, T_base_tcp_mm=np.eye(4),
            image_timestamp_ns=2_000_000_000, pose_timestamp_ns=2_000_000_000,
            now_ns=2_000_000_000)
        self.assertTrue(observation['valid'], observation)
        self.assertFalse(observation['orientation_observed'])
        np.testing.assert_allclose(np.asarray(observation['T_base_bottle_mm'])[:2, 3], xy, atol=1.)
        plan = plan_alignment(self.profile, self.c, observation, self.ref, now_ns=2_000_000_000)
        self.assertTrue(plan['valid'], plan)
        np.testing.assert_allclose(np.asarray(plan['goal_tcp_mm'])[:2, 3], xy, atol=1.)
        self.assertFalse(plan['contact_allowed'])

    def test_negative_background_required_and_confusion_rejected(self):
        with self.assertRaises(ValueError):
            build_appearance_profile(self.rgb, self.mask, target_id='white_bottle', simulated=True,
                                     evidence={'source': 'synthetic'}, appearance_mode='white')
        image = self.rgb.copy()
        image[5:25, 5:25] = [150, 150, 150]
        negative = np.zeros_like(self.mask); negative[5:25, 5:25] = 255
        with self.assertRaises(ValueError):
            build_appearance_profile(image, self.mask, target_id='white_bottle', simulated=True,
                evidence={'source': 'synthetic'}, background_mask=negative, appearance_mode='white')

    def test_blank_and_multiple_objects_rejected(self):
        self.assertFalse(segment_bottle(np.full_like(self.rgb, 220), self.appearance)['valid'])
        mask = self.mask_at(self.ref[:2, 3] + [-45, 0]) | self.mask_at(self.ref[:2, 3] + [45, 0])
        result = segment_bottle(self.colour(mask), self.appearance)
        self.assertFalse(result['valid'], result)
        self.assertIn('multiple_body_colour_candidates', result['reasons'])

    def test_white_profile_is_immutable_and_not_markerless_green(self):
        self.assertEqual(self.appearance['schema'], 'known_white_bottle_appearance_v1')
        changed = deepcopy(self.appearance)
        changed['color_model']['value_range'][1] += 1
        with self.assertRaises(ValueError):
            validate_appearance_profile(changed)
        changed = deepcopy(self.c)
        changed.pop('calibration_sha256')
        changed['target_id'] = 'green_bottle'
        with self.assertRaises(ValueError):
            validate_bottle_calibration(changed)

    def test_white_cli_build_detect_and_no_hardware(self):
        from bottle_scan.vision_stage1 import main
        with TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
                patch('bottle_scan.backend.load_hardware', side_effect=AssertionError('no hardware')):
            path = Path(temporary)
            for name, value in [('image.png', cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)),
                                ('foreground.png', self.mask), ('background.png', self.background)]:
                self.assertTrue(cv2.imwrite(str(path/name), value))
            (path/'evidence.json').write_text(json.dumps({'source': 'synthetic only'}))
            args = ['build-appearance', '--image', str(path/'image.png'),
                    '--mask', str(path/'foreground.png'), '--background-mask', str(path/'background.png'),
                    '--appearance-mode', 'white', '--target-id', 'white_bottle', '--simulated',
                    '--evidence', str(path/'evidence.json'), '--out', str(path/'appearance.json')]
            self.assertEqual(main(args), 0)
            args = ['detect-bottle', '--image', str(path/'image.png'),
                    '--appearance', str(path/'appearance.json'), '--out', str(path/'report.json'),
                    '--mask-out', str(path/'detected.png')]
            self.assertEqual(main(args), 0)
            self.assertTrue(json.loads((path/'report.json').read_text())['valid'])
            self.assertTrue((path/'detected.png').exists())


if __name__ == '__main__': unittest.main()
