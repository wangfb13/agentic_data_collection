"""Known-colour bottle segmentation tests; synthetic pixels, no hardware."""
from copy import deepcopy
import hashlib
import json
import unittest

import cv2
import numpy as np

from bottle_scan.bottle_appearance import (build_appearance_profile,
    validate_appearance_profile, segment_bottle, appearance_digest)


def scene(hue=78, saturation=150, value=110, *, x=95, y=55, width=44, height=115):
    hsv = np.full((240, 320, 3), [0, 0, 210], np.uint8)
    mask = np.zeros((240, 320), np.uint8)
    cv2.rectangle(mask, (x, y), (x+width-1, y+height-1), 255, -1)
    hsv[mask != 0] = [hue, saturation, value]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), mask


def model(image=None, mask=None, **kwargs):
    if image is None:
        image, mask = scene()
    return build_appearance_profile(image, mask, target_id='current_green_bottle',
        simulated=True, evidence={'source': 'synthetic test pixels'}, **kwargs)


def reseal(profile):
    profile = deepcopy(profile); profile.pop('content_sha256', None)
    profile['content_sha256'] = hashlib.sha256(json.dumps(profile, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return profile


class BottleAppearanceTests(unittest.TestCase):
    def test_explicit_training_and_mask_recovery(self):
        image, mask = scene()
        profile = model(image, mask)
        result = segment_bottle(image, profile)
        self.assertTrue(result['valid'], result)
        np.testing.assert_array_equal(result['mask'], mask)
        self.assertEqual(result['bbox_xywh'], [95, 55, 44, 115])
        self.assertEqual(result['appearance_sha256'], appearance_digest(profile))
        self.assertFalse(result['semantic_identity_verified'])
        self.assertTrue(result['simulated'])

    def test_profile_round_trip_and_detached_validation(self):
        profile = model()
        self.assertEqual(validate_appearance_profile(json.loads(json.dumps(profile))), profile)
        returned = validate_appearance_profile(profile)
        returned['settings']['hue_margin'] = 99.
        self.assertNotEqual(profile, returned)

    def test_small_hue_and_illumination_changes(self):
        profile = model()
        image, mask = scene(hue=81, saturation=130, value=140)
        result = segment_bottle(image, profile)
        self.assertTrue(result['valid'], result)
        np.testing.assert_array_equal(result['mask'], mask)

    def test_hue_wrap_is_continuous(self):
        image, mask = scene(hue=179)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[100:130, 95:139, 0] = 1
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        profile = model(image, mask)
        image2, mask2 = scene(hue=0)
        result = segment_bottle(image2, profile)
        self.assertTrue(result['valid'], result)
        np.testing.assert_array_equal(result['mask'], mask2)

    def test_target_missing_or_wrong_colour_rejects(self):
        profile = model()
        for image in (np.full((240, 320, 3), 210, np.uint8), scene(hue=20)[0], scene(value=230)[0]):
            with self.subTest():
                result = segment_bottle(image, profile)
                self.assertFalse(result['valid'])
                self.assertIsNone(result['mask'])
                self.assertIn('no_body_colour_candidate', result['reasons'])

    def test_two_colour_matches_are_ambiguous_not_largest_selected(self):
        image, mask = scene()
        colour = image[70, 110].copy()
        image[60:155, 200:235] = colour
        result = segment_bottle(image, model())
        self.assertFalse(result['valid'])
        self.assertEqual(len(result['candidates']), 2)
        self.assertIn('multiple_body_colour_candidates', result['reasons'])

    def test_same_colour_background_training_is_rejected(self):
        image, foreground = scene()
        background = np.zeros(foreground.shape, np.uint8)
        background[70:130, 200:240] = 255
        image[background != 0] = image[70, 110]
        with self.assertRaisesRegex(ValueError, 'background confuses'):
            model(image, foreground, background_mask=background)

    def test_explicit_negative_pixels_are_recorded(self):
        image, foreground = scene()
        background = np.zeros(foreground.shape, np.uint8); background[20:40, 20:70] = 255
        profile = model(image, foreground, background_mask=background)
        self.assertEqual(profile['training']['background_match_fraction'], 0.)
        self.assertEqual(profile['training']['background_pixels'], 1000)

    def test_clipped_and_roi_edge_components_rejected(self):
        profile = model()
        clipped = scene(x=0)[0]
        self.assertIn('component_touches_image_or_roi_border', segment_bottle(clipped, profile)['reasons'])
        image, foreground = scene()
        profile = model(image, foreground, settings={'search_roi_xywh': [95, 40, 90, 150]})
        result = segment_bottle(image, profile)
        self.assertFalse(result['valid'])
        self.assertIn('component_touches_image_or_roi_border', result['reasons'])

    def test_holes_are_not_filled(self):
        image, _ = scene()
        image[90:135, 105:129] = 210
        result = segment_bottle(image, model())
        self.assertFalse(result['valid'])
        self.assertIn('component_holes_or_occlusion', result['reasons'])
        self.assertIsNone(result['mask'])

    def test_deep_notch_rejected_as_occluded(self):
        image, _ = scene()
        image[85:140, 95:125] = 210
        result = segment_bottle(image, model())
        self.assertFalse(result['valid'])
        self.assertIn('component_nonconvex_or_occluded', result['reasons'])

    def test_tiny_noise_does_not_create_a_second_target(self):
        image, mask = scene()
        image[15:18, 15:18] = image[70, 110]
        result = segment_bottle(image, model())
        self.assertTrue(result['valid'], result)
        np.testing.assert_array_equal(result['mask'], mask)

    def test_image_geometry_dtype_and_missing_mask_reject(self):
        profile = model()
        for image in (np.zeros((120, 160, 3), np.uint8), np.zeros((240, 320), np.uint8), np.zeros((240, 320, 3), float)):
            self.assertFalse(segment_bottle(image, profile)['valid'])
        image, _ = scene()
        with self.assertRaises(ValueError):
            model(image, np.zeros((240, 320), np.uint8))
        with self.assertRaises(ValueError):
            model(image, np.ones((12, 12), np.uint8))

    def test_profile_mutation_and_resealed_invalid_values_reject(self):
        profile = model()
        bad = deepcopy(profile); bad['color_model']['hue_center'] += 1
        with self.assertRaisesRegex(ValueError, 'sha256'):
            segment_bottle(scene()[0], bad)
        for mutate in (lambda x: x.update(simulated='false'),
                       lambda x: x['settings'].update(min_component_area_px=True),
                       lambda x: x['color_model'].update(value_range=[0., 999.]),
                       lambda x: x['settings'].update(search_roi_xywh=[-1, 0, 10, 10])):
            bad = deepcopy(profile); mutate(bad)
            with self.assertRaises(ValueError):
                validate_appearance_profile(reseal(bad))
        bad = deepcopy(profile); bad['color_model']['hue_radius'] = float('nan')
        with self.assertRaises(ValueError):
            validate_appearance_profile(bad)

    def test_background_overlap_and_uncoloured_training_reject(self):
        image, foreground = scene()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            model(image, foreground, background_mask=foreground)
        gray = np.full_like(image, 130)
        with self.assertRaises(ValueError):
            model(gray, foreground)

    def test_unknown_settings_are_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'unknown appearance settings'):
            model(settings={'magic': 1})


if __name__ == '__main__':
    unittest.main()
