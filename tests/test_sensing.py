"""Offline numerical/storage tests. No robot SDK or network imports."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from bottle_scan.sensing import TemplateGuard, dd_descriptors


class SensingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.images = {name: rng.integers(0, 256, (110, 140, 3), dtype=np.uint8)
                       for name in ("wrist_1", "wrist_2")}
        self.rois = {name: (40, 35, 28, 32) for name in self.images}

    def shifted(self, dx):
        return {name: cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (140, 110))
                for name, image in self.images.items()}

    def test_fixed_reference_and_threshold_loop(self):
        guard = TemplateGuard(self.images, self.rois)
        before = guard.reference_sha256
        self.assertFalse(guard.compare(self.images)["passed"])
        calibration = guard.calibrate(self.shifted(10), [self.images, self.images])
        self.assertEqual(calibration["cameras"]["wrist_1"]["threshold_px"], 5.)
        self.assertFalse(guard.compare(self.shifted(10))["passed"])
        self.assertTrue(guard.compare(self.shifted(4))["passed"])
        self.assertTrue(guard.compare(self.images)["passed"])
        self.assertEqual(before, guard.reference_sha256)
        with tempfile.TemporaryDirectory() as directory:
            guard.save(directory)
            restored = TemplateGuard.load(directory)
            self.assertEqual(before, restored.reference_sha256)
            self.assertTrue(restored.compare(self.images)["passed"])
            with self.assertRaises(FileExistsError):
                guard.save(directory)

    def test_single_camera_failure_fails_closed(self):
        guard = TemplateGuard(self.images, self.rois, {n: 5 for n in self.images})
        moved = dict(self.images)
        moved["wrist_2"] = self.shifted(8)["wrist_2"]
        self.assertFalse(guard.compare(moved)["passed"])
        self.assertFalse(guard.compare({"wrist_1": self.images["wrist_1"]})["passed"])
        moved["wrist_2"] = np.zeros_like(moved["wrist_2"])
        self.assertFalse(guard.compare(moved)["passed"])

    def test_flat_and_noisy_calibration_rejected(self):
        with self.assertRaises(ValueError):
            TemplateGuard({n: np.zeros_like(x) for n, x in self.images.items()}, self.rois)
        guard = TemplateGuard(self.images, self.rois)
        with self.assertRaises(ValueError):
            guard.calibrate(self.shifted(2), [self.shifted(1), self.shifted(1)])
        with self.assertRaises(ValueError):
            guard.calibrate(self.shifted(10))
        self.assertEqual(guard.thresholds_px, {})

    def test_repeated_texture_rejected(self):
        tile = self.images["wrist_1"][35:67, 40:68]
        repeated = np.tile(tile, (4, 5, 1))
        images = {"a": repeated, "b": repeated}
        guard = TemplateGuard(images, {n: (28, 32, 28, 32) for n in images}, {n: 5 for n in images})
        self.assertFalse(guard.compare(images)["passed"])

    def test_dd_centroid_and_units(self):
        baseline = np.ones((12, 14, 3), dtype=np.float32)
        dd = baseline.copy()
        dd[4, 7, 0] += 2
        result = dd_descriptors(dd, baseline, pixel_pitch_mm=(.1, .2))
        self.assertEqual(result["centroid_px"], [7., 4.])
        np.testing.assert_allclose(result["centroid_mm"], [.7, .8])
        self.assertEqual(result["area_px"], 1)
        self.assertAlmostEqual(result["area_mm2"], .02)
        self.assertEqual(result["edge_fraction"], 0.)
        self.assertIsNone(dd_descriptors(baseline, baseline)["centroid_px"])
        self.assertIsNone(dd_descriptors(dd, baseline)["centroid_mm"])

    def test_edge_fraction_uses_deformation_weight_not_pixel_count(self):
        dd = np.zeros((9, 9), dtype=np.float32)
        dd[0, 4], dd[4, 4] = 9., 1.
        result = dd_descriptors(dd, contact_threshold=.1, edge_width_px=1)
        self.assertAlmostEqual(result["edge_fraction"], .9)
        self.assertAlmostEqual(result["edge_area_fraction"], .5)


class RecordingTests(unittest.TestCase):
    def test_raw_dtype_shape_timestamps_and_source_ids(self):
        import h5py
        from bottle_scan.recording import H5Writer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"streams.h5"
            raw = np.random.default_rng(2).normal(size=(17, 19, 3)).astype("float32")
            source_timestamp = 1_800_000_000_123_456_789
            with H5Writer(path) as writer:
                writer.append("tactile/left/dd", raw, 500, source_timestamp,
                              {"source_sequence": 31, "stage": "PROBE"})
                writer.append("tactile/left/dd", raw+1, 600, attrs={"source_sequence": 34})
                writer.append("tactile/right/dd", raw, 510)
                with self.assertRaises(ValueError):
                    writer.append("tactile/left/dd", raw.astype("float64"), 700)
                with self.assertRaises(ValueError):
                    writer.append("tactile/left/dd", raw, 400)
            with h5py.File(path, "r") as h5:
                group = h5["streams/tactile/left/dd"]
                self.assertEqual(group["data"].dtype, raw.dtype)
                self.assertEqual(group["data"].shape, (2, 17, 19, 3))
                np.testing.assert_array_equal(group["data"][0], raw)
                self.assertEqual(group["data"].compression, "gzip")
                self.assertEqual(group.attrs["committed_count"], 2)
                self.assertEqual(json.loads(group["metadata_json"][0])["device_timestamp_raw"], source_timestamp)
                self.assertEqual(json.loads(group["metadata_json"][1])["source_sequence"], 34)
                self.assertTrue(np.isnan(group["device_timestamp"][1]))
                np.testing.assert_array_equal(group["sequence"], [0, 1])

    def test_jpeg_bytes_and_per_camera_csv(self):
        from bottle_scan.recording import JpegWriter
        image = np.random.default_rng(8).integers(0, 256, (12, 15, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", image)
        with tempfile.TemporaryDirectory() as directory:
            with JpegWriter(directory) as writer:
                first = writer.append("wrist_1", encoded.tobytes(), 100, attrs={"phase": "CHECK"})
                writer.append("wrist_2", image, 99)
            self.assertEqual((Path(directory)/first).read_bytes(), encoded.tobytes())
            with (Path(directory)/"timestamps.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["sequence"] for r in rows], ["0", "0"])
            self.assertEqual(rows[0]["device_timestamp"], "")
            self.assertEqual(json.loads(rows[0]["metadata_json"])["encoding"], "source_jpeg_verbatim")

    def test_receiver_validated_jpeg_checks_bytes_without_redecoding(self):
        import hashlib
        from unittest.mock import patch
        from bottle_scan.recording import JpegWriter
        image = np.random.default_rng(18).integers(0,256,(24,31,3),dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg",image)
        payload=encoded.tobytes();attrs={"receiver_validated_jpeg_sha256":hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            with JpegWriter(directory) as writer:
                with patch("bottle_scan.recording.cv2.imdecode",side_effect=AssertionError("redundant decode")):
                    path=writer.append("right_wrist",payload,100,attrs=attrs)
                with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                    writer.append("right_wrist",payload+b"changed",101,attrs=attrs)
                with self.assertRaisesRegex(ValueError,"valid JPEG"):
                    writer.append("right_wrist",b"invalid",101)
            self.assertEqual((Path(directory)/path).read_bytes(),payload)
            with (Path(directory)/"timestamps.csv").open() as stream:
                row=next(csv.DictReader(stream))
            self.assertEqual(json.loads(row["metadata_json"])["jpeg_validation"],"receiver_decode_and_writer_sha256")


if __name__ == "__main__":
    unittest.main()
