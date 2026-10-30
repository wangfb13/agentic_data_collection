import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from bottle_scan.config import demo_config
from bottle_scan.window_descriptors import enrich_trial_windows, source_frame_gaps


class WindowTests(unittest.TestCase):
    def make_data(self, directory):
        cfg = demo_config()
        cfg["probe"] = {"window_frames": 5}
        cfg["hold"] = {"window_s": .4}
        cfg["tactile"].update(depth_channel=0, pixel_pitch_mm="TODO", normal_axis=2, normal_signs=[1, -1])
        cfg["force"]["unit"] = "sdk"
        cfg["force"]["target_definition"] = "mean"
        reference = {side+kind: np.full((4, 12, 1), value) for side in ("left", "right")
                     for kind, value in (("_dd", 0.), ("_dd_std", .1))}
        meta = {"stages": [{"name": "PROBE", "start_ns": 0, "end_ns": 900_000_000},
                            {"name": "HOLD", "start_ns": 1_000_000_000, "end_ns": 2_000_000_000}],
                "auto": {"probe_stable_window": {"start_ns": 200_000_000, "end_ns": 800_000_000},
                         "probe_not_settled": False}}
        path = Path(directory)/"streams.h5"
        with h5py.File(path, "w") as handle:
            fields = []
            for index in range(21):
                image = np.zeros((4, 12, 1))
                image[1, index % 10 + 1, 0] = index+1
                if index == 7:
                    image[0, 0, 0] = np.nan
                fields.append(image)
            for side, sign in (("left", 1), ("right", -1)):
                for kind in ("dd", "wrench"):
                    group = handle.create_group("streams/"+kind+"_"+side)
                    data = fields if kind == "dd" else [[0, 0, sign*(index+.5), 0, 0, 0] for index in range(21)]
                    group.create_dataset("data", data=np.asarray(data))
                    group.create_dataset("monotonic_ns", data=np.arange(21)*100_000_000)
                    group.create_dataset("metadata_json", data=[json.dumps({"fid": 100+index+(3 if index >= 6 else 0),
                        "unit": "sdk"}) for index in range(21)], dtype=h5py.string_dtype())
                    group.attrs["committed_count"] = 21
        return cfg, meta, reference, path

    def test_last_valid_stable_frames_median_and_exact_force_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, meta, reference, path = self.make_data(directory)
            result = enrich_trial_windows(meta, path, cfg, reference)
            window = result["descriptor_windows"]["PROBE"]["left"]
            self.assertEqual(window["dd_indices"], [3, 4, 5, 6, 8])
            self.assertEqual(window["median"]["centroid_u_px"], 6)
            self.assertEqual(window["median"]["depth_weight_sum"], 6)
            self.assertIsNone(window["median"]["centroid_u_mm"])
            self.assertEqual(window["dd_frame_gaps"]["missing_frames"], 3)
            self.assertEqual(window["invalid_frames"][0]["index"], 7)
            self.assertEqual([entry["normal_force"] for entry in window["frames"]], [3.5, 4.5, 5.5, 6.5, 8.5])
            self.assertEqual(window["median"]["normal_force_sdk"], 5.5)
            self.assertIsNone(window["median"]["cop_u_mm"])
            self.assertAlmostEqual(result["auto"]["probe_snr"]["left"], 6/4.8)
            self.assertEqual(window["selection"], "last_valid_stable_frames")
            self.assertIsNone(window["fallback_reason"])
            # JSON is safe for session.atomic_json(allow_nan=False).
            json.dumps(result, allow_nan=False)

    def test_hold_uses_central_time_window_and_short_probe_does_not_extend(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, meta, reference, path = self.make_data(directory)
            meta["auto"]["probe_stable_window"]["start_ns"] = 600_000_000
            result = enrich_trial_windows(meta, path, cfg, reference)
            probe = result["descriptor_windows"]["PROBE"]["left"]
            hold = result["descriptor_windows"]["HOLD"]["right"]
            self.assertEqual(probe["dd_indices"], [6, 8])
            self.assertTrue(probe["short_window"])
            self.assertEqual(hold["dd_indices"], [13, 14, 15, 16, 17])
            self.assertEqual(hold["median"]["normal_force_sdk"], 15.5)

    def test_unsettled_fallback_is_explicit_and_noise_is_not_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, meta, reference, path = self.make_data(directory)
            meta["auto"]["probe_not_settled"] = True
            del reference["left_dd_std"]
            reference["right_dd_std"][:] = 0
            result = enrich_trial_windows(meta, path, cfg, reference)
            for side in ("left", "right"):
                record = result["descriptor_windows"]["PROBE"][side]
                self.assertEqual(record["fallback_reason"], "probe_not_settled")
                self.assertTrue(record["probe_not_settled"])
                self.assertEqual(record["dd_indices"], [4, 5, 6, 8, 9])
                self.assertIsNone(result["auto"]["probe_snr"][side])
                self.assertTrue(record["snr"]["unavailable_reason"])

    def test_reference_missing_does_not_mask_original_trial_fault(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, meta, _, path = self.make_data(directory)
            meta["auto"]["exceptions"] = [{"error": "original stop"}]
            result = enrich_trial_windows(meta, path, cfg, None)
            self.assertEqual(result["auto"]["exceptions"], [{"error": "original stop"}])
            self.assertEqual(result["descriptor_windows"]["PROBE"]["left"]["selected_frames"], 0)
            self.assertIsNone(result["auto"]["probe_snr"]["left"])

    def test_unit_mismatch_refuses_force_association(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, meta, reference, path = self.make_data(directory)
            cfg["force"]["unit"] = "N"
            result = enrich_trial_windows(meta, path, cfg, reference)
            record = result["descriptor_windows"]["PROBE"]["left"]
            self.assertIsNone(record["median"]["normal_force_N"])
            self.assertTrue(all(entry["force_association"] == "stale_or_invalid_or_unit_mismatch" for entry in record["frames"]))

    def test_real_six_component_units_keep_force_with_unknown_moments(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, meta, reference, path = self.make_data(directory)
            cfg["force"]["unit"] = "N"
            cfg["tactile"]["force_unit"] = "N"
            cfg["tactile"]["moment_unit"] = "sdk"
            with h5py.File(path, "r+") as handle:
                for side in ("left", "right"):
                    dataset = handle["streams/wrench_"+side+"/metadata_json"]
                    for index in range(len(dataset)):
                        dataset[index] = json.dumps({"unit": ["N"]*3+["sdk"]*3})
            result = enrich_trial_windows(meta, path, cfg, reference)
            window = result["descriptor_windows"]["PROBE"]["left"]
            self.assertEqual(window["median"]["normal_force_N"], 5.5)
            self.assertEqual(window["frames"][0]["force_association"], "nearest_host_timestamp")
            self.assertIsNone(window["median"]["cop_u_mm"])

    def test_no_source_ids_report_unknown_gap_count(self):
        report = source_frame_gaps({"times": [1, 2, 3], "metadata": [{}, {}, {}]})
        self.assertIsNone(report["missing_frames"])
        self.assertEqual(report["source_id_missing_count"], 3)


if __name__ == "__main__":
    unittest.main()
