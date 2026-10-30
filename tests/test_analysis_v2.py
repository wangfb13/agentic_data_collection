import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from bottle_scan.analysis import (_outcomes, analyze_session, analyze_sessions,
    dd_descriptors, _projection)
from bottle_scan.config import demo_config
from bottle_scan.geometry import grasp_pose_offset


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_session(path, translation=0, invalid_first=False, unit="sdk", pitch="TODO", g_x=80):
    path.mkdir()
    cfg = demo_config()
    cfg["T0"][0][3] += translation
    cfg["tool"]["grasp_center_in_tcp"][0][3] = g_x
    cfg["tactile"]["pixel_pitch_mm"] = pitch
    cfg["force"].update(unit=unit, target_definition="mean")
    cfg["probe"] = {"window_frames": 5}
    cfg["hold"] = {"window_s": .5}
    reference = np.zeros((4, 10, 3))
    np.savez(path/"ref.npz", left_dd=reference, right_dd=reference,
        left_dd_std=np.full_like(reference, .01), right_dd_std=np.full_like(reference, .01))
    (path/"config_snapshot.yaml").write_text(json.dumps(cfg))
    (path/"session_meta.json").write_text(json.dumps({"T0": cfg["T0"],
        "dd_processing": {"service": "synthetic", "version": "test", "dtype": "float32"},
        "references": [{"path": "ref.npz", "monotonic_ns": 0}]}))
    for i, offset in enumerate((-10, 10)):
        trial = path/"trials"/("trial_"+str(i))
        trial.mkdir(parents=True)
        target = grasp_pose_offset(cfg["T0"], cfg, [offset, 0, 0])
        meta = {"trial_id": trial.name, "task": "contact", "status": "completed", "bottle": "light",
            "T_target": target.tolist(), "offsets": {"a_mm": 999, "c_mm": 0, "l_mm": 0},
            "stages": [{"name": "PROBE", "start_ns": 0, "end_ns": 600_000_000},
                {"name": "HOLD", "start_ns": 700_000_000, "end_ns": 1_300_000_000}],
            "auto": {"bilateral_contact": True, "force_reached": True, "normal_pre_lift": [10, 12],
                "normal_hold": [11, 13], "probe_not_settled": False,
                "probe_stable_window": {"start_ns": 100_000_000, "end_ns": 600_000_000},
                "approach_translation_error_mm": .4, "approach_rotation_error_deg": .2,
                "contact_peak": [14, 15], "lower_unload_height_difference_mm": .5,
                "control_loop_periods": {"arm": {"mean_s": .02, "p95_s": .021, "max_s": .022}}},
            "position_checks": [{"passed": True, "cameras": {"left_wrist": {"offset_px": 1.5},
                "right_wrist": {"offset_px": 2.5}}}],
            "operator": {"label": "x" if i == 0 and invalid_first else ""}}
        (trial/"meta.json").write_text(json.dumps(meta))
        image = reference.copy()
        image[2, 3+2*i, 0] = 2
        times = np.arange(14)*100_000_000
        with h5py.File(trial/"streams.h5", "w") as handle:
            for side in ("left", "right"):
                for kind in ("dd", "wrench"):
                    group = handle.create_group("streams/"+kind+"_"+side)
                    group.create_dataset("data", data=np.asarray([image]*14 if kind == "dd" else [[0, 0, 10, 0, 0, 0]]*14))
                    group.create_dataset("monotonic_ns", data=times)
                    group.create_dataset("metadata_json", data=[json.dumps({"source_sequence": index, "unit": unit}) for index in range(14)], dtype=h5py.string_dtype())
                    group.attrs["committed_count"] = 14
    return cfg


class AnalysisV2Tests(unittest.TestCase):
    def test_sdk_outcomes_never_emit_N_columns_or_reinterpret_legacy_N(self):
        meta = {"auto": {"normal_pre_lift": [3, 4], "normal_hold": [5, 7], "hold_vertical_tangent_sum": 9}}
        result = _outcomes(meta, "sdk")
        self.assertEqual(result["normal_change_sdk"], [2, 3])
        self.assertEqual(result["hold_vertical_tangent_sum_sdk"], 9)
        self.assertFalse(any(key.endswith("_N") for key in result))
        legacy = _outcomes({"auto": {"normal_pre_lift_N": [3, 4]}}, "sdk")
        self.assertIsNone(legacy["normal_pre_lift_sdk"])

    def test_unknown_pitch_keeps_pixels_and_explicit_pixel_moments(self):
        image = np.zeros((5, 7))
        image[1, 2], image[3, 4] = 2, 2
        result = dd_descriptors(image, np.zeros_like(image), depth_channel=None,
            threshold=.1, pixel_pitch_mm="TODO")
        self.assertEqual(result["centroid_u_px"], 3)
        self.assertEqual(result["moment_uu_px2"], 1)
        self.assertTrue(math.isnan(result["moment_uu_mm2"]))
        self.assertTrue(math.isnan(result["centroid_u_mm"]))
        projected = _projection(result, [[1, 0], [0, 1]])
        self.assertEqual(projected["centroid_a_px"], 0)
        self.assertTrue(math.isnan(projected["centroid_a_mm"]))

    def test_combined_offsets_group_identity_quality_and_pixel_slopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, output = root/"one", root/"two", root/"combined"
            write_session(first)
            write_session(second, translation=150, invalid_first=True)
            report = analyze_sessions([first, second], output)
            self.assertEqual(report["trials"], 4)
            self.assertEqual(report["descriptor_rows"], 16)
            summary = rows(output/"summary.csv")
            self.assertEqual({float(row["a_mm"]) for row in summary}, {-10, 10})
            self.assertEqual(len({row["session"] for row in summary}), 2)
            self.assertTrue(all(row["offset_basis"].startswith("target_grasp_center") for row in summary))
            self.assertEqual(summary[0]["check_left_wrist_max_distance_px"], "1.5")
            self.assertEqual(summary[0]["contact_peak_right_sdk"], "15")
            self.assertEqual(summary[0]["arm_loop_p95_s"], "0.021")
            self.assertGreater(float(summary[0]["probe_snr_left"]), 0)
            self.assertEqual(summary[0]["dd_left_frame_gaps"], "0")
            fits = [row for row in rows(output/"linear_fits.csv") if row["axis"] == "a"]
            self.assertTrue(fits)
            self.assertTrue(all(row["slope_unit"] == "px/mm" for row in fits))
            self.assertTrue(all(abs(float(row["slope"])-.1) < 1e-9 for row in fits))
            cells = [row for row in rows(output/"heatmap_cells.csv") if row["metric"] == "normal_completion"]
            self.assertEqual({float(row["a_mm"]): int(row["n_trials"]) for row in cells}, {-10: 1, 10: 2})
            for child in (output/"per_session").iterdir():
                self.assertTrue((child/"linear_fits.csv").exists())
                self.assertTrue((child/"heatmap_light.png").exists())
            descriptors = rows(output/"descriptors.csv")
            self.assertTrue(all(row["normal_force_sdk"] == "10.0" for row in descriptors))
            self.assertFalse(any(key.endswith("_N") for key in descriptors[0]))

    def test_mismatches_are_reported_and_known_pitch_also_exports_mm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root/"one", root/"two"
            write_session(first)
            write_session(second, unit="N", pitch=[.2, .3], g_x=90)
            report = analyze_sessions([first, second], root/"combined")
            mismatch = "\n".join(report["warnings"])
            for field in ("G", "force_unit", "sensor_pixel_pitch_mm"):
                self.assertIn("MISMATCH: "+field, mismatch)
            fits = rows(root/"combined"/"per_session"/"02_two"/"linear_fits.csv")
            self.assertEqual({row["slope_unit"] for row in fits}, {"px/mm", "mm/mm"})
            self.assertTrue((root/"combined"/"centroid_vs_a_left_px.png").exists())


if __name__ == "__main__":
    unittest.main()
