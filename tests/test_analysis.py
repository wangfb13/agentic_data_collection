"""Known-geometry offline checks; no robotics, ROS, or sensor imports."""
import math
import csv
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from bottle_scan.analysis import (_outcomes, analyze_session, center_of_pressure, dd_descriptors, linear_fit,
                                  nearest_reference, pose_errors)


class DescriptorTests(unittest.TestCase):
    def test_baseline_pitch_centroid_and_edge(self):
        reference = np.full((5, 7, 3), 4.0)
        raw = reference.copy()
        raw[2, 3, 2] += 2
        raw[2, 4, 2] += 2
        raw[2, 6, 2] += 4
        raw[0, 0, 2] -= 2  # negative residual is not contact
        original = raw.copy()
        desc = dd_descriptors(raw, reference, depth_channel=2, threshold=1,
                              pixel_pitch_mm=[.5, .25], edge_width_px=1)
        self.assertEqual(desc["depth_weight_sum"], 8)
        self.assertEqual(desc["area_mm2"], 3 * .5 * .25)
        self.assertEqual(desc["edge_fraction"], .5)
        self.assertAlmostEqual(desc["centroid_u_mm"], ((3 * 2 + 4 * 2 + 6 * 4) / 8 - 3) * .5)
        self.assertEqual(desc["centroid_v_mm"], 0)
        self.assertEqual(desc["principal_angle_deg"], 0)
        self.assertTrue(math.isinf(desc["aspect_ratio"]))
        np.testing.assert_array_equal(raw, original)

    def test_empty_and_unknown_units_are_not_fabricated(self):
        field = np.zeros((3, 4))
        desc = dd_descriptors(field, field, depth_channel=None, threshold=.1,
                              pixel_pitch_mm="TODO", edge_width_px=1)
        self.assertTrue(math.isnan(desc["area_mm2"]))
        self.assertTrue(math.isnan(desc["centroid_u_mm"]))
        self.assertTrue(math.isnan(desc["edge_fraction"]))
        with self.assertRaisesRegex(ValueError, "explicitly configured"):
            dd_descriptors(np.zeros((3, 4, 3)), np.zeros((3, 4, 3)), depth_channel=None,
                           threshold=.1, pixel_pitch_mm=.1)

    def test_cop_torque_conversion_and_plane(self):
        # Force [2,3,10] applied at [4,5,1] mm => M=[47,-38,2] Nmm.
        cop = center_of_pressure(np.array([2, 3, 10, .047, -.038, .002]),
                                 force_unit="N", moment_unit="Nm", plane_z_mm=1)
        self.assertAlmostEqual(cop["cop_u_mm"], 4)
        self.assertAlmostEqual(cop["cop_v_mm"], 5)
        invalid = center_of_pressure(np.ones(6), force_unit="sdk_raw", moment_unit="Nm", plane_z_mm=0)
        self.assertTrue(math.isnan(invalid["cop_u_mm"]))
        other_normal = center_of_pressure(np.ones(6), force_unit="N", moment_unit="Nm", plane_z_mm=0, normal_axis=0)
        self.assertTrue(math.isnan(other_normal["cop_u_mm"]))

    def test_fit_known_mapping_and_undefined_r2(self):
        fit = linear_fit([0, 1, 2, 3], [2, 5, 8, 11])
        self.assertAlmostEqual(fit["slope"], 3)
        self.assertAlmostEqual(fit["intercept"], 2)
        self.assertAlmostEqual(fit["r2"], 1)
        self.assertTrue(math.isnan(linear_fit([1, 1], [2, 4])["slope"]))
        self.assertTrue(math.isnan(linear_fit([0, 1], [4, 4])["r2"]))

    def test_reference_is_nearest_with_earlier_tie(self):
        entries = [{"path": "first", "monotonic_ns": 100}, {"path": "second", "monotonic_ns": 200}]
        self.assertEqual(nearest_reference(entries, 160)["path"], "second")
        self.assertEqual(nearest_reference(entries, 150)["path"], "first")

    def test_normal_changes_preserve_each_finger(self):
        outcomes = _outcomes({"auto": {"normal_pre_lift_N": [2, 3], "normal_hold_N": [1.5, 3.25]}})
        self.assertEqual(outcomes["normal_change_left_N"], -.5)
        self.assertEqual(outcomes["normal_change_right_N"], .25)
        self.assertEqual(outcomes["normal_change_N"], [-.5, .25])
        scalar = _outcomes({"auto": {"normal_pre_lift_N": 5, "normal_hold_N": 4}})
        self.assertEqual(scalar["normal_change_N"], -1)
        self.assertTrue(math.isnan(scalar["normal_pre_lift_left_N"]))

    def test_pose_pairing_is_causal_and_rejects_stale_commands(self):
        command = np.eye(4)
        measured = command.copy()
        measured[0, 3] = 3
        measured[1, 3] = 4
        measured[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        commands = {"times": np.array([100_000_000, 300_000_000]), "data": [command, command], "metadata": []}
        actual = {"times": np.array([50_000_000, 200_000_000, 900_000_000]),
                  "data": [measured] * 3, "metadata": [{"stage": "HOLD"}] * 3}
        errors = pose_errors(commands, actual, max_age_s=.2)
        self.assertEqual(len(errors), 1)
        self.assertAlmostEqual(errors[0]["translation_mm"], 5)
        self.assertAlmostEqual(errors[0]["rotation_deg"], 90)

    def test_offline_session_outputs_and_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            trial = session / "trials" / "trial_0000"
            trial.mkdir(parents=True)
            baseline = np.ones((5, 7, 3), dtype=np.float32)
            np.savez(session / "ref.npz", left_dd=baseline, right_dd=baseline)
            (session / "session_meta.json").write_text(json.dumps({"references": [{"path": "ref.npz", "monotonic_ns": 0}]}))
            (session / "config_snapshot.yaml").write_text(json.dumps({
                "tactile": {"depth_channel": 0, "pixel_pitch_mm": [.2, .2], "force_unit": "N", "moment_unit": "Nmm", "cop_plane_z_mm": 0, "normal_axis": 2},
                "analysis": {"centroid_to_al": {"left": [[1, 0], [0, 1]], "right": [[1, 0], [0, 1]]}}}))
            (trial / "meta.json").write_text(json.dumps({
                "bottle": "light", "offsets": {"a_mm": 0, "l_mm": 0, "c_mm": 0},
                "stages": [{"name": "PROBE", "start_ns": 10, "end_ns": 20}, {"name": "HOLD", "start_ns": 30, "end_ns": 50}],
                "auto": {"bilateral_contact": True, "force_reached": True, "contact_times_ns": [3, 4], "contact_widths_mm": [30, 29]},
                "operator": {"label": "", "note": "synthetic test"}}))
            skipped = session / "trials" / "trial_0001"
            skipped.mkdir()
            (skipped / "meta.json").write_text(json.dumps({"task": "contact", "bottle": "light", "status": "skipped_safety", "auto": {}}))
            field = baseline.copy()
            field[2, 4, 0] += 2
            with h5py.File(trial / "streams.h5", "w") as handle:
                for name, data in (("dd_left", [field, field, field * 100]), ("dd_right", [field, field, field * 100]),
                                   ("wrench_left", [[0, 0, 2, 0, -1, 0]] * 3), ("wrench_right", [[0, 0, 2, 0, -1, 0]] * 3),
                                   ("tcp_command", [np.eye(4)] * 3), ("tcp_measured", [np.eye(4)] * 3)):
                    group = handle.create_group("streams/" + name)
                    group.create_dataset("data", data=np.asarray(data))
                    group.create_dataset("monotonic_ns", data=[20, 40, 50])
                    group.create_dataset("metadata_json", data=[json.dumps({"stage": stage}) for stage in ("PROBE", "HOLD", "HOLD")], dtype=h5py.string_dtype())
                    group.attrs["committed_count"] = 2
            report = analyze_session(session)
            self.assertEqual(report["descriptor_rows"], 4)
            self.assertEqual(report["pose_error_rows"], 2)
            self.assertFalse(report["warnings"])
            with (session / "analysis" / "descriptors.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(float(row["depth_weight_sum"]) == 2 for row in rows))
            with (session / "analysis" / "heatmap_cells.csv").open() as stream:
                cells = list(csv.DictReader(stream))
            self.assertTrue(all(int(row["n_trials"]) == 1 for row in cells))
            for output in ("summary.csv", "linear_fits.csv", "heatmap_cells.csv", "pose_error_by_stage.csv", "heatmap_light.png", "zero_offset_drift.png"):
                self.assertTrue((session / "analysis" / output).is_file())


if __name__ == "__main__":
    unittest.main()
