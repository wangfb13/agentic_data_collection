import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import validate_force
from tests.test_force_control import force_config


class ValidateForceTests(unittest.TestCase):
    def test_dry_run_does_not_load_or_connect_hardware(self):
        with patch.object(validate_force, "load_config", return_value=force_config()), \
                patch("bottle_scan.backend.load_hardware", side_effect=AssertionError("no hardware")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(validate_force.main(["--dry-run", "--demo"]), 0)
        self.assertIn("DRY-RUN", output.getvalue())
        self.assertIn('"repetitions": 2', output.getvalue())

    def test_simulated_validation_reuses_runner_without_arm_motion(self):
        cfg = force_config()
        cfg["gripper"]["close_speed_mm_s"] = 20.
        cfg["close"]["max_overtravel_mm"] = 3.  # Deliberately synthetic test fixture.
        cfg["force"]["validation"]["repetitions"] = 1
        from bottle_scan.backend import SimBackend
        original = SimBackend.move_linear
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(validate_force, "load_config", return_value=cfg), \
                patch.object(SimBackend, "move_linear", side_effect=AssertionError("validation must not move arms")), \
                contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "force_report.json"
            self.assertEqual(validate_force.main(["--simulate", "--output", str(path)]), 0)
            report = json.loads(path.read_text())
            self.assertTrue(report["simulated"])
            self.assertEqual(report["operator_confirmation"], "SIMULATION_ONLY")
            self.assertTrue(report["passed"])
            self.assertEqual(report["control_signal"], "synthetic_controller_emulation")
            self.assertEqual(len(report["trials"]), 1)
            self.assertTrue(path.with_suffix(".csv").exists())

    def test_cancel_before_first_phase_sends_no_hold_or_open(self):
        from bottle_scan.backend import SimBackend
        from bottle_scan.runtime import Runner
        from bottle_scan.session import Session
        cfg=force_config()
        with tempfile.TemporaryDirectory() as tmp:
            session=Session.create(Path(tmp)/"session",cfg,cfg["T0"],simulated=True)
            backend=SimBackend(cfg,session.T0)
            runner=Runner(session,backend,single_step=True,unattended=False)
            folder=Path(tmp)/"cycle";folder.mkdir()
            with patch("builtins.input",side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                validate_force.run_cycle(runner,folder,1)
            self.assertEqual(backend.calls,[])

    def test_comparison_columns_explicitly_keep_unit_and_source(self):
        from tests.test_force_control import passing_report
        report = passing_report(force_config())
        report.update(evaluation_signal="39020", control_signal="39010")
        report["unit"] = "sdk"
        rows = validate_force.comparison_rows([report])
        self.assertIn("left_contact_peak_sdk", rows[0])
        self.assertNotIn("left_contact_peak_N", rows[0])
        self.assertEqual(rows[0]["control_signal"], "39010")
        self.assertEqual(rows[0]["evaluation_signal"], "39020")


if __name__ == "__main__": unittest.main()
