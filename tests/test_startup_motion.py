"""A5 startup/return policy with virtual feedback, never a ROS connection."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from bottle_scan.backend import SimBackend
from bottle_scan.config import demo_config
from bottle_scan.geometry import translate_local
from bottle_scan.runtime import Runner, SafetyFault, SkippedPoint
from bottle_scan.session import Session


class StartupMotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = demo_config()
        self.session = Session.create(Path(self.tmp.name)/"session", self.cfg,
                                      np.array(self.cfg["T0"]), simulated=True)
        self.backend = SimBackend(self.cfg, self.session.T0)
        self.runner = Runner(self.session, self.backend, single_step=False)
        self.target = translate_local(self.session.T0, -self.cfg["check"]["retreat_mm"], "a", self.cfg["axes"])

    def test_real_confirmation_required_even_without_single_step(self):
        self.backend.simulated = False  # Only changes prompt policy; no ROS exists.
        with patch("builtins.input", return_value="cancel") as ask, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SafetyFault, "not confirmed"):
                self.runner.confirm_startup_move(self.target)
        self.assertIn("MOVE", ask.call_args.args[0])
        self.assertFalse(self.runner.actuation_started)
        self.assertEqual(self.backend.calls, [])

    def test_confirmed_preview_does_not_itself_publish(self):
        self.backend.simulated = False
        with patch("builtins.input", return_value="MOVE"), contextlib.redirect_stdout(io.StringIO()):
            self.runner.confirm_startup_move(self.target)
        self.assertEqual(self.backend.calls, [])
        self.assertTrue(self.session.meta["startup_moves"][-1]["path_check_passed"])

    def test_start_changed_while_operator_waited_refuses_without_motion(self):
        self.backend.simulated = False
        def reply(_):
            self.backend.pose[1, 3] += 2.
            self.backend.step(0.)
            return "MOVE"
        with patch("builtins.input", side_effect=reply), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SafetyFault, "changed"):
                self.runner.confirm_startup_move(self.target)
        self.assertEqual(self.backend.calls, [])

    def test_invalid_initial_path_does_not_prompt_or_publish(self):
        self.backend.simulated = False
        target = self.target.copy()
        target[2, 3] = -100.
        with patch("builtins.input", side_effect=AssertionError("unsafe target must not prompt")), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SkippedPoint):
                self.runner.confirm_startup_move(target)
        self.assertEqual(self.backend.calls, [])

    def test_visual_end_rejects_path_before_any_gripper_command(self):
        self.runner.guard = Mock()
        with patch("bottle_scan.runtime.check_segment", return_value=["test obstacle"]):
            with self.assertRaisesRegex(SkippedPoint, "remain at current position"):
                self.runner.check_bottle("visual_end")
        self.assertEqual(self.backend.calls, [])


if __name__ == "__main__":
    unittest.main()
