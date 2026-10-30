"""Preparation UI waits and reference previews; no device/runtime imports."""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import prepare


class ObservationBackend:
    simulated = False

    def __init__(self):
        self.clock_ns = 10_000_000_000
        self.images_read = 0
        self.image_pair = {name: np.full((12, 17, 3), value, np.uint8)
                           for name, value in (("left_wrist", 35), ("right_wrist", 83))}

    def monotonic_ns(self):
        return self.clock_ns

    def images(self):
        self.images_read += 1
        return {name: image.copy() for name, image in self.image_pair.items()}


class TimedObservationRunner:
    """Minimal phase-clock model matching Runner.phase/tick, without motors."""
    def __init__(self):
        self.backend = ObservationBackend()
        self.cfg = {"motion": {"settle_s": .3, "stage_timeouts_s": {"BASELINE": 1., "CHECK": 1.}}}
        self.stage = "BASELINE"
        self.stage_started_ns = 0
        self.phases = []
        self.waits = []
        self.ask = Mock(return_value="")

    @contextmanager
    def phase(self, name):
        self.stage = name
        self.stage_started_ns = self.backend.monotonic_ns()
        self.phases.append((name, self.stage_started_ns))
        yield

    def wait(self, seconds, *, noncontact):
        if not noncontact:
            raise AssertionError("Preparation observation must monitor noncontact")
        self.backend.clock_ns += round(seconds*1e9)
        elapsed_ns = self.backend.monotonic_ns()-self.stage_started_ns
        if elapsed_ns > self.cfg["motion"]["stage_timeouts_s"][self.stage]*1e9:
            raise RuntimeError(self.stage+" timeout")
        self.waits.append((self.stage, elapsed_ns))


class PreparationInteractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @contextmanager
    def no_device_imports(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name in ("backend", "bottle_scan.backend", "runtime", "bottle_scan.runtime", "rospy", "dmrobotics"):
                raise AssertionError("Unexpected device/runtime import: "+name)
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            yield

    def test_long_roi_and_displacement_waits_begin_fresh_observation_phases(self):
        runner = TimedObservationRunner()
        original_cfg = copy.deepcopy(runner.cfg)
        with self.no_device_imports():
            # Human GUI/file selection takes five minutes after BASELINE.
            runner.backend.clock_ns += 300_000_000_000
            noise = prepare.capture_observation_pairs(runner, 3)
            # Human 2 mm displacement takes another five minutes.
            runner.backend.clock_ns += 300_000_000_000
            moved = prepare.capture_observation_pairs(runner, 1)
        self.assertEqual(len(noise), 3)
        self.assertEqual(len(moved), 1)
        self.assertEqual([name for name, _ in runner.phases], ["CHECK", "CHECK"])
        self.assertEqual([elapsed for _, elapsed in runner.waits], [300_000_000, 600_000_000, 900_000_000, 300_000_000])
        self.assertEqual(runner.cfg, original_cfg)
        self.assertLess(noise[-1][0], moved[0][0])
        runner.ask.assert_not_called()

    def test_actual_observation_timeout_remains_enforced(self):
        runner = TimedObservationRunner()
        with self.assertRaisesRegex(RuntimeError, "CHECK timeout"):
            prepare.capture_observation_pairs(runner, 4)
        self.assertEqual(runner.cfg["motion"]["stage_timeouts_s"]["CHECK"], 1.)

    def test_preview_is_native_lossless_and_separate_from_fixed_guard(self):
        references = self.root/"references"
        fixed = references/"bottle_position"
        fixed.mkdir(parents=True)
        immutable = fixed/"template_guard.json"
        immutable.write_bytes(b"original immutable guard")
        rng = np.random.default_rng(29)
        pair = {name: rng.integers(0, 256, (23, 31, 3), dtype=np.uint8)
                for name in ("left_wrist", "right_wrist")}
        originals = {name: image.copy() for name, image in pair.items()}
        directory = references/"bottle_position_preview"
        with self.no_device_imports():
            prepare.save_reference_preview(pair, directory, 123456)
        for name, expected in originals.items():
            decoded = cv2.cvtColor(cv2.imread(str(directory/(name+".png"))), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(decoded, expected)
            np.testing.assert_array_equal(pair[name], expected)
        metadata = json.loads((directory/"preview.json").read_text())
        self.assertEqual(metadata["host_pair_received_ns"], 123456)
        self.assertEqual(metadata["images"]["left_wrist"]["shape"], [23, 31, 3])
        self.assertEqual(immutable.read_bytes(), b"original immutable guard")
        before = (directory/"left_wrist.png").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare.save_reference_preview(pair, directory, 999)
        self.assertEqual((directory/"left_wrist.png").read_bytes(), before)

    def test_missing_roi_json_waits_without_ticks_until_operator_creates_it(self):
        runner = TimedObservationRunner()
        path = self.root/"operator"/"rois.json"
        wanted = {"left_wrist": [1, 2, 6, 7], "right_wrist": [3, 4, 6, 7]}
        def respond(message):
            self.assertIn(str(path.resolve()), message)
            self.assertFalse(runner.waits)
            self.assertFalse(runner.phases)
            self.assertEqual(runner.backend.images_read, 0)
            runner.backend.clock_ns += 600_000_000_000
            if runner.ask.call_count == 2:
                path.parent.mkdir()
                path.write_text(json.dumps(wanted))
            return ""
        runner.ask.side_effect = respond
        with self.no_device_imports():
            result = prepare.load_or_wait_rois(path, runner)
        self.assertEqual(result, wanted)
        self.assertEqual(runner.ask.call_count, 2)
        # The next observation independently resets its timer after the wait.
        prepare.capture_observation_pairs(runner, 1)
        self.assertEqual(runner.waits[-1][1], 300_000_000)

    def test_existing_roi_json_loads_without_prompt_and_errors_propagate(self):
        runner = TimedObservationRunner()
        path = self.root/"existing.json"
        wanted = {"left_wrist": [1, 2, 6, 7], "right_wrist": [3, 4, 6, 7]}
        path.write_text(json.dumps(wanted))
        self.assertEqual(prepare.load_or_wait_rois(path, runner), wanted)
        runner.ask.assert_not_called()
        path.write_text("invalid JSON")
        with self.assertRaises(json.JSONDecodeError):
            prepare.load_or_wait_rois(path, runner)
        runner.ask.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            prepare.load_or_wait_rois(self.root/"absent.json", runner)

    def test_unattended_missing_roi_does_not_busy_loop(self):
        runner = TimedObservationRunner()
        runner.backend.simulated = True
        with self.assertRaises(FileNotFoundError):
            prepare.load_or_wait_rois(self.root/"absent.json", runner)
        runner.ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
